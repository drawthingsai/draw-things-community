import DataModels
import Diffusion
import Fickling
import Foundation
import ModelZoo
import NNC
import WeightsCache
import ZIPFoundation

public final class ModelImporter {
  public enum Error: Swift.Error {
    case tensorWritesFailed
    case textEncoder(Swift.Error)
    case autoencoder(Swift.Error)
    case noTextEncoder
  }
  private let filePath: String
  private let modelName: String
  private let isTextEncoderCustomized: Bool
  private let autoencoderFilePath: String?
  private let textEncoderFilePath: String?
  private let textEncoder2FilePath: String?
  private var access: Int = 0
  private var expectedTotalAccess: Int = 0
  private var progress: ((Float) -> Void)? = nil
  public init(
    filePath: String, modelName: String, isTextEncoderCustomized: Bool,
    autoencoderFilePath: String?, textEncoderFilePath: String?,
    textEncoder2FilePath: String?
  ) {
    self.filePath = filePath
    self.modelName = modelName
    self.isTextEncoderCustomized = isTextEncoderCustomized
    self.autoencoderFilePath = autoencoderFilePath
    self.textEncoderFilePath = textEncoderFilePath
    self.textEncoder2FilePath = textEncoder2FilePath
  }

  public func `import`(
    versionCheck: @escaping (ModelVersion) -> Void, progress: @escaping (Float) -> Void
  ) throws -> ([String], ModelVersion, SamplerModifier, InspectionResult) {
    self.progress = progress
    defer { self.progress = nil }
    Interpreter.inflateInterrupter = self.interrupt
    defer { Interpreter.inflateInterrupter = nil }
    self.access = 0
    return try self.internalImport(versionCheck: versionCheck)
  }

  private func interrupt() -> Bool {
    access += 1
    progress?(0.05 + Float(access) / Float(max(expectedTotalAccess, 1)) * 0.95)
    return false
  }

  public struct InspectionResult {
    public var version: ModelVersion
    public var archive: TensorArchive
    public var stateDict: [String: TensorDescriptor]
    public var modifier: SamplerModifier
    public var inputChannels: Int
    public var isDiffusersFormat: Bool
    public var hasEncoderHidProj: Bool
    public var hasGuidanceEmbed: Bool
    public var numberOfTensors: Int
    public var qkNorm: Bool
    public var dualAttentionLayers: [Int]
    public var distilledGuidanceLayers: Int
    public init(
      version: ModelVersion, archive: TensorArchive, stateDict: [String: TensorDescriptor],
      modifier: SamplerModifier, inputChannels: Int, isDiffusersFormat: Bool,
      hasEncoderHidProj: Bool, hasGuidanceEmbed: Bool, qkNorm: Bool, dualAttentionLayers: [Int],
      distilledGuidanceLayers: Int, numberOfTensors: Int
    ) {
      self.version = version
      self.archive = archive
      self.stateDict = stateDict
      self.modifier = modifier
      self.inputChannels = inputChannels
      self.isDiffusersFormat = isDiffusersFormat
      self.hasEncoderHidProj = hasEncoderHidProj
      self.hasGuidanceEmbed = hasGuidanceEmbed
      self.qkNorm = qkNorm
      self.dualAttentionLayers = dualAttentionLayers
      self.distilledGuidanceLayers = distilledGuidanceLayers
      self.numberOfTensors = numberOfTensors
    }
  }

  public func inspect() throws -> InspectionResult {
    var archive: TensorArchive
    var stateDict: [String: TensorDescriptor]
    if let safeTensors = SafeTensors(url: URL(fileURLWithPath: filePath)) {
      archive = safeTensors
      let states = safeTensors.states
      stateDict = states
      for (key, value) in states {
        if key.hasPrefix("encoder.") || key.hasPrefix("decoder.")
          || key.hasPrefix("post_quant_conv.") || key.hasPrefix("quant_conv.")
        {
          stateDict["first_stage_model.\(key)"] = value
        }
      }
    } else if let zipArchive = Archive(url: URL(fileURLWithPath: filePath), accessMode: .read) {
      archive = zipArchive
      let rootObject = try Interpreter.unpickle(zip: zipArchive)
      let originalStateDict =
        rootObject["state_dict"] as? Interpreter.Dictionary ?? rootObject["module"]
        as? Interpreter.Dictionary ?? rootObject
      stateDict = [String: TensorDescriptor]()
      originalStateDict.forEach { key, value in
        guard let value = value as? TensorDescriptor else { return }
        stateDict[key] = value
        if key.hasPrefix("encoder.") || key.hasPrefix("decoder.")
          || key.hasPrefix("post_quant_conv.") || key.hasPrefix("quant_conv.")
        {
          stateDict["first_stage_model.\(key)"] = value
        }
      }
    } else {
      throw UnpickleError.dataNotFound
    }
    let isSvdI2v = stateDict.keys.contains { $0.contains("time_mixer") }
    var isWurstchenStageC = stateDict.keys.contains { $0.contains("clip_txt_mapper.") }
    var isSD3Large = stateDict.keys.contains {
      $0.contains("joint_blocks.37.context_block.")
        || $0.contains("transformer_blocks.36.ff_context.")
    }
    var isSD3Medium =
      !isSD3Large
      && stateDict.keys.contains {
        ($0.contains("joint_blocks.23.context_block.")
          || $0.contains("transformer_blocks.22.ff_context."))
      }
    var isPixArtSigmaXL =
      !isSD3Large
      && stateDict.keys.contains {
        $0.contains("blocks.27.cross_attn.kv_") || $0.contains("transformer_blocks.27.attn2.to_")
      }
    var isHunyuan = stateDict.keys.contains {
      $0.contains("double_blocks.19.img_attn_qkv.")
        || $0.contains("single_transformer_blocks.39.")
    }
    var isFlux1 =
      !isHunyuan
      && stateDict.keys.contains {
        $0.contains("double_blocks.18.img_attn.qkv.")
          || $0.contains("single_transformer_blocks.37.")
      }
    var isWan21_14B = stateDict.keys.contains {
      $0.contains("blocks.39.cross_attn.v.") || $0.contains("blocks.39.attn2.to_v.")
    }
    var isWan21_1_3B =
      !isWan21_14B
      && stateDict.keys.contains {
        $0.contains("blocks.29.cross_attn.v.") || $0.contains("blocks.29.attn2.to_v.")
      }
    var isHiDream = stateDict.keys.contains {
      $0.contains("double_stream_blocks.15.block.ff_i.experts.0.")
        || $0.contains("single_stream_blocks.31.block.ff_i.experts.0.")
    }
    let modifier: SamplerModifier
    let modelVersion: ModelVersion
    let inputDim: Int
    let isDiffusersFormat: Bool
    let expectedTotalAccess: Int
    var distilledGuidanceLayers: Int = 0
    // This is for SD v1, v2 and SDXL.
    if let tokey = stateDict[
      "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight"]
      ?? stateDict["down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.weight"],
      let inputConv2d = stateDict["model.diffusion_model.input_blocks.0.0.weight"]
        ?? stateDict["conv_in.weight"]
    {

      inputDim = inputConv2d.shape.count >= 2 ? inputConv2d.shape[1] : 4
      switch inputDim {
      case 9:
        modifier = .inpainting
      case 8:
        if isSvdI2v {
          // For Stable Video Diffusion.
          modifier = .none
        } else {
          modifier = .editing
        }
      case 5:
        modifier = .depth
      default:
        modifier = .none
      }
      switch tokey.shape.last {
      case 2048:
        if !stateDict.keys.contains(where: {
          $0.contains("mid_block.attentions.0.transformer_blocks.")
            || $0.contains("middle_block.1.transformer_blocks.")
        }) {
          modelVersion = .ssd1b
          expectedTotalAccess = 944
        } else {
          modelVersion = .sdxlBase
          expectedTotalAccess = 1680
        }
      case 1280:
        modelVersion = .sdxlRefiner
        expectedTotalAccess = 1220
      case 1024:
        if isSvdI2v {
          modelVersion = .svdI2v
          expectedTotalAccess = 686
        } else {
          modelVersion = .v2
          expectedTotalAccess = 686
        }
      case 768:
        modelVersion = .v1
        expectedTotalAccess = 686
      default:
        throw UnpickleError.tensorNotFound
      }
      isDiffusersFormat = stateDict.keys.contains { $0.hasPrefix("mid_block.") }
      isWurstchenStageC = false
      isPixArtSigmaXL = false
      isSD3Medium = false
      isSD3Large = false
      isFlux1 = false
      isHunyuan = false
      isWan21_14B = false
      isWan21_1_3B = false
      isHiDream = false
    } else if isWurstchenStageC {
      modelVersion = .wurstchenStageC
      modifier = .none
      inputDim = 16
      expectedTotalAccess = 1550
      isDiffusersFormat = false
    } else if isPixArtSigmaXL {
      modelVersion = .pixart
      modifier = .none
      inputDim = 4
      expectedTotalAccess = 754
      isDiffusersFormat = stateDict.keys.contains { $0.contains("transformer_blocks.27.") }
    } else if isSD3Medium {
      modelVersion = .sd3
      modifier = .none
      inputDim = 16
      expectedTotalAccess = 1157
      isDiffusersFormat = stateDict.keys.contains {
        $0.contains("transformer_blocks.22.ff_context.")
      }
    } else if isSD3Large {
      modelVersion = .sd3Large
      modifier = .none
      inputDim = 16
      expectedTotalAccess = 1981
      isDiffusersFormat = stateDict.keys.contains {
        $0.contains("transformer_blocks.36.ff_context.")
      }
    } else if isFlux1 {
      modelVersion = .flux1
      modifier = .none
      inputDim = 16
      expectedTotalAccess = 1732
      isDiffusersFormat = stateDict.keys.contains {
        $0.contains("single_transformer_blocks.37.")
      }
      // If it is flux, check for distilled guidance layer (Chroma).
      distilledGuidanceLayers = 0
      while stateDict.keys.contains(where: {
        $0.contains("distilled_guidance_layer.layers.\(distilledGuidanceLayers).")
      }) {
        distilledGuidanceLayers += 1
      }
    } else if isHunyuan {
      modelVersion = .hunyuanVideo
      modifier = .none
      inputDim = 16
      expectedTotalAccess = 1870
      isDiffusersFormat = stateDict.keys.contains {
        $0.contains("single_transformer_blocks.39.")
      }
    } else if isWan21_14B {
      modelVersion = .wan21_14b
      modifier =
        stateDict.keys.contains {
          $0.contains("blocks.39.cross_attn.v_img.") || $0.contains("blocks.39.attn2.add_v_proj.")
        } ? .inpainting : .none
      inputDim = 16
      expectedTotalAccess = 1306 + (modifier == .inpainting ? 208 : 0)
      isDiffusersFormat = stateDict.keys.contains {
        $0.contains("blocks.39.attn2.to_v.")
      }
    } else if isWan21_1_3B {
      modelVersion = .wan21_1_3b
      modifier =
        stateDict.keys.contains {
          $0.contains("blocks.29.cross_attn.v_img.") || $0.contains("blocks.29.attn2.add_v_proj.")
        } ? .inpainting : .none
      inputDim = 16
      expectedTotalAccess = 986 + (modifier == .inpainting ? 158 : 0)
      isDiffusersFormat = stateDict.keys.contains {
        $0.contains("blocks.29.attn2.to_v.")
      }
    } else if isHiDream {
      modelVersion = .hiDreamI1
      modifier = .none
      inputDim = 16
      expectedTotalAccess = 1857
      isDiffusersFormat = true  // Only Diffusers format available.
    } else {
      throw UnpickleError.tensorNotFound
    }
    let keys = stateDict.keys
    let hasEncoderHidProj = keys.contains { $0 == "encoder_hid_proj.weight" }
    let hasGuidanceEmbed = keys.contains {
      $0.contains(".guidance_embedder.") || $0.contains("guidance_in.")
    }
    let qkNorm = keys.contains {
      $0.contains(".ln_k.") || $0.contains(".ln_q.") || $0.contains(".norm_k.")
        || $0.contains(".norm_q.")
    }
    let dualAttentionLayers = (0..<38).filter { i in
      keys.contains {
        $0.contains(".\(i).x_block.attn2.") || $0.contains("_blocks.\(i).attn2.")
      }
    }
    return InspectionResult(
      version: modelVersion, archive: archive, stateDict: stateDict, modifier: modifier,
      inputChannels: inputDim, isDiffusersFormat: isDiffusersFormat,
      hasEncoderHidProj: hasEncoderHidProj, hasGuidanceEmbed: hasGuidanceEmbed,
      qkNorm: qkNorm, dualAttentionLayers: dualAttentionLayers,
      distilledGuidanceLayers: distilledGuidanceLayers, numberOfTensors: expectedTotalAccess
    )
  }

  private func internalImport(versionCheck: @escaping (ModelVersion) -> Void) throws -> (
    [String], ModelVersion, SamplerModifier, InspectionResult
  ) {
    let inspectionResult = try inspect()
    var archive = inspectionResult.archive
    var stateDict = inspectionResult.stateDict
    var modifier = inspectionResult.modifier
    let modelVersion = inspectionResult.version
    let inputDim = inspectionResult.inputChannels
    let isDiffusersFormat = inspectionResult.isDiffusersFormat
    let hasEncoderHidProj = inspectionResult.hasEncoderHidProj
    let qkNorm = inspectionResult.qkNorm
    let dualAttentionLayers = inspectionResult.dualAttentionLayers
    let distilledGuidanceLayers = inspectionResult.distilledGuidanceLayers
    expectedTotalAccess = inspectionResult.numberOfTensors
    versionCheck(modelVersion)
    progress?(0.05)
    if isTextEncoderCustomized {
      switch modelVersion {
      case .v1:
        expectedTotalAccess += 196
      case .v2:
        expectedTotalAccess += 280
      case .sdxlBase, .ssd1b:
        expectedTotalAccess += 388 + 196
      case .sdxlRefiner:
        expectedTotalAccess += 388
      case .sd3, .sd3Large, .pixart, .auraflow, .flux1, .hunyuanVideo:
        throw Error.noTextEncoder
      case .svdI2v:
        throw Error.noTextEncoder
      case .wurstchenStageC, .wurstchenStageB:
        throw Error.noTextEncoder
      case .wan21_1_3b, .wan21_14b:
        throw Error.noTextEncoder
      case .hiDreamI1:
        throw Error.noTextEncoder
      case .kandinsky21:
        fatalError()
      }
    }
    if autoencoderFilePath != nil {
      expectedTotalAccess += 248
    }
    let graph = DynamicGraph()
    var filePaths = [String]()
    if isTextEncoderCustomized {
      do {
        try graph.withNoGrad {
          let tokensTensor = graph.variable(.CPU, .C(2 * 77), of: Int32.self)
          let positionTensor = graph.variable(.CPU, .C(2 * 77), of: Int32.self)
          let causalAttentionMask = graph.variable(.CPU, .NHWC(1, 1, 77, 77), of: FloatType.self)
          var textModel: Model
          var textModelReader: PythonReader
          var textEncoderArchive = archive
          var filePath: String
          var textEncoderStateDict = stateDict
          if let textEncoderFilePath = textEncoderFilePath {
            if let safeTensors = SafeTensors(url: URL(fileURLWithPath: textEncoderFilePath)) {
              textEncoderArchive = safeTensors
              textEncoderStateDict = safeTensors.states
              for (key, value) in textEncoderStateDict {
                textEncoderStateDict["cond_stage_model.transformer.\(key)"] = value
              }
            } else if let zipArchive = Archive(
              url: URL(fileURLWithPath: textEncoderFilePath), accessMode: .read)
            {
              textEncoderArchive = zipArchive
              let rootObject = try Interpreter.unpickle(zip: zipArchive)
              let originalStateDict =
                rootObject["state_dict"] as? Interpreter.Dictionary ?? rootObject["module"]
                as? Interpreter.Dictionary ?? rootObject
              textEncoderStateDict = [String: TensorDescriptor]()
              originalStateDict.forEach { key, value in
                guard let value = value as? TensorDescriptor else { return }
                textEncoderStateDict["cond_stage_model.transformer.\(key)"] = value
                textEncoderStateDict[key] = value
              }
            } else {
              throw UnpickleError.dataNotFound
            }
          }
          switch modelVersion {
          case .v1, .sdxlBase, .ssd1b:
            (textModel, textModelReader) = CLIPTextModel(
              FloatType.self, injectEmbeddings: false,
              vocabularySize: 49408, maxLength: 77, maxTokenLength: 77, embeddingSize: 768,
              numLayers: 12, numHeads: 12,
              batchSize: 2, intermediateSize: 3072, usesFlashAttention: false)
            filePath = ModelZoo.filePathForModelDownloaded("\(modelName)_clip_vit_l14_f16.ckpt")
          case .v2:
            (textModel, textModelReader) = OpenCLIPTextModel(
              FloatType.self, injectEmbeddings: false,
              vocabularySize: 49408, maxLength: 77, maxTokenLength: 77, embeddingSize: 1024,
              numLayers: 23, numHeads: 16,
              batchSize: 2, intermediateSize: 4096, usesFlashAttention: false)
            filePath = ModelZoo.filePathForModelDownloaded(
              "\(modelName)_open_clip_vit_h14_f16.ckpt")
          case .sdxlRefiner:
            (textModel, textModelReader) = OpenCLIPTextModel(
              FloatType.self, injectEmbeddings: false,
              vocabularySize: 49408, maxLength: 77, maxTokenLength: 77, embeddingSize: 1280,
              numLayers: 32, numHeads: 20, batchSize: 2,
              intermediateSize: 5120, usesFlashAttention: false, outputPenultimate: true)
            filePath = ModelZoo.filePathForModelDownloaded(
              "\(modelName)_open_clip_vit_bigg14_f16.ckpt")
          case .sd3, .sd3Large, .pixart, .auraflow, .flux1, .kandinsky21, .svdI2v, .wurstchenStageC,
            .wurstchenStageB, .hunyuanVideo, .wan21_1_3b, .wan21_14b, .hiDreamI1:
            fatalError()
          }
          if modelVersion == .sdxlBase || modelVersion == .sdxlRefiner {
            for (key, value) in textEncoderStateDict {
              if key.hasPrefix("conditioner.embedders.0.") {
                textEncoderStateDict["cond_stage_model." + key.dropFirst(24)] = value
              }
            }
          }
          textModel.compile(inputs: tokensTensor, positionTensor, causalAttentionMask)
          try textModelReader(textEncoderStateDict, textEncoderArchive)

          try graph.openStore(filePath) { store in
            store.removeAll()
            if let text_projection = textEncoderStateDict["cond_stage_model.model.text_projection"]
              ?? textEncoderStateDict["cond_stage_model.text_projection"]
            {
              try textEncoderArchive.with(text_projection) {
                store.write("text_projection", tensor: $0)
              }
            }
            store.write("text_model", model: textModel)
          }
          try graph.openStore(filePath) {
            switch modelVersion {
            case .v1, .sdxlBase, .ssd1b:
              if $0.keys.count < 196 {
                throw Error.tensorWritesFailed
              }
            case .v2:
              if $0.keys.count < 372 {
                throw Error.tensorWritesFailed
              }
            case .sdxlRefiner:
              if $0.keys.count < 517 {
                throw Error.tensorWritesFailed
              }
            case .sd3, .sd3Large, .pixart, .auraflow, .flux1, .kandinsky21, .svdI2v,
              .wurstchenStageC, .wurstchenStageB, .hunyuanVideo, .wan21_1_3b, .wan21_14b,
              .hiDreamI1:
              fatalError()
            }
          }
          filePaths.append(filePath)
          if modelVersion == .sdxlBase {
            // SDXL Base has two text encoders.
            (textModel, textModelReader) = OpenCLIPTextModel(
              FloatType.self, injectEmbeddings: false,
              vocabularySize: 49408, maxLength: 77, maxTokenLength: 77, embeddingSize: 1280,
              numLayers: 32, numHeads: 20, batchSize: 2,
              intermediateSize: 5120, usesFlashAttention: false, outputPenultimate: true)
            filePath = ModelZoo.filePathForModelDownloaded(
              "\(modelName)_open_clip_vit_bigg14_f16.ckpt")
            var textEncoder2StateDict = stateDict
            var textEncoder2Archive: TensorArchive = archive
            var textEncoder2TextProjectionTransposed = false
            if let textEncoder2FilePath = textEncoder2FilePath {
              if let safeTensors = SafeTensors(url: URL(fileURLWithPath: textEncoder2FilePath)) {
                textEncoder2Archive = safeTensors
                textEncoder2StateDict = safeTensors.states
                for (key, value) in textEncoder2StateDict {
                  if key.hasPrefix("text_projection") {
                    textEncoder2StateDict["cond_stage_model.model.text_projection"] = value
                    if key.hasSuffix(".weight") {
                      textEncoder2TextProjectionTransposed = true
                    }
                  } else {
                    textEncoder2StateDict["cond_stage_model.transformer.\(key)"] = value
                  }
                }
              } else if let zipArchive = Archive(
                url: URL(fileURLWithPath: textEncoder2FilePath), accessMode: .read)
              {
                textEncoder2Archive = zipArchive
                let rootObject = try Interpreter.unpickle(zip: zipArchive)
                textEncoder2StateDict = [String: TensorDescriptor]()
                let originalStateDict =
                  rootObject["state_dict"] as? Interpreter.Dictionary ?? rootObject["module"]
                  as? Interpreter.Dictionary ?? rootObject
                originalStateDict.forEach { key, value in
                  guard let value = value as? TensorDescriptor else { return }
                  if key.hasPrefix("text_projection") {
                    textEncoder2StateDict["cond_stage_model.model.text_projection"] = value
                    if key.hasSuffix(".weight") {
                      textEncoder2TextProjectionTransposed = true
                    }
                  } else {
                    textEncoder2StateDict["cond_stage_model.transformer.\(key)"] = value
                  }
                  textEncoder2StateDict[key] = value
                }
              } else {
                throw UnpickleError.dataNotFound
              }
            }
            for (key, value) in textEncoder2StateDict {
              if key.hasPrefix("conditioner.embedders.1.") {
                textEncoder2StateDict["cond_stage_model." + key.dropFirst(24)] = value
              }
            }
            textModel.compile(inputs: tokensTensor, positionTensor, causalAttentionMask)
            try textModelReader(textEncoder2StateDict, textEncoder2Archive)
            try graph.openStore(filePath) { store in
              store.removeAll()
              if let text_projection = textEncoder2StateDict[
                "cond_stage_model.model.text_projection"]
                ?? textEncoder2StateDict["cond_stage_model.text_projection"]
              {
                if textEncoder2TextProjectionTransposed {
                  try textEncoder2Archive.with(text_projection) { tensor in
                    let textProjection = graph.variable(
                      Tensor<FloatType>(
                        from: tensor)
                    ).toGPU(0)
                    store.write(
                      "text_projection", tensor: textProjection.transposed(0, 1).toCPU().rawValue)
                  }
                } else {
                  try textEncoder2Archive.with(text_projection) {
                    store.write("text_projection", tensor: $0)
                  }
                }
              }
              store.write("text_model", model: textModel)
            }
            try graph.openStore(filePath) {
              if $0.keys.count != 517 {
                throw Error.tensorWritesFailed
              }
            }
            filePaths.append(filePath)
          }
        }
      } catch {
        throw Error.textEncoder(error)
      }
    }
    let conditionalLength: Int
    let batchSize: Int
    switch modelVersion {
    case .v1:
      conditionalLength = 768
      batchSize = 2
    case .v2:
      conditionalLength = 1024
      batchSize = 2
    case .svdI2v:
      conditionalLength = 1024
      batchSize = 1
    case .sdxlBase, .sdxlRefiner, .ssd1b:
      conditionalLength = 1280
      batchSize = 2
    case .wurstchenStageC:
      conditionalLength = 1280
      batchSize = 2
    case .pixart:
      conditionalLength = 4096
      batchSize = 2
    case .sd3, .sd3Large:
      conditionalLength = 4096
      batchSize = 2
    case .auraflow:
      conditionalLength = 2048
      batchSize = 2
    case .flux1:
      conditionalLength = 4096
      batchSize = 1
    case .hunyuanVideo:
      conditionalLength = 4096
      batchSize = 1
    case .wan21_1_3b, .wan21_14b:
      conditionalLength = 4096
      batchSize = 1
    case .hiDreamI1:
      conditionalLength = 4096
      batchSize = 1
    case .kandinsky21, .wurstchenStageB:
      fatalError()
    }
    try graph.withNoGrad {
      var tEmb: DynamicGraph.Tensor<FloatType>? = graph.variable(
        Tensor<FloatType>(
          from: timeEmbedding(
            timestep: 981, batchSize: batchSize,
            embeddingSize: modelVersion == .sdxlRefiner ? 384 : 320,
            maxPeriod: 10_000)
        ))
      let xTensor = graph.variable(.CPU, .NHWC(batchSize, 64, 64, inputDim), of: FloatType.self)
      let cTensor = graph.variable(.CPU, .HWC(batchSize, 77, conditionalLength), of: FloatType.self)
      var cArr = [cTensor]
      let unet: Model
      var unetReader: PythonReader?
      let unetMapper: ModelWeightMapper?
      let filePath = ModelZoo.filePathForModelDownloaded("\(self.modelName)_f16.ckpt")
      switch modelVersion {
      case .sdxlBase, .sdxlRefiner, .ssd1b, .svdI2v, .wurstchenStageB, .wurstchenStageC, .pixart,
        .sd3, .sd3Large, .auraflow:
        let fixedEncoder = UNetFixedEncoder<FloatType>(
          filePath: "", version: modelVersion, modifier: .none,
          dualAttentionLayers: dualAttentionLayers,
          usesFlashAttention: false, zeroNegativePrompt: false,
          isQuantizedModel: false, canRunLoRASeparately: false, externalOnDemand: false,
          deviceProperties: DeviceProperties(
            isFreadPreferred: true, memoryCapacity: .high, isNHWCPreferred: true),
          weightsCache: WeightsCache(maxTotalCacheSize: 0, memorySubsystem: .UMA))
        cArr.insert(
          graph.variable(.CPU, .HWC(batchSize, 77, 768), of: FloatType.self),
          at: 0)
        cArr.append(
          graph.variable(.CPU, .WC(batchSize, 1280), of: FloatType.self))
        for c in cArr {
          c.full(0)
        }
        let vectors: [DynamicGraph.Tensor<FloatType>]
        switch modelVersion {
        case .sdxlBase, .ssd1b:
          vectors = [graph.variable(.CPU, .WC(batchSize, 2816), of: FloatType.self)]
        case .sdxlRefiner:
          vectors = [graph.variable(.CPU, .WC(batchSize, 2560), of: FloatType.self)]
        case .svdI2v:
          vectors = [graph.variable(.CPU, .WC(batchSize, 768), of: FloatType.self)]
        case .wurstchenStageC, .wurstchenStageB, .pixart, .sd3, .sd3Large, .auraflow, .flux1,
          .hunyuanVideo, .wan21_1_3b, .wan21_14b, .hiDreamI1:
          vectors = []
        case .kandinsky21, .v1, .v2:
          fatalError()
        }
        // These values doesn't matter, it won't affect the model shape, just the input vector.
        cArr =
          vectors
          + fixedEncoder.encode(
            isCfgEnabled: true, textGuidanceScale: 3.5, guidanceEmbed: 3.5,
            isGuidanceEmbedEnabled: false, distilledGuidanceLayers: 0,
            textEncoding: cArr.map({ $0.toGPU(0) }), timesteps: [0], batchSize: batchSize,
            startHeight: 64, startWidth: 64,
            tokenLengthUncond: 77, tokenLengthCond: 77, lora: [],
            tiledDiffusion: TiledConfiguration(
              isEnabled: false, tileSize: .init(width: 0, height: 0), tileOverlap: 0),
            teaCache: TeaCacheConfiguration(
              coefficients: (0, 0, 0, 0, 0), steps: 0...0, threshold: 0, maxSkipSteps: 0),
            injectedControls: []
          ).0.map({ DynamicGraph.Tensor<FloatType>($0).toCPU() })
        if modelVersion == .svdI2v {
          // Only take the first half (positive part).
          cArr = Array(cArr[0..<(1 + (cArr.count - 1) / 2)])
        }
      case .flux1:
        cArr =
          [
            graph.variable(
              Tensor<FloatType>(
                from: Flux1RotaryPositionEmbedding(
                  height: 32, width: 32, tokenLength: 256, channels: 128)))
          ]
          + Flux1FixedOutputShapes(
            batchSize: (1, 1), tokenLength: 256, channels: 3072, layers: (19, 38),
            contextPreloaded: true
          ).map {
            graph.variable(.CPU, format: .NHWC, shape: $0, of: FloatType.self)
          }
      case .hunyuanVideo:
        let (rot0, rot1) = HunyuanRotaryPositionEmbedding(
          height: 64, width: 64, time: 1, tokenLength: 20, channels: 128)
        cArr =
          [
            graph.variable(Tensor<FloatType>(from: rot0)),
            graph.variable(Tensor<FloatType>(from: rot1)),
          ]
          + HunyuanFixedOutputShapes(
            batchSize: batchSize, channels: 3072, layers: (20, 40), textLength: 20
          ).map {
            graph.variable(.CPU, format: .NHWC, shape: $0, of: FloatType.self)
          }
      case .wan21_1_3b:
        let rot = Tensor<FloatType>(
          from: WanRotaryPositionEmbedding(
            height: 64, width: 64, time: 1, channels: 128)
        )
        cArr =
          [graph.variable(rot)]
          + WanFixedOutputShapes(
            timesteps: 1, batchSize: (1, 1), channels: 1_536, layers: 30, textLength: 512,
            injectImage: true
          ).map {
            graph.variable(.CPU, format: .NHWC, shape: $0, of: FloatType.self)
          }
      case .wan21_14b:
        let rot = Tensor<FloatType>(
          from: WanRotaryPositionEmbedding(
            height: 64, width: 64, time: 1, channels: 128)
        )
        cArr =
          [graph.variable(rot)]
          + WanFixedOutputShapes(
            timesteps: 1, batchSize: (1, 1), channels: 5_120, layers: 40, textLength: 512,
            injectImage: true
          ).map {
            graph.variable(.CPU, format: .NHWC, shape: $0, of: FloatType.self)
          }
      case .hiDreamI1:
        cArr =
          [
            graph.variable(
              Tensor<FloatType>(
                from: HiDreamRotaryPositionEmbedding(
                  height: 32, width: 32, tokenLength: 128, channels: 128)))
          ]
          + HiDreamFixedOutputShapes(
            timesteps: 1, layers: (16, 32), textLength: 128
          ).map {
            graph.variable(.CPU, format: .NHWC, shape: $0, of: FloatType.self)
          }
      case .kandinsky21, .v1, .v2:
        break
      }
      let unetFixed: Model?
      let unetFixedMapper: ModelWeightMapper?
      switch modelVersion {
      case .v1:
        (unet, unetReader) = UNet(
          batchSize: batchSize, embeddingLength: (77, 77), startWidth: 64, startHeight: 64,
          usesFlashAttention: .none, injectControls: false, injectT2IAdapters: false,
          injectIPAdapterLengths: [], injectAttentionKV: false)
        unetMapper = nil
        (unetFixed, unetFixedMapper) = (nil, nil)
      case .v2:
        (unet, unetReader) = UNetv2(
          batchSize: batchSize, embeddingLength: (77, 77), startWidth: 64, startHeight: 64,
          upcastAttention: false, usesFlashAttention: .none, injectControls: false)
        unetMapper = nil
        (unetFixed, unetFixedMapper) = (nil, nil)
      case .sdxlBase:
        (unet, _, unetMapper) = UNetXL(
          batchSize: batchSize, startHeight: 64, startWidth: 64,
          channels: [320, 640, 1280], inputAttentionRes: [2: [2, 2], 4: [10, 10]],
          middleAttentionBlocks: 10, outputAttentionRes: [2: [2, 2, 2], 4: [10, 10, 10]],
          embeddingLength: (77, 77), injectIPAdapterLengths: [],
          upcastAttention: ([:], false, [:]), usesFlashAttention: .none, injectControls: false,
          isTemporalMixEnabled: false, of: FloatType.self)
        unetReader = nil
        (unetFixed, _, unetFixedMapper) = UNetXLFixed(
          batchSize: batchSize, startHeight: 64, startWidth: 64, channels: [320, 640, 1280],
          embeddingLength: (77, 77), inputAttentionRes: [2: [2, 2], 4: [10, 10]],
          middleAttentionBlocks: 10, outputAttentionRes: [2: [2, 2, 2], 4: [10, 10, 10]],
          usesFlashAttention: .none, isTemporalMixEnabled: false)
      case .ssd1b:
        (unet, _, unetMapper) = UNetXL(
          batchSize: batchSize, startHeight: 64, startWidth: 64,
          channels: [320, 640, 1280], inputAttentionRes: [2: [2, 2], 4: [4, 4]],
          middleAttentionBlocks: 0, outputAttentionRes: [2: [2, 1, 1], 4: [4, 4, 10]],
          embeddingLength: (77, 77), injectIPAdapterLengths: [],
          upcastAttention: ([:], false, [:]), usesFlashAttention: .none, injectControls: false,
          isTemporalMixEnabled: false, of: FloatType.self)
        unetReader = nil
        (unetFixed, _, unetFixedMapper) = UNetXLFixed(
          batchSize: batchSize, startHeight: 64, startWidth: 64, channels: [320, 640, 1280],
          embeddingLength: (77, 77), inputAttentionRes: [2: [2, 2], 4: [4, 4]],
          middleAttentionBlocks: 0, outputAttentionRes: [2: [2, 1, 1], 4: [4, 4, 10]],
          usesFlashAttention: .none, isTemporalMixEnabled: false)
      case .sdxlRefiner:
        (unet, _, unetMapper) =
          UNetXL(
            batchSize: batchSize, startHeight: 64, startWidth: 64,
            channels: [384, 768, 1536, 1536], inputAttentionRes: [2: [4, 4], 4: [4, 4]],
            middleAttentionBlocks: 4, outputAttentionRes: [2: [4, 4, 4], 4: [4, 4, 4]],
            embeddingLength: (77, 77), injectIPAdapterLengths: [],
            upcastAttention: ([:], false, [:]), usesFlashAttention: .none, injectControls: false,
            isTemporalMixEnabled: false, of: FloatType.self
          )
        unetReader = nil
        (unetFixed, _, unetFixedMapper) = UNetXLFixed(
          batchSize: batchSize, startHeight: 64, startWidth: 64, channels: [384, 768, 1536, 1536],
          embeddingLength: (77, 77), inputAttentionRes: [2: [4, 4], 4: [4, 4]],
          middleAttentionBlocks: 4, outputAttentionRes: [2: [4, 4, 4], 4: [4, 4, 4]],
          usesFlashAttention: .none, isTemporalMixEnabled: false)
      case .svdI2v:
        (unet, _, unetMapper) =
          UNetXL(
            batchSize: batchSize, startHeight: 64, startWidth: 64,
            channels: [320, 640, 1280, 1280],
            inputAttentionRes: [1: [1, 1], 2: [1, 1], 4: [1, 1]], middleAttentionBlocks: 1,
            outputAttentionRes: [1: [1, 1, 1], 2: [1, 1, 1], 4: [1, 1, 1]], embeddingLength: (1, 1),
            injectIPAdapterLengths: [], upcastAttention: ([:], false, [1: [0, 1, 2]]),
            usesFlashAttention: .none, injectControls: false,
            isTemporalMixEnabled: true, of: FloatType.self
          )
        unetReader = nil
        (unetFixed, _, unetFixedMapper) = UNetXLFixed(
          batchSize: batchSize, startHeight: 64, startWidth: 64,
          channels: [320, 640, 1280, 1280], embeddingLength: (1, 1),
          inputAttentionRes: [1: [1, 1], 2: [1, 1], 4: [1, 1]], middleAttentionBlocks: 1,
          outputAttentionRes: [1: [1, 1, 1], 2: [1, 1, 1], 4: [1, 1, 1]], usesFlashAttention: .none,
          isTemporalMixEnabled: true)
      case .wurstchenStageC:
        (unet, unetMapper) = WurstchenStageC(
          batchSize: batchSize, height: 24, width: 24,
          t: (77 + 8, 77 + 8),
          usesFlashAttention: .none)
        unetReader = nil
        (unetFixed, unetFixedMapper) = WurstchenStageCFixed(
          batchSize: batchSize, t: (77 + 8, 77 + 8),
          usesFlashAttention: .none)
      case .pixart:
        (unetMapper, unet) = PixArt(
          batchSize: batchSize, height: 64, width: 64, channels: 1152, layers: 28,
          tokenLength: (77, 77), usesFlashAttention: false, of: FloatType.self)
        unetReader = nil
        (unetFixedMapper, unetFixed) = PixArtFixed(
          batchSize: batchSize, channels: 1152, layers: 28, tokenLength: (77, 77),
          usesFlashAttention: false, of: FloatType.self)
      case .sd3:
        (unetMapper, unet) = MMDiT(
          batchSize: batchSize, t: 77, height: 64, width: 64, channels: 1536, layers: 24,
          upcast: false, qkNorm: qkNorm, dualAttentionLayers: dualAttentionLayers,
          posEmbedMaxSize: 192, usesFlashAttention: .none, of: FloatType.self)
        unetReader = nil
        (unetFixedMapper, unetFixed) = MMDiTFixed(
          batchSize: batchSize, channels: 1536, layers: 24, dualAttentionLayers: dualAttentionLayers
        )
      case .sd3Large:
        (unetMapper, unet) = MMDiT(
          batchSize: batchSize, t: 77, height: 64, width: 64, channels: 2432, layers: 38,
          upcast: true, qkNorm: true, dualAttentionLayers: [], posEmbedMaxSize: 192,
          usesFlashAttention: .none, of: FloatType.self)
        unetReader = nil
        (unetFixedMapper, unetFixed) = MMDiTFixed(
          batchSize: batchSize, channels: 2432, layers: 38, dualAttentionLayers: [])
      case .flux1:
        (unetMapper, unet) = Flux1(
          batchSize: batchSize, tokenLength: 256, height: 64, width: 64, channels: 3072,
          layers: (19, 38), usesFlashAttention: .scaleMerged, contextPreloaded: true,
          injectControls: false, injectIPAdapterLengths: [:], outputResidual: false,
          inputResidual: false)
        if distilledGuidanceLayers > 0 {
          (unetFixedMapper, unetFixed) = ChromaFixed(
            channels: 3072, distilledGuidanceLayers: distilledGuidanceLayers, layers: (19, 38),
            contextPreloaded: true)
        } else {
          (unetFixedMapper, unetFixed) = Flux1Fixed(
            batchSize: (batchSize, batchSize), channels: 3072, layers: (19, 38),
            contextPreloaded: true, guidanceEmbed: true)
        }
      case .hunyuanVideo:
        (unetMapper, unet) = Hunyuan(
          time: 1, height: 64, width: 64, textLength: 20, channels: 3072, layers: (20, 40),
          usesFlashAttention: .scaleMerged, outputResidual: false, inputResidual: false)
        (unetFixedMapper, unetFixed) = HunyuanFixed(
          timesteps: batchSize, channels: 3072, layers: (20, 40), textLength: (0, 20))
      case .wan21_1_3b:
        (unetMapper, unet) = Wan(
          channels: 1_536, layers: 30, vaceLayers: [], intermediateSize: 8_960, time: 1, height: 64,
          width: 64,
          textLength: 512, causalInference: 0, injectImage: true, usesFlashAttention: true,
          outputResidual: false, inputResidual: false)
        (unetFixedMapper, unetFixed) = WanFixed(
          timesteps: 1, batchSize: (1, 1), channels: 1_536, layers: 30, vaceLayers: [],
          textLength: 512, injectImage: true)
      case .wan21_14b:
        (unetMapper, unet) = Wan(
          channels: 5_120, layers: 40, vaceLayers: [], intermediateSize: 13_824, time: 1,
          height: 64, width: 64,
          textLength: 512, causalInference: 0, injectImage: true, usesFlashAttention: true,
          outputResidual: false, inputResidual: false)
        (unetFixedMapper, unetFixed) = WanFixed(
          timesteps: 1, batchSize: (1, 1), channels: 5_120, layers: 40, vaceLayers: [],
          textLength: 512, injectImage: true)
      case .hiDreamI1:
        (unet, unetMapper) = HiDream(
          batchSize: 1, height: 64, width: 64, textLength: (128, 128), layers: (16, 32),
          usesFlashAttention: true, outputResidual: false, inputResidual: false)
        (unetFixed, unetFixedMapper) = HiDreamFixed(
          timesteps: 1, layers: (16, 32), outputTimesteps: false)
      case .auraflow:
        fatalError()
      case .kandinsky21, .wurstchenStageB:
        fatalError()
      }
      let crossattn: [DynamicGraph.Tensor<FloatType>]
      switch modelVersion {
      case .sdxlBase, .ssd1b:
        crossattn = [graph.variable(.CPU, .HWC(batchSize, 77, 2048), of: FloatType.self)]
      case .sdxlRefiner:
        crossattn = [graph.variable(.CPU, .HWC(batchSize, 77, 1280), of: FloatType.self)]
      case .svdI2v:
        let numFramesEmb = [320, 640, 1280, 1280].map { embeddingSize in
          let tensors = (0..<batchSize).map {
            graph.variable(
              timeEmbedding(
                timestep: Float($0), batchSize: 1, embeddingSize: embeddingSize, maxPeriod: 10_000)
            )
          }
          return DynamicGraph.Tensor<FloatType>(
            from: Concat(axis: 0)(inputs: tensors[0], Array(tensors[1...]))[0].as(of: Float.self))
        }
        crossattn =
          [graph.variable(.CPU, .HWC(batchSize, 1, 1024), of: FloatType.self)] + numFramesEmb
      case .wurstchenStageC:
        crossattn = [
          graph.variable(.CPU, .HWC(batchSize, 77, 1280), of: FloatType.self),
          graph.variable(.CPU, .HWC(batchSize, 1, 1280), of: FloatType.self),
          graph.variable(.CPU, .HWC(batchSize, 1, 1280), of: FloatType.self),
        ]
      case .pixart:
        crossattn = [
          graph.variable(
            Tensor<FloatType>(
              from: timeEmbedding(
                timestep: 1000, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000)
            ).reshaped(.HWC(1, 1, 256))
          ), graph.variable(.CPU, .HWC(batchSize, 77, 4096), of: FloatType.self),
        ]
        tEmb = nil
      case .sd3, .sd3Large:
        crossattn = [
          graph.variable(.CPU, .WC(batchSize, 2048), of: FloatType.self),
          graph.variable(
            Tensor<FloatType>(
              from: timeEmbedding(
                timestep: 1000, batchSize: batchSize, embeddingSize: 256, maxPeriod: 10_000)
            ).reshaped(.WC(batchSize, 256))
          ), graph.variable(.CPU, .HWC(batchSize, 154, 4096), of: FloatType.self),
        ]
        tEmb = nil
      case .flux1:
        if distilledGuidanceLayers > 0 {
          crossattn = [
            graph.variable(.CPU, .HWC(batchSize, 256, 4096), of: FloatType.self),
            graph.variable(.CPU, .HWC(batchSize, 344, 64), of: FloatType.self),
          ]
        } else {
          crossattn = [
            graph.variable(.CPU, .HWC(batchSize, 256, 4096), of: FloatType.self),
            graph.variable(.CPU, .WC(batchSize, 256), of: FloatType.self),
            graph.variable(.CPU, .WC(batchSize, 768), of: FloatType.self),
            graph.variable(.CPU, .WC(batchSize, 256), of: FloatType.self),
          ]
        }
        tEmb = nil
      case .hunyuanVideo:
        crossattn = [
          graph.variable(.CPU, .HWC(batchSize, 20, 4096), of: FloatType.self),
          graph.variable(.CPU, .WC(batchSize, 256), of: FloatType.self),
          graph.variable(.CPU, .WC(batchSize, 768), of: FloatType.self),
          graph.variable(.CPU, .WC(batchSize, 256), of: FloatType.self),
        ]
        tEmb = nil
      case .wan21_1_3b, .wan21_14b:
        crossattn = [
          graph.variable(.CPU, .HWC(1, 512, 4096), of: FloatType.self),
          graph.variable(.CPU, .WC(1, 256), of: FloatType.self),
          graph.variable(.CPU, .HWC(1, 257, 1280), of: FloatType.self),
        ]
        tEmb = nil
      case .hiDreamI1:
        crossattn =
          [
            graph.variable(.CPU, .WC(1, 256), of: FloatType.self),
            graph.variable(.CPU, .WC(1, 2048), of: FloatType.self),
            graph.variable(.CPU, .HWC(1, 128, 4096), of: FloatType.self),
          ]
          + (0..<32).map { _ in
            graph.variable(.CPU, .HWC(1, 128, 4096), of: FloatType.self)  // Llama encoder hidden states.
          }
        tEmb = nil
      case .auraflow:
        fatalError()
      case .v1, .v2, .kandinsky21, .wurstchenStageB:
        crossattn = []
      }
      if modelVersion == .v1 || modelVersion == .v2 {
        for (key, value) in stateDict {
          if !key.hasPrefix("model.diffusion_model.") {
            stateDict["model.diffusion_model.\(key)"] = value
          }
        }
        for (key, value) in stateDict {
          for name in DiffusersMapping.UNetPartials {
            if key.contains(name.1) {
              var newKey = key.replacingOccurrences(of: name.1, with: name.0)
              if key.contains(".resnets.") {
                for name in DiffusersMapping.ResNetsPartials {
                  if newKey.contains(name.1) {
                    newKey = newKey.replacingOccurrences(of: name.1, with: name.0)
                    break
                  }
                }
              }
              stateDict[newKey] = value
              break
            }
          }
        }
      } else if modelVersion == .wurstchenStageC {
        // Remove the model.diffusion_model prefix.
        for (key, value) in stateDict {
          if key.hasPrefix("model.diffusion_model.") {
            stateDict[String(key.dropFirst(22))] = value
          }
        }
      } else if modelVersion == .sd3 || modelVersion == .sd3Large {
        // Remove the model prefix.
        for (key, value) in stateDict {
          if key.hasPrefix("model.") {
            stateDict[String(key.dropFirst(6))] = value
          }
        }
      } else if modelVersion == .pixart {
        // Remove the model.diffusion_model / model prefix.
        for (key, value) in stateDict {
          if key.hasPrefix("model.diffusion_model.") {
            stateDict[String(key.dropFirst(22))] = value
          } else if key.hasPrefix("model.") {
            stateDict[String(key.dropFirst(6))] = value
          }
        }
      } else if modelVersion == .flux1 {
        // Remove the model.diffusion_model / model prefix.
        for (key, value) in stateDict {
          if key.hasPrefix("model.diffusion_model.") {
            stateDict[String(key.dropFirst(22))] = value
          } else if key.hasPrefix("model.") {
            stateDict[String(key.dropFirst(6))] = value
          }
        }
      } else if modelVersion == .hiDreamI1 {
        // Remove the model.diffusion_model / model prefix.
        for (key, value) in stateDict {
          if key.hasPrefix("model.diffusion_model.") {
            stateDict[String(key.dropFirst(22))] = value
          } else if key.hasPrefix("model.") {
            stateDict[String(key.dropFirst(6))] = value
          }
        }
      }
      // In case it is not on high performance device and it is SDXL model, read the parameters directly from the mapping.
      if let unetReader = unetReader {
        let inputs: [DynamicGraph.Tensor<FloatType>] = [xTensor] + (tEmb.map { [$0] } ?? []) + cArr
        unet.compile(inputs: inputs)
        try! unetReader(stateDict, archive)
        graph.openStore(filePath) {
          if unetFixed == nil {
            $0.removeAll()
          }
          $0.write("unet", model: unet)
        }
      } else if let unetMapper = unetMapper, let unetFixedMapper = unetFixedMapper,
        let unetFixed = unetFixed
      {
        try graph.openStore(filePath) { store in
          let UNetMapping: ModelWeightMapping
          let UNetMappingFixed: ModelWeightMapping
          let modelPrefix: String
          let modelPrefixFixed: String
          switch modelVersion {
          case .sdxlBase:
            if isDiffusersFormat {
              let inputs: [DynamicGraph.Tensor<FloatType>] =
                [xTensor] + (tEmb.map { [$0] } ?? []) + cArr
              unet.compile(inputs: inputs)
              unetFixed.compile(inputs: crossattn)
              UNetMapping = unetMapper(.diffusers)
              UNetMappingFixed = unetFixedMapper(.diffusers)
            } else {
              UNetMapping = StableDiffusionMapping.UNetXLBase
              UNetMappingFixed = StableDiffusionMapping.UNetXLBaseFixed
            }
            modelPrefix = "unet"
            modelPrefixFixed = "unet_fixed"
          case .sdxlRefiner:
            if isDiffusersFormat {
              let inputs: [DynamicGraph.Tensor<FloatType>] =
                [xTensor] + (tEmb.map { [$0] } ?? []) + cArr
              unet.compile(inputs: inputs)
              unetFixed.compile(inputs: crossattn)
              UNetMapping = unetMapper(.diffusers)
              UNetMappingFixed = unetFixedMapper(.diffusers)
            } else {
              UNetMapping = StableDiffusionMapping.UNetXLRefiner
              UNetMappingFixed = StableDiffusionMapping.UNetXLRefinerFixed
            }
            modelPrefix = "unet"
            modelPrefixFixed = "unet_fixed"
          case .ssd1b:
            let inputs: [DynamicGraph.Tensor<FloatType>] =
              [xTensor] + (tEmb.map { [$0] } ?? []) + cArr
            unet.compile(inputs: inputs)
            unetFixed.compile(inputs: crossattn)
            UNetMapping = unetMapper(isDiffusersFormat ? .diffusers : .generativeModels)
            UNetMappingFixed = unetFixedMapper(isDiffusersFormat ? .diffusers : .generativeModels)
            modelPrefix = "unet"
            modelPrefixFixed = "unet_fixed"
          case .svdI2v:
            let inputs: [DynamicGraph.Tensor<FloatType>] =
              [xTensor] + (tEmb.map { [$0] } ?? []) + cArr
            unet.compile(inputs: inputs)
            unetFixed.compile(inputs: crossattn)
            UNetMapping = unetMapper(isDiffusersFormat ? .diffusers : .generativeModels)
            UNetMappingFixed = unetFixedMapper(isDiffusersFormat ? .diffusers : .generativeModels)
            modelPrefix = "unet"
            modelPrefixFixed = "unet_fixed"
          case .wurstchenStageC:
            let inputs: [DynamicGraph.Tensor<FloatType>] =
              [xTensor] + (tEmb.map { [$0] } ?? []) + cArr
            unet.compile(inputs: inputs)
            unetFixed.compile(inputs: crossattn)
            UNetMapping = unetMapper(.generativeModels)
            UNetMappingFixed = unetFixedMapper(.generativeModels)
            modelPrefix = "stage_c"
            modelPrefixFixed = "stage_c_fixed"
          case .pixart, .sd3, .sd3Large, .flux1, .hunyuanVideo, .wan21_14b, .wan21_1_3b, .hiDreamI1:
            let inputs: [DynamicGraph.Tensor<FloatType>] =
              [xTensor] + (tEmb.map { [$0] } ?? []) + cArr
            unet.compile(inputs: inputs)
            unetFixed.compile(inputs: crossattn)
            UNetMapping = unetMapper(isDiffusersFormat ? .diffusers : .generativeModels)
            UNetMappingFixed = unetFixedMapper(isDiffusersFormat ? .diffusers : .generativeModels)
            modelPrefix = "dit"
            modelPrefixFixed = "dit"
          case .auraflow:
            fatalError()
          case .v1, .v2, .kandinsky21, .wurstchenStageB:
            fatalError()
          }
          func reverseMapping(original: ModelWeightMapping) -> [String: [String]] {
            var reversed: [String: [(Int, String)]] = [:]
            for (key, values) in original {
              for value in values {
                reversed[value, default: []].append((values.index, key))
              }
            }
            return reversed.mapValues {
              $0.sorted(by: { $0.0 < $1.0 }).map(\.1)
            }
          }
          let reverseUNetMapping = reverseMapping(original: UNetMapping)
          let reverseUNetMappingFixed = reverseMapping(original: UNetMappingFixed)
          try store.withTransaction {
            if let encoderHidProjWeightDescriptor = stateDict["encoder_hid_proj.weight"],
              let encoderHidProjBiasDescriptor = stateDict["encoder_hid_proj.bias"]
            {
              try archive.with(encoderHidProjWeightDescriptor) { tensor in
                let tensor = Tensor<FloatType>(from: tensor)
                store.write("__encoder_hid_proj__[t-0-0]", tensor: tensor)
              }
              try archive.with(encoderHidProjBiasDescriptor) { tensor in
                let tensor = Tensor<FloatType>(from: tensor)
                store.write("__encoder_hid_proj__[t-0-1]", tensor: tensor)
              }
            }
            var consumed = Set<String>()
            for (key, value) in UNetMapping.sorted(by: { $0.key < $1.key }) {
              guard let _ = stateDict[key], !consumed.contains(key) else {
                continue
              }
              let values = value.count == 1 ? (reverseUNetMapping[value[0]] ?? [key]) : [key]
              consumed.formUnion(values)
              let tensorDescriptors = values.compactMap { stateDict[$0] }
              try archive.with(tensorDescriptors) { tensors in
                guard !tensors.isEmpty else { return }
                let tensor: Tensor<FloatType>
                if tensors.count == 1 {
                  tensor = Tensor<FloatType>(from: tensors[0])
                } else {
                  let shape = [tensors.count] + Array(tensors[0].shape)
                  var combined = Tensor<FloatType>(.CPU, format: .NCHW, shape: TensorShape(shape))
                  for (i, tensor) in tensors.enumerated() {
                    let shape = tensor.shape
                    if shape.count == 4 {
                      combined[
                        i..<(i + 1), 0..<shape[0], 0..<shape[1], 0..<shape[2], 0..<shape[3]] =
                        Tensor<FloatType>(from: tensor).reshaped(
                          format: .NCHW, shape: TensorShape([1] + Array(shape)))
                    } else if shape.count == 3 {
                      combined[i..<(i + 1), 0..<shape[0], 0..<shape[1], 0..<shape[2]] = Tensor<
                        FloatType
                      >(from: tensor).reshaped(
                        format: .NCHW, shape: TensorShape([1] + Array(shape)))
                    } else if shape.count == 2 {
                      combined[i..<(i + 1), 0..<shape[0], 0..<shape[1]] = Tensor<FloatType>(
                        from: tensor
                      ).reshaped(format: .NCHW, shape: TensorShape([1] + Array(shape)))
                    } else if shape.count == 1 {
                      combined[i..<(i + 1), 0..<shape[0]] = Tensor<FloatType>(from: tensor)
                        .reshaped(format: .NCHW, shape: TensorShape([1] + Array(shape)))
                    }
                  }
                  tensor = combined
                }
                if value.count > 1 {
                  value.write(
                    graph: graph,
                    to: store, tensor: tensor, format: value.format, isDiagonalUp: false,
                    isDiagonalDown: false
                  ) {
                    let _ = interrupt()
                    return "__\(modelPrefix)__[\($0)]"
                  }
                } else if let name = value.first {
                  // For FLUX.1 model, we can inspect x_embedder for whether it is a inpainting, depth, or canny.
                  if name == "t-x_embedder-0-0" && modelVersion == .flux1 {
                    let shape = tensor.shape
                    if shape[1] == 384 {
                      // This is an inpainting model.
                      modifier = .inpainting
                    } else if shape[1] == 128 {
                      modifier = .depth  // We cannot differentiate depth or canny, assuming it is depth.
                    }
                    store.write("__\(modelPrefix)__[\(name)]", tensor: tensor)
                  } else if name.contains("time_mixer") {
                    var f32Tensor = Tensor<Float>(from: tensor)
                    // Apply sigmoid transformation.
                    f32Tensor[0] = 1.0 / (1.0 + expf(-f32Tensor[0]))
                    store.write(
                      "__\(modelPrefix)__[\(name)]", tensor: Tensor<FloatType>(from: f32Tensor))
                  } else {
                    value.write(
                      graph: graph,
                      to: store, tensor: tensor, format: value.format, isDiagonalUp: false,
                      isDiagonalDown: false
                    ) {
                      let _ = interrupt()
                      return "__\(modelPrefix)__[\($0)]"
                    }
                  }
                  let _ = interrupt()
                }
              }
            }
            for (key, value) in UNetMappingFixed.sorted(by: { $0.key < $1.key }) {
              guard let _ = stateDict[key], !consumed.contains(key) else {
                continue
              }
              let values = value.count == 1 ? (reverseUNetMappingFixed[value[0]] ?? [key]) : [key]
              consumed.formUnion(values)
              let tensorDescriptors = values.compactMap { stateDict[$0] }
              try archive.with(tensorDescriptors) { tensors in
                guard !tensors.isEmpty else { return }
                let tensor: Tensor<FloatType>
                if tensors.count == 1 {
                  tensor = Tensor<FloatType>(from: tensors[0])
                } else {
                  let shape = [tensors.count] + Array(tensors[0].shape)
                  var combined = Tensor<FloatType>(.CPU, format: .NCHW, shape: TensorShape(shape))
                  for (i, tensor) in tensors.enumerated() {
                    let shape = tensor.shape
                    if shape.count == 3 {
                      combined[
                        i..<(i + 1), 0..<shape[0], 0..<shape[1], 0..<shape[2], 0..<shape[3]] =
                        Tensor<FloatType>(from: tensor).reshaped(
                          format: .NCHW, shape: TensorShape([1] + Array(shape)))
                    } else if shape.count == 3 {
                      combined[i..<(i + 1), 0..<shape[0], 0..<shape[1], 0..<shape[2]] = Tensor<
                        FloatType
                      >(from: tensor).reshaped(
                        format: .NCHW, shape: TensorShape([1] + Array(shape)))
                    } else if shape.count == 2 {
                      combined[i..<(i + 1), 0..<shape[0], 0..<shape[1]] = Tensor<FloatType>(
                        from: tensor
                      ).reshaped(format: .NCHW, shape: TensorShape([1] + Array(shape)))
                    } else if shape.count == 1 {
                      combined[i..<(i + 1), 0..<shape[0]] = Tensor<FloatType>(from: tensor)
                        .reshaped(format: .NCHW, shape: TensorShape([1] + Array(shape)))
                    }
                  }
                  tensor = combined
                }
                if value.count > 1 {
                  value.write(
                    graph: graph,
                    to: store, tensor: tensor, format: value.format, isDiagonalUp: false,
                    isDiagonalDown: false
                  ) {
                    let _ = interrupt()
                    return "__\(modelPrefixFixed)__[\($0)]"
                  }
                } else if let name = value.first {
                  store.write("__\(modelPrefixFixed)__[\(name)]", tensor: tensor)
                  let _ = interrupt()
                }
              }
            }
          }
        }
      }
      try graph.openStore(filePath) {
        switch modelVersion {
        case .v1, .v2:
          if $0.keys.count != 718 {
            throw Error.tensorWritesFailed
          }
        case .sdxlBase:
          if $0.keys.count != 1820 + (hasEncoderHidProj ? 2 : 0) {
            throw Error.tensorWritesFailed
          }
        case .ssd1b:
          if $0.keys.count != 1012 + (hasEncoderHidProj ? 2 : 0) {
            throw Error.tensorWritesFailed
          }
        case .sdxlRefiner:
          if $0.keys.count != 1308 + (hasEncoderHidProj ? 2 : 0) {
            throw Error.tensorWritesFailed
          }
        case .svdI2v:
          if $0.keys.count != 1396 {
            throw Error.tensorWritesFailed
          }
        case .wurstchenStageC:
          // The first number is the number of tensors in stage c file, the second is the ones with effnet and previewer.
          if $0.keys.count != 1550 && $0.keys.count != 1550 + 374 {
            throw Error.tensorWritesFailed
          }
        case .pixart:
          if $0.keys.count != 754 {
            throw Error.tensorWritesFailed
          }
        case .sd3:
          if $0.keys.count != 1157 + (qkNorm ? 24 * 4 : 0) + dualAttentionLayers.count
            * (4 * 2 + 2 + 3 * 2)
          {
            throw Error.tensorWritesFailed
          }
        case .sd3Large:
          if $0.keys.count != 1981 {
            throw Error.tensorWritesFailed
          }
        case .flux1:
          let count = $0.keys.count
          if count != 1732 && count != 1728 && count != 1041 + max(distilledGuidanceLayers, 1) * 4 {
            throw Error.tensorWritesFailed
          }
        case .hunyuanVideo:
          if $0.keys.count != 1870 {
            throw Error.tensorWritesFailed
          }
        case .wan21_1_3b:
          let count = $0.keys.count
          if count != 986 && count != 1144 {
            throw Error.tensorWritesFailed
          }
        case .wan21_14b:
          let count = $0.keys.count
          if count != 1306 && count != 1514 {
            throw Error.tensorWritesFailed
          }
        case .hiDreamI1:
          if $0.keys.count != 1857 {
            throw Error.tensorWritesFailed
          }
        case .auraflow:
          fatalError()
        case .kandinsky21, .wurstchenStageB:
          fatalError()
        }
      }
      filePaths.append(filePath)
    }
    if let autoencoderFilePath = autoencoderFilePath {
      do {
        if autoencoderFilePath != filePath {
          // Redo the stateDict
          stateDict.removeAll()
          if let safeTensors = SafeTensors(url: URL(fileURLWithPath: autoencoderFilePath)) {
            archive = safeTensors
            let states = safeTensors.states
            stateDict = states
            for (key, value) in states {
              if key.hasPrefix("encoder.") || key.hasPrefix("decoder.")
                || key.hasPrefix("post_quant_conv.") || key.hasPrefix("quant_conv.")
              {
                stateDict["first_stage_model.\(key)"] = value
              }
            }
          } else if let zipArchive = Archive(
            url: URL(fileURLWithPath: autoencoderFilePath), accessMode: .read)
          {
            archive = zipArchive
            let rootObject = try Interpreter.unpickle(zip: zipArchive)
            guard
              let originalStateDict =
                (rootObject["state_dict"] as? Interpreter.Dictionary ?? rootObject["module"]
                  as? Interpreter.Dictionary)
            else {
              throw UnpickleError.dataNotFound
            }
            originalStateDict.forEach { key, value in
              guard let value = value as? TensorDescriptor else { return }
              stateDict[key] = value
              if key.hasPrefix("encoder.") || key.hasPrefix("decoder.")
                || key.hasPrefix("post_quant_conv.") || key.hasPrefix("quant_conv.")
              {
                stateDict["first_stage_model.\(key)"] = value
              }
            }
          } else {
            throw UnpickleError.dataNotFound
          }
        }
        try graph.withNoGrad {
          let encoderTensor = graph.variable(.CPU, .NHWC(1, 512, 512, 3), of: FloatType.self)
          let decoderTensor = graph.variable(.CPU, .NHWC(1, 64, 64, inputDim), of: FloatType.self)
          let (encoder, encoderReader, encoderWeightMapper) = Encoder(
            channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: 64,
            startHeight: 64, usesFlashAttention: false, format: .NHWC, outputChannels: inputDim)
          let (decoder, decoderReader, decoderWeightMapper) = Decoder(
            channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: 64,
            startHeight: 64, inputChannels: inputDim, highPrecisionKeysAndValues: false,
            usesFlashAttention: false, paddingFinalConvLayer: false, format: .NHWC)
          decoder.compile(inputs: decoderTensor)
          encoder.compile(inputs: encoderTensor)
          let filePath = ModelZoo.filePathForModelDownloaded("\(modelName)_vae_f16.ckpt")
          // This is the name of diffusers.
          if stateDict.keys.contains(where: { $0.contains(".mid_block.") }) {
            // For diffusers, we use the weight mapper.
            try graph.openStore(filePath) { store in
              store.removeAll()
              try store.withTransaction {
                let decoderMapping = decoderWeightMapper(.diffusers)
                for (key, value) in decoderMapping {
                  guard let tensorDescriptor = stateDict[key] else {
                    continue
                  }
                  try archive.with(tensorDescriptor) { tensor in
                    let tensor = Tensor<FloatType>(from: tensor)
                    if let name = value.first {
                      store.write("__decoder__[\(name)]", tensor: tensor)
                    }
                  }
                }
                let encoderMapping = encoderWeightMapper(.diffusers)
                for (key, value) in encoderMapping {
                  guard let tensorDescriptor = stateDict[key] else {
                    continue
                  }
                  try archive.with(tensorDescriptor) { tensor in
                    let tensor = Tensor<FloatType>(from: tensor)
                    if let name = value.first {
                      store.write("__encoder__[\(name)]", tensor: tensor)
                    }
                  }
                }
              }
            }
          } else {
            try decoderReader(stateDict, archive)
            try encoderReader(stateDict, archive)
            graph.openStore(filePath) {
              $0.removeAll()
              $0.write("decoder", model: decoder)
              $0.write("encoder", model: encoder)
            }
          }
          try graph.openStore(filePath) {
            if $0.keys.count != 248 {
              throw Error.tensorWritesFailed
            }
          }
          filePaths.append(filePath)
        }
      } catch {
        throw Error.autoencoder(error)
      }
    }
    return (filePaths, modelVersion, modifier, inspectionResult)
  }
}

extension ModelImporter {
  public static func merge(efficientNetAndPreviewer: String, into file: String) {
    let graph = DynamicGraph()
    graph.openStore(efficientNetAndPreviewer, flags: .readOnly) { fromStore in
      let keys = fromStore.keys.filter {
        $0.hasPrefix("__effnet__[") || $0.hasPrefix("__previewer__[")
      }
      graph.openStore(file, flags: .truncateWhenClose) { toStore in
        for key in keys {
          guard let tensor = fromStore.read(key, codec: [.q6p, .q8p, .ezm7, .externalData]) else {
            continue
          }
          toStore.write(key, tensor: tensor)
        }
      }
    }
  }
}

extension ModelImporter {

  public static func inferModelSpecification(
    modelName: String,
    fileName: String,
    fileNames: [String],
    modelVersion: ModelVersion,
    modifier: SamplerModifier,
    inspectionResult: ModelImporter.InspectionResult,
    prefix: String,
    objective: Denoiser.Objective?,
    conditioning: Denoiser.Conditioning?,
    noiseDiscretization: ModelZoo.NoiseDiscretization?,
    upcastAttention: Bool,
    finetuneScale: UInt16
  ) -> (
    specification: ModelZoo.Specification,
    additionalModels: [(name: String, subtitle: String, file: String)],
    wurstchenStageCEffNetAndPreviewer: String?
  ) {

    // Extract text encoders based on model version
    var clipEncoder: String? = nil
    var additionalClipEncoders: [String]? = nil
    var t5Encoder: String? = nil
    let textEncoder: String?

    switch modelVersion {
    case .v1:
      textEncoder = fileNames.first {
        $0.hasSuffix("_clip_vit_l14_f16.ckpt")
      }
    case .v2, .svdI2v:
      textEncoder = fileNames.first {
        $0.hasSuffix("_open_clip_vit_h14_f16.ckpt")
      }
    case .sdxlBase, .ssd1b:
      clipEncoder = fileNames.first {
        $0.hasSuffix("_clip_vit_l14_f16.ckpt")
      }
      textEncoder = fileNames.first {
        $0.hasSuffix("_open_clip_vit_bigg14_f16.ckpt")
      }
    case .sd3:
      clipEncoder = fileNames.first {
        $0.hasSuffix("_clip_vit_l14_f16.ckpt")
      }
      textEncoder = fileNames.first {
        $0.hasSuffix("_open_clip_vit_bigg14_f16.ckpt")
      }
    case .sd3Large:
      clipEncoder = fileNames.first {
        $0.hasSuffix("_clip_vit_l14_f16.ckpt")
      }
      textEncoder = fileNames.first {
        $0.hasSuffix("_open_clip_vit_bigg14_f16.ckpt")
      }
    case .pixart:
      textEncoder = fileNames.first {
        $0.hasSuffix("_t5_xxl_encoder_f16.ckpt")
      }
    case .auraflow:
      textEncoder = fileNames.first {
        $0.hasSuffix("_pile_t5_xl_encoder_f16.ckpt")
      }
    case .sdxlRefiner:
      textEncoder = fileNames.first {
        $0.hasSuffix("_open_clip_vit_bigg14_f16.ckpt")
      }
    case .flux1:
      textEncoder = fileNames.first {
        $0.hasSuffix("_t5_xxl_encoder_f16.ckpt")
      }
      clipEncoder = fileNames.first {
        $0.hasSuffix("_clip_vit_l14_f16.ckpt")
      }
    case .hunyuanVideo:
      textEncoder = fileNames.first {
        $0.hasSuffix("_llava_llama_3_8b_v1.1_q8p.ckpt")
      }
      clipEncoder = fileNames.first {
        $0.hasSuffix("_clip_vit_l14_f16.ckpt")
      }
    case .wan21_1_3b, .wan21_14b:
      textEncoder = fileNames.first {
        $0.hasSuffix("_umt5_xxl_encoder_q8p.ckpt")
      }
      clipEncoder = fileNames.first {
        $0.hasSuffix("_open_clip_xlm_roberta_large_vit_h14_f16.ckpt")
      }
    case .hiDreamI1:
      textEncoder = fileNames.first {
        $0.hasSuffix("_llama_3.1_8b_instruct_f16.ckpt")
      }
      clipEncoder = fileNames.first {
        $0.hasSuffix("_long_clip_vit_l14_f16.ckpt")
      }
      additionalClipEncoders =
        (fileNames.first {
          $0.hasSuffix("_long_open_clip_vit_bigg14_f16.ckpt")
        }).map { [$0] }
      t5Encoder = fileNames.first {
        $0.hasSuffix("_t5_xxl_encoder_f16.ckpt")
      }
    case .wurstchenStageC:
      textEncoder = nil
    case .kandinsky21, .wurstchenStageB:
      fatalError()
    }

    let autoencoder = fileNames.first { $0.hasSuffix("_vae_f16.ckpt") }

    // Create base specification
    var specification = ModelZoo.Specification(
      name: modelName, file: "\(fileName)_f16.ckpt", prefix: prefix, version: modelVersion,
      upcastAttention: upcastAttention, defaultScale: finetuneScale,
      textEncoder: textEncoder, autoencoder: autoencoder, modifier: modifier,
      clipEncoder: clipEncoder, additionalClipEncoders: additionalClipEncoders,
      t5Encoder: t5Encoder, conditioning: conditioning, objective: objective,
      noiseDiscretization: noiseDiscretization
    )

    // Apply version-specific configurations
    var wurstchenStageCEffNetAndPreviewer: String? = nil
    var additionalModels = [(name: String, subtitle: String, file: String)]()

    switch modelVersion {
    case .v1:
      break
    case .v2:
      if specification.textEncoder == nil {
        specification.textEncoder = "open_clip_vit_h14_f16.ckpt"
      }
    case .sdxlBase, .sdxlRefiner, .ssd1b:
      if specification.textEncoder == nil {
        specification.textEncoder = "open_clip_vit_bigg14_f16.ckpt"
      }
      if specification.clipEncoder == nil {
        specification.clipEncoder = "clip_vit_l14_f16.ckpt"
      }
      if specification.autoencoder == nil {
        specification.autoencoder = "sdxl_vae_v1.0_f16.ckpt"
      }
    case .svdI2v:
      if specification.textEncoder == nil {
        specification.textEncoder = "open_clip_vit_h14_vision_model_f16.ckpt"
      }
      if specification.clipEncoder == nil {
        specification.clipEncoder = "open_clip_vit_h14_visual_proj_f16.ckpt"
      }
      specification.conditioning = .noise
      specification.objective = .v
      specification.noiseDiscretization = .edm(.init(sigmaMax: 700.0))
    case .wurstchenStageC:
      if specification.textEncoder == nil {
        specification.textEncoder = "open_clip_vit_bigg14_f16.ckpt"
      }
      if specification.autoencoder == nil {
        specification.autoencoder = "wurstchen_3.0_stage_a_hq_f16.ckpt"
      }
      specification.stageModels = ["wurstchen_3.0_stage_b_q6p_q8p.ckpt"]
      if ModelZoo.isModelDownloaded("wurstchen_3.0_stage_c_f32_f16.ckpt") {
        wurstchenStageCEffNetAndPreviewer = "wurstchen_3.0_stage_c_f32_f16.ckpt"
      } else if ModelZoo.isModelDownloaded("wurstchen_3.0_stage_c_f32_q6p_q8p.ckpt") {
        wurstchenStageCEffNetAndPreviewer = "wurstchen_3.0_stage_c_f32_q6p_q8p.ckpt"
      } else {
        wurstchenStageCEffNetAndPreviewer = "wurstchen_3.0_stage_c_effnet_previewer_f32_f16.ckpt"
        additionalModels.append(
          (
            name: "Stable Cascade (Würstchen v3.0) EfficientNet and Previewer",
            subtitle: ModelZoo.humanReadableNameForVersion(.wurstchenStageC),
            file: "wurstchen_3.0_stage_c_effnet_previewer_f32_f16.ckpt"
          ))
      }
    case .pixart:
      if specification.textEncoder == nil {
        specification.textEncoder = "t5_xxl_encoder_q6p.ckpt"
      }
      if specification.autoencoder == nil {
        specification.autoencoder = "sdxl_vae_v1.0_f16.ckpt"
      }
      specification.noiseDiscretization = .ddpm(
        .init(linearStart: 0.0001, linearEnd: 0.02, timesteps: 1_000, linspace: .linearWrtBeta))
    case .auraflow:
      if specification.textEncoder == nil {
        specification.textEncoder = "pile_t5_xl_encoder_q8p.ckpt"
      }
      if specification.autoencoder == nil {
        specification.autoencoder = "sdxl_vae_v1.0_f16.ckpt"
      }
      specification.objective = .u(conditionScale: 1000)
      specification.noiseDiscretization = .rf(
        .init(sigmaMin: 0, sigmaMax: 1, conditionScale: 1_000))
    case .flux1:
      if specification.textEncoder == nil {
        specification.textEncoder = "t5_xxl_encoder_q6p.ckpt"
      }
      if specification.clipEncoder == nil {
        specification.clipEncoder = "clip_vit_l14_f16.ckpt"
      }
      if specification.autoencoder == nil {
        specification.autoencoder = "flux_1_vae_f16.ckpt"
      }
      specification.highPrecisionAutoencoder = true
      specification.objective = .u(conditionScale: 1000)
      specification.noiseDiscretization = .rf(
        .init(sigmaMin: 0, sigmaMax: 1, conditionScale: 1_000))
      if inspectionResult.hasGuidanceEmbed {
        specification.guidanceEmbed = true
      }
      // For FLUX.1, the hires fix trigger scale is 1.5 of the finetune scale.
      specification.hiresFixScale = (finetuneScale * 3 + 1) / 2
      if inspectionResult.distilledGuidanceLayers > 0 {
        specification.mmdit = ModelZoo.Specification.MMDiT(
          qkNorm: true, dualAttentionLayers: [],
          distilledGuidanceLayers: inspectionResult.distilledGuidanceLayers)
      }
    case .hunyuanVideo:
      if specification.textEncoder == nil {
        specification.textEncoder = "llava_llama_3_8b_v1.1_q8p.ckpt"
      }
      if specification.clipEncoder == nil {
        specification.clipEncoder = "clip_vit_l14_f16.ckpt"
      }
      if specification.autoencoder == nil {
        specification.autoencoder = "hunyuan_video_vae_f16.ckpt"
      }
      specification.objective = .u(conditionScale: 1000)
      specification.noiseDiscretization = .rf(
        .init(sigmaMin: 0, sigmaMax: 1, conditionScale: 1_000))
      specification.guidanceEmbed = true
      // For Hunyuan, the hires fix trigger scale is 1.5 of the finetune scale.
      specification.hiresFixScale = (finetuneScale * 3 + 1) / 2
    case .wan21_1_3b, .wan21_14b:
      if specification.textEncoder == nil {
        specification.textEncoder = "umt5_xxl_encoder_q8p.ckpt"
      }
      if modifier == .inpainting && specification.clipEncoder == nil {
        specification.clipEncoder = "open_clip_xlm_roberta_large_vit_h14_f16.ckpt"
      }
      if specification.autoencoder == nil {
        specification.autoencoder = "wan_v2.1_video_vae_f16.ckpt"
      }
      specification.objective = .u(conditionScale: 1000)
      specification.noiseDiscretization = .rf(
        .init(sigmaMin: 0, sigmaMax: 1, conditionScale: 1_000))
      // For Wan, the hires fix trigger scale is 1.5 of the finetune scale.
      specification.hiresFixScale = (finetuneScale * 3 + 1) / 2
    case .hiDreamI1:
      if specification.textEncoder == nil {
        specification.textEncoder = "llama_3.1_8b_instruct_q8p.ckpt"
      }
      if specification.clipEncoder == nil {
        specification.clipEncoder = "long_clip_vit_l14_f16.ckpt"
      }
      if specification.additionalClipEncoders == nil {
        specification.additionalClipEncoders = ["long_open_clip_vit_bigg14_f16.ckpt"]
      }
      if specification.t5Encoder == nil {
        specification.t5Encoder = "t5_xxl_encoder_q6p.ckpt"
      }
      if specification.autoencoder == nil {
        specification.autoencoder = "flux_1_vae_f16.ckpt"
      }
      specification.highPrecisionAutoencoder = true
      specification.objective = .u(conditionScale: 1000)
      specification.noiseDiscretization = .rf(
        .init(sigmaMin: 0, sigmaMax: 1, conditionScale: 1_000))
      // For HiDream, the hires fix trigger scale is 1.5 of the finetune scale.
      specification.hiresFixScale = (finetuneScale * 3 + 1) / 2
    case .sd3, .sd3Large:
      if specification.textEncoder == nil {
        specification.textEncoder = "open_clip_vit_bigg14_f16.ckpt"
      }
      if specification.clipEncoder == nil {
        specification.clipEncoder = "clip_vit_l14_f16.ckpt"
      }
      if specification.t5Encoder == nil {
        specification.t5Encoder = "t5_xxl_encoder_q6p.ckpt"
      }
      if specification.autoencoder == nil {
        specification.autoencoder = "sd3_vae_f16.ckpt"
      }
      specification.mmdit = ModelZoo.Specification.MMDiT(
        qkNorm: inspectionResult.qkNorm, dualAttentionLayers: inspectionResult.dualAttentionLayers
      )
      specification.objective = .u(conditionScale: 1000)
      specification.noiseDiscretization = .rf(
        .init(sigmaMin: 0, sigmaMax: 1, conditionScale: 1_000))
    case .kandinsky21, .wurstchenStageB:
      fatalError()
    }

    return (specification, additionalModels, wurstchenStageCEffNetAndPreviewer)
  }
}
