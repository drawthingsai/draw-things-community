import DataModels
import Diffusion
import Fickling
import Foundation
import ModelZoo
import NNC
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
    public init(
      version: ModelVersion, archive: TensorArchive, stateDict: [String: TensorDescriptor],
      modifier: SamplerModifier, inputChannels: Int, isDiffusersFormat: Bool,
      hasEncoderHidProj: Bool, hasGuidanceEmbed: Bool, numberOfTensors: Int
    ) {
      self.version = version
      self.archive = archive
      self.stateDict = stateDict
      self.modifier = modifier
      self.inputChannels = inputChannels
      self.isDiffusersFormat = isDiffusersFormat
      self.hasEncoderHidProj = hasEncoderHidProj
      self.hasGuidanceEmbed = hasGuidanceEmbed
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
      let originalStateDict = rootObject["state_dict"] as? Interpreter.Dictionary ?? rootObject
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
    var isPixArtSigmaXL = stateDict.keys.contains {
      ($0.contains("blocks.27.") || $0.contains("transformer_blocks.27."))
        && !($0.contains("single_transformer_blocks.27.")) && !($0.contains("single_blocks.27."))
    }
    var isSD3 = stateDict.keys.contains {
      $0.contains("joint_blocks.23.context_block.")
        || $0.contains("transformer_blocks.22.ff_context.")
    }
    var isFlux1 = stateDict.keys.contains {
      $0.contains("double_blocks.18.img_attn.qkv.")
        || $0.contains("single_transformer_blocks.37.")
    }
    let modifier: SamplerModifier
    let modelVersion: ModelVersion
    let inputDim: Int
    let isDiffusersFormat: Bool
    let expectedTotalAccess: Int
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
      isSD3 = false
      isFlux1 = false
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
    } else if isSD3 {
      modelVersion = .sd3
      modifier = .none
      inputDim = 16
      expectedTotalAccess = 1157
      isDiffusersFormat = stateDict.keys.contains {
        $0.contains("transformer_blocks.22.ff_context.")
      }
    } else if isFlux1 {
      modelVersion = .flux1
      modifier = .none
      inputDim = 16
      expectedTotalAccess = 1732
      isDiffusersFormat = stateDict.keys.contains {
        $0.contains("single_transformer_blocks.37.")
      }
    } else {
      throw UnpickleError.tensorNotFound
    }
    let hasEncoderHidProj = stateDict.keys.contains { $0 == "encoder_hid_proj.weight" }
    let hasGuidanceEmbed = stateDict.keys.contains {
      $0.contains(".guidance_embedder.") || $0.contains("guidance_in.")
    }
    return InspectionResult(
      version: modelVersion, archive: archive, stateDict: stateDict, modifier: modifier,
      inputChannels: inputDim, isDiffusersFormat: isDiffusersFormat,
      hasEncoderHidProj: hasEncoderHidProj, hasGuidanceEmbed: hasGuidanceEmbed,
      numberOfTensors: expectedTotalAccess)
  }

  private func internalImport(versionCheck: @escaping (ModelVersion) -> Void) throws -> (
    [String], ModelVersion, SamplerModifier, InspectionResult
  ) {
    let inspectionResult = try inspect()
    var archive = inspectionResult.archive
    var stateDict = inspectionResult.stateDict
    let modifier = inspectionResult.modifier
    let modelVersion = inspectionResult.version
    let inputDim = inspectionResult.inputChannels
    let isDiffusersFormat = inspectionResult.isDiffusersFormat
    let hasEncoderHidProj = inspectionResult.hasEncoderHidProj
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
      case .sd3, .pixart, .auraflow, .flux1:
        throw Error.noTextEncoder
      case .svdI2v:
        throw Error.noTextEncoder
      case .wurstchenStageC, .wurstchenStageB:
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
                rootObject["state_dict"] as? Interpreter.Dictionary ?? rootObject
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
          case .sd3, .pixart, .auraflow, .flux1, .kandinsky21, .svdI2v, .wurstchenStageC,
            .wurstchenStageB:
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
            case .sd3, .pixart, .auraflow, .flux1, .kandinsky21, .svdI2v, .wurstchenStageC,
              .wurstchenStageB:
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
                  rootObject["state_dict"] as? Interpreter.Dictionary ?? rootObject
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
    case .sd3:
      conditionalLength = 4096
      batchSize = 2
    case .auraflow:
      conditionalLength = 2048
      batchSize = 2
    case .flux1:
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
        .sd3, .auraflow:
        let fixedEncoder = UNetFixedEncoder<FloatType>(
          filePath: "", version: modelVersion, usesFlashAttention: false, zeroNegativePrompt: false,
          is8BitModel: false, canRunLoRASeparately: false, externalOnDemand: false)
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
        case .wurstchenStageC, .wurstchenStageB, .pixart, .sd3, .auraflow, .flux1:
          vectors = []
        case .kandinsky21, .v1, .v2:
          fatalError()
        }
        // These values doesn't matter, it won't affect the model shape, just the input vector.
        cArr =
          vectors
          + fixedEncoder.encode(
            isCfgEnabled: true, textGuidanceScale: 3.5, guidanceEmbed: 3.5,
            isGuidanceEmbedEnabled: false,
            textEncoding: cArr.map({ $0.toGPU(0) }), timesteps: [0], batchSize: batchSize,
            startHeight: 64, startWidth: 64,
            tokenLengthUncond: 77, tokenLengthCond: 77, lora: [],
            tiledDiffusion: TiledConfiguration(
              isEnabled: false, tileSize: .init(width: 0, height: 0), tileOverlap: 0)
          ).0.map({ $0.toCPU() })
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
            batchSize: (1, 1), channels: 3072, layers: (19, 38), guidanceEmbed: true
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
          injectIPAdapterLengths: [], injectAttentionKV: false, outputSpatialAttnInput: false)
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
          usesFlashAttention: .none, of: FloatType.self)
        unetReader = nil
        (unetFixedMapper, unetFixed) = MMDiTFixed(batchSize: batchSize, channels: 1536, layers: 24)
      case .flux1:
        (unetMapper, unet) = Flux1(
          batchSize: batchSize, tokenLength: 256, height: 64, width: 64, channels: 3072,
          layers: (19, 38),
          usesFlashAttention: .scaleMerged)
        (unetFixedMapper, unetFixed) = Flux1Fixed(
          batchSize: (batchSize, batchSize), channels: 3072, layers: (19, 38), guidanceEmbed: true)
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
      case .sd3:
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
        crossattn = [
          graph.variable(.CPU, .HWC(batchSize, 256, 4096), of: FloatType.self),
          graph.variable(.CPU, .WC(batchSize, 256), of: FloatType.self),
          graph.variable(.CPU, .WC(batchSize, 768), of: FloatType.self),
          graph.variable(.CPU, .WC(batchSize, 256), of: FloatType.self),
        ]
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
      } else if modelVersion == .sd3 {
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
          case .pixart:
            let inputs: [DynamicGraph.Tensor<FloatType>] =
              [xTensor] + (tEmb.map { [$0] } ?? []) + cArr
            unet.compile(inputs: inputs)
            unetFixed.compile(inputs: crossattn)
            UNetMapping = unetMapper(isDiffusersFormat ? .diffusers : .generativeModels)
            UNetMappingFixed = unetFixedMapper(isDiffusersFormat ? .diffusers : .generativeModels)
            modelPrefix = "dit"
            modelPrefixFixed = "dit"
          case .sd3:
            let inputs: [DynamicGraph.Tensor<FloatType>] =
              [xTensor] + (tEmb.map { [$0] } ?? []) + cArr
            unet.compile(inputs: inputs)
            unetFixed.compile(inputs: crossattn)
            UNetMapping = unetMapper(isDiffusersFormat ? .diffusers : .generativeModels)
            UNetMappingFixed = unetFixedMapper(isDiffusersFormat ? .diffusers : .generativeModels)
            modelPrefix = "dit"
            modelPrefixFixed = "dit"
          case .flux1:
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
            for (key, value) in UNetMapping {
              guard let tensorDescriptor = stateDict[key] else {
                continue
              }
              try archive.with(tensorDescriptor) { tensor in
                if value.count > 1 {
                  let tensor = Tensor<FloatType>(from: tensor)
                  value.write(to: store, tensor: tensor, format: value.format, isDiagonal: false) {
                    let _ = interrupt()
                    return "__\(modelPrefix)__[\($0)]"
                  }
                } else if let name = value.first {
                  if name.contains("time_mixer") {
                    var f32Tensor = Tensor<Float>(from: tensor)
                    // Apply sigmoid transformation.
                    f32Tensor[0] = 1.0 / (1.0 + expf(-f32Tensor[0]))
                    store.write(
                      "__\(modelPrefix)__[\(name)]", tensor: Tensor<FloatType>(from: f32Tensor))
                  } else {
                    let tensor = Tensor<FloatType>(from: tensor)
                    store.write("__\(modelPrefix)__[\(name)]", tensor: tensor)
                  }
                  let _ = interrupt()
                }
              }
            }
            for (key, value) in UNetMappingFixed {
              guard let tensorDescriptor = stateDict[key] else {
                continue
              }
              try archive.with(tensorDescriptor) { tensor in
                let tensor = Tensor<FloatType>(from: tensor)
                if value.count > 1 {
                  value.write(to: store, tensor: tensor, format: value.format, isDiagonal: false) {
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
          if $0.keys.count != 1157 {
            throw Error.tensorWritesFailed
          }
        case .flux1:
          if $0.keys.count != 1732 && $0.keys.count != 1728 {
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
            guard let originalStateDict = rootObject["state_dict"] as? Interpreter.Dictionary else {
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
          let decoderTensor = graph.variable(.CPU, .NHWC(1, 64, 64, 4), of: FloatType.self)
          let (encoder, encoderReader, encoderWeightMapper) = Encoder(
            channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: 64,
            startHeight: 64, usesFlashAttention: false)
          let (decoder, decoderReader, decoderWeightMapper) = Decoder(
            channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: 64,
            startHeight: 64, highPrecisionKeysAndValues: false, usesFlashAttention: false,
            paddingFinalConvLayer: false)
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
