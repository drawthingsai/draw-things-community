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
  }
  private let filePath: String
  private let modelName: String
  private let isTextEncoderCustomized: Bool
  private let autoencoderFilePath: String?
  private var access: Int = 0
  private var expectedTotalAccess: Int = 0
  private var progress: ((Float) -> Void)? = nil
  public init(
    filePath: String, modelName: String, isTextEncoderCustomized: Bool, autoencoderFilePath: String?
  ) {
    self.filePath = filePath
    self.modelName = modelName
    self.isTextEncoderCustomized = isTextEncoderCustomized
    self.autoencoderFilePath = autoencoderFilePath
  }

  public func `import`(
    versionCheck: @escaping (ModelVersion) -> Void, progress: @escaping (Float) -> Void
  ) throws -> ([String], ModelVersion, SamplerModifier) {
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

  private func internalImport(versionCheck: @escaping (ModelVersion) -> Void) throws -> (
    [String], ModelVersion, SamplerModifier
  ) {
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
    guard
      let tokey = stateDict[
        "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight"]
        ?? stateDict["down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.weight"],
      let inputConv2d = stateDict["model.diffusion_model.input_blocks.0.0.weight"]
        ?? stateDict["conv_in.weight"]
    else {
      throw UnpickleError.tensorNotFound
    }
    let inputDim = inputConv2d.shape.count >= 2 ? inputConv2d.shape[1] : 4
    let modifier: SamplerModifier
    switch inputDim {
    case 9:
      modifier = .inpainting
    case 8:
      modifier = .editing
    case 5:
      modifier = .depth
    default:
      modifier = .none
    }
    let modelVersion: ModelVersion
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
      modelVersion = .v2
      expectedTotalAccess = 686
    case 768:
      modelVersion = .v1
      expectedTotalAccess = 686
    default:
      throw UnpickleError.tensorNotFound
    }
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
      case .kandinsky21, .svdI2v, .wurstchenStageC, .wurstchenStageB:
        fatalError()
      }
    }
    if autoencoderFilePath != nil {
      expectedTotalAccess += 248
    }
    versionCheck(modelVersion)
    progress?(0.05)
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
          var filePath: String
          var textStateDict = stateDict
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
          case .kandinsky21, .svdI2v, .wurstchenStageC, .wurstchenStageB:
            fatalError()
          }
          if modelVersion == .sdxlBase || modelVersion == .sdxlRefiner {
            for (key, value) in stateDict {
              if key.hasPrefix("conditioner.embedders.0.") {
                textStateDict["cond_stage_model." + key.dropFirst(24)] = value
              }
            }
          }
          textModel.compile(inputs: tokensTensor, positionTensor, causalAttentionMask)
          try textModelReader(textStateDict, archive)
          try graph.openStore(filePath) { store in
            store.removeAll()
            if let text_projection = textStateDict["cond_stage_model.model.text_projection"]
              ?? textStateDict["cond_stage_model.text_projection"]
            {
              try archive.with(text_projection) {
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
            case .kandinsky21, .svdI2v, .wurstchenStageC, .wurstchenStageB:
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
            textStateDict = stateDict
            for (key, value) in stateDict {
              if key.hasPrefix("conditioner.embedders.1.") {
                textStateDict["cond_stage_model." + key.dropFirst(24)] = value
              }
            }
            textModel.compile(inputs: tokensTensor, positionTensor, causalAttentionMask)
            try textModelReader(textStateDict, archive)
            try graph.openStore(filePath) { store in
              store.removeAll()
              if let text_projection = textStateDict["cond_stage_model.model.text_projection"]
                ?? textStateDict["cond_stage_model.text_projection"]
              {
                try archive.with(text_projection) {
                  store.write("text_projection", tensor: $0)
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
    switch modelVersion {
    case .v1:
      conditionalLength = 768
    case .v2:
      conditionalLength = 1024
    case .sdxlBase, .sdxlRefiner, .ssd1b:
      conditionalLength = 1280
    case .kandinsky21, .svdI2v, .wurstchenStageC, .wurstchenStageB:
      fatalError()
    }
    try graph.withNoGrad {
      let tEmb = graph.variable(
        Tensor<FloatType>(
          from: timeEmbedding(
            timestep: 981, batchSize: 2, embeddingSize: modelVersion == .sdxlRefiner ? 384 : 320,
            maxPeriod: 10_000)
        ))
      let xTensor = graph.variable(.CPU, .NHWC(2, 64, 64, inputDim), of: FloatType.self)
      let cTensor = graph.variable(.CPU, .HWC(2, 77, conditionalLength), of: FloatType.self)
      var cArr = [cTensor]
      let unet: Model
      let unetReader: PythonReader
      let unetMapper: ModelWeightMapper?
      let filePath = ModelZoo.filePathForModelDownloaded("\(self.modelName)_f16.ckpt")
      if modelVersion == .sdxlBase || modelVersion == .sdxlRefiner || modelVersion == .ssd1b {
        let fixedEncoder = UNetFixedEncoder<FloatType>(
          filePath: "", version: modelVersion, usesFlashAttention: false, zeroNegativePrompt: false)
        cArr.insert(
          graph.variable(.CPU, .HWC(2, 77, 768), of: FloatType.self),
          at: 0)
        cArr.append(
          graph.variable(.CPU, .WC(2, 1280), of: FloatType.self))
        for c in cArr {
          c.full(0)
        }
        let vector: DynamicGraph.Tensor<FloatType>
        if modelVersion == .sdxlBase || modelVersion == .ssd1b {
          vector = graph.variable(.CPU, .WC(2, 2816), of: FloatType.self)
        } else {
          vector = graph.variable(.CPU, .WC(2, 2560), of: FloatType.self)
        }
        // These values doesn't matter, it won't affect the model shape, just the input vector.
        cArr =
          [vector]
          + fixedEncoder.encode(
            textEncoding: cArr.map({ $0.toGPU(0) }), batchSize: 2, startHeight: 64, startWidth: 64,
            tokenLengthUncond: 77, tokenLengthCond: 77, lora: []
          ).0.map({ $0.toCPU() })
      }
      let unetFixed: Model?
      let unetFixedReader: PythonReader?
      let unetFixedMapper: ModelWeightMapper?
      switch modelVersion {
      case .v1:
        (unet, unetReader) = UNet(
          batchSize: 2, embeddingLength: (77, 77), startWidth: 64, startHeight: 64,
          usesFlashAttention: .none, injectControls: false, injectT2IAdapters: false,
          injectIPAdapterLengths: [])
        unetMapper = nil
        (unetFixed, unetFixedReader, unetFixedMapper) = (nil, nil, nil)
      case .v2:
        (unet, unetReader) = UNetv2(
          batchSize: 2, embeddingLength: (77, 77), startWidth: 64, startHeight: 64,
          upcastAttention: false, usesFlashAttention: .none, injectControls: false)
        unetMapper = nil
        (unetFixed, unetFixedReader, unetFixedMapper) = (nil, nil, nil)
      case .sdxlBase:
        (unet, unetReader, unetMapper) = UNetXL(
          batchSize: 2, startHeight: 64, startWidth: 64,
          channels: [320, 640, 1280], inputAttentionRes: [2: [2, 2], 4: [10, 10]],
          middleAttentionBlocks: 10, outputAttentionRes: [2: [2, 2, 2], 4: [10, 10, 10]],
          embeddingLength: (77, 77), injectIPAdapterLengths: [],
          upcastAttention: ([:], false, [:]), usesFlashAttention: .none, injectControls: false,
          isTemporalMixEnabled: false, of: FloatType.self)
        (unetFixed, unetFixedReader, unetFixedMapper) = UNetXLFixed(
          batchSize: 2, startHeight: 64, startWidth: 64, channels: [320, 640, 1280],
          embeddingLength: (77, 77), inputAttentionRes: [2: [2, 2], 4: [10, 10]],
          middleAttentionBlocks: 10, outputAttentionRes: [2: [2, 2, 2], 4: [10, 10, 10]],
          usesFlashAttention: .none, isTemporalMixEnabled: false)
      case .ssd1b:
        (unet, unetReader, unetMapper) = UNetXL(
          batchSize: 2, startHeight: 64, startWidth: 64,
          channels: [320, 640, 1280], inputAttentionRes: [2: [2, 2], 4: [4, 4]],
          middleAttentionBlocks: 0, outputAttentionRes: [2: [2, 1, 1], 4: [4, 4, 10]],
          embeddingLength: (77, 77), injectIPAdapterLengths: [],
          upcastAttention: ([:], false, [:]), usesFlashAttention: .none, injectControls: false,
          isTemporalMixEnabled: false, of: FloatType.self)
        (unetFixed, unetFixedReader, unetFixedMapper) = UNetXLFixed(
          batchSize: 2, startHeight: 64, startWidth: 64, channels: [320, 640, 1280],
          embeddingLength: (77, 77), inputAttentionRes: [2: [2, 2], 4: [4, 4]],
          middleAttentionBlocks: 0, outputAttentionRes: [2: [2, 1, 1], 4: [4, 4, 10]],
          usesFlashAttention: .none, isTemporalMixEnabled: false)
      case .sdxlRefiner:
        (unet, unetReader, unetMapper) =
          UNetXL(
            batchSize: 2, startHeight: 64, startWidth: 64,
            channels: [384, 768, 1536, 1536], inputAttentionRes: [2: [4, 4], 4: [4, 4]],
            middleAttentionBlocks: 4, outputAttentionRes: [2: [4, 4, 4], 4: [4, 4, 4]],
            embeddingLength: (77, 77), injectIPAdapterLengths: [],
            upcastAttention: ([:], false, [:]), usesFlashAttention: .none, injectControls: false,
            isTemporalMixEnabled: false, of: FloatType.self
          )
        (unetFixed, unetFixedReader, unetFixedMapper) = UNetXLFixed(
          batchSize: 2, startHeight: 64, startWidth: 64, channels: [384, 768, 1536, 1536],
          embeddingLength: (77, 77), inputAttentionRes: [2: [4, 4], 4: [4, 4]],
          middleAttentionBlocks: 4, outputAttentionRes: [2: [4, 4, 4], 4: [4, 4, 4]],
          usesFlashAttention: .none, isTemporalMixEnabled: false)
      case .kandinsky21, .svdI2v, .wurstchenStageC, .wurstchenStageB:
        fatalError()
      }
      let crossattn: DynamicGraph.Tensor<FloatType>?
      switch modelVersion {
      case .sdxlBase, .ssd1b:
        crossattn = graph.variable(.CPU, .HWC(2, 77, 2048), of: FloatType.self)
      case .sdxlRefiner:
        crossattn = graph.variable(.CPU, .HWC(2, 77, 1280), of: FloatType.self)
      case .v1, .v2, .kandinsky21, .svdI2v, .wurstchenStageC, .wurstchenStageB:
        crossattn = nil
      }
      let isDiffusersFormat = stateDict.keys.contains { $0.hasPrefix("mid_block.") }
      if !isDiffusersFormat, let unetFixed = unetFixed, let unetFixedReader = unetFixedReader,
        let crossattn = crossattn
      {
        unetFixed.compile(inputs: crossattn)
        try unetFixedReader(stateDict, archive)
        graph.openStore(filePath) {
          $0.removeAll()
          $0.write("unet_fixed", model: unetFixed)
        }
      }
      // In case it is not on high performance device and it is SDXL model, read the parameters directly from the mapping.
      if (modelVersion == .sdxlBase || modelVersion == .sdxlRefiner || modelVersion == .ssd1b)
        && (!DeviceCapability.isHighPerformance || isDiffusersFormat),
        let unetMapper = unetMapper, let unetFixedMapper = unetFixedMapper,
        let unetFixed = unetFixed, let crossattn = crossattn
      {
        try graph.openStore(filePath) { store in
          let UNetMapping: [String: [String]]
          let UNetMappingFixed: [String: [String]]
          switch modelVersion {
          case .sdxlBase:
            if isDiffusersFormat {
              unet.compile(inputs: [xTensor, tEmb] + cArr)
              unetFixed.compile(inputs: crossattn)
              UNetMapping = unetMapper(.diffusers)
              UNetMappingFixed = unetFixedMapper(.diffusers)
            } else {
              UNetMapping = StableDiffusionMapping.UNetXLBase
              UNetMappingFixed = StableDiffusionMapping.UNetXLBaseFixed
            }
          case .sdxlRefiner:
            if isDiffusersFormat {
              unet.compile(inputs: [xTensor, tEmb] + cArr)
              unetFixed.compile(inputs: crossattn)
              UNetMapping = unetMapper(.diffusers)
              UNetMappingFixed = unetFixedMapper(.diffusers)
            } else {
              UNetMapping = StableDiffusionMapping.UNetXLRefiner
              UNetMappingFixed = StableDiffusionMapping.UNetXLRefinerFixed
            }
          case .ssd1b:
            unet.compile(inputs: [xTensor, tEmb] + cArr)
            unetFixed.compile(inputs: crossattn)
            UNetMapping = unetMapper(isDiffusersFormat ? .diffusers : .generativeModels)
            UNetMappingFixed = unetFixedMapper(isDiffusersFormat ? .diffusers : .generativeModels)
          case .v1, .v2, .kandinsky21, .svdI2v, .wurstchenStageC, .wurstchenStageB:
            fatalError()
          }
          try store.withTransaction {
            for (key, value) in UNetMapping {
              guard let tensorDescriptor = stateDict[key] else {
                continue
              }
              try archive.with(tensorDescriptor) { tensor in
                let tensor = Tensor<FloatType>(from: tensor)
                if value.count > 1 {
                  let count = tensor.shape[0] / value.count
                  for (i, name) in value.enumerated() {
                    if tensor.shape.count > 1 {
                      store.write(
                        "__unet__[\(name)]",
                        tensor: tensor[(i * count)..<((i + 1) * count), 0..<tensor.shape[1]]
                          .copied())
                    } else {
                      store.write(
                        "__unet__[\(name)]",
                        tensor: tensor[(i * count)..<((i + 1) * count)].copied())
                    }
                  }
                } else if let name = value.first {
                  store.write("__unet__[\(name)]", tensor: tensor)
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
                  let count = tensor.shape[0] / value.count
                  for (i, name) in value.enumerated() {
                    if tensor.shape.count > 1 {
                      store.write(
                        "__unet_fixed__[\(name)]",
                        tensor: tensor[(i * count)..<((i + 1) * count), 0..<tensor.shape[1]]
                          .copied())
                    } else {
                      store.write(
                        "__unet_fixed__[\(name)]",
                        tensor: tensor[(i * count)..<((i + 1) * count)].copied())
                    }
                  }
                } else if let name = value.first {
                  store.write("__unet_fixed__[\(name)]", tensor: tensor)
                }
              }
            }
          }
        }
      } else {
        unet.compile(inputs: [xTensor, tEmb] + cArr)
        try unetReader(stateDict, archive)
        graph.openStore(filePath) {
          if unetFixed == nil && unetFixedReader == nil {
            $0.removeAll()
          }
          $0.write("unet", model: unet)
        }
      }
      try graph.openStore(filePath) {
        switch modelVersion {
        case .v1, .v2:
          if $0.keys.count != 718 {
            throw Error.tensorWritesFailed
          }
        case .sdxlBase:
          if $0.keys.count != 1820 {
            throw Error.tensorWritesFailed
          }
        case .ssd1b:
          if $0.keys.count != 1012 {
            throw Error.tensorWritesFailed
          }
        case .sdxlRefiner:
          if $0.keys.count != 1308 {
            throw Error.tensorWritesFailed
          }
        case .kandinsky21, .svdI2v, .wurstchenStageC, .wurstchenStageB:
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
            startHeight: 64, usesFlashAttention: false, paddingFinalConvLayer: false)
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
    return (filePaths, modelVersion, modifier)
  }
}
