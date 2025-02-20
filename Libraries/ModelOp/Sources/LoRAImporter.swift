import Diffusion
import Fickling
import Foundation
import ModelZoo
import NNC
import ZIPFoundation

public enum LoRAImporter {
  public enum Error: Swift.Error {
    case modelVersionFailed
  }
  public static func modelWeightsMapping(
    by version: ModelVersion, qkNorm: Bool, dualAttentionLayers: [Int], format: [ModelWeightFormat]
  ) -> (ModelWeightMapping, ModelWeightMapping) {
    let graph = DynamicGraph()
    var UNetMapping = ModelWeightMapping()
    var UNetMappingFixed = ModelWeightMapping()
    guard version != .v1 && version != .v2 else {
      // Legacy compatibility.
      UNetMapping = StableDiffusionMapping.UNet
      let UNetMappingKeys = UNetMapping.keys
      if format.contains(.diffusers) {
        for key in UNetMappingKeys {
          var diffusersKey = key
          for name in DiffusersMapping.UNetPartials {
            if key.contains(name.0) {
              diffusersKey = key.replacingOccurrences(of: name.0, with: name.1)
              break
            }
          }
          if diffusersKey.contains(".resnets.") {
            for name in DiffusersMapping.ResNetsPartials {
              if diffusersKey.contains(name.0) {
                diffusersKey = diffusersKey.replacingOccurrences(of: name.0, with: name.1)
                break
              }
            }
          }
          if diffusersKey != key {
            UNetMapping[diffusersKey] = UNetMapping[key]
          }
        }
      }
      return (UNetMappingFixed, UNetMapping)
    }
    let unet: Model
    let unetFixed: Model
    let unetMapper: ModelWeightMapper
    let unetFixedMapper: ModelWeightMapper
    switch version {
    case .sdxlBase:
      UNetMapping = StableDiffusionMapping.UNetXLBase
      UNetMappingFixed = StableDiffusionMapping.UNetXLBaseFixed
      (unet, _, unetMapper) = UNetXL(
        batchSize: 2, startHeight: 64, startWidth: 64,
        channels: [320, 640, 1280], inputAttentionRes: [2: [2, 2], 4: [10, 10]],
        middleAttentionBlocks: 10, outputAttentionRes: [2: [2, 2, 2], 4: [10, 10, 10]],
        embeddingLength: (77, 77), injectIPAdapterLengths: [],
        upcastAttention: ([:], false, [:]), usesFlashAttention: .none, injectControls: false,
        isTemporalMixEnabled: false, of: FloatType.self)
      (unetFixed, _, unetFixedMapper) = UNetXLFixed(
        batchSize: 2, startHeight: 64, startWidth: 64, channels: [320, 640, 1280],
        embeddingLength: (77, 77), inputAttentionRes: [2: [2, 2], 4: [10, 10]],
        middleAttentionBlocks: 10, outputAttentionRes: [2: [2, 2, 2], 4: [10, 10, 10]],
        usesFlashAttention: .none, isTemporalMixEnabled: false)
    case .sdxlRefiner:
      UNetMapping = StableDiffusionMapping.UNetXLRefiner
      UNetMappingFixed = StableDiffusionMapping.UNetXLRefinerFixed
      (unet, _, unetMapper) =
        UNetXL(
          batchSize: 2, startHeight: 64, startWidth: 64,
          channels: [384, 768, 1536, 1536], inputAttentionRes: [2: [4, 4], 4: [4, 4]],
          middleAttentionBlocks: 4, outputAttentionRes: [2: [4, 4, 4], 4: [4, 4, 4]],
          embeddingLength: (77, 77), injectIPAdapterLengths: [],
          upcastAttention: ([:], false, [:]), usesFlashAttention: .none, injectControls: false,
          isTemporalMixEnabled: false, of: FloatType.self
        )
      (unetFixed, _, unetFixedMapper) = UNetXLFixed(
        batchSize: 2, startHeight: 64, startWidth: 64, channels: [384, 768, 1536, 1536],
        embeddingLength: (77, 77), inputAttentionRes: [2: [4, 4], 4: [4, 4]],
        middleAttentionBlocks: 4, outputAttentionRes: [2: [4, 4, 4], 4: [4, 4, 4]],
        usesFlashAttention: .none, isTemporalMixEnabled: false)
    case .sd3:
      (unetMapper, unet) = MMDiT(
        batchSize: 2, t: 77, height: 64, width: 64, channels: 1536, layers: 24,
        upcast: false, qkNorm: qkNorm, dualAttentionLayers: dualAttentionLayers,
        posEmbedMaxSize: 192 /* This value doesn't matter */,
        usesFlashAttention: .none, of: FloatType.self)
      (unetFixedMapper, unetFixed) = MMDiTFixed(
        batchSize: 2, channels: 1536, layers: 24, dualAttentionLayers: dualAttentionLayers)
    case .sd3Large:
      (unetMapper, unet) = MMDiT(
        batchSize: 2, t: 77, height: 64, width: 64, channels: 2432, layers: 38,
        upcast: true, qkNorm: true, dualAttentionLayers: [], posEmbedMaxSize: 192,
        usesFlashAttention: .none, of: FloatType.self)
      (unetFixedMapper, unetFixed) = MMDiTFixed(
        batchSize: 2, channels: 2432, layers: 38, dualAttentionLayers: [])
    case .pixart:
      (unetMapper, unet) = PixArt(
        batchSize: 2, height: 64, width: 64, channels: 1152, layers: 28,
        tokenLength: (77, 77), usesFlashAttention: false, of: FloatType.self)
      (unetFixedMapper, unetFixed) = PixArtFixed(
        batchSize: 2, channels: 1152, layers: 28, tokenLength: (77, 77),
        usesFlashAttention: false, of: FloatType.self)
    case .flux1:
      (unetMapper, unet) = Flux1(
        batchSize: 1, tokenLength: 256, height: 64, width: 64, channels: 3072, layers: (19, 38),
        usesFlashAttention: .scaleMerged, contextPreloaded: true, injectControls: false,
        injectIPAdapterLengths: [:])
      (unetFixedMapper, unetFixed) = Flux1Fixed(
        batchSize: (1, 1), channels: 3072, layers: (19, 38), contextPreloaded: true,
        guidanceEmbed: true)
    case .hunyuanVideo:
      (unetMapper, unet) = Hunyuan(
        time: 1, height: 64, width: 64, textLength: 20, channels: 3072, layers: (20, 40),
        usesFlashAttention: .scaleMerged)
      (unetFixedMapper, unetFixed) = HunyuanFixed(
        timesteps: 1, channels: 3072, layers: (20, 40), textLength: (0, 20))
    case .auraflow:
      fatalError()
    case .v1, .v2, .kandinsky21, .svdI2v, .wurstchenStageC, .wurstchenStageB:
      fatalError()
    case .ssd1b:
      (unet, _, unetMapper) = UNetXL(
        batchSize: 2, startHeight: 64, startWidth: 64,
        channels: [320, 640, 1280], inputAttentionRes: [2: [2, 2], 4: [4, 4]],
        middleAttentionBlocks: 0, outputAttentionRes: [2: [2, 1, 1], 4: [4, 4, 10]],
        embeddingLength: (77, 77), injectIPAdapterLengths: [],
        upcastAttention: ([:], false, [:]), usesFlashAttention: .none, injectControls: false,
        isTemporalMixEnabled: false, of: FloatType.self)
      (unetFixed, _, unetFixedMapper) = UNetXLFixed(
        batchSize: 2, startHeight: 64, startWidth: 64, channels: [320, 640, 1280],
        embeddingLength: (77, 77), inputAttentionRes: [2: [2, 2], 4: [4, 4]],
        middleAttentionBlocks: 0, outputAttentionRes: [2: [2, 1, 1], 4: [4, 4, 10]],
        usesFlashAttention: .none, isTemporalMixEnabled: false)
    }
    return graph.withNoGrad {
      let inputDim: Int
      let conditionalLength: Int
      switch version {
      case .v1:
        inputDim = 4
        conditionalLength = 768
      case .v2:
        inputDim = 4
        conditionalLength = 1024
      case .sdxlBase, .sdxlRefiner, .ssd1b:
        inputDim = 4
        conditionalLength = 1280
      case .sd3, .sd3Large:
        inputDim = 16
        conditionalLength = 4096
      case .pixart:
        inputDim = 4
        conditionalLength = 4096
      case .auraflow:
        inputDim = 4
        conditionalLength = 2048
      case .flux1:
        inputDim = 16
        conditionalLength = 4096
      case .hunyuanVideo:
        inputDim = 16
        conditionalLength = 4096
      case .kandinsky21, .svdI2v, .wurstchenStageC, .wurstchenStageB:
        fatalError()
      }
      let crossattn: [DynamicGraph.Tensor<FloatType>]
      let tEmb: DynamicGraph.Tensor<FloatType>?
      let isCfgEnabled: Bool
      let isGuidanceEmbedEnabled: Bool
      switch version {
      case .sdxlBase, .ssd1b:
        isCfgEnabled = true
        isGuidanceEmbedEnabled = false
        crossattn = [graph.variable(.CPU, .HWC(2, 77, 2048), of: FloatType.self)]
        tEmb = graph.variable(
          Tensor<FloatType>(
            from: timeEmbedding(
              timestep: 981, batchSize: 2, embeddingSize: 320,
              maxPeriod: 10_000)
          ))
      case .sdxlRefiner:
        isCfgEnabled = true
        isGuidanceEmbedEnabled = false
        crossattn = [graph.variable(.CPU, .HWC(2, 77, 1280), of: FloatType.self)]
        tEmb = graph.variable(
          Tensor<FloatType>(
            from: timeEmbedding(
              timestep: 981, batchSize: 2, embeddingSize: 384,
              maxPeriod: 10_000)
          ))
      case .pixart:
        isCfgEnabled = true
        isGuidanceEmbedEnabled = false
        crossattn = [
          graph.variable(
            Tensor<FloatType>(
              from: timeEmbedding(
                timestep: 1000, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000)
            ).reshaped(.HWC(1, 1, 256))
          ), graph.variable(.CPU, .HWC(2, 77, 4096), of: FloatType.self),
        ]
        tEmb = nil
      case .sd3, .sd3Large:
        isCfgEnabled = true
        isGuidanceEmbedEnabled = false
        crossattn = [
          graph.variable(.CPU, .WC(2, 2048), of: FloatType.self),
          graph.variable(
            Tensor<FloatType>(
              from: timeEmbedding(
                timestep: 1000, batchSize: 2, embeddingSize: 256, maxPeriod: 10_000)
            ).reshaped(.WC(2, 256))
          ), graph.variable(.CPU, .HWC(2, 154, 4096), of: FloatType.self),
        ]
        tEmb = nil
      case .flux1:
        isCfgEnabled = false
        isGuidanceEmbedEnabled = true
        crossattn = [
          graph.variable(.CPU, .HWC(1, 256, 4096), of: FloatType.self),
          graph.variable(.CPU, .WC(1, 256), of: FloatType.self),
          graph.variable(.CPU, .WC(1, 768), of: FloatType.self),
          graph.variable(.CPU, .WC(1, 256), of: FloatType.self),
        ]
        tEmb = nil
      case .hunyuanVideo:
        isCfgEnabled = false
        isGuidanceEmbedEnabled = true
        crossattn = [
          graph.variable(.CPU, .HWC(1, 20, 4096), of: FloatType.self),
          graph.variable(.CPU, .WC(1, 256), of: FloatType.self),
          graph.variable(.CPU, .WC(1, 768), of: FloatType.self),
          graph.variable(.CPU, .WC(1, 256), of: FloatType.self),
        ]
        tEmb = nil
      case .auraflow:
        fatalError()
      case .v1, .v2, .kandinsky21, .svdI2v, .wurstchenStageC, .wurstchenStageB:
        fatalError()
      }
      unetFixed.compile(inputs: crossattn)
      let xTensor = graph.variable(
        .CPU, .NHWC(isCfgEnabled ? 2 : 1, 64, 64, inputDim), of: FloatType.self)
      var cArr: [DynamicGraph.Tensor<FloatType>]
      if version == .flux1 {  // This logic should be switch and for auraflow too.
        cArr = [
          graph.variable(.CPU, .HWC(1, 256, 4096), of: FloatType.self),
          graph.variable(.CPU, .WC(1, 768), of: FloatType.self),
        ]
      } else {
        let cTensor = graph.variable(
          .CPU, .HWC(isCfgEnabled ? 2 : 1, 77, conditionalLength), of: FloatType.self)
        cArr = [cTensor]
        cArr.insert(
          graph.variable(.CPU, .HWC(isCfgEnabled ? 2 : 1, 77, 768), of: FloatType.self),
          at: 0)
        cArr.append(
          graph.variable(.CPU, .WC(isCfgEnabled ? 2 : 1, 1280), of: FloatType.self))
      }
      let fixedEncoder = UNetFixedEncoder<FloatType>(
        filePath: "", version: version, dualAttentionLayers: dualAttentionLayers,
        usesFlashAttention: false,
        zeroNegativePrompt: false, isQuantizedModel: false, canRunLoRASeparately: false,
        externalOnDemand: false)
      for c in cArr {
        c.full(0)
      }
      // These values doesn't matter, it won't affect the model shape, just the input vector.
      let vectors: [DynamicGraph.Tensor<FloatType>]
      switch version {
      case .sdxlBase, .ssd1b:
        vectors = [graph.variable(.CPU, .WC(2, 2816), of: FloatType.self)]
      case .sdxlRefiner:
        vectors = [graph.variable(.CPU, .WC(2, 2560), of: FloatType.self)]
      case .svdI2v:
        vectors = [graph.variable(.CPU, .WC(2, 768), of: FloatType.self)]
      case .wurstchenStageC, .wurstchenStageB, .pixart, .sd3, .sd3Large, .auraflow, .flux1,
        .hunyuanVideo:
        vectors = []
      case .kandinsky21, .v1, .v2:
        fatalError()
      }
      switch version {
      case .sdxlBase, .ssd1b, .sdxlRefiner, .svdI2v, .wurstchenStageC, .wurstchenStageB, .pixart,
        .sd3, .sd3Large, .auraflow:
        // These values doesn't matter, it won't affect the model shape, just the input vector.
        cArr =
          vectors
          + fixedEncoder.encode(
            isCfgEnabled: isCfgEnabled, textGuidanceScale: 3.5, guidanceEmbed: 3.5,
            isGuidanceEmbedEnabled: isGuidanceEmbedEnabled,
            textEncoding: cArr.map({ $0.toGPU(0) }), timesteps: [0],
            batchSize: isCfgEnabled ? 2 : 1, startHeight: 64,
            startWidth: 64,
            tokenLengthUncond: 77, tokenLengthCond: 77, lora: [],
            tiledDiffusion: TiledConfiguration(
              isEnabled: false, tileSize: .init(width: 0, height: 0), tileOverlap: 0),
            injectedControls: []
          ).0.map({ $0.toCPU() })
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
            batchSize: 1, channels: 3072, layers: (20, 40), textLength: 20
          ).map {
            graph.variable(.CPU, format: .NHWC, shape: $0, of: FloatType.self)
          }
      case .kandinsky21, .v1, .v2:
        fatalError()
      }
      let inputs: [DynamicGraph.Tensor<FloatType>] = [xTensor] + (tEmb.map { [$0] } ?? []) + cArr
      unet.compile(inputs: inputs)
      // Otherwise we use our fixed dictionary.
      if version != .sdxlBase && version != .sdxlRefiner {
        let mappingFixed = unetFixedMapper(.generativeModels)
        let mapping = unetMapper(.generativeModels)
        UNetMappingFixed = [:]
        for (key, value) in mappingFixed {
          guard !key.hasPrefix("diffusion_model.") else {
            UNetMappingFixed[key] = value
            continue
          }
          UNetMappingFixed["diffusion_model.\(key)"] = value
        }
        UNetMapping = [:]
        for (key, value) in mapping {
          guard !key.hasPrefix("diffusion_model.") else {
            UNetMapping[key] = value
            continue
          }
          UNetMapping["diffusion_model.\(key)"] = value
        }
      }
      if format.contains(.diffusers) {
        let diffusersUNetMappingFixed = unetFixedMapper(.diffusers)
        let diffusersUNetMapping = unetMapper(.diffusers)
        if !format.contains(.generativeModels) {
          // Remove generativeModels ones.
          UNetMappingFixed = diffusersUNetMappingFixed
          UNetMapping = diffusersUNetMapping
        } else {
          UNetMappingFixed.merge(diffusersUNetMappingFixed) { v, _ in v }
          UNetMapping.merge(diffusersUNetMapping) { v, _ in v }
        }
      }
      return (UNetMappingFixed, UNetMapping)
    }
  }

  private static func isDiagonalUp(_ tensor: Tensor<FloatType>, weights: ModelWeightElement) -> Bool
  {
    // Check if this tensor is diagonal. If it is, marking it and prepare to slice the corresponding lora_down.weight.
    // First, check if the shape is right.
    guard weights.format == .O else { return false }
    let shape = tensor.shape
    var isDiagonal = false
    if shape.count == 2 && (shape[1] % weights.count) == 0 {
      let estimatedInputShape0 = shape[0] / weights.count
      let inputShape1 = shape[1] / weights.count
      isDiagonal = true
      for i in 0..<weights.count {
        let startOffset = weights.offsets?[i] ?? i * estimatedInputShape0
        let endOffset =
          i < weights.count - 1
          ? (weights.offsets?[i + 1] ?? (i + 1) * estimatedInputShape0) : shape[0]
        for j in 0..<weights.count {
          // Now, only diag should have any value.
          guard i != j else { continue }
          let offDiagTensor = tensor[
            startOffset..<endOffset,
            (j * inputShape1)..<((j + 1) * inputShape1)
          ].copied()
          isDiagonal = offDiagTensor.withUnsafeBytes {
            guard let ptr = $0.baseAddress else { return false }
            let val = ptr.assumingMemoryBound(to: UInt8.self)
            for k in 0..<$0.count {
              if val[k] != 0 {
                return false
              }
            }
            return true
          }
          if !isDiagonal {
            break
          }
        }
        if !isDiagonal {
          break
        }
      }
    }
    return isDiagonal
  }

  private static func isDiagonalDown(_ tensor: Tensor<FloatType>, weights: ModelWeightElement)
    -> Bool
  {
    // Check if this tensor is diagonal. If it is, marking it and prepare to slice the corresponding lora_down.weight.
    // First, check if the shape is right.
    guard weights.format == .I else { return false }
    let shape = tensor.shape
    var isDiagonal = false
    if shape.count == 2 && (shape[1] % weights.count) == 0 {
      let inputShape0 = shape[0] / weights.count
      let estimatedInputShape1 = shape[1] / weights.count
      isDiagonal = true
      for i in 0..<weights.count {
        for j in 0..<weights.count {
          // Now, only diag should have any value.
          guard i != j else { continue }
          let startOffset = weights.offsets?[j] ?? j * estimatedInputShape1
          let endOffset =
            j < weights.count - 1
            ? (weights.offsets?[j + 1] ?? (j + 1) * estimatedInputShape1) : shape[1]
          let offDiagTensor = tensor[
            (i * inputShape0)..<((i + 1) * inputShape0),
            startOffset..<endOffset
          ].copied()
          isDiagonal = offDiagTensor.withUnsafeBytes {
            guard let ptr = $0.baseAddress else { return false }
            let val = ptr.assumingMemoryBound(to: UInt8.self)
            for k in 0..<$0.count {
              if val[k] != 0 {
                return false
              }
            }
            return true
          }
          if !isDiagonal {
            break
          }
        }
        if !isDiagonal {
          break
        }
      }
    }
    return isDiagonal
  }

  static private func findKeysAndValues(_ dictionary: ModelWeightMapping, keys: [String]) -> (
    String, ModelWeightElement
  )? {
    for key in keys {
      guard let value = dictionary[key] else { continue }
      return (key, value)
    }
    return nil
  }

  public static func `import`(
    downloadedFile: String, name: String, filename: String, scaleFactor: Double,
    forceVersion: ModelVersion?, progress: (Float) -> Void
  ) throws -> (ModelVersion, Bool, Int, Bool) {
    let filePath =
      downloadedFile.starts(with: "/")
      ? downloadedFile : ModelZoo.filePathForDownloadedFile(downloadedFile)
    let archive: TensorArchive
    var stateDict: [String: TensorDescriptor]
    if let safeTensors = SafeTensors(url: URL(fileURLWithPath: filePath)) {
      archive = safeTensors
      let states = safeTensors.states
      stateDict = states
    } else if let zipArchive = Archive(url: URL(fileURLWithPath: filePath), accessMode: .read) {
      archive = zipArchive
      let rootObject = try Interpreter.unpickle(zip: zipArchive)
      let originalStateDict = rootObject["state_dict"] as? Interpreter.Dictionary ?? rootObject
      stateDict = [String: TensorDescriptor]()
      originalStateDict.forEach { key, value in
        guard let value = value as? TensorDescriptor else { return }
        stateDict[key] = value
      }
    } else {
      throw UnpickleError.dataNotFound
    }
    let keys = stateDict.keys
    // Fix for one LoRA formulation (commonly found in LyCORIS)
    for key in keys {
      guard key.hasSuffix(".lora_A.weight") || key.hasSuffix(".lora_B.weight") else { continue }
      var components = key.components(separatedBy: ".")
      guard components.count >= 3 else { continue }
      if components[1] == "base_model" {
        components.remove(at: 1)
      }
      if components[0] == "base_model" {
        components.remove(at: 0)
      }
      let newKey = components[0..<(components.count - 2)].joined(separator: "_")
      let isUp = key.hasSuffix(".lora_B.weight")
      if isUp {
        stateDict[newKey + ".lora_up.weight"] = stateDict[key]
      } else {
        stateDict[newKey + ".lora_down.weight"] = stateDict[key]
      }
    }
    // Fix the LoRA formulation for diffusers.
    for key in keys {
      guard key.hasSuffix(".lora.up.weight") || key.hasSuffix(".lora.down.weight") else { continue }
      var components = key.components(separatedBy: ".")
      guard components.count >= 4 else { continue }
      if components[1] == "base_model" {
        components.remove(at: 1)
      }
      if components[0] == "base_model" {
        components.remove(at: 0)
      }
      let newKey = components[0..<(components.count - 3)].joined(separator: "_")
      let isUp = key.hasSuffix(".lora.up.weight")
      if isUp {
        stateDict[newKey + ".lora_up.weight"] = stateDict[key]
      } else {
        stateDict[newKey + ".lora_down.weight"] = stateDict[key]
      }
    }
    // Fix for another LoRA formulation (commonly found in SD-Forge, particularly, layerdiffuse).
    for key in keys {
      guard key.hasSuffix("::lora::0") || key.hasSuffix("::lora::1") else { continue }
      let components = key.components(separatedBy: ".")
      guard components.count > 2 else { continue }
      let newKey = components[0..<(components.count - 1)].joined(separator: "_")
      let isUp = key.hasSuffix("::lora::0")
      if isUp {
        stateDict[newKey + ".lora_up.weight"] = stateDict[key]
      } else {
        stateDict[newKey + ".lora_down.weight"] = stateDict[key]
      }
    }
    // Fix for another LoRA formulation (Foocus, SD-Forge, diff based).
    for key in keys {
      guard key.hasSuffix("::diff::0") else { continue }
      let components = key.components(separatedBy: ".")
      guard components.count > 2 else { continue }
      let newKey =
        components[0..<(components.count - 1)].joined(separator: "_") + "."
        + components[components.count - 1]
      stateDict[newKey.dropLast(9) + ".diff"] = stateDict[key]
    }
    let modelVersion: ModelVersion = try {
      let isSD3Large = stateDict.keys.contains {
        $0.contains("joint_blocks_37_attn_")
          || $0.contains("transformer_blocks_37_attn_")
      }
      let isSD3Medium = stateDict.keys.contains {
        $0.contains("joint_blocks_23_context_block_")
          || $0.contains("transformer_blocks_22_ff_context_")
      }
      let isPixArtSigmaXL = stateDict.keys.contains {
        ($0.contains("blocks_27_") || $0.contains("transformer_blocks_27_"))
          && !($0.contains("single_transformer_blocks_27_")) && !($0.contains("single_blocks_27_"))
      }
      let isFlux1 = stateDict.keys.contains {
        $0.contains("double_blocks.18.img_attn.qkv.")
          || $0.contains("double_blocks_18_img_attn_qkv.")
          || $0.contains("single_transformer_blocks_37_")
      }
      let isHunyuan = stateDict.keys.contains {
        $0.contains("double_blocks.19.img_attn_qkv.")
          || $0.contains("double_blocks_19_img_attn_qkv.")
          || $0.contains("double_blocks.19.img_attn.qkv.")
          || $0.contains("single_transformer_blocks_39_")
      }
      let isSDOrSDXL = stateDict.keys.contains {
        $0.hasSuffix(
          "down_blocks_1_attentions_0_transformer_blocks_0_attn2_to_k.lora_down.weight")
          || $0.hasSuffix(
            "up_blocks_1_attentions_0_transformer_blocks_0_attn2_to_k.lora_down.weight")
          || $0.hasSuffix("input_blocks_4_1_transformer_blocks_0_attn2_to_k.lora_down.weight")
          || $0.hasSuffix(
            "down_blocks_1_attentions_0_transformer_blocks_0_attn2_to_k.hada_w1_b")
          || $0.hasSuffix("input_blocks_4_1_transformer_blocks_0_attn2_to_k.hada_w1_b")
        // For models we can only infer from CLIP, we let user pick which model that should be.
        // || $0.hasSuffix("encoder_layers_0_self_attn_k_proj.lora_down.weight")
        // || $0.hasSuffix("encoder_layers_0_self_attn_k_proj.hada_w1_b")
      }
      // Only confident about these if there is no ambiguity. If there are, we will use force version value.
      switch (isSDOrSDXL, isSD3Medium, isSD3Large, isPixArtSigmaXL, isFlux1, isHunyuan) {
      case (true, false, false, false, false, false):
        if let tokey = stateDict.first(where: {
          $0.key.hasSuffix(
            "down_blocks_1_attentions_0_transformer_blocks_0_attn2_to_k.lora_down.weight")
            || $0.key.hasSuffix(
              "up_blocks_1_attentions_0_transformer_blocks_0_attn2_to_k.lora_down.weight")
            || $0.key.hasSuffix("input_blocks_4_1_transformer_blocks_0_attn2_to_k.lora_down.weight")
            || $0.key.hasSuffix(
              "down_blocks_1_attentions_0_transformer_blocks_0_attn2_to_k.hada_w1_b")
            || $0.key.hasSuffix("input_blocks_4_1_transformer_blocks_0_attn2_to_k.hada_w1_b")
        })?.value
          ?? stateDict.first(where: {
            $0.key.hasSuffix("encoder_layers_0_self_attn_k_proj.lora_down.weight")
              || $0.key.hasSuffix("encoder_layers_0_self_attn_k_proj.hada_w1_b")
          })?.value
        {
          switch tokey.shape.last {
          case 2048:
            if !stateDict.keys.contains(where: {
              $0.contains("mid_block_attentions_0_transformer_blocks_")
                || $0.contains("middle_block_1_transformer_blocks_")
            }) {
              return .ssd1b
            } else {
              return .sdxlBase
            }
          case 1280:
            return .sdxlRefiner
          case 1024:
            return .v2
          case 768:
            // Check if it has lora_te2, if it does, this might be text-encoder only SDXL Base.
            if stateDict.contains(where: { $0.key.hasPrefix("lora_te2_") }) {
              return .sdxlBase
            } else {
              return .v1
            }
          default:
            if let forceVersion = forceVersion {
              return forceVersion
            }
            throw Error.modelVersionFailed
          }
        } else {
          if let forceVersion = forceVersion {
            return forceVersion
          }
          throw Error.modelVersionFailed
        }
      case (false, true, false, false, false, false):
        return .sd3
      case (false, false, true, false, false, false):
        return .sd3Large
      case (false, false, false, true, false, false):
        return .pixart
      case (false, false, false, false, true, false):
        return .flux1
      case (false, false, false, false, false, true):
        return .hunyuanVideo
      default:
        if let forceVersion = forceVersion {
          return forceVersion
        }
        throw Error.modelVersionFailed
      }
    }()
    let graph = DynamicGraph()
    var textModelMapping1: ModelWeightMapping
    var textModelMapping2: ModelWeightMapping
    switch modelVersion {
    case .v1:
      textModelMapping1 = StableDiffusionMapping.CLIPTextModel
      textModelMapping2 = [:]
    case .v2:
      textModelMapping1 = StableDiffusionMapping.OpenCLIPTextModel
      textModelMapping2 = [:]
    case .sdxlBase, .ssd1b:
      textModelMapping1 = StableDiffusionMapping.CLIPTextModel
      textModelMapping2 = StableDiffusionMapping.OpenCLIPTextModelG
      textModelMapping2.merge(StableDiffusionMapping.OpenCLIPTextModelGTransformers) { v, _ in v }
    case .sdxlRefiner:
      textModelMapping1 = StableDiffusionMapping.OpenCLIPTextModelG
      textModelMapping1.merge(StableDiffusionMapping.OpenCLIPTextModelGTransformers) { v, _ in v }
      textModelMapping2 = [:]
    case .sd3, .sd3Large:
      // We don't support LoRA on text encoder for SD3 yet.
      textModelMapping1 = [:]
      textModelMapping2 = [:]
    case .pixart:
      textModelMapping1 = graph.withNoGrad {
        let (t5Mapper, t5) = T5ForConditionalGeneration(b: 1, t: 2, of: FloatType.self)
        let relativePositionBuckets = relativePositionBuckets(
          sequenceLength: 2, numBuckets: 32, maxDistance: 128
        ).toGPU(0)
        let tokens = Tensor<Int32>([0, 0], .GPU(0), .C(2))
        t5.compile(inputs: graph.variable(tokens), graph.variable(relativePositionBuckets))
        return t5Mapper(.generativeModels)
      }
      textModelMapping2 = [:]
    case .flux1:
      textModelMapping1 = [:]
      textModelMapping2 = [:]
    case .hunyuanVideo:
      textModelMapping1 = [:]
      textModelMapping2 = [:]
    case .auraflow:
      fatalError()
    case .kandinsky21, .svdI2v, .wurstchenStageC, .wurstchenStageB:
      fatalError()
    }
    var swapTE2 = false
    if modelVersion == .sdxlBase {
      // Inspect further to see if it has both te1 / te2 model, and whether one of them is CLIP or OpenCLIPTextModelG.
      // If both te2 and te1 exists, check if te2 is 2048, if not, then check if t1 is. If it is,
      // we need to swap.
      // Otherwise, if only te2 exists, check if it is 768, if it is, it is CLIP, we need to swap.
      // If te2 doesn't exists, then we need to decide whether te1 / te is 2048, if it is, we need
      // to swap too.
      let te1tokeys = stateDict.first(where: {
        ($0.key.hasSuffix("encoder_layers_0_self_attn_k_proj.lora_down.weight")
          || $0.key.hasSuffix("encoder_layers_0_self_attn_k_proj.hada_w1_b"))
          && $0.key.contains("_te1_")
      })?.value
      let te2tokeys = stateDict.first(where: {
        ($0.key.hasSuffix("encoder_layers_0_self_attn_k_proj.lora_down.weight")
          || $0.key.hasSuffix("encoder_layers_0_self_attn_k_proj.hada_w1_b"))
          && $0.key.contains("_te2_")
      })?.value
      let tetokeys = stateDict.first(where: {
        ($0.key.hasSuffix("encoder_layers_0_self_attn_k_proj.lora_down.weight")
          || $0.key.hasSuffix("encoder_layers_0_self_attn_k_proj.hada_w1_b"))
          && $0.key.contains("_te_")
      })?.value
      if let te1tokeys = te1tokeys, let te2tokeys = te2tokeys {
        if te2tokeys.shape.last == 1280 {
          // Everything matches.
        } else if te1tokeys.shape.last == 1280 {
          // We need to swap te1 / te2.
          swapTE2 = true
        }
      } else if let te2tokeys = te2tokeys {
        if te2tokeys.shape.last == 768 {
          swapTE2 = true
        }
      } else if let te1tokeys = te1tokeys, te1tokeys.shape.last == 1280 {
        // In this case, we treat the te1 as OpenCLIP-G
        swapTE2 = true
      } else if let tetokeys = tetokeys, tetokeys.shape.last == 1280 {
        // In this case, we treat the te as OpenCLIP-G
        swapTE2 = true
      }
    }
    let textModelMapping1Keys = textModelMapping1.keys
    for key in textModelMapping1Keys {
      let value = textModelMapping1[key]
      let parts = key.components(separatedBy: ".")
      for i in 0...2 {
        guard parts.count - 1 > i else { break }
        textModelMapping1[
          parts[i..<(parts.count - 1)].joined(separator: "_") + "." + parts[parts.count - 1]] =
          value
      }
    }
    let textModelMapping2Keys = textModelMapping2.keys
    for key in textModelMapping2Keys {
      let value = textModelMapping2[key]
      let parts = key.components(separatedBy: ".")
      for i in 0...2 {
        guard parts.count - 1 > i else { break }
        textModelMapping2[
          parts[i..<(parts.count - 1)].joined(separator: "_") + "." + parts[parts.count - 1]] =
          value
      }
    }
    let qkNorm = keys.contains {
      ($0.contains(".ln_k.") || $0.contains(".ln_q.") || $0.contains(".norm_k.")
        || $0.contains(".norm_q."))
        || ($0.contains("_ln_k_") || $0.contains("_ln_q_") || $0.contains("_norm_k_")
          || $0.contains("_norm_q_"))
    }
    let dualAttentionLayers = (0..<38).filter { i in
      if let key = keys.first(where: {
        ($0.contains("_\(i)_x_block_adaLN_modulation_1")
          || $0.contains(".\(i).x_block.adaLN_modulation_1")
          || $0.contains("_\(i)_norm1_linear") || $0.contains(".\(i).norm1.linear"))
          && $0.contains("lora_up")
      }),
        let value = stateDict[key]
      {
        return value.shape[0] == 1536 * 9
      }
      return keys.contains {
        ($0.contains(".\(i).x_block.attn2.") || $0.contains("_blocks.\(i).attn2."))
          || ($0.contains("_\(i)_x_block_attn2_") || $0.contains("_blocks_\(i)_attn2_"))
      }
    }
    // Prepare UNet mapping.
    var (UNetMappingFixed, UNetMapping) = modelWeightsMapping(
      by: modelVersion, qkNorm: qkNorm, dualAttentionLayers: dualAttentionLayers,
      format: [.diffusers, .generativeModels])
    let UNetMappingKeys = UNetMapping.keys
    for key in UNetMappingKeys {
      let value = UNetMapping[key]
      let parts = key.components(separatedBy: ".")
      if parts[0] == "model" {
        if parts.count > 3 {
          UNetMapping[
            parts[2..<(parts.count - 1)].joined(separator: "_") + "." + parts[parts.count - 1]] =
            value
        }
        if parts.count > 2 {
          UNetMapping[
            parts[1..<(parts.count - 1)].joined(separator: "_") + "." + parts[parts.count - 1]] =
            value
        }
      } else if parts[0] == "diffusion_model" {
        if parts.count > 2 {
          UNetMapping[
            parts[1..<(parts.count - 1)].joined(separator: "_") + "." + parts[parts.count - 1]] =
            value
        }
      } else {
        if parts.count > 1 {
          UNetMapping[
            parts[0..<(parts.count - 1)].joined(separator: "_") + "." + parts[parts.count - 1]] =
            value
        }
      }
    }
    let UNetMappingFixedKeys = UNetMappingFixed.keys
    for key in UNetMappingFixedKeys {
      let value = UNetMappingFixed[key]
      let parts = key.components(separatedBy: ".")
      if parts[0] == "model" {
        if parts.count > 3 {
          UNetMappingFixed[
            parts[2..<(parts.count - 1)].joined(separator: "_") + "." + parts[parts.count - 1]] =
            value
        }
        if parts.count > 2 {
          UNetMappingFixed[
            parts[1..<(parts.count - 1)].joined(separator: "_") + "." + parts[parts.count - 1]] =
            value
        }
      } else if parts[0] == "diffusion_model" {
        if parts.count > 2 {
          UNetMappingFixed[
            parts[1..<(parts.count - 1)].joined(separator: "_") + "." + parts[parts.count - 1]] =
            value
        }
      } else {
        if parts.count > 1 {
          UNetMappingFixed[
            parts[0..<(parts.count - 1)].joined(separator: "_") + "." + parts[parts.count - 1]] =
            value
        }
      }
    }
    var didImportTIEmbedding = false
    var textEmbeddingLength = 0
    var isLoHa = false
    try graph.openStore(LoRAZoo.filePathForModelDownloaded(filename)) { store in
      store.removeAll()
      switch modelVersion {
      case .v1, .v2, .kandinsky21, .svdI2v:
        if let tensorDesc = stateDict["emb_params"] {
          try archive.with(tensorDesc) {
            let tensor = Tensor<FloatType>(from: $0)
            textEmbeddingLength = tensorDesc.shape.count > 1 ? tensorDesc.shape[0] : 1
            store.write("string_to_param", tensor: tensor)
            didImportTIEmbedding = true
          }
        }
      case .sd3, .sd3Large:
        if let tensorDescClipG = stateDict["clip_g"] {
          try archive.with(tensorDescClipG) {
            let tensor = Tensor<FloatType>(from: $0)
            store.write("string_to_param_clip_g", tensor: tensor)
            textEmbeddingLength = tensorDescClipG.shape.count > 1 ? tensorDescClipG.shape[0] : 1
            didImportTIEmbedding = true
          }
        }
        if let tensorDescClipL = stateDict["clip_l"] {
          try archive.with(tensorDescClipL) {
            let tensor = Tensor<FloatType>(from: $0)
            store.write("string_to_param_clip_l", tensor: tensor)
            let textEmbeddingLengthClipL =
              tensorDescClipL.shape.count > 1 ? tensorDescClipL.shape[0] : 1
            guard textEmbeddingLengthClipL == textEmbeddingLength else {
              textEmbeddingLength = 0
              return
            }
            didImportTIEmbedding = true
          }
        }
        if let tensorDescT5XXL = stateDict["t5_xxl"] {
          try archive.with(tensorDescT5XXL) {
            let tensor = Tensor<FloatType>(from: $0)
            store.write("string_to_param_t5_xxl", tensor: tensor)
            let textEmbeddingLengthT5XXL =
              tensorDescT5XXL.shape.count > 1 ? tensorDescT5XXL.shape[0] : 1
            guard textEmbeddingLengthT5XXL == textEmbeddingLength else {
              textEmbeddingLength = 0
              return
            }
            didImportTIEmbedding = true
          }
        }
      case .pixart:
        if let tensorDescT5XXL = stateDict["t5_xxl"] {
          try archive.with(tensorDescT5XXL) {
            let tensor = Tensor<FloatType>(from: $0)
            store.write("string_to_param_t5_xxl", tensor: tensor)
            let textEmbeddingLengthT5XXL =
              tensorDescT5XXL.shape.count > 1 ? tensorDescT5XXL.shape[0] : 1
            guard textEmbeddingLengthT5XXL == textEmbeddingLength else {
              textEmbeddingLength = 0
              return
            }
            didImportTIEmbedding = true
          }
        }
      case .flux1:
        if let tensorDescT5XXL = stateDict["t5_xxl"] {
          try archive.with(tensorDescT5XXL) {
            let tensor = Tensor<FloatType>(from: $0)
            store.write("string_to_param_t5_xxl", tensor: tensor)
            let textEmbeddingLengthT5XXL =
              tensorDescT5XXL.shape.count > 1 ? tensorDescT5XXL.shape[0] : 1
            guard textEmbeddingLengthT5XXL == textEmbeddingLength else {
              textEmbeddingLength = 0
              return
            }
            didImportTIEmbedding = true
          }
        }
      case .hunyuanVideo:
        if let tensorDescLlama = stateDict["llama"] {
          try archive.with(tensorDescLlama) {
            let tensor = Tensor<FloatType>(from: $0)
            store.write("string_to_param_llama", tensor: tensor)
            let textEmbeddingLengthLlama =
              tensorDescLlama.shape.count > 1 ? tensorDescLlama.shape[0] : 1
            guard textEmbeddingLengthLlama == textEmbeddingLength else {
              textEmbeddingLength = 0
              return
            }
            didImportTIEmbedding = true
          }
        }
      case .auraflow:
        fatalError()
      case .sdxlBase, .sdxlRefiner, .ssd1b, .wurstchenStageC, .wurstchenStageB:
        if let tensorDescClipG = stateDict["clip_g"] {
          try archive.with(tensorDescClipG) {
            let tensor = Tensor<FloatType>(from: $0)
            store.write("string_to_param_clip_g", tensor: tensor)
            textEmbeddingLength = tensorDescClipG.shape.count > 1 ? tensorDescClipG.shape[0] : 1
            didImportTIEmbedding = true
          }
        }
        if let tensorDescClipL = stateDict["clip_l"] {
          try archive.with(tensorDescClipL) {
            let tensor = Tensor<FloatType>(from: $0)
            store.write("string_to_param_clip_l", tensor: tensor)
            let textEmbeddingLengthClipL =
              tensorDescClipL.shape.count > 1 ? tensorDescClipL.shape[0] : 1
            guard textEmbeddingLengthClipL == textEmbeddingLength else {
              textEmbeddingLength = 0
              return
            }
            didImportTIEmbedding = true
          }
        }
      }
      let modelPrefix: String
      let modelPrefixFixed: String
      switch modelVersion {
      case .v1, .v2, .kandinsky21, .sdxlBase, .sdxlRefiner, .ssd1b, .svdI2v:
        modelPrefix = "unet"
        modelPrefixFixed = "unet_fixed"
      case .wurstchenStageC, .wurstchenStageB:
        modelPrefix = "stage_c"
        modelPrefixFixed = "stage_c_fixed"
      case .sd3, .sd3Large, .pixart, .auraflow, .flux1, .hunyuanVideo:
        modelPrefix = "dit"
        modelPrefixFixed = "dit"
      }
      try store.withTransaction {
        let total = stateDict.count
        var diagonalUpMatrixKeys = Set<String>()
        var diagonalDownsForRedefine = [(String, String, ModelWeightElement)]()
        for (i, (key, descriptor)) in stateDict.enumerated() {
          let parts = key.components(separatedBy: "_")
          guard parts.count > 2 else { continue }
          let te2 = parts[1] == "te2"
          // Try to remove the prefixes.
          let newKeys: [String] = (0...2).compactMap {
            let newParts = String(parts[$0..<parts.count].joined(separator: "_")).components(
              separatedBy: ".")  // Remove the first two.
            guard newParts.count > 1 else { return nil }
            return newParts[0..<(newParts.count > 2 ? newParts.count - 2 : newParts.count - 1)]
              .joined(
                separator: ".") + ".weight"
          }
          if let (newKey, unetParams) = Self.findKeysAndValues(UNetMapping, keys: newKeys) {
            if key.hasSuffix("up.weight") {
              let scalar = try stateDict[
                String(key.prefix(upTo: key.index(key.endIndex, offsetBy: -14))) + "alpha"
              ].map {
                return try archive.with($0) {
                  return Tensor<Float32>(from: $0)[0]
                }
              }
              try archive.with(descriptor) {
                var tensor = Tensor<FloatType>(from: $0)
                let loraDim = Float(tensor.shape[1])
                let isDiagonalUp =
                  unetParams.count > 1 ? Self.isDiagonalUp(tensor, weights: unetParams) : false
                if let scalar = scalar, abs(scalar - loraDim) > 1e-5 {
                  tensor = Tensor<FloatType>(
                    from: (Float(Double(scalar / loraDim) * scaleFactor.squareRoot())
                      * graph.variable(Tensor<Float>(from: tensor)))
                      .rawValue)
                } else if scaleFactor != 1 {
                  tensor = Tensor<FloatType>(
                    from: (Float(scaleFactor.squareRoot())
                      * graph.variable(Tensor<Float>(from: tensor)))
                      .rawValue)
                }
                if isDiagonalUp {
                  diagonalUpMatrixKeys.insert(newKey)
                }
                unetParams.write(
                  to: store, tensor: tensor, format: .O, isDiagonalUp: isDiagonalUp,
                  isDiagonalDown: false
                ) {
                  "__\(modelPrefix)__[\($0)]__up__"
                }
              }
            } else if key.hasSuffix("hada_w1_a") || key.hasSuffix("hada_w2_a") {
              let scalar = try stateDict[
                String(key.prefix(upTo: key.index(key.endIndex, offsetBy: -9))) + "alpha"
              ].map {
                return try archive.with($0) {
                  return Tensor<Float32>(from: $0)[0]
                }
              }
              let wSuffix = key.hasSuffix("hada_w1_a") ? "w1_a" : "w2_a"
              try archive.with(descriptor) {
                var tensor = Tensor<FloatType>(from: $0)
                let loraDim = Float(tensor.shape[1])
                if let scalar = scalar {
                  tensor = Tensor<FloatType>(
                    from: (Float(
                      Double(scalar / loraDim).squareRoot()
                        * scaleFactor.squareRoot().squareRoot())
                      * graph.variable(Tensor<Float>(from: tensor)))
                      .rawValue)
                } else if scaleFactor != 1 {
                  tensor = Tensor<FloatType>(
                    from: (Float(scaleFactor.squareRoot().squareRoot())
                      * graph.variable(Tensor<Float>(from: tensor)))
                      .rawValue)
                }
                unetParams.write(
                  to: store, tensor: tensor, format: .O, isDiagonalUp: false, isDiagonalDown: false
                ) {
                  "__\(modelPrefix)__[\($0)]__\(wSuffix)__"
                }
              }
              isLoHa = true
            }
          } else if let (newKey, unetParams) = Self.findKeysAndValues(
            UNetMappingFixed, keys: newKeys)
          {
            if key.hasSuffix("up.weight") {
              let scalar = try stateDict[
                String(key.prefix(upTo: key.index(key.endIndex, offsetBy: -14))) + "alpha"
              ].map {
                return try archive.with($0) {
                  return Tensor<Float32>(from: $0)[0]
                }
              }
              try archive.with(descriptor) {
                var tensor = Tensor<FloatType>(from: $0)
                let loraDim = Float(tensor.shape[1])
                let isDiagonalUp =
                  unetParams.count > 1 ? Self.isDiagonalUp(tensor, weights: unetParams) : false
                if let scalar = scalar, abs(scalar - loraDim) > 1e-5 {
                  tensor = Tensor<FloatType>(
                    from: (Float(Double(scalar / loraDim) * scaleFactor.squareRoot())
                      * graph.variable(Tensor<Float>(from: tensor)))
                      .rawValue)
                } else if scaleFactor != 1 {
                  tensor = Tensor<FloatType>(
                    from: (Float(scaleFactor.squareRoot())
                      * graph.variable(Tensor<Float>(from: tensor)))
                      .rawValue)
                }
                if isDiagonalUp {
                  diagonalUpMatrixKeys.insert(newKey)
                }
                unetParams.write(
                  to: store, tensor: tensor, format: .O, isDiagonalUp: isDiagonalUp,
                  isDiagonalDown: false
                ) {
                  "__\(modelPrefixFixed)__[\($0)]__up__"
                }
              }
            } else if key.hasSuffix("hada_w1_a") || key.hasSuffix("hada_w2_a") {
              let scalar = try stateDict[
                String(key.prefix(upTo: key.index(key.endIndex, offsetBy: -9))) + "alpha"
              ].map {
                return try archive.with($0) {
                  return Tensor<Float32>(from: $0)[0]
                }
              }
              let wSuffix = key.hasSuffix("hada_w1_a") ? "w1_a" : "w2_a"
              try archive.with(descriptor) {
                var tensor = Tensor<FloatType>(from: $0)
                let loraDim = Float(tensor.shape[1])
                if let scalar = scalar {
                  tensor = Tensor<FloatType>(
                    from: (Float(
                      Double(scalar / loraDim).squareRoot()
                        * scaleFactor.squareRoot().squareRoot())
                      * graph.variable(Tensor<Float>(from: tensor)))
                      .rawValue)
                } else if scaleFactor != 1 {
                  tensor = Tensor<FloatType>(
                    from: (Float(scaleFactor.squareRoot().squareRoot())
                      * graph.variable(Tensor<Float>(from: tensor)))
                      .rawValue)
                }
                unetParams.write(
                  to: store, tensor: tensor, format: .O, isDiagonalUp: false, isDiagonalDown: false
                ) {
                  "__\(modelPrefixFixed)__[\($0)]__\(wSuffix)__"
                }
              }
              isLoHa = true
            }
          } else {
            let textModelMapping: ModelWeightMapping
            if te2 != swapTE2 {
              textModelMapping = textModelMapping2
            } else {
              textModelMapping = textModelMapping1
            }
            if let (_, textParams) = Self.findKeysAndValues(textModelMapping, keys: newKeys),
              let name = textParams.first
            {
              if key.hasSuffix("up.weight") {
                let scalar = try stateDict[
                  String(key.prefix(upTo: key.index(key.endIndex, offsetBy: -14))) + "alpha"
                ].map {
                  return try archive.with($0) {
                    return Tensor<Float32>(from: $0)[0]
                  }
                }
                try archive.with(descriptor) {
                  var tensor = Tensor<FloatType>(from: $0)
                  let loraDim = Float(tensor.shape[1])
                  if let scalar = scalar, abs(scalar - loraDim) > 1e-5 {
                    tensor = Tensor<FloatType>(
                      from: (Float(Double(scalar / loraDim) * scaleFactor.squareRoot())
                        * graph.variable(Tensor<Float>(from: tensor)))
                        .rawValue)
                  } else if scaleFactor != 1 {
                    tensor = Tensor<FloatType>(
                      from: (Float(scaleFactor.squareRoot())
                        * graph.variable(Tensor<Float>(from: tensor)))
                        .rawValue)
                  }
                  if te2 != swapTE2 {
                    store.write("__te2__text_model__[\(name)]__up__", tensor: tensor)
                  } else {
                    store.write("__text_model__[\(name)]__up__", tensor: tensor)
                  }
                }
              } else if key.hasSuffix("hada_w1_a") || key.hasSuffix("hada_w2_a") {
                let scalar = try stateDict[
                  String(key.prefix(upTo: key.index(key.endIndex, offsetBy: -9))) + "alpha"
                ].map {
                  return try archive.with($0) {
                    return Tensor<Float32>(from: $0)[0]
                  }
                }
                let wSuffix = key.hasSuffix("hada_w1_a") ? "w1_a" : "w2_a"
                try archive.with(descriptor) {
                  var tensor = Tensor<FloatType>(from: $0)
                  let loraDim = Float(tensor.shape[1])
                  if let scalar = scalar {
                    tensor = Tensor<FloatType>(
                      from: (Float(
                        Double(scalar / loraDim).squareRoot()
                          * scaleFactor.squareRoot().squareRoot())
                        * graph.variable(Tensor<Float>(from: tensor)))
                        .rawValue)
                  } else if scaleFactor != 1 {
                    tensor = Tensor<FloatType>(
                      from: (Float(scaleFactor.squareRoot().squareRoot())
                        * graph.variable(Tensor<Float>(from: tensor)))
                        .rawValue)
                  }
                  if te2 != swapTE2 {
                    store.write("__te2__text_model__[\(name)]__\(wSuffix)__", tensor: tensor)
                  } else {
                    store.write("__text_model__[\(name)]__\(wSuffix)__", tensor: tensor)
                  }
                }
                isLoHa = true
              }
            }
          }
          progress(Float(i + 1) / Float(total * 2))
        }
        for (i, (key, descriptor)) in stateDict.enumerated() {
          let parts = key.components(separatedBy: "_")
          guard parts.count > 2 else { continue }
          let te2 = parts[1] == "te2"
          // Try to remove the prefixes.
          let newKeys: [String] = (0...2).flatMap { (index) -> [String] in
            let newParts = String(parts[index..<parts.count].joined(separator: "_")).components(
              separatedBy: ".")  // Remove the first two.
            guard newParts.count > 1 else { return [] }
            var newKey =
              newParts[0..<(newParts.count > 2 ? newParts.count - 2 : newParts.count - 1)].joined(
                separator: ".")
            if (newParts[newParts.count - 1] == "diff" || newParts[newParts.count - 1] == "diff_b")
              && newParts.count > 2
            {
              newKey = newKey + "." + newParts[newParts.count - 2]
              if newParts[newParts.count - 1] == "diff_b" {
                return [newKey, newKey + ".bias"]
              }
              return [newKey, newKey + ".weight"]
            } else {
              newKey = newKey + ".weight"
              return [newKey]
            }
          }
          if let (newKey, unetParams) = Self.findKeysAndValues(UNetMapping, keys: newKeys) {
            if key.hasSuffix("down.weight") {
              try archive.with(descriptor) {
                var tensor = Tensor<FloatType>(from: $0)
                let isDiagonalDown = Self.isDiagonalDown(tensor, weights: unetParams)
                if isDiagonalDown {
                  diagonalDownsForRedefine.append(
                    (
                      String(key.prefix(upTo: key.index(key.endIndex, offsetBy: -16))), modelPrefix,
                      unetParams
                    ))
                }
                if scaleFactor != 1 {
                  tensor = Tensor<FloatType>(
                    from: (Float(scaleFactor.squareRoot())
                      * graph.variable(Tensor<Float>(from: tensor)))
                      .rawValue)
                }
                unetParams.write(
                  to: store, tensor: tensor, format: .I,
                  isDiagonalUp: diagonalUpMatrixKeys.contains(newKey),
                  isDiagonalDown: isDiagonalDown
                ) {
                  return "__\(modelPrefix)__[\($0)]__down__"
                }
              }
            } else if key.hasSuffix("mid.weight") {
              try archive.with(descriptor) {
                let tensor = Tensor<FloatType>(from: $0)
                for name in unetParams {
                  store.write("__\(modelPrefix)__[\(name)]__mid__", tensor: tensor)
                }
              }
            } else if key.hasSuffix("hada_w1_b") || key.hasSuffix("hada_w2_b") {
              let wSuffix = key.hasSuffix("hada_w1_b") ? "w1_b" : "w2_b"
              try archive.with(descriptor) {
                var tensor = Tensor<FloatType>(from: $0)
                if scaleFactor != 1 {
                  tensor = Tensor<FloatType>(
                    from: (Float(scaleFactor.squareRoot().squareRoot())
                      * graph.variable(Tensor<Float>(from: tensor)))
                      .rawValue)
                }
                for name in unetParams {
                  store.write("__\(modelPrefix)__[\(name)]__\(wSuffix)__", tensor: tensor)
                }
              }
              isLoHa = true
            } else if key.hasSuffix(".diff") || key.hasSuffix(".diff_b") {
              try archive.with(descriptor) {
                let tensor = Tensor<FloatType>(from: $0)
                /* One-off code to import 8-channel to 9-channel as if it is a inpainting LoRA.
                let shape = tensor.shape
                if shape[1] == 8 && shape[2] == 3 && shape[3] == 3 {
                  // Expand this tensor to 9 in the middle section.
                  var newTensor = Tensor<FloatType>(.CPU, .NCHW(shape[0], 9, 3, 3))
                  newTensor.withUnsafeMutableBytes {
                    guard let f16 = $0.baseAddress?.assumingMemoryBound(to: FloatType.self) else { return }
                    memset(f16, 0, 2 * shape[0] * 9 * 3 * 3)
                  }
                  newTensor[0..<shape[0], 0..<4, 0..<3, 0..<3] = tensor[0..<shape[0], 0..<4, 0..<3, 0..<3]
                  newTensor[0..<shape[0], 5..<9, 0..<3, 0..<3] = tensor[0..<shape[0], 4..<8, 0..<3, 0..<3]
                  tensor = newTensor
                }
                 */
                unetParams.write(
                  to: store, tensor: tensor, format: .O, isDiagonalUp: false, isDiagonalDown: false
                ) {
                  return "__\(modelPrefix)__[\($0)]"
                }
              }
            }
          } else if let (newKey, unetParams) = Self.findKeysAndValues(
            UNetMappingFixed, keys: newKeys)
          {
            if key.hasSuffix("down.weight") {
              try archive.with(descriptor) {
                var tensor = Tensor<FloatType>(from: $0)
                let isDiagonalDown = Self.isDiagonalDown(tensor, weights: unetParams)
                if isDiagonalDown {
                  diagonalDownsForRedefine.append(
                    (
                      String(key.prefix(upTo: key.index(key.endIndex, offsetBy: -16))),
                      modelPrefixFixed, unetParams
                    ))
                }
                if scaleFactor != 1 {
                  tensor = Tensor<FloatType>(
                    from: (Float(scaleFactor.squareRoot())
                      * graph.variable(Tensor<Float>(from: tensor)))
                      .rawValue)
                }
                unetParams.write(
                  to: store, tensor: tensor, format: .I,
                  isDiagonalUp: diagonalUpMatrixKeys.contains(newKey),
                  isDiagonalDown: isDiagonalDown
                ) {
                  return "__\(modelPrefixFixed)__[\($0)]__down__"
                }
              }
            } else if key.hasSuffix("mid.weight") {
              try archive.with(descriptor) {
                let tensor = Tensor<FloatType>(from: $0)
                for name in unetParams {
                  store.write("__\(modelPrefixFixed)__[\(name)]__mid__", tensor: tensor)
                }
              }
            } else if key.hasSuffix("hada_w1_b") || key.hasSuffix("hada_w2_b") {
              let wSuffix = key.hasSuffix("hada_w1_b") ? "w1_b" : "w2_b"
              try archive.with(descriptor) {
                var tensor = Tensor<FloatType>(from: $0)
                if scaleFactor != 1 {
                  tensor = Tensor<FloatType>(
                    from: (Float(scaleFactor.squareRoot().squareRoot())
                      * graph.variable(Tensor<Float>(from: tensor)))
                      .rawValue)
                }
                for name in unetParams {
                  store.write("__\(modelPrefixFixed)__[\(name)]__\(wSuffix)__", tensor: tensor)
                }
              }
              isLoHa = true
            } else if key.hasSuffix(".diff") || key.hasSuffix(".diff_b") {
              try archive.with(descriptor) {
                let tensor = Tensor<FloatType>(from: $0)
                unetParams.write(
                  to: store, tensor: tensor, format: .O, isDiagonalUp: false, isDiagonalDown: false
                ) {
                  return "__\(modelPrefixFixed)__[\($0)]"
                }
              }
            }
          } else {
            let textModelMapping: ModelWeightMapping
            if te2 != swapTE2 {
              textModelMapping = textModelMapping2
            } else {
              textModelMapping = textModelMapping1
            }
            if let (_, textParams) = Self.findKeysAndValues(textModelMapping, keys: newKeys),
              let name = textParams.first
            {
              if key.hasSuffix("down.weight") {
                try archive.with(descriptor) {
                  var tensor = Tensor<FloatType>(from: $0)
                  if scaleFactor != 1 {
                    tensor = Tensor<FloatType>(
                      from: (Float(scaleFactor.squareRoot())
                        * graph.variable(Tensor<Float>(from: tensor)))
                        .rawValue)
                  }
                  if te2 != swapTE2 {
                    store.write("__te2__text_model__[\(name)]__down__", tensor: tensor)
                  } else {
                    store.write("__text_model__[\(name)]__down__", tensor: tensor)
                  }
                }
              } else if key.hasSuffix("hada_w1_b") || key.hasSuffix("hada_w2_b") {
                let wSuffix = key.hasSuffix("hada_w1_b") ? "w1_b" : "w2_b"
                try archive.with(descriptor) {
                  var tensor = Tensor<FloatType>(from: $0)
                  if scaleFactor != 1 {
                    tensor = Tensor<FloatType>(
                      from: (Float(scaleFactor.squareRoot().squareRoot())
                        * graph.variable(Tensor<Float>(from: tensor)))
                        .rawValue)
                  }
                  if te2 != swapTE2 {
                    store.write("__te2__text_model__[\(name)]__\(wSuffix)__", tensor: tensor)
                  } else {
                    store.write("__text_model__[\(name)]__\(wSuffix)__", tensor: tensor)
                  }
                }
                isLoHa = true
              } else if key.hasSuffix(".diff") || key.hasSuffix(".diff_b") {
                try archive.with(descriptor) {
                  let tensor = Tensor<FloatType>(from: $0)
                  if te2 != swapTE2 {
                    store.write("__te2__text_model__[\(name)]", tensor: tensor)
                  } else {
                    store.write("__text_model__[\(name)]", tensor: tensor)
                  }
                }
              }
            }
          }
          progress(Float(i + total + 1) / Float(total * 2))
        }
        for diagonalDownForRedefine in diagonalDownsForRedefine {
          guard let descriptor = stateDict[diagonalDownForRedefine.0 + "lora_up.weight"] else {
            continue
          }
          let scalar = try stateDict[
            diagonalDownForRedefine.0 + "alpha"
          ].map {
            return try archive.with($0) {
              return Tensor<Float32>(from: $0)[0]
            }
          }
          try archive.with(descriptor) {
            var tensor = Tensor<FloatType>(from: $0)
            let loraDim = Float(tensor.shape[1])
            let unetParams = diagonalDownForRedefine.2
            if let scalar = scalar, abs(scalar - loraDim) > 1e-5 {
              tensor = Tensor<FloatType>(
                from: (Float(Double(scalar / loraDim) * scaleFactor.squareRoot())
                  * graph.variable(Tensor<Float>(from: tensor)))
                  .rawValue)
            } else if scaleFactor != 1 {
              tensor = Tensor<FloatType>(
                from: (Float(scaleFactor.squareRoot())
                  * graph.variable(Tensor<Float>(from: tensor)))
                  .rawValue)
            }
            let prefix = diagonalDownForRedefine.1
            unetParams.write(
              to: store, tensor: tensor, format: .O, isDiagonalUp: false,
              isDiagonalDown: true
            ) {
              "__\(prefix)__[\($0)]__up__"
            }
          }
        }
      }
    }
    return (modelVersion, didImportTIEmbedding, textEmbeddingLength, isLoHa)
  }
}
