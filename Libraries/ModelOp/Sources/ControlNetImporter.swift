import Diffusion
import Fickling
import Foundation
import ModelZoo
import NNC
import ZIPFoundation

public final class ControlNetImporter {
  private var access: Int = 0
  private var expectedTotalAccess: Int = 0
  private var progress: ((Float) -> Void)? = nil

  public init() {}

  private func interrupt() -> Bool {
    access += 1
    progress?(0.05 + Float(access) / Float(max(expectedTotalAccess, 1)) * 0.95)
    return false
  }

  private static func numberOfTransformBlocks(prefix: String, upTo: Int, keys: [String]) -> Int {
    for i in 0..<upTo {
      let suffix = "\(prefix).transformer_blocks.\(i).attn2.to_k.weight"
      if !keys.contains(where: { $0.hasSuffix(suffix) }) {
        return i
      }
    }
    return upTo
  }

  public func `import`(
    downloadedFile: String, name: String, filename: String, modifier: ControlHintType,
    progress: @escaping (Float) -> Void
  ) throws -> (transformerBlocks: [Int], version: ModelVersion, type: ControlType) {
    Interpreter.inflateInterrupter = interrupt
    defer {
      Interpreter.inflateInterrupter = nil
    }
    let filePath =
      downloadedFile.starts(with: "/")
      ? downloadedFile : ModelZoo.filePathForDownloadedFile(downloadedFile)
    let archive: TensorArchive
    var stateDict: [String: TensorDescriptor]
    let removePrefix = "control_model."
    if let safeTensors = SafeTensors(url: URL(fileURLWithPath: filePath)) {
      archive = safeTensors
      let states = safeTensors.states
      stateDict = states
      for (key, value) in states {
        var newKey = key
        if key.hasPrefix(removePrefix) {
          let prefixRange = key.startIndex..<key.index(key.startIndex, offsetBy: removePrefix.count)
          newKey = String(key[prefixRange.upperBound...])
        }
        if !key.hasPrefix("model.control_model.") {
          stateDict["model.control_model.\(newKey)"] = value
        }
      }
    } else if let zipArchive = Archive(url: URL(fileURLWithPath: filePath), accessMode: .read) {
      archive = zipArchive
      let rootObject = try Interpreter.unpickle(zip: zipArchive)
      let originalStateDict = rootObject["state_dict"] as? Interpreter.Dictionary ?? rootObject
      stateDict = [String: TensorDescriptor]()
      originalStateDict.forEach { key, value in
        guard let value = value as? TensorDescriptor else { return }
        var newKey = key
        if key.hasPrefix(removePrefix) {
          let prefixRange = key.startIndex..<key.index(key.startIndex, offsetBy: removePrefix.count)
          newKey = String(key[prefixRange.upperBound...])
        }
        if !key.hasPrefix("model.control_model.") {
          stateDict["model.control_model.\(newKey)"] = value
        } else {
          stateDict[key] = value
        }
      }
    } else {
      throw UnpickleError.dataNotFound
    }
    progress(0.05)
    guard
      let tokey = stateDict.first(where: {
        $0.key.hasSuffix("transformer_blocks.0.attn2.to_k.weight")
          // In case there is no transformer blocks at all, we probing the label emb layer for SDXL models.
          || $0.key.hasSuffix("add_embedding.linear_1.weight")
          || $0.key.hasSuffix("label_emb.0.0.weight")
          // Probing for ControlNet LoRA.
          || $0.key.hasSuffix("transformer_blocks.0.attn2.to_k.down")
          || $0.key.hasSuffix("label_emb.0.0.down")
      })?.value
    else {
      throw UnpickleError.tensorNotFound
    }
    var modelVersion: ModelVersion = .v1
    switch tokey.shape.last {
    case 768:
      modelVersion = .v1
    case 1024:
      modelVersion = .v2
    case 2048, 2816:
      modelVersion = .sdxlBase
    case 1280, 2560:
      modelVersion = .sdxlRefiner
      throw UnpickleError.tensorNotFound
    default:
      throw UnpickleError.tensorNotFound
    }
    let graph = DynamicGraph()
    let (hintNet, reader, mapper) = HintNet(channels: 320)
    let tokensTensor = graph.variable(.CPU, .NHWC(1, 512, 512, 3), of: FloatType.self)
    let x = graph.variable(.CPU, .NHWC(2, 64, 64, 4), of: FloatType.self)
    let hint = graph.variable(.CPU, .NHWC(2, 64, 64, 320), of: FloatType.self)
    let tembed = graph.variable(.CPU, .WC(2, 320), of: FloatType.self)
    let dim: Int
    switch modelVersion {
    case .v1:
      dim = 768
    case .v2:
      dim = 1024
    case .sdxlBase, .ssd1b:
      dim = 2048
    case .sd3, .kandinsky21, .svdI2v, .sdxlRefiner, .wurstchenStageC, .wurstchenStageB:
      fatalError()
    }
    let c = graph.variable(.CPU, .HWC(2, 77, dim), of: FloatType.self)
    hintNet.compile(inputs: tokensTensor)
    let controlNet: Model
    let controlNetReader: PythonReader?
    let controlNetMapper: ModelWeightMapper?
    let controlNetFixed: Model?
    let controlNetFixedMapper: ModelWeightMapper?
    let isDiffusersFormat = stateDict.keys.contains { $0.hasPrefix("mid_block.") }
    let isControlNetLoRA = stateDict.keys.contains {
      $0.hasSuffix("label_emb.0.0.down") || $0.hasSuffix("transformer_blocks.0.attn2.to_k.down")
    }
    var transformerBlocks = [Int]()
    switch modelVersion {
    case .v1:
      (controlNet, controlNetReader) = ControlNet(
        batchSize: 2, embeddingLength: (77, 77), startWidth: 64, startHeight: 64,
        usesFlashAttention: .scaleMerged)
      controlNetMapper = nil
      controlNetFixed = nil
      controlNetFixedMapper = nil
    case .v2:
      (controlNet, controlNetReader) = ControlNetv2(
        batchSize: 2, embeddingLength: (77, 77), startWidth: 64, startHeight: 64,
        upcastAttention: false, usesFlashAttention: .scaleMerged)
      controlNetMapper = nil
      controlNetFixed = nil
      controlNetFixedMapper = nil
    case .sdxlBase:
      let inputAttentionRes: KeyValuePairs<Int, [Int]>
      let middleAttentionBlocks: Int
      if isDiffusersFormat {
        // For diffusers format ControlNet, there is small / mid / full size.
        // For small size, there is no transformer blocks, for mid size, there is one transformer blocks.
        // To make this generic, we will get "transform_blocks" and configure the network accordingly.
        let keys = Array(stateDict.keys)
        let downBlocks1AttentionRes = Self.numberOfTransformBlocks(
          prefix: "down_blocks.1.attentions.0", upTo: 2, keys: keys)
        let downBlocks2AttentionRes = Self.numberOfTransformBlocks(
          prefix: "down_blocks.2.attentions.0", upTo: 10, keys: keys)
        inputAttentionRes = [
          2: [downBlocks1AttentionRes, downBlocks1AttentionRes],
          4: [downBlocks2AttentionRes, downBlocks2AttentionRes],
        ]
        middleAttentionBlocks = Self.numberOfTransformBlocks(
          prefix: "mid_block.attentions.0", upTo: 10, keys: keys)
        if downBlocks1AttentionRes != 2 || downBlocks2AttentionRes != 10
          || middleAttentionBlocks != 10
        {
          transformerBlocks = [
            0, downBlocks1AttentionRes, downBlocks2AttentionRes, middleAttentionBlocks,
          ]
        }
      } else {
        inputAttentionRes = [2: [2, 2], 4: [10, 10]]
        middleAttentionBlocks = 10
      }
      if isControlNetLoRA {
        let LoRAConfiguration = LoRANetworkConfiguration(rank: 64, scale: 1, highPrecision: false)
        (controlNet, controlNetMapper) = LoRAControlNetXL(
          batchSize: 2, startWidth: 64, startHeight: 64, channels: [320, 640, 1280],
          embeddingLength: (77, 77), inputAttentionRes: inputAttentionRes,
          middleAttentionBlocks: middleAttentionBlocks, usesFlashAttention: .none,
          LoRAConfiguration: LoRAConfiguration)
        if transformerBlocks.isEmpty || transformerBlocks.contains(where: { $0 > 0 }) {
          (controlNetFixed, controlNetFixedMapper) = LoRAControlNetXLFixed(
            batchSize: 2, startHeight: 64, startWidth: 64, channels: [320, 640, 1280],
            embeddingLength: (77, 77), inputAttentionRes: inputAttentionRes,
            middleAttentionBlocks: middleAttentionBlocks, usesFlashAttention: .none,
            LoRAConfiguration: LoRAConfiguration)
        } else {
          controlNetFixed = nil
          controlNetFixedMapper = nil
        }
      } else {
        (controlNet, _, controlNetMapper) = ControlNetXL(
          batchSize: 2, startWidth: 64, startHeight: 64, channels: [320, 640, 1280],
          embeddingLength: (77, 77), inputAttentionRes: inputAttentionRes,
          middleAttentionBlocks: middleAttentionBlocks, usesFlashAttention: .none)
        if transformerBlocks.isEmpty || transformerBlocks.contains(where: { $0 > 0 }) {
          (controlNetFixed, _, controlNetFixedMapper) = ControlNetXLFixed(
            batchSize: 2, startHeight: 64, startWidth: 64, channels: [320, 640, 1280],
            embeddingLength: (77, 77), inputAttentionRes: inputAttentionRes,
            middleAttentionBlocks: middleAttentionBlocks, usesFlashAttention: .none)
        } else {
          controlNetFixed = nil
          controlNetFixedMapper = nil
        }
      }
      controlNetReader = nil
    case .sd3, .kandinsky21, .ssd1b, .svdI2v, .sdxlRefiner, .wurstchenStageC, .wurstchenStageB:
      fatalError()
    }

    if let controlNetMapper = controlNetMapper {
      try graph.openStore(ModelZoo.filePathForModelDownloaded(filename)) { store in
        let mapping: [String: [String]]
        let mappingFixed: [String: [String]]
        let hintMapping: [String: [String]]
        switch modelVersion {
        case .sdxlBase:
          let vector = graph.variable(.GPU(0), .WC(2, 2816), of: FloatType.self)
          // These values doesn't matter, it won't affect the model shape, just the input vector.
          let kvs: [DynamicGraph.Tensor<FloatType>]
          if let controlNetFixed = controlNetFixed {
            let crossattn = graph.variable(.GPU(0), .HWC(2, 77, 2048), of: FloatType.self)
            kvs = controlNetFixed(inputs: crossattn).map { $0.as(of: FloatType.self) }
          } else {
            kvs = []
          }
          let cArr = [vector] + kvs
          controlNet.compile(inputs: [x, hint, tembed] + cArr)
          if isDiffusersFormat {
            mapping = controlNetMapper(.diffusers)
            if let controlNetFixedMapper = controlNetFixedMapper {
              mappingFixed = controlNetFixedMapper(.diffusers)
            } else {
              mappingFixed = [:]
            }
            hintMapping = mapper(.diffusers)
          } else {
            mapping = controlNetMapper(.generativeModels)
            if let controlNetFixedMapper = controlNetFixedMapper {
              mappingFixed = controlNetFixedMapper(.generativeModels)
            } else {
              mappingFixed = [:]
            }
            hintMapping = mapper(.generativeModels)
          }
        case .sd3, .sdxlRefiner, .ssd1b, .v1, .v2, .kandinsky21, .svdI2v, .wurstchenStageC,
          .wurstchenStageB:
          fatalError()
        }
        store.removeAll()
        expectedTotalAccess = mapping.count + mappingFixed.count + hintMapping.count
        access = 0
        try store.withTransaction {
          for (key, value) in mapping {
            guard let tensorDescriptor = stateDict[key] else {
              continue
            }
            try archive.with(tensorDescriptor) { tensor in
              let tensor = Tensor<FloatType>(from: tensor)
              if value.count > 1 {
                if key.hasSuffix(".down") {
                  for name in value {
                    store.write("__controlnet__[\(name)]", tensor: tensor)
                  }
                } else {
                  let count = tensor.shape[0] / value.count
                  for (i, name) in value.enumerated() {
                    if tensor.shape.count > 1 {
                      store.write(
                        "__controlnet__[\(name)]",
                        tensor: tensor[(i * count)..<((i + 1) * count), 0..<tensor.shape[1]]
                          .copied())
                    } else {
                      store.write(
                        "__controlnet__[\(name)]",
                        tensor: tensor[(i * count)..<((i + 1) * count)].copied())
                    }
                  }
                }
              } else if let name = value.first {
                store.write("__controlnet__[\(name)]", tensor: tensor)
              }
            }
            access += 1
            progress(0.05 + Float(access) / Float(max(expectedTotalAccess, 1)) * 0.95)
          }
          for (key, value) in mappingFixed {
            guard let tensorDescriptor = stateDict[key] else {
              continue
            }
            try archive.with(tensorDescriptor) { tensor in
              let tensor = Tensor<FloatType>(from: tensor)
              if value.count > 1 {
                if key.hasSuffix(".down") {
                  for name in value {
                    store.write("__controlnet_fixed__[\(name)]", tensor: tensor)
                  }
                } else {
                  let count = tensor.shape[0] / value.count
                  for (i, name) in value.enumerated() {
                    if tensor.shape.count > 1 {
                      store.write(
                        "__controlnet_fixed__[\(name)]",
                        tensor: tensor[(i * count)..<((i + 1) * count), 0..<tensor.shape[1]]
                          .copied())
                    } else {
                      store.write(
                        "__controlnet_fixed__[\(name)]",
                        tensor: tensor[(i * count)..<((i + 1) * count)].copied())
                    }
                  }
                }
              } else if let name = value.first {
                store.write("__controlnet_fixed__[\(name)]", tensor: tensor)
              }
            }
            access += 1
            progress(0.05 + Float(access) / Float(max(expectedTotalAccess, 1)) * 0.95)
          }
          for (key, value) in hintMapping {
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
                      "__hintnet__[\(name)]",
                      tensor: tensor[(i * count)..<((i + 1) * count), 0..<tensor.shape[1]]
                        .copied())
                  } else {
                    store.write(
                      "__hintnet__[\(name)]",
                      tensor: tensor[(i * count)..<((i + 1) * count)].copied())
                  }
                }
              } else if let name = value.first {
                store.write("__hintnet__[\(name)]", tensor: tensor)
              }
            }
            access += 1
            progress(0.05 + Float(access) / Float(max(expectedTotalAccess, 1)) * 0.95)
          }
        }
      }
    } else if let controlNetReader = controlNetReader {
      controlNet.compile(inputs: x, hint, tembed, c)
      access = 0
      // This is the same for v1, v2.
      expectedTotalAccess = 340
      self.progress = progress
      try reader(stateDict, archive)
      try controlNetReader(stateDict, archive)
      self.progress = nil
      graph.openStore(ModelZoo.filePathForModelDownloaded(filename)) { store in
        store.removeAll()
        store.write("hintnet", model: hintNet)
        store.write("controlnet", model: controlNet)
      }
      progress(1)
    }
    return (transformerBlocks, modelVersion, isControlNetLoRA ? .controlnetlora : .controlnet)
  }
}
