import Foundation
import NNC

public struct UNetFixedEncoder<FloatType: TensorNumeric & BinaryFloatingPoint> {
  public let filePath: String
  public let version: ModelVersion
  public let usesFlashAttention: Bool
  public let zeroNegativePrompt: Bool
  public init(
    filePath: String, version: ModelVersion, usesFlashAttention: Bool, zeroNegativePrompt: Bool
  ) {
    self.filePath = filePath
    self.version = version
    self.usesFlashAttention = usesFlashAttention
    self.zeroNegativePrompt = zeroNegativePrompt
  }
}

extension UNetFixedEncoder {
  public func vector(
    textEmbedding: DynamicGraph.Tensor<FloatType>, originalSize: (width: Int, height: Int),
    cropTopLeft: (top: Int, left: Int), targetSize: (width: Int, height: Int),
    aestheticScore: Float, negativeOriginalSize: (width: Int, height: Int),
    negativeAestheticScore: Float, fpsId: Int, motionBucketId: Int, condAug: Float
  ) -> [DynamicGraph.Tensor<FloatType>] {
    let graph = textEmbedding.graph
    let batchSize = textEmbedding.shape[0]
    switch version {
    case .sdxlBase, .ssd1b:
      let originalHeight = Tensor<FloatType>(
        from: timeEmbedding(
          timestep: Float(originalSize.height), batchSize: 1, embeddingSize: 256, maxPeriod: 10_000)
      )
      let originalWidth = Tensor<FloatType>(
        from: timeEmbedding(
          timestep: Float(originalSize.width), batchSize: 1, embeddingSize: 256, maxPeriod: 10_000))
      let cropTop = Tensor<FloatType>(
        from: timeEmbedding(
          timestep: Float(cropTopLeft.top), batchSize: 1, embeddingSize: 256, maxPeriod: 10_000))
      let cropLeft = Tensor<FloatType>(
        from: timeEmbedding(
          timestep: Float(cropTopLeft.left), batchSize: 1, embeddingSize: 256, maxPeriod: 10_000))
      let targetHeight = Tensor<FloatType>(
        from: timeEmbedding(
          timestep: Float(targetSize.height), batchSize: 1, embeddingSize: 256, maxPeriod: 10_000))
      let targetWidth = Tensor<FloatType>(
        from: timeEmbedding(
          timestep: Float(originalSize.width), batchSize: 1, embeddingSize: 256, maxPeriod: 10_000))
      let negativeOriginalHeight = Tensor<FloatType>(
        from: timeEmbedding(
          timestep: Float(negativeOriginalSize.height), batchSize: 1, embeddingSize: 256,
          maxPeriod: 10_000)
      )
      let negativeOriginalWidth = Tensor<FloatType>(
        from: timeEmbedding(
          timestep: Float(negativeOriginalSize.width), batchSize: 1, embeddingSize: 256,
          maxPeriod: 10_000))
      var vector = graph.variable(.GPU(0), .WC(batchSize, 2816), of: FloatType.self)
      vector[0..<batchSize, 0..<1280] = textEmbedding
      if zeroNegativePrompt && (batchSize % 2) == 0 && (version == .sdxlBase || version == .ssd1b) {
        vector[0..<(batchSize / 2), 0..<1280].full(0)
      }
      for i in 0..<batchSize {
        if i < batchSize / 2 {
          vector[i..<(i + 1), 1280..<1536] = graph.variable(negativeOriginalHeight.toGPU(0))
          vector[i..<(i + 1), 1536..<1792] = graph.variable(negativeOriginalWidth.toGPU(0))
        } else {
          vector[i..<(i + 1), 1280..<1536] = graph.variable(originalHeight.toGPU(0))
          vector[i..<(i + 1), 1536..<1792] = graph.variable(originalWidth.toGPU(0))
        }
        vector[i..<(i + 1), 1792..<2048] = graph.variable(cropTop.toGPU(0))
        vector[i..<(i + 1), 2048..<2304] = graph.variable(cropLeft.toGPU(0))
        vector[i..<(i + 1), 2304..<2560] = graph.variable(targetHeight.toGPU(0))
        vector[i..<(i + 1), 2560..<2816] = graph.variable(targetWidth.toGPU(0))
      }
      return [vector]
    case .sdxlRefiner:
      let originalHeight = Tensor<FloatType>(
        from: timeEmbedding(
          timestep: Float(originalSize.height), batchSize: 1, embeddingSize: 256, maxPeriod: 10_000)
      )
      let originalWidth = Tensor<FloatType>(
        from: timeEmbedding(
          timestep: Float(originalSize.width), batchSize: 1, embeddingSize: 256, maxPeriod: 10_000))
      let cropTop = Tensor<FloatType>(
        from: timeEmbedding(
          timestep: Float(cropTopLeft.top), batchSize: 1, embeddingSize: 256, maxPeriod: 10_000))
      let cropLeft = Tensor<FloatType>(
        from: timeEmbedding(
          timestep: Float(cropTopLeft.left), batchSize: 1, embeddingSize: 256, maxPeriod: 10_000))
      let negativeOriginalHeight = Tensor<FloatType>(
        from: timeEmbedding(
          timestep: Float(negativeOriginalSize.height), batchSize: 1, embeddingSize: 256,
          maxPeriod: 10_000)
      )
      let negativeOriginalWidth = Tensor<FloatType>(
        from: timeEmbedding(
          timestep: Float(negativeOriginalSize.width), batchSize: 1, embeddingSize: 256,
          maxPeriod: 10_000))
      let scoreVector = Tensor<FloatType>(
        from: timeEmbedding(
          timestep: aestheticScore, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000))
      let negativeScoreVector = Tensor<FloatType>(
        from: timeEmbedding(
          timestep: negativeAestheticScore, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000))
      var vector = graph.variable(.GPU(0), .WC(batchSize, 2560), of: FloatType.self)
      vector[0..<batchSize, 0..<1280] = textEmbedding
      if zeroNegativePrompt && (batchSize % 2) == 0 && (version == .sdxlBase || version == .ssd1b) {
        vector[0..<(batchSize / 2), 0..<1280].full(0)
      }
      for i in 0..<batchSize {
        if i < batchSize / 2 {
          vector[i..<(i + 1), 1280..<1536] = graph.variable(negativeOriginalHeight.toGPU(0))
          vector[i..<(i + 1), 1536..<1792] = graph.variable(negativeOriginalWidth.toGPU(0))
        } else {
          vector[i..<(i + 1), 1280..<1536] = graph.variable(originalHeight.toGPU(0))
          vector[i..<(i + 1), 1536..<1792] = graph.variable(originalWidth.toGPU(0))
        }
        vector[i..<(i + 1), 1792..<2048] = graph.variable(cropTop.toGPU(0))
        vector[i..<(i + 1), 2048..<2304] = graph.variable(cropLeft.toGPU(0))
        if i < batchSize / 2 {
          vector[i..<(i + 1), 2304..<2560] = graph.variable(negativeScoreVector.toGPU(0))
        } else {
          vector[i..<(i + 1), 2304..<2560] = graph.variable(scoreVector.toGPU(0))
        }
      }
      return [vector]
    case .svdI2v:
      let fpsId = Tensor<FloatType>(
        from: timeEmbedding(
          timestep: Float(fpsId), batchSize: 1, embeddingSize: 256, maxPeriod: 10_000))
      let motionBucketId = Tensor<FloatType>(
        from: timeEmbedding(
          timestep: Float(motionBucketId), batchSize: 1, embeddingSize: 256, maxPeriod: 10_000))
      let condAug = Tensor<FloatType>(
        from: timeEmbedding(timestep: condAug, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000))
      var vector = graph.variable(.GPU(0), .WC(1, 768), of: FloatType.self)
      vector[0..<1, 0..<256] = graph.variable(fpsId.toGPU(0))
      vector[0..<1, 256..<512] = graph.variable(motionBucketId.toGPU(0))
      vector[0..<1, 512..<768] = graph.variable(condAug.toGPU(0))
      return [vector]
    case .wurstchenStageC, .wurstchenStageB:
      // We don't need other vectors for sampling.
      return []
    case .v1, .v2, .sd3, .kandinsky21:
      fatalError()
    }
  }
  public func encode(
    textEncoding: [DynamicGraph.Tensor<FloatType>], batchSize: Int, startHeight: Int,
    startWidth: Int, tokenLengthUncond: Int, tokenLengthCond: Int, lora: [LoRAConfiguration]
  ) -> ([DynamicGraph.Tensor<FloatType>], ModelWeightMapper?) {
    let graph = textEncoding[0].graph
    let lora = lora.filter { $0.version == version }
    switch version {
    case .sdxlBase, .ssd1b:
      let batchSize = textEncoding[0].shape[0]
      let maxTokenLength = textEncoding[0].shape[1]
      var crossattn = graph.variable(
        textEncoding[0].kind, .HWC(batchSize, maxTokenLength, 2048), of: FloatType.self)
      crossattn[0..<batchSize, 0..<maxTokenLength, 0..<768] = textEncoding[0]
      crossattn[0..<batchSize, 0..<maxTokenLength, 768..<2048] = textEncoding[1]
      if zeroNegativePrompt && (batchSize % 2) == 0 && (version == .sdxlBase || version == .ssd1b) {
        crossattn[0..<(batchSize / 2), 0..<maxTokenLength, 0..<2048].full(0)
      }
      let unetBaseFixed: Model
      let unetBaseFixedWeightMapper: ModelWeightMapper
      if version == .sdxlBase {
        (unetBaseFixed, _, unetBaseFixedWeightMapper) =
          UNetXLFixed(
            batchSize: batchSize, startHeight: startHeight, startWidth: startWidth,
            channels: [320, 640, 1280], embeddingLength: (tokenLengthUncond, tokenLengthCond),
            inputAttentionRes: [2: [2, 2], 4: [10, 10]], middleAttentionBlocks: 10,
            outputAttentionRes: [2: [2, 2, 2], 4: [10, 10, 10]],
            usesFlashAttention: usesFlashAttention ? .scaleMerged : .none,
            isTemporalMixEnabled: false
          )
      } else {
        precondition(version == .ssd1b)
        (unetBaseFixed, _, unetBaseFixedWeightMapper) =
          UNetXLFixed(
            batchSize: batchSize, startHeight: startHeight, startWidth: startWidth,
            channels: [320, 640, 1280], embeddingLength: (tokenLengthUncond, tokenLengthCond),
            inputAttentionRes: [2: [2, 2], 4: [4, 4]], middleAttentionBlocks: 0,
            outputAttentionRes: [2: [2, 1, 1], 4: [4, 4, 10]],
            usesFlashAttention: usesFlashAttention ? .scale1 : .none, isTemporalMixEnabled: false
          )
      }
      unetBaseFixed.compile(inputs: crossattn)
      graph.openStore(
        filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
      ) { store in
        if lora.count > 0 {
          LoRALoader<FloatType>.openStore(graph, lora: lora) { loader in
            store.read(
              "unet_fixed", model: unetBaseFixed, codec: [.q6p, .q8p, .ezm7, .jit, .externalData]
            ) {
              name, _, _, shape in
              return loader.mergeLoRA(graph, name: name, store: store, shape: shape)
            }
          }
        } else {
          store.read(
            "unet_fixed", model: unetBaseFixed, codec: [.q6p, .q8p, .ezm7, .jit, .externalData])
        }
      }
      return (
        unetBaseFixed(inputs: crossattn).map { $0.as(of: FloatType.self) },
        unetBaseFixedWeightMapper
      )
    case .sdxlRefiner:
      let batchSize = textEncoding[0].shape[0]
      let (unetRefinerFixed, _, unetRefinerFixedWeightMapper) = UNetXLFixed(
        batchSize: batchSize, startHeight: startHeight, startWidth: startWidth,
        channels: [384, 768, 1536, 1536], embeddingLength: (tokenLengthUncond, tokenLengthCond),
        inputAttentionRes: [2: [4, 4], 4: [4, 4]], middleAttentionBlocks: 4,
        outputAttentionRes: [2: [4, 4, 4], 4: [4, 4, 4]],
        usesFlashAttention: usesFlashAttention ? .scaleMerged : .none, isTemporalMixEnabled: false
      )
      unetRefinerFixed.compile(inputs: textEncoding[1])
      graph.openStore(
        filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
      ) { store in
        if lora.count > 0 {
          LoRALoader<FloatType>.openStore(graph, lora: lora) { loader in
            store.read(
              "unet_fixed", model: unetRefinerFixed,
              codec: [.q6p, .q8p, .ezm7, .jit, .externalData]
            ) {
              name, _, _, shape in
              return loader.mergeLoRA(graph, name: name, store: store, shape: shape)
            }
          }
        } else {
          store.read(
            "unet_fixed", model: unetRefinerFixed, codec: [.q6p, .q8p, .ezm7, .jit, .externalData])
        }
      }
      return (
        unetRefinerFixed(inputs: textEncoding[1]).map { $0.as(of: FloatType.self) },
        unetRefinerFixedWeightMapper
      )
    case .svdI2v:
      let numFramesEmb = [320, 640, 1280, 1280].map { embeddingSize in
        let tensors = (0..<batchSize).map {
          graph.variable(
            timeEmbedding(
              timestep: Float($0), batchSize: 1, embeddingSize: embeddingSize, maxPeriod: 10_000)
          ).toGPU(0)
        }
        return DynamicGraph.Tensor<FloatType>(
          from: Concat(axis: 0)(inputs: tensors[0], Array(tensors[1...]))[0].as(of: Float.self))
      }
      let (unetFixed, _, unetFixedWeightMapper) = UNetXLFixed(
        batchSize: 1, startHeight: startHeight, startWidth: startWidth,
        channels: [320, 640, 1280, 1280], embeddingLength: (1, 1),
        inputAttentionRes: [1: [1, 1], 2: [1, 1], 4: [1, 1]], middleAttentionBlocks: 1,
        outputAttentionRes: [1: [1, 1, 1], 2: [1, 1, 1], 4: [1, 1, 1]], usesFlashAttention: .none,
        isTemporalMixEnabled: true
      )
      let crossattn = textEncoding[0].reshaped(.HWC(1, 1, 1024))
      unetFixed.compile(inputs: [crossattn] + numFramesEmb)
      graph.openStore(
        filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
      ) {
        $0.read("unet_fixed", model: unetFixed, codec: [.jit, .q6p, .q8p, .ezm7, .externalData])
      }
      var kvs = unetFixed(inputs: crossattn, numFramesEmb).map { $0.as(of: FloatType.self) }
      let zeroProj = graph.variable(like: crossattn)
      zeroProj.full(0)
      kvs.append(
        contentsOf: unetFixed(inputs: zeroProj, numFramesEmb).map { $0.as(of: FloatType.self) })
      return (kvs, unetFixedWeightMapper)
    case .v1, .v2, .sd3, .kandinsky21:
      return (textEncoding, nil)
    case .wurstchenStageC:
      let batchSize = textEncoding[0].shape[0]
      let emptyImage = graph.variable(.GPU(0), .HWC(batchSize, 1, 1280), of: FloatType.self)
      emptyImage.full(0)
      let (stageCFixed, _) = WurstchenStageCFixed(
        batchSize: batchSize, t: (tokenLengthUncond + 8, tokenLengthCond + 8),
        usesFlashAttention: usesFlashAttention ? .scaleMerged : .none)
      stageCFixed.compile(
        inputs: textEncoding[0], textEncoding[1].reshaped(.HWC(batchSize, 1, 1280)), emptyImage)
      graph.openStore(
        filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
      ) { store in
        if lora.count > 0 {
          // TODO: Do the name remapping.
          LoRALoader<FloatType>.openStore(graph, lora: lora) { loader in
            store.read(
              "stage_c_fixed", model: stageCFixed, codec: [.q6p, .q8p, .ezm7, .jit, .externalData]
            ) {
              name, _, _, shape in
              var name = name
              if name.hasPrefix("__stage_c_fixed__")
                && (name.contains(".keys") || name.contains(".values"))
              {
                name = "__stage_c__" + name.dropFirst(17)
              }
              return loader.mergeLoRA(graph, name: name, store: store, shape: shape)
            }
          }
        } else {
          store.read(
            "stage_c_fixed", model: stageCFixed, codec: [.q6p, .q8p, .ezm7, .jit, .externalData]
          ) {
            name, _, _, _ in
            guard name.hasPrefix("__stage_c_fixed__") else { return .continue(name) }
            guard name.contains(".keys") || name.contains(".values") else { return .continue(name) }
            let name = "__stage_c__" + name.dropFirst(17)
            return .continue(name)
          }
        }
      }
      return (
        stageCFixed(inputs: textEncoding[0], textEncoding[1], emptyImage).map {
          $0.as(of: FloatType.self)
        },
        nil
      )
    case .wurstchenStageB:
      let cfgChannelsAndBatchSize = textEncoding[0].shape[0]
      let effnetHeight = textEncoding[textEncoding.count - 1].shape[1]
      let effnetWidth = textEncoding[textEncoding.count - 1].shape[2]
      let (stageBFixed, _) = WurstchenStageBFixed(
        batchSize: cfgChannelsAndBatchSize, height: startHeight, width: startWidth,
        effnetHeight: effnetHeight, effnetWidth: effnetWidth,
        usesFlashAttention: usesFlashAttention ? .scaleMerged : .none)
      var effnet = graph.variable(
        .GPU(0), .NHWC(cfgChannelsAndBatchSize, effnetHeight, effnetWidth, 16), of: FloatType.self)
      if batchSize != cfgChannelsAndBatchSize {
        effnet.full(0)
        effnet[
          (cfgChannelsAndBatchSize - batchSize)..<cfgChannelsAndBatchSize, 0..<effnetHeight,
          0..<effnetWidth, 0..<16] = textEncoding[textEncoding.count - 1]
      } else {
        effnet[0..<batchSize, 0..<effnetHeight, 0..<effnetWidth, 0..<16] =
          textEncoding[textEncoding.count - 1]
      }
      let pixels = graph.variable(
        .GPU(0), .NHWC(cfgChannelsAndBatchSize, 8, 8, 3), of: FloatType.self)
      pixels.full(0)
      stageBFixed.compile(
        inputs: effnet, pixels, textEncoding[1].reshaped(.HWC(cfgChannelsAndBatchSize, 1, 1280)))
      graph.openStore(
        filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
      ) { store in
        if lora.count > 0 {
          LoRALoader<FloatType>.openStore(graph, lora: lora) { loader in
            store.read(
              "stage_b_fixed", model: stageBFixed, codec: [.q6p, .q8p, .ezm7, .jit, .externalData]
            ) {
              name, _, _, shape in
              return loader.mergeLoRA(graph, name: name, store: store, shape: shape)
            }
          }
        } else {
          store.read(
            "stage_b_fixed", model: stageBFixed, codec: [.q6p, .q8p, .ezm7, .jit, .externalData]
          ) {
            name, _, _, _ in
            guard name.hasPrefix("__stage_b_fixed__") else { return .continue(name) }
            guard name.contains(".keys") || name.contains(".values") else { return .continue(name) }
            let name = "__stage_b__" + name.dropFirst(17)
            return .continue(name)
          }
        }
      }
      return (
        stageBFixed(inputs: effnet, pixels, textEncoding[1]).map { $0.as(of: FloatType.self) },
        nil
      )
    }
  }
}
