import Collections
import Foundation
import NNC
import WeightsCache

public struct UNetFixedEncoder<FloatType: TensorNumeric & BinaryFloatingPoint> {
  public let filePath: String
  public let version: ModelVersion
  public let modifier: SamplerModifier
  public let dualAttentionLayers: [Int]
  public let usesFlashAttention: Bool
  public let zeroNegativePrompt: Bool
  public let isQuantizedModel: Bool
  public let canRunLoRASeparately: Bool
  public let externalOnDemand: Bool
  public let deviceProperties: DeviceProperties
  private let weightsCache: WeightsCache
  public init(
    filePath: String, version: ModelVersion, modifier: SamplerModifier, dualAttentionLayers: [Int],
    usesFlashAttention: Bool, zeroNegativePrompt: Bool, isQuantizedModel: Bool,
    canRunLoRASeparately: Bool, externalOnDemand: Bool, deviceProperties: DeviceProperties,
    weightsCache: WeightsCache
  ) {
    self.filePath = filePath
    self.version = version
    self.modifier = modifier
    self.dualAttentionLayers = dualAttentionLayers
    self.usesFlashAttention = usesFlashAttention
    self.zeroNegativePrompt = zeroNegativePrompt
    self.isQuantizedModel = isQuantizedModel
    self.canRunLoRASeparately = canRunLoRASeparately
    self.externalOnDemand = externalOnDemand
    self.deviceProperties = deviceProperties
    self.weightsCache = weightsCache
  }
}

extension UNetFixedEncoder {
  static func isFixedEncoderRequired(version: ModelVersion) -> Bool {
    switch version {
    case .sdxlBase, .sdxlRefiner, .ssd1b, .svdI2v, .sd3, .sd3Large, .pixart, .auraflow, .flux1,
      .wurstchenStageC, .wurstchenStageB, .hunyuanVideo, .wan21_1_3b, .wan21_14b, .hiDreamI1:
      return true
    case .v1, .v2, .kandinsky21:
      return false
    }
  }

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
      let textEmbeddingShape = textEmbedding.shape
      var vector = graph.variable(
        .GPU(0), .WC(batchSize, textEmbeddingShape[1] + 1536), of: FloatType.self)
      vector[0..<batchSize, 0..<textEmbeddingShape[1]] = textEmbedding
      if zeroNegativePrompt && (batchSize % 2) == 0 && (version == .sdxlBase || version == .ssd1b) {
        vector[0..<(batchSize / 2), 0..<textEmbeddingShape[1]].full(0)
      }
      for i in 0..<batchSize {
        if i < batchSize / 2 {
          vector[i..<(i + 1), textEmbeddingShape[1]..<(textEmbeddingShape[1] + 256)] =
            graph.variable(negativeOriginalHeight.toGPU(0))
          vector[i..<(i + 1), (textEmbeddingShape[1] + 256)..<(textEmbeddingShape[1] + 512)] =
            graph.variable(negativeOriginalWidth.toGPU(0))
        } else {
          vector[i..<(i + 1), textEmbeddingShape[1]..<(textEmbeddingShape[1] + 256)] =
            graph.variable(originalHeight.toGPU(0))
          vector[i..<(i + 1), (textEmbeddingShape[1] + 256)..<(textEmbeddingShape[1] + 512)] =
            graph.variable(originalWidth.toGPU(0))
        }
        vector[i..<(i + 1), (textEmbeddingShape[1] + 512)..<(textEmbeddingShape[1] + 768)] =
          graph.variable(cropTop.toGPU(0))
        vector[i..<(i + 1), (textEmbeddingShape[1] + 768)..<(textEmbeddingShape[1] + 1024)] =
          graph.variable(cropLeft.toGPU(0))
        vector[i..<(i + 1), (textEmbeddingShape[1] + 1024)..<(textEmbeddingShape[1] + 1280)] =
          graph.variable(targetHeight.toGPU(0))
        vector[i..<(i + 1), (textEmbeddingShape[1] + 1280)..<(textEmbeddingShape[1] + 1536)] =
          graph.variable(targetWidth.toGPU(0))
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
    case .sd3, .sd3Large, .pixart, .auraflow, .flux1, .hunyuanVideo, .wan21_1_3b, .wan21_14b,
      .hiDreamI1:
      return []
    case .v1, .v2, .kandinsky21:
      fatalError()
    }
  }
  public func encode(
    isCfgEnabled: Bool, textGuidanceScale: Float, guidanceEmbed: Float,
    isGuidanceEmbedEnabled: Bool, distilledGuidanceLayers: Int,
    textEncoding: [DynamicGraph.Tensor<FloatType>],
    timesteps: [Float], batchSize: Int, startHeight: Int, startWidth: Int, tokenLengthUncond: Int,
    tokenLengthCond: Int, lora: [LoRAConfiguration], tiledDiffusion: TiledConfiguration,
    teaCache teaCacheConfiguration: TeaCacheConfiguration,
    injectedControls: [(
      model: ControlModel<FloatType>, hints: [([DynamicGraph.Tensor<FloatType>], Float)]
    )]
  ) -> ([DynamicGraph.AnyTensor], ModelWeightMapper?) {
    let graph = textEncoding[0].graph
    let lora = lora.filter { $0.version == version }
    let externalData: DynamicGraph.Store.Codec =
      externalOnDemand
      ? .externalOnDemand : .externalData(deviceProperties.isFreadPreferred ? .fread : .mmap)
    switch version {
    case .sdxlBase, .ssd1b:
      let batchSize = textEncoding[0].shape[0]
      let maxTokenLength = textEncoding[0].shape[1]
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
      var textEncoding = textEncoding
      graph.openStore(
        filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
      ) {
        if $0.read(like: "__encoder_hid_proj__[t-0-0]") != nil {
          let encoderHidProj = Dense(count: 2_048)
          encoderHidProj.compile(inputs: textEncoding[0])
          $0.read(
            "encoder_hid_proj", model: encoderHidProj,
            codec: [.jit, .q6p, .q8p, .ezm7, externalData])
          textEncoding = encoderHidProj(inputs: textEncoding[0]).map { $0.as(of: FloatType.self) }
        }
      }
      var crossattn = graph.variable(
        textEncoding[0].kind, .HWC(batchSize, maxTokenLength, 2048), of: FloatType.self)
      if textEncoding.count >= 2 {
        crossattn[0..<batchSize, 0..<maxTokenLength, 0..<768] = textEncoding[0]
        crossattn[0..<batchSize, 0..<maxTokenLength, 768..<2048] = textEncoding[1]
      } else {
        crossattn[0..<batchSize, 0..<maxTokenLength, 0..<2048] = textEncoding[0]
      }
      if zeroNegativePrompt && isCfgEnabled && (version == .sdxlBase || version == .ssd1b) {
        crossattn[0..<(batchSize / 2), 0..<maxTokenLength, 0..<2048].full(0)
      }
      unetBaseFixed.maxConcurrency = .limit(4)
      unetBaseFixed.compile(inputs: crossattn)
      graph.openStore(
        filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
      ) { store in
        if lora.count > 0 {
          LoRALoader.openStore(graph, lora: lora) { loader in
            store.read(
              "unet_fixed", model: unetBaseFixed, codec: [.q6p, .q8p, .ezm7, .jit, externalData]
            ) {
              name, dataType, _, shape in
              return loader.mergeLoRA(
                graph, name: name, store: store, dataType: dataType, shape: shape,
                of: FloatType.self)
            }
          }
        } else {
          store.read(
            "unet_fixed", model: unetBaseFixed, codec: [.q6p, .q8p, .ezm7, .jit, externalData])
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
      unetRefinerFixed.maxConcurrency = .limit(4)
      unetRefinerFixed.compile(inputs: textEncoding[1])
      graph.openStore(
        filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
      ) { store in
        if lora.count > 0 {
          LoRALoader.openStore(graph, lora: lora) { loader in
            store.read(
              "unet_fixed", model: unetRefinerFixed,
              codec: [.q6p, .q8p, .ezm7, .jit, externalData]
            ) {
              name, dataType, _, shape in
              return loader.mergeLoRA(
                graph, name: name, store: store, dataType: dataType, shape: shape,
                of: FloatType.self)
            }
          }
        } else {
          store.read(
            "unet_fixed", model: unetRefinerFixed, codec: [.q6p, .q8p, .ezm7, .jit, externalData])
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
      unetFixed.maxConcurrency = .limit(4)
      unetFixed.compile(inputs: [crossattn] + numFramesEmb)
      graph.openStore(
        filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
      ) {
        $0.read("unet_fixed", model: unetFixed, codec: [.jit, .q6p, .q8p, .ezm7, externalData])
      }
      var kvs = unetFixed(inputs: crossattn, numFramesEmb).map { $0.as(of: FloatType.self) }
      let zeroProj = graph.variable(like: crossattn)
      zeroProj.full(0)
      kvs.append(
        contentsOf: unetFixed(inputs: zeroProj, numFramesEmb).map { $0.as(of: FloatType.self) })
      return (kvs, unetFixedWeightMapper)
    case .v1, .v2, .kandinsky21:
      return (textEncoding, nil)
    case .pixart:
      let tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
      let tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
      let h = tiledHeight / 2
      let w = tiledWidth / 2
      let posEmbed = graph.variable(
        Tensor<FloatType>(from: sinCos2DPositionEmbedding(height: h, width: w, embeddingSize: 1152))
          .reshaped(.HWC(1, h * w, 1152)).toGPU(0))
      precondition(timesteps.count > 0)
      var timeEmbeds = graph.variable(
        .GPU(0), .WC(timesteps.count, 256), of: FloatType.self)
      for (i, timestep) in timesteps.enumerated() {
        let timeEmbed = graph.variable(
          Tensor<FloatType>(
            from: timeEmbedding(
              timestep: timestep, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000)
          ).toGPU(0))
        timeEmbeds[i..<(i + 1), 0..<256] = timeEmbed
      }
      timeEmbeds = timeEmbeds.reshaped(.HWC(timesteps.count, 1, 256))
      var c = textEncoding[0]
      let cBatchSize = c.shape[0]
      if zeroNegativePrompt && isCfgEnabled {
        let oldC = c
        c = graph.variable(like: c)
        c.full(0)
        let shape = c.shape
        c[batchSize..<(batchSize * 2), 0..<shape[1], 0..<shape[2]] =
          oldC[batchSize..<(batchSize * 2), 0..<shape[1], 0..<shape[2]]
      }
      let (_, unetFixed) = PixArtFixed(
        batchSize: cBatchSize, channels: 1152, layers: 28,
        tokenLength: (tokenLengthUncond, tokenLengthCond),
        usesFlashAttention: usesFlashAttention, of: FloatType.self)
      unetFixed.maxConcurrency = .limit(4)
      unetFixed.compile(inputs: timeEmbeds, c)
      graph.openStore(
        filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
      ) { store in
        if lora.count > 0 {
          LoRALoader.openStore(graph, lora: lora) { loader in
            store.read(
              "dit", model: unetFixed, codec: [.q6p, .q8p, .ezm7, .jit, externalData]
            ) {
              name, dataType, _, shape in
              return loader.mergeLoRA(
                graph, name: name, store: store, dataType: dataType, shape: shape,
                of: FloatType.self)
            }
          }
        } else {
          store.read("dit", model: unetFixed, codec: [.jit, .q6p, .q8p, .ezm7, externalData])
        }
      }
      return (
        [posEmbed] + unetFixed(inputs: timeEmbeds, c).map { $0.as(of: FloatType.self) },
        nil
      )
    case .auraflow:
      var c = textEncoding[0]
      if c.shape[1] < 256 {
        let oldC = c
        c = graph.variable(.GPU(0), .HWC(oldC.shape[0], 256, oldC.shape[2]))
        c.full(0)
        c[0..<oldC.shape[0], 0..<oldC.shape[1], 0..<oldC.shape[2]] = oldC
      }
      // Load the unetFixed.
      let cBatchSize = c.shape[0]
      precondition(timesteps.count > 0)
      let (_, unetFixed) = AuraFlowFixed(
        batchSize: (cBatchSize, cBatchSize * timesteps.count), channels: 3072, layers: (4, 32),
        of: FloatType.self)
      var timeEmbeds = graph.variable(
        .GPU(0), .WC(cBatchSize * timesteps.count, 256), of: FloatType.self)
      for (i, timestep) in timesteps.enumerated() {
        let timeEmbed = graph.variable(
          Tensor<FloatType>(
            from: timeEmbedding(
              timestep: timestep, batchSize: cBatchSize, embeddingSize: 256, maxPeriod: 10_000)
          ).toGPU(0))
        timeEmbeds[(i * cBatchSize)..<((i + 1) * cBatchSize), 0..<256] = timeEmbed
      }
      unetFixed.maxConcurrency = .limit(4)
      unetFixed.compile(inputs: c, timeEmbeds)
      if !weightsCache.detach("\(filePath):[fixed]", to: unetFixed.parameters) {
        graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) { store in
          store.read("dit", model: unetFixed, codec: [.jit, .q6p, .q8p, .ezm7, externalData])
        }
      }
      let result = unetFixed(inputs: c, timeEmbeds).map { $0.as(of: FloatType.self) }
      weightsCache.attach("\(filePath):[fixed]", from: unetFixed.parameters)
      return (result, nil)
    case .sd3, .sd3Large:
      var c: DynamicGraph.Tensor<FloatType>
      var pooled: DynamicGraph.Tensor<FloatType>
      if textEncoding.count >= 4 {
        let c0 = textEncoding[0]
        let c1 = textEncoding[1]
        let c2 = textEncoding[2]
        pooled = textEncoding[3]
        let cBatchSize = c0.shape[0]
        let maxLength = c0.shape[1]
        let t5Length = c2.shape[1]
        c = graph.variable(
          .GPU(0), .HWC(cBatchSize, maxLength + t5Length, 4096), of: FloatType.self)
        c.full(0)
        if zeroNegativePrompt && isCfgEnabled {
          let oldPooled = pooled
          pooled = graph.variable(like: oldPooled)
          pooled.full(0)
          pooled[batchSize..<(batchSize * 2), 0..<2048] =
            oldPooled[batchSize..<(batchSize * 2), 0..<2048]
          c[batchSize..<(batchSize * 2), 0..<maxLength, 0..<768] =
            c0[batchSize..<(batchSize * 2), 0..<maxLength, 0..<768]
          c[batchSize..<(batchSize * 2), 0..<maxLength, 768..<2048] =
            c1[batchSize..<(batchSize * 2), 0..<maxLength, 0..<1280]
          c[batchSize..<(batchSize * 2), maxLength..<(maxLength + t5Length), 0..<4096] =
            c2[batchSize..<(batchSize * 2), 0..<t5Length, 0..<4096]
        } else {
          c[0..<cBatchSize, 0..<maxLength, 0..<768] = c0
          c[0..<cBatchSize, 0..<maxLength, 768..<2048] = c1
          c[0..<cBatchSize, maxLength..<(maxLength + t5Length), 0..<4096] = c2
        }
      } else {
        let c0 = textEncoding[0]
        let c1 = textEncoding[1]
        pooled = textEncoding[2]
        let cBatchSize = c0.shape[0]
        let maxLength = c0.shape[1]
        c = graph.variable(.GPU(0), .HWC(cBatchSize, maxLength, 4096), of: FloatType.self)
        c.full(0)
        if zeroNegativePrompt && isCfgEnabled {
          let oldPooled = pooled
          pooled = graph.variable(like: oldPooled)
          pooled.full(0)
          pooled[batchSize..<(batchSize * 2), 0..<2048] =
            oldPooled[batchSize..<(batchSize * 2), 0..<2048]
          c[batchSize..<(batchSize * 2), 0..<maxLength, 0..<768] =
            c0[batchSize..<(batchSize * 2), 0..<maxLength, 0..<768]
          c[batchSize..<(batchSize * 2), 0..<maxLength, 768..<2048] =
            c1[batchSize..<(batchSize * 2), 0..<maxLength, 0..<1280]
        } else {
          c[0..<cBatchSize, 0..<maxLength, 0..<768] = c0
          c[0..<cBatchSize, 0..<maxLength, 768..<2048] = c1
        }
      }
      // Load the unetFixed.
      precondition(timesteps.count > 0)
      let cBatchSize = c.shape[0]
      let unetFixed: Model
      switch version {
      case .sd3:
        (_, unetFixed) = MMDiTFixed(
          batchSize: cBatchSize * timesteps.count, channels: 1536, layers: 24,
          dualAttentionLayers: dualAttentionLayers)
      case .sd3Large:
        (_, unetFixed) = MMDiTFixed(
          batchSize: cBatchSize * timesteps.count, channels: 2432, layers: 38,
          dualAttentionLayers: [])
      case .v1, .v2, .auraflow, .flux1, .kandinsky21, .pixart, .sdxlBase, .sdxlRefiner, .ssd1b,
        .svdI2v, .wurstchenStageB, .wurstchenStageC, .hunyuanVideo, .wan21_1_3b, .wan21_14b,
        .hiDreamI1:
        fatalError()
      }
      var timeEmbeds = graph.variable(
        .GPU(0), .WC(cBatchSize * timesteps.count, 256), of: FloatType.self)
      var pooleds = graph.variable(
        .GPU(0), .WC(cBatchSize * timesteps.count, 2048), of: FloatType.self)
      for (i, timestep) in timesteps.enumerated() {
        let timeEmbed = graph.variable(
          Tensor<FloatType>(
            from: timeEmbedding(
              timestep: timestep, batchSize: cBatchSize, embeddingSize: 256, maxPeriod: 10_000)
          ).toGPU(0))
        timeEmbeds[(i * cBatchSize)..<((i + 1) * cBatchSize), 0..<256] = timeEmbed
        pooleds[(i * cBatchSize)..<((i + 1) * cBatchSize), 0..<2048] = pooled
      }
      unetFixed.maxConcurrency = .limit(4)
      unetFixed.compile(inputs: c, timeEmbeds, pooleds)
      if lora.count > 0 || !weightsCache.detach("\(filePath):[fixed]", to: unetFixed.parameters) {
        graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) { store in
          if lora.count > 0 {
            LoRALoader.openStore(graph, lora: lora) { loader in
              store.read(
                "dit", model: unetFixed, codec: [.q6p, .q8p, .ezm7, .jit, externalData]
              ) {
                name, dataType, _, shape in
                return loader.mergeLoRA(
                  graph, name: name, store: store, dataType: dataType, shape: shape,
                  of: FloatType.self)
              }
            }
          } else {
            store.read("dit", model: unetFixed, codec: [.jit, .q6p, .q8p, .ezm7, externalData])
          }
        }
      }
      let result = unetFixed(inputs: c, timeEmbeds, pooleds).map { $0.as(of: FloatType.self) }
      if lora.isEmpty {
        weightsCache.attach("\(filePath):[fixed]", from: unetFixed.parameters)
      }
      return (result, nil)
    case .wurstchenStageC:
      let batchSize = textEncoding[0].shape[0]
      let emptyImage = graph.variable(.GPU(0), .HWC(batchSize, 1, 1280), of: FloatType.self)
      emptyImage.full(0)
      let (stageCFixed, _) = WurstchenStageCFixed(
        batchSize: batchSize, t: (tokenLengthUncond + 8, tokenLengthCond + 8),
        usesFlashAttention: usesFlashAttention ? .scaleMerged : .none)
      stageCFixed.maxConcurrency = .limit(4)
      stageCFixed.compile(
        inputs: textEncoding[0], textEncoding[1].reshaped(.HWC(batchSize, 1, 1280)), emptyImage)
      graph.openStore(
        filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
      ) { store in
        if lora.count > 0 {
          // TODO: Do the name remapping.
          LoRALoader.openStore(graph, lora: lora) { loader in
            store.read(
              "stage_c_fixed", model: stageCFixed, codec: [.q6p, .q8p, .ezm7, .jit, externalData]
            ) {
              name, dataType, _, shape in
              var name = name
              if name.hasPrefix("__stage_c_fixed__")
                && (name.contains(".keys") || name.contains(".values"))
              {
                name = "__stage_c__" + name.dropFirst(17)
              }
              return loader.mergeLoRA(
                graph, name: name, store: store, dataType: dataType, shape: shape,
                of: FloatType.self)
            }
          }
        } else {
          store.read(
            "stage_c_fixed", model: stageCFixed, codec: [.q6p, .q8p, .ezm7, .jit, externalData]
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
    case .flux1:
      let textEncoding = ControlModel<FloatType>.modifyTextEncoding(
        textEncoding: textEncoding, isCfgEnabled: isCfgEnabled, batchSize: batchSize,
        injecteds: injectedControls)
      let c0 = textEncoding[0]
      var pooled = textEncoding[1]
      let cBatchSize = c0.shape[0]
      let t5Length = c0.shape[1]
      var c = graph.variable(
        .GPU(0), .HWC(cBatchSize, t5Length, 4096), of: FloatType.self)
      c.full(0)
      if zeroNegativePrompt && isCfgEnabled {
        let oldPooled = pooled
        pooled = graph.variable(like: oldPooled)
        pooled.full(0)
        pooled[batchSize..<(batchSize * 2), 0..<768] =
          oldPooled[batchSize..<(batchSize * 2), 0..<768]
        c[batchSize..<(batchSize * 2), 0..<t5Length, 0..<4096] =
          c0[batchSize..<(batchSize * 2), 0..<t5Length, 0..<4096]
      } else {
        c[0..<cBatchSize, 0..<t5Length, 0..<4096] = c0
      }
      precondition(timesteps.count > 0)
      // Load the unetFixed.
      // TODO: This is not ideal because we opened it twice, but hopefully it is OK for now until we are at 300ms domain.
      let isGuidanceEmbedSupported =
        (try?
          (graph.openStore(
            filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
          ) {
            return $0.read(like: "__dit__[t-guidance_embedder_0-0-1]") != nil
          }).get()) ?? (distilledGuidanceLayers > 0)
      let unetFixed: Model
      let lora = Array(
        (OrderedDictionary<String, LoRAConfiguration>(
          lora.filter({ $0.version == version }).map {
            ($0.file, $0)
          }
        ) {
          LoRAConfiguration(
            file: $0.file, weight: $0.weight + $1.weight, version: $0.version, isLoHa: $0.isLoHa,
            modifier: $0.modifier)
        })
        .values
      ).filter { $0.weight != 0 }
      let (rankOfLoRA, filesRequireMerge) = LoRALoader.rank(
        graph, of: lora.map { $0.file }, modelFile: filePath)
      let isLoHa = lora.contains { $0.isLoHa }
      var configuration = LoRANetworkConfiguration(rank: rankOfLoRA, scale: 1, highPrecision: false)
      let runLoRASeparatelyIsPreferred = isQuantizedModel || externalOnDemand
      let shouldRunLoRASeparately =
        !lora.isEmpty && !isLoHa && runLoRASeparatelyIsPreferred && rankOfLoRA > 0
        && canRunLoRASeparately
      if shouldRunLoRASeparately {
        let keys = LoRALoader.keys(graph, of: lora.map { $0.file }, modelFile: filePath)
        configuration.keys = keys
        if distilledGuidanceLayers > 0 {
          (_, unetFixed) = LoRAChromaFixed(
            channels: 3072, distilledGuidanceLayers: distilledGuidanceLayers, layers: (19, 38),
            LoRAConfiguration: configuration, contextPreloaded: true)
        } else {
          (_, unetFixed) = LoRAFlux1Fixed(
            batchSize: (cBatchSize, cBatchSize * timesteps.count), channels: 3072, layers: (19, 38),
            LoRAConfiguration: configuration, contextPreloaded: true,
            guidanceEmbed: isGuidanceEmbedSupported)
        }
      } else {
        if distilledGuidanceLayers > 0 {
          (_, unetFixed) = ChromaFixed(
            channels: 3072, distilledGuidanceLayers: distilledGuidanceLayers, layers: (19, 38),
            contextPreloaded: true)
        } else {
          (_, unetFixed) = Flux1Fixed(
            batchSize: (cBatchSize, cBatchSize * timesteps.count), channels: 3072, layers: (19, 38),
            contextPreloaded: true, guidanceEmbed: isGuidanceEmbedSupported)
        }
      }
      let restInputs: [DynamicGraph.Tensor<FloatType>]
      if distilledGuidanceLayers > 0 {
        var conditionEmbeds = graph.variable(
          .GPU(0), .HWC(cBatchSize * timesteps.count, 344, 64), of: FloatType.self)
        let guidanceScale = isGuidanceEmbedEnabled ? textGuidanceScale : 0
        let guidanceEmbed = graph.variable(
          Tensor<FloatType>(
            from: timeEmbedding(
              timestep: guidanceScale * 1_000, batchSize: cBatchSize * 344, embeddingSize: 16,
              maxPeriod: 10_000)
          ).toGPU(0)
        ).reshaped(.HWC(cBatchSize, 344, 16))
        var modEmbeds = graph.variable(.GPU(0), .HWC(cBatchSize, 344, 32), of: FloatType.self)
        for i in 0..<344 {
          let modEmbed = graph.variable(
            Tensor<FloatType>(
              from: timeEmbedding(
                timestep: Float(i * 1_000), batchSize: cBatchSize, embeddingSize: 32,
                maxPeriod: 10_000)
            ).toGPU(0)
          ).reshaped(.HWC(cBatchSize, 1, 32))
          modEmbeds[0..<cBatchSize, i..<(i + 1), 0..<32] = modEmbed
        }
        for (i, timestep) in timesteps.enumerated() {
          let timeEmbed = graph.variable(
            Tensor<FloatType>(
              from: timeEmbedding(
                timestep: timestep, batchSize: cBatchSize * 344, embeddingSize: 16,
                maxPeriod: 10_000)
            ).toGPU(0)
          ).reshaped(.HWC(cBatchSize, 344, 16))
          conditionEmbeds[(i * cBatchSize)..<((i + 1) * cBatchSize), 0..<344, 0..<16] = timeEmbed
          conditionEmbeds[(i * cBatchSize)..<((i + 1) * cBatchSize), 0..<344, 16..<32] =
            guidanceEmbed
          conditionEmbeds[(i * cBatchSize)..<((i + 1) * cBatchSize), 0..<344, 32..<64] = modEmbeds
        }
        restInputs = [conditionEmbeds]
      } else {
        var timeEmbeds = graph.variable(
          .GPU(0), .WC(cBatchSize * timesteps.count, 256), of: FloatType.self)
        var pooleds = graph.variable(
          .GPU(0), .WC(cBatchSize * timesteps.count, 768), of: FloatType.self)
        var guidanceEmbeds: DynamicGraph.Tensor<FloatType>?
        if isGuidanceEmbedSupported {
          guidanceEmbeds = graph.variable(
            .GPU(0), .WC(cBatchSize * timesteps.count, 256), of: FloatType.self)
        } else {
          guidanceEmbeds = nil
        }
        for (i, timestep) in timesteps.enumerated() {
          let timeEmbed = graph.variable(
            Tensor<FloatType>(
              from: timeEmbedding(
                timestep: timestep, batchSize: cBatchSize, embeddingSize: 256, maxPeriod: 10_000)
            ).toGPU(0))
          timeEmbeds[(i * cBatchSize)..<((i + 1) * cBatchSize), 0..<256] = timeEmbed
          pooleds[(i * cBatchSize)..<((i + 1) * cBatchSize), 0..<768] = pooled
          if var guidanceEmbeds = guidanceEmbeds {
            let guidanceScale = isGuidanceEmbedEnabled ? textGuidanceScale : guidanceEmbed
            let guidanceEmbed = graph.variable(
              Tensor<FloatType>(
                from: timeEmbedding(
                  timestep: guidanceScale * 1_000, batchSize: cBatchSize, embeddingSize: 256,
                  maxPeriod: 10_000)
              ).toGPU(0))
            guidanceEmbeds[(i * cBatchSize)..<((i + 1) * cBatchSize), 0..<256] = guidanceEmbed
          }
        }
        restInputs = [timeEmbeds, pooleds] + (guidanceEmbeds.map { [$0] } ?? [])
      }
      unetFixed.maxConcurrency = .limit(4)
      unetFixed.compile(inputs: [c] + restInputs)
      let loadedFromWeightsCache = weightsCache.detach(
        "\(filePath):[fixed]", to: unetFixed.parameters)
      graph.openStore(
        filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
      ) { store in
        if !lora.isEmpty {
          if shouldRunLoRASeparately {
            let mapping: [Int: Int] = [Int: Int](
              uniqueKeysWithValues: (0..<(19 + 38)).map {
                return ($0, $0)
              })
            LoRALoader.openStore(graph, lora: lora) { loader in
              store.read("dit", model: unetFixed, codec: [.jit, .q6p, .q8p, .ezm7, externalData]) {
                name, dataType, format, shape in
                let result = loader.concatenateLoRA(
                  graph, LoRAMapping: mapping, filesRequireMerge: filesRequireMerge, name: name,
                  store: store, dataType: dataType, format: format, shape: shape, of: FloatType.self
                )
                switch result {
                case .continue(let updatedName, _, _):
                  guard updatedName == name else { return result }
                  if !loadedFromWeightsCache {
                    return result
                  } else {
                    return .fail  // Don't need to load.
                  }
                case .fail, .final(_):
                  return result
                }
              }
            }
          } else {
            LoRALoader.openStore(graph, lora: lora) { loader in
              store.read("dit", model: unetFixed, codec: [.jit, .q6p, .q8p, .ezm7, externalData]) {
                name, dataType, _, shape in
                return loader.mergeLoRA(
                  graph, name: name, store: store, dataType: dataType, shape: shape,
                  of: FloatType.self)
              }
            }
          }
        } else if !loadedFromWeightsCache {
          store.read("dit", model: unetFixed, codec: [.jit, .q6p, .q8p, .ezm7, externalData])
        }
      }
      let conditions = unetFixed(inputs: c, restInputs).map { $0.as(of: FloatType.self) }
      if lora.isEmpty || shouldRunLoRASeparately {
        weightsCache.attach("\(filePath):[fixed]", from: unetFixed.parameters)
      }
      let h = startHeight / 2
      let w = startWidth / 2
      let rot = Tensor<FloatType>(
        from: Flux1RotaryPositionEmbedding(
          height: h, width: w, tokenLength: t5Length, channels: 128)
      ).toGPU(0)
      return ([graph.variable(rot)] + conditions, nil)
    case .hunyuanVideo:
      let c0 = textEncoding[0]
      let pooled = textEncoding[1]
      let cBatchSize = c0.shape[0]
      let llama3Length = c0.shape[1]
      let h = startHeight / 2
      let w = startWidth / 2
      let embeddings = HunyuanRotaryPositionEmbedding(
        height: h, width: w, time: batchSize, tokenLength: llama3Length, channels: 128)
      let (rot0, rot1) = (
        Tensor<FloatType>(from: embeddings.0).toGPU(0),
        Tensor<FloatType>(from: embeddings.1).toGPU(0)
      )
      let isGuidanceEmbedSupported =
        (try?
          (graph.openStore(
            filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
          ) {
            return $0.read(like: "__dit__[t-guidance_embedder_0-0-1]") != nil
          }).get()) ?? false
      let guidanceEmbeds: DynamicGraph.Tensor<FloatType>?
      if isGuidanceEmbedSupported {
        let guidanceScale = isGuidanceEmbedEnabled ? textGuidanceScale : guidanceEmbed
        guidanceEmbeds = graph.variable(
          Tensor<FloatType>(
            from: timeEmbedding(
              timestep: guidanceScale * 1_000, batchSize: 1, embeddingSize: 256,
              maxPeriod: 10_000)
          ).toGPU(0))
      } else {
        guidanceEmbeds = nil
      }
      var timeEmbeds = graph.variable(
        .GPU(0), .WC(timesteps.count * cBatchSize, 256), of: FloatType.self)
      var c = graph.variable(
        .GPU(0),
        .WC(timesteps.count * ((isCfgEnabled ? tokenLengthUncond : 0) + tokenLengthCond), 4096),
        of: FloatType.self)
      var expandedPooled = pooled
      var c00 = c0.reshaped(.WC(llama3Length, 4096))
      var c01 = c00
      if isCfgEnabled {
        expandedPooled = graph.variable(
          .GPU(0), .WC(timesteps.count * cBatchSize, 768), of: FloatType.self)
        c00 = c0[0..<1, 0..<tokenLengthUncond, 0..<4096].copied().reshaped(
          .WC(tokenLengthUncond, 4096))
        c01 = c0[1..<2, 0..<tokenLengthCond, 0..<4096].copied().reshaped(.WC(tokenLengthCond, 4096))
      }
      for (i, timestep) in timesteps.enumerated() {
        let timeEmbed = graph.variable(
          Tensor<FloatType>(
            from: timeEmbedding(
              timestep: timestep, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000)
          ).toGPU(0))
        if isCfgEnabled {
          timeEmbeds[i..<(i + 1), 0..<256] = timeEmbed
          timeEmbeds[(i + timesteps.count)..<(i + timesteps.count + 1), 0..<256] = timeEmbed
          expandedPooled[i..<(i + 1), 0..<768] = pooled[0..<1, 0..<768]
          expandedPooled[(i + timesteps.count)..<(i + timesteps.count + 1), 0..<768] =
            pooled[1..<2, 0..<768]
          c[(i * tokenLengthUncond)..<((i + 1) * tokenLengthUncond), 0..<4096] = c00
          c[
            (i * tokenLengthCond + tokenLengthUncond * timesteps.count)..<((i + 1) * tokenLengthCond
              + tokenLengthUncond * timesteps.count), 0..<4096] = c01
        } else {
          timeEmbeds[i..<(i + 1), 0..<256] = timeEmbed
          c[(i * llama3Length)..<((i + 1) * llama3Length), 0..<4096] = c01
        }
      }
      let unetFixed: Model
      let lora = Array(
        (OrderedDictionary<String, LoRAConfiguration>(
          lora.filter({ $0.version == version }).map {
            ($0.file, $0)
          }
        ) {
          LoRAConfiguration(
            file: $0.file, weight: $0.weight + $1.weight, version: $0.version, isLoHa: $0.isLoHa,
            modifier: $0.modifier)
        })
        .values
      ).filter { $0.weight != 0 }
      let (rankOfLoRA, filesRequireMerge) = LoRALoader.rank(
        graph, of: lora.map { $0.file }, modelFile: filePath)
      let isLoHa = lora.contains { $0.isLoHa }
      var configuration = LoRANetworkConfiguration(rank: rankOfLoRA, scale: 1, highPrecision: false)
      let runLoRASeparatelyIsPreferred = isQuantizedModel || externalOnDemand
      let shouldRunLoRASeparately =
        !lora.isEmpty && !isLoHa && runLoRASeparatelyIsPreferred && rankOfLoRA > 0
        && canRunLoRASeparately
      if shouldRunLoRASeparately {
        let keys = LoRALoader.keys(graph, of: lora.map { $0.file }, modelFile: filePath)
        configuration.keys = keys
        unetFixed =
          LoRAHunyuanFixed(
            timesteps: timesteps.count, channels: 3072, layers: (20, 40),
            textLength: (isCfgEnabled ? tokenLengthUncond : 0, tokenLengthCond),
            LoRAConfiguration: configuration
          ).1
      } else {
        unetFixed =
          HunyuanFixed(
            timesteps: timesteps.count, channels: 3072, layers: (20, 40),
            textLength: (isCfgEnabled ? tokenLengthUncond : 0, tokenLengthCond)
          ).1
      }
      unetFixed.maxConcurrency = .limit(4)
      unetFixed.compile(
        inputs: [c, timeEmbeds, expandedPooled] + (guidanceEmbeds.map { [$0] } ?? []))
      let loadedFromWeightsCache = weightsCache.detach(
        "\(filePath):[fixed]", to: unetFixed.parameters)
      graph.openStore(
        filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
      ) { store in
        if !lora.isEmpty {
          if !isLoHa && runLoRASeparatelyIsPreferred && rankOfLoRA > 0 && canRunLoRASeparately {
            let mapping: [Int: Int] = [Int: Int](
              uniqueKeysWithValues: (0..<(20 + 40)).map {
                return ($0, $0)
              })
            LoRALoader.openStore(graph, lora: lora) { loader in
              store.read("dit", model: unetFixed, codec: [.jit, .q6p, .q8p, .ezm7, externalData]) {
                name, dataType, format, shape in
                let result = loader.concatenateLoRA(
                  graph, LoRAMapping: mapping, filesRequireMerge: filesRequireMerge, name: name,
                  store: store, dataType: dataType, format: format, shape: shape, of: FloatType.self
                )
                switch result {
                case .continue(let updatedName, _, _):
                  guard updatedName == name else { return result }
                  if !loadedFromWeightsCache {
                    return result
                  } else {
                    return .fail  // Skip loading.
                  }
                case .fail, .final(_):
                  return result
                }
              }
            }
          } else {
            LoRALoader.openStore(graph, lora: lora) { loader in
              store.read("dit", model: unetFixed, codec: [.jit, .q6p, .q8p, .ezm7, externalData]) {
                name, dataType, _, shape in
                return loader.mergeLoRA(
                  graph, name: name, store: store, dataType: dataType, shape: shape,
                  of: FloatType.self)
              }
            }
          }
        } else if !loadedFromWeightsCache {
          store.read("dit", model: unetFixed, codec: [.jit, .q6p, .q8p, .ezm7, externalData])
        }
      }
      let conditions = unetFixed(
        inputs: c, [timeEmbeds, expandedPooled] + (guidanceEmbeds.map { [$0] } ?? [])
      ).map { $0.as(of: FloatType.self) }
      if lora.isEmpty || shouldRunLoRASeparately {
        weightsCache.attach("\(filePath):[fixed]", from: unetFixed.parameters)
      }
      return (
        [graph.variable(rot0), graph.variable(rot1)] + conditions, nil
      )
    case .wan21_1_3b, .wan21_14b:
      let h = startHeight / 2
      let w = startWidth / 2
      let rot = Tensor<FloatType>(
        from: WanRotaryPositionEmbedding(
          height: h, width: w, time: batchSize, channels: 128)
      ).toGPU(0)
      let c0 = textEncoding[0]
      let textLength = max(c0.shape[1], 512)
      var c = graph.variable(
        .GPU(0), .HWC(isCfgEnabled ? 2 : 1, textLength, 4_096), of: FloatType.self)
      c.full(0)
      if isCfgEnabled {
        c[0..<2, 0..<c0.shape[1], 0..<4096] = c0
      } else {
        c[0..<1, 0..<c0.shape[1], 0..<4096] = c0[
          (c0.shape[0] - 1)..<c0.shape[0], 0..<c0.shape[1], 0..<4096
        ].copied()
      }
      var timeEmbeds = graph.variable(.GPU(0), .WC(timesteps.count, 256), of: Float.self)
      for (i, timestep) in timesteps.enumerated() {
        let timeEmbed = graph.variable(
          timeEmbedding(
            timestep: timestep, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000
          ).toGPU(0))
        timeEmbeds[i..<(i + 1), 0..<256] = timeEmbed
      }
      let injectImage = textEncoding.count >= 2
      let c1: DynamicGraph.Tensor<FloatType>? = injectImage ? textEncoding[1] : nil
      let unetFixed: Model
      let lora = Array(
        (OrderedDictionary<String, LoRAConfiguration>(
          lora.filter({ $0.version == version }).map {
            ($0.file, $0)
          }
        ) {
          LoRAConfiguration(
            file: $0.file, weight: $0.weight + $1.weight, version: $0.version, isLoHa: $0.isLoHa,
            modifier: $0.modifier)
        })
        .values
      ).filter { $0.weight != 0 }
      let (rankOfLoRA, filesRequireMerge) = LoRALoader.rank(
        graph, of: lora.map { $0.file }, modelFile: filePath)
      let isLoHa = lora.contains { $0.isLoHa }
      var configuration = LoRANetworkConfiguration(rank: rankOfLoRA, scale: 1, highPrecision: false)
      let runLoRASeparatelyIsPreferred = isQuantizedModel || externalOnDemand
      let shouldRunLoRASeparately =
        !lora.isEmpty && !isLoHa && runLoRASeparatelyIsPreferred && rankOfLoRA > 0
        && canRunLoRASeparately
      let vaceContext: DynamicGraph.Tensor<FloatType>? =
        (injectedControls.first {
          $0.model.type == .controlnet
        })?.hints.first?.0.first
      let vaceLayers: [Int]
      if version == .wan21_1_3b {
        vaceLayers = vaceContext == nil ? [] : (0..<15).map { $0 * 2 }
      } else {
        vaceLayers = vaceContext == nil ? [] : (0..<8).map { $0 * 5 }
      }
      if shouldRunLoRASeparately {
        let keys = LoRALoader.keys(graph, of: lora.map { $0.file }, modelFile: filePath)
        configuration.keys = keys
        if version == .wan21_1_3b {
          unetFixed =
            LoRAWanFixed(
              timesteps: timesteps.count, batchSize: (isCfgEnabled ? 2 : 1, 1), channels: 1_536,
              layers: 30, vaceLayers: vaceLayers, textLength: textLength, injectImage: injectImage,
              LoRAConfiguration: configuration
            ).1
        } else {
          unetFixed =
            LoRAWanFixed(
              timesteps: timesteps.count, batchSize: (isCfgEnabled ? 2 : 1, 1), channels: 5_120,
              layers: 40, vaceLayers: vaceLayers, textLength: textLength, injectImage: injectImage,
              LoRAConfiguration: configuration
            ).1
        }
      } else {
        if version == .wan21_1_3b {
          unetFixed =
            WanFixed(
              timesteps: timesteps.count, batchSize: (isCfgEnabled ? 2 : 1, 1), channels: 1_536,
              layers: 30, vaceLayers: vaceLayers, textLength: textLength, injectImage: injectImage
            ).1
        } else {
          unetFixed =
            WanFixed(
              timesteps: timesteps.count, batchSize: (isCfgEnabled ? 2 : 1, 1), channels: 5_120,
              layers: 40, vaceLayers: vaceLayers, textLength: textLength, injectImage: injectImage
            ).1
        }
      }
      unetFixed.maxConcurrency = .limit(4)
      unetFixed.compile(
        inputs: [c, timeEmbeds] + (vaceContext.map { [$0] } ?? []) + (c1.map { [$0] } ?? []))
      let loadedFromWeightsCache = weightsCache.detach(
        "\(filePath):[fixed]", to: unetFixed.parameters)
      graph.openStore(
        filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
      ) { store in
        ControlModelLoader<FloatType>.openStore(
          graph, injectControlModels: injectedControls.map { $0.model },
          version: version
        ) { controlModelLoader in
          if !lora.isEmpty {
            if shouldRunLoRASeparately {
              let mapping: [Int: Int] = [Int: Int](
                uniqueKeysWithValues: (0..<40).map {
                  return ($0, $0)
                })
              LoRALoader.openStore(graph, lora: lora) { loader in
                store.read("dit", model: unetFixed, codec: [.jit, .q6p, .q8p, .ezm7, externalData])
                {
                  name, dataType, format, shape in
                  if let result = controlModelLoader.loadMergedWeight(name: name) {
                    return result
                  }
                  let result: DynamicGraph.Store.ModelReaderResult
                  if dataType == .Float32 {
                    // Keeping at higher precision for LoRA loading.
                    result = loader.concatenateLoRA(
                      graph, LoRAMapping: mapping, filesRequireMerge: filesRequireMerge, name: name,
                      store: store, dataType: dataType, format: format, shape: shape,
                      of: Float32.self
                    )
                  } else {
                    result = loader.concatenateLoRA(
                      graph, LoRAMapping: mapping, filesRequireMerge: filesRequireMerge, name: name,
                      store: store, dataType: dataType, format: format, shape: shape,
                      of: FloatType.self)
                  }
                  switch result {
                  case .continue(let updatedName, _, _):
                    guard updatedName == name else { return result }
                    if !loadedFromWeightsCache {
                      return result
                    } else {
                      return .fail  // Skip loading.
                    }
                  case .fail, .final(_):
                    return result
                  }
                }
              }
            } else {
              LoRALoader.openStore(graph, lora: lora) { loader in
                store.read("dit", model: unetFixed, codec: [.jit, .q6p, .q8p, .ezm7, externalData])
                {
                  name, dataType, _, shape in
                  if let result = controlModelLoader.loadMergedWeight(name: name) {
                    return result
                  }
                  if dataType == .Float32 {
                    // Keeping at higher precision for LoRA loading.
                    return loader.mergeLoRA(
                      graph, name: name, store: store, dataType: dataType, shape: shape,
                      of: Float32.self)
                  } else {
                    return loader.mergeLoRA(
                      graph, name: name, store: store, dataType: dataType, shape: shape,
                      of: FloatType.self)
                  }
                }
              }
            }
          } else {
            store.read("dit", model: unetFixed, codec: [.jit, .q6p, .q8p, .ezm7, externalData]) {
              name, _, _, _ in
              if let result = controlModelLoader.loadMergedWeight(name: name) {
                return result
              }
              if !loadedFromWeightsCache {
                return .continue(name)
              } else {
                return .fail  // Skip.
              }
            }
          }
        }
      }
      var conditions: [DynamicGraph.AnyTensor] = unetFixed(
        inputs: c, [timeEmbeds] + (vaceContext.map { [$0] } ?? []) + (c1.map { [$0] } ?? []))
      if lora.isEmpty || shouldRunLoRASeparately {
        weightsCache.attach("\(filePath):[fixed]", from: unetFixed.parameters)
      }
      conditions = conditions.enumerated().flatMap {
        switch $0.offset {
        case 0..<(vaceContext == nil ? 6 : 7), (conditions.count - 2)..<conditions.count:
          return [$0.element]
        default:  // Need to separate them into two elements.
          let shape = $0.element.shape
          guard shape[0] > 1 else {
            return [$0.element]
          }
          let value = DynamicGraph.Tensor<FloatType>($0.element)
          return [
            value[0..<1, 0..<shape[1], 0..<shape[2], 0..<shape[3]].copied(),
            value[1..<2, 0..<shape[1], 0..<shape[2], 0..<shape[3]].copied(),
          ]
        }
      }
      if vaceContext != nil {
        // Insert empty weight into the conditions.
        let contextScale = graph.variable(.GPU(0), .C(1), of: FloatType.self)
        contextScale.full(1)
        conditions.insert(contextScale, at: 6)
      }
      return ([graph.variable(rot)] + conditions, nil)
    case .hiDreamI1:
      let h = startHeight / 2
      let w = startWidth / 2
      var pooled = textEncoding[0]
      let cBatchSize = pooled.shape[0]
      let t5 = textEncoding[1]
      let llama3 = Array(textEncoding[2...])
      let t5Length = t5.shape[1]
      let llama3Length = llama3[0].shape[1]
      let rot = Tensor<FloatType>(
        from: HiDreamRotaryPositionEmbedding(
          height: h, width: modifier == .editing ? w * 2 : w,
          tokenLength: t5Length + llama3Length * 2, channels: 128)
      ).toGPU(0)
      if zeroNegativePrompt && isCfgEnabled {
        let oldPooled = pooled
        pooled = graph.variable(like: oldPooled)
        pooled.full(0)
        pooled[batchSize..<(batchSize * 2), 0..<2048] =
          oldPooled[batchSize..<(batchSize * 2), 0..<2048]
      }
      precondition(timesteps.count > 0)
      let unetFixed: Model
      let lora = Array(
        (OrderedDictionary<String, LoRAConfiguration>(
          lora.filter({ $0.version == version }).map {
            ($0.file, $0)
          }
        ) {
          LoRAConfiguration(
            file: $0.file, weight: $0.weight + $1.weight, version: $0.version, isLoHa: $0.isLoHa,
            modifier: $0.modifier)
        })
        .values
      ).filter { $0.weight != 0 }
      let (rankOfLoRA, filesRequireMerge) = LoRALoader.rank(
        graph, of: lora.map { $0.file }, modelFile: filePath)
      let isLoHa = lora.contains { $0.isLoHa }
      var configuration = LoRANetworkConfiguration(rank: rankOfLoRA, scale: 1, highPrecision: false)
      let runLoRASeparatelyIsPreferred = isQuantizedModel || externalOnDemand
      let shouldRunLoRASeparately =
        !lora.isEmpty && !isLoHa && runLoRASeparatelyIsPreferred && rankOfLoRA > 0
        && canRunLoRASeparately
      let isTeaCacheEnabled = teaCacheConfiguration.threshold > 0
      if shouldRunLoRASeparately {
        let keys = LoRALoader.keys(graph, of: lora.map { $0.file }, modelFile: filePath)
        configuration.keys = keys
        (unetFixed, _) = LoRAHiDreamFixed(
          timesteps: cBatchSize * timesteps.count, layers: (16, 32),
          outputTimesteps: isTeaCacheEnabled,
          LoRAConfiguration: configuration)
      } else {
        (unetFixed, _) = HiDreamFixed(
          timesteps: cBatchSize * timesteps.count, layers: (16, 32),
          outputTimesteps: isTeaCacheEnabled)
      }
      var timeEmbeds = graph.variable(
        .GPU(0), .WC(cBatchSize * timesteps.count, 256), of: FloatType.self)
      var pooleds = graph.variable(
        .GPU(0), .WC(cBatchSize * timesteps.count, 2048), of: FloatType.self)
      for (i, timestep) in timesteps.enumerated() {
        let timeEmbed = graph.variable(
          Tensor<FloatType>(
            from: timeEmbedding(
              timestep: timestep, batchSize: cBatchSize, embeddingSize: 256, maxPeriod: 10_000)
          ).toGPU(0))
        timeEmbeds[(i * cBatchSize)..<((i + 1) * cBatchSize), 0..<256] = timeEmbed
        pooleds[(i * cBatchSize)..<((i + 1) * cBatchSize), 0..<2048] = pooled
      }
      unetFixed.maxConcurrency = .limit(4)
      unetFixed.compile(inputs: [timeEmbeds, pooleds, t5] + llama3)
      let loadedFromWeightsCache = weightsCache.detach(
        "\(filePath):[fixed]", to: unetFixed.parameters)
      graph.openStore(
        filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
      ) { store in
        if !lora.isEmpty {
          if shouldRunLoRASeparately {
            let mapping: [Int: Int] = [Int: Int](
              uniqueKeysWithValues: (0..<48).map {
                return ($0, $0)
              })
            LoRALoader.openStore(graph, lora: lora) { loader in
              store.read("dit", model: unetFixed, codec: [.jit, .q6p, .q8p, .ezm7, externalData]) {
                name, dataType, format, shape in
                let result = loader.concatenateLoRA(
                  graph, LoRAMapping: mapping, filesRequireMerge: filesRequireMerge, name: name,
                  store: store, dataType: dataType, format: format, shape: shape, of: FloatType.self
                )
                switch result {
                case .continue(let updatedName, _, _):
                  guard updatedName == name else { return result }
                  if !loadedFromWeightsCache {
                    return result
                  } else {
                    return .fail  // Skip loading.
                  }
                case .fail, .final(_):
                  return result
                }
              }
            }
          } else {
            LoRALoader.openStore(graph, lora: lora) { loader in
              store.read("dit", model: unetFixed, codec: [.jit, .q6p, .q8p, .ezm7, externalData]) {
                name, dataType, _, shape in
                return loader.mergeLoRA(
                  graph, name: name, store: store, dataType: dataType, shape: shape,
                  of: FloatType.self)
              }
            }
          }
        } else if !loadedFromWeightsCache {
          store.read("dit", model: unetFixed, codec: [.jit, .q6p, .q8p, .ezm7, externalData])
        }
      }
      let conditions = unetFixed(
        inputs: timeEmbeds, [pooleds, t5] + llama3
      )
      weightsCache.attach("\(filePath):[fixed]", from: unetFixed.parameters)
      return ([graph.variable(rot)] + conditions, nil)
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
      stageBFixed.maxConcurrency = .limit(4)
      stageBFixed.compile(
        inputs: effnet, pixels, textEncoding[1].reshaped(.HWC(cfgChannelsAndBatchSize, 1, 1280)))
      graph.openStore(
        filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
      ) { store in
        if lora.count > 0 {
          LoRALoader.openStore(graph, lora: lora) { loader in
            store.read(
              "stage_b_fixed", model: stageBFixed, codec: [.q6p, .q8p, .ezm7, .jit, externalData]
            ) {
              name, dataType, _, shape in
              return loader.mergeLoRA(
                graph, name: name, store: store, dataType: dataType, shape: shape,
                of: FloatType.self)
            }
          }
        } else {
          store.read(
            "stage_b_fixed", model: stageBFixed, codec: [.q6p, .q8p, .ezm7, .jit, externalData]
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
