import NNC

public struct FirstStage<FloatType: TensorNumeric & BinaryFloatingPoint> {
  public let filePath: String
  public let version: ModelVersion
  public let highPrecisionKeysAndValues: Bool
  public let externalOnDemand: Bool
  public let tiledDecoding: TiledConfiguration
  public let tiledDiffusion: TiledConfiguration
  private let alternativeUsesFlashAttention: Bool
  private let alternativeFilePath: String?
  private let alternativeDecoderVersion: AlternativeDecoderVersion?
  private let latentsScaling:
    (mean: [Float]?, std: [Float]?, scalingFactor: Float, shiftFactor: Float?)
  private let highPrecisionFallback: Bool
  public init(
    filePath: String, version: ModelVersion,
    latentsScaling: (mean: [Float]?, std: [Float]?, scalingFactor: Float, shiftFactor: Float?),
    highPrecisionKeysAndValues: Bool, highPrecisionFallback: Bool,
    tiledDecoding: TiledConfiguration,
    tiledDiffusion: TiledConfiguration, externalOnDemand: Bool, alternativeUsesFlashAttention: Bool,
    alternativeFilePath: String?, alternativeDecoderVersion: AlternativeDecoderVersion?
  ) {
    self.filePath = filePath
    self.version = version
    self.latentsScaling = latentsScaling
    self.highPrecisionKeysAndValues = highPrecisionKeysAndValues
    self.highPrecisionFallback = highPrecisionFallback
    self.externalOnDemand = externalOnDemand
    self.tiledDecoding = tiledDecoding
    self.tiledDiffusion = tiledDiffusion
    self.alternativeUsesFlashAttention = alternativeUsesFlashAttention
    self.alternativeFilePath = alternativeFilePath
    self.alternativeDecoderVersion = alternativeDecoderVersion
  }
}

extension FirstStage {
  private func decode(
    _ x: DynamicGraph.Tensor<FloatType>, decoder existingDecoder: Model?, highPrecision: Bool
  )
    -> (DynamicGraph.Tensor<FloatType>, Model)
  {
    let shape = x.shape
    let batchSize = shape[0]
    let startHeight = shape[1]
    let startWidth = shape[2]
    let graph = x.graph
    let scalingFactor = latentsScaling.scalingFactor
    let z: DynamicGraph.Tensor<FloatType>
    if let latentsMean = latentsScaling.mean, let latentsStd = latentsScaling.std,
      latentsMean.count >= 4, latentsStd.count >= 4
    {
      let mean = graph.variable(
        Tensor<FloatType>(latentsMean.map { FloatType($0) }, .GPU(0), .NHWC(1, 1, 1, 4)))
      let std = graph.variable(
        Tensor<FloatType>(
          latentsStd.map { FloatType($0 / scalingFactor) }, .GPU(0), .NHWC(1, 1, 1, 4)))
      z = std .* x + mean
    } else if let shiftFactor = latentsScaling.shiftFactor {
      z = x / scalingFactor + shiftFactor
    } else {
      z = x / scalingFactor
    }
    let decoder: Model
    var transparentDecoder: Model? = nil
    let queueWatermark = DynamicGraph.queueWatermark
    if version == .kandinsky21 || version == .hunyuanVideo {
      DynamicGraph.queueWatermark = min(0, queueWatermark)
    }
    defer {
      if version == .kandinsky21 || version == .hunyuanVideo {
        DynamicGraph.queueWatermark = queueWatermark
      }
    }
    let scaleFactorZ: Int
    let scaleFactor: Int
    switch version {
    case .v1, .v2, .sd3, .sd3Large, .pixart, .auraflow, .flux1, .sdxlBase, .sdxlRefiner, .ssd1b,
      .svdI2v, .kandinsky21:
      scaleFactor = 8
      scaleFactorZ = 1
    case .hunyuanVideo:
      scaleFactor = 8
      scaleFactorZ = 4
    case .wurstchenStageB, .wurstchenStageC:
      scaleFactor = 4
      scaleFactorZ = 1
    }
    var decodingTileSize = (
      depth: 0,
      width: min(tiledDecoding.tileSize.width * (64 / scaleFactor), startWidth),
      height: min(tiledDecoding.tileSize.height * (64 / scaleFactor), startHeight)
    )
    // Cut by half because we will * 2 later.
    var decodingTileOverlap = (tiledDecoding.tileOverlap * (64 / scaleFactor) + 1) / 2
    var tiledDecoding =
      tiledDecoding.isEnabled
      && (startWidth > decodingTileSize.width || startHeight > decodingTileSize.height)
    let externalData: DynamicGraph.Store.Codec =
      externalOnDemand ? .externalOnDemand : .externalData
    let outputChannels: Int
    let causalAttentionMask: DynamicGraph.Tensor<Float>?
    switch version {
    case .v1, .v2, .sdxlBase, .sdxlRefiner, .ssd1b, .svdI2v, .pixart, .auraflow:
      let startWidth = tiledDecoding ? decodingTileSize.width : startWidth
      let startHeight = tiledDecoding ? decodingTileSize.height : startHeight
      decoder =
        existingDecoder
        ?? Decoder(
          channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: startWidth,
          startHeight: startHeight, highPrecisionKeysAndValues: highPrecisionKeysAndValues,
          usesFlashAttention: false, paddingFinalConvLayer: true
        ).0
      if existingDecoder == nil {
        decoder.maxConcurrency = .limit(4)
        if highPrecision {
          decoder.compile(
            inputs: DynamicGraph.Tensor<Float>(
              from: z[0..<1, 0..<startHeight, 0..<startWidth, 0..<shape[3]]))
        } else {
          decoder.compile(inputs: z[0..<1, 0..<startHeight, 0..<startWidth, 0..<shape[3]])
        }
        graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) {
          $0.read("decoder", model: decoder, codec: [.jit, externalData])
        }
      }
      if let alternativeFilePath = alternativeFilePath, alternativeDecoderVersion == .transparent {
        let decoder = TransparentVAEDecoder(
          startHeight: startHeight * 8, startWidth: startWidth * 8,
          usesFlashAttention: alternativeUsesFlashAttention ? .scaleMerged : .none)
        decoder.maxConcurrency = .limit(4)
        if highPrecision {
          let pixels = graph.variable(
            .GPU(0), .NHWC(1, startHeight * 8, startWidth * 8, 3), of: Float.self)
          decoder.compile(
            inputs: pixels,
            DynamicGraph.Tensor<Float>(
              from: z[0..<1, 0..<startHeight, 0..<startWidth, 0..<shape[3]]))
        } else {
          let pixels = graph.variable(
            .GPU(0), .NHWC(1, startHeight * 8, startWidth * 8, 3), of: FloatType.self)
          decoder.compile(inputs: pixels, z[0..<1, 0..<startHeight, 0..<startWidth, 0..<shape[3]])
        }
        graph.openStore(
          alternativeFilePath, flags: .readOnly,
          externalStore: TensorData.externalStore(filePath: alternativeFilePath)
        ) {
          $0.read("decoder", model: decoder, codec: [.jit, .ezm7, externalData])
        }
        transparentDecoder = decoder
        outputChannels = 4
      } else {
        outputChannels = 3
      }
      causalAttentionMask = nil
    case .sd3, .sd3Large, .flux1:
      let startWidth = tiledDecoding ? decodingTileSize.width : startWidth
      let startHeight = tiledDecoding ? decodingTileSize.height : startHeight
      decoder =
        existingDecoder
        ?? Decoder(
          channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: startWidth,
          startHeight: startHeight, highPrecisionKeysAndValues: highPrecisionKeysAndValues,
          usesFlashAttention: false, paddingFinalConvLayer: true,
          quantLayer: false
        ).0
      if existingDecoder == nil {
        decoder.maxConcurrency = .limit(4)
        if highPrecision {
          decoder.compile(
            inputs: DynamicGraph.Tensor<Float>(
              from: z[0..<1, 0..<startHeight, 0..<startWidth, 0..<shape[3]]))
        } else {
          decoder.compile(inputs: z[0..<1, 0..<startHeight, 0..<startWidth, 0..<shape[3]])
        }
        graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) {
          $0.read("decoder", model: decoder, codec: [.jit, externalData])
        }
      }
      outputChannels = 3
      causalAttentionMask = nil
    case .hunyuanVideo:
      var startDepth = shape[0]
      var startWidth = tiledDecoding ? decodingTileSize.width : startWidth
      var startHeight = tiledDecoding ? decodingTileSize.height : startHeight
      if startWidth > 32 || startHeight > 32 || startDepth > 15 {
        // We turn on tiled decoding forcefully.
        if !tiledDecoding {
          decodingTileOverlap = 4
        }
        tiledDecoding = true
        startWidth = min(startWidth, 32)
        startHeight = min(startHeight, 32)
        decodingTileSize.width = startWidth
        decodingTileSize.height = startHeight
        startDepth = min(startDepth, 15)
        decodingTileSize.depth = startDepth
      }
      decoder =
        existingDecoder
        ?? DecoderCausal3D(
          channels: [128, 256, 512, 512], numRepeat: 2, startWidth: startWidth,
          startHeight: startHeight, startDepth: startDepth
        ).1
      var mask = Tensor<Float>(
        Array(repeating: 0, count: startDepth * startDepth), .CPU,
        .NHWC(startDepth, 1, startDepth, 1))
      for i in 0..<(startDepth - 1) {
        for j in (i + 1)..<startDepth {
          mask[i, 0, j, 0] = -Float.greatestFiniteMagnitude
        }
      }
      causalAttentionMask = graph.variable(mask.toGPU(0))
      if existingDecoder == nil {
        decoder.maxConcurrency = .limit(4)
        if highPrecision {
          decoder.compile(
            inputs: [
              DynamicGraph.Tensor<Float>(
                from: z[0..<startDepth, 0..<startHeight, 0..<startWidth, 0..<shape[3]])
            ] + (causalAttentionMask.map { [$0] } ?? []))
        } else {
          decoder.compile(
            inputs: [z[0..<startDepth, 0..<startHeight, 0..<startWidth, 0..<shape[3]]]
              + (causalAttentionMask.map { [DynamicGraph.Tensor<FloatType>(from: $0)] } ?? []))
        }
        graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) {
          $0.read("decoder", model: decoder, codec: [.jit, externalData])
        }
      }
      outputChannels = 3
    case .kandinsky21:
      let startWidth = tiledDecoding ? decodingTileSize.width : startWidth
      let startHeight = tiledDecoding ? decodingTileSize.height : startHeight
      decoder =
        existingDecoder
        ?? MOVQDecoderKandinsky(
          zChannels: 4, channels: 128, channelMult: [1, 2, 2, 4], numResBlocks: 2,
          startHeight: startHeight, startWidth: startWidth, attnResolutions: Set([32]))
      if existingDecoder == nil {
        decoder.maxConcurrency = .limit(4)
        if highPrecision {
          decoder.compile(
            inputs: DynamicGraph.Tensor<Float>(
              from: z[0..<1, 0..<startHeight, 0..<startWidth, 0..<shape[3]]))
        } else {
          decoder.compile(inputs: z[0..<1, 0..<startHeight, 0..<startWidth, 0..<shape[3]])
        }
        graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) {
          $0.read("movq", model: decoder, codec: [.jit, externalData])
        }
      }
      outputChannels = 3
      causalAttentionMask = nil
    case .wurstchenStageC, .wurstchenStageB:
      let startWidth = tiledDecoding ? decodingTileSize.width : startWidth
      let startHeight = tiledDecoding ? decodingTileSize.height : startHeight
      decoder =
        existingDecoder
        ?? WurstchenStageADecoder(batchSize: 1, height: startHeight * 2, width: startWidth * 2).0
      if existingDecoder == nil {
        decoder.maxConcurrency = .limit(4)
        if highPrecision {
          decoder.compile(
            inputs: DynamicGraph.Tensor<Float>(
              from: z[0..<1, 0..<startHeight, 0..<startWidth, 0..<shape[3]]))
        } else {
          decoder.compile(inputs: z[0..<1, 0..<startHeight, 0..<startWidth, 0..<shape[3]])
        }
        graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) {
          $0.read("decoder", model: decoder, codec: [.jit, externalData])
        }
      }
      outputChannels = 3
      causalAttentionMask = nil
    }
    // Hunyuan just do the decoding with the batch.
    guard batchSize > 1 && version != .hunyuanVideo else {
      if highPrecision {
        let result: DynamicGraph.Tensor<Float>
        if tiledDecoding {
          result = tiledDecode(
            DynamicGraph.Tensor<Float>(from: z), causalAttentionMask: causalAttentionMask,
            decoder: decoder,
            transparentDecoder: transparentDecoder, tileSize: decodingTileSize,
            tileOverlap: decodingTileOverlap, outputChannels: outputChannels,
            scaleFactor: (scaleFactor, scaleFactorZ)
          )
        } else {
          result = internalDecode(
            DynamicGraph.Tensor<Float>(from: z), causalAttentionMask: causalAttentionMask,
            decoder: decoder,
            transparentDecoder: transparentDecoder)
        }
        let shape = result.shape
        return (
          DynamicGraph.Tensor<FloatType>(
            from: result[0..<shape[0], 0..<shape[1], 0..<shape[2], 0..<outputChannels]),
          decoder
        )
      } else {
        let result: DynamicGraph.Tensor<FloatType>
        if tiledDecoding {
          result = tiledDecode(
            z,
            causalAttentionMask: causalAttentionMask.map {
              DynamicGraph.Tensor<FloatType>(from: $0)
            }, decoder: decoder, transparentDecoder: transparentDecoder, tileSize: decodingTileSize,
            tileOverlap: decodingTileOverlap, outputChannels: outputChannels,
            scaleFactor: (scaleFactor, scaleFactorZ)
          )
        } else {
          result = internalDecode(
            z,
            causalAttentionMask: causalAttentionMask.map {
              DynamicGraph.Tensor<FloatType>(from: $0)
            }, decoder: decoder, transparentDecoder: transparentDecoder)
        }
        let shape = result.shape
        return (
          result[0..<shape[0], 0..<shape[1], 0..<shape[2], 0..<outputChannels].copied(), decoder
        )
      }
    }
    var result = graph.variable(
      .GPU(0),
      .NHWC(batchSize, startHeight * scaleFactor, startWidth * scaleFactor, outputChannels),
      of: FloatType.self)
    for i in 0..<batchSize {
      let zEnc = z[i..<(i + 1), 0..<startHeight, 0..<startWidth, 0..<shape[3]].copied()
      if highPrecision {
        let partial: DynamicGraph.Tensor<Float>
        if tiledDecoding {
          partial = tiledDecode(
            DynamicGraph.Tensor<Float>(from: zEnc), causalAttentionMask: causalAttentionMask,
            decoder: decoder,
            transparentDecoder: transparentDecoder, tileSize: decodingTileSize,
            tileOverlap: decodingTileOverlap, outputChannels: outputChannels,
            scaleFactor: (scaleFactor, scaleFactorZ)
          )
        } else {
          partial = internalDecode(
            DynamicGraph.Tensor<Float>(from: zEnc), causalAttentionMask: causalAttentionMask,
            decoder: decoder,
            transparentDecoder: transparentDecoder)
        }
        let shape = partial.shape
        result[
          i..<(i + 1), 0..<(startHeight * scaleFactor), 0..<(startWidth * scaleFactor),
          0..<outputChannels] =
          DynamicGraph
          .Tensor<FloatType>(from: partial[0..<1, 0..<shape[1], 0..<shape[2], 0..<outputChannels])
      } else {
        let partial: DynamicGraph.Tensor<FloatType>
        if tiledDecoding {
          partial = tiledDecode(
            zEnc,
            causalAttentionMask: causalAttentionMask.map {
              DynamicGraph.Tensor<FloatType>(from: $0)
            }, decoder: decoder, transparentDecoder: transparentDecoder,
            tileSize: decodingTileSize, tileOverlap: decodingTileOverlap,
            outputChannels: outputChannels, scaleFactor: (scaleFactor, scaleFactorZ))
        } else {
          partial = internalDecode(
            zEnc,
            causalAttentionMask: causalAttentionMask.map {
              DynamicGraph.Tensor<FloatType>(from: $0)
            }, decoder: decoder, transparentDecoder: transparentDecoder)
        }
        let shape = partial.shape
        result[
          i..<(i + 1), 0..<(startHeight * scaleFactor), 0..<(startWidth * scaleFactor),
          0..<outputChannels] =
          partial[0..<1, 0..<shape[1], 0..<shape[2], 0..<outputChannels]
      }
    }
    return (result, decoder)
  }

  public func decode(_ x: DynamicGraph.Tensor<FloatType>, decoder existingDecoder: Model?)
    -> (DynamicGraph.Tensor<FloatType>, Model)
  {
    let (result, decoder) = decode(x, decoder: existingDecoder, highPrecision: false)
    if highPrecisionFallback && isNaN(result.rawValue.toCPU()) {
      let (highPrecisionResult, _) = decode(x, decoder: nil, highPrecision: true)
      return (highPrecisionResult, decoder)
    }
    return (result, decoder)
  }

  public func sampleFromDistribution(
    _ parameters: DynamicGraph.Tensor<FloatType>, noise: DynamicGraph.Tensor<FloatType>? = nil
  ) -> (DynamicGraph.Tensor<FloatType>, DynamicGraph.Tensor<FloatType>) {
    let shape = parameters.shape
    let batchSize = shape[0]
    let startHeight = parameters.shape[1]
    let startWidth = parameters.shape[2]
    let graph = parameters.graph
    let channels = parameters.shape[3] / 2
    let mean = parameters[0..<batchSize, 0..<startHeight, 0..<startWidth, 0..<channels]
    let logvar = parameters[
      0..<batchSize, 0..<startHeight, 0..<startWidth, channels..<(channels * 2)
    ].copied().clamped(
      -30...20)
    let std = Functional.exp(0.5 * logvar)
    let n: DynamicGraph.Tensor<FloatType>
    if let noise = noise {
      n = noise
    } else {
      n = graph.variable(
        .GPU(0), .NHWC(batchSize, startHeight, startWidth, channels), of: FloatType.self)
      n.randn(std: 1, mean: 0)
    }
    return (scale(mean + std .* n), mean)
  }

  public func sample(_ x: DynamicGraph.Tensor<FloatType>, encoder: Model?) -> (
    DynamicGraph.Tensor<FloatType>, DynamicGraph.Tensor<FloatType>, Model
  ) {
    let (parameters, encoder) = encode(x, encoder: encoder)
    guard version != .kandinsky21 && version != .wurstchenStageC else {
      return (parameters, parameters, encoder)
    }
    guard version != .wurstchenStageB else {
      // For stage b, we need to scale it properly.
      let sample = scale(parameters)
      return (sample, sample, encoder)
    }
    let (sample, mean) = sampleFromDistribution(parameters)
    return (sample, mean, encoder)
  }

  public func scale(_ x: DynamicGraph.Tensor<FloatType>) -> DynamicGraph.Tensor<FloatType> {
    let graph = x.graph
    let scalingFactor = latentsScaling.scalingFactor
    if let latentsMean = latentsScaling.mean, let latentsStd = latentsScaling.std,
      latentsMean.count >= 4, latentsStd.count >= 4
    {
      let mean = graph.variable(
        Tensor<FloatType>(latentsMean.map { FloatType($0) }, .GPU(0), .NHWC(1, 1, 1, 4)))
      let invStd = graph.variable(
        Tensor<FloatType>(
          latentsStd.map { FloatType(scalingFactor / $0) }, .GPU(0), .NHWC(1, 1, 1, 4)))
      return invStd .* (x - mean)
    } else if let shiftFactor = latentsScaling.shiftFactor {
      return (x - shiftFactor) * scalingFactor
    } else {
      return x * scalingFactor
    }
  }

  public func encode(_ x: DynamicGraph.Tensor<FloatType>, encoder existingEncoder: Model?)
    -> (DynamicGraph.Tensor<FloatType>, Model)
  {
    let (result, encoder) = encode(x, encoder: existingEncoder, highPrecision: false)
    if highPrecisionFallback && isNaN(result.rawValue.toCPU()) {
      let (highPrecisionResult, _) = encode(x, encoder: nil, highPrecision: true)
      return (highPrecisionResult, encoder)
    }
    return (result, encoder)
  }

  private func encode(
    _ x: DynamicGraph.Tensor<FloatType>, encoder existingEncoder: Model?, highPrecision: Bool
  )
    -> (DynamicGraph.Tensor<FloatType>, Model)
  {
    let shape = x.shape
    let batchSize = shape[0]
    precondition(shape[1] % 32 == 0)
    precondition(shape[2] % 32 == 0)
    let graph = x.graph
    let encoder: Model
    let queueWatermark = DynamicGraph.queueWatermark
    if version == .hunyuanVideo {
      DynamicGraph.queueWatermark = min(0, queueWatermark)
    }
    defer {
      if version == .hunyuanVideo {
        DynamicGraph.queueWatermark = queueWatermark
      }
    }
    let outputChannels: Int
    var x = x
    let externalData: DynamicGraph.Store.Codec =
      externalOnDemand ? .externalOnDemand : .externalData
    let scaleFactor: Int
    let scaleFactorZ: Int
    switch version {
    case .v1, .v2, .sd3, .sd3Large, .pixart, .auraflow, .flux1, .sdxlBase, .sdxlRefiner, .ssd1b,
      .svdI2v, .kandinsky21:
      scaleFactor = 8
      scaleFactorZ = 1
    case .hunyuanVideo:
      scaleFactor = 8
      scaleFactorZ = 4
    case .wurstchenStageC:
      scaleFactor = 32
      scaleFactorZ = 1
    case .wurstchenStageB:
      scaleFactor = 4
      scaleFactorZ = 1
    }
    let startHeight = shape[1] / scaleFactor
    let startWidth = shape[2] / scaleFactor
    var encodingTileSize = (
      depth: 0,
      width: min(tiledDiffusion.tileSize.width * (64 / scaleFactor), startWidth),
      height: min(tiledDiffusion.tileSize.height * (64 / scaleFactor), startHeight)
    )
    // Cut by half because we will * 2 later.
    var encodingTileOverlap = (tiledDiffusion.tileOverlap * (64 / scaleFactor) + 1) / 2
    var tiledEncoding =
      tiledDiffusion.isEnabled
      && (startWidth > encodingTileSize.width || startHeight > encodingTileSize.height)
    let causalAttentionMask: DynamicGraph.Tensor<Float>?
    switch version {
    case .v1, .v2, .sdxlBase, .sdxlRefiner, .ssd1b, .svdI2v, .pixart, .auraflow:
      let startWidth = tiledEncoding ? encodingTileSize.width : startWidth
      let startHeight = tiledEncoding ? encodingTileSize.height : startHeight
      encoder =
        existingEncoder
        ?? Encoder(
          channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: startWidth,
          startHeight: startHeight, usesFlashAttention: false
        ).0
      if existingEncoder == nil {
        encoder.maxConcurrency = .limit(4)
        if highPrecision {
          encoder.compile(
            inputs: DynamicGraph.Tensor<Float>(
              from: x[0..<1, 0..<(startHeight * 8), 0..<(startWidth * 8), 0..<shape[3]]))
        } else {
          encoder.compile(
            inputs: x[0..<1, 0..<(startHeight * 8), 0..<(startWidth * 8), 0..<shape[3]])
        }
        graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) {
          $0.read("encoder", model: encoder, codec: [.jit, externalData])
        }
      }
      outputChannels = 8
      causalAttentionMask = nil
    case .sd3, .sd3Large, .flux1:
      let startWidth = tiledEncoding ? encodingTileSize.width : startWidth
      let startHeight = tiledEncoding ? encodingTileSize.height : startHeight
      encoder =
        existingEncoder
        ?? Encoder(
          channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: startWidth,
          startHeight: startHeight, usesFlashAttention: false, quantLayer: false, outputChannels: 16
        ).0
      // Don't use FP32 for SD3 / FLUX.1 encoding pass.
      if existingEncoder == nil {
        encoder.maxConcurrency = .limit(4)
        if highPrecision {
          encoder.compile(
            inputs: DynamicGraph.Tensor<Float>(
              from: x[0..<1, 0..<(startHeight * 8), 0..<(startWidth * 8), 0..<shape[3]]))
        } else {
          encoder.compile(
            inputs: x[0..<1, 0..<(startHeight * 8), 0..<(startWidth * 8), 0..<shape[3]])
        }
        graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) {
          $0.read("encoder", model: encoder, codec: [.jit, externalData])
        }
      }
      outputChannels = 32
      causalAttentionMask = nil
    case .hunyuanVideo:
      var startDepth = (shape[0] - 1) / 4 + 1
      var startWidth = tiledEncoding ? encodingTileSize.width : startWidth
      var startHeight = tiledEncoding ? encodingTileSize.height : startHeight
      if startWidth > 32 || startHeight > 32 || startDepth > 15 {
        // We turn on tiled decoding forcefully.
        if !tiledEncoding {
          encodingTileOverlap = 4
        }
        tiledEncoding = true
        startWidth = min(startWidth, 32)
        startHeight = min(startHeight, 32)
        encodingTileSize.width = startWidth
        encodingTileSize.height = startHeight
        startDepth = min(startDepth, 15)
        encodingTileSize.depth = startDepth
      }
      encoder =
        existingEncoder
        ?? EncoderCausal3D(
          channels: [128, 256, 512, 512], numRepeat: 2, startWidth: startWidth,
          startHeight: startHeight, startDepth: startDepth
        ).1
      var mask = Tensor<Float>(
        Array(repeating: 0, count: startDepth * startDepth), .CPU,
        .NHWC(startDepth, 1, startDepth, 1))
      for i in 0..<(startDepth - 1) {
        for j in (i + 1)..<startDepth {
          mask[i, 0, j, 0] = -Float.greatestFiniteMagnitude
        }
      }
      let batchSize = (startDepth - 1) * 4 + 1
      causalAttentionMask = graph.variable(mask.toGPU(0))
      if existingEncoder == nil {
        encoder.maxConcurrency = .limit(4)
        if highPrecision {
          encoder.compile(
            inputs: [
              DynamicGraph.Tensor<Float>(
                from: x[0..<batchSize, 0..<(startHeight * 8), 0..<(startWidth * 8), 0..<shape[3]])
            ] + (causalAttentionMask.map { [$0] } ?? []))
        } else {
          encoder.compile(
            inputs: [x[0..<batchSize, 0..<(startHeight * 8), 0..<(startWidth * 8), 0..<shape[3]]]
              + (causalAttentionMask.map { [DynamicGraph.Tensor<FloatType>(from: $0)] } ?? []))
        }
        graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) {
          $0.read("encoder", model: encoder, codec: [.jit, externalData])
        }
      }
      outputChannels = 32
    case .kandinsky21:
      let startWidth = tiledEncoding ? encodingTileSize.width : startWidth
      let startHeight = tiledEncoding ? encodingTileSize.height : startHeight
      encoder =
        existingEncoder
        ?? EncoderKandinsky(
          zChannels: 4, channels: 128, channelMult: [1, 2, 2, 4], numResBlocks: 2,
          startHeight: startHeight * 8, startWidth: startWidth * 8, attnResolutions: Set([32]))
      if existingEncoder == nil {
        encoder.maxConcurrency = .limit(4)
        encoder.compile(inputs: x[0..<1, 0..<(startHeight * 8), 0..<(startWidth * 8), 0..<shape[3]])
        graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) {
          $0.read("encoder", model: encoder, codec: [.jit, externalData])
        }
      }
      outputChannels = 4
      causalAttentionMask = nil
    case .wurstchenStageC:
      encodingTileSize.height = startHeight
      encodingTileSize.width = startWidth
      encodingTileOverlap = 0
      encoder = existingEncoder ?? EfficientNetEncoder()
      if existingEncoder == nil {
        encoder.maxConcurrency = .limit(4)
        encoder.compile(inputs: x[0..<1, 0..<shape[1], 0..<shape[2], 0..<shape[3]])
        graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) {
          $0.read("effnet", model: encoder, codec: [.q6p, .q8p, .ezm7, .jit, externalData])
        }
      }
      // Normalize from -1...1 to the EffNet range.
      let mean = graph.variable(
        Tensor<FloatType>(
          [
            FloatType(2 * 0.485 - 1), FloatType(2 * 0.456 - 1),
            FloatType(2 * 0.406 - 1),
          ], .GPU(0), .NHWC(1, 1, 1, 3)))
      let invStd = graph.variable(
        Tensor<FloatType>(
          [
            FloatType(0.5 / 0.229), FloatType(0.5 / 0.224), FloatType(0.5 / 0.225),
          ],
          .GPU(0), .NHWC(1, 1, 1, 3)))
      x = (x - mean) .* invStd
      outputChannels = 16
      causalAttentionMask = nil
    case .wurstchenStageB:
      let startWidth = tiledEncoding ? encodingTileSize.width : startWidth
      let startHeight = tiledEncoding ? encodingTileSize.height : startHeight
      encoder = existingEncoder ?? WurstchenStageAEncoder(batchSize: 1).0
      if existingEncoder == nil {
        encoder.maxConcurrency = .limit(4)
        encoder.compile(inputs: x[0..<1, 0..<(startHeight * 4), 0..<(startWidth * 4), 0..<shape[3]])
        graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) {
          $0.read("encoder", model: encoder, codec: [.jit, .q6p, .q8p, .ezm7, externalData])
        }
      }
      x = (x + 1) * 0.5
      outputChannels = 4
      causalAttentionMask = nil
    }
    guard batchSize > 1 && version != .hunyuanVideo else {
      if highPrecision {
        if tiledEncoding {
          return (
            DynamicGraph.Tensor<FloatType>(
              from: tiledEncode(
                DynamicGraph.Tensor<Float>(from: x), causalAttentionMask: causalAttentionMask,
                encoder: encoder,
                tileSize: encodingTileSize,
                tileOverlap: encodingTileOverlap,
                scaleFactor: (scaleFactor, scaleFactorZ), outputChannels: outputChannels)), encoder
          )
        } else {
          return (
            DynamicGraph.Tensor<FloatType>(
              from: encoder(
                inputs: DynamicGraph.Tensor<Float>(from: x),
                (causalAttentionMask.map { [$0] } ?? []))[0].as(of: Float.self)),
            encoder
          )
        }
      } else {
        if tiledEncoding {
          return (
            tiledEncode(
              x,
              causalAttentionMask: causalAttentionMask.map {
                DynamicGraph.Tensor<FloatType>(from: $0)
              }, encoder: encoder, tileSize: encodingTileSize,
              tileOverlap: encodingTileOverlap, scaleFactor: (scaleFactor, scaleFactorZ),
              outputChannels: outputChannels),
            encoder
          )
        } else {
          return (
            encoder(
              inputs: x,
              (causalAttentionMask.map { [DynamicGraph.Tensor<FloatType>(from: $0)] } ?? []))[0].as(
                of: FloatType.self), encoder
          )
        }
      }
    }
    var result = graph.variable(
      .GPU(0), .NHWC(batchSize, startHeight, startWidth, outputChannels), of: FloatType.self)
    for i in 0..<batchSize {
      let z = x[i..<(i + 1), 0..<shape[1], 0..<shape[2], 0..<shape[3]].copied()
      if highPrecision {
        if tiledEncoding {
          result[i..<(i + 1), 0..<startHeight, 0..<startWidth, 0..<outputChannels] = DynamicGraph
            .Tensor<
              FloatType
            >(
              from: tiledEncode(
                DynamicGraph.Tensor<Float>(from: z), causalAttentionMask: causalAttentionMask,
                encoder: encoder,
                tileSize: encodingTileSize,
                tileOverlap: encodingTileOverlap,
                scaleFactor: (scaleFactor, scaleFactorZ), outputChannels: outputChannels))
        } else {
          result[i..<(i + 1), 0..<startHeight, 0..<startWidth, 0..<outputChannels] = DynamicGraph
            .Tensor<
              FloatType
            >(
              from: encoder(
                inputs: DynamicGraph.Tensor<Float>(from: z),
                (causalAttentionMask.map { [$0] } ?? []))[0].as(
                  of: Float.self))
        }
      } else {
        if tiledEncoding {
          result[i..<(i + 1), 0..<startHeight, 0..<startWidth, 0..<outputChannels] = tiledEncode(
            z,
            causalAttentionMask: causalAttentionMask.map {
              DynamicGraph.Tensor<FloatType>(from: $0)
            }, encoder: encoder, tileSize: encodingTileSize,
            tileOverlap: encodingTileOverlap, scaleFactor: (scaleFactor, scaleFactorZ),
            outputChannels: outputChannels)
        } else {
          result[i..<(i + 1), 0..<startHeight, 0..<startWidth, 0..<outputChannels] = encoder(
            inputs: z,
            (causalAttentionMask.map { [DynamicGraph.Tensor<FloatType>(from: $0)] } ?? []))[0].as(
              of: FloatType.self)
        }
      }
    }
    return (result, encoder)
  }
}

extension FirstStage {
  private func internalDecode<T: TensorNumeric & BinaryFloatingPoint>(
    _ z: DynamicGraph.Tensor<T>, causalAttentionMask: DynamicGraph.Tensor<T>?, decoder: Model,
    transparentDecoder: Model?
  ) -> DynamicGraph.Tensor<T> {
    let pixel = decoder(inputs: z, causalAttentionMask.map { [$0] } ?? [])[0].as(of: T.self)
    guard let transparentDecoder = transparentDecoder else { return pixel }
    let pixelShape = pixel.shape
    let rgb = pixel[0..<1, 0..<pixelShape[1], 0..<pixelShape[2], 0..<3].copied()
    // Normalize to between 0 and 1
    var result = transparentDecoder(inputs: (rgb + 1) * 0.5, z)[0].as(of: T.self)
    // At this point, result should have 4 channels. The last 3 is RGB. Do so.
    let shape = result.shape
    // Only scale RGB channels to -1 and 1.
    let alpha = result[0..<1, 0..<shape[1], 0..<shape[2], 0..<1].clamped(0...1)
    result[0..<1, 0..<shape[1], 0..<shape[2], 1..<shape[3]] =
      (result[0..<1, 0..<shape[1], 0..<shape[2], 1..<shape[3]].clamped(0...1) - 0.5) * 2
    result[0..<1, 0..<shape[1], 0..<shape[2], 0..<1] = alpha
    return result
  }

  private func tiledDecode<T: TensorNumeric & BinaryFloatingPoint>(
    _ z: DynamicGraph.Tensor<T>, causalAttentionMask: DynamicGraph.Tensor<T>?, decoder: Model,
    transparentDecoder: Model?,
    tileSize: (width: Int, height: Int), tileOverlap: Int, outputChannels: Int,
    scaleFactor: (spatial: Int, temporal: Int)
  ) -> DynamicGraph.Tensor<T> {
    let shape = z.shape
    let channels = shape[3]
    let batchSize = shape[0]
    // tile overlap shouldn't be bigger than 1/3 of either height or width (otherwise we cannot make much progress).
    let tileOverlap = min(
      min(
        tileOverlap,
        Int((Double(tileSize.height / 3) / Double(64 / scaleFactor.spatial)).rounded(.down))
          * (64 / scaleFactor.spatial)),
      Int((Double(tileSize.width / 3) / Double(64 / scaleFactor.spatial)).rounded(.down))
        * (64 / scaleFactor.spatial)
    )
    let yTiles =
      (shape[1] - tileOverlap * 2 + (tileSize.height - tileOverlap * 2) - 1)
      / (tileSize.height - tileOverlap * 2)
    let xTiles =
      (shape[2] - tileOverlap * 2 + (tileSize.width - tileOverlap * 2) - 1)
      / (tileSize.width - tileOverlap * 2)
    let graph = z.graph
    var decodedRawValues = [Tensor<T>]()
    for y in 0..<yTiles {
      let yOfs = y * (tileSize.height - tileOverlap * 2) + (y > 0 ? tileOverlap : 0)
      let (inputStartYPad, inputEndYPad) = paddedTileStartAndEnd(
        iOfs: yOfs, length: shape[1], tileSize: tileSize.height, tileOverlap: tileOverlap)
      for x in 0..<xTiles {
        let xOfs = x * (tileSize.width - tileOverlap * 2) + (x > 0 ? tileOverlap : 0)
        let (inputStartXPad, inputEndXPad) = paddedTileStartAndEnd(
          iOfs: xOfs, length: shape[2], tileSize: tileSize.width, tileOverlap: tileOverlap)
        decodedRawValues.append(
          internalDecode(
            z[
              0..<batchSize, inputStartYPad..<inputEndYPad, inputStartXPad..<inputEndXPad,
              0..<channels
            ].copied(), causalAttentionMask: causalAttentionMask, decoder: decoder,
            transparentDecoder: transparentDecoder
          ).rawValue.toCPU())
      }
    }
    let (xWeightsAndIndexes, yWeightsAndIndexes) = xyTileWeightsAndIndexes(
      width: shape[2] * scaleFactor.spatial, height: shape[1] * scaleFactor.spatial, xTiles: xTiles,
      yTiles: yTiles,
      tileSize: (
        width: tileSize.width * scaleFactor.spatial, height: tileSize.height * scaleFactor.spatial
      ),
      tileOverlap: tileOverlap * scaleFactor.spatial)
    let resultBatchSize = (shape[0] - 1) * scaleFactor.temporal + 1
    var result = Tensor<T>(
      .CPU,
      .NHWC(
        resultBatchSize, shape[1] * scaleFactor.spatial, shape[2] * scaleFactor.spatial,
        outputChannels))
    let inputChannels = decodedRawValues.first?.shape[3] ?? outputChannels
    result.withUnsafeMutableBytes {
      guard var fp = $0.baseAddress?.assumingMemoryBound(to: T.self) else { return }
      for t in 0..<resultBatchSize {
        for j in 0..<(shape[1] * scaleFactor.spatial) {
          let yWeightAndIndex = yWeightsAndIndexes[j]
          for i in 0..<(shape[2] * scaleFactor.spatial) {
            let xWeightAndIndex = xWeightsAndIndexes[i]
            for k in 0..<outputChannels {
              fp[k] = 0
            }
            let tOffset =
              t * tileSize.width * scaleFactor.spatial * inputChannels * tileSize.height
              * scaleFactor.spatial
            for y in yWeightAndIndex {
              let yOffset =
                y.offset * tileSize.width * scaleFactor.spatial * inputChannels + tOffset
              for x in xWeightAndIndex {
                let weight = T(x.weight * y.weight)
                let index = y.index * xTiles + x.index
                let tensor = decodedRawValues[index]
                tensor.withUnsafeBytes {
                  guard var v = $0.baseAddress?.assumingMemoryBound(to: T.self) else { return }
                  // Note that while result is outputChannels, this is padded to 4 i.e. channels.
                  v += x.offset * inputChannels + yOffset
                  for k in 0..<outputChannels {
                    fp[k] += v[k] * weight
                  }
                }
              }
            }
            fp += outputChannels
          }
        }
      }
    }
    return graph.variable(result.toGPU(0))
  }

  private func tiledDecode<T: TensorNumeric & BinaryFloatingPoint>(
    _ z: DynamicGraph.Tensor<T>, causalAttentionMask: DynamicGraph.Tensor<T>?, decoder: Model,
    transparentDecoder: Model?,
    tileSize: (depth: Int, width: Int, height: Int), tileOverlap: Int, outputChannels: Int,
    scaleFactor: (spatial: Int, temporal: Int)
  ) -> DynamicGraph.Tensor<T> {
    guard tileSize.depth > 1 && tileSize.depth < z.shape[0] else {
      return tiledDecode(
        z, causalAttentionMask: causalAttentionMask, decoder: decoder,
        transparentDecoder: transparentDecoder,
        tileSize: (width: tileSize.width, height: tileSize.height), tileOverlap: tileOverlap,
        outputChannels: outputChannels, scaleFactor: scaleFactor)
    }
    let shape = z.shape
    let channels = shape[3]
    let batchSize = shape[0]
    let graph = z.graph
    // tile overlap shouldn't be bigger than 1/3 of either height or width (otherwise we cannot make much progress).
    let tileOverlap = min(
      min(
        tileOverlap,
        Int((Double(tileSize.height / 3) / Double(64 / scaleFactor.spatial)).rounded(.down))
          * (64 / scaleFactor.spatial)),
      Int((Double(tileSize.width / 3) / Double(64 / scaleFactor.spatial)).rounded(.down))
        * (64 / scaleFactor.spatial)
    )
    let yTiles =
      (shape[1] - tileOverlap * 2 + (tileSize.height - tileOverlap * 2) - 1)
      / (tileSize.height - tileOverlap * 2)
    let xTiles =
      (shape[2] - tileOverlap * 2 + (tileSize.width - tileOverlap * 2) - 1)
      / (tileSize.width - tileOverlap * 2)
    let (xWeightsAndIndexes, yWeightsAndIndexes) = xyTileWeightsAndIndexes(
      width: shape[2] * scaleFactor.spatial, height: shape[1] * scaleFactor.spatial, xTiles: xTiles,
      yTiles: yTiles,
      tileSize: (
        width: tileSize.width * scaleFactor.spatial, height: tileSize.height * scaleFactor.spatial
      ),
      tileOverlap: tileOverlap * scaleFactor.spatial)
    let resultBatchSize = (shape[0] - 1) * scaleFactor.temporal + 1
    var result = Tensor<T>(
      Array(
        repeating: 0,
        count: resultBatchSize * shape[1] * scaleFactor.spatial * shape[2] * scaleFactor.spatial
          * outputChannels), .CPU,
      .NHWC(
        resultBatchSize, shape[1] * scaleFactor.spatial, shape[2] * scaleFactor.spatial,
        outputChannels))
    // Hard-code overlapping 16 frames in time.
    for t in stride(from: 0, to: batchSize, by: max(1, tileSize.depth - 5)) {
      var decodedRawValues = [Tensor<T>]()
      let tStart = min(t, batchSize - tileSize.depth)
      for y in 0..<yTiles {
        let yOfs = y * (tileSize.height - tileOverlap * 2) + (y > 0 ? tileOverlap : 0)
        let (inputStartYPad, inputEndYPad) = paddedTileStartAndEnd(
          iOfs: yOfs, length: shape[1], tileSize: tileSize.height, tileOverlap: tileOverlap)
        for x in 0..<xTiles {
          let xOfs = x * (tileSize.width - tileOverlap * 2) + (x > 0 ? tileOverlap : 0)
          let (inputStartXPad, inputEndXPad) = paddedTileStartAndEnd(
            iOfs: xOfs, length: shape[2], tileSize: tileSize.width, tileOverlap: tileOverlap)
          decodedRawValues.append(
            internalDecode(
              z[
                tStart..<(tStart + tileSize.depth), inputStartYPad..<inputEndYPad,
                inputStartXPad..<inputEndXPad,
                0..<channels
              ].copied(), causalAttentionMask: causalAttentionMask, decoder: decoder,
              transparentDecoder: transparentDecoder
            ).rawValue.toCPU())
        }
      }
      let tileDecodedDepth = ((tileSize.depth - 1) * scaleFactor.temporal) + 1
      let inputChannels = decodedRawValues.first?.shape[3] ?? outputChannels
      result.withUnsafeMutableBytes {
        guard let rfp = $0.baseAddress?.assumingMemoryBound(to: T.self) else { return }
        // Due to hard-code and causal convolution, we skip the first 8 frames and only mix the rest 8 frames.
        let tDecodedStart: Int
        let isLast = t + tileSize.depth >= batchSize
        if t == 0 {
          tDecodedStart = 0
        } else if t + tileSize.depth > batchSize {
          tDecodedStart = (t - (batchSize - tileSize.depth)) * scaleFactor.temporal + 9
        } else {
          tDecodedStart = 9
        }
        for tDecoded in tDecodedStart..<tileDecodedDepth {
          var fp =
            rfp + (tDecoded + tStart * scaleFactor.temporal) * shape[1] * scaleFactor.spatial
            * shape[2]
            * scaleFactor.spatial * outputChannels
          let tWeight: Float
          if tDecoded - tDecodedStart < 8 && tDecodedStart != 0 {
            tWeight = min((Float(tDecoded - tDecodedStart) + 0.5) / 8, 1)
          } else if tileDecodedDepth - tDecoded <= 8 && !isLast {
            tWeight = min((Float(tileDecodedDepth - tDecoded) - 0.5) / 8, 1)
          } else {
            tWeight = 1
          }
          for j in 0..<(shape[1] * scaleFactor.spatial) {
            let yWeightAndIndex = yWeightsAndIndexes[j]
            for i in 0..<(shape[2] * scaleFactor.spatial) {
              let xWeightAndIndex = xWeightsAndIndexes[i]
              let tOffset =
                tDecoded * tileSize.width * scaleFactor.spatial * inputChannels * tileSize.height
                * scaleFactor.spatial
              for y in yWeightAndIndex {
                let yOffset =
                  y.offset * tileSize.width * scaleFactor.spatial * inputChannels + tOffset
                for x in xWeightAndIndex {
                  let weight = T(x.weight * y.weight * tWeight)
                  let index = y.index * xTiles + x.index
                  let tensor = decodedRawValues[index]
                  tensor.withUnsafeBytes {
                    guard var v = $0.baseAddress?.assumingMemoryBound(to: T.self) else { return }
                    // Note that while result is outputChannels, this is padded to 4 i.e. channels.
                    v += x.offset * inputChannels + yOffset
                    for k in 0..<outputChannels {
                      fp[k] += v[k] * weight
                    }
                  }
                }
              }
              fp += outputChannels
            }
          }
        }
      }
    }
    return graph.variable(result.toGPU(0))
  }

  private func tiledEncode<T: TensorNumeric & BinaryFloatingPoint>(
    _ z: DynamicGraph.Tensor<T>, causalAttentionMask: DynamicGraph.Tensor<T>?, encoder: Model,
    tileSize: (width: Int, height: Int), tileOverlap: Int,
    scaleFactor: (spatial: Int, temporal: Int),
    outputChannels: Int
  ) -> DynamicGraph.Tensor<T> {
    let shape = z.shape
    let channels = shape[3]
    let batchSize = shape[0]
    // tile overlap shouldn't be bigger than 1/3 of either height or width (otherwise we cannot make much progress).
    let tileOverlap = min(
      min(tileOverlap, Int((Double(tileSize.height / 3) / 8).rounded(.down)) * 8),
      Int((Double(tileSize.width / 3) / 8).rounded(.down)) * 8)
    let startHeight = shape[1] / scaleFactor.spatial
    let startWidth = shape[2] / scaleFactor.spatial
    let yTiles =
      (startHeight - tileOverlap * 2 + (tileSize.height - tileOverlap * 2) - 1)
      / (tileSize.height - tileOverlap * 2)
    let xTiles =
      (startWidth - tileOverlap * 2 + (tileSize.width - tileOverlap * 2) - 1)
      / (tileSize.width - tileOverlap * 2)
    let graph = z.graph
    var encodedRawValues = [Tensor<T>]()
    for y in 0..<yTiles {
      let yOfs = y * (tileSize.height - tileOverlap * 2) + (y > 0 ? tileOverlap : 0)
      let (inputStartYPad, inputEndYPad) = paddedTileStartAndEnd(
        iOfs: yOfs * scaleFactor.spatial, length: shape[1],
        tileSize: tileSize.height * scaleFactor.spatial,
        tileOverlap: tileOverlap * scaleFactor.spatial)
      for x in 0..<xTiles {
        let xOfs = x * (tileSize.width - tileOverlap * 2) + (x > 0 ? tileOverlap : 0)
        let (inputStartXPad, inputEndXPad) = paddedTileStartAndEnd(
          iOfs: xOfs * scaleFactor.spatial, length: shape[2],
          tileSize: tileSize.width * scaleFactor.spatial,
          tileOverlap: tileOverlap * scaleFactor.spatial)
        encodedRawValues.append(
          encoder(
            inputs:
              z[
                0..<batchSize, inputStartYPad..<inputEndYPad, inputStartXPad..<inputEndXPad,
                0..<channels
              ].copied(), (causalAttentionMask.map { [$0] } ?? []))[0].as(of: T.self).rawValue
            .toCPU())
      }
    }
    let (xWeightsAndIndexes, yWeightsAndIndexes) = xyTileWeightsAndIndexes(
      width: startWidth, height: startHeight, xTiles: xTiles, yTiles: yTiles,
      tileSize: (width: tileSize.width, height: tileSize.height), tileOverlap: tileOverlap)
    let resultBatchSize = (shape[0] - 1) / scaleFactor.temporal + 1
    var result = Tensor<T>(.CPU, .NHWC(resultBatchSize, startHeight, startWidth, outputChannels))
    result.withUnsafeMutableBytes {
      guard var fp = $0.baseAddress?.assumingMemoryBound(to: T.self) else { return }
      for t in 0..<resultBatchSize {
        for j in 0..<startHeight {
          let yWeightAndIndex = yWeightsAndIndexes[j]
          for i in 0..<startWidth {
            let xWeightAndIndex = xWeightsAndIndexes[i]
            for k in 0..<outputChannels {
              fp[k] = 0
            }
            let tOffset = t * tileSize.width * outputChannels * tileSize.height
            for y in yWeightAndIndex {
              let yOffset = y.offset * tileSize.width * outputChannels + tOffset
              for x in xWeightAndIndex {
                let weight = T(x.weight * y.weight)
                let index = y.index * xTiles + x.index
                let tensor = encodedRawValues[index]
                tensor.withUnsafeBytes {
                  guard var v = $0.baseAddress?.assumingMemoryBound(to: T.self) else { return }
                  // Note that while result is outputChannels, this is padded to 4 i.e. channels.
                  v = v + x.offset * outputChannels + yOffset
                  for k in 0..<outputChannels {
                    fp[k] += v[k] * weight
                  }
                }
              }
            }
            fp += outputChannels
          }
        }
      }
    }
    return graph.variable(result.toGPU(0))
  }

  private func tiledEncode<T: TensorNumeric & BinaryFloatingPoint>(
    _ z: DynamicGraph.Tensor<T>, causalAttentionMask: DynamicGraph.Tensor<T>?, encoder: Model,
    tileSize: (depth: Int, width: Int, height: Int), tileOverlap: Int,
    scaleFactor: (spatial: Int, temporal: Int),
    outputChannels: Int
  ) -> DynamicGraph.Tensor<T> {
    guard tileSize.depth > 1 && tileSize.depth < (z.shape[0] - 1) / scaleFactor.temporal + 1 else {
      return tiledEncode(
        z, causalAttentionMask: causalAttentionMask, encoder: encoder,
        tileSize: (width: tileSize.width, height: tileSize.height), tileOverlap: tileOverlap,
        scaleFactor: scaleFactor, outputChannels: outputChannels)
    }
    let shape = z.shape
    let channels = shape[3]
    let batchSize = shape[0]
    // tile overlap shouldn't be bigger than 1/3 of either height or width (otherwise we cannot make much progress).
    let tileOverlap = min(
      min(tileOverlap, Int((Double(tileSize.height / 3) / 8).rounded(.down)) * 8),
      Int((Double(tileSize.width / 3) / 8).rounded(.down)) * 8)
    let startHeight = shape[1] / scaleFactor.spatial
    let startWidth = shape[2] / scaleFactor.spatial
    let yTiles =
      (startHeight - tileOverlap * 2 + (tileSize.height - tileOverlap * 2) - 1)
      / (tileSize.height - tileOverlap * 2)
    let xTiles =
      (startWidth - tileOverlap * 2 + (tileSize.width - tileOverlap * 2) - 1)
      / (tileSize.width - tileOverlap * 2)
    let graph = z.graph
    var encodedRawValues = [Tensor<T>]()
    let resultBatchSize = (shape[0] - 1) / scaleFactor.temporal + 1
    var result = Tensor<T>(.CPU, .NHWC(resultBatchSize, startHeight, startWidth, outputChannels))
    let tileDecodedDepth = ((tileSize.depth - 1) * scaleFactor.temporal) + 1
    for t in stride(from: 0, to: resultBatchSize, by: max(1, tileSize.depth - 5)) {
      let tDecodedStart = min(t * scaleFactor.temporal, batchSize - tileDecodedDepth)
      for y in 0..<yTiles {
        let yOfs = y * (tileSize.height - tileOverlap * 2) + (y > 0 ? tileOverlap : 0)
        let (inputStartYPad, inputEndYPad) = paddedTileStartAndEnd(
          iOfs: yOfs * scaleFactor.spatial, length: shape[1],
          tileSize: tileSize.height * scaleFactor.spatial,
          tileOverlap: tileOverlap * scaleFactor.spatial)
        for x in 0..<xTiles {
          let xOfs = x * (tileSize.width - tileOverlap * 2) + (x > 0 ? tileOverlap : 0)
          let (inputStartXPad, inputEndXPad) = paddedTileStartAndEnd(
            iOfs: xOfs * scaleFactor.spatial, length: shape[2],
            tileSize: tileSize.width * scaleFactor.spatial,
            tileOverlap: tileOverlap * scaleFactor.spatial)
          encodedRawValues.append(
            encoder(
              inputs:
                z[
                  tDecodedStart..<(tDecodedStart + tileDecodedDepth), inputStartYPad..<inputEndYPad,
                  inputStartXPad..<inputEndXPad,
                  0..<channels
                ].copied(), (causalAttentionMask.map { [$0] } ?? []))[0].as(of: T.self).rawValue
              .toCPU())
        }
      }
      let (xWeightsAndIndexes, yWeightsAndIndexes) = xyTileWeightsAndIndexes(
        width: startWidth, height: startHeight, xTiles: xTiles, yTiles: yTiles,
        tileSize: (width: tileSize.width, height: tileSize.height), tileOverlap: tileOverlap)
      result.withUnsafeMutableBytes {
        guard let rfp = $0.baseAddress?.assumingMemoryBound(to: T.self) else { return }
        let tEncodedStart: Int
        let isLast = t + tileSize.depth >= resultBatchSize
        if t == 0 {
          tEncodedStart = 0
        } else if t + tileSize.depth > resultBatchSize {
          tEncodedStart = (t - (resultBatchSize - tileSize.depth)) + 2
        } else {
          tEncodedStart = 2
        }
        for tEncoded in tEncodedStart..<tileSize.depth {
          var fp =
            rfp + (tEncoded + tDecodedStart / scaleFactor.temporal) * tileSize.width
            * tileSize.height * outputChannels
          let tWeight: Float
          if tEncoded - tEncodedStart < 2 && tEncodedStart != 0 {
            tWeight = min((Float(tEncoded - tEncodedStart) + 0.5) / 2, 1)
          } else if tileSize.depth - tEncoded <= 2 && !isLast {
            tWeight = min((Float(tileSize.depth - tEncoded) - 0.5) / 2, 1)
          } else {
            tWeight = 1
          }
          for j in 0..<startHeight {
            let yWeightAndIndex = yWeightsAndIndexes[j]
            for i in 0..<startWidth {
              let xWeightAndIndex = xWeightsAndIndexes[i]
              let tOffset = tEncoded * tileSize.width * outputChannels * tileSize.height
              for y in yWeightAndIndex {
                let yOffset = y.offset * tileSize.width * outputChannels + tOffset
                for x in xWeightAndIndex {
                  let weight = T(x.weight * y.weight * tWeight)
                  let index = y.index * xTiles + x.index
                  let tensor = encodedRawValues[index]
                  tensor.withUnsafeBytes {
                    guard var v = $0.baseAddress?.assumingMemoryBound(to: T.self) else { return }
                    // Note that while result is outputChannels, this is padded to 4 i.e. channels.
                    v = v + x.offset * outputChannels + yOffset
                    for k in 0..<outputChannels {
                      fp[k] += v[k] * weight
                    }
                  }
                }
              }
              fp += outputChannels
            }
          }
        }
      }
    }
    return graph.variable(result.toGPU(0))
  }
}
