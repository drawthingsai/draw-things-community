import NNC

public struct FirstStage<FloatType: TensorNumeric & BinaryFloatingPoint> {
  public let filePath: String
  public let version: ModelVersion
  public let highPrecision: Bool
  public let externalOnDemand: Bool
  private let latentsScaling: (mean: [Float]?, std: [Float]?, scalingFactor: Float)
  private let highPrecisionFallback: Bool
  private let tiledDecoding: Bool
  private let decodingTileSize: (width: Int, height: Int)
  private let decodingTileOverlap: Int
  public init(
    filePath: String, version: ModelVersion,
    latentsScaling: (mean: [Float]?, std: [Float]?, scalingFactor: Float),
    highPrecision: Bool, highPrecisionFallback: Bool, tiledDecoding: Bool,
    decodingTileSize: (width: Int, height: Int), decodingTileOverlap: Int,
    externalOnDemand: Bool
  ) {
    self.filePath = filePath
    self.version = version
    self.latentsScaling = latentsScaling
    self.highPrecision = highPrecision
    self.highPrecisionFallback = highPrecisionFallback
    self.externalOnDemand = externalOnDemand
    self.tiledDecoding = tiledDecoding
    self.decodingTileSize = decodingTileSize
    self.decodingTileOverlap = decodingTileOverlap
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
    } else {
      z = x / scalingFactor
    }
    let decoder: Model
    let isMemoryEfficient = DynamicGraph.memoryEfficient
    if version == .kandinsky21 {
      DynamicGraph.memoryEfficient = true
    }
    defer {
      if version == .kandinsky21 {
        DynamicGraph.memoryEfficient = isMemoryEfficient
      }
    }
    // Tiled decoding is only applicable for SD / SDXL VAE (Wurstchen's Stage A is too tiny to matter).
    let decodingTileSize = (
      width: min(decodingTileSize.width * 8, startWidth),
      height: min(decodingTileSize.height * 8, startHeight)
    )
    let decodingTileOverlap = decodingTileOverlap * 8
    var tiledDecoding =
      tiledDecoding
      && (startWidth > decodingTileSize.width || startHeight > decodingTileSize.height)
    let zoomFactor: Int
    let externalData: DynamicGraph.Store.Codec =
      externalOnDemand ? .externalOnDemand : .externalData
    switch version {
    case .v1, .v2, .sdxlBase, .sdxlRefiner, .ssd1b, .svdI2v:
      let startWidth = tiledDecoding ? decodingTileSize.width : startWidth
      let startHeight = tiledDecoding ? decodingTileSize.height : startHeight
      decoder =
        existingDecoder
        ?? Decoder(
          channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: startWidth,
          startHeight: startHeight, usesFlashAttention: false, paddingFinalConvLayer: true
        ).0
      if existingDecoder == nil {
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
      zoomFactor = 8
    case .kandinsky21:
      tiledDecoding = false
      decoder =
        existingDecoder
        ?? MOVQDecoderKandinsky(
          zChannels: 4, channels: 128, channelMult: [1, 2, 2, 4], numResBlocks: 2,
          startHeight: startHeight, startWidth: startWidth, attnResolutions: Set([32]))
      if existingDecoder == nil {
        decoder.compile(inputs: z[0..<1, 0..<startHeight, 0..<startWidth, 0..<shape[3]])
        graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) {
          $0.read("movq", model: decoder, codec: [.jit, externalData])
        }
      }
      zoomFactor = 8
    case .wurstchenStageC, .wurstchenStageB:
      tiledDecoding = false
      decoder =
        existingDecoder
        ?? WurstchenStageADecoder(batchSize: 1, height: startHeight * 2, width: startWidth * 2)
      if existingDecoder == nil {
        decoder.compile(inputs: z[0..<1, 0..<startHeight, 0..<startWidth, 0..<shape[3]])
        graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) {
          $0.read("decoder", model: decoder, codec: [.jit, externalData])
        }
      }
      zoomFactor = 4
    }
    guard batchSize > 1 else {
      if highPrecision {
        let result: DynamicGraph.Tensor<Float>
        if tiledDecoding {
          result = tiledDecode(
            DynamicGraph.Tensor<Float>(from: z), decoder: decoder, tileSize: decodingTileSize,
            tileOverlap: decodingTileOverlap)
        } else {
          result = decoder(inputs: DynamicGraph.Tensor<Float>(from: z))[0].as(of: Float.self)
        }
        let shape = result.shape
        return (
          DynamicGraph.Tensor<FloatType>(
            from: result[0..<shape[0], 0..<shape[1], 0..<shape[2], 0..<3]),
          decoder
        )
      } else {
        let result: DynamicGraph.Tensor<FloatType>
        if tiledDecoding {
          result = tiledDecode(
            z, decoder: decoder, tileSize: decodingTileSize, tileOverlap: decodingTileOverlap)
        } else {
          result = decoder(inputs: z)[0].as(of: FloatType.self)
        }
        let shape = result.shape
        return (result[0..<shape[0], 0..<shape[1], 0..<shape[2], 0..<3].copied(), decoder)
      }
    }
    var result = graph.variable(
      .GPU(0), .NHWC(batchSize, startHeight * zoomFactor, startWidth * zoomFactor, 3),
      of: FloatType.self)
    for i in 0..<batchSize {
      let zEnc = z[i..<(i + 1), 0..<startHeight, 0..<startWidth, 0..<shape[3]].copied()
      if highPrecision {
        let partial: DynamicGraph.Tensor<Float>
        if tiledDecoding {
          partial = tiledDecode(
            DynamicGraph.Tensor<Float>(from: zEnc), decoder: decoder, tileSize: decodingTileSize,
            tileOverlap: decodingTileOverlap)
        } else {
          partial = decoder(
            inputs: DynamicGraph.Tensor<Float>(from: zEnc))[0].as(of: Float.self)
        }
        let shape = partial.shape
        result[i..<(i + 1), 0..<(startHeight * zoomFactor), 0..<(startWidth * zoomFactor), 0..<3] =
          DynamicGraph
          .Tensor<FloatType>(from: partial[0..<1, 0..<shape[1], 0..<shape[2], 0..<3])
      } else {
        let partial: DynamicGraph.Tensor<FloatType>
        if tiledDecoding {
          partial = tiledDecode(
            zEnc, decoder: decoder, tileSize: decodingTileSize, tileOverlap: decodingTileOverlap)
        } else {
          partial = decoder(inputs: zEnc)[0].as(of: FloatType.self)
        }
        let shape = partial.shape
        result[i..<(i + 1), 0..<(startHeight * zoomFactor), 0..<(startWidth * zoomFactor), 0..<3] =
          partial[0..<1, 0..<shape[1], 0..<shape[2], 0..<3]
      }
    }
    return (result, decoder)
  }

  public func decode(_ x: DynamicGraph.Tensor<FloatType>, decoder existingDecoder: Model?)
    -> (DynamicGraph.Tensor<FloatType>, Model)
  {
    guard !highPrecision else {
      return decode(x, decoder: existingDecoder, highPrecision: true)
    }
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
    let mean = parameters[0..<batchSize, 0..<startHeight, 0..<startWidth, 0..<4]
    let logvar = parameters[0..<batchSize, 0..<startHeight, 0..<startWidth, 4..<8].copied().clamped(
      -30...20)
    let std = Functional.exp(0.5 * logvar)
    let n: DynamicGraph.Tensor<FloatType>
    if let noise = noise {
      n = noise
    } else {
      n = graph.variable(
        .GPU(0), .NHWC(batchSize, startHeight, startWidth, 4), of: FloatType.self)
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
    } else {
      return x * scalingFactor
    }
  }

  public func encode(_ x: DynamicGraph.Tensor<FloatType>, encoder existingEncoder: Model?)
    -> (DynamicGraph.Tensor<FloatType>, Model)
  {
    let shape = x.shape
    let batchSize = shape[0]
    precondition(shape[1] % 32 == 0)
    let startHeight: Int
    precondition(shape[2] % 32 == 0)
    let startWidth: Int
    let graph = x.graph
    let encoder: Model
    let outputChannels: Int
    var x = x
    let externalData: DynamicGraph.Store.Codec =
      externalOnDemand ? .externalOnDemand : .externalData
    switch version {
    case .v1, .v2, .sdxlBase, .sdxlRefiner, .ssd1b, .svdI2v:
      startHeight = shape[1] / 8
      startWidth = shape[2] / 8
      encoder =
        existingEncoder
        ?? Encoder(
          channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: startWidth,
          startHeight: startHeight, usesFlashAttention: false
        ).0
      if existingEncoder == nil {
        if highPrecision {
          encoder.compile(
            inputs: DynamicGraph.Tensor<Float>(
              from: x[0..<1, 0..<shape[1], 0..<shape[2], 0..<shape[3]]))
        } else {
          encoder.compile(inputs: x[0..<1, 0..<shape[1], 0..<shape[2], 0..<shape[3]])
        }
        graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) {
          $0.read("encoder", model: encoder, codec: [.jit, externalData])
        }
      }
      outputChannels = 8
    case .kandinsky21:
      startHeight = shape[1] / 8
      startWidth = shape[2] / 8
      encoder =
        existingEncoder
        ?? EncoderKandinsky(
          zChannels: 4, channels: 128, channelMult: [1, 2, 2, 4], numResBlocks: 2,
          startHeight: startHeight * 8, startWidth: startWidth * 8, attnResolutions: Set([32]))
      if existingEncoder == nil {
        encoder.compile(inputs: x[0..<1, 0..<shape[1], 0..<shape[2], 0..<shape[3]])
        graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) {
          $0.read("encoder", model: encoder, codec: [.jit, externalData])
        }
      }
      outputChannels = 4
    case .wurstchenStageC:
      startHeight = shape[1] / 32
      startWidth = shape[2] / 32
      encoder = existingEncoder ?? EfficientNetEncoder()
      if existingEncoder == nil {
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
    case .wurstchenStageB:
      startHeight = shape[1] / 4
      startWidth = shape[2] / 4
      encoder = existingEncoder ?? WurstchenStageAEncoder(batchSize: 1)
      if existingEncoder == nil {
        encoder.compile(inputs: x[0..<1, 0..<shape[1], 0..<shape[2], 0..<shape[3]])
        graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) {
          $0.read("encoder", model: encoder, codec: [.jit, .q6p, .q8p, .ezm7, externalData])
        }
      }
      x = (x + 1) * 0.5
      outputChannels = 4
    }
    guard batchSize > 1 else {
      if highPrecision {
        return (
          DynamicGraph.Tensor<FloatType>(
            from: encoder(inputs: DynamicGraph.Tensor<Float>(from: x))[0].as(of: Float.self)),
          encoder
        )
      } else {
        return (encoder(inputs: x)[0].as(of: FloatType.self), encoder)
      }
    }
    var result = graph.variable(
      .GPU(0), .NHWC(batchSize, startHeight, startWidth, outputChannels), of: FloatType.self)
    for i in 0..<batchSize {
      let z = x[i..<(i + 1), 0..<shape[1], 0..<shape[2], 0..<shape[3]].copied()
      if highPrecision {
        result[i..<(i + 1), 0..<startHeight, 0..<startWidth, 0..<outputChannels] = DynamicGraph
          .Tensor<
            FloatType
          >(
            from: encoder(inputs: DynamicGraph.Tensor<Float>(from: z))[0].as(
              of: Float.self))
      } else {
        result[i..<(i + 1), 0..<startHeight, 0..<startWidth, 0..<outputChannels] = encoder(
          inputs: z)[0].as(
            of: FloatType.self)
      }
    }
    return (result, encoder)
  }
}

extension FirstStage {
  private func paddedTileStartAndEnd(iOfs: Int, length: Int, tileSize: Int, tileOverlap: Int) -> (
    paddedStart: Int, paddedEnd: Int
  ) {
    let inputEnd = min(iOfs + tileSize - tileOverlap * 2, length)
    var inputStartPad = iOfs - tileOverlap
    var inputEndPad = inputEnd + tileOverlap
    if inputStartPad < 0 {
      inputStartPad = 0
      inputEndPad = tileSize
      precondition(inputEndPad <= length)
    } else if inputEndPad > length {
      inputEndPad = length
      inputStartPad = length - tileSize
      precondition(inputStartPad >= 0)
    }
    return (inputStartPad, inputEndPad)
  }

  private func tiledDecode<T: TensorNumeric & BinaryFloatingPoint>(
    _ z: DynamicGraph.Tensor<T>, decoder: Model, tileSize: (width: Int, height: Int),
    tileOverlap: Int
  ) -> DynamicGraph.Tensor<T> {
    // Assuming batch is 1.
    let shape = z.shape
    let channels = shape[3]
    precondition(shape[0] == 1)
    // tile overlap shouldn't be bigger than 1/3 of either height or width (otherwise we cannot make much progress).
    let tileOverlap = min(
      min(tileOverlap, Int((Double(tileSize.height / 3) / 8).rounded(.down)) * 8),
      Int((Double(tileSize.width / 3) / 8).rounded(.down)) * 8)
    let yTiles =
      (shape[1] - tileOverlap * 2 + (tileSize.height - tileOverlap * 2) - 1)
      / (tileSize.height - tileOverlap * 2)
    let xTiles =
      (shape[2] - tileOverlap * 2 + (tileSize.width - tileOverlap * 2) - 1)
      / (tileSize.width - tileOverlap * 2)
    let graph = z.graph
    let streamContext = StreamContext(.GPU(0))
    let decodedImages = graph.withStream(streamContext) {
      var decodedImages = [DynamicGraph.Tensor<T>]()
      for y in 0..<yTiles {
        let yOfs = y * (tileSize.height - tileOverlap * 2) + (y > 0 ? tileOverlap : 0)
        let (inputStartYPad, inputEndYPad) = paddedTileStartAndEnd(
          iOfs: yOfs, length: shape[1], tileSize: tileSize.height, tileOverlap: tileOverlap)
        for x in 0..<xTiles {
          let xOfs = x * (tileSize.width - tileOverlap * 2) + (x > 0 ? tileOverlap : 0)
          let (inputStartXPad, inputEndXPad) = paddedTileStartAndEnd(
            iOfs: xOfs, length: shape[2], tileSize: tileSize.width, tileOverlap: tileOverlap)
          decodedImages.append(
            decoder(
              inputs: z[
                0..<1, inputStartYPad..<inputEndYPad, inputStartXPad..<inputEndXPad, 0..<channels
              ].copied())[0].as(of: T.self))
        }
      }
      return decodedImages
    }
    let decodedRawValues = decodedImages.map { $0.rawValue.toCPU() }
    var result = Tensor<T>(.CPU, .NHWC(1, shape[1] * 8, shape[2] * 8, channels))
    var yWeightsAndIndexes = [[(weight: Float, index: Int, offset: Int)]]()
    for j in 0..<(shape[1] * 8) {
      var weightAndIndex = [(weight: Float, index: Int, offset: Int)]()
      let y1 = min(
        max((j - tileOverlap * 8) / ((tileSize.height - tileOverlap * 2) * 8), 0), yTiles - 1)
      let y1Ofs = y1 * (tileSize.height - tileOverlap * 2) * 8 + (y1 > 0 ? tileOverlap * 8 : 0)
      let (inputStartY1Pad, inputEndY1Pad) = paddedTileStartAndEnd(
        iOfs: y1Ofs, length: shape[1] * 8, tileSize: tileSize.height * 8,
        tileOverlap: tileOverlap * 8)
      if j >= inputStartY1Pad && j < inputEndY1Pad {
        weightAndIndex.append(
          (
            weight: Float(min(j - inputStartY1Pad, inputEndY1Pad - j)), index: y1,
            offset: j - inputStartY1Pad
          ))
      }
      if y1 + 1 < yTiles {
        let y2Ofs = (y1 + 1) * (tileSize.height - tileOverlap * 2) * 8 + tileOverlap * 8
        let (inputStartY2Pad, inputEndY2Pad) = paddedTileStartAndEnd(
          iOfs: y2Ofs, length: shape[1] * 8, tileSize: tileSize.height * 8,
          tileOverlap: tileOverlap * 8)
        if j >= inputStartY2Pad && j < inputEndY2Pad {
          weightAndIndex.append(
            (
              weight: Float(min(j - inputStartY2Pad, inputEndY2Pad - j)), index: y1 + 1,
              offset: j - inputStartY2Pad
            ))
        }
      }
      if y1 - 1 >= 0 {
        let y0Ofs =
          (y1 - 1) * (tileSize.height - tileOverlap * 2) * 8 + (y1 - 1 > 0 ? tileOverlap * 8 : 0)
        let (inputStartY0Pad, inputEndY0Pad) = paddedTileStartAndEnd(
          iOfs: y0Ofs, length: shape[1] * 8, tileSize: tileSize.height * 8,
          tileOverlap: tileOverlap * 8)
        if j >= inputStartY0Pad && j < inputEndY0Pad {
          weightAndIndex.append(
            (
              weight: Float(min(j - inputStartY0Pad, inputEndY0Pad - j)), index: y1 - 1,
              offset: j - inputStartY0Pad
            ))
        }
      }
      // Now normalize the weights.
      let totalWeight = weightAndIndex.reduce(0) { $0 + $1.weight }
      yWeightsAndIndexes.append(
        weightAndIndex.map {
          if totalWeight > 0 {  // Fix boundary condition.
            return (weight: $0.weight / totalWeight, index: $0.index, offset: $0.offset)
          } else {
            return (weight: 1, index: $0.index, offset: $0.offset)
          }
        })
    }
    var xWeightsAndIndexes = [[(weight: Float, index: Int, offset: Int)]]()
    for i in 0..<(shape[2] * 8) {
      var weightAndIndex = [(weight: Float, index: Int, offset: Int)]()
      let x1 = min(
        max((i - tileOverlap * 8) / ((tileSize.width - tileOverlap * 2) * 8), 0), xTiles - 1)
      let x1Ofs = x1 * (tileSize.width - tileOverlap * 2) * 8 + (x1 > 0 ? tileOverlap * 8 : 0)
      let (inputStartX1Pad, inputEndX1Pad) = paddedTileStartAndEnd(
        iOfs: x1Ofs, length: shape[2] * 8, tileSize: tileSize.width * 8,
        tileOverlap: tileOverlap * 8)
      if i >= inputStartX1Pad && i < inputEndX1Pad {
        weightAndIndex.append(
          (
            weight: Float(min(i - inputStartX1Pad, inputEndX1Pad - i)), index: x1,
            offset: i - inputStartX1Pad
          ))
      }
      if x1 + 1 < xTiles {
        let x2Ofs = (x1 + 1) * (tileSize.width - tileOverlap * 2) * 8 + tileOverlap * 8
        let (inputStartX2Pad, inputEndX2Pad) = paddedTileStartAndEnd(
          iOfs: x2Ofs, length: shape[2] * 8, tileSize: tileSize.width * 8,
          tileOverlap: tileOverlap * 8)
        if i >= inputStartX2Pad && i < inputEndX2Pad {
          weightAndIndex.append(
            (
              weight: Float(min(i - inputStartX2Pad, inputEndX2Pad - i)), index: x1 + 1,
              offset: i - inputStartX2Pad
            ))
        }
      }
      if x1 - 1 >= 0 {
        let x0Ofs =
          (x1 - 1) * (tileSize.width - tileOverlap * 2) * 8 + (x1 - 1 > 0 ? tileOverlap * 8 : 0)
        let (inputStartX0Pad, inputEndX0Pad) = paddedTileStartAndEnd(
          iOfs: x0Ofs, length: shape[2] * 8, tileSize: tileSize.width * 8,
          tileOverlap: tileOverlap * 8)
        if i >= inputStartX0Pad && i < inputEndX0Pad {
          weightAndIndex.append(
            (
              weight: Float(min(i - inputStartX0Pad, inputEndX0Pad - i)), index: x1 - 1,
              offset: i - inputStartX0Pad
            ))
        }
      }
      // Now normalize the weights.
      let totalWeight = weightAndIndex.reduce(0) { $0 + $1.weight }
      xWeightsAndIndexes.append(
        weightAndIndex.map {
          if totalWeight > 0 {  // Fix boundary condition.
            return (weight: $0.weight / totalWeight, index: $0.index, offset: $0.offset)
          } else {
            return (weight: 1, index: $0.index, offset: $0.offset)
          }
        })
    }
    result.withUnsafeMutableBytes {
      guard var fp = $0.baseAddress?.assumingMemoryBound(to: T.self) else { return }
      for j in 0..<(shape[1] * 8) {
        let yWeightAndIndex = yWeightsAndIndexes[j]
        for i in 0..<(shape[2] * 8) {
          let xWeightAndIndex = xWeightsAndIndexes[i]
          for k in 0..<channels {
            fp[k] = 0
          }
          for y in yWeightAndIndex {
            for x in xWeightAndIndex {
              let weight = T(x.weight * y.weight)
              let index = y.index * xTiles + x.index
              let tensor = decodedRawValues[index]
              tensor.withUnsafeBytes {
                guard var v = $0.baseAddress?.assumingMemoryBound(to: T.self) else { return }
                v = v + x.offset * channels + y.offset * tileSize.width * 8 * channels
                for k in 0..<channels {
                  fp[k] += v[k] * weight
                }
              }
            }
          }
          fp += channels
        }
      }
    }
    return graph.variable(result.toGPU(0))
  }
}
