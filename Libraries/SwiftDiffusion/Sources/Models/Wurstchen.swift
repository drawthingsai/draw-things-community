import Foundation
import NNC

private func ResBlock(
  prefix: String, batchSize: Int, channels: Int, skip: Bool, of dataType: DataType? = nil
) -> Model {
  let x = Input()
  let depthwise = Convolution(
    groups: channels, filters: channels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW,
    name: "resblock")
  var out = depthwise(x)
  let norm = LayerNorm(epsilon: 1e-6, axis: [3], elementwiseAffine: false)
  out = norm(out)
  if let dataType = dataType {
    // Do the rest of the computation at the data type of specified.
    out = out.to(dataType)
  }
  let xSkip: Input?
  if skip {
    let xSkipIn = Input()
    out = Functional.concat(axis: 3, out, xSkipIn)
    xSkip = xSkipIn
  } else {
    xSkip = nil
  }
  let convIn = Convolution(
    groups: 1, filters: channels * 4, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW,
    name: "resblock")
  out = convIn(out).GELU()
  let Gx = out.reduced(.norm2, axis: [1, 2])
  let Nx = Gx .* (1 / Gx.reduced(.mean, axis: [3])) + 1e-6
  let gamma = Parameter<FloatType>(
    .GPU(0), .NHWC(1, 1, 1, channels * 4), initBound: 1, name: "resblock")
  let beta = Parameter<FloatType>(
    .GPU(0), .NHWC(1, 1, 1, channels * 4), initBound: 1, name: "resblock")
  out = gamma .* (out .* Nx) + beta + out
  let convOut = Convolution(
    groups: 1, filters: channels, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW)
  if dataType != nil {
    out = convOut(out).to(of: x) + x
  } else {
    out = convOut(out) + x
  }
  if let xSkip = xSkip {
    return Model([x, xSkip], [out])
  } else {
    return Model([x], [out])
  }
}

private func TimestepBlock(
  prefix: String, batchSize: Int, timeEmbedSize: Int, channels: Int, tConds: [String]
) -> Model {
  let x = Input()
  let rEmbed = Input()
  let mapper = Dense(count: channels * 2, name: "timestepblock")
  var gate = mapper(
    rEmbed.reshaped(
      [batchSize, timeEmbedSize], offset: [0, 0], strides: [timeEmbedSize * (tConds.count + 1), 1]))
  var otherMappers = [Model]()
  for i in 0..<tConds.count {
    let otherMapper = Dense(count: channels * 2, name: "timestepblock")
    gate =
      gate
      + otherMapper(
        rEmbed.reshaped(
          [batchSize, timeEmbedSize], offset: [0, timeEmbedSize * (i + 1)],
          strides: [timeEmbedSize * (tConds.count + 1), 1]))
    otherMappers.append(otherMapper)
  }
  var out: Model.IO = x
  out =
    out
    .* (1
      + gate.reshaped(
        [batchSize, 1, 1, channels], offset: [0, 0, 0, 0],
        strides: [channels * 2, channels * 2, channels * 2, 1]))
    + gate.reshaped(
      [batchSize, 1, 1, channels], offset: [0, 0, 0, channels],
      strides: [channels * 2, channels * 2, channels * 2, 1])
  return Model([x, rEmbed], [out])
}

private func MultiHeadAttention(
  prefix: String, k: Int, h: Int, b: Int, hw: Int, t: (Int, Int),
  usesFlashAttention: FlashAttentionLevel
) -> Model {
  let x = Input()
  let key = Input()
  let value = Input()
  let tokeys = Dense(count: k * h, name: "\(prefix).keys")
  let toqueries = Dense(count: k * h, name: "queries")
  let tovalues = Dense(count: k * h, name: "\(prefix).values")
  var keys = tokeys(x).reshaped([b, hw, h, k])
  var queries: Model.IO
  if usesFlashAttention != .scaleMerged {
    queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, hw, h, k])
  } else {
    queries = toqueries(x).reshaped([b, hw, h, k])
  }
  var values = tovalues(x).reshaped([b, hw, h, k])
  if usesFlashAttention == .scale1 || usesFlashAttention == .scaleMerged {
    keys = Functional.concat(axis: 1, keys, key)
    values = Functional.concat(axis: 1, values, value)
    if t.0 == t.1 || b == 1 {
      let scaledDotProductAttention: ScaledDotProductAttention
      if usesFlashAttention == .scale1 {
        scaledDotProductAttention = ScaledDotProductAttention(
          scale: 1, multiHeadOutputProjectionFused: true, name: "unifyheads")
      } else {
        scaledDotProductAttention = ScaledDotProductAttention(
          scale: 1.0 / Float(k).squareRoot(),
          multiHeadOutputProjectionFused: true, name: "unifyheads")
      }
      let out = scaledDotProductAttention(queries, keys, values).reshaped([b, hw, k * h])
      return Model([x, key, value], [out])
    } else {
      let b0 = b / 2
      let keys0 = keys.reshaped(
        [b0, hw + t.0, h, k], offset: [0, 0, 0, 0],
        strides: [(hw + max(t.0, t.1)) * h * k, h * k, k, 1]
      )
      let queries0 = queries.reshaped([b0, hw, h, k])
      let values0 = values.reshaped(
        [b0, hw + t.0, h, k], offset: [0, 0, 0, 0],
        strides: [(hw + max(t.0, t.1)) * h * k, h * k, k, 1]
      )
      let out0 = ScaledDotProductAttention(scale: 1.0 / Float(k).squareRoot())(
        queries0, keys0, values0
      ).reshaped([b0, hw, h * k])
      let keys1 = keys.reshaped(
        [b0, hw + t.1, h, k], offset: [b0, 0, 0, 0],
        strides: [(hw + max(t.0, t.1)) * h * k, h * k, k, 1]
      )
      let queries1 = queries.reshaped(
        [b0, hw, h, k], offset: [b0, 0, 0, 0], strides: [h * hw * k, h * k, k, 1])
      let values1 = values.reshaped(
        [b0, hw + t.1, h, k], offset: [b0, 0, 0, 0],
        strides: [(hw + max(t.0, t.1)) * h * k, h * k, k, 1]
      )
      let out1 = ScaledDotProductAttention(scale: 1.0 / Float(k).squareRoot())(
        queries1, keys1, values1
      ).reshaped([b0, hw, h * k])
      var out = Functional.concat(axis: 0, out0, out1)
      let unifyheads = Dense(count: k * h, name: "unifyheads")
      out = unifyheads(out)
      return (Model([x, key, value], [out]))
    }
  } else {
    queries = queries.transposed(1, 2)
    if t.0 == t.1 || b == 1 {
      // We don't need to implement split head because these are very small values.
      keys = keys.transposed(1, 2)
      keys = Functional.concat(axis: 2, keys, key)
      values = values.transposed(1, 2)
      values = Functional.concat(axis: 2, values, value)
      var dot = Matmul(transposeB: (2, 3))(queries, keys)
      dot = dot.reshaped([b * h * hw, hw + t.0])
      dot = dot.softmax()
      dot = dot.reshaped([b, h, hw, hw + t.0])
      var out = dot * values
      out = out.reshaped([b, h, hw, k]).transposed(1, 2).reshaped([b, hw, h * k])
      let unifyheads = Dense(count: k * h, name: "unifyheads")
      out = unifyheads(out)
      return Model([x, key, value], [out])
    } else {
      keys = Functional.concat(axis: 1, keys, key)
      values = Functional.concat(axis: 1, values, value)
      let b0 = b / 2
      var keys0 = keys.reshaped(
        [b0, hw + t.0, h, k], offset: [0, 0, 0, 0],
        strides: [(hw + max(t.0, t.1)) * h * k, h * k, k, 1]
      )
      let queries0 = queries.reshaped([b0, h, hw, k])
      var values0 = values.reshaped(
        [b0, hw + t.0, h, k], offset: [0, 0, 0, 0],
        strides: [(hw + max(t.0, t.1)) * h * k, h * k, k, 1]
      )
      keys0 = keys0.transposed(1, 2)
      values0 = values0.transposed(1, 2)
      var dot0 = Matmul(transposeB: (2, 3))(queries0, keys0)
      dot0 = dot0.reshaped([b0 * h * hw, hw + t.0])
      dot0 = dot0.softmax()
      dot0 = dot0.reshaped([b0, h, hw, hw + t.0])
      var out0 = dot0 * values0
      out0 = out0.reshaped([b0, h, hw, k]).transposed(1, 2).reshaped([b0, hw, h * k])
      var keys1 = keys.reshaped(
        [b0, hw + t.1, h, k], offset: [b0, 0, 0, 0],
        strides: [(hw + max(t.0, t.1)) * h * k, h * k, k, 1]
      )
      let queries1 = queries.reshaped(
        [b0, h, hw, k], offset: [b0, 0, 0, 0], strides: [h * hw * k, hw * k, k, 1])
      var values1 = values.reshaped(
        [b0, hw + t.1, h, k], offset: [b0, 0, 0, 0],
        strides: [(hw + max(t.0, t.1)) * h * k, h * k, k, 1]
      )
      keys1 = keys1.transposed(1, 2)
      values1 = values1.transposed(1, 2)
      var dot1 = Matmul(transposeB: (2, 3))(queries1, keys1)
      dot1 = dot1.reshaped([b0 * h * hw, hw + t.1])
      dot1 = dot1.softmax()
      dot1 = dot1.reshaped([b0, h, hw, hw + t.1])
      var out1 = dot1 * values1
      out1 = out1.reshaped([b0, h, hw, k]).transposed(1, 2).reshaped([b0, hw, h * k])
      var out = Functional.concat(axis: 0, out0, out1)
      let unifyheads = Dense(count: k * h, name: "unifyheads")
      out = unifyheads(out).reshaped([b, hw, h * k])
      return Model([x, key, value], [out])
    }
  }
}

private func AttnBlock(
  prefix: String, batchSize: Int, channels: Int, nHead: Int, height: Int, width: Int, t: (Int, Int),
  usesFlashAttention: FlashAttentionLevel, of dataType: DataType? = nil
) -> Model {
  let x = Input()
  let key = Input()
  let value = Input()
  let norm = LayerNorm(epsilon: 1e-6, axis: [3], elementwiseAffine: false)
  var out = norm(x).reshaped([batchSize, height * width, channels])
  if let dataType = dataType {
    out = out.to(dataType)
  }
  let k = channels / nHead
  let multiHeadAttention = MultiHeadAttention(
    prefix: prefix, k: k, h: nHead, b: batchSize, hw: height * width, t: t,
    usesFlashAttention: usesFlashAttention)
  let attnOut = multiHeadAttention(out, key, value).identity().reshaped([
    batchSize, height, width, channels,
  ])
  if dataType != nil {
    out = x + attnOut.to(of: x)
  } else {
    out = x + attnOut
  }
  return Model([x, key, value], [out])
}

func AttnBlockFixed(
  prefix: String, batchSize: Int, channels: Int, nHead: Int, t: (Int, Int),
  usesFlashAttention: FlashAttentionLevel
) -> Model {
  let kv = Input()
  let kvMapper = Dense(count: channels, name: "kv_mapper")
  let kvOut = kvMapper(kv.swish())
  let tokeys = Dense(count: channels, name: "\(prefix).keys")
  let tovalues = Dense(count: channels, name: "\(prefix).values")
  let k = channels / nHead
  if t.0 == t.1 {
    switch usesFlashAttention {
    case .none:
      let keys = tokeys(kvOut).reshaped([batchSize, t.0, nHead, k]).transposed(1, 2)
      let values = tovalues(kvOut).reshaped([batchSize, t.0, nHead, k]).transposed(1, 2)
      return Model([kv], [keys, values])
    case .scale1, .scaleMerged:
      let keys = tokeys(kvOut).reshaped([batchSize, t.0, nHead, k])
      let values = tovalues(kvOut).reshaped([batchSize, t.0, nHead, k])
      return Model([kv], [keys, values])
    }
  } else {
    let keys = tokeys(kvOut).reshaped([batchSize, max(t.0, t.1), nHead, k])
    let values = tovalues(kvOut).reshaped([batchSize, max(t.0, t.1), nHead, k])
    return Model([kv], [keys, values])
  }
}

func WurstchenStageCFixed(batchSize: Int, t: (Int, Int), usesFlashAttention: FlashAttentionLevel)
  -> Model
{
  let clipText = Input()
  let clipTextPooled = Input()
  let clipImg = Input()
  let clipTextMapper = Dense(count: 2048, name: "clip_text_mapper")
  let clipTextMapped = clipTextMapper(clipText)
  let clipTextPooledMapper = Dense(count: 2048 * 4, name: "clip_text_pool_mapper")
  let clipTextPooledMapped = clipTextPooledMapper(clipTextPooled).reshaped([batchSize, 4, 2048])
  let clipImgMapper = Dense(count: 2048 * 4, name: "clip_image_mapper")
  let clipImgMapped = clipImgMapper(clipImg).reshaped([batchSize, 4, 2048])
  let clipNorm = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let clip = clipNorm(
    Functional.concat(axis: 1, clipTextPooledMapped, clipImgMapped, clipTextMapped))
  let blocks: [[Int]] = [[8, 24], [24, 8]]
  var outs = [Model.IO]()
  for i in 0..<2 {
    for j in 0..<blocks[0][i] {
      let attnBlockFixed = AttnBlockFixed(
        prefix: "down_blocks.\(i).\(j * 3 + 2)", batchSize: batchSize, channels: 2048, nHead: 32,
        t: t, usesFlashAttention: usesFlashAttention)
      let out = attnBlockFixed(clip)
      outs.append(out)
    }
  }
  for i in 0..<2 {
    for j in 0..<blocks[1][i] {
      let attnBlockFixed = AttnBlockFixed(
        prefix: "up_blocks.\(i).\(j * 3 + 2)", batchSize: batchSize, channels: 2048, nHead: 32,
        t: t, usesFlashAttention: usesFlashAttention)
      let out = attnBlockFixed(clip)
      outs.append(out)
    }
  }
  return Model([clipText, clipTextPooled, clipImg], outs)
}

func WurstchenStageC(
  batchSize: Int, height: Int, width: Int, t: (Int, Int), usesFlashAttention: FlashAttentionLevel
) -> Model {
  let x = Input()
  let rEmbed = Input()
  let conv2d = Convolution(
    groups: 1, filters: 2048, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW)
  var out = conv2d(x)
  let normIn = LayerNorm(epsilon: 1e-6, axis: [3], elementwiseAffine: false)
  out = normIn(out)

  let blocks: [[Int]] = [[8, 24], [24, 8]]
  var levelOutputs = [Model.IO]()
  var kvs = [Input]()
  for i in 0..<2 {
    if i > 0 {
      let norm = LayerNorm(epsilon: 1e-6, axis: [3], elementwiseAffine: false)
      out = norm(out)
      let downscaler = Convolution(
        groups: 1, filters: 2048, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW)
      out = downscaler(out)
    }
    for j in 0..<blocks[0][i] {
      let resBlock = ResBlock(
        prefix: "down_blocks.\(i).\(j * 3)", batchSize: batchSize, channels: 2048, skip: false)
      out = resBlock(out)
      let timestepBlock = TimestepBlock(
        prefix: "down_blocks.\(i).\(j * 3 + 1)", batchSize: batchSize, timeEmbedSize: 64,
        channels: 2048, tConds: ["sca", "crp"])
      out = timestepBlock(out, rEmbed)
      let attnBlock = AttnBlock(
        prefix: "down_blocks.\(i).\(j * 3 + 2)", batchSize: batchSize, channels: 2048, nHead: 32,
        height: height, width: width, t: t, usesFlashAttention: usesFlashAttention)
      let key = Input()
      let value = Input()
      out = attnBlock(out, key, value)
      kvs.append(key)
      kvs.append(value)
    }
    if i < 2 - 1 {
      levelOutputs.append(out)
    }
  }

  var skip: Model.IO? = nil
  for i in 0..<2 {
    for j in 0..<blocks[1][i] {
      // For the last layers, we start to accumulate values through the residual connection, and that will exceed FP16.
      // Thus, we convert to FP32 until the normalization layer of attention block so attention is done at FP16 but
      // cheap computations such as ResBlock / timestepBlock are done in FP32.
      if i == 1 && j > 0 {
        out = out.to(.Float32)
      }
      let resBlock: Model
      if i == 2 - 1 && j > 0 {
        // Even input is Float32, we will do the rest of the computation of ResBlock in Float16.
        resBlock = ResBlock(
          prefix: "up_blocks.\(i).\(j * 3)", batchSize: batchSize, channels: 2048,
          skip: skip != nil, of: .Float16)
      } else {
        resBlock = ResBlock(
          prefix: "up_blocks.\(i).\(j * 3)", batchSize: batchSize, channels: 2048, skip: skip != nil
        )
      }
      if let skip = skip {
        out = resBlock(out, skip)
      } else {
        out = resBlock(out)
      }
      skip = nil
      // No normalization layer in Timestep block, still do this in Float32.
      let timestepBlock = TimestepBlock(
        prefix: "up_blocks.\(i).\(j * 3 + 1)", batchSize: batchSize, timeEmbedSize: 64,
        channels: 2048, tConds: ["sca", "crp"])
      var rEmbed: Model.IO = rEmbed
      if i == 2 - 1 && j > 0 {
        rEmbed = rEmbed.to(.Float32)
      }
      out = timestepBlock(out, rEmbed)
      let attnBlockDataType: DataType?
      if i == 2 - 1 && j > 0 {
        attnBlockDataType = .Float16
      } else {
        attnBlockDataType = nil
      }
      let attnBlock = AttnBlock(
        prefix: "up_blocks.\(i).\(j * 3 + 2)", batchSize: batchSize, channels: 2048, nHead: 32,
        height: height, width: width, t: t, usesFlashAttention: usesFlashAttention,
        of: attnBlockDataType)
      let key = Input()
      let value = Input()
      out = attnBlock(out, key, value)
      kvs.append(key)
      kvs.append(value)
    }
    if i < 2 - 1 {
      let norm = LayerNorm(epsilon: 1e-6, axis: [3], elementwiseAffine: false)
      out = norm(out)
      let upscaler = Convolution(
        groups: 1, filters: 2048, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW)
      out = upscaler(out)
      skip = levelOutputs.removeLast()
    }
  }

  let normOut = LayerNorm(epsilon: 1e-6, axis: [3], elementwiseAffine: false)
  out = normOut(out).to(of: x)
  let convOut = Convolution(
    groups: 1, filters: 16, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW)
  out = convOut(out)

  return Model([x, rEmbed] + kvs, [out])
}

private func SpatialMapper(prefix: String, cHidden: Int) -> Model {
  let x = Input()
  let convIn = Convolution(
    groups: 1, filters: cHidden * 4, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW)
  var out = convIn(x)
  let convOut = Convolution(
    groups: 1, filters: cHidden, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW)
  out = convOut(out.GELU())
  let normOut = LayerNorm(epsilon: 1e-6, axis: [3], elementwiseAffine: false)
  out = normOut(out)
  return Model([x], [out])
}

func WurstchenStageBFixed(
  batchSize: Int, height: Int, width: Int, effnetHeight: Int, effnetWidth: Int,
  usesFlashAttention: FlashAttentionLevel
) -> Model {
  let effnet = Input()
  let pixels = Input()
  let clip = Input()
  let cHidden: [Int] = [320, 640, 1280, 1280]
  let effnetMapper = SpatialMapper(
    prefix: "effnet_mapper", cHidden: cHidden[0])
  var out = effnetMapper(
    Upsample(
      .bilinear, widthScale: Float(width / 2) / Float(effnetWidth),
      heightScale: Float(height / 2) / Float(effnetHeight), alignCorners: true)(effnet))
  let pixelsMapper = SpatialMapper(
    prefix: "pixels_mapper", cHidden: cHidden[0])
  out =
    out
    + Upsample(
      .bilinear, widthScale: Float(width / 2) / Float(8), heightScale: Float(height / 2) / Float(8),
      alignCorners: true)(pixelsMapper(pixels))
  var outs = [out]
  let clipMapper = Dense(count: 1280 * 4)
  let clipMapped = clipMapper(clip).reshaped([batchSize, 4, 1280])
  let clipNorm = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let clipNormed = clipNorm(clipMapped)
  let blocks: [[Int]] = [[2, 6, 28, 6], [6, 28, 6, 2]]
  let attentions: [[Bool]] = [[false, false, true, true], [true, true, false, false]]
  for i in 0..<4 {
    let attention = attentions[0][i]
    for j in 0..<blocks[0][i] {
      if attention {
        let attnBlockFixed = AttnBlockFixed(
          prefix: "down_blocks.\(i).\(j * 3 + 2)", batchSize: batchSize, channels: cHidden[i],
          nHead: 20, t: (4, 4), usesFlashAttention: usesFlashAttention)
        let out = attnBlockFixed(clipNormed)
        outs.append(out)
      }
    }
  }
  for i in 0..<4 {
    let attention = attentions[1][i]
    for j in 0..<blocks[1][i] {
      if attention {
        let attnBlockFixed = AttnBlockFixed(
          prefix: "up_blocks.\(i).\(j * 3 + 2)", batchSize: batchSize, channels: cHidden[3 - i],
          nHead: 20, t: (4, 4), usesFlashAttention: usesFlashAttention)
        let out = attnBlockFixed(clipNormed)
        outs.append(out)
      }
    }
  }
  return Model([effnet, pixels, clip], outs)
}

func WurstchenStageB(
  batchSize: Int, cIn: Int, height: Int, width: Int, usesFlashAttention: FlashAttentionLevel
)
  -> Model
{
  let x = Input()
  let rEmbed = Input()
  let effnetAndPixels = Input()
  let cHidden: [Int] = [320, 640, 1280, 1280]
  let conv2d = Convolution(
    groups: 1, filters: cHidden[0], filterSize: [2, 2], hint: Hint(stride: [2, 2]), format: .OIHW)
  var out = conv2d(x)
  let normIn = LayerNorm(epsilon: 1e-6, axis: [3], elementwiseAffine: false)
  out = normIn(out) + effnetAndPixels
  let blocks: [[Int]] = [[2, 6, 28, 6], [6, 28, 6, 2]]
  let attentions: [[Bool]] = [[false, false, true, true], [true, true, false, false]]
  var levelOutputs = [Model.IO]()
  var height = height / 2
  var width = width / 2
  var kvs = [Input]()
  for i in 0..<4 {
    if i > 0 {
      let norm = LayerNorm(epsilon: 1e-6, axis: [3], elementwiseAffine: false)
      out = norm(out)
      let downscaler = Convolution(
        groups: 1, filters: cHidden[i], filterSize: [2, 2], hint: Hint(stride: [2, 2]),
        format: .OIHW)
      out = downscaler(out)
      height = height / 2
      width = width / 2
    }
    let attention = attentions[0][i]
    for j in 0..<blocks[0][i] {
      let resBlock = ResBlock(
        prefix: "down_blocks.\(i).\(j * (attention ? 3 : 2))", batchSize: batchSize,
        channels: cHidden[i], skip: false)
      out = resBlock(out)
      let timestepBlock = TimestepBlock(
        prefix: "down_blocks.\(i).\(j * (attention ? 3 : 2) + 1)", batchSize: batchSize,
        timeEmbedSize: 64,
        channels: cHidden[i], tConds: ["sca"])
      out = timestepBlock(out, rEmbed)
      if attention {
        let attnBlock = AttnBlock(
          prefix: "down_blocks.\(i).\(j * 3 + 2)", batchSize: batchSize, channels: cHidden[i],
          nHead: 20, height: height, width: width, t: (4, 4), usesFlashAttention: usesFlashAttention
        )
        let key = Input()
        let value = Input()
        out = attnBlock(out, key, value)
        kvs.append(key)
        kvs.append(value)
      }
    }
    if i < 4 - 1 {
      levelOutputs.append(out)
    }
  }
  var skip: Model.IO? = nil
  let blockRepeat: [Int] = [3, 3, 2, 2]
  for i in 0..<4 {
    let cSkip = skip
    skip = nil
    let attention = attentions[1][i]
    var resBlocks = [Model]()
    var timestepBlocks = [Model]()
    var attnBlocks = [Model]()
    var keyAndValue = [(Input, Input)]()
    for j in 0..<blocks[1][i] {
      let resBlock = ResBlock(
        prefix: "up_blocks.\(i).\(j * (attention ? 3 : 2))", batchSize: batchSize,
        channels: cHidden[3 - i], skip: j == 0 && cSkip != nil)
      resBlocks.append(resBlock)
      let timestepBlock = TimestepBlock(
        prefix: "up_blocks.\(i).\(j * (attention ? 3 : 2) + 1)", batchSize: batchSize,
        timeEmbedSize: 64,
        channels: cHidden[3 - i], tConds: ["sca"])
      timestepBlocks.append(timestepBlock)
      if attention {
        let attnBlock = AttnBlock(
          prefix: "up_blocks.\(i).\(j * 3 + 2)", batchSize: batchSize, channels: cHidden[3 - i],
          nHead: 20, height: height, width: width, t: (4, 4), usesFlashAttention: usesFlashAttention
        )
        attnBlocks.append(attnBlock)
        keyAndValue.append((Input(), Input()))
      }
    }
    kvs.append(contentsOf: keyAndValue.flatMap { [$0.0, $0.1] })
    for j in 0..<blockRepeat[i] {
      for k in 0..<blocks[1][i] {
        if k == 0, let cSkip = cSkip {
          out = resBlocks[k](out, cSkip)
        } else {
          out = resBlocks[k](out)
        }
        out = timestepBlocks[k](out, rEmbed)
        if attention {
          out = attnBlocks[k](out, keyAndValue[k].0, keyAndValue[k].1)
        }
      }
      // repmap.
      if j < blockRepeat[i] - 1 {
        let repmap = Convolution(
          groups: 1, filters: cHidden[3 - i], filterSize: [1, 1], hint: Hint(stride: [1, 1]),
          format: .OIHW)
        out = repmap(out)
      }
    }
    if i < 4 - 1 {
      let norm = LayerNorm(epsilon: 1e-6, axis: [3], elementwiseAffine: false)
      out = norm(out)
      let upscaler = ConvolutionTranspose(
        groups: 1, filters: cHidden[2 - i], filterSize: [2, 2], hint: Hint(stride: [2, 2]),
        format: .OIHW)
      out = upscaler(out)
      skip = levelOutputs.removeLast()
      height = height * 2
      width = width * 2
    }
  }
  let normOut = LayerNorm(epsilon: 1e-6, axis: [3], elementwiseAffine: false)
  out = normOut(out)
  let convOut = Convolution(
    groups: 1, filters: 16, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW)
  out = convOut(out).reshaped([batchSize, height, width, 4, 2, 2]).transposed(3, 4).transposed(4, 5)
    .transposed(2, 3).reshaped([batchSize, height * 2, width * 2, 4])  // This is the same as .permuted(0, 1, 4, 2, 5, 3).

  return Model([x, rEmbed, effnetAndPixels] + kvs, [out])
}

private func StageAResBlock(prefix: String, channels: Int) -> Model {
  let x = Input()
  let gammas = Parameter<FloatType>(.GPU(0), .NHWC(1, 1, 1, 6), initBound: 1)
  let norm1 = LayerNorm(epsilon: 1e-6, axis: [3], elementwiseAffine: false)
  var out =
    norm1(x) .* (1 + gammas.reshaped([1, 1, 1, 1], offset: [0, 0, 0, 0], strides: [6, 6, 6, 1]))
    + gammas.reshaped([1, 1, 1, 1], offset: [0, 0, 0, 1], strides: [6, 6, 6, 1])
  let depthwise = Convolution(
    groups: channels, filters: channels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1]), format: .OIHW)
  out = x + depthwise(out.padded(.replication, begin: [0, 1, 1, 0], end: [0, 1, 1, 0]))
    .* gammas.reshaped([1, 1, 1, 1], offset: [0, 0, 0, 2], strides: [6, 6, 6, 1])
  let norm2 = LayerNorm(epsilon: 1e-6, axis: [3], elementwiseAffine: false)
  let xTemp =
    norm2(out)
    .* (1 + gammas.reshaped([1, 1, 1, 1], offset: [0, 0, 0, 3], strides: [6, 6, 6, 1]))
    + gammas.reshaped([1, 1, 1, 1], offset: [0, 0, 0, 4], strides: [6, 6, 6, 1])
  let convIn = Convolution(
    groups: 1, filters: channels * 4, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW)
  let convOut = Convolution(
    groups: 1, filters: channels, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW)
  out = out + convOut(convIn(xTemp).GELU())
    .* gammas.reshaped([1, 1, 1, 1], offset: [0, 0, 0, 5], strides: [6, 6, 6, 1])
  return Model([x], [out])
}

func WurstchenStageAEncoder(batchSize: Int) -> Model {
  let x = Input()
  let cHidden = [192, 384]
  let convIn = Convolution(
    groups: 1, filters: cHidden[0], filterSize: [2, 2], hint: Hint(stride: [2, 2]), format: .OIHW)
  var out = convIn(x)
  var j = 0
  for i in 0..<cHidden.count {
    if i > 0 {
      let conv2d = Convolution(
        groups: 1, filters: cHidden[i], filterSize: [4, 4],
        hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
      out = conv2d(out)
      j += 1
    }
    let resBlock = StageAResBlock(
      prefix: "down_blocks.\(j)", channels: cHidden[i])
    out = resBlock(out)
    j += 1
  }
  let conv2d = Convolution(
    groups: 1, filters: 4, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW)
  out = conv2d(out)
  return Model([x], [out])
}

func WurstchenStageADecoder(batchSize: Int, height: Int, width: Int) -> Model {
  let x = Input()
  let cHidden = [384, 192]
  let convIn = Convolution(
    groups: 1, filters: cHidden[0], filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW)
  var out = convIn(x)
  var j = 1
  for i in 0..<cHidden.count {
    for _ in 0..<(i == 0 ? 12 : 1) {
      let resBlock = StageAResBlock(
        prefix: "up_blocks.\(j)", channels: cHidden[i])
      out = resBlock(out)
      j += 1
    }
    if i < cHidden.count - 1 {
      let conv2d = ConvolutionTranspose(
        groups: 1, filters: cHidden[i + 1], filterSize: [4, 4],
        hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
      out = conv2d(out)
      j += 1
    }
  }
  let convOut = Convolution(
    groups: 1, filters: 12, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW)
  out =
    convOut(out).reshaped([batchSize, height, width, 3, 2, 2]).transposed(3, 4).transposed(4, 5)
    .transposed(2, 3).reshaped([batchSize, height * 2, width * 2, 3]) * 2 - 1  // This is the same as .permuted(0, 1, 4, 2, 5, 3).
  return Model([x], [out])
}

func rEmbedding(timesteps: Float, batchSize: Int, embeddingSize: Int, maxPeriod: Int) -> Tensor<
  Float
> {
  precondition(embeddingSize % 2 == 0)
  var embedding = Tensor<Float>(.CPU, .WC(batchSize, embeddingSize))
  let r = timesteps * Float(maxPeriod)
  let half = embeddingSize / 2
  for i in 0..<half {
    let freq: Float = exp(-log(Float(maxPeriod)) * Float(i) / Float(half - 1)) * r
    let sinFreq = sin(freq)
    let cosFreq = cos(freq)
    for j in 0..<batchSize {
      embedding[j, i] = sinFreq
      embedding[j, i + half] = cosFreq
    }
  }
  return embedding
}

private func FusedMBConv(
  prefix: String, outChannels: Int, stride: Int, filterSize: Int, skip: Bool,
  expandChannels: Int? = nil
) -> Model {
  let x = Input()
  var out: Model.IO = x
  let convOut: Model
  if let expandChannels = expandChannels {
    let conv = Convolution(
      groups: 1, filters: expandChannels, filterSize: [filterSize, filterSize],
      hint: Hint(
        stride: [stride, stride],
        border: Hint.Border(
          begin: [(filterSize - 1) / 2, (filterSize - 1) / 2],
          end: [(filterSize - 1) / 2, (filterSize - 1) / 2])), format: .OIHW)
    out = conv(out).swish()
    convOut = Convolution(
      groups: 1, filters: outChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW
    )
    out = convOut(out)
  } else {
    convOut = Convolution(
      groups: 1, filters: outChannels, filterSize: [filterSize, filterSize],
      hint: Hint(
        stride: [stride, stride],
        border: Hint.Border(
          begin: [(filterSize - 1) / 2, (filterSize - 1) / 2],
          end: [(filterSize - 1) / 2, (filterSize - 1) / 2])), format: .OIHW)
    out = convOut(out).swish()
  }
  if skip {
    out = x + out
  }
  return Model([x], [out])
}

private func MBConv(
  prefix: String, stride: Int, filterSize: Int, inChannels: Int, expandChannels: Int,
  outChannels: Int
) -> Model {
  let x = Input()
  var out: Model.IO = x
  if expandChannels != inChannels {
    let conv = Convolution(
      groups: 1, filters: expandChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]),
      format: .OIHW)
    out = conv(out).swish()
  }

  let depthwise = Convolution(
    groups: expandChannels, filters: expandChannels, filterSize: [filterSize, filterSize],
    hint: Hint(
      stride: [stride, stride],
      border: Hint.Border(
        begin: [(filterSize - 1) / 2, (filterSize - 1) / 2],
        end: [(filterSize - 1) / 2, (filterSize - 1) / 2])), format: .OIHW)
  out = depthwise(out).swish()

  // Squeeze and Excitation
  var scale = out.reduced(.mean, axis: [1, 2])
  let fc1 = Convolution(
    groups: 1, filters: inChannels / 4, filterSize: [1, 1], hint: Hint(stride: [1, 1]),
    format: .OIHW)
  scale = fc1(scale).swish()
  let fc2 = Convolution(
    groups: 1, filters: expandChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]),
    format: .OIHW)
  scale = fc2(scale).sigmoid()
  out = scale .* out

  let convOut = Convolution(
    groups: 1, filters: outChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW)
  out = convOut(out)

  if inChannels == outChannels && stride == 1 {
    out = x + out
  }
  return Model([x], [out])
}

func EfficientNetEncoder() -> Model {
  let x = Input()
  let conv = Convolution(
    groups: 1, filters: 24, filterSize: [3, 3],
    hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  var out = conv(x).swish()
  // 1.
  let backbone_1_0 = FusedMBConv(
    prefix: "backbone.1.0", outChannels: 24, stride: 1, filterSize: 3, skip: true)
  out = backbone_1_0(out)
  let backbone_1_1 = FusedMBConv(
    prefix: "backbone.1.1", outChannels: 24, stride: 1, filterSize: 3, skip: true)
  out = backbone_1_1(out)
  // 2.
  let backbone_2_0 = FusedMBConv(
    prefix: "backbone.2.0", outChannels: 48, stride: 2, filterSize: 3, skip: false,
    expandChannels: 96)
  out = backbone_2_0(out)
  for i in 1..<4 {
    let backbone_2_x = FusedMBConv(
      prefix: "backbone.2.\(i)", outChannels: 48, stride: 1, filterSize: 3, skip: true,
      expandChannels: 192)
    out = backbone_2_x(out)
  }
  // 3.
  let backbone_3_0 = FusedMBConv(
    prefix: "backbone.3.0", outChannels: 64, stride: 2, filterSize: 3, skip: false,
    expandChannels: 192)
  out = backbone_3_0(out)
  for i in 1..<4 {
    let backbone_3_x = FusedMBConv(
      prefix: "backbone.3.\(i)", outChannels: 64, stride: 1, filterSize: 3, skip: true,
      expandChannels: 256)
    out = backbone_3_x(out)
  }
  // 4.
  let backbone_4_0 = MBConv(
    prefix: "backbone.4.0", stride: 2, filterSize: 3, inChannels: 64, expandChannels: 256,
    outChannels: 128)
  out = backbone_4_0(out)
  for i in 1..<6 {
    let backbone_4_x = MBConv(
      prefix: "backbone.4.\(i)", stride: 1, filterSize: 3, inChannels: 128, expandChannels: 512,
      outChannels: 128)
    out = backbone_4_x(out)
  }
  // 5.
  let backbone_5_0 = MBConv(
    prefix: "backbone.5.0", stride: 1, filterSize: 3, inChannels: 128, expandChannels: 768,
    outChannels: 160)
  out = backbone_5_0(out)
  for i in 1..<9 {
    let backbone_5_x = MBConv(
      prefix: "backbone.5.\(i)", stride: 1, filterSize: 3, inChannels: 160, expandChannels: 960,
      outChannels: 160)
    out = backbone_5_x(out)
  }
  // 6.
  let backbone_6_0 = MBConv(
    prefix: "backbone.6.0", stride: 2, filterSize: 3, inChannels: 160, expandChannels: 960,
    outChannels: 256)
  out = backbone_6_0(out)
  for i in 1..<15 {
    let backbone_6_x = MBConv(
      prefix: "backbone.6.\(i)", stride: 1, filterSize: 3, inChannels: 256, expandChannels: 1536,
      outChannels: 256)
    out = backbone_6_x(out)
  }
  let convOut = Convolution(
    groups: 1, filters: 1280, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW)
  out = convOut(out).swish()
  let mapper = Convolution(
    groups: 1, filters: 16, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW)
  out = mapper(out)
  return Model([x], [out])
}

func WurstchenStageCPreviewer() -> Model {
  let x = Input()
  let conv1 = Convolution(
    groups: 1, filters: 512, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW)
  var out = conv1(x).GELU()
  let norm1 = Convolution(
    groups: 512, filters: 512, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW)
  out = norm1(out)

  let conv2 = Convolution(
    groups: 1, filters: 512, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = conv2(out).GELU()
  let norm2 = Convolution(
    groups: 512, filters: 512, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW)
  out = norm2(out)

  let conv3 = ConvolutionTranspose(
    groups: 1, filters: 256, filterSize: [2, 2], hint: Hint(stride: [2, 2]), format: .OIHW)
  out = conv3(out).GELU()
  let norm3 = Convolution(
    groups: 256, filters: 256, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW)
  out = norm3(out)

  let conv4 = Convolution(
    groups: 1, filters: 256, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = conv4(out).GELU()
  let norm4 = Convolution(
    groups: 256, filters: 256, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW)
  out = norm4(out)

  let conv5 = ConvolutionTranspose(
    groups: 1, filters: 128, filterSize: [2, 2], hint: Hint(stride: [2, 2]), format: .OIHW)
  out = conv5(out).GELU()
  let norm5 = Convolution(
    groups: 128, filters: 128, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW)
  out = norm5(out)

  let conv6 = Convolution(
    groups: 1, filters: 128, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = conv6(out).GELU()
  let norm6 = Convolution(
    groups: 128, filters: 128, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW)
  out = norm6(out)

  let conv7 = ConvolutionTranspose(
    groups: 1, filters: 128, filterSize: [2, 2], hint: Hint(stride: [2, 2]), format: .OIHW)
  out = conv7(out).GELU()
  let norm7 = Convolution(
    groups: 128, filters: 128, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW)
  out = norm7(out)

  let conv8 = Convolution(
    groups: 1, filters: 128, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = conv8(out).GELU()

  let convOut = Convolution(
    groups: 1, filters: 3, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW)
  out = convOut(out)

  return Model([x], [out])
}
