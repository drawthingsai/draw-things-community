import Foundation
import NNC

public func ZImageRotaryPositionEmbedding(
  height: Int, width: Int, tokenLength: Int, heads: Int = 1
) -> Tensor<Float> {
  var rotTensor = Tensor<Float>(.CPU, .NHWC(1, height * width + tokenLength, heads, 128))
  let dim0 = 32
  let dim1 = 48
  let dim2 = 48
  for y in 0..<height {
    for x in 0..<width {
      let i = y * width + x
      for j in 0..<heads {
        for k in 0..<(dim0 / 2) {
          let theta = Double(tokenLength + 1) * 1.0 / pow(256, Double(k) * 2 / Double(dim0))
          let sintheta = sin(theta)
          let costheta = cos(theta)
          rotTensor[0, i, j, k * 2] = Float(costheta)
          rotTensor[0, i, j, k * 2 + 1] = Float(sintheta)
        }
        for k in 0..<(dim1 / 2) {
          let theta =
            Double(y) * 1.0 / pow(256, Double(k) * 2 / Double(dim1))
          let sintheta = sin(theta)
          let costheta = cos(theta)
          rotTensor[0, i, j, (k + (dim0 / 2)) * 2] = Float(costheta)
          rotTensor[0, i, j, (k + (dim0 / 2)) * 2 + 1] = Float(sintheta)
        }
        for k in 0..<(dim2 / 2) {
          let theta =
            Double(x) * 1.0 / pow(256, Double(k) * 2 / Double(dim2))
          let sintheta = sin(theta)
          let costheta = cos(theta)
          rotTensor[0, i, j, (k + (dim0 / 2) + (dim1 / 2)) * 2] = Float(costheta)
          rotTensor[0, i, j, (k + (dim0 / 2) + (dim1 / 2)) * 2 + 1] = Float(sintheta)
        }
      }
    }
  }
  let offset = height * width
  for i in 0..<tokenLength {
    for j in 0..<heads {
      for k in 0..<(dim0 / 2) {
        let theta = Double(i + 1) * 1.0 / pow(256, Double(k) * 2 / Double(dim0))
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, offset + i, j, k * 2] = Float(costheta)
        rotTensor[0, offset + i, j, k * 2 + 1] = Float(sintheta)
      }
      for k in 0..<(dim1 / 2) {
        let theta = Double(0) * 1.0 / pow(256, Double(k) * 2 / Double(dim1))
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, offset + i, j, (k + (dim0 / 2)) * 2] = Float(costheta)
        rotTensor[0, offset + i, j, (k + (dim0 / 2)) * 2 + 1] = Float(sintheta)
      }
      for k in 0..<(dim2 / 2) {
        let theta = Double(0) * 1.0 / pow(256, Double(k) * 2 / Double(dim2))
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, offset + i, j, (k + (dim0 / 2) + (dim1 / 2)) * 2] = Float(costheta)
        rotTensor[0, offset + i, j, (k + (dim0 / 2) + (dim1 / 2)) * 2 + 1] = Float(sintheta)
      }
    }
  }
  return rotTensor
}

private func MLPEmbedder(channels: Int, intermediateSize: Int, name: String) -> (
  Model, Model, Model
) {
  let x = Input()
  let fc0 = Dense(count: intermediateSize, name: "\(name)_embedder_0")
  var out = fc0(x).swish()
  let fc2 = Dense(count: channels, name: "\(name)_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

private func FeedForward(
  hiddenSize: Int, intermediateSize: Int, scaleFactor: Float?, name: String = ""
)
  -> (
    Model, Model, Model, Model
  )
{
  let x = Input()
  let w1 = Dense(count: intermediateSize, noBias: true, name: "\(name)_gate_proj")
  let w3 = Dense(count: intermediateSize, noBias: true, name: "\(name)_up_proj")
  var out = w3(x)
  if let scaleFactor = scaleFactor {
    out = (1 / scaleFactor) * out
  }
  out = out .* w1(x).swish()
  let w2 = Dense(count: hiddenSize, noBias: true, name: "\(name)_down_proj")
  out = w2(out).to(.Float32)
  if let scaleFactor = scaleFactor {
    out = out * scaleFactor
  }
  return (w1, w2, w3, Model([x], [out], name: name))
}

private func ZImageTransformerBlock(
  prefix: String, name: String, k: Int, h: Int, b: Int, t: (keyValue: Int, query: Int),
  segments: [Int], scaleFactor: (Float, Float),
  modulation: Bool, usesFlashAttention: FlashAttentionLevel
) -> (Model, ModelWeightMapper) {
  let x = Input()
  let chunks: [Input]
  if modulation {
    chunks = (0..<4).map { _ in Input() }
  } else {
    chunks = []
  }
  let rot = Input()
  let tokeys = Dense(count: k * h, noBias: true, name: name.isEmpty ? "k" : "\(name)_k")
  let toqueries = Dense(count: k * h, noBias: true, name: name.isEmpty ? "q" : "\(name)_q")
  let tovalues = Dense(count: k * h, noBias: true, name: name.isEmpty ? "v" : "\(name)_v")
  let attentionNorm1 = RMSNorm(
    epsilon: 1e-5, axis: [2], name: name.isEmpty ? "attention_norm1" : "\(name)_attention_norm_1")
  var out = attentionNorm1(x)
  if modulation {
    out = chunks[0] .* out
  }
  out = out.to(.Float16)
  var keys = tokeys(out).reshaped([b, t.keyValue, h, k])
  var queries: Model.IO
  if t.keyValue != t.query {
    queries = toqueries(
      out.reshaped([b, t.query, k * h], offset: [0, 0, 0], strides: [t.0 * h * k, h * k, 1])
    ).reshaped([b, t.query, h, k])
  } else {
    queries = toqueries(out).reshaped([b, t.0, h, k])
  }
  let normK = RMSNorm(epsilon: 1e-5, axis: [3], name: name.isEmpty ? "norm_k" : "\(name)_norm_k")
  keys = normK(keys)
  let normQ = RMSNorm(epsilon: 1e-5, axis: [3], name: name.isEmpty ? "norm_q" : "\(name)_norm_q")
  queries = normQ(queries)
  if scaleFactor.0 > 1 {
    out = (1 / scaleFactor.0) * out
  }
  var values = tovalues(out)
  values = values.reshaped([b, t.0, h, k])
  if t.keyValue != t.query {
    queries = Functional.cmul(
      left: queries,
      right: rot.reshaped(
        [1, t.query, 1, k], offset: [0, 0, 0, 0], strides: [t.keyValue * k, k, k, 1]))
  } else {
    queries = Functional.cmul(left: queries, right: rot)
  }
  keys = Functional.cmul(left: keys, right: rot)
  switch usesFlashAttention {
  case .scale1:
    queries = (1.0 / Float(k).squareRoot().squareRoot()) * queries
    keys = (1.0 / Float(k).squareRoot().squareRoot()) * keys
    let scaledDotProductAttention = ScaledDotProductAttention(scale: 1)
    if segments.count > 1 {
      var offset = 0
      var outs = [Model.IO]()
      for segment in segments {
        let query = queries.reshaped(
          [b, segment, h, k], offset: [0, offset, 0, 0], strides: [t.query * h * k, h * k, k, 1])
        let key = keys.reshaped(
          [b, segment, h, k], offset: [0, offset, 0, 0], strides: [t.keyValue * h * k, h * k, k, 1])
        let value = values.reshaped(
          [b, segment, h, k], offset: [0, offset, 0, 0], strides: [t.keyValue * h * k, h * k, k, 1])
        outs.append(scaledDotProductAttention(query, key, value))
        offset += segment
      }
      out = Concat(axis: 1)(outs)
    } else {
      out = scaledDotProductAttention(
        queries, keys, values
      )
    }
  case .scaleMerged:
    let scaledDotProductAttention = ScaledDotProductAttention(scale: 1.0 / Float(k).squareRoot())
    if segments.count > 1 {
      var offset = 0
      var outs = [Model.IO]()
      for segment in segments {
        let query = queries.reshaped(
          [b, segment, h, k], offset: [0, offset, 0, 0], strides: [t.query * h * k, h * k, k, 1])
        let key = keys.reshaped(
          [b, segment, h, k], offset: [0, offset, 0, 0], strides: [t.keyValue * h * k, h * k, k, 1])
        let value = values.reshaped(
          [b, segment, h, k], offset: [0, offset, 0, 0], strides: [t.keyValue * h * k, h * k, k, 1])
        outs.append(scaledDotProductAttention(query, key, value))
        offset += segment
      }
      out = Concat(axis: 1)(outs)
    } else {
      out = scaledDotProductAttention(
        queries, keys, values
      )
    }
  case .none:
    queries = (1.0 / Float(k).squareRoot().squareRoot()) * queries
    keys = (1.0 / Float(k).squareRoot().squareRoot()) * keys
    if segments.count > 1 {
      var offset = 0
      var finalOuts = [Model.IO]()
      for segment in segments {
        var subQueries = queries.reshaped(
          [b, segment, h, k], offset: [0, offset, 0, 0], strides: [t.query * h * k, h * k, k, 1])
        var subKeys = keys.reshaped(
          [b, segment, h, k], offset: [0, offset, 0, 0], strides: [t.keyValue * h * k, h * k, k, 1])
        var subValues = values.reshaped(
          [b, segment, h, k], offset: [0, offset, 0, 0], strides: [t.keyValue * h * k, h * k, k, 1])
        subKeys = subKeys.transposed(1, 2)
        subQueries = subQueries.transposed(1, 2)
        subValues = subValues.transposed(1, 2)
        var outs = [Model.IO]()
        for i in 0..<(b * h) {
          let key = subKeys.reshaped(
            [1, segment, k], offset: [i, 0, 0], strides: [segment * k, k, 1])
          let query = subQueries.reshaped(
            [1, segment, k], offset: [i, 0, 0], strides: [segment * k, k, 1])
          let value = subValues.reshaped(
            [1, segment, k], offset: [i, 0, 0], strides: [segment * k, k, 1])
          var dot = Matmul(transposeB: (1, 2))(query, key)
          if let last = outs.last {
            dot.add(dependencies: [last])
          }
          dot = dot.reshaped([segment, segment])
          dot = dot.softmax()
          dot = dot.reshaped([1, segment, segment])
          outs.append(dot * value)
        }
        finalOuts.append(Concat(axis: 0)(outs).reshaped([b, h, segment, k]).transposed(1, 2))
        offset += segment
      }
      out = Concat(axis: 1)(finalOuts)
    } else {
      keys = keys.transposed(1, 2)
      queries = queries.transposed(1, 2)
      values = values.transposed(1, 2)
      var outs = [Model.IO]()
      for i in 0..<(b * h) {
        let key = keys.reshaped(
          [1, t.keyValue, k], offset: [i, 0, 0], strides: [t.keyValue * k, k, 1])
        let query = queries.reshaped(
          [1, t.query, k], offset: [i, 0, 0], strides: [t.query * k, k, 1])
        let value = values.reshaped(
          [1, t.keyValue, k], offset: [i, 0, 0], strides: [t.keyValue * k, k, 1])
        var dot = Matmul(transposeB: (1, 2))(query, key)
        if let last = outs.last {
          dot.add(dependencies: [last])
        }
        dot = dot.reshaped([t.query, t.keyValue])
        dot = dot.softmax()
        dot = dot.reshaped([1, t.query, t.keyValue])
        outs.append(dot * value)
      }
      out = Concat(axis: 0)(outs).reshaped([b, h, t.query, k]).transposed(1, 2)
    }
  }
  let xIn: Model.IO
  if t.keyValue > t.query {
    xIn = x.reshaped(
      [b, t.query, h * k], offset: [0, 0, 0], strides: [t.keyValue * h * k, h * k, 1])
    out = out.reshaped([b, t.query, h * k])
  } else {
    xIn = x
    out = out.reshaped([b, t.0, h * k])
  }
  let unifyheads = Dense(count: k * h, noBias: true, name: name.isEmpty ? "o" : "\(name)_o")
  out = unifyheads(out).to(of: xIn)
  if scaleFactor.0 > 1 {
    out = scaleFactor.0 * out
  }
  let attentionNorm2 = RMSNorm(
    epsilon: 1e-5, axis: [2], name: name.isEmpty ? "attention_norm2" : "\(name)_attention_norm2")
  out = attentionNorm2(out)
  if modulation {
    out = chunks[1] .* out
  }
  out = xIn + out
  let (w1, w2, w3, ffn) = FeedForward(
    hiddenSize: h * k, intermediateSize: 10_240, scaleFactor: scaleFactor.1,
    name: name.isEmpty ? "ffn" : "\(name)_ffn")
  let feedForwardNorm1 = RMSNorm(
    epsilon: 1e-5, axis: [2], name: name.isEmpty ? "ffn_norm1" : "\(name)_ffn_norm1")
  let feedForwardNorm2 = RMSNorm(
    epsilon: 1e-5, axis: [2], name: name.isEmpty ? "ffn_norm2" : "\(name)_ffn_norm2")
  let residual = out
  out = feedForwardNorm1(out)
  if modulation {
    out = chunks[2] .* out
  }
  out = out.to(.Float16)
  out = feedForwardNorm2(ffn(out))  // Already converted to Float32.
  if modulation {
    out = chunks[3] .* out
  }
  out = residual + out
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).attention.to_q.weight"] = [toqueries.weight.name]
    mapping["\(prefix).attention.to_k.weight"] = [tokeys.weight.name]
    mapping["\(prefix).attention.norm_q.weight"] = [normQ.weight.name]
    mapping["\(prefix).attention.norm_k.weight"] = [normK.weight.name]
    mapping["\(prefix).attention.to_v.weight"] = [tovalues.weight.name]
    mapping["\(prefix).attention.to_out.0.weight"] = [unifyheads.weight.name]
    mapping["\(prefix).attention_norm1.weight"] = [attentionNorm1.weight.name]
    mapping["\(prefix).attention_norm2.weight"] = [attentionNorm2.weight.name]
    mapping["\(prefix).feed_forward.w1.weight"] = [w1.weight.name]
    mapping["\(prefix).feed_forward.w2.weight"] = [w2.weight.name]
    mapping["\(prefix).feed_forward.w3.weight"] = [w3.weight.name]
    mapping["\(prefix).ffn_norm1.weight"] = [feedForwardNorm1.weight.name]
    mapping["\(prefix).ffn_norm2.weight"] = [feedForwardNorm2.weight.name]
    return mapping
  }
  return (Model([x, rot] + chunks, [out]), mapper)
}

func ZImage(
  batchSize: Int, height: Int, width: Int, textLength: Int, channels: Int, layers: Int,
  usesFlashAttention: FlashAttentionLevel
) -> (
  Model, ModelWeightMapper
) {
  let x = Input()
  let rot = Input()
  let txtIn = Input()
  let imgIn = Dense(count: channels, name: "x_embedder")
  let h = height / 2
  let w = width / 2
  var xOut = imgIn(
    x.reshaped([batchSize, h, 2, w, 2, 16]).permuted(0, 1, 3, 2, 4, 5).contiguous()
      .reshaped([batchSize, h * w, 2 * 2 * 16], format: .NHWC)
  ).to(.Float32)
  var mappers = [ModelWeightMapper]()
  var adaLNChunks = [Input]()
  let xRot = rot.reshaped([1, h * w, 1, 128])
  for i in 0..<2 {
    let chunks = (0..<4).map { _ in Input() }
    let (block, mapper) = ZImageTransformerBlock(
      prefix: "noise_refiner.\(i)", name: "noise_refiner", k: 128, h: channels / 128, b: batchSize,
      t: (keyValue: h * w, query: h * w), segments: [], scaleFactor: (4, 32), modulation: true,
      usesFlashAttention: usesFlashAttention)
    xOut = block([xOut, xRot] + chunks)
    adaLNChunks.append(contentsOf: chunks)
    mappers.append(mapper)
  }
  var out = Functional.concat(axis: 1, xOut, txtIn)
  let rotResized = rot.reshaped(.NHWC(1, h * w + textLength, 1, 128))
  for i in 0..<layers {
    let chunks = (0..<4).map { _ in Input() }
    let (block, mapper) = ZImageTransformerBlock(
      prefix: "layers.\(i)", name: "", k: 128, h: channels / 128, b: batchSize,
      t: (keyValue: h * w + textLength, query: i == layers - 1 ? h * w : h * w + textLength),
      segments: [], scaleFactor: (4, 32), modulation: true, usesFlashAttention: usesFlashAttention)
    out = block([out, rotResized] + chunks)
    adaLNChunks.append(contentsOf: chunks)
    mappers.append(mapper)
  }
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let scale = Input()
  adaLNChunks.append(scale)
  let projOut = Dense(count: 2 * 2 * 16, name: "linear_final")
  out = scale .* normFinal(out).to(.Float16)
  out = (-projOut(out)).reshaped([batchSize, h, w, 2, 2, 16]).permuted(0, 1, 3, 2, 4, 5)
    .contiguous()
    .reshaped([
      batchSize, h * 2, w * 2, 16,
    ])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["all_x_embedder.2-1.weight"] = [imgIn.weight.name]
    mapping["all_x_embedder.2-1.bias"] = [imgIn.bias.name]
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping["all_final_layer.2-1.linear.weight"] = [projOut.weight.name]
    mapping["all_final_layer.2-1.linear.bias"] = [projOut.bias.name]
    return mapping
  }
  return (Model([x, rot, txtIn] + adaLNChunks, [out]), mapper)
}

private func ZImageTransformerBlockFixed(
  prefix: String, name: String, channels: Int
) -> (Model, ModelWeightMapper) {
  let tEmbed = Input()
  let adaLNs = (0..<4).map {
    Dense(count: channels, name: name.isEmpty ? "ada_ln_\($0)" : "\(name)_ada_ln_\($0)")
  }
  var chunks = adaLNs.map { $0(tEmbed) }
  chunks[0] = (1 + chunks[0]).to(.Float32)
  chunks[1] = chunks[1].tanh().to(.Float32)
  chunks[2] = (1 + chunks[2]).to(.Float32)
  chunks[3] = chunks[3].tanh().to(.Float32)
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).adaLN_modulation.0.weight"] = ModelWeightElement(
      (0..<4).map { adaLNs[$0].weight.name })
    mapping["\(prefix).adaLN_modulation.0.bias"] = ModelWeightElement(
      (0..<4).map { adaLNs[$0].bias.name })
    return mapping
  }
  return (Model([tEmbed], chunks), mapper)
}

func ZImageFixed(
  batchSize: Int, tokenLength: (Int, Int), channels: Int, layers: Int,
  usesFlashAttention: FlashAttentionLevel
) -> (
  Model, ModelWeightMapper
) {
  let txt = Input()
  let txtRot = Input()
  let t = Input()
  let txtNorm = RMSNorm(epsilon: 1e-5, axis: [2], name: "cap_norm")
  let txtIn = Dense(count: channels, name: "cap_embedder")
  var txtOut = txtIn(txtNorm(txt))
  let roundUpTokenLength = ((tokenLength.0 + 31) / 32 * 32, (tokenLength.1 + 31) / 32 * 32)
  let capPadTokens: Input?
  if roundUpTokenLength.0 != tokenLength.0 || roundUpTokenLength.1 != tokenLength.1 {
    let padTokens = Input()
    if tokenLength.0 > 0 {
      var txtOut0 = txtOut.reshaped(
        [batchSize, tokenLength.0, channels], offset: [0, 0, 0],
        strides: [(tokenLength.0 + tokenLength.1) * channels, channels, 1]
      ).contiguous()
      if roundUpTokenLength.0 != tokenLength.0 {
        let padTokens0 = padTokens.reshaped(
          [batchSize, roundUpTokenLength.0 - tokenLength.0, channels], offset: [0, 0, 0],
          strides: [
            (roundUpTokenLength.0 - tokenLength.0 + roundUpTokenLength.1 - tokenLength.1)
              * channels, channels, 1,
          ]
        ).contiguous()
        txtOut0 = Functional.concat(axis: 1, txtOut0, padTokens0, flags: .disableOpt)
      }
      var txtOut1 = txtOut.reshaped(
        [batchSize, tokenLength.1, channels], offset: [0, tokenLength.0, 0],
        strides: [(tokenLength.0 + tokenLength.1) * channels, channels, 1]
      ).contiguous()
      if roundUpTokenLength.1 != tokenLength.1 {
        let padTokens1 = padTokens.reshaped(
          [batchSize, roundUpTokenLength.1 - tokenLength.1, channels],
          offset: [0, roundUpTokenLength.0 - tokenLength.0, 0],
          strides: [
            (roundUpTokenLength.0 - tokenLength.0 + roundUpTokenLength.1 - tokenLength.1)
              * channels, channels, 1,
          ]
        ).contiguous()
        txtOut1 = Functional.concat(axis: 1, txtOut1, padTokens1, flags: .disableOpt)
      }
      txtOut = Functional.concat(axis: 1, txtOut0, txtOut1, flags: .disableOpt)
    } else {
      if roundUpTokenLength.1 != tokenLength.1 {
        txtOut = Functional.concat(axis: 1, txtOut, padTokens, flags: .disableOpt)
      }
    }
    capPadTokens = padTokens
  } else {
    capPadTokens = nil
  }
  txtOut = txtOut.to(.Float32)
  let (timeInMlp0, timeInMlp2, timeIn) = MLPEmbedder(
    channels: 256, intermediateSize: 1024, name: "t")
  let tOut = timeIn(t)
  var mappers = [ModelWeightMapper]()
  let segments: [Int]
  if roundUpTokenLength.0 > 0 {
    segments = [roundUpTokenLength.0, roundUpTokenLength.1]
  } else {
    segments = []
  }
  for i in 0..<2 {
    let (block, mapper) = ZImageTransformerBlock(
      prefix: "context_refiner.\(i)", name: "context_refiner", k: 128, h: channels / 128,
      b: batchSize,
      t: (
        keyValue: roundUpTokenLength.0 + roundUpTokenLength.1,
        query: roundUpTokenLength.0 + roundUpTokenLength.1
      ),
      segments: segments, scaleFactor: (2, 2), modulation: false,
      usesFlashAttention: usesFlashAttention)
    txtOut = block(txtOut, txtRot)
    mappers.append(mapper)
  }
  var outs = [txtOut]
  for i in 0..<2 {
    let (block, mapper) = ZImageTransformerBlockFixed(
      prefix: "noise_refiner.\(i)", name: "noise_refiner", channels: channels)
    outs.append(block(tOut))
    mappers.append(mapper)
  }
  for i in 0..<layers {
    let (block, mapper) = ZImageTransformerBlockFixed(
      prefix: "layers.\(i)", name: "", channels: channels)
    outs.append(block(tOut))
    mappers.append(mapper)
  }
  let scale = Dense(count: channels, name: "ada_ln_final")
  outs.append(1 + scale(tOut.swish()))
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["cap_embedder.0.weight"] = [txtNorm.weight.name]
    mapping["cap_embedder.1.weight"] = [txtIn.weight.name]
    mapping["cap_embedder.1.bias"] = [txtIn.bias.name]
    mapping["t_embedder.mlp.0.weight"] = [timeInMlp0.weight.name]
    mapping["t_embedder.mlp.0.bias"] = [timeInMlp0.bias.name]
    mapping["t_embedder.mlp.2.weight"] = [timeInMlp2.weight.name]
    mapping["t_embedder.mlp.2.bias"] = [timeInMlp2.bias.name]
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping["all_final_layer.2-1.adaLN_modulation.1.weight"] = [scale.weight.name]
    mapping["all_final_layer.2-1.adaLN_modulation.1.bias"] = [scale.bias.name]
    return mapping
  }
  return (Model([txt, txtRot, t] + (capPadTokens.map { [$0] } ?? []), outs), mapper)
}
