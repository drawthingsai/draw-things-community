import DiffusionMappings
import Foundation
import NNC

private func qwenVLMRoPEAngles(
  position: (Int, Int, Int), headDim: Int, theta: Double = 5_000_000
) -> [Double] {
  let half = headDim / 2
  var angles = [Double](repeating: 0, count: half)
  for i in 0..<half {
    angles[i] = Double(position.0) / pow(theta, Double(i * 2) / Double(headDim))
  }
  let positions = [position.0, position.1, position.2]
  let sections = [24, 20, 20]
  for axis in 1...2 {
    let length = sections[axis] * 3
    var i = axis
    while i < length {
      angles[i] = Double(positions[axis]) / pow(theta, Double(i * 2) / Double(headDim))
      i += 3
    }
  }
  return angles
}

public func QwenVLRotaryEmbedding<FloatType: TensorNumeric & BinaryFloatingPoint>(
  sequenceLength: Int, of dataType: FloatType.Type = FloatType.self
) -> Tensor<FloatType> {
  let headDim = 128
  let half = headDim / 2
  var rotary = Tensor<FloatType>(.CPU, .NHWC(1, sequenceLength, 1, headDim))
  for i in 0..<sequenceLength {
    let angles = qwenVLMRoPEAngles(position: (i, i, i), headDim: headDim)
    for k in 0..<half {
      rotary[0, i, 0, k * 2] = FloatType(cos(angles[k]))
      rotary[0, i, 0, k * 2 + 1] = FloatType(sin(angles[k]))
    }
  }
  return rotary
}

public func Ideogram4TimeEmbedding<FloatType: TensorNumeric & BinaryFloatingPoint>(
  timestep: Float, dim: Int = 4_608, of dataType: FloatType.Type = FloatType.self
) -> Tensor<FloatType> {
  precondition(dim % 2 == 0)
  let half = dim / 2
  let scaledTimestep = 10_000 * timestep
  let frequencyScale = log(Float(10_000)) / Float(half - 1)
  var embedding = Tensor<FloatType>(.CPU, .WC(1, dim))
  for i in 0..<half {
    let value = scaledTimestep * exp(Float(i) * -frequencyScale)
    embedding[0, i] = FloatType(sin(value))
    embedding[0, i + half] = FloatType(cos(value))
  }
  return embedding
}

public func Ideogram4RotaryPositionEmbedding<FloatType: TensorNumeric & BinaryFloatingPoint>(
  textLength: Int, gridHeight: Int, gridWidth: Int, of dataType: FloatType.Type = FloatType.self
) -> Tensor<FloatType> {
  let headDim = 256
  let half = headDim / 2
  let imagePositionOffset = 65_536
  var rotary = Tensor<FloatType>(
    .CPU, .NHWC(1, textLength + gridHeight * gridWidth, 1, headDim))
  for i in 0..<textLength {
    let angles = qwenVLMRoPEAngles(position: (i, i, i), headDim: headDim)
    for k in 0..<half {
      rotary[0, i, 0, k * 2] = FloatType(cos(angles[k]))
      rotary[0, i, 0, k * 2 + 1] = FloatType(sin(angles[k]))
    }
  }
  for y in 0..<gridHeight {
    for x in 0..<gridWidth {
      let i = textLength + y * gridWidth + x
      let angles = qwenVLMRoPEAngles(
        position: (imagePositionOffset, imagePositionOffset + y, imagePositionOffset + x),
        headDim: headDim)
      for k in 0..<half {
        rotary[0, i, 0, k * 2] = FloatType(cos(angles[k]))
        rotary[0, i, 0, k * 2 + 1] = FloatType(sin(angles[k]))
      }
    }
  }
  return rotary
}

public func Ideogram4IndicatorIDs(textLength: Int, imageLength: Int) -> Tensor<Int32> {
  var indicator = Tensor<Int32>(.CPU, .C(textLength + imageLength))
  for i in 0..<textLength {
    indicator[i] = 0
  }
  for i in textLength..<(textLength + imageLength) {
    indicator[i] = 1
  }
  return indicator
}

private func Ideogram4Attention(
  prefix: String, tokenLength: Int, usesFlashAttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let rot = Input()
  let q = Dense(count: 4_608, noBias: true, name: "q")
  let k = Dense(count: 4_608, noBias: true, name: "k")
  let v = Dense(count: 4_608, noBias: true, name: "v")
  var queries = q(x).reshaped([1, tokenLength, 18, 256])
  let normQ = RMSNorm(epsilon: 1e-5, axis: [3], name: "norm_q")
  queries = normQ(queries)
  var keys = k(x).reshaped([1, tokenLength, 18, 256])
  let normK = RMSNorm(epsilon: 1e-5, axis: [3], name: "norm_k")
  keys = normK(keys)
  let values = v(x).reshaped([1, tokenLength, 18, 256])
  keys = Functional.cmul(left: keys, right: rot)
  queries = Functional.cmul(left: queries, right: rot)
  var out: Model.IO
  switch usesFlashAttention {
  case .none:
    let transposedKeys = keys.permuted(0, 2, 1, 3)
    let transposedQueries = ((1.0 / Float(256).squareRoot()) * queries).permuted(0, 2, 1, 3)
    let transposedValues = values.permuted(0, 2, 1, 3)
    var dot = Matmul(transposeB: (2, 3))(transposedQueries, transposedKeys)
    dot = dot.reshaped([18 * tokenLength, tokenLength])
    dot = dot.softmax()
    dot = dot.reshaped([1, 18, tokenLength, tokenLength])
    out = dot * transposedValues
    out = out.reshaped([1, 18, tokenLength, 256]).transposed(1, 2).reshaped([
      tokenLength, 4_608,
    ])
  case .scale1:
    let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
    out = scaledDotProductAttention(
      (1.0 / Float(256).squareRoot()) * queries, keys, values
    ).reshaped([tokenLength, 4_608])
  case .scaleMerged, .quantized:
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1.0 / Float(256).squareRoot(),
      flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
    out = scaledDotProductAttention(queries, keys, values).reshaped([tokenLength, 4_608])
  }
  let o = Dense(count: 4_608, noBias: true, name: "o")
  out = o(out)
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).attention.qkv.weight"] = ModelWeightElement(
      [q.weight.name, k.weight.name, v.weight.name], offsets: [0, 4_608, 9_216])
    mapping["\(prefix).attention.o.weight"] = [o.weight.name]
    mapping["\(prefix).attention.norm_q.weight"] = ModelWeightElement(
      [normQ.weight.name], interleaved: true, numberOfHeads: 1, headDimension: 256)
    mapping["\(prefix).attention.norm_k.weight"] = ModelWeightElement(
      [normK.weight.name], interleaved: true, numberOfHeads: 1, headDimension: 256)
    return mapping
  }
  return (mapper, Model([x, rot], [out]))
}

private func Ideogram4MLP(prefix: String) -> (ModelWeightMapper, Model) {
  let x = Input()
  let w1 = Dense(count: 12_288, noBias: true, name: "w1")
  let w3 = Dense(count: 12_288, noBias: true, name: "w3")
  let w2 = Dense(count: 4_608, noBias: true, name: "w2")
  let out = w2(w1(x).swish() .* w3(x))
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).feed_forward.w1.weight"] = [w1.weight.name]
    mapping["\(prefix).feed_forward.w2.weight"] = [w2.weight.name]
    mapping["\(prefix).feed_forward.w3.weight"] = [w3.weight.name]
    return mapping
  }
  return (mapper, Model([x], [out]))
}

private func Ideogram4Block(
  prefix: String, tokenLength: Int, usesFlashAttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let rot = Input()
  let adaln = Input()
  let adaLNs = (0..<4).map { Dense(count: 4_608, name: "ada_ln_\($0)") }
  let scaleMSA = 1 + adaLNs[0](adaln)
  let gateMSA = adaLNs[1](adaln).tanh()
  let scaleMLP = 1 + adaLNs[2](adaln)
  let gateMLP = adaLNs[3](adaln).tanh()

  let attentionNorm1 = RMSNorm(epsilon: 1e-5, axis: [1], name: "attention_norm1")
  let attnIn = (attentionNorm1(x) .* scaleMSA).to(FloatType.dataType)
  let (attentionMapper, attention) = Ideogram4Attention(
    prefix: prefix, tokenLength: tokenLength, usesFlashAttention: usesFlashAttention)
  let attentionNorm2 = RMSNorm(epsilon: 1e-5, axis: [1], name: "attention_norm2")
  var out = x + gateMSA.to(of: x) .* attentionNorm2(attention(attnIn, rot)).to(of: x)
  let ffnNorm1 = RMSNorm(epsilon: 1e-5, axis: [1], name: "ffn_norm1")
  let (mlpMapper, mlp) = Ideogram4MLP(prefix: prefix)
  let ffnNorm2 = RMSNorm(epsilon: 1e-5, axis: [1], name: "ffn_norm2")
  out =
    out
    + gateMLP.to(of: out)
    .* ffnNorm2(mlp((ffnNorm1(out) .* scaleMLP).to(FloatType.dataType))).to(of: out)
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping.merge(attentionMapper(format)) { v, _ in v }
    mapping.merge(mlpMapper(format)) { v, _ in v }
    mapping["\(prefix).adaln_modulation.weight"] = ModelWeightElement(
      adaLNs.map { $0.weight.name }, offsets: [0, 4_608, 9_216, 13_824])
    mapping["\(prefix).adaln_modulation.bias"] = ModelWeightElement(
      adaLNs.map { $0.bias.name }, offsets: [0, 4_608, 9_216, 13_824])
    mapping["\(prefix).attention_norm1.weight"] = [attentionNorm1.weight.name]
    mapping["\(prefix).attention_norm2.weight"] = [attentionNorm2.weight.name]
    mapping["\(prefix).ffn_norm1.weight"] = [ffnNorm1.weight.name]
    mapping["\(prefix).ffn_norm2.weight"] = [ffnNorm2.weight.name]
    return mapping
  }
  return (mapper, Model([x, rot, adaln], [out]))
}

public func Ideogram4(
  batchSize: Int, height: Int, width: Int, textLength: Int,
  usesFlashAttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  precondition(batchSize == 1)
  precondition(height % 2 == 0 && width % 2 == 0)
  let h = height / 2
  let w = width / 2
  let imageLength = h * w
  let tokenLength = textLength + imageLength
  let x = Input()
  let textFeatures = Input()
  let indicatorEmbedding = Input()
  let rot = Input()
  let tEmbed = Input()
  let xImage = x.reshaped([1, h, 2, w, 2, 32]).permuted(0, 1, 3, 2, 4, 5).contiguous()
    .reshaped([imageLength, 128])
  let inputProj = Dense(count: 4_608, name: "input_proj")
  let imageOut = inputProj(xImage)
  let llmNorm = RMSNorm(epsilon: 1e-6, axis: [1], name: "llm_cond_norm")
  let llmProj = Dense(count: 4_608, name: "llm_cond_proj")
  let textOut = llmProj(llmNorm(textFeatures).to(FloatType.dataType))
  var out = Functional.concat(axis: 0, textOut, imageOut)
  out = (out + indicatorEmbedding).to(FloatType.dataType)

  let tMlpIn = Dense(count: 4_608, name: "t_embedding_mlp_in")
  let tMlpOut = Dense(count: 4_608, name: "t_embedding_mlp_out")
  let adalnProj = Dense(count: 512, name: "adaln_proj")
  let adaln = adalnProj(tMlpOut(tMlpIn(tEmbed).swish())).swish()

  var blockMappers = [ModelWeightMapper]()
  for i in 0..<34 {
    let (mapper, block) = Ideogram4Block(
      prefix: "layers.\(i)", tokenLength: tokenLength, usesFlashAttention: usesFlashAttention)
    out = block(out, rot, adaln)
    blockMappers.append(mapper)
  }

  let norm = LayerNorm(epsilon: 1e-6, axis: [1], elementwiseAffine: false)
  let finalAdaln = Dense(count: 4_608, name: "final_ada_ln")
  let finalLinear = Dense(count: 128, name: "final_linear")
  out = finalLinear((norm(out) .* (1 + finalAdaln(adaln.swish()))).to(FloatType.dataType)).to(
    .Float32)
  out = out.reshaped([imageLength, 128], offset: [textLength, 0], strides: [128, 1]).copied()
  out = out.reshaped([1, h, w, 2, 2, 32]).permuted(0, 1, 3, 2, 4, 5).contiguous().reshaped([
    1, height, width, 32,
  ]).to(FloatType.dataType)

  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for blockMapper in blockMappers {
      mapping.merge(blockMapper(format)) { v, _ in v }
    }
    mapping["input_proj.weight"] = [inputProj.weight.name]
    mapping["input_proj.bias"] = [inputProj.bias.name]
    mapping["llm_cond_norm.weight"] = [llmNorm.weight.name]
    mapping["llm_cond_proj.weight"] = [llmProj.weight.name]
    mapping["llm_cond_proj.bias"] = [llmProj.bias.name]
    mapping["t_embedding.mlp_in.weight"] = [tMlpIn.weight.name]
    mapping["t_embedding.mlp_in.bias"] = [tMlpIn.bias.name]
    mapping["t_embedding.mlp_out.weight"] = [tMlpOut.weight.name]
    mapping["t_embedding.mlp_out.bias"] = [tMlpOut.bias.name]
    mapping["adaln_proj.weight"] = [adalnProj.weight.name]
    mapping["adaln_proj.bias"] = [adalnProj.bias.name]
    mapping["final_layer.adaln_modulation.weight"] = [finalAdaln.weight.name]
    mapping["final_layer.adaln_modulation.bias"] = [finalAdaln.bias.name]
    mapping["final_layer.linear.weight"] = [finalLinear.weight.name]
    mapping["final_layer.linear.bias"] = [finalLinear.bias.name]
    return mapping
  }
  return (mapper, Model([x, textFeatures, indicatorEmbedding, rot, tEmbed], [out]))
}
