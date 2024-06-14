import NNC

private func T5TextEmbedding(vocabularySize: Int, embeddingSize: Int, name: String) -> Model {
  let tokenEmbed = Embedding(
    FloatType.self, vocabularySize: vocabularySize, embeddingSize: embeddingSize, name: name)
  return tokenEmbed
}

private func T5LayerSelfAttention(k: Int, h: Int, b: Int, t: Int, outFeatures: Int) -> (
  Model, Model, Model, Model, Model
) {
  let x = Input()
  let positionBias = Input()
  let tokeys = Dense(count: k * h, noBias: true, name: "k")
  let toqueries = Dense(count: k * h, noBias: true, name: "q")
  let tovalues = Dense(count: k * h, noBias: true, name: "v")
  let keys = tokeys(x).reshaped([b, t, h, k]).permuted(0, 2, 1, 3)
  // No scaling the queries.
  let queries = toqueries(x).reshaped([b, t, h, k])
    .permuted(0, 2, 1, 3)
  let values = tovalues(x).reshaped([b, t, h, k]).permuted(0, 2, 1, 3)
  var dot = Matmul(transposeB: (2, 3))(queries, keys) + positionBias
  dot = dot.reshaped([b * h * t, t])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, t, t])
  var out = dot * values
  out = out.reshaped([b, h, t, k]).transposed(1, 2).reshaped([b * t, h * k])
  let unifyheads = Dense(count: outFeatures, noBias: true, name: "o")
  out = unifyheads(out)
  return (tokeys, toqueries, tovalues, unifyheads, Model([x, positionBias], [out]))
}

private func T5DenseGatedActDense(hiddenSize: Int, intermediateSize: Int) -> (
  Model, Model, Model, Model
) {
  let x = Input()
  let wi_0 = Dense(count: intermediateSize, noBias: true, name: "w0")
  let wi_1 = Dense(count: intermediateSize, noBias: true, name: "w1")
  var out = wi_1(x).to(.Float32) .* wi_0(x).GELU(approximate: .tanh).to(.Float32)
  let wo = Dense(count: hiddenSize, noBias: true, name: "wo")
  // Need to apply a scale factor if T5 has to work with Float16.
  let scaleFactor: Float = 8
  out = scaleFactor * wo(((1 / scaleFactor) * out).to(of: x)).to(.Float32)
  return (wi_0, wi_1, wo, Model([x], [out]))
}

private func T5Block(
  prefix: String, k: Int, h: Int, b: Int, t: Int, outFeatures: Int, intermediateSize: Int
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let positionBias = Input()
  let norm1 = RMSNorm(epsilon: 1e-6, axis: [1], name: "norm1")
  let (tokeys, toqueries, tovalues, unifyheads, attention) = T5LayerSelfAttention(
    k: k, h: h, b: b, t: t, outFeatures: outFeatures)
  var out = x + attention(norm1(x).to(FloatType.dataType), positionBias).to(of: x)
  let norm2 = RMSNorm(epsilon: 1e-6, axis: [1], name: "norm2")
  let (wi_0, wi_1, wo, ff) = T5DenseGatedActDense(
    hiddenSize: outFeatures, intermediateSize: intermediateSize)
  out = out + ff(norm2(out).to(FloatType.dataType))
  let mapper: ModelWeightMapper = { _ in
    var mapping = [String: [String]]()
    mapping["\(prefix).layer.0.layer_norm.weight"] = [norm1.weight.name]
    mapping["\(prefix).layer.0.SelfAttention.k.weight"] = [tokeys.weight.name]
    mapping["\(prefix).layer.0.SelfAttention.q.weight"] = [toqueries.weight.name]
    mapping["\(prefix).layer.0.SelfAttention.v.weight"] = [tovalues.weight.name]
    mapping["\(prefix).layer.0.SelfAttention.o.weight"] = [unifyheads.weight.name]
    mapping["\(prefix).layer.1.layer_norm.weight"] = [norm2.weight.name]
    mapping["\(prefix).layer.1.DenseReluDense.wi_0.weight"] = [wi_0.weight.name]
    mapping["\(prefix).layer.1.DenseReluDense.wi_1.weight"] = [wi_1.weight.name]
    mapping["\(prefix).layer.1.DenseReluDense.wo.weight"] = [wo.weight.name]
    return mapping
  }
  return (mapper, Model([x, positionBias], [out]))
}

func T5ForConditionalGeneration(b: Int, t: Int) -> (ModelWeightMapper, Model) {
  let x = Input()
  let relativePositionBuckets = Input()
  let textEmbed = T5TextEmbedding(vocabularySize: 32_128, embeddingSize: 4_096, name: "shared")
  var out = textEmbed(x).to(.Float32)
  let relativePositionEmbedding = Embedding(
    FloatType.self, vocabularySize: 32, embeddingSize: 64, name: "relative_position_embedding")
  let positionBias = relativePositionEmbedding(relativePositionBuckets).reshaped([1, t, t, 64])
    .permuted(0, 3, 1, 2).contiguous()
  var mappers = [ModelWeightMapper]()
  for i in 0..<24 {
    let (mapper, block) = T5Block(
      prefix: "encoder.block.\(i)", k: 64, h: 64, b: b, t: t, outFeatures: 4_096,
      intermediateSize: 10_240)
    out = block(out, positionBias)
    mappers.append(mapper)
  }
  let finalNorm = RMSNorm(epsilon: 1e-6, axis: [1], name: "final_norm")
  out = finalNorm(out).to(FloatType.dataType)
  let mapper: ModelWeightMapper = { format in
    var mapping = [String: [String]]()
    mapping["shared.weight"] = [textEmbed.weight.name]
    mapping[
      "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
    ] = [relativePositionEmbedding.weight.name]
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping["encoder.final_layer_norm.weight"] = [finalNorm.weight.name]
    return mapping
  }
  return (mapper, Model([x, relativePositionBuckets], [out]))
}
