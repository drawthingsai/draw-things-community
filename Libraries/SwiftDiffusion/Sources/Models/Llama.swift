import Foundation
import NNC

func Llama3RotaryEmbedding<FloatType: TensorNumeric & BinaryFloatingPoint>(
  sequenceLength: Int, of dataType: FloatType.Type = FloatType.self
) -> Tensor<FloatType> {
  var rotary = Tensor<FloatType>(.CPU, .NHWC(1, sequenceLength, 1, 128))
  for i in 0..<sequenceLength {
    for k in 0..<64 {
      let theta = Double(i) * 1.0 / pow(500_000, Double(k) * 2 / 128)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotary[0, i, 0, k * 2] = FloatType(costheta)
      rotary[0, i, 0, k * 2 + 1] = FloatType(sintheta)
    }
  }
  return rotary
}

private func SelfAttention(prefix: String, k: Int, h: Int, hk: Int, b: Int, t: Int) -> Model {
  let x = Input()
  let rot = Input()
  let causalAttentionMask = Input()
  let tokeys = Dense(count: k * hk, noBias: true, name: "k_proj")
  let toqueries = Dense(count: k * h, noBias: true, name: "q_proj")
  let tovalues = Dense(count: k * hk, noBias: true, name: "v_proj")
  var keys = tokeys(x).reshaped([b, t, hk, k])
  var queries = toqueries(x).reshaped([b, t, h, k])
  let values = tovalues(x).reshaped([b, t, hk, k])
  queries = Functional.cmul(left: queries, right: rot)
  keys = Functional.cmul(left: keys, right: rot)
  var out = ScaledDotProductAttention(
    scale: 1.0 / Float(k).squareRoot(), isCausal: true, hasAttentionMask: true)(
      queries, keys, values, causalAttentionMask
    ).reshaped([b * t, h * k])
  let unifyheads = Dense(count: k * h, noBias: true, name: "out_proj")
  out = unifyheads(out)
  return Model([x, rot, causalAttentionMask], [out])
}

private func FeedForward(hiddenSize: Int, intermediateSize: Int, name: String = "") -> (
  Model, Model, Model, Model
) {
  let x = Input()
  let w1 = Dense(count: intermediateSize, noBias: true, name: "\(name)_gate_proj")
  let w3 = Dense(count: intermediateSize, noBias: true, name: "\(name)_up_proj")
  var out = w3(x) .* w1(x).swish()
  let w2 = Dense(count: hiddenSize, noBias: true, name: "\(name)_down_proj")
  out = w2(out)
  return (w1, w2, w3, Model([x], [out], name: name))
}

private func TransformerBlock(prefix: String, k: Int, h: Int, hk: Int, b: Int, t: Int, MLP: Int)
  -> Model
{
  let x = Input()
  let rot = Input()
  let causalAttentionMask = Input()
  let norm1 = RMSNorm(epsilon: 1e-5, axis: [1], name: "input_layernorm")
  var out = norm1(x)
  let attention = SelfAttention(prefix: prefix, k: k, h: h, hk: hk, b: b, t: t)
  out = attention(out, rot, causalAttentionMask) + x
  let residual = out
  let norm2 = RMSNorm(epsilon: 1e-5, axis: [1], name: "post_attention_layernorm")
  out = norm2(out)
  let (_, _, _, ffn) = FeedForward(hiddenSize: h * k, intermediateSize: MLP, name: "mlp")
  out = residual + ffn(out)
  return Model([x, rot, causalAttentionMask], [out])
}

private func TextEmbedding<T: TensorNumeric>(
  _ dataType: T.Type, batchSize: Int, vocabularySize: Int, embeddingSize: Int
) -> Model {
  let tokens = Input()
  let tokenEmbed = Embedding(
    T.self, vocabularySize: vocabularySize, embeddingSize: embeddingSize, name: "tok_embeddings")
  let embedding = tokenEmbed(tokens)
  return Model([tokens], [embedding])
}

func Llama3<T: TensorNumeric>(
  _ dataType: T.Type, vocabularySize: Int, width: Int, tokenLength: (Int, Int),
  layers: Int, MLP: Int, heads: Int, outputHiddenStates: Int?, batchSize: Int
) -> Model {
  let tokens = Input()
  let rot = Input()
  let causalAttentionMask = Input()
  let embedding = TextEmbedding(
    T.self, batchSize: batchSize, vocabularySize: vocabularySize, embeddingSize: width)
  var out = embedding(tokens)
  let textEmbedding: Input?
  if tokenLength.1 > tokenLength.0 {
    let additionalEmbedding = Input()
    out = Functional.concat(
      axis: 1, out.reshaped([batchSize, tokenLength.0, width]), additionalEmbedding
    ).reshaped([batchSize * tokenLength.1, width])
    textEmbedding = additionalEmbedding
  } else {
    textEmbedding = nil
  }
  var hiddenStates: Model.IO? = nil
  for i in 0..<layers {
    let layer = TransformerBlock(
      prefix: "layers.\(i)", k: width / heads, h: heads, hk: heads / 4, b: batchSize,
      t: tokenLength.1, MLP: MLP)
    out = layer(out, rot, causalAttentionMask)
    if let outputHiddenStates = outputHiddenStates, outputHiddenStates == i {
      hiddenStates = out
    }
  }
  let norm = RMSNorm(epsilon: 1e-5, axis: [1], name: "norm")
  out = norm(out)
  return Model(
    [tokens, rot, causalAttentionMask] + (textEmbedding.flatMap { [$0] } ?? []),
    (hiddenStates.map { [$0] } ?? []) + [out])
}
