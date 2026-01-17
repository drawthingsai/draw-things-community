import Foundation
import NNC

public func Qwen3RotaryEmbedding<FloatType: TensorNumeric & BinaryFloatingPoint>(
  sequenceLength: Int, of dataType: FloatType.Type = FloatType.self
) -> Tensor<FloatType> {
  var rotary = Tensor<FloatType>(.CPU, .NHWC(1, sequenceLength, 1, 128))
  for i in 0..<sequenceLength {
    for k in 0..<64 {
      let theta = Double(i) * 1.0 / pow(1_000_000, Double(k) * 2 / 128)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotary[0, i, 0, k * 2] = FloatType(costheta)
      rotary[0, i, 0, k * 2 + 1] = FloatType(sintheta)
    }
  }
  return rotary
}

private func SelfAttention(
  prefix: String, width: Int, k: Int, h: Int, hk: Int, b: Int, t: Int, usesFlashAttention: Bool
) -> Model {
  let x = Input()
  let rot = Input()
  let causalAttentionMask = Input()
  let tokeys = Dense(count: k * hk, noBias: true, name: "k_proj")
  let toqueries = Dense(count: k * h, noBias: true, name: "q_proj")
  let tovalues = Dense(count: k * hk, noBias: true, name: "v_proj")
  var keys = tokeys(x).reshaped([b, t, hk, k])
  var queries = toqueries(x).reshaped([b, t, h, k])
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "norm_k")
  keys = normK(keys)
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "norm_q")
  queries = normQ(queries)
  var values = tovalues(x).reshaped([b, t, hk, k])
  queries = (1.0 / Float(k).squareRoot().squareRoot()) * Functional.cmul(left: queries, right: rot)
  keys = (1.0 / Float(k).squareRoot().squareRoot()) * Functional.cmul(left: keys, right: rot)
  var out: Model.IO
  if usesFlashAttention {
    out = ScaledDotProductAttention(scale: 1, isCausal: true, hasAttentionMask: true)(
      queries, keys, values, causalAttentionMask
    ).reshaped([b * t, h * k])
  } else {
    values = values.transposed(1, 2)
    queries = queries.transposed(1, 2)
    keys = keys.transposed(1, 2)
    var outs = [Model.IO]()
    for i in 0..<hk {
      let query = queries.reshaped(
        [b, h / hk, t, k], offset: [0, i * (h / hk), 0, 0], strides: [h * t * k, t * k, k, 1]
      ).contiguous()
      let key = keys.reshaped(
        [b, 1, t, k], offset: [0, i, 0, 0], strides: [hk * t * k, t * k, k, 1]
      ).contiguous()
      let value = values.reshaped(
        [b, 1, t, k], offset: [0, i, 0, 0], strides: [hk * t * k, t * k, k, 1]
      ).contiguous()
      var dot = Matmul(transposeB: (2, 3))(query, key)
      if let last = outs.last {
        dot.add(dependencies: [last])
      }
      dot = dot + causalAttentionMask
      dot = dot.reshaped([b * (h / hk) * t, t])
      dot = dot.softmax()
      dot = dot.reshaped([b, h / hk, t, t])
      let out = dot * value
      outs.append(out)
    }
    out = Concat(axis: 1)(outs).reshaped([b, h, t, k]).transposed(1, 2).reshaped([b * t, h * k])
  }
  let unifyheads = Dense(count: width, noBias: true, name: "out_proj")
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

private func TransformerBlock(
  prefix: String, width: Int, k: Int, h: Int, hk: Int, b: Int, t: Int, MLP: Int,
  usesFlashAttention: Bool
)
  -> Model
{
  let x = Input()
  let rot = Input()
  let causalAttentionMask = Input()
  let norm1 = RMSNorm(epsilon: 1e-6, axis: [1], name: "input_layernorm")
  var out = norm1(x)
  let attention = SelfAttention(
    prefix: prefix, width: width, k: k, h: h, hk: hk, b: b, t: t,
    usesFlashAttention: usesFlashAttention)
  out = attention(out, rot, causalAttentionMask) + x
  let residual = out
  let norm2 = RMSNorm(epsilon: 1e-6, axis: [1], name: "post_attention_layernorm")
  out = norm2(out)
  let (_, _, _, ffn) = FeedForward(hiddenSize: width, intermediateSize: MLP, name: "mlp")
  out = residual + ffn(out)
  return Model([x, rot, causalAttentionMask], [out])
}

private func TextEmbedding<T: TensorNumeric & BinaryFloatingPoint>(
  _ dataType: T.Type, batchSize: Int, vocabularySize: Int, maxLength: Int, embeddingSize: Int
) -> Model {
  let tokens = Input()
  let tokenEmbed = Embedding(
    T.self, vocabularySize: vocabularySize, embeddingSize: embeddingSize, name: "tok_embeddings")
  let embedding = tokenEmbed(tokens)
  return Model([tokens], [embedding])
}

public func Qwen3<T: TensorNumeric & BinaryFloatingPoint>(
  _ dataType: T.Type, vocabularySize: Int, maxLength: Int, width: Int, tokenLength: Int,
  layers: Int, MLP: Int, heads: Int, outputHiddenStates: [Int], batchSize: Int,
  usesFlashAttention: Bool
) -> Model {
  let tokens = Input()
  let rot = Input()
  let causalAttentionMask = Input()
  let embedding = TextEmbedding(
    T.self, batchSize: batchSize, vocabularySize: vocabularySize, maxLength: maxLength,
    embeddingSize: width)
  var out = embedding(tokens)
  var hiddenStates = [Model.IO]()
  for i in 0..<layers {
    let layer = TransformerBlock(
      prefix: "layers.\(i)", width: width, k: 128, h: heads, hk: 8, b: batchSize,
      t: tokenLength, MLP: MLP, usesFlashAttention: usesFlashAttention)
    out = layer(out, rot, causalAttentionMask)
    if outputHiddenStates.contains(i) {
      hiddenStates.append(out)
    }
  }
  let norm = RMSNorm(epsilon: 1e-6, axis: [1], name: "norm")
  out = norm(out)
  return Model([tokens, rot, causalAttentionMask], hiddenStates + [out])
}
