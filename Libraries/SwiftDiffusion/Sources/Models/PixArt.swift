import Foundation
import NNC

func sinCos2DPositionEmbedding(height: Int, width: Int, embeddingSize: Int) -> Tensor<Float> {
  precondition(embeddingSize % 4 == 0)
  var embedding = Tensor<Float>(.CPU, .HWC(height, width, embeddingSize))
  let halfOfHalf = embeddingSize / 4
  let omega: [Double] = (0..<halfOfHalf).map {
    pow(Double(1.0 / 10000), Double($0) / Double(halfOfHalf))
  }
  for i in 0..<height {
    let y = Double(i) / 2
    for j in 0..<width {
      let x = Double(j) / 2
      for k in 0..<halfOfHalf {
        let xFreq = x * omega[k]
        embedding[i, j, k] = Float(sin(xFreq))
        embedding[i, j, k + halfOfHalf] = Float(cos(xFreq))
        let yFreq = y * omega[k]
        embedding[i, j, k + 2 * halfOfHalf] = Float(sin(yFreq))
        embedding[i, j, k + 3 * halfOfHalf] = Float(cos(yFreq))
      }
    }
  }
  return embedding
}

private func TimeEmbedder(channels: Int) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: channels, name: "t_embedder_0")
  var out = fc0(x).swish()
  let fc2 = Dense(count: channels, name: "t_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

private func MLP(hiddenSize: Int, intermediateSize: Int, name: String) -> (Model, Model, Model) {
  let x = Input()
  let fc1 = Dense(count: intermediateSize, name: "\(name)_fc1")
  var out = GELU(approximate: .tanh)(fc1(x))
  let fc2 = Dense(count: hiddenSize, name: "\(name)_fc2")
  out = fc2(out)
  return (fc1, fc2, Model([x], [out]))
}

private func SelfAttention(k: Int, h: Int, b: Int, t: Int) -> (
  Model, Model, Model, Model, Model
) {
  let x = Input()
  let tokeys = Dense(count: k * h, name: "k")
  let toqueries = Dense(count: k * h, name: "q")
  let tovalues = Dense(count: k * h, name: "v")
  let keys = tokeys(x).reshaped([b, t, h, k]).transposed(1, 2)
  // No scaling the queries.
  let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, t, h, k])
    .transposed(1, 2)
  let values = tovalues(x).reshaped([b, t, h, k]).transposed(1, 2)
  var dot = Matmul(transposeB: (2, 3))(queries, keys)
  dot = dot.reshaped([b * h * t, t])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, t, t])
  var out = dot * values
  out = out.reshaped([b, h, t, k]).transposed(1, 2).reshaped([b, t, h * k])
  let unifyheads = Dense(count: k * h, name: "o")
  out = unifyheads(out)
  return (tokeys, toqueries, tovalues, unifyheads, Model([x], [out]))
}

private func CrossAttention(k: Int, h: Int, b: Int, hw: Int, t: Int) -> (
  Model, Model, Model, Model, Model
) {
  let x = Input()
  let context = Input()
  let tokeys = Dense(count: k * h, name: "c_k")
  let toqueries = Dense(count: k * h, name: "c_q")
  let tovalues = Dense(count: k * h, name: "c_v")
  let keys = tokeys(context).reshaped([b, t, h, k]).transposed(1, 2)
  let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, hw, h, k])
    .transposed(1, 2)
  let values = tovalues(context).reshaped([b, t, h, k]).transposed(1, 2)
  var dot = Matmul(transposeB: (2, 3))(queries, keys)
  dot = dot.reshaped([b * h * hw, t])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, hw, t])
  var out = dot * values
  out = out.reshaped([b, h, hw, k]).transposed(1, 2).reshaped([b, hw, h * k])
  let unifyheads = Dense(count: k * h, name: "c_o")
  out = unifyheads(out)
  return (tokeys, toqueries, tovalues, unifyheads, Model([x, context], [out]))
}

func PixArtMSBlock(prefix: String, k: Int, h: Int, b: Int, hw: Int, t: Int) -> (
  ModelWeightMapper, Model
) {
  let x = Input()
  let context = Input()
  let shiftMsa = Input()
  let scaleMsa = Input()
  let gateMsa = Input()
  let shiftMlp = Input()
  let scaleMlp = Input()
  let gateMlp = Input()
  let norm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let (tokeys1, toqueries1, tovalues1, unifyheads1, attn) = SelfAttention(k: k, h: h, b: b, t: hw)
  let shiftMsaShift = Parameter<Float>(.GPU(0), .CHW(1, 1, k * h), name: "scale_shift_table_0")
  let scaleMsaShift = Parameter<Float>(.GPU(0), .CHW(1, 1, k * h), name: "scale_shift_table_1")
  let gateMsaShift = Parameter<Float>(.GPU(0), .HWC(1, 1, k * h), name: "scale_shift_table_2")
  var out =
    x + (gateMsa + gateMsaShift)
    .* attn(norm1(x) .* (scaleMsa + scaleMsaShift) + (shiftMsa + shiftMsaShift))
  let (tokeys2, toqueries2, tovalues2, unifyheads2, crossAttn) = CrossAttention(
    k: k, h: h, b: b, hw: hw, t: t)
  out = out + crossAttn(out, context)
  let (fc1, fc2, mlp) = MLP(hiddenSize: k * h, intermediateSize: k * h * 4, name: "mlp")
  let norm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let shiftMlpShift = Parameter<Float>(.GPU(0), .CHW(1, 1, k * h), name: "scale_shift_table_3")
  let scaleMlpShift = Parameter<Float>(.GPU(0), .CHW(1, 1, k * h), name: "scale_shift_table_4")
  let gateMlpShift = Parameter<Float>(.GPU(0), .CHW(1, 1, k * h), name: "scale_shift_table_5")
  out = out + (gateMlp + gateMlpShift)
    .* mlp(norm2(out) .* (scaleMlp + scaleMlpShift) + (shiftMlp + shiftMlpShift))
  let mapper: ModelWeightMapper = { _ in
    var mapping = [String: [String]]()
    mapping["\(prefix).attn.qkv.weight"] = [
      toqueries1.weight.name, tokeys1.weight.name, tovalues1.weight.name,
    ]
    mapping["\(prefix).attn.qkv.bias"] = [
      toqueries1.bias.name, tokeys1.bias.name, tovalues1.bias.name,
    ]
    mapping["\(prefix).attn.proj.weight"] = [unifyheads1.weight.name]
    mapping["\(prefix).attn.proj.bias"] = [unifyheads1.bias.name]
    mapping["\(prefix).scale_shift_table"] = [
      shiftMsaShift.weight.name, scaleMsaShift.weight.name, gateMsaShift.weight.name,
      shiftMlpShift.weight.name, scaleMlpShift.weight.name, gateMlpShift.weight.name,
    ]
    mapping["\(prefix).cross_attn.q_linear.weight"] = [toqueries2.weight.name]
    mapping["\(prefix).cross_attn.q_linear.bias"] = [toqueries2.bias.name]
    mapping["\(prefix).cross_attn.kv_linear.weight"] = [tokeys2.weight.name, tovalues2.weight.name]
    mapping["\(prefix).cross_attn.kv_linear.bias"] = [tokeys2.bias.name, tovalues2.bias.name]
    mapping["\(prefix).cross_attn.proj.weight"] = [unifyheads2.weight.name]
    mapping["\(prefix).cross_attn.proj.bias"] = [unifyheads2.bias.name]
    mapping["\(prefix).mlp.fc1.weight"] = [fc1.weight.name]
    mapping["\(prefix).mlp.fc1.bias"] = [fc1.bias.name]
    mapping["\(prefix).mlp.fc2.weight"] = [fc2.weight.name]
    mapping["\(prefix).mlp.fc2.bias"] = [fc2.bias.name]
    return mapping
  }
  return (
    mapper, Model([x, context, shiftMsa, scaleMsa, gateMsa, shiftMlp, scaleMlp, gateMlp], [out])
  )
}

func PixArt(b: Int, h: Int, w: Int) -> (ModelWeightMapper, Model) {
  let x = Input()
  let posEmbed = Input()
  let t = Input()
  let y = Input()
  let xEmbedder = Convolution(
    groups: 1, filters: 1152, filterSize: [2, 2],
    hint: Hint(stride: [2, 2]), name: "x_embedder")
  var out = xEmbedder(x).reshaped([b, 1152, h * w]).transposed(1, 2) + posEmbed
  let (tMlp0, tMlp2, tEmbedder) = TimeEmbedder(channels: 1152)
  let t0 = tEmbedder(t)
  let t1 = t0.swish().reshaped([b, 1, 1152])
  let tBlock = (0..<6).map { Dense(count: 1152, name: "t_block_\($0)") }
  var adaln = tBlock.map { $0(t1) }
  adaln[1] = 1 + adaln[1]
  adaln[4] = 1 + adaln[4]
  let (fc1, fc2, yEmbedder) = MLP(hiddenSize: 1152, intermediateSize: 1152, name: "y_embedder")
  let y0 = yEmbedder(y)
  var mappers = [ModelWeightMapper]()
  for i in 0..<28 {
    let (mapper, block) = PixArtMSBlock(
      prefix: "blocks.\(i)", k: 72, h: 16, b: 2, hw: h * w, t: 300)
    out = block(out, y0, adaln[0], adaln[1], adaln[2], adaln[3], adaln[4], adaln[5])
    mappers.append(mapper)
  }
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let shiftShift = Parameter<Float>(.GPU(0), .CHW(1, 1, 1152), name: "final_scale_shift_table_0")
  let scaleShift = Parameter<Float>(.GPU(0), .CHW(1, 1, 1152), name: "final_scale_shift_table_1")
  let tt = t0.reshaped([1, 1, 1152])  // PixArt uses chunk, but that always assumes t0 is the same, which is true.
  out = (scaleShift + 1 + tt) .* normFinal(out) + (shiftShift + tt)
  let linear = Dense(count: 2 * 2 * 8, name: "linear")
  out = linear(out)
  // Unpatchify
  out = out.reshaped([b, h, w, 2, 2, 8]).permuted(0, 5, 1, 3, 2, 4).contiguous().reshaped([
    b, 8, h * 2, w * 2,
  ])
  let mapper: ModelWeightMapper = { format in
    var mapping = [String: [String]]()
    mapping["x_embedder.proj.weight"] = [xEmbedder.weight.name]
    mapping["x_embedder.proj.bias"] = [xEmbedder.bias.name]
    mapping["t_embedder.mlp.0.weight"] = [tMlp0.weight.name]
    mapping["t_embedder.mlp.0.bias"] = [tMlp0.bias.name]
    mapping["t_embedder.mlp.2.weight"] = [tMlp2.weight.name]
    mapping["t_embedder.mlp.2.bias"] = [tMlp2.bias.name]
    mapping["t_block.1.weight"] = tBlock.map { $0.weight.name }
    mapping["t_block.1.bias"] = tBlock.map { $0.bias.name }
    mapping["y_embedder.y_proj.fc1.weight"] = [fc1.weight.name]
    mapping["y_embedder.y_proj.fc1.bias"] = [fc1.bias.name]
    mapping["y_embedder.y_proj.fc2.weight"] = [fc2.weight.name]
    mapping["y_embedder.y_proj.fc2.bias"] = [fc2.bias.name]
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping["final_layer.scale_shift_table"] = [shiftShift.weight.name, scaleShift.weight.name]
    mapping["final_layer.linear.weight"] = [linear.weight.name]
    mapping["final_layer.linear.bias"] = [linear.bias.name]
    return mapping
  }
  return (mapper, Model([x, posEmbed, t, y], [out]))
}
