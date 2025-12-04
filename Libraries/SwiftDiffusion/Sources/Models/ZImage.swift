import Foundation
import NNC

public func ZImageRotaryPositionEmbedding(
  height: Int, width: Int, tokenLength: Int, heads: Int = 1
)
  -> (Tensor<Float>, Tensor<Float>)
{
  var xRotTensor = Tensor<Float>(.CPU, .NHWC(1, height * width, heads, 128))
  var txtRotTensor = Tensor<Float>(.CPU, .NHWC(1, tokenLength, heads, 128))
  let dim0 = 32
  let dim1 = 48
  let dim2 = 48
  for i in 0..<tokenLength {
    for j in 0..<heads {
      for k in 0..<(dim0 / 2) {
        let theta = Double(i + 1) * 1.0 / pow(256, Double(k) * 2 / Double(dim0))
        let sintheta = sin(theta)
        let costheta = cos(theta)
        txtRotTensor[0, i, j, k * 2] = Float(costheta)
        txtRotTensor[0, i, j, k * 2 + 1] = Float(sintheta)
      }
      for k in 0..<(dim1 / 2) {
        let theta = Double(0) * 1.0 / pow(256, Double(k) * 2 / Double(dim1))
        let sintheta = sin(theta)
        let costheta = cos(theta)
        txtRotTensor[0, i, j, (k + (dim0 / 2)) * 2] = Float(costheta)
        txtRotTensor[0, i, j, (k + (dim0 / 2)) * 2 + 1] = Float(sintheta)
      }
      for k in 0..<(dim2 / 2) {
        let theta = Double(0) * 1.0 / pow(256, Double(k) * 2 / Double(dim2))
        let sintheta = sin(theta)
        let costheta = cos(theta)
        txtRotTensor[0, i, j, (k + (dim0 / 2) + (dim1 / 2)) * 2] = Float(costheta)
        txtRotTensor[0, i, j, (k + (dim0 / 2) + (dim1 / 2)) * 2 + 1] = Float(sintheta)
      }
    }
  }
  for y in 0..<height {
    for x in 0..<width {
      let i = y * width + x
      for j in 0..<heads {
        for k in 0..<(dim0 / 2) {
          let theta = Double(tokenLength + 1) * 1.0 / pow(256, Double(k) * 2 / Double(dim0))
          let sintheta = sin(theta)
          let costheta = cos(theta)
          xRotTensor[0, i, j, k * 2] = Float(costheta)
          xRotTensor[0, i, j, k * 2 + 1] = Float(sintheta)
        }
        for k in 0..<(dim1 / 2) {
          let theta =
            Double(y) * 1.0 / pow(256, Double(k) * 2 / Double(dim1))
          let sintheta = sin(theta)
          let costheta = cos(theta)
          xRotTensor[0, i, j, (k + (dim0 / 2)) * 2] = Float(costheta)
          xRotTensor[0, i, j, (k + (dim0 / 2)) * 2 + 1] = Float(sintheta)
        }
        for k in 0..<(dim2 / 2) {
          let theta =
            Double(x) * 1.0 / pow(256, Double(k) * 2 / Double(dim2))
          let sintheta = sin(theta)
          let costheta = cos(theta)
          xRotTensor[0, i, j, (k + (dim0 / 2) + (dim1 / 2)) * 2] = Float(costheta)
          xRotTensor[0, i, j, (k + (dim0 / 2) + (dim1 / 2)) * 2 + 1] = Float(sintheta)
        }
      }
    }
  }
  return (xRotTensor, txtRotTensor)
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
  prefix: String, name: String, k: Int, h: Int, b: Int, t: (Int, Int), scaleFactor: (Float, Float),
  modulation: Bool
) -> (Model, ModelWeightMapper) {
  let x = Input()
  let tEmbed: Input?
  let chunks: [Model.IO]
  let adaLNs: [Model]
  if modulation {
    let t = Input()
    adaLNs = (0..<4).map {
      Dense(count: k * h, name: name.isEmpty ? "ada_ln_\($0)" : "\(name)_ada_ln_\($0)")
    }
    chunks = adaLNs.map { $0(t) }
    tEmbed = t
  } else {
    tEmbed = nil
    adaLNs = []
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
    out = (chunks[0] + 1).to(of: out) .* out
  }
  out = out.to(.Float16)
  var keys = tokeys(out).reshaped([b, t.0, h, k])
  var queries = toqueries(out).reshaped([b, t.0, h, k])
  let normK = RMSNorm(epsilon: 1e-5, axis: [3], name: name.isEmpty ? "norm_k" : "\(name)_norm_k")
  keys = normK(keys)
  let normQ = RMSNorm(epsilon: 1e-5, axis: [3], name: name.isEmpty ? "norm_q" : "\(name)_norm_q")
  queries = normQ(queries)
  if scaleFactor.0 > 1 {
    out = (1 / scaleFactor.0) * out
  }
  var values = tovalues(out)
  values = values.reshaped([b, t.0, h, k])
  queries = (1.0 / Float(k).squareRoot().squareRoot()) * Functional.cmul(left: queries, right: rot)
  keys = (1.0 / Float(k).squareRoot().squareRoot()) * Functional.cmul(left: keys, right: rot)
  out = ScaledDotProductAttention(scale: 1)(
    queries, keys, values
  )
  let xIn: Model.IO
  if t.0 > t.1 {
    xIn = x.reshaped([b, t.1, h * k], offset: [0, 0, 0], strides: [t.0 * h * k, h * k, 1])
    out = out.reshaped([b, t.1, h * k], offset: [0, 0, 0], strides: [t.0 * h * k, h * k, 1])
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
    out = chunks[1].tanh().to(of: out) .* out
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
    out = (chunks[2] + 1).to(of: out) .* out
  }
  out = out.to(.Float16)
  out = feedForwardNorm2(ffn(out))  // Already converted to Float32.
  if modulation {
    out = chunks[3].tanh().to(of: out) .* out
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
    if modulation {
      mapping["\(prefix).adaLN_modulation.0.weight"] = ModelWeightElement(
        (0..<4).map { adaLNs[$0].weight.name })
      mapping["\(prefix).adaLN_modulation.0.bias"] = ModelWeightElement(
        (0..<4).map { adaLNs[$0].bias.name })
    }
    return mapping
  }
  return (Model([x, rot] + (tEmbed.map { [$0] } ?? []), [out]), mapper)
}

func ZImage(
  batchSize: Int, height: Int, width: Int, textLength: Int, channels: Int, layers: Int,
  usesFlashAttention: FlashAttentionLevel
) -> (
  Model, ModelWeightMapper
) {
  let x = Input()
  let xRot = Input()
  let txt = Input()
  let txtRot = Input()
  let t = Input()
  let imgIn = Dense(count: channels, name: "x_embedder")
  let h = height / 2
  let w = width / 2
  var xOut = imgIn(
    x.reshaped([batchSize, h, 2, w, 2, 16]).permuted(0, 1, 3, 2, 4, 5).contiguous()
      .reshaped([batchSize, h * w, 2 * 2 * 16], format: .NHWC)
  ).to(.Float32)
  let txtNorm = RMSNorm(epsilon: 1e-5, axis: [2], name: "cap_norm")
  let txtIn = Dense(count: channels, name: "cap_embedder")
  var txtOut = txtIn(txtNorm(txt)).to(.Float32)
  let (timeInMlp0, timeInMlp2, timeIn) = MLPEmbedder(
    channels: 256, intermediateSize: 1024, name: "t")
  let tOut = timeIn(t)
  var mappers = [ModelWeightMapper]()
  for i in 0..<2 {
    let (block, mapper) = ZImageTransformerBlock(
      prefix: "context_refiner.\(i)", name: "context_refiner", k: 128, h: channels / 128,
      b: batchSize,
      t: (textLength, textLength), scaleFactor: (2, 2), modulation: false)
    txtOut = block(txtOut, txtRot)
    mappers.append(mapper)
  }
  for i in 0..<2 {
    let (block, mapper) = ZImageTransformerBlock(
      prefix: "noise_refiner.\(i)", name: "noise_refiner", k: 128, h: channels / 128, b: 1,
      t: (h * w, h * w),
      scaleFactor: (4, 32), modulation: true)
    xOut = block(xOut, xRot, tOut)
    mappers.append(mapper)
  }
  var out = Functional.concat(axis: 1, xOut, txtOut)
  let rot = Functional.concat(axis: 1, xRot, txtRot)
  for i in 0..<layers {
    let (block, mapper) = ZImageTransformerBlock(
      prefix: "layers.\(i)", name: "", k: 128, h: channels / 128, b: 1,
      t: (h * w + textLength, i == layers - 1 ? h * w : h * w + textLength), scaleFactor: (4, 32),
      modulation: true)
    out = block(out, rot, tOut)
    mappers.append(mapper)
  }
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let scale = Dense(count: channels, name: "ada_ln_final")
  let projOut = Dense(count: 2 * 2 * 16, name: "linear_final")
  out = (1 + scale(tOut.swish())) .* normFinal(out).to(.Float16)
  out = (-projOut(out)).reshaped([batchSize, h, w, 2, 2, 16]).permuted(0, 1, 3, 2, 4, 5)
    .contiguous()
    .reshaped([
      batchSize, h * 2, w * 2, 16,
    ])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["all_x_embedder.2-1.weight"] = [imgIn.weight.name]
    mapping["all_x_embedder.2-1.bias"] = [imgIn.bias.name]
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
    mapping["all_final_layer.2-1.linear.weight"] = [projOut.weight.name]
    mapping["all_final_layer.2-1.linear.bias"] = [projOut.bias.name]
    return mapping
  }
  return (Model([x, xRot, txt, txtRot, t], [out, txtOut]), mapper)
}
