import Foundation
import NNC

func Flux1RotaryPositionEmbedding(height: Int, width: Int, tokenLength: Int, channels: Int)
  -> Tensor<Float>
{
  var rotTensor = Tensor<Float>(.CPU, .NHWC(1, height * width + tokenLength, 1, channels))
  let dim0 = channels / 8
  let dim1 = channels * 7 / 16
  let dim2 = dim1
  assert(channels % 16 == 0)
  for i in 0..<tokenLength {
    for k in 0..<(dim0 / 2) {
      let theta = 0 * 1.0 / pow(10_000, Double(k) * 2 / Double(dim0))
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotTensor[0, i, 0, k * 2] = Float(costheta)
      rotTensor[0, i, 0, k * 2 + 1] = Float(sintheta)
    }
    for k in 0..<(dim1 / 2) {
      let theta = 0 * 1.0 / pow(10_000, Double(k) * 2 / Double(dim1))
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotTensor[0, i, 0, (k + (dim0 / 2)) * 2] = Float(costheta)
      rotTensor[0, i, 0, (k + (dim0 / 2)) * 2 + 1] = Float(sintheta)
    }
    for k in 0..<(dim2 / 2) {
      let theta = 0 * 1.0 / pow(10_000, Double(k) * 2 / 28)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotTensor[0, i, 0, (k + (dim0 / 2) + (dim1 / 2)) * 2] = Float(costheta)
      rotTensor[0, i, 0, (k + (dim0 / 2) + (dim1 / 2)) * 2 + 1] = Float(sintheta)
    }
  }
  for y in 0..<height {
    for x in 0..<width {
      let i = y * height + x + tokenLength
      for k in 0..<(dim0 / 2) {
        let theta = 0 * 1.0 / pow(10_000, Double(k) * 2 / Double(dim0))
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, i, 0, k * 2] = Float(costheta)
        rotTensor[0, i, 0, k * 2 + 1] = Float(sintheta)
      }
      for k in 0..<(dim1 / 2) {
        let theta = Double(y) * 1.0 / pow(10_000, Double(k) * 2 / Double(dim1))
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, i, 0, (k + (dim0 / 2)) * 2] = Float(costheta)
        rotTensor[0, i, 0, (k + (dim0 / 2)) * 2 + 1] = Float(sintheta)
      }
      for k in 0..<(dim2 / 2) {
        let theta = Double(x) * 1.0 / pow(10_000, Double(k) * 2 / Double(dim2))
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, i, 0, (k + (dim0 / 2) + (dim1 / 2)) * 2] = Float(costheta)
        rotTensor[0, i, 0, (k + (dim0 / 2) + (dim1 / 2)) * 2 + 1] = Float(sintheta)
      }
    }
  }
  return rotTensor
}

private func MLPEmbedder(channels: Int, name: String) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: channels, name: "\(name)_embedder_0")
  var out = fc0(x).swish()
  let fc2 = Dense(count: channels, name: "\(name)_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

private func FeedForward(hiddenSize: Int, intermediateSize: Int, name: String) -> (
  Model, Model, Model
) {
  let x = Input()
  let linear1 = Dense(count: intermediateSize, name: "\(name)_linear1")
  var out = linear1(x).GELU(approximate: .tanh)
  let outProjection = Dense(count: hiddenSize, name: "\(name)_out_proj")
  out = outProjection(out)
  return (linear1, outProjection, Model([x], [out]))
}

private func JointTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, t: Int, hw: Int, contextBlockPreOnly: Bool,
  usesFlashAtttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  let context = Input()
  let x = Input()
  let c = Input()
  let rot = Input()
  let contextAdaLNs = (0..<(contextBlockPreOnly ? 2 : 6)).map {
    Dense(count: k * h, name: "context_ada_ln_\($0)")
  }
  let contextChunks = contextAdaLNs.map { $0(c) }
  let contextNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var contextOut = (1 + contextChunks[1]) .* contextNorm1(context) + contextChunks[0]
  let contextToKeys = Dense(count: k * h, name: "c_k")
  let contextToQueries = Dense(count: k * h, name: "c_q")
  let contextToValues = Dense(count: k * h, name: "c_v")
  var contextK = contextToKeys(contextOut).reshaped([b, t, h, k])
  let normAddedK = RMSNorm(epsilon: 1e-6, axis: [3], name: "c_norm_k")
  contextK = normAddedK(contextK)
  var contextQ = contextToQueries(contextOut).reshaped([b, t, h, k])
  let normAddedQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "c_norm_q")
  contextQ = normAddedQ(contextQ)
  let contextV = contextToValues(contextOut).reshaped([b, t, h, k])
  let xAdaLNs = (0..<6).map { Dense(count: k * h, name: "x_ada_ln_\($0)") }
  let xChunks = xAdaLNs.map { $0(c) }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = (1 + xChunks[1]) .* xNorm1(x) + xChunks[0]
  let xToKeys = Dense(count: k * h, name: "x_k")
  let xToQueries = Dense(count: k * h, name: "x_q")
  let xToValues = Dense(count: k * h, name: "x_v")
  var xK = xToKeys(xOut).reshaped([b, hw, h, k])
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_k")
  xK = normK(xK)
  var xQ = xToQueries(xOut).reshaped([b, hw, h, k])
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_q")
  xQ = normQ(xQ)
  let xV = xToValues(xOut).reshaped([b, hw, h, k])
  var keys = Functional.concat(axis: 1, contextK, xK)
  var values = Functional.concat(axis: 1, contextV, xV)
  var queries = Functional.concat(axis: 1, contextQ, xQ)
  queries = Functional.cmul(left: queries, right: rot)
  keys = Functional.cmul(left: keys, right: rot)
  // Now run attention.
  var out: Model.IO
  switch usesFlashAtttention {
  case .none:
    keys = keys.transposed(1, 2)
    queries = ((1.0 / Float(k).squareRoot()) * queries)
      .transposed(1, 2)
    values = values.transposed(1, 2)
    if b * h <= 256 {
      var outs = [Model.IO]()
      for i in 0..<(b * h) {
        let key = keys.reshaped([1, t + hw, k], offset: [i, 0, 0], strides: [(t + hw) * k, k, 1])
        let query = queries.reshaped(
          [1, t + hw, k], offset: [i, 0, 0], strides: [(t + hw) * k, k, 1])
        let value = values.reshaped(
          [1, t + hw, k], offset: [i, 0, 0], strides: [(t + hw) * k, k, 1])
        var dot = Matmul(transposeB: (1, 2))(query, key)
        if let last = outs.last {
          dot.add(dependencies: [last])
        }
        dot = dot.reshaped([t + hw, t + hw])
        dot = dot.softmax()
        dot = dot.reshaped([1, t + hw, t + hw])
        outs.append(dot * value)
      }
      out = Concat(axis: 0)(outs)
      out = out.reshaped([b, h, t + hw, k]).transposed(1, 2).reshaped([b, t + hw, h * k])
    } else {
      var dot = Matmul(transposeB: (2, 3))(queries, keys)
      dot = dot.reshaped([b * h * (t + hw), t + hw])
      dot = dot.softmax()
      dot = dot.reshaped([b, h, (t + hw), t + hw])
      out = dot * values
      out = out.reshaped([b, h, (t + hw), k]).transposed(1, 2).reshaped([b, (t + hw), h * k])
    }
  case .scale1:
    queries = (1.0 / Float(k).squareRoot()) * queries
    let scaledDotProductAttention = ScaledDotProductAttention(scale: 1)
    out = scaledDotProductAttention(queries, keys, values).reshaped([b, (t + hw), k * h])
  case .scaleMerged:
    let scaledDotProductAttention = ScaledDotProductAttention(scale: 1.0 / Float(k).squareRoot())
    out = scaledDotProductAttention(queries, keys, values).reshaped([b, (t + hw), k * h])
  }
  let contextUnifyheads: Model?
  if !contextBlockPreOnly {
    contextOut = out.reshaped([b, t, h * k], strides: [(t + hw) * h * k, h * k, 1]).contiguous()
    let unifyheads = Dense(count: k * h, name: "c_o")
    contextOut = unifyheads(contextOut)
    contextUnifyheads = unifyheads
  } else {
    contextUnifyheads = nil
  }
  xOut = out.reshaped([b, hw, h * k], offset: [0, t, 0], strides: [(t + hw) * h * k, h * k, 1])
    .contiguous()
  let xUnifyheads = Dense(count: k * h, name: "x_o")
  xOut = xUnifyheads(xOut)
  if !contextBlockPreOnly {
    contextOut = context + contextChunks[2] .* contextOut
  }
  xOut = x + xChunks[2] .* xOut
  // Attentions are now. Now run MLP.
  let contextLinear1: Model?
  let contextOutProjection: Model?
  if !contextBlockPreOnly {
    let contextFF: Model
    (contextLinear1, contextOutProjection, contextFF) = FeedForward(
      hiddenSize: k * h, intermediateSize: k * h * 4, name: "c")
    let contextNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
    contextOut = contextOut + contextChunks[5]
      .* contextFF(contextNorm2(contextOut) .* (1 + contextChunks[4]) + contextChunks[3])
  } else {
    contextLinear1 = nil
    contextOutProjection = nil
  }
  let (xLinear1, xOutProjection, xFF) = FeedForward(
    hiddenSize: k * h, intermediateSize: k * h * 4, name: "x")
  let xNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  xOut = xOut + xChunks[5] .* xFF(xNorm2(xOut) .* (1 + xChunks[4]) + xChunks[3])
  let mapper: ModelWeightMapper = { _ in
    var mapping: [String: [String]] = [:]
    mapping["\(prefix).txt_attn.qkv.weight"] = [
      contextToQueries.weight.name, contextToKeys.weight.name, contextToValues.weight.name,
    ]
    mapping["\(prefix).txt_attn.qkv.bias"] = [
      contextToQueries.bias.name, contextToKeys.bias.name, contextToValues.bias.name,
    ]
    mapping["\(prefix).txt_attn.norm.key_norm.scale"] = [normAddedK.weight.name]
    mapping["\(prefix).txt_attn.norm.query_norm.scale"] = [normAddedQ.weight.name]
    mapping["\(prefix).img_attn.qkv.weight"] = [
      xToQueries.weight.name, xToKeys.weight.name, xToValues.weight.name,
    ]
    mapping["\(prefix).img_attn.qkv.bias"] = [
      xToQueries.bias.name, xToKeys.bias.name, xToValues.bias.name,
    ]
    mapping["\(prefix).img_attn.norm.key_norm.scale"] = [normK.weight.name]
    mapping["\(prefix).img_attn.norm.query_norm.scale"] = [normQ.weight.name]
    if let contextUnifyheads = contextUnifyheads {
      mapping["\(prefix).txt_attn.proj.weight"] = [contextUnifyheads.weight.name]
      mapping["\(prefix).txt_attn.proj.bias"] = [contextUnifyheads.bias.name]
    }
    mapping["\(prefix).img_attn.proj.weight"] = [xUnifyheads.weight.name]
    mapping["\(prefix).img_attn.proj.bias"] = [xUnifyheads.bias.name]
    if let contextLinear1 = contextLinear1,
      let contextOutProjection = contextOutProjection
    {
      mapping["\(prefix).txt_mlp.0.weight"] = [contextLinear1.weight.name]
      mapping["\(prefix).txt_mlp.0.bias"] = [contextLinear1.bias.name]
      mapping["\(prefix).txt_mlp.2.weight"] = [contextOutProjection.weight.name]
      mapping["\(prefix).txt_mlp.2.bias"] = [contextOutProjection.bias.name]
    }
    mapping["\(prefix).img_mlp.0.weight"] = [xLinear1.weight.name]
    mapping["\(prefix).img_mlp.0.bias"] = [xLinear1.bias.name]
    mapping["\(prefix).img_mlp.2.weight"] = [xOutProjection.weight.name]
    mapping["\(prefix).img_mlp.2.bias"] = [xOutProjection.bias.name]
    mapping["\(prefix).txt_mod.lin.weight"] = (0..<(contextBlockPreOnly ? 2 : 6)).map {
      contextAdaLNs[$0].weight.name
    }
    mapping["\(prefix).txt_mod.lin.bias"] = (0..<(contextBlockPreOnly ? 2 : 6)).map {
      contextAdaLNs[$0].bias.name
    }
    mapping["\(prefix).img_mod.lin.weight"] = (0..<6).map { xAdaLNs[$0].weight.name }
    mapping["\(prefix).img_mod.lin.bias"] = (0..<6).map { xAdaLNs[$0].bias.name }
    return mapping
  }
  if !contextBlockPreOnly {
    return (mapper, Model([context, x, c, rot], [contextOut, xOut]))
  } else {
    return (mapper, Model([context, x, c, rot], [xOut]))
  }
}

private func SingleTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, t: Int, hw: Int, contextBlockPreOnly: Bool,
  usesFlashAtttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let c = Input()
  let rot = Input()
  let xAdaLNs = (0..<3).map { Dense(count: k * h, name: "x_ada_ln_\($0)") }
  let xChunks = xAdaLNs.map { $0(c) }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = (1 + xChunks[1]) .* xNorm1(x) + xChunks[0]
  let xToKeys = Dense(count: k * h, name: "x_k")
  let xToQueries = Dense(count: k * h, name: "x_q")
  let xToValues = Dense(count: k * h, name: "x_v")
  var xK = xToKeys(xOut).reshaped([b, t + hw, h, k])
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_k")
  xK = normK(xK)
  var xQ = xToQueries(xOut).reshaped([b, t + hw, h, k])
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_q")
  xQ = normQ(xQ)
  let xV = xToValues(xOut).reshaped([b, t + hw, h, k])
  var keys = xK
  var values = xV
  var queries = xQ
  queries = Functional.cmul(left: queries, right: rot)
  keys = Functional.cmul(left: keys, right: rot)
  // Now run attention.
  var out: Model.IO
  switch usesFlashAtttention {
  case .none:
    keys = keys.transposed(1, 2)
    queries = ((1.0 / Float(k).squareRoot()) * queries)
      .transposed(1, 2)
    values = values.transposed(1, 2)
    if b * h <= 256 {
      var outs = [Model.IO]()
      for i in 0..<(b * h) {
        let key = keys.reshaped([1, t + hw, k], offset: [i, 0, 0], strides: [(t + hw) * k, k, 1])
        let query = queries.reshaped(
          [1, t + hw, k], offset: [i, 0, 0], strides: [(t + hw) * k, k, 1])
        let value = values.reshaped(
          [1, t + hw, k], offset: [i, 0, 0], strides: [(t + hw) * k, k, 1])
        var dot = Matmul(transposeB: (1, 2))(query, key)
        if let last = outs.last {
          dot.add(dependencies: [last])
        }
        dot = dot.reshaped([t + hw, t + hw])
        dot = dot.softmax()
        dot = dot.reshaped([1, t + hw, t + hw])
        outs.append(dot * value)
      }
      out = Concat(axis: 0)(outs)
      out = out.reshaped([b, h, t + hw, k]).transposed(1, 2).reshaped([b, t + hw, h * k])
    } else {
      var dot = Matmul(transposeB: (2, 3))(queries, keys)
      dot = dot.reshaped([b * h * (t + hw), t + hw])
      dot = dot.softmax()
      dot = dot.reshaped([b, h, (t + hw), t + hw])
      out = dot * values
      out = out.reshaped([b, h, (t + hw), k]).transposed(1, 2).reshaped([b, (t + hw), h * k])
    }
  case .scale1:
    queries = (1.0 / Float(k).squareRoot()) * queries
    let scaledDotProductAttention = ScaledDotProductAttention(scale: 1)
    out = scaledDotProductAttention(queries, keys, values).reshaped([b, (t + hw), k * h])
  case .scaleMerged:
    let scaledDotProductAttention = ScaledDotProductAttention(scale: 1.0 / Float(k).squareRoot())
    out = scaledDotProductAttention(queries, keys, values).reshaped([b, (t + hw), k * h])
  }
  var xIn: Model.IO = x
  if contextBlockPreOnly {
    out = out.reshaped([b, hw, h * k], offset: [0, t, 0], strides: [(t + hw) * h * k, h * k, 1])
      .contiguous()
    xIn = x.reshaped([b, hw, h * k], offset: [0, t, 0], strides: [(t + hw) * h * k, h * k, 1])
      .contiguous()
    xOut = xOut.reshaped(
      [b, hw, h * k], offset: [0, t, 0], strides: [(t + hw) * h * k, h * k, 1]
    )
    .contiguous()
  }
  let xUnifyheads = Dense(count: k * h, noBias: true, name: "x_o")
  let xLinear1 = Dense(count: k * h * 4, name: "x_linear1")
  let xOutProjection = Dense(count: k * h, name: "x_out_proj")
  out = xUnifyheads(out) + xOutProjection(xLinear1(xOut).GELU(approximate: .tanh))
  out = xIn + xChunks[2] .* out
  let mapper: ModelWeightMapper = { _ in
    var mapping: [String: [String]] = [:]
    mapping["\(prefix).linear1.weight"] = [
      xToQueries.weight.name, xToKeys.weight.name, xToValues.weight.name, xLinear1.weight.name,
    ]
    mapping["\(prefix).linear1.bias"] = [
      xToQueries.bias.name, xToKeys.bias.name, xToValues.bias.name, xLinear1.bias.name,
    ]
    mapping["\(prefix).norm.key_norm.scale"] = [normK.weight.name]
    mapping["\(prefix).norm.query_norm.scale"] = [normQ.weight.name]
    mapping["\(prefix).linear2.weight"] = [xUnifyheads.weight.name, xOutProjection.weight.name]
    mapping["\(prefix).linear2.bias"] = [xOutProjection.bias.name]
    mapping["\(prefix).modulation.lin.weight"] = (0..<3).map { xAdaLNs[$0].weight.name }
    mapping["\(prefix).modulation.lin.bias"] = (0..<3).map { xAdaLNs[$0].bias.name }
    return mapping
  }
  return (mapper, Model([x, c, rot], [out]))
}

private func Flux1(
  batchSize: Int, tokenLength: Int, height: Int, width: Int, channels: Int, layers: (Int, Int),
  usesFlashAttention: FlashAttentionLevel, guidanceEmbed: Bool = false,
  of: FloatType.Type = FloatType.self
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let t = Input()
  let y = Input()
  let contextIn = Input()
  let rot = Input()
  let guidance: Input?
  let h = height / 2
  let w = width / 2
  let xEmbedder = Dense(count: channels, name: "x_embedder")
  var out = xEmbedder(x)
  let (tMlp0, tMlp2, tEmbedder) = MLPEmbedder(channels: channels, name: "t")
  var vec = tEmbedder(t)
  let gMlp0: Model?
  let gMlp2: Model?
  if guidanceEmbed {
    let (mlp0, mlp2, gEmbedder) = MLPEmbedder(channels: channels, name: "guidance")
    let g = Input()
    vec = vec + gEmbedder(g)
    guidance = g
    gMlp0 = mlp0
    gMlp2 = mlp2
  } else {
    gMlp0 = nil
    gMlp2 = nil
    guidance = nil
  }
  let (yMlp0, yMlp2, yEmbedder) = MLPEmbedder(channels: channels, name: "vector")
  vec = vec + yEmbedder(y)
  let contextEmbedder = Dense(count: channels, name: "context_embedder")
  var context = contextEmbedder(contextIn)
  let c = vec.reshaped([batchSize, 1, channels]).swish()
  var mappers = [ModelWeightMapper]()
  for i in 0..<layers.0 {
    let (mapper, block) = JointTransformerBlock(
      prefix: "double_blocks.\(i)", k: 128, h: channels / 128, b: batchSize, t: tokenLength,
      hw: h * w,
      contextBlockPreOnly: false, usesFlashAtttention: usesFlashAttention)
    let blockOut = block(context, out, c, rot)
    context = blockOut[0]
    out = blockOut[1]
    mappers.append(mapper)
  }
  out = Functional.concat(axis: 1, context, out)
  for i in 0..<layers.1 {
    let (mapper, block) = SingleTransformerBlock(
      prefix: "single_blocks.\(i)", k: 128, h: channels / 128, b: batchSize, t: tokenLength,
      hw: h * w,
      contextBlockPreOnly: i == layers.1 - 1, usesFlashAtttention: usesFlashAttention)
    out = block(out, c, rot)
    mappers.append(mapper)
  }
  let scale = Dense(count: channels, name: "ada_ln_0")
  let shift = Dense(count: channels, name: "ada_ln_1")
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  out = (1 + scale(c)) .* normFinal(out) + shift(c)
  let projOut = Dense(count: 2 * 2 * 16, name: "linear")
  out = projOut(out)
  let mapper: ModelWeightMapper = { format in
    var mapping = [String: [String]]()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping["img_in.weight"] = [xEmbedder.weight.name]
    mapping["img_in.bias"] = [xEmbedder.bias.name]
    mapping["time_in.in_layer.weight"] = [tMlp0.weight.name]
    mapping["time_in.in_layer.bias"] = [tMlp0.bias.name]
    mapping["time_in.out_layer.weight"] = [tMlp2.weight.name]
    mapping["time_in.out_layer.bias"] = [tMlp2.bias.name]
    if let gMlp0 = gMlp0, let gMlp2 = gMlp2 {
      mapping["guidance_in.in_layer.weight"] = [gMlp0.weight.name]
      mapping["guidance_in.in_layer.bias"] = [gMlp0.bias.name]
      mapping["guidance_in.out_layer.weight"] = [gMlp2.weight.name]
      mapping["guidance_in.out_layer.bias"] = [gMlp2.bias.name]
    }
    mapping["vector_in.in_layer.weight"] = [yMlp0.weight.name]
    mapping["vector_in.in_layer.bias"] = [yMlp0.bias.name]
    mapping["vector_in.out_layer.weight"] = [yMlp2.weight.name]
    mapping["vector_in.out_layer.bias"] = [yMlp2.bias.name]
    mapping["txt_in.weight"] = [contextEmbedder.weight.name]
    mapping["txt_in.bias"] = [contextEmbedder.bias.name]
    mapping["final_layer.adaLN_modulation.1.weight"] = [shift.weight.name, scale.weight.name]
    mapping["final_layer.adaLN_modulation.1.bias"] = [shift.bias.name, scale.bias.name]
    mapping["final_layer.linear.weight"] = [projOut.weight.name]
    mapping["final_layer.linear.bias"] = [projOut.bias.name]
    return mapping
  }
  return (mapper, Model([x, t, y, contextIn, rot] + (guidance.map { [$0] } ?? []), [out]))
}
