import NNC

private func TimeEmbedder(channels: Int) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: channels, name: "t_embedder_0")
  var out = fc0(x).swish()
  let fc2 = Dense(count: channels, name: "t_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

private func VectorEmbedder(channels: Int) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: channels, name: "y_embedder_0")
  var out = fc0(x).swish()
  let fc2 = Dense(count: channels, name: "y_embedder_1")
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

private func JointTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, t: Int, hw: Int, contextBlockPreOnly: Bool,
  usesFlashAtttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  let context = Input()
  let x = Input()
  let c = Input()
  let contextAdaLNs = (0..<(contextBlockPreOnly ? 2 : 6)).map {
    Dense(count: k * h, name: "context_ada_ln_\($0)")
  }
  let contextChunks = contextAdaLNs.map { $0(c) }
  let contextNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var contextOut = (1 + contextChunks[1]) .* contextNorm1(context) + contextChunks[0]
  let contextToKeys = Dense(count: k * h, name: "c_k")
  let contextToQueries = Dense(count: k * h, name: "c_q")
  let contextToValues = Dense(count: k * h, name: "c_v")
  let contextK = contextToKeys(contextOut)
  let contextQ = contextToQueries(contextOut)
  let contextV = contextToValues(contextOut)
  let xAdaLNs = (0..<6).map { Dense(count: k * h, name: "x_ada_ln_\($0)") }
  let xChunks = xAdaLNs.map { $0(c) }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = (1 + xChunks[1]) .* xNorm1(x) + xChunks[0]
  let xToKeys = Dense(count: k * h, name: "x_k")
  let xToQueries = Dense(count: k * h, name: "x_q")
  let xToValues = Dense(count: k * h, name: "x_v")
  let xK = xToKeys(xOut)
  let xQ = xToQueries(xOut)
  let xV = xToValues(xOut)
  var keys = Functional.concat(axis: 1, contextK, xK)
  var values = Functional.concat(axis: 1, contextV, xV)
  var queries = Functional.concat(axis: 1, contextQ, xQ)
  // Now run attention.
  var out: Model.IO
  switch usesFlashAtttention {
  case .none:
    keys = keys.reshaped([b, t + hw, h, k]).permuted(0, 2, 1, 3)
    queries = ((1.0 / Float(k).squareRoot()) * queries).reshaped([b, t + hw, h, k])
      .permuted(0, 2, 1, 3)
    values = values.reshaped([b, t + hw, h, k]).permuted(0, 2, 1, 3)
    var dot = Matmul(transposeB: (2, 3))(queries, keys)
    dot = dot.reshaped([b * h * (t + hw), t + hw])
    dot = dot.softmax()
    dot = dot.reshaped([b, h, (t + hw), t + hw])
    out = dot * values
    out = out.reshaped([b, h, (t + hw), k]).transposed(1, 2).reshaped([b, (t + hw), h * k])
  case .scale1:
    keys = keys.reshaped([b, t + hw, h, k])
    queries = ((1.0 / Float(k).squareRoot()) * queries).reshaped([b, t + hw, h, k])
    values = values.reshaped([b, t + hw, h, k])
    let scaledDotProductAttention = ScaledDotProductAttention(scale: 1)
    out = scaledDotProductAttention(queries, keys, values).reshaped([b, (t + hw), k * h])
  case .scaleMerged:
    keys = keys.reshaped([b, t + hw, h, k])
    queries = queries.reshaped([b, t + hw, h, k])
    values = values.reshaped([b, t + hw, h, k])
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
  let contextFc1: Model?
  let contextFc2: Model?
  if !contextBlockPreOnly {
    let contextMlp: Model
    (contextFc1, contextFc2, contextMlp) = MLP(
      hiddenSize: k * h, intermediateSize: k * h * 4, name: "c")
    let contextNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
    contextOut = contextOut + contextChunks[5]
      .* contextMlp(contextNorm2(contextOut) .* (1 + contextChunks[4]) + contextChunks[3])
  } else {
    contextFc1 = nil
    contextFc2 = nil
  }
  let (xFc1, xFc2, xMlp) = MLP(hiddenSize: k * h, intermediateSize: k * h * 4, name: "x")
  let xNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  xOut = xOut + xChunks[5] .* xMlp(xNorm2(xOut) .* (1 + xChunks[4]) + xChunks[3])
  let mapper: ModelWeightMapper = { _ in
    var mapping = [String: [String]]()
    mapping["\(prefix).context_block.attn.qkv.weight"] = [
      contextToQueries.weight.name, contextToKeys.weight.name, contextToValues.weight.name,
    ]
    mapping["\(prefix).context_block.attn.qkv.bias"] = [
      contextToQueries.bias.name, contextToKeys.bias.name, contextToValues.bias.name,
    ]
    mapping["\(prefix).x_block.attn.qkv.weight"] = [
      xToQueries.weight.name, xToKeys.weight.name, xToValues.weight.name,
    ]
    mapping["\(prefix).x_block.attn.qkv.bias"] = [
      xToQueries.bias.name, xToKeys.bias.name, xToValues.bias.name,
    ]
    if let contextUnifyheads = contextUnifyheads {
      mapping["\(prefix).context_block.attn.proj.weight"] = [contextUnifyheads.weight.name]
      mapping["\(prefix).context_block.attn.proj.bias"] = [contextUnifyheads.bias.name]
    }
    mapping["\(prefix).x_block.attn.proj.weight"] = [xUnifyheads.weight.name]
    mapping["\(prefix).x_block.attn.proj.bias"] = [xUnifyheads.bias.name]
    if let contextFc1 = contextFc1, let contextFc2 = contextFc2 {
      mapping["\(prefix).context_block.mlp.fc1.weight"] = [contextFc1.weight.name]
      mapping["\(prefix).context_block.mlp.fc1.bias"] = [contextFc1.bias.name]
      mapping["\(prefix).context_block.mlp.fc2.weight"] = [contextFc2.weight.name]
      mapping["\(prefix).context_block.mlp.fc2.bias"] = [contextFc2.bias.name]
    }
    mapping["\(prefix).x_block.mlp.fc1.weight"] = [xFc1.weight.name]
    mapping["\(prefix).x_block.mlp.fc1.bias"] = [xFc1.bias.name]
    mapping["\(prefix).x_block.mlp.fc2.weight"] = [xFc2.weight.name]
    mapping["\(prefix).x_block.mlp.fc2.bias"] = [xFc2.bias.name]
    mapping[
      "\(prefix).context_block.adaLN_modulation.1.weight"
    ] = (0..<(contextBlockPreOnly ? 2 : 6)).map { contextAdaLNs[$0].weight.name }
    mapping[
      "\(prefix).context_block.adaLN_modulation.1.bias"
    ] = (0..<(contextBlockPreOnly ? 2 : 6)).map { contextAdaLNs[$0].bias.name }
    mapping["\(prefix).x_block.adaLN_modulation.1.weight"] = (0..<6).map { xAdaLNs[$0].weight.name }
    mapping["\(prefix).x_block.adaLN_modulation.1.bias"] = (0..<6).map { xAdaLNs[$0].weight.name }
    return mapping
  }
  if !contextBlockPreOnly {
    return (mapper, Model([context, x, c], [contextOut, xOut]))
  } else {
    return (mapper, Model([context, x, c], [xOut]))
  }
}

func MMDiT(b: Int, t: Int, h: Int, w: Int, layers: Int, usesFlashAttention: FlashAttentionLevel)
  -> (ModelWeightMapper, Model)
{
  let x = Input()
  let timestep = Input()
  let y = Input()
  let contextIn = Input()
  let xEmbedder = Convolution(
    groups: 1, filters: 1536, filterSize: [2, 2],
    hint: Hint(stride: [2, 2]), format: .OIHW, name: "x_embedder")
  var out = xEmbedder(x).reshaped([b, 1536, h * w]).transposed(1, 2)
  let posEmbed = Parameter<FloatType>(.GPU(0), .NHWC(1, 192, 192, 1536), name: "pos_embed")
  let spatialPosEmbed = posEmbed.reshaped(
    [1, h, w, 1536], offset: [0, (192 - h) / 2, (192 - w) / 2, 0],
    strides: [192 * 192 * 1536, 192 * 1536, 1536, 1]
  ).contiguous().reshaped([1, h * w, 1536])
  out = spatialPosEmbed + out
  let (tMlp0, tMlp2, tEmbedder) = TimeEmbedder(channels: 1536)
  let (yMlp0, yMlp2, yEmbedder) = VectorEmbedder(channels: 1536)
  let c = (tEmbedder(timestep) + yEmbedder(y)).reshaped([b, 1, 1536]).swish()
  let contextEmbedder = Dense(count: 1536, name: "context_embedder")
  var context = contextEmbedder(contextIn)
  var mappers = [ModelWeightMapper]()
  for i in 0..<layers {
    let (mapper, block) = JointTransformerBlock(
      prefix: "diffusion_model.joint_blocks.\(i)", k: 64, h: 24, b: b, t: t, hw: h * w,
      contextBlockPreOnly: i == layers - 1, usesFlashAtttention: usesFlashAttention)
    let blockOut = block(context, out, c)
    if i == layers - 1 {
      out = blockOut
    } else {
      context = blockOut[0]
      out = blockOut[1]
    }
    mappers.append(mapper)
  }
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let shift = Dense(count: 1536, name: "ada_ln_0")
  let scale = Dense(count: 1536, name: "ada_ln_1")
  out = (1 + scale(c)) .* normFinal(out) + shift(c)
  let linear = Dense(count: 2 * 2 * 16, name: "linear")
  out = linear(out)
  // Unpatchify
  out = out.reshaped([b, h, w, 2, 2, 16]).permuted(0, 1, 3, 2, 4, 5).contiguous().reshaped([
    b, h * 2, w * 2, 16,
  ])
  let mapper: ModelWeightMapper = { format in
    var mapping = [String: [String]]()
    mapping["diffusion_model.x_embedder.proj.weight"] = [xEmbedder.weight.name]
    mapping["diffusion_model.x_embedder.proj.bias"] = [xEmbedder.bias.name]
    mapping["diffusion_model.pos_embed"] = [posEmbed.weight.name]
    mapping["diffusion_model.t_embedder.mlp.0.weight"] = [tMlp0.weight.name]
    mapping["diffusion_model.t_embedder.mlp.0.bias"] = [tMlp0.bias.name]
    mapping["diffusion_model.t_embedder.mlp.2.weight"] = [tMlp2.weight.name]
    mapping["diffusion_model.t_embedder.mlp.2.bias"] = [tMlp2.bias.name]
    mapping["diffusion_model.y_embedder.mlp.0.weight"] = [yMlp0.weight.name]
    mapping["diffusion_model.y_embedder.mlp.0.bias"] = [yMlp0.bias.name]
    mapping["diffusion_model.y_embedder.mlp.2.weight"] = [yMlp2.weight.name]
    mapping["diffusion_model.y_embedder.mlp.2.bias"] = [yMlp2.bias.name]
    mapping["diffusion_model.context_embedder.weight"] = [contextEmbedder.weight.name]
    mapping["diffusion_model.context_embedder.bias"] = [contextEmbedder.bias.name]
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping[
      "diffusion_model.final_layer.adaLN_modulation.1.weight"
    ] = [shift.weight.name, scale.weight.name]
    mapping[
      "diffusion_model.final_layer.adaLN_modulation.1.bias"
    ] = [shift.bias.name, scale.bias.name]
    mapping["diffusion_model.final_layer.linear.weight"] = [linear.weight.name]
    mapping["diffusion_model.final_layer.linear.bias"] = [linear.bias.name]
    return mapping
  }
  return (mapper, Model([x, timestep, contextIn, y], [out]))
}
