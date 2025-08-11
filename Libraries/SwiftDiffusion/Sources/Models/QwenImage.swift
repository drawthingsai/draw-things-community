import Foundation
import NNC

public func QwenImageRotaryPositionEmbedding(
  height: Int, width: Int, tokenLength: Int, channels: Int, heads: Int = 1
)
  -> Tensor<Float>
{
  var rotTensor = Tensor<Float>(.CPU, .NHWC(1, height * width + tokenLength, heads, channels))
  let dim0 = channels / 8
  let dim1 = channels * 7 / 16
  let dim2 = dim1
  assert(channels % 16 == 0)
  let maxImgIdx = max(height / 2, width / 2)
  let imageLength = height * width
  for i in 0..<tokenLength {
    for j in 0..<heads {
      for k in 0..<(dim0 / 2) {
        let theta = Double(i + maxImgIdx) * 1.0 / pow(10_000, Double(k) * 2 / Double(dim0))
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, i + imageLength, j, k * 2] = Float(costheta)
        rotTensor[0, i + imageLength, j, k * 2 + 1] = Float(sintheta)
      }
      for k in 0..<(dim1 / 2) {
        let theta = Double(i + maxImgIdx) * 1.0 / pow(10_000, Double(k) * 2 / Double(dim1))
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, i + imageLength, j, (k + (dim0 / 2)) * 2] = Float(costheta)
        rotTensor[0, i + imageLength, j, (k + (dim0 / 2)) * 2 + 1] = Float(sintheta)
      }
      for k in 0..<(dim2 / 2) {
        let theta = Double(i + maxImgIdx) * 1.0 / pow(10_000, Double(k) * 2 / Double(dim2))
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, i + imageLength, j, (k + (dim0 / 2) + (dim1 / 2)) * 2] = Float(costheta)
        rotTensor[0, i + imageLength, j, (k + (dim0 / 2) + (dim1 / 2)) * 2 + 1] = Float(sintheta)
      }
    }
  }
  for y in 0..<height {
    for x in 0..<width {
      let i = y * width + x
      for j in 0..<heads {
        for k in 0..<(dim0 / 2) {
          let theta = 0 * 1.0 / pow(10_000, Double(k) * 2 / Double(dim0))
          let sintheta = sin(theta)
          let costheta = cos(theta)
          rotTensor[0, i, j, k * 2] = Float(costheta)
          rotTensor[0, i, j, k * 2 + 1] = Float(sintheta)
        }
        for k in 0..<(dim1 / 2) {
          let theta =
            Double(y - (height - height / 2)) * 1.0 / pow(10_000, Double(k) * 2 / Double(dim1))
          let sintheta = sin(theta)
          let costheta = cos(theta)
          rotTensor[0, i, j, (k + (dim0 / 2)) * 2] = Float(costheta)
          rotTensor[0, i, j, (k + (dim0 / 2)) * 2 + 1] = Float(sintheta)
        }
        for k in 0..<(dim2 / 2) {
          let theta =
            Double(x - (width - width / 2)) * 1.0 / pow(10_000, Double(k) * 2 / Double(dim2))
          let sintheta = sin(theta)
          let costheta = cos(theta)
          rotTensor[0, i, j, (k + (dim0 / 2) + (dim1 / 2)) * 2] = Float(costheta)
          rotTensor[0, i, j, (k + (dim0 / 2) + (dim1 / 2)) * 2 + 1] = Float(sintheta)
        }
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

private func FeedForward(hiddenSize: Int, intermediateSize: Int, scaleFactor: Float, name: String)
  -> (
    Model, Model, Model
  )
{
  let x = Input()
  let linear1 = Dense(count: intermediateSize, name: "\(name)_linear1")
  var out = linear1(x).GELU(approximate: .tanh)
  out = (1.0 / scaleFactor) * out
  // The scale down is integrated into out proj bias.
  let outProjection = Dense(count: hiddenSize, name: "\(name)_out_proj")
  out = scaleFactor * outProjection(out).to(.Float32)
  return (linear1, outProjection, Model([x], [out]))
}

private func JointTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, t: Int, hw: Int, contextBlockPreOnly: Bool,
  usesFlashAttention: FlashAttentionLevel, scaleFactor: (Float, Float)
) -> (ModelWeightMapper, Model) {
  let context = Input()
  let x = Input()
  let c = Input()
  let rot = Input()
  let contextAdaLNs = (0..<(contextBlockPreOnly ? 2 : 6)).map {
    Dense(count: k * h, name: "context_ada_ln_\($0)")
  }
  let contextChunks = contextAdaLNs.map { $0(c) }
  let contextNorm1 = LayerNorm(
    epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var contextOut =
    ((1 + contextChunks[1].to(of: context)) .* contextNorm1(context)
      + contextChunks[0].to(of: context))
  let contextToKeys = Dense(count: k * h, name: "c_k")
  let contextToQueries = Dense(count: k * h, name: "c_q")
  let contextToValues = Dense(count: k * h, name: "c_v")
  let downcastContextOut = ((1.0 / 8) * contextOut).to(.Float16)
  var contextK = contextToKeys(downcastContextOut).reshaped([b, t, h, k])
  let normAddedK = RMSNorm(
    epsilon: 1e-6 / (8.0 * 8.0 /* This is to remove the scale down factor */), axis: [3],
    name: "c_norm_k")
  contextK = normAddedK(contextK)
  var contextQ = contextToQueries(downcastContextOut).reshaped([b, t, h, k])
  let normAddedQ = RMSNorm(
    epsilon: 1e-6 / (8.0 * 8.0 /* This is to remove the scale down factor */), axis: [3],
    name: "c_norm_q")
  contextQ = normAddedQ(contextQ)
  let contextV = contextToValues(downcastContextOut).reshaped([b, t, h, k])
  let xAdaLNs = (0..<6).map { Dense(count: k * h, name: "x_ada_ln_\($0)") }
  let xChunks = xAdaLNs.map { $0(c) }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = xNorm1(x)
  xOut = ((1 + xChunks[1].to(of: x)) .* xOut + xChunks[0].to(of: x))
  let xToKeys = Dense(count: k * h, name: "x_k")
  let xToQueries = Dense(count: k * h, name: "x_q")
  let xToValues = Dense(count: k * h, name: "x_v")
  let downcastXOut = ((1.0 / 8) * xOut).to(.Float16)
  var xK = xToKeys(downcastXOut).reshaped([b, hw, h, k])
  let normK = RMSNorm(
    epsilon: 1e-6 / (8.0 * 8.0 /* This is to remove the scale down factor */), axis: [3],
    name: "x_norm_k")
  xK = normK(xK)
  var xQ = xToQueries(downcastXOut).reshaped([b, hw, h, k])
  let normQ = RMSNorm(
    epsilon: 1e-6 / (8.0 * 8.0 /* This is to remove the scale down factor */), axis: [3],
    name: "x_norm_q")
  xQ = normQ(xQ)
  let xV = xToValues(downcastXOut).reshaped([b, hw, h, k])
  var keys = Functional.concat(axis: 1, xK, contextK)
  var values = Functional.concat(axis: 1, xV, contextV)
  var queries = Functional.concat(axis: 1, xQ, contextQ)
  queries = Functional.cmul(left: queries, right: rot)
  keys = Functional.cmul(left: keys, right: rot)
  // Now run attention.
  var out: Model.IO
  switch usesFlashAttention {
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
    let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
    out = scaledDotProductAttention(queries, keys, values).reshaped([b, (t + hw), k * h])
  case .scaleMerged:
    out = ScaledDotProductAttention(scale: 1.0 / Float(k).squareRoot(), flags: [.Float16])(
      queries, keys, values
    ).reshaped([b, t + hw, h * k])
  }
  let contextUnifyheads: Model?
  if !contextBlockPreOnly {
    contextOut = out.reshaped(
      [b, t, h * k], offset: [0, hw, 0], strides: [(t + hw) * h * k, h * k, 1]
    ).contiguous()
    let unifyheads = Dense(count: k * h, name: "c_o")
    contextOut =
      (8 * scaleFactor.0) * unifyheads((1.0 / scaleFactor.0) * contextOut).to(of: context)
    contextUnifyheads = unifyheads
  } else {
    contextUnifyheads = nil
  }
  xOut = out.reshaped([b, hw, h * k], strides: [(t + hw) * h * k, h * k, 1])
    .contiguous()
  let xUnifyheads = Dense(count: k * h, name: "x_o")
  xOut = (8 * scaleFactor.0) * xUnifyheads((1.0 / scaleFactor.0) * xOut).to(of: x)
  if !contextBlockPreOnly {
    contextOut = context + (contextChunks[2]).to(of: context) .* contextOut
  }
  xOut = x + (xChunks[2]).to(of: x) .* xOut
  // Attentions are now. Now run MLP.
  let contextLinear1: Model?
  let contextOutProjection: Model?
  if !contextBlockPreOnly {
    let contextFF: Model
    (contextLinear1, contextOutProjection, contextFF) = FeedForward(
      hiddenSize: k * h, intermediateSize: k * h * 4, scaleFactor: scaleFactor.1, name: "c")
    let contextNorm2 = LayerNorm(
      epsilon: 1e-6, axis: [2], elementwiseAffine: false)
    contextOut =
      contextOut
      + contextChunks[5].to(of: contextOut)
      .* contextFF(
        (contextNorm2(contextOut) .* (1 + contextChunks[4].to(of: contextOut))
          + contextChunks[3].to(of: contextOut)).to(.Float16)
      ).to(
        of: contextOut)
  } else {
    contextLinear1 = nil
    contextOutProjection = nil
  }
  let (xLinear1, xOutProjection, xFF) = FeedForward(
    hiddenSize: k * h, intermediateSize: k * h * 4, scaleFactor: scaleFactor.1, name: "x")
  let xNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  xOut =
    xOut
    + xChunks[5].to(of: xOut)
    .* xFF((xNorm2(xOut) .* (1 + xChunks[4].to(of: xOut)) + xChunks[3].to(of: xOut)).to(.Float16))
    .to(of: xOut)
  let reader: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).attn.add_q_proj.weight"] = [contextToQueries.weight.name]
    mapping["\(prefix).attn.add_q_proj.bias"] = ModelWeightElement(
      [contextToQueries.bias.name], scale: 1.0 / 8)
    mapping["\(prefix).attn.add_k_proj.weight"] = [contextToKeys.weight.name]
    mapping["\(prefix).attn.add_k_proj.bias"] = ModelWeightElement(
      [contextToKeys.bias.name], scale: 1.0 / 8)
    mapping["\(prefix).attn.add_v_proj.weight"] = [contextToValues.weight.name]
    mapping["\(prefix).attn.add_v_proj.bias"] = ModelWeightElement(
      [contextToValues.bias.name], scale: 1.0 / 8)
    mapping["\(prefix).attn.norm_added_k.weight"] = [normAddedK.weight.name]
    mapping["\(prefix).attn.norm_added_q.weight"] = [normAddedQ.weight.name]
    mapping["\(prefix).attn.to_q.weight"] = [xToQueries.weight.name]
    mapping["\(prefix).attn.to_q.bias"] = ModelWeightElement([xToQueries.bias.name], scale: 1.0 / 8)
    mapping["\(prefix).attn.to_k.weight"] = [xToKeys.weight.name]
    mapping["\(prefix).attn.to_k.bias"] = ModelWeightElement([xToKeys.bias.name], scale: 1.0 / 8)
    mapping["\(prefix).attn.to_v.weight"] = [xToValues.weight.name]
    mapping["\(prefix).attn.to_v.bias"] = ModelWeightElement([xToValues.bias.name], scale: 1.0 / 8)
    mapping["\(prefix).attn.norm_k.weight"] = [normK.weight.name]
    mapping["\(prefix).attn.norm_q.weight"] = [normQ.weight.name]
    if let contextUnifyheads = contextUnifyheads {
      mapping["\(prefix).attn.to_add_out.weight"] = [contextUnifyheads.weight.name]
      mapping["\(prefix).attn.to_add_out.bias"] = ModelWeightElement(
        [contextUnifyheads.bias.name], scale: 1.0 / (8 * scaleFactor.0))
    }
    mapping["\(prefix).attn.to_out.0.weight"] = [xUnifyheads.weight.name]
    mapping["\(prefix).attn.to_out.0.bias"] = ModelWeightElement(
      [xUnifyheads.bias.name], scale: 1.0 / (8 * scaleFactor.0))
    if let contextLinear1 = contextLinear1,
      let contextOutProjection = contextOutProjection
    {
      mapping["\(prefix).txt_mlp.net.0.proj.weight"] = [contextLinear1.weight.name]
      mapping["\(prefix).txt_mlp.net.0.proj.bias"] = [contextLinear1.bias.name]
      mapping[
        "\(prefix).txt_mlp.net.2.weight"
      ] = [contextOutProjection.weight.name]
      mapping[
        "\(prefix).txt_mlp.net.2.bias"
      ] = ModelWeightElement([contextOutProjection.bias.name], scale: 1.0 / scaleFactor.1)
    }
    mapping["\(prefix).img_mlp.net.0.proj.weight"] = [xLinear1.weight.name]
    mapping["\(prefix).img_mlp.net.0.proj.bias"] = [xLinear1.bias.name]
    mapping["\(prefix).img_mlp.net.2.weight"] = [xOutProjection.weight.name]
    mapping["\(prefix).img_mlp.net.2.bias"] = ModelWeightElement(
      [xOutProjection.bias.name], scale: 1.0 / scaleFactor.1)
    mapping[
      "\(prefix).txt_mod.1.weight"
    ] = ModelWeightElement(
      (0..<(contextBlockPreOnly ? 2 : 6)).map { contextAdaLNs[$0].weight.name })
    mapping[
      "\(prefix).txt_mod.1.bias"
    ] = ModelWeightElement((0..<(contextBlockPreOnly ? 2 : 6)).map { contextAdaLNs[$0].bias.name })
    mapping["\(prefix).img_mod.1.weight"] = ModelWeightElement(
      (0..<6).map { xAdaLNs[$0].weight.name })
    mapping["\(prefix).img_mod.1.bias"] = ModelWeightElement((0..<6).map { xAdaLNs[$0].bias.name })
    return mapping
  }
  if !contextBlockPreOnly {
    return (reader, Model([x, context, c, rot], [xOut, contextOut]))
  } else {
    return (reader, Model([x, context, c, rot], [xOut]))
  }
}

func QwenImage(
  height: Int, width: Int, textLength: Int, layers: Int, usesFlashAttention: FlashAttentionLevel
) -> (
  ModelWeightMapper, Model
) {
  let x = Input()
  let rot = Input()
  let txt = Input()
  let t = Input()
  let imgIn = Dense(count: 3072, name: "x_embedder")
  let txtNorm = RMSNorm(epsilon: 1e-6, axis: [2], name: "context_norm")
  let txtIn = Dense(count: 3_072, name: "context_embedder")
  let (timeInMlp0, timeInMlp2, timeIn) = MLPEmbedder(channels: 3_072, name: "t")
  var vec = timeIn(t)
  vec = vec.reshaped([1, 1, 3072]).swish()
  var context = txtIn(txtNorm(txt))
  var mappers = [ModelWeightMapper]()
  context = context.to(.Float32)
  var out = imgIn(x).to(.Float32)
  let h = height / 2
  let w = width / 2
  for i in 0..<layers {
    let (mapper, block) = JointTransformerBlock(
      prefix: "transformer_blocks.\(i)", k: 128, h: 24, b: 1, t: textLength, hw: h * w,
      contextBlockPreOnly: i == layers - 1, usesFlashAttention: usesFlashAttention,
      scaleFactor: (i >= layers - 16 ? 16 : 2, i >= layers - 1 ? 256 : 16))
    let blockOut = block(out, context, vec, rot)
    if i == layers - 1 {
      out = blockOut
    } else {
      out = blockOut[0]
      context = blockOut[1]
    }
    mappers.append(mapper)
  }
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let scale = Dense(count: 3072, name: "ada_ln_0")
  let shift = Dense(count: 3072, name: "ada_ln_1")
  out = (1 + scale(vec)) .* normFinal(out).to(.Float16) + shift(vec)
  let projOut = Dense(count: 2 * 2 * 16, name: "linear")
  out = projOut(out)
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["img_in.weight"] = [imgIn.weight.name]
    mapping["img_in.bias"] = [imgIn.bias.name]
    mapping["txt_norm.weight"] = [txtNorm.weight.name]
    mapping["txt_in.weight"] = [txtIn.weight.name]
    mapping["txt_in.bias"] = [txtIn.bias.name]
    mapping[
      "time_text_embed.timestep_embedder.linear_1.weight"
    ] = [timeInMlp0.weight.name]
    mapping[
      "time_text_embed.timestep_embedder.linear_1.bias"
    ] = [timeInMlp0.bias.name]
    mapping[
      "time_text_embed.timestep_embedder.linear_2.weight"
    ] = [timeInMlp2.weight.name]
    mapping[
      "time_text_embed.timestep_embedder.linear_2.bias"
    ] = [timeInMlp2.bias.name]
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping["norm_out.linear.weight"] = [scale.weight.name, shift.weight.name]
    mapping["norm_out.linear.bias"] = [scale.bias.name, shift.bias.name]
    mapping["proj_out.weight"] = [projOut.weight.name]
    mapping["proj_out.bias"] = [projOut.bias.name]
    return mapping
  }
  return (mapper, Model([x, rot, t, txt], [out]))
}
