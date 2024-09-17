import Foundation
import NNC

public func Flux1RotaryPositionEmbedding(height: Int, width: Int, tokenLength: Int, channels: Int)
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
      let theta = 0 * 1.0 / pow(10_000, Double(k) * 2 / Double(dim2))
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotTensor[0, i, 0, (k + (dim0 / 2) + (dim1 / 2)) * 2] = Float(costheta)
      rotTensor[0, i, 0, (k + (dim0 / 2) + (dim1 / 2)) * 2 + 1] = Float(sintheta)
    }
  }
  for y in 0..<height {
    for x in 0..<width {
      let i = y * width + x + tokenLength
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

private func FeedForward(hiddenSize: Int, intermediateSize: Int, upcast: Bool, name: String) -> (
  Model, Model, Model
) {
  let x = Input()
  let linear1 = Dense(count: intermediateSize, name: "\(name)_linear1")
  var out = linear1(x).GELU(approximate: .tanh)
  if upcast {
    let scaleFactor: Float = 8
    out = (1 / scaleFactor) * out
  }
  let outProjection = Dense(count: hiddenSize, flags: [.Float32], name: "\(name)_out_proj")
  out = outProjection(out)
  if upcast {
    let scaleFactor: Float = 8
    out = out.to(.Float32) * scaleFactor
  }
  return (linear1, outProjection, Model([x], [out]))
}

private func JointTransformerBlock(
  prefix: (String, String), k: Int, h: Int, b: Int, t: Int, hw: Int, contextBlockPreOnly: Bool,
  upcast: Bool, usesFlashAtttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  let context = Input()
  let x = Input()
  let rot = Input()
  let contextChunks = (0..<(contextBlockPreOnly ? 2 : 6)).map { _ in
    Input()
  }
  let contextNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var contextOut = contextChunks[1] .* contextNorm1(context).to(.Float16) + contextChunks[0]
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
  let xChunks = (0..<6).map { _ in Input() }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = xChunks[1] .* xNorm1(x).to(.Float16) + xChunks[0]
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
    let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
    out = scaledDotProductAttention(queries, keys, values).reshaped([b, (t + hw), k * h])
  case .scaleMerged:
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1.0 / Float(k).squareRoot(), flags: [.Float16])
    out = scaledDotProductAttention(queries, keys, values).reshaped([b, (t + hw), k * h])
  }
  let contextUnifyheads: Model?
  if !contextBlockPreOnly {
    contextOut = out.reshaped([b, t, h * k], strides: [(t + hw) * h * k, h * k, 1])
    let unifyheads = Dense(count: k * h, name: "c_o")
    contextOut = unifyheads(contextOut)
    contextUnifyheads = unifyheads
  } else {
    contextUnifyheads = nil
  }
  xOut = out.reshaped([b, hw, h * k], offset: [0, t, 0], strides: [(t + hw) * h * k, h * k, 1])
  let xUnifyheads = Dense(count: k * h, name: "x_o")
  xOut = xUnifyheads(xOut)
  if !contextBlockPreOnly {
    contextOut = context + (contextChunks[2] .* contextOut).to(of: context)
  }
  xOut = x + (xChunks[2] .* xOut).to(of: x)
  // Attentions are now. Now run MLP.
  let contextLinear1: Model?
  let contextOutProjection: Model?
  if !contextBlockPreOnly {
    let contextFF: Model
    (contextLinear1, contextOutProjection, contextFF) = FeedForward(
      hiddenSize: k * h, intermediateSize: k * h * 4, upcast: upcast, name: "c")
    let contextNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
    if upcast {
      contextOut = contextOut + contextChunks[5].to(of: contextOut)
        .* contextFF(contextNorm2(contextOut).to(.Float16) .* contextChunks[4] + contextChunks[3])
    } else {
      contextOut =
        contextOut
        + (contextChunks[5]
        .* contextFF(contextNorm2(contextOut).to(.Float16) .* contextChunks[4] + contextChunks[3]))
        .to(of: contextOut)
    }
  } else {
    contextLinear1 = nil
    contextOutProjection = nil
  }
  let (xLinear1, xOutProjection, xFF) = FeedForward(
    hiddenSize: k * h, intermediateSize: k * h * 4, upcast: upcast, name: "x")
  let xNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  if upcast {
    xOut = xOut + xChunks[5].to(of: xOut)
      .* xFF(xNorm2(xOut).to(.Float16) .* xChunks[4] + xChunks[3])
  } else {
    xOut =
      xOut + (xChunks[5] .* xFF(xNorm2(xOut).to(.Float16) .* xChunks[4] + xChunks[3])).to(of: xOut)
  }
  let mapper: ModelWeightMapper = { format in
    var mapping: [String: ModelWeightElement] = [:]
    switch format {
    case .generativeModels:
      mapping["\(prefix.0).txt_attn.qkv.weight"] = [
        contextToQueries.weight.name, contextToKeys.weight.name, contextToValues.weight.name,
      ]
      mapping["\(prefix.0).txt_attn.qkv.bias"] = [
        contextToQueries.bias.name, contextToKeys.bias.name, contextToValues.bias.name,
      ]
      mapping["\(prefix.0).txt_attn.norm.key_norm.scale"] = [normAddedK.weight.name]
      mapping["\(prefix.0).txt_attn.norm.query_norm.scale"] = [normAddedQ.weight.name]
      mapping["\(prefix.0).img_attn.qkv.weight"] = [
        xToQueries.weight.name, xToKeys.weight.name, xToValues.weight.name,
      ]
      mapping["\(prefix.0).img_attn.qkv.bias"] = [
        xToQueries.bias.name, xToKeys.bias.name, xToValues.bias.name,
      ]
      mapping["\(prefix.0).img_attn.norm.key_norm.scale"] = [normK.weight.name]
      mapping["\(prefix.0).img_attn.norm.query_norm.scale"] = [normQ.weight.name]
      if let contextUnifyheads = contextUnifyheads {
        mapping["\(prefix.0).txt_attn.proj.weight"] = [contextUnifyheads.weight.name]
        mapping["\(prefix.0).txt_attn.proj.bias"] = [contextUnifyheads.bias.name]
      }
      mapping["\(prefix.0).img_attn.proj.weight"] = [xUnifyheads.weight.name]
      mapping["\(prefix.0).img_attn.proj.bias"] = [xUnifyheads.bias.name]
      if let contextLinear1 = contextLinear1,
        let contextOutProjection = contextOutProjection
      {
        mapping["\(prefix.0).txt_mlp.0.weight"] = [contextLinear1.weight.name]
        mapping["\(prefix.0).txt_mlp.0.bias"] = [contextLinear1.bias.name]
        mapping["\(prefix.0).txt_mlp.2.weight"] = [contextOutProjection.weight.name]
        mapping["\(prefix.0).txt_mlp.2.bias"] = [contextOutProjection.bias.name]
      }
      mapping["\(prefix.0).img_mlp.0.weight"] = [xLinear1.weight.name]
      mapping["\(prefix.0).img_mlp.0.bias"] = [xLinear1.bias.name]
      mapping["\(prefix.0).img_mlp.2.weight"] = [xOutProjection.weight.name]
      mapping["\(prefix.0).img_mlp.2.bias"] = [xOutProjection.bias.name]
    case .diffusers:
      mapping["\(prefix.1).attn.add_q_proj.weight"] = [contextToQueries.weight.name]
      mapping["\(prefix.1).attn.add_q_proj.bias"] = [contextToQueries.bias.name]
      mapping["\(prefix.1).attn.add_k_proj.weight"] = [contextToKeys.weight.name]
      mapping["\(prefix.1).attn.add_k_proj.bias"] = [contextToKeys.bias.name]
      mapping["\(prefix.1).attn.add_v_proj.weight"] = [contextToValues.weight.name]
      mapping["\(prefix.1).attn.add_v_proj.bias"] = [contextToValues.bias.name]
      mapping["\(prefix.1).attn.norm_added_k.weight"] = [normAddedK.weight.name]
      mapping["\(prefix.1).attn.norm_added_q.weight"] = [normAddedQ.weight.name]
      mapping["\(prefix.1).attn.to_q.weight"] = [xToQueries.weight.name]
      mapping["\(prefix.1).attn.to_q.bias"] = [xToQueries.bias.name]
      mapping["\(prefix.1).attn.to_k.weight"] = [xToKeys.weight.name]
      mapping["\(prefix.1).attn.to_k.bias"] = [xToKeys.bias.name]
      mapping["\(prefix.1).attn.to_v.weight"] = [xToValues.weight.name]
      mapping["\(prefix.1).attn.to_v.bias"] = [xToValues.bias.name]
      mapping["\(prefix.1).attn.norm_k.weight"] = [normK.weight.name]
      mapping["\(prefix.1).attn.norm_q.weight"] = [normQ.weight.name]
      if let contextUnifyheads = contextUnifyheads {
        mapping["\(prefix.1).attn.to_add_out.weight"] = [contextUnifyheads.weight.name]
        mapping["\(prefix.1).attn.to_add_out.bias"] = [contextUnifyheads.bias.name]
      }
      mapping["\(prefix.1).attn.to_out.0.weight"] = [xUnifyheads.weight.name]
      mapping["\(prefix.1).attn.to_out.0.bias"] = [xUnifyheads.bias.name]
      if let contextLinear1 = contextLinear1,
        let contextOutProjection = contextOutProjection
      {
        mapping["\(prefix.1).ff_context.net.0.proj.weight"] = [contextLinear1.weight.name]
        mapping["\(prefix.1).ff_context.net.0.proj.bias"] = [contextLinear1.bias.name]
        mapping["\(prefix.1).ff_context.net.2.weight"] = [contextOutProjection.weight.name]
        mapping["\(prefix.1).ff_context.net.2.bias"] = [contextOutProjection.bias.name]
      }
      mapping["\(prefix.1).ff.net.0.proj.weight"] = [xLinear1.weight.name]
      mapping["\(prefix.1).ff.net.0.proj.bias"] = [xLinear1.bias.name]
      mapping["\(prefix.1).ff.net.2.weight"] = [xOutProjection.weight.name]
      mapping["\(prefix.1).ff.net.2.bias"] = [xOutProjection.bias.name]
    }
    return mapping
  }
  if !contextBlockPreOnly {
    return (mapper, Model([context, x, rot] + contextChunks + xChunks, [contextOut, xOut]))
  } else {
    return (mapper, Model([context, x, rot] + contextChunks + xChunks, [xOut]))
  }
}

private func SingleTransformerBlock(
  prefix: (String, String), k: Int, h: Int, b: Int, t: Int, hw: Int, contextBlockPreOnly: Bool,
  usesFlashAtttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let rot = Input()
  let xChunks = (0..<3).map { _ in Input() }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = xChunks[1] .* xNorm1(x).to(.Float16) + xChunks[0]
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
    let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
    out = scaledDotProductAttention(queries, keys, values).reshaped([b, (t + hw), k * h])
  case .scaleMerged:
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1.0 / Float(k).squareRoot(), flags: [.Float16])
    out = scaledDotProductAttention(queries, keys, values).reshaped([b, (t + hw), k * h])
  }
  var xIn: Model.IO = x
  if contextBlockPreOnly {
    out = out.reshaped([b, hw, h * k], offset: [0, t, 0], strides: [(t + hw) * h * k, h * k, 1])
    xIn = x.reshaped([b, hw, h * k], offset: [0, t, 0], strides: [(t + hw) * h * k, h * k, 1])
      .contiguous()
    xOut = xOut.reshaped(
      [b, hw, h * k], offset: [0, t, 0], strides: [(t + hw) * h * k, h * k, 1]
    )
  }
  let xUnifyheads = Dense(count: k * h, noBias: true, name: "x_o")
  let xLinear1 = Dense(count: k * h * 4, name: "x_linear1")
  let xOutProjection = Dense(count: k * h, flags: [.Float32], name: "x_out_proj")
  out = xUnifyheads(out) + xOutProjection(xLinear1(xOut).GELU(approximate: .tanh))
  out = xIn + (xChunks[2] .* out).to(of: xIn)
  let mapper: ModelWeightMapper = { format in
    var mapping: ModelWeightMapping = [:]
    switch format {
    case .generativeModels:
      mapping["\(prefix.0).linear1.weight"] = ModelWeightElement(
        [
          xToQueries.weight.name, xToKeys.weight.name, xToValues.weight.name, xLinear1.weight.name,
        ], offsets: [0, k * h, k * h * 2, k * h * 3])
      mapping["\(prefix.0).linear1.bias"] = ModelWeightElement(
        [
          xToQueries.bias.name, xToKeys.bias.name, xToValues.bias.name, xLinear1.bias.name,
        ], offsets: [0, k * h, k * h * 2, k * h * 3])
      mapping["\(prefix.0).norm.key_norm.scale"] = [normK.weight.name]
      mapping["\(prefix.0).norm.query_norm.scale"] = [normQ.weight.name]
      mapping["\(prefix.0).linear2.weight"] = ModelWeightElement(
        [xUnifyheads.weight.name, xOutProjection.weight.name], format: .I, offsets: [0, k * h])
      mapping["\(prefix.0).linear2.bias"] = [xOutProjection.bias.name]
    case .diffusers:
      mapping["\(prefix.1).attn.to_q.weight"] = [xToQueries.weight.name]
      mapping["\(prefix.1).attn.to_q.bias"] = [xToQueries.bias.name]
      mapping["\(prefix.1).attn.to_k.weight"] = [xToKeys.weight.name]
      mapping["\(prefix.1).attn.to_k.bias"] = [xToKeys.bias.name]
      mapping["\(prefix.1).attn.to_v.weight"] = [xToValues.weight.name]
      mapping["\(prefix.1).attn.to_v.bias"] = [xToValues.bias.name]
      mapping["\(prefix.1).proj_mlp.weight"] = [xLinear1.weight.name]
      mapping["\(prefix.1).proj_mlp.bias"] = [xLinear1.bias.name]
      mapping["\(prefix.1).attn.norm_k.weight"] = [normK.weight.name]
      mapping["\(prefix.1).attn.norm_q.weight"] = [normQ.weight.name]
      mapping["\(prefix.1).proj_out.weight"] = ModelWeightElement(
        [xUnifyheads.weight.name, xOutProjection.weight.name], format: .I, offsets: [0, k * h])
      mapping["\(prefix.1).proj_out.bias"] = [xOutProjection.bias.name]
    }
    return mapping
  }
  return (mapper, Model([x, rot] + xChunks, [out]))
}

public func Flux1(
  batchSize: Int, tokenLength: Int, height: Int, width: Int, channels: Int, layers: (Int, Int),
  usesFlashAttention: FlashAttentionLevel, injectControls: Bool
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let contextIn = Input()
  let rot = Input()
  let h = height / 2
  let w = width / 2
  let xEmbedder = Convolution(
    groups: 1, filters: channels, filterSize: [2, 2],
    hint: Hint(stride: [2, 2]), format: .OIHW, name: "x_embedder")
  var out = xEmbedder(x).reshaped([batchSize, h * w, channels]).to(.Float32)
  var adaLNChunks = [Input]()
  var injectedControls = [Input]()
  var mappers = [ModelWeightMapper]()
  var context = contextIn.to(.Float32)
  for i in 0..<layers.0 {
    let contextChunks = (0..<6).map { _ in Input() }
    let xChunks = (0..<6).map { _ in Input() }
    let (mapper, block) = JointTransformerBlock(
      prefix: ("double_blocks.\(i)", "transformer_blocks.\(i)"), k: 128, h: channels / 128,
      b: batchSize, t: tokenLength,
      hw: h * w, contextBlockPreOnly: false, upcast: i > (layers.0 - 3),
      usesFlashAtttention: usesFlashAttention
    )
    let blockOut = block([context, out, rot] + contextChunks + xChunks)
    context = blockOut[0]
    out = blockOut[1]
    if injectControls {
      let injectedControl = Input()
      let injectedControlFP32 = injectedControl.to(.Float32)
      injectedControlFP32.add(dependencies: [blockOut])  // out has no associated nodes, use blockOut instead.
      let scaleFactor: Float = 8
      out = out + (injectedControlFP32 * scaleFactor)
      injectedControls.append(injectedControl)
    }
    adaLNChunks.append(contentsOf: contextChunks + xChunks)
    mappers.append(mapper)
  }
  out = Functional.concat(axis: 1, context, out)
  for i in 0..<layers.1 {
    let xChunks = (0..<3).map { _ in Input() }
    let (mapper, block) = SingleTransformerBlock(
      prefix: ("single_blocks.\(i)", "single_transformer_blocks.\(i)"), k: 128, h: channels / 128,
      b: batchSize, t: tokenLength,
      hw: h * w, contextBlockPreOnly: i == layers.1 - 1, usesFlashAtttention: usesFlashAttention)
    out = block([out, rot] + xChunks)
    if injectControls {
      let injectedControl = Input()
      let injectedControlFP32 = injectedControl.to(.Float32)
      injectedControlFP32.add(dependencies: [out])
      let scaleFactor: Float = 8
      if i == layers.1 - 1 {
        out = out + (injectedControlFP32 * scaleFactor)
      } else {
        let encoderHiddenStates = out.reshaped(
          [batchSize, tokenLength, channels], offset: [0, 0, 0],
          strides: [(tokenLength + h * w) * channels, channels, 1])
        var hiddenStates = out.reshaped(
          [batchSize, h * w, channels], offset: [0, tokenLength, 0],
          strides: [(tokenLength + h * w) * channels, channels, 1])
        hiddenStates = hiddenStates + (injectedControlFP32 * scaleFactor)
        out = Functional.concat(axis: 1, encoderHiddenStates, hiddenStates)
      }
      injectedControls.append(injectedControl)
    }
    adaLNChunks.append(contentsOf: xChunks)
    mappers.append(mapper)
  }
  let shift = Input()
  let scale = Input()
  adaLNChunks.append(contentsOf: [shift, scale])
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  out = scale .* normFinal(out).to(.Float16) + shift
  let projOut = Dense(count: 2 * 2 * 16, name: "linear")
  out = projOut(out)
  // Unpatchify
  out = out.reshaped([batchSize, h, w, 16, 2, 2]).permuted(0, 1, 4, 2, 5, 3).contiguous().reshaped([
    batchSize, h * 2, w * 2, 16,
  ])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    switch format {
    case .generativeModels:
      mapping["img_in.weight"] = [xEmbedder.weight.name]
      mapping["img_in.bias"] = [xEmbedder.bias.name]
      mapping["final_layer.linear.weight"] = [projOut.weight.name]
      mapping["final_layer.linear.bias"] = [projOut.bias.name]
    case .diffusers:
      mapping["x_embedder.weight"] = [xEmbedder.weight.name]
      mapping["x_embedder.bias"] = [xEmbedder.bias.name]
      mapping["proj_out.weight"] = [projOut.weight.name]
      mapping["proj_out.bias"] = [projOut.bias.name]
    }
    return mapping
  }
  return (mapper, Model([x, rot, contextIn] + adaLNChunks + injectedControls, [out]))
}

private func LoRAMLPEmbedder(channels: Int, configuration: LoRANetworkConfiguration, name: String)
  -> (Model, Model, Model)
{
  let x = Input()
  let fc0 = LoRADense(count: channels, configuration: configuration, name: "\(name)_embedder_0")
  var out = fc0(x).swish()
  let fc2 = LoRADense(count: channels, configuration: configuration, name: "\(name)_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

private func LoRAFeedForward(
  hiddenSize: Int, intermediateSize: Int, upcast: Bool, configuration: LoRANetworkConfiguration,
  index: Int, name: String
) -> (
  Model, Model, Model
) {
  let x = Input()
  let linear1 = LoRADense(
    count: intermediateSize, configuration: configuration, index: index, name: "\(name)_linear1")
  var out = linear1(x).GELU(approximate: .tanh)
  if upcast {
    let scaleFactor: Float = 8
    out = (1 / scaleFactor) * out
  }
  let outProjection = LoRADense(
    count: hiddenSize, configuration: configuration, flags: [.Float32], index: index,
    name: "\(name)_out_proj")
  out = outProjection(out)
  if upcast {
    let scaleFactor: Float = 8
    out = out.to(.Float32) * scaleFactor
  }
  return (linear1, outProjection, Model([x], [out]))
}

private func LoRAJointTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, t: Int, hw: Int, contextBlockPreOnly: Bool,
  upcast: Bool, usesFlashAtttention: FlashAttentionLevel, layerIndex: Int,
  configuration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let context = Input()
  let x = Input()
  let rot = Input()
  let contextChunks = (0..<(contextBlockPreOnly ? 2 : 6)).map { _ in
    Input()
  }
  let contextNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var contextOut = contextChunks[1] .* contextNorm1(context).to(.Float16) + contextChunks[0]
  let contextToKeys = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "c_k")
  let contextToQueries = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "c_q")
  let contextToValues = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "c_v")
  var contextK = contextToKeys(contextOut).reshaped([b, t, h, k])
  let normAddedK = RMSNorm(epsilon: 1e-6, axis: [3], name: "c_norm_k")
  contextK = normAddedK(contextK)
  var contextQ = contextToQueries(contextOut).reshaped([b, t, h, k])
  let normAddedQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "c_norm_q")
  contextQ = normAddedQ(contextQ)
  let contextV = contextToValues(contextOut).reshaped([b, t, h, k])
  let xChunks = (0..<6).map { _ in Input() }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = xChunks[1] .* xNorm1(x).to(.Float16) + xChunks[0]
  let xToKeys = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "x_k")
  let xToQueries = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "x_q")
  let xToValues = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "x_v")
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
    let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
    out = scaledDotProductAttention(queries, keys, values).reshaped([b, (t + hw), k * h])
  case .scaleMerged:
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1.0 / Float(k).squareRoot(), flags: [.Float16])
    out = scaledDotProductAttention(queries, keys, values).reshaped([b, (t + hw), k * h])
  }
  let contextUnifyheads: Model?
  if !contextBlockPreOnly {
    contextOut = out.reshaped([b, t, h * k], strides: [(t + hw) * h * k, h * k, 1])
    let unifyheads = LoRADense(
      count: k * h, configuration: configuration, index: layerIndex, name: "c_o")
    contextOut = unifyheads(contextOut)
    contextUnifyheads = unifyheads
  } else {
    contextUnifyheads = nil
  }
  xOut = out.reshaped([b, hw, h * k], offset: [0, t, 0], strides: [(t + hw) * h * k, h * k, 1])
  let xUnifyheads = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "x_o")
  xOut = xUnifyheads(xOut)
  if !contextBlockPreOnly {
    contextOut = context + (contextChunks[2] .* contextOut).to(of: context)
  }
  xOut = x + (xChunks[2] .* xOut).to(of: x)
  // Attentions are now. Now run MLP.
  let contextLinear1: Model?
  let contextOutProjection: Model?
  if !contextBlockPreOnly {
    let contextFF: Model
    (contextLinear1, contextOutProjection, contextFF) = LoRAFeedForward(
      hiddenSize: k * h, intermediateSize: k * h * 4, upcast: upcast, configuration: configuration,
      index: layerIndex, name: "c")
    let contextNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
    if upcast {
      contextOut = contextOut + contextChunks[5].to(of: contextOut)
        .* contextFF(contextNorm2(contextOut).to(.Float16) .* contextChunks[4] + contextChunks[3])
    } else {
      contextOut =
        contextOut
        + (contextChunks[5]
        .* contextFF(contextNorm2(contextOut).to(.Float16) .* contextChunks[4] + contextChunks[3]))
        .to(of: contextOut)
    }
  } else {
    contextLinear1 = nil
    contextOutProjection = nil
  }
  let (xLinear1, xOutProjection, xFF) = LoRAFeedForward(
    hiddenSize: k * h, intermediateSize: k * h * 4, upcast: upcast, configuration: configuration,
    index: layerIndex, name: "x")
  let xNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  if upcast {
    xOut = xOut + xChunks[5].to(of: xOut)
      .* xFF(xNorm2(xOut).to(.Float16) .* xChunks[4] + xChunks[3])
  } else {
    xOut =
      xOut + (xChunks[5] .* xFF(xNorm2(xOut).to(.Float16) .* xChunks[4] + xChunks[3])).to(of: xOut)
  }
  let mapper: ModelWeightMapper = { _ in
    var mapping: ModelWeightMapping = [:]
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
    return mapping
  }
  if !contextBlockPreOnly {
    return (mapper, Model([context, x, rot] + contextChunks + xChunks, [contextOut, xOut]))
  } else {
    return (mapper, Model([context, x, rot] + contextChunks + xChunks, [xOut]))
  }
}

private func LoRASingleTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, t: Int, hw: Int, contextBlockPreOnly: Bool,
  usesFlashAtttention: FlashAttentionLevel, layerIndex: Int, configuration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let rot = Input()
  let xChunks = (0..<3).map { _ in Input() }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = xChunks[1] .* xNorm1(x).to(.Float16) + xChunks[0]
  let xToKeys = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "x_k")
  let xToQueries = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "x_q")
  let xToValues = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "x_v")
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
    let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
    out = scaledDotProductAttention(queries, keys, values).reshaped([b, (t + hw), k * h])
  case .scaleMerged:
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1.0 / Float(k).squareRoot(), flags: [.Float16])
    out = scaledDotProductAttention(queries, keys, values).reshaped([b, (t + hw), k * h])
  }
  var xIn: Model.IO = x
  if contextBlockPreOnly {
    out = out.reshaped([b, hw, h * k], offset: [0, t, 0], strides: [(t + hw) * h * k, h * k, 1])
    xIn = x.reshaped([b, hw, h * k], offset: [0, t, 0], strides: [(t + hw) * h * k, h * k, 1])
      .contiguous()
    xOut = xOut.reshaped(
      [b, hw, h * k], offset: [0, t, 0], strides: [(t + hw) * h * k, h * k, 1]
    )
  }
  let xUnifyheads = LoRADense(
    count: k * h, configuration: configuration, noBias: true, index: layerIndex, name: "x_o")
  let xLinear1 = LoRADense(
    count: k * h * 4, configuration: configuration, index: layerIndex, name: "x_linear1")
  let xOutProjection = LoRADense(
    count: k * h, configuration: configuration, flags: [.Float32], index: layerIndex,
    name: "x_out_proj")
  out = xUnifyheads(out) + xOutProjection(xLinear1(xOut).GELU(approximate: .tanh))
  out = xIn + (xChunks[2] .* out).to(of: xIn)
  let mapper: ModelWeightMapper = { _ in
    var mapping: ModelWeightMapping = [:]
    mapping["\(prefix).linear1.weight"] = [
      xToQueries.weight.name, xToKeys.weight.name, xToValues.weight.name, xLinear1.weight.name,
    ]
    mapping["\(prefix).linear1.bias"] = [
      xToQueries.bias.name, xToKeys.bias.name, xToValues.bias.name, xLinear1.bias.name,
    ]
    mapping["\(prefix).norm.key_norm.scale"] = [normK.weight.name]
    mapping["\(prefix).norm.query_norm.scale"] = [normQ.weight.name]
    mapping["\(prefix).linear2.weight"] = ModelWeightElement(
      [xUnifyheads.weight.name, xOutProjection.weight.name], format: .I)
    mapping["\(prefix).linear2.bias"] = [xOutProjection.bias.name]
    return mapping
  }
  return (mapper, Model([x, rot] + xChunks, [out]))
}

public func LoRAFlux1(
  batchSize: Int, tokenLength: Int, height: Int, width: Int, channels: Int, layers: (Int, Int),
  usesFlashAttention: FlashAttentionLevel, injectControls: Bool,
  LoRAConfiguration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let contextIn = Input()
  let rot = Input()
  let h = height / 2
  let w = width / 2
  let xEmbedder = LoRAConvolution(
    groups: 1, filters: channels, filterSize: [2, 2], configuration: LoRAConfiguration,
    hint: Hint(stride: [2, 2]), format: .OIHW, name: "x_embedder")
  var out = xEmbedder(x).reshaped([batchSize, h * w, channels]).to(.Float32)
  var adaLNChunks = [Input]()
  var injectedControls = [Input]()
  var mappers = [ModelWeightMapper]()
  var context = contextIn.to(.Float32)
  for i in 0..<layers.0 {
    let contextChunks = (0..<6).map { _ in Input() }
    let xChunks = (0..<6).map { _ in Input() }
    let (mapper, block) = LoRAJointTransformerBlock(
      prefix: "double_blocks.\(i)", k: 128, h: channels / 128, b: batchSize, t: tokenLength,
      hw: h * w, contextBlockPreOnly: false, upcast: i > 16,
      usesFlashAtttention: usesFlashAttention, layerIndex: i, configuration: LoRAConfiguration
    )
    let blockOut = block([context, out, rot] + contextChunks + xChunks)
    context = blockOut[0]
    out = blockOut[1]
    if injectControls {
      let injectedControl = Input()
      let injectedControlFP32 = injectedControl.to(.Float32)
      injectedControlFP32.add(dependencies: [blockOut])  // out has no associated nodes, use blockOut instead.
      let scaleFactor: Float = 8
      out = out + (injectedControlFP32 * scaleFactor)
      injectedControls.append(injectedControl)
    }
    adaLNChunks.append(contentsOf: contextChunks + xChunks)
    mappers.append(mapper)
  }
  out = Functional.concat(axis: 1, context, out)
  for i in 0..<layers.1 {
    let xChunks = (0..<3).map { _ in Input() }
    let (mapper, block) = LoRASingleTransformerBlock(
      prefix: "single_blocks.\(i)", k: 128, h: channels / 128, b: batchSize, t: tokenLength,
      hw: h * w, contextBlockPreOnly: i == layers.1 - 1, usesFlashAtttention: usesFlashAttention,
      layerIndex: i + layers.0, configuration: LoRAConfiguration)
    out = block([out, rot] + xChunks)
    if injectControls {
      let injectedControl = Input()
      let injectedControlFP32 = injectedControl.to(.Float32)
      injectedControlFP32.add(dependencies: [out])
      let scaleFactor: Float = 8
      if i == layers.1 - 1 {
        out = out + (injectedControlFP32 * scaleFactor)
      } else {
        let encoderHiddenStates = out.reshaped(
          [batchSize, tokenLength, channels], offset: [0, 0, 0],
          strides: [(tokenLength + h * w) * channels, channels, 1])
        var hiddenStates = out.reshaped(
          [batchSize, h * w, channels], offset: [0, tokenLength, 0],
          strides: [(tokenLength + h * w) * channels, channels, 1])
        hiddenStates = hiddenStates + (injectedControlFP32 * scaleFactor)
        out = Functional.concat(axis: 1, encoderHiddenStates, hiddenStates)
      }
      injectedControls.append(injectedControl)
    }
    adaLNChunks.append(contentsOf: xChunks)
    mappers.append(mapper)
  }
  let shift = Input()
  let scale = Input()
  adaLNChunks.append(contentsOf: [shift, scale])
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  out = scale .* normFinal(out).to(.Float16) + shift
  let projOut = LoRADense(count: 2 * 2 * 16, configuration: LoRAConfiguration, name: "linear")
  out = projOut(out)
  // Unpatchify
  out = out.reshaped([batchSize, h, w, 16, 2, 2]).permuted(0, 1, 4, 2, 5, 3).contiguous().reshaped([
    batchSize, h * 2, w * 2, 16,
  ])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping["img_in.weight"] = [xEmbedder.weight.name]
    mapping["img_in.bias"] = [xEmbedder.bias.name]
    mapping["final_layer.linear.weight"] = [projOut.weight.name]
    mapping["final_layer.linear.bias"] = [projOut.bias.name]
    return mapping
  }
  return (mapper, Model([x, rot, contextIn] + adaLNChunks + injectedControls, [out]))
}

private func JointTransformerBlockFixed(
  prefix: (String, String), k: Int, h: Int, contextBlockPreOnly: Bool
) -> (ModelWeightMapper, Model) {
  let c = Input()
  let contextAdaLNs = (0..<(contextBlockPreOnly ? 2 : 6)).map {
    Dense(count: k * h, name: "context_ada_ln_\($0)")
  }
  var contextChunks = contextAdaLNs.map { $0(c) }
  contextChunks[1] = 1 + contextChunks[1]
  let xAdaLNs = (0..<6).map { Dense(count: k * h, name: "x_ada_ln_\($0)") }
  var xChunks = xAdaLNs.map { $0(c) }
  xChunks[1] = 1 + xChunks[1]
  if !contextBlockPreOnly {
    contextChunks[4] = 1 + contextChunks[4]
  }
  xChunks[4] = 1 + xChunks[4]
  let mapper: ModelWeightMapper = { format in
    var mapping: ModelWeightMapping = [:]
    switch format {
    case .generativeModels:
      mapping["\(prefix.0).txt_mod.lin.weight"] = ModelWeightElement(
        (0..<(contextBlockPreOnly ? 2 : 6)).map {
          contextAdaLNs[$0].weight.name
        })
      mapping["\(prefix.0).txt_mod.lin.bias"] = ModelWeightElement(
        (0..<(contextBlockPreOnly ? 2 : 6)).map {
          contextAdaLNs[$0].bias.name
        })
      mapping["\(prefix.0).img_mod.lin.weight"] = ModelWeightElement(
        (0..<6).map { xAdaLNs[$0].weight.name })
      mapping["\(prefix.0).img_mod.lin.bias"] = ModelWeightElement(
        (0..<6).map { xAdaLNs[$0].bias.name })
    case .diffusers:
      mapping["\(prefix.1).norm1_context.linear.weight"] = ModelWeightElement(
        (0..<(contextBlockPreOnly ? 2 : 6)).map {
          contextAdaLNs[$0].weight.name
        })
      mapping["\(prefix.1).norm1_context.linear.bias"] = ModelWeightElement(
        (0..<(contextBlockPreOnly ? 2 : 6)).map {
          contextAdaLNs[$0].bias.name
        })
      mapping["\(prefix.1).norm1.linear.weight"] = ModelWeightElement(
        (0..<6).map { xAdaLNs[$0].weight.name })
      mapping["\(prefix.1).norm1.linear.bias"] = ModelWeightElement(
        (0..<6).map { xAdaLNs[$0].bias.name })
    }
    return mapping
  }
  return (mapper, Model([c], contextChunks + xChunks))
}

private func SingleTransformerBlockFixed(
  prefix: (String, String), k: Int, h: Int
) -> (ModelWeightMapper, Model) {
  let c = Input()
  let xAdaLNs = (0..<3).map { Dense(count: k * h, name: "x_ada_ln_\($0)") }
  var xChunks = xAdaLNs.map { $0(c) }
  xChunks[1] = 1 + xChunks[1]
  let mapper: ModelWeightMapper = { format in
    var mapping: ModelWeightMapping = [:]
    switch format {
    case .generativeModels:
      mapping["\(prefix.0).modulation.lin.weight"] = ModelWeightElement(
        (0..<3).map { xAdaLNs[$0].weight.name })
      mapping["\(prefix.0).modulation.lin.bias"] = ModelWeightElement(
        (0..<3).map { xAdaLNs[$0].bias.name })
    case .diffusers:
      mapping["\(prefix.1).norm.linear.weight"] = ModelWeightElement(
        (0..<3).map { xAdaLNs[$0].weight.name })
      mapping["\(prefix.1).norm.linear.bias"] = ModelWeightElement(
        (0..<3).map { xAdaLNs[$0].bias.name })
    }
    return mapping
  }
  return (mapper, Model([c], xChunks))
}

public func Flux1Fixed(
  batchSize: (Int, Int), channels: Int, layers: (Int, Int),
  guidanceEmbed: Bool = false
) -> (ModelWeightMapper, Model) {
  let timestep = Input()
  let y = Input()
  let contextIn = Input()
  let guidance: Input?
  let (tMlp0, tMlp2, tEmbedder) = MLPEmbedder(channels: channels, name: "t")
  var vec = tEmbedder(timestep)
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
  var outs = [Model.IO]()
  let contextEmbedder = Dense(count: channels, name: "context_embedder")
  let context = contextEmbedder(contextIn)
  outs.append(context)
  let c = vec.reshaped([batchSize.1, 1, channels]).swish()
  var mappers = [ModelWeightMapper]()
  for i in 0..<layers.0 {
    let (mapper, block) = JointTransformerBlockFixed(
      prefix: ("double_blocks.\(i)", "transformer_blocks.\(i)"), k: 128, h: channels / 128,
      contextBlockPreOnly: false)
    let blockOut = block(c)
    mappers.append(mapper)
    outs.append(blockOut)
  }
  for i in 0..<layers.1 {
    let (mapper, block) = SingleTransformerBlockFixed(
      prefix: ("single_blocks.\(i)", "single_transformer_blocks.\(i)"), k: 128, h: channels / 128)
    let blockOut = block(c)
    mappers.append(mapper)
    outs.append(blockOut)
  }
  let scale = Dense(count: channels, name: "ada_ln_0")
  let shift = Dense(count: channels, name: "ada_ln_1")
  outs.append(contentsOf: [shift(c), 1 + scale(c)])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    switch format {
    case .generativeModels:
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
    case .diffusers:
      mapping["time_text_embed.timestep_embedder.linear_1.weight"] = [tMlp0.weight.name]
      mapping["time_text_embed.timestep_embedder.linear_1.bias"] = [tMlp0.bias.name]
      mapping["time_text_embed.timestep_embedder.linear_2.weight"] = [tMlp2.weight.name]
      mapping["time_text_embed.timestep_embedder.linear_2.bias"] = [tMlp2.bias.name]
      if let gMlp0 = gMlp0, let gMlp2 = gMlp2 {
        mapping["time_text_embed.guidance_embedder.linear_1.weight"] = [gMlp0.weight.name]
        mapping["time_text_embed.guidance_embedder.linear_1.bias"] = [gMlp0.bias.name]
        mapping["time_text_embed.guidance_embedder.linear_2.weight"] = [gMlp2.weight.name]
        mapping["time_text_embed.guidance_embedder.linear_2.bias"] = [gMlp2.bias.name]
      }
      mapping["time_text_embed.text_embedder.linear_1.weight"] = [yMlp0.weight.name]
      mapping["time_text_embed.text_embedder.linear_1.bias"] = [yMlp0.bias.name]
      mapping["time_text_embed.text_embedder.linear_2.weight"] = [yMlp2.weight.name]
      mapping["time_text_embed.text_embedder.linear_2.bias"] = [yMlp2.bias.name]
      mapping["context_embedder.weight"] = [contextEmbedder.weight.name]
      mapping["context_embedder.bias"] = [contextEmbedder.bias.name]
      mapping["norm_out.linear.weight"] = [shift.weight.name, scale.weight.name]
      mapping["norm_out.linear.bias"] = [shift.bias.name, scale.bias.name]
    }
    return mapping
  }
  return (mapper, Model([contextIn, timestep, y] + (guidance.map { [$0] } ?? []), outs))
}

private func JointTransformerBlockFixedOutputShapes(
  prefix: String, batchSize: Int, k: Int, h: Int, contextBlockPreOnly: Bool
) -> [TensorShape] {
  let contextOutputShapes = (0..<(contextBlockPreOnly ? 2 : 6)).map { _ in
    TensorShape([batchSize, 1, k * h])
  }
  let xOutputShapes = (0..<6).map { _ in TensorShape([batchSize, 1, k * h]) }
  return contextOutputShapes + xOutputShapes
}

private func SingleTransformerBlockFixedOutputShapes(
  prefix: String, batchSize: Int, k: Int, h: Int
) -> [TensorShape] {
  let xOutputShapes = (0..<3).map { _ in TensorShape([batchSize, 1, k * h]) }
  return xOutputShapes
}

public func Flux1FixedOutputShapes(
  batchSize: (Int, Int), channels: Int, layers: (Int, Int),
  guidanceEmbed: Bool = false
) -> [TensorShape] {
  var outs = [TensorShape]()
  outs.append(TensorShape([batchSize.0, 256, channels]))
  for i in 0..<layers.0 {
    let outputShapes = JointTransformerBlockFixedOutputShapes(
      prefix: "double_blocks.\(i)", batchSize: batchSize.1, k: 128, h: channels / 128,
      contextBlockPreOnly: false)
    outs.append(contentsOf: outputShapes)
  }
  for i in 0..<layers.1 {
    let outputShapes = SingleTransformerBlockFixedOutputShapes(
      prefix: "single_blocks.\(i)", batchSize: batchSize.1, k: 128, h: channels / 128)
    outs.append(contentsOf: outputShapes)
  }
  outs.append(contentsOf: [
    TensorShape([batchSize.1, 1, channels]), TensorShape([batchSize.1, 1, channels]),
  ])
  return outs
}

private func LoRAJointTransformerBlockFixed(
  prefix: String, k: Int, h: Int, contextBlockPreOnly: Bool, layerIndex: Int,
  configuration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let c = Input()
  let contextAdaLNs = (0..<(contextBlockPreOnly ? 2 : 6)).map {
    LoRADense(
      count: k * h, configuration: configuration, index: layerIndex, name: "context_ada_ln_\($0)")
  }
  var contextChunks = contextAdaLNs.map { $0(c) }
  contextChunks[1] = 1 + contextChunks[1]
  let xAdaLNs = (0..<6).map {
    LoRADense(count: k * h, configuration: configuration, index: layerIndex, name: "x_ada_ln_\($0)")
  }
  var xChunks = xAdaLNs.map { $0(c) }
  xChunks[1] = 1 + xChunks[1]
  if !contextBlockPreOnly {
    contextChunks[4] = 1 + contextChunks[4]
  }
  xChunks[4] = 1 + xChunks[4]
  let mapper: ModelWeightMapper = { _ in
    var mapping: ModelWeightMapping = [:]
    mapping["\(prefix).txt_mod.lin.weight"] = ModelWeightElement(
      (0..<(contextBlockPreOnly ? 2 : 6)).map {
        contextAdaLNs[$0].weight.name
      })
    mapping["\(prefix).txt_mod.lin.bias"] = ModelWeightElement(
      (0..<(contextBlockPreOnly ? 2 : 6)).map {
        contextAdaLNs[$0].bias.name
      })
    mapping["\(prefix).img_mod.lin.weight"] = ModelWeightElement(
      (0..<6).map { xAdaLNs[$0].weight.name })
    mapping["\(prefix).img_mod.lin.bias"] = ModelWeightElement(
      (0..<6).map { xAdaLNs[$0].bias.name })
    return mapping
  }
  return (mapper, Model([c], contextChunks + xChunks))
}

private func LoRASingleTransformerBlockFixed(
  prefix: String, k: Int, h: Int, layerIndex: Int, configuration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let c = Input()
  let xAdaLNs = (0..<3).map {
    LoRADense(count: k * h, configuration: configuration, index: layerIndex, name: "x_ada_ln_\($0)")
  }
  var xChunks = xAdaLNs.map { $0(c) }
  xChunks[1] = 1 + xChunks[1]
  let mapper: ModelWeightMapper = { _ in
    var mapping: ModelWeightMapping = [:]
    mapping["\(prefix).modulation.lin.weight"] = ModelWeightElement(
      (0..<3).map { xAdaLNs[$0].weight.name })
    mapping["\(prefix).modulation.lin.bias"] = ModelWeightElement(
      (0..<3).map { xAdaLNs[$0].bias.name })
    return mapping
  }
  return (mapper, Model([c], xChunks))
}

public func LoRAFlux1Fixed(
  batchSize: (Int, Int), channels: Int, layers: (Int, Int),
  LoRAConfiguration: LoRANetworkConfiguration,
  guidanceEmbed: Bool = false
) -> (ModelWeightMapper, Model) {
  let timestep = Input()
  let y = Input()
  let contextIn = Input()
  let guidance: Input?
  let (tMlp0, tMlp2, tEmbedder) = LoRAMLPEmbedder(
    channels: channels, configuration: LoRAConfiguration, name: "t")
  var vec = tEmbedder(timestep)
  let gMlp0: Model?
  let gMlp2: Model?
  if guidanceEmbed {
    let (mlp0, mlp2, gEmbedder) = LoRAMLPEmbedder(
      channels: channels, configuration: LoRAConfiguration, name: "guidance")
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
  let (yMlp0, yMlp2, yEmbedder) = LoRAMLPEmbedder(
    channels: channels, configuration: LoRAConfiguration, name: "vector")
  vec = vec + yEmbedder(y)
  var outs = [Model.IO]()
  let contextEmbedder = LoRADense(
    count: channels, configuration: LoRAConfiguration, name: "context_embedder")
  let context = contextEmbedder(contextIn)
  outs.append(context)
  let c = vec.reshaped([batchSize.1, 1, channels]).swish()
  var mappers = [ModelWeightMapper]()
  for i in 0..<layers.0 {
    let (mapper, block) = LoRAJointTransformerBlockFixed(
      prefix: "double_blocks.\(i)", k: 128, h: channels / 128,
      contextBlockPreOnly: false, layerIndex: i, configuration: LoRAConfiguration)
    let blockOut = block(c)
    mappers.append(mapper)
    outs.append(blockOut)
  }
  for i in 0..<layers.1 {
    let (mapper, block) = LoRASingleTransformerBlockFixed(
      prefix: "single_blocks.\(i)", k: 128, h: channels / 128, layerIndex: i + layers.0,
      configuration: LoRAConfiguration)
    let blockOut = block(c)
    mappers.append(mapper)
    outs.append(blockOut)
  }
  let scale = LoRADense(count: channels, configuration: LoRAConfiguration, name: "ada_ln_0")
  let shift = LoRADense(count: channels, configuration: LoRAConfiguration, name: "ada_ln_1")
  outs.append(contentsOf: [shift(c), 1 + scale(c)])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
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
    return mapping
  }
  return (mapper, Model([contextIn, timestep, y] + (guidance.map { [$0] } ?? []), outs))
}

public func ControlNetFlux1(
  union: Bool,
  batchSize: Int, tokenLength: Int, height: Int, width: Int, channels: Int, layers: (Int, Int),
  usesFlashAttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  let tokenLength = union ? tokenLength + 1 : tokenLength
  let x = Input()
  let controlnetX = Input()
  let contextIn = Input()
  let rot = Input()
  let h = height / 2
  let w = width / 2
  let xEmbedder = Convolution(
    groups: 1, filters: channels, filterSize: [2, 2],
    hint: Hint(stride: [2, 2]), format: .OIHW, name: "x_embedder")
  // Doing this here such that when we do tiled diffusion, we deal with smaller input even though there are some redundant compute.
  let controlnetXEmbedder = Convolution(
    groups: 1, filters: channels, filterSize: [2, 2],
    hint: Hint(stride: [2, 2]), format: .OIHW, name: "controlnet_x_embedder")
  var out =
    xEmbedder(x).reshaped([batchSize, h * w, channels]).to(.Float32)
    + controlnetXEmbedder(controlnetX).reshaped([batchSize, h * w, channels]).to(.Float32)
  var adaLNChunks = [Input]()
  var mappers = [ModelWeightMapper]()
  var context = contextIn.to(.Float32)
  var zeroConvs = [Dense]()
  var outs = [Model.IO]()
  for i in 0..<layers.0 {
    let contextChunks = (0..<6).map { _ in Input() }
    let xChunks = (0..<6).map { _ in Input() }
    let (mapper, block) = JointTransformerBlock(
      prefix: ("double_blocks.\(i)", "transformer_blocks.\(i)"), k: 128, h: channels / 128,
      b: batchSize, t: tokenLength,
      hw: h * w, contextBlockPreOnly: false, upcast: i > (layers.0 - 3),
      usesFlashAtttention: usesFlashAttention
    )
    let blockOut = block([context, out, rot] + contextChunks + xChunks)
    context = blockOut[0]
    out = blockOut[1]
    adaLNChunks.append(contentsOf: contextChunks + xChunks)
    mappers.append(mapper)
    let zeroConv = Dense(count: channels, name: "zero_conv")
    if let last = outs.last {
      blockOut.add(dependencies: [last])  // out has no associated nodes, use blockOut instead.
    }
    let scaleFactor: Float = 1 / 8  // We already scaled bias for zero conv.
    outs.append(zeroConv((out * scaleFactor).to(of: controlnetX)))
    zeroConvs.append(zeroConv)
  }
  out = Functional.concat(axis: 1, context, out)
  for i in 0..<layers.1 {
    let xChunks = (0..<3).map { _ in Input() }
    let (mapper, block) = SingleTransformerBlock(
      prefix: ("single_blocks.\(i)", "single_transformer_blocks.\(i)"), k: 128, h: channels / 128,
      b: batchSize, t: tokenLength,
      hw: h * w, contextBlockPreOnly: i == layers.1 - 1, usesFlashAtttention: usesFlashAttention)
    out = block([out, rot] + xChunks)
    adaLNChunks.append(contentsOf: xChunks)
    mappers.append(mapper)
    let zeroConv = Dense(count: channels, name: "zero_conv")
    if let last = outs.last {
      out.add(dependencies: [last])
    }
    let scaleFactor: Float = 1 / 8
    if i == layers.1 - 1 {
      outs.append(zeroConv((out * scaleFactor).to(of: controlnetX)))
    } else {
      outs.append(
        zeroConv(
          (out.reshaped(
            [batchSize, h * w, channels], offset: [0, tokenLength, 0],
            strides: [(tokenLength + h * w) * channels, channels, 1]) * scaleFactor).to(
              of: controlnetX)))
    }
    zeroConvs.append(zeroConv)
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    switch format {
    case .generativeModels:
      mapping["img_in.weight"] = [xEmbedder.weight.name]
      mapping["img_in.bias"] = [xEmbedder.bias.name]
    case .diffusers:
      mapping["x_embedder.weight"] = [xEmbedder.weight.name]
      mapping["x_embedder.bias"] = [xEmbedder.bias.name]
      mapping["controlnet_x_embedder.weight"] = [controlnetXEmbedder.weight.name]
      mapping["controlnet_x_embedder.bias"] = [controlnetXEmbedder.bias.name]
      for i in 0..<layers.0 {
        mapping["controlnet_blocks.\(i).weight"] = [zeroConvs[i].weight.name]
        mapping["controlnet_blocks.\(i).bias"] = [zeroConvs[i].bias.name]
      }
      for i in 0..<layers.1 {
        mapping["controlnet_single_blocks.\(i).weight"] = [zeroConvs[i + layers.0].weight.name]
        mapping["controlnet_single_blocks.\(i).bias"] = [zeroConvs[i + layers.0].bias.name]
      }
    }
    return mapping
  }
  return (mapper, Model([x, controlnetX, rot, contextIn] + adaLNChunks, outs))
}

public func ControlNetFlux1Fixed<FloatType: TensorNumeric & BinaryFloatingPoint>(
  union: Bool,
  batchSize: (Int, Int), channels: Int, layers: (Int, Int), guidanceEmbed: Bool = false,
  of: FloatType.Type = FloatType.self
) -> (ModelWeightMapper, Model) {
  let timestep = Input()
  let y = Input()
  let contextIn = Input()
  let guidance: Input?
  let (tMlp0, tMlp2, tEmbedder) = MLPEmbedder(channels: channels, name: "t")
  var vec = tEmbedder(timestep)
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
  var outs = [Model.IO]()
  let contextEmbedder = Dense(count: channels, name: "context_embedder")
  var context = contextEmbedder(contextIn)
  let controlnetMode: Input?
  let controlnetModeEmbedder: Embedding?
  if union {
    let modeEmbedder = Embedding(
      FloatType.self, vocabularySize: 10, embeddingSize: channels, name: "controlnet_mode_embedder")
    let mode = Input()
    context = Functional.concat(axis: 1, modeEmbedder(mode).reshaped([1, 1, channels]), context)
    controlnetMode = mode
    controlnetModeEmbedder = modeEmbedder
  } else {
    controlnetMode = nil
    controlnetModeEmbedder = nil
  }
  outs.append(context)
  let c = vec.reshaped([batchSize.1, 1, channels]).swish()
  var mappers = [ModelWeightMapper]()
  for i in 0..<layers.0 {
    let (mapper, block) = JointTransformerBlockFixed(
      prefix: ("double_blocks.\(i)", "transformer_blocks.\(i)"), k: 128, h: channels / 128,
      contextBlockPreOnly: false)
    let blockOut = block(c)
    mappers.append(mapper)
    outs.append(blockOut)
  }
  for i in 0..<layers.1 {
    let (mapper, block) = SingleTransformerBlockFixed(
      prefix: ("single_blocks.\(i)", "single_transformer_blocks.\(i)"), k: 128, h: channels / 128)
    let blockOut = block(c)
    mappers.append(mapper)
    outs.append(blockOut)
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    switch format {
    case .generativeModels:
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
    case .diffusers:
      mapping["time_text_embed.timestep_embedder.linear_1.weight"] = [tMlp0.weight.name]
      mapping["time_text_embed.timestep_embedder.linear_1.bias"] = [tMlp0.bias.name]
      mapping["time_text_embed.timestep_embedder.linear_2.weight"] = [tMlp2.weight.name]
      mapping["time_text_embed.timestep_embedder.linear_2.bias"] = [tMlp2.bias.name]
      if let gMlp0 = gMlp0, let gMlp2 = gMlp2 {
        mapping["time_text_embed.guidance_embedder.linear_1.weight"] = [gMlp0.weight.name]
        mapping["time_text_embed.guidance_embedder.linear_1.bias"] = [gMlp0.bias.name]
        mapping["time_text_embed.guidance_embedder.linear_2.weight"] = [gMlp2.weight.name]
        mapping["time_text_embed.guidance_embedder.linear_2.bias"] = [gMlp2.bias.name]
      }
      mapping["time_text_embed.text_embedder.linear_1.weight"] = [yMlp0.weight.name]
      mapping["time_text_embed.text_embedder.linear_1.bias"] = [yMlp0.bias.name]
      mapping["time_text_embed.text_embedder.linear_2.weight"] = [yMlp2.weight.name]
      mapping["time_text_embed.text_embedder.linear_2.bias"] = [yMlp2.bias.name]
      mapping["context_embedder.weight"] = [contextEmbedder.weight.name]
      mapping["context_embedder.bias"] = [contextEmbedder.bias.name]
      if let controlnetModeEmbedder = controlnetModeEmbedder {
        mapping["controlnet_mode_embedder.weight"] = [controlnetModeEmbedder.weight.name]
        mapping["controlnet_mode_embedder.bias"] = [controlnetModeEmbedder.bias.name]
      }
    }
    return mapping
  }
  return (
    mapper,
    Model(
      (controlnetMode.map { [$0] } ?? []) + [contextIn, timestep, y]
        + (guidance.map { [$0] } ?? []), outs)
  )
}
