import Foundation
import NNC

public func QwenImageRotaryPositionEmbedding(
  height: Int, width: Int, tokenLength: Int, referenceSizes: [(height: Int, width: Int)],
  channels: Int, heads: Int = 1
)
  -> Tensor<Float>
{
  var rotTensor = Tensor<Float>(
    .CPU,
    .NHWC(
      1, height * width + tokenLength + referenceSizes.reduce(0) { $0 + $1.height * $1.width },
      heads, channels))
  let dim0 = channels / 8
  let dim1 = channels * 7 / 16
  let dim2 = dim1
  assert(channels % 16 == 0)
  let maxImgIdx = max(height / 2, width / 2)
  let imageLength = height * width + referenceSizes.reduce(0) { $0 + $1.height * $1.width }
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
  var index = width * height
  var h = 0
  var w = 0
  for referenceSize in referenceSizes {
    let height = referenceSize.height
    let width = referenceSize.width
    var hOffset = 0
    var wOffset = 0
    if height + h > width + w {
      wOffset = w
    } else {
      hOffset = h
    }
    for y in 0..<height {
      for x in 0..<width {
        let i = y * width + x + index
        for j in 0..<heads {
          for k in 0..<(dim0 / 2) {
            let theta = 1 * 1.0 / pow(10_000, Double(k) * 2 / Double(dim0))  // Use time index at 1.
            let sintheta = sin(theta)
            let costheta = cos(theta)
            rotTensor[0, i, j, k * 2] = Float(costheta)
            rotTensor[0, i, j, k * 2 + 1] = Float(sintheta)
          }
          for k in 0..<(dim1 / 2) {
            let theta = Double(y + hOffset) * 1.0 / pow(10_000, Double(k) * 2 / Double(dim1))
            let sintheta = sin(theta)
            let costheta = cos(theta)
            rotTensor[0, i, j, (k + (dim0 / 2)) * 2] = Float(costheta)
            rotTensor[0, i, j, (k + (dim0 / 2)) * 2 + 1] = Float(sintheta)
          }
          for k in 0..<(dim2 / 2) {
            let theta = Double(x + wOffset) * 1.0 / pow(10_000, Double(k) * 2 / Double(dim2))
            let sintheta = sin(theta)
            let costheta = cos(theta)
            rotTensor[0, i, j, (k + (dim0 / 2) + (dim1 / 2)) * 2] = Float(costheta)
            rotTensor[0, i, j, (k + (dim0 / 2) + (dim1 / 2)) * 2 + 1] = Float(sintheta)
          }
        }
      }
    }
    index += height * width
    h = max(h, height + hOffset)
    w = max(w, width + wOffset)
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

private func FeedForward(
  hiddenSize: Int, intermediateSize: Int, scaleFactor: Float, isBF16: Bool, name: String
)
  -> (
    Model, Model, Model
  )
{
  let x = Input()
  let linear1 = Dense(count: intermediateSize, flags: [.Float16], name: "\(name)_linear1")
  var out = linear1(x).GELU(approximate: .tanh)
  if isBF16 {
    out = out.to(.BFloat16)
  } else {
    out = (1.0 / scaleFactor) * out
  }
  // The scale down is integrated into out proj bias.
  let outProjection = Dense(count: hiddenSize, flags: [.Float32], name: "\(name)_out_proj")
  if isBF16 {
    out = outProjection(out).to(.Float32)
  } else {
    out = scaleFactor * outProjection(out).to(.Float32)
  }
  return (linear1, outProjection, Model([x], [out]))
}

private func JointTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, t: Int, hw: Int, referenceSequenceLength: Int,
  contextBlockPreOnly: Bool, usesFlashAttention: FlashAttentionLevel, scaleFactor: (Float, Float),
  isBF16: Bool
) -> (ModelWeightMapper, Model) {
  let context = Input()
  let x = Input()
  let rot = Input()
  let contextChunks = (0..<(contextBlockPreOnly ? 2 : 6)).map { _ in
    Input()
  }
  let xChunks = (0..<6).map { _ in Input() }
  let contextNorm1 = LayerNorm(
    epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var contextOut = contextChunks[1] .* contextNorm1(context) + contextChunks[0]
  let contextToKeys = Dense(count: k * h, name: "c_k")
  let contextToQueries = Dense(count: k * h, flags: [.Float16], name: "c_q")
  let contextToValues = Dense(count: k * h, name: "c_v")
  let downcastContextOut: Model.IO
  if isBF16 {
    downcastContextOut = ((1.0 / 8) * contextOut).to(.Float16)  // scale down factor not merged. values path doesn't use the scale factor.
  } else {
    downcastContextOut = contextOut.to(.Float16)  // scale down factor merged into contextChunks.
  }
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
  let contextV = contextToValues(isBF16 ? contextOut.to(.BFloat16) : downcastContextOut).reshaped([
    b, t, h, k,
  ])
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = xNorm1(x)
  xOut = xChunks[1] .* xOut + xChunks[0]
  let xToKeys = Dense(count: k * h, name: "x_k")
  let xToQueries = Dense(count: k * h, flags: [.Float16], name: "x_q")
  let xToValues = Dense(count: k * h, name: "x_v")
  let downcastXOut: Model.IO
  if isBF16 {
    downcastXOut = ((1.0 / 8) * xOut).to(.Float16)  // scale down factor not merged. Values path doesn't use the scale factor.
  } else {
    downcastXOut = xOut.to(.Float16)  // scale down factor merged into xChunks.
  }
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
  let xV = xToValues(isBF16 ? xOut.to(.BFloat16) : downcastXOut).reshaped([b, hw, h, k])
  var keys = Functional.concat(axis: 1, xK, contextK)
  var values = Functional.concat(axis: 1, xV, contextV)
  var queries = Functional.concat(axis: 1, xQ, contextQ)
  queries = Functional.cmul(left: queries, right: rot)
  keys = Functional.cmul(left: keys, right: rot)
  if isBF16 {
    queries = queries.to(.BFloat16)
    keys = keys.to(.BFloat16)
  }
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
    contextOut = unifyheads(isBF16 ? contextOut : (1.0 / scaleFactor.0) * contextOut).to(
      of: context)  // scale up factor merged into contextChunks.
    contextUnifyheads = unifyheads
  } else {
    contextUnifyheads = nil
  }
  let xIn: Model.IO
  if contextBlockPreOnly, referenceSequenceLength > 0 {
    xIn = x.reshaped([b, hw - referenceSequenceLength, h * k], strides: [hw * h * k, h * k, 1])
      .contiguous()
    xOut = out.reshaped(
      [b, hw - referenceSequenceLength, h * k], strides: [(t + hw) * h * k, h * k, 1]
    )
    .contiguous()
  } else {
    xIn = x
    xOut = out.reshaped([b, hw, h * k], strides: [(t + hw) * h * k, h * k, 1])
      .contiguous()
  }
  let xUnifyheads = Dense(count: k * h, name: "x_o")
  xOut = xUnifyheads(isBF16 ? xOut : (1.0 / scaleFactor.0) * xOut).to(of: x)  // scale up factor merged into xChunks.
  if !contextBlockPreOnly {
    contextOut = context + contextChunks[2] .* contextOut
  }
  xOut = xIn + xChunks[2] .* xOut
  // Attentions are now. Now run MLP.
  let contextLinear1: Model?
  let contextOutProjection: Model?
  if !contextBlockPreOnly {
    let contextFF: Model
    (contextLinear1, contextOutProjection, contextFF) = FeedForward(
      hiddenSize: k * h, intermediateSize: k * h * 4, scaleFactor: scaleFactor.1, isBF16: isBF16,
      name: "c")
    let contextNorm2 = LayerNorm(
      epsilon: 1e-6, axis: [2], elementwiseAffine: false)
    contextOut =
      contextOut
      + contextChunks[5]
      .* contextFF(
        (contextNorm2(contextOut) .* contextChunks[4] + contextChunks[3]).to(.Float16)
      ).to(
        of: contextOut)
  } else {
    contextLinear1 = nil
    contextOutProjection = nil
  }
  let (xLinear1, xOutProjection, xFF) = FeedForward(
    hiddenSize: k * h, intermediateSize: k * h * 4, scaleFactor: scaleFactor.1, isBF16: isBF16,
    name: "x")
  let xNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  xOut =
    xOut
    + xChunks[5]
    .* xFF((xNorm2(xOut) .* xChunks[4] + xChunks[3]).to(.Float16))
    .to(of: xOut)
  let mapper: ModelWeightMapper = { _ in
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
    return mapping
  }
  if !contextBlockPreOnly {
    return (mapper, Model([x, context, rot] + contextChunks + xChunks, [xOut, contextOut]))
  } else {
    return (mapper, Model([x, context, rot] + contextChunks + xChunks, [xOut]))
  }
}

public func QwenImage(
  batchSize: Int, height: Int, width: Int, textLength: Int, referenceSequenceLength: Int,
  channels: Int, layers: Int,
  usesFlashAttention: FlashAttentionLevel, isBF16: Bool, activationFfnScaling: [Int: Int]
) -> (
  ModelWeightMapper, Model
) {
  let x = Input()
  let rot = Input()
  let contextIn = Input()
  var adaLNChunks = [Input]()
  let imgIn = Convolution(
    groups: 1, filters: channels, filterSize: [2, 2],
    hint: Hint(stride: [2, 2]), format: .OIHW, name: "x_embedder")
  let referenceLatents: Input?
  let h = height / 2
  let w = width / 2
  var out: Model.IO
  if referenceSequenceLength > 0 {
    let latents = Input()
    out = Functional.concat(
      axis: 1, imgIn(x).reshaped([batchSize, h * w, channels]), latents, flags: [.disableOpt]
    ).to(.Float32)
    referenceLatents = latents
  } else {
    out = imgIn(x).reshaped(.HWC(batchSize, h * w, channels)).to(.Float32)
    referenceLatents = nil
  }
  var mappers = [ModelWeightMapper]()
  var context = contextIn.to(.Float32)
  let rotResized = rot.reshaped(.NHWC(1, h * w + referenceSequenceLength + textLength, 1, 128))
  for i in 0..<layers {
    let contextBlockPreOnly = i == layers - 1
    let contextChunks = (0..<(contextBlockPreOnly ? 2 : 6)).map { _ in Input() }
    let xChunks = (0..<6).map { _ in Input() }
    let ffnScaling = activationFfnScaling[i] ?? 1
    let (mapper, block) = JointTransformerBlock(
      prefix: "transformer_blocks.\(i)", k: 128, h: channels / 128, b: batchSize, t: textLength,
      hw: h * w + referenceSequenceLength, referenceSequenceLength: referenceSequenceLength,
      contextBlockPreOnly: contextBlockPreOnly, usesFlashAttention: usesFlashAttention,
      scaleFactor: (
        i >= layers - 16 ? 16 : 2,
        i >= layers - 1 ? Float(256 * ffnScaling) : Float(16 * ffnScaling)
      ), isBF16: isBF16)
    let blockOut = block([out, context, rotResized] + contextChunks + xChunks)
    if i == layers - 1 {
      out = blockOut
    } else {
      out = blockOut[0]
      context = blockOut[1]
    }
    adaLNChunks.append(contentsOf: contextChunks + xChunks)
    mappers.append(mapper)
  }
  let shift = Input()
  let scale = Input()
  adaLNChunks.append(contentsOf: [shift, scale])
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  out = scale .* normFinal(out).to(.Float16) + shift
  let projOut = Dense(count: 2 * 2 * 16, name: "linear")
  out = projOut(out).reshaped([batchSize, h, w, 16, 2, 2]).permuted(0, 1, 4, 2, 5, 3).contiguous()
    .reshaped([
      batchSize, h * 2, w * 2, 16,
    ])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["img_in.weight"] = [imgIn.weight.name]
    mapping["img_in.bias"] = [imgIn.bias.name]
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping["proj_out.weight"] = [projOut.weight.name]
    mapping["proj_out.bias"] = [projOut.bias.name]
    return mapping
  }
  return (
    mapper,
    Model([x, rot] + (referenceLatents.map { [$0] } ?? []) + [contextIn] + adaLNChunks, [out])
  )
}

private func JointTransformerBlockFixed(
  prefix: String, k: Int, h: Int, contextBlockPreOnly: Bool, scaleFactor: (Float, Float),
  isBF16: Bool
) -> (ModelWeightMapper, Model) {
  let c = Input()
  let contextAdaLNs = (0..<(contextBlockPreOnly ? 2 : 6)).map {
    Dense(count: k * h, name: "context_ada_ln_\($0)")
  }
  var contextChunks = contextAdaLNs.map { $0(c) }
  let xAdaLNs = (0..<6).map { Dense(count: k * h, name: "x_ada_ln_\($0)") }
  var xChunks = xAdaLNs.map { $0(c) }
  if isBF16 {
    contextChunks = contextChunks.map { $0.to(.Float32) }
    contextChunks[1] = 1 + contextChunks[1]
    xChunks = xChunks.map { $0.to(.Float32) }
    xChunks[1] = 1 + xChunks[1]
    if !contextBlockPreOnly {
      contextChunks[4] = 1 + contextChunks[4]
    }
    xChunks[4] = 1 + xChunks[4]
  } else {
    // Merge scale factor into the adaLN.
    contextChunks[0] = (1.0 / 8) * contextChunks[0].to(.Float32)
    contextChunks[1] = (1.0 / 8) * (1 + contextChunks[1].to(.Float32))
    xChunks[0] = (1.0 / 8) * xChunks[0].to(.Float32)
    xChunks[1] = (1.0 / 8) * (1 + xChunks[1].to(.Float32))
    xChunks[2] = (8 * scaleFactor.0) * xChunks[2].to(.Float32)
    if !contextBlockPreOnly {
      contextChunks[2] = (8 * scaleFactor.0) * contextChunks[2].to(.Float32)
      contextChunks[3] = contextChunks[3].to(.Float32)
      contextChunks[4] = 1 + contextChunks[4].to(.Float32)
      contextChunks[5] = contextChunks[5].to(.Float32)
    }
    xChunks[3] = xChunks[3].to(.Float32)
    xChunks[4] = 1 + xChunks[4].to(.Float32)
    xChunks[5] = xChunks[5].to(.Float32)
  }
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
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
  return (mapper, Model([c], contextChunks + xChunks))
}

public func QwenImageFixed(
  timesteps: Int, channels: Int, layers: Int, isBF16: Bool, activationFfnScaling: [Int: Int],
  numberOfReferenceImages: Int
) -> (
  ModelWeightMapper, Model
) {
  let txt = Input()
  var outs = [Model.IO]()
  var referenceImages = [Input]()
  if numberOfReferenceImages > 0 {
    let imgIn = Convolution(
      groups: 1, filters: channels, filterSize: [2, 2],
      hint: Hint(stride: [2, 2]), format: .OIHW, name: "x_embedder")
    for _ in 0..<numberOfReferenceImages {
      let x = Input()
      let out = imgIn(x)
      referenceImages.append(x)
      outs.append(out)
    }
  }
  let txtNorm = RMSNorm(epsilon: 1e-6, axis: [2], name: "context_norm")
  let txtIn = Dense(count: channels, name: "context_embedder")
  let context = txtIn(txtNorm(txt))
  outs.append(context)
  var mappers = [ModelWeightMapper]()
  let t: Input?
  let timeInMlp0: Model?
  let timeInMlp2: Model?
  let scale: Model?
  let shift: Model?
  if layers > 0 {
    let tIn = Input()
    let (inMlp0, inMlp2, timeIn) = MLPEmbedder(channels: channels, name: "t")
    var vec = timeIn(tIn)
    vec = vec.reshaped([timesteps, 1, channels]).swish()
    for i in 0..<layers {
      let ffnScaling = activationFfnScaling[i] ?? 1
      let (mapper, block) = JointTransformerBlockFixed(
        prefix: "transformer_blocks.\(i)", k: 128, h: channels / 128,
        contextBlockPreOnly: i == layers - 1,
        scaleFactor: (
          i >= layers - 16 ? 16 : 2,
          i >= layers - 1 ? Float(256 * ffnScaling) : Float(16 * ffnScaling)
        ), isBF16: isBF16)
      let blockOut = block(vec)
      mappers.append(mapper)
      outs.append(blockOut)
    }
    let scaleIn = Dense(count: channels, name: "ada_ln_0")
    let shiftIn = Dense(count: channels, name: "ada_ln_1")
    outs.append(contentsOf: [shiftIn(vec), 1 + scaleIn(vec)])
    t = tIn
    timeInMlp0 = inMlp0
    timeInMlp2 = inMlp2
    scale = scaleIn
    shift = shiftIn
  } else {
    t = nil
    timeInMlp0 = nil
    timeInMlp2 = nil
    scale = nil
    shift = nil
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["txt_norm.weight"] = [txtNorm.weight.name]
    mapping["txt_in.weight"] = [txtIn.weight.name]
    mapping["txt_in.bias"] = [txtIn.bias.name]
    if let timeInMlp0 = timeInMlp0 {
      mapping[
        "time_text_embed.timestep_embedder.linear_1.weight"
      ] = [timeInMlp0.weight.name]
      mapping[
        "time_text_embed.timestep_embedder.linear_1.bias"
      ] = [timeInMlp0.bias.name]
    }
    if let timeInMlp2 = timeInMlp2 {
      mapping[
        "time_text_embed.timestep_embedder.linear_2.weight"
      ] = [timeInMlp2.weight.name]
      mapping[
        "time_text_embed.timestep_embedder.linear_2.bias"
      ] = [timeInMlp2.bias.name]
    }
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    if let scale = scale, let shift = shift {
      mapping["norm_out.linear.weight"] = [scale.weight.name, shift.weight.name]
      mapping["norm_out.linear.bias"] = [scale.bias.name, shift.bias.name]
    }
    return mapping
  }
  return (mapper, Model([txt] + referenceImages + (t.map { [$0] } ?? []), outs))
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

public func QwenImageFixedOutputShapes(
  batchSize: Int, textLength: Int, channels: Int, layers: Int
) -> [TensorShape] {
  var outs = [TensorShape]()
  outs.append(TensorShape([batchSize, textLength, channels]))
  for i in 0..<layers {
    let contextBlockPreOnly = i == layers - 1
    let outputShapes = JointTransformerBlockFixedOutputShapes(
      prefix: "transformer_blocks.\(i)", batchSize: batchSize, k: 128, h: channels / 128,
      contextBlockPreOnly: contextBlockPreOnly)
    outs.append(contentsOf: outputShapes)
  }
  outs.append(contentsOf: [
    TensorShape([batchSize, 1, channels]), TensorShape([batchSize, 1, channels]),
  ])
  return outs
}

private func LoRAMLPEmbedder(
  channels: Int, configuration: LoRANetworkConfiguration, index: Int, name: String
) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = LoRADense(
    count: channels, configuration: configuration, index: index, name: "\(name)_embedder_0")
  var out = fc0(x).swish()
  let fc2 = LoRADense(
    count: channels, configuration: configuration, index: index, name: "\(name)_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

private func LoRAFeedForward(
  hiddenSize: Int, intermediateSize: Int, scaleFactor: Float, isBF16: Bool,
  configuration: LoRANetworkConfiguration, index: Int, name: String
)
  -> (
    Model, Model, Model
  )
{
  let x = Input()
  let linear1 = LoRADense(
    count: intermediateSize, configuration: configuration, flags: [.Float16], index: index,
    name: "\(name)_linear1")
  var out = linear1(x).GELU(approximate: .tanh)
  if isBF16 {
    out = out.to(.BFloat16)
  } else {
    out = (1.0 / scaleFactor) * out
  }
  // The scale down is integrated into out proj bias.
  let outProjection = LoRADense(
    count: hiddenSize, configuration: configuration, flags: [.Float32], index: index,
    name: "\(name)_out_proj")
  if isBF16 {
    out = outProjection(out).to(.Float32)
  } else {
    out = scaleFactor * outProjection(out).to(.Float32)
  }
  return (linear1, outProjection, Model([x], [out]))
}

private func LoRAJointTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, t: Int, hw: Int, referenceSequenceLength: Int,
  contextBlockPreOnly: Bool, usesFlashAttention: FlashAttentionLevel, scaleFactor: (Float, Float),
  isBF16: Bool, layerIndex: Int, configuration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let context = Input()
  let x = Input()
  let rot = Input()
  let contextChunks = (0..<(contextBlockPreOnly ? 2 : 6)).map { _ in
    Input()
  }
  let xChunks = (0..<6).map { _ in Input() }
  let contextNorm1 = LayerNorm(
    epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var contextOut = contextChunks[1] .* contextNorm1(context) + contextChunks[0]
  let contextToKeys = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "c_k")
  let contextToQueries = LoRADense(
    count: k * h, configuration: configuration, flags: [.Float16], index: layerIndex, name: "c_q")
  let contextToValues = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "c_v")
  let downcastContextOut: Model.IO
  if isBF16 {
    downcastContextOut = ((1.0 / 8) * contextOut).to(.Float16)  // scale down factor not merged. Values path doesn't use the scale factor.
  } else {
    downcastContextOut = contextOut.to(.Float16)  // scale down factor merged into contextChunks.
  }
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
  let contextV = contextToValues(isBF16 ? contextOut.to(.BFloat16) : downcastContextOut).reshaped([
    b, t, h, k,
  ])
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = xNorm1(x)
  xOut = xChunks[1] .* xOut + xChunks[0]
  let xToKeys = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "x_k")
  let xToQueries = LoRADense(
    count: k * h, configuration: configuration, flags: [.Float16], index: layerIndex, name: "x_q")
  let xToValues = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "x_v")
  let downcastXOut: Model.IO
  if isBF16 {
    downcastXOut = ((1.0 / 8) * xOut).to(.Float16)  // scale down factor not merged. Values path doesn't use the scale factor.
  } else {
    downcastXOut = xOut.to(.Float16)  // scale down factor merged into xChunks.
  }
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
  let xV = xToValues(isBF16 ? xOut.to(.BFloat16) : downcastXOut).reshaped([b, hw, h, k])
  var keys = Functional.concat(axis: 1, xK, contextK)
  var values = Functional.concat(axis: 1, xV, contextV)
  var queries = Functional.concat(axis: 1, xQ, contextQ)
  queries = Functional.cmul(left: queries, right: rot)
  keys = Functional.cmul(left: keys, right: rot)
  if isBF16 {
    queries = queries.to(.BFloat16)
    keys = keys.to(.BFloat16)
  }
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
    let unifyheads = LoRADense(
      count: k * h, configuration: configuration, index: layerIndex, name: "c_o")
    contextOut = unifyheads(isBF16 ? contextOut : (1.0 / scaleFactor.0) * contextOut).to(
      of: context)  // scale up factor merged into contextChunks.
    contextUnifyheads = unifyheads
  } else {
    contextUnifyheads = nil
  }
  let xIn: Model.IO
  if contextBlockPreOnly, referenceSequenceLength > 0 {
    xIn = x.reshaped([b, hw - referenceSequenceLength, h * k], strides: [hw * h * k, h * k, 1])
      .contiguous()
    xOut = out.reshaped(
      [b, hw - referenceSequenceLength, h * k], strides: [(t + hw) * h * k, h * k, 1]
    )
    .contiguous()
  } else {
    xIn = x
    xOut = out.reshaped([b, hw, h * k], strides: [(t + hw) * h * k, h * k, 1])
      .contiguous()
  }
  let xUnifyheads = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "x_o")
  xOut = xUnifyheads(isBF16 ? xOut : (1.0 / scaleFactor.0) * xOut).to(of: x)  // scale up factor merged into xChunks.
  if !contextBlockPreOnly {
    contextOut = context + contextChunks[2] .* contextOut
  }
  xOut = xIn + xChunks[2] .* xOut
  // Attentions are now. Now run MLP.
  let contextLinear1: Model?
  let contextOutProjection: Model?
  if !contextBlockPreOnly {
    let contextFF: Model
    (contextLinear1, contextOutProjection, contextFF) = LoRAFeedForward(
      hiddenSize: k * h, intermediateSize: k * h * 4, scaleFactor: scaleFactor.1,
      isBF16: isBF16, configuration: configuration, index: layerIndex, name: "c")
    let contextNorm2 = LayerNorm(
      epsilon: 1e-6, axis: [2], elementwiseAffine: false)
    contextOut =
      contextOut
      + contextChunks[5]
      .* contextFF(
        (contextNorm2(contextOut) .* contextChunks[4] + contextChunks[3]).to(.Float16)
      ).to(
        of: contextOut)
  } else {
    contextLinear1 = nil
    contextOutProjection = nil
  }
  let (xLinear1, xOutProjection, xFF) = LoRAFeedForward(
    hiddenSize: k * h, intermediateSize: k * h * 4, scaleFactor: scaleFactor.1,
    isBF16: isBF16, configuration: configuration, index: layerIndex, name: "x")
  let xNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  xOut =
    xOut
    + xChunks[5]
    .* xFF((xNorm2(xOut) .* xChunks[4] + xChunks[3]).to(.Float16))
    .to(of: xOut)
  let mapper: ModelWeightMapper = { _ in
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
    return mapping
  }
  if !contextBlockPreOnly {
    return (mapper, Model([x, context, rot] + contextChunks + xChunks, [xOut, contextOut]))
  } else {
    return (mapper, Model([x, context, rot] + contextChunks + xChunks, [xOut]))
  }
}

public func LoRAQwenImage(
  batchSize: Int, height: Int, width: Int, textLength: Int, referenceSequenceLength: Int,
  channels: Int, layers: Int, usesFlashAttention: FlashAttentionLevel, isBF16: Bool,
  activationFfnScaling: [Int: Int], LoRAConfiguration: LoRANetworkConfiguration
) -> (
  ModelWeightMapper, Model
) {
  let x = Input()
  let rot = Input()
  let contextIn = Input()
  var adaLNChunks = [Input]()
  let imgIn = LoRAConvolution(
    groups: 1, filters: channels, filterSize: [2, 2], configuration: LoRAConfiguration,
    hint: Hint(stride: [2, 2]), format: .OIHW, name: "x_embedder")
  let referenceLatents: Input?
  let h = height / 2
  let w = width / 2
  var out: Model.IO
  if referenceSequenceLength > 0 {
    let latents = Input()
    out = Functional.concat(
      axis: 1, imgIn(x).reshaped([batchSize, h * w, channels]), latents, flags: [.disableOpt]
    ).to(.Float32)
    referenceLatents = latents
  } else {
    out = imgIn(x).reshaped(.HWC(batchSize, h * w, channels)).to(.Float32)
    referenceLatents = nil
  }
  var mappers = [ModelWeightMapper]()
  var context = contextIn.to(.Float32)
  let rotResized = rot.reshaped(.NHWC(1, h * w + referenceSequenceLength + textLength, 1, 128))
  for i in 0..<layers {
    let contextBlockPreOnly = i == layers - 1
    let contextChunks = (0..<(contextBlockPreOnly ? 2 : 6)).map { _ in Input() }
    let xChunks = (0..<6).map { _ in Input() }
    let ffnScaling = activationFfnScaling[i] ?? 1
    let (mapper, block) = LoRAJointTransformerBlock(
      prefix: "transformer_blocks.\(i)", k: 128, h: channels / 128, b: batchSize, t: textLength,
      hw: h * w + referenceSequenceLength, referenceSequenceLength: referenceSequenceLength,
      contextBlockPreOnly: contextBlockPreOnly, usesFlashAttention: usesFlashAttention,
      scaleFactor: (
        i >= layers - 16 ? 16 : 2,
        i >= layers - 1 ? Float(256 * ffnScaling) : Float(16 * ffnScaling)
      ), isBF16: isBF16, layerIndex: i, configuration: LoRAConfiguration)
    let blockOut = block([out, context, rotResized] + contextChunks + xChunks)
    if i == layers - 1 {
      out = blockOut
    } else {
      out = blockOut[0]
      context = blockOut[1]
    }
    adaLNChunks.append(contentsOf: contextChunks + xChunks)
    mappers.append(mapper)
  }
  let shift = Input()
  let scale = Input()
  adaLNChunks.append(contentsOf: [shift, scale])
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  out = scale .* normFinal(out).to(.Float16) + shift
  let projOut = LoRADense(
    count: 2 * 2 * 16, configuration: LoRAConfiguration, index: 0, name: "linear")
  out = projOut(out).reshaped([batchSize, h, w, 16, 2, 2]).permuted(0, 1, 4, 2, 5, 3).contiguous()
    .reshaped([
      batchSize, h * 2, w * 2, 16,
    ])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["img_in.weight"] = [imgIn.weight.name]
    mapping["img_in.bias"] = [imgIn.bias.name]
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping["proj_out.weight"] = [projOut.weight.name]
    mapping["proj_out.bias"] = [projOut.bias.name]
    return mapping
  }
  return (
    mapper,
    Model([x, rot] + (referenceLatents.map { [$0] } ?? []) + [contextIn] + adaLNChunks, [out])
  )
}

private func LoRAJointTransformerBlockFixed(
  prefix: String, k: Int, h: Int, contextBlockPreOnly: Bool, scaleFactor: (Float, Float),
  isBF16: Bool, layerIndex: Int, configuration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let c = Input()
  let contextAdaLNs = (0..<(contextBlockPreOnly ? 2 : 6)).map {
    LoRADense(
      count: k * h, configuration: configuration, index: layerIndex, name: "context_ada_ln_\($0)")
  }
  var contextChunks = contextAdaLNs.map { $0(c) }
  let xAdaLNs = (0..<6).map {
    LoRADense(count: k * h, configuration: configuration, index: layerIndex, name: "x_ada_ln_\($0)")
  }
  var xChunks = xAdaLNs.map { $0(c) }
  if isBF16 {
    contextChunks = contextChunks.map { $0.to(.Float32) }
    xChunks = xChunks.map { $0.to(.Float32) }
    contextChunks[1] = 1 + contextChunks[1]
    xChunks[1] = 1 + xChunks[1]
    if !contextBlockPreOnly {
      contextChunks[4] = 1 + contextChunks[4]
    }
    xChunks[4] = 1 + xChunks[4]
  } else {
    // Merge scale factor into the adaLN.
    contextChunks[0] = (1.0 / 8) * contextChunks[0].to(.Float32)
    contextChunks[1] = (1.0 / 8) * (1 + contextChunks[1].to(.Float32))
    xChunks[0] = (1.0 / 8) * xChunks[0].to(.Float32)
    xChunks[1] = (1.0 / 8) * (1 + xChunks[1].to(.Float32))
    xChunks[2] = (8 * scaleFactor.0) * xChunks[2].to(.Float32)
    if !contextBlockPreOnly {
      contextChunks[2] = (8 * scaleFactor.0) * contextChunks[2].to(.Float32)
      contextChunks[3] = contextChunks[3].to(.Float32)
      contextChunks[4] = 1 + contextChunks[4].to(.Float32)
      contextChunks[5] = contextChunks[5].to(.Float32)
    }
    xChunks[3] = xChunks[3].to(.Float32)
    xChunks[4] = 1 + xChunks[4].to(.Float32)
    xChunks[5] = xChunks[5].to(.Float32)
  }
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
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
  return (mapper, Model([c], contextChunks + xChunks))
}

public func LoRAQwenImageFixed(
  timesteps: Int, channels: Int, layers: Int, isBF16: Bool,
  activationFfnScaling: [Int: Int], numberOfReferenceImages: Int,
  LoRAConfiguration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let txt = Input()
  var outs = [Model.IO]()
  var referenceImages = [Input]()
  if numberOfReferenceImages > 0 {
    let imgIn = LoRAConvolution(
      groups: 1, filters: channels, filterSize: [2, 2], configuration: LoRAConfiguration,
      hint: Hint(stride: [2, 2]), format: .OIHW, name: "x_embedder")
    for _ in 0..<numberOfReferenceImages {
      let x = Input()
      let out = imgIn(x)
      referenceImages.append(x)
      outs.append(out)
    }
  }
  let txtNorm = RMSNorm(epsilon: 1e-6, axis: [2], name: "context_norm")
  let txtIn = LoRADense(
    count: channels, configuration: LoRAConfiguration, index: 0, name: "context_embedder")
  let context = txtIn(txtNorm(txt))
  outs.append(context)
  var mappers = [ModelWeightMapper]()
  let t: Input?
  let timeInMlp0: Model?
  let timeInMlp2: Model?
  let scale: Model?
  let shift: Model?
  if layers > 0 {
    let tIn = Input()
    let (inMlp0, inMlp2, timeIn) = LoRAMLPEmbedder(
      channels: channels, configuration: LoRAConfiguration, index: 0, name: "t")
    var vec = timeIn(tIn)
    vec = vec.reshaped([timesteps, 1, channels]).swish()
    for i in 0..<layers {
      let ffnScaling = activationFfnScaling[i] ?? 1
      let (mapper, block) = LoRAJointTransformerBlockFixed(
        prefix: "transformer_blocks.\(i)", k: 128, h: channels / 128,
        contextBlockPreOnly: i == layers - 1,
        scaleFactor: (
          i >= layers - 16 ? 16 : 2,
          i >= layers - 1 ? Float(256 * ffnScaling) : Float(16 * ffnScaling)
        ), isBF16: isBF16, layerIndex: i, configuration: LoRAConfiguration)
      let blockOut = block(vec)
      mappers.append(mapper)
      outs.append(blockOut)
    }
    let scaleIn = LoRADense(
      count: channels, configuration: LoRAConfiguration, index: 0, name: "ada_ln_0")
    let shiftIn = LoRADense(
      count: channels, configuration: LoRAConfiguration, index: 0, name: "ada_ln_1")
    outs.append(contentsOf: [shiftIn(vec), 1 + scaleIn(vec)])
    t = tIn
    timeInMlp0 = inMlp0
    timeInMlp2 = inMlp2
    scale = scaleIn
    shift = shiftIn
  } else {
    t = nil
    timeInMlp0 = nil
    timeInMlp2 = nil
    scale = nil
    shift = nil
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["txt_norm.weight"] = [txtNorm.weight.name]
    mapping["txt_in.weight"] = [txtIn.weight.name]
    mapping["txt_in.bias"] = [txtIn.bias.name]
    if let timeInMlp0 = timeInMlp0 {
      mapping[
        "time_text_embed.timestep_embedder.linear_1.weight"
      ] = [timeInMlp0.weight.name]
      mapping[
        "time_text_embed.timestep_embedder.linear_1.bias"
      ] = [timeInMlp0.bias.name]
    }
    if let timeInMlp2 = timeInMlp2 {
      mapping[
        "time_text_embed.timestep_embedder.linear_2.weight"
      ] = [timeInMlp2.weight.name]
      mapping[
        "time_text_embed.timestep_embedder.linear_2.bias"
      ] = [timeInMlp2.bias.name]
    }
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    if let scale = scale, let shift = shift {
      mapping["norm_out.linear.weight"] = [scale.weight.name, shift.weight.name]
      mapping["norm_out.linear.bias"] = [scale.bias.name, shift.bias.name]
    }
    return mapping
  }
  return (mapper, Model([txt] + referenceImages + (t.map { [$0] } ?? []), outs))
}
