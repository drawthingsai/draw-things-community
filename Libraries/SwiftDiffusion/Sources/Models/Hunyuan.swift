import Foundation
import NNC

func HunyuanRotaryPositionEmbedding(
  height: Int, width: Int, time: Int, tokenLength: Int, channels: Int, heads: Int = 1
)
  -> (Tensor<Float>, Tensor<Float>)
{
  var rotNdTensor0 = Tensor<Float>(.CPU, .NHWC(1, time * height * width, heads, channels))
  var rotNdTensor1 = Tensor<Float>(
    .CPU, .NHWC(1, time * height * width + tokenLength, heads, channels))
  let dim0 = channels / 8
  let dim1 = channels * 7 / 16
  let dim2 = dim1
  assert(channels % 16 == 0)
  for t in 0..<time {
    for y in 0..<height {
      for x in 0..<width {
        let i = t * height * width + y * width + x + tokenLength
        for j in 0..<heads {
          for k in 0..<(dim0 / 2) {
            let theta = 0 * 1.0 / pow(10_000, Double(k) * 2 / Double(dim0))
            let sintheta = sin(theta)
            let costheta = cos(theta)
            rotNdTensor0[0, i, j, k * 2] = Float(costheta)
            rotNdTensor0[0, i, j, k * 2 + 1] = Float(sintheta)
            rotNdTensor1[0, i, j, k * 2] = Float(costheta)
            rotNdTensor1[0, i, j, k * 2 + 1] = Float(sintheta)
          }
          for k in 0..<(dim1 / 2) {
            let theta = Double(y) * 1.0 / pow(10_000, Double(k) * 2 / Double(dim1))
            let sintheta = sin(theta)
            let costheta = cos(theta)
            rotNdTensor0[0, i, j, (k + (dim0 / 2)) * 2] = Float(costheta)
            rotNdTensor0[0, i, j, (k + (dim0 / 2)) * 2 + 1] = Float(sintheta)
            rotNdTensor1[0, i, j, (k + (dim0 / 2)) * 2] = Float(costheta)
            rotNdTensor1[0, i, j, (k + (dim0 / 2)) * 2 + 1] = Float(sintheta)
          }
          for k in 0..<(dim2 / 2) {
            let theta = Double(x) * 1.0 / pow(10_000, Double(k) * 2 / Double(dim2))
            let sintheta = sin(theta)
            let costheta = cos(theta)
            rotNdTensor0[0, i, j, (k + (dim0 / 2) + (dim1 / 2)) * 2] = Float(costheta)
            rotNdTensor0[0, i, j, (k + (dim0 / 2) + (dim1 / 2)) * 2 + 1] = Float(sintheta)
            rotNdTensor1[0, i, j, (k + (dim0 / 2) + (dim1 / 2)) * 2] = Float(costheta)
            rotNdTensor1[0, i, j, (k + (dim0 / 2) + (dim1 / 2)) * 2 + 1] = Float(sintheta)
          }
        }
      }
    }
  }
  for i in (time * height * width)..<(time * height * width + tokenLength) {
    for k in 0..<(channels / 2) {
      rotNdTensor1[0, i, 0, k * 2] = 1
      rotNdTensor1[0, i, 0, k * 2 + 1] = 0
    }
  }
  return (rotNdTensor0, rotNdTensor1)
}

private func MLPEmbedder(channels: Int, name: String) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: channels, name: "\(name)_embedder_0")
  var out = fc0(x).swish()
  let fc2 = Dense(count: channels, name: "\(name)_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

private func RefinerSelfAttention(prefix: String, k: Int, h: Int, hk: Int, b: Int, t: Int) -> (
  Model, ModelWeightMapper
) {
  let x = Input()
  let tokeys = Dense(count: k * hk, name: "refiner_k_proj")
  let toqueries = Dense(count: k * h, name: "refiner_q_proj")
  let tovalues = Dense(count: k * hk, name: "refiner_v_proj")
  let keys = tokeys(x).reshaped([b, t, hk, k])
  let queries = toqueries(x).reshaped([b, t, h, k])
  let values = tovalues(x).reshaped([b, t, hk, k])
  var out = ScaledDotProductAttention(scale: 1.0 / Float(k).squareRoot())(
    queries, keys, values
  ).reshaped([b, t, h * k])
  let unifyheads = Dense(count: k * h, name: "refiner_out_proj")
  out = unifyheads(out)
  let mapper: ModelWeightMapper = { _ in
    // The rotary in Llama is first half and second half, we can be clever and do the extra transpose here to use with cmul.
    return ModelWeightMapping()
  }
  return (Model([x], [out]), mapper)
}

private func IndividualRefinerBlock(prefix: String, t: Int) -> (Model, ModelWeightMapper) {
  let x = Input()
  let c = Input()
  let norm1 = LayerNorm(epsilon: 1e-6, axis: [2], name: "refiner_norm1")
  let gateMsa = Dense(count: 3_072, name: "refiner_ada_ln_msa")
  let (attention, attentionMapper) = RefinerSelfAttention(
    prefix: prefix, k: 128, h: 24, hk: 24, b: 1, t: t)
  var out = x + attention(norm1(x)) .* gateMsa(c)
  let norm2 = LayerNorm(epsilon: 1e-6, axis: [2], name: "refiner_norm2")
  let mlp0 = Dense(count: 3_072 * 4, name: "refiner_mlp_0")
  let mlp1 = Dense(count: 3_072, name: "refiner_mlp_1")
  let gateMlp = Dense(count: 3_072, name: "refiner_ada_ln_mlp")
  out = out + mlp1(mlp0(norm2(out)).swish()) .* gateMlp(c)
  let mapper: ModelWeightMapper = { format in
    return attentionMapper(format)
  }
  return (Model([x, c], [out]), mapper)
}

private func FeedForward(hiddenSize: Int, intermediateSize: Int, upcast: Bool, name: String) -> (
  Model, Model, Model
) {
  let x = Input()
  let linear1 = Dense(count: intermediateSize, name: "\(name)_linear1")
  var out = linear1(x).GELU(approximate: .tanh)
  // The scale down is integrated into out proj bias.
  if upcast {
    let scaleFactor: Float = 8
    out = (1 / scaleFactor) * out
  }
  let outProjection = Dense(count: hiddenSize, name: "\(name)_out_proj")
  out = outProjection(out)
  if upcast {
    let scaleFactor: Float = 8
    out = out.to(.Float32) * scaleFactor
  }
  return (linear1, outProjection, Model([x], [out]))
}

private func JointTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, t: Int, hw: Int, contextBlockPreOnly: Bool, upcast: Bool
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
  var contextOut = (1 + contextChunks[1]) .* contextNorm1(context).to(.Float16) + contextChunks[0]
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
  var xOut = (1 + xChunks[1]) .* xNorm1(x).to(.Float16) + xChunks[0]
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
  xQ = Functional.cmul(left: xQ, right: rot)
  xK = Functional.cmul(left: xK, right: rot)
  let keys = Functional.concat(axis: 1, xK, contextK)
  let values = Functional.concat(axis: 1, xV, contextV)
  let queries = Functional.concat(axis: 1, xQ, contextQ)
  // Now run attention.
  let out = ScaledDotProductAttention(scale: 1.0 / Float(k).squareRoot(), flags: [.Float16])(
    queries, keys, values
  ).reshaped([b, t + hw, h * k])
  let contextUnifyheads: Model?
  if !contextBlockPreOnly {
    contextOut = out.reshaped(
      [b, t, h * k], offset: [0, hw, 0], strides: [(t + hw) * h * k, h * k, 1]
    ).contiguous()
    let unifyheads = Dense(count: k * h, name: "c_o")
    contextOut = unifyheads(contextOut)
    contextUnifyheads = unifyheads
  } else {
    contextUnifyheads = nil
  }
  xOut = out.reshaped([b, hw, h * k], strides: [(t + hw) * h * k, h * k, 1]).contiguous()
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
    contextOut =
      contextOut
      + ((upcast ? contextChunks[5].to(of: contextOut) : contextChunks[5])
      .* contextFF(
        contextNorm2(contextOut).to(.Float16) .* (1 + contextChunks[4]) + contextChunks[3])).to(
        of: contextOut)
  } else {
    contextLinear1 = nil
    contextOutProjection = nil
  }
  let (xLinear1, xOutProjection, xFF) = FeedForward(
    hiddenSize: k * h, intermediateSize: k * h * 4, upcast: upcast, name: "x")
  let xNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  xOut =
    xOut
    + ((upcast ? xChunks[5].to(of: xOut) : xChunks[5])
    .* xFF(xNorm2(xOut).to(.Float16) .* (1 + xChunks[4]) + xChunks[3])).to(of: xOut)
  let mapper: ModelWeightMapper = { _ in
    return ModelWeightMapping()
  }
  if !contextBlockPreOnly {
    return (mapper, Model([x, context, c, rot], [xOut, contextOut]))
  } else {
    return (mapper, Model([x, context, c, rot], [xOut]))
  }
}

private func SingleTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, t: Int, hw: Int, contextBlockPreOnly: Bool
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let c = Input()
  let rot = Input()
  let xAdaLNs = (0..<3).map { Dense(count: k * h, name: "x_ada_ln_\($0)") }
  let xChunks = xAdaLNs.map { $0(c) }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = (1 + xChunks[1]) .* xNorm1(x).to(.Float16) + xChunks[0]
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
  let queries = Functional.cmul(left: xQ, right: rot)
  let keys = Functional.cmul(left: xK, right: rot)
  let values = xV
  // Now run attention.
  let scaledDotProductAttention = ScaledDotProductAttention(
    scale: 1.0 / Float(k).squareRoot(), flags: [.Float16])
  var out = scaledDotProductAttention(queries, keys, values).reshaped([b, (t + hw), k * h])
  var xIn: Model.IO = x
  if contextBlockPreOnly {
    out = out.reshaped([b, hw, h * k], strides: [(t + hw) * h * k, h * k, 1])
      .contiguous()
    xIn = x.reshaped([b, hw, h * k], strides: [(t + hw) * h * k, h * k, 1])
      .contiguous()
    xOut = xOut.reshaped(
      [b, hw, h * k], strides: [(t + hw) * h * k, h * k, 1]
    )
    .contiguous()
  }
  let xUnifyheads = Dense(count: k * h, noBias: true, name: "x_o")
  let xLinear1 = Dense(count: k * h * 4, name: "x_linear1")
  let xOutProjection = Dense(count: k * h, name: "x_out_proj")
  out = xUnifyheads(out) + xOutProjection(xLinear1(xOut).GELU(approximate: .tanh))
  out = xIn + xChunks[2].to(of: xIn) .* out.to(of: xIn)
  let mapper: ModelWeightMapper = { _ in
    return ModelWeightMapping()
  }
  return (mapper, Model([x, c, rot], [out]))
}

func Hunyuan(time: Int, height: Int, width: Int, textLength: Int) -> (ModelWeightMapper, Model) {
  let x = Input()
  let rot = Input()
  let imgIn = Dense(count: 3072, name: "x_embedder")
  let txt = Input()
  let t = Input()
  let vector = Input()
  let guidanceEmbed = Input()
  let (tMlp0, tMlp2, timeEmbedder) = MLPEmbedder(channels: 3_072, name: "txt_in_t")
  var c = txt.reduced(.mean, axis: [1])
  let (cLinear1, cLinear2, contextEmbedder) = MLPEmbedder(channels: 3_072, name: "c")
  c = timeEmbedder(t) + contextEmbedder(c)
  c = c.reshaped([1, 1, 3072]).swish()
  let inputEmbedder = Dense(count: 3_072, name: "input_embedder")
  var context = inputEmbedder(txt)
  var mappers = [ModelWeightMapper]()
  for i in 0..<2 {
    let (block, mapper) = IndividualRefinerBlock(
      prefix: "txt_in.individual_token_refiner.blocks.\(i)", t: textLength)
    context = block(context, c)
    mappers.append(mapper)
  }
  context = context.to(.Float32)
  var out = imgIn(x).to(.Float32)
  let (timeInMlp0, timeInMlp2, timeIn) = MLPEmbedder(channels: 3_072, name: "t")
  let (vMlp0, vMlp2, vectorIn) = MLPEmbedder(channels: 3_072, name: "vector")
  let (gMlp0, gMlp2, guidanceIn) = MLPEmbedder(channels: 3_072, name: "guidance")
  var vec = timeIn(t) + vectorIn(vector) + guidanceIn(guidanceEmbed)
  vec = vec.reshaped([1, 1, 3072]).swish()
  let h = height / 2
  let w = width / 2
  for i in 0..<20 {
    let (mapper, block) = JointTransformerBlock(
      prefix: "double_blocks.\(i)", k: 128, h: 24, b: 1, t: textLength, hw: time * h * w,
      contextBlockPreOnly: false, upcast: true)
    let blockOut = block(out, context, vec, rot)
    out = blockOut[0]
    context = blockOut[1]
    mappers.append(mapper)
  }
  let rot2 = Input()
  out = Functional.concat(axis: 1, out, context)
  for i in 0..<40 {
    let (mapper, block) = SingleTransformerBlock(
      prefix: "single_blocks.\(i)", k: 128, h: 24, b: 1, t: textLength, hw: time * h * w,
      contextBlockPreOnly: i == 39)
    out = block(out, vec, rot2)
    mappers.append(mapper)
  }
  let scale = Dense(count: 3072, name: "ada_ln_0")
  let shift = Dense(count: 3072, name: "ada_ln_1")
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  out = (1 + scale(vec)) .* normFinal(out).to(.Float16) + shift(vec)
  let projOut = Dense(count: 2 * 2 * 16, name: "linear")
  out = projOut(out)
  let mapper: ModelWeightMapper = { _ in
    return ModelWeightMapping()
  }
  return (mapper, Model([x, rot, rot2, txt, t, vector, guidanceEmbed], [out]))
}
