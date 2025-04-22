import Foundation
import NNC

public func HiDreamRotaryPositionEmbedding(
  height: Int, width: Int, tokenLength: Int, channels: Int, heads: Int = 1
)
  -> Tensor<Float>
{
  var rotTensor = Tensor<Float>(.CPU, .NHWC(1, tokenLength + height * width, heads, channels))
  let dim1 = (channels / 8) * 2
  let dim2 = dim1
  let dim0 = channels - dim1 - dim2
  assert(channels % 16 == 0)
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
          let theta = Double(y) * 1.0 / pow(10_000, Double(k) * 2 / Double(dim1))
          let sintheta = sin(theta)
          let costheta = cos(theta)
          rotTensor[0, i, j, (k + (dim0 / 2)) * 2] = Float(costheta)
          rotTensor[0, i, j, (k + (dim0 / 2)) * 2 + 1] = Float(sintheta)
        }
        for k in 0..<(dim2 / 2) {
          let theta = Double(x) * 1.0 / pow(10_000, Double(k) * 2 / Double(dim2))
          let sintheta = sin(theta)
          let costheta = cos(theta)
          rotTensor[0, i, j, (k + (dim0 / 2) + (dim1 / 2)) * 2] = Float(costheta)
          rotTensor[0, i, j, (k + (dim0 / 2) + (dim1 / 2)) * 2 + 1] = Float(sintheta)
        }
      }
    }
  }
  for i in (height * width)..<(height * width + tokenLength) {
    for j in 0..<heads {
      for k in 0..<(dim0 / 2) {
        let theta = 0 * 1.0 / pow(10_000, Double(k) * 2 / Double(dim0))
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, i, j, k * 2] = Float(costheta)
        rotTensor[0, i, j, k * 2 + 1] = Float(sintheta)
      }
      for k in 0..<(dim1 / 2) {
        let theta = 0 * 1.0 / pow(10_000, Double(k) * 2 / Double(dim1))
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, i, j, (k + (dim0 / 2)) * 2] = Float(costheta)
        rotTensor[0, i, j, (k + (dim0 / 2)) * 2 + 1] = Float(sintheta)
      }
      for k in 0..<(dim2 / 2) {
        let theta = 0 * 1.0 / pow(10_000, Double(k) * 2 / Double(dim2))
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, i, j, (k + (dim0 / 2) + (dim1 / 2)) * 2] = Float(costheta)
        rotTensor[0, i, j, (k + (dim0 / 2) + (dim1 / 2)) * 2 + 1] = Float(sintheta)
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
  Model, Model, Model, Model
) {
  let x = Input()
  let w1 = Dense(count: intermediateSize, noBias: true, name: "\(name)_w1")
  let w3 = Dense(count: intermediateSize, noBias: true, name: "\(name)_w3")
  var out = w1(x).swish() .* w3(x)
  // The scale down is integrated into out proj bias.
  if upcast {
    let scaleFactor: Float = 4
    out = (1 / scaleFactor) * out
  }
  let w2 = Dense(count: hiddenSize, noBias: true, name: "\(name)_w2")
  out = w2(out)
  if upcast {
    let scaleFactor: Float = 4
    out = out.to(.Float32) * scaleFactor
  } else {
    out = out.to(.Float32)
  }
  return (w1, w2, w3, Model([x], [out]))
}

private func MoEFeedForward(
  segments: Int, tokenLength: Int, hiddenSize: Int, intermediateSize: Int, upcast: Bool,
  name: String
) -> (
  Model, Model, Model, Model, Model
) {
  let x = Input()
  let gate = Dense(count: segments, noBias: true, name: "\(name)_gate")
  let route = gate(x).reshaped([tokenLength, 4]).softmax().partitioned(
    kth: 2, axis: 1, descending: true)
  var weights = route[0].reshaped([tokenLength * 2])
  let experts = route[1].reshaped([tokenLength * 2])  // This is to select into experts.
  let sort = experts.sorted(axis: 0, descending: false)
  let sortIndices = sort[1]
  weights = IndexSelect()(weights, sortIndices)  // Reorder the weights by the sorting order.
  let expertIds = sort[0].uniqueConsecutive(count: segments)
  let indices = 0.5 * sortIndices  // Scale it to 0..<tokenLength.
  let gathered = IndexSelect()(x.reshaped([tokenLength, hiddenSize]), indices)
  let w1 = SegmentedDense(
    segments: segments, count: intermediateSize, noBias: true, name: "\(name)_w1")
  let w3 = SegmentedDense(
    segments: segments, count: intermediateSize, noBias: true, name: "\(name)_w3")
  var out = w1(gathered, expertIds).swish() .* w3(gathered, expertIds)
  // The scale down is integrated into out proj bias.
  if upcast {
    let scaleFactor: Float = 4
    out = (1 / scaleFactor) * out
  }
  let w2 = SegmentedDense(segments: segments, count: hiddenSize, noBias: true, name: "\(name)_w2")
  out = w2(out, expertIds)
  // Out is tokenLength * 2, now multiply weights and scale back.
  out = out .* weights.reshaped([tokenLength * 2, 1])
  out = Functional.scatterAdd(count: tokenLength, out, index: indices)
  if upcast {
    let scaleFactor: Float = 4
    out = out.to(.Float32) * scaleFactor
  } else {
    out = out.to(.Float32)
  }
  return (gate, w1, w2, w3, Model([x], [out]))
}

private func JointTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, t: (Int, Int), hw: Int, contextBlockPreOnly: Bool,
  upcast: Bool
) -> (ModelWeightMapper, Model) {
  let context = Input()
  let x = Input()
  let rot = Input()
  let contextChunks = (0..<(contextBlockPreOnly ? 2 : 6)).map { _ in
    Input()
  }
  let xChunks = (0..<6).map { _ in Input() }
  let contextNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var contextOut = contextChunks[1] .* contextNorm1(context).to(.Float16) + contextChunks[0]
  let contextToKeys = Dense(count: k * h, name: "c_k")
  let contextToQueries = Dense(count: k * h, name: "c_q")
  let contextToValues = Dense(count: k * h, name: "c_v")
  var contextK = contextToKeys(contextOut)
  let normAddedK = RMSNorm(epsilon: 1e-5, axis: [2], name: "c_norm_k")
  contextK = normAddedK(contextK).reshaped([b, t.1, h, k])
  var contextQ = contextToQueries(contextOut)
  let normAddedQ = RMSNorm(epsilon: 1e-5, axis: [2], name: "c_norm_q")
  contextQ = normAddedQ(contextQ).reshaped([b, t.1, h, k])
  let contextV = contextToValues(contextOut).reshaped([b, t.1, h, k])
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = xChunks[1] .* xNorm1(x).to(.Float16) + xChunks[0]
  let xToKeys = Dense(count: k * h, name: "x_k")
  let xToQueries = Dense(count: k * h, name: "x_q")
  let xToValues = Dense(count: k * h, name: "x_v")
  var xK = xToKeys(xOut)
  let normK = RMSNorm(epsilon: 1e-5, axis: [2], name: "x_norm_k")
  xK = normK(xK).reshaped([b, hw, h, k])
  var xQ = xToQueries(xOut)
  let normQ = RMSNorm(epsilon: 1e-5, axis: [2], name: "x_norm_q")
  xQ = normQ(xQ).reshaped([b, hw, h, k])
  let xV = xToValues(xOut).reshaped([b, hw, h, k])
  var keys = Functional.concat(axis: 1, xK, contextK)
  let values = Functional.concat(axis: 1, xV, contextV)
  var queries = Functional.concat(axis: 1, xQ, contextQ)
  queries = (1.0 / Float(k).squareRoot().squareRoot()) * Functional.cmul(left: queries, right: rot)
  keys = (1.0 / Float(k).squareRoot().squareRoot()) * Functional.cmul(left: keys, right: rot)
  // Now run attention.
  let out = ScaledDotProductAttention(scale: 1, flags: [.Float16])(
    queries, keys, values
  ).reshaped([b, t.1 + hw, h * k])
  let contextUnifyheads: Model?
  if !contextBlockPreOnly {
    contextOut = out.reshaped(
      [b, t.0, h * k], offset: [0, hw, 0], strides: [(t.1 + hw) * h * k, h * k, 1]
    ).contiguous()
    let unifyheads = Dense(count: k * h, name: "c_o")
    contextOut = unifyheads(contextOut)
    contextUnifyheads = unifyheads
  } else {
    contextUnifyheads = nil
  }
  xOut = out.reshaped([b, hw, h * k], strides: [(t.1 + hw) * h * k, h * k, 1]).contiguous()
  let xUnifyheads = Dense(count: k * h, name: "x_o")
  xOut = xUnifyheads(xOut)
  if !contextBlockPreOnly {
    contextOut =
      context.reshaped([b, t.0, h * k], strides: [t.1 * h * k, h * k, 1]).contiguous()
      + (contextChunks[2] .* contextOut).to(of: context)
  }
  xOut = x + (xChunks[2] .* xOut).to(of: x)
  // Attentions are now. Now run MLP.
  let contextW1: Model?
  let contextW2: Model?
  let contextW3: Model?
  if !contextBlockPreOnly {
    let contextFF: Model
    (contextW1, contextW2, contextW3, contextFF) = FeedForward(
      hiddenSize: k * h, intermediateSize: 6_912, upcast: upcast, name: "c")
    let contextNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
    contextOut =
      contextOut
      + (contextChunks[5].to(of: contextOut)
      .* contextFF(
        contextNorm2(contextOut).to(.Float16) .* contextChunks[4] + contextChunks[3])).to(
        of: contextOut)
  } else {
    contextW1 = nil
    contextW2 = nil
    contextW3 = nil
  }
  let (xSharedW1, xSharedW2, xSharedW3, xSharedFF) = FeedForward(
    hiddenSize: k * h, intermediateSize: 3_584, upcast: upcast, name: "x_shared")
  let (xMoEGate, xMoEW1, xMoEW2, xMoEW3, xMoEFF) = MoEFeedForward(
    segments: 4, tokenLength: b * hw, hiddenSize: k * h, intermediateSize: 6_912, upcast: upcast,
    name: "x_moe")
  let xNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let xIn = xNorm2(xOut).to(.Float16) .* xChunks[4] + xChunks[3]
  xOut =
    xOut
    + (xChunks[5].to(of: xOut) .* (xSharedFF(xIn) + xMoEFF(xIn).reshaped([b, hw, h * k]))).to(
      of: xOut)
  let mapper: ModelWeightMapper = { _ in
    ModelWeightMapping()
  }
  if !contextBlockPreOnly {
    return (mapper, Model([x, context, rot] + contextChunks + xChunks, [xOut, contextOut]))
  } else {
    return (mapper, Model([x, context, rot] + contextChunks + xChunks, [xOut]))
  }
}

private func SingleTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, t: (Int, Int), hw: Int, contextBlockPreOnly: Bool,
  upcast: Bool
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let rot = Input()
  let xChunks = (0..<6).map { _ in Input() }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = xChunks[1] .* xNorm1(x).to(.Float16) + xChunks[0]
  let xToKeys = Dense(count: k * h, name: "x_k")
  let xToQueries = Dense(count: k * h, name: "x_q")
  let xToValues = Dense(count: k * h, name: "x_v")
  var xK = xToKeys(xOut)
  let normK = RMSNorm(epsilon: 1e-5, axis: [2], name: "x_norm_k")
  xK = normK(xK).reshaped([b, hw + t.1, h, k])
  var xQ = xToQueries(xOut)
  let normQ = RMSNorm(epsilon: 1e-5, axis: [2], name: "x_norm_q")
  xQ = normQ(xQ).reshaped([b, hw + t.1, h, k])
  let xV = xToValues(xOut).reshaped([b, hw + t.1, h, k])
  xQ = (1.0 / Float(k).squareRoot().squareRoot()) * Functional.cmul(left: xQ, right: rot)
  xK = (1.0 / Float(k).squareRoot().squareRoot()) * Functional.cmul(left: xK, right: rot)
  // Now run attention.
  let out = ScaledDotProductAttention(scale: 1, flags: [.Float16])(
    xQ, xK, xV
  ).reshaped([b, t.1 + hw, h * k])
  var xIn: Model.IO = x
  let xLength: Int
  if contextBlockPreOnly {
    xOut = out.reshaped([b, hw, h * k], strides: [(t.1 + hw) * h * k, h * k, 1]).contiguous()
    xIn = xIn.reshaped([b, hw, h * k], strides: [(t.1 + hw) * h * k, h * k, 1]).contiguous()
    xLength = hw
  } else {
    xOut = out.reshaped([b, hw + t.0, h * k], strides: [(t.1 + hw) * h * k, h * k, 1]).contiguous()
    xIn = xIn.reshaped([b, hw + t.0, h * k], strides: [(t.1 + hw) * h * k, h * k, 1]).contiguous()
    xLength = hw + t.0
  }
  let xUnifyheads = Dense(count: k * h, name: "x_o")
  xOut = xUnifyheads(xOut)
  xOut = xIn + (xChunks[2] .* xOut).to(of: xIn)
  // Attentions are now. Now run MLP.
  let (xSharedW1, xSharedW2, xSharedW3, xSharedFF) = FeedForward(
    hiddenSize: k * h, intermediateSize: 3_584, upcast: upcast, name: "x_shared")
  let (xMoEGate, xMoEW1, xMoEW2, xMoEW3, xMoEFF) = MoEFeedForward(
    segments: 4, tokenLength: b * xLength, hiddenSize: k * h, intermediateSize: 6_912,
    upcast: upcast,
    name: "x_moe")
  let xNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let xFFIn = xNorm2(xOut).to(.Float16) .* xChunks[4] + xChunks[3]
  xOut =
    xOut
    + (xChunks[5].to(of: xOut) .* (xSharedFF(xFFIn) + xMoEFF(xFFIn).reshaped([b, xLength, h * k])))
    .to(of: xOut)
  let mapper: ModelWeightMapper = { _ in
    ModelWeightMapping()
  }
  return (mapper, Model([x, rot] + xChunks, [xOut]))
}

func HiDream(batchSize: Int, height: Int, width: Int, textLength: (Int, Int), layers: (Int, Int))
  -> (
    Model, ModelWeightMapper
  )
{
  let x = Input()
  let rot = Input()
  let h = height / 2
  let w = width / 2
  let imgIn = Dense(count: 2_560, name: "x_embedder")
  var out = imgIn(
    x.reshaped([batchSize, h, 2, w, 2, 16]).permuted(0, 1, 3, 2, 4, 5).contiguous()
      .reshaped([batchSize, h * w, 2 * 2 * 16])
  ).to(.Float32)
  let encoderHiddenStates = (0..<49).map { _ in Input() }
  var context = encoderHiddenStates[encoderHiddenStates.count - 1].to(.Float32)
  var mappers = [ModelWeightMapper]()
  var adaLNChunks = [Input]()
  for i in 0..<layers.0 {
    let contextChunks = (0..<6).map { _ in Input() }
    let xChunks = (0..<6).map { _ in Input() }
    let contextIn = Functional.concat(axis: 1, context, encoderHiddenStates[i].to(.Float32))
    let (mapper, block) = JointTransformerBlock(
      prefix: "double_stream_blocks.\(i).block", k: 128, h: 20, b: batchSize,
      t: (textLength.0 + textLength.1, textLength.0 + textLength.1 * 2), hw: h * w,
      contextBlockPreOnly: false, upcast: i > 12)
    let blockOut = block([out, contextIn, rot] + contextChunks + xChunks)
    out = blockOut[0]
    context = blockOut[1]
    adaLNChunks.append(contentsOf: contextChunks + xChunks)
    mappers.append(mapper)
  }
  out = Functional.concat(axis: 1, out, context)
  for i in 0..<layers.1 {
    let xChunks = (0..<6).map { _ in Input() }
    let xIn = Functional.concat(axis: 1, out, encoderHiddenStates[layers.0 + i].to(.Float32))
    let (mapper, block) = SingleTransformerBlock(
      prefix: "single_stream_blocks.\(i).block", k: 128, h: 20, b: batchSize,
      t: (textLength.0 + textLength.1, textLength.0 + textLength.1 * 2), hw: h * w,
      contextBlockPreOnly: i == layers.1 - 1, upcast: false)
    out = block([xIn, rot] + xChunks)
    adaLNChunks.append(contentsOf: xChunks)
    mappers.append(mapper)
  }
  let shift = Input()
  let scale = Input()
  adaLNChunks.append(contentsOf: [shift, scale])
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  out = scale .* normFinal(out).to(.Float16) + shift
  let projOut = Dense(count: 2 * 2 * 16, name: "linear")
  out = (-projOut(out)).reshaped([batchSize, h, w, 2, 2, 16]).permuted(0, 1, 3, 2, 4, 5)
    .contiguous()
    .reshaped([
      batchSize, h * 2, w * 2, 16,
    ])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    return mapping
  }
  return (
    Model([x, rot] + encoderHiddenStates + adaLNChunks, [out]), mapper
  )
}

private func JointTransformerBlockFixed(
  prefix: String, k: Int, h: Int, contextBlockPreOnly: Bool
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
  let mapper: ModelWeightMapper = { _ in
    ModelWeightMapping()
  }
  return (mapper, Model([c], contextChunks + xChunks))
}

private func SingleTransformerBlockFixed(
  prefix: String, k: Int, h: Int
) -> (ModelWeightMapper, Model) {
  let c = Input()
  let xAdaLNs = (0..<6).map { Dense(count: k * h, name: "x_ada_ln_\($0)") }
  var xChunks = xAdaLNs.map { $0(c) }
  xChunks[1] = 1 + xChunks[1]
  xChunks[4] = 1 + xChunks[4]
  let mapper: ModelWeightMapper = { _ in
    ModelWeightMapping()
  }
  return (mapper, Model([c], xChunks))
}

func HiDreamFixed(timesteps: Int, layers: (Int, Int)) -> (
  Model, ModelWeightMapper
) {
  let t = Input()
  let vector = Input()
  let (tMlp0, tMlp2, timeEmbedder) = MLPEmbedder(channels: 2_560, name: "t")
  let (pMlp0, pMlp2, pooledEmbedder) = MLPEmbedder(channels: 2_560, name: "p")
  var vec = timeEmbedder(t) + pooledEmbedder(vector)
  let t5EncoderHiddenStates = Input()
  let llamaEncoderHiddenStates = (0..<32).map { _ in Input() }
  let captionProjections = (0..<49).map { _ in
    Dense(count: 2_560, noBias: true, name: "caption_projection")
  }
  var encoderHiddenStates = [Model.IO]()
  for i in 0..<48 {
    encoderHiddenStates.append(
      captionProjections[i](llamaEncoderHiddenStates[min(i, llamaEncoderHiddenStates.count - 1)]))
  }
  let t5HiddenStates = captionProjections[48](t5EncoderHiddenStates)
  encoderHiddenStates.append(
    Functional.concat(
      axis: 1, t5HiddenStates, encoderHiddenStates[encoderHiddenStates.count - 1]
    ))
  vec = vec.reshaped([timesteps, 1, 2_560]).swish()
  var mappers = [ModelWeightMapper]()
  var outs = [Model.IO]()
  for i in 0..<layers.0 {
    let (mapper, block) = JointTransformerBlockFixed(
      prefix: "double_stream_blocks.\(i).block", k: 128, h: 20, contextBlockPreOnly: false)
    mappers.append(mapper)
    let blockOut = block(vec)
    outs.append(blockOut)
  }
  for i in 0..<layers.1 {
    let (mapper, block) = SingleTransformerBlockFixed(
      prefix: "single_stream_blocks.\(i).block", k: 128, h: 20)
    mappers.append(mapper)
    let blockOut = block(vec)
    outs.append(blockOut)
  }
  let scale = Dense(count: 2_560, name: "ada_ln_0")
  let shift = Dense(count: 2_560, name: "ada_ln_1")
  outs.append(contentsOf: [shift(vec), 1 + scale(vec)])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    return mapping
  }
  return (
    Model(
      [t, vector, t5EncoderHiddenStates] + llamaEncoderHiddenStates, encoderHiddenStates + outs),
    mapper
  )
}
