import Foundation
import NNC

public func LTX2RotaryPositionEmbedding1D(
  tokenLength: Int, maxLength: Int, channels: Int, headDimension: Int
)
  -> Tensor<Float>
{
  var rotTensor = Tensor<Float>(.CPU, .HWC(1, tokenLength, channels))
  for i in 0..<tokenLength {
    let fractionalPosition = Double(i) / Double(maxLength) * 2 - 1
    for j in 0..<(channels / 2) {
      let theta: Double = pow(10_000, Double(j) / Double(channels / 2 - 1)) * .pi * 0.5
      let freq = theta * fractionalPosition
      let cosFreq = cos(freq)
      let sinFreq = sin(freq)
      rotTensor[0, i, j * 2] = Float(cosFreq)
      rotTensor[0, i, j * 2 + 1] = Float(sinFreq)
    }
  }
  return rotTensor.reshaped(.NHWC(1, tokenLength, channels / headDimension, headDimension))
}

public func LTX2VideoAudioRotaryPositionEmbedding(
  time: Int, height: Int, width: Int, audioTime: Int, channels: (Int, Int), numberOfHeads: Int
) -> (Tensor<Float>, Tensor<Float>, Tensor<Float>) {
  let tokenLength = time * height * width
  let audioLength = (time - 1) * 8 + 1
  var rotVideoTensor = Tensor<Float>(.CPU, .HWC(1, tokenLength, channels.0))
  let dim0 = channels.0 / 3 / 2
  let thetas0: [Double] = (0..<dim0).map { pow(10_000, Double($0) / Double(dim0 - 1)) * .pi * 0.5 }
  let pad = channels.0 / 2 - dim0 * 3
  rotVideoTensor.withUnsafeMutableBytes {
    guard let fp32 = $0.baseAddress?.assumingMemoryBound(to: Float.self) else { return }
    DispatchQueue.concurrentPerform(iterations: time) { i in
      let fractionFrame: Double = Double(max(0, i * 8 - 7) + i * 8 + 1) / 500 - 1
      for y in 0..<height {
        let fractionY: Double = (Double(y) + 0.5) / 32 - 1
        for x in 0..<width {
          let fractionX: Double = (Double(x) + 0.5) / 32 - 1
          let idx = i * height * width + y * width + x
          let fp = fp32 + idx * channels.0
          for j in 0..<pad {
            fp[j * 2] = 1
            fp[j * 2 + 1] = 0
          }
          for j in 0..<dim0 {
            let thetaFractionFrame = thetas0[j] * fractionFrame
            fp[j * 6 + pad * 2] = Float(cos(thetaFractionFrame))
            fp[j * 6 + pad * 2 + 1] = Float(sin(thetaFractionFrame))
            let thetaFractionY = thetas0[j] * fractionY
            fp[j * 6 + pad * 2 + 2] = Float(cos(thetaFractionY))
            fp[j * 6 + pad * 2 + 3] = Float(sin(thetaFractionY))
            let thetaFractionX = thetas0[j] * fractionX
            fp[j * 6 + pad * 2 + 4] = Float(cos(thetaFractionX))
            fp[j * 6 + pad * 2 + 5] = Float(sin(thetaFractionX))
          }
        }
      }
    }
  }
  var rotAudioTensor = Tensor<Float>(.CPU, .HWC(1, audioLength, channels.1))
  let dim1 = channels.1 / 2
  let thetas1: [Double] = (0..<dim1).map { pow(10_000, Double($0) / Double(dim1 - 1)) * .pi * 0.5 }
  rotAudioTensor.withUnsafeMutableBytes {
    guard let fp32 = $0.baseAddress?.assumingMemoryBound(to: Float.self) else { return }
    DispatchQueue.concurrentPerform(iterations: audioLength) { i in
      let fractionPosition: Double = Double(max(i * 4 - 3, 0) + i * 4 + 1) / 2000 - 1
      let fp = fp32 + i * channels.1
      for j in 0..<dim1 {
        let theta = thetas1[j] * fractionPosition
        fp[j * 2] = Float(cos(theta))
        fp[j * 2 + 1] = Float(sin(theta))
      }
    }
  }
  var rotVideoToAudioTensor = Tensor<Float>(.CPU, .HWC(1, tokenLength, channels.1))
  rotVideoToAudioTensor.withUnsafeMutableBytes {
    guard let fp32 = $0.baseAddress?.assumingMemoryBound(to: Float.self) else { return }
    DispatchQueue.concurrentPerform(iterations: time) { i in
      let fractionFrame: Double = Double(max(0, i * 8 - 7) + i * 8 + 1) / 500 - 1
      for y in 0..<height {
        for x in 0..<width {
          let idx = i * height * width + y * width + x
          let fp = fp32 + idx * channels.1
          for j in 0..<dim1 {
            let theta = thetas1[j] * fractionFrame
            fp[j * 2] = Float(cos(theta))
            fp[j * 2 + 1] = Float(sin(theta))
          }
        }
      }
    }
  }
  return (
    rotVideoTensor.reshaped(.NHWC(1, tokenLength, numberOfHeads, channels.0 / numberOfHeads)),
    rotAudioTensor.reshaped(.NHWC(1, audioLength, numberOfHeads, channels.1 / numberOfHeads)),
    rotVideoToAudioTensor.reshaped(.NHWC(1, tokenLength, numberOfHeads, channels.1 / numberOfHeads))
  )
}

public func LTX2ExtractAudioFramesAndHeight(_ shape: TensorShape) -> (Int, Int) {
  let audioFrames = (shape[0] - 1) * 8 + 1
  let audioHeight = (audioFrames + shape[2] * shape[0] - 1) / (shape[2] * shape[0])
  return (audioFrames, audioHeight)
}

private func BasicTransformerBlock1D(prefix: String, k: Int, h: Int, b: Int, t: Int) -> (
  ModelWeightMapper, Model
) {
  let x = Input()
  let norm = RMSNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let normX = norm(x).to(.Float16)
  let rot = Input()
  let toKeys = Dense(count: k * h, name: "to_k")
  let toQueries = Dense(count: k * h, name: "to_q")
  let toValues = Dense(count: k * h, name: "to_v")
  var keys = toKeys(normX)
  let normK = RMSNorm(epsilon: 1e-6, axis: [2], name: "norm_k")
  keys = normK(keys).reshaped([b, t, h, k])
  var queries = toQueries(normX)
  let normQ = RMSNorm(epsilon: 1e-6, axis: [2], name: "norm_q")
  queries = normQ(queries).reshaped([b, t, h, k])
  let values = toValues(normX).reshaped([b, t, h, k])
  queries = (1 / Float(k).squareRoot().squareRoot()) * Functional.cmul(left: queries, right: rot)
  keys = (1 / Float(k).squareRoot().squareRoot()) * Functional.cmul(left: keys, right: rot)
  // Now run attention.
  let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
  var out = scaledDotProductAttention(queries, keys, values).reshaped([b, t, k * h])
  let unifyheads = Dense(count: k * h, name: "to_o")
  out = unifyheads(out).to(of: x) + x
  let residual = out
  let upProj = Dense(count: k * h * 4, name: "up_proj")
  out = (1.0 / 8.0) * upProj(norm(out).to(.Float16)).GELU(approximate: .tanh)
  let downProj = Dense(count: k * h, name: "down_proj")
  out = Add(leftScalar: 8, rightScalar: 1)(downProj(out).to(of: residual), residual)
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).attn1.to_q.weight"] = [toQueries.weight.name]
    mapping["\(prefix).attn1.to_q.bias"] = [toQueries.bias.name]
    mapping["\(prefix).attn1.to_k.weight"] = [toKeys.weight.name]
    mapping["\(prefix).attn1.to_k.bias"] = [toKeys.bias.name]
    mapping["\(prefix).attn1.to_v.weight"] = [toValues.weight.name]
    mapping["\(prefix).attn1.to_v.bias"] = [toValues.bias.name]
    mapping["\(prefix).attn1.to_out.0.weight"] = [unifyheads.weight.name]
    mapping["\(prefix).attn1.to_out.0.bias"] = [unifyheads.bias.name]
    mapping["\(prefix).attn1.k_norm.weight"] = [normK.weight.name]
    mapping["\(prefix).attn1.q_norm.weight"] = [normQ.weight.name]
    mapping["\(prefix).ff.net.0.proj.weight"] = [upProj.weight.name]
    mapping["\(prefix).ff.net.0.proj.bias"] = [upProj.bias.name]
    mapping["\(prefix).ff.net.2.weight"] = [downProj.weight.name]
    mapping["\(prefix).ff.net.2.bias"] = [downProj.bias.name]
    return mapping
  }
  return (mapper, Model([x, rot], [out]))
}

func Embedding1DConnector(prefix: String, layers: Int, tokenLength: Int) -> (
  Model, ModelWeightMapper
) {
  let x = Input()
  let rot = Input()
  var mappers = [ModelWeightMapper]()
  var out: Model.IO = x.to(.Float32)
  for i in 0..<layers {
    let (mapper, block) = BasicTransformerBlock1D(
      prefix: "\(prefix).transformer_1d_blocks.\(i)", k: 128, h: 30, b: 1, t: tokenLength)
    out = block(out, rot)
    mappers.append(mapper)
  }
  let norm = RMSNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  out = norm(out).to(.Float16)
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    return mapping
  }
  return (Model([x, rot], [out]), mapper)
}

private func GELUMLPEmbedder(channels: Int, name: String) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: channels, name: "\(name)_embedder_0")
  var out = fc0(x).GELU(approximate: .tanh)
  let fc2 = Dense(count: channels, name: "\(name)_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

private func MLPEmbedder(channels: Int, name: String) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: channels, name: "\(name)_embedder_0")
  var out = fc0(x).swish()
  let fc2 = Dense(count: channels, name: "\(name)_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

private func LTX2SelfAttention(prefix: String, k: Int, h: Int, b: Int, t: Int, name: String) -> (
  ModelWeightMapper, Model
) {
  let x = Input()
  let rot = Input()
  let toKeys = Dense(count: k * h, name: "\(name)_k")
  let toQueries = Dense(count: k * h, name: "\(name)_q")
  let toValues = Dense(count: k * h, name: "\(name)_v")
  var keys = toKeys(x)
  let normK = RMSNorm(epsilon: 1e-6, axis: [2], name: "\(name)_norm_k")
  keys = normK(keys).reshaped([b, t, h, k])
  var queries = toQueries(x)
  let normQ = RMSNorm(epsilon: 1e-6, axis: [2], name: "\(name)_norm_q")
  queries = normQ(queries).reshaped([b, t, h, k])
  let values = toValues(x).reshaped([b, t, h, k])
  queries = (1 / Float(k).squareRoot().squareRoot()) * Functional.cmul(left: queries, right: rot)
  keys = (1 / Float(k).squareRoot().squareRoot()) * Functional.cmul(left: keys, right: rot)
  // Now run attention.
  let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
  var out = scaledDotProductAttention(queries, keys, values).reshaped([b, t, k * h])
  let unifyheads = Dense(count: k * h, name: "\(name)_o")
  out = unifyheads(out)
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).to_q.weight"] = [toQueries.weight.name]
    mapping["\(prefix).to_q.bias"] = [toQueries.bias.name]
    mapping["\(prefix).to_k.weight"] = [toKeys.weight.name]
    mapping["\(prefix).to_k.bias"] = [toKeys.bias.name]
    mapping["\(prefix).to_v.weight"] = [toValues.weight.name]
    mapping["\(prefix).to_v.bias"] = [toValues.bias.name]
    mapping["\(prefix).to_out.0.weight"] = [unifyheads.weight.name]
    mapping["\(prefix).to_out.0.bias"] = [unifyheads.bias.name]
    mapping["\(prefix).k_norm.weight"] = [normK.weight.name]
    mapping["\(prefix).q_norm.weight"] = [normQ.weight.name]
    return mapping
  }
  return (mapper, Model([x, rot], [out]))
}

private func LTX2CrossAttention(
  prefix: String, k: (Int, Int, Int), h: Int, b: Int, t: (Int, Int), positionEmbedding: Bool,
  KV: Bool, name: String
) -> (ModelWeightMapper, Model) {
  let x = Input()
  var context = [Input()]
  let rot: Input?
  let rotK: Input?
  if positionEmbedding {
    rot = Input()
    rotK = Input()
  } else {
    rot = nil
    rotK = nil
  }
  let toKeys: Model?
  let toValues: Model?
  let normK: Model?
  var keys: Model.IO
  let values: Model.IO
  if KV {
    keys = context[0]
    context.append(Input())
    values = context[1]
    toKeys = nil
    toValues = nil
    normK = nil
  } else {
    let toKeysLocal = Dense(count: k.1 * h, name: "\(name)_k")
    let toValuesLocal = Dense(count: k.1 * h, name: "\(name)_v")
    keys = toKeysLocal(context[0])
    let normKLocal = RMSNorm(epsilon: 1e-6, axis: [2], name: "\(name)_norm_k")
    keys = normKLocal(keys).reshaped([b, t.1, h, k.1])
    keys = (1 / Float(k.1).squareRoot().squareRoot()) * keys
    values = toValuesLocal(context[0]).reshaped([b, t.1, h, k.1])
    toKeys = toKeysLocal
    toValues = toValuesLocal
    normK = normKLocal
  }
  let toQueries = Dense(count: k.1 * h, name: "\(name)_q")
  var queries = toQueries(x)
  let normQ = RMSNorm(epsilon: 1e-6, axis: [2], name: "\(name)_norm_q")
  queries = normQ(queries).reshaped([b, t.0, h, k.1])
  if let rot = rot {
    queries = Functional.cmul(left: queries, right: rot)
  }
  if let rotK = rotK {
    keys = Functional.cmul(left: keys, right: rotK)
  }
  queries = (1 / Float(k.1).squareRoot().squareRoot()) * queries
  // Now run attention.
  let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
  var out = scaledDotProductAttention(queries, keys, values).reshaped([b, t.0, k.1 * h])
  let unifyheads = Dense(count: k.0 * h, name: "\(name)_o")
  out = unifyheads(out)
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).to_q.weight"] = [toQueries.weight.name]
    mapping["\(prefix).to_q.bias"] = [toQueries.bias.name]
    if let toKeys = toKeys {
      mapping["\(prefix).to_k.weight"] = [toKeys.weight.name]
      mapping["\(prefix).to_k.bias"] = [toKeys.bias.name]
    }
    if let toValues = toValues {
      mapping["\(prefix).to_v.weight"] = [toValues.weight.name]
      mapping["\(prefix).to_v.bias"] = [toValues.bias.name]
    }
    mapping["\(prefix).to_out.0.weight"] = [unifyheads.weight.name]
    mapping["\(prefix).to_out.0.bias"] = [unifyheads.bias.name]
    if let normK = normK {
      mapping["\(prefix).k_norm.weight"] = [normK.weight.name]
    }
    mapping["\(prefix).q_norm.weight"] = [normQ.weight.name]
    return mapping
  }
  return (
    mapper, Model([x] + (rot.map { [$0] } ?? []) + context + (rotK.map { [$0] } ?? []), [out])
  )
}

private func FeedForward(hiddenSize: Int, intermediateSize: Int, name: String) -> (
  Model, Model, Model
) {
  let x = Input()
  let linear1 = Dense(count: intermediateSize, name: "\(name)_linear1")
  var out = linear1(x).GELU(approximate: .tanh)
  // The scale down is integrated into out proj bias.
  let outProjection = Dense(count: hiddenSize, name: "\(name)_out_proj")
  out = outProjection(out)
  return (linear1, outProjection, Model([x], [out]))
}

private func LTX2TransformerBlock(
  prefix: String, k: (Int, Int), h: Int, b: Int, t: Int, time: Int, hw: Int, a: Int,
  tokenModulation: Bool
) -> (ModelWeightMapper, Model) {
  let vx = Input()
  let ax = Input()
  let cvK = Input()
  let cvV = Input()
  let caK = Input()
  let caV = Input()
  let rot = Input()
  let rotA = Input()
  let rotCX = Input()
  let modulations = (0..<22).map { _ in Input() }
  let norm = RMSNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var out: Model.IO
  if tokenModulation {
    out = (norm(vx).reshaped([time, hw, k.0 * h]) .* modulations[0] + modulations[1]).reshaped([
      1, time * hw, k.0 * h,
    ])
  } else {
    out = norm(vx) .* modulations[0] + modulations[1]
  }
  let (attn1Mapper, attn1) = LTX2SelfAttention(
    prefix: "\(prefix).attn1", k: k.0, h: h, b: b, t: time * hw, name: "x")
  if tokenModulation {
    out =
      vx
      + (attn1(out.to(.Float16), rot).to(of: vx).reshaped([time, hw, k.0 * h]) .* modulations[2])
      .reshaped([1, time * hw, k.0 * h])
  } else {
    out = vx + attn1(out.to(.Float16), rot).to(of: vx) .* modulations[2]
  }
  let (attn2Mapper, attn2) = LTX2CrossAttention(
    prefix: "\(prefix).attn2", k: (k.0, k.0, k.0), h: h, b: b, t: (time * hw, t),
    positionEmbedding: false,
    KV: true, name: "cv")
  let normOut = norm(out).to(.Float16)
  out = out + attn2(normOut, cvK, cvV).to(of: out)
  let (audioAttn1Mapper, audioAttn1) = LTX2SelfAttention(
    prefix: "\(prefix).audio_attn1", k: k.1, h: h, b: b, t: a, name: "a")
  var aOut =
    norm(ax) .* modulations[3] + modulations[4]
  aOut = ax + audioAttn1(aOut.to(.Float16), rotA).to(of: ax)
    .* modulations[5]
  let (audioAttn2Mapper, audioAttn2) = LTX2CrossAttention(
    prefix: "\(prefix).audio_attn2", k: (k.1, k.1, k.1), h: h, b: b, t: (a, t),
    positionEmbedding: false, KV: true, name: "ca")
  let normAOut = norm(aOut).to(.Float16)
  aOut = aOut + audioAttn2(normAOut, caK, caV).to(of: aOut)
  let vxNorm3 = norm(out)
  let axNorm3 = norm(aOut)
  let (audioToVideoAttnMapper, audioToVideoAttn) = LTX2CrossAttention(
    prefix: "\(prefix).audio_to_video_attn", k: (k.0, k.1, k.1), h: h, b: b, t: (time * hw, a),
    positionEmbedding: true, KV: false, name: "ax")
  let vxScaled: Model.IO
  if tokenModulation {
    vxScaled = (vxNorm3.reshaped([time, hw, k.0 * h]) .* modulations[6] + modulations[7]).reshaped([
      1, time * hw, k.0 * h,
    ])
  } else {
    vxScaled = vxNorm3 .* modulations[6] + modulations[7]
  }
  let axScaled =
    axNorm3 .* modulations[8] + modulations[9]
  if tokenModulation {
    out =
      out
      + (audioToVideoAttn(vxScaled.to(.Float16), rotCX, axScaled.to(.Float16), rotA).to(of: out)
      .reshaped([time, hw, k.0 * h]) .* modulations[10]).reshaped([1, time * hw, k.0 * h])
  } else {
    out =
      out + audioToVideoAttn(vxScaled.to(.Float16), rotCX, axScaled.to(.Float16), rotA).to(of: out)
      .* modulations[10]
  }
  let (videoToAudioAttnMapper, videoToAudioAttn) = LTX2CrossAttention(
    prefix: "\(prefix).video_to_audio_attn", k: (k.1, k.1, k.0), h: h, b: b, t: (a, time * hw),
    positionEmbedding: true, KV: false, name: "xa")
  let audioVxScaled: Model.IO
  if tokenModulation {
    audioVxScaled = (vxNorm3.reshaped([time, hw, k.0 * h]) .* modulations[11] + modulations[12])
      .reshaped([1, time * hw, k.0 * h])
  } else {
    audioVxScaled = vxNorm3 .* modulations[11] + modulations[12]
  }
  let audioAxScaled =
    axNorm3 .* modulations[13] + modulations[14]
  aOut =
    aOut
    + videoToAudioAttn(audioAxScaled.to(.Float16), rotA, audioVxScaled.to(.Float16), rotCX).to(
      of: aOut)
    .* modulations[15]
  // Now attention done, do MLP.
  let (xLinear1, xOutProjection, xFF) = FeedForward(
    hiddenSize: 4096, intermediateSize: 4096 * 4, name: "x")
  let lastVxScaled: Model.IO
  if tokenModulation {
    lastVxScaled = (norm(out).reshaped([time, hw, k.0 * h]) .* modulations[16] + modulations[17])
      .reshaped([1, time * hw, k.0 * h])
    out =
      out
      + (xFF(lastVxScaled.to(.Float16)).to(of: out).reshaped([time, hw, k.0 * h]) .* modulations[18])
      .reshaped([1, time * hw, k.0 * h])
  } else {
    lastVxScaled =
      norm(out) .* modulations[16] + modulations[17]
    out = out + xFF(lastVxScaled.to(.Float16)).to(of: out) .* modulations[18]
  }
  let lastAxScaled =
    norm(aOut) .* modulations[19] + modulations[20]
  let (audioLinear1, audioOutProjection, audioFF) = FeedForward(
    hiddenSize: 2048, intermediateSize: 2048 * 4, name: "a")
  aOut = aOut + audioFF(lastAxScaled.to(.Float16)).to(of: aOut)
    .* modulations[21]
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping.merge(attn1Mapper(format)) { v, _ in v }
    mapping.merge(attn2Mapper(format)) { v, _ in v }
    mapping.merge(audioAttn1Mapper(format)) { v, _ in v }
    mapping.merge(audioAttn2Mapper(format)) { v, _ in v }
    mapping.merge(audioToVideoAttnMapper(format)) { v, _ in v }
    mapping.merge(videoToAudioAttnMapper(format)) { v, _ in v }
    mapping["\(prefix).ff.net.0.proj.weight"] = [xLinear1.weight.name]
    mapping["\(prefix).ff.net.0.proj.bias"] = [xLinear1.bias.name]
    mapping["\(prefix).ff.net.2.weight"] = [xOutProjection.weight.name]
    mapping["\(prefix).ff.net.2.bias"] = [xOutProjection.bias.name]
    mapping["\(prefix).audio_ff.net.0.proj.weight"] = [audioLinear1.weight.name]
    mapping["\(prefix).audio_ff.net.0.proj.bias"] = [audioLinear1.bias.name]
    mapping["\(prefix).audio_ff.net.2.weight"] = [audioOutProjection.weight.name]
    mapping["\(prefix).audio_ff.net.2.bias"] = [audioOutProjection.bias.name]
    return mapping
  }
  var inputs: [Input] = [vx, rot, cvK, cvV, ax, rotA, caK, caV, rotCX]
  inputs.append(contentsOf: modulations)
  return (mapper, Model(inputs, [out, aOut]))
}

private func LTX2AdaLNSingle(
  prefix: String, timesteps: Int, channels: Int, count: Int, outputEmbedding: Bool, name: String,
  t: Input
) -> (
  ModelWeightMapper, Model.IO?, [Model.IO]
) {
  let (tMlp0, tMlp2, tEmbedder) = MLPEmbedder(channels: channels, name: name)
  let adaLNSingles = (0..<count).map { Dense(count: channels, name: "\(name)_adaln_single_\($0)") }
  var tOut = tEmbedder(t).reshaped([timesteps, 1, channels])
  let tEmb: Model.IO?
  if outputEmbedding {
    tEmb = tOut.to(.Float32)
  } else {
    tEmb = nil
  }
  tOut = tOut.swish()
  let chunks = adaLNSingles.map { $0(tOut).to(.Float32) }
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping[
      "\(prefix).emb.timestep_embedder.linear_1.weight"
    ] = [tMlp0.weight.name]
    mapping[
      "\(prefix).emb.timestep_embedder.linear_1.bias"
    ] = [tMlp0.bias.name]
    mapping[
      "\(prefix).emb.timestep_embedder.linear_2.weight"
    ] = [tMlp2.weight.name]
    mapping[
      "\(prefix).emb.timestep_embedder.linear_2.bias"
    ] = [tMlp2.bias.name]
    mapping[
      "\(prefix).linear.weight"
    ] = ModelWeightElement(adaLNSingles.map { $0.weight.name })
    mapping[
      "\(prefix).linear.bias"
    ] = ModelWeightElement(adaLNSingles.map { $0.bias.name })
    return mapping
  }
  return (mapper, tEmb, chunks)
}

func LTX2(
  time: Int, h: Int, w: Int, textLength: Int, audioFrames: Int, channels: (Int, Int), layers: Int,
  tokenModulation: Bool
) -> (
  ModelWeightMapper, Model
) {
  let x = Input()
  let rot = Input()
  let rotA = Input()
  let rotCX = Input()
  let xEmbedder = Convolution(
    groups: 1, filters: channels.0, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW,
    name: "x_embedder")
  var out = xEmbedder(x).reshaped(.HWC(1, time * h * w, channels.0)).to(.Float32)
  let a = Input()
  let aEmbedder = Dense(count: channels.1, name: "a_embedder")
  var aOut = aEmbedder(a).to(.Float32)
  var mappers = [ModelWeightMapper]()
  let hw = h * w
  var modulationsAndKVs = [Input]()
  for i in 0..<layers {
    let (mapper, block) = LTX2TransformerBlock(
      prefix: "transformer_blocks.\(i)", k: (channels.0 / 32, channels.1 / 32), h: 32, b: 1,
      t: textLength, time: time, hw: hw, a: audioFrames, tokenModulation: tokenModulation)
    let cvK = Input()
    let cvV = Input()
    let caK = Input()
    let caV = Input()
    let modulations = (0..<22).map { _ in Input() }
    let blockOut = block([out, rot, cvK, cvV, aOut, rotA, caK, caV, rotCX] + modulations)
    mappers.append(mapper)
    out = blockOut[0]
    aOut = blockOut[1]
    modulationsAndKVs.append(contentsOf: [cvK, cvV, caK, caV])
    modulationsAndKVs.append(contentsOf: modulations)
  }
  let scaleShiftModulations = [Input(), Input()]
  let normOut = RMSNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  if tokenModulation {
    out = normOut(out).reshaped([time, hw, channels.0]) .* scaleShiftModulations[0]
      + scaleShiftModulations[1]
  } else {
    out = normOut(out) .* scaleShiftModulations[0] + scaleShiftModulations[1]
  }
  modulationsAndKVs.append(contentsOf: scaleShiftModulations)
  let projOut = Dense(count: 128, name: "proj_out")
  out = projOut(out.to(.Float16))
  let audioScaleShiftModulations = [Input(), Input()]
  aOut = normOut(aOut) .* audioScaleShiftModulations[0] + audioScaleShiftModulations[1]
  modulationsAndKVs.append(contentsOf: audioScaleShiftModulations)
  let audioProjOut = Dense(count: 128, name: "audio_proj_out")
  aOut = audioProjOut(aOut.to(.Float16))
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["patchify_proj.weight"] = [xEmbedder.weight.name]
    mapping["patchify_proj.bias"] = [xEmbedder.bias.name]
    mapping["audio_patchify_proj.weight"] = [aEmbedder.weight.name]
    mapping["audio_patchify_proj.bias"] = [aEmbedder.bias.name]
    mapping["proj_out.weight"] = [projOut.weight.name]
    mapping["proj_out.bias"] = [projOut.bias.name]
    mapping["audio_proj_out.weight"] = [audioProjOut.weight.name]
    mapping["audio_proj_out.bias"] = [audioProjOut.bias.name]
    return mapping
  }
  return (mapper, Model([x, a, rot, rotA, rotCX] + modulationsAndKVs, [out, aOut]))
}

private func LTX2CrossAttentionFixed(
  prefix: String, k: (Int, Int, Int), h: Int, b: Int, t: Int, positionEmbedding: Bool,
  name: String
) -> (ModelWeightMapper, Model) {
  let context = Input()
  let toKeys = Dense(count: k.1 * h, name: "\(name)_k")
  let toValues = Dense(count: k.1 * h, name: "\(name)_v")
  var keys = toKeys(context)
  let normK = RMSNorm(epsilon: 1e-6, axis: [2], name: "\(name)_norm_k")
  keys = normK(keys).reshaped([b, t, h, k.1])
  let values = toValues(context).reshaped([b, t, h, k.1])
  keys = (1 / Float(k.1).squareRoot().squareRoot()) * keys
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).to_k.weight"] = [toKeys.weight.name]
    mapping["\(prefix).to_k.bias"] = [toKeys.bias.name]
    mapping["\(prefix).to_v.weight"] = [toValues.weight.name]
    mapping["\(prefix).to_v.bias"] = [toValues.bias.name]
    mapping["\(prefix).k_norm.weight"] = [normK.weight.name]
    return mapping
  }
  return (mapper, Model([context], [keys, values]))
}

private func LTX2TransformerBlockFixed(
  prefix: String, k: (Int, Int), h: Int, b: Int, t: Int
) -> (ModelWeightMapper, Model) {
  let cv = Input()
  let ca = Input()
  let (attn2Mapper, attn2) = LTX2CrossAttentionFixed(
    prefix: "\(prefix).attn2", k: (k.0, k.0, k.0), h: h, b: b, t: t, positionEmbedding: false,
    name: "cv")
  var outs = [Model.IO]()
  outs.append(attn2(cv))
  let (audioAttn2Mapper, audioAttn2) = LTX2CrossAttentionFixed(
    prefix: "\(prefix).audio_attn2", k: (k.1, k.1, k.1), h: h, b: b, t: t,
    positionEmbedding: false, name: "ca")
  outs.append(audioAttn2(ca))
  let timesteps = (0..<6).map { _ in Input() }
  let attn1Modulations = (0..<6).map {
    Parameter<Float>(.GPU(0), .HWC(1, 1, k.0 * h), name: "attn1_ada_ln_\($0)")
  }
  outs.append(attn1Modulations[1] + timesteps[1])
  outs.append(attn1Modulations[0] + timesteps[0])
  outs.append(attn1Modulations[2] + timesteps[2])
  let audioTimesteps = (0..<6).map { _ in Input() }
  let audioAttn1Modulations = (0..<6).map {
    Parameter<Float>(.GPU(0), .HWC(1, 1, k.1 * h), name: "audio_attn1_ada_ln_\($0)")
  }
  outs.append(audioAttn1Modulations[1] + audioTimesteps[1])
  outs.append(audioAttn1Modulations[0] + audioTimesteps[0])
  outs.append(audioAttn1Modulations[2] + audioTimesteps[2])
  let caScaleShiftTimesteps = (0..<4).map { _ in Input() }
  let caGateTimesteps = Input()
  let audioToVideoAttnModulations = (0..<5).map {
    if $0 < 2 {
      return Parameter<Float>(
        .GPU(0), .HWC(1, 1, k.0 * h), name: "audio_to_video_attn_ada_ln_\($0)")
    } else if $0 < 4 {
      return Parameter<Float>(
        .GPU(0), .HWC(1, 1, k.1 * h), name: "audio_to_video_attn_ada_ln_\($0)")
    } else {
      return Parameter<Float>(
        .GPU(0), .HWC(1, 1, k.0 * h), name: "audio_to_video_attn_ada_ln_\($0)")
    }
  }
  outs.append(audioToVideoAttnModulations[1] + caScaleShiftTimesteps[1])
  outs.append(audioToVideoAttnModulations[0] + caScaleShiftTimesteps[0])
  outs.append(audioToVideoAttnModulations[3] + caScaleShiftTimesteps[3])
  outs.append(audioToVideoAttnModulations[2] + caScaleShiftTimesteps[2])
  outs.append(audioToVideoAttnModulations[4] + caGateTimesteps)
  let audioCaScaleShiftTimesteps = (0..<4).map { _ in Input() }
  let audioCaGateTimesteps = Input()
  let videoToAudioAttnModulations = (0..<5).map {
    if $0 < 2 {
      return Parameter<Float>(
        .GPU(0), .HWC(1, 1, k.0 * h), name: "video_to_audio_attn_ada_ln_\($0)")
    } else {
      return Parameter<Float>(
        .GPU(0), .HWC(1, 1, k.1 * h), name: "video_to_audio_attn_ada_ln_\($0)")
    }
  }
  outs.append(videoToAudioAttnModulations[1] + audioCaScaleShiftTimesteps[1])
  outs.append(videoToAudioAttnModulations[0] + audioCaScaleShiftTimesteps[0])
  outs.append(videoToAudioAttnModulations[3] + audioCaScaleShiftTimesteps[3])
  outs.append(videoToAudioAttnModulations[2] + audioCaScaleShiftTimesteps[2])
  outs.append(videoToAudioAttnModulations[4] + audioCaGateTimesteps)
  outs.append(attn1Modulations[4] + timesteps[4])
  outs.append(attn1Modulations[3] + timesteps[3])
  outs.append(attn1Modulations[5] + timesteps[5])
  outs.append(audioAttn1Modulations[4] + audioTimesteps[4])
  outs.append(audioAttn1Modulations[3] + audioTimesteps[3])
  outs.append(audioAttn1Modulations[5] + audioTimesteps[5])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).scale_shift_table"] = ModelWeightElement(
      (0..<6).map { attn1Modulations[$0].weight.name })
    mapping.merge(attn2Mapper(format)) { v, _ in v }
    mapping["\(prefix).audio_scale_shift_table"] = ModelWeightElement(
      (0..<6).map { audioAttn1Modulations[$0].weight.name })
    mapping.merge(audioAttn2Mapper(format)) { v, _ in v }
    mapping["\(prefix).scale_shift_table_a2v_ca_audio"] = [
      audioToVideoAttnModulations[3].weight.name, audioToVideoAttnModulations[2].weight.name,
      videoToAudioAttnModulations[3].weight.name, videoToAudioAttnModulations[2].weight.name,
      videoToAudioAttnModulations[4].weight.name,
    ]
    mapping["\(prefix).scale_shift_table_a2v_ca_video"] = [
      audioToVideoAttnModulations[1].weight.name, audioToVideoAttnModulations[0].weight.name,
      videoToAudioAttnModulations[1].weight.name, videoToAudioAttnModulations[0].weight.name,
      audioToVideoAttnModulations[4].weight.name,
    ]
    return mapping
  }
  var inputs: [Input] = [cv, ca]
  inputs.append(contentsOf: timesteps + audioTimesteps)
  inputs.append(contentsOf: caScaleShiftTimesteps + [caGateTimesteps])
  inputs.append(contentsOf: audioCaScaleShiftTimesteps + [audioCaGateTimesteps])
  return (mapper, Model(inputs, outs))
}

func LTX2Fixed(
  time: Int, textLength: Int, audioFrames: Int, timesteps: Int, channels: (Int, Int), layers: Int
) -> (
  ModelWeightMapper, Model
) {
  let txt = Input()
  let aTxt = Input()
  let (contextMlp0, contextMlp2, contextEmbedder) = GELUMLPEmbedder(
    channels: channels.0, name: "context")
  let txtOut = contextEmbedder(txt)
  let (aContextMlp0, aContextMlp2, aContextEmbedder) = GELUMLPEmbedder(
    channels: channels.1, name: "a_context")
  let aTxtOut = aContextEmbedder(aTxt)
  let t = Input()
  let (txMapper, txEmb, txEmbChunks) = LTX2AdaLNSingle(
    prefix: "adaln_single", timesteps: timesteps, channels: channels.0, count: 6,
    outputEmbedding: true,
    name: "tx", t: t)
  let (taMapper, taEmb, taEmbChunks) = LTX2AdaLNSingle(
    prefix: "audio_adaln_single", timesteps: timesteps, channels: channels.1, count: 6,
    outputEmbedding: true, name: "ta", t: t)
  let (caMapper, _, tcxEmbChunks) = LTX2AdaLNSingle(
    prefix: "av_ca_video_scale_shift_adaln_single", timesteps: timesteps, channels: channels.0,
    count: 4,
    outputEmbedding: false, name: "tcx", t: t)
  let (audioCaMapper, _, tcaEmbChunks) = LTX2AdaLNSingle(
    prefix: "av_ca_audio_scale_shift_adaln_single", timesteps: timesteps, channels: channels.1,
    count: 4,
    outputEmbedding: false, name: "tca", t: t)
  let (gateMapper, _, a2vEmbChunks) = LTX2AdaLNSingle(
    prefix: "av_ca_a2v_gate_adaln_single", timesteps: timesteps, channels: channels.0, count: 1,
    outputEmbedding: false, name: "a2v", t: t)
  let (audioGateMapper, _, v2aEmbChunks) = LTX2AdaLNSingle(
    prefix: "av_ca_v2a_gate_adaln_single", timesteps: timesteps, channels: channels.1, count: 1,
    outputEmbedding: false, name: "v2a", t: t)
  var mappers = [ModelWeightMapper]()
  var outs = [Model.IO]()
  let timesteps_1 = txEmbChunks[1] + 1
  let audioTimesteps_1 = taEmbChunks[1] + 1
  let timesteps_4 = txEmbChunks[4] + 1
  let audioTimesteps_4 = taEmbChunks[4] + 1
  let caScaleShiftTimesteps_1 = tcxEmbChunks[0] + 1
  let caScaleShiftTimesteps_3 = tcaEmbChunks[0] + 1
  let audioCaScaleShiftTimesteps_1 = tcxEmbChunks[2] + 1
  let audioCaScaleShiftTimesteps_3 = tcaEmbChunks[2] + 1
  for i in 0..<layers {
    let (mapper, block) = LTX2TransformerBlockFixed(
      prefix: "transformer_blocks.\(i)", k: (channels.0 / 32, channels.1 / 32), h: 32, b: 1,
      t: textLength)
    let blockOut = block(
      txtOut, aTxtOut, txEmbChunks[0], timesteps_1, txEmbChunks[2], txEmbChunks[3],
      timesteps_4, txEmbChunks[5], taEmbChunks[0], audioTimesteps_1, taEmbChunks[2],
      taEmbChunks[3], audioTimesteps_4, taEmbChunks[5], tcxEmbChunks[1], caScaleShiftTimesteps_1,
      tcaEmbChunks[1], caScaleShiftTimesteps_3, a2vEmbChunks[0], tcxEmbChunks[3],
      audioCaScaleShiftTimesteps_1, tcaEmbChunks[3], audioCaScaleShiftTimesteps_3, v2aEmbChunks[0])
    mappers.append(mapper)
    outs.append(blockOut)
  }
  let scaleShiftModulations = (0..<2).map {
    Parameter<Float>(.GPU(0), .HWC(1, 1, channels.0), name: "norm_out_ada_ln_\($0)")
  }
  if let txEmb = txEmb {
    outs.append(1 + scaleShiftModulations[1] + txEmb)
    outs.append(scaleShiftModulations[0] + txEmb)
  }
  let audioScaleShiftModulations = (0..<2).map {
    Parameter<Float>(.GPU(0), .HWC(1, 1, channels.1), name: "audio_norm_out_ada_ln_\($0)")
  }
  if let taEmb = taEmb {
    outs.append(1 + audioScaleShiftModulations[1] + taEmb)
    outs.append(audioScaleShiftModulations[0] + taEmb)
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["caption_projection.linear_1.weight"] = [contextMlp0.weight.name]
    mapping["caption_projection.linear_1.bias"] = [contextMlp0.bias.name]
    mapping["caption_projection.linear_2.weight"] = [contextMlp2.weight.name]
    mapping["caption_projection.linear_2.bias"] = [contextMlp2.bias.name]
    mapping[
      "audio_caption_projection.linear_1.weight"
    ] = [aContextMlp0.weight.name]
    mapping[
      "audio_caption_projection.linear_1.bias"
    ] = [aContextMlp0.bias.name]
    mapping[
      "audio_caption_projection.linear_2.weight"
    ] = [aContextMlp2.weight.name]
    mapping[
      "audio_caption_projection.linear_2.bias"
    ] = [aContextMlp2.bias.name]
    mapping.merge(txMapper(format)) { v, _ in v }
    mapping.merge(taMapper(format)) { v, _ in v }
    mapping.merge(caMapper(format)) { v, _ in v }
    mapping.merge(audioCaMapper(format)) { v, _ in v }
    mapping.merge(gateMapper(format)) { v, _ in v }
    mapping.merge(audioGateMapper(format)) { v, _ in v }
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping["scale_shift_table"] = ModelWeightElement(
      (0..<2).map { scaleShiftModulations[$0].weight.name })
    mapping["audio_scale_shift_table"] = ModelWeightElement(
      (0..<2).map { audioScaleShiftModulations[$0].weight.name })
    return mapping
  }
  return (mapper, Model([txt, aTxt, t], outs))
}
