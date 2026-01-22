import NNC

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
  var out: Model.IO = x
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
  prefix: String, k: (Int, Int, Int), h: Int, b: Int, t: (Int, Int), name: String
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let context = Input()
  let rot = Input()
  let rotK = Input()
  let toKeys = Dense(count: k.1 * h, name: "\(name)_k")
  let toQueries = Dense(count: k.1 * h, name: "\(name)_q")
  let toValues = Dense(count: k.1 * h, name: "\(name)_v")
  var keys = toKeys(context)
  let normK = RMSNorm(epsilon: 1e-6, axis: [2], name: "\(name)_norm_k")
  keys = normK(keys).reshaped([b, t.1, h, k.1])
  var queries = toQueries(x)
  let normQ = RMSNorm(epsilon: 1e-6, axis: [2], name: "\(name)_norm_q")
  queries = normQ(queries).reshaped([b, t.0, h, k.1])
  let values = toValues(context).reshaped([b, t.1, h, k.1])
  queries = (1 / Float(k.1).squareRoot().squareRoot()) * Functional.cmul(left: queries, right: rot)
  keys = (1 / Float(k.1).squareRoot().squareRoot()) * Functional.cmul(left: keys, right: rotK)
  // Now run attention.
  let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
  var out = scaledDotProductAttention(queries, keys, values).reshaped([b, t.0, k.1 * h])
  let unifyheads = Dense(count: k.0 * h, name: "\(name)_o")
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
  return (mapper, Model([x, rot, context, rotK], [out]))
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
  prefix: String, k: Int, h: Int, b: Int, t: Int, hw: Int, a: Int, intermediateSize: Int
) -> (ModelWeightMapper, Model) {
  let vx = Input()
  let ax = Input()
  let cv = Input()
  let ca = Input()
  let rot = Input()
  let rotC = Input()
  let rotA = Input()
  let rotAC = Input()
  let rotCX = Input()
  let timesteps = (0..<6).map { _ in Input() }
  let attn1Modulations = (0..<6).map {
    Parameter<Float>(.GPU(0), .HWC(1, 1, k * h), name: "attn1_ada_ln_\($0)")
  }
  let norm = RMSNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var out =
    norm(vx) .* (1 + (attn1Modulations[1] + timesteps[1])) + (attn1Modulations[0] + timesteps[0])
  let (attn1Mapper, attn1) = LTX2SelfAttention(
    prefix: "\(prefix).attn1", k: k, h: h, b: b, t: hw, name: "x")
  out = vx + attn1(out.to(.Float16), rot).to(of: vx) .* (attn1Modulations[2] + timesteps[2])
  let (attn2Mapper, attn2) = LTX2CrossAttention(
    prefix: "\(prefix).attn2", k: (k, k, k), h: h, b: b, t: (hw, t), name: "cv")
  let normOut = norm(out).to(.Float16)
  out = out + attn2(normOut, rot, cv, rotC).to(of: out)
  let audioTimesteps = (0..<6).map { _ in Input() }
  let audioAttn1Modulations = (0..<6).map {
    Parameter<Float>(.GPU(0), .HWC(1, 1, k / 2 * h), name: "audio_attn1_ada_ln_\($0)")
  }
  let (audioAttn1Mapper, audioAttn1) = LTX2SelfAttention(
    prefix: "\(prefix).audio_attn1", k: k / 2, h: h, b: b, t: a, name: "a")
  var aOut =
    norm(ax) .* (1 + (audioAttn1Modulations[1] + audioTimesteps[1]))
    + (audioAttn1Modulations[0] + audioTimesteps[0])
  aOut = ax + audioAttn1(aOut.to(.Float16), rotA).to(of: ax)
    .* (audioAttn1Modulations[2] + audioTimesteps[2])
  let (audioAttn2Mapper, audioAttn2) = LTX2CrossAttention(
    prefix: "\(prefix).audio_attn2", k: (k / 2, k / 2, k / 2), h: h, b: b, t: (a, t), name: "ca")
  let normAOut = norm(aOut).to(.Float16)
  aOut = aOut + audioAttn2(normAOut, rotA, ca, rotAC).to(of: aOut)
  let vxNorm3 = norm(out)
  let axNorm3 = norm(aOut)
  let (audioToVideoAttnMapper, audioToVideoAttn) = LTX2CrossAttention(
    prefix: "\(prefix).audio_to_video_attn", k: (k, k / 2, k / 2), h: h, b: b, t: (hw, a),
    name: "ax")
  let caScaleShiftTimesteps = (0..<4).map { _ in Input() }
  let caGateTimesteps = Input()
  let audioToVideoAttnModulations = (0..<5).map {
    if $0 < 2 {
      return Parameter<Float>(
        .GPU(0), .HWC(1, 1, k * h), name: "audio_to_video_attn_ada_ln_\($0)")
    } else if $0 < 4 {
      return Parameter<Float>(
        .GPU(0), .HWC(1, 1, k / 2 * h), name: "audio_to_video_attn_ada_ln_\($0)")
    } else {
      return Parameter<Float>(
        .GPU(0), .HWC(1, 1, k * h), name: "audio_to_video_attn_ada_ln_\($0)")
    }
  }
  let vxScaled =
    vxNorm3 .* (1 + (audioToVideoAttnModulations[1] + caScaleShiftTimesteps[1]))
    + (audioToVideoAttnModulations[0] + caScaleShiftTimesteps[0])
  let axScaled =
    axNorm3 .* (1 + (audioToVideoAttnModulations[3] + caScaleShiftTimesteps[3]))
    + (audioToVideoAttnModulations[2] + caScaleShiftTimesteps[2])
  out =
    out + audioToVideoAttn(vxScaled.to(.Float16), rotCX, axScaled.to(.Float16), rotA).to(of: out)
    .* (audioToVideoAttnModulations[4] + caGateTimesteps)
  let (videoToAudioAttnMapper, videoToAudioAttn) = LTX2CrossAttention(
    prefix: "\(prefix).video_to_audio_attn", k: (k / 2, k / 2, k), h: h, b: b, t: (a, hw),
    name: "xa")
  let audioCaScaleShiftTimesteps = (0..<4).map { _ in Input() }
  let audioCaGateTimesteps = Input()
  let videoToAudioAttnModulations = (0..<5).map {
    if $0 < 2 {
      return Parameter<Float>(
        .GPU(0), .HWC(1, 1, k * h), name: "video_to_audio_attn_ada_ln_\($0)")
    } else {
      return Parameter<Float>(
        .GPU(0), .HWC(1, 1, k / 2 * h), name: "video_to_audio_attn_ada_ln_\($0)")
    }
  }
  let audioVxScaled =
    vxNorm3 .* (1 + (videoToAudioAttnModulations[1] + audioCaScaleShiftTimesteps[1]))
    + (videoToAudioAttnModulations[0] + audioCaScaleShiftTimesteps[0])
  let audioAxScaled =
    axNorm3 .* (1 + (videoToAudioAttnModulations[3] + audioCaScaleShiftTimesteps[3]))
    + (videoToAudioAttnModulations[2] + audioCaScaleShiftTimesteps[2])
  aOut =
    aOut
    + videoToAudioAttn(audioAxScaled.to(.Float16), rotA, audioVxScaled.to(.Float16), rotCX).to(
      of: aOut)
    .* (videoToAudioAttnModulations[4] + audioCaGateTimesteps)
  // Now attention done, do MLP.
  let (xLinear1, xOutProjection, xFF) = FeedForward(
    hiddenSize: 4096, intermediateSize: 4096 * 4, name: "x")
  let lastVxScaled =
    norm(out) .* (1 + (attn1Modulations[4] + timesteps[4])) + (attn1Modulations[3] + timesteps[3])
  out = out + xFF(lastVxScaled.to(.Float16)).to(of: out) .* (attn1Modulations[5] + timesteps[5])
  let lastAxScaled =
    norm(aOut) .* (1 + (audioAttn1Modulations[4] + audioTimesteps[4]))
    + (audioAttn1Modulations[3] + audioTimesteps[3])
  let (audioLinear1, audioOutProjection, audioFF) = FeedForward(
    hiddenSize: 2048, intermediateSize: 2048 * 4, name: "a")
  aOut = aOut + audioFF(lastAxScaled.to(.Float16)).to(of: aOut)
    .* (audioAttn1Modulations[5] + audioTimesteps[5])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).scale_shift_table"] = ModelWeightElement(
      (0..<6).map { attn1Modulations[$0].weight.name })
    mapping.merge(attn1Mapper(format)) { v, _ in v }
    mapping.merge(attn2Mapper(format)) { v, _ in v }
    mapping["\(prefix).audio_scale_shift_table"] = ModelWeightElement(
      (0..<6).map { audioAttn1Modulations[$0].weight.name })
    mapping.merge(audioAttn1Mapper(format)) { v, _ in v }
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
  var inputs: [Input] = [vx, rot, cv, rotC, ax, rotA, ca, rotAC, rotCX]
  inputs.append(contentsOf: timesteps + audioTimesteps)
  inputs.append(contentsOf: caScaleShiftTimesteps + [caGateTimesteps])
  inputs.append(contentsOf: audioCaScaleShiftTimesteps + [audioCaGateTimesteps])
  return (mapper, Model(inputs, [out, aOut]))
}

private func LTX2AdaLNSingle(
  prefix: String, channels: Int, count: Int, outputEmbedding: Bool, name: String, t: Input
) -> (
  ModelWeightMapper, Model.IO?, [Model.IO]
) {
  let (tMlp0, tMlp2, tEmbedder) = MLPEmbedder(channels: channels, name: name)
  let adaLNSingles = (0..<count).map { Dense(count: channels, name: "\(name)_adaln_single_\($0)") }
  var tOut = tEmbedder(t).reshaped([1, 1, channels])
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

func LTX2(b: Int, h: Int, w: Int) -> (ModelWeightMapper, Model) {
  let x = Input()
  let rot = Input()
  let rotC = Input()
  let rotA = Input()
  let rotAC = Input()
  let rotCX = Input()
  let xEmbedder = Dense(count: 4096, name: "x_embedder")
  let (contextMlp0, contextMlp2, contextEmbedder) = GELUMLPEmbedder(channels: 4096, name: "context")
  var out = xEmbedder(x).to(.Float32)
  let txt = Input()
  let txtOut = contextEmbedder(txt)
  let a = Input()
  let aEmbedder = Dense(count: 2048, name: "a_embedder")
  let (aContextMlp0, aContextMlp2, aContextEmbedder) = GELUMLPEmbedder(
    channels: 2048, name: "a_context")
  var aOut = aEmbedder(a).to(.Float32)
  let aTxt = Input()
  let aTxtOut = aContextEmbedder(aTxt)
  let t = Input()
  let (txMapper, txEmb, txEmbChunks) = LTX2AdaLNSingle(
    prefix: "adaln_single", channels: 4096, count: 6, outputEmbedding: true, name: "tx", t: t)
  let (taMapper, taEmb, taEmbChunks) = LTX2AdaLNSingle(
    prefix: "audio_adaln_single", channels: 2048, count: 6, outputEmbedding: true, name: "ta", t: t)
  let (caMapper, _, tcxEmbChunks) = LTX2AdaLNSingle(
    prefix: "av_ca_video_scale_shift_adaln_single", channels: 4096, count: 4,
    outputEmbedding: false, name: "tcx", t: t)
  let (audioCaMapper, _, tcaEmbChunks) = LTX2AdaLNSingle(
    prefix: "av_ca_audio_scale_shift_adaln_single", channels: 2048, count: 4,
    outputEmbedding: false, name: "tca", t: t)
  let (gateMapper, _, a2vEmbChunks) = LTX2AdaLNSingle(
    prefix: "av_ca_a2v_gate_adaln_single", channels: 4096, count: 1, outputEmbedding: false,
    name: "a2v", t: t)
  let (audioGateMapper, _, v2aEmbChunks) = LTX2AdaLNSingle(
    prefix: "av_ca_v2a_gate_adaln_single", channels: 2048, count: 1, outputEmbedding: false,
    name: "v2a", t: t)
  var mappers = [ModelWeightMapper]()
  for i in 0..<48 {
    let (mapper, block) = LTX2TransformerBlock(
      prefix: "transformer_blocks.\(i)", k: 128, h: 32, b: 1, t: 1024, hw: 6144, a: 121,
      intermediateSize: 0)
    let blockOut = block(
      out, rot, txtOut, rotC, aOut, rotA, aTxtOut, rotAC, rotCX,
      txEmbChunks[0], txEmbChunks[1], txEmbChunks[2], txEmbChunks[3], txEmbChunks[4],
      txEmbChunks[5],
      taEmbChunks[0], taEmbChunks[1], taEmbChunks[2], taEmbChunks[3], taEmbChunks[4],
      taEmbChunks[5],
      tcxEmbChunks[1], tcxEmbChunks[0], tcaEmbChunks[1], tcaEmbChunks[0], a2vEmbChunks[0],
      tcxEmbChunks[3], tcxEmbChunks[2], tcaEmbChunks[3], tcaEmbChunks[2], v2aEmbChunks[0])
    mappers.append(mapper)
    out = blockOut[0]
    aOut = blockOut[1]
  }
  let scaleShiftModulations = (0..<2).map {
    Parameter<Float>(.GPU(0), .HWC(1, 1, 4096), name: "norm_out_ada_ln_\($0)")
  }
  let normOut = RMSNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  if let txEmb = txEmb {
    out = normOut(out) .* (1 + (scaleShiftModulations[1] + txEmb))
      + (scaleShiftModulations[0] + txEmb)
  }
  let projOut = Dense(count: 128, name: "proj_out")
  out = projOut(out.to(.Float16))
  let audioScaleShiftModulations = (0..<2).map {
    Parameter<Float>(.GPU(0), .HWC(1, 1, 2048), name: "audio_norm_out_ada_ln_\($0)")
  }
  if let taEmb = taEmb {
    aOut = normOut(aOut) .* (1 + (audioScaleShiftModulations[1] + taEmb))
      + (audioScaleShiftModulations[0] + taEmb)
  }
  let audioProjOut = Dense(count: 128, name: "audio_proj_out")
  aOut = audioProjOut(aOut.to(.Float16))
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["patchify_proj.weight"] = [xEmbedder.weight.name]
    mapping["patchify_proj.bias"] = [xEmbedder.bias.name]
    mapping["caption_projection.linear_1.weight"] = [contextMlp0.weight.name]
    mapping["caption_projection.linear_1.bias"] = [contextMlp0.bias.name]
    mapping["caption_projection.linear_2.weight"] = [contextMlp2.weight.name]
    mapping["caption_projection.linear_2.bias"] = [contextMlp2.bias.name]
    mapping["audio_patchify_proj.weight"] = [aEmbedder.weight.name]
    aEmbedder.weight.to(.unifiedMemory)
    mapping["audio_patchify_proj.bias"] = [aEmbedder.bias.name]
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
    mapping["proj_out.weight"] = [projOut.weight.name]
    mapping["proj_out.bias"] = [projOut.bias.name]
    mapping["audio_proj_out.weight"] = [audioProjOut.weight.name]
    mapping["audio_proj_out.bias"] = [audioProjOut.bias.name]
    return mapping
  }
  return (mapper, Model([x, a, txt, aTxt, t, rot, rotC, rotA, rotAC, rotCX], [out, aOut]))
}
