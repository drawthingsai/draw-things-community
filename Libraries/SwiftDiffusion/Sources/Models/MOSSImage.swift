import DiffusionMappings
import Foundation
import NNC

/// Rotary embedding for the image followed by the text, one row per prompt. Each token has 4 axes
/// (unused, text, y, x) of 32 channels. Image tokens take the text length on the text axis and
/// centered coordinates on the y / x axes, thus, each prompt has its own rotary embedding.
public func MOSSImageRotaryPositionEmbedding(
  height: Int, width: Int, tokenLengths: [Int]
) -> Tensor<Float> {
  let maxTokenLength = tokenLengths.max() ?? 0
  var rotTensor = Tensor<Float>(
    .CPU, .NHWC(tokenLengths.count, height * width + maxTokenLength, 1, 128))
  func setPositions(_ n: Int, _ i: Int, _ positions: [Double]) {
    for (axis, position) in positions.enumerated() {
      for k in 0..<16 {
        let theta = position / pow(10_000, Double(k) / 16)
        rotTensor[n, i, 0, (k + axis * 16) * 2] = Float(cos(theta))
        rotTensor[n, i, 0, (k + axis * 16) * 2 + 1] = Float(sin(theta))
      }
    }
  }
  for (n, tokenLength) in tokenLengths.enumerated() {
    for y in 0..<height {
      for x in 0..<width {
        setPositions(
          n, y * width + x,
          [
            0, Double(tokenLength), Double(y) - Double(height - 1) / 2,
            Double(x) - Double(width - 1) / 2,
          ])
      }
    }
    for i in 0..<maxTokenLength {
      setPositions(n, height * width + i, [0, Double(i), 0, 0])
    }
  }
  return rotTensor
}

// 4 * tanh(gate / 4) * sigmoid(gate) * 25 * tanh(hidden / 25), both factors are soft capped.
private func MOSSImageSiTU(_ hidden: Model.IO, _ gate: Model.IO) -> Model.IO {
  return (gate.tanh(softCap: 4) .* gate.sigmoid()) .* hidden.tanh(softCap: 25)
}

private func MOSSImageAttention(
  queries: Model.IO, keys: Model.IO, values: Model.IO, b: Int, t: Int, h: Int, k: Int,
  usesFlashAttention: FlashAttentionLevel
) -> Model.IO {
  var queries = queries
  var keys = keys
  var values = values
  switch usesFlashAttention {
  case .none:
    keys = keys.transposed(1, 2)
    queries = ((1.0 / Float(k).squareRoot()) * queries).transposed(1, 2)
    values = values.transposed(1, 2)
    if b * h <= 256 {
      var outs = [Model.IO]()
      for i in 0..<(b * h) {
        let key = keys.reshaped([1, t, k], offset: [i, 0, 0], strides: [t * k, k, 1])
        let query = queries.reshaped([1, t, k], offset: [i, 0, 0], strides: [t * k, k, 1])
        let value = values.reshaped([1, t, k], offset: [i, 0, 0], strides: [t * k, k, 1])
        var dot = Matmul(transposeB: (1, 2))(query, key)
        if let last = outs.last {
          dot.add(dependencies: [last])
        }
        dot = dot.reshaped([t, t])
        dot = dot.softmax()
        dot = dot.reshaped([1, t, t])
        outs.append(dot * value)
      }
      let out = Concat(axis: 0)(outs)
      return out.reshaped([b, h, t, k]).transposed(1, 2).reshaped([b, t, h * k])
    } else {
      var dot = Matmul(transposeB: (2, 3))(queries, keys)
      dot = dot.reshaped([b * h * t, t])
      dot = dot.softmax()
      dot = dot.reshaped([b, h, t, t])
      let out = dot * values
      return out.reshaped([b, h, t, k]).transposed(1, 2).reshaped([b, t, h * k])
    }
  case .scale1:
    queries = (1.0 / Float(k).squareRoot()) * queries
    let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
    return scaledDotProductAttention(queries, keys, values).reshaped([b, t, k * h])
  case .scaleMerged, .quantized:
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1.0 / Float(k).squareRoot(), useHadamard: true,
      flags: usesFlashAttention == .quantized ? [.Int8] : [.Float16])
    return scaledDotProductAttention(queries, keys, values).reshaped([b, t, k * h])
  }
}

// Prompts are packed one after another as segments, each attends only to itself with text
// positions starting from 0, thus, rot only needs the longest segment.
private func MOSSImageTextBlock<T: TensorNumeric>(
  _ dataType: T.Type, prefix: String, k: Int, h: Int, b: Int, segments: [Int],
  intermediateSize: Int, usesFlashAttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let rot = Input()
  let t = segments.reduce(0, +)
  let maxSegment = segments.max() ?? 0
  let norm1 = RMSNorm(epsilon: 1e-6, axis: [2], name: "context_norm1")
  let xOut = norm1(x).to(T.dataType)
  let toQueries = Dense(count: k * h, name: "context_q")
  let toKeys = Dense(count: k * h, name: "context_k")
  let toValues = Dense(count: k * h, name: "context_v")
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "context_norm_q")
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "context_norm_k")
  let queries = normQ(toQueries(xOut).reshaped([b, t, h, k]))
  let keys = normK(toKeys(xOut).reshaped([b, t, h, k]))
  let values = toValues(xOut).reshaped([b, t, h, k])
  var offset = 0
  var outs = [Model.IO]()
  for segment in segments {
    func slice(_ x: Model.IO) -> Model.IO {
      return x.reshaped(
        [b, segment, h, k], offset: [0, offset, 0, 0], strides: [t * h * k, h * k, k, 1]
      ).contiguous()
    }
    let segmentRot = rot.reshaped(
      [1, segment, 1, k], strides: [maxSegment * k, k, k, 1]
    ).contiguous()
    outs.append(
      MOSSImageAttention(
        queries: Functional.cmul(left: slice(queries), right: segmentRot),
        keys: Functional.cmul(left: slice(keys), right: segmentRot), values: slice(values), b: b,
        t: segment, h: h, k: k, usesFlashAttention: usesFlashAttention))
    offset += segment
  }
  let attention = outs.count > 1 ? Concat(axis: 1)(outs) : outs[0]
  let unifyheads = Dense(count: k * h, name: "context_o")
  var out = x + unifyheads(attention).to(.Float32)
  let norm2 = RMSNorm(epsilon: 1e-6, axis: [2], name: "context_norm2")
  let ffnIn = norm2(out).to(T.dataType)
  let ffnGate = Dense(count: intermediateSize, name: "context_ffn_gate_proj")
  let ffnUp = Dense(count: intermediateSize, name: "context_ffn_up_proj")
  let ffnDown = Dense(count: k * h, name: "context_ffn_down_proj")
  out = out + ffnDown(MOSSImageSiTU(ffnUp(ffnIn), ffnGate(ffnIn))).to(.Float32)
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).norm1.weight"] = [norm1.weight.name]
    mapping["\(prefix).norm2.weight"] = [norm2.weight.name]
    mapping["\(prefix).q_norm.weight"] = ModelWeightElement(
      [normQ.weight.name], interleavedIndices: [0], numberOfHeads: 1, headDimension: k)
    mapping["\(prefix).k_norm.weight"] = ModelWeightElement(
      [normK.weight.name], interleavedIndices: [0], numberOfHeads: 1, headDimension: k)
    mapping["\(prefix).qkv.weight"] = ModelWeightElement(
      [toQueries.weight.name, toKeys.weight.name, toValues.weight.name],
      interleavedIndices: [0, 1], numberOfHeads: h, headDimension: k)
    mapping["\(prefix).qkv.bias"] = ModelWeightElement(
      [toQueries.bias.name, toKeys.bias.name, toValues.bias.name],
      interleavedIndices: [0, 1], numberOfHeads: h, headDimension: k)
    mapping["\(prefix).proj.weight"] = [unifyheads.weight.name]
    mapping["\(prefix).proj.bias"] = [unifyheads.bias.name]
    mapping["\(prefix).ffn_gate.weight"] = [ffnGate.weight.name]
    mapping["\(prefix).ffn_gate.bias"] = [ffnGate.bias.name]
    mapping["\(prefix).ffn_up.weight"] = [ffnUp.weight.name]
    mapping["\(prefix).ffn_up.bias"] = [ffnUp.bias.name]
    mapping["\(prefix).ffn_down.weight"] = [ffnDown.weight.name]
    mapping["\(prefix).ffn_down.bias"] = [ffnDown.bias.name]
    return mapping
  }
  return (mapper, Model([x, rot], [out]))
}

private func MOSSImageBlock<T: TensorNumeric>(
  _ dataType: T.Type, prefix: String, k: Int, h: Int, b: Int, t: Int, outputLength: Int,
  intermediateSize: Int, usesFlashAttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let rot = Input()
  // Attention shift, scale, gate, then FFN shift, scale, gate. The scales already have 1 added.
  let chunks = (0..<6).map { _ in Input() }
  let norm1 = RMSNorm(epsilon: 1e-6, axis: [2], name: "x_norm1")
  var xOut = (norm1(x) .* chunks[1] + chunks[0]).to(T.dataType)
  let toQueries = Dense(count: k * h, noBias: true, name: "x_q")
  let toKeys = Dense(count: k * h, noBias: true, name: "x_k")
  let toValues = Dense(count: k * h, noBias: true, name: "x_v")
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_q")
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_k")
  var queries = normQ(toQueries(xOut).reshaped([b, t, h, k]))
  var keys = normK(toKeys(xOut).reshaped([b, t, h, k]))
  let values = toValues(xOut).reshaped([b, t, h, k])
  queries = Functional.cmul(left: queries, right: rot)
  keys = Functional.cmul(left: keys, right: rot)
  var out = MOSSImageAttention(
    queries: queries, keys: keys, values: values, b: b, t: t, h: h, k: k,
    usesFlashAttention: usesFlashAttention)
  var xIn: Model.IO = x
  if outputLength < t {
    // Only the image tokens are needed from the last block.
    out = out.reshaped([b, outputLength, h * k], strides: [t * h * k, h * k, 1]).contiguous()
    xIn = x.reshaped([b, outputLength, h * k], strides: [t * h * k, h * k, 1]).contiguous()
  }
  let unifyheads = Dense(count: k * h, noBias: true, name: "x_o")
  out = xIn + unifyheads(out).to(.Float32) .* chunks[2]
  let norm2 = RMSNorm(epsilon: 1e-6, axis: [2], name: "x_norm2")
  xOut = (norm2(out) .* chunks[4] + chunks[3]).to(T.dataType)
  let ffnUp = Dense(count: intermediateSize, noBias: true, name: "ffn_up_proj")
  let ffnGate = Dense(count: intermediateSize, noBias: true, name: "ffn_gate_proj")
  let ffnDown = Dense(count: k * h, noBias: true, name: "ffn_down_proj")
  out = out + ffnDown(MOSSImageSiTU(ffnUp(xOut), ffnGate(xOut))).to(.Float32) .* chunks[5]
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).norm1.weight"] = [norm1.weight.name]
    mapping["\(prefix).norm2.weight"] = [norm2.weight.name]
    mapping["\(prefix).q_norm.weight"] = ModelWeightElement(
      [normQ.weight.name], interleavedIndices: [0], numberOfHeads: 1, headDimension: k)
    mapping["\(prefix).k_norm.weight"] = ModelWeightElement(
      [normK.weight.name], interleavedIndices: [0], numberOfHeads: 1, headDimension: k)
    mapping["\(prefix).proj.q.weight"] = ModelWeightElement(
      [toQueries.weight.name], interleavedIndices: [0], numberOfHeads: h, headDimension: k)
    mapping["\(prefix).proj.k.weight"] = ModelWeightElement(
      [toKeys.weight.name], interleavedIndices: [0], numberOfHeads: h, headDimension: k)
    mapping["\(prefix).proj.v.weight"] = [toValues.weight.name]
    mapping["\(prefix).proj.o.weight"] = [unifyheads.weight.name]
    // The packed projection is the hidden (up) half followed by the gate half.
    mapping["\(prefix).proj.ffn_up_gate.weight"] = [ffnUp.weight.name, ffnGate.weight.name]
    mapping["\(prefix).proj.ffn_down.weight"] = [ffnDown.weight.name]
    return mapping
  }
  return (mapper, Model([x, rot] + chunks, [out]))
}

/// The main MOSS Image transformer. Inputs are the latent, the rotary embedding, the refined text
/// from `MOSSImageFixed`, 6 shared modulation chunks, the head shift / scale, and one rotary
/// embedding per skippable block. With `skipsLayers`, these rotary embeddings are 0-length ([0]),
/// the skippable blocks run on 0-length tensors and leave every token unchanged.
public func MOSSImage<T: TensorNumeric>(
  _ dataType: T.Type, batchSize: Int, tokenLength: Int, height: Int, width: Int, channels: Int,
  layers: Int, skippableLayers: [Int], skipsLayers: Bool, usesFlashAttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  precondition(height % 2 == 0 && width % 2 == 0)
  let x = Input()
  let rot = Input()
  let contextIn = Input()
  let chunks = (0..<6).map { _ in Input() }
  let shift = Input()
  let scale = Input()
  let skippableRots = skippableLayers.map { _ in Input() }
  let h = height / 2
  let w = width / 2
  let k = 128
  let heads = channels / k
  let xEmbedder = Dense(count: channels, name: "x_embedder")
  var out = xEmbedder(
    x.reshaped([batchSize, h, 2, w, 2, 32]).permuted(0, 1, 3, 5, 2, 4).contiguous().reshaped(
      [batchSize, h * w, 32 * 2 * 2], format: .NHWC)
  ).to(.Float32)
  let sequenceLength = h * w + tokenLength
  out = Functional.concat(axis: 1, out, contextIn)
  var mappers = [ModelWeightMapper]()
  for i in 0..<layers {
    let skippableIndex = skippableLayers.firstIndex(of: i)
    let (mapper, block) = MOSSImageBlock(
      T.self, prefix: "layers.\(i)", k: k, h: heads, b: batchSize, t: sequenceLength,
      outputLength: i == layers - 1 ? h * w : sequenceLength, intermediateSize: 8_192,
      usesFlashAttention: usesFlashAttention)
    if let skippableIndex = skippableIndex {
      // Both branches build the same graph so the model builder keeps the weights when switching
      // between them. A skipped block runs on 0-length tensors, and its output is dropped as the
      // 0-length part of the concat.
      let blockOut = block(
        [
          out.reshaped(skipsLayers ? [0] : [batchSize, sequenceLength, channels]),
          skippableRots[skippableIndex],
        ] + chunks)
      out = Functional.concat(
        axis: 1, blockOut.reshaped(skipsLayers ? [0] : [batchSize, sequenceLength, channels]),
        out.reshaped(skipsLayers ? [batchSize, sequenceLength, channels] : [0]))
    } else {
      out = block([out, rot] + chunks)
    }
    mappers.append(mapper)
  }
  let normFinal = RMSNorm(epsilon: 1e-6, axis: [2], name: "final_norm")
  out = (normFinal(out) .* scale + shift).to(T.dataType)
  let projOut = Dense(count: 2 * 2 * 32, noBias: true, name: "final_linear")
  out = projOut(out)
  // Unpatchify
  out = out.reshaped([batchSize, h, w, 32, 2, 2]).permuted(0, 1, 4, 2, 5, 3).contiguous().reshaped([
    batchSize, h * 2, w * 2, 32,
  ])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping["video_embedder.weight"] = [xEmbedder.weight.name]
    mapping["video_embedder.bias"] = [xEmbedder.bias.name]
    mapping["final_norm_video.weight"] = [normFinal.weight.name]
    mapping["final_linear_video.weight"] = [projOut.weight.name]
    return mapping
  }
  return (
    mapper,
    Model([x, rot, contextIn] + chunks + [shift, scale] + skippableRots, [out])
  )
}

/// The text refiner, the time embedding and the modulations shared by every block. Prompts are
/// packed one after another in the context (one segment per prompt) with the text rotary embedding
/// of the longest prompt. Outputs are the refined context in FP32, then the 6 shared modulation
/// chunks and the head shift / scale, each is [timesteps, 1, channels] in FP32 with 1 already added
/// to the scales.
public func MOSSImageFixed<T: TensorNumeric>(
  _ dataType: T.Type, batchSize: Int, tokenLengths: [Int], timesteps: Int, channels: Int,
  usesFlashAttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  let contextIn = Input()
  let rot = Input()
  let t = Input()
  let contextEmbedder = Dense(count: channels, name: "context_embedder")
  var context = contextEmbedder(contextIn).to(.Float32)
  var mappers = [ModelWeightMapper]()
  for i in 0..<2 {
    let (mapper, block) = MOSSImageTextBlock(
      T.self, prefix: "text_embedder.blocks.\(i)", k: 128, h: channels / 128, b: batchSize,
      segments: tokenLengths, intermediateSize: 8_192, usesFlashAttention: usesFlashAttention)
    context = block(context, rot)
    mappers.append(mapper)
  }
  let tMlp0 = Dense(count: channels, name: "time_embedder_0")
  let tMlp2 = Dense(count: channels, name: "time_embedder_2")
  // The output layer is kept in FP32 to avoid FP16 underflow of its weights.
  let tOut = tMlp2(tMlp0(t).swish().to(.Float32)).reshaped([timesteps, 1, channels])
  let tIn = tOut.to(.Float16)
  let adaLNs = (0..<6).map { Dense(count: channels, name: "x_ada_ln_\($0)") }
  var chunks = adaLNs.map { $0(tIn).to(.Float32) }
  chunks[1] = 1 + chunks[1]
  chunks[4] = 1 + chunks[4]
  let shift = Dense(count: channels, name: "ada_ln_1")
  let scale = Dense(count: channels, name: "ada_ln_0")
  chunks.append(shift(tIn).to(.Float32))
  chunks.append(1 + scale(tIn).to(.Float32))
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping["text_embedder.proj.weight"] = [contextEmbedder.weight.name]
    mapping["text_embedder.proj.bias"] = [contextEmbedder.bias.name]
    mapping["time_embedding.0.weight"] = [tMlp0.weight.name]
    mapping["time_embedding.0.bias"] = [tMlp0.bias.name]
    mapping["time_embedding.2.weight"] = [tMlp2.weight.name]
    mapping["time_embedding.2.bias"] = [tMlp2.bias.name]
    mapping["shared_block_modulation.weight"] = ModelWeightElement(
      adaLNs.map { $0.weight.name })
    mapping["shared_block_modulation.bias"] = ModelWeightElement(adaLNs.map { $0.bias.name })
    mapping["video_head_modulation.weight"] = [shift.weight.name, scale.weight.name]
    mapping["video_head_modulation.bias"] = [shift.bias.name, scale.bias.name]
    return mapping
  }
  return (mapper, Model([contextIn, rot, t], [context] + chunks))
}

private func LoRAMOSSImageTextBlock<T: TensorNumeric>(
  _ dataType: T.Type, prefix: String, k: Int, h: Int, b: Int, segments: [Int],
  intermediateSize: Int, usesFlashAttention: FlashAttentionLevel, layerIndex: Int,
  configuration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let rot = Input()
  let t = segments.reduce(0, +)
  let maxSegment = segments.max() ?? 0
  let norm1 = RMSNorm(epsilon: 1e-6, axis: [2], name: "context_norm1")
  let xOut = norm1(x).to(T.dataType)
  let toQueries = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "context_q")
  let toKeys = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "context_k")
  let toValues = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "context_v")
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "context_norm_q")
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "context_norm_k")
  let queries = normQ(toQueries(xOut).reshaped([b, t, h, k]))
  let keys = normK(toKeys(xOut).reshaped([b, t, h, k]))
  let values = toValues(xOut).reshaped([b, t, h, k])
  var offset = 0
  var outs = [Model.IO]()
  for segment in segments {
    func slice(_ x: Model.IO) -> Model.IO {
      return x.reshaped(
        [b, segment, h, k], offset: [0, offset, 0, 0], strides: [t * h * k, h * k, k, 1]
      ).contiguous()
    }
    let segmentRot = rot.reshaped(
      [1, segment, 1, k], strides: [maxSegment * k, k, k, 1]
    ).contiguous()
    outs.append(
      MOSSImageAttention(
        queries: Functional.cmul(left: slice(queries), right: segmentRot),
        keys: Functional.cmul(left: slice(keys), right: segmentRot), values: slice(values), b: b,
        t: segment, h: h, k: k, usesFlashAttention: usesFlashAttention))
    offset += segment
  }
  let attention = outs.count > 1 ? Concat(axis: 1)(outs) : outs[0]
  let unifyheads = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "context_o")
  var out = x + unifyheads(attention).to(.Float32)
  let norm2 = RMSNorm(epsilon: 1e-6, axis: [2], name: "context_norm2")
  let ffnIn = norm2(out).to(T.dataType)
  let ffnGate = LoRADense(
    count: intermediateSize, configuration: configuration, index: layerIndex,
    name: "context_ffn_gate_proj")
  let ffnUp = LoRADense(
    count: intermediateSize, configuration: configuration, index: layerIndex,
    name: "context_ffn_up_proj")
  let ffnDown = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "context_ffn_down_proj")
  out = out + ffnDown(MOSSImageSiTU(ffnUp(ffnIn), ffnGate(ffnIn))).to(.Float32)
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).norm1.weight"] = [norm1.weight.name]
    mapping["\(prefix).norm2.weight"] = [norm2.weight.name]
    mapping["\(prefix).q_norm.weight"] = ModelWeightElement(
      [normQ.weight.name], interleavedIndices: [0], numberOfHeads: 1, headDimension: k)
    mapping["\(prefix).k_norm.weight"] = ModelWeightElement(
      [normK.weight.name], interleavedIndices: [0], numberOfHeads: 1, headDimension: k)
    mapping["\(prefix).qkv.weight"] = ModelWeightElement(
      [toQueries.weight.name, toKeys.weight.name, toValues.weight.name],
      interleavedIndices: [0, 1], numberOfHeads: h, headDimension: k)
    mapping["\(prefix).qkv.bias"] = ModelWeightElement(
      [toQueries.bias.name, toKeys.bias.name, toValues.bias.name],
      interleavedIndices: [0, 1], numberOfHeads: h, headDimension: k)
    mapping["\(prefix).proj.weight"] = [unifyheads.weight.name]
    mapping["\(prefix).proj.bias"] = [unifyheads.bias.name]
    mapping["\(prefix).ffn_gate.weight"] = [ffnGate.weight.name]
    mapping["\(prefix).ffn_gate.bias"] = [ffnGate.bias.name]
    mapping["\(prefix).ffn_up.weight"] = [ffnUp.weight.name]
    mapping["\(prefix).ffn_up.bias"] = [ffnUp.bias.name]
    mapping["\(prefix).ffn_down.weight"] = [ffnDown.weight.name]
    mapping["\(prefix).ffn_down.bias"] = [ffnDown.bias.name]
    return mapping
  }
  return (mapper, Model([x, rot], [out]))
}

private func LoRAMOSSImageBlock<T: TensorNumeric>(
  _ dataType: T.Type, prefix: String, k: Int, h: Int, b: Int, t: Int, outputLength: Int,
  intermediateSize: Int, usesFlashAttention: FlashAttentionLevel, layerIndex: Int,
  configuration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let rot = Input()
  // Attention shift, scale, gate, then FFN shift, scale, gate. The scales already have 1 added.
  let chunks = (0..<6).map { _ in Input() }
  let norm1 = RMSNorm(epsilon: 1e-6, axis: [2], name: "x_norm1")
  var xOut = (norm1(x) .* chunks[1] + chunks[0]).to(T.dataType)
  let toQueries = LoRADense(
    count: k * h, configuration: configuration, noBias: true, index: layerIndex, name: "x_q")
  let toKeys = LoRADense(
    count: k * h, configuration: configuration, noBias: true, index: layerIndex, name: "x_k")
  let toValues = LoRADense(
    count: k * h, configuration: configuration, noBias: true, index: layerIndex, name: "x_v")
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_q")
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_k")
  var queries = normQ(toQueries(xOut).reshaped([b, t, h, k]))
  var keys = normK(toKeys(xOut).reshaped([b, t, h, k]))
  let values = toValues(xOut).reshaped([b, t, h, k])
  queries = Functional.cmul(left: queries, right: rot)
  keys = Functional.cmul(left: keys, right: rot)
  var out = MOSSImageAttention(
    queries: queries, keys: keys, values: values, b: b, t: t, h: h, k: k,
    usesFlashAttention: usesFlashAttention)
  var xIn: Model.IO = x
  if outputLength < t {
    // Only the image tokens are needed from the last block.
    out = out.reshaped([b, outputLength, h * k], strides: [t * h * k, h * k, 1]).contiguous()
    xIn = x.reshaped([b, outputLength, h * k], strides: [t * h * k, h * k, 1]).contiguous()
  }
  let unifyheads = LoRADense(
    count: k * h, configuration: configuration, noBias: true, index: layerIndex, name: "x_o")
  out = xIn + unifyheads(out).to(.Float32) .* chunks[2]
  let norm2 = RMSNorm(epsilon: 1e-6, axis: [2], name: "x_norm2")
  xOut = (norm2(out) .* chunks[4] + chunks[3]).to(T.dataType)
  let ffnUp = LoRADense(
    count: intermediateSize, configuration: configuration, noBias: true, index: layerIndex,
    name: "ffn_up_proj")
  let ffnGate = LoRADense(
    count: intermediateSize, configuration: configuration, noBias: true, index: layerIndex,
    name: "ffn_gate_proj")
  let ffnDown = LoRADense(
    count: k * h, configuration: configuration, noBias: true, index: layerIndex,
    name: "ffn_down_proj")
  out = out + ffnDown(MOSSImageSiTU(ffnUp(xOut), ffnGate(xOut))).to(.Float32) .* chunks[5]
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).norm1.weight"] = [norm1.weight.name]
    mapping["\(prefix).norm2.weight"] = [norm2.weight.name]
    mapping["\(prefix).q_norm.weight"] = ModelWeightElement(
      [normQ.weight.name], interleavedIndices: [0], numberOfHeads: 1, headDimension: k)
    mapping["\(prefix).k_norm.weight"] = ModelWeightElement(
      [normK.weight.name], interleavedIndices: [0], numberOfHeads: 1, headDimension: k)
    mapping["\(prefix).proj.q.weight"] = ModelWeightElement(
      [toQueries.weight.name], interleavedIndices: [0], numberOfHeads: h, headDimension: k)
    mapping["\(prefix).proj.k.weight"] = ModelWeightElement(
      [toKeys.weight.name], interleavedIndices: [0], numberOfHeads: h, headDimension: k)
    mapping["\(prefix).proj.v.weight"] = [toValues.weight.name]
    mapping["\(prefix).proj.o.weight"] = [unifyheads.weight.name]
    // The packed projection is the hidden (up) half followed by the gate half.
    mapping["\(prefix).proj.ffn_up_gate.weight"] = [ffnUp.weight.name, ffnGate.weight.name]
    mapping["\(prefix).proj.ffn_down.weight"] = [ffnDown.weight.name]
    return mapping
  }
  return (mapper, Model([x, rot] + chunks, [out]))
}

public func LoRAMOSSImage<T: TensorNumeric>(
  _ dataType: T.Type, batchSize: Int, tokenLength: Int, height: Int, width: Int, channels: Int,
  layers: Int, skippableLayers: [Int], skipsLayers: Bool, usesFlashAttention: FlashAttentionLevel,
  LoRAConfiguration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  precondition(height % 2 == 0 && width % 2 == 0)
  let x = Input()
  let rot = Input()
  let contextIn = Input()
  let chunks = (0..<6).map { _ in Input() }
  let shift = Input()
  let scale = Input()
  let skippableRots = skippableLayers.map { _ in Input() }
  let h = height / 2
  let w = width / 2
  let k = 128
  let heads = channels / k
  let xEmbedder = LoRADense(
    count: channels, configuration: LoRAConfiguration, index: 0, name: "x_embedder")
  var out = xEmbedder(
    x.reshaped([batchSize, h, 2, w, 2, 32]).permuted(0, 1, 3, 5, 2, 4).contiguous().reshaped(
      [batchSize, h * w, 32 * 2 * 2], format: .NHWC)
  ).to(.Float32)
  let sequenceLength = h * w + tokenLength
  out = Functional.concat(axis: 1, out, contextIn)
  var mappers = [ModelWeightMapper]()
  for i in 0..<layers {
    let skippableIndex = skippableLayers.firstIndex(of: i)
    let (mapper, block) = LoRAMOSSImageBlock(
      T.self, prefix: "layers.\(i)", k: k, h: heads, b: batchSize, t: sequenceLength,
      outputLength: i == layers - 1 ? h * w : sequenceLength, intermediateSize: 8_192,
      usesFlashAttention: usesFlashAttention, layerIndex: i, configuration: LoRAConfiguration)
    if let skippableIndex = skippableIndex {
      // Both branches build the same graph so the model builder keeps the weights when switching
      // between them. A skipped block runs on 0-length tensors, and its output is dropped as the
      // 0-length part of the concat.
      let blockOut = block(
        [
          out.reshaped(skipsLayers ? [0] : [batchSize, sequenceLength, channels]),
          skippableRots[skippableIndex],
        ] + chunks)
      out = Functional.concat(
        axis: 1, blockOut.reshaped(skipsLayers ? [0] : [batchSize, sequenceLength, channels]),
        out.reshaped(skipsLayers ? [batchSize, sequenceLength, channels] : [0]))
    } else {
      out = block([out, rot] + chunks)
    }
    mappers.append(mapper)
  }
  let normFinal = RMSNorm(epsilon: 1e-6, axis: [2], name: "final_norm")
  out = (normFinal(out) .* scale + shift).to(T.dataType)
  let projOut = LoRADense(
    count: 2 * 2 * 32, configuration: LoRAConfiguration, noBias: true, index: 0,
    name: "final_linear")
  out = projOut(out)
  // Unpatchify
  out = out.reshaped([batchSize, h, w, 32, 2, 2]).permuted(0, 1, 4, 2, 5, 3).contiguous().reshaped([
    batchSize, h * 2, w * 2, 32,
  ])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping["video_embedder.weight"] = [xEmbedder.weight.name]
    mapping["video_embedder.bias"] = [xEmbedder.bias.name]
    mapping["final_norm_video.weight"] = [normFinal.weight.name]
    mapping["final_linear_video.weight"] = [projOut.weight.name]
    return mapping
  }
  return (
    mapper,
    Model(
      [x, rot, contextIn] + chunks + [shift, scale] + skippableRots, [out], trainable: false)
  )
}

public func LoRAMOSSImageFixed<T: TensorNumeric>(
  _ dataType: T.Type, batchSize: Int, tokenLengths: [Int], timesteps: Int, channels: Int,
  usesFlashAttention: FlashAttentionLevel, LoRAConfiguration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let contextIn = Input()
  let rot = Input()
  let t = Input()
  let contextEmbedder = LoRADense(
    count: channels, configuration: LoRAConfiguration, index: 0, name: "context_embedder")
  var context = contextEmbedder(contextIn).to(.Float32)
  var mappers = [ModelWeightMapper]()
  for i in 0..<2 {
    let (mapper, block) = LoRAMOSSImageTextBlock(
      T.self, prefix: "text_embedder.blocks.\(i)", k: 128, h: channels / 128, b: batchSize,
      segments: tokenLengths, intermediateSize: 8_192, usesFlashAttention: usesFlashAttention,
      layerIndex: i, configuration: LoRAConfiguration)
    context = block(context, rot)
    mappers.append(mapper)
  }
  let tMlp0 = LoRADense(
    count: channels, configuration: LoRAConfiguration, index: 0, name: "time_embedder_0")
  let tMlp2 = LoRADense(
    count: channels, configuration: LoRAConfiguration, index: 0, name: "time_embedder_2")
  // The output layer is kept in FP32 to avoid FP16 underflow of its weights.
  let tOut = tMlp2(tMlp0(t).swish().to(.Float32)).reshaped([timesteps, 1, channels])
  let tIn = tOut.to(.Float16)
  let adaLNs = (0..<6).map {
    LoRADense(
      count: channels, configuration: LoRAConfiguration, index: 0, name: "x_ada_ln_\($0)")
  }
  var chunks = adaLNs.map { $0(tIn).to(.Float32) }
  chunks[1] = 1 + chunks[1]
  chunks[4] = 1 + chunks[4]
  let shift = LoRADense(
    count: channels, configuration: LoRAConfiguration, index: 0, name: "ada_ln_1")
  let scale = LoRADense(
    count: channels, configuration: LoRAConfiguration, index: 0, name: "ada_ln_0")
  chunks.append(shift(tIn).to(.Float32))
  chunks.append(1 + scale(tIn).to(.Float32))
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping["text_embedder.proj.weight"] = [contextEmbedder.weight.name]
    mapping["text_embedder.proj.bias"] = [contextEmbedder.bias.name]
    mapping["time_embedding.0.weight"] = [tMlp0.weight.name]
    mapping["time_embedding.0.bias"] = [tMlp0.bias.name]
    mapping["time_embedding.2.weight"] = [tMlp2.weight.name]
    mapping["time_embedding.2.bias"] = [tMlp2.bias.name]
    mapping["shared_block_modulation.weight"] = ModelWeightElement(
      adaLNs.map { $0.weight.name })
    mapping["shared_block_modulation.bias"] = ModelWeightElement(adaLNs.map { $0.bias.name })
    mapping["video_head_modulation.weight"] = [shift.weight.name, scale.weight.name]
    mapping["video_head_modulation.bias"] = [shift.bias.name, scale.bias.name]
    return mapping
  }
  return (mapper, Model([contextIn, rot, t], [context] + chunks, trainable: false))
}
