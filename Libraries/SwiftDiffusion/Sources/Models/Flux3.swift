import DiffusionMappings
import Foundation
import NNC

public func Flux3RotaryPositionEmbedding(
  height: Int, width: Int, tokenLength: Int, referenceSizes: [(height: Int, width: Int)],
  channels: Int, heads: Int = 1
)
  -> Tensor<Float>
{
  var rotTensor = Tensor<Float>(
    .CPU,
    .NHWC(
      1, height * width + referenceSizes.reduce(0) { $0 + $1.height * $1.width } + tokenLength,
      heads, channels))
  let dim = channels / 4
  assert(channels % 16 == 0)
  // Each token has 4 position axes (unused, y, x, text / reference index), each takes channels / 4.
  func setPositions(_ i: Int, _ positions: [Double]) {
    for j in 0..<heads {
      for (axis, position) in positions.enumerated() {
        for k in 0..<(dim / 2) {
          let theta = position * 1.0 / pow(10_000, Double(k) * 2 / Double(dim))
          let sintheta = sin(theta)
          let costheta = cos(theta)
          rotTensor[0, i, j, (k + axis * (dim / 2)) * 2] = Float(costheta)
          rotTensor[0, i, j, (k + axis * (dim / 2)) * 2 + 1] = Float(sintheta)
        }
      }
    }
  }
  for y in 0..<height {
    for x in 0..<width {
      setPositions(y * width + x, [0, Double(y) * 0.5, Double(x) * 0.5, 0])
    }
  }
  var index = height * width
  for (n, referenceSize) in referenceSizes.enumerated() {
    let height = referenceSize.height
    let width = referenceSize.width
    for y in 0..<height {
      for x in 0..<width {
        setPositions(
          y * width + x + index, [0, Double(y) * 0.5, Double(x) * 0.5, Double(-1 - n)])
      }
    }
    index += height * width
  }
  for i in 0..<tokenLength {
    setPositions(i + index, [0, 0, 0, Double(i)])
  }
  return rotTensor
}

private func MLPEmbedder(channels: Int, name: String) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: channels, noBias: true, name: "\(name)_embedder_0")
  var out = fc0(x).swish()
  let fc2 = Dense(count: channels, noBias: true, name: "\(name)_embedder_2")
  out = fc2(out.to(.Float32))  // The output layer is kept in FP32.
  return (fc0, fc2, Model([x], [out]))
}

private func SingleTransformerBlock(
  prefix: String, name: String, k: Int, h: Int, b: Int, segments: [Int],
  contextBlockPreOnly: Bool, usesFlashAttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let rot = Input()
  // Each segment (image, reference, text) has its own shift, scale and gate. The last block only
  // outputs the first segment (image), thus, other segments don't need the gate.
  let chunks = segments.indices.map { i in
    (0..<(contextBlockPreOnly && i > 0 ? 2 : 3)).map { _ in Input() }
  }
  let t = segments.reduce(0, +)
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = xNorm1(x)
  if segments.count > 1 {
    var offset = 0
    var outs = [Model.IO]()
    for (i, segment) in segments.enumerated() {
      let segmentOut = xOut.reshaped(
        [b, segment, k * h], offset: [0, offset, 0], strides: [t * k * h, k * h, 1]
      ).contiguous()
      outs.append((segmentOut .* chunks[i][1] + chunks[i][0]).to(.Float16))
      offset += segment
    }
    xOut = Concat(axis: 1)(outs)
  } else {
    xOut = (xOut .* chunks[0][1] + chunks[0][0]).to(.Float16)
  }
  let xToKeys = Dense(count: k * h, noBias: true, name: "\(name)_k")
  let xToQueries = Dense(count: k * h, noBias: true, flags: [.Float16], name: "\(name)_q")
  let xToValues = Dense(count: k * h, noBias: true, name: "\(name)_v")
  var keys = xToKeys(xOut).reshaped([b, t, h, k])
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "\(name)_norm_k")
  keys = normK(keys)
  var queries = xToQueries(xOut).reshaped([b, t, h, k])
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "\(name)_norm_q")
  queries = normQ(queries)
  var values = xToValues(xOut).reshaped([b, t, h, k])
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
      out = Concat(axis: 0)(outs)
      out = out.reshaped([b, h, t, k]).transposed(1, 2).reshaped([b, t, h * k])
    } else {
      var dot = Matmul(transposeB: (2, 3))(queries, keys)
      dot = dot.reshaped([b * h * t, t])
      dot = dot.softmax()
      dot = dot.reshaped([b, h, t, t])
      out = dot * values
      out = out.reshaped([b, h, t, k]).transposed(1, 2).reshaped([b, t, h * k])
    }
  case .scale1:
    queries = (1.0 / Float(k).squareRoot()) * queries
    let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
    out = scaledDotProductAttention(queries, keys, values).reshaped([b, t, k * h])
  case .scaleMerged, .quantized:
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1.0 / Float(k).squareRoot(), useHadamard: true,
      flags: usesFlashAttention == .quantized ? [.Int8] : [.Float16])
    out = scaledDotProductAttention(queries, keys, values).reshaped([b, t, k * h])
  }
  var xIn: Model.IO = x
  if contextBlockPreOnly {
    // Only the first segment (image) is needed from the last block.
    let hw = segments[0]
    out = out.reshaped([b, hw, h * k], strides: [t * h * k, h * k, 1]).contiguous()
    xIn = x.reshaped([b, hw, h * k], strides: [t * h * k, h * k, 1]).contiguous()
    xOut = xOut.reshaped([b, hw, h * k], strides: [t * h * k, h * k, 1]).contiguous()
  }
  let xUnifyheads = Dense(count: k * h, noBias: true, name: "\(name)_o")
  let xW1 = Dense(count: k * h * 3, noBias: true, flags: [.Float16], name: "\(name)_w1")
  let xW3 = Dense(count: k * h * 3, noBias: true, name: "\(name)_w3")
  let xW2 = Dense(count: k * h, noBias: true, name: "\(name)_w2")
  // Scale the gated product down by 8 to keep FP16 in range, and restore it in FP32.
  out = Add(rightScalar: 8)(
    xUnifyheads(out).to(.Float32),
    xW2(Functional.swishMul(value: xW3(xOut), gate: xW1(xOut), scale: 1.0 / 8)).to(.Float32))
  if segments.count > 1 && !contextBlockPreOnly {
    var offset = 0
    var outs = [Model.IO]()
    for (i, segment) in segments.enumerated() {
      let segmentIn = x.reshaped(
        [b, segment, k * h], offset: [0, offset, 0], strides: [t * k * h, k * h, 1]
      ).contiguous()
      let segmentOut = out.reshaped(
        [b, segment, k * h], offset: [0, offset, 0], strides: [t * k * h, k * h, 1]
      ).contiguous()
      outs.append(segmentIn + segmentOut .* chunks[i][2])
      offset += segment
    }
    out = Concat(axis: 1)(outs)
  } else {
    out = xIn + out .* chunks[0][2]
  }
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).q_proj.weight"] = [xToQueries.weight.name]
    mapping["\(prefix).k_proj.weight"] = [xToKeys.weight.name]
    mapping["\(prefix).v_proj.weight"] = [xToValues.weight.name]
    mapping["\(prefix).norm.query_norm.scale"] = [normQ.weight.name]
    mapping["\(prefix).norm.key_norm.scale"] = [normK.weight.name]
    mapping["\(prefix).attn_out.weight"] = [xUnifyheads.weight.name]
    mapping["\(prefix).mlp_in.weight"] = [xW1.weight.name, xW3.weight.name]
    mapping["\(prefix).mlp_out.weight"] = [xW2.weight.name]
    return mapping
  }
  return (mapper, Model([x, rot] + chunks.flatMap { $0 }, [out]))
}

public func Flux3(
  batchSize: Int, tokenLength: Int, referenceSequenceLength: Int, height: Int, width: Int,
  channels: Int, layers: (Int, Int), usesFlashAttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let contextIn = Input()
  let rot = Input()
  let h = height / 2
  let w = width / 2
  let xEmbedder = Dense(count: channels, noBias: true, name: "x_embedder")
  var out = xEmbedder(
    x.reshaped([batchSize, h, 2, w, 2, 32]).permuted(0, 1, 3, 5, 2, 4).contiguous().reshaped(
      [batchSize, h * w, 32 * 2 * 2], format: .NHWC)
  ).to(.Float32)
  let referenceLatents: Input? = referenceSequenceLength > 0 ? Input() : nil
  var referenceOut = referenceLatents?.to(.Float32)
  var context = contextIn.to(.Float32)
  let contextRefinerChunks = (0..<3).map { _ in Input() }
  let xRefinerChunks = (0..<3).map { _ in Input() }
  let referenceRefinerChunks = (0..<(referenceSequenceLength > 0 ? 3 : 0)).map { _ in Input() }
  // Sequence is image, reference, text.
  let sequenceLength = h * w + referenceSequenceLength + tokenLength
  let rotResized = rot.reshaped(.NHWC(1, sequenceLength, 1, 128))
  let xRot = rotResized.reshaped(
    [1, h * w, 1, 128], strides: [sequenceLength * 128, 128, 128, 1]
  ).contiguous()
  let referenceRot =
    referenceSequenceLength > 0
    ? rotResized.reshaped(
      [1, referenceSequenceLength, 1, 128], offset: [0, h * w, 0, 0],
      strides: [sequenceLength * 128, 128, 128, 1]
    ).contiguous() : nil
  let contextRot = rotResized.reshaped(
    [1, tokenLength, 1, 128], offset: [0, h * w + referenceSequenceLength, 0, 0],
    strides: [sequenceLength * 128, 128, 128, 1]
  ).contiguous()
  var mappers = [ModelWeightMapper]()
  for i in 0..<layers.0 {
    let (contextMapper, contextBlock) = SingleTransformerBlock(
      prefix: "txt_mode_blocks.\(i)", name: "context_refiner", k: 128, h: channels / 128,
      b: batchSize, segments: [tokenLength], contextBlockPreOnly: false,
      usesFlashAttention: usesFlashAttention)
    context = contextBlock([context, contextRot] + contextRefinerChunks)
    mappers.append(contextMapper)
    let (xMapper, xBlock) = SingleTransformerBlock(
      prefix: "content_mode_blocks.image.\(i)", name: "x_refiner", k: 128, h: channels / 128,
      b: batchSize, segments: [h * w], contextBlockPreOnly: false,
      usesFlashAttention: usesFlashAttention)
    out = xBlock([out, xRot] + xRefinerChunks)
    mappers.append(xMapper)
    if let referenceIn = referenceOut, let referenceRot = referenceRot {
      let (referenceMapper, referenceBlock) = SingleTransformerBlock(
        prefix: "content_mode_blocks.image_cond.\(i)", name: "ref_refiner", k: 128,
        h: channels / 128, b: batchSize, segments: [referenceSequenceLength],
        contextBlockPreOnly: false, usesFlashAttention: usesFlashAttention)
      referenceOut = referenceBlock([referenceIn, referenceRot] + referenceRefinerChunks)
      mappers.append(referenceMapper)
    }
  }
  let contextChunks = (0..<3).map { _ in Input() }
  let xChunks = (0..<3).map { _ in Input() }
  let referenceChunks = (0..<(referenceSequenceLength > 0 ? 3 : 0)).map { _ in Input() }
  let segments: [Int]
  if let referenceOut = referenceOut {
    out = Functional.concat(axis: 1, out, referenceOut, context)
    segments = [h * w, referenceSequenceLength, tokenLength]
  } else {
    out = Functional.concat(axis: 1, out, context)
    segments = [h * w, tokenLength]
  }
  for i in 0..<layers.1 {
    let (mapper, block) = SingleTransformerBlock(
      prefix: "single_blocks.\(i)", name: "x", k: 128, h: channels / 128, b: batchSize,
      segments: segments, contextBlockPreOnly: i == layers.1 - 1,
      usesFlashAttention: usesFlashAttention)
    if i == layers.1 - 1 {
      out = block(
        [out, rotResized] + xChunks + Array(referenceChunks.prefix(2))
          + Array(contextChunks.prefix(2)))
    } else {
      out = block([out, rotResized] + xChunks + referenceChunks + contextChunks)
    }
    mappers.append(mapper)
  }
  let scale = Input()
  let shift = Input()
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  out = (normFinal(out) .* scale + shift).to(.Float16)
  let projOut = Dense(count: 2 * 2 * 32, noBias: true, name: "linear")
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
    mapping["emb_in.image.weight"] = [xEmbedder.weight.name]
    mapping["final_layer.image.linear.weight"] = [projOut.weight.name]
    return mapping
  }
  var inputs: [Input] = [x, rot] + (referenceLatents.map { [$0] } ?? []) + [contextIn]
  inputs.append(contentsOf: contextRefinerChunks + xRefinerChunks + referenceRefinerChunks)
  inputs.append(contentsOf: contextChunks + xChunks + referenceChunks)
  inputs.append(contentsOf: [scale, shift])
  return (mapper, Model(inputs, [out]))
}

public func Flux3Fixed(
  timesteps: Int, channels: Int, referenceSizes: [(height: Int, width: Int)], flowMap: Bool
) -> (ModelWeightMapper, Model) {
  let contextIn = Input()
  // t has timesteps + 1 rows, the last row is timestep 0, which is used by text and references.
  let t = Input()
  // The base model is not a flow map model, thus, it doesn't have the next timestep.
  let nextT: Input? = flowMap ? Input() : nil
  var referenceImages = [Input]()
  var referenceOutputs = [Model.IO]()
  let referenceEmbedder: Model?
  if !referenceSizes.isEmpty {
    let embedder = Dense(count: channels, noBias: true, name: "ref_embedder")
    for referenceSize in referenceSizes {
      let x = Input()
      let h = referenceSize.height / 2
      let w = referenceSize.width / 2
      let out = embedder(
        x.reshaped([1, h, 2, w, 2, 32]).permuted(0, 1, 3, 5, 2, 4).contiguous().reshaped(
          [1, h * w, 32 * 2 * 2], format: .NHWC))
      referenceImages.append(x)
      referenceOutputs.append(out)
    }
    referenceEmbedder = embedder
  } else {
    referenceEmbedder = nil
  }
  let contextEmbedder = Dense(count: channels, noBias: true, name: "context_embedder")
  // Scale the text features down by 4 to keep FP16 in range, and restore it in FP32.
  let context = 4 * contextEmbedder((1.0 / 4) * contextIn).to(.Float32)
  let (tMlp0, tMlp2, tEmbedder) = MLPEmbedder(channels: channels, name: "time")
  let tOut = tEmbedder(t)
  var xVec = tOut.reshaped([timesteps, 1, channels], strides: [channels, channels, 1])
    .contiguous()
  var contextVec = tOut.reshaped(
    [1, 1, channels], offset: [timesteps, 0, 0], strides: [channels, channels, 1]
  ).contiguous()
  let nextTMlp0: Model?
  let nextTMlp2: Model?
  if let nextT = nextT {
    let (mlp0, mlp2, nextTEmbedder) = MLPEmbedder(channels: channels, name: "next_time")
    let nextTOut = nextTEmbedder(nextT)
    // The next timestep (flow map) conditions every token, including text and references.
    xVec = xVec + nextTOut
    contextVec = nextTOut + contextVec
    nextTMlp0 = mlp0
    nextTMlp2 = mlp2
  } else {
    // Text and references are conditioned on timestep 0 only, repeat it for every timestep.
    if timesteps > 1 {
      contextVec = Concat(axis: 0)(Array(repeating: contextVec, count: timesteps))
    }
    nextTMlp0 = nil
    nextTMlp2 = nil
  }
  let xVecIn = xVec.swish().to(.Float16)
  let contextVecIn = contextVec.swish().to(.Float16)
  // Each modulation has its own shift, scale and gate.
  func chunks(_ adaLNs: [Model], _ vec: Model.IO) -> [Model.IO] {
    var chunks = adaLNs.map { $0(vec).to(.Float32) }
    chunks[1] = 1 + chunks[1]
    return chunks
  }
  let contextRefinerAdaLNs = (0..<3).map {
    Dense(count: channels, noBias: true, name: "context_refiner_ada_ln_\($0)")
  }
  let xRefinerAdaLNs = (0..<3).map {
    Dense(count: channels, noBias: true, name: "x_refiner_ada_ln_\($0)")
  }
  let contextAdaLNs = (0..<3).map {
    Dense(count: channels, noBias: true, name: "context_ada_ln_\($0)")
  }
  let xAdaLNs = (0..<3).map { Dense(count: channels, noBias: true, name: "x_ada_ln_\($0)") }
  let referenceRefinerAdaLNs: [Model]
  let referenceAdaLNs: [Model]
  if !referenceSizes.isEmpty {
    referenceRefinerAdaLNs = (0..<3).map {
      Dense(count: channels, noBias: true, name: "ref_refiner_ada_ln_\($0)")
    }
    referenceAdaLNs = (0..<3).map {
      Dense(count: channels, noBias: true, name: "ref_ada_ln_\($0)")
    }
  } else {
    referenceRefinerAdaLNs = []
    referenceAdaLNs = []
  }
  let shift = Dense(count: channels, noBias: true, name: "final_ada_ln_0")
  let scale = Dense(count: channels, noBias: true, name: "final_ada_ln_1")
  let finalVec = xVec.swish()  // This is kept in FP32.
  let finalChunks = [1 + scale(finalVec), shift(finalVec)]
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    if let referenceEmbedder = referenceEmbedder {
      mapping["emb_in.image_cond.weight"] = [referenceEmbedder.weight.name]
    }
    mapping["txt_in.weight"] = [contextEmbedder.weight.name]
    mapping["time_in.in_layer.weight"] = [tMlp0.weight.name]
    mapping["time_in.out_layer.weight"] = [tMlp2.weight.name]
    if let nextTMlp0 = nextTMlp0, let nextTMlp2 = nextTMlp2 {
      mapping["ref_time_in.in_layer.weight"] = [nextTMlp0.weight.name]
      mapping["ref_time_in.out_layer.weight"] = [nextTMlp2.weight.name]
    }
    mapping["early_stream_modulations.txt.lin.weight"] = ModelWeightElement(
      contextRefinerAdaLNs.map { $0.weight.name })
    mapping["early_stream_modulations.image.lin.weight"] = ModelWeightElement(
      xRefinerAdaLNs.map { $0.weight.name })
    mapping["single_stream_modulations.txt.lin.weight"] = ModelWeightElement(
      contextAdaLNs.map { $0.weight.name })
    mapping["single_stream_modulations.image.lin.weight"] = ModelWeightElement(
      xAdaLNs.map { $0.weight.name })
    if !referenceSizes.isEmpty {
      mapping["early_stream_modulations.image_cond.lin.weight"] = ModelWeightElement(
        referenceRefinerAdaLNs.map { $0.weight.name })
      mapping["single_stream_modulations.image_cond.lin.weight"] = ModelWeightElement(
        referenceAdaLNs.map { $0.weight.name })
    }
    mapping["final_layer.image.adaLN_modulation.1.weight"] = [shift.weight.name, scale.weight.name]
    return mapping
  }
  var outputs = referenceOutputs
  outputs.append(context)
  outputs.append(contentsOf: chunks(contextRefinerAdaLNs, contextVecIn))
  outputs.append(contentsOf: chunks(xRefinerAdaLNs, xVecIn))
  if !referenceSizes.isEmpty {
    outputs.append(contentsOf: chunks(referenceRefinerAdaLNs, contextVecIn))
  }
  outputs.append(contentsOf: chunks(contextAdaLNs, contextVecIn))
  outputs.append(contentsOf: chunks(xAdaLNs, xVecIn))
  if !referenceSizes.isEmpty {
    outputs.append(contentsOf: chunks(referenceAdaLNs, contextVecIn))
  }
  outputs.append(contentsOf: finalChunks)
  return (
    mapper, Model([contextIn] + referenceImages + [t] + (nextT.map { [$0] } ?? []), outputs)
  )
}

public func Flux3FixedOutputShapes(
  tokenLength: Int, referenceSequenceLength: Int, channels: Int
) -> [TensorShape] {
  var outs = [TensorShape]()
  if referenceSequenceLength > 0 {
    outs.append(TensorShape([1, referenceSequenceLength, channels]))
  }
  outs.append(TensorShape([1, tokenLength, channels]))
  for _ in 0..<(3 * (referenceSequenceLength > 0 ? 6 : 4) + 2) {
    outs.append(TensorShape([1, 1, channels]))
  }
  return outs
}

private func LoRAMLPEmbedder(
  channels: Int, configuration: LoRANetworkConfiguration, index: Int, name: String
) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = LoRADense(
    count: channels, configuration: configuration, noBias: true, index: index,
    name: "\(name)_embedder_0")
  var out = fc0(x).swish()
  let fc2 = LoRADense(
    count: channels, configuration: configuration, noBias: true, index: index,
    name: "\(name)_embedder_2")
  out = fc2(out.to(.Float32))  // The output layer is kept in FP32.
  return (fc0, fc2, Model([x], [out]))
}

private func LoRASingleTransformerBlock(
  prefix: String, name: String, k: Int, h: Int, b: Int, segments: [Int],
  contextBlockPreOnly: Bool, usesFlashAttention: FlashAttentionLevel, layerIndex: Int,
  configuration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let rot = Input()
  // Each segment (image, reference, text) has its own shift, scale and gate. The last block only
  // outputs the first segment (image), thus, other segments don't need the gate.
  let chunks = segments.indices.map { i in
    (0..<(contextBlockPreOnly && i > 0 ? 2 : 3)).map { _ in Input() }
  }
  let t = segments.reduce(0, +)
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = xNorm1(x)
  if segments.count > 1 {
    var offset = 0
    var outs = [Model.IO]()
    for (i, segment) in segments.enumerated() {
      let segmentOut = xOut.reshaped(
        [b, segment, k * h], offset: [0, offset, 0], strides: [t * k * h, k * h, 1]
      ).contiguous()
      outs.append((segmentOut .* chunks[i][1] + chunks[i][0]).to(.Float16))
      offset += segment
    }
    xOut = Concat(axis: 1)(outs)
  } else {
    xOut = (xOut .* chunks[0][1] + chunks[0][0]).to(.Float16)
  }
  let xToKeys = LoRADense(
    count: k * h, configuration: configuration, noBias: true, index: layerIndex, name: "\(name)_k")
  let xToQueries = LoRADense(
    count: k * h, configuration: configuration, noBias: true, flags: [.Float16], index: layerIndex,
    name: "\(name)_q")
  let xToValues = LoRADense(
    count: k * h, configuration: configuration, noBias: true, index: layerIndex, name: "\(name)_v")
  var keys = xToKeys(xOut).reshaped([b, t, h, k])
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "\(name)_norm_k")
  keys = normK(keys)
  var queries = xToQueries(xOut).reshaped([b, t, h, k])
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "\(name)_norm_q")
  queries = normQ(queries)
  var values = xToValues(xOut).reshaped([b, t, h, k])
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
      out = Concat(axis: 0)(outs)
      out = out.reshaped([b, h, t, k]).transposed(1, 2).reshaped([b, t, h * k])
    } else {
      var dot = Matmul(transposeB: (2, 3))(queries, keys)
      dot = dot.reshaped([b * h * t, t])
      dot = dot.softmax()
      dot = dot.reshaped([b, h, t, t])
      out = dot * values
      out = out.reshaped([b, h, t, k]).transposed(1, 2).reshaped([b, t, h * k])
    }
  case .scale1:
    queries = (1.0 / Float(k).squareRoot()) * queries
    let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
    out = scaledDotProductAttention(queries, keys, values).reshaped([b, t, k * h])
  case .scaleMerged, .quantized:
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1.0 / Float(k).squareRoot(), useHadamard: true,
      flags: usesFlashAttention == .quantized ? [.Int8] : [.Float16])
    out = scaledDotProductAttention(queries, keys, values).reshaped([b, t, k * h])
  }
  var xIn: Model.IO = x
  if contextBlockPreOnly {
    // Only the first segment (image) is needed from the last block.
    let hw = segments[0]
    out = out.reshaped([b, hw, h * k], strides: [t * h * k, h * k, 1]).contiguous()
    xIn = x.reshaped([b, hw, h * k], strides: [t * h * k, h * k, 1]).contiguous()
    xOut = xOut.reshaped([b, hw, h * k], strides: [t * h * k, h * k, 1]).contiguous()
  }
  let xUnifyheads = LoRADense(
    count: k * h, configuration: configuration, noBias: true, index: layerIndex, name: "\(name)_o")
  let xW1 = LoRADense(
    count: k * h * 3, configuration: configuration, noBias: true, flags: [.Float16],
    index: layerIndex, name: "\(name)_w1")
  let xW3 = LoRADense(
    count: k * h * 3, configuration: configuration, noBias: true, index: layerIndex,
    name: "\(name)_w3")
  let xW2 = LoRADense(
    count: k * h, configuration: configuration, noBias: true, index: layerIndex,
    name: "\(name)_w2")
  // Scale the gated product down by 8 to keep FP16 in range, and restore it in FP32.
  out = Add(rightScalar: 8)(
    xUnifyheads(out).to(.Float32),
    xW2(Functional.swishMul(value: xW3(xOut), gate: xW1(xOut), scale: 1.0 / 8)).to(.Float32))
  if segments.count > 1 && !contextBlockPreOnly {
    var offset = 0
    var outs = [Model.IO]()
    for (i, segment) in segments.enumerated() {
      let segmentIn = x.reshaped(
        [b, segment, k * h], offset: [0, offset, 0], strides: [t * k * h, k * h, 1]
      ).contiguous()
      let segmentOut = out.reshaped(
        [b, segment, k * h], offset: [0, offset, 0], strides: [t * k * h, k * h, 1]
      ).contiguous()
      outs.append(segmentIn + segmentOut .* chunks[i][2])
      offset += segment
    }
    out = Concat(axis: 1)(outs)
  } else {
    out = xIn + out .* chunks[0][2]
  }
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).q_proj.weight"] = [xToQueries.weight.name]
    mapping["\(prefix).k_proj.weight"] = [xToKeys.weight.name]
    mapping["\(prefix).v_proj.weight"] = [xToValues.weight.name]
    mapping["\(prefix).norm.query_norm.scale"] = [normQ.weight.name]
    mapping["\(prefix).norm.key_norm.scale"] = [normK.weight.name]
    mapping["\(prefix).attn_out.weight"] = [xUnifyheads.weight.name]
    mapping["\(prefix).mlp_in.weight"] = [xW1.weight.name, xW3.weight.name]
    mapping["\(prefix).mlp_out.weight"] = [xW2.weight.name]
    return mapping
  }
  return (mapper, Model([x, rot] + chunks.flatMap { $0 }, [out]))
}

public func LoRAFlux3(
  batchSize: Int, tokenLength: Int, referenceSequenceLength: Int, height: Int, width: Int,
  channels: Int, layers: (Int, Int), usesFlashAttention: FlashAttentionLevel,
  LoRAConfiguration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let contextIn = Input()
  let rot = Input()
  let h = height / 2
  let w = width / 2
  let xEmbedder = LoRADense(
    count: channels, configuration: LoRAConfiguration, noBias: true, index: 0,
    name: "x_embedder")
  var out = xEmbedder(
    x.reshaped([batchSize, h, 2, w, 2, 32]).permuted(0, 1, 3, 5, 2, 4).contiguous().reshaped(
      [batchSize, h * w, 32 * 2 * 2], format: .NHWC)
  ).to(.Float32)
  let referenceLatents: Input? = referenceSequenceLength > 0 ? Input() : nil
  var referenceOut = referenceLatents?.to(.Float32)
  var context = contextIn.to(.Float32)
  let contextRefinerChunks = (0..<3).map { _ in Input() }
  let xRefinerChunks = (0..<3).map { _ in Input() }
  let referenceRefinerChunks = (0..<(referenceSequenceLength > 0 ? 3 : 0)).map { _ in Input() }
  // Sequence is image, reference, text.
  let sequenceLength = h * w + referenceSequenceLength + tokenLength
  let rotResized = rot.reshaped(.NHWC(1, sequenceLength, 1, 128))
  let xRot = rotResized.reshaped(
    [1, h * w, 1, 128], strides: [sequenceLength * 128, 128, 128, 1]
  ).contiguous()
  let referenceRot =
    referenceSequenceLength > 0
    ? rotResized.reshaped(
      [1, referenceSequenceLength, 1, 128], offset: [0, h * w, 0, 0],
      strides: [sequenceLength * 128, 128, 128, 1]
    ).contiguous() : nil
  let contextRot = rotResized.reshaped(
    [1, tokenLength, 1, 128], offset: [0, h * w + referenceSequenceLength, 0, 0],
    strides: [sequenceLength * 128, 128, 128, 1]
  ).contiguous()
  var mappers = [ModelWeightMapper]()
  for i in 0..<layers.0 {
    let (contextMapper, contextBlock) = LoRASingleTransformerBlock(
      prefix: "txt_mode_blocks.\(i)", name: "context_refiner", k: 128, h: channels / 128,
      b: batchSize, segments: [tokenLength], contextBlockPreOnly: false,
      usesFlashAttention: usesFlashAttention, layerIndex: i, configuration: LoRAConfiguration)
    context = contextBlock([context, contextRot] + contextRefinerChunks)
    if LoRAConfiguration.gradientCheckpointingTransformerLayer {
      contextBlock.gradientCheckpointing = true
    }
    mappers.append(contextMapper)
    let (xMapper, xBlock) = LoRASingleTransformerBlock(
      prefix: "content_mode_blocks.image.\(i)", name: "x_refiner", k: 128, h: channels / 128,
      b: batchSize, segments: [h * w], contextBlockPreOnly: false,
      usesFlashAttention: usesFlashAttention, layerIndex: i, configuration: LoRAConfiguration)
    out = xBlock([out, xRot] + xRefinerChunks)
    if LoRAConfiguration.gradientCheckpointingTransformerLayer {
      xBlock.gradientCheckpointing = true
    }
    mappers.append(xMapper)
    if let referenceIn = referenceOut, let referenceRot = referenceRot {
      let (referenceMapper, referenceBlock) = LoRASingleTransformerBlock(
        prefix: "content_mode_blocks.image_cond.\(i)", name: "ref_refiner", k: 128,
        h: channels / 128, b: batchSize, segments: [referenceSequenceLength],
        contextBlockPreOnly: false, usesFlashAttention: usesFlashAttention, layerIndex: i,
        configuration: LoRAConfiguration)
      referenceOut = referenceBlock([referenceIn, referenceRot] + referenceRefinerChunks)
      if LoRAConfiguration.gradientCheckpointingTransformerLayer {
        referenceBlock.gradientCheckpointing = true
      }
      mappers.append(referenceMapper)
    }
  }
  let contextChunks = (0..<3).map { _ in Input() }
  let xChunks = (0..<3).map { _ in Input() }
  let referenceChunks = (0..<(referenceSequenceLength > 0 ? 3 : 0)).map { _ in Input() }
  let segments: [Int]
  if let referenceOut = referenceOut {
    out = Functional.concat(axis: 1, out, referenceOut, context)
    segments = [h * w, referenceSequenceLength, tokenLength]
  } else {
    out = Functional.concat(axis: 1, out, context)
    segments = [h * w, tokenLength]
  }
  for i in 0..<layers.1 {
    let (mapper, block) = LoRASingleTransformerBlock(
      prefix: "single_blocks.\(i)", name: "x", k: 128, h: channels / 128, b: batchSize,
      segments: segments, contextBlockPreOnly: i == layers.1 - 1,
      usesFlashAttention: usesFlashAttention, layerIndex: i, configuration: LoRAConfiguration)
    if i == layers.1 - 1 {
      out = block(
        [out, rotResized] + xChunks + Array(referenceChunks.prefix(2))
          + Array(contextChunks.prefix(2)))
    } else {
      out = block([out, rotResized] + xChunks + referenceChunks + contextChunks)
    }
    if LoRAConfiguration.gradientCheckpointingTransformerLayer {
      block.gradientCheckpointing = true
    }
    mappers.append(mapper)
  }
  let scale = Input()
  let shift = Input()
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  out = (normFinal(out) .* scale + shift).to(.Float16)
  let projOut = LoRADense(
    count: 2 * 2 * 32, configuration: LoRAConfiguration, noBias: true, index: 0, name: "linear")
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
    mapping["emb_in.image.weight"] = [xEmbedder.weight.name]
    mapping["final_layer.image.linear.weight"] = [projOut.weight.name]
    return mapping
  }
  var inputs: [Input] = [x, rot] + (referenceLatents.map { [$0] } ?? []) + [contextIn]
  inputs.append(contentsOf: contextRefinerChunks + xRefinerChunks + referenceRefinerChunks)
  inputs.append(contentsOf: contextChunks + xChunks + referenceChunks)
  inputs.append(contentsOf: [scale, shift])
  return (mapper, Model(inputs, [out], trainable: false))
}

public func LoRAFlux3Fixed(
  timesteps: Int, channels: Int, referenceSizes: [(height: Int, width: Int)], flowMap: Bool,
  LoRAConfiguration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let contextIn = Input()
  // t has timesteps + 1 rows, the last row is timestep 0, which is used by text and references.
  let t = Input()
  // The base model is not a flow map model, thus, it doesn't have the next timestep.
  let nextT: Input? = flowMap ? Input() : nil
  var referenceImages = [Input]()
  var referenceOutputs = [Model.IO]()
  let referenceEmbedder: Model?
  if !referenceSizes.isEmpty {
    let embedder = LoRADense(
      count: channels, configuration: LoRAConfiguration, noBias: true, index: 0,
      name: "ref_embedder")
    for referenceSize in referenceSizes {
      let x = Input()
      let h = referenceSize.height / 2
      let w = referenceSize.width / 2
      let out = embedder(
        x.reshaped([1, h, 2, w, 2, 32]).permuted(0, 1, 3, 5, 2, 4).contiguous().reshaped(
          [1, h * w, 32 * 2 * 2], format: .NHWC))
      referenceImages.append(x)
      referenceOutputs.append(out)
    }
    referenceEmbedder = embedder
  } else {
    referenceEmbedder = nil
  }
  let contextEmbedder = LoRADense(
    count: channels, configuration: LoRAConfiguration, noBias: true, index: 0,
    name: "context_embedder")
  // Scale the text features down by 4 to keep FP16 in range, and restore it in FP32.
  let context = 4 * contextEmbedder((1.0 / 4) * contextIn).to(.Float32)
  let (tMlp0, tMlp2, tEmbedder) = LoRAMLPEmbedder(
    channels: channels, configuration: LoRAConfiguration, index: 0, name: "time")
  let tOut = tEmbedder(t)
  var xVec = tOut.reshaped([timesteps, 1, channels], strides: [channels, channels, 1])
    .contiguous()
  var contextVec = tOut.reshaped(
    [1, 1, channels], offset: [timesteps, 0, 0], strides: [channels, channels, 1]
  ).contiguous()
  let nextTMlp0: Model?
  let nextTMlp2: Model?
  if let nextT = nextT {
    let (mlp0, mlp2, nextTEmbedder) = LoRAMLPEmbedder(
      channels: channels, configuration: LoRAConfiguration, index: 0, name: "next_time")
    let nextTOut = nextTEmbedder(nextT)
    // The next timestep (flow map) conditions every token, including text and references.
    xVec = xVec + nextTOut
    contextVec = nextTOut + contextVec
    nextTMlp0 = mlp0
    nextTMlp2 = mlp2
  } else {
    // Text and references are conditioned on timestep 0 only, repeat it for every timestep.
    if timesteps > 1 {
      contextVec = Concat(axis: 0)(Array(repeating: contextVec, count: timesteps))
    }
    nextTMlp0 = nil
    nextTMlp2 = nil
  }
  let xVecIn = xVec.swish().to(.Float16)
  let contextVecIn = contextVec.swish().to(.Float16)
  // Each modulation has its own shift, scale and gate.
  func chunks(_ adaLNs: [Model], _ vec: Model.IO) -> [Model.IO] {
    var chunks = adaLNs.map { $0(vec).to(.Float32) }
    chunks[1] = 1 + chunks[1]
    return chunks
  }
  let contextRefinerAdaLNs = (0..<3).map {
    LoRADense(
      count: channels, configuration: LoRAConfiguration, noBias: true, index: 0,
      name: "context_refiner_ada_ln_\($0)")
  }
  let xRefinerAdaLNs = (0..<3).map {
    LoRADense(
      count: channels, configuration: LoRAConfiguration, noBias: true, index: 0,
      name: "x_refiner_ada_ln_\($0)")
  }
  let contextAdaLNs = (0..<3).map {
    LoRADense(
      count: channels, configuration: LoRAConfiguration, noBias: true, index: 0,
      name: "context_ada_ln_\($0)")
  }
  let xAdaLNs = (0..<3).map {
    LoRADense(
      count: channels, configuration: LoRAConfiguration, noBias: true, index: 0,
      name: "x_ada_ln_\($0)")
  }
  let referenceRefinerAdaLNs: [Model]
  let referenceAdaLNs: [Model]
  if !referenceSizes.isEmpty {
    referenceRefinerAdaLNs = (0..<3).map {
      LoRADense(
        count: channels, configuration: LoRAConfiguration, noBias: true, index: 0,
        name: "ref_refiner_ada_ln_\($0)")
    }
    referenceAdaLNs = (0..<3).map {
      LoRADense(
        count: channels, configuration: LoRAConfiguration, noBias: true, index: 0,
        name: "ref_ada_ln_\($0)")
    }
  } else {
    referenceRefinerAdaLNs = []
    referenceAdaLNs = []
  }
  let shift = LoRADense(
    count: channels, configuration: LoRAConfiguration, noBias: true, index: 0,
    name: "final_ada_ln_0")
  let scale = LoRADense(
    count: channels, configuration: LoRAConfiguration, noBias: true, index: 0,
    name: "final_ada_ln_1")
  let finalVec = xVec.swish()  // This is kept in FP32.
  let finalChunks = [1 + scale(finalVec), shift(finalVec)]
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    if let referenceEmbedder = referenceEmbedder {
      mapping["emb_in.image_cond.weight"] = [referenceEmbedder.weight.name]
    }
    mapping["txt_in.weight"] = [contextEmbedder.weight.name]
    mapping["time_in.in_layer.weight"] = [tMlp0.weight.name]
    mapping["time_in.out_layer.weight"] = [tMlp2.weight.name]
    if let nextTMlp0 = nextTMlp0, let nextTMlp2 = nextTMlp2 {
      mapping["ref_time_in.in_layer.weight"] = [nextTMlp0.weight.name]
      mapping["ref_time_in.out_layer.weight"] = [nextTMlp2.weight.name]
    }
    mapping["early_stream_modulations.txt.lin.weight"] = ModelWeightElement(
      contextRefinerAdaLNs.map { $0.weight.name })
    mapping["early_stream_modulations.image.lin.weight"] = ModelWeightElement(
      xRefinerAdaLNs.map { $0.weight.name })
    mapping["single_stream_modulations.txt.lin.weight"] = ModelWeightElement(
      contextAdaLNs.map { $0.weight.name })
    mapping["single_stream_modulations.image.lin.weight"] = ModelWeightElement(
      xAdaLNs.map { $0.weight.name })
    if !referenceSizes.isEmpty {
      mapping["early_stream_modulations.image_cond.lin.weight"] = ModelWeightElement(
        referenceRefinerAdaLNs.map { $0.weight.name })
      mapping["single_stream_modulations.image_cond.lin.weight"] = ModelWeightElement(
        referenceAdaLNs.map { $0.weight.name })
    }
    mapping["final_layer.image.adaLN_modulation.1.weight"] = [shift.weight.name, scale.weight.name]
    return mapping
  }
  var outputs = referenceOutputs
  outputs.append(context)
  outputs.append(contentsOf: chunks(contextRefinerAdaLNs, contextVecIn))
  outputs.append(contentsOf: chunks(xRefinerAdaLNs, xVecIn))
  if !referenceSizes.isEmpty {
    outputs.append(contentsOf: chunks(referenceRefinerAdaLNs, contextVecIn))
  }
  outputs.append(contentsOf: chunks(contextAdaLNs, contextVecIn))
  outputs.append(contentsOf: chunks(xAdaLNs, xVecIn))
  if !referenceSizes.isEmpty {
    outputs.append(contentsOf: chunks(referenceAdaLNs, contextVecIn))
  }
  outputs.append(contentsOf: finalChunks)
  return (
    mapper,
    Model(
      [contextIn] + referenceImages + [t] + (nextT.map { [$0] } ?? []), outputs, trainable: false)
  )
}
