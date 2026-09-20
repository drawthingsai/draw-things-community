import Foundation
import NNC

// Follow V4 Flash: FP32 residuals, mHC and FFNs; FP16 attention and caches.

/// Adjacent-pair complex RoPE. Compressed-attention layers apply YaRN to both
/// raw and global KV; layers 0 and 1 use ordinary RoPE with theta 10,000.
public func DeepSeek4_1RotaryEmbedding<FloatType: TensorNumeric>(
  sequenceLength: Int, cachedTokenLength: Int = 0, positionStride: Int = 1,
  compressed: Bool, headDim: Int = 512,
  configuration: DeepSeek4_1ModelConfiguration = .deepSeekV4_1Flash,
  of dataType: FloatType.Type = Float.self
) -> Tensor<FloatType> {
  precondition(sequenceLength > 0 && cachedTokenLength >= 0 && positionStride > 0)
  let rotaryDim = configuration.rotaryDim
  precondition(rotaryDim > 0 && rotaryDim.isMultiple(of: 2) && headDim >= rotaryDim)
  precondition((headDim - rotaryDim).isMultiple(of: 2))
  let base = compressed ? configuration.compressedRopeTheta : configuration.ropeTheta
  let low = max(
    floor(
      Double(rotaryDim)
        * log(
          Double(configuration.ropeOriginalContext) / (configuration.ropeYarnBetaFast * 2 * .pi))
        / (2 * log(base))), 0)
  let high = min(
    ceil(
      Double(rotaryDim)
        * log(
          Double(configuration.ropeOriginalContext) / (configuration.ropeYarnBetaSlow * 2 * .pi))
        / (2 * log(base))), Double(rotaryDim - 1))
  var frequencies = [Float]()
  for i in 0..<(rotaryDim / 2) {
    var frequency = 1 / pow(Float(base), Float(2 * i) / Float(rotaryDim))
    if compressed {
      let ramp = min(max((Float(i) - Float(low)) / Float(max(high - low, 1e-3)), 0), 1)
      let smooth = 1 - ramp
      frequency =
        frequency / Float(configuration.ropeScaleFactor) * (1 - smooth) + frequency * smooth
    }
    frequencies.append(frequency)
  }
  var values = Array(repeating: Float(0), count: sequenceLength * headDim)
  for row in 0..<sequenceLength {
    for pair in 0..<((headDim - rotaryDim) / 2) {
      values[row * headDim + pair * 2] = 1
    }
    for i in 0..<(rotaryDim / 2) {
      let angle = Float(cachedTokenLength + row * positionStride) * frequencies[i]
      values[row * headDim + headDim - rotaryDim + 2 * i] = cos(angle)
      values[row * headDim + headDim - rotaryDim + 2 * i + 1] = sin(angle)
    }
  }
  return Tensor<FloatType>(from: Tensor<Float>(values, .CPU, .NHWC(1, sequenceLength, 1, headDim)))
}

/// The Engram table is fetched separately. Inputs are the HC residual and the
/// concatenated, dequantized embedding rows for these tokens, in hash-column order.
public func DeepSeek4_1Engram<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, prefix: String, tokenLength: Int,
  configuration: DeepSeek4_1ModelConfiguration = .deepSeekV4_1Flash
) -> Model {
  precondition(tokenLength > 0)
  let residual = Input()
  let embeddings = Input()
  let hidden = configuration.hiddenSize
  let hc = configuration.hcCount
  let embedding = embeddings.reshaped([tokenLength, configuration.engramEmbeddingSize])
    .to(FloatType.dataType)
  let kv = Dense(count: (hc + 1) * hidden, noBias: true, name: "\(prefix).wkv")(
    embedding
  ).to(.Float32)
  let keys = kv.reshaped([tokenLength, hc * hidden], strides: [(hc + 1) * hidden, 1])
    .contiguous().reshaped([tokenLength, hc, hidden])
  let value = kv.reshaped(
    [tokenLength, hidden], offset: [0, hc * hidden],
    strides: [(hc + 1) * hidden, 1]
  ).contiguous().reshaped([tokenLength, 1, hidden])
  let qWeight = Parameter<Float>(.GPU(0), .WC(hc, hidden), name: "\(prefix).q_weight")
  let kWeight = Parameter<Float>(.GPU(0), .WC(hc, hidden), name: "\(prefix).k_weight")
  let q = RMSNorm(epsilon: configuration.normEpsilon, axis: [2], elementwiseAffine: false)(residual)
  let k = RMSNorm(epsilon: configuration.normEpsilon, axis: [2], elementwiseAffine: false)(keys)
  let weights = (qWeight .* kWeight).reshaped([1, hc, hidden])
  let dot = (q .* k .* weights).reduced(.sum, axis: [2]) * (1 / Float(hidden).squareRoot())
  let gate = SignedSquareRoot(minimumMagnitude: 1e-6)(dot).sigmoid()
  let output = residual + gate .* value
  return Model([residual, embeddings], [output])
}

/// Inputs: FP32 HC residual and incoming pre-mix. Outputs: next pre-mix, post-mix,
/// combination matrix, and the residual collapsed with the *incoming* pre-mix.
public func DeepSeek4_1HCMix(
  prefix: String, tokenLength: Int,
  configuration: DeepSeek4_1ModelConfiguration = .deepSeekV4_1Flash
) -> Model {
  precondition(tokenLength > 0)
  let residual = Input()
  let incomingPre = Input()
  let hc = configuration.hcCount
  let hidden = configuration.hiddenSize
  let flat = residual.reshaped([tokenLength, hc * hidden])
  let inverseNorm = ((flat .* flat).reduced(.mean, axis: [1]) + configuration.normEpsilon)
    .squareRoot().reciprocal()
  // Project before applying the scalar inverse norm, as in the source model.
  let mix =
    Dense(count: configuration.hcMixDim, noBias: true, flags: [.Float32], name: "\(prefix)_fn")(
      flat) .* inverseNorm
  let scale = Parameter<Float>(.GPU(0), .C(3), name: "\(prefix)_scale")
  let base = Parameter<Float>(.GPU(0), .C(configuration.hcMixDim), name: "\(prefix)_base")
  let parts = HyperConnection(
    count: hc, sinkhornIterations: configuration.hcSinkhornIterations,
    epsilon: configuration.hcEpsilon, operation: .split)(mix, scale, base)
  let collapsed = (residual .* incomingPre.reshaped([tokenLength, hc, 1]))
    .reduced(.sum, axis: [1]).reshaped([tokenLength, hidden])
  return Model([residual, incomingPre], [parts[0], parts[1], parts[2], collapsed])
}

/// Inputs: normalized sublayer input and RoPE. Outputs: rotated query, raw
/// KV, and the normalized low-rank query used by index-source layers.
public func DeepSeek4_1AttentionProjection(
  prefix: String, tokenLength: Int,
  configuration: DeepSeek4_1ModelConfiguration = .deepSeekV4_1Flash
) -> Model {
  let x = Input()
  let rotary = Input()
  let dim = configuration.attentionHeadDim
  let heads = configuration.attentionHeads
  let low = Dense(count: configuration.queryLowRank, noBias: true, name: "\(prefix).wq_a")(
    x
  )
  let rank = RMSNorm(epsilon: configuration.normEpsilon, axis: [1], name: "\(prefix).q_norm")(low)
  let query = Dense(count: heads * dim, noBias: true, name: "\(prefix).wq_b")(
    rank
  )
  .reshaped([1, tokenLength, heads, dim])
  let rotatedQuery = Functional.cmul(left: query, right: rotary.reshaped([1, tokenLength, 1, dim]))
  let raw = Dense(count: dim, noBias: true, name: "\(prefix).wkv")(x)
  let normalized = RMSNorm(
    epsilon: configuration.normEpsilon, axis: [1], name: "\(prefix).kv_norm")(raw)
    .reshaped([1, tokenLength, 1, dim])
  let rotatedKV = Functional.cmul(
    left: normalized, right: rotary.reshaped([1, tokenLength, 1, dim])
  )
  return Model([x, rotary], [rotatedQuery, rotatedKV, rank])
}

/// Inputs: attention head outputs and RoPE. The grouped output projection is
/// stored as [groups, low rank, heads per group * head dimension], as in DeepSeek4.
public func DeepSeek4_1AttentionOutput<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, prefix: String, tokenLength: Int,
  configuration: DeepSeek4_1ModelConfiguration = .deepSeekV4_1Flash
) -> Model {
  let heads = Input()
  let rotary = Input()
  let dim = configuration.attentionHeadDim
  let groups = configuration.attentionOutputGroups
  let groupDim = configuration.attentionHeads / groups * dim
  let unrotated = Functional.cmul(
    left: heads.reshaped([tokenLength, configuration.attentionHeads, dim]),
    right: rotary.reshaped([tokenLength, 1, dim]), conjugate: true
  )
  .reshaped([tokenLength, groups, 1, groupDim])
  let woA = Parameter<FloatType>(
    .GPU(0), .HWC(groups, configuration.attentionLowRank, groupDim), name: "\(prefix).wo_a")
  let low = Matmul(transposeB: (1, 2))(unrotated, woA)
    .reshaped([tokenLength, configuration.attentionOutputLowDim])
  let output = Dense(count: configuration.hiddenSize, noBias: true, name: "\(prefix).wo_b")(
    low
  )
  return Model([heads, rotary], [output])
}

/// Projects complete groups to pre-RoPE global latents. Retain an incomplete
/// group's input outside this model and prepend it on the next invocation.
public func DeepSeek4_1Compressor(
  prefix: String, x: Model.IO, tokenLength: Int, compressionRatio: Int,
  dependencies: [Model.IO] = [],
  configuration: DeepSeek4_1ModelConfiguration = .deepSeekV4_1Flash
) -> Model.IO {
  precondition(compressionRatio == 1 || compressionRatio == 2)
  precondition(tokenLength >= 0 && tokenLength.isMultiple(of: compressionRatio))
  let headDim = configuration.attentionHeadDim
  let rows = tokenLength / compressionRatio
  let projected = Dense(count: headDim, noBias: true, name: "\(prefix).wkv")(x)
  projected.add(dependencies: dependencies)
  let pooled: Model.IO
  if compressionRatio == 2 {
    let scores = Dense(count: headDim, noBias: true, name: "\(prefix).wgate")(x)
    scores.add(dependencies: dependencies)
    let weights = scores.reshaped([rows, compressionRatio, headDim]).transposed(1, 2).contiguous()
      .reshaped([rows * headDim, compressionRatio]).softmax()
      .reshaped([rows, headDim, compressionRatio])
    let values = projected.reshaped([rows, compressionRatio, headDim]).transposed(1, 2).contiguous()
    pooled = (weights .* values).reduced(.sum, axis: [2]).reshaped([rows, headDim])
  } else {
    pooled = projected
  }
  let output = RMSNorm(epsilon: configuration.normEpsilon, axis: [1], name: "\(prefix).norm")(
    pooled
  )
  return output
}

/// Inputs: normalized sublayer input, low-rank query, and indexer RoPE.
/// Outputs: rotated query heads and scaled head weights.
public func DeepSeek4_1IndexerQuery(
  prefix: String, tokenLength: Int,
  configuration: DeepSeek4_1ModelConfiguration = .deepSeekV4_1Flash
) -> Model {
  let x = Input()
  let rank = Input()
  let rotary = Input()
  let heads = configuration.indexerHeads
  let dim = configuration.indexerHeadDim
  let query = Dense(count: heads * dim, noBias: true, name: "\(prefix).wq_b")(
    rank
  )
  .reshaped([tokenLength, heads, dim])
  let rotated = Functional.cmul(left: query, right: rotary.reshaped([tokenLength, 1, dim]))
  let weights =
    Dense(count: heads, noBias: true, name: "\(prefix).weights_proj")(x)
    * (1 / Float(dim * heads).squareRoot())
  return Model([x, rank, rotary], [rotated, weights])
}

/// Inputs: compressed latent before RoPE and the RoPE at each group's first token.
public func DeepSeek4_1IndexerKey(
  prefix: String, rowCount: Int,
  configuration: DeepSeek4_1ModelConfiguration = .deepSeekV4_1Flash
) -> Model {
  precondition(rowCount >= 0)
  let latent = Input()
  let rotary = Input()
  let dim = configuration.indexerHeadDim
  let key = Dense(count: dim, noBias: true, name: "\(prefix).wk")(latent)
  let normalized = RMSNorm(epsilon: configuration.normEpsilon, axis: [1], name: "\(prefix).k_norm")(
    key
  )
  let rotated = Functional.cmul(left: normalized, right: rotary.reshaped([rowCount, dim]))
  return Model([latent, rotary], [rotated])
}

/// Inputs: index query [tokens, heads, dim], shared index keys [rows, dim],
/// head weights [tokens, heads], and (after the candidate source) its int32 block IDs.
/// Returns global row ids in position order with -1 for unreachable positions.
/// The candidate source returns [tokens, candidateTopKBlocks] IDs for later indexers;
/// its own selection still scores all reachable rows. Empty key tensors keep
/// the same graph when no compression group has completed.
public func DeepSeek4_1IndexerSelection(
  tokenLength: Int, keyLength: Int, cachedTokenLength: Int, layerIndex: Int,
  configuration: DeepSeek4_1ModelConfiguration = .deepSeekV4_1Flash
) -> Model {
  precondition(tokenLength > 0 && keyLength >= 0 && cachedTokenLength >= 0)
  precondition(configuration.indexSourceLayers.contains(layerIndex))
  let ratio = configuration.compressionRatios[layerIndex]
  precondition(ratio > 0 && keyLength == (cachedTokenLength + tokenLength) / ratio)
  let query = Input()
  let keys = Input()
  let weights = Input()
  let isSource = layerIndex == configuration.candidateSourceLayer
  let candidates = layerIndex > configuration.candidateSourceLayer ? Input() : nil
  let candidatePool: ScaledDotProductArgPartition.CandidatePool?
  if isSource {
    candidatePool = .produce(
      size: configuration.candidateBlockSize, count: configuration.candidateTopKBlocks)
  } else if candidates != nil {
    candidatePool = .consume(
      size: configuration.candidateBlockSize, count: configuration.candidateTopKBlocks)
  } else {
    candidatePool = nil
  }
  let selection = ScaledDotProductArgPartition(
    kth: max(1, min(configuration.indexerTopK, keyLength)), scale: 1, isCausal: true,
    compressionRatio: ratio, queryOffset: cachedTokenLength,
    sortIndices: true, candidatePool: candidatePool)
  var inputs = [query, keys, weights]
  var arguments = [
    query.reshaped([tokenLength, configuration.indexerHeads, configuration.indexerHeadDim]),
    keys.reshaped([keyLength, configuration.indexerHeadDim]),
    weights.reshaped([tokenLength, configuration.indexerHeads]),
  ]
  if let candidates = candidates {
    inputs.append(candidates)
    arguments.append(candidates)
  }
  let outputs = selection(arguments)
  return Model(inputs, isSource ? [outputs[0], outputs[1]] : [outputs[0]])
}

/// Sliding-window attention. Inputs: hidden, rotary, raw KV. Cache buffers are
/// caller-owned; raw KV prepends the last min(window, cachedTokenLength) rows.
public func DeepSeek4_1SWAttention<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, tokenLength: Int, cachedTokenLength: Int, layerIndex: Int,
  cachedRawTokenLength: Int? = nil,
  configuration: DeepSeek4_1ModelConfiguration = .deepSeekV4_1Flash
) -> Model {
  precondition(tokenLength > 0 && cachedTokenLength >= 0)
  precondition(configuration.compressionRatios[layerIndex] == 0)
  let prefix = "layers.\(layerIndex).attn"
  let dim = configuration.attentionHeadDim
  let cachedRaw = cachedRawTokenLength ?? min(configuration.rawWindow, cachedTokenLength)
  let rawRows = cachedRaw + tokenLength
  let x = Input()
  let rotary = Input()
  let rawCache = Input()
  let projection = DeepSeek4_1AttentionProjection(
    prefix: prefix, tokenLength: tokenLength, configuration: configuration)(x, rotary)
  let rawWrite = projection[1].moved(
    to: rawCache.reshaped(
      [1, tokenLength, 1, dim], offset: [0, cachedRaw, 0, 0],
      strides: [rawRows * dim, dim, dim, 1]), flags: [.disableOpt])
  let sinks = Parameter<FloatType>(
    .GPU(0), .NHWC(1, 1, configuration.attentionHeads, 1), name: "\(prefix).attn_sink")
  let heads = ScaledDotProductAttention(
    scale: 1 / Float(dim).squareRoot(), isCausal: true, hasAttentionSinks: true,
    slidingWindow: configuration.rawWindow)(projection[0], rawCache, rawCache, sinks)
  heads.add(dependencies: [rawWrite])
  let output = DeepSeek4_1AttentionOutput(
    dataType, prefix: prefix, tokenLength: tokenLength, configuration: configuration)(
      heads, rotary)
  return Model([x, rotary, rawCache], [output])
}

/// Compressed attention that computes this chunk's global-row indices.
/// Raw-window and selected global KV share one softmax with the attention sink.
/// Inputs: hidden, rotary, raw KV, global KV, index KV; KV-source layers then
/// take compressor input, compressor rotary, and indexer compressor rotary;
/// followed by indexer rotary, and candidates after the candidate-source layer.
/// Outputs: hidden, indices, and candidates on the candidate-source layer.
public func DeepSeek4_1CompressedSparseAttention<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, tokenLength: Int, cachedTokenLength: Int, layerIndex: Int,
  cachedRawTokenLength: Int? = nil,
  configuration: DeepSeek4_1ModelConfiguration = .deepSeekV4_1Flash
) -> Model {
  precondition(tokenLength > 0 && cachedTokenLength >= 0)
  precondition(configuration.indexSourceLayers.contains(layerIndex))
  let prefix = "layers.\(layerIndex).attn"
  let dim = configuration.attentionHeadDim
  let hidden = configuration.hiddenSize
  let indexDim = configuration.indexerHeadDim
  let ratio = configuration.compressionRatios[layerIndex]
  precondition(ratio == 1 || ratio == 2)
  let cachedRaw = cachedRawTokenLength ?? min(configuration.rawWindow, cachedTokenLength)
  let rawRows = cachedRaw + tokenLength
  let globalRows = (cachedTokenLength + tokenLength) / ratio
  let x = Input()
  let rotary = Input()
  let rawCache = Input()
  let globalCache = Input()
  let indexCache = Input()
  var inputs = [x, rotary, rawCache, globalCache, indexCache]
  let projection = DeepSeek4_1AttentionProjection(
    prefix: prefix, tokenLength: tokenLength, configuration: configuration)(x, rotary)
  let rawWrite = projection[1].moved(
    to: rawCache.reshaped(
      [1, tokenLength, 1, dim], offset: [0, cachedRaw, 0, 0],
      strides: [rawRows * dim, dim, dim, 1]), flags: [.disableOpt])
  var cacheWrites = [rawWrite]
  var indexWrite: Model.IO?
  if configuration.kvSourceLayers.contains(layerIndex) {
    let previousRows = cachedTokenLength / ratio
    let emittedRows = globalRows - previousRows
    let remainder = cachedTokenLength % ratio
    let compressorCache = Input()
    let compressorRotary = Input()
    let indexerCompressorRotary = Input()
    inputs.append(contentsOf: [compressorCache, compressorRotary, indexerCompressorRotary])
    let write = x.moved(
      to: compressorCache.reshaped(
        [tokenLength, hidden], offset: [remainder, 0], strides: [hidden, 1]),
      flags: [.disableOpt])
    cacheWrites.append(write)
    // Keep compressor/key projections and cache writes in every graph. Empty
    // compression groups flow through as zero-sized tensors, as in V4 Flash.
    let completeInput = compressorCache.reshaped(
      emittedRows > 0 ? [emittedRows * ratio, hidden] : [0])
    let latent = DeepSeek4_1Compressor(
      prefix: "\(prefix).compressor", x: completeInput, tokenLength: emittedRows * ratio,
      compressionRatio: ratio, dependencies: [write], configuration: configuration)
    let key = DeepSeek4_1IndexerKey(
      prefix: "\(prefix).indexer", rowCount: emittedRows,
      configuration: configuration)(latent, indexerCompressorRotary)
    let keyWrite = key.moved(
      to: emittedRows > 0
        ? indexCache.reshaped(
          [emittedRows, indexDim], offset: [previousRows, 0], strides: [indexDim, 1])
        : indexCache.reshaped([0]), flags: [.disableOpt])
    indexWrite = keyWrite
    cacheWrites.append(keyWrite)
    // Both projections read the original latent. Global KV rotation must not
    // overwrite the indexer's pre-RoPE input.
    let rotated = Functional.cmul(
      left: latent, right: compressorRotary.reshaped([emittedRows, dim]))
    let globalWrite = rotated.moved(
      to: emittedRows > 0
        ? globalCache.reshaped(
          [emittedRows, dim], offset: [previousRows, 0], strides: [dim, 1])
        : globalCache.reshaped([0]), flags: [.disableOpt])
    cacheWrites.append(globalWrite)
  }
  let indexerRotary = Input()
  inputs.append(indexerRotary)
  let query = DeepSeek4_1IndexerQuery(
    prefix: "\(prefix).indexer", tokenLength: tokenLength,
    configuration: configuration)(x, projection[2], indexerRotary)
  var args = [query[0], indexCache, query[1]]
  if layerIndex > configuration.candidateSourceLayer {
    let candidates = Input()
    inputs.append(candidates)
    args.append(candidates)
  }
  let selected = DeepSeek4_1IndexerSelection(
    tokenLength: tokenLength, keyLength: globalRows, cachedTokenLength: cachedTokenLength,
    layerIndex: layerIndex, configuration: configuration)(args)
  if let indexWrite { selected.add(dependencies: [indexWrite]) }
  let global = globalCache.reshaped(globalRows > 0 ? [1, globalRows, 1, dim] : [0])
  let sinks = Parameter<FloatType>(
    .GPU(0), .NHWC(1, 1, configuration.attentionHeads, 1), name: "\(prefix).attn_sink")
  let heads = SparseIndexedAttention(
    scale: 1 / Float(dim).squareRoot(), isCausal: true, hasAttentionSinks: true,
    slidingWindow: configuration.rawWindow)(
      projection[0], rawCache, rawCache, global, global, selected[0], sinks)
  heads.add(dependencies: cacheWrites)
  let output = DeepSeek4_1AttentionOutput(
    dataType, prefix: prefix, tokenLength: tokenLength, configuration: configuration)(
      heads, rotary)
  var outputs = [output, selected[0]]
  if layerIndex == configuration.candidateSourceLayer {
    outputs.append(selected[1])
  }
  return Model(inputs, outputs)
}

/// Reuses global KV and the latest index source's selection for this chunk.
/// Still attends to raw-window and selected global KV with one shared softmax.
/// Inputs: hidden, rotary, raw KV, global KV, indices. Output: hidden.
public func DeepSeek4_1SharedAttention<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, tokenLength: Int, cachedTokenLength: Int, layerIndex: Int,
  cachedRawTokenLength: Int? = nil,
  configuration: DeepSeek4_1ModelConfiguration = .deepSeekV4_1Flash
) -> Model {
  precondition(tokenLength > 0 && cachedTokenLength >= 0)
  precondition(!configuration.indexSourceLayers.contains(layerIndex))
  precondition(!configuration.kvSourceLayers.contains(layerIndex))
  let prefix = "layers.\(layerIndex).attn"
  let dim = configuration.attentionHeadDim
  let ratio = configuration.compressionRatios[layerIndex]
  precondition(ratio == 1 || ratio == 2)
  let cachedRaw = cachedRawTokenLength ?? min(configuration.rawWindow, cachedTokenLength)
  let rawRows = cachedRaw + tokenLength
  let globalRows = (cachedTokenLength + tokenLength) / ratio
  let x = Input()
  let rotary = Input()
  let rawCache = Input()
  let globalCache = Input()
  let indices = Input()
  let projection = DeepSeek4_1AttentionProjection(
    prefix: prefix, tokenLength: tokenLength, configuration: configuration)(x, rotary)
  let rawWrite = projection[1].moved(
    to: rawCache.reshaped(
      [1, tokenLength, 1, dim], offset: [0, cachedRaw, 0, 0],
      strides: [rawRows * dim, dim, dim, 1]), flags: [.disableOpt])
  let global = globalCache.reshaped(globalRows > 0 ? [1, globalRows, 1, dim] : [0])
  let sinks = Parameter<FloatType>(
    .GPU(0), .NHWC(1, 1, configuration.attentionHeads, 1), name: "\(prefix).attn_sink")
  let heads = SparseIndexedAttention(
    scale: 1 / Float(dim).squareRoot(), isCausal: true, hasAttentionSinks: true,
    slidingWindow: configuration.rawWindow)(
      projection[0], rawCache, rawCache, global, global, indices, sinks)
  heads.add(dependencies: [rawWrite])
  let output = DeepSeek4_1AttentionOutput(
    dataType, prefix: prefix, tokenLength: tokenLength, configuration: configuration)(
      heads, rotary)
  return Model([x, rotary, rawCache, globalCache, indices], [output])
}

private func DeepSeek4_1SharedFFN(
  prefix: String, x: Model.IO, dependencies: [Model.IO] = [],
  configuration: DeepSeek4_1ModelConfiguration
) -> Model.IO {
  let hidden = SwiGLU(
    count: configuration.sharedIntermediateSize, clamp: 10, name: "\(prefix).shared_experts")(x)
  hidden.add(dependencies: dependencies)
  return Dense(
    count: configuration.hiddenSize, noBias: true, name: "\(prefix).shared_experts.w2")(hidden)
}

/// FP32 expert evaluation and accumulation with fused SwiGLU, as in V4 Flash.
/// When resident slots are fewer than experts, load the three expert banks with
/// `.jit` and `.externalOnDemand` so MoEWeightStreaming can read their source regions.
public func DeepSeek4_1MoE<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, prefix: String, tokenLength: Int, layerIndex: Int,
  configuration: DeepSeek4_1ModelConfiguration = .deepSeekV4_1Flash
) -> Model {
  precondition(tokenLength > 0 && layerIndex >= 0 && layerIndex < configuration.layers)
  let x = Input()
  let logits = Dense(
    count: configuration.expertCount, noBias: true, name: "\(prefix).gate")(
      x.to(FloatType.dataType))
  let bias = Parameter<Float>(.GPU(0), .C(configuration.expertCount), name: "\(prefix).gate.bias")
  let prepared = MoERouting(
    kth: configuration.routedExperts, weightScale: 1.5, normalizationEpsilon: 1e-20,
    singleInputToken: true, name: "\(prefix).routing")(logits.to(of: x), bias, x)
  let pairs = tokenLength * configuration.routedExperts
  let shared: Model.IO
  let routed: Model.IO
  if configuration.expertResidentSlots[layerIndex] == configuration.expertCount {
    shared = DeepSeek4_1SharedFFN(prefix: prefix, x: x, configuration: configuration)
    let hidden = SegmentedSwiGLU(
      segments: configuration.expertCount, count: configuration.expertIntermediateSize,
      clamp: 10, name: "\(prefix).experts")([
        prepared[0], prepared[3], prepared[4], prepared[1].reshaped([pairs, 1]),
      ])
    routed = SegmentedDense(
      segments: configuration.expertCount, count: configuration.hiddenSize,
      noBias: true, name: "\(prefix).experts.w2")(hidden, prepared[3], prepared[4])
  } else {
    precondition(configuration.expertResidentSlots[layerIndex] >= configuration.routedExperts)
    let gateWeight = Parameter<Float>(
      .GPU(0),
      .HWC(
        configuration.expertCount, configuration.expertIntermediateSize, configuration.hiddenSize),
      name: "\(prefix).experts.streaming_gate")()
    let upWeight = Parameter<Float>(
      .GPU(0),
      .HWC(
        configuration.expertCount, configuration.expertIntermediateSize, configuration.hiddenSize),
      name: "\(prefix).experts.streaming_up")()
    let downWeight = Parameter<Float>(
      .GPU(0),
      .HWC(
        configuration.expertCount, configuration.hiddenSize, configuration.expertIntermediateSize),
      name: "\(prefix).experts.w2")()
    let resident = MoEWeightStreaming(
      residentSlots: configuration.expertResidentSlots[layerIndex],
      routingWidth: configuration.routedExperts, name: "\(prefix).expert_streaming")([
        prepared[3], prepared[4], prepared[1].reshaped([pairs, 1]), gateWeight, upWeight,
        downWeight,
      ])
    let residentIndices = resident[0]
    let residentCounts = resident[1]
    let residentScales = resident[2]
    let residentGate = resident[3]
    let residentUp = resident[4]
    let residentDown = resident[5]
    shared = DeepSeek4_1SharedFFN(
      prefix: prefix, x: x, dependencies: [residentGate, residentUp, residentDown],
      configuration: configuration)
    let hidden = SegmentedSwiGLU(
      segments: 0, count: configuration.expertIntermediateSize, clamp: 10, functional: true,
      name: "\(prefix).experts.functional")([
        prepared[0], residentIndices, residentCounts, residentGate, residentUp, residentScales,
      ])
    hidden.add(dependencies: [shared])
    routed = SegmentedDense(
      segments: 0, count: configuration.hiddenSize, noBias: true, functional: true,
      name: "\(prefix).experts.functional.w2")([
        hidden, residentIndices, residentCounts, residentDown,
      ])
  }
  let scattered = Functional.scatterAdd(
    count: tokenLength, countPerOutput: configuration.routedExperts,
    routed, index: prepared[2]
  ).reshaped([tokenLength, configuration.hiddenSize])
  return Model([x], [scattered + shared])
}

/// Expands each text embedding into the initial HC residual streams. The caller
/// supplies the reference's initial one-hot pre-mix [1, 0, 0, 0].
public func DeepSeek4_1Embedding<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, tokenLength: Int,
  configuration: DeepSeek4_1ModelConfiguration = .deepSeekV4_1Flash
) -> Model {
  let tokens = Input()
  let embedding = Embedding(
    dataType, vocabularySize: configuration.vocabularySize,
    embeddingSize: configuration.hiddenSize, name: "embed")(tokens)
    .reshaped([tokenLength, configuration.hiddenSize], format: .NHWC)
  let concat = Concat(axis: 1)
  concat.flags = [.disableOpt]
  return Model(
    [tokens],
    [
      concat(Array(repeating: embedding, count: configuration.hcCount)).to(.Float32)
        .reshaped([tokenLength, configuration.hcCount, configuration.hiddenSize])
    ])
}

/// A text transformer layer, including its optional Engram contribution. The
/// caller supplies the previous FFN's pre-mix and the attention cache buffers.
/// This layer's attention pre-mix collapses its FFN input; its FFN pre-mix is
/// returned for the next layer (or the final output head).
/// outputTokenLength keeps the input attention/cache updates but evaluates the
/// FFN and returns residual/pre-mix for only that many trailing query tokens.
/// Inputs: residual, incoming pre-mix, optional Engram embeddings, then the
/// attention variant's inputs excluding hidden. Outputs: residual, next pre-mix,
/// followed by indices and optional candidates on index-source layers.
public func DeepSeek4_1Layer<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, tokenLength: Int, cachedTokenLength: Int, layerIndex: Int,
  cachedRawTokenLength: Int? = nil, outputTokenLength: Int? = nil,
  configuration: DeepSeek4_1ModelConfiguration = .deepSeekV4_1Flash
) -> Model {
  let prefix = "layers.\(layerIndex)"
  let residual = Input()
  let incomingPre = Input()
  var inputs = [residual, incomingPre]
  let beforeAttention: Model.IO
  if configuration.engramLayers.contains(layerIndex) {
    let engramEmbeddings = Input()
    inputs.append(engramEmbeddings)
    let engram = DeepSeek4_1Engram(
      dataType, prefix: "\(prefix).engram", tokenLength: tokenLength,
      configuration: configuration)(
        residual, engramEmbeddings)
    beforeAttention = engram[0]
  } else {
    beforeAttention = residual
  }
  let attnMix = DeepSeek4_1HCMix(
    prefix: "\(prefix).hc_attn", tokenLength: tokenLength, configuration: configuration)(
      beforeAttention, incomingPre)
  let normalizedAttention = RMSNorm(
    epsilon: configuration.normEpsilon, axis: [1], name: "\(prefix).attn_norm")(
      attnMix[3]
    ).to(FloatType.dataType)
  let rotary = Input()
  let rawKeyValue = Input()
  inputs.append(contentsOf: [rotary, rawKeyValue])
  let attention: Model.IO
  var indices: Model.IO?
  var candidates: Model.IO?
  if configuration.compressionRatios[layerIndex] == 0 {
    attention = DeepSeek4_1SWAttention(
      dataType, tokenLength: tokenLength, cachedTokenLength: cachedTokenLength,
      layerIndex: layerIndex, cachedRawTokenLength: cachedRawTokenLength,
      configuration: configuration)(
        normalizedAttention, rotary, rawKeyValue)
  } else if configuration.indexSourceLayers.contains(layerIndex) {
    let globalKeyValue = Input()
    let indexKeyValue = Input()
    inputs.append(contentsOf: [globalKeyValue, indexKeyValue])
    var args: [Model.IO] = [
      normalizedAttention, rotary, rawKeyValue, globalKeyValue, indexKeyValue,
    ]
    if configuration.kvSourceLayers.contains(layerIndex) {
      let compressorInput = Input()
      let compressorRotary = Input()
      let indexerCompressorRotary = Input()
      inputs.append(contentsOf: [compressorInput, compressorRotary, indexerCompressorRotary])
      args.append(contentsOf: [compressorInput, compressorRotary, indexerCompressorRotary])
    }
    let indexerRotary = Input()
    inputs.append(indexerRotary)
    args.append(indexerRotary)
    if layerIndex > configuration.candidateSourceLayer {
      let candidates = Input()
      inputs.append(candidates)
      args.append(candidates)
    }
    let result = DeepSeek4_1CompressedSparseAttention(
      dataType, tokenLength: tokenLength, cachedTokenLength: cachedTokenLength,
      layerIndex: layerIndex, cachedRawTokenLength: cachedRawTokenLength,
      configuration: configuration)(args)
    attention = result[0]
    indices = result[1]
    if layerIndex == configuration.candidateSourceLayer { candidates = result[2] }
  } else {
    let globalKeyValue = Input()
    let indices = Input()
    inputs.append(contentsOf: [globalKeyValue, indices])
    attention = DeepSeek4_1SharedAttention(
      dataType, tokenLength: tokenLength, cachedTokenLength: cachedTokenLength,
      layerIndex: layerIndex, cachedRawTokenLength: cachedRawTokenLength,
      configuration: configuration)(
        normalizedAttention, rotary, rawKeyValue, globalKeyValue, indices)
  }
  let afterAttention = HyperConnection(count: configuration.hcCount, operation: .expand)(
    attention, beforeAttention, attnMix[1], attnMix[2])[0]
  let ffnTokenLength = outputTokenLength ?? tokenLength
  precondition(ffnTokenLength > 0 && ffnTokenLength <= tokenLength)
  let ffnResidual: Model.IO
  let incomingFFNPre: Model.IO
  if outputTokenLength != nil {
    let hc = configuration.hcCount
    let hidden = configuration.hiddenSize
    ffnResidual = afterAttention.reshaped(
      [ffnTokenLength, hc, hidden], offset: [tokenLength - ffnTokenLength, 0, 0],
      strides: [hc * hidden, hidden, 1]
    ).contiguous()
    incomingFFNPre = attnMix[0].reshaped(
      [ffnTokenLength, hc], offset: [tokenLength - ffnTokenLength, 0], strides: [hc, 1]
    ).contiguous()
  } else {
    ffnResidual = afterAttention
    incomingFFNPre = attnMix[0]
  }
  let ffnMix = DeepSeek4_1HCMix(
    prefix: "\(prefix).hc_ffn", tokenLength: ffnTokenLength, configuration: configuration)(
      ffnResidual, incomingFFNPre)
  let normalizedFFN = RMSNorm(
    epsilon: configuration.normEpsilon, axis: [1], name: "\(prefix).ffn_norm")(
      ffnMix[3]
    )
  let ffn = DeepSeek4_1MoE(
    dataType, prefix: "\(prefix).ffn", tokenLength: ffnTokenLength, layerIndex: layerIndex,
    configuration: configuration)(normalizedFFN)
  let expanded = HyperConnection(count: configuration.hcCount, operation: .expand)(
    ffn, ffnResidual, ffnMix[1], ffnMix[2])[0]
  var outputs = [expanded, ffnMix[0]]
  if let indices { outputs.append(indices) }
  if let candidates { outputs.append(candidates) }
  return Model(inputs, outputs)
}

/// Collapses the final residual with the last FFN's returned pre-mix. V4.1 has
/// no learned HC head projection. Normalization is FP32; the vocabulary
/// projection uses the activation datatype, as in V4 Flash.
public func DeepSeek4_1OutputHead<FloatType: TensorNumeric>(
  x: Model.IO, incomingPre: Model.IO, tokenLength: Int,
  configuration: DeepSeek4_1ModelConfiguration = .deepSeekV4_1Flash,
  of dataType: FloatType.Type
) -> Model.IO {
  let collapsed =
    (x .* incomingPre.reshaped([tokenLength, configuration.hcCount, 1]))
    .reduced(.sum, axis: [1]).reshaped([tokenLength, configuration.hiddenSize])
  let norm = RMSNorm(epsilon: configuration.normEpsilon, axis: [1], name: "norm")
  let head = Dense(count: configuration.vocabularySize, noBias: true, name: "head")
  return head(norm(collapsed).to(FloatType.dataType))
}

/// Input rows at each layer, followed by the final output row count. Once the
/// last KV source has published the complete chunk, decoder FFNs only retain
/// the suffix needed by the remaining sliding windows. Attention still sees
/// the preceding window rows, so the retained outputs and final raw KV are valid.
public func DeepSeek4_1CausalLMTokenLengths(
  tokenLength: Int, boundedReplay: Bool = false,
  configuration: DeepSeek4_1ModelConfiguration = .deepSeekV4_1Flash
) -> [Int] {
  precondition(tokenLength > 0 && configuration.layers > 0)
  if boundedReplay {
    precondition(configuration.kvSourceLayers.last == configuration.candidateSourceLayer)
    precondition(configuration.engramLayers.allSatisfy { $0 <= configuration.candidateSourceLayer })
    precondition(
      configuration.compressionRatios[configuration.candidateSourceLayer...].allSatisfy { $0 == 1 })
  }
  var lengths = [tokenLength]
  for layer in 0..<configuration.layers {
    let rows =
      boundedReplay && layer >= configuration.candidateSourceLayer
      ? min(tokenLength, 1 + (configuration.layers - 1 - layer) * (configuration.rawWindow - 1))
      : tokenLength
    lengths.append(layer == configuration.layers - 1 ? 1 : rows)
  }
  return lengths
}

private func DeepSeek4_1Prefix<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, tokenLength: Int, cachedTokenLength: Int, boundedReplay: Bool,
  cachedRawTokenLengths: [Int],
  configuration: DeepSeek4_1ModelConfiguration
) -> (inputs: [Input], hidden: Model.IO, pre: Model.IO) {
  precondition(tokenLength > 0 && cachedTokenLength >= 0)
  precondition(configuration.layers > 0)
  precondition(cachedRawTokenLengths.count == configuration.layers)
  precondition(cachedRawTokenLengths.allSatisfy { $0 >= 0 })
  let tokens = Input()
  let incomingPre = Input()
  var inputs = [tokens, incomingPre]
  let ratios = configuration.compressionRatios.prefix(configuration.layers)
  let rawRotary = ratios.contains(0) ? Input() : nil
  if let rawRotary { inputs.append(rawRotary) }
  let usesCompressedRotary = ratios.contains { $0 > 0 }
  let compressedRotary = usesCompressedRotary ? Input() : nil
  let indexerRotary = usesCompressedRotary ? Input() : nil
  if let compressedRotary, let indexerRotary {
    inputs.append(contentsOf: [compressedRotary, indexerRotary])
  }
  var compressorRotaries = [(ratio: Int, rotary: Input, indexerRotary: Input)]()
  for ratio in Set(ratios.filter { $0 > 0 }).sorted() {
    let rotary = Input()
    let indexerRotary = Input()
    compressorRotaries.append((ratio, rotary, indexerRotary))
    inputs.append(contentsOf: [rotary, indexerRotary])
  }
  var hidden = DeepSeek4_1Embedding(
    dataType, tokenLength: tokenLength, configuration: configuration)(tokens)
  var pre: Model.IO = incomingPre
  var globalKeyValue: Input?
  var indexKeyValue: Input?
  var indices: Model.IO?
  var candidates: Model.IO?
  let lengths = DeepSeek4_1CausalLMTokenLengths(
    tokenLength: tokenLength, boundedReplay: boundedReplay, configuration: configuration)
  for layer in 0..<configuration.layers {
    let layerTokenLength = lengths[layer]
    // After trimming, the first window-1 attention outputs are discarded by
    // the FFN slice. Retained outputs only read valid current-chunk raw rows;
    // the caller's raw-cache prefix length stays independent of query position.
    let layerCachedTokenLength = cachedTokenLength + tokenLength - layerTokenLength
    let ratio = configuration.compressionRatios[layer]
    let engramEmbeddings = configuration.engramLayers.contains(layer) ? Input() : nil
    if let engramEmbeddings { inputs.append(engramEmbeddings) }
    let rawKeyValue = Input()
    inputs.append(rawKeyValue)
    let compressorInput: Input?
    if configuration.kvSourceLayers.contains(layer) {
      let global = Input()
      let index = Input()
      let compressor = Input()
      inputs.append(contentsOf: [global, index, compressor])
      globalKeyValue = global
      indexKeyValue = index
      compressorInput = compressor
    } else {
      compressorInput = nil
    }
    // Reader layers reuse the latest source's cache and selection directly.
    let compressorRotary = compressorRotaries.first { $0.ratio == ratio }
    var args = [hidden, pre]
    if let engramEmbeddings { args.append(engramEmbeddings) }
    var rotary: Model.IO = (ratio > 0 ? compressedRotary : rawRotary)!
    if boundedReplay && layer > configuration.candidateSourceLayer {
      let dim = configuration.attentionHeadDim
      rotary = rotary.reshaped(
        [layerTokenLength, dim], offset: [tokenLength - layerTokenLength, 0], strides: [dim, 1]
      ).contiguous()
    }
    args.append(contentsOf: [rotary, rawKeyValue])
    if ratio > 0 {
      args.append(globalKeyValue!)
      if configuration.indexSourceLayers.contains(layer) {
        args.append(indexKeyValue!)
        if let compressorInput {
          args.append(contentsOf: [
            compressorInput, compressorRotary!.rotary, compressorRotary!.indexerRotary,
          ])
        }
        var queryRotary: Model.IO = indexerRotary!
        if boundedReplay && layer > configuration.candidateSourceLayer {
          let dim = configuration.indexerHeadDim
          queryRotary = queryRotary.reshaped(
            [layerTokenLength, dim], offset: [tokenLength - layerTokenLength, 0], strides: [dim, 1]
          ).contiguous()
        }
        args.append(queryRotary)
        if layer > configuration.candidateSourceLayer { args.append(candidates!) }
      } else {
        args.append(indices!)
      }
    }
    let outputs = DeepSeek4_1Layer(
      dataType, tokenLength: layerTokenLength, cachedTokenLength: layerCachedTokenLength,
      layerIndex: layer, cachedRawTokenLength: cachedRawTokenLengths[layer],
      outputTokenLength: (boundedReplay && layer >= configuration.candidateSourceLayer)
        || layer == configuration.layers - 1 ? lengths[layer + 1] : nil,
      configuration: configuration)(args)
    hidden = outputs[0]
    pre = outputs[1]
    if configuration.indexSourceLayers.contains(layer) {
      indices = outputs[2]
      if layer == configuration.candidateSourceLayer { candidates = outputs[3] }
    }
    if boundedReplay && layer >= configuration.candidateSourceLayer
      && layer < configuration.layers - 1
    {
      let rows = lengths[layer + 1]
      if !configuration.indexSourceLayers.contains(layer + 1), let selection = indices {
        let width = max(
          1, min(configuration.indexerTopK, (cachedTokenLength + tokenLength) / ratio))
        indices = selection.reshaped(
          [rows, width], offset: [layerTokenLength - rows, 0], strides: [width, 1]
        ).contiguous()
      }
      if layer < configuration.indexSourceLayers.last!, let pool = candidates {
        let width = configuration.candidateTopKBlocks
        candidates = pool.reshaped(
          [rows, width], offset: [layerTokenLength - rows, 0], strides: [width, 1]
        ).contiguous()
      }
    }
  }
  return (inputs, hidden, pre)
}

/// Complete text model, returning FP16/FP32 logits for the last query token.
/// Inputs: tokens, initial FP32 pre-mix [T, hc] with rows [1, 0, ...], raw rotary
/// (if used), compressed and indexer rotary (if used), then compressor/indexer
/// rotary pairs in ascending compression-ratio order. Each layer appends its
/// optional Engram embeddings and raw KV; KV-source layers also append global
/// KV, index KV, and compressor input. Rotary tensors and caches use the activation
/// dtype. Cache buffers are updated in place; callers retain the raw-window tail
/// and unfinished compressor input. Pass each layer's stored raw-row count when
/// binding a growing cache prefix. Input order depends only on the architecture.
public func DeepSeek4_1CausalLM<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, tokenLength: Int, cachedTokenLength: Int = 0,
  cachedRawTokenLengths: [Int]? = nil, boundedReplay: Bool = false,
  configuration: DeepSeek4_1ModelConfiguration = .deepSeekV4_1Flash
) -> Model {
  let prefix = DeepSeek4_1Prefix(
    dataType, tokenLength: tokenLength, cachedTokenLength: cachedTokenLength,
    boundedReplay: boundedReplay,
    cachedRawTokenLengths: cachedRawTokenLengths
      ?? Array(
        repeating: min(cachedTokenLength, configuration.rawWindow), count: configuration.layers),
    configuration: configuration)
  let output = DeepSeek4_1OutputHead(
    x: prefix.hidden, incomingPre: prefix.pre, tokenLength: 1,
    configuration: configuration, of: dataType)
  return Model(prefix.inputs, [output])
}
