import Foundation
import NNC

#if canImport(SQLite3)
  import SQLite3
#else
  import _SQLite3Shims
#endif

/// GLM-5.3-Flash text decoder: Kimi Delta Attention (KDA) linear attention, NoPE
/// multi-latent attention with a KPool DSA indexer, sigmoid-routed MoE, and four
/// manifold-constrained hyper-connection (mHC) streams.
public struct GLM5_3ModelConfiguration: Codable, Sendable {
  public var vocabularySize = 154_880
  public var hiddenSize = 4_096
  public var layers = 45
  public var hcCount = 4
  public var hcSinkhornIterations = 20
  public var hcEpsilon: Float = 1e-6
  public var normEpsilon: Float = 1e-5
  public var linearHeads = 64
  public var linearHeadDim = 128
  public var linearConvKernel = 4
  public var linearGateLowerBound: Float = -5
  public var attentionHeads = 64
  public var queryLowRank = 1_536
  public var keyValueLowRank = 512
  public var queryKeyHeadDim = 256
  public var valueHeadDim = 256
  public var indexerHeads = 32
  public var indexerHeadDim = 128
  public var indexerTopK = 2_048
  public var indexerPool = 4
  public var denseLayers = 3
  public var intermediateSize = 12_288
  public var expertCount = 288
  public var routedExperts = 8
  public var expertIntermediateSize = 2_048
  public var sharedIntermediateSize = 2_048
  public var routedScalingFactor: Float = 2.5
  public var swigluLimit: Float = 10
  /// Decode expert pool slots of each sparse layer (`denseLayers..<layers`). A layer with fewer
  /// than `expertCount` streams its routed experts through MoEWeightStreaming.
  public var expertResidentSlots = Array(repeating: 288, count: 42)

  public init() {}

  public static let glm5_3Flash = GLM5_3ModelConfiguration()

  public var hcMixDim: Int { (2 + hcCount) * hcCount }
  public var linearDim: Int { linearHeads * linearHeadDim }
  public var linearConvDim: Int { 3 * linearDim }

  /// Three KDA layers precede each DSA layer; the final layer (44) is KDA.
  public func isLinearAttentionLayer(_ layerIndex: Int) -> Bool {
    precondition(layerIndex >= 0 && layerIndex < layers)
    return layerIndex % 4 != 3
  }

  public func isSparseLayer(_ layerIndex: Int) -> Bool {
    precondition(layerIndex >= 0 && layerIndex < layers)
    return layerIndex >= denseLayers
  }

  /// Whether sparse layer `layerIndex` streams its routed experts.
  public func isStreamingLayer(_ layerIndex: Int) -> Bool {
    isSparseLayer(layerIndex) && expertResidentSlots[layerIndex - denseLayers] < expertCount
  }
}

// Follow the reference: FP32 mHC residual streams, mixing, router and KDA gates.
// Projections, attention and caches use the activation datatype.

/// Expands token embeddings, then injected image embeddings, then the tokens after the image,
/// into the initial `hcCount` identical FP32 residual streams. Any input may be empty;
/// `tokenLength` counts all three.
public func GLM5_3Embedding<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, tokenLength: Int,
  configuration: GLM5_3ModelConfiguration = .glm5_3Flash
) -> Model {
  let tokens = Input()
  let injected = Input()
  let postTokens = Input()
  let embed = Embedding(
    dataType, vocabularySize: configuration.vocabularySize,
    embeddingSize: configuration.hiddenSize, name: "embed_tokens")
  // Pin NHWC so Upsample's width is the stream axis, not the hidden axis.
  let embedding = Concat(axis: 0)(embed(tokens), injected, embed(postTokens))
    .reshaped(.NHWC(1, tokenLength, 1, configuration.hiddenSize))
  return Model(
    [tokens, injected, postTokens],
    [
      Upsample(.nearest, widthScale: Float(configuration.hcCount), heightScale: 1)(embedding)
        .to(.Float32)
        .reshaped([tokenLength, configuration.hcCount, configuration.hiddenSize])
    ])
}

/// Inputs: FP32 residual streams `[T, hc, hidden]`. One fused op holds the RMSNorm before the
/// mixing projection, the split, the collapse with this sublayer's pre-mix, and the sublayer RMSNorm
/// `normName`. Outputs: post-mix, combination matrix, and the normalized collapse.
public func GLM5_3HCMix(
  prefix: String, normName: String, residual: Model.IO, tokenLength: Int,
  configuration: GLM5_3ModelConfiguration = .glm5_3Flash
) -> (Model.IO, Model.IO, Model.IO) {
  let hc = configuration.hcCount
  let hidden = configuration.hiddenSize
  let projection = Parameter<Float>(
    .GPU(0), .WC(configuration.hcMixDim, hc * hidden), name: "\(prefix)_fn")
  let scale = Parameter<Float>(.GPU(0), .C(3), name: "\(prefix)_scale")
  let base = Parameter<Float>(.GPU(0), .C(configuration.hcMixDim), name: "\(prefix)_base")
  let normScale = Parameter<Float>(.GPU(0), .C(hidden), name: normName)
  let parts = HyperConnection(
    count: hc, sinkhornIterations: configuration.hcSinkhornIterations,
    epsilon: configuration.hcEpsilon, operation: .mixOwnPre(normEpsilon: configuration.normEpsilon))(
      residual, projection, scale, base, normScale)
  return (parts[1], parts[2], parts[3])
}

/// KDA. Inputs: normalized hidden `[T, hidden]`, the causal convolution state
/// `[1, kernel - 1, 1, 3 * linearDim]` and the FP32 recurrent state
/// `[1, heads, headDim (value), headDim (key)]`, then the buffers the next states go to, which
/// must not overlap them (a graph cannot write what it reads). Output: the projection. The next
/// convolution state is the first `kernel - 1` rows of `convStateNext` and the next recurrent state
/// the first `heads` block of `recurrentStateNext`. With state checkpoints the buffers also hold
/// the states before each of the last rows, so a speculative verify can keep any prefix of its
/// rows: `kernel - 1 + checkpoints` convolution rows that end with the last token, and
/// `1 + checkpoints` recurrent blocks, the state after every row followed by the states before
/// each of the last rows.
public func GLM5_3LinearAttention<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, prefix: String, x: Model.IO, convState: Model.IO,
  recurrentState: Model.IO, convStateNext: Model.IO, recurrentStateNext: Model.IO,
  tokenLength: Int, stateCheckpointCount: Int = 0,
  configuration: GLM5_3ModelConfiguration = .glm5_3Flash
) -> Model.IO {
  precondition(stateCheckpointCount >= 0 && stateCheckpointCount < tokenLength)
  let heads = configuration.linearHeads
  let headDim = configuration.linearHeadDim
  let linearDim = configuration.linearDim
  let convDim = configuration.linearConvDim
  let kernel = configuration.linearConvKernel
  // The short convolution takes SiLU and returns the next rows; queries, keys and values are views
  // of its output. GatedDelta applies FLA's l2norm, x / sqrt(sum(x^2) + 1e-6), with the query's
  // 1 / sqrt(headDim) attention scale, the gate and sigmoid(beta) as it reads them, as FLA's KDA
  // kernel does.
  let qkv = Dense(count: convDim, noBias: true, name: "\(prefix).qkv_proj")(x)
  let conv = CausalConvolution(
    kernelSize: kernel, silu: true, stateCheckpointCount: stateCheckpointCount,
    name: "\(prefix).conv1d")(qkv, previousRows: convState)
  // The move folds into the convolution, which writes the next rows into the buffer.
  let convRows = kernel - 1 + stateCheckpointCount
  let convWrite = conv.nextRows.moved(to: convStateNext.reshaped(.NHWC(1, convRows, 1, convDim)))
  func split(_ index: Int) -> Model.IO {
    conv.output.reshaped(
      [1, tokenLength, heads, headDim], offset: [0, 0, index * heads, 0],
      strides: [tokenLength * convDim, convDim, headDim, 1])
  }
  // Log decay per key channel: lowerBound * sigmoid(A * (f + dt_bias)), A = exp(A_log).
  let forget = Dense(count: linearDim, noBias: true, name: "\(prefix).f_b_proj")(
    Dense(count: headDim, noBias: true, name: "\(prefix).f_a_proj")(x)
  ).reshaped([1, tokenLength, heads, headDim])
  let dtBias = Parameter<Float>(.GPU(0), .C(linearDim), name: "\(prefix).dt_bias")
  // The checkpoint stores A = exp(A_log), so decode does not take the exp of a constant.
  let a = Parameter<Float>(.GPU(0), .NHWC(1, 1, heads, 1), name: "\(prefix).A")
  let beta = Dense(count: heads, noBias: true, name: "\(prefix).b_proj")(x)
    .reshaped([1, tokenLength, heads])
  let recurrent = GatedDelta(
    logDecay: true, stateCheckpointCount: stateCheckpointCount,
    queryKeyL2Norm: true, queryScale: 1 / Float(headDim).squareRoot(),
    sigmoidGateLowerBound: configuration.linearGateLowerBound, sigmoidBeta: true,
    name: "\(prefix).gated_delta")(
      queries: split(0), keys: split(1), values: split(2), gate: forget, beta: beta,
      state: recurrentState, gateBias: dtBias, gateScale: a)
  // The move folds into GatedDelta, which writes its states straight into the buffer.
  let stateWrite = recurrent.state.moved(
    to: recurrentStateNext.reshaped(
      .NHWC(1, (1 + stateCheckpointCount) * heads, headDim, headDim)))
  let gate = Dense(count: linearDim, noBias: true, name: "\(prefix).g_b_proj")(
    Dense(count: headDim, noBias: true, name: "\(prefix).g_a_proj")(x)
  ).reshaped([tokenLength * heads, headDim])
  // The reference gated norm runs in FP32 and gates with sigmoid, unlike Qwen3.5's SiLU; the
  // fused kernel computes in FP32 and rounds once.
  let gated = RMSNormGated(
    epsilon: configuration.normEpsilon, axis: [1], sigmoid: true, name: "\(prefix).o_norm")(
      recurrent.output.reshaped([tokenLength * heads, headDim]), gate)
  let out = Dense(count: configuration.hiddenSize, noBias: true, name: "\(prefix).o_proj")(
    gated.reshaped([tokenLength, linearDim]))
  out.add(dependencies: [convWrite, stateWrite])
  return out
}

/// How much of the checkpoint to map with `DynamicGraph.setWholeFileMappingSizeLimit`: up to the
/// end of the last text tensor loaded whole, which is everything but the expert banks of streaming
/// layers, so those stay out of the pinned mapping; 0 maps the whole checkpoint, and 1 (when the
/// layout cannot be read) maps each tensor on its own. Tensors kept
/// outside SQLite record their position past the trailer, whose start is the header's user version
/// (DeepSeek4_1TextGenerator.residentMappingSize).
public func GLM5_3ResidentMappingSize(
  checkpoint filePath: String, configuration: GLM5_3ModelConfiguration
)
  -> Int
{
  guard (0..<configuration.layers).contains(where: { configuration.isStreamingLayer($0) }) else {
    return 0
  }
  var database: OpaquePointer?
  defer { sqlite3_close(database) }
  var trailer: OpaquePointer?
  defer { sqlite3_finalize(trailer) }
  var statement: OpaquePointer?
  defer { sqlite3_finalize(statement) }
  guard sqlite3_open_v2(filePath, &database, SQLITE_OPEN_READONLY, nil) == SQLITE_OK,
    sqlite3_prepare_v2(database, "PRAGMA user_version", -1, &trailer, nil) == SQLITE_OK,
    sqlite3_step(trailer) == SQLITE_ROW,
    sqlite3_prepare_v2(database, "SELECT name, type, data FROM tensors", -1, &statement, nil)
      == SQLITE_OK
  else { return 1 }
  let trailerOffset = Int(UInt32(truncatingIfNeeded: sqlite3_column_int64(trailer, 0)))
  var end = 0
  while sqlite3_step(statement) == SQLITE_ROW {
    guard let name = sqlite3_column_text(statement, 0).map({ String(cString: $0) }),
      name.hasPrefix("__text_model__["),
      (sqlite3_column_int64(statement, 1) >> 32) & 0x1000_0000 != 0,
      let data = sqlite3_column_blob(statement, 2)
    else { continue }
    if name.contains(".mlp.experts"),
      let layer = name.components(separatedBy: "layers.").last?.components(separatedBy: ".").first
        .flatMap(Int.init), configuration.isStreamingLayer(layer)
    {
      continue
    }
    // Codecs with a block size lead with it, in a word of its own.
    let bytes = Int(sqlite3_column_bytes(statement, 2))
    guard bytes == 16 || bytes == 24 else { return 1 }
    let offset = data.loadUnaligned(fromByteOffset: bytes - 16, as: UInt64.self)
    let length = data.loadUnaligned(fromByteOffset: bytes - 8, as: UInt64.self)
    end = max(end, trailerOffset + Int(offset) + Int(length))
  }
  // On Metal a tensor's buffer takes up whole pages, however short its stored bytes are, and
  // has to fit inside the mapping. Elsewhere the limit is ignored.
  let pageSize = Int(getpagesize())
  return (end + pageSize - 1) & -pageSize
}

/// KPool DSA indexer. Writes this chunk's per-token indexer key and pooling gate into the
/// indexer cache, pools the 4-token groups this chunk completes with a softmax over `gate + ape`
/// into the pooled cache `[maximum pools, indexerHeadDim]`, and selects up to
/// `indexerTopK / indexerPool` pools per query among those it can see.
/// Returns pool ids `[T, kth]` ascending, padded with -1.
public func GLM5_3Indexer<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, prefix: String, x: Model.IO, queryResidual: Model.IO,
  indexerCache: Model.IO, pooledCache: Model.IO, tokenLength: Int, cachedTokenLength: Int,
  configuration: GLM5_3ModelConfiguration = .glm5_3Flash
) -> Model.IO {
  let heads = configuration.indexerHeads
  let dim = configuration.indexerHeadDim
  let pool = configuration.indexerPool
  let totalTokenLength = cachedTokenLength + tokenLength
  let pools = totalTokenLength / pool
  // A pool only reads its own rows, so pools complete before this chunk stay as they are.
  let firstPool = cachedTokenLength / pool
  let newPools = pools - firstPool
  let query = Dense(count: heads * dim, noBias: true, name: "\(prefix).wq_b")(queryResidual)
  let key = LayerNorm(epsilon: 1e-6, axis: [1], name: "\(prefix).k_norm")(
    Dense(count: dim, noBias: true, name: "\(prefix).wk")(x))
  let gate = Dense(count: dim, noBias: true, name: "\(prefix).index_kpool_compress_gate")(x)
  let weights = Dense(count: heads, noBias: true, name: "\(prefix).weights_proj")(x)
  // Cache rows are [key | gate]. Without disableOpt the concat lets LayerNorm write a
  // strided slice of its output, which that kernel does not support (only row 0 lands).
  let concat = Concat(axis: 1)
  concat.flags = [.disableOpt]
  let rows = concat(key, gate).reshaped(.NHWC(1, tokenLength, 1, 2 * dim))
  let cacheWrite = rows.moved(
    to: indexerCache.reshaped(
      .NHWC(1, tokenLength, 1, 2 * dim), offset: [0, cachedTokenLength, 0, 0],
      strides: [totalTokenLength * 2 * dim, 2 * dim, 2 * dim, 1]),
    flags: [.disableOpt])
  // Keep one graph for every chunk: with no pool to complete, the same ops (and the ape
  // parameter) run on zero-sized tensors, and with none at all selection pads with -1.
  // 2-D views with the cache's row stride take ccv's strided-copy fast path; 3-D views fall
  // back to building a new MPSGraph on every call.
  let keys = indexerCache.reshaped(
    [newPools * pool, dim], offset: [firstPool * pool, 0], strides: [2 * dim, 1]
  ).contiguous().reshaped([newPools, pool, dim])
  let logits = indexerCache.reshaped(
    [newPools * pool, dim], offset: [firstPool * pool, dim], strides: [2 * dim, 1]
  ).contiguous().reshaped([newPools, pool, dim])
  keys.add(dependencies: [cacheWrite])
  logits.add(dependencies: [cacheWrite])
  let ape = Parameter<FloatType>(
    .GPU(0), .HWC(1, pool, dim), name: "\(prefix).index_kpool_compress_ape")
  let probabilities = (logits + ape).transposed(1, 2).contiguous()
    .reshaped([newPools * dim, pool]).softmax().reshaped([newPools, dim, pool])
  let pooledWrite = (probabilities .* keys.transposed(1, 2).contiguous())
    .reduced(.sum, axis: [2]).reshaped([newPools, dim], format: .NHWC).moved(
      to: pooledCache.reshaped(
        [newPools, dim], offset: [firstPool, 0], strides: [dim, 1], format: .NHWC),
      flags: [.disableOpt])
  let pooled = pooledCache.reshaped([pools, dim], offset: [0, 0], strides: [dim, 1])
  pooled.add(dependencies: [pooledWrite])
  // GLM scores sum_h w_h * relu(q_h . k / sqrt(dim)) / sqrt(heads); the positive scales
  // factor out of the ReLU into one score scale.
  return ScaledDotProductArgPartition(
    kth: max(1, min(configuration.indexerTopK / pool, pools)),
    scale: 1 / (Float(dim).squareRoot() * Float(heads).squareRoot()), isCausal: true,
    compressionRatio: pool, queryOffset: cachedTokenLength, sortIndices: true)(
      query.reshaped([tokenLength, heads, dim]), pooled, weights.reshaped([tokenLength, heads]))
}

/// NoPE multi-latent attention with KPool DSA, in the absorbed form: queries move into the
/// latent space through each head's key projection, attention reads the 512-wide latent
/// cache as both key and value, and each head's value projection applies afterwards.
/// Blocked sparse indexed attention attends to the indexer's selected 4-token pools plus
/// the query's own incomplete pool, as the reference's top-k with its visible tail.
/// Inputs: normalized hidden `[T, hidden]`, the latent cache bound as exactly
/// `[1, cachedTokenLength + T, 1, keyValueLowRank]`, the indexer cache bound as
/// `[1, cachedTokenLength + T, 1, 2 * indexerHeadDim]`, and the pooled cache
/// `[maximum pools, indexerHeadDim]`; this call writes the first two's last T rows and the pools
/// it completes.
public func GLM5_3Attention<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, prefix: String, x: Model.IO, latentCache: Model.IO,
  indexerCache: Model.IO, pooledCache: Model.IO, tokenLength: Int, cachedTokenLength: Int,
  configuration: GLM5_3ModelConfiguration = .glm5_3Flash
) -> Model.IO {
  let heads = configuration.attentionHeads
  let qkDim = configuration.queryKeyHeadDim
  let vDim = configuration.valueHeadDim
  let rank = configuration.keyValueLowRank
  let totalTokenLength = cachedTokenLength + tokenLength
  let queryResidual = RMSNorm(
    epsilon: configuration.normEpsilon, axis: [1], name: "\(prefix).q_a_layernorm")(
      Dense(count: configuration.queryLowRank, noBias: true, name: "\(prefix).q_a_proj")(x))
  // The absorbed products batch over heads with token-major rows, [T, heads, 1, ...], as
  // DeepSeek V4.1's wo_a does, so no transposes run.
  let query = Dense(count: heads * qkDim, noBias: true, name: "\(prefix).q_b_proj")(
    queryResidual
  ).reshaped([tokenLength, heads, 1, qkDim])
  // Stored [heads, rank, qkDim] so the contraction is on the last dimension, as with the
  // value projection (and DeepSeek V4.1's wo_a); rowwise quantized codecs need that.
  let keyProjection = Parameter<FloatType>(
    .GPU(0), .HWC(heads, rank, qkDim), name: "\(prefix).kv_b_proj_k")
  let valueProjection = Parameter<FloatType>(
    .GPU(0), .HWC(heads, vDim, rank), name: "\(prefix).kv_b_proj_v")
  let absorbedQuery = Matmul(transposeB: (1, 2))(query, keyProjection)
    .reshaped(.NHWC(1, tokenLength, heads, rank))
  let latent = RMSNorm(
    epsilon: configuration.normEpsilon, axis: [1], name: "\(prefix).kv_a_layernorm")(
      Dense(count: rank, noBias: true, name: "\(prefix).kv_a_proj_with_mqa")(x))
  // As with the DeepSeek4 caches, the caller binds exactly the visible rows. Pin NHWC on
  // both sides: a format mismatch turns this copy into an out-of-bounds transpose.
  let latentWrite = latent.reshaped(.NHWC(1, tokenLength, 1, rank)).moved(
    to: latentCache.reshaped(
      .NHWC(1, tokenLength, 1, rank), offset: [0, cachedTokenLength, 0, 0],
      strides: [totalTokenLength * rank, rank, rank, 1]),
    flags: [.disableOpt])
  let indices = GLM5_3Indexer(
    dataType, prefix: "\(prefix).indexer", x: x, queryResidual: queryResidual,
    indexerCache: indexerCache, pooledCache: pooledCache, tokenLength: tokenLength,
    cachedTokenLength: cachedTokenLength, configuration: configuration)
  let latents = latentCache.reshaped(.NHWC(1, totalTokenLength, 1, rank))
  let attention = SparseIndexedAttention(
    scale: 1 / Float(qkDim).squareRoot(), isCausal: true, blockSize: configuration.indexerPool)(
      absorbedQuery, latents, latents, latents, latents, indices)
  attention.add(dependencies: [latentWrite])
  let out = Matmul(transposeB: (1, 2))(
    attention.reshaped([tokenLength, heads, 1, rank]), valueProjection
  ).reshaped([tokenLength, heads * vDim])
  return Dense(count: configuration.hiddenSize, noBias: true, name: "\(prefix).o_proj")(out)
}

/// Dense SwiGLU MLP with the reference's gate/up clamping.
public func GLM5_3MLP(
  prefix: String, x: Model.IO, intermediateSize: Int,
  configuration: GLM5_3ModelConfiguration = .glm5_3Flash
) -> Model.IO {
  let hidden = SwiGLU(count: intermediateSize, clamp: configuration.swigluLimit, name: prefix)(x)
  return Dense(count: configuration.hiddenSize, noBias: true, name: "\(prefix).down_proj")(hidden)
}

/// Sigmoid-scored MoE with a shared expert. Experts evaluate and accumulate in FP32.
/// Input: normalized FP32 hidden `[T, hidden]`. Outputs: the routed experts' sum and the shared
/// expert, which the layer adds in its expansion.
/// With `residentSlots` below `expertCount`, the routed experts stream through a decode pool of that
/// many slots: loaders read their banks as `.externalOnDemand`, and the shared
/// expert runs on its own, before the routed experts wait for their weights. With `outputRouting`,
/// `routing` holds the pairs' tokens grouped by expert, then each group's expert and count
/// (MoERouting); otherwise it is empty, since an output nothing reads breaks the graph build.
public func GLM5_3MoE<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, prefix: String, x: Model.IO, tokenLength: Int,
  residentSlots: Int? = nil, outputRouting: Bool = false,
  configuration: GLM5_3ModelConfiguration = .glm5_3Flash
) -> (routed: Model.IO, shared: Model.IO, routing: [Model.IO]) {
  let logits = Dense(count: configuration.expertCount, noBias: true, name: "\(prefix).gate")(x)
  let bias = Parameter<Float>(
    .GPU(0), .C(configuration.expertCount), name: "\(prefix).gate.e_score_correction_bias")
  let prepared = MoERouting(
    kth: configuration.routedExperts, weightScale: configuration.routedScalingFactor,
    normalizationEpsilon: 1e-20, singleInputToken: true, sigmoid: true,
    name: "\(prefix).routing")(logits, bias, x)
  let pairs = tokenLength * configuration.routedExperts
  if let residentSlots, residentSlots < configuration.expertCount {
    precondition(residentSlots >= configuration.routedExperts)
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
      name: "\(prefix).experts.down_proj")()
    let resident = MoEWeightStreaming(
      residentSlots: residentSlots, routingWidth: configuration.routedExperts,
      name: "\(prefix).expert_streaming")([
        prepared[3], prepared[4], prepared[1].reshaped([pairs, 1]), gateWeight, upWeight,
        downWeight,
      ])
    let sharedHidden = SwiGLU(
      count: configuration.sharedIntermediateSize, clamp: configuration.swigluLimit,
      name: "\(prefix).shared_experts")(x)
    sharedHidden.add(dependencies: [resident[3], resident[4], resident[5]])
    let shared = Dense(
      count: configuration.hiddenSize, noBias: true, name: "\(prefix).shared_experts.down_proj")(
        sharedHidden)
    let hidden = SegmentedSwiGLU(
      segments: 0, count: configuration.expertIntermediateSize, clamp: configuration.swigluLimit,
      functional: true, name: "\(prefix).experts.functional")([
        prepared[0], resident[0], resident[1], resident[3], resident[4], resident[2],
      ])
    hidden.add(dependencies: [shared])
    let routed = SegmentedDense(
      segments: 0, count: configuration.hiddenSize, noBias: true, functional: true,
      name: "\(prefix).experts.functional.down_proj")([
        hidden, resident[0], resident[1], resident[5],
      ])
    let scattered = Functional.scatterAdd(
      count: tokenLength, countPerOutput: configuration.routedExperts, routed, index: prepared[2]
    ).reshaped([tokenLength, configuration.hiddenSize])
    return (scattered, shared, outputRouting ? [prepared[2], prepared[3], prepared[4]] : [])
  }
  // The shared expert joins the routed decode dispatch with the routed experts' intermediate
  // width. Its gate and up weights follow theirs as `experts-0-2` and `experts-0-3`, which
  // loaders map to the checkpoint's `shared_experts-0-0` and `shared_experts-0-1`.
  precondition(configuration.sharedIntermediateSize == configuration.expertIntermediateSize)
  let hidden = SegmentedSwiGLU(
    segments: configuration.expertCount, count: configuration.expertIntermediateSize,
    clamp: configuration.swigluLimit, sharedExpertFused: true, name: "\(prefix).experts")([
      prepared[0], prepared[3], prepared[4], prepared[1].reshaped([pairs, 1]), x,
    ])
  // The shared expert's down projection joins the routed one, as `experts.down_proj-0-1`, which
  // loaders map to the checkpoint's `shared_experts.down_proj-0-0`.
  let down = SegmentedDense(
    segments: configuration.expertCount, count: configuration.hiddenSize, noBias: true,
    sharedExpertFused: true, name: "\(prefix).experts.down_proj")(
      hidden[0], prepared[3], prepared[4], hidden[1])
  let scattered = Functional.scatterAdd(
    count: tokenLength, countPerOutput: configuration.routedExperts, down[0], index: prepared[2]
  ).reshaped([tokenLength, configuration.hiddenSize])
  return (scattered, down[1], outputRouting ? [prepared[2], prepared[3], prepared[4]] : [])
}

/// One decoder layer. Inputs: FP32 residual streams `[T, hc, hidden]`, followed by
/// the convolution and recurrent states and the buffers their next values go to for KDA layers
/// (GLM5_3LinearAttention), or the latent and indexer caches for attention layers. Output:
/// residual streams.
public func GLM5_3Layer<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, tokenLength: Int, cachedTokenLength: Int, layerIndex: Int,
  stateCheckpointCount: Int = 0, outputRouting: Bool = false,
  configuration: GLM5_3ModelConfiguration = .glm5_3Flash
) -> Model {
  let prefix = "layers.\(layerIndex)"
  let residual = Input()
  var inputs = [residual]
  let (attnPost, attnCombination, attnNormalized) = GLM5_3HCMix(
    prefix: "\(prefix).hc_attn", normName: "\(prefix).input_layernorm", residual: residual,
    tokenLength: tokenLength, configuration: configuration)
  let normalizedAttention = attnNormalized.to(FloatType.dataType)
  let attention: Model.IO
  if configuration.isLinearAttentionLayer(layerIndex) {
    let states = (0..<4).map { _ in Input() }
    inputs.append(contentsOf: states)
    attention = GLM5_3LinearAttention(
      dataType, prefix: "\(prefix).self_attn", x: normalizedAttention, convState: states[0],
      recurrentState: states[1], convStateNext: states[2], recurrentStateNext: states[3],
      tokenLength: tokenLength, stateCheckpointCount: stateCheckpointCount,
      configuration: configuration)
  } else {
    let latentCache = Input()
    let indexerCache = Input()
    let pooledCache = Input()
    inputs.append(contentsOf: [latentCache, indexerCache, pooledCache])
    attention = GLM5_3Attention(
      dataType, prefix: "\(prefix).self_attn", x: normalizedAttention, latentCache: latentCache,
      indexerCache: indexerCache, pooledCache: pooledCache, tokenLength: tokenLength,
      cachedTokenLength: cachedTokenLength, configuration: configuration)
  }
  // Expansion reads the FP16 projection directly.
  let afterAttention = HyperConnection(count: configuration.hcCount, operation: .expand)(
    attention, residual, attnPost, attnCombination)[0]
  let (ffnPost, ffnCombination, normalizedFFN) = GLM5_3HCMix(
    prefix: "\(prefix).hc_ffn", normName: "\(prefix).post_attention_layernorm",
    residual: afterAttention, tokenLength: tokenLength, configuration: configuration)
  let expanded: Model.IO
  var routing = [Model.IO]()
  if configuration.isSparseLayer(layerIndex) {
    let moe = GLM5_3MoE(
      dataType, prefix: "\(prefix).mlp", x: normalizedFFN, tokenLength: tokenLength,
      residentSlots: configuration.expertResidentSlots[layerIndex - configuration.denseLayers],
      outputRouting: outputRouting, configuration: configuration)
    routing = moe.routing
    expanded =
      HyperConnection(count: configuration.hcCount, operation: .expandSum)(
        moe.routed, afterAttention, ffnPost, ffnCombination, moe.shared)[0]
  } else {
    let ffn = GLM5_3MLP(
      prefix: "\(prefix).mlp", x: normalizedFFN.to(FloatType.dataType),
      intermediateSize: configuration.intermediateSize, configuration: configuration)
    expanded =
      HyperConnection(count: configuration.hcCount, operation: .expand)(
        ffn, afterAttention, ffnPost, ffnCombination)[0]
  }
  return Model(inputs, [expanded] + routing)
}

/// Complete text model. Inputs: token IDs, image embeddings `[rows, hidden]` placed after
/// them, and the token IDs after the image (any may be empty; together they make T rows), so a
/// chunk holds at most one image. Then per layer in order the KDA
/// convolution and recurrent states and the two buffers their next values go to
/// (GLM5_3LinearAttention), or the latent and indexer caches bound as exactly
/// `[1, cachedTokenLength + T, 1, keyValueLowRank]` and `[1, cachedTokenLength + T, 1,
/// 2 * indexerHeadDim]` (updated in place). Outputs: logits for the last `logitTokenLength`
/// tokens, then each layer's residual streams when `outputResiduals` is set, then, with
/// `targetLayers`, the mean streams entering those layers for the last `targetTokenLength` tokens,
/// concatenated as FP32 `[targetTokenLength, targetLayers.count * hidden]`. With `outputRouting`,
/// last come each sparse layer's MoERouting token indices, expert ids and counts (Int32).
/// `stateCheckpointCount` also keeps the KDA states before each of the last rows. Input order
/// depends only on the architecture, so one graph serves every chunk.
public func GLM5_3CausalLM<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, tokenLength: Int, cachedTokenLength: Int = 0,
  logitTokenLength: Int = 1, outputResiduals: Bool = false, targetLayers: [Int] = [],
  targetTokenLength: Int = 0, stateCheckpointCount: Int = 0, outputRouting: Bool = false,
  configuration: GLM5_3ModelConfiguration = .glm5_3Flash
) -> Model {
  precondition(logitTokenLength >= 1 && logitTokenLength <= tokenLength)
  precondition(targetLayers.isEmpty == (targetTokenLength == 0) && targetTokenLength <= tokenLength)
  let hc = configuration.hcCount
  let hidden = configuration.hiddenSize
  let tokens = Input()
  let injected = Input()
  let postTokens = Input()
  var inputs = [tokens, injected, postTokens]
  var residuals = [Model.IO]()
  var residual = GLM5_3Embedding(
    dataType, tokenLength: tokenLength, configuration: configuration)(
      tokens, injected, postTokens)
  var targets = [Model.IO]()
  var routing = [Model.IO]()
  for layerIndex in 0..<configuration.layers {
    if targetLayers.contains(layerIndex) {
      targets.append(
        residual.reshaped(
          [targetTokenLength, hc, hidden], offset: [tokenLength - targetTokenLength, 0, 0],
          strides: [hc * hidden, hidden, 1]
        ).reduced(.mean, axis: [1]).reshaped([targetTokenLength, hidden]))
    }
    let layerInputs = (0..<(configuration.isLinearAttentionLayer(layerIndex) ? 4 : 3)).map { _ in
      Input()
    }
    inputs.append(contentsOf: layerInputs)
    let layerOutputRouting = outputRouting && configuration.isSparseLayer(layerIndex)
    let layerOutputs = GLM5_3Layer(
      dataType, tokenLength: tokenLength, cachedTokenLength: cachedTokenLength,
      layerIndex: layerIndex, stateCheckpointCount: stateCheckpointCount,
      outputRouting: layerOutputRouting, configuration: configuration)([residual] + layerInputs)
    residual = layerOutputRouting ? layerOutputs[0] : layerOutputs
    if layerOutputRouting { routing.append(contentsOf: (1...3).map { layerOutputs[$0] }) }
    if outputResiduals { residuals.append(residual) }
  }
  let last = residual.reshaped(
    [logitTokenLength, hc, hidden], offset: [tokenLength - logitTokenLength, 0, 0],
    strides: [hc * hidden, hidden, 1])
  let logits = GLM5_3OutputHead(
    dataType, tokenLength: logitTokenLength, configuration: configuration)(last)
  var targetOutputs = [Model.IO]()
  if !targets.isEmpty {
    let concat = Concat(axis: 1)
    concat.flags = [.disableOpt]
    targetOutputs.append(concat(targets))
  }
  return Model(inputs, [logits] + residuals + targetOutputs + routing)
}

/// GLM-5.3 collapses the final streams with an unweighted mean, unlike DeepSeek-V4's
/// learned pre-mix. Normalization is FP32; the vocabulary projection uses the
/// activation datatype.
public func GLM5_3OutputHead<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, tokenLength: Int,
  configuration: GLM5_3ModelConfiguration = .glm5_3Flash
) -> Model {
  let x = Input()
  let collapsed = x.reduced(.mean, axis: [1]).reshaped([tokenLength, configuration.hiddenSize])
  let norm = RMSNorm(epsilon: configuration.normEpsilon, axis: [1], name: "norm")
  let head = Dense(count: configuration.vocabularySize, noBias: true, name: "lm_head")
  return Model([x], [head(norm(collapsed).to(FloatType.dataType))])
}
