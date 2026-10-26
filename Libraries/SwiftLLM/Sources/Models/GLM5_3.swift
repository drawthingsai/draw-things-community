import Foundation
import NNC

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
}

// Follow the reference: FP32 mHC residual streams, mixing, router and KDA gates.
// Projections, attention and caches use the activation datatype.

/// Expands token embeddings, followed by injected image embeddings, into the initial
/// `hcCount` identical FP32 residual streams. Either input may be empty; `tokenLength`
/// counts both.
public func GLM5_3Embedding<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, tokenLength: Int,
  configuration: GLM5_3ModelConfiguration = .glm5_3Flash
) -> Model {
  let tokens = Input()
  let injected = Input()
  let embed = Embedding(
    dataType, vocabularySize: configuration.vocabularySize,
    embeddingSize: configuration.hiddenSize, name: "embed_tokens")
  // Pin NHWC so Upsample's width is the stream axis, not the hidden axis.
  let embedding = Concat(axis: 0)(embed(tokens), injected)
    .reshaped(.NHWC(1, tokenLength, 1, configuration.hiddenSize))
  return Model(
    [tokens, injected],
    [
      Upsample(.nearest, widthScale: Float(configuration.hcCount), heightScale: 1)(embedding)
        .to(.Float32)
        .reshaped([tokenLength, configuration.hcCount, configuration.hiddenSize])
    ])
}

/// Inputs: FP32 residual streams `[T, hc, hidden]`. Outputs: post-mix, combination
/// matrix, and the residual collapsed with this sublayer's pre-mix.
public func GLM5_3HCMix(
  prefix: String, residual: Model.IO, tokenLength: Int,
  configuration: GLM5_3ModelConfiguration = .glm5_3Flash
) -> (Model.IO, Model.IO, Model.IO) {
  let hc = configuration.hcCount
  let hidden = configuration.hiddenSize
  let normalized = RMSNorm(
    epsilon: configuration.normEpsilon, axis: [1], elementwiseAffine: false)(
      residual.reshaped([tokenLength, hc * hidden]))
  let mix = Dense(count: configuration.hcMixDim, noBias: true, name: "\(prefix)_fn")(normalized)
  let scale = Parameter<Float>(.GPU(0), .C(3), name: "\(prefix)_scale")
  let base = Parameter<Float>(.GPU(0), .C(configuration.hcMixDim), name: "\(prefix)_base")
  let parts = HyperConnection(
    count: hc, sinkhornIterations: configuration.hcSinkhornIterations,
    epsilon: configuration.hcEpsilon, operation: .splitWeightedSum)(mix, scale, base, residual)
  return (parts[0], parts[1], parts[2].reshaped([tokenLength, hidden]))
}

/// KDA. Inputs: normalized hidden `[T, hidden]`, the causal convolution state
/// `[1, kernel - 1, 1, 3 * linearDim]` and the FP32 recurrent state
/// `[1, heads, headDim (value), headDim (key)]`. Outputs: projection, next
/// convolution state and next recurrent state.
public func GLM5_3LinearAttention<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, prefix: String, x: Model.IO, convState: Model.IO,
  recurrentState: Model.IO, tokenLength: Int,
  configuration: GLM5_3ModelConfiguration = .glm5_3Flash
) -> (Model.IO, Model.IO, Model.IO) {
  let heads = configuration.linearHeads
  let headDim = configuration.linearHeadDim
  let linearDim = configuration.linearDim
  let convDim = configuration.linearConvDim
  let kernel = configuration.linearConvKernel
  // Pin NHWC: the convolution and cache copies are format-sensitive.
  let qkv = Dense(count: convDim, noBias: true, name: "\(prefix).qkv_proj")(x)
    .reshaped(.NHWC(1, tokenLength, 1, convDim))
  let concat = Concat(axis: 1)
  concat.flags = [.disableOpt]
  let convContext = concat(convState, qkv)
  let convStateNext = convContext.reshaped(
    [1, kernel - 1, 1, convDim], offset: [0, tokenLength, 0, 0],
    strides: [(kernel - 1 + tokenLength) * convDim, convDim, convDim, 1]
  ).contiguous()
  // Tiny chunks must keep unit stride; automatic hint inference can choose a larger stride.
  let conv = Convolution(
    groups: convDim, filters: convDim, filterSize: [kernel, 1], noBias: true,
    hint: Hint(stride: [1, 1]), format: .OIHW, name: "\(prefix).conv1d")(convContext)
    .reshaped([tokenLength, convDim]).swish()
  func split(_ index: Int) -> Model.IO {
    conv.reshaped(
      [tokenLength, linearDim], offset: [0, index * linearDim], strides: [convDim, 1]
    )
    .contiguous().reshaped([1, tokenLength, heads, headDim])
  }
  // FLA's l2norm divides by sqrt(sum(x^2) + 1e-6); RMSNorm folds in the head size and
  // the query's 1 / sqrt(headDim) attention scale.
  let epsilon = 1e-6 / Float(headDim)
  let query = RMSNorm(
    epsilon: epsilon, axis: [3], elementwiseAffine: false, scale: 1 / Float(headDim))(split(0))
  let key = RMSNorm(
    epsilon: epsilon, axis: [3], elementwiseAffine: false,
    scale: 1 / Float(headDim).squareRoot())(split(1))
  let value = split(2)
  // Log decay per key channel: lowerBound * sigmoid(exp(A_log) * (f + dt_bias)).
  let forget = Dense(count: linearDim, noBias: true, name: "\(prefix).f_b_proj")(
    Dense(count: headDim, noBias: true, name: "\(prefix).f_a_proj")(x)
  ).to(.Float32)
  let dtBias = Parameter<Float>(.GPU(0), .C(linearDim), name: "\(prefix).dt_bias")
  let aLog = Parameter<Float>(.GPU(0), .NHWC(1, 1, heads, 1), name: "\(prefix).A_log")
  let logDecay =
    configuration.linearGateLowerBound
    * ((forget + dtBias.reshaped([1, linearDim])).reshaped([1, tokenLength, heads, headDim])
    .* aLog.exp()).sigmoid()
  let beta = Dense(count: heads, noBias: true, name: "\(prefix).b_proj")(x).to(.Float32)
    .sigmoid().reshaped([1, tokenLength, heads])
  let recurrent = GatedDelta(logDecay: true, name: "\(prefix).gated_delta")(
    queries: query, keys: key, values: value, decay: logDecay, beta: beta,
    state: recurrentState)
  let gate = Dense(count: linearDim, noBias: true, name: "\(prefix).g_b_proj")(
    Dense(count: headDim, noBias: true, name: "\(prefix).g_a_proj")(x)
  ).reshaped([tokenLength * heads, headDim]).to(.Float32)
  // The reference gated norm runs in FP32 and gates with sigmoid, unlike Qwen3.5's SiLU.
  let normalized = RMSNorm(
    epsilon: configuration.normEpsilon, axis: [1], name: "\(prefix).o_norm")(
      recurrent.output.reshaped([tokenLength * heads, headDim]).to(.Float32))
  let gated = (normalized .* gate.sigmoid()).reshaped([tokenLength, linearDim])
  let out = Dense(count: configuration.hiddenSize, noBias: true, name: "\(prefix).o_proj")(
    gated.to(FloatType.dataType))
  return (out, convStateNext, recurrent.state)
}

/// KPool DSA indexer. Writes this chunk's per-token indexer key and pooling gate into the
/// indexer cache, pools every complete 4-token group with a softmax over `gate + ape`, and
/// selects up to `indexerTopK / indexerPool` pools per query among those it can see.
/// Returns pool ids `[T, kth]` ascending, padded with -1.
public func GLM5_3Indexer<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, prefix: String, x: Model.IO, queryResidual: Model.IO,
  indexerCache: Model.IO, tokenLength: Int, cachedTokenLength: Int,
  configuration: GLM5_3ModelConfiguration = .glm5_3Flash
) -> Model.IO {
  let heads = configuration.indexerHeads
  let dim = configuration.indexerHeadDim
  let pool = configuration.indexerPool
  let totalTokenLength = cachedTokenLength + tokenLength
  let pools = totalTokenLength / pool
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
  // Keep one graph for every chunk: with no complete pool yet, the same ops (and the ape
  // parameter) run on zero-sized tensors, and selection pads with -1.
  // 2-D views with the cache's row stride take ccv's strided-copy fast path; 3-D views fall
  // back to building a new MPSGraph on every call.
  let keys = indexerCache.reshaped(
    [pools * pool, dim], offset: [0, 0], strides: [2 * dim, 1]
  ).contiguous().reshaped([pools, pool, dim])
  let logits = indexerCache.reshaped(
    [pools * pool, dim], offset: [0, dim], strides: [2 * dim, 1]
  ).contiguous().reshaped([pools, pool, dim])
  keys.add(dependencies: [cacheWrite])
  logits.add(dependencies: [cacheWrite])
  let ape = Parameter<FloatType>(
    .GPU(0), .HWC(1, pool, dim), name: "\(prefix).index_kpool_compress_ape")
  let probabilities = (logits + ape).transposed(1, 2).contiguous()
    .reshaped([pools * dim, pool]).softmax().reshaped([pools, dim, pool])
  let pooled = (probabilities .* keys.transposed(1, 2).contiguous())
    .reduced(.sum, axis: [2]).reshaped([pools, dim])
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
/// `[1, cachedTokenLength + T, 1, keyValueLowRank]` and the indexer cache bound as
/// `[1, cachedTokenLength + T, 1, 2 * indexerHeadDim]`; this call writes their last T rows.
public func GLM5_3Attention<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, prefix: String, x: Model.IO, latentCache: Model.IO,
  indexerCache: Model.IO, tokenLength: Int, cachedTokenLength: Int,
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
  let query = Dense(count: heads * qkDim, noBias: true, name: "\(prefix).q_b_proj")(
    queryResidual
  ).reshaped([tokenLength, heads, qkDim]).transposed(0, 1).contiguous()
  // Stored [heads, rank, qkDim] so the contraction is on the last dimension, as with the
  // value projection (and DeepSeek V4.1's wo_a); rowwise quantized codecs need that.
  let keyProjection = Parameter<FloatType>(
    .GPU(0), .HWC(heads, rank, qkDim), name: "\(prefix).kv_b_proj_k")
  let valueProjection = Parameter<FloatType>(
    .GPU(0), .HWC(heads, vDim, rank), name: "\(prefix).kv_b_proj_v")
  let absorbedQuery = Matmul(transposeB: (1, 2))(query, keyProjection).transposed(0, 1).contiguous()
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
    indexerCache: indexerCache, tokenLength: tokenLength, cachedTokenLength: cachedTokenLength,
    configuration: configuration)
  let latents = latentCache.reshaped(.NHWC(1, totalTokenLength, 1, rank))
  let attention = SparseIndexedAttention(
    scale: 1 / Float(qkDim).squareRoot(), isCausal: true, blockSize: configuration.indexerPool)(
      absorbedQuery, latents, latents, latents, latents, indices)
  attention.add(dependencies: [latentWrite])
  let out = Matmul(transposeB: (1, 2))(
    attention.reshaped([tokenLength, heads, rank]).transposed(0, 1).contiguous(), valueProjection
  ).transposed(0, 1).contiguous().reshaped([tokenLength, heads * vDim])
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
/// Input: normalized FP32 hidden `[T, hidden]`.
public func GLM5_3MoE<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, prefix: String, x: Model.IO, tokenLength: Int,
  configuration: GLM5_3ModelConfiguration = .glm5_3Flash
) -> Model.IO {
  let logits = Dense(count: configuration.expertCount, noBias: true, name: "\(prefix).gate")(x)
  let bias = Parameter<Float>(
    .GPU(0), .C(configuration.expertCount), name: "\(prefix).gate.e_score_correction_bias")
  let prepared = MoERouting(
    kth: configuration.routedExperts, weightScale: configuration.routedScalingFactor,
    normalizationEpsilon: 1e-20, singleInputToken: true, sigmoid: true,
    name: "\(prefix).routing")(logits, bias, x)
  let pairs = tokenLength * configuration.routedExperts
  let shared = GLM5_3MLP(
    prefix: "\(prefix).shared_experts", x: x,
    intermediateSize: configuration.sharedIntermediateSize, configuration: configuration)
  let hidden = SegmentedSwiGLU(
    segments: configuration.expertCount, count: configuration.expertIntermediateSize,
    clamp: configuration.swigluLimit, name: "\(prefix).experts")([
      prepared[0], prepared[3], prepared[4], prepared[1].reshaped([pairs, 1]),
    ])
  let routed = SegmentedDense(
    segments: configuration.expertCount, count: configuration.hiddenSize, noBias: true,
    name: "\(prefix).experts.down_proj")(hidden, prepared[3], prepared[4])
  let scattered = Functional.scatterAdd(
    count: tokenLength, countPerOutput: configuration.routedExperts, routed, index: prepared[2]
  ).reshaped([tokenLength, configuration.hiddenSize])
  return scattered + shared
}

/// One decoder layer. Inputs: FP32 residual streams `[T, hc, hidden]`, followed by
/// the convolution and recurrent states for KDA layers, or the latent and indexer caches
/// for attention layers. Outputs: residual streams, then the next KDA states.
public func GLM5_3Layer<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, tokenLength: Int, cachedTokenLength: Int, layerIndex: Int,
  configuration: GLM5_3ModelConfiguration = .glm5_3Flash
) -> Model {
  let prefix = "layers.\(layerIndex)"
  let residual = Input()
  var inputs = [residual]
  var outputs = [Model.IO]()
  let (attnPost, attnCombination, attnCollapsed) = GLM5_3HCMix(
    prefix: "\(prefix).hc_attn", residual: residual, tokenLength: tokenLength,
    configuration: configuration)
  let normalizedAttention = RMSNorm(
    epsilon: configuration.normEpsilon, axis: [1], name: "\(prefix).input_layernorm")(
      attnCollapsed
    ).to(FloatType.dataType)
  let attention: Model.IO
  if configuration.isLinearAttentionLayer(layerIndex) {
    let convState = Input()
    let recurrentState = Input()
    inputs.append(contentsOf: [convState, recurrentState])
    let (out, convStateNext, recurrentStateNext) = GLM5_3LinearAttention(
      dataType, prefix: "\(prefix).self_attn", x: normalizedAttention, convState: convState,
      recurrentState: recurrentState, tokenLength: tokenLength, configuration: configuration)
    attention = out
    outputs.append(contentsOf: [convStateNext, recurrentStateNext])
  } else {
    let latentCache = Input()
    let indexerCache = Input()
    inputs.append(contentsOf: [latentCache, indexerCache])
    attention = GLM5_3Attention(
      dataType, prefix: "\(prefix).self_attn", x: normalizedAttention, latentCache: latentCache,
      indexerCache: indexerCache, tokenLength: tokenLength, cachedTokenLength: cachedTokenLength,
      configuration: configuration)
  }
  let afterAttention = HyperConnection(count: configuration.hcCount, operation: .expand)(
    attention.to(.Float32), residual, attnPost, attnCombination)[0]
  let (ffnPost, ffnCombination, ffnCollapsed) = GLM5_3HCMix(
    prefix: "\(prefix).hc_ffn", residual: afterAttention, tokenLength: tokenLength,
    configuration: configuration)
  let normalizedFFN = RMSNorm(
    epsilon: configuration.normEpsilon, axis: [1], name: "\(prefix).post_attention_layernorm")(
      ffnCollapsed)
  let ffn: Model.IO
  if configuration.isSparseLayer(layerIndex) {
    ffn = GLM5_3MoE(
      dataType, prefix: "\(prefix).mlp", x: normalizedFFN, tokenLength: tokenLength,
      configuration: configuration)
  } else {
    ffn = GLM5_3MLP(
      prefix: "\(prefix).mlp", x: normalizedFFN.to(FloatType.dataType),
      intermediateSize: configuration.intermediateSize, configuration: configuration
    ).to(.Float32)
  }
  let expanded = HyperConnection(count: configuration.hcCount, operation: .expand)(
    ffn, afterAttention, ffnPost, ffnCombination)[0]
  return Model(inputs, [expanded] + outputs)
}

/// Complete text model. Inputs: token IDs, image embeddings `[rows, hidden]` placed after
/// them (one is usually empty; together they make T rows), then per layer in order the KDA
/// convolution and recurrent states, or the latent and indexer caches bound as exactly
/// `[1, cachedTokenLength + T, 1, keyValueLowRank]` and `[1, cachedTokenLength + T, 1,
/// 2 * indexerHeadDim]` (updated in place). Outputs: logits
/// for the last `logitTokenLength` tokens, the next KDA states in layer order, then each
/// layer's residual streams when `outputResiduals` is set. Input order depends only on
/// the architecture, so one graph serves every chunk.
public func GLM5_3CausalLM<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, tokenLength: Int, cachedTokenLength: Int = 0,
  logitTokenLength: Int = 1, outputResiduals: Bool = false,
  configuration: GLM5_3ModelConfiguration = .glm5_3Flash
) -> Model {
  precondition(logitTokenLength >= 1 && logitTokenLength <= tokenLength)
  let hc = configuration.hcCount
  let hidden = configuration.hiddenSize
  let tokens = Input()
  let injected = Input()
  var inputs = [tokens, injected]
  var states = [Model.IO]()
  var residuals = [Model.IO]()
  var residual = GLM5_3Embedding(
    dataType, tokenLength: tokenLength, configuration: configuration)(tokens, injected)
  for layerIndex in 0..<configuration.layers {
    let layerInputs = [Input(), Input()]
    inputs.append(contentsOf: layerInputs)
    let outputs = GLM5_3Layer(
      dataType, tokenLength: tokenLength, cachedTokenLength: cachedTokenLength,
      layerIndex: layerIndex, configuration: configuration)([residual] + layerInputs)
    residual = outputs[0]
    if configuration.isLinearAttentionLayer(layerIndex) {
      states.append(contentsOf: [outputs[1], outputs[2]])
    }
    if outputResiduals { residuals.append(residual) }
  }
  let last = residual.reshaped(
    [logitTokenLength, hc, hidden], offset: [tokenLength - logitTokenLength, 0, 0],
    strides: [hc * hidden, hidden, 1])
  let logits = GLM5_3OutputHead(
    dataType, tokenLength: logitTokenLength, configuration: configuration)(last)
  return Model(inputs, [logits] + states + residuals)
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
