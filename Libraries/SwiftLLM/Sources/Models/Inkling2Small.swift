import Foundation
import NNC

#if canImport(SQLite3)
  import SQLite3
#else
  import _SQLite3Shims
#endif

/// Inkling 2 Small text decoder: 48 layers of grouped-query attention, five sliding-window layers
/// to each full-attention one, with a learned relative-position bias, per-token attention sinks,
/// L2-clipped queries and keys, short causal convolutions around every sublayer, sigmoid-routed
/// MoE with softcapped SwiGLU experts, and block attention residuals: each sublayer reads a
/// softmax mixture, over depth, of earlier block states and the current partial block.
public struct Inkling2SmallConfiguration: Codable, Sendable {
  public var vocabularySize = 201_024
  /// Rows of the unembedding that hold real tokens; logits beyond are dropped.
  public var unpaddedVocabularySize = 200_058
  public var hiddenSize = 4_608
  public var layers = 48
  public var attentionHeads = 48
  public var keyValueHeads = 8
  public var headDim = 128
  /// Per-head width of the relative states mixed into the relative-position bias.
  public var relativeDim = 16
  /// Backward distances the relative-position bias covers.
  public var relativeExtent = 1_024
  public var slidingWindow = 1_024
  public var logScalingFloor = 128_000
  public var logScalingAlpha: Float = 0.1
  public var normEpsilon: Float = 1e-6
  public var queryKeyClip: Float = 22.627416997969522
  public var relativeClip: Float = 8
  public var convolutionKernel = 4
  /// Layers summed into one block before it becomes a source of the attention residuals.
  public var layersPerBlock = 6
  public var expertCount = 384
  public var routedExperts = 8
  public var expertIntermediateSize = 2_304
  public var gateSoftcap: Float = 4
  public var upSoftcap: Float = 25
  public var routedScalingFactor: Float = 8
  public var logitSoftcap: Float = 30
  /// Decode expert pool slots of each layer. A layer with fewer than `expertCount` streams its
  /// routed experts through MoEWeightStreaming.
  public var expertResidentSlots = Array(repeating: 384, count: 48)

  public init() {}

  public static let inkling2Small = Inkling2SmallConfiguration()

  /// Model inputs of each layer, in order: the key and value caches bound from their first row
  /// through this chunk's rows, the four convolution rows (key and value in the activation
  /// datatype, post-attention and post-MLP in FP32), each with at least `convolutionOffset` rows
  /// before its kernel - 1, and the four buffers their next values go to, which must not overlap
  /// them.
  public static let layerInputCount = 10

  public var queryDim: Int { attentionHeads * headDim }
  public var keyValueDim: Int { keyValueHeads * headDim }

  /// Five sliding-window layers precede each full-attention layer.
  public func isFullAttentionLayer(_ layerIndex: Int) -> Bool {
    precondition(layerIndex >= 0 && layerIndex < layers)
    return (layerIndex + 1) % layersPerBlock == 0
  }

  /// A block starts at this layer: its input joins the block states, and its sublayers restart
  /// the partial block.
  public func isBlockBoundary(_ layerIndex: Int) -> Bool {
    precondition(layerIndex >= 0 && layerIndex < layers)
    return layerIndex % layersPerBlock == 0
  }

  /// Whether the first `layers` layers include a full-attention layer.
  public var hasFullAttentionLayer: Bool { layers >= layersPerBlock }

  /// Whether layer `layerIndex` streams its routed experts.
  public func isStreamingLayer(_ layerIndex: Int) -> Bool {
    expertResidentSlots[layerIndex] < expertCount
  }

  /// Cached rows a chunk's attention reads: all of them in full-attention layers, the last
  /// `slidingWindow - 1` in sliding-window layers. A sliding-window cache may be bound with more
  /// (`cachedSlidingTokenLength`), so it only needs compacting once in a while.
  public func cachedKeyCount(fullAttention: Bool, cachedTokenLength: Int) -> Int {
    fullAttention ? cachedTokenLength : min(cachedTokenLength, slidingWindow - 1)
  }

  /// Full-attention layers scale queries and relative states by
  /// 1 + alpha * log(max(1, (position + 1) / floor)).
  public func logScale(position: Int) -> Float {
    1 + logScalingAlpha * Foundation.log(max(1, Float(position + 1) / Float(logScalingFloor)))
  }
}

/// How much of the checkpoint to map with `DynamicGraph.setWholeFileMappingSizeLimit`: up to the
/// end of the last text tensor loaded whole, which is everything but the expert banks of streaming
/// layers, so those stay out of the pinned mapping; 0 maps the whole checkpoint, and 1 (when the
/// layout cannot be read) maps each tensor on its own. Tensors kept outside SQLite record their
/// position past the trailer, whose start is the header's user version (GLM5_3ResidentMappingSize).
public func Inkling2SmallResidentMappingSize(
  checkpoint filePath: String, configuration: Inkling2SmallConfiguration
) -> Int {
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
        .flatMap(Int.init), layer < configuration.layers, configuration.isStreamingLayer(layer)
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

/// Full-attention log scales for a chunk's positions, `[T, 1]`.
public func Inkling2SmallLogScale<FloatType: TensorNumeric & BinaryFloatingPoint>(
  _ dataType: FloatType.Type, cachedTokenLength: Int, tokenLength: Int,
  configuration: Inkling2SmallConfiguration = .inkling2Small
) -> Tensor<FloatType> {
  Tensor<FloatType>(
    (0..<tokenLength).map { FloatType(configuration.logScale(position: cachedTokenLength + $0)) },
    kind: .CPU, format: .NHWC, shape: [tokenLength, 1])
}

// As in the reference, residuals, block states, attention residuals, the router and experts are
// FP32. Attention runs in the activation datatype; its norms and convolutions accumulate in FP32.
// Every short convolution holds the reference's residual and scale in its stored weights, and
// writes its next rows into their buffer: the move folds into the convolution.

/// Attention of one layer. Inputs: the normalized query, key and value inputs (the same for
/// sliding-window layers) `[T, hidden]` in the activation datatype; the key and value caches
/// `[1, cachedKeyCount + T, keyValueHeads, headDim]`, the cached rows before this chunk's, which
/// this call writes; the convolution rows `[k, v]` and the buffers their next values go to; and,
/// in full-attention layers, the log scales `[T, 1]`. Output: the output projection `[T, hidden]`.
///
/// Each query adds its own relative-position bias, mixed from its relative states, to the last
/// `relativeExtent` keys, and its own sink. The learned scales before the reference's L2 clips
/// are in the stored query and relative projections.
public func Inkling2SmallAttention<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, prefix: String, layerIndex: Int, queryInput: Model.IO,
  keyInput: Model.IO, valueInput: Model.IO, keyCache: Model.IO, valueCache: Model.IO,
  convolutionRows: [Model.IO], nextConvolutionRows: [Model.IO], logScale: Model.IO?,
  tokenLength: Int, cachedKeyCount: Int, convolutionCheckpoints: Int = 0,
  configuration: Inkling2SmallConfiguration = .inkling2Small
) -> Model.IO {
  let isFull = configuration.isFullAttentionLayer(layerIndex)
  precondition(isFull == (logScale != nil))
  let T = tokenLength
  let heads = configuration.attentionHeads
  let kvHeads = configuration.keyValueHeads
  let headDim = configuration.headDim
  let kvDim = configuration.keyValueDim
  let relativeDim = configuration.relativeDim
  let kernel = configuration.convolutionKernel
  let keyCount = cachedKeyCount + T
  // An L2 clip of n values at a threshold is an RMSNorm of scale threshold / sqrt(n) that never
  // scales up.
  var query = RMSNorm(
    epsilon: 1e-12, axis: [2], elementwiseAffine: false,
    scale: configuration.queryKeyClip / Float(headDim).squareRoot(), clip: true)(
      Dense(count: heads * headDim, noBias: true, name: "\(prefix).q_proj")(queryInput).reshaped([
        T, heads, headDim,
      ]))
  var relative = RMSNorm(
    epsilon: 1e-12, axis: [2], elementwiseAffine: false,
    scale: configuration.relativeClip / Float(relativeDim).squareRoot(), clip: true)(
      Dense(count: heads * relativeDim, noBias: true, name: "\(prefix).r_proj")(queryInput)
        .reshaped([T, heads, relativeDim]))
  if let logScale {
    let scale = logScale.reshaped([T, 1, 1])
    query = query .* scale
    relative = relative .* scale
  }
  // Full-attention layers add each token's dynamic weights to the key and value convolutions, one
  // set to a key/value head.
  let nextRows = kernel - 1 + convolutionCheckpoints
  let keyConvolution = CausalConvolution(
    kernelSize: kernel, stateCheckpointCount: convolutionCheckpoints, hasDynamicWeights: isFull,
    name: "\(prefix).k_sconv")(
      [Dense(count: kvDim, noBias: true, name: "\(prefix).k_proj")(keyInput), convolutionRows[0]]
        + (isFull
          ? [
            Dense(count: kvHeads * kernel, noBias: true, name: "\(prefix).k_sconv.weight_proj")(
              keyInput)
          ] : []))
  let key = RMSNorm(
    epsilon: 1e-12, axis: [2], elementwiseAffine: false,
    scale: configuration.queryKeyClip / Float(headDim).squareRoot(), clip: true)(
      keyConvolution[0].reshaped([T, kvHeads, headDim]))
  key.add(dependencies: [
    keyConvolution[1].moved(to: nextConvolutionRows[0].reshaped(.NHWC(1, nextRows, 1, kvDim)))
  ])
  let valueConvolution = CausalConvolution(
    kernelSize: kernel, stateCheckpointCount: convolutionCheckpoints, hasDynamicWeights: isFull,
    name: "\(prefix).v_sconv")(
      [Dense(count: kvDim, noBias: true, name: "\(prefix).v_proj")(valueInput), convolutionRows[1]]
        + (isFull
          ? [
            Dense(count: kvHeads * kernel, noBias: true, name: "\(prefix).v_sconv.weight_proj")(
              valueInput)
          ] : []))
  let value = RMSNorm(epsilon: configuration.normEpsilon, axis: [2], elementwiseAffine: false)(
    valueConvolution[0].reshaped([T, kvHeads, headDim]))
  value.add(dependencies: [
    valueConvolution[1].moved(to: nextConvolutionRows[1].reshaped(.NHWC(1, nextRows, 1, kvDim)))
  ])
  // Row d of the stored projection is the bias profile of the key d positions back.
  let bias = Dense(
    count: configuration.relativeExtent, noBias: true, name: "\(prefix).rel_logits_proj")(
      relative.reshaped([T * heads, relativeDim]))
  let sinks = Dense(count: heads, noBias: true, name: "\(prefix).sink_proj")(queryInput)
  let attention = ScaledDotProductAttention(
    scale: 1 / Float(headDim).squareRoot(), isCausal: true, hasAttentionSinks: true,
    hasRelativeBias: true, slidingWindow: isFull ? 0 : configuration.slidingWindow)(
      query.reshaped(.NHWC(1, T, heads, headDim)), keyCache, valueCache,
      sinks.reshaped([1, T, heads]),
      bias.reshaped(.NHWC(1, T, heads, configuration.relativeExtent)))
  // This chunk's rows are the last T of the bound caches.
  attention.add(dependencies: [
    key.reshaped(.NHWC(1, T, kvHeads, headDim)).moved(
      to: keyCache.reshaped(
        .NHWC(1, T, kvHeads, headDim), offset: [0, keyCount - T, 0, 0],
        strides: [keyCount * kvDim, kvDim, headDim, 1]), flags: [.disableOpt]),
    value.reshaped(.NHWC(1, T, kvHeads, headDim)).moved(
      to: valueCache.reshaped(
        .NHWC(1, T, kvHeads, headDim), offset: [0, keyCount - T, 0, 0],
        strides: [keyCount * kvDim, kvDim, headDim, 1]), flags: [.disableOpt]),
  ])
  let gate = Dense(count: heads * headDim, noBias: true, name: "\(prefix).gate_proj")(queryInput)
  return Dense(count: configuration.hiddenSize, noBias: true, name: "\(prefix).o_proj")(
    attention.reshaped([T, heads * headDim]) .* Sigmoid()(gate))
}

/// Sigmoid-scored MoE with softcapped SwiGLU experts, 4 tanh(g / 4) sigmoid(g) 25 tanh(u / 25),
/// evaluated in FP32. Input: normalized FP32 hidden `[T, hidden]`. Returns the output `[T, hidden]`,
/// the routing, whose outputs group the routed rows by expert (MoERouting: the rows, their
/// weights, their tokens `[8T]`, and the experts of the groups with their row counts), and the
/// weighted SwiGLU output of the rows, the input of the down projection `[8T, intermediate]`.
/// With `residentSlots` below `expertCount`, the routed experts stream through a decode pool of
/// that many slots: loaders read their banks as `.externalOnDemand`.
public func Inkling2SmallMoE(
  prefix: String, x: Model.IO, tokenLength: Int, residentSlots: Int? = nil,
  configuration: Inkling2SmallConfiguration = .inkling2Small
) -> (output: Model.IO, routing: Model.IO, hidden: Model.IO) {
  let logits = Dense(count: configuration.expertCount, noBias: true, name: "\(prefix).gate")(x)
  let bias = Parameter<Float>(
    .GPU(0), .C(configuration.expertCount), name: "\(prefix).gate.e_score_correction_bias")
  let prepared = MoERouting(
    kth: configuration.routedExperts, weightScale: configuration.routedScalingFactor,
    normalizationEpsilon: 1e-20, sigmoid: true, name: "\(prefix).routing")(logits, bias, x)
  let pairs = tokenLength * configuration.routedExperts
  let experts = configuration.expertCount
  let intermediate = configuration.expertIntermediateSize
  let hidden: Model.IO
  let down: Model.IO
  if let residentSlots, residentSlots < experts {
    precondition(residentSlots >= configuration.routedExperts)
    // Loaders map `experts.streaming_gate` and `experts.streaming_up` to the stored gate and up
    // banks, `experts-0-0` and `experts-0-1`.
    let resident = MoEWeightStreaming(
      residentSlots: residentSlots, routingWidth: configuration.routedExperts,
      name: "\(prefix).expert_streaming")([
        prepared[3], prepared[4], prepared[1].reshaped([pairs, 1]),
        Parameter<Float>(
          .GPU(0), .HWC(experts, intermediate, configuration.hiddenSize),
          name: "\(prefix).experts.streaming_gate")(),
        Parameter<Float>(
          .GPU(0), .HWC(experts, intermediate, configuration.hiddenSize),
          name: "\(prefix).experts.streaming_up")(),
        Parameter<Float>(
          .GPU(0), .HWC(experts, configuration.hiddenSize, intermediate),
          name: "\(prefix).experts.down_proj")(),
      ])
    hidden = SegmentedSwiGLU(
      segments: 0, count: intermediate, gateSoftCap: configuration.gateSoftcap,
      upSoftCap: configuration.upSoftcap, functional: true, name: "\(prefix).experts.functional")([
        prepared[0], resident[0], resident[1], resident[3], resident[4], resident[2],
      ])
    down = SegmentedDense(
      segments: 0, count: configuration.hiddenSize, noBias: true, functional: true,
      name: "\(prefix).experts.functional.down_proj")([
        hidden, resident[0], resident[1], resident[5],
      ])
  } else {
    // The gate and up weights are `experts-0-0` and `experts-0-1`.
    hidden = SegmentedSwiGLU(
      segments: experts, count: intermediate, gateSoftCap: configuration.gateSoftcap,
      upSoftCap: configuration.upSoftcap, name: "\(prefix).experts")(
        prepared[0], prepared[3], prepared[4], prepared[1].reshaped([pairs, 1]))
    down = SegmentedDense(
      segments: experts, count: configuration.hiddenSize, noBias: true,
      name: "\(prefix).experts.down_proj")(hidden, prepared[3], prepared[4])
  }
  let output = Functional.scatterAdd(
    count: tokenLength, countPerOutput: configuration.routedExperts, down, index: prepared[2]
  ).reshaped([tokenLength, configuration.hiddenSize])
  return (output, prepared, hidden)
}

/// One decoder layer. `hidden` is the FP32 partial block `[T, hidden]` and `blocks` the block
/// states `[n, T, hidden]`, which a block boundary extends. Each sublayer reads an attention
/// residual: a softmax mixture, by a learned query, of the normalized block states and partial
/// block. Returns the next partial block, and the expert input `[T, hidden]` with the MoE's
/// routing and SwiGLU output (Inkling2SmallMoE).
public func Inkling2SmallLayer<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, layerIndex: Int, hidden: Model.IO, blocks: inout Model.IO?,
  inputs: [Model.IO], logScale: Model.IO?, tokenLength: Int, cachedKeyCount: Int,
  convolutionCheckpoints: Int = 0, convolutionOffset: Int = 0,
  configuration: Inkling2SmallConfiguration = .inkling2Small
) -> (output: Model.IO, experts: (input: Model.IO, routing: Model.IO, hidden: Model.IO)) {
  precondition(inputs.count == Inkling2SmallConfiguration.layerInputCount)
  let prefix = "layers.\(layerIndex)"
  let T = tokenLength
  let h = configuration.hiddenSize
  let epsilon = configuration.normEpsilon
  let kernel = configuration.convolutionKernel
  let isFull = configuration.isFullAttentionLayer(layerIndex)
  // The convolution rows sit `convolutionOffset` rows into their inputs: the rows after a verify's
  // last kept token, which it wrote with those after its other tokens. The batch of one takes the
  // contiguous stride, so the view stays contiguous and the convolutions keep their fast kernels.
  let previousRows = zip(
    inputs[2..<6], [configuration.keyValueDim, configuration.keyValueDim, h, h]
  )
  .map { input, dim in
    input.reshaped(
      .NHWC(1, kernel - 1, 1, dim), offset: [0, convolutionOffset, 0, 0],
      strides: [(kernel - 1) * dim, dim, dim, 1])
  }
  // Layer 0 reads the embedding directly; later layers mix the block states. Full-attention
  // layers mix and normalize their query, key and value inputs separately.
  let sources =
    layerIndex == 0
    ? Array(repeating: hidden, count: isFull ? 3 : 1)
    : AttentionResidual(
      queries: isFull ? 3 : 1, epsilon: epsilon, scale: 1 / Float(h).squareRoot(),
      name: "\(prefix).attention_input")(blocks: blocks!, partial: hidden)
  let normalized = zip(isFull ? ["q_", "k_", "v_"] : [""], sources).map {
    RMSNorm(epsilon: epsilon, axis: [1], name: "\(prefix).\($0)input_layernorm")($1).to(
      FloatType.dataType)
  }
  let attention = Inkling2SmallAttention(
    dataType, prefix: "\(prefix).self_attn", layerIndex: layerIndex, queryInput: normalized[0],
    keyInput: normalized[isFull ? 1 : 0], valueInput: normalized[isFull ? 2 : 0],
    keyCache: inputs[0], valueCache: inputs[1],
    convolutionRows: [previousRows[0], previousRows[1]],
    nextConvolutionRows: [inputs[6], inputs[7]], logScale: isFull ? logScale : nil,
    tokenLength: T, cachedKeyCount: cachedKeyCount,
    convolutionCheckpoints: convolutionCheckpoints, configuration: configuration)
  let nextRows = kernel - 1 + convolutionCheckpoints
  let afterAttention = CausalConvolution(
    kernelSize: kernel, stateCheckpointCount: convolutionCheckpoints,
    name: "\(prefix).post_attention_convolution")(
      attention.to(of: hidden), previousRows: previousRows[2])
  let partial: Model.IO
  if configuration.isBlockBoundary(layerIndex) {
    let block = hidden.reshaped([1, T, h])
    if let previous = blocks {
      let concat = Concat(axis: 0)
      concat.flags = [.disableOpt]
      blocks = concat(previous, block)
    } else {
      blocks = block
    }
    partial = afterAttention.output
  } else {
    partial = hidden + afterAttention.output
  }
  let mlpInput = AttentionResidual(
    queries: 1, epsilon: epsilon, scale: 1 / Float(h).squareRoot(), name: "\(prefix).mlp_input")(
      blocks: blocks!, partial: partial)[0]
  mlpInput.add(dependencies: [
    afterAttention.nextRows.moved(to: inputs[8].reshaped(.NHWC(1, nextRows, 1, h)))
  ])
  let expertInput = RMSNorm(
    epsilon: epsilon, axis: [1], name: "\(prefix).post_attention_layernorm")(mlpInput)
  let moe = Inkling2SmallMoE(
    prefix: "\(prefix).mlp", x: expertInput, tokenLength: T,
    residentSlots: configuration.expertResidentSlots[layerIndex], configuration: configuration)
  let afterMLP = CausalConvolution(
    kernelSize: kernel, stateCheckpointCount: convolutionCheckpoints,
    name: "\(prefix).post_mlp_convolution")(moe.output, previousRows: previousRows[3])
  let out = partial + afterMLP.output
  out.add(dependencies: [
    afterMLP.nextRows.moved(to: inputs[9].reshaped(.NHWC(1, nextRows, 1, h)))
  ])
  return (out, (expertInput, moe.routing, moe.hidden))
}

/// Complete text model. Inputs: token IDs, image embeddings `[rows, hidden]` placed after them,
/// and the token IDs after the image (any may be empty; together they make T rows); then, when a
/// full-attention layer is among `layers`, the log scales (Inkling2SmallLogScale); then per layer
/// the `Inkling2SmallConfiguration.layerInputCount` inputs. `cachedSlidingTokenLength` is the
/// cached rows the sliding-window caches are bound with, at least the last `slidingWindow - 1`
/// and by default exactly those. Outputs: softcapped logits over the unpadded vocabulary for the
/// last `logitTokenLength` tokens in the activation datatype, then each layer's partial block
/// `[T, hidden]` when `outputResiduals` is set, then per layer: when `outputExpertActivations` is
/// set, five as calibration reads them: the expert input `[T, hidden]`, the token of each routed
/// row `[8T]`, the rows' weighted SwiGLU output `[8T, intermediate]`, and the experts of the
/// groups of rows with their row counts; when `outputRouting` is set, three as routing traces read
/// them: the token of each routed row, and the experts of the groups with their row counts; then,
/// with `draftFeatureLayers`, the drafter's target features of every row `[T, layers * hidden]`
/// in the activation datatype: for each of those layers, in order, the partial block entering it,
/// or, for the layer count, the final normalized output. `convolutionCheckpoints` makes every
/// convolution's next rows `kernel - 1 + convolutionCheckpoints` rows, so the rows after each of the
/// chunk's last `convolutionCheckpoints + 1` tokens can be picked afterwards, as the next call's
/// `convolutionOffset`. Input order depends only on the configuration, so one graph serves every
/// chunk.
public func Inkling2SmallCausalLM<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, tokenLength: Int, cachedTokenLength: Int = 0,
  cachedSlidingTokenLength: Int? = nil, logitTokenLength: Int = 1, outputResiduals: Bool = false,
  outputExpertActivations: Bool = false, outputRouting: Bool = false,
  draftFeatureLayers: [Int] = [], convolutionCheckpoints: Int = 0, convolutionOffset: Int = 0,
  configuration: Inkling2SmallConfiguration = .inkling2Small
) -> Model {
  precondition(convolutionCheckpoints < tokenLength || convolutionCheckpoints == 0)
  precondition(logitTokenLength >= 1 && logitTokenLength <= tokenLength)
  let cachedSliding =
    cachedSlidingTokenLength
    ?? configuration.cachedKeyCount(fullAttention: false, cachedTokenLength: cachedTokenLength)
  precondition(
    cachedSliding <= cachedTokenLength
      && cachedSliding
        >= configuration.cachedKeyCount(fullAttention: false, cachedTokenLength: cachedTokenLength)
  )
  let h = configuration.hiddenSize
  let T = tokenLength
  let tokens = Input()
  let injected = Input()
  let postTokens = Input()
  var inputs = [tokens, injected, postTokens]
  var logScale: Input? = nil
  if configuration.hasFullAttentionLayer {
    logScale = Input()
    inputs.append(logScale!)
  }
  // The stored table holds the reference's embed_norm(embed(token)): the RMSNorm only depends on
  // the token, so the converter applies it. Injected image embeddings replace normalized ones.
  let embed = Embedding(
    dataType, vocabularySize: configuration.vocabularySize, embeddingSize: h,
    name: "embed_tokens")
  var hidden = Concat(axis: 0)(embed(tokens), injected, embed(postTokens)).to(.Float32)
  var blocks: Model.IO? = nil
  var residuals = [Model.IO]()
  var layerOutputs = [Model.IO]()
  var targetFeatures = [Model.IO]()
  for layerIndex in 0..<configuration.layers {
    if draftFeatureLayers.contains(layerIndex) {
      targetFeatures.append(hidden.to(FloatType.dataType))
    }
    let layerInputs = (0..<Inkling2SmallConfiguration.layerInputCount).map { _ in Input() }
    inputs += layerInputs
    let layer = Inkling2SmallLayer(
      dataType, layerIndex: layerIndex, hidden: hidden, blocks: &blocks, inputs: layerInputs,
      logScale: logScale, tokenLength: T,
      cachedKeyCount: configuration.isFullAttentionLayer(layerIndex)
        ? cachedTokenLength : cachedSliding, convolutionCheckpoints: convolutionCheckpoints,
      convolutionOffset: convolutionOffset, configuration: configuration)
    hidden = layer.output
    if outputResiduals { residuals.append(hidden) }
    // Routing's outputs are indexed only when they are outputs too.
    let experts = layer.experts
    if outputExpertActivations {
      layerOutputs += [
        experts.input, experts.routing[2], experts.hidden, experts.routing[3], experts.routing[4],
      ]
    }
    if outputRouting {
      layerOutputs += [experts.routing[2], experts.routing[3], experts.routing[4]]
    }
  }
  // Logits for the last rows of the final mixture. The final target feature normalizes every row, the logits
  // the last of them.
  let final = AttentionResidual(
    queries: 1, epsilon: configuration.normEpsilon, scale: 1 / Float(h).squareRoot(),
    name: "final_input")(blocks: blocks!, partial: hidden)[0]
  let norm = RMSNorm(epsilon: configuration.normEpsilon, axis: [1], name: "norm")
  let normalized: Model.IO
  if draftFeatureLayers.contains(configuration.layers) {
    let rows = norm(final)
    targetFeatures.append(rows.to(FloatType.dataType))
    normalized = rows.reshaped(
      [logitTokenLength, h], offset: [T - logitTokenLength, 0], strides: [h, 1])
  } else {
    normalized = norm(
      final.reshaped([logitTokenLength, h], offset: [T - logitTokenLength, 0], strides: [h, 1]))
  }
  let logits = Dense(count: configuration.unpaddedVocabularySize, noBias: true, name: "lm_head")(
    normalized.to(FloatType.dataType)
  ).tanh(softCap: configuration.logitSoftcap)
  precondition(targetFeatures.count == draftFeatureLayers.count)
  return Model(
    inputs,
    [logits] + residuals + layerOutputs
      + (targetFeatures.isEmpty ? [] : [Concat(axis: 1)(targetFeatures)]))
}
