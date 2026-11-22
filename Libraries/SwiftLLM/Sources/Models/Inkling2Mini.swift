import Foundation
import NNC

#if canImport(SQLite3)
  import SQLite3
#else
  import _SQLite3Shims
#endif

/// Inkling 2 Mini text decoder: Inkling 2 Small's layer (grouped-query attention with a learned
/// relative-position bias, per-token sinks and L2-clipped queries and keys, short causal
/// convolutions around every sublayer, sigmoid-routed MoE with softcapped SwiGLU experts, block
/// attention residuals, five sliding-window layers to each full-attention one) at half the width,
/// as a looped transformer: of its 24 layers, layers 6 to 17 run three times over the same
/// weights. Every pass restarts from the states before the loop. A pass after the first gates a
/// message into each sublayer input: a softmax mixture, by learned queries, of the block states
/// the pass before left.
public struct Inkling2MiniConfiguration: Codable, Sendable {
  public var vocabularySize = 201_024
  /// Rows of the unembedding that hold real tokens; logits beyond are dropped.
  public var unpaddedVocabularySize = 200_058
  public var hiddenSize = 2_304
  public var layers = 24
  /// The looped layers, `loopStart..<loopEnd`, run `loopPasses` times. Both bounds start a block.
  public var loopStart = 6
  public var loopEnd = 18
  public var loopPasses = 3
  /// Width of the mix of a sublayer input and its message that the message gates read.
  public var messageGateRank = 64
  public var attentionHeads = 24
  public var keyValueHeads = 8
  public var headDim = 128
  /// Per-head width of the relative states mixed into the relative-position bias.
  public var relativeDim = 16
  /// Backward distances the relative-position bias covers.
  public var relativeExtent = 1_024
  public var slidingWindow = 1_024
  public var logScalingFloor = 8_192
  public var logScalingAlpha: Float = 0.1
  public var normEpsilon: Float = 1e-6
  public var queryKeyClip: Float = 22.627416997969522
  public var relativeClip: Float = 8
  public var convolutionKernel = 4
  /// Layers summed into one block before it becomes a source of the attention residuals.
  public var layersPerBlock = 6
  public var expertCount = 128
  public var routedExperts = 8
  public var expertIntermediateSize = 1_152
  /// Rows of an expert's stored gate and up projections, and columns of its down projection: the
  /// converter pads the checkpoint's `expertIntermediateSize` with zeros to a multiple of 256,
  /// which ccv's kernels for GGUF-packed weights need of both dimensions of a weight.
  public var expertStoredIntermediateSize = 1_280
  public var gateSoftcap: Float = 4
  public var upSoftcap: Float = 25
  public var routedScalingFactor: Float = 8
  public var logitSoftcap: Float = 30
  /// Decode expert pool slots of each layer. A layer with fewer than `expertCount` streams its
  /// routed experts through MoEWeightStreaming; a looped layer's passes share its pool.
  public var expertResidentSlots = Array(repeating: 128, count: 24)

  public init() {}

  public static let inkling2Mini = Inkling2MiniConfiguration()

  /// Model inputs of each layer application, in order: the key and value caches bound from their
  /// first row through this chunk's rows, the four convolution rows (key and value in the
  /// activation datatype, post-attention and post-MLP in FP32), each with at least
  /// `convolutionOffset` rows before its kernel - 1, and the four buffers their next values go to,
  /// which must not overlap them.
  public static let layerInputCount = 10

  public var queryDim: Int { attentionHeads * headDim }
  public var keyValueDim: Int { keyValueHeads * headDim }

  /// Times a layer runs for each token, 48: a looped layer once a pass. Every application has its
  /// own attention caches and convolution rows.
  public var layerApplications: Int { layers + (loopPasses - 1) * (loopEnd - loopStart) }

  /// The layer the `application`-th one runs, and, for a looped layer, which pass it is in.
  public func layer(application: Int) -> (index: Int, pass: Int) {
    precondition(application >= 0 && application < layerApplications)
    let loopLayers = loopEnd - loopStart
    guard application >= loopStart else { return (application, 0) }
    guard application < loopStart + loopPasses * loopLayers else {
      return (application - (loopPasses - 1) * loopLayers, 0)
    }
    return (
      loopStart + (application - loopStart) % loopLayers, (application - loopStart) / loopLayers
    )
  }

  /// Five sliding-window layers precede each full-attention layer.
  public func isFullAttentionLayer(_ layerIndex: Int) -> Bool {
    precondition(layerIndex >= 0 && layerIndex < layers)
    return (layerIndex + 1) % layersPerBlock == 0
  }

  /// Whether the `application`-th layer application runs a full-attention layer.
  public func isFullAttention(application: Int) -> Bool {
    isFullAttentionLayer(layer(application: application).index)
  }

  /// A block starts at this layer: its input joins the block states, and its sublayers restart
  /// the partial block.
  public func isBlockBoundary(_ layerIndex: Int) -> Bool {
    precondition(layerIndex >= 0 && layerIndex < layers)
    return layerIndex % layersPerBlock == 0
  }

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
public func Inkling2MiniResidentMappingSize(
  checkpoint filePath: String, configuration: Inkling2MiniConfiguration
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
public func Inkling2MiniLogScale<FloatType: TensorNumeric & BinaryFloatingPoint>(
  _ dataType: FloatType.Type, cachedTokenLength: Int, tokenLength: Int,
  configuration: Inkling2MiniConfiguration = .inkling2Mini
) -> Tensor<FloatType> {
  Tensor<FloatType>(
    (0..<tokenLength).map { FloatType(configuration.logScale(position: cachedTokenLength + $0)) },
    kind: .CPU, format: .NHWC, shape: [tokenLength, 1])
}

// As in the reference, residuals, block states, attention residuals, messages and their gates, the
// router and experts are FP32. Attention runs in the activation datatype; its norms and
// convolutions accumulate in FP32. Every short convolution holds the reference's residual and
// scale in its stored weights, and writes its next rows into their buffer: the move folds into the
// convolution.

/// One layer's models, which own its weights: a looped layer applies them once a pass, each time
/// to that pass's caches and convolution rows.
private struct Inkling2MiniLayer {
  let configuration: Inkling2MiniConfiguration
  let layerIndex: Int
  let convolutionCheckpoints: Int
  let attentionInput: AttentionResidual?
  let inputNorms: [Model]
  let queryProjection: Model
  let relativeProjection: Model
  let keyProjection: Model
  let valueProjection: Model
  let keyConvolution: Model
  let valueConvolution: Model
  let keyDynamicWeights: Model?
  let valueDynamicWeights: Model?
  let relativeBias: Model
  let sinkProjection: Model
  let gateProjection: Model
  let outputProjection: Model
  let postAttentionConvolution: CausalConvolution
  let mlpInput: AttentionResidual
  let postAttentionNorm: Model
  let router: Model
  let routerBias: Parameter<Float>
  /// SegmentedSwiGLU and SegmentedDense owning the routed experts' weights or, in a streaming
  /// layer, their functional forms, which read the weights of `streamingBanks`.
  let experts: Model
  let expertsDown: Model
  /// In a streaming layer, the stored banks MoEWeightStreaming loads its decode pool from: loaders
  /// map `experts.streaming_gate` and `experts.streaming_up` to the stored gate and up banks,
  /// `experts-0-0` and `experts-0-1`, and read the banks as `.externalOnDemand`.
  let streamingBanks: (gate: Parameter<Float>, up: Parameter<Float>, down: Parameter<Float>)?
  let postMLPConvolution: CausalConvolution
  /// In looped layers: the four queries of the query, key, value and MLP messages, and the gate of
  /// each, which owns the down, forget and gain projections.
  let messageInput: AttentionResidual?
  let messageGates: [MessageGate]

  init(layerIndex: Int, convolutionCheckpoints: Int, configuration: Inkling2MiniConfiguration) {
    self.configuration = configuration
    self.layerIndex = layerIndex
    self.convolutionCheckpoints = convolutionCheckpoints
    let prefix = "layers.\(layerIndex)"
    let attention = "\(prefix).self_attn"
    let h = configuration.hiddenSize
    let epsilon = configuration.normEpsilon
    let kernel = configuration.convolutionKernel
    let isFull = configuration.isFullAttentionLayer(layerIndex)
    // Layer 0 reads the embedding directly; later layers mix the block states. Full-attention
    // layers mix and normalize their query, key and value inputs separately.
    attentionInput =
      layerIndex == 0
      ? nil
      : AttentionResidual(
        queries: isFull ? 3 : 1, epsilon: epsilon, scale: 1 / Float(h).squareRoot(),
        name: "\(prefix).attention_input")
    inputNorms = (isFull ? ["q_", "k_", "v_"] : [""]).map {
      RMSNorm(epsilon: epsilon, axis: [1], name: "\(prefix).\($0)input_layernorm")
    }
    queryProjection = Dense(
      count: configuration.queryDim, noBias: true, name: "\(attention).q_proj")
    relativeProjection = Dense(
      count: configuration.attentionHeads * configuration.relativeDim, noBias: true,
      name: "\(attention).r_proj")
    keyProjection = Dense(
      count: configuration.keyValueDim, noBias: true, name: "\(attention).k_proj")
    valueProjection = Dense(
      count: configuration.keyValueDim, noBias: true, name: "\(attention).v_proj")
    // Full-attention layers add each token's dynamic weights to the key and value convolutions, one
    // set to a key/value head.
    keyConvolution = CausalConvolution(
      kernelSize: kernel, stateCheckpointCount: convolutionCheckpoints, hasDynamicWeights: isFull,
      name: "\(attention).k_sconv")
    valueConvolution = CausalConvolution(
      kernelSize: kernel, stateCheckpointCount: convolutionCheckpoints, hasDynamicWeights: isFull,
      name: "\(attention).v_sconv")
    keyDynamicWeights =
      isFull
      ? Dense(
        count: configuration.keyValueHeads * kernel, noBias: true,
        name: "\(attention).k_sconv.weight_proj") : nil
    valueDynamicWeights =
      isFull
      ? Dense(
        count: configuration.keyValueHeads * kernel, noBias: true,
        name: "\(attention).v_sconv.weight_proj") : nil
    // Row d of the stored projection is the bias profile of the key d positions back.
    relativeBias = Dense(
      count: configuration.relativeExtent, noBias: true, name: "\(attention).rel_logits_proj")
    sinkProjection = Dense(
      count: configuration.attentionHeads, noBias: true, name: "\(attention).sink_proj")
    gateProjection = Dense(
      count: configuration.queryDim, noBias: true, name: "\(attention).gate_proj")
    outputProjection = Dense(count: h, noBias: true, name: "\(attention).o_proj")
    postAttentionConvolution = CausalConvolution(
      kernelSize: kernel, stateCheckpointCount: convolutionCheckpoints,
      name: "\(prefix).post_attention_convolution")
    mlpInput = AttentionResidual(
      queries: 1, epsilon: epsilon, scale: 1 / Float(h).squareRoot(), name: "\(prefix).mlp_input")
    postAttentionNorm = RMSNorm(
      epsilon: epsilon, axis: [1], name: "\(prefix).post_attention_layernorm")
    router = Dense(count: configuration.expertCount, noBias: true, name: "\(prefix).mlp.gate")
    routerBias = Parameter<Float>(
      .GPU(0), .C(configuration.expertCount), name: "\(prefix).mlp.gate.e_score_correction_bias")
    let intermediate = configuration.expertStoredIntermediateSize
    if configuration.isStreamingLayer(layerIndex) {
      let count = configuration.expertCount
      streamingBanks = (
        Parameter<Float>(
          .GPU(0), .HWC(count, intermediate, h), name: "\(prefix).mlp.experts.streaming_gate"),
        Parameter<Float>(
          .GPU(0), .HWC(count, intermediate, h), name: "\(prefix).mlp.experts.streaming_up"),
        Parameter<Float>(
          .GPU(0), .HWC(count, h, intermediate), name: "\(prefix).mlp.experts.down_proj")
      )
      experts = SegmentedSwiGLU(
        segments: 0, count: intermediate, gateSoftCap: configuration.gateSoftcap,
        upSoftCap: configuration.upSoftcap, functional: true,
        name: "\(prefix).mlp.experts.functional")
      expertsDown = SegmentedDense(
        segments: 0, count: h, noBias: true, functional: true,
        name: "\(prefix).mlp.experts.functional.down_proj")
    } else {
      streamingBanks = nil
      // The gate and up weights are `experts-0-0` and `experts-0-1`.
      experts = SegmentedSwiGLU(
        segments: configuration.expertCount, count: intermediate,
        gateSoftCap: configuration.gateSoftcap, upSoftCap: configuration.upSoftcap,
        name: "\(prefix).mlp.experts")
      expertsDown = SegmentedDense(
        segments: configuration.expertCount, count: h, noBias: true,
        name: "\(prefix).mlp.experts.down_proj")
    }
    postMLPConvolution = CausalConvolution(
      kernelSize: kernel, stateCheckpointCount: convolutionCheckpoints,
      name: "\(prefix).post_mlp_convolution")
    if layerIndex >= configuration.loopStart && layerIndex < configuration.loopEnd {
      messageInput = AttentionResidual(
        queries: 4, epsilon: epsilon, scale: 1 / Float(h).squareRoot(),
        name: "\(prefix).message_input")
      messageGates = ["q", "k", "v", "mlp"].map {
        MessageGate(
          rank: configuration.messageGateRank, epsilon: epsilon,
          name: "\(prefix).\($0)_message_gate")
      }
    } else {
      messageInput = nil
      messageGates = []
    }
  }

  /// One application of the layer. `hidden` is the FP32 partial block `[T, hidden]`; `blocks` the
  /// block states `[n, T, hidden]` its attention input mixes (none at layer 0) and `nextBlocks`
  /// those its MLP input mixes, which a block boundary extends with `hidden`. Each sublayer reads
  /// an attention residual: a softmax mixture, by a learned query, of the normalized block states
  /// and partial block. `messages` are the query, key, value and MLP messages of a pass after the
  /// first, gated into the attention residuals. `inputs` are the application's
  /// `Inkling2MiniConfiguration.layerInputCount` inputs and, in full-attention layers, `logScale`
  /// the log scales `[T, 1]`. Returns the next partial block, and the expert input `[T, hidden]`
  /// with the MoE's routing, whose outputs group the routed rows by expert (MoERouting: the rows,
  /// their weights, their tokens `[8T]`, and the experts of the groups with their row counts), and
  /// the weighted SwiGLU output of the rows, the input of the down projection
  /// `[8T, stored intermediate]`.
  func callAsFunction<FloatType: TensorNumeric>(
    _ dataType: FloatType.Type, hidden: Model.IO, blocks: Model.IO?, nextBlocks: Model.IO,
    messages: [Model.IO]?, inputs: [Model.IO], logScale: Model.IO?, tokenLength: Int,
    cachedKeyCount: Int, convolutionOffset: Int
  ) -> (output: Model.IO, experts: (input: Model.IO, routing: Model.IO, hidden: Model.IO)) {
    precondition(inputs.count == Inkling2MiniConfiguration.layerInputCount)
    let T = tokenLength
    let h = configuration.hiddenSize
    let heads = configuration.attentionHeads
    let kvHeads = configuration.keyValueHeads
    let headDim = configuration.headDim
    let kvDim = configuration.keyValueDim
    let relativeDim = configuration.relativeDim
    let epsilon = configuration.normEpsilon
    let kernel = configuration.convolutionKernel
    let isFull = configuration.isFullAttentionLayer(layerIndex)
    precondition(isFull == (logScale != nil))
    // The convolution rows sit `convolutionOffset` rows into their inputs: the rows after a
    // verify's last kept token, which it wrote with those after its other tokens. The batch of one
    // takes the contiguous stride, so the view stays contiguous and the convolutions keep their
    // fast kernels.
    let previousRows = zip(inputs[2..<6], [kvDim, kvDim, h, h]).map { input, dim in
      input.reshaped(
        .NHWC(1, kernel - 1, 1, dim), offset: [0, convolutionOffset, 0, 0],
        strides: [(kernel - 1) * dim, dim, dim, 1])
    }
    var sources =
      attentionInput.map { $0(blocks: blocks!, partial: hidden) }
      ?? Array(repeating: hidden, count: isFull ? 3 : 1)
    if let messages {
      // The query, key and value inputs each take their own message, so a sliding-window layer's
      // one mixture becomes three inputs too.
      sources = (0..<3).map {
        messageGates[$0](state: sources[isFull ? $0 : 0], message: messages[$0])
      }
    }
    let normalized = sources.enumerated().map {
      inputNorms[isFull ? $0.offset : 0]($0.element).to(FloatType.dataType)
    }
    let queryInput = normalized[0]
    let keyInput = normalized[normalized.count == 3 ? 1 : 0]
    let valueInput = normalized[normalized.count == 3 ? 2 : 0]

    // Each query adds its own relative-position bias, mixed from its relative states, to the last
    // `relativeExtent` keys, and its own sink. The learned scales before the reference's L2 clips
    // are in the stored query and relative projections. An L2 clip of n values at a threshold is
    // an RMSNorm of scale threshold / sqrt(n) that never scales up.
    let keyCount = cachedKeyCount + T
    var query = RMSNorm(
      epsilon: 1e-12, axis: [2], elementwiseAffine: false,
      scale: configuration.queryKeyClip / Float(headDim).squareRoot(), clip: true)(
        queryProjection(queryInput).reshaped([T, heads, headDim]))
    var relative = RMSNorm(
      epsilon: 1e-12, axis: [2], elementwiseAffine: false,
      scale: configuration.relativeClip / Float(relativeDim).squareRoot(), clip: true)(
        relativeProjection(queryInput).reshaped([T, heads, relativeDim]))
    if let logScale {
      let scale = logScale.reshaped([T, 1, 1])
      query = query .* scale
      relative = relative .* scale
    }
    let nextRows = kernel - 1 + convolutionCheckpoints
    let keyRows = keyConvolution(
      [keyProjection(keyInput), previousRows[0]] + (keyDynamicWeights.map { [$0(keyInput)] } ?? []))
    let key = RMSNorm(
      epsilon: 1e-12, axis: [2], elementwiseAffine: false,
      scale: configuration.queryKeyClip / Float(headDim).squareRoot(), clip: true)(
        keyRows[0].reshaped([T, kvHeads, headDim]))
    key.add(dependencies: [
      keyRows[1].moved(to: inputs[6].reshaped(.NHWC(1, nextRows, 1, kvDim)))
    ])
    let valueRows = valueConvolution(
      [valueProjection(valueInput), previousRows[1]]
        + (valueDynamicWeights.map { [$0(valueInput)] } ?? []))
    let value = RMSNorm(epsilon: epsilon, axis: [2], elementwiseAffine: false)(
      valueRows[0].reshaped([T, kvHeads, headDim]))
    value.add(dependencies: [
      valueRows[1].moved(to: inputs[7].reshaped(.NHWC(1, nextRows, 1, kvDim)))
    ])
    let attended = ScaledDotProductAttention(
      scale: 1 / Float(headDim).squareRoot(), isCausal: true, hasAttentionSinks: true,
      hasRelativeBias: true, slidingWindow: isFull ? 0 : configuration.slidingWindow)(
        query.reshaped(.NHWC(1, T, heads, headDim)), inputs[0], inputs[1],
        sinkProjection(queryInput).reshaped([1, T, heads]),
        relativeBias(relative.reshaped([T * heads, relativeDim])).reshaped(
          .NHWC(1, T, heads, configuration.relativeExtent)))
    // This chunk's rows are the last T of the bound caches.
    attended.add(dependencies: [
      key.reshaped(.NHWC(1, T, kvHeads, headDim)).moved(
        to: inputs[0].reshaped(
          .NHWC(1, T, kvHeads, headDim), offset: [0, keyCount - T, 0, 0],
          strides: [keyCount * kvDim, kvDim, headDim, 1]), flags: [.disableOpt]),
      value.reshaped(.NHWC(1, T, kvHeads, headDim)).moved(
        to: inputs[1].reshaped(
          .NHWC(1, T, kvHeads, headDim), offset: [0, keyCount - T, 0, 0],
          strides: [keyCount * kvDim, kvDim, headDim, 1]), flags: [.disableOpt]),
    ])
    let attention = outputProjection(
      attended.reshaped([T, heads * headDim]) .* Sigmoid()(gateProjection(queryInput)))
    let afterAttention = postAttentionConvolution(
      attention.to(of: hidden), previousRows: previousRows[2])
    let partial =
      configuration.isBlockBoundary(layerIndex)
      ? afterAttention.output : hidden + afterAttention.output
    let mixed = mlpInput(blocks: nextBlocks, partial: partial)[0]
    mixed.add(dependencies: [
      afterAttention.nextRows.moved(to: inputs[8].reshaped(.NHWC(1, nextRows, 1, h)))
    ])
    let expertInput = postAttentionNorm(
      messages.map { messageGates[3](state: mixed, message: $0[3]) } ?? mixed)
    // Sigmoid-scored MoE with softcapped SwiGLU experts, 4 tanh(g / 4) sigmoid(g) 25 tanh(u / 25),
    // evaluated in FP32.
    let routing = MoERouting(
      kth: configuration.routedExperts, weightScale: configuration.routedScalingFactor,
      normalizationEpsilon: 1e-20, sigmoid: true, name: "layers.\(layerIndex).mlp.routing")(
        router(expertInput), routerBias, expertInput)
    let rowWeights = routing[1].reshaped([T * configuration.routedExperts, 1])
    let expertHidden: Model.IO
    let expertOutput: Model.IO
    if let streamingBanks {
      // The pool belongs to the stored gate bank, so a looped layer's passes share it.
      let resident = MoEWeightStreaming(
        residentSlots: configuration.expertResidentSlots[layerIndex],
        routingWidth: configuration.routedExperts, name: "layers.\(layerIndex).mlp.expert_streaming"
      )([
        routing[3], routing[4], rowWeights, streamingBanks.gate(), streamingBanks.up(),
        streamingBanks.down(),
      ])
      expertHidden = experts([
        routing[0], resident[0], resident[1], resident[3], resident[4], resident[2],
      ])
      expertOutput = expertsDown([expertHidden, resident[0], resident[1], resident[5]])
    } else {
      expertHidden = experts(routing[0], routing[3], routing[4], rowWeights)
      expertOutput = expertsDown(expertHidden, routing[3], routing[4])
    }
    let afterMLP = postMLPConvolution(
      Functional.scatterAdd(
        count: T, countPerOutput: configuration.routedExperts, expertOutput, index: routing[2]
      ).reshaped([T, h]), previousRows: previousRows[3])
    let out = partial + afterMLP.output
    out.add(dependencies: [
      afterMLP.nextRows.moved(to: inputs[9].reshaped(.NHWC(1, nextRows, 1, h)))
    ])
    return (out, (expertInput, routing, expertHidden))
  }
}

/// Complete text model. Inputs: token IDs, image embeddings `[rows, hidden]` placed after them,
/// and the token IDs after the image (any may be empty; together they make T rows); then the log
/// scales (Inkling2MiniLogScale); then, per layer application in the order they run (layers 0-5,
/// each pass over layers 6-17, layers 18-23), the `Inkling2MiniConfiguration.layerInputCount`
/// inputs. `cachedSlidingTokenLength` is the cached rows the sliding-window caches are bound with,
/// at least the last `slidingWindow - 1` and by default exactly those. Outputs: softcapped logits
/// over the unpadded vocabulary for the last `logitTokenLength` tokens in the activation datatype,
/// then each application's partial block `[T, hidden]` when `outputResiduals` is set, then per
/// application: when `outputExpertActivations` is set, five as calibration reads them: the expert
/// input `[T, hidden]`, the token of each routed row `[8T]`, the rows' weighted SwiGLU output
/// `[8T, stored intermediate]`, and the experts of the groups of rows with their row counts; when
/// `outputRouting` is set, three as routing traces read them: the token of each routed row, and the
/// experts of the groups with their row counts; then, with `draftFeatureLayers`, the drafter's
/// target features of every row `[T, layers * hidden]` in the activation datatype: for each of
/// those applications, in order, the partial block entering it, or, for the application count, the
/// final normalized output. `convolutionCheckpoints` makes every convolution's next rows
/// `kernel - 1 + convolutionCheckpoints` rows, so the rows after each of the chunk's last
/// `convolutionCheckpoints + 1` tokens can be picked afterwards, as the next call's
/// `convolutionOffset`. Input order depends only on the configuration, so one graph serves every
/// chunk.
public func Inkling2MiniCausalLM<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, tokenLength: Int, cachedTokenLength: Int = 0,
  cachedSlidingTokenLength: Int? = nil, logitTokenLength: Int = 1, outputResiduals: Bool = false,
  outputExpertActivations: Bool = false, outputRouting: Bool = false,
  draftFeatureLayers: [Int] = [], convolutionCheckpoints: Int = 0, convolutionOffset: Int = 0,
  configuration: Inkling2MiniConfiguration = .inkling2Mini
) -> Model {
  precondition(convolutionCheckpoints < tokenLength || convolutionCheckpoints == 0)
  precondition(logitTokenLength >= 1 && logitTokenLength <= tokenLength)
  // The loop spans two blocks, so each pass leaves one block state of its own, its second
  // block's input, for the next pass's messages.
  precondition(
    configuration.loopStart > 0 && configuration.isBlockBoundary(configuration.loopStart)
      && configuration.loopEnd == configuration.loopStart + 2 * configuration.layersPerBlock
      && configuration.loopEnd <= configuration.layers)
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
  let logScale = Input()
  var inputs = [tokens, injected, postTokens, logScale]
  // The stored table holds the reference's embed_norm(embed(token)): the RMSNorm only depends on
  // the token, so the converter applies it. Injected image embeddings replace normalized ones.
  let embed = Embedding(
    dataType, vocabularySize: configuration.vocabularySize, embeddingSize: h,
    name: "embed_tokens")
  var hidden = Concat(axis: 0)(embed(tokens), injected, embed(postTokens)).to(.Float32)
  let layers = (0..<configuration.layers).map {
    Inkling2MiniLayer(
      layerIndex: $0, convolutionCheckpoints: convolutionCheckpoints, configuration: configuration)
  }
  var blocks: Model.IO? = nil
  // The states every pass over the loop starts from, and the block states after its first layer,
  // which are the same in every pass.
  var loopEntry: (hidden: Model.IO, blocks: Model.IO)? = nil
  var loopBlocks: Model.IO? = nil
  // What the pass before left for this pass's messages: its second block's input, and its final
  // partial block.
  var passBlock: Model.IO? = nil
  var previousPass: (blocks: Model.IO, partial: Model.IO)? = nil
  var residuals = [Model.IO]()
  var layerOutputs = [Model.IO]()
  var targetFeatures = [Model.IO]()
  for application in 0..<configuration.layerApplications {
    let (layerIndex, pass) = configuration.layer(application: application)
    if layerIndex == configuration.loopStart {
      if pass == 0 {
        loopEntry = (hidden, blocks!)
      } else {
        previousPass = (passBlock!, hidden)
        hidden = loopEntry!.hidden
        blocks = loopEntry!.blocks
      }
    }
    if draftFeatureLayers.contains(application) {
      targetFeatures.append(hidden.to(FloatType.dataType))
    }
    var nextBlocks = blocks
    if layerIndex == configuration.loopStart, let loopBlocks {
      nextBlocks = loopBlocks
    } else if configuration.isBlockBoundary(layerIndex) {
      let block = hidden.reshaped([1, T, h])
      if let blocks {
        let concat = Concat(axis: 0)
        concat.flags = [.disableOpt]
        nextBlocks = concat(blocks, block)
      } else {
        nextBlocks = block
      }
      if layerIndex == configuration.loopStart {
        loopBlocks = nextBlocks
      } else if layerIndex > configuration.loopStart && layerIndex < configuration.loopEnd {
        passBlock = block
      }
    }
    let layerInputs = (0..<Inkling2MiniConfiguration.layerInputCount).map { _ in Input() }
    inputs += layerInputs
    let isFull = configuration.isFullAttentionLayer(layerIndex)
    let layer = layers[layerIndex](
      dataType, hidden: hidden, blocks: blocks, nextBlocks: nextBlocks!,
      messages: pass > 0
        ? layers[layerIndex].messageInput!(
          blocks: previousPass!.blocks, partial: previousPass!.partial) : nil, inputs: layerInputs,
      logScale: isFull ? logScale : nil, tokenLength: T,
      cachedKeyCount: isFull ? cachedTokenLength : cachedSliding,
      convolutionOffset: convolutionOffset)
    hidden = layer.output
    blocks = nextBlocks
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
  // Logits for the last rows of the final mixture. The final target feature normalizes every row,
  // the logits the last of them.
  let final = AttentionResidual(
    queries: 1, epsilon: configuration.normEpsilon, scale: 1 / Float(h).squareRoot(),
    name: "final_input")(blocks: blocks!, partial: hidden)[0]
  let norm = RMSNorm(epsilon: configuration.normEpsilon, axis: [1], name: "norm")
  let normalized: Model.IO
  if draftFeatureLayers.contains(configuration.layerApplications) {
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

extension Inkling2SmallVisionConfiguration {
  /// Inkling 2 Mini's vision tower is Inkling 2 Small's, projected to Mini's width.
  public static let inkling2Mini: Inkling2SmallVisionConfiguration = {
    var configuration = Inkling2SmallVisionConfiguration()
    configuration.outputHiddenSize = 2_304
    return configuration
  }()
}
