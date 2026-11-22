import Foundation
import NNC

/// Inkling 2 Small's DSpark drafter (`mtp.safetensors`), and Inkling 2 Mini's: dense layers fill a
/// block [anchor, mask x 7] in one pass, attending both ways to themselves and to keys and values made from the
/// target's features of the committed positions, which each layer mixes from the target layers with
/// its own softmax weights and normalizes (DFlare). Layers in `localLayers` see the windowRows
/// positions before the block, the others every committed position. A recurrent (`rnn_hidden`)
/// Markov head then corrects the shared head's logits position by position.
public struct Inkling2SmallDSparkConfiguration: Codable, Sendable {
  public var layers = 5
  public var queryHeads = 48
  public var keyValueHeads = 8
  public var headDim = 128
  public var intermediateSize = 18_432
  public var localLayers = [0, 1, 3, 4]
  public var windowRows = 1_024
  public var blockSize = 8
  public var markovRank = 256
  /// The target features: the target's AttnRes partial entering these layers, then, for the text
  /// model's layer count, its final normalized output.
  public var targetLayers = [5, 11, 17, 23, 29, 35, 41, 48]
  public var ropeTheta = 1_000_000.0
  public var normEpsilon: Float = 1e-6
  /// The text model's width, which the drafter's residual stream shares, and its vocabulary and
  /// logit soft cap: the drafter embeds with the text model's `embed_tokens` and picks with its
  /// `lm_head`.
  public var hiddenSize = 4_608
  public var vocabularySize = 201_024
  public var unpaddedVocabularySize = 200_058
  public var logitSoftcap: Float = 30

  public init() {}

  public static let inkling2Small = Inkling2SmallDSparkConfiguration()

  /// Inkling 2 Mini's drafter. Its target layers count layer applications: the looped layers'
  /// three passes are applications 6 through 41, and 48, the application count, is the final
  /// normalized output.
  public static let inkling2Mini: Inkling2SmallDSparkConfiguration = {
    var draft = Inkling2SmallDSparkConfiguration()
    draft.layers = 3
    draft.queryHeads = 24
    draft.intermediateSize = 9_216
    draft.localLayers = [0, 2]
    draft.targetLayers = [5, 23, 41, 48]
    draft.hiddenSize = 2_304
    return draft
  }()

  public var proposals: Int { blockSize - 1 }
  public var keyValueWidth: Int { keyValueHeads * headDim }
  public func isLocal(_ layer: Int) -> Bool { localLayers.contains(layer) }
}

/// Rotary rows [rows, headDim] for positions from `firstPosition`, each pair (cos, sin) of one
/// complex rotation. The converter interleaves the checkpoint's half-split head layout to match.
public func Inkling2SmallDSparkRotaryEmbedding<FloatType: TensorNumeric & BinaryFloatingPoint>(
  rows: Int, firstPosition: Int, draft: Inkling2SmallDSparkConfiguration = .inkling2Small,
  of dataType: FloatType.Type = FloatType.self
) -> Tensor<FloatType> {
  let half = draft.headDim / 2
  let inverseFrequencies = (0..<half).map {
    1 / pow(draft.ropeTheta, Double($0 * 2) / Double(draft.headDim))
  }
  var values = [FloatType]()
  values.reserveCapacity(rows * draft.headDim)
  for row in 0..<rows {
    let position = Double(firstPosition + row)
    for frequency in inverseFrequencies {
      values.append(FloatType(cos(position * frequency)))
      values.append(FloatType(sin(position * frequency)))
    }
  }
  return Tensor(values, .CPU, .WC(rows, draft.headDim))
}

/// The RMS-normalized mixture, by the softmax weights in row `row` of `fuse_alpha`, of the target
/// features [rows, target layers, hidden] (FP32), in FP32.
private func Inkling2SmallDSparkMix(
  _ features: Model.IO, alpha: Model.IO, row: Int, rows: Int, normName: String, hidden: Int,
  draft: Inkling2SmallDSparkConfiguration
) -> Model.IO {
  let targetLayers = draft.targetLayers.count
  let weights = alpha.reshaped(
    [1, targetLayers, 1], offset: [row, 0, 0], strides: [targetLayers, 1, 1])
  return RMSNorm(epsilon: draft.normEpsilon, axis: [1], name: normName)(
    (features .* weights).reduced(.sum, axis: [1]).reshaped([rows, hidden]))
}

/// Writes the drafter's keys and values of `rows` committed positions, as a prefill chunk gives
/// them, into each layer's caches: at `globalRow` for the layers that see every position, at
/// `localRow` for the window layers. A draft writes the rows the last verify evaluated the same way.
/// Inputs: the target features [rows, target layers * hidden] (FP16), their rotary [rows, head
/// dim], then each layer's key cache and value cache [rows bound, keyValueHeads * head dim].
/// Outputs: the first value of each cache [1, 2 * layers] (FP32), read only so the writes land.
public func Inkling2SmallDSparkContext<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, rows: Int, globalRow: Int, localRow: Int,
  draft: Inkling2SmallDSparkConfiguration = .inkling2Small
) -> Model {
  precondition(rows > 0 && globalRow >= 0 && localRow >= 0)
  let hidden = draft.hiddenSize
  let width = draft.keyValueWidth
  let targetLayers = draft.targetLayers.count
  let targetFeatures = Input()
  let rotary = Input()
  let keyCaches = (0..<draft.layers).map { _ in Input() }
  let valueCaches = (0..<draft.layers).map { _ in Input() }
  let alpha = Parameter<Float>(
    .GPU(0), .WC(draft.layers + 1, targetLayers), name: "dspark.fuse_alpha")
  let features = targetFeatures.to(.Float32).reshaped([rows, targetLayers, hidden])
  var reads = [Model.IO]()
  for layer in 0..<draft.layers {
    let prefix = "dspark.layers.\(layer)"
    let context = Inkling2SmallDSparkMix(
      features, alpha: alpha.io, row: layer, rows: rows, normName: "\(prefix).dflare_norm",
      hidden: hidden, draft: draft
    ).to(dataType.dataType)
    let keys = Functional.cmul(
      left: RMSNorm(epsilon: draft.normEpsilon, axis: [3], name: "\(prefix).self_attn.k_norm_ctx")(
        Dense(count: width, noBias: true, name: "\(prefix).self_attn.k_proj_ctx")(context)
          .reshaped([1, rows, draft.keyValueHeads, draft.headDim])),
      right: rotary.reshaped([1, rows, 1, draft.headDim]))
    let values = Dense(count: width, noBias: true, name: "\(prefix).self_attn.v_proj_ctx")(context)
    let row = draft.isLocal(layer) ? localRow : globalRow
    // Without disableOpt a move folds into its producer, which a graph that reads nothing back
    // then drops with the write.
    let writes = [
      keys.reshaped([rows, width]).moved(
        to: keyCaches[layer].reshaped([rows, width], offset: [row, 0], strides: [width, 1]),
        flags: [.disableOpt]),
      values.moved(
        to: valueCaches[layer].reshaped([rows, width], offset: [row, 0], strides: [width, 1]),
        flags: [.disableOpt]),
    ]
    // A cache the graph only writes, through views at offsets, is not bound to the input it is
    // given, and the writes are lost. Reading one value of each after the writes binds it.
    for cache in [keyCaches[layer], valueCaches[layer]] {
      let read = cache.reshaped([1, 1], offset: [0, 0], strides: [width, 1]).to(.Float32)
      read.add(dependencies: writes)
      reads.append(read)
    }
  }
  // Dependencies of a model's output are not kept, so the reads go out through a concatenation.
  return Model(
    [targetFeatures, rotary] + zip(keyCaches, valueCaches).flatMap { [$0, $1] },
    [Concat(axis: 1)(reads)])
}

/// One draft. Each layer first writes the keys and values of the main rows, the rows the last
/// verify evaluated, then the block's after them, and attends both ways over a view of its caches.
/// A layer that sees every position reads rows 0 through the block, as its cache row is the
/// position; a window layer reads windowRows + mainRows + blockSize rows from `localOffset`: the
/// window before the first main row, the main rows, then the block. The mask closes the main rows
/// the verify did not keep, which the block's positions take, and, for a window layer, the
/// positions before the windowRows ones before the block. As the main rows do not depend on what the
/// verify keeps, the graph builds while the verify runs. The caches are bound whole, as a bound
/// tensor can't start past its buffer's base, so the views' starts are part of the graph.
/// The residual stream runs in FP32, the projections, attention and caches in `FloatType`. The
/// head is the text model's `lm_head`, and the embedding its `embed_tokens` (normalized rows).
/// Inputs: the main rows' target features [mainRows, target layers * hidden] (FP16), the last kept
/// row's [1, target layers * hidden], the main rows' rotary [mainRows, head dim], the anchor [1]
/// (Int32), the block's rotary [blockSize, head dim], the window layers' additive mask [1, 1,
/// blockSize, windowRows + mainRows + blockSize] (a column per view row), zeros [1, 1, blockSize,
/// globalRow] that open the other layers' view up to the main rows, the min-p window [1, 1] (FP32:
/// logits further than this below a position's best are never picked), then each layer's key cache
/// and value cache. A temperature of 0 drafts greedily; a positive one samples
/// softmax(logits / temperature) within the window.
/// Outputs: the drafted tokens [proposals] (Int32, each picked after the Markov head's correction
/// from the token before), the logits they were drawn from [proposals, vocabulary] (FP32), and each
/// row's sum of exp(logit - best logit) [proposals, 1] (FP32), whose inverse is the probability
/// of its best token.
public func Inkling2SmallDSpark<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, mainRows: Int, globalRow: Int, localOffset: Int,
  temperature: Float, draft: Inkling2SmallDSparkConfiguration = .inkling2Small
) -> Model {
  precondition(mainRows > 0 && globalRow >= 0 && localOffset >= 0)
  let hidden = draft.hiddenSize
  let dim = draft.headDim
  let heads = draft.queryHeads
  let kvHeads = draft.keyValueHeads
  let width = draft.keyValueWidth
  let block = draft.blockSize
  let epsilon = draft.normEpsilon
  let vocabulary = draft.unpaddedVocabularySize
  let localView = draft.windowRows + mainRows + block
  let globalView = globalRow + mainRows + block
  let targetLayers = draft.targetLayers.count
  let targetFeatures = Input()
  let lastFeatures = Input()
  let mainRotary = Input()
  let anchor = Input()
  let blockRotary = Input()
  let mask = Input()
  let openPrefix = Input()
  let minPWindow = Input()
  let keyCaches = (0..<draft.layers).map { _ in Input() }
  let valueCaches = (0..<draft.layers).map { _ in Input() }
  let alpha = Parameter<Float>(
    .GPU(0), .WC(draft.layers + 1, targetLayers), name: "dspark.fuse_alpha")
  let features = targetFeatures.to(.Float32).reshaped([mainRows, targetLayers, hidden])
  // The other layers' mask: open through the rows before the main rows, then as the window
  // layers' over the main rows and the block. Their view grows with the context, so it takes the
  // zeros from a buffer kept across drafts rather than a mask built and uploaded each draft, and the
  // attention takes a contiguous mask, which the concatenation writes. In two dimensions, where a
  // strided copy has a fast kernel.
  let globalMask = Concat(axis: 1)(
    openPrefix.reshaped([block, globalRow]),
    mask.reshaped(
      [block, mainRows + block], offset: [0, draft.windowRows], strides: [localView, 1])
  ).reshaped([1, 1, block, globalView])
  // The anchor's normalized embedding, then the mask rows: `mask_embed` is not a row of the
  // embedding the drafter shares with the text model, so the block can't embed a mask token.
  let maskEmbedding = Parameter<Float>(.GPU(0), .WC(1, hidden), name: "dspark.mask_embed")
  var out = Concat(axis: 0)(
    [
      Embedding(
        dataType, vocabularySize: draft.vocabularySize, embeddingSize: hidden,
        name: "embed_tokens")(
          anchor
        ).to(.Float32)
    ] + Array(repeating: maskEmbedding.io, count: block - 1))
  for layer in 0..<draft.layers {
    let prefix = "dspark.layers.\(layer)"
    let x = RMSNorm(epsilon: epsilon, axis: [1], name: "\(prefix).input_layernorm")(out).to(
      dataType.dataType)
    let keys = Functional.cmul(
      left: RMSNorm(epsilon: epsilon, axis: [3], name: "\(prefix).self_attn.k_norm")(
        Dense(count: width, noBias: true, name: "\(prefix).self_attn.k_proj")(x)
          .reshaped([1, block, kvHeads, dim])),
      right: blockRotary.reshaped([1, block, 1, dim]))
    let values = Dense(count: width, noBias: true, name: "\(prefix).self_attn.v_proj")(x)
    // The main rows' keys and values, from their target features as Inkling2SmallDSparkContext
    // makes them, then the block's, all write into the caches in place, and attention reads them
    // after.
    let context = Inkling2SmallDSparkMix(
      features, alpha: alpha.io, row: layer, rows: mainRows, normName: "\(prefix).dflare_norm",
      hidden: hidden, draft: draft
    ).to(dataType.dataType)
    let mainKeys = Functional.cmul(
      left: RMSNorm(epsilon: epsilon, axis: [3], name: "\(prefix).self_attn.k_norm_ctx")(
        Dense(count: width, noBias: true, name: "\(prefix).self_attn.k_proj_ctx")(context)
          .reshaped([1, mainRows, kvHeads, dim])),
      right: mainRotary.reshaped([1, mainRows, 1, dim]))
    let mainValues = Dense(count: width, noBias: true, name: "\(prefix).self_attn.v_proj_ctx")(
      context)
    let isLocal = draft.isLocal(layer)
    let mainRow = isLocal ? localOffset + draft.windowRows : globalRow
    let keyCache = keyCaches[layer]
    let valueCache = valueCaches[layer]
    let writes = [
      mainKeys.reshaped([mainRows, width]).moved(
        to: keyCache.reshaped([mainRows, width], offset: [mainRow, 0], strides: [width, 1])),
      mainValues.moved(
        to: valueCache.reshaped([mainRows, width], offset: [mainRow, 0], strides: [width, 1])),
      keys.reshaped([block, width]).moved(
        to: keyCache.reshaped(
          [block, width], offset: [mainRow + mainRows, 0], strides: [width, 1])),
      values.moved(
        to: valueCache.reshaped(
          [block, width], offset: [mainRow + mainRows, 0], strides: [width, 1])),
    ]
    let query = Functional.cmul(
      left: RMSNorm(epsilon: epsilon, axis: [3], name: "\(prefix).self_attn.q_norm")(
        Dense(count: heads * dim, noBias: true, name: "\(prefix).self_attn.q_proj")(x)
          .reshaped([1, block, heads, dim])),
      right: blockRotary.reshaped([1, block, 1, dim]))
    let viewRows = isLocal ? localView : globalView
    let viewShape: TensorShape = [1, viewRows, kvHeads, dim]
    let viewStrides: TensorShape = [viewRows * width, width, dim, 1]
    let viewOffset: TensorShape = [0, isLocal ? localOffset : 0, 0, 0]
    let viewKeys = keyCache.reshaped(viewShape, offset: viewOffset, strides: viewStrides)
    let viewValues = valueCache.reshaped(viewShape, offset: viewOffset, strides: viewStrides)
    let attended = ScaledDotProductAttention(
      scale: 1 / Float(dim).squareRoot(), hasAttentionMask: true, flags: [.Float16])(
        query, viewKeys, viewValues, isLocal ? mask : globalMask)
    attended.add(dependencies: writes)
    out =
      out
      + Dense(count: hidden, noBias: true, name: "\(prefix).self_attn.o_proj")(
        attended.reshaped([block, heads * dim])
      ).to(.Float32)
    let y = RMSNorm(epsilon: epsilon, axis: [1], name: "\(prefix).post_attention_layernorm")(out)
      .to(dataType.dataType)
    let gate = Dense(count: draft.intermediateSize, noBias: true, name: "\(prefix).mlp.gate_proj")(
      y)
    let up = Dense(count: draft.intermediateSize, noBias: true, name: "\(prefix).mlp.up_proj")(y)
    out =
      out
      + Dense(count: hidden, noBias: true, name: "\(prefix).mlp.down_proj")(
        Functional.swishMul(value: up, gate: gate)
      ).to(.Float32)
  }
  let proposals = draft.proposals
  let rank = draft.markovRank
  // The anchor's row is not a prediction.
  let predicted = RMSNorm(epsilon: epsilon, axis: [1], name: "dspark.norm")(
    out.reshaped([proposals, hidden], offset: [1, 0], strides: [hidden, 1]))
  let baseLogits = Dense(count: vocabulary, noBias: true, name: "lm_head")(
    predicted.to(dataType.dataType)
  ).to(.Float32)
  // The recurrent head's input from the predicted rows and the last kept row's target features,
  // with the bias, for every position at once; each step adds its state and the token before.
  let feature = Inkling2SmallDSparkMix(
    lastFeatures.to(.Float32).reshaped([1, targetLayers, hidden]), alpha: alpha.io,
    row: draft.layers, rows: 1, normName: "dspark.markov_head.dflare_norm", hidden: hidden,
    draft: draft)
  let context =
    Dense(count: 3 * rank, noBias: true, name: "dspark.markov_head.rnn_hidden")(predicted)
    + Dense(count: 3 * rank, name: "dspark.markov_head.rnn_feature")(feature)
  let markovEmbedding = Embedding(
    dataType, vocabularySize: draft.vocabularySize, embeddingSize: rank,
    name: "dspark.markov_head.markov_w1")
  let previousProjection = Dense(
    count: 3 * rank, noBias: true, name: "dspark.markov_head.rnn_previous")
  let stateProjection = Dense(count: 3 * rank, noBias: true, name: "dspark.markov_head.rnn_state")
  // hidden_out folded through the head, [vocabulary, rank] and its bias.
  let fold = Dense(count: vocabulary, name: "dspark.markov_head.fold")
  var previous: Model.IO = anchor.reshaped([1])
  var state: Model.IO? = nil
  var picks = [Model.IO]()
  var biasedRows = [Model.IO]()
  var partitions = [Model.IO]()
  for position in 0..<proposals {
    // The projections come first: a sum is computed in place over its first operand read for the
    // last time, and over the row of `context` it would be a view, whose slices below compiling the
    // graph asserts against.
    var raw = previousProjection(markovEmbedding(previous).to(.Float32))
    if let state {
      raw = raw + stateProjection(state)
    }
    raw = raw + context.reshaped([1, 3 * rank], offset: [position, 0], strides: [3 * rank, 1])
    let latent = raw.reshaped([1, rank], offset: [0, 2 * rank], strides: [3 * rank, 1]).tanh()
    // gate * state + (1 - gate) * candidate, from a zero state at the first position. The last
    // position's state is never read, and an unread node breaks the build.
    if position + 1 < proposals {
      let gate = raw.reshaped([1, rank], offset: [0, 0], strides: [3 * rank, 1]).sigmoid()
      let candidate = raw.reshaped([1, rank], offset: [0, rank], strides: [3 * rank, 1]).tanh()
      if let current = state {
        state = candidate + gate .* (current - candidate)
      } else {
        state = candidate - gate .* candidate
      }
    }
    let biased =
      (baseLogits.reshaped([1, vocabulary], offset: [position, 0], strides: [vocabulary, 1])
      + fold(latent.to(dataType.dataType)).to(.Float32)).tanh(softCap: draft.logitSoftcap)
    biasedRows.append(biased)
    let centered = biased - biased.reduced(.max, axis: [1])
    // 1 / sum is the top token's probability, the draft's confidence.
    partitions.append(centered.exp().reduced(.sum, axis: [1]))
    // Gumbel-max over the min-p window samples softmax(biased / temperature) there: a logit below
    // the window takes a penalty no noise overcomes.
    let window = centered + minPWindow
    previous = (biased - (window * -1e6).ReLU()).gumbelArgmax(axis: 1, scale: temperature)
      .reshaped([1])
    picks.append(previous)
  }
  return Model(
    [targetFeatures, lastFeatures, mainRotary, anchor, blockRotary, mask, openPrefix, minPWindow]
      + zip(keyCaches, valueCaches).flatMap { [$0, $1] },
    [Concat(axis: 0)(picks), Concat(axis: 0)(biasedRows), Concat(axis: 0)(partitions)])
}
