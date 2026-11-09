import Foundation
import NNC

/// Inco AI's DFlash2 drafter for GLM-5.3-Flash: five Qwen3-style layers fill a block
/// [token, mask x 7] in one pass, the block attending both ways to itself and to the
/// windowRows - 1 positions before it. Keys and values of those positions come from the mean mHC
/// streams leaving target layers 5, 14, 24, 33 and 42, through `fc`. Two-tap dynamic convolutions
/// wrap each attention and MLP. The drafter's own head (the target's) gives each proposal its top
/// candidates, and a rank-256 selector walks them in order.
public struct GLM5_3DFlash2Configuration: Codable, Sendable {
  public var layers = 5
  public var queryHeads = 32
  public var keyValueHeads = 8
  public var headDim = 128
  public var blockSize = 8
  public var convolutionGroupSize = 16
  public var intermediateSize = 12_288
  /// The mean streams entering these layers, i.e. leaving DFlash2's target_layer_ids.
  public var targetLayers = [6, 15, 25, 34, 43]
  public var maskTokenID: Int32 = 154_856
  public var selectorRank = 256
  public var selectorCandidates = 16
  public var windowRows = 2_048
  public var ropeTheta = 10_000.0
  public var normEpsilon: Float = 1e-5

  public init() {}

  public static let glm5_3Flash = GLM5_3DFlash2Configuration()

  public var proposals: Int { blockSize - 1 }
  public var keyValueWidth: Int { keyValueHeads * headDim }
}

/// Rotary rows [rows, headDim] for logical positions from `firstPosition`, each pair (cos, sin) of
/// one complex rotation. The converter interleaves the checkpoint's half-split head layout to match.
public func GLM5_3DFlash2RotaryEmbedding<FloatType: TensorNumeric & BinaryFloatingPoint>(
  rows: Int, firstPosition: Int, draft: GLM5_3DFlash2Configuration = .glm5_3Flash,
  of dataType: FloatType.Type = FloatType.self
) -> Tensor<FloatType> {
  var rotary = Tensor<FloatType>(.CPU, .WC(rows, draft.headDim))
  for row in 0..<rows {
    let position = Double(firstPosition + row)
    for k in 0..<(draft.headDim / 2) {
      let theta = position / pow(draft.ropeTheta, Double(k * 2) / Double(draft.headDim))
      rotary[row, k * 2] = FloatType(cos(theta))
      rotary[row, k * 2 + 1] = FloatType(sin(theta))
    }
  }
  return rotary
}

/// The kernels of DFlash2's two-tap convolutions around one sub-layer, [blockSize, 4 * hidden] in
/// FP32. Stage s (0 on the sub-layer's input, 1 on its output) and tap k (0 the row itself, 1 the
/// row before it) own columns (2s + k) * hidden ..< (2s + k + 1) * hidden: a weight per channel,
/// plus one per row and group of channels that `kernel_projection` makes from the sub-layer's
/// normalized input.
private func GLM5_3DFlash2ConvolutionKernels(
  prefix: String, normalized: Model.IO, configuration: GLM5_3ModelConfiguration,
  draft: GLM5_3DFlash2Configuration
) -> Model.IO {
  let hidden = configuration.hiddenSize
  let groupSize = draft.convolutionGroupSize
  let groups = hidden / groupSize
  let base = Parameter<BFloat16>(
    .GPU(0), .HWC(2, 2, hidden), initBound: 0, name: "\(prefix).base_kernel")
  let rows = Dense(count: 4 * groups, noBias: true, name: "\(prefix).kernel_projection")(
    normalized)
  return
    (base.io.to(.Float32).reshaped([1, 4, groups, groupSize])
    + rows.to(.Float32).reshaped([draft.blockSize, 4, groups, 1])).reshaped([
      draft.blockSize, 4 * hidden,
    ])
}

/// One stage of a two-tap convolution along the block: row t takes itself and row t - 1 (zero
/// before row 0) through the kernels of GLM5_3DFlash2ConvolutionKernels, in FP32.
private func GLM5_3DFlash2Convolution(
  _ x: Model.IO, kernels: Model.IO, stage: Int, hidden: Int, rows: Int
) -> Model.IO {
  let x = x.to(.Float32)
  let current = kernels.reshaped(
    [rows, hidden], offset: [0, 2 * stage * hidden], strides: [4 * hidden, 1])
  let before = kernels.reshaped(
    [rows - 1, hidden], offset: [1, (2 * stage + 1) * hidden], strides: [4 * hidden, 1])
  let previous = x.reshaped([rows - 1, hidden], offset: [0, 0], strides: [hidden, 1]) .* before
  return x .* current + previous.padded(.zero, begin: [1, 0], end: [0, 0])
}

/// One draft, shaped as DeepSeek4_1DSpark: the block [anchor, mask x 7] attends, both ways, to
/// itself and to the windowRows - 1 positions before the anchor. Each layer keeps the keys and
/// values of the committed positions in caches it reads through a view of windowRows - 1 +
/// mainRows + blockSize rows from `historyOffset`: the window before the first main row, then rows
/// the draft fills in place with the main rows' keys and values (the target rows through `fc`) and
/// with the block's. The mask leaves open only the block and the window ending at the last
/// committed main row. The caches are bound whole, as a bound tensor can't start past its buffer's
/// base, so the view's start is part of the graph. The layers run in BF16, as trained: the residual
/// stream reaches millions, past FP16. The two-tap convolutions run in FP32; the embedding, head,
/// rotary, attention and caches in `FloatType`, the target's.
/// Inputs: target rows [mainRows, targets * hidden] (FP32; committed rows first, then any the mask
/// closes), their rotary [mainRows, head dim], the block tokens [blockSize] ([anchor, mask...]),
/// their rotary [blockSize, head dim], the additive attention mask [1, 1, blockSize, view rows] (a
/// column per view row), then each layer's key cache and value cache [at least historyOffset + view
/// rows, keyValueHeads * head dim]. A temperature of 0 drafts greedily; a positive one samples
/// softmax(score / temperature) over each position's candidates.
/// Outputs: the drafted tokens [proposals] (Int32, picked in order after the selector's edge
/// from the token before), each position's peak probability at temperature 1 [proposals, 1]
/// (FP32), and its candidates' scores [proposals, selectorCandidates] (FP32, edges included)
/// and token ids (Int32).
public func GLM5_3DFlash2<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, mainRows: Int, temperature: Float, historyOffset: Int,
  configuration: GLM5_3ModelConfiguration = .glm5_3Flash,
  draft: GLM5_3DFlash2Configuration = .glm5_3Flash
) -> Model {
  let windowRows = draft.windowRows - 1
  precondition(mainRows > 0 && historyOffset >= 0)
  let hidden = configuration.hiddenSize
  let dim = draft.headDim
  let heads = draft.queryHeads
  let kvHeads = draft.keyValueHeads
  let width = draft.keyValueWidth
  let block = draft.blockSize
  let epsilon = draft.normEpsilon
  let historyRows = windowRows + mainRows + block
  let mainHidden = Input()
  let mainRotary = Input()
  let tokens = Input()
  let blockRotary = Input()
  let mask = Input()
  let keyHistories = (0..<draft.layers).map { _ in Input() }
  let valueHistories = (0..<draft.layers).map { _ in Input() }
  let mainX = RMSNorm(epsilon: epsilon, axis: [1], name: "hidden_norm")(
    Dense(count: hidden, noBias: true, name: "fc")(mainHidden.to(.BFloat16)))
  var out = Embedding(
    dataType, vocabularySize: configuration.vocabularySize, embeddingSize: hidden,
    name: "embed_tokens")(tokens).to(.BFloat16)
  for layer in 0..<draft.layers {
    let prefix = "layers.\(layer)"
    let attentionNormalized = RMSNorm(
      epsilon: epsilon, axis: [1], name: "\(prefix).input_layernorm")(out)
    let attentionKernels = GLM5_3DFlash2ConvolutionKernels(
      prefix: "\(prefix).attention_conv", normalized: attentionNormalized,
      configuration: configuration, draft: draft)
    let x = GLM5_3DFlash2Convolution(
      attentionNormalized, kernels: attentionKernels, stage: 0, hidden: hidden, rows: block
    ).to(.BFloat16)
    let wk = Dense(count: width, noBias: true, name: "\(prefix).self_attn.k_proj")
    let wv = Dense(count: width, noBias: true, name: "\(prefix).self_attn.v_proj")
    let kNorm = RMSNorm(epsilon: epsilon, axis: [3], name: "\(prefix).self_attn.k_norm")
    let mainKeys = Functional.cmul(
      left: kNorm(wk(mainX).reshaped([1, mainRows, kvHeads, dim])).to(dataType.dataType),
      right: mainRotary.reshaped([1, mainRows, 1, dim]))
    let blockKeys = Functional.cmul(
      left: kNorm(wk(x).reshaped([1, block, kvHeads, dim])).to(dataType.dataType),
      right: blockRotary.reshaped([1, block, 1, dim]))
    // All four write into the view in place, and attention reads it after them.
    let keys = keyHistories[layer]
    let values = valueHistories[layer]
    let writes = [
      mainKeys.reshaped([mainRows, width]).moved(
        to: keys.reshaped(
          [mainRows, width], offset: [historyOffset + windowRows, 0], strides: [width, 1])),
      wv(mainX).to(dataType.dataType).moved(
        to: values.reshaped(
          [mainRows, width], offset: [historyOffset + windowRows, 0], strides: [width, 1])),
      blockKeys.reshaped([block, width]).moved(
        to: keys.reshaped(
          [block, width], offset: [historyOffset + windowRows + mainRows, 0], strides: [width, 1])
      ),
      wv(x).to(dataType.dataType).moved(
        to: values.reshaped(
          [block, width], offset: [historyOffset + windowRows + mainRows, 0], strides: [width, 1])
      ),
    ]
    let query = Functional.cmul(
      left: RMSNorm(epsilon: epsilon, axis: [3], name: "\(prefix).self_attn.q_norm")(
        Dense(count: heads * dim, noBias: true, name: "\(prefix).self_attn.q_proj")(x)
          .reshaped([1, block, heads, dim])
      ).to(dataType.dataType), right: blockRotary.reshaped([1, block, 1, dim]))
    let viewShape: TensorShape = [1, historyRows, kvHeads, dim]
    let viewStrides: TensorShape = [historyRows * width, width, dim, 1]
    let attended = ScaledDotProductAttention(
      scale: 1 / Float(dim).squareRoot(), hasAttentionMask: true, flags: [.Float16])(
        query, keys.reshaped(viewShape, offset: [0, historyOffset, 0, 0], strides: viewStrides),
        values.reshaped(viewShape, offset: [0, historyOffset, 0, 0], strides: viewStrides), mask)
    attended.add(dependencies: writes)
    let attention = Dense(count: hidden, noBias: true, name: "\(prefix).self_attn.o_proj")(
      attended.reshaped([block, heads * dim]).to(.BFloat16))
    let afterAttention =
      out
      + GLM5_3DFlash2Convolution(
        attention, kernels: attentionKernels, stage: 1, hidden: hidden, rows: block
      ).to(.BFloat16)
    let ffnNormalized = RMSNorm(
      epsilon: epsilon, axis: [1], name: "\(prefix).post_attention_layernorm")(afterAttention)
    let ffnKernels = GLM5_3DFlash2ConvolutionKernels(
      prefix: "\(prefix).mlp_conv", normalized: ffnNormalized, configuration: configuration,
      draft: draft)
    let y = GLM5_3DFlash2Convolution(
      ffnNormalized, kernels: ffnKernels, stage: 0, hidden: hidden, rows: block
    ).to(.BFloat16)
    let gate = Dense(count: draft.intermediateSize, noBias: true, name: "\(prefix).mlp.gate_proj")(
      y)
    let up = Dense(count: draft.intermediateSize, noBias: true, name: "\(prefix).mlp.up_proj")(y)
    let ffn = Dense(count: hidden, noBias: true, name: "\(prefix).mlp.down_proj")(
      Functional.swishMul(value: up, gate: gate))
    out =
      afterAttention
      + GLM5_3DFlash2Convolution(ffn, kernels: ffnKernels, stage: 1, hidden: hidden, rows: block)
      .to(.BFloat16)
  }
  let proposals = draft.proposals
  let candidates = draft.selectorCandidates
  let rank = draft.selectorRank
  let collapsed = RMSNorm(epsilon: epsilon, axis: [1], name: "norm")(
    out.reshaped([proposals, hidden], offset: [1, 0], strides: [hidden, 1]))
  let top = Dense(count: configuration.vocabularySize, noBias: true, name: "lm_head")(
    collapsed.to(dataType.dataType)
  ).partitioned(kth: candidates, axis: 1, descending: true)
  let selectorRows = Dense(
    count: rank, noBias: true, name: "candidate_selector.hidden_projection")(collapsed).to(
      .Float32)
  // Each position's candidates score their logit plus the bilinear edge from the token picked
  // before it, through the position's selector row, so the picks run in order.
  let predecessorCodebook = Embedding(
    BFloat16.self, vocabularySize: configuration.vocabularySize, embeddingSize: rank,
    name: "candidate_selector.predecessor_codebook")
  let successorCodebook = Embedding(
    BFloat16.self, vocabularySize: configuration.vocabularySize, embeddingSize: rank,
    name: "candidate_selector.successor_codebook")
  let successors = successorCodebook(top[1].reshaped([proposals * candidates])).to(.Float32)
  var previous: Model.IO = tokens.reshaped([1], offset: [0], strides: [1])
  var picks = [Model.IO]()
  var peaks = [Model.IO]()
  var scores = [Model.IO]()
  for position in 0..<proposals {
    let context =
      predecessorCodebook(previous).to(.Float32)
      .* selectorRows.reshaped([1, rank], offset: [position, 0], strides: [rank, 1])
    let edges = Matmul(transposeB: (0, 1))(
      context,
      successors.reshaped(
        [candidates, rank], offset: [position * candidates, 0], strides: [rank, 1]))
    let score =
      top[0].reshaped([1, candidates], offset: [position, 0], strides: [candidates, 1]).to(
        .Float32) + edges
    scores.append(score)
    peaks.append(score.softmax().reduced(.max, axis: [1]))
    previous = IndexSelect()(
      top[1].reshaped([candidates, 1], offset: [position * candidates, 0], strides: [1, 1]),
      score.gumbelArgmax(axis: 1, scale: temperature).reshaped([1])
    ).reshaped([1])
    picks.append(previous)
  }
  return Model(
    [mainHidden, mainRotary, tokens, blockRotary, mask]
      + zip(keyHistories, valueHistories).flatMap { [$0, $1] },
    [
      Concat(axis: 0)(picks), Concat(axis: 0)(peaks), Concat(axis: 0)(scores),
      top[1],
    ])
}
