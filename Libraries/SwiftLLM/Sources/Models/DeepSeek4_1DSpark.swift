import Foundation
import NNC

/// DeepSeek V4.1's DSpark drafter: three sliding-window stages, stored after the trunk as
/// layers 40 to 42, draft a block [token, noise x 4] in one pass. Stage 0 reads the mean HC
/// stream entering the target layers through `main_proj`; each stage keeps its own window of
/// the committed positions' keys, and the last stage adds a rank-256 Markov bias per position
/// and a confidence score.
public struct DeepSeek4_1DSparkConfiguration: Codable, Sendable {
  public var stages = 3
  public var blockSize = 5
  public var noiseTokenID: Int32 = 128_799
  public var targetLayers = [37, 38, 39]
  public var markovRank = 256
  public var expertCount = 128
  public var routedExperts = 3

  public static let deepSeekV4_1Flash = DeepSeek4_1DSparkConfiguration()
}

/// One draft. Each stage keeps the keys of the committed positions in a cache it reads through a
/// view of rawWindow + mainRows + blockSize rows: the rawWindow positions before the first
/// committed row, then rows the draft fills in place with the main rows' keys and with the
/// block's. Attention spans the view with no causal mask, and the mask leaves open only the block
/// and the rawWindow positions ending at the last committed row, so the caller moves the view
/// along the cache while the graph keeps its shape.
/// Inputs: target rows [mainRows, targets * hidden] (FP32; committed rows first, then any the
/// mask closes), their raw RoPE [mainRows, head dim], the block tokens [blockSize] ([pending,
/// noise...]), their raw RoPE, the one-hot pre-mix [blockSize, hc], the additive attention mask
/// [1, 1, blockSize, rawWindow + mainRows + blockSize] (a column per view row), the min-p window
/// [1, 1] (FP32: logits further than this below a position's best are never picked), then each
/// stage's view [rawWindow + mainRows + blockSize, head dim]. A temperature of 0 drafts greedily;
/// a positive one samples softmax(logits / temperature) within the window. The stages always run
/// the whole block; only its first sequenceLength - 1 positions, at most blockSize, reach the head.
/// Outputs: the drafted tokens [sequenceLength - 1] (Int32, picked in order after the Markov bias of
/// the token before), the confidence logits [sequenceLength - 1, 1] (FP32), and the logits the picks
/// were drawn from [sequenceLength - 1, vocabulary] (FP32, Markov bias included).
public func DeepSeek4_1DSpark<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, mainRows: Int, temperature: Float, sequenceLength: Int,
  configuration: DeepSeek4_1ModelConfiguration = .deepSeekV4_1Flash,
  draft: DeepSeek4_1DSparkConfiguration = .deepSeekV4_1Flash
) -> Model {
  let windowRows = configuration.rawWindow
  precondition(mainRows > 0)
  precondition((2...(draft.blockSize + 1)).contains(sequenceLength))
  let hidden = configuration.hiddenSize
  let dim = configuration.attentionHeadDim
  let heads = configuration.attentionHeads
  let hc = configuration.hcCount
  let block = draft.blockSize
  let historyRows = windowRows + mainRows + block
  let mainHidden = Input()
  let mainRotary = Input()
  let tokens = Input()
  let blockRotary = Input()
  let identityPre = Input()
  let mask = Input()
  let minPWindow = Input()
  let histories = (0..<draft.stages).map { _ in Input() }
  let first = "layers.\(configuration.layers)"
  // A few target rows reach past FP16's range. main_norm undoes any per-row scale, so the rows
  // go to unit RMS in FP32 first.
  let mainX = RMSNorm(
    epsilon: configuration.normEpsilon, axis: [1], name: "\(first).main_norm")(
      Dense(count: hidden, noBias: true, name: "\(first).main_proj")(
        RMSNorm(epsilon: configuration.normEpsilon, axis: [1], elementwiseAffine: false)(
          mainHidden
        ).to(FloatType.dataType)))
  // The stages route over their own experts, numbered after the trunk's layers.
  var stageConfiguration = configuration
  stageConfiguration.layers = configuration.layers + draft.stages
  stageConfiguration.expertCount = draft.expertCount
  stageConfiguration.routedExperts = draft.routedExperts
  stageConfiguration.expertResidentSlots = Array(
    repeating: draft.expertCount, count: stageConfiguration.layers)
  let embedding = Embedding(
    dataType, vocabularySize: configuration.vocabularySize, embeddingSize: hidden, name: "embed")(
      tokens
    ).reshaped([1, block, 1, hidden])
  var residual = Upsample(.nearest, widthScale: Float(hc), heightScale: 1)(embedding)
    .to(.Float32).reshaped([block, hc, hidden])
  var pre: Model.IO = identityPre
  for stage in 0..<draft.stages {
    let layerIndex = configuration.layers + stage
    let prefix = "layers.\(layerIndex)"
    let (attnPre, attnPost, attnCombination, attnNormalized) = DeepSeek4_1HCMix(
      prefix: "\(prefix).hc_attn", normName: "\(prefix).attn_norm", residual: residual,
      incomingPre: pre, tokenLength: block, configuration: configuration)
    let x = attnNormalized.to(FloatType.dataType)
    let wkv = Dense(count: dim, noBias: true, name: "\(prefix).attn.wkv")
    let kvNorm = RMSNorm(
      epsilon: configuration.normEpsilon, axis: [1], name: "\(prefix).attn.kv_norm")
    let mainKeys = Functional.cmul(left: kvNorm(wkv(mainX)), right: mainRotary)
    let blockKeys = Functional.cmul(left: kvNorm(wkv(x)), right: blockRotary)
    // Both write into the view in place, and attention reads it after them.
    let history = histories[stage]
    let mainWrite = mainKeys.moved(
      to: history.reshaped([mainRows, dim], offset: [windowRows, 0], strides: [dim, 1]))
    let blockWrite = blockKeys.moved(
      to: history.reshaped([block, dim], offset: [windowRows + mainRows, 0], strides: [dim, 1]))
    let low = Dense(count: configuration.queryLowRank, noBias: true, name: "\(prefix).attn.wq_a")(x)
    let rank = RMSNorm(
      epsilon: configuration.normEpsilon, axis: [1], name: "\(prefix).attn.q_norm")(low)
    let query = Functional.cmul(
      left: Dense(count: heads * dim, noBias: true, name: "\(prefix).attn.wq_b")(rank)
        .reshaped([1, block, heads, dim]),
      right: blockRotary.reshaped([1, block, 1, dim]))
    let keys = history.reshaped([1, historyRows, 1, dim])
    let sinks = Parameter<FloatType>(
      .GPU(0), .NHWC(1, 1, heads, 1), name: "\(prefix).attn.attn_sink")
    let attended = ScaledDotProductAttention(
      scale: 1 / Float(dim).squareRoot(), hasAttentionMask: true, hasAttentionSinks: true)(
        query, keys, keys, mask, sinks)
    attended.add(dependencies: [mainWrite, blockWrite])
    let attention = DeepSeek4_1AttentionOutput(
      dataType, prefix: "\(prefix).attn", heads: attended, rotary: blockRotary, tokenLength: block,
      configuration: configuration)
    let afterAttention = HyperConnection(count: hc, operation: .expand)(
      attention, residual, attnPost, attnCombination)[0]
    let (ffnPre, ffnPost, ffnCombination, normalizedFFN) = DeepSeek4_1HCMix(
      prefix: "\(prefix).hc_ffn", normName: "\(prefix).ffn_norm", residual: afterAttention,
      incomingPre: attnPre, tokenLength: block, configuration: configuration)
    let moe = DeepSeek4_1MoE(
      dataType, prefix: "\(prefix).ffn", tokenLength: block, layerIndex: layerIndex,
      configuration: stageConfiguration)
    let ffn = moe.model(normalizedFFN)
    residual =
      HyperConnection(count: hc, operation: .expandSum)(
        ffn[0], afterAttention, ffnPost, ffnCombination, ffn[1])[0]
    pre = ffnPre
  }
  let last = "layers.\(configuration.layers + draft.stages - 1)"
  let collapsed = (residual .* pre.reshaped([block, hc, 1])).reduced(.sum, axis: [1])
    .reshaped([sequenceLength - 1, hidden], offset: [0, 0], strides: [hidden, 1])
  let logits = Dense(count: configuration.vocabularySize, noBias: true, name: "head")(
    RMSNorm(epsilon: configuration.normEpsilon, axis: [1], name: "\(last).norm")(collapsed)
      .to(FloatType.dataType)
  ).to(.Float32)
  // Each position adds the Markov bias of the token before it, so the picks run in order.
  let markovEmbedding = Embedding(
    Float.self, vocabularySize: configuration.vocabularySize, embeddingSize: draft.markovRank,
    name: "\(last).markov_head.embed")
  let markovHead = Dense(
    count: configuration.vocabularySize, noBias: true, name: "\(last).markov_head.head")
  var previous: Model.IO = tokens.reshaped([1], offset: [0], strides: [1])
  var picks = [Model.IO]()
  var markovEmbeddings = [Model.IO]()
  var biasedRows = [Model.IO]()
  for position in 0..<(sequenceLength - 1) {
    let embedded = markovEmbedding(previous)
    markovEmbeddings.append(embedded)
    let row = logits.reshaped(
      [1, configuration.vocabularySize], offset: [position, 0],
      strides: [configuration.vocabularySize, 1])
    let biased = row + markovHead(embedded)
    biasedRows.append(biased)
    // Gumbel-max over the min-p window samples softmax(biased / temperature) there: a logit below
    // the window takes a penalty no noise overcomes.
    let window = biased - biased.reduced(.max, axis: [1]) + minPWindow
    previous = (biased - (window * -1e6).ReLU()).gumbelArgmax(axis: 1, scale: temperature)
      .reshaped([1])
    picks.append(previous)
  }
  let confidence = Dense(count: 1, noBias: true, name: "\(last).confidence_head.proj")(
    Concat(axis: 1)(collapsed, Concat(axis: 0)(markovEmbeddings)))
  return Model(
    [mainHidden, mainRotary, tokens, blockRotary, identityPre, mask, minPWindow] + histories,
    [Concat(axis: 0)(picks), confidence, Concat(axis: 0)(biasedRows)])
}
