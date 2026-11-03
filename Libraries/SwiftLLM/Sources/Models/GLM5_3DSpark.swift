import Foundation
import NNC

/// Red Hat's DSpark drafter for GLM-5.3-Flash: five Qwen3-style layers fill a block
/// [token, mask x 7] in one pass, causally within the block. Keys and values of the committed
/// positions come from the mean mHC stream entering the target layers, through `fc`. A rank-256
/// Markov bias per position and a confidence score follow, as in DeepSeek V4.1's drafter.
public struct GLM5_3DSparkConfiguration: Codable, Sendable {
  public var layers = 5
  public var heads = 64
  public var headDim = 64
  public var blockSize = 8
  public var intermediateSize = 12_288
  public var targetLayers = [20, 28, 32, 36, 40, 44]
  public var maskTokenID: Int32 = 154_856
  public var windowRows = 2_048
  public var markovRank = 256
  public var ropeTheta = 10_000.0

  public init() {}

  public static let glm5_3Flash = GLM5_3DSparkConfiguration()

  public var keyValueWidth: Int { heads * headDim }
}

/// Rotary rows [rows, headDim] for logical positions from `firstPosition`, each pair (cos, sin) of
/// one complex rotation. The converter interleaves the checkpoint's half-split head layout to match.
public func GLM5_3DSparkRotaryEmbedding<FloatType: TensorNumeric & BinaryFloatingPoint>(
  rows: Int, firstPosition: Int, draft: GLM5_3DSparkConfiguration = .glm5_3Flash,
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

/// The additive attention mask [1, 1, blockSize, windowRows + mainRows + blockSize] of a view whose
/// row r holds position first - windowRows + r. A block row attends to the windowRows positions
/// before the block, none before position 0, and to itself and the block rows before it.
public func GLM5_3DSparkAttentionMask(
  mainRows: Int, committedRows: Int, first: Int, draft: GLM5_3DSparkConfiguration = .glm5_3Flash
) -> Tensor<Float16> {
  let block = draft.blockSize
  let viewRows = draft.windowRows + mainRows + block
  let open = max(committedRows, draft.windowRows - first)
  var mask = [Float16](repeating: 0, count: block * viewRows)
  for row in 0..<block {
    for column in Array(0..<open)
      + Array((draft.windowRows + committedRows)..<(draft.windowRows + mainRows))
      + Array((draft.windowRows + mainRows + row + 1)..<viewRows)
    {
      mask[row * viewRows + column] = -Float16.greatestFiniteMagnitude
    }
  }
  return Tensor(mask, kind: .CPU, format: .NHWC, shape: [1, 1, block, viewRows])
}

/// One draft. Each layer keeps the keys and values of the committed positions in caches it reads
/// through views of windowRows + mainRows + blockSize rows: the windowRows positions before the
/// first main row, then rows the draft fills in place with the main rows' and with the block's.
/// Inputs: target rows [mainRows, targets * hidden] (FP32; committed rows first, then any the mask
/// closes), their rotary [mainRows, headDim], the block tokens [blockSize] ([pending, mask...]),
/// their rotary, the additive attention mask, the min-p window [1, 1] (FP32: logits further than
/// this below a position's best are never picked), then each layer's key and value views
/// [viewRows, heads * headDim]. A temperature of 0 drafts greedily; a positive one samples
/// softmax(logits / temperature) within the window. The layers always run the whole block; only its
/// first sequenceLength - 1 positions, at most blockSize, reach the head.
/// Outputs: the drafted tokens [sequenceLength - 1] (Int32, picked in order after the Markov bias of
/// the token before), the confidence logits [sequenceLength - 1, 1] (FP32), and the logits the picks
/// were drawn from [sequenceLength - 1, vocabulary] (FP32, Markov bias included).
public func GLM5_3DSpark<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, mainRows: Int, temperature: Float, sequenceLength: Int,
  configuration: GLM5_3ModelConfiguration = .glm5_3Flash,
  draft: GLM5_3DSparkConfiguration = .glm5_3Flash
) -> Model {
  precondition(mainRows > 0)
  precondition((2...(draft.blockSize + 1)).contains(sequenceLength))
  let hidden = configuration.hiddenSize
  let epsilon = configuration.normEpsilon
  let heads = draft.heads
  let dim = draft.headDim
  let width = draft.keyValueWidth
  let block = draft.blockSize
  let viewRows = draft.windowRows + mainRows + block
  let mainHidden = Input()
  let mainRotary = Input()
  let tokens = Input()
  let blockRotary = Input()
  let mask = Input()
  let minPWindow = Input()
  var inputs: [Input] = [mainHidden, mainRotary, tokens, blockRotary, mask, minPWindow]
  // A few target rows can reach past FP16's range. hidden_norm undoes any per-row scale, so the
  // rows go to unit RMS in FP32 first.
  let mainX = RMSNorm(epsilon: epsilon, axis: [1], name: "hidden_norm")(
    Dense(count: hidden, noBias: true, name: "fc")(
      RMSNorm(epsilon: epsilon, axis: [1], elementwiseAffine: false)(mainHidden)
        .to(FloatType.dataType)))
  var out = Embedding(
    dataType, vocabularySize: configuration.vocabularySize, embeddingSize: hidden,
    name: "embed_tokens")(
      tokens
    ).reshaped([block, hidden])
  let rotaryOfMain = mainRotary.reshaped([1, mainRows, 1, dim])
  let rotaryOfBlock = blockRotary.reshaped([1, block, 1, dim])
  let viewStrides: TensorShape = [viewRows * width, width, dim, 1]
  for layer in 0..<draft.layers {
    let prefix = "layers.\(layer)"
    let keysIn = Input()
    let valuesIn = Input()
    inputs.append(contentsOf: [keysIn, valuesIn])
    let keyProjection = Dense(count: width, noBias: true, name: "\(prefix).self_attn.k_proj")
    let valueProjection = Dense(count: width, noBias: true, name: "\(prefix).self_attn.v_proj")
    let keyNorm = RMSNorm(epsilon: epsilon, axis: [3], name: "\(prefix).self_attn.k_norm")
    let normalized = RMSNorm(epsilon: epsilon, axis: [1], name: "\(prefix).input_layernorm")(out)
    func keys(_ x: Model.IO, rows: Int, rotary: Model.IO) -> Model.IO {
      Functional.cmul(
        left: keyNorm(keyProjection(x).reshaped([1, rows, heads, dim], format: .NHWC)),
        right: rotary)
    }
    func values(_ x: Model.IO, rows: Int) -> Model.IO {
      valueProjection(x).reshaped([1, rows, heads, dim], format: .NHWC)
    }
    let writes = [
      keys(mainX, rows: mainRows, rotary: rotaryOfMain).moved(
        to: keysIn.reshaped(
          [1, mainRows, heads, dim], offset: [0, draft.windowRows, 0, 0], strides: viewStrides)),
      values(mainX, rows: mainRows).moved(
        to: valuesIn.reshaped(
          [1, mainRows, heads, dim], offset: [0, draft.windowRows, 0, 0], strides: viewStrides)),
      keys(normalized, rows: block, rotary: rotaryOfBlock).moved(
        to: keysIn.reshaped(
          [1, block, heads, dim], offset: [0, draft.windowRows + mainRows, 0, 0],
          strides: viewStrides)),
      values(normalized, rows: block).moved(
        to: valuesIn.reshaped(
          [1, block, heads, dim], offset: [0, draft.windowRows + mainRows, 0, 0],
          strides: viewStrides)),
    ]
    let queries = Functional.cmul(
      left: RMSNorm(epsilon: epsilon, axis: [3], name: "\(prefix).self_attn.q_norm")(
        Dense(count: width, noBias: true, name: "\(prefix).self_attn.q_proj")(normalized)
          .reshaped([1, block, heads, dim], format: .NHWC)),
      right: rotaryOfBlock)
    let viewShape: TensorShape = [1, viewRows, heads, dim]
    // Reads go through aliases: the caller moves the views along its caches, and rebinding an input
    // read directly leaves it at the first view's offset.
    let attended = ScaledDotProductAttention(
      scale: 1 / Float(dim).squareRoot(), hasAttentionMask: true, flags: [.Float16])(
        queries, keysIn.reshaped(viewShape, offset: [0, 0, 0, 0], strides: viewStrides),
        valuesIn.reshaped(viewShape, offset: [0, 0, 0, 0], strides: viewStrides), mask)
    attended.add(dependencies: writes)
    let residual =
      out
      + Dense(count: hidden, noBias: true, name: "\(prefix).self_attn.o_proj")(
        attended.reshaped([block, width]))
    let feedForward = SwiGLU(count: draft.intermediateSize, name: "\(prefix).mlp")(
      RMSNorm(epsilon: epsilon, axis: [1], name: "\(prefix).post_attention_layernorm")(residual))
    out =
      residual + Dense(count: hidden, noBias: true, name: "\(prefix).mlp.down_proj")(feedForward)
  }
  let head = RMSNorm(epsilon: epsilon, axis: [1], name: "norm")(
    out.reshaped([sequenceLength - 1, hidden], offset: [0, 0], strides: [hidden, 1]))
  let logits = Dense(count: configuration.vocabularySize, noBias: true, name: "lm_head")(head)
    .to(.Float32)
  // Each position adds the Markov bias of the token before it, so the picks run in order.
  let markovEmbedding = Embedding(
    Float.self, vocabularySize: configuration.vocabularySize, embeddingSize: draft.markovRank,
    name: "markov_head.markov_w1")
  let markovHead = Dense(
    count: configuration.vocabularySize, noBias: true, name: "markov_head.markov_w2")
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
  let confidence = Dense(count: 1, name: "confidence_head.proj")(
    Concat(axis: 1)(head.to(.Float32), Concat(axis: 0)(markovEmbeddings)))
  return Model(inputs, [Concat(axis: 0)(picks), confidence, Concat(axis: 0)(biasedRows)])
}
