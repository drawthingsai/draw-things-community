import DiffusionMappings
import Foundation
import NNC

public func ErnieImageTimeEmbedding(
  timestep: Float, batchSize: Int, embeddingSize: Int, maxPeriod: Int = 10_000
) -> Tensor<Float> {
  precondition(embeddingSize % 2 == 0)
  var embedding = Tensor<Float>(.CPU, .WC(batchSize, embeddingSize))
  let half = embeddingSize / 2
  for i in 0..<half {
    let exponent = -log(Float(maxPeriod)) * Float(i) / Float(half)
    let freq = exp(exponent)
    let value = timestep * freq
    let sinValue = sin(value)
    let cosValue = cos(value)
    for j in 0..<batchSize {
      embedding[j, i] = sinValue
      embedding[j, i + half] = cosValue
    }
  }
  return embedding
}

public func ErnieImageRotaryPositionEmbedding(
  height: Int, width: Int, textLength: Int, heads: Int = 1
) -> (Tensor<Float>, Tensor<Float>) {
  let imageLength = height * width
  var cosTensor = Tensor<Float>(
    .CPU, .NHWC(1, imageLength + textLength, heads, 128))
  var sinTensor = Tensor<Float>(
    .CPU, .NHWC(1, imageLength + textLength, heads, 128))

  for token in 0..<(imageLength + textLength) {
    let positions: (Double, Double, Double)
    if token < imageLength {
      positions = (Double(textLength), Double(token / width), Double(token % width))
    } else {
      positions = (Double(token - imageLength), 0, 0)
    }
    for j in 0..<heads {
      for k in 0..<64 {
        let angle: Double
        if k < 16 {
          angle = positions.0 / pow(256.0, Double(k * 2) / 32.0)
        } else if k < 40 {
          angle = positions.1 / pow(256.0, Double((k - 16) * 2) / 48.0)
        } else {
          angle = positions.2 / pow(256.0, Double((k - 40) * 2) / 48.0)
        }
        let c = Float(cos(angle))
        let s = Float(sin(angle))
        cosTensor[0, token, j, k * 2] = c
        cosTensor[0, token, j, k * 2 + 1] = c
        // Bake the non-interleaved rotate_half sign into the swapped-half multiplier.
        let signedS = k < 32 ? -s : s
        sinTensor[0, token, j, k * 2] = signedS
        sinTensor[0, token, j, k * 2 + 1] = signedS
      }
    }
  }
  return (cosTensor, sinTensor)
}

public func ErnieImageFixed(tokenLength: Int, timesteps: Int, channels: Int) -> Model {
  let txt = Input()
  let t = Input()
  let textProj = Dense(count: channels, noBias: true, name: "c_embedder")
  let projectedText = textProj(txt)
  let timeMlp0 = Dense(count: channels, name: "time_embedder_0")
  let timeMlp2 = Dense(count: channels, name: "time_embedder_1")
  let timeCondition = timeMlp2(timeMlp0(t).swish())
  let timeModulation = timeCondition.swish()
  let adaLNs = (0..<6).map { Dense(count: channels, name: "ada_ln_\($0)") }
  let shiftMSA = adaLNs[0](timeModulation)
  let scaleMSA = 1 + adaLNs[1](timeModulation)
  let gateMSA = adaLNs[2](timeModulation)
  let shiftMLP = adaLNs[3](timeModulation)
  let scaleMLP = 1 + adaLNs[4](timeModulation)
  let gateMLP = adaLNs[5](timeModulation)
  let finalScale = Dense(count: channels, name: "scale")
  let finalShift = Dense(count: channels, name: "shift")
  return Model(
    [txt, t],
    [
      projectedText, shiftMSA, scaleMSA, gateMSA, shiftMLP, scaleMLP, gateMLP,
      1 + finalScale(timeCondition), finalShift(timeCondition),
    ])
}

private func ErnieImageFeedForward(hiddenSize: Int, intermediateSize: Int, name: String) -> (
  Model, Model, Model, Model
) {
  let x = Input()
  let gate = Dense(count: intermediateSize, noBias: true, name: "\(name)_gate_proj")
  let up = Dense(count: intermediateSize, noBias: true, name: "\(name)_up_proj")
  var out = up(x) .* gate(x).GELU()
  let down = Dense(count: hiddenSize, noBias: true, name: "\(name)_down_proj")
  out = down(out).to(.Float32)
  return (gate, down, up, Model([x], [out]))
}

private func ErnieImageApplyRotary(
  _ x: Model.IO, cos: Model.IO, sin: Model.IO, batchSize: Int, tokenLength: Int, heads: Int,
  headDim: Int
) -> Model.IO {
  let halfDim = headDim / 2
  let firstHalf = x.reshaped(
    [batchSize, tokenLength, heads, halfDim], offset: [0, 0, 0, 0],
    strides: [tokenLength * heads * headDim, heads * headDim, headDim, 1]
  ).copied()
  let secondHalf = x.reshaped(
    [batchSize, tokenLength, heads, halfDim], offset: [0, 0, 0, halfDim],
    strides: [tokenLength * heads * headDim, heads * headDim, headDim, 1]
  ).copied()
  let rotated = Functional.concat(axis: 3, secondHalf, firstHalf)
  return x .* cos + rotated .* sin
}

private func ErnieImageTransformerBlock(
  prefix: String, hiddenSize: Int, k: Int, h: Int, batchSize: Int, t: (keyValue: Int, query: Int),
  intermediateSize: Int, usesFlashAttention: FlashAttentionLevel
) -> Model {
  let x = Input()
  let rotCos = Input()
  let rotSin = Input()
  let shiftMSA = Input()
  let scaleMSA = Input()
  let gateMSA = Input()
  let shiftMLP = Input()
  let scaleMLP = Input()
  let gateMLP = Input()

  let attentionNorm = RMSNorm(epsilon: 1e-6, axis: [2], name: "attention_norm1")
  var out = attentionNorm(x)
  out = (scaleMSA.to(of: out) .* out + shiftMSA.to(of: out)).to(.Float16)
  let tokeys = Dense(count: k * h, noBias: true, name: "k")
  let toqueries = Dense(count: k * h, noBias: true, name: "q")
  let tovalues = Dense(count: k * h, noBias: true, name: "v")
  var keys = tokeys(out).reshaped([batchSize, t.keyValue, h, k])
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "norm_k")
  keys = normK(keys)
  var queries: Model.IO
  if t.keyValue != t.query {
    queries = toqueries(
      out.reshaped(
        [batchSize, t.query, k * h], offset: [0, 0, 0], strides: [t.keyValue * h * k, h * k, 1])
    ).reshaped([batchSize, t.query, h, k])
  } else {
    queries = toqueries(out).reshaped([batchSize, t.keyValue, h, k])
  }
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "norm_q")
  queries = normQ(queries)
  let values = tovalues(out).reshaped([batchSize, t.keyValue, h, k])
  let queryCos: Model.IO
  let querySin: Model.IO
  if t.keyValue != t.query {
    queryCos = rotCos.reshaped(
      [1, t.query, 1, k], offset: [0, 0, 0, 0], strides: [t.keyValue * k, k, k, 1])
    querySin = rotSin.reshaped(
      [1, t.query, 1, k], offset: [0, 0, 0, 0], strides: [t.keyValue * k, k, k, 1])
  } else {
    queryCos = rotCos
    querySin = rotSin
  }
  queries = ErnieImageApplyRotary(
    queries, cos: queryCos, sin: querySin, batchSize: batchSize, tokenLength: t.query, heads: h,
    headDim: k)
  keys = ErnieImageApplyRotary(
    keys, cos: rotCos, sin: rotSin, batchSize: batchSize, tokenLength: t.keyValue, heads: h,
    headDim: k)

  var attentionOut: Model.IO
  switch usesFlashAttention {
  case .none:
    let transposedKeys = keys.transposed(1, 2)
    let transposedQueries = ((1.0 / Float(k).squareRoot()) * queries).transposed(1, 2)
    let transposedValues = values.transposed(1, 2)
    var dot = Matmul(transposeB: (2, 3))(transposedQueries, transposedKeys)
    dot = dot.reshaped([batchSize * h * t.query, t.keyValue])
    dot = dot.softmax()
    dot = dot.reshaped([batchSize, h, t.query, t.keyValue])
    attentionOut = (dot * transposedValues).transposed(1, 2).reshaped([
      batchSize, t.query, hiddenSize,
    ])
  case .scale1:
    let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
    attentionOut = scaledDotProductAttention(
      (1.0 / Float(k).squareRoot()) * queries, keys, values
    ).reshaped([batchSize, t.query, hiddenSize])
  case .scaleMerged, .quantized:
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1.0 / Float(k).squareRoot(),
      flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
    attentionOut = scaledDotProductAttention(
      queries, keys, values
    ).reshaped([batchSize, t.query, hiddenSize])
  }
  let xIn: Model.IO
  if t.keyValue > t.query {
    xIn = x.reshaped(
      [batchSize, t.query, h * k], offset: [0, 0, 0], strides: [t.keyValue * h * k, h * k, 1])
  } else {
    xIn = x
  }

  let attentionOutProj = Dense(count: hiddenSize, noBias: true, name: "o")
  attentionOut = attentionOutProj(attentionOut).to(.Float32)
  out = xIn + gateMSA.to(of: attentionOut) .* attentionOut

  let mlpNorm = RMSNorm(epsilon: 1e-6, axis: [2], name: "attention_norm2")
  var mlpIn = mlpNorm(out)
  mlpIn = (scaleMLP.to(of: mlpIn) .* mlpIn + shiftMLP.to(of: mlpIn)).to(.Float16)
  let (_, _, _, ffn) = ErnieImageFeedForward(
    hiddenSize: hiddenSize, intermediateSize: intermediateSize, name: "ffn")
  let mlpOut = ffn(mlpIn)
  out = out + gateMLP.to(of: mlpOut) .* mlpOut.to(of: out)

  return Model(
    [x, rotCos, rotSin, shiftMSA, scaleMSA, gateMSA, shiftMLP, scaleMLP, gateMLP], [out])
}

public func ErnieImage(
  batchSize: Int, height: Int, width: Int, textLength: Int,
  layers: Int, channels: Int, intermediateSize: Int, usesFlashAttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let rotCos = Input()
  let rotSin = Input()
  let txt = Input()
  let shiftMSA = Input()
  let scaleMSA = Input()
  let gateMSA = Input()
  let shiftMLP = Input()
  let scaleMLP = Input()
  let gateMLP = Input()
  let finalScale = Input()
  let finalShift = Input()
  let patchHeight = height / 2
  let patchWidth = width / 2
  let headDim = 128
  precondition(channels % headDim == 0)
  let heads = channels / headDim
  let xEmbedder = Convolution(
    groups: 1, filters: channels, filterSize: [2, 2],
    hint: Hint(stride: [2, 2], border: Hint.Border(begin: [0, 0], end: [0, 0])), format: .OIHW,
    name: "x_embedder")
  let img = xEmbedder(x).reshaped([batchSize, patchHeight * patchWidth, channels]).to(
    .Float32)

  let sequenceLength = patchHeight * patchWidth + textLength
  let rotCosResized = rotCos.reshaped([1, sequenceLength, 1, 128])
  let rotSinResized = rotSin.reshaped([1, sequenceLength, 1, 128])
  var out = Functional.concat(axis: 1, img, txt.to(.Float32))
  for i in 0..<layers {
    let block = ErnieImageTransformerBlock(
      prefix: "layers.\(i)", hiddenSize: channels, k: headDim, h: heads, batchSize: batchSize,
      t: (
        keyValue: sequenceLength, query: i == layers - 1 ? patchHeight * patchWidth : sequenceLength
      ),
      intermediateSize: intermediateSize, usesFlashAttention: usesFlashAttention)
    out = block(
      out, rotCosResized, rotSinResized, shiftMSA, scaleMSA, gateMSA, shiftMLP, scaleMLP,
      gateMLP)
  }

  let finalNorm = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false, name: "final_norm")
  out = (finalScale.to(of: out) .* finalNorm(out) + finalShift.to(of: out)).to(.Float16)
  let finalLinear = Dense(count: 128, name: "linear")
  out = finalLinear(out)

  let unpatchified = out.reshaped(
    [batchSize, patchHeight, patchWidth, 32, 2, 2]
  ).permuted(0, 1, 4, 2, 5, 3).contiguous().reshaped([batchSize, height, width, 32])

  return (
    { _ in [:] },
    Model(
      [
        x, rotCos, rotSin, txt, shiftMSA, scaleMSA, gateMSA, shiftMLP, scaleMLP, gateMLP,
        finalScale, finalShift,
      ], [unpatchified])
  )
}
