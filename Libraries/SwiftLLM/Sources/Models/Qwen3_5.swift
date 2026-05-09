import Foundation
import NNC

public struct Qwen3_5ModelConfiguration: Sendable {
  public var vocabularySize: Int
  public var hiddenSize: Int
  public var intermediateSize: Int
  public var layers: Int
  public var fullAttentionInterval: Int
  public var attentionHeads: Int
  public var keyValueHeads: Int
  public var attentionHeadDim: Int
  public var rotaryDim: Int
  public var mropeSection: (temporal: Int, height: Int, width: Int)
  public var ropeTheta: Double
  public var linearNumKeyHeads: Int
  public var linearNumValueHeads: Int
  public var linearKeyHeadDim: Int
  public var linearValueHeadDim: Int
  public var linearConvKernel: Int

  public init(
    vocabularySize: Int, hiddenSize: Int, intermediateSize: Int,
    layers: Int, fullAttentionInterval: Int, attentionHeads: Int,
    keyValueHeads: Int, attentionHeadDim: Int, rotaryDim: Int,
    mropeSection: (temporal: Int, height: Int, width: Int),
    ropeTheta: Double, linearNumKeyHeads: Int,
    linearNumValueHeads: Int, linearKeyHeadDim: Int,
    linearValueHeadDim: Int, linearConvKernel: Int
  ) {
    self.vocabularySize = vocabularySize
    self.hiddenSize = hiddenSize
    self.intermediateSize = intermediateSize
    self.layers = layers
    self.fullAttentionInterval = fullAttentionInterval
    self.attentionHeads = attentionHeads
    self.keyValueHeads = keyValueHeads
    self.attentionHeadDim = attentionHeadDim
    self.rotaryDim = rotaryDim
    self.mropeSection = mropeSection
    self.ropeTheta = ropeTheta
    self.linearNumKeyHeads = linearNumKeyHeads
    self.linearNumValueHeads = linearNumValueHeads
    self.linearKeyHeadDim = linearKeyHeadDim
    self.linearValueHeadDim = linearValueHeadDim
    self.linearConvKernel = linearConvKernel
  }

  public var linearKeyDim: Int { linearNumKeyHeads * linearKeyHeadDim }
  public var linearValueDim: Int { linearNumValueHeads * linearValueHeadDim }
  public var linearConvDim: Int { linearKeyDim * 2 + linearValueDim }

  public func isLinearAttentionLayer(_ layerIndex: Int) -> Bool {
    return (layerIndex + 1) % fullAttentionInterval != 0
  }
}

extension Qwen3_5ModelConfiguration {
  public static let qwen3_6_27B = Qwen3_5ModelConfiguration(
    vocabularySize: 248_320, hiddenSize: 5_120, intermediateSize: 17_408,
    layers: 64, fullAttentionInterval: 4, attentionHeads: 24,
    keyValueHeads: 4, attentionHeadDim: 256, rotaryDim: 64,
    mropeSection: (temporal: 11, height: 11, width: 10),
    ropeTheta: 10_000_000, linearNumKeyHeads: 16,
    linearNumValueHeads: 48, linearKeyHeadDim: 128,
    linearValueHeadDim: 128, linearConvKernel: 4)

  public static let qwen3_5_4B = Qwen3_5ModelConfiguration(
    vocabularySize: 248_320, hiddenSize: 2_560, intermediateSize: 9_216,
    layers: 32, fullAttentionInterval: 4,
    attentionHeads: 16, keyValueHeads: 4, attentionHeadDim: 256, rotaryDim: 64,
    mropeSection: (temporal: 11, height: 11, width: 10),
    ropeTheta: 10_000_000, linearNumKeyHeads: 16, linearNumValueHeads: 32,
    linearKeyHeadDim: 128, linearValueHeadDim: 128, linearConvKernel: 4)
}

public func Qwen3_5RotaryEmbedding<FloatType: TensorNumeric & BinaryFloatingPoint>(
  sequenceLength: Int, cachedTokenLength: Int = 0,
  configuration: Qwen3_5ModelConfiguration,
  of dataType: FloatType.Type = FloatType.self
) -> Tensor<FloatType> {
  let halfRotaryDim = configuration.rotaryDim / 2
  let halfHeadDim = configuration.attentionHeadDim / 2
  var rotary = Tensor<FloatType>(
    Array(repeating: FloatType.zero, count: sequenceLength * configuration.attentionHeadDim), .CPU,
    .NHWC(1, sequenceLength, 1, configuration.attentionHeadDim))
  for i in 0..<sequenceLength {
    let position = Double(cachedTokenLength + i)
    for k in 0..<halfRotaryDim {
      let theta =
        position / pow(configuration.ropeTheta, Double(k) * 2 / Double(configuration.rotaryDim))
      rotary[0, i, 0, k * 2] = FloatType(cos(theta))
      rotary[0, i, 0, k * 2 + 1] = FloatType(sin(theta))
    }
    for k in halfRotaryDim..<halfHeadDim {
      rotary[0, i, 0, k * 2] = 1
    }
  }
  return rotary
}

private func Qwen3_5FullAttention(
  prefix: String, configuration: Qwen3_5ModelConfiguration, batchSize: Int,
  tokenLength: Int, cachedTokenLength: Int, lastTokenOnly: Bool
) -> Model {
  let x = Input()
  let rotary = Input()
  let kIn = Input()
  let vIn = Input()
  let totalTokenLength = cachedTokenLength + tokenLength
  let headDim = configuration.attentionHeadDim
  let heads = configuration.attentionHeads
  let keyValueHeads = configuration.keyValueHeads
  let queryDim = heads * headDim
  let toqueries = Dense(
    count: queryDim, noBias: true, name: "\(prefix).self_attn.q_proj")
  let toqueryGates = Dense(
    count: queryDim, noBias: true, name: "\(prefix).self_attn.q_gate_proj")
  let tokeys = Dense(
    count: keyValueHeads * headDim, noBias: true, name: "\(prefix).self_attn.k_proj")
  let tovalues = Dense(
    count: keyValueHeads * headDim, noBias: true, name: "\(prefix).self_attn.v_proj")
  let queriesIn: Model.IO
  let queriesLength: Int
  if lastTokenOnly {
    queriesIn = x.reshaped(
      [1, configuration.hiddenSize], offset: [tokenLength - 1, 0],
      strides: [configuration.hiddenSize, 1])
    queriesLength = 1
  } else {
    queriesIn = x
    queriesLength = tokenLength
  }
  var queries = toqueries(queriesIn).reshaped([batchSize, queriesLength, heads, headDim])
  let gate = toqueryGates(queriesIn).reshaped([batchSize * queriesLength, queryDim])
  var keys = tokeys(x).reshaped([
    batchSize, tokenLength, keyValueHeads, headDim,
  ])
  let values = tovalues(x).reshaped([
    batchSize, tokenLength, keyValueHeads, headDim,
  ])
  let qNorm = RMSNorm(epsilon: 1e-6, axis: [3], name: "\(prefix).self_attn.q_norm")
  queries = qNorm(queries)
  let kNorm = RMSNorm(epsilon: 1e-6, axis: [3], name: "\(prefix).self_attn.k_norm")
  keys = kNorm(keys)
  if lastTokenOnly {
    queries = Functional.cmul(
      left: queries,
      right: rotary.reshaped(
        [1, 1, 1, headDim], offset: [0, tokenLength - 1, 0, 0],
        strides: [tokenLength * headDim, headDim, headDim, 1]))
  } else {
    queries = Functional.cmul(left: queries, right: rotary)
  }
  keys = Functional.cmul(left: keys, right: rotary)
  let kOut = keys.moved(
    to: kIn.reshaped(
      [batchSize, tokenLength, keyValueHeads, headDim],
      offset: [0, cachedTokenLength, 0, 0],
      strides: [totalTokenLength * keyValueHeads * headDim, keyValueHeads * headDim, headDim, 1]))
  let vOut = values.moved(
    to: vIn.reshaped(
      [batchSize, tokenLength, keyValueHeads, headDim],
      offset: [0, cachedTokenLength, 0, 0],
      strides: [totalTokenLength * keyValueHeads * headDim, keyValueHeads * headDim, headDim, 1]))
  var out = ScaledDotProductAttention(
    scale: 1.0 / Float(headDim).squareRoot(), isCausal: true,
    flags: tokenLength > 1 ? [.Int8, .Float16] : [.Float16])(
      queries, kIn, vIn
    )
  out.add(dependencies: [kOut, vOut])
  out = out.reshaped([batchSize * queriesLength, heads * headDim])
  out = out .* gate.sigmoid()
  let outProj = Dense(
    count: configuration.hiddenSize, noBias: true, name: "\(prefix).self_attn.o_proj")
  out = outProj(out)
  return Model([x, rotary, kIn, vIn], [out])
}

private func Qwen3_5LinearAttention<FloatType: TensorNumeric>(
  prefix: String, configuration: Qwen3_5ModelConfiguration, batchSize: Int, tokenLength: Int,
  x: Model.IO, convState: Model.IO, recurrentState: Model.IO, of: FloatType.Type = FloatType.self
) -> [Model.IO] {
  precondition(batchSize == 1)
  let keyHeads = configuration.linearNumKeyHeads
  let valueHeads = configuration.linearNumValueHeads
  let keyHeadDim = configuration.linearKeyHeadDim
  let valueHeadDim = configuration.linearValueHeadDim
  let keyDim = configuration.linearKeyDim
  let valueDim = configuration.linearValueDim
  let convDim = configuration.linearConvDim
  let inProjQKV = Dense(
    count: convDim, noBias: true, name: "\(prefix).linear_attn.in_proj_qkv")
  let inProjZ = Dense(
    count: valueDim, noBias: true, name: "\(prefix).linear_attn.in_proj_z")
  let inProjB = Dense(
    count: valueHeads, noBias: true, name: "\(prefix).linear_attn.in_proj_b")
  let inProjA = Dense(
    count: valueHeads, noBias: true, flags: [.Float16],
    name: "\(prefix).linear_attn.in_proj_a")
  let aScale = Parameter<Float>(
    .GPU(0), .C(valueHeads), initBound: 0, name: "\(prefix).linear_attn.A_log")
  let dtBias = Parameter<FloatType>(
    .GPU(0), .C(valueHeads), initBound: 0, name: "\(prefix).linear_attn.dt_bias")
  let mixedQKV = inProjQKV(x).reshaped([batchSize, tokenLength, 1, convDim])
  let convStateLength = configuration.linearConvKernel - 1
  let concat = Concat(axis: 1)
  concat.flags = [.disableOpt]
  let convContext = concat(convState, mixedQKV)
  let conv1d = Convolution(
    groups: convDim, filters: convDim, filterSize: [configuration.linearConvKernel, 1],
    noBias: true, format: .OIHW,
    name: "\(prefix).linear_attn.conv1d.weight")
  let convStateNext = convContext.reshaped(
    [batchSize, convStateLength, 1, convDim], offset: [0, tokenLength, 0, 0],
    strides: [(convStateLength + tokenLength) * convDim, convDim, convDim, 1]
  ).contiguous()
  let convPre = conv1d(convContext).reshaped([tokenLength, convDim])
  let convOut = convPre.swish()
  var query = convOut.reshaped(
    [tokenLength, keyDim], offset: [0, 0], strides: [convDim, 1]
  ).contiguous().reshaped([batchSize, tokenLength, keyHeads, keyHeadDim])
  var key = convOut.reshaped(
    [tokenLength, keyDim], offset: [0, keyDim], strides: [convDim, 1]
  ).contiguous().reshaped([batchSize, tokenLength, keyHeads, keyHeadDim])
  let value = convOut.reshaped(
    [tokenLength, valueDim], offset: [0, keyDim * 2], strides: [convDim, 1]
  ).contiguous().reshaped([batchSize, tokenLength, valueHeads, valueHeadDim])
  let qkNormEpsilon = 1e-6 / Float(keyHeadDim)
  let queryNorm = RMSNorm(
    epsilon: qkNormEpsilon, axis: [3], elementwiseAffine: false,
    scale: 1.0 / Float(keyHeadDim))
  let keyNorm = RMSNorm(
    epsilon: qkNormEpsilon, axis: [3], elementwiseAffine: false,
    scale: 1.0 / Float(keyHeadDim).squareRoot())
  key = keyNorm(key)
  query = queryNorm(query)
  let z = inProjZ(x).reshaped([
    batchSize, tokenLength, valueHeads, valueHeadDim,
  ])
  let beta = inProjB(x).reshaped([
    batchSize, tokenLength, valueHeads,
  ]).sigmoid()
  let a = inProjA(x).reshaped([batchSize, tokenLength, valueHeads])
  let dt = (a + dtBias.reshaped([1, 1, valueHeads])).softplus().to(.Float32)
  let g = Functional.mul(left: aScale.reshaped([1, 1, valueHeads]), right: dt, scalar: -1)
  let recurrent = GatedDelta(logDecay: true, name: "\(prefix).linear_attn.gated_delta")(
    queries: query, keys: key, values: value, decay: g, beta: beta,
    state: recurrentState)
  let recurrentOut = recurrent.output.reshaped([tokenLength, valueHeads, valueHeadDim])
  var out = recurrentOut.reshaped([batchSize * tokenLength * valueHeads, valueHeadDim])
  let norm = RMSNormGated(epsilon: 1e-6, axis: [1], name: "\(prefix).linear_attn.norm")
  let zGate = z.reshaped([batchSize * tokenLength * valueHeads, valueHeadDim])
  let gatedOut = norm(out, zGate)
  out = gatedOut.reshaped([batchSize * tokenLength, valueDim])
  let outProj = Dense(
    count: configuration.hiddenSize, noBias: true, name: "\(prefix).linear_attn.out_proj")
  out = outProj(out.to(of: x))
  let convStateOutput = convStateNext
  let recurrentStateOutput = recurrent.state
  return [out, convStateOutput, recurrentStateOutput]
}

private func Qwen3_5DecoderPost(
  prefix: String, x: Model.IO, configuration: Qwen3_5ModelConfiguration
) -> (out: Model.IO, normOut: Model.IO, feedForwardOut: Model.IO) {
  let norm = RMSNorm(epsilon: 1e-6, axis: [1], name: "\(prefix).post_attention_layernorm")
  let normOut = norm(x)
  let mlpGate = Dense(
    count: configuration.intermediateSize, noBias: true, name: "\(prefix).mlp.gate_proj")
  let mlpUp = Dense(
    count: configuration.intermediateSize, noBias: true, name: "\(prefix).mlp.up_proj")
  let mlpDown = Dense(
    count: configuration.hiddenSize, noBias: true, name: "\(prefix).mlp.down_proj")
  let gateOut = mlpGate(normOut)
  let upOut = mlpUp(normOut)
  let hidden = Functional.swishMul(value: upOut, gate: gateOut)
  let feedForwardOut = mlpDown(hidden)
  let out = x + feedForwardOut
  return (out, normOut, feedForwardOut)
}

public func Qwen3_5CausalLM<T: TensorNumeric>(
  _ dataType: T.Type, tokenLength: Int, cachedTokenLength: Int = 0,
  configuration: Qwen3_5ModelConfiguration, batchSize: Int = 1,
  outputHiddenStates: Bool = false, includeLogits: Bool = true, outputCacheStates: Bool = false,
  outputFinalState: Bool = true, tieEmbedding: Bool = false, injectEmbeddings: Bool = false
) -> Model {
  let tokens = Input()
  let tokenEmbed = Embedding(
    T.self, vocabularySize: configuration.vocabularySize, embeddingSize: configuration.hiddenSize,
    name: "model.language_model.embed_tokens")
  let textEmbedding: Model.IO
  var inputs: [Input] = [tokens]
  if injectEmbeddings {
    let tokenMask = Input()
    let injectedEmbeddings = Input()
    textEmbedding = tokenEmbed(tokens) .* tokenMask + injectedEmbeddings
    inputs.append(contentsOf: [tokenMask, injectedEmbeddings])
  } else {
    textEmbedding = tokenEmbed(tokens)
  }
  let hasFullAttentionLayer = (0..<configuration.layers).contains {
    !configuration.isLinearAttentionLayer($0)
  }
  let rotary = hasFullAttentionLayer ? Input() : nil
  var out: Model.IO = textEmbedding
  if let rotary = rotary {
    inputs.append(rotary)
  }
  var outputs = [Model.IO]()
  var cacheOutputs = [Model.IO]()
  for i in 0..<configuration.layers {
    let prefix = "model.language_model.layers.\(i)"
    let isLinearAttentionLayer = configuration.isLinearAttentionLayer(i)
    let residual = out
    let norm1 = RMSNorm(epsilon: 1e-6, axis: [1], name: "\(prefix).input_layernorm")
    let norm1Out = norm1(out)
    if isLinearAttentionLayer {
      let convState = Input()
      let recurrentState = Input()
      let linearOut = Qwen3_5LinearAttention(
        prefix: prefix, configuration: configuration, batchSize: batchSize,
        tokenLength: tokenLength, x: norm1Out, convState: convState,
        recurrentState: recurrentState, of: dataType)
      let afterMixer = linearOut[0].to(of: residual) + residual
      let post = Qwen3_5DecoderPost(
        prefix: prefix, x: afterMixer, configuration: configuration)
      out = post.out
      if outputCacheStates {
        cacheOutputs.append(contentsOf: [linearOut[1], linearOut[2]])
      }
      inputs.append(contentsOf: [convState, recurrentState])
    } else {
      let kIn = Input()
      let vIn = Input()
      let attention = Qwen3_5FullAttention(
        prefix: prefix, configuration: configuration, batchSize: batchSize,
        tokenLength: tokenLength, cachedTokenLength: cachedTokenLength,
        lastTokenOnly: i == configuration.layers - 1)
      let mixerOut = attention(norm1Out, rotary!, kIn, vIn)
      let residualOut =
        i == configuration.layers - 1
        ? residual.reshaped(
          [1, configuration.hiddenSize], offset: [tokenLength - 1, 0],
          strides: [configuration.hiddenSize, 1])
        : residual
      let afterMixer = mixerOut.to(of: residualOut) + residualOut
      let post = Qwen3_5DecoderPost(
        prefix: prefix, x: afterMixer, configuration: configuration)
      out = post.out
      inputs.append(contentsOf: [kIn, vIn])
    }
    if outputHiddenStates {
      outputs.append(out.to(T.dataType))
    }
  }
  if !outputFinalState && outputCacheStates {
    // Cache-only prefill still needs a live dependency on the final decoder state.
    outputs.append(out)
  }
  if outputFinalState {
    let finalNorm = RMSNorm(epsilon: 1e-6, axis: [1], name: "model.language_model.norm")
    var finalOut = finalNorm(out)
    if includeLogits {
      if tieEmbedding {
        finalOut = Matmul(transposeB: (0, 1))(finalOut, tokenEmbed.weight)
      } else {
        let lmHead = Dense(
          count: configuration.vocabularySize, noBias: true, name: "lm_head")
        finalOut = lmHead(finalOut)
      }
    }
    outputs.append(finalOut)
  }
  outputs.append(contentsOf: cacheOutputs)
  return Model(inputs, outputs)
}
