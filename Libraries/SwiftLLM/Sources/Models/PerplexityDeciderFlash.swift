import Foundation
import NNC

private func PerplexityDeciderFlashFullAttention(
  prefix: String, configuration: Qwen3_5ModelConfiguration, tokenLength: Int
) -> Model {
  let x = Input()
  let rotary = Input()
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
  var queries = toqueries(x).reshaped([1, tokenLength, heads, headDim], format: .NHWC)
  let gate = toqueryGates(x)
  var keys = tokeys(x).reshaped([1, tokenLength, keyValueHeads, headDim], format: .NHWC)
  let values = tovalues(x).reshaped([1, tokenLength, keyValueHeads, headDim], format: .NHWC)
  let qNorm = RMSNorm(epsilon: 1e-6, axis: [3], name: "\(prefix).self_attn.q_norm")
  queries = qNorm(queries)
  let kNorm = RMSNorm(epsilon: 1e-6, axis: [3], name: "\(prefix).self_attn.k_norm")
  keys = kNorm(keys)
  queries = Functional.cmul(left: queries, right: rotary)
  keys = Functional.cmul(left: keys, right: rotary)
  // Noncausal: every token attends to the whole request, so the state also reads the questions after it.
  var out = ScaledDotProductAttention(
    scale: 1.0 / Float(headDim).squareRoot(), isCausal: false, flags: [.Float16])(
      queries, keys, values)
  out = out.reshaped([tokenLength, queryDim])
  out = out .* gate.sigmoid()
  let outProj = Dense(
    count: configuration.hiddenSize, noBias: true, name: "\(prefix).self_attn.o_proj")
  out = outProj(out)
  return Model([x, rotary], [out])
}

/// pplx-decider-flash: the Qwen3.5 text model with noncausal full-attention layers, and a readout head applied
/// to the final hidden state at each question's marker token. The readout scores `optionCount` answer codes; a
/// question with n options uses the first n.
///
/// Inputs are the tokens, the marker positions (Int32), the rotary embedding, then a zero conv state and a zero
/// recurrent state for each linear-attention layer, which stay causal. Outputs are the readout logits, one row per
/// marker, then the linear-attention states, which are output only so that no node in the graph dangles.
public func PerplexityDeciderFlash<T: TensorNumeric>(
  _ dataType: T.Type, tokenLength: Int, configuration: Qwen3_5ModelConfiguration,
  optionCount: Int = 255
) -> Model {
  let tokens = Input()
  let markers = Input()
  let rotary = Input()
  let tokenEmbed = Embedding(
    T.self, vocabularySize: configuration.vocabularySize, embeddingSize: configuration.hiddenSize,
    name: "language_model.embed_tokens")
  var out = tokenEmbed(tokens)
  var inputs: [Input] = [tokens, markers, rotary]
  var stateOutputs = [Model.IO]()
  for i in 0..<configuration.layers {
    let prefix = "language_model.layers.\(i)"
    let residual = out
    let norm1 = RMSNorm(epsilon: 1e-6, axis: [1], name: "\(prefix).input_layernorm")
    let norm1Out = norm1(out)
    let mixerOut: Model.IO
    if configuration.isLinearAttentionLayer(i) {
      let convState = Input()
      let recurrentState = Input()
      let linearOut = Qwen3_5LinearAttention(
        prefix: prefix, configuration: configuration, batchSize: 1, tokenLength: tokenLength,
        x: norm1Out, convState: convState, recurrentState: recurrentState, of: dataType)
      mixerOut = linearOut[0]
      stateOutputs.append(contentsOf: [linearOut[1], linearOut[2]])
      inputs.append(contentsOf: [convState, recurrentState])
    } else {
      let attention = PerplexityDeciderFlashFullAttention(
        prefix: prefix, configuration: configuration, tokenLength: tokenLength)
      mixerOut = attention(norm1Out, rotary)
    }
    out = Qwen3_5DecoderPost(
      prefix: prefix, x: mixerOut.to(of: residual) + residual, configuration: configuration)
  }
  let finalNorm = RMSNorm(epsilon: 1e-6, axis: [1], name: "language_model.norm")
  let readout = Dense(count: optionCount, noBias: true, name: "readout")
  let logits = readout(finalNorm(Functional.indexSelect(input: out, index: markers)))
  return Model(inputs, [logits] + stateOutputs)
}
