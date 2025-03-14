import Foundation
import NNC

public func WanRotaryPositionEmbedding(
  height: Int, width: Int, time: Int, channels: Int, heads: Int = 1
)
  -> Tensor<Float>
{
  var rotTensor = Tensor<Float>(.CPU, .NHWC(1, time * height * width, heads, channels))
  let dim1 = (channels / 6) * 2
  let dim2 = dim1
  let dim0 = channels - dim1 - dim2
  assert(channels % 16 == 0)
  for t in 0..<time {
    for y in 0..<height {
      for x in 0..<width {
        let i = t * height * width + y * width + x
        for j in 0..<heads {
          for k in 0..<(dim0 / 2) {
            let theta = Double(t) * 1.0 / pow(10_000, Double(k) * 2 / Double(dim0))
            let sintheta = sin(theta)
            let costheta = cos(theta)
            rotTensor[0, i, j, k * 2] = Float(costheta)
            rotTensor[0, i, j, k * 2 + 1] = Float(sintheta)
          }
          for k in 0..<(dim1 / 2) {
            let theta = Double(y) * 1.0 / pow(10_000, Double(k) * 2 / Double(dim1))
            let sintheta = sin(theta)
            let costheta = cos(theta)
            rotTensor[0, i, j, (k + (dim0 / 2)) * 2] = Float(costheta)
            rotTensor[0, i, j, (k + (dim0 / 2)) * 2 + 1] = Float(sintheta)
          }
          for k in 0..<(dim2 / 2) {
            let theta = Double(x) * 1.0 / pow(10_000, Double(k) * 2 / Double(dim2))
            let sintheta = sin(theta)
            let costheta = cos(theta)
            rotTensor[0, i, j, (k + (dim0 / 2) + (dim1 / 2)) * 2] = Float(costheta)
            rotTensor[0, i, j, (k + (dim0 / 2) + (dim1 / 2)) * 2 + 1] = Float(sintheta)
          }
        }
      }
    }
  }
  return rotTensor
}

private func FeedForward(hiddenSize: Int, intermediateSize: Int, upcast: Bool, name: String) -> (
  Model, Model, Model
) {
  let x = Input()
  let linear1 = Dense(count: intermediateSize, name: "\(name)_linear1")
  var out = linear1(x).GELU(approximate: .tanh)
  // The scale down is integrated into out proj bias.
  if upcast {
    let scaleFactor: Float = 8
    out = (1 / scaleFactor) * out
  }
  let outProjection = Dense(count: hiddenSize, name: "\(name)_out_proj")
  out = outProjection(out)
  if upcast {
    let scaleFactor: Float = 8
    out = out.to(.Float32) * scaleFactor
  }
  return (linear1, outProjection, Model([x], [out]))
}

private func WanAttentionBlock(
  prefix: String, k: Int, h: Int, b: Int, t: Int, hw: Int, intermediateSize: Int
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let rot = Input()
  let c = (0..<6).map { _ in Input() }
  let modulations = (0..<6).map {
    Parameter<Float>(.GPU(0), .HWC(1, 1, k * h), name: "attn_ada_ln_\($0)")
  }
  let chunks = zip(c, modulations).map { $0 + $1 }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = ((1 + chunks[1]) .* xNorm1(x) + chunks[0]).to(.Float16)
  let xToKeys = Dense(count: k * h, name: "x_k")
  let xToQueries = Dense(count: k * h, name: "x_q")
  let xToValues = Dense(count: k * h, name: "x_v")
  var xK = xToKeys(xOut)
  let normK = RMSNorm(epsilon: 1e-6, axis: [2], name: "x_norm_k")
  xK = normK(xK).reshaped([b, hw, h, k])
  var xQ = xToQueries(xOut)
  let normQ = RMSNorm(epsilon: 1e-6, axis: [2], name: "x_norm_q")
  xQ = normQ(xQ).reshaped([b, hw, h, k])
  let xV = xToValues(xOut).reshaped([b, hw, h, k])
  let queries = (1 / Float(k).squareRoot().squareRoot()) * Functional.cmul(left: xQ, right: rot)
  let keys = (1 / Float(k).squareRoot().squareRoot()) * Functional.cmul(left: xK, right: rot)
  let values = xV
  // Now run attention.
  let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
  var out = scaledDotProductAttention(queries, keys, values).reshaped([b, hw, k * h])
  let xUnifyheads = Dense(count: k * h, name: "x_o")
  out = xUnifyheads(out)
  out = x + chunks[2] .* out.to(of: x)
  let xNorm3 = LayerNorm(epsilon: 1e-6, axis: [2], name: "x_norm_3")
  xOut = xNorm3(out).to(.Float16)
  let xToContextQueries = Dense(count: k * h, name: "x_c_q")
  let cK = Input()
  var cQ = xToContextQueries(xOut)
  let contextNormQ = RMSNorm(epsilon: 1e-6, axis: [2], name: "x_c_norm_q")
  cQ = contextNormQ(cQ).reshaped([b, hw, h, k])
  let cV = Input()
  let crossAttention = ScaledDotProductAttention(
    scale: 1 / Float(k).squareRoot(), flags: [.Float16])
  let crossOut = crossAttention(cQ, cK, cV).reshaped([b, hw, k * h])
  let contextUnifyheads = Dense(count: k * h, name: "c_o")
  out = out + contextUnifyheads(crossOut).to(of: out)
  let xNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let (xLinear1, xOutProjection, xFF) = FeedForward(
    hiddenSize: k * h, intermediateSize: intermediateSize, upcast: false, name: "x")
  out =
    out + xFF(((1 + chunks[4]) .* xNorm2(out) + chunks[3]).to(.Float16)).to(of: out) .* chunks[5]
  let mapper: ModelWeightMapper = { _ in
    return ModelWeightMapping()
  }
  return (mapper, Model([x, rot] + c + [cK, cV], [out]))
}

private func TimeEmbedder(channels: Int, name: String) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: channels, name: "\(name)_embedder_0")
  var out = fc0(x).swish()
  let fc2 = Dense(count: channels, name: "\(name)_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

private func MLPEmbedder(channels: Int, name: String) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: channels, name: "\(name)_embedder_0")
  var out = fc0(x).GELU(approximate: .tanh)
  let fc2 = Dense(count: channels, name: "\(name)_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

func Wan(
  channels: Int, layers: Int, intermediateSize: Int, time: Int, height: Int, width: Int,
  textLength: Int
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let imgIn = Convolution(
    groups: 1, filters: channels, filterSize: [2, 2],
    hint: Hint(stride: [2, 2]), format: .OIHW, name: "x_embedder")
  let rot = Input()
  let tOut = (0..<6).map { _ in Input() }
  let h = height / 2
  let w = width / 2
  var mappers = [ModelWeightMapper]()
  var out = imgIn(x).reshaped([1, time * h * w, channels]).to(.Float32)
  var contextIn = [Input]()
  for i in 0..<layers {
    let (mapper, block) = WanAttentionBlock(
      prefix: "blocks.\(i)", k: 128, h: channels / 128, b: 1, t: textLength, hw: time * h * w,
      intermediateSize: intermediateSize)
    let contextK = Input()
    let contextV = Input()
    out = block([out, rot] + tOut + [contextK, contextV])
    contextIn.append(contentsOf: [contextK, contextV])
    mappers.append(mapper)
  }
  let scale = Input()
  let shift = Input()
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  out = (scale .* normFinal(out) + shift).to(.Float16)
  let projOut = Dense(count: 2 * 2 * 16, name: "linear")
  out = projOut(out).reshaped([time, h, w, 2, 2, 16]).permuted(0, 1, 3, 2, 4, 5).contiguous()
    .reshaped([
      time, h * 2, w * 2, 16,
    ])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    return mapping
  }
  return (mapper, Model([x, rot] + tOut + contextIn + [scale, shift], [out]))
}

private func WanAttentionBlockFixed(
  prefix: String, k: Int, h: Int, b: Int, t: Int
) -> (ModelWeightMapper, Model) {
  let context = Input()
  // Now run attention.
  let contextToKeys = Dense(count: k * h, name: "c_k")
  let contextToValues = Dense(count: k * h, name: "c_v")
  var cK = contextToKeys(context)
  let contextNormK = RMSNorm(epsilon: 1e-6, axis: [2], name: "c_norm_k")
  cK = contextNormK(cK).reshaped([b, t, h, k])
  let cV = contextToValues(context).reshaped([b, t, h, k])
  let mapper: ModelWeightMapper = { _ in
    return ModelWeightMapping()
  }
  return (mapper, Model([context], [cK, cV]))
}

func WanFixed(timesteps: Int, batchSize: Int, channels: Int, layers: Int, textLength: Int) -> (
  ModelWeightMapper, Model
) {
  let txt = Input()
  let (cLinear1, cLinear2, contextEmbedder) = MLPEmbedder(channels: channels, name: "c")
  let context = contextEmbedder(txt)
  let t = Input()
  let (timeInMlp0, timeInMlp2, timeIn) = TimeEmbedder(channels: channels, name: "t")
  let vector = timeIn(t).reshaped([timesteps, 1, channels])
  let vectorIn = vector.swish()
  let timeProjections = (0..<6).map { Dense(count: channels, name: "ada_ln_\($0)") }
  var outs = timeProjections.map { $0(vectorIn).identity().identity().identity() }  // Have duplicate name ada_ln_0 / ada_ln_1, now have to push the order to make sure the proper weights are loaded :(.
  var mappers = [ModelWeightMapper]()
  for i in 0..<layers {
    let (mapper, block) = WanAttentionBlockFixed(
      prefix: "blocks.\(i)", k: 128, h: channels / 128, b: batchSize, t: textLength)
    outs.append(block(context))
    mappers.append(mapper)
  }
  let scale = Parameter<Float>(.GPU(0), .HWC(1, 1, channels), name: "ada_ln_0")
  let shift = Parameter<Float>(.GPU(0), .HWC(1, 1, channels), name: "ada_ln_1")
  outs.append(1 + scale + vector)
  outs.append(vector + shift)
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    return mapping
  }
  return (mapper, Model([txt, t], outs))
}

private func LoRAFeedForward(
  hiddenSize: Int, intermediateSize: Int, upcast: Bool, configuration: LoRANetworkConfiguration,
  index: Int, name: String
) -> (
  Model, Model, Model
) {
  let x = Input()
  let linear1 = LoRADense(
    count: intermediateSize, configuration: configuration, index: index, name: "\(name)_linear1")
  var out = linear1(x).GELU(approximate: .tanh)
  // The scale down is integrated into out proj bias.
  if upcast {
    let scaleFactor: Float = 8
    out = (1 / scaleFactor) * out
  }
  let outProjection = LoRADense(
    count: hiddenSize, configuration: configuration, index: index, name: "\(name)_out_proj")
  out = outProjection(out)
  if upcast {
    let scaleFactor: Float = 8
    out = out.to(.Float32) * scaleFactor
  }
  return (linear1, outProjection, Model([x], [out]))
}

private func LoRAWanAttentionBlock(
  prefix: String, k: Int, h: Int, b: Int, t: Int, hw: Int, intermediateSize: Int, layerIndex: Int,
  configuration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let rot = Input()
  let c = (0..<6).map { _ in Input() }
  let modulations = (0..<6).map {
    Parameter<Float>(.GPU(0), .HWC(1, 1, k * h), name: "attn_ada_ln_\($0)")
  }
  let chunks = zip(c, modulations).map { $0 + $1 }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = ((1 + chunks[1]) .* xNorm1(x) + chunks[0]).to(.Float16)
  let xToKeys = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "x_k")
  let xToQueries = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "x_q")
  let xToValues = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "x_v")
  var xK = xToKeys(xOut)
  let normK = RMSNorm(epsilon: 1e-6, axis: [2], name: "x_norm_k")
  xK = normK(xK).reshaped([b, hw, h, k])
  var xQ = xToQueries(xOut)
  let normQ = RMSNorm(epsilon: 1e-6, axis: [2], name: "x_norm_q")
  xQ = normQ(xQ).reshaped([b, hw, h, k])
  let xV = xToValues(xOut).reshaped([b, hw, h, k])
  let queries = (1 / Float(k).squareRoot().squareRoot()) * Functional.cmul(left: xQ, right: rot)
  let keys = (1 / Float(k).squareRoot().squareRoot()) * Functional.cmul(left: xK, right: rot)
  let values = xV
  // Now run attention.
  let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
  var out = scaledDotProductAttention(queries, keys, values).reshaped([b, hw, k * h])
  let xUnifyheads = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "x_o")
  out = xUnifyheads(out)
  out = x + chunks[2] .* out.to(of: x)
  let xNorm3 = LayerNorm(epsilon: 1e-6, axis: [2], name: "x_norm_3")
  xOut = xNorm3(out).to(.Float16)
  let xToContextQueries = Dense(count: k * h, name: "x_c_q")
  let cK = Input()
  var cQ = xToContextQueries(xOut)
  let contextNormQ = RMSNorm(epsilon: 1e-6, axis: [2], name: "x_c_norm_q")
  cQ = contextNormQ(cQ).reshaped([b, hw, h, k])
  let cV = Input()
  let crossAttention = ScaledDotProductAttention(
    scale: 1 / Float(k).squareRoot(), flags: [.Float16])
  let crossOut = crossAttention(cQ, cK, cV).reshaped([b, hw, k * h])
  let contextUnifyheads = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "c_o")
  out = out + contextUnifyheads(crossOut).to(of: out)
  let xNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let (xLinear1, xOutProjection, xFF) = LoRAFeedForward(
    hiddenSize: k * h, intermediateSize: intermediateSize, upcast: false,
    configuration: configuration, index: layerIndex, name: "x")
  out =
    out + xFF(((1 + chunks[4]) .* xNorm2(out) + chunks[3]).to(.Float16)).to(of: out) .* chunks[5]
  let mapper: ModelWeightMapper = { _ in
    return ModelWeightMapping()
  }
  return (mapper, Model([x, rot] + c + [cK, cV], [out]))
}

private func LoRATimeEmbedder(channels: Int, configuration: LoRANetworkConfiguration, name: String)
  -> (Model, Model, Model)
{
  let x = Input()
  let fc0 = LoRADense(count: channels, configuration: configuration, name: "\(name)_embedder_0")
  var out = fc0(x).swish()
  let fc2 = LoRADense(count: channels, configuration: configuration, name: "\(name)_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

private func LoRAMLPEmbedder(channels: Int, configuration: LoRANetworkConfiguration, name: String)
  -> (Model, Model, Model)
{
  let x = Input()
  let fc0 = LoRADense(count: channels, configuration: configuration, name: "\(name)_embedder_0")
  var out = fc0(x).GELU(approximate: .tanh)
  let fc2 = LoRADense(count: channels, configuration: configuration, name: "\(name)_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

func LoRAWan(
  channels: Int, layers: Int, intermediateSize: Int, time: Int, height: Int, width: Int,
  textLength: Int, LoRAConfiguration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let imgIn = LoRAConvolution(
    groups: 1, filters: channels, filterSize: [2, 2], configuration: LoRAConfiguration,
    hint: Hint(stride: [2, 2]), format: .OIHW, name: "x_embedder")
  let rot = Input()
  let tOut = (0..<6).map { _ in Input() }
  let h = height / 2
  let w = width / 2
  var mappers = [ModelWeightMapper]()
  var out = imgIn(x).reshaped([1, time * h * w, channels]).to(.Float32)
  var contextIn = [Input]()
  for i in 0..<layers {
    let (mapper, block) = LoRAWanAttentionBlock(
      prefix: "blocks.\(i)", k: 128, h: channels / 128, b: 1, t: textLength, hw: time * h * w,
      intermediateSize: intermediateSize, layerIndex: i, configuration: LoRAConfiguration)
    let contextK = Input()
    let contextV = Input()
    out = block([out, rot] + tOut + [contextK, contextV])
    contextIn.append(contentsOf: [contextK, contextV])
    mappers.append(mapper)
  }
  let scale = Input()
  let shift = Input()
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  out = (scale .* normFinal(out) + shift).to(.Float16)
  let projOut = LoRADense(
    count: 2 * 2 * 16, configuration: LoRAConfiguration, index: 0, name: "linear")
  out = projOut(out).reshaped([time, h, w, 2, 2, 16]).permuted(0, 1, 3, 2, 4, 5).contiguous()
    .reshaped([
      time, h * 2, w * 2, 16,
    ])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    return mapping
  }
  return (mapper, Model([x, rot] + tOut + contextIn + [scale, shift], [out]))
}

private func LoRAWanAttentionBlockFixed(
  prefix: String, k: Int, h: Int, b: Int, t: Int, layerIndex: Int,
  configuration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let context = Input()
  // Now run attention.
  let contextToKeys = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "c_k")
  let contextToValues = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "c_v")
  var cK = contextToKeys(context)
  let contextNormK = RMSNorm(epsilon: 1e-6, axis: [2], name: "c_norm_k")
  cK = contextNormK(cK).reshaped([b, t, h, k])
  let cV = contextToValues(context).reshaped([b, t, h, k])
  let mapper: ModelWeightMapper = { _ in
    return ModelWeightMapping()
  }
  return (mapper, Model([context], [cK, cV]))
}

func LoRAWanFixed(
  timesteps: Int, batchSize: Int, channels: Int, layers: Int, textLength: Int,
  LoRAConfiguration: LoRANetworkConfiguration
) -> (
  ModelWeightMapper, Model
) {
  let txt = Input()
  let (cLinear1, cLinear2, contextEmbedder) = LoRAMLPEmbedder(
    channels: channels, configuration: LoRAConfiguration, name: "c")
  let context = contextEmbedder(txt)
  let t = Input()
  let (timeInMlp0, timeInMlp2, timeIn) = LoRATimeEmbedder(
    channels: channels, configuration: LoRAConfiguration, name: "t")
  let vector = timeIn(t).reshaped([timesteps, 1, channels])
  let vectorIn = vector.swish()
  let timeProjections = (0..<6).map {
    LoRADense(count: channels, configuration: LoRAConfiguration, name: "ada_ln_\($0)")
  }
  var outs = timeProjections.map { $0(vectorIn).identity().identity().identity() }  // Have duplicate name ada_ln_0 / ada_ln_1, now have to push the order to make sure the proper weights are loaded :(.
  var mappers = [ModelWeightMapper]()
  for i in 0..<layers {
    let (mapper, block) = LoRAWanAttentionBlockFixed(
      prefix: "blocks.\(i)", k: 128, h: channels / 128, b: batchSize, t: textLength, layerIndex: i,
      configuration: LoRAConfiguration)
    outs.append(block(context))
    mappers.append(mapper)
  }
  let scale = Parameter<Float>(.GPU(0), .HWC(1, 1, channels), name: "ada_ln_0")
  let shift = Parameter<Float>(.GPU(0), .HWC(1, 1, channels), name: "ada_ln_1")
  outs.append(1 + scale + vector)
  outs.append(vector + shift)
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    return mapping
  }
  return (mapper, Model([txt, t], outs))
}
