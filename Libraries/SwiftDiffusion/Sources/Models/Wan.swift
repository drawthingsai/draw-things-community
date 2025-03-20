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
  prefix: String, k: Int, h: Int, b: Int, t: Int, hw: Int, intermediateSize: Int, injectImage: Bool
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
  var crossOut = crossAttention(cQ, cK, cV).reshaped([b, hw, k * h])
  var injectedImageKVs: [Model.IO]
  if injectImage {
    let cImgK = Input()
    let cImgV = Input()
    let crossAttentionImg = ScaledDotProductAttention(
      scale: 1 / Float(k).squareRoot(), flags: [.Float16])
    let crossOutImg = crossAttentionImg(cQ, cImgK, cImgV).reshaped([b, hw, k * h])
    crossOutImg.add(dependencies: [crossOut])
    crossOut = crossOut + crossOutImg
    injectedImageKVs = [cImgK, cImgV]
  } else {
    injectedImageKVs = []
  }
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
  return (mapper, Model([x, rot] + c + [cK, cV] + injectedImageKVs, [out]))
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

private func MLPProj(inChannels: Int, outChannels: Int, name: String) -> (
  Model, Model, Model, Model, Model
) {
  let x = Input()
  let ln1 = LayerNorm(epsilon: 1e-5, axis: [2], name: "\(name)_layer_norm_0")
  let fc0 = Dense(count: inChannels, name: "\(name)_embedder_0")
  var out = fc0(ln1(x)).GELU()
  let fc2 = Dense(count: outChannels, name: "\(name)_embedder_1")
  let ln2 = LayerNorm(epsilon: 1e-5, axis: [2], name: "\(name)_layer_norm_1")
  out = ln2(fc2(out))
  return (ln1, ln2, fc0, fc2, Model([x], [out]))
}

func Wan(
  channels: Int, layers: Int, intermediateSize: Int, time: Int, height: Int, width: Int,
  textLength: Int, injectImage: Bool
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
      intermediateSize: intermediateSize, injectImage: injectImage)
    let contextK = Input()
    let contextV = Input()
    contextIn.append(contentsOf: [contextK, contextV])
    if injectImage {
      let contextImgK = Input()
      let contextImgV = Input()
      out = block([out, rot] + tOut + [contextK, contextV, contextImgK, contextImgV])
      contextIn.append(contentsOf: [contextImgK, contextImgV])
    } else {
      out = block([out, rot] + tOut + [contextK, contextV])
    }
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
  prefix: String, k: Int, h: Int, b: Int, t: (Int, Int), injectImage: Bool
) -> (ModelWeightMapper, Model) {
  let context = Input()
  // Now run attention.
  let contextToKeys = Dense(count: k * h, name: "c_k")
  let contextToValues = Dense(count: k * h, name: "c_v")
  var cK = contextToKeys(context)
  let contextNormK = RMSNorm(epsilon: 1e-6, axis: [2], name: "c_norm_k")
  cK = contextNormK(cK).reshaped([b, t.0, h, k])
  let cV = contextToValues(context).reshaped([b, t.0, h, k])
  var ins = [context]
  var outs = [cK, cV]
  if injectImage {
    let contextImg = Input()
    let contextImgToKeys = Dense(count: k * h, name: "c_img_k")
    let contextImgToValues = Dense(count: k * h, name: "c_img_v")
    var cImgK = contextImgToKeys(contextImg)
    let contextImgNormK = RMSNorm(epsilon: 1e-6, axis: [2], name: "c_img_norm_k")
    cImgK = contextImgNormK(cImgK).reshaped([b, t.1, h, k])
    let cImgV = contextImgToValues(contextImg).reshaped([b, t.1, h, k])
    ins.append(contextImg)
    outs.append(contentsOf: [cImgK, cImgV])
  }
  let mapper: ModelWeightMapper = { _ in
    return ModelWeightMapping()
  }
  return (mapper, Model(ins, outs))
}

func WanFixed(
  timesteps: Int, batchSize: Int, channels: Int, layers: Int, textLength: Int, injectImage: Bool
) -> (
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
  var ins = [txt, t]
  let contextImg: Model.IO?
  if injectImage {
    let img = Input()
    let (clipLn1, clipLn2, clipMlp0, clipMlp2, clipIn) = MLPProj(
      inChannels: 1_280, outChannels: channels, name: "clip")
    contextImg = clipIn(img)
    ins.append(img)
  } else {
    contextImg = nil
  }
  var mappers = [ModelWeightMapper]()
  for i in 0..<layers {
    let (mapper, block) = WanAttentionBlockFixed(
      prefix: "blocks.\(i)", k: 128, h: channels / 128, b: batchSize, t: (textLength, 257),
      injectImage: injectImage)
    if let contextImg = contextImg {
      outs.append(block(context, contextImg))
    } else {
      outs.append(block(context))
    }
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
  return (mapper, Model(ins, outs))
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
  prefix: String, k: Int, h: Int, b: Int, t: Int, hw: Int, intermediateSize: Int, injectImage: Bool,
  layerIndex: Int,
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
  let xToContextQueries = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "x_c_q")
  let cK = Input()
  var cQ = xToContextQueries(xOut)
  let contextNormQ = RMSNorm(epsilon: 1e-6, axis: [2], name: "x_c_norm_q")
  cQ = contextNormQ(cQ).reshaped([b, hw, h, k])
  let cV = Input()
  let crossAttention = ScaledDotProductAttention(
    scale: 1 / Float(k).squareRoot(), flags: [.Float16])
  var crossOut = crossAttention(cQ, cK, cV).reshaped([b, hw, k * h])
  var injectedImageKVs: [Model.IO]
  if injectImage {
    let cImgK = Input()
    let cImgV = Input()
    let crossAttentionImg = ScaledDotProductAttention(
      scale: 1 / Float(k).squareRoot(), flags: [.Float16])
    let crossOutImg = crossAttentionImg(cQ, cImgK, cImgV).reshaped([b, hw, k * h])
    crossOutImg.add(dependencies: [crossOut])
    crossOut = crossOut + crossOutImg
    injectedImageKVs = [cImgK, cImgV]
  } else {
    injectedImageKVs = []
  }
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
  return (mapper, Model([x, rot] + c + [cK, cV] + injectedImageKVs, [out]))
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

private func LoRAMLPProj(
  inChannels: Int, outChannels: Int, configuration: LoRANetworkConfiguration, name: String
) -> (Model, Model, Model, Model, Model) {
  let x = Input()
  let ln1 = LayerNorm(epsilon: 1e-5, axis: [2], name: "\(name)_layer_norm_0")
  let fc0 = LoRADense(count: inChannels, configuration: configuration, name: "\(name)_embedder_0")
  var out = fc0(ln1(x)).GELU()
  let fc2 = LoRADense(count: outChannels, configuration: configuration, name: "\(name)_embedder_1")
  let ln2 = LayerNorm(epsilon: 1e-5, axis: [2], name: "\(name)_layer_norm_1")
  out = ln2(fc2(out))
  return (ln1, ln2, fc0, fc2, Model([x], [out]))
}

func LoRAWan(
  channels: Int, layers: Int, intermediateSize: Int, time: Int, height: Int, width: Int,
  textLength: Int, injectImage: Bool, LoRAConfiguration: LoRANetworkConfiguration
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
      intermediateSize: intermediateSize, injectImage: injectImage, layerIndex: i,
      configuration: LoRAConfiguration)
    let contextK = Input()
    let contextV = Input()
    contextIn.append(contentsOf: [contextK, contextV])
    if injectImage {
      let contextImgK = Input()
      let contextImgV = Input()
      out = block([out, rot] + tOut + [contextK, contextV, contextImgK, contextImgV])
      contextIn.append(contentsOf: [contextImgK, contextImgV])
    } else {
      out = block([out, rot] + tOut + [contextK, contextV])
    }
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
  prefix: String, k: Int, h: Int, b: Int, t: (Int, Int), injectImage: Bool, layerIndex: Int,
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
  cK = contextNormK(cK).reshaped([b, t.0, h, k])
  let cV = contextToValues(context).reshaped([b, t.0, h, k])
  var ins = [context]
  var outs = [cK, cV]
  if injectImage {
    let contextImg = Input()
    let contextImgToKeys = LoRADense(
      count: k * h, configuration: configuration, index: layerIndex, name: "c_img_k")
    let contextImgToValues = LoRADense(
      count: k * h, configuration: configuration, index: layerIndex, name: "c_img_v")
    var cImgK = contextImgToKeys(contextImg)
    let contextImgNormK = RMSNorm(epsilon: 1e-6, axis: [2], name: "c_img_norm_k")
    cImgK = contextImgNormK(cImgK).reshaped([b, t.1, h, k])
    let cImgV = contextImgToValues(contextImg).reshaped([b, t.1, h, k])
    ins.append(contextImg)
    outs.append(contentsOf: [cImgK, cImgV])
  }
  let mapper: ModelWeightMapper = { _ in
    return ModelWeightMapping()
  }
  return (mapper, Model(ins, outs))
}

func LoRAWanFixed(
  timesteps: Int, batchSize: Int, channels: Int, layers: Int, textLength: Int, injectImage: Bool,
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
  var ins = [txt, t]
  let contextImg: Model.IO?
  if injectImage {
    let img = Input()
    let (clipLn1, clipLn2, clipMlp0, clipMlp2, clipIn) = LoRAMLPProj(
      inChannels: 1_280, outChannels: channels, configuration: LoRAConfiguration, name: "clip")
    contextImg = clipIn(img)
    ins.append(img)
  } else {
    contextImg = nil
  }
  var mappers = [ModelWeightMapper]()
  for i in 0..<layers {
    let (mapper, block) = LoRAWanAttentionBlockFixed(
      prefix: "blocks.\(i)", k: 128, h: channels / 128, b: batchSize, t: (textLength, 257),
      injectImage: injectImage, layerIndex: i, configuration: LoRAConfiguration)
    if let contextImg = contextImg {
      outs.append(block(context, contextImg))
    } else {
      outs.append(block(context))
    }
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
  return (mapper, Model(ins, outs))
}
