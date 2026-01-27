import NNC

private func ResBlock1(prefix: String, channels: Int, kernelSize: Int) -> (ModelWeightMapper, Model)
{
  let x = Input()
  var out: Model.IO = x
  var mappers = [ModelWeightMapper]()
  for (i, k) in [1, 3, 5].enumerated() {
    let residual = out
    out = out.leakyReLU(negativeSlope: 0.1)
    let conv1 = Convolution(
      groups: 1, filters: channels, filterSize: [1, kernelSize], dilation: [1, k],
      hint: Hint(
        stride: [1, 1],
        border: Hint.Border(
          begin: [0, (kernelSize - 1) * k / 2], end: [0, (kernelSize - 1) * k / 2])),
      name: "resnet_conv1")
    out = conv1(out)
    out = out.leakyReLU(negativeSlope: 0.1)
    let conv2 = Convolution(
      groups: 1, filters: channels, filterSize: [1, kernelSize],
      hint: Hint(
        stride: [1, 1],
        border: Hint.Border(begin: [0, (kernelSize - 1) / 2], end: [0, (kernelSize - 1) / 2])),
      name: "resnet_conv2")
    out = conv2(out) + residual
    let idx = i
    mappers.append { _ in
      var mapping = ModelWeightMapping()
      mapping["\(prefix).convs1.\(idx).weight"] = [conv1.weight.name]
      mapping["\(prefix).convs1.\(idx).bias"] = [conv1.bias.name]
      mapping["\(prefix).convs2.\(idx).weight"] = [conv2.weight.name]
      mapping["\(prefix).convs2.\(idx).bias"] = [conv2.bias.name]
      return mapping
    }
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    return mapping
  }
  return (mapper, Model([x], [out]))
}

func LTX2Vocoder(layers: [(channels: Int, kernelSize: Int, stride: Int, padding: Int)])
  -> (
    ModelWeightMapper, Model
  )
{
  let x = Input()
  let convPre = Convolution(
    groups: 1, filters: 1024, filterSize: [1, 7],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 3], end: [0, 3])), name: "conv_pre")
  var out = convPre(x)
  var mappers = [ModelWeightMapper]()
  for (i, layer) in layers.enumerated() {
    out = out.leakyReLU(negativeSlope: 0.1)
    let up = ConvolutionTranspose(
      groups: 1, filters: layer.channels, filterSize: [1, layer.kernelSize],
      hint: Hint(
        stride: [1, layer.stride],
        border: Hint.Border(begin: [0, layer.padding], end: [0, layer.padding])), name: "up")
    out = up(out)
    let upIdx = i
    mappers.append { _ in
      var mapping = ModelWeightMapping()
      mapping["ups.\(upIdx).weight"] = [up.weight.name]
      mapping["ups.\(upIdx).bias"] = [up.bias.name]
      return mapping
    }
    let resBlock1 = [
      ResBlock1(prefix: "resblocks.\(i * 3)", channels: layer.channels, kernelSize: 3),
      ResBlock1(prefix: "resblocks.\(i * 3 + 1)", channels: layer.channels, kernelSize: 7),
      ResBlock1(prefix: "resblocks.\(i * 3 + 2)", channels: layer.channels, kernelSize: 11),
    ]
    mappers.append(resBlock1[0].0)
    mappers.append(resBlock1[1].0)
    mappers.append(resBlock1[2].0)
    out = (1.0 / 3) * (resBlock1[0].1(out) + resBlock1[1].1(out) + resBlock1[2].1(out))
  }
  let convPost = Convolution(
    groups: 1, filters: 2, filterSize: [1, 7],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 3], end: [0, 3])), name: "conv_post")
  out = convPost(out.leakyReLU(negativeSlope: 0.01)).tanh()
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["conv_pre.weight"] = [convPre.weight.name]
    mapping["conv_pre.bias"] = [convPre.bias.name]
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping["conv_post.weight"] = [convPost.weight.name]
    mapping["conv_post.bias"] = [convPost.bias.name]
    return mapping
  }
  return (mapper, Model([x], [out]))
}
