import DiffusionMappings
import NNC

enum LTX2VocoderActivation: Equatable {
  case leakyReLU
  case activation1d
}

enum LTX2VocoderFinalActivation: Equatable {
  case clamp
  case tanh
}

private func SnakeBeta(prefix: String, channels: Int) -> (ModelWeightMapper, Model) {
  let x = Input()
  let alpha = Parameter<Float>(
    .GPU(0), .NCHW(1, channels, 1, 1), trainable: false, name: "snake_alpha")
  let beta = Parameter<Float>(
    .GPU(0), .NCHW(1, channels, 1, 1), trainable: false, name: "snake_beta")
  let out = x + beta .* (x .* alpha).sin().pow(2)
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).alpha"] = [alpha.weight.name]
    mapping["\(prefix).beta"] = [beta.weight.name]
    return mapping
  }
  return (mapper, Model([x], [out]))
}

private func Activation1d(prefix: String, channels: Int, width: Int) -> (ModelWeightMapper, Model) {
  let x = Input()
  let upWidth = width * 2
  let upPad = 12 / 2 - 1
  let upInputWidth = width + 2 * upPad
  let upRawWidth = (upInputWidth - 1) * 2 + 12
  let upPadLeft = upPad * 2 + (12 - 2) / 2
  let upsample = ConvolutionTranspose(
    groups: 1, filters: 1, filterSize: [1, 12], noBias: true, hint: Hint(stride: [1, 2]),
    format: .OIHW, name: "act_upsample")
  let (snakeMapper, snake) = SnakeBeta(prefix: "\(prefix).act", channels: channels)
  let downsample = Convolution(
    groups: 1, filters: 1, filterSize: [1, 12], noBias: true, hint: Hint(stride: [1, 2]),
    format: .OIHW, name: "act_downsample")
  var out = x.reshaped([channels, 1, 1, width])
  out = out.padded(.replicate, begin: [0, 0, 0, upPad], end: [0, 0, 0, upPad])
  out = Float(2) * upsample(out)
  out = out.reshaped(
    [channels, 1, 1, upWidth], offset: [0, 0, 0, upPadLeft],
    strides: [upRawWidth, upRawWidth, upRawWidth, 1]
  ).contiguous()
  out = out.reshaped([1, channels, 1, upWidth])
  out = snake(out)
  out = out.reshaped([channels, 1, 1, upWidth])
  out = downsample(out.padded(.replicate, begin: [0, 0, 0, 5], end: [0, 0, 0, 6]))
  out = out.reshaped([1, channels, 1, width])
  let mapper: ModelWeightMapper = { format in
    var mapping = snakeMapper(format)
    mapping["\(prefix).upsample.filter"] = [upsample.weight.name]
    mapping["\(prefix).downsample.lowpass.filter"] = [downsample.weight.name]
    return mapping
  }
  return (mapper, Model([x], [out]))
}

private func ResBlock1(
  prefix: String, channels: Int, width: Int, kernelSize: Int, dilations: [Int],
  activation: LTX2VocoderActivation
) -> (ModelWeightMapper, Model) {
  let x = Input()
  var out: Model.IO = x
  var mappers = [ModelWeightMapper]()
  for (i, dilation) in dilations.enumerated() {
    let residual = out
    switch activation {
    case .leakyReLU:
      out = out.leakyReLU(negativeSlope: 0.1)
    case .activation1d:
      let (mapper, activation1d) = Activation1d(
        prefix: "\(prefix).acts1.\(i)", channels: channels, width: width)
      out = activation1d(out)
      mappers.append(mapper)
    }
    let conv1 = Convolution(
      groups: 1, filters: channels, filterSize: [1, kernelSize], dilation: [1, dilation],
      hint: Hint(
        stride: [1, 1],
        border: Hint.Border(
          begin: [0, (kernelSize - 1) * dilation / 2], end: [0, (kernelSize - 1) * dilation / 2])),
      format: .OIHW, name: "resnet_conv1")
    out = conv1(out)
    switch activation {
    case .leakyReLU:
      out = out.leakyReLU(negativeSlope: 0.1)
    case .activation1d:
      let (mapper, activation1d) = Activation1d(
        prefix: "\(prefix).acts2.\(i)", channels: channels, width: width)
      out = activation1d(out)
      mappers.append(mapper)
    }
    let conv2 = Convolution(
      groups: 1, filters: channels, filterSize: [1, kernelSize],
      hint: Hint(
        stride: [1, 1],
        border: Hint.Border(begin: [0, (kernelSize - 1) / 2], end: [0, (kernelSize - 1) / 2])),
      format: .OIHW, name: "resnet_conv2")
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

func LTX2Vocoder(
  width: Int, initialChannels: Int,
  layers: [(channels: Int, kernelSize: Int, stride: Int, padding: Int)], resblockKernelSizes: [Int],
  resblockDilations: [[Int]], activation: LTX2VocoderActivation, finalConvBias: Bool,
  finalActivation: LTX2VocoderFinalActivation
)
  -> (
    ModelWeightMapper, Model
  )
{
  precondition(!resblockKernelSizes.isEmpty)
  precondition(resblockKernelSizes.count == resblockDilations.count)
  let x = Input()
  let convPre = Convolution(
    groups: 1, filters: initialChannels, filterSize: [1, 7],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 3], end: [0, 3])), format: .OIHW,
    name: "conv_pre")
  var out = convPre(x)
  var currentWidth = width
  var mappers = [ModelWeightMapper]()
  for (i, layer) in layers.enumerated() {
    if activation == .leakyReLU {
      out = out.leakyReLU(negativeSlope: 0.1)
    }
    let up = ConvolutionTranspose(
      groups: 1, filters: layer.channels, filterSize: [1, layer.kernelSize],
      hint: Hint(
        stride: [1, layer.stride],
        border: Hint.Border(begin: [0, layer.padding], end: [0, layer.padding])), format: .OIHW,
      name: "up")
    out = up(out)
    currentWidth *= layer.stride
    let upIdx = i
    mappers.append { _ in
      var mapping = ModelWeightMapping()
      mapping["ups.\(upIdx).weight"] = [up.weight.name]
      mapping["ups.\(upIdx).bias"] = [up.bias.name]
      return mapping
    }
    let resBlocks = resblockKernelSizes.enumerated().map {
      ResBlock1(
        prefix: "resblocks.\(i * resblockKernelSizes.count + $0.offset)", channels: layer.channels,
        width: currentWidth, kernelSize: $0.element, dilations: resblockDilations[$0.offset],
        activation: activation)
    }
    for resBlock in resBlocks {
      mappers.append(resBlock.0)
    }
    out =
      (1.0 / Float(resBlocks.count))
      * resBlocks.dropFirst().reduce(resBlocks[0].1(out)) { $0 + $1.1(out) }
  }
  if activation == .leakyReLU {
    out = out.leakyReLU(negativeSlope: 0.01)
  } else {
    let (mapper, actPost) = Activation1d(
      prefix: "act_post", channels: layers.last?.channels ?? initialChannels, width: currentWidth)
    out = actPost(out)
    mappers.append(mapper)
  }
  let convPost = Convolution(
    groups: 1, filters: 2, filterSize: [1, 7], noBias: !finalConvBias,
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 3], end: [0, 3])), format: .OIHW,
    name: "conv_post")
  out = convPost(out)
  switch finalActivation {
  case .clamp:
    out = out.clamped(-1...1)
  case .tanh:
    out = out.tanh()
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["conv_pre.weight"] = [convPre.weight.name]
    mapping["conv_pre.bias"] = [convPre.bias.name]
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping["conv_post.weight"] = [convPost.weight.name]
    if finalConvBias {
      mapping["conv_post.bias"] = [convPost.bias.name]
    }
    return mapping
  }
  return (mapper, Model([x], [out]))
}
