import NNC

private func ResnetBlockCausal3D(
  prefix: String, channels: Int, depth: Int, height: Int, width: Int, isCausal: Bool
) -> (
  ModelWeightMapper, Model
) {
  let x = Input()
  let norm1 = RMSNorm(epsilon: 1e-8, axis: [0], elementwiseAffine: false, name: "resnet_norm1")
  var out = norm1(x.reshaped([channels, depth, height, width])).reshaped([
    1, channels, depth, height, width,
  ])
  out = out.swish()
  let conv1 = Convolution(
    groups: 1, filters: channels, filterSize: [3, 3, 3],
    hint: Hint(
      stride: [1, 1, 1],
      border: Hint.Border(
        begin: [0, isCausal ? 1 : 0, isCausal ? 1 : 0],
        end: [0, isCausal ? 1 : 0, isCausal ? 1 : 0])),
    name: "resnet_conv1")
  out = out.padded(
    .replicate, begin: [0, 0, isCausal ? 2 : 1, 0, 0], end: [0, 0, isCausal ? 0 : 1, 0, 0])
  if !isCausal {
    out = out.padded(.reflect, begin: [0, 0, 0, 1, 1], end: [0, 0, 0, 1, 1])
  }
  out = conv1(out)
  let norm2 = RMSNorm(epsilon: 1e-8, axis: [0], elementwiseAffine: false, name: "resnet_norm2")
  out = norm2(out.reshaped([channels, depth, height, width])).reshaped([
    1, channels, depth, height, width,
  ])
  out = out.swish()
  let conv2 = Convolution(
    groups: 1, filters: channels, filterSize: [3, 3, 3],
    hint: Hint(
      stride: [1, 1, 1],
      border: Hint.Border(
        begin: [0, isCausal ? 1 : 0, isCausal ? 1 : 0],
        end: [0, isCausal ? 1 : 0, isCausal ? 1 : 0])),
    name: "resnet_conv2")
  out = out.padded(
    .replicate, begin: [0, 0, isCausal ? 2 : 1, 0, 0], end: [0, 0, isCausal ? 0 : 1, 0, 0])
  if !isCausal {
    out = out.padded(.reflect, begin: [0, 0, 0, 1, 1], end: [0, 0, 0, 1, 1])
  }
  out = conv2(out)
  out = x + out
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).conv1.conv.weight"] = [conv1.weight.name]
    mapping["\(prefix).conv1.conv.bias"] = [conv1.bias.name]
    mapping["\(prefix).conv2.conv.weight"] = [conv2.weight.name]
    mapping["\(prefix).conv2.conv.bias"] = [conv2.bias.name]
    return mapping
  }
  return (mapper, Model([x], [out]))
}

func LTX2VideoDecoderCausal3D(
  channels: [Int], numRepeat: Int, startWidth: Int, startHeight: Int,
  startDepth: Int
)
  -> (ModelWeightMapper, Model)
{
  let x = Input()
  var previousChannel = channels[channels.count - 1]
  let convIn = Convolution(
    groups: 1, filters: previousChannel, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
    name: "conv_in")
  var out = convIn(
    x.padded(.replicate, begin: [0, 0, 1, 0, 0], end: [0, 0, 1, 0, 0]).padded(
      .reflect, begin: [0, 0, 0, 1, 1], end: [0, 0, 0, 1, 1]))
  var mappers = [ModelWeightMapper]()
  var j = 0
  var depth = startDepth
  var height = startHeight
  var width = startWidth
  for channel in channels.reversed() {
    if previousChannel != channel {
      // Convolution & reshape.
      let conv = Convolution(
        groups: 1, filters: channel * 8, filterSize: [3, 3, 3],
        hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
        name: "depth_to_space_upsample")
      var residual = out.reshaped([previousChannel / 8, 2, 2, 2, depth, height, width]).permuted(
        0, 4, 1, 5, 2, 6, 3
      ).contiguous().reshaped(
        [1, previousChannel / 8, depth * 2 - 1, height * 2, width * 2], offset: [0, 0, 1, 0, 0],
        strides: [
          previousChannel * depth * height * width, depth * 2 * height * 2 * width * 2,
          height * 2 * width * 2, width * 2, 1,
        ]
      ).contiguous()
      residual = Functional.concat(
        axis: 1, residual, residual, residual, residual, flags: [.disableOpt])
      out = conv(
        out.padded(.replicate, begin: [0, 0, 1, 0, 0], end: [0, 0, 1, 0, 0]).padded(
          .reflect, begin: [0, 0, 0, 1, 1], end: [0, 0, 0, 1, 1]))
      let upBlocks = j
      mappers.append { _ in
        var mapping = ModelWeightMapping()
        mapping["up_blocks.\(upBlocks).conv.conv.weight"] = [conv.weight.name]
        mapping["up_blocks.\(upBlocks).conv.conv.bias"] = [conv.bias.name]
        return mapping
      }
      out =
        residual
        + out.reshaped([channel, 2, 2, 2, depth, height, width]).permuted(0, 4, 1, 5, 2, 6, 3)
        .contiguous().reshaped(
          [1, channel, depth * 2 - 1, height * 2, width * 2], offset: [0, 0, 1, 0, 0],
          strides: [
            channel * depth * 2 * height * 2 * width * 2, depth * 2 * height * 2 * width * 2,
            height * 2 * width * 2, width * 2, 1,
          ]
        ).contiguous()
      j += 1
      depth = depth * 2 - 1
      height = height * 2
      width = width * 2
      previousChannel = channel
    }
    for i in 0..<numRepeat {
      let (mapper, block) = ResnetBlockCausal3D(
        prefix: "up_blocks.\(j).res_blocks.\(i)", channels: channel, depth: depth, height: height,
        width: width, isCausal: false)
      out = block(out)
      mappers.append(mapper)
    }
    j += 1
  }
  let normOut = RMSNorm(epsilon: 1e-8, axis: [0], elementwiseAffine: false, name: "norm_out")
  out = normOut(out.reshaped([previousChannel, depth, height, width])).reshaped([
    1, previousChannel, depth, height, width,
  ]).swish()
  let convOut = Convolution(
    groups: 1, filters: 48, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
    name: "conv_out")
  out = convOut(
    out.padded(.replicate, begin: [0, 0, 1, 0, 0], end: [0, 0, 1, 0, 0]).padded(
      .reflect, begin: [0, 0, 0, 1, 1], end: [0, 0, 0, 1, 1]))
  // LTXV weirdly, did "b (c p r q) f h w -> b c (f p) (h q) (w r)"
  out = out.reshaped([3, 4, 4, depth, height, width]).permuted(0, 3, 4, 2, 5, 1).contiguous()
    .reshaped([3, depth, height * 4, width * 4])

  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["conv_in.conv.weight"] = [convIn.weight.name]
    mapping["conv_in.conv.bias"] = [convIn.bias.name]
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping["conv_out.conv.weight"] = [convOut.weight.name]
    mapping["conv_out.conv.bias"] = [convOut.bias.name]
    return mapping
  }
  return (mapper, Model([x], [out]))
}

func LTX2VideoEncoderCausal3D(
  layers: [(channels: Int, numRepeat: Int, stride: (Int, Int, Int))], startWidth: Int,
  startHeight: Int,
  startDepth: Int
)
  -> (ModelWeightMapper, Model)
{
  let x = Input()
  var height = startHeight
  var width = startWidth
  var depth = startDepth
  for layer in layers {
    depth = (depth - 1) * layer.stride.0 + 1
    height *= layer.stride.1
    width *= layer.stride.2
  }
  // LTXV weirdly, did "b (c p r q) f h w -> b c (f p) (h q) (w r)"
  var out = x.reshaped([3, depth, height, 4, width, 4]).permuted(0, 5, 3, 1, 2, 4).contiguous()
    .reshaped([1, 48, depth, height, width])
  var previousChannel = layers[0].channels
  let convIn = Convolution(
    groups: 1, filters: previousChannel, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
    name: "conv_in")
  out = convIn(out.padded(.replicate, begin: [0, 0, 2, 0, 0], end: [0, 0, 0, 0, 0]))
  var j = 0
  var mappers = [ModelWeightMapper]()
  for layer in layers {
    let channels = layer.channels
    if layer.stride.0 > 1 || layer.stride.1 > 1 || layer.stride.2 > 1 {
      // Convolution & reshape.
      let conv = Convolution(
        groups: 1, filters: channels / (layer.stride.0 * layer.stride.1 * layer.stride.2),
        filterSize: [3, 3, 3],
        hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
        name: "space_to_depth_downsample")
      if layer.stride.0 == 1 {
        var residual = out.reshaped([
          previousChannel, depth, height / layer.stride.1, layer.stride.1, width / layer.stride.2,
          layer.stride.2,
        ]).permuted(0, 3, 5, 1, 2, 4).contiguous().reshaped([
          1, channels, previousChannel * layer.stride.1 * layer.stride.2 / channels, depth,
          height / layer.stride.1, width / layer.stride.2,
        ])
        residual = residual.reduced(.mean, axis: [2]).reshaped([
          1, channels, depth, height / layer.stride.1, width / layer.stride.2,
        ])
        out = conv(out.padded(.replicate, begin: [0, 0, 2, 0, 0]))
        out = out.reshaped([
          channels / (layer.stride.1 * layer.stride.2), depth, height / layer.stride.1,
          layer.stride.1, width / layer.stride.2, layer.stride.2,
        ]).permuted(0, 3, 5, 1, 2, 4).contiguous().reshaped([
          1, channels, depth, height / layer.stride.1, width / layer.stride.2,
        ])
        out = residual + out
        height = height / layer.stride.1
        width = width / layer.stride.2
      } else if layer.stride.1 == 1 && layer.stride.2 == 1 {
        var residual = out.padded(.replicate, begin: [0, 0, 1, 0, 0]).reshaped([
          previousChannel, (depth + 1) / layer.stride.0, layer.stride.0, height, width,
        ]).permuted(0, 2, 1, 3, 4).contiguous()
        if previousChannel * layer.stride.0 / channels > 1 {
          residual = residual.reshaped([
            1, channels, previousChannel * layer.stride.0 / channels, (depth + 1) / layer.stride.0,
            height, width,
          ])
          residual = residual.reduced(.mean, axis: [2]).reshaped([
            1, channels, (depth + 1) / layer.stride.0, height, width,
          ])
        } else {
          residual = residual.reshaped([1, channels, (depth + 1) / layer.stride.0, height, width])
        }
        out = conv(out.padded(.replicate, begin: [0, 0, 3, 0, 0]))
        out = out.reshaped([
          channels / layer.stride.0, (depth + 1) / layer.stride.0, layer.stride.0, height, width,
        ]).permuted(0, 2, 1, 3, 4).contiguous().reshaped([
          1, channels, (depth + 1) / layer.stride.0, height, width,
        ])
        out = residual + out
        depth = (depth + 1) / layer.stride.0
      } else {
        var residual = out.padded(.replicate, begin: [0, 0, 1, 0, 0]).reshaped([
          previousChannel, (depth + 1) / layer.stride.0, layer.stride.0, height / layer.stride.1,
          layer.stride.1, width / layer.stride.2, layer.stride.2,
        ]).permuted(0, 2, 4, 6, 1, 3, 5).contiguous().reshaped([
          1, channels,
          previousChannel * layer.stride.0 * layer.stride.1 * layer.stride.2 / channels,
          (depth + 1) / layer.stride.0, height / layer.stride.1, width / layer.stride.2,
        ])
        residual = residual.reduced(.mean, axis: [2]).reshaped([
          1, channels, (depth + 1) / layer.stride.0, height / layer.stride.1,
          width / layer.stride.2,
        ])
        out = conv(out.padded(.replicate, begin: [0, 0, 3, 0, 0]))
        out = out.reshaped([
          channels / (layer.stride.0 * layer.stride.1 * layer.stride.2),
          (depth + 1) / layer.stride.0, layer.stride.0, height / layer.stride.1, layer.stride.1,
          width / layer.stride.2, layer.stride.2,
        ]).permuted(0, 2, 4, 6, 1, 3, 5).contiguous().reshaped([
          1, channels, (depth + 1) / layer.stride.0, height / layer.stride.1,
          width / layer.stride.2,
        ])
        out = residual + out
        depth = (depth + 1) / layer.stride.0
        height = height / layer.stride.1
        width = width / layer.stride.2
      }
      previousChannel = channels
      let downBlocks = j
      mappers.append { _ in
        var mapping = ModelWeightMapping()
        mapping["down_blocks.\(downBlocks).conv.conv.weight"] = [conv.weight.name]
        mapping["down_blocks.\(downBlocks).conv.conv.bias"] = [conv.bias.name]
        return mapping
      }
      j += 1
    }
    for i in 0..<layer.numRepeat {
      let (mapper, block) = ResnetBlockCausal3D(
        prefix: "down_blocks.\(j).res_blocks.\(i)", channels: channels, depth: depth,
        height: height, width: width, isCausal: true)
      out = block(out)
      mappers.append(mapper)
    }
    j += 1
  }
  let normOut = RMSNorm(epsilon: 1e-8, axis: [0], elementwiseAffine: false, name: "norm_out")
  out = normOut(out.reshaped([previousChannel, depth, height, width])).reshaped([
    1, previousChannel, depth, height, width,
  ]).swish()
  let convOut = Convolution(
    groups: 1, filters: 129, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
    name: "conv_out")
  out = convOut(out.padded(.replicate, begin: [0, 0, 2, 0, 0]))
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["conv_in.conv.weight"] = [convIn.weight.name]
    mapping["conv_in.conv.bias"] = [convIn.bias.name]
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping["conv_out.conv.weight"] = [convOut.weight.name]
    mapping["conv_out.conv.bias"] = [convOut.bias.name]
    return mapping
  }
  return (mapper, Model([x], [out]))
}
