import NNC

private func ResnetBlockCausal3D(
  prefix: String, inChannels: Int, outChannels: Int, shortcut: Bool, depth: Int, height: Int,
  width: Int
) -> (
  ModelWeightMapper, Model
) {
  let x = Input()
  let norm1 = GroupNorm(axis: 0, groups: 32, epsilon: 1e-6, reduce: [1, 2, 3])
  var out = norm1(x.reshaped([inChannels, depth, height, width])).reshaped([
    1, inChannels, depth, height, width,
  ])
  out = Swish()(out)
  let conv1 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])))
  out = conv1(out.padded(.replication, begin: [0, 0, 2, 1, 1], end: [0, 0, 0, 1, 1]))
  let norm2 = GroupNorm(axis: 0, groups: 32, epsilon: 1e-6, reduce: [1, 2, 3])
  out = norm2(out.reshaped([outChannels, depth, height, width])).reshaped([
    1, outChannels, depth, height, width,
  ])
  out = Swish()(out)
  let conv2 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])))
  out = conv2(out.padded(.replication, begin: [0, 0, 2, 1, 1], end: [0, 0, 0, 1, 1]))
  let ninShortcut: Model?
  if shortcut {
    let nin = Convolution(
      groups: 1, filters: outChannels, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]))
    out = nin(x) + out
    ninShortcut = nin
  } else {
    ninShortcut = nil
    out = x + out
  }
  let mapper: ModelWeightMapper = { _ in
    return ModelWeightMapping()
  }
  return (mapper, Model([x], [out]))
}

private func AttnBlockCausal3D(
  prefix: String, inChannels: Int, depth: Int, height: Int, width: Int
) -> (
  ModelWeightMapper, Model
) {
  let x = Input()
  let causalAttentionMask = Input()
  let norm = GroupNorm(axis: 0, groups: 32, epsilon: 1e-6, reduce: [1, 2, 3])
  var out = norm(x.reshaped([inChannels, depth, height, width])).reshaped([
    1, inChannels, depth, height, width,
  ])
  let hw = width * height * depth
  let tokeys = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]))
  let k = tokeys(out).reshaped([1, inChannels, hw])
  let toqueries = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]))
  let q = ((1.0 / Float(inChannels).squareRoot()) * toqueries(out)).reshaped([
    1, inChannels, hw,
  ])
  var dot =
    Matmul(transposeA: (1, 2))(q, k).reshaped([
      depth, height * width, depth, height * width,
    ]) + causalAttentionMask
  dot = dot.reshaped([hw, hw])
  dot = dot.softmax()
  dot = dot.reshaped([1, hw, hw])
  let tovalues = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]))
  let v = tovalues(out).reshaped([1, inChannels, hw])
  out = Matmul(transposeB: (1, 2))(v, dot)
  let projOut = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]))
  out = x + projOut(out.reshaped([1, inChannels, depth, height, width]))
  let mapper: ModelWeightMapper = { _ in
    return ModelWeightMapping()
  }
  return (mapper, Model([x, causalAttentionMask], [out]))
}

func EncoderCausal3D(
  channels: [Int], numRepeat: Int, startWidth: Int, startHeight: Int, startDepth: Int
)
  -> (ModelWeightMapper, Model)
{
  let x = Input()
  let causalAttentionMask = Input()
  var previousChannel = channels[0]
  let convIn = Convolution(
    groups: 1, filters: previousChannel, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])))
  var out = convIn(x.padded(.replication, begin: [0, 0, 2, 1, 1], end: [0, 0, 0, 1, 1]))
  var mappers = [ModelWeightMapper]()
  var height = startHeight
  var width = startWidth
  var depth = startDepth
  for i in 1..<channels.count {
    height *= 2
    width *= 2
    if i > 1 {
      depth = (depth - 1) * 2 + 1
    }
  }
  for (i, channel) in channels.enumerated() {
    for j in 0..<numRepeat {
      let (mapper, block) = ResnetBlockCausal3D(
        prefix: "encoder.down_blocks.\(i).resnets.\(j)", inChannels: previousChannel,
        outChannels: channel,
        shortcut: previousChannel != channel, depth: depth, height: height, width: width)
      mappers.append(mapper)
      out = block(out)
      previousChannel = channel
    }
    if i < channels.count - 1 {
      // Conv always pad left first, then right, and pad top first then bottom.
      // Thus, we cannot have (0, 1, 0, 1) (left 0, right 1, top 0, bottom 1) padding as in
      // Stable Diffusion. Instead, we pad to (2, 1, 2, 1) and simply discard the first row and first column.
      height /= 2
      width /= 2
      let strideZ: Int
      if i > 0 {
        depth = (depth - 1) / 2 + 1
        strideZ = 2
      } else {
        strideZ = 1
      }
      let conv2d = Convolution(
        groups: 1, filters: channel, filterSize: [3, 3, 3],
        hint: Hint(
          stride: [strideZ, 2, 2], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])))
      out = conv2d(out.padded(.replication, begin: [0, 0, 2, 1, 1], end: [0, 0, 0, 1, 1]))
      let downLayer = i
      let mapper: ModelWeightMapper = { _ in
        return ModelWeightMapping()
      }
      mappers.append(mapper)
    }
  }
  let (midBlockMapper1, midBlock1) = ResnetBlockCausal3D(
    prefix: "encoder.mid_block.resnets.0", inChannels: previousChannel,
    outChannels: previousChannel,
    shortcut: false, depth: depth, height: height, width: width)
  out = midBlock1(out)
  let (midAttnMapper1, midAttn1) = AttnBlockCausal3D(
    prefix: "encoder.mid_block.attentions.0", inChannels: previousChannel, depth: depth,
    height: height, width: width)
  out = midAttn1(out, causalAttentionMask)
  let (midBlockMapper2, midBlock2) = ResnetBlockCausal3D(
    prefix: "encoder.mid_block.resnets.1", inChannels: previousChannel,
    outChannels: previousChannel,
    shortcut: false, depth: depth, height: height, width: width)
  out = midBlock2(out)
  let normOut = GroupNorm(axis: 0, groups: 32, epsilon: 1e-6, reduce: [1, 2, 3])
  out = normOut(out.reshaped([previousChannel, depth, height, width])).reshaped([
    1, previousChannel, depth, height, width,
  ])
  out = out.swish()
  let convOut = Convolution(
    groups: 1, filters: 32, filterSize: [3, 3, 3])
  out = convOut(out.padded(.replication, begin: [0, 0, 2, 1, 1], end: [0, 0, 0, 1, 1]))
  let quantConv = Convolution(groups: 1, filters: 32, filterSize: [1, 1, 1])
  out = quantConv(out)
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping.merge(midBlockMapper1(format)) { v, _ in v }
    mapping.merge(midAttnMapper1(format)) { v, _ in v }
    mapping.merge(midBlockMapper2(format)) { v, _ in v }
    return mapping
  }
  return (mapper, Model([x, causalAttentionMask], [out]))
}

func DecoderCausal3D(
  channels: [Int], numRepeat: Int, startWidth: Int, startHeight: Int,
  startDepth: Int
)
  -> (ModelWeightMapper, Model)
{
  let x = Input()
  let causalAttentionMask = Input()
  var previousChannel = channels[channels.count - 1]
  let postQuantConv = Convolution(groups: 1, filters: 16, filterSize: [1, 1, 1])
  var out = postQuantConv(x)
  let convIn = Convolution(
    groups: 1, filters: previousChannel, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])))
  out = convIn(out.padded(.replication, begin: [0, 0, 2, 1, 1], end: [0, 0, 0, 1, 1]))
  let (midBlockMapper1, midBlock1) = ResnetBlockCausal3D(
    prefix: "decoder.mid_block.resnets.0", inChannels: previousChannel,
    outChannels: previousChannel, shortcut: false, depth: startDepth, height: startHeight,
    width: startWidth)
  out = midBlock1(out)
  let (midAttnMapper1, midAttn1) = AttnBlockCausal3D(
    prefix: "decoder.mid_block.attentions.0", inChannels: previousChannel, depth: startDepth,
    height: startHeight, width: startWidth)
  out = midAttn1(out, causalAttentionMask)
  let (midBlockMapper2, midBlock2) = ResnetBlockCausal3D(
    prefix: "decoder.mid_block.resnets.1", inChannels: previousChannel,
    outChannels: previousChannel, shortcut: false, depth: startDepth, height: startHeight,
    width: startWidth)
  out = midBlock2(out)
  var width = startWidth
  var height = startHeight
  var depth = startDepth
  var mappers = [ModelWeightMapper]()
  for (i, channel) in channels.enumerated().reversed() {
    for j in 0..<numRepeat + 1 {
      let (mapper, block) = ResnetBlockCausal3D(
        prefix: "decoder.up_blocks.\(channels.count - 1 - i).resnets.\(j)",
        inChannels: previousChannel, outChannels: channel,
        shortcut: previousChannel != channel, depth: depth, height: height, width: width)
      mappers.append(mapper)
      out = block(out)
      previousChannel = channel
    }
    if i > 0 {
      out = Upsample(.nearest, widthScale: 2, heightScale: 2)(
        out.reshaped([channel, depth, height, width])
      ).reshaped([1, channel, depth, height * 2, width * 2])
      width *= 2
      height *= 2
      if i < channels.count - 1 {  // Scale time too.
        let first = out.reshaped(
          [channel, 1, height * width], strides: [depth * height * width, height * width, 1]
        ).contiguous()
        let more = out.reshaped(
          [channel, (depth - 1), 1, height * width], offset: [0, 1, 0, 0],
          strides: [depth * height * width, height * width, height * width, 1]
        ).contiguous()
        out = Functional.concat(
          axis: 1, first,
          Functional.concat(axis: 2, more, more).reshaped([
            channel, (depth - 1) * 2, height * width,
          ]))
        depth = 1 + (depth - 1) * 2
        out = out.reshaped([1, channel, depth, height, width])
      }
      let conv2d = Convolution(
        groups: 1, filters: channel, filterSize: [3, 3, 3],
        hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])))
      out = conv2d(out.padded(.replication, begin: [0, 0, 2, 1, 1], end: [0, 0, 0, 1, 1]))
      let upLayer = channels.count - 1 - i
      let mapper: ModelWeightMapper = { _ in
        return ModelWeightMapping()
      }
      mappers.append(mapper)
    }
  }
  let normOut = GroupNorm(axis: 0, groups: 32, epsilon: 1e-6, reduce: [1, 2, 3])
  out = normOut(out.reshaped([channels[0], depth, height, width])).reshaped([
    1, channels[0], depth, height, width,
  ])
  out = out.swish()
  let convOut = Convolution(
    groups: 1, filters: 3, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])))
  out = convOut(out.padded(.replication, begin: [0, 0, 2, 1, 1], end: [0, 0, 0, 1, 1]))
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping.merge(midBlockMapper1(format)) { v, _ in v }
    mapping.merge(midAttnMapper1(format)) { v, _ in v }
    mapping.merge(midBlockMapper2(format)) { v, _ in v }
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    return mapping
  }
  return (mapper, Model([x, causalAttentionMask], [out]))
}
