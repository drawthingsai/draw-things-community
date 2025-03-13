import NNC

private struct ResnetBlockCausal3D {
  private let norm1: Model
  private let conv1: Model
  private let norm2: Model
  private let conv2: Model
  private let ninShortcut: Model?
  init(outChannels: Int, shortcut: Bool) {
    norm1 = RMSNorm(epsilon: 1e-6, axis: [3], name: "resnet_norm1")
    conv1 = Convolution(
      groups: 1, filters: outChannels, filterSize: [3, 3, 3],
      hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
      format: .OIHW, name: "resnet_conv1")
    norm2 = RMSNorm(epsilon: 1e-6, axis: [3], name: "resnet_norm2")
    conv2 = Convolution(
      groups: 1, filters: outChannels, filterSize: [3, 3, 3],
      hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
      format: .OIHW, name: "resnet_conv2")
    if shortcut {
      ninShortcut = Convolution(
        groups: 1, filters: outChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]),
        format: .OIHW, name: "resnet_shortcut")
    } else {
      ninShortcut = nil
    }
  }
  private var conv1Inputs: Model.IO? = nil
  private var conv2Inputs: Model.IO? = nil
  mutating func callAsFunction(
    input x: Model.IO, prefix: String, inChannels: Int, outChannels: Int, shortcut: Bool,
    depth: Int, height: Int, width: Int, inputsOnly: Bool
  ) -> (
    ModelWeightMapper, Model.IO
  ) {
    var out = norm1(x.reshaped([depth, height, width, inChannels]))
    var pre = out.swish()
    if let conv1Inputs = conv1Inputs {
      out = conv1(
        Functional.concat(axis: 0, conv1Inputs, pre, flags: [.disableOpt]).padded(
          .zero, begin: [0, 1, 1, 0], end: [0, 1, 1, 0]
        ).reshaped([
          1, depth + 2, height + 2, width + 2, inChannels,
        ]))
    } else {
      out = conv1(
        pre.padded(.zero, begin: [2, 1, 1, 0], end: [0, 1, 1, 0]).reshaped([
          1, depth + 2, height + 2, width + 2, inChannels,
        ]))
    }
    if !inputsOnly {
      conv1Inputs = pre.reshaped(
        [2, height, width, inChannels], offset: [depth - 2, 0, 0, 0],
        strides: [height * width * inChannels, width * inChannels, inChannels, 1]
      ).contiguous()
    } else {
      conv1Inputs = nil
    }
    out = norm2(out.reshaped([depth, height, width, outChannels]))
    pre = out.swish()
    if let conv2Inputs = conv2Inputs {
      out = conv2(
        Functional.concat(axis: 0, conv2Inputs, pre, flags: [.disableOpt]).padded(
          .zero, begin: [0, 1, 1, 0], end: [0, 1, 1, 0]
        ).reshaped([
          1, depth + 2, height + 2, width + 2, outChannels,
        ]))
    } else {
      out = conv2(
        pre.padded(.zero, begin: [2, 1, 1, 0], end: [0, 1, 1, 0]).reshaped([
          1, depth + 2, height + 2, width + 2, outChannels,
        ]))
    }
    out = out.reshaped([depth, height, width, outChannels])
    if !inputsOnly {
      conv2Inputs = pre.reshaped(
        [2, height, width, outChannels], offset: [depth - 2, 0, 0, 0],
        strides: [height * width * outChannels, width * outChannels, outChannels, 1]
      ).contiguous()
    } else {
      conv2Inputs = nil
    }
    if let ninShortcut = ninShortcut {
      out = ninShortcut(x) + out
    } else {
      out = x + out
    }
    let mapper: ModelWeightMapper = { _ in
      return ModelWeightMapping()
    }
    return (mapper, out)
  }
}

private struct AttnBlockCausal3D {
  private let norm: Model
  private let toqueries: Model
  private let tokeys: Model
  private let tovalues: Model
  private let projOut: Model
  init(inChannels: Int) {
    norm = RMSNorm(epsilon: 1e-6, axis: [3], name: "attn_norm")
    tokeys = Convolution(
      groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]),
      format: .OIHW, name: "to_k")
    toqueries = Convolution(
      groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]),
      format: .OIHW, name: "to_q")
    tovalues = Convolution(
      groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]),
      format: .OIHW, name: "to_v")
    projOut = Convolution(
      groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]),
      format: .OIHW, name: "proj_out")
  }
  func callAsFunction(
    prefix: String, inChannels: Int, depth: Int, height: Int, width: Int
  ) -> (
    ModelWeightMapper, Model
  ) {
    let x = Input()
    var out = norm(x.reshaped([depth, height, width, inChannels]))
    let hw = width * height
    let k = tokeys(out).reshaped([depth, hw, inChannels])
    let q = ((1.0 / Float(inChannels).squareRoot()) * toqueries(out)).reshaped([
      depth, hw, inChannels,
    ])
    var dot =
      Matmul(transposeB: (1, 2))(q, k)
    dot = dot.reshaped([depth * hw, hw])
    dot = dot.softmax()
    dot = dot.reshaped([depth, hw, hw])
    let v = tovalues(out).reshaped([depth, hw, inChannels])
    out = dot * v
    out = x + projOut(out.reshaped([depth, height, width, inChannels]))
    let mapper: ModelWeightMapper = { _ in
      return ModelWeightMapping()
    }
    return (mapper, Model([x], [out]))
  }
}

func WanDecoderCausal3D(
  channels: [Int], numRepeat: Int, startWidth: Int, startHeight: Int,
  startDepth: Int, paddingFinalConvLayer: Bool
)
  -> (ModelWeightMapper, Model)
{
  let x = Input()
  var previousChannel = channels[channels.count - 1]
  let postQuantConv = Convolution(
    groups: 1, filters: 16, filterSize: [1, 1], hint: Hint(stride: [1, 1]),
    format: .OIHW, name: "post_quant_conv")
  let postQuantX = postQuantConv(x)
  let convIn = Convolution(
    groups: 1, filters: previousChannel, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
    format: .OIHW, name: "conv_in")
  let convOut = Convolution(
    groups: 1, filters: paddingFinalConvLayer ? 4 : 3, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
    format: .OIHW, name: "conv_out")
  let normOut = RMSNorm(epsilon: 1e-6, axis: [3], name: "norm_out")
  var finalOut: Model.IO? = nil
  var midBlock1Builder = ResnetBlockCausal3D(outChannels: previousChannel, shortcut: false)
  let midAttn1Builder = AttnBlockCausal3D(inChannels: previousChannel)
  var midBlock2Builder = ResnetBlockCausal3D(outChannels: previousChannel, shortcut: false)
  var upBlockBuilders = [ResnetBlockCausal3D]()
  for (i, channel) in channels.enumerated().reversed() {
    for _ in 0..<numRepeat + 1 {
      upBlockBuilders.append(
        ResnetBlockCausal3D(outChannels: channel, shortcut: previousChannel != channel))
      previousChannel = channel
    }
    if i > 0 {
      previousChannel = channel / 2
    }
  }
  let timeConvs = (0..<(channels.count - 2)).map { i in
    return Convolution(
      groups: 1, filters: channels[channels.count - i - 1] * 2, filterSize: [3, 1, 1],
      hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
      format: .OIHW, name: "time_conv")
  }
  let upsampleConv2d = (0..<(channels.count - 1)).map { i in
    return Convolution(
      groups: 1, filters: channels[channels.count - i - 1] / 2, filterSize: [3, 3],
      hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])),
      format: .OIHW, name: "upsample")
  }
  var timeInputs: [Model.IO?] = Array(repeating: nil, count: channels.count - 2)
  var convOutInputs: Model.IO? = nil
  for d in stride(from: 0, to: max(startDepth - 1, 1), by: 2) {
    previousChannel = channels[channels.count - 1]
    var out: Model.IO
    if d == 0 {
      out = postQuantX.reshaped(
        [min(startDepth - d, 3), startHeight, startWidth, 16], offset: [d, 0, 0, 0],
        strides: [startHeight * startWidth * 16, startWidth * 16, 16, 1]
      ).contiguous()
      out = convIn(
        out.padded(.zero, begin: [2, 1, 1, 0], end: [0, 1, 1, 0]).reshaped([
          1, min(startDepth - d, 3) + 2, startHeight + 2, startWidth + 2, 16,
        ])
      ).reshaped([min(startDepth - d, 3), startHeight, startWidth, previousChannel])
    } else {
      out = postQuantX.reshaped(
        [min(startDepth - (d - 1), 4), startHeight, startWidth, 16],
        offset: [d - 1, 0, 0, 0],
        strides: [startHeight * startWidth * 16, startWidth * 16, 16, 1]
      ).contiguous()
      if let last = finalOut {
        out.add(dependencies: [last])
      }
      out = convIn(
        out.padded(.zero, begin: [0, 1, 1, 0], end: [0, 1, 1, 0]).reshaped([
          1, min(startDepth - (d - 1), 4), startHeight + 2, startWidth + 2, 16,
        ])
      ).reshaped([min(startDepth - (d - 1), 4) - 2, startHeight, startWidth, previousChannel])
    }
    let inputsOnly = startDepth - 1 - d <= 2  // This is the last one.
    var width = startWidth
    var height = startHeight
    var depth = d > 0 ? min(startDepth - 1 - d, 2) : min(startDepth, 3)
    let (midBlockMapper1, midBlock1Out) = midBlock1Builder(
      input: out,
      prefix: "decoder.middle.0", inChannels: previousChannel,
      outChannels: previousChannel, shortcut: false, depth: depth, height: height,
      width: width, inputsOnly: inputsOnly)
    out = midBlock1Out
    let (midAttnMapper1, midAttn1) = midAttn1Builder(
      prefix: "decoder.middle.1", inChannels: previousChannel, depth: depth,
      height: height, width: width)
    out = midAttn1(out)
    let (midBlockMapper2, midBlock2Out) = midBlock2Builder(
      input: out,
      prefix: "decoder.middle.2", inChannels: previousChannel,
      outChannels: previousChannel, shortcut: false, depth: depth, height: height,
      width: width, inputsOnly: inputsOnly)
    out = midBlock2Out
    var mappers = [ModelWeightMapper]()
    var j = 0
    var k = 0
    for (i, channel) in channels.enumerated().reversed() {
      for _ in 0..<numRepeat + 1 {
        let (mapper, blockOut) = upBlockBuilders[j](
          input: out,
          prefix: "decoder.upsamples.\(k)",
          inChannels: previousChannel, outChannels: channel,
          shortcut: previousChannel != channel, depth: depth, height: height, width: width,
          inputsOnly: inputsOnly)
        mappers.append(mapper)
        out = blockOut
        previousChannel = channel
        j += 1
        k += 1
      }
      if i > 0 {
        if i > 1 && startDepth > 1 {  // Need to bump up on the depth axis.
          if d == 0 {  // Special case for first frame.
            let first = out.reshaped(
              [1, height, width, channel],
              strides: [height * width * channel, width * channel, channel, 1]
            ).contiguous()
            let more = out.reshaped(
              [(depth - 1), height, width, channel], offset: [1, 0, 0, 0],
              strides: [height * width * channel, width * channel, channel, 1]
            ).contiguous()
            var expanded = timeConvs[channels.count - i - 1](
              more.padded(.zero, begin: [2, 0, 0, 0], end: [0, 0, 0, 0]).reshaped([
                1, depth + 1, height, width, channel,
              ]))
            if !inputsOnly {
              timeInputs[channels.count - i - 1] = out.reshaped(
                [2, height, width, channel], offset: [depth - 2, 0, 0, 0],
                strides: [height * width * channel, width * channel, channel, 1]
              ).contiguous()
            }
            let upLayer = k
            let mapper: ModelWeightMapper = { _ in
              return ModelWeightMapping()
            }
            mappers.append(mapper)
            expanded = expanded.reshaped([depth - 1, height, width, 2, channel]).permuted(
              0, 3, 1, 2, 4
            )
            .contiguous().reshaped([2 * (depth - 1), height, width, channel])
            out = Functional.concat(axis: 0, first, expanded)
            depth = 1 + (depth - 1) * 2
            out = out.reshaped([depth, height, width, channel])
          } else if let timeInput = timeInputs[channels.count - i - 1] {
            let more = out.reshaped([depth, height, width, channel])
            let expanded = timeConvs[channels.count - i - 1](
              Functional.concat(axis: 0, timeInput, more, flags: [.disableOpt]).reshaped([
                1, depth + 2, height, width, channel,
              ]))
            if !inputsOnly {
              timeInputs[channels.count - i - 1] = out.reshaped(
                [2, height, width, channel], offset: [depth - 2, 0, 0, 0],
                strides: [height * width * channel, width * channel, channel, 1]
              ).contiguous()
            }
            let upLayer = k
            let mapper: ModelWeightMapper = { _ in
              return ModelWeightMapping()
            }
            mappers.append(mapper)
            out = expanded.reshaped([depth, height, width, 2, channel]).permuted(0, 3, 1, 2, 4)
              .contiguous().reshaped([2 * depth, height, width, channel])
            depth = depth * 2
          }
        }
        out = Upsample(.nearest, widthScale: 2, heightScale: 2)(
          out.reshaped([depth, height, width, channel])
        ).reshaped([depth, height * 2, width * 2, channel])
        width *= 2
        height *= 2
        out = upsampleConv2d[channels.count - i - 1](out)
        previousChannel = channel / 2
        let upLayer = k
        let mapper: ModelWeightMapper = { _ in
          return ModelWeightMapping()
        }
        mappers.append(mapper)
        k += 1
      }
    }
    out = normOut(out.reshaped([depth, height, width, channels[0]]))
    let pre = out.swish()
    if let convOutInputs = convOutInputs {
      out = convOut(
        Functional.concat(axis: 0, convOutInputs, pre, flags: [.disableOpt]).padded(
          .zero, begin: [0, 1, 1, 0], end: [0, 1, 1, 0]
        ).reshaped([
          1, depth + 2, height + 2, width + 2, channels[0],
        ])
      ).reshaped([
        depth, height, width, paddingFinalConvLayer ? 4 : 3,
      ])
    } else {
      out = convOut(
        pre.padded(.zero, begin: [2, 1, 1, 0], end: [0, 1, 1, 0]).reshaped([
          1, depth + 2, height + 2, width + 2, channels[0],
        ])
      ).reshaped([
        depth, height, width, paddingFinalConvLayer ? 4 : 3,
      ])
    }
    if !inputsOnly {
      convOutInputs = pre.reshaped(
        [2, height, width, channels[0]], offset: [depth - 2, 0, 0, 0],
        strides: [height * width * channels[0], width * channels[0], channels[0], 1]
      ).contiguous()
    }
    if let otherOut = finalOut {
      finalOut = Functional.concat(axis: 0, otherOut, out, flags: [.disableOpt])
    } else {
      finalOut = out
    }
  }
  let out = finalOut!
  let mapper: ModelWeightMapper = { _ in
    return ModelWeightMapping()
  }
  return (mapper, Model([x], [out]))
}

func WanEncoderCausal3D(
  channels: [Int], numRepeat: Int, startWidth: Int, startHeight: Int, startDepth: Int
)
  -> (ModelWeightMapper, Model)
{
  let x = Input()
  var previousChannel = channels[0]
  let convIn = Convolution(
    groups: 1, filters: previousChannel, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
    format: .OIHW, name: "conv_in")
  var out = convIn(x.padded(.zero, begin: [0, 0, 2, 1, 1], end: [0, 0, 0, 1, 1]))
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
  var k = 0
  for (i, channel) in channels.enumerated() {
    for _ in 0..<numRepeat {
      var builder = ResnetBlockCausal3D(outChannels: channel, shortcut: previousChannel != channel)
      let (mapper, blockOut) = builder(
        input: out,
        prefix: "encoder.downsamples.\(k)", inChannels: previousChannel,
        outChannels: channel,
        shortcut: previousChannel != channel, depth: depth, height: height, width: width,
        inputsOnly: true)
      mappers.append(mapper)
      out = blockOut
      previousChannel = channel
      k += 1
    }
    if i < channels.count - 1 {
      height /= 2
      width /= 2
      let conv2d = Convolution(
        groups: 1, filters: channel, filterSize: [1, 3, 3],
        hint: Hint(
          stride: [1, 2, 2], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
        format: .OIHW, name: "downsample")
      out = conv2d(out.padded(.zero, begin: [0, 0, 0, 0, 0], end: [0, 0, 0, 1, 1]))
      let downLayer = k
      let mapper: ModelWeightMapper = { _ in
        return ModelWeightMapping()
      }
      mappers.append(mapper)
      if i > 0 && depth > 1 {
        let first = out.reshaped(
          [1, channel, 1, height, width],
          strides: [depth * height * width, depth * height * width, height * width, width, 1]
        ).contiguous()
        let timeConv = Convolution(
          groups: 1, filters: channel, filterSize: [3, 1, 1],
          hint: Hint(
            stride: [2, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
          format: .OIHW, name: "time_conv")
        let shrunk = timeConv(out)
        let upLayer = k
        let mapper: ModelWeightMapper = { _ in
          return ModelWeightMapping()
        }
        mappers.append(mapper)
        depth = (depth - 1) / 2 + 1
        out = Functional.concat(axis: 2, first, shrunk)
      }
      k += 1
    }
  }
  var midBlock1Builder = ResnetBlockCausal3D(outChannels: previousChannel, shortcut: false)
  let (midBlockMapper1, midBlock1Out) = midBlock1Builder(
    input: out,
    prefix: "encoder.middle.0", inChannels: previousChannel,
    outChannels: previousChannel,
    shortcut: false, depth: depth, height: height, width: width, inputsOnly: true)
  out = midBlock1Out
  var midAttn1Builder = AttnBlockCausal3D(inChannels: previousChannel)
  let (midAttnMapper1, midAttn1) = midAttn1Builder(
    prefix: "encoder.middle.1", inChannels: previousChannel, depth: depth,
    height: height, width: width)
  out = midAttn1(out)
  var midBlock2Builder = ResnetBlockCausal3D(outChannels: previousChannel, shortcut: false)
  let (midBlockMapper2, midBlock2Out) = midBlock2Builder(
    input: out,
    prefix: "encoder.middle.2", inChannels: previousChannel,
    outChannels: previousChannel,
    shortcut: false, depth: depth, height: height, width: width, inputsOnly: true)
  out = midBlock2Out
  let normOut = RMSNorm(epsilon: 1e-6, axis: [3], name: "norm_out")
  out = normOut(out.reshaped([previousChannel, depth, height, width])).reshaped([
    1, previousChannel, depth, height, width,
  ])
  out = out.swish()
  let convOut = Convolution(
    groups: 1, filters: 32, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
    format: .OIHW, name: "conv_out")
  out = convOut(out.padded(.zero, begin: [0, 0, 2, 1, 1], end: [0, 0, 0, 1, 1]))
  let quantConv = Convolution(
    groups: 1, filters: 32, filterSize: [1, 1, 1], format: .OIHW, name: "quant_conv")
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
  return (mapper, Model([x], [out]))
}
