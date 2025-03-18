import NNC

private struct NHWCResnetBlockCausal3D {
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
      let inputs = pre.reshaped(
        [2, height, width, inChannels], offset: [depth - 2, 0, 0, 0],
        strides: [height * width * inChannels, width * inChannels, inChannels, 1]
      ).copied()
      conv1Inputs = inputs
      out.add(dependencies: [inputs])  // This makes sure the copy is done before the convolution, freeing the activations holds there.
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
    if !inputsOnly {
      let inputs = pre.reshaped(
        [2, height, width, outChannels], offset: [depth - 2, 0, 0, 0],
        strides: [height * width * outChannels, width * outChannels, outChannels, 1]
      ).copied()
      conv2Inputs = inputs
      out.add(dependencies: [inputs])  // This makes sure the copy is done before the convolution, freeing the activations holds there.
    } else {
      conv2Inputs = nil
    }
    out = out.reshaped([depth, height, width, outChannels])
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

private struct NHWCAttnBlockCausal3D {
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

private func NHWCWanDecoderCausal3D(
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
  var last: Model.IO? = nil
  var midBlock1Builder = NHWCResnetBlockCausal3D(outChannels: previousChannel, shortcut: false)
  let midAttn1Builder = NHWCAttnBlockCausal3D(inChannels: previousChannel)
  var midBlock2Builder = NHWCResnetBlockCausal3D(outChannels: previousChannel, shortcut: false)
  var upBlockBuilders = [NHWCResnetBlockCausal3D]()
  for (i, channel) in channels.enumerated().reversed() {
    for _ in 0..<numRepeat + 1 {
      upBlockBuilders.append(
        NHWCResnetBlockCausal3D(outChannels: channel, shortcut: previousChannel != channel))
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
  var outs = [Model.IO]()
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
      ).contiguous().padded(.zero, begin: [0, 1, 1, 0], end: [0, 1, 1, 0])
      if let last = last {
        out.add(dependencies: [last])
      }
      out = convIn(
        out.reshaped([
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
              let inputs = out.reshaped(
                [2, height, width, channel], offset: [depth - 2, 0, 0, 0],
                strides: [height * width * channel, width * channel, channel, 1]
              ).copied()
              timeInputs[channels.count - i - 1] = inputs
              expanded.add(dependencies: [inputs])
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
              let inputs = out.reshaped(
                [2, height, width, channel], offset: [depth - 2, 0, 0, 0],
                strides: [height * width * channel, width * channel, channel, 1]
              ).copied()
              timeInputs[channels.count - i - 1] = inputs
              expanded.add(dependencies: [inputs])
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
      )
    } else {
      out = convOut(
        pre.padded(.zero, begin: [2, 1, 1, 0], end: [0, 1, 1, 0]).reshaped([
          1, depth + 2, height + 2, width + 2, channels[0],
        ])
      )
    }
    if !inputsOnly {
      let inputs = pre.reshaped(
        [2, height, width, channels[0]], offset: [depth - 2, 0, 0, 0],
        strides: [height * width * channels[0], width * channels[0], channels[0], 1]
      ).copied()
      convOutInputs = inputs
      out.add(dependencies: [inputs])
    }
    last = out
    outs.append(
      out.reshaped([
        depth, height, width, paddingFinalConvLayer ? 4 : 3,
      ]))
  }
  let out: Model.IO
  if outs.count > 1 {
    out = Concat(axis: 0)(outs)
  } else {
    out = outs[0]
  }
  let mapper: ModelWeightMapper = { _ in
    return ModelWeightMapping()
  }
  return (mapper, Model([x], [out]))
}

private func NHWCWanEncoderCausal3D(
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
  var downBlockBuilders = [NHWCResnetBlockCausal3D]()
  var timeConvs = [Convolution]()
  var downsampleConv2d = [Convolution]()
  for (i, channel) in channels.enumerated() {
    for _ in 0..<numRepeat {
      downBlockBuilders.append(
        NHWCResnetBlockCausal3D(outChannels: channel, shortcut: previousChannel != channel))
      previousChannel = channel
    }
    if i < channels.count - 1 {
      downsampleConv2d.append(
        Convolution(
          groups: 1, filters: channel, filterSize: [3, 3],
          hint: Hint(
            stride: [2, 2], border: Hint.Border(begin: [0, 0], end: [0, 0])),
          format: .OIHW, name: "downsample"))
      if i > 0 && startDepth > 1 {
        timeConvs.append(
          Convolution(
            groups: 1, filters: channel, filterSize: [3, 1, 1],
            hint: Hint(
              stride: [2, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
            format: .OIHW, name: "time_conv"))
      }
    }
  }
  var timeInputs: [Model.IO?] = Array(repeating: nil, count: channels.count - 2)
  var midBlock1Builder = NHWCResnetBlockCausal3D(outChannels: previousChannel, shortcut: false)
  var midAttn1Builder = NHWCAttnBlockCausal3D(inChannels: previousChannel)
  var midBlock2Builder = NHWCResnetBlockCausal3D(outChannels: previousChannel, shortcut: false)
  let normOut = RMSNorm(epsilon: 1e-6, axis: [3], name: "norm_out")
  let endHeight = height
  let endWidth = width
  let endDepth = depth
  var outs = [Model.IO]()
  let input = x
  for d in stride(from: 0, to: max(startDepth - 1, 1), by: 2) {
    previousChannel = channels[0]
    height = endHeight
    width = endWidth
    var out: Model.IO
    if d == 0 {
      depth = min(endDepth, 9)
      out = input.reshaped(
        [depth, height, width, 3],
        strides: [height * width * 3, width * 3, 3, 1]
      ).contiguous().padded(.zero, begin: [2, 1, 1, 0], end: [0, 1, 1, 0])
    } else {
      depth = min(endDepth - (d * 4 + 1), 8)
      out = input.reshaped(
        [depth + 2, height, width, 3], offset: [d * 4 - 1, 0, 0, 0],
        strides: [height * width * 3, width * 3, 3, 1]
      ).contiguous().padded(.zero, begin: [0, 1, 1, 0], end: [0, 1, 1, 0])
    }
    if let last = outs.last {
      out.add(dependencies: [last])
    }
    out = convIn(out.reshaped([1, depth + 2, height + 2, width + 2, 3])).reshaped([
      depth, height, width, previousChannel,
    ])
    let inputsOnly = startDepth - 1 - d <= 2  // This is the last one.
    var j = 0
    var k = 0
    for (i, channel) in channels.enumerated() {
      for _ in 0..<numRepeat {
        let (mapper, blockOut) = downBlockBuilders[j](
          input: out,
          prefix: "encoder.downsamples.\(k)", inChannels: previousChannel,
          outChannels: channel,
          shortcut: previousChannel != channel, depth: depth, height: height, width: width,
          inputsOnly: inputsOnly)
        mappers.append(mapper)
        out = blockOut
        previousChannel = channel
        j += 1
        k += 1
      }
      if i < channels.count - 1 {
        height /= 2
        width /= 2
        out = downsampleConv2d[i](out.padded(.zero, begin: [0, 0, 0, 0], end: [0, 1, 1, 0]))
        let downLayer = k
        let mapper: ModelWeightMapper = { _ in
          return ModelWeightMapping()
        }
        mappers.append(mapper)
        if i > 0 && startDepth > 1 {
          if d == 0 {
            let first = out.reshaped(
              [1, height, width, channel],
              strides: [height * width * channel, width * channel, channel, 1]
            ).contiguous()
            let shrunk = timeConvs[i - 1](out.reshaped([1, depth, height, width, channel]))
              .reshaped([(depth - 1) / 2, height, width, channel])
            if !inputsOnly {
              let input = out.reshaped(
                [1, height, width, channel], offset: [depth - 1, 0, 0, 0],
                strides: [height * width * channel, width * channel, channel, 1]
              ).copied()
              timeInputs[i - 1] = input
              shrunk.add(dependencies: [input])
            }
            let upLayer = k
            let mapper: ModelWeightMapper = { _ in
              return ModelWeightMapping()
            }
            mappers.append(mapper)
            depth = (depth - 1) / 2 + 1
            out = Functional.concat(axis: 0, first, shrunk)
          } else if let timeInput = timeInputs[i - 1] {
            let shrunk = timeConvs[i - 1](
              Functional.concat(axis: 0, timeInput, out, flags: [.disableOpt]).reshaped([
                1, depth + 1, height, width, channel,
              ]))
            if !inputsOnly {
              let input = out.reshaped(
                [1, height, width, channel], offset: [depth - 1, 0, 0, 0],
                strides: [height * width * channel, width * channel, channel, 1]
              ).copied()
              timeInputs[i - 1] = input
              shrunk.add(dependencies: [input])
            }
            depth = depth / 2
            out = shrunk.reshaped([depth, height, width, channel])
          }
        }
        k += 1
      }
    }
    let (midBlockMapper1, midBlock1Out) = midBlock1Builder(
      input: out,
      prefix: "encoder.middle.0", inChannels: previousChannel,
      outChannels: previousChannel,
      shortcut: false, depth: depth, height: height, width: width, inputsOnly: inputsOnly)
    out = midBlock1Out
    let (midAttnMapper1, midAttn1) = midAttn1Builder(
      prefix: "encoder.middle.1", inChannels: previousChannel, depth: depth,
      height: height, width: width)
    out = midAttn1(out)
    let (midBlockMapper2, midBlock2Out) = midBlock2Builder(
      input: out,
      prefix: "encoder.middle.2", inChannels: previousChannel,
      outChannels: previousChannel,
      shortcut: false, depth: depth, height: height, width: width, inputsOnly: inputsOnly)
    out = midBlock2Out
    out = normOut(out.reshaped([depth, height, width, previousChannel]))
    out = out.swish()
    outs.append(out)
  }
  height = startHeight
  width = startWidth
  depth = startDepth
  var out: Model.IO
  if outs.count > 1 {
    out = Concat(axis: 0)(outs)
  } else {
    out = outs[0]
  }
  let convOut = Convolution(
    groups: 1, filters: 32, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
    format: .OIHW, name: "conv_out")
  out = convOut(
    out.padded(.zero, begin: [2, 1, 1, 0], end: [0, 1, 1, 0]).reshaped([
      1, depth + 2, height + 2, width + 2, previousChannel,
    ])
  ).reshaped([depth, height, width, 32])
  let quantConv = Convolution(
    groups: 1, filters: 32, filterSize: [1, 1], format: .OIHW, name: "quant_conv")
  out = quantConv(out)
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    return mapping
  }
  return (mapper, Model([x], [out]))
}

private struct NCHWResnetBlockCausal3D {
  private let norm1: Model
  private let conv1: Model
  private let norm2: Model
  private let conv2: Model
  private let ninShortcut: Model?
  init(outChannels: Int, shortcut: Bool) {
    norm1 = RMSNorm(epsilon: 1e-6, axis: [0], name: "resnet_norm1")
    conv1 = Convolution(
      groups: 1, filters: outChannels, filterSize: [3, 3, 3],
      hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
      format: .OIHW, name: "resnet_conv1")
    norm2 = RMSNorm(epsilon: 1e-6, axis: [0], name: "resnet_norm2")
    conv2 = Convolution(
      groups: 1, filters: outChannels, filterSize: [3, 3, 3],
      hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
      format: .OIHW, name: "resnet_conv2")
    if shortcut {
      ninShortcut = Convolution(
        groups: 1, filters: outChannels, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]),
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
    var out = norm1(x.reshaped([inChannels, depth, height, width]))
    var pre = out.swish()
    if let conv1Inputs = conv1Inputs {
      out = conv1(
        Functional.concat(axis: 1, conv1Inputs, pre, flags: [.disableOpt]).padded(
          .zero, begin: [0, 0, 1, 1], end: [0, 0, 1, 1]
        ).reshaped([
          1, inChannels, depth + 2, height + 2, width + 2,
        ]))
    } else {
      out = conv1(
        pre.padded(.zero, begin: [0, 2, 1, 1], end: [0, 0, 1, 1]).reshaped([
          1, inChannels, depth + 2, height + 2, width + 2,
        ]))
    }
    if !inputsOnly {
      let inputs = pre.reshaped(
        [inChannels, 2, height, width], offset: [0, depth - 2, 0, 0],
        strides: [depth * height * width, height * width, width, 1]
      ).copied()
      conv1Inputs = inputs
      out.add(dependencies: [inputs])  // This makes sure the copy is done before the convolution, freeing the activations holds there.
    } else {
      conv1Inputs = nil
    }
    out = norm2(out.reshaped([outChannels, depth, height, width]))
    pre = out.swish()
    if let conv2Inputs = conv2Inputs {
      out = conv2(
        Functional.concat(axis: 1, conv2Inputs, pre, flags: [.disableOpt]).padded(
          .zero, begin: [0, 0, 1, 1], end: [0, 0, 1, 1]
        ).reshaped([
          1, outChannels, depth + 2, height + 2, width + 2,
        ]))
    } else {
      out = conv2(
        pre.padded(.zero, begin: [0, 2, 1, 1], end: [0, 0, 1, 1]).reshaped([
          1, outChannels, depth + 2, height + 2, width + 2,
        ]))
    }
    if !inputsOnly {
      let inputs = pre.reshaped(
        [outChannels, 2, height, width], offset: [0, depth - 2, 0, 0],
        strides: [depth * height * width, height * width, width, 1]
      ).copied()
      conv2Inputs = inputs
      out.add(dependencies: [inputs])  // This makes sure the copy is done before the convolution, freeing the activations holds there.
    } else {
      conv2Inputs = nil
    }
    out = out.reshaped([outChannels, depth, height, width])
    if let ninShortcut = ninShortcut {
      out =
        ninShortcut(x.reshaped([1, inChannels, depth, height, width])).reshaped([
          outChannels, depth, height, width,
        ]) + out
    } else {
      out = x + out
    }
    let mapper: ModelWeightMapper = { _ in
      return ModelWeightMapping()
    }
    return (mapper, out)
  }
}

private struct NCHWAttnBlockCausal3D {
  private let norm: Model
  private let toqueries: Model
  private let tokeys: Model
  private let tovalues: Model
  private let projOut: Model
  init(inChannels: Int) {
    norm = RMSNorm(epsilon: 1e-6, axis: [0], name: "attn_norm")
    tokeys = Convolution(
      groups: 1, filters: inChannels, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]),
      format: .OIHW, name: "to_k")
    toqueries = Convolution(
      groups: 1, filters: inChannels, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]),
      format: .OIHW, name: "to_q")
    tovalues = Convolution(
      groups: 1, filters: inChannels, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]),
      format: .OIHW, name: "to_v")
    projOut = Convolution(
      groups: 1, filters: inChannels, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]),
      format: .OIHW, name: "proj_out")
  }
  func callAsFunction(
    prefix: String, inChannels: Int, depth: Int, height: Int, width: Int
  ) -> (
    ModelWeightMapper, Model
  ) {
    let x = Input()
    var out = norm(x.reshaped([inChannels, depth, height, width])).reshaped([
      1, inChannels, depth, height, width,
    ])
    let hw = width * height
    let k = tokeys(out).reshaped([inChannels, depth, hw]).transposed(0, 1)
    let q = ((1.0 / Float(inChannels).squareRoot()) * toqueries(out)).reshaped([
      inChannels, depth, hw,
    ]).transposed(0, 1)
    var dot =
      Matmul(transposeA: (1, 2))(q, k)
    dot = dot.reshaped([depth * hw, hw])
    dot = dot.softmax()
    dot = dot.reshaped([depth, hw, hw])
    let v = tovalues(out).reshaped([inChannels, depth, hw]).transposed(0, 1)
    out = Matmul(transposeB: (1, 2))(v, dot)
    out =
      x
      + projOut(out.transposed(0, 1).reshaped([1, inChannels, depth, height, width])).reshaped([
        inChannels, depth, height, width,
      ])
    let mapper: ModelWeightMapper = { _ in
      return ModelWeightMapping()
    }
    return (mapper, Model([x], [out]))
  }
}

func NCHWWanDecoderCausal3D(
  channels: [Int], numRepeat: Int, startWidth: Int, startHeight: Int,
  startDepth: Int, paddingFinalConvLayer: Bool
)
  -> (ModelWeightMapper, Model)
{
  let x = Input()
  var previousChannel = channels[channels.count - 1]
  let postQuantConv = Convolution(
    groups: 1, filters: 16, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]),
    format: .OIHW, name: "post_quant_conv")
  let postQuantX = postQuantConv(
    x.permuted(3, 0, 1, 2).contiguous().reshaped(
      [1, 16, startDepth, startHeight, startWidth], format: .NCHW)
  ).reshaped([16, startDepth, startHeight, startWidth])
  let convIn = Convolution(
    groups: 1, filters: previousChannel, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
    format: .OIHW, name: "conv_in")
  let convOut = Convolution(
    groups: 1, filters: paddingFinalConvLayer ? 4 : 3, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
    format: .OIHW, name: "conv_out")
  let normOut = RMSNorm(epsilon: 1e-6, axis: [0], name: "norm_out")
  var last: Model.IO? = nil
  var midBlock1Builder = NCHWResnetBlockCausal3D(outChannels: previousChannel, shortcut: false)
  let midAttn1Builder = NCHWAttnBlockCausal3D(inChannels: previousChannel)
  var midBlock2Builder = NCHWResnetBlockCausal3D(outChannels: previousChannel, shortcut: false)
  var upBlockBuilders = [NCHWResnetBlockCausal3D]()
  for (i, channel) in channels.enumerated().reversed() {
    for _ in 0..<numRepeat + 1 {
      upBlockBuilders.append(
        NCHWResnetBlockCausal3D(outChannels: channel, shortcut: previousChannel != channel))
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
      groups: 1, filters: channels[channels.count - i - 1] / 2, filterSize: [1, 3, 3],
      hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
      format: .OIHW, name: "upsample")
  }
  var timeInputs: [Model.IO?] = Array(repeating: nil, count: channels.count - 2)
  var convOutInputs: Model.IO? = nil
  var outs = [Model.IO]()
  for d in stride(from: 0, to: max(startDepth - 1, 1), by: 2) {
    previousChannel = channels[channels.count - 1]
    var out: Model.IO
    if d == 0 {
      out = postQuantX.reshaped(
        [16, min(startDepth - d, 3), startHeight, startWidth], offset: [0, d, 0, 0],
        strides: [startDepth * startHeight * startWidth, startHeight * startWidth, startWidth, 1]
      ).contiguous()
      out = convIn(
        out.padded(.zero, begin: [0, 2, 1, 1], end: [0, 0, 1, 1]).reshaped([
          1, 16, min(startDepth - d, 3) + 2, startHeight + 2, startWidth + 2,
        ])
      ).reshaped([previousChannel, min(startDepth - d, 3), startHeight, startWidth])
    } else {
      out = postQuantX.reshaped(
        [16, min(startDepth - (d - 1), 4), startHeight, startWidth],
        offset: [0, d - 1, 0, 0],
        strides: [startDepth * startHeight * startWidth, startHeight * startWidth, startWidth, 1]
      ).contiguous().padded(.zero, begin: [0, 0, 1, 1], end: [0, 0, 1, 1])
      if let last = last {
        out.add(dependencies: [last])
      }
      out = convIn(
        out.reshaped([
          1, 16, min(startDepth - (d - 1), 4), startHeight + 2, startWidth + 2,
        ])
      ).reshaped([previousChannel, min(startDepth - (d - 1), 4) - 2, startHeight, startWidth])
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
              [channel, 1, height, width],
              strides: [depth * height * width, height * width, width, 1]
            ).contiguous()
            let more = out.reshaped(
              [channel, (depth - 1), height, width], offset: [0, 1, 0, 0],
              strides: [depth * height * width, height * width, width, 1]
            ).contiguous()
            var expanded = timeConvs[channels.count - i - 1](
              more.padded(.zero, begin: [0, 2, 0, 0], end: [0, 0, 0, 0]).reshaped([
                1, channel, depth + 1, height, width,
              ]))
            if !inputsOnly {
              let inputs = out.reshaped(
                [channel, 2, height, width], offset: [0, depth - 2, 0, 0],
                strides: [depth * height * width, height * width, width, 1]
              ).copied()
              timeInputs[channels.count - i - 1] = inputs
              expanded.add(dependencies: [inputs])
            }
            let upLayer = k
            let mapper: ModelWeightMapper = { _ in
              return ModelWeightMapping()
            }
            mappers.append(mapper)
            expanded = expanded.reshaped([2, channel, depth - 1, height, width]).permuted(
              1, 2, 0, 3, 4
            )
            .contiguous().reshaped([channel, 2 * (depth - 1), height, width])
            out = Functional.concat(axis: 1, first, expanded)
            depth = 1 + (depth - 1) * 2
            out = out.reshaped([channel, depth, height, width])
          } else if let timeInput = timeInputs[channels.count - i - 1] {
            let more = out.reshaped([channel, depth, height, width])
            let expanded = timeConvs[channels.count - i - 1](
              Functional.concat(axis: 1, timeInput, more, flags: [.disableOpt]).reshaped([
                1, channel, depth + 2, height, width,
              ]))
            if !inputsOnly {
              let inputs = out.reshaped(
                [channel, 2, height, width], offset: [0, depth - 2, 0, 0],
                strides: [depth * height * width, height * width, width, 1]
              ).copied()
              timeInputs[channels.count - i - 1] = inputs
              expanded.add(dependencies: [inputs])
            }
            let upLayer = k
            let mapper: ModelWeightMapper = { _ in
              return ModelWeightMapping()
            }
            mappers.append(mapper)
            out = expanded.reshaped([2, channel, depth, height, width]).permuted(1, 2, 0, 3, 4)
              .contiguous().reshaped([channel, 2 * depth, height, width])
            depth = depth * 2
          }
        }
        out = Upsample(.nearest, widthScale: 2, heightScale: 2)(
          out.reshaped([channel, depth, height, width])
        ).reshaped([1, channel, depth, height * 2, width * 2])
        width *= 2
        height *= 2
        previousChannel = channel / 2
        out = upsampleConv2d[channels.count - i - 1](out).reshaped([
          previousChannel, depth, height, width,
        ])
        let upLayer = k
        let mapper: ModelWeightMapper = { _ in
          return ModelWeightMapping()
        }
        mappers.append(mapper)
        k += 1
      }
    }
    out = normOut(out.reshaped([channels[0], depth, height, width]))
    let pre = out.swish()
    if let convOutInputs = convOutInputs {
      out = convOut(
        Functional.concat(axis: 1, convOutInputs, pre, flags: [.disableOpt]).padded(
          .zero, begin: [0, 0, 1, 1], end: [0, 0, 1, 1]
        ).reshaped([
          1, channels[0], depth + 2, height + 2, width + 2,
        ])
      )
    } else {
      out = convOut(
        pre.padded(.zero, begin: [0, 2, 1, 1], end: [0, 0, 1, 1]).reshaped([
          1, channels[0], depth + 2, height + 2, width + 2,
        ])
      )
    }
    if !inputsOnly {
      let inputs = pre.reshaped(
        [channels[0], 2, height, width], offset: [0, depth - 2, 0, 0],
        strides: [depth * height * width, height * width, width, 1]
      ).copied()
      convOutInputs = inputs
      out.add(dependencies: [inputs])
    }
    last = out
    outs.append(
      out.reshaped([
        paddingFinalConvLayer ? 4 : 3, depth, height, width,
      ]))
  }
  var out: Model.IO
  if outs.count > 1 {
    let concat = Concat(axis: 1)
    concat.flags = [.disableOpt]
    out = concat(outs)
  } else {
    out = outs[0]
  }
  out = out.permuted(1, 2, 3, 0).contiguous().reshaped(
    .NHWC((startDepth - 1) * 4 + 1, startHeight * 8, startWidth * 8, paddingFinalConvLayer ? 4 : 3))
  let mapper: ModelWeightMapper = { _ in
    return ModelWeightMapping()
  }
  return (mapper, Model([x], [out]))
}

private func NCHWWanEncoderCausal3D(
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
  var downBlockBuilders = [NCHWResnetBlockCausal3D]()
  var timeConvs = [Convolution]()
  var downsampleConv2d = [Convolution]()
  for (i, channel) in channels.enumerated() {
    for _ in 0..<numRepeat {
      downBlockBuilders.append(
        NCHWResnetBlockCausal3D(outChannels: channel, shortcut: previousChannel != channel))
      previousChannel = channel
    }
    if i < channels.count - 1 {
      downsampleConv2d.append(
        Convolution(
          groups: 1, filters: channel, filterSize: [1, 3, 3],
          hint: Hint(
            stride: [1, 2, 2], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
          format: .OIHW, name: "downsample"))
      if i > 0 && startDepth > 1 {
        timeConvs.append(
          Convolution(
            groups: 1, filters: channel, filterSize: [3, 1, 1],
            hint: Hint(
              stride: [2, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
            format: .OIHW, name: "time_conv"))
      }
    }
  }
  var timeInputs: [Model.IO?] = Array(repeating: nil, count: channels.count - 2)
  var midBlock1Builder = NCHWResnetBlockCausal3D(outChannels: previousChannel, shortcut: false)
  var midAttn1Builder = NCHWAttnBlockCausal3D(inChannels: previousChannel)
  var midBlock2Builder = NCHWResnetBlockCausal3D(outChannels: previousChannel, shortcut: false)
  let normOut = RMSNorm(epsilon: 1e-6, axis: [0], name: "norm_out")
  let endHeight = height
  let endWidth = width
  let endDepth = depth
  var outs = [Model.IO]()
  let input = x.permuted(3, 0, 1, 2).contiguous().reshaped(.NCHW(3, endDepth, endHeight, endWidth))  // s4nnc cannot bind tensor properly with offsets. Making a copy to workaround this issue.
  for d in stride(from: 0, to: max(startDepth - 1, 1), by: 2) {
    previousChannel = channels[0]
    height = endHeight
    width = endWidth
    var out: Model.IO
    if d == 0 {
      depth = min(endDepth, 9)
      out = input.reshaped(
        [3, depth, height, width],
        strides: [endDepth * height * width, height * width, width, 1]
      ).contiguous().padded(.zero, begin: [0, 2, 1, 1], end: [0, 0, 1, 1])
    } else {
      depth = min(endDepth - (d * 4 + 1), 8)
      out = input.reshaped(
        [3, depth + 2, height, width], offset: [0, d * 4 - 1, 0, 0],
        strides: [endDepth * height * width, height * width, width, 1]
      ).contiguous().padded(.zero, begin: [0, 0, 1, 1], end: [0, 0, 1, 1])
    }
    if let last = outs.last {
      out.add(dependencies: [last])
    }
    out = convIn(out.reshaped([1, 3, depth + 2, height + 2, width + 2])).reshaped([
      previousChannel, depth, height, width,
    ])
    let inputsOnly = startDepth - 1 - d <= 2  // This is the last one.
    var j = 0
    var k = 0
    for (i, channel) in channels.enumerated() {
      for _ in 0..<numRepeat {
        let (mapper, blockOut) = downBlockBuilders[j](
          input: out,
          prefix: "encoder.downsamples.\(k)", inChannels: previousChannel,
          outChannels: channel,
          shortcut: previousChannel != channel, depth: depth, height: height, width: width,
          inputsOnly: inputsOnly)
        mappers.append(mapper)
        out = blockOut
        previousChannel = channel
        j += 1
        k += 1
      }
      if i < channels.count - 1 {
        out = downsampleConv2d[i](
          out.padded(.zero, begin: [0, 0, 0, 0], end: [0, 0, 1, 1]).reshaped([
            1, previousChannel, depth, height + 1, width + 1,
          ]))
        height /= 2
        width /= 2
        out = out.reshaped([channel, depth, height, width])
        let downLayer = k
        let mapper: ModelWeightMapper = { _ in
          return ModelWeightMapping()
        }
        mappers.append(mapper)
        if i > 0 && startDepth > 1 {
          if d == 0 {
            let first = out.reshaped(
              [channel, 1, height, width],
              strides: [depth * height * width, height * width, width, 1]
            ).contiguous()
            let shrunk = timeConvs[i - 1](out.reshaped([1, channel, depth, height, width]))
              .reshaped([channel, (depth - 1) / 2, height, width])
            if !inputsOnly {
              let input = out.reshaped(
                [channel, 1, height, width], offset: [0, depth - 1, 0, 0],
                strides: [depth * height * width, height * width, width, 1]
              ).copied()
              timeInputs[i - 1] = input
              shrunk.add(dependencies: [input])
            }
            let upLayer = k
            let mapper: ModelWeightMapper = { _ in
              return ModelWeightMapping()
            }
            mappers.append(mapper)
            depth = (depth - 1) / 2 + 1
            out = Functional.concat(axis: 1, first, shrunk)
          } else if let timeInput = timeInputs[i - 1] {
            let shrunk = timeConvs[i - 1](
              Functional.concat(axis: 1, timeInput, out, flags: [.disableOpt]).reshaped([
                1, channel, depth + 1, height, width,
              ]))
            if !inputsOnly {
              let input = out.reshaped(
                [channel, 1, height, width], offset: [0, depth - 1, 0, 0],
                strides: [depth * height * width, height * width, width, 1]
              ).copied()
              timeInputs[i - 1] = input
              shrunk.add(dependencies: [input])
            }
            depth = depth / 2
            out = shrunk.reshaped([channel, depth, height, width])
          }
        }
        k += 1
      }
    }
    let (midBlockMapper1, midBlock1Out) = midBlock1Builder(
      input: out,
      prefix: "encoder.middle.0", inChannels: previousChannel,
      outChannels: previousChannel,
      shortcut: false, depth: depth, height: height, width: width, inputsOnly: inputsOnly)
    out = midBlock1Out
    let (midAttnMapper1, midAttn1) = midAttn1Builder(
      prefix: "encoder.middle.1", inChannels: previousChannel, depth: depth,
      height: height, width: width)
    out = midAttn1(out)
    let (midBlockMapper2, midBlock2Out) = midBlock2Builder(
      input: out,
      prefix: "encoder.middle.2", inChannels: previousChannel,
      outChannels: previousChannel,
      shortcut: false, depth: depth, height: height, width: width, inputsOnly: inputsOnly)
    out = midBlock2Out
    out = normOut(out.reshaped([previousChannel, depth, height, width]))
    out = out.swish()
    outs.append(out)
  }
  height = startHeight
  width = startWidth
  depth = startDepth
  var out: Model.IO
  if outs.count > 1 {
    let concat = Concat(axis: 1)
    concat.flags = [.disableOpt]
    out = concat(outs)
  } else {
    out = outs[0]
  }
  let convOut = Convolution(
    groups: 1, filters: 32, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
    format: .OIHW, name: "conv_out")
  out = convOut(
    out.padded(.zero, begin: [0, 2, 1, 1], end: [0, 0, 1, 1]).reshaped([
      1, previousChannel, depth + 2, height + 2, width + 2,
    ])
  ).reshaped([1, 32, depth, height, width])
  let quantConv = Convolution(
    groups: 1, filters: 32, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]), format: .OIHW,
    name: "quant_conv")
  out = quantConv(out).permuted(0, 2, 3, 4, 1).contiguous().reshaped(
    .NHWC(depth, height, width, 32))
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    return mapping
  }
  return (mapper, Model([x], [out]))
}

func WanDecoderCausal3D(
  channels: [Int], numRepeat: Int, startWidth: Int, startHeight: Int,
  startDepth: Int, paddingFinalConvLayer: Bool, format: TensorFormat
) -> (ModelWeightMapper, Model) {
  switch format {
  case .NHWC:
    return NHWCWanDecoderCausal3D(
      channels: channels, numRepeat: numRepeat, startWidth: startWidth, startHeight: startHeight,
      startDepth: startDepth, paddingFinalConvLayer: paddingFinalConvLayer)
  case .NCHW:
    return NCHWWanDecoderCausal3D(
      channels: channels, numRepeat: numRepeat, startWidth: startWidth, startHeight: startHeight,
      startDepth: startDepth, paddingFinalConvLayer: paddingFinalConvLayer)
  case .CHWN:
    fatalError()
  }
}

func WanEncoderCausal3D(
  channels: [Int], numRepeat: Int, startWidth: Int, startHeight: Int, startDepth: Int,
  format: TensorFormat
) -> (ModelWeightMapper, Model) {
  switch format {
  case .NHWC:
    return NHWCWanEncoderCausal3D(
      channels: channels, numRepeat: numRepeat, startWidth: startWidth, startHeight: startHeight,
      startDepth: startDepth)
  case .NCHW:
    return NCHWWanEncoderCausal3D(
      channels: channels, numRepeat: numRepeat, startWidth: startWidth, startHeight: startHeight,
      startDepth: startDepth)
  case .CHWN:
    fatalError()
  }
}
