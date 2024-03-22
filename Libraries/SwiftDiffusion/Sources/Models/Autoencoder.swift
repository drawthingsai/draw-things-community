import NNC

/// Autoencoder

func ResnetBlock(prefix: (String, String), outChannels: Int, shortcut: Bool)
  -> (Model, PythonReader, ModelWeightMapper)
{
  let x = Input()
  let norm1 = GroupNorm(axis: 3, groups: 32, epsilon: 1e-6, reduce: [1, 2])
  var out = norm1(x)
  out = out.swish()
  let conv1 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = conv1(out)
  let norm2 = GroupNorm(axis: 3, groups: 32, epsilon: 1e-6, reduce: [1, 2])
  out = norm2(out)
  out = out.swish()
  let conv2 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = conv2(out)
  let ninShortcut: Model?
  if shortcut {
    let nin = Convolution(
      groups: 1, filters: outChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW
    )
    out = nin(x) + out
    ninShortcut = nin
  } else {
    out = x + out
    ninShortcut = nil
  }
  let reader: PythonReader = { stateDict, archive in
    guard let norm1_weight = stateDict["first_stage_model.\(prefix.0).norm1.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let norm1_bias = stateDict["first_stage_model.\(prefix.0).norm1.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try norm1.parameters(for: .weight).copy(from: norm1_weight, zip: archive, of: FloatType.self)
    try norm1.parameters(for: .bias).copy(from: norm1_bias, zip: archive, of: FloatType.self)
    guard let conv1_weight = stateDict["first_stage_model.\(prefix.0).conv1.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let conv1_bias = stateDict["first_stage_model.\(prefix.0).conv1.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try conv1.parameters(for: .weight).copy(from: conv1_weight, zip: archive, of: FloatType.self)
    try conv1.parameters(for: .bias).copy(from: conv1_bias, zip: archive, of: FloatType.self)
    guard let norm2_weight = stateDict["first_stage_model.\(prefix.0).norm2.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let norm2_bias = stateDict["first_stage_model.\(prefix.0).norm2.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try norm2.parameters(for: .weight).copy(from: norm2_weight, zip: archive, of: FloatType.self)
    try norm2.parameters(for: .bias).copy(from: norm2_bias, zip: archive, of: FloatType.self)
    guard let conv2_weight = stateDict["first_stage_model.\(prefix.0).conv2.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let conv2_bias = stateDict["first_stage_model.\(prefix.0).conv2.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try conv2.parameters(for: .weight).copy(from: conv2_weight, zip: archive, of: FloatType.self)
    try conv2.parameters(for: .bias).copy(from: conv2_bias, zip: archive, of: FloatType.self)
    if let ninShortcut = ninShortcut {
      guard let nin_shortcut_weight = stateDict["first_stage_model.\(prefix.0).nin_shortcut.weight"]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard let nin_shortcut_bias = stateDict["first_stage_model.\(prefix.0).nin_shortcut.bias"]
      else {
        throw UnpickleError.tensorNotFound
      }
      try ninShortcut.parameters(for: .weight).copy(
        from: nin_shortcut_weight, zip: archive, of: FloatType.self)
      try ninShortcut.parameters(for: .bias).copy(
        from: nin_shortcut_bias, zip: archive, of: FloatType.self)
    }
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = [String: [String]]()
    switch format {
    case .generativeModels:
      mapping["first_stage_model.\(prefix.0).norm1.weight"] = [norm1.weight.name]
      mapping["first_stage_model.\(prefix.0).norm1.bias"] = [norm1.bias.name]
      mapping["first_stage_model.\(prefix.0).conv1.weight"] = [conv1.weight.name]
      mapping["first_stage_model.\(prefix.0).conv1.bias"] = [conv1.bias.name]
      mapping["first_stage_model.\(prefix.0).norm2.weight"] = [norm2.weight.name]
      mapping["first_stage_model.\(prefix.0).norm2.bias"] = [norm2.bias.name]
      mapping["first_stage_model.\(prefix.0).conv2.weight"] = [conv2.weight.name]
      mapping["first_stage_model.\(prefix.0).conv2.bias"] = [conv2.bias.name]
      if let ninShortcut = ninShortcut {
        mapping["first_stage_model.\(prefix.0).nin_shortcut.weight"] = [ninShortcut.weight.name]
        mapping["first_stage_model.\(prefix.0).nin_shortcut.bias"] = [ninShortcut.bias.name]
      }
    case .diffusers:
      mapping["\(prefix.1).norm1.weight"] = [norm1.weight.name]
      mapping["\(prefix.1).norm1.bias"] = [norm1.bias.name]
      mapping["\(prefix.1).conv1.weight"] = [conv1.weight.name]
      mapping["\(prefix.1).conv1.bias"] = [conv1.bias.name]
      mapping["\(prefix.1).norm2.weight"] = [norm2.weight.name]
      mapping["\(prefix.1).norm2.bias"] = [norm2.bias.name]
      mapping["\(prefix.1).conv2.weight"] = [conv2.weight.name]
      mapping["\(prefix.1).conv2.bias"] = [conv2.bias.name]
      if let ninShortcut = ninShortcut {
        mapping["\(prefix.1).conv_shortcut.weight"] = [ninShortcut.weight.name]
        mapping["\(prefix.1).conv_shortcut.bias"] = [ninShortcut.bias.name]
      }
    }
    return mapping

  }
  return (Model([x], [out]), reader, mapper)
}

func AttnBlock(
  prefix: (String, String), inChannels: Int, batchSize: Int, width: Int, height: Int,
  usesFlashAttention: Bool
) -> (
  Model, PythonReader, ModelWeightMapper
) {
  let x = Input()
  let norm = GroupNorm(axis: 3, groups: 32, epsilon: 1e-6, reduce: [1, 2])
  var out = norm(x)
  let hw = width * height
  let toqueries = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW)
  let tokeys = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW)
  let tovalues = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW)
  let projOut: Model
  if usesFlashAttention {
    let q = toqueries(out).reshaped([batchSize, hw, inChannels]).identity().identity()
    let k = tokeys(out).reshaped([batchSize, hw, inChannels]).identity()
    let v = tovalues(out).reshaped([batchSize, hw, inChannels])
    projOut = ScaledDotProductAttention(
      scale: 1.0 / Float(inChannels).squareRoot(), multiHeadOutputProjectionFused: true)
    out = projOut(q, k, v).reshaped([batchSize, height, width, inChannels])
    out = x + out
  } else {
    let k = tokeys(out).reshaped([batchSize, hw, inChannels])
    let q = ((1.0 / Float(inChannels).squareRoot()) * toqueries(out)).reshaped([
      batchSize, hw, inChannels,
    ])
    let v = tovalues(out).reshaped([batchSize, hw, inChannels])
    var dot = Matmul(transposeB: (1, 2))(q, k)
    dot = dot.reshaped([batchSize * hw, hw])
    dot = dot.softmax()
    dot = dot.reshaped([batchSize, hw, hw])
    out = dot * v
    projOut = Convolution(
      groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW)
    out = x + projOut(out.reshaped([batchSize, height, width, inChannels]))
  }
  let reader: PythonReader = { stateDict, archive in
    guard let norm_weight = stateDict["first_stage_model.\(prefix.0).norm.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let norm_bias = stateDict["first_stage_model.\(prefix.0).norm.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try norm.parameters(for: .weight).copy(from: norm_weight, zip: archive, of: FloatType.self)
    try norm.parameters(for: .bias).copy(from: norm_bias, zip: archive, of: FloatType.self)
    guard let k_weight = stateDict["first_stage_model.\(prefix.0).k.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let k_bias = stateDict["first_stage_model.\(prefix.0).k.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try tokeys.parameters(for: .weight).copy(from: k_weight, zip: archive, of: FloatType.self)
    try tokeys.parameters(for: .bias).copy(from: k_bias, zip: archive, of: FloatType.self)
    guard let q_weight = stateDict["first_stage_model.\(prefix.0).q.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let q_bias = stateDict["first_stage_model.\(prefix.0).q.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try toqueries.parameters(for: .weight).copy(from: q_weight, zip: archive, of: FloatType.self)
    try toqueries.parameters(for: .bias).copy(from: q_bias, zip: archive, of: FloatType.self)
    guard let v_weight = stateDict["first_stage_model.\(prefix.0).v.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let v_bias = stateDict["first_stage_model.\(prefix.0).v.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try tovalues.parameters(for: .weight).copy(from: v_weight, zip: archive, of: FloatType.self)
    try tovalues.parameters(for: .bias).copy(from: v_bias, zip: archive, of: FloatType.self)
    guard let proj_out_weight = stateDict["first_stage_model.\(prefix.0).proj_out.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let proj_out_bias = stateDict["first_stage_model.\(prefix.0).proj_out.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try projOut.parameters(for: .weight).copy(
      from: proj_out_weight, zip: archive, of: FloatType.self)
    try projOut.parameters(for: .bias).copy(from: proj_out_bias, zip: archive, of: FloatType.self)
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = [String: [String]]()
    switch format {
    case .generativeModels:
      mapping["first_stage_model.\(prefix.0).norm.weight"] = [norm.weight.name]
      mapping["first_stage_model.\(prefix.0).norm.bias"] = [norm.bias.name]
      mapping["first_stage_model.\(prefix.0).q.weight"] = [toqueries.weight.name]
      mapping["first_stage_model.\(prefix.0).q.bias"] = [toqueries.bias.name]
      mapping["first_stage_model.\(prefix.0).k.weight"] = [tokeys.weight.name]
      mapping["first_stage_model.\(prefix.0).k.bias"] = [tokeys.bias.name]
      mapping["first_stage_model.\(prefix.0).v.weight"] = [tovalues.weight.name]
      mapping["first_stage_model.\(prefix.0).v.bias"] = [tovalues.bias.name]
      mapping["first_stage_model.\(prefix.0).proj_out.weight"] = [projOut.weight.name]
      mapping["first_stage_model.\(prefix.0).proj_out.bias"] = [projOut.bias.name]
    case .diffusers:
      mapping["\(prefix.1).group_norm.weight"] = [norm.weight.name]
      mapping["\(prefix.1).group_norm.bias"] = [norm.bias.name]
      mapping["\(prefix.1).to_q.weight"] = [toqueries.weight.name]
      mapping["\(prefix.1).to_q.bias"] = [toqueries.bias.name]
      mapping["\(prefix.1).to_k.weight"] = [tokeys.weight.name]
      mapping["\(prefix.1).to_k.bias"] = [tokeys.bias.name]
      mapping["\(prefix.1).to_v.weight"] = [tovalues.weight.name]
      mapping["\(prefix.1).to_v.bias"] = [tovalues.bias.name]
      mapping["\(prefix.1).to_out.0.weight"] = [projOut.weight.name]
      mapping["\(prefix.1).to_out.0.bias"] = [projOut.bias.name]
    }
    return mapping

  }
  return (Model([x], [out]), reader, mapper)
}

public func Encoder(
  channels: [Int], numRepeat: Int, batchSize: Int, startWidth: Int, startHeight: Int,
  usesFlashAttention: Bool
)
  -> (Model, PythonReader, ModelWeightMapper)
{
  let x = Input()
  var previousChannel = channels[0]
  let convIn = Convolution(
    groups: 1, filters: previousChannel, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  var out = convIn(x)
  var readers = [PythonReader]()
  var mappers = [ModelWeightMapper]()
  var height = startHeight
  var width = startWidth
  for _ in 1..<channels.count {
    height *= 2
    width *= 2
  }
  for (i, channel) in channels.enumerated() {
    for j in 0..<numRepeat {
      let (block, reader, mapper) = ResnetBlock(
        prefix: ("encoder.down.\(i).block.\(j)", "encoder.down_blocks.\(i).resnets.\(j)"),
        outChannels: channel,
        shortcut: previousChannel != channel)
      readers.append(reader)
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
      let conv2d = Convolution(
        groups: 1, filters: channel, filterSize: [3, 3],
        hint: Hint(stride: [2, 2], border: Hint.Border(begin: [2, 2], end: [1, 1])), format: .OIHW)
      out = conv2d(out).reshaped(
        [batchSize, height, width, channel], offset: [0, 1, 1, 0],
        strides: [channel * (height + 1) * (width + 1), (width + 1) * channel, channel, 1])
      let downLayer = i
      let reader: PythonReader = { stateDict, archive in
        guard
          let conv_weight = stateDict[
            "first_stage_model.encoder.down.\(downLayer).downsample.conv.weight"]
        else {
          throw UnpickleError.tensorNotFound
        }
        guard
          let conv_bias = stateDict[
            "first_stage_model.encoder.down.\(downLayer).downsample.conv.bias"]
        else {
          throw UnpickleError.tensorNotFound
        }
        try conv2d.parameters(for: .weight).copy(
          from: conv_weight, zip: archive, of: FloatType.self)
        try conv2d.parameters(for: .bias).copy(from: conv_bias, zip: archive, of: FloatType.self)
      }
      readers.append(reader)
      let mapper: ModelWeightMapper = { format in
        var mapping = [String: [String]]()
        switch format {
        case .generativeModels:
          mapping["first_stage_model.encoder.down.\(downLayer).downsample.conv.weight"] = [
            conv2d.weight.name
          ]
          mapping["first_stage_model.encoder.down.\(downLayer).downsample.conv.bias"] = [
            conv2d.bias.name
          ]
        case .diffusers:
          mapping["encoder.down_blocks.\(downLayer).downsamplers.0.conv.weight"] = [
            conv2d.weight.name
          ]
          mapping["encoder.down_blocks.\(downLayer).downsamplers.0.conv.bias"] = [conv2d.bias.name]
        }
        return mapping
      }
      mappers.append(mapper)
    }
  }
  let (midBlock1, midBlockReader1, midBlockMapper1) = ResnetBlock(
    prefix: ("encoder.mid.block_1", "encoder.mid_block.resnets.0"), outChannels: previousChannel,
    shortcut: false)
  out = midBlock1(out)
  let (midAttn1, midAttnReader1, midAttnMapper1) = AttnBlock(
    prefix: ("encoder.mid.attn_1", "encoder.mid_block.attentions.0"), inChannels: previousChannel,
    batchSize: batchSize,
    width: startWidth, height: startHeight, usesFlashAttention: usesFlashAttention)
  out = midAttn1(out)
  let (midBlock2, midBlockReader2, midBlockMapper2) = ResnetBlock(
    prefix: ("encoder.mid.block_2", "encoder.mid_block.resnets.1"), outChannels: previousChannel,
    shortcut: false)
  out = midBlock2(out)
  let normOut = GroupNorm(axis: 3, groups: 32, epsilon: 1e-6, reduce: [1, 2])
  out = normOut(out)
  out = out.swish()
  let convOut = Convolution(
    groups: 1, filters: 8, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = convOut(out)
  let quantConv2d = Convolution(
    groups: 1, filters: 8, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW)
  out = quantConv2d(out)
  let reader: PythonReader = { stateDict, archive in
    guard let conv_in_weight = stateDict["first_stage_model.encoder.conv_in.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let conv_in_bias = stateDict["first_stage_model.encoder.conv_in.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try convIn.parameters(for: .weight).copy(from: conv_in_weight, zip: archive, of: FloatType.self)
    try convIn.parameters(for: .bias).copy(from: conv_in_bias, zip: archive, of: FloatType.self)
    for reader in readers {
      try reader(stateDict, archive)
    }
    try midBlockReader1(stateDict, archive)
    try midAttnReader1(stateDict, archive)
    try midBlockReader2(stateDict, archive)
    guard let norm_out_weight = stateDict["first_stage_model.encoder.norm_out.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let norm_out_bias = stateDict["first_stage_model.encoder.norm_out.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try normOut.parameters(for: .weight).copy(
      from: norm_out_weight, zip: archive, of: FloatType.self)
    try normOut.parameters(for: .bias).copy(from: norm_out_bias, zip: archive, of: FloatType.self)
    guard let conv_out_weight = stateDict["first_stage_model.encoder.conv_out.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let conv_out_bias = stateDict["first_stage_model.encoder.conv_out.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try convOut.parameters(for: .weight).copy(
      from: conv_out_weight, zip: archive, of: FloatType.self)
    try convOut.parameters(for: .bias).copy(from: conv_out_bias, zip: archive, of: FloatType.self)
    guard let quant_conv_weight = stateDict["first_stage_model.quant_conv.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let quant_conv_bias = stateDict["first_stage_model.quant_conv.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try quantConv2d.parameters(for: .weight).copy(
      from: quant_conv_weight, zip: archive, of: FloatType.self)
    try quantConv2d.parameters(for: .bias).copy(
      from: quant_conv_bias, zip: archive, of: FloatType.self)
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = [String: [String]]()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping.merge(midBlockMapper1(format)) { v, _ in v }
    mapping.merge(midAttnMapper1(format)) { v, _ in v }
    mapping.merge(midBlockMapper2(format)) { v, _ in v }
    switch format {
    case .generativeModels:
      mapping["first_stage_model.encoder.conv_in.weight"] = [convIn.weight.name]
      mapping["first_stage_model.encoder.conv_in.bias"] = [convIn.bias.name]
      mapping["first_stage_model.encoder.norm_out.weight"] = [normOut.weight.name]
      mapping["first_stage_model.encoder.norm_out.bias"] = [normOut.bias.name]
      mapping["first_stage_model.encoder.conv_out.weight"] = [convOut.weight.name]
      mapping["first_stage_model.encoder.conv_out.bias"] = [convOut.bias.name]
      mapping["first_stage_model.quant_conv.weight"] = [quantConv2d.weight.name]
      mapping["first_stage_model.quant_conv.bias"] = [quantConv2d.bias.name]
    case .diffusers:
      mapping["encoder.conv_in.weight"] = [convIn.weight.name]
      mapping["encoder.conv_in.bias"] = [convIn.bias.name]
      mapping["encoder.conv_norm_out.weight"] = [normOut.weight.name]
      mapping["encoder.conv_norm_out.bias"] = [normOut.bias.name]
      mapping["encoder.conv_out.weight"] = [convOut.weight.name]
      mapping["encoder.conv_out.bias"] = [convOut.bias.name]
      mapping["quant_conv.weight"] = [quantConv2d.weight.name]
      mapping["quant_conv.bias"] = [quantConv2d.bias.name]
    }
    return mapping

  }
  return (Model([x], [out]), reader, mapper)
}

public func Decoder(
  channels: [Int], numRepeat: Int, batchSize: Int, startWidth: Int, startHeight: Int,
  usesFlashAttention: Bool, paddingFinalConvLayer: Bool
)
  -> (Model, PythonReader, ModelWeightMapper)
{
  let x = Input()
  let postQuantConv2d = Convolution(
    groups: 1, filters: 4, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW)
  var out = postQuantConv2d(x)
  var previousChannel = channels[channels.count - 1]
  let convIn = Convolution(
    groups: 1, filters: previousChannel, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = convIn(out)
  let (midBlock1, midBlockReader1, midBlockMapper1) = ResnetBlock(
    prefix: ("decoder.mid.block_1", "decoder.mid_block.resnets.0"), outChannels: previousChannel,
    shortcut: false)
  out = midBlock1(out)
  let (midAttn1, midAttnReader1, midAttnMapper1) = AttnBlock(
    prefix: ("decoder.mid.attn_1", "decoder.mid_block.attentions.0"), inChannels: previousChannel,
    batchSize: batchSize,
    width: startWidth, height: startHeight, usesFlashAttention: usesFlashAttention)
  out = midAttn1(out)
  let (midBlock2, midBlockReader2, midBlockMapper2) = ResnetBlock(
    prefix: ("decoder.mid.block_2", "decoder.mid_block.resnets.1"), outChannels: previousChannel,
    shortcut: false)
  out = midBlock2(out)
  var readers = [PythonReader]()
  var mappers = [ModelWeightMapper]()
  for (i, channel) in channels.enumerated().reversed() {
    for j in 0..<numRepeat + 1 {
      let (block, reader, mapper) = ResnetBlock(
        prefix: (
          "decoder.up.\(i).block.\(j)", "decoder.up_blocks.\(channels.count - 1 - i).resnets.\(j)"
        ), outChannels: channel,
        shortcut: previousChannel != channel)
      readers.append(reader)
      mappers.append(mapper)
      out = block(out)
      previousChannel = channel
    }
    if i > 0 {
      out = Upsample(.nearest, widthScale: 2, heightScale: 2)(out)
      let conv2d = Convolution(
        groups: 1, filters: channel, filterSize: [3, 3],
        hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
      out = conv2d(out)
      let upLayer = i
      let reader: PythonReader = { stateDict, archive in
        guard
          let conv_weight = stateDict[
            "first_stage_model.decoder.up.\(upLayer).upsample.conv.weight"]
        else {
          throw UnpickleError.tensorNotFound
        }
        guard
          let conv_bias = stateDict["first_stage_model.decoder.up.\(upLayer).upsample.conv.bias"]
        else {
          throw UnpickleError.tensorNotFound
        }
        try conv2d.parameters(for: .weight).copy(
          from: conv_weight, zip: archive, of: FloatType.self)
        try conv2d.parameters(for: .bias).copy(from: conv_bias, zip: archive, of: FloatType.self)
      }
      readers.append(reader)
      let mapper: ModelWeightMapper = { format in
        var mapping = [String: [String]]()
        switch format {
        case .generativeModels:
          mapping["first_stage_model.decoder.up.\(upLayer).upsample.conv.weight"] = [
            conv2d.weight.name
          ]
          mapping["first_stage_model.decoder.up.\(upLayer).upsample.conv.bias"] = [conv2d.bias.name]
        case .diffusers:
          mapping["decoder.up_blocks.\(channels.count - 1 - upLayer).upsamplers.0.conv.weight"] = [
            conv2d.weight.name
          ]
          mapping["decoder.up_blocks.\(channels.count - 1 - upLayer).upsamplers.0.conv.bias"] = [
            conv2d.bias.name
          ]
        }
        return mapping
      }
      mappers.append(mapper)
    }
  }
  let normOut = GroupNorm(axis: 3, groups: 32, epsilon: 1e-6, reduce: [1, 2])
  out = normOut(out)
  out = out.swish()
  let convOut = Convolution(
    groups: 1, filters: paddingFinalConvLayer ? 4 : 3, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = convOut(out)
  let reader: PythonReader = { stateDict, archive in
    guard let post_quant_conv_weight = stateDict["first_stage_model.post_quant_conv.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let post_quant_conv_bias = stateDict["first_stage_model.post_quant_conv.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try postQuantConv2d.parameters(for: .weight).copy(
      from: post_quant_conv_weight, zip: archive, of: FloatType.self)
    try postQuantConv2d.parameters(for: .bias).copy(
      from: post_quant_conv_bias, zip: archive, of: FloatType.self)
    guard let conv_in_weight = stateDict["first_stage_model.decoder.conv_in.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let conv_in_bias = stateDict["first_stage_model.decoder.conv_in.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try convIn.parameters(for: .weight).copy(from: conv_in_weight, zip: archive, of: FloatType.self)
    try convIn.parameters(for: .bias).copy(from: conv_in_bias, zip: archive, of: FloatType.self)
    try midBlockReader1(stateDict, archive)
    try midAttnReader1(stateDict, archive)
    try midBlockReader2(stateDict, archive)
    for reader in readers {
      try reader(stateDict, archive)
    }
    guard let norm_out_weight = stateDict["first_stage_model.decoder.norm_out.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let norm_out_bias = stateDict["first_stage_model.decoder.norm_out.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try normOut.parameters(for: .weight).copy(
      from: norm_out_weight, zip: archive, of: FloatType.self)
    try normOut.parameters(for: .bias).copy(from: norm_out_bias, zip: archive, of: FloatType.self)
    guard let conv_out_weight = stateDict["first_stage_model.decoder.conv_out.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let conv_out_bias = stateDict["first_stage_model.decoder.conv_out.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try convOut.parameters(for: .weight).copy(
      from: conv_out_weight, zip: archive, of: FloatType.self)
    try convOut.parameters(for: .bias).copy(from: conv_out_bias, zip: archive, of: FloatType.self)
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = [String: [String]]()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping.merge(midBlockMapper1(format)) { v, _ in v }
    mapping.merge(midAttnMapper1(format)) { v, _ in v }
    mapping.merge(midBlockMapper2(format)) { v, _ in v }
    switch format {
    case .generativeModels:
      mapping["first_stage_model.post_quant_conv.weight"] = [postQuantConv2d.weight.name]
      mapping["first_stage_model.post_quant_conv.bias"] = [postQuantConv2d.bias.name]
      mapping["first_stage_model.decoder.conv_in.weight"] = [convIn.weight.name]
      mapping["first_stage_model.decoder.conv_in.bias"] = [convIn.bias.name]
      mapping["first_stage_model.decoder.norm_out.weight"] = [normOut.weight.name]
      mapping["first_stage_model.decoder.norm_out.bias"] = [normOut.bias.name]
      mapping["first_stage_model.decoder.conv_out.weight"] = [convOut.weight.name]
      mapping["first_stage_model.decoder.conv_out.bias"] = [convOut.bias.name]
    case .diffusers:
      mapping["post_quant_conv.weight"] = [postQuantConv2d.weight.name]
      mapping["post_quant_conv.bias"] = [postQuantConv2d.bias.name]
      mapping["decoder.conv_in.weight"] = [convIn.weight.name]
      mapping["decoder.conv_in.bias"] = [convIn.bias.name]
      mapping["decoder.conv_norm_out.weight"] = [normOut.weight.name]
      mapping["decoder.conv_norm_out.bias"] = [normOut.bias.name]
      mapping["decoder.conv_out.weight"] = [convOut.weight.name]
      mapping["decoder.conv_out.bias"] = [convOut.bias.name]
    }
    return mapping

  }
  return (Model([x], [out]), reader, mapper)
}
