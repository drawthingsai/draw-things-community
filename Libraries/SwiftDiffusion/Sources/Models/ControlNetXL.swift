import NNC

func InputBlocks<FloatType: TensorNumeric & BinaryFloatingPoint>(
  channels: [Int], numRepeat: Int, numHeadChannels: Int, batchSize: Int, startHeight: Int,
  startWidth: Int, embeddingLength: (Int, Int), attentionRes: [Int: [Int]],
  injectIPAdapterLengths: [Int], upcastAttention: [Int: [Int]],
  usesFlashAttention: FlashAttentionLevel,
  isTemporalMixEnabled: Bool, x: Model.IO, hint: Model.IO, emb: Model.IO,
  of: FloatType.Type = FloatType.self
) -> (PythonReader, ModelWeightMapper, [(Model.IO, Int)], Model.IO, [Input]) {
  let conv2d = Convolution(
    groups: 1, filters: channels[0], filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  var out = conv2d(x) + hint
  var layerStart = 1
  var height = startHeight
  var width = startWidth
  var previousChannel = channels[0]
  var ds = 1
  var passLayers = [(out, previousChannel)]
  var kvs = [Input]()
  var readers = [PythonReader]()
  var mappers = [ModelWeightMapper]()
  for (i, channel) in channels.enumerated() {
    let attentionBlock = attentionRes[ds, default: Array(repeating: 0, count: numRepeat)]
    let upcastAttention = Set(upcastAttention[ds, default: []])
    for j in 0..<numRepeat {
      let (reader, mapper, inputLayer) = BlockLayer(
        prefix: ("model.control_model.input_blocks.\(layerStart)", "down_blocks.\(i)"),
        repeatStart: j, skipConnection: previousChannel != channel,
        attentionBlock: attentionBlock[j], channels: channel, numHeadChannels: numHeadChannels,
        batchSize: batchSize, height: height, width: width, embeddingLength: embeddingLength,
        intermediateSize: channel * 4, injectIPAdapterLengths: injectIPAdapterLengths,
        upcastAttention: upcastAttention.contains(j), usesFlashAttention: usesFlashAttention,
        flags: .Float16, isTemporalMixEnabled: isTemporalMixEnabled, of: FloatType.self)
      previousChannel = channel
      var c: [Input]
      if embeddingLength.0 == 1 && embeddingLength.1 == 1 {
        c = (0..<(attentionBlock[j] * (injectIPAdapterLengths.count + 1))).map { _ in Input() }
        if isTemporalMixEnabled && attentionBlock[j] > 0 {
          c.append(contentsOf: (0..<(attentionBlock[j] + 1)).map { _ in Input() })
        }
      } else {
        c = (0..<(attentionBlock[j] * (injectIPAdapterLengths.count + 1) * 2)).map { _ in Input() }
        if isTemporalMixEnabled && attentionBlock[j] > 0 {
          c.append(contentsOf: (0..<(attentionBlock[j] * 2 + 1)).map { _ in Input() })
        }
      }
      out = inputLayer([out, emb] + c)
      kvs.append(contentsOf: c)
      passLayers.append((out, channel))
      readers.append(reader)
      mappers.append(mapper)
      layerStart += 1
    }
    if i != channels.count - 1 {
      let downsample = Convolution(
        groups: 1, filters: channel, filterSize: [3, 3],
        hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [0, 0])), format: .OIHW)
      out = downsample(out)
      passLayers.append((out, channel))
      let downLayer = layerStart
      let reader: PythonReader = { stateDict, archive in
        guard
          let op_weight = stateDict["model.control_model.input_blocks.\(downLayer).0.op.weight"]
        else {
          throw UnpickleError.tensorNotFound
        }
        guard let op_bias = stateDict["model.control_model.input_blocks.\(downLayer).0.op.bias"]
        else {
          throw UnpickleError.tensorNotFound
        }
        try downsample.weight.copy(from: op_weight, zip: archive, of: FloatType.self)
        try downsample.bias.copy(from: op_bias, zip: archive, of: FloatType.self)
      }
      readers.append(reader)
      let mapper: ModelWeightMapper = { format in
        switch format {
        case .generativeModels:
          return [
            "model.control_model.input_blocks.\(downLayer).0.op.weight": [downsample.weight.name],
            "model.control_model.input_blocks.\(downLayer).0.op.bias": [downsample.bias.name],
          ]
        case .diffusers:
          return [
            "down_blocks.\(i).downsamplers.0.conv.weight": [downsample.weight.name],
            "down_blocks.\(i).downsamplers.0.conv.bias": [downsample.bias.name],
          ]
        }
      }
      mappers.append(mapper)
      height = height / 2
      width = width / 2
      layerStart += 1
      ds *= 2
    }
  }
  let reader: PythonReader = { stateDict, archive in
    guard let input_blocks_0_0_weight = stateDict["model.control_model.input_blocks.0.0.weight"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard let input_blocks_0_0_bias = stateDict["model.control_model.input_blocks.0.0.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    try conv2d.weight.copy(from: input_blocks_0_0_weight, zip: archive, of: FloatType.self)
    try conv2d.bias.copy(from: input_blocks_0_0_bias, zip: archive, of: FloatType.self)
    for reader in readers {
      try reader(stateDict, archive)
    }
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    switch format {
    case .generativeModels:
      mapping["model.control_model.input_blocks.0.0.weight"] = [conv2d.weight.name]
      mapping["model.control_model.input_blocks.0.0.bias"] = [conv2d.bias.name]
    case .diffusers:
      mapping["conv_in.weight"] = [conv2d.weight.name]
      mapping["conv_in.bias"] = [conv2d.bias.name]
    }
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    return mapping
  }
  return (reader, mapper, passLayers, out, kvs)
}

public func ControlNetXL(
  batchSize: Int, startWidth: Int, startHeight: Int, channels: [Int], embeddingLength: (Int, Int),
  inputAttentionRes: KeyValuePairs<Int, [Int]>, middleAttentionBlocks: Int,
  usesFlashAttention: FlashAttentionLevel
) -> (Model, PythonReader, ModelWeightMapper) {
  let x = Input()
  let hint = Input()
  let t_emb = Input()
  let y = Input()
  let inputAttentionRes = [Int: [Int]](
    uniqueKeysWithValues: inputAttentionRes.map { ($0.key, $0.value) })
  let (timeFc0, timeFc2, timeEmbed) = TimeEmbed(modelChannels: channels[0])
  let (labelFc0, labelFc2, labelEmbed) = LabelEmbed(modelChannels: channels[0])
  let emb = timeEmbed(t_emb) + labelEmbed(y)
  let middleBlockSizeMult = 1 << (channels.count - 1)
  let (inputReader, inputMapper, inputs, inputBlocks, inputKVs) = InputBlocks(
    channels: channels, numRepeat: 2, numHeadChannels: 64, batchSize: batchSize,
    startHeight: startHeight, startWidth: startWidth, embeddingLength: embeddingLength,
    attentionRes: inputAttentionRes, injectIPAdapterLengths: [],
    upcastAttention: [:], usesFlashAttention: usesFlashAttention,
    isTemporalMixEnabled: false, x: x, hint: hint, emb: emb, of: FloatType.self)
  var out = inputBlocks
  let (middleReader, middleMapper, middleBlock, middleKVs) = MiddleBlock(
    prefix: "model.control_model",
    channels: channels.last!, numHeadChannels: 64, batchSize: batchSize,
    height: startHeight / middleBlockSizeMult, width: startWidth / middleBlockSizeMult,
    embeddingLength: embeddingLength, attentionBlock: middleAttentionBlocks,
    retainingNormProjInProjOutAndSecondResNetWhenNoAttentionBlocks: true,  // For ControlNet SDXL Small, even without attention blocks, we still have the second resnet, norm, proj_in, proj_out in the network.
    injectIPAdapterLengths: [], upcastAttention: false,
    usesFlashAttention: usesFlashAttention, isTemporalMixEnabled: false, x: out,
    emb: emb, of: FloatType.self)
  out = middleBlock
  var zeroConvs = [Model]()
  var outputs = [Model.IO]()
  for i in 0..<inputs.count {
    let channel = inputs[i].1
    let zeroConv = Convolution(
      groups: 1, filters: channel, filterSize: [1, 1],
      hint: Hint(stride: [1, 1]), format: .OIHW)
    outputs.append(zeroConv(inputs[i].0))
    zeroConvs.append(zeroConv)
  }
  let middleBlockOut = Convolution(
    groups: 1, filters: 1280, filterSize: [1, 1],
    hint: Hint(stride: [1, 1]), format: .OIHW)
  outputs.append(middleBlockOut(out))
  let reader: PythonReader = { stateDict, archive in
    guard let time_embed_0_weight = stateDict["model.control_model.time_embed.0.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let time_embed_0_bias = stateDict["model.control_model.time_embed.0.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let time_embed_2_weight = stateDict["model.control_model.time_embed.2.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let time_embed_2_bias = stateDict["model.control_model.time_embed.2.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try timeFc0.weight.copy(from: time_embed_0_weight, zip: archive, of: FloatType.self)
    try timeFc0.bias.copy(from: time_embed_0_bias, zip: archive, of: FloatType.self)
    try timeFc2.weight.copy(from: time_embed_2_weight, zip: archive, of: FloatType.self)
    try timeFc2.bias.copy(from: time_embed_2_bias, zip: archive, of: FloatType.self)
    guard let label_emb_0_0_weight = stateDict["model.control_model.label_emb.0.0.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let label_emb_0_0_bias = stateDict["model.control_model.label_emb.0.0.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let label_emb_0_2_weight = stateDict["model.control_model.label_emb.0.2.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let label_emb_0_2_bias = stateDict["model.control_model.label_emb.0.2.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try labelFc0.weight.copy(from: label_emb_0_0_weight, zip: archive, of: FloatType.self)
    try labelFc0.bias.copy(from: label_emb_0_0_bias, zip: archive, of: FloatType.self)
    try labelFc2.weight.copy(from: label_emb_0_2_weight, zip: archive, of: FloatType.self)
    try labelFc2.bias.copy(from: label_emb_0_2_bias, zip: archive, of: FloatType.self)
    try inputReader(stateDict, archive)
    try middleReader(stateDict, archive)
    for (i, zeroConv) in zeroConvs.enumerated() {
      guard
        let zero_convs_weight =
          stateDict["model.control_model.zero_convs.\(i).0.weight"]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard
        let zero_convs_bias =
          stateDict["model.control_model.zero_convs.\(i).0.bias"]
      else {
        throw UnpickleError.tensorNotFound
      }
      try zeroConv.weight.copy(from: zero_convs_weight, zip: archive, of: FloatType.self)
      try zeroConv.bias.copy(from: zero_convs_bias, zip: archive, of: FloatType.self)
    }
    guard
      let middle_block_out_0_weight =
        stateDict["model.control_model.middle_block_out.0.weight"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let middle_block_out_0_bias =
        stateDict["model.control_model.middle_block_out.0.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    try middleBlockOut.weight.copy(
      from: middle_block_out_0_weight, zip: archive, of: FloatType.self)
    try middleBlockOut.bias.copy(from: middle_block_out_0_bias, zip: archive, of: FloatType.self)
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping.merge(inputMapper(format)) { v, _ in v }
    mapping.merge(middleMapper(format)) { v, _ in v }
    switch format {
    case .generativeModels:
      mapping["model.control_model.time_embed.0.weight"] = [timeFc0.weight.name]
      mapping["model.control_model.time_embed.0.bias"] = [timeFc0.bias.name]
      mapping["model.control_model.time_embed.2.weight"] = [timeFc2.weight.name]
      mapping["model.control_model.time_embed.2.bias"] = [timeFc2.bias.name]
      mapping["model.control_model.label_emb.0.0.weight"] = [labelFc0.weight.name]
      mapping["model.control_model.label_emb.0.0.bias"] = [labelFc0.bias.name]
      mapping["model.control_model.label_emb.0.2.weight"] = [labelFc2.weight.name]
      mapping["model.control_model.label_emb.0.2.bias"] = [labelFc2.bias.name]
      for (i, zeroConv) in zeroConvs.enumerated() {
        mapping["model.control_model.zero_convs.\(i).0.weight"] = [zeroConv.weight.name]
        mapping["model.control_model.zero_convs.\(i).0.bias"] = [zeroConv.bias.name]
      }
      mapping["model.control_model.middle_block_out.0.weight"] = [middleBlockOut.weight.name]
      mapping["model.control_model.middle_block_out.0.bias"] = [middleBlockOut.bias.name]
    case .diffusers:
      mapping["time_embedding.linear_1.weight"] = [timeFc0.weight.name]
      mapping["time_embedding.linear_1.bias"] = [timeFc0.bias.name]
      mapping["time_embedding.linear_2.weight"] = [timeFc2.weight.name]
      mapping["time_embedding.linear_2.bias"] = [timeFc2.bias.name]
      mapping["add_embedding.linear_1.weight"] = [labelFc0.weight.name]
      mapping["add_embedding.linear_1.bias"] = [labelFc0.bias.name]
      mapping["add_embedding.linear_2.weight"] = [labelFc2.weight.name]
      mapping["add_embedding.linear_2.bias"] = [labelFc2.bias.name]
      for (i, zeroConv) in zeroConvs.enumerated() {
        mapping["controlnet_down_blocks.\(i).weight"] = [zeroConv.weight.name]
        mapping["controlnet_down_blocks.\(i).bias"] = [zeroConv.bias.name]
      }
      mapping["controlnet_mid_block.weight"] = [middleBlockOut.weight.name]
      mapping["controlnet_mid_block.bias"] = [middleBlockOut.bias.name]
    }
    return mapping
  }
  return (Model([x, hint, t_emb, y] + inputKVs + middleKVs, outputs), reader, mapper)
}

public func ControlNetXLFixed(
  batchSize: Int, startHeight: Int, startWidth: Int, channels: [Int], embeddingLength: (Int, Int),
  inputAttentionRes: KeyValuePairs<Int, [Int]>, middleAttentionBlocks: Int,
  usesFlashAttention: FlashAttentionLevel
) -> (Model, PythonReader, ModelWeightMapper) {
  let c = Input()
  let inputAttentionRes = [Int: [Int]](
    uniqueKeysWithValues: inputAttentionRes.map { ($0.key, $0.value) })
  let (inputReader, inputMapper, inputBlocks) = InputBlocksFixed(
    prefix: "model.control_model",
    channels: channels, numRepeat: 2, numHeadChannels: 64, batchSize: batchSize,
    startHeight: startHeight, startWidth: startWidth, embeddingLength: embeddingLength,
    attentionRes: inputAttentionRes, usesFlashAttention: usesFlashAttention,
    isTemporalMixEnabled: false, c: c, numFrames: [])
  var out = inputBlocks
  let middleBlockSizeMult = 1 << (channels.count - 1)
  let middleReader: PythonReader?
  let middleMapper: ModelWeightMapper?
  if middleAttentionBlocks > 0 {
    let (reader, mapper, middleBlock) = MiddleBlockFixed(
      prefix: "model.control_model",
      channels: channels.last!, numHeadChannels: 64, batchSize: batchSize,
      height: startHeight / middleBlockSizeMult, width: startWidth / middleBlockSizeMult,
      embeddingLength: embeddingLength, attentionBlock: middleAttentionBlocks,
      usesFlashAttention: usesFlashAttention, isTemporalMixEnabled: false, c: c,
      numFrames: [])
    out.append(middleBlock)
    middleReader = reader
    middleMapper = mapper
  } else {
    middleReader = nil
    middleMapper = nil
  }
  let reader: PythonReader = { stateDict, archive in
    try inputReader(stateDict, archive)
    try middleReader?(stateDict, archive)
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping.merge(inputMapper(format)) { v, _ in v }
    if let middleMapper = middleMapper {
      mapping.merge(middleMapper(format)) { v, _ in v }
    }
    return mapping
  }
  return (Model([c], out), reader, mapper)
}

func LoRAInputBlocks(
  channels: [Int], numRepeat: Int, numHeadChannels: Int, batchSize: Int, startHeight: Int,
  startWidth: Int, embeddingLength: (Int, Int), attentionRes: [Int: [Int]],
  injectIPAdapterLengths: [Int], upcastAttention: [Int: [Int]],
  usesFlashAttention: FlashAttentionLevel,
  LoRAConfiguration: LoRANetworkConfiguration,
  x: Model.IO, hint: Model.IO, emb: Model.IO
) -> (ModelWeightMapper, [(Model.IO, Int)], Model.IO, [Input]) {
  let conv2d = LoRAConvolution(
    groups: 1, filters: channels[0], filterSize: [3, 3], configuration: LoRAConfiguration,
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  var out = conv2d(x) + hint
  var layerStart = 1
  var height = startHeight
  var width = startWidth
  var previousChannel = channels[0]
  var ds = 1
  var passLayers = [(out, previousChannel)]
  var kvs = [Input]()
  var mappers = [ModelWeightMapper]()
  for (i, channel) in channels.enumerated() {
    let attentionBlock = attentionRes[ds, default: Array(repeating: 0, count: numRepeat)]
    let upcastAttention = Set(upcastAttention[ds, default: []])
    for j in 0..<numRepeat {
      let (mapper, inputLayer) = LoRABlockLayer(
        prefix: ("model.control_model.input_blocks.\(layerStart)", "down_blocks.\(i)"),
        repeatStart: j, skipConnection: previousChannel != channel,
        attentionBlock: attentionBlock[j], channels: channel, numHeadChannels: numHeadChannels,
        batchSize: batchSize, height: height, width: width, embeddingLength: embeddingLength,
        intermediateSize: channel * 4, injectIPAdapterLengths: injectIPAdapterLengths,
        upcastAttention: upcastAttention.contains(j), usesFlashAttention: usesFlashAttention,
        LoRAConfiguration: LoRAConfiguration)
      previousChannel = channel
      var c: [Input]
      if embeddingLength.0 == 1 && embeddingLength.1 == 1 {
        c = (0..<(attentionBlock[j] * (injectIPAdapterLengths.count + 1))).map { _ in Input() }
      } else {
        c = (0..<(attentionBlock[j] * (injectIPAdapterLengths.count + 1) * 2)).map { _ in Input() }
      }
      out = inputLayer([out, emb] + c)
      kvs.append(contentsOf: c)
      passLayers.append((out, channel))
      mappers.append(mapper)
      layerStart += 1
    }
    if i != channels.count - 1 {
      let downsample = LoRAConvolution(
        groups: 1, filters: channel, filterSize: [3, 3], configuration: LoRAConfiguration,
        hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [0, 0])), format: .OIHW)
      out = downsample(out)
      passLayers.append((out, channel))
      let downLayer = layerStart
      let mapper: ModelWeightMapper = { format in
        switch format {
        case .generativeModels:
          return [
            "model.control_model.input_blocks.\(downLayer).0.op.down": [
              downsample.parameters(for: .index(0)).name
            ],
            "model.control_model.input_blocks.\(downLayer).0.op.up": [
              downsample.parameters(for: .index(1)).name
            ],
            "model.control_model.input_blocks.\(downLayer).0.op.weight": [
              downsample.parameters(for: .index(2)).name
            ],
            "model.control_model.input_blocks.\(downLayer).0.op.bias": [downsample.bias.name],
          ]
        case .diffusers:
          return [
            "down_blocks.\(i).downsamplers.0.conv.down": [
              downsample.parameters(for: .index(0)).name
            ],
            "down_blocks.\(i).downsamplers.0.conv.up": [downsample.parameters(for: .index(1)).name],
            "down_blocks.\(i).downsamplers.0.conv.weight": [
              downsample.parameters(for: .index(2)).name
            ],
            "down_blocks.\(i).downsamplers.0.conv.bias": [downsample.bias.name],
          ]
        }
      }
      mappers.append(mapper)
      height = height / 2
      width = width / 2
      layerStart += 1
      ds *= 2
    }
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    switch format {
    case .generativeModels:
      mapping["model.control_model.input_blocks.0.0.down"] = [
        conv2d.parameters(for: .index(0)).name
      ]
      mapping["model.control_model.input_blocks.0.0.up"] = [conv2d.parameters(for: .index(1)).name]
      mapping["model.control_model.input_blocks.0.0.weight"] = [
        conv2d.parameters(for: .index(2)).name
      ]
      mapping["model.control_model.input_blocks.0.0.bias"] = [conv2d.bias.name]
    case .diffusers:
      mapping["conv_in.down"] = [conv2d.parameters(for: .index(0)).name]
      mapping["conv_in.up"] = [conv2d.parameters(for: .index(1)).name]
      mapping["conv_in.weight"] = [conv2d.parameters(for: .index(2)).name]
      mapping["conv_in.bias"] = [conv2d.bias.name]
    }
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    return mapping
  }
  return (mapper, passLayers, out, kvs)
}

public func LoRAControlNetXL(
  batchSize: Int, startWidth: Int, startHeight: Int, channels: [Int], embeddingLength: (Int, Int),
  inputAttentionRes: KeyValuePairs<Int, [Int]>, middleAttentionBlocks: Int,
  usesFlashAttention: FlashAttentionLevel,
  LoRAConfiguration: LoRANetworkConfiguration
) -> (Model, ModelWeightMapper) {
  let x = Input()
  let hint = Input()
  let t_emb = Input()
  let y = Input()
  let inputAttentionRes = [Int: [Int]](
    uniqueKeysWithValues: inputAttentionRes.map { ($0.key, $0.value) })
  let (timeFc0, timeFc2, timeEmbed) = LoRATimeEmbed(
    modelChannels: channels[0], LoRAConfiguration: LoRAConfiguration)
  let (labelFc0, labelFc2, labelEmbed) = LoRALabelEmbed(
    modelChannels: channels[0], LoRAConfiguration: LoRAConfiguration)
  let emb = timeEmbed(t_emb) + labelEmbed(y)
  let middleBlockSizeMult = 1 << (channels.count - 1)
  let (inputMapper, inputs, inputBlocks, inputKVs) = LoRAInputBlocks(
    channels: channels, numRepeat: 2, numHeadChannels: 64, batchSize: batchSize,
    startHeight: startHeight, startWidth: startWidth, embeddingLength: embeddingLength,
    attentionRes: inputAttentionRes, injectIPAdapterLengths: [],
    upcastAttention: [:], usesFlashAttention: usesFlashAttention,
    LoRAConfiguration: LoRAConfiguration, x: x, hint: hint, emb: emb)
  var out = inputBlocks
  let (middleMapper, middleBlock, middleKVs) = LoRAMiddleBlock(
    prefix: "model.control_model",
    channels: channels.last!, numHeadChannels: 64, batchSize: batchSize,
    height: startHeight / middleBlockSizeMult, width: startWidth / middleBlockSizeMult,
    embeddingLength: embeddingLength, attentionBlock: middleAttentionBlocks,
    retainingNormProjInProjOutAndSecondResNetWhenNoAttentionBlocks: true,  // For ControlNet SDXL Small, even without attention blocks, we still have the second resnet, norm, proj_in, proj_out in the network.
    injectIPAdapterLengths: [], upcastAttention: false,
    usesFlashAttention: usesFlashAttention, LoRAConfiguration: LoRAConfiguration, x: out, emb: emb)
  out = middleBlock
  var zeroConvs = [Model]()
  var outputs = [Model.IO]()
  for i in 0..<inputs.count {
    let channel = inputs[i].1
    let zeroConv = Convolution(
      groups: 1, filters: channel, filterSize: [1, 1],
      hint: Hint(stride: [1, 1]), format: .OIHW)
    outputs.append(zeroConv(inputs[i].0))
    zeroConvs.append(zeroConv)
  }
  let middleBlockOut = Convolution(
    groups: 1, filters: 1280, filterSize: [1, 1],
    hint: Hint(stride: [1, 1]), format: .OIHW)
  outputs.append(middleBlockOut(out))
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping.merge(inputMapper(format)) { v, _ in v }
    mapping.merge(middleMapper(format)) { v, _ in v }
    switch format {
    case .generativeModels:
      mapping["model.control_model.time_embed.0.down"] = [timeFc0.parameters(for: .index(0)).name]
      mapping["model.control_model.time_embed.0.up"] = [timeFc0.parameters(for: .index(1)).name]
      mapping["model.control_model.time_embed.0.weight"] = [timeFc0.parameters(for: .index(2)).name]
      mapping["model.control_model.time_embed.0.bias"] = [timeFc0.bias.name]
      mapping["model.control_model.time_embed.2.down"] = [timeFc2.parameters(for: .index(0)).name]
      mapping["model.control_model.time_embed.2.up"] = [timeFc2.parameters(for: .index(1)).name]
      mapping["model.control_model.time_embed.2.weight"] = [timeFc2.parameters(for: .index(2)).name]
      mapping["model.control_model.time_embed.2.bias"] = [timeFc2.bias.name]
      mapping["model.control_model.label_emb.0.0.down"] = [labelFc0.parameters(for: .index(0)).name]
      mapping["model.control_model.label_emb.0.0.up"] = [labelFc0.parameters(for: .index(1)).name]
      mapping["model.control_model.label_emb.0.0.weight"] = [
        labelFc0.parameters(for: .index(2)).name
      ]
      mapping["model.control_model.label_emb.0.0.bias"] = [labelFc0.bias.name]
      mapping["model.control_model.label_emb.0.2.down"] = [labelFc2.parameters(for: .index(0)).name]
      mapping["model.control_model.label_emb.0.2.up"] = [labelFc2.parameters(for: .index(1)).name]
      mapping["model.control_model.label_emb.0.2.weight"] = [
        labelFc2.parameters(for: .index(2)).name
      ]
      mapping["model.control_model.label_emb.0.2.bias"] = [labelFc2.bias.name]
      for (i, zeroConv) in zeroConvs.enumerated() {
        mapping["model.control_model.zero_convs.\(i).0.weight"] = [zeroConv.weight.name]
        mapping["model.control_model.zero_convs.\(i).0.bias"] = [zeroConv.bias.name]
      }
      mapping["model.control_model.middle_block_out.0.weight"] = [middleBlockOut.weight.name]
      mapping["model.control_model.middle_block_out.0.bias"] = [middleBlockOut.bias.name]
    case .diffusers:
      mapping["time_embedding.linear_1.down"] = [timeFc0.parameters(for: .index(0)).name]
      mapping["time_embedding.linear_1.up"] = [timeFc0.parameters(for: .index(1)).name]
      mapping["time_embedding.linear_1.weight"] = [timeFc0.parameters(for: .index(2)).name]
      mapping["time_embedding.linear_1.bias"] = [timeFc0.bias.name]
      mapping["time_embedding.linear_2.down"] = [timeFc2.parameters(for: .index(0)).name]
      mapping["time_embedding.linear_2.up"] = [timeFc2.parameters(for: .index(1)).name]
      mapping["time_embedding.linear_2.weight"] = [timeFc2.parameters(for: .index(2)).name]
      mapping["time_embedding.linear_2.bias"] = [timeFc2.bias.name]
      mapping["add_embedding.linear_1.down"] = [labelFc0.parameters(for: .index(0)).name]
      mapping["add_embedding.linear_1.up"] = [labelFc0.parameters(for: .index(1)).name]
      mapping["add_embedding.linear_1.weight"] = [labelFc0.parameters(for: .index(2)).name]
      mapping["add_embedding.linear_1.bias"] = [labelFc0.bias.name]
      mapping["add_embedding.linear_2.down"] = [labelFc2.parameters(for: .index(0)).name]
      mapping["add_embedding.linear_2.up"] = [labelFc2.parameters(for: .index(1)).name]
      mapping["add_embedding.linear_2.weight"] = [labelFc2.parameters(for: .index(2)).name]
      mapping["add_embedding.linear_2.bias"] = [labelFc2.bias.name]
      for (i, zeroConv) in zeroConvs.enumerated() {
        mapping["controlnet_down_blocks.\(i).weight"] = [zeroConv.weight.name]
        mapping["controlnet_down_blocks.\(i).bias"] = [zeroConv.bias.name]
      }
      mapping["controlnet_mid_block.weight"] = [middleBlockOut.weight.name]
      mapping["controlnet_mid_block.bias"] = [middleBlockOut.bias.name]
    }
    return mapping
  }
  return (Model([x, hint, t_emb, y] + inputKVs + middleKVs, outputs), mapper)
}

public func LoRAControlNetXLFixed(
  batchSize: Int, startHeight: Int, startWidth: Int, channels: [Int], embeddingLength: (Int, Int),
  inputAttentionRes: KeyValuePairs<Int, [Int]>, middleAttentionBlocks: Int,
  usesFlashAttention: FlashAttentionLevel,
  LoRAConfiguration: LoRANetworkConfiguration
) -> (Model, ModelWeightMapper) {
  let c = Input()
  let inputAttentionRes = [Int: [Int]](
    uniqueKeysWithValues: inputAttentionRes.map { ($0.key, $0.value) })
  let (inputMapper, inputBlocks) = LoRAInputBlocksFixed(
    prefix: "model.control_model",
    channels: channels, numRepeat: 2, numHeadChannels: 64, batchSize: batchSize,
    startHeight: startHeight, startWidth: startWidth, embeddingLength: embeddingLength,
    attentionRes: inputAttentionRes, usesFlashAttention: usesFlashAttention,
    LoRAConfiguration: LoRAConfiguration, c: c)
  var out = inputBlocks
  let middleBlockSizeMult = 1 << (channels.count - 1)
  let middleMapper: ModelWeightMapper?
  if middleAttentionBlocks > 0 {
    let (mapper, middleBlock) = LoRAMiddleBlockFixed(
      prefix: "model.control_model",
      channels: channels.last!, numHeadChannels: 64, batchSize: batchSize,
      height: startHeight / middleBlockSizeMult, width: startWidth / middleBlockSizeMult,
      embeddingLength: embeddingLength, attentionBlock: middleAttentionBlocks,
      usesFlashAttention: usesFlashAttention, LoRAConfiguration: LoRAConfiguration, c: c)
    out.append(middleBlock)
    middleMapper = mapper
  } else {
    middleMapper = nil
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping.merge(inputMapper(format)) { v, _ in v }
    if let middleMapper = middleMapper {
      mapping.merge(middleMapper(format)) { v, _ in v }
    }
    return mapping
  }
  return (Model([c], out), mapper)
}
