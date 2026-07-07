import Foundation
import NNC

public func Krea2RotaryPositionEmbedding<FloatType: TensorNumeric & BinaryFloatingPoint>(
  textLength: Int, gridHeight: Int, gridWidth: Int, of dataType: FloatType.Type = FloatType.self
) -> Tensor<FloatType> {
  let ropeTheta: Double = 1_000
  let imageLength = gridHeight * gridWidth
  let tokenLength = imageLength + textLength
  var rotary = Tensor<FloatType>(.CPU, .NHWC(1, tokenLength, 1, 128))
  let dim0 = 32
  let dim1 = 48
  let dim2 = 48
  for y in 0..<gridHeight {
    for x in 0..<gridWidth {
      let i = y * gridWidth + x
      for k in 0..<(dim0 / 2) {
        let theta = 0 * 1.0 / pow(ropeTheta, Double(k) * 2 / Double(dim0))
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotary[0, i, 0, k * 2] = FloatType(costheta)
        rotary[0, i, 0, k * 2 + 1] = FloatType(sintheta)
      }
      for k in 0..<(dim1 / 2) {
        let theta = Double(y) * 1.0 / pow(ropeTheta, Double(k) * 2 / Double(dim1))
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotary[0, i, 0, (k + (dim0 / 2)) * 2] = FloatType(costheta)
        rotary[0, i, 0, (k + (dim0 / 2)) * 2 + 1] = FloatType(sintheta)
      }
      for k in 0..<(dim2 / 2) {
        let theta = Double(x) * 1.0 / pow(ropeTheta, Double(k) * 2 / Double(dim2))
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotary[0, i, 0, (k + (dim0 / 2) + (dim1 / 2)) * 2] = FloatType(costheta)
        rotary[0, i, 0, (k + (dim0 / 2) + (dim1 / 2)) * 2 + 1] = FloatType(sintheta)
      }
    }
  }
  let tokenOffset = imageLength
  for i in 0..<textLength {
    for k in 0..<(dim0 / 2) {
      let theta = 0 * 1.0 / pow(ropeTheta, Double(k) * 2 / Double(dim0))
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotary[0, i + tokenOffset, 0, k * 2] = FloatType(costheta)
      rotary[0, i + tokenOffset, 0, k * 2 + 1] = FloatType(sintheta)
    }
    for k in 0..<(dim1 / 2) {
      let theta = 0 * 1.0 / pow(ropeTheta, Double(k) * 2 / Double(dim1))
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotary[0, i + tokenOffset, 0, (k + (dim0 / 2)) * 2] = FloatType(costheta)
      rotary[0, i + tokenOffset, 0, (k + (dim0 / 2)) * 2 + 1] = FloatType(sintheta)
    }
    for k in 0..<(dim2 / 2) {
      let theta = 0 * 1.0 / pow(ropeTheta, Double(k) * 2 / Double(dim2))
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotary[0, i + tokenOffset, 0, (k + (dim0 / 2) + (dim1 / 2)) * 2] = FloatType(costheta)
      rotary[0, i + tokenOffset, 0, (k + (dim0 / 2) + (dim1 / 2)) * 2 + 1] =
        FloatType(sintheta)
    }
  }
  return rotary
}

private func Krea2SwiGLU(hiddenSize: Int, intermediateSize: Int, name: String = "") -> Model {
  let x = Input()
  let gate = Dense(count: intermediateSize, noBias: true, name: "\(name)gate")
  let up = Dense(count: intermediateSize, noBias: true, name: "\(name)up")
  let down = Dense(count: hiddenSize, noBias: true, name: "\(name)down")
  let out = down(gate(x).swish() .* up(x))
  return Model([x], [out])
}

private func Krea2RepeatKeyValueHeads(
  _ x: Model.IO, batchSize: Int, tokenLength: Int, keyValueHeads: Int, heads: Int, headDim: Int
) -> Model.IO {
  precondition(heads % keyValueHeads == 0)
  guard heads != keyValueHeads else { return x }
  let repeats = heads / keyValueHeads
  var parts = [Model.IO]()
  parts.reserveCapacity(heads)
  for head in 0..<keyValueHeads {
    let slice = x.reshaped(
      [batchSize, tokenLength, 1, headDim], offset: [0, 0, head, 0],
      strides: [tokenLength * keyValueHeads * headDim, keyValueHeads * headDim, headDim, 1]
    ).contiguous()
    for _ in 0..<repeats {
      parts.append(slice)
    }
  }
  return Concat(axis: 2)(parts)
}

private func Krea2ExplicitAttention(
  queries: Model.IO, keys: Model.IO, values: Model.IO, batchSize: Int, queryLength: Int,
  keyValueLength: Int, heads: Int, keyValueHeads: Int, headDim: Int
) -> Model.IO {
  let repeatedKeys = Krea2RepeatKeyValueHeads(
    keys, batchSize: batchSize, tokenLength: keyValueLength, keyValueHeads: keyValueHeads,
    heads: heads, headDim: headDim)
  let repeatedValues = Krea2RepeatKeyValueHeads(
    values, batchSize: batchSize, tokenLength: keyValueLength, keyValueHeads: keyValueHeads,
    heads: heads, headDim: headDim)
  let scaledQueries = ((1.0 / Float(headDim).squareRoot()) * queries).transposed(1, 2)
    .contiguous()
  let transposedKeys = repeatedKeys.transposed(1, 2).contiguous()
  let transposedValues = repeatedValues.transposed(1, 2).contiguous()
  var dot = Matmul(transposeB: (2, 3))(scaledQueries, transposedKeys)
  dot = dot.reshaped([batchSize * heads * queryLength, keyValueLength]).softmax()
  dot = dot.reshaped([batchSize, heads, queryLength, keyValueLength])
  var out = dot * transposedValues
  out = out.reshaped([batchSize, heads, queryLength, headDim]).transposed(1, 2)
  return out.reshaped([batchSize, queryLength, heads * headDim])
}

private func Krea2Attention(
  hiddenSize: Int, heads: Int, keyValueHeads: Int, batchSize: Int, tokenLength: Int, rotary: Bool,
  queryLength: Int? = nil, segments: [Int] = [], usesFlashAttention: FlashAttentionLevel,
  namePrefix: String = ""
) -> Model {
  let x = Input()
  let rot = rotary ? Input() : nil
  let queryLength = queryLength ?? tokenLength
  precondition(queryLength <= tokenLength)
  precondition(
    segments.isEmpty || (queryLength == tokenLength && segments.reduce(0, +) == tokenLength))
  let headDim = hiddenSize / heads
  let toQ = Dense(count: headDim * heads, noBias: true, name: "\(namePrefix)to_q")
  let toK = Dense(
    count: headDim * keyValueHeads, noBias: true, name: "\(namePrefix)to_k")
  let toV = Dense(
    count: headDim * keyValueHeads, noBias: true, name: "\(namePrefix)to_v")
  let toGate = Dense(count: hiddenSize, noBias: true, name: "\(namePrefix)to_gate")
  let queryIn: Model.IO =
    queryLength < tokenLength
    ? x.reshaped(
      [batchSize, queryLength, hiddenSize], offset: [0, 0, 0],
      strides: [tokenLength * hiddenSize, hiddenSize, 1]
    ).contiguous() : x
  var queries = toQ(queryIn).reshaped([batchSize, queryLength, heads, headDim])
  let normQ = RMSNorm(epsilon: 1e-5, axis: [3], name: "\(namePrefix)norm_q")
  queries = normQ(queries)
  var keys = toK(x).reshaped([batchSize, tokenLength, keyValueHeads, headDim])
  let normK = RMSNorm(epsilon: 1e-5, axis: [3], name: "\(namePrefix)norm_k")
  keys = normK(keys)
  let values = toV(x).reshaped([batchSize, tokenLength, keyValueHeads, headDim])
  if let rot = rot {
    let queryRot: Model.IO =
      queryLength < tokenLength
      ? rot.reshaped(
        [1, queryLength, 1, headDim], offset: [0, 0, 0, 0],
        strides: [tokenLength * headDim, headDim, headDim, 1]
      ) : rot
    queries = Functional.cmul(left: queries, right: queryRot)
    keys = Functional.cmul(left: keys, right: rot)
  }
  let attention: Model.IO
  switch usesFlashAttention {
  case .none:
    if segments.count > 1 {
      var offset = 0
      var outs = [Model.IO]()
      for segment in segments {
        let query = queries.reshaped(
          [batchSize, segment, heads, headDim], offset: [0, offset, 0, 0],
          strides: [queryLength * heads * headDim, heads * headDim, headDim, 1]
        ).contiguous()
        let key = keys.reshaped(
          [batchSize, segment, keyValueHeads, headDim], offset: [0, offset, 0, 0],
          strides: [tokenLength * keyValueHeads * headDim, keyValueHeads * headDim, headDim, 1]
        ).contiguous()
        let value = values.reshaped(
          [batchSize, segment, keyValueHeads, headDim], offset: [0, offset, 0, 0],
          strides: [tokenLength * keyValueHeads * headDim, keyValueHeads * headDim, headDim, 1]
        ).contiguous()
        outs.append(
          Krea2ExplicitAttention(
            queries: query, keys: key, values: value, batchSize: batchSize,
            queryLength: segment, keyValueLength: segment, heads: heads,
            keyValueHeads: keyValueHeads, headDim: headDim))
        offset += segment
      }
      let concat = Concat(axis: 1)
      concat.flags = .disableOpt
      attention = concat(outs)
    } else {
      attention = Krea2ExplicitAttention(
        queries: queries, keys: keys, values: values, batchSize: batchSize,
        queryLength: queryLength, keyValueLength: tokenLength, heads: heads,
        keyValueHeads: keyValueHeads, headDim: headDim)
    }
  case .scale1:
    let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
    if segments.count > 1 {
      var offset = 0
      var outs = [Model.IO]()
      for segment in segments {
        let query = queries.reshaped(
          [batchSize, segment, heads, headDim], offset: [0, offset, 0, 0],
          strides: [queryLength * heads * headDim, heads * headDim, headDim, 1]
        ).contiguous()
        let key = keys.reshaped(
          [batchSize, segment, keyValueHeads, headDim], offset: [0, offset, 0, 0],
          strides: [tokenLength * keyValueHeads * headDim, keyValueHeads * headDim, headDim, 1]
        ).contiguous()
        let value = values.reshaped(
          [batchSize, segment, keyValueHeads, headDim], offset: [0, offset, 0, 0],
          strides: [tokenLength * keyValueHeads * headDim, keyValueHeads * headDim, headDim, 1]
        ).contiguous()
        outs.append(
          scaledDotProductAttention((1.0 / Float(headDim).squareRoot()) * query, key, value))
        offset += segment
      }
      let concat = Concat(axis: 1)
      concat.flags = .disableOpt
      attention = concat(outs).reshaped([batchSize, queryLength, hiddenSize])
    } else {
      attention = scaledDotProductAttention(
        (1.0 / Float(headDim).squareRoot()) * queries, keys, values
      ).reshaped([batchSize, queryLength, hiddenSize])
    }
  case .scaleMerged, .quantized:
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1.0 / Float(headDim).squareRoot(),
      flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
    if segments.count > 1 {
      var offset = 0
      var outs = [Model.IO]()
      for segment in segments {
        let query = queries.reshaped(
          [batchSize, segment, heads, headDim], offset: [0, offset, 0, 0],
          strides: [queryLength * heads * headDim, heads * headDim, headDim, 1]
        ).contiguous()
        let key = keys.reshaped(
          [batchSize, segment, keyValueHeads, headDim], offset: [0, offset, 0, 0],
          strides: [tokenLength * keyValueHeads * headDim, keyValueHeads * headDim, headDim, 1]
        ).contiguous()
        let value = values.reshaped(
          [batchSize, segment, keyValueHeads, headDim], offset: [0, offset, 0, 0],
          strides: [tokenLength * keyValueHeads * headDim, keyValueHeads * headDim, headDim, 1]
        ).contiguous()
        outs.append(scaledDotProductAttention(query, key, value))
        offset += segment
      }
      let concat = Concat(axis: 1)
      concat.flags = .disableOpt
      attention = concat(outs).reshaped([
        batchSize, queryLength, hiddenSize,
      ])
    } else {
      attention = scaledDotProductAttention(queries, keys, values).reshaped([
        batchSize, queryLength, hiddenSize,
      ])
    }
  }
  let gate = toGate(queryIn).sigmoid()
  let toOut = Dense(count: hiddenSize, noBias: true, name: "\(namePrefix)to_out")
  let out = toOut(attention .* gate)
  if let rot = rot {
    return Model([x, rot], [out])
  }
  return Model([x], [out])
}

private func Krea2TextFusionBlock(
  batchSize: Int, tokenLength: Int, segments: [Int] = [],
  usesFlashAttention: FlashAttentionLevel, namePrefix: String
)
  -> Model
{
  let x = Input()
  let norm1 = RMSNorm(epsilon: 1e-5, axis: [2], name: "\(namePrefix)norm1")
  let attention = Krea2Attention(
    hiddenSize: 2_560, heads: 20, keyValueHeads: 20, batchSize: batchSize,
    tokenLength: tokenLength, rotary: false, segments: segments,
    usesFlashAttention: usesFlashAttention,
    namePrefix: "\(namePrefix)attn_")
  var out = x + attention(norm1(x).to(.Float16)).to(of: x)
  let norm2 = RMSNorm(epsilon: 1e-5, axis: [2], name: "\(namePrefix)norm2")
  let ff = Krea2SwiGLU(hiddenSize: 2_560, intermediateSize: 6_912, name: "\(namePrefix)ff_")
  out = out + ff(norm2(out).to(.Float16)).to(of: out)
  return Model([x], [out])
}

private func Krea2TextFusion(
  batchSize: Int, textLength: (Int, Int), usesFlashAttention: FlashAttentionLevel
) -> Model {
  let x = Input()
  let totalTextLength = textLength.0 + textLength.1
  let segments = textLength.0 > 0 ? [textLength.0, textLength.1] : []
  var out = x.reshaped([batchSize * totalTextLength, 12, 2_560])
  for _ in 0..<2 {
    let block = Krea2TextFusionBlock(
      batchSize: batchSize * totalTextLength, tokenLength: 12,
      usesFlashAttention: usesFlashAttention,
      namePrefix: "text_fusion_layerwise_")
    out = block(out)
  }
  out = out.reshaped([batchSize, totalTextLength, 12, 2_560]).permuted(0, 1, 3, 2)
  let projector = Dense(count: 1, noBias: true, name: "text_fusion_projector")
  out = out.reshaped([batchSize * totalTextLength * 2_560, 12])
  out = projector(out).reshaped([batchSize, totalTextLength, 2_560])
  for _ in 0..<2 {
    let block = Krea2TextFusionBlock(
      batchSize: batchSize, tokenLength: totalTextLength, segments: segments,
      usesFlashAttention: usesFlashAttention, namePrefix: "text_fusion_refiner_")
    out = block(out)
  }
  return Model([x], [out])
}

private func Krea2TextProjection() -> Model {
  let x = Input()
  let norm = RMSNorm(epsilon: 1e-5, axis: [2], name: "norm")
  let linear1 = Dense(count: 6_144, name: "linear_1")
  let linear2 = Dense(count: 6_144, name: "linear_2")
  let out = linear2(linear1(norm(x).to(.Float16)).GELU(approximate: .tanh))
  return Model([x], [out])
}

public func Krea2TextFusionAdapter(
  batchSize: Int, textLength: (Int, Int), usesFlashAttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  let text = Input()
  let textFusion = Krea2TextFusion(
    batchSize: batchSize, textLength: textLength, usesFlashAttention: usesFlashAttention)
  let mapper: ModelWeightMapper = { _ in ModelWeightMapping() }
  return (mapper, Model([text], [textFusion(text)]))
}

private func Krea2TransformerBlock(
  batchSize: Int, tokenLength: Int, imageLength: Int, contextBlockPreOnly: Bool,
  usesFlashAttention: FlashAttentionLevel
) -> Model {
  precondition(!contextBlockPreOnly || imageLength <= tokenLength)
  let x = Input()
  let prescaleMod = Input()
  let preshiftMod = Input()
  let pregateMod = Input()
  let postscaleMod = Input()
  let postshiftMod = Input()
  let postgateMod = Input()
  let rot = Input()
  let prescaleTable = Parameter<FloatType>(.GPU(0), .CHW(1, 1, 6_144), name: "scale_shift_table_0")
  let preshiftTable = Parameter<FloatType>(.GPU(0), .CHW(1, 1, 6_144), name: "scale_shift_table_1")
  let pregateTable = Parameter<FloatType>(.GPU(0), .CHW(1, 1, 6_144), name: "scale_shift_table_2")
  let postscaleTable = Parameter<FloatType>(.GPU(0), .CHW(1, 1, 6_144), name: "scale_shift_table_3")
  let postshiftTable = Parameter<FloatType>(.GPU(0), .CHW(1, 1, 6_144), name: "scale_shift_table_4")
  let postgateTable = Parameter<FloatType>(.GPU(0), .CHW(1, 1, 6_144), name: "scale_shift_table_5")
  let prescale = prescaleMod + prescaleTable
  let preshift = preshiftMod + preshiftTable
  let pregate = pregateMod + pregateTable
  let postscale = postscaleMod + postscaleTable
  let postshift = postshiftMod + postshiftTable
  let postgate = postgateMod + postgateTable
  let norm1 = RMSNorm(epsilon: 1e-5, axis: [2], name: "norm1")
  let attention = Krea2Attention(
    hiddenSize: 6_144, heads: 48, keyValueHeads: 12, batchSize: batchSize,
    tokenLength: tokenLength, rotary: true,
    queryLength: contextBlockPreOnly ? imageLength : tokenLength,
    usesFlashAttention: usesFlashAttention)
  let xIn: Model.IO =
    contextBlockPreOnly
    ? x.reshaped(
      [batchSize, imageLength, 6_144], offset: [0, 0, 0],
      strides: [tokenLength * 6_144, 6_144, 1]
    ).contiguous() : x
  var out =
    xIn
    + (pregate .* attention(((1 + prescale) .* norm1(x).to(.Float16) + preshift), rot)).to(
      of: xIn)
  let norm2 = RMSNorm(epsilon: 1e-5, axis: [2], name: "norm2")
  let ff = Krea2SwiGLU(hiddenSize: 6_144, intermediateSize: 16_384)
  out =
    out + (postgate .* ff((1 + postscale) .* norm2(out).to(.Float16) + postshift)).to(of: out)
  return Model(
    [x, prescaleMod, preshiftMod, pregateMod, postscaleMod, postshiftMod, postgateMod, rot],
    [out])
}

public func Krea2Fixed(timesteps: Int) -> (ModelWeightMapper, Model) {
  precondition(timesteps > 0)
  let text = Input()
  let tEmbed = Input()
  let timeLinear1 = Dense(count: 6_144, name: "time_embed_linear_1")
  let timeLinear2 = Dense(count: 6_144, name: "time_embed_linear_2")
  let timeModProj0 = Dense(count: 6_144, name: "time_mod_proj_0")
  let timeModProj1 = Dense(count: 6_144, name: "time_mod_proj_1")
  let timeModProj2 = Dense(count: 6_144, name: "time_mod_proj_2")
  let timeModProj3 = Dense(count: 6_144, name: "time_mod_proj_3")
  let timeModProj4 = Dense(count: 6_144, name: "time_mod_proj_4")
  let timeModProj5 = Dense(count: 6_144, name: "time_mod_proj_5")
  let temb = timeLinear2(timeLinear1(tEmbed).GELU(approximate: .tanh))
  let tembMod = temb.GELU(approximate: .tanh)
  let prescaleMod = timeModProj0(tembMod)
  let preshiftMod = timeModProj1(tembMod)
  let pregateMod = timeModProj2(tembMod)
  let postscaleMod = timeModProj3(tembMod)
  let postshiftMod = timeModProj4(tembMod)
  let postgateMod = timeModProj5(tembMod)
  let txtIn = Krea2TextProjection()
  var outs = [txtIn(text)]
  outs.append(contentsOf: [
    prescaleMod, preshiftMod, pregateMod, postscaleMod, postshiftMod, postgateMod, temb,
  ])
  let mapper: ModelWeightMapper = { _ in ModelWeightMapping() }
  return (mapper, Model([text, tEmbed], outs))
}

public func Krea2(
  batchSize: Int, height: Int, width: Int, textLength: Int,
  usesFlashAttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  precondition(height % 2 == 0 && width % 2 == 0)
  let h = height / 2
  let w = width / 2
  let imageLength = h * w
  let tokenLength = textLength + imageLength
  let x = Input()
  let text = Input()
  let rot = Input()
  var inputs: [Input] = [x, text, rot]
  let packedImage = x.reshaped([batchSize, h, 2, w, 2, 16]).permuted(0, 1, 3, 5, 2, 4)
    .contiguous().reshaped([batchSize, imageLength, 64])
  let imgIn = Dense(count: 6_144, name: "img_in")
  let imageOut = imgIn(packedImage)
  var out = Functional.concat(axis: 1, imageOut, text).to(.Float32)
  let rotResized = rot.reshaped([1, tokenLength, 1, 128])
  let prescaleMod = Input()
  let preshiftMod = Input()
  let pregateMod = Input()
  let postscaleMod = Input()
  let postshiftMod = Input()
  let postgateMod = Input()
  inputs.append(contentsOf: [
    prescaleMod, preshiftMod, pregateMod, postscaleMod, postshiftMod, postgateMod,
  ])
  for i in 0..<28 {
    let block = Krea2TransformerBlock(
      batchSize: batchSize, tokenLength: tokenLength, imageLength: imageLength,
      contextBlockPreOnly: i == 27, usesFlashAttention: usesFlashAttention)
    out = block(
      out, prescaleMod, preshiftMod, pregateMod, postscaleMod, postshiftMod, postgateMod,
      rotResized)
  }
  out = out.reshaped([batchSize, imageLength, 6_144]).contiguous()
  let temb = Input()
  inputs.append(temb)
  let finalScaleTable = Parameter<FloatType>(
    .GPU(0), .CHW(1, 1, 6_144), name: "final_scale_shift_table_0")
  let finalShiftTable = Parameter<FloatType>(
    .GPU(0), .CHW(1, 1, 6_144), name: "final_scale_shift_table_1")
  let finalScale = temb + finalScaleTable
  let finalShift = temb + finalShiftTable
  let finalNorm = RMSNorm(epsilon: 1e-5, axis: [2], name: "final_norm")
  let finalLinear = Dense(count: 64, name: "final_linear")
  out = finalLinear((1 + finalScale) .* finalNorm(out).to(.Float16) + finalShift)
  out = out.reshaped([batchSize, h, w, 16, 2, 2]).permuted(0, 1, 4, 2, 5, 3).contiguous()
    .reshaped([batchSize, height, width, 16])
  let mapper: ModelWeightMapper = { _ in ModelWeightMapping() }
  return (mapper, Model(inputs, [out.to(of: x)]))
}
