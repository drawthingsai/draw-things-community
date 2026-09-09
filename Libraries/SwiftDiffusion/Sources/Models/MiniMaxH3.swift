// MiniMax H3 joint video/audio denoiser graph definitions.
import Foundation
import NNC

public enum MiniMaxH3Configuration {
  public static let videoChannels = 24
  public static let audioChannels = 32
  public static let spatialCompression = 16
  public static let framesPerSecond = 24
  public static let audioSampleRate = 32_000
  public static let videoFlowShift: Float = 12
  public static let audioFlowShift: Float = 3
}

public func MiniMaxH3AudioHeight(videoLatentFrames: Int, latentWidth: Int) -> Int {
  precondition(latentWidth > 0)
  precondition(
    videoLatentFrames == 1 || videoLatentFrames == 2
      || (videoLatentFrames >= 7 && (videoLatentFrames - 2) % 5 == 0))
  let frames =
    videoLatentFrames == 1 ? 1 : (videoLatentFrames - 2) / 5 * 17 + 5
  let rows =
    2
    * Int(
      (Double(frames) / Double(MiniMaxH3Configuration.framesPerSecond) * 40).rounded())
  let rowSize =
    videoLatentFrames * latentWidth * MiniMaxH3Configuration.videoChannels
  let audioSize = rows * MiniMaxH3Configuration.audioChannels
  var height = (audioSize + rowSize - 1) / rowSize
  while height * rowSize % MiniMaxH3Configuration.audioChannels != 0 {
    height += 1
  }
  return height
}

func MiniMaxH3AudioSigma(forVideoSigma sigma: Float) -> Float {
  let base =
    sigma
    / (MiniMaxH3Configuration.videoFlowShift
      - (MiniMaxH3Configuration.videoFlowShift - 1) * sigma)
  return MiniMaxH3Configuration.audioFlowShift * base
    / (1 + (MiniMaxH3Configuration.audioFlowShift - 1) * base)
}

public func MiniMaxH3RotaryEmbedding(
  textLength: Int, audioLength: Int = 0, videoFrames: Int = 0, videoHeight: Int = 0,
  videoWidth: Int = 0, referenceImages: [(height: Int, width: Int, position: Float)] = [],
  videoPosition: Float? = nil, visionTokenRanges: [Range<Int>] = []
) -> Tensor<Float> {
  precondition(audioLength % 2 == 0)
  precondition(videoHeight % 2 == 0)
  precondition(videoWidth % 2 == 0)
  let audioLatents = audioLength / 2
  let height = videoHeight / 2
  let width = videoWidth / 2
  let rowsPerFrame = height * width
  let videoLength = videoFrames * rowsPerFrame
  let conditionLength = referenceImages.reduce(0) { $0 + $1.height / 2 * ($1.width / 2) }
  let visionLength = visionTokenRanges.reduce(0) { $0 + $1.count }
  let mediaPosition = videoPosition ?? Float(textLength)
  let sequenceLength = textLength + conditionLength + audioLength + videoLength
  let squareRootArea = max(1, sqrt(Double(videoHeight * videoWidth)))
  let heightRatio = Double(videoHeight) / squareRootArea
  let widthRatio = Double(videoWidth) / squareRootArea
  let heightLeft = (1 - heightRatio) / 2
  let widthLeft = (1 - widthRatio) / 2
  // These match numpy.linspace(start, stop, count, endpoint=False), not torch.linspace.
  let heightGrid = (0..<height).map {
    Float((heightLeft + heightRatio * Double($0) / Double(height)) * 32)
  }
  let widthGrid = (0..<width).map {
    Float((widthLeft + widthRatio * Double($0) / Double(width)) * 32)
  }
  let frameRescale = 5.0 / 3.0
  let framesPerLatent = [1.0, 4.0, 4.0, 4.0, 4.0]
  var temporalGrid = [Float](repeating: 0, count: videoFrames)
  var temporalPosition = Double(mediaPosition)
  for frame in 0..<videoFrames {
    temporalGrid[frame] = Float(temporalPosition)
    temporalPosition += frameRescale * framesPerLatent[frame % framesPerLatent.count]
  }
  let inverseFrequencies = (0..<16).map {
    pow(10_000, -Double($0) * 2 / 32)
  }
  var rotary = Tensor<Float>(.CPU, .NHWC(1, sequenceLength, 1, 128))
  rotary.withUnsafeMutableBytes {
    guard let fp32 = $0.baseAddress?.assumingMemoryBound(to: Float.self) else { return }
    func write(row: Int, position: (Float, Float, Float)) {
      let rowStart = row * 128
      for axis in 0..<3 {
        let value = axis == 0 ? position.0 : (axis == 1 ? position.1 : position.2)
        for frequency in 0..<16 {
          let angle = Double(value) * inverseFrequencies[frequency]
          let offset = rowStart + (axis * 16 + frequency) * 2
          fp32[offset] = Float(cos(angle))
          fp32[offset + 1] = Float(sin(angle))
        }
      }
      for index in 96..<128 {
        fp32[rowStart + index] = index % 2 == 0 ? 1 : 0
      }
    }
    var textRow = 0
    var visionRow = textLength - visionLength + conditionLength + audioLength
    for token in 0..<textLength {
      if visionTokenRanges.contains(where: { $0.contains(token) }) {
        write(row: visionRow, position: (Float(token), 0, 0))
        visionRow += 1
      } else {
        write(row: textRow, position: (Float(token), 0, 0))
        textRow += 1
      }
    }
    var referenceRow = textLength - visionLength
    for image in referenceImages {
      precondition(
        image.height > 0 && image.width > 0 && image.height % 2 == 0 && image.width % 2 == 0)
      let area = sqrt(Double(image.height * image.width))
      let hRatio = Double(image.height) / area
      let wRatio = Double(image.width) / area
      for y in 0..<(image.height / 2) {
        for x in 0..<(image.width / 2) {
          write(
            row: referenceRow,
            position: (
              image.position,
              Float(((1 - hRatio) / 2 + hRatio * Double(y) / Double(image.height / 2)) * 32),
              Float(((1 - wRatio) / 2 + wRatio * Double(x) / Double(image.width / 2)) * 32)
            ))
          referenceRow += 1
        }
      }
    }
    let audioOffset = textLength - visionLength + conditionLength
    for channel in 0..<(audioLatents > 0 ? 2 : 0) {
      let spatialPosition = channel == 0 ? widthGrid.first! : widthGrid.last!
      for index in 0..<audioLatents {
        write(
          row: audioOffset + channel * audioLatents + index,
          position: (mediaPosition + Float(index), 0, spatialPosition))
      }
    }
    let videoOffset = audioOffset + audioLength + visionLength
    for frame in 0..<videoFrames {
      for y in 0..<height {
        for x in 0..<width {
          write(
            row: videoOffset + frame * rowsPerFrame + y * width + x,
            position: (temporalGrid[frame], heightGrid[y], widthGrid[x]))
        }
      }
    }
  }
  return rotary
}

private func H3Attention(
  hiddenSize: Int, sequenceLength: Int, isRoPEEnabled: Bool, scaleFactor: Int,
  usesFlashAttention: FlashAttentionLevel, segments: [Int] = [], name: String = ""
) -> Model {
  let x = Input()
  let rot = isRoPEEnabled ? Input() : nil
  let toQ = Dense(
    count: 7_168, noBias: true, name: name.isEmpty ? "q" : "\(name)_q")
  let toK = Dense(
    count: 7_168, noBias: true, name: name.isEmpty ? "k" : "\(name)_k")
  let toV = Dense(
    count: 7_168, noBias: true, name: name.isEmpty ? "v" : "\(name)_v")
  let normQ = RMSNorm(
    epsilon: 1e-5, axis: [3], name: name.isEmpty ? "norm_q" : "\(name)_norm_q")
  let normK = RMSNorm(
    epsilon: 1e-5, axis: [3], name: name.isEmpty ? "norm_k" : "\(name)_norm_k")
  let qProjected = toQ(x)
  let kProjected = toK(x)
  let vProjected = scaleFactor > 1 ? toV((1 / Float(scaleFactor)) * x) : toV(x)
  var q = normQ(
    qProjected.reshaped(.NHWC(1, sequenceLength, 56, 128)))
  var k = normK(
    kProjected.reshaped(.NHWC(1, sequenceLength, 56, 128)))
  let v = vProjected.reshaped(.NHWC(1, sequenceLength, 56, 128))
  if let rot {
    q = Functional.cmul(left: q, right: rot)
    k = Functional.cmul(left: k, right: rot)
  }
  q = (1 / Float(128).squareRoot()) * q
  let attention = ScaledDotProductAttention(
    scale: 1,
    flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
  let attentionOutput: Model.IO
  if segments.isEmpty {
    attentionOutput = attention(q, k, v)
  } else {
    precondition(segments.reduce(0, +) == sequenceLength)
    var offset = 0
    let outputs = segments.map { length in
      let inputs = [q, k, v].map {
        $0.reshaped(
          [1, length, 56, 128], offset: [0, offset, 0, 0],
          strides: [sequenceLength * 7_168, 7_168, 128, 1]
        ).contiguous()
      }
      offset += length
      return attention(inputs)
    }
    attentionOutput = Concat(axis: 1)(outputs)
  }
  let attended = attentionOutput.reshaped([1, sequenceLength, 7_168])
  let out = Dense(
    count: hiddenSize, noBias: true, name: name.isEmpty ? "o" : "\(name)_o")
  let projected: Model.IO
  if scaleFactor > 1 {
    projected = Float(scaleFactor) * out(attended).to(.Float32)
  } else {
    projected = out(attended).to(.Float32)
  }
  if let rot {
    return Model([x, rot], [projected])
  }
  return Model([x], [projected])
}

private func H3SwiGLU(hiddenSize: Int, scaleFactor: Int? = nil, name: String = "") -> Model {
  let x = Input()
  let up = Dense(
    count: 14_336, noBias: true, name: name.isEmpty ? "up" : "\(name)_up")
  let gate = Dense(
    count: 14_336, noBias: true, name: name.isEmpty ? "gate" : "\(name)_gate")
  let down = Dense(
    count: hiddenSize, noBias: true, name: name.isEmpty ? "down" : "\(name)_down")
  let upOutput = scaleFactor.map { up((1 / Float($0)) * x) } ?? up(x)
  let out = down(Functional.swishMul(value: upOutput, gate: gate(x))).to(.Float32)
  return Model([x], [out])
}

private func H3TransformerBlock(
  hiddenSize: Int, textLength: Int, audioLength: Int, videoLength: Int,
  conditionLength: Int,
  usesFlashAttention: FlashAttentionLevel, scaleFactor: Int?
) -> Model {
  let sequenceLength = textLength + conditionLength + audioLength + videoLength
  let x = Input()
  let rot = Input()
  let modulationCount = conditionLength > 0 ? 4 : 3
  let modulations = (0..<(6 * modulationCount)).map { _ in Input() }
  // Modulation order: video, text, audio, and (when present) the fixed video keyframe.
  var spans: [(range: Range<Int>, modality: Int)] = [(0..<textLength, 1)]
  if conditionLength > 0 {
    spans.append((textLength..<(textLength + conditionLength), 3))
  }
  spans += [
    ((textLength + conditionLength)..<(textLength + conditionLength + audioLength), 2),
    ((textLength + conditionLength + audioLength)..<sequenceLength, 0),
  ]
  spans.removeAll { $0.range.isEmpty }
  let norm1 = RMSNorm(epsilon: 1e-5, axis: [2], name: "norm1")
  let attention = H3Attention(
    hiddenSize: hiddenSize, sequenceLength: sequenceLength, isRoPEEnabled: true, scaleFactor: 8,
    usesFlashAttention: usesFlashAttention)
  let normed1 = norm1(x).to(.Float16)
  let attentionInputs = spans.map { span in
    let modality = span.modality
    let shift = modulations[modality]
    let scale = 1 + modulations[modulationCount + modality]
    return
      (normed1.reshaped(
        [1, span.range.count, hiddenSize], offset: [0, span.range.lowerBound, 0],
        strides: [sequenceLength * hiddenSize, hiddenSize, 1]
      ).contiguous()
      .* scale + shift)
  }
  let attentionInput = Concat(axis: 1)(attentionInputs)
  let attentionOutput = attention(attentionInput, rot)
  let gatedAttention = spans.map { span in
    return modulations[2 * modulationCount + span.modality].to(.Float32)
      .* attentionOutput.reshaped(
        [1, span.range.count, hiddenSize], offset: [0, span.range.lowerBound, 0],
        strides: [sequenceLength * hiddenSize, hiddenSize, 1]
      ).contiguous()
  }
  var out =
    x
    + Concat(axis: 1)(gatedAttention)
  let norm2 = RMSNorm(epsilon: 1e-5, axis: [2], name: "norm2")
  let feedForward = H3SwiGLU(hiddenSize: hiddenSize, scaleFactor: scaleFactor)
  let normed2 = norm2(out).to(.Float16)
  let feedForwardInputs = spans.map { span in
    let modality = span.modality
    let shift = modulations[3 * modulationCount + modality]
    let scale = 1 + modulations[4 * modulationCount + modality]
    return
      (normed2.reshaped(
        [1, span.range.count, hiddenSize], offset: [0, span.range.lowerBound, 0],
        strides: [sequenceLength * hiddenSize, hiddenSize, 1]
      ).contiguous()
      .* scale + shift)
  }
  let feedForwardInput = Concat(axis: 1)(feedForwardInputs)
  let rawFeedForwardOutput = feedForward(feedForwardInput)
  let gatedFeedForward = spans.map { span in
    return modulations[5 * modulationCount + span.modality].to(.Float32)
      .* rawFeedForwardOutput.reshaped(
        [1, span.range.count, hiddenSize], offset: [0, span.range.lowerBound, 0],
        strides: [sequenceLength * hiddenSize, hiddenSize, 1]
      ).contiguous()
  }
  let feedForwardOutput = Concat(axis: 1)(gatedFeedForward)
  if let scaleFactor {
    out = out + Float(scaleFactor) * feedForwardOutput
  } else {
    out = out + feedForwardOutput
  }
  return Model([x, rot] + modulations, [out])
}

private func H3TokenRefinerBlock(
  hiddenSize: Int, sequenceLength: Int, usesFlashAttention: FlashAttentionLevel, segments: [Int]
) -> Model {
  let x = Input()
  let norm1 = RMSNorm(epsilon: 1e-5, axis: [2], name: "refiner_norm1")
  let attention = H3Attention(
    hiddenSize: hiddenSize, sequenceLength: sequenceLength, isRoPEEnabled: false, scaleFactor: 1,
    usesFlashAttention: usesFlashAttention, segments: segments, name: "refiner")
  let attentionInput = norm1(x).to(.Float16)
  let attentionOutput = attention(attentionInput)
  var out = x + attentionOutput
  let norm2 = RMSNorm(epsilon: 1e-5, axis: [2], name: "refiner_norm2")
  let feedForward = H3SwiGLU(hiddenSize: hiddenSize, name: "refiner")
  out = out + feedForward(norm2(out).to(.Float16))
  return Model([x], [out])
}

private func H3TokenRefiner(
  hiddenSize: Int, sequenceLength: Int, usesFlashAttention: FlashAttentionLevel, segments: [Int]
) -> Model {
  let x = Input()
  var out: Model.IO = x
  for _ in 0..<2 {
    let block = H3TokenRefinerBlock(
      hiddenSize: hiddenSize, sequenceLength: sequenceLength,
      usesFlashAttention: usesFlashAttention, segments: segments)
    out = block(out)
  }
  let norm = RMSNorm(epsilon: 1e-5, axis: [2], name: "refiner_final_norm")
  out = norm(out)
  return Model([x], [out])
}

private func H3TimestepEmbedding(hiddenSize: Int) -> Model {
  let frequencies = Input()
  let linear1 = Dense(count: hiddenSize, name: "time_embedder_linear_1")
  let linear2 = Dense(count: 2_688, name: "time_embedder_linear_2")
  let out = linear2(linear1(frequencies).swish())
  return Model([frequencies], [out])
}

public func MiniMaxH3Fixed(
  timesteps: Int, hiddenSize: Int, layers: Int, textLength: (Int, Int),
  usesFlashAttention: FlashAttentionLevel, referenceImageCount: Int = 0,
  visionLength: Int = 0
) -> Model {
  precondition(timesteps > 0 && hiddenSize > 0 && layers > 0)
  precondition(textLength.0 >= 0 && textLength.1 > 0)
  let text = Input()
  let referenceImages = (0..<referenceImageCount).map { _ in Input() }
  let paddedLength = max(textLength.0, textLength.1)
  let contextInput: Model.IO
  if textLength.0 > 0 {
    // Remove CFG padding before refinement; vision rows follow the padded text in each branch.
    let branches = [textLength.0, textLength.1].enumerated().map { batch, length in
      var branch = text.reshaped(
        [1, length - visionLength, 5_120], offset: [batch, 0, 0],
        strides: [paddedLength * 5_120, 5_120, 1]
      ).contiguous()
      if visionLength > 0 {
        branch = Functional.concat(
          axis: 1, branch,
          text.reshaped(
            [1, visionLength, 5_120], offset: [batch, paddedLength - visionLength, 0],
            strides: [paddedLength * 5_120, 5_120, 1]
          ).contiguous())
      }
      return branch
    }
    contextInput = Concat(axis: 1)(branches)
  } else {
    contextInput = text
  }
  let textInput = Dense(count: hiddenSize, name: "context_embedder")
  let refiner = H3TokenRefiner(
    hiddenSize: hiddenSize, sequenceLength: textLength.0 + textLength.1,
    usesFlashAttention: usesFlashAttention,
    segments: textLength.0 > 0 ? [textLength.0, textLength.1] : [])
  // UNetFixedEncoder scales the text bias by 1/4 to match the pre-scaled input.
  let textProjected = 4 * textInput(0.25 * contextInput).to(.Float32)
  var context = refiner(textProjected)
  if textLength.0 > 0 {
    let refined = context
    var offset = 0
    let branches = [textLength.0, textLength.1].map { length in
      var branch = refined.reshaped(
        [1, length - visionLength, hiddenSize], offset: [0, offset, 0],
        strides: [(textLength.0 + textLength.1) * hiddenSize, hiddenSize, 1]
      ).contiguous()
      if length < paddedLength {
        branch = branch.padded(.zero, begin: [0, 0, 0], end: [0, paddedLength - length, 0])
      }
      if visionLength > 0 {
        branch = Functional.concat(
          axis: 1, branch,
          refined.reshaped(
            [1, visionLength, hiddenSize], offset: [0, offset + length - visionLength, 0],
            strides: [(textLength.0 + textLength.1) * hiddenSize, hiddenSize, 1]
          ).contiguous())
      }
      offset += length
      return branch
    }
    context = Concat(axis: 0)(branches)
  }
  let timestepFrequencies = Input()
  let timeEmbedding = H3TimestepEmbedding(hiddenSize: hiddenSize)
  let activatedTimestep = timeEmbedding(timestepFrequencies).swish()
  let timestepCount = referenceImages.isEmpty ? 2 : 3
  var outputs = [context]
  if !referenceImages.isEmpty {
    let xEmbedder = Convolution(
      groups: 1, filters: hiddenSize, filterSize: [2, 2],
      hint: Hint(stride: [2, 2]), format: .OIHW, name: "proj_in")
    outputs += referenceImages.map { xEmbedder($0) }
  }
  for _ in 0..<layers {
    for chunk in 0..<6 {
      var condition: Model.IO? = nil
      for modality in 0..<3 {
        let projected = Dense(
          count: hiddenSize, name: "adaln_\(chunk)_\(modality)"
        )(activatedTimestep)
        let timestep = modality == 2 ? 1 : 0
        outputs.append(
          projected.reshaped(
            [timesteps, 1, hiddenSize], offset: [0, timestep, 0],
            strides: [timestepCount * hiddenSize, hiddenSize, 1]
          ).contiguous())
        if !referenceImages.isEmpty && modality == 0 {
          condition = projected.reshaped(
            [timesteps, 1, hiddenSize], offset: [0, 2, 0],
            strides: [timestepCount * hiddenSize, hiddenSize, 1]
          ).contiguous()
        }
      }
      if let condition { outputs.append(condition) }
    }
  }
  let outputShift = Dense(count: hiddenSize, name: "norm_out_shift")
  let outputScale = Dense(count: hiddenSize, name: "norm_out_scale")
  let outputShifts = outputShift(activatedTimestep)
  let outputScales = outputScale(activatedTimestep)
  for timestep in 0..<2 {
    outputs.append(
      outputShifts.reshaped(
        [timesteps, 1, hiddenSize], offset: [0, timestep, 0],
        strides: [timestepCount * hiddenSize, hiddenSize, 1]
      ).contiguous())
    outputs.append(
      outputScales.reshaped(
        [timesteps, 1, hiddenSize], offset: [0, timestep, 0],
        strides: [timestepCount * hiddenSize, hiddenSize, 1]
      ).contiguous())
  }
  precondition(
    outputs.count == layers * (referenceImages.isEmpty ? 18 : 24) + 5 + referenceImages.count)
  return Model([text, timestepFrequencies] + referenceImages, outputs)
}

public func MiniMaxH3(
  hiddenSize: Int, layers: Int, textLength: Int, audioLength: Int, videoFrames: Int,
  videoHeight: Int, videoWidth: Int, usesFlashAttention: FlashAttentionLevel,
  referenceImageSizes: [(height: Int, width: Int)] = [], visionLength: Int = 0
) -> Model {
  precondition(hiddenSize > 0 && layers > 0)
  precondition(videoHeight % 2 == 0 && videoWidth % 2 == 0)
  let video = Input()
  let audio = Input()
  let contextRows = Input()
  let rot = Input()
  let referenceImages = referenceImageSizes.map { _ in Input() }
  let perLayerConditions = referenceImages.isEmpty ? 18 : 24
  let fixedConditions = (0..<(layers * perLayerConditions + 4)).map { _ in Input() }
  let videoLength = videoFrames * videoHeight / 2 * videoWidth / 2
  let conditionLength = referenceImageSizes.reduce(0) { $0 + $1.height / 2 * ($1.width / 2) }
  let sequenceLength = textLength + conditionLength + audioLength + videoLength

  let xEmbedder = Convolution(
    groups: 1, filters: hiddenSize, filterSize: [2, 2],
    hint: Hint(stride: [2, 2]), format: .OIHW, name: "proj_in")
  let audioInput = Dense(count: hiddenSize, name: "audio_proj_in")
  let audioRows = audioInput(audio).to(.Float32)
  let videoRows = xEmbedder(video).reshaped(.HWC(1, videoLength, hiddenSize)).to(.Float32)
  let textRows: Model.IO
  if visionLength > 0 {
    textRows = contextRows.reshaped(
      [1, textLength - visionLength, hiddenSize],
      strides: [textLength * hiddenSize, hiddenSize, 1]
    ).contiguous()
  } else {
    textRows = contextRows
  }
  // Each modulation group stays contiguous throughout all transformer blocks.
  var rows = [textRows]
  for (image, size) in zip(referenceImages, referenceImageSizes) {
    rows.append(
      image.reshaped(.HWC(1, size.height / 2 * (size.width / 2), hiddenSize)).to(.Float32))
  }
  rows.append(audioRows)
  if visionLength > 0 {
    rows.append(
      contextRows.reshaped(
        [1, visionLength, hiddenSize], offset: [0, textLength - visionLength, 0],
        strides: [textLength * hiddenSize, hiddenSize, 1]
      ).contiguous())
  }
  rows.append(videoRows)
  var out = Concat(axis: 1)(rows)
  for layer in 0..<layers {
    // Keep FFN outliers in FP16 range; the residual branch restores the factor in FP32.
    let scaleFactor: Int
    switch layer {
    case 0, 4, 11, 31, 32:
      scaleFactor = 4
    case 1, 7, 27, 33, 34, 35, 37, 38, 46:
      scaleFactor = 8
    case 18, 30, 40, 41, 42:
      scaleFactor = 16
    case 47:
      scaleFactor = 32
    case 13, 36, 43, 44, 48, 49:
      scaleFactor = 64
    case 39:
      scaleFactor = 512
    case 45:
      scaleFactor = 1_024
    default:
      scaleFactor = 2
    }
    let block = H3TransformerBlock(
      hiddenSize: hiddenSize, textLength: textLength - visionLength, audioLength: audioLength,
      videoLength: visionLength + videoLength, conditionLength: conditionLength,
      usesFlashAttention: usesFlashAttention,
      scaleFactor: scaleFactor)
    let conditionOffset = layer * perLayerConditions
    out =
      block(
        [out, rot]
          + Array(
            fixedConditions[
              conditionOffset..<(conditionOffset + perLayerConditions)
            ]))[0]
  }

  let outputNorm = RMSNorm(epsilon: 1e-5, axis: [2], name: "norm_out")
  let normalized = outputNorm(out)
  let finalConditionOffset = layers * perLayerConditions
  let videoShift = fixedConditions[finalConditionOffset]
  let videoScale = fixedConditions[finalConditionOffset + 1]
  let audioShift = fixedConditions[finalConditionOffset + 2]
  let audioScale = fixedConditions[finalConditionOffset + 3]
  let audioOutRows =
    normalized.reshaped(
      [1, audioLength, hiddenSize], offset: [0, textLength - visionLength + conditionLength, 0],
      strides: [sequenceLength * hiddenSize, hiddenSize, 1]
    ).contiguous() .* (1 + audioScale).to(.Float32) + audioShift.to(.Float32)
  let videoOutRows =
    normalized.reshaped(
      [1, videoLength, hiddenSize], offset: [0, textLength + conditionLength + audioLength, 0],
      strides: [sequenceLength * hiddenSize, hiddenSize, 1]
    ).contiguous() .* (1 + videoScale).to(.Float32) + videoShift.to(.Float32)
  let videoOutput = Dense(count: 96, name: "proj_out")
  let audioOutput = Dense(count: 32, name: "audio_proj_out")
  let projectedVideo = -videoOutput(videoOutRows).reshaped([
    videoFrames, videoHeight / 2, videoWidth / 2, 24, 2, 2,
  ]).permuted(0, 1, 4, 2, 5, 3).contiguous().reshaped([
    videoFrames, videoHeight, videoWidth, 24,
  ])
  let projectedAudio = -audioOutput(audioOutRows)
  return Model(
    [video, audio, contextRows, rot] + referenceImages
      + fixedConditions,
    [projectedVideo.to(of: video), projectedAudio.to(of: audio)])
}
