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
  textLength: Int, audioLength: Int, videoFrames: Int, videoHeight: Int, videoWidth: Int
) -> Tensor<Float> {
  precondition(audioLength % 2 == 0)
  precondition(videoHeight % 2 == 0)
  precondition(videoWidth % 2 == 0)
  let audioLatents = audioLength / 2
  let height = videoHeight / 2
  let width = videoWidth / 2
  let rowsPerFrame = height * width
  let videoLength = videoFrames * rowsPerFrame
  let sequenceLength = textLength + audioLength + videoLength
  let squareRootArea = sqrt(Double(videoHeight * videoWidth))
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
  var temporalPosition = Double(textLength)
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
    for token in 0..<textLength {
      write(row: token, position: (Float(token), 0, 0))
    }
    let audioOffset = textLength
    for channel in 0..<2 {
      let spatialPosition = channel == 0 ? widthGrid.first! : widthGrid.last!
      for index in 0..<audioLatents {
        write(
          row: audioOffset + channel * audioLatents + index,
          position: (Float(textLength + index), 0, spatialPosition))
      }
    }
    let videoOffset = textLength + audioLength
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
  usesFlashAttention: FlashAttentionLevel, name: String = ""
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
  let attentionOutput = attention(q, k, v)
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
  usesFlashAttention: FlashAttentionLevel, scaleFactor: Int?
) -> Model {
  let sequenceLength = textLength + audioLength + videoLength
  let x = Input()
  let rot = Input()
  let modulations = (0..<18).map { _ in Input() }
  let offsets = [0, textLength, textLength + audioLength]
  let lengths = [textLength, audioLength, videoLength]
  // Token order is text, audio, video, while AdaLN modality order is video, text, audio.
  let modalities = [1, 2, 0]
  let norm1 = RMSNorm(epsilon: 1e-5, axis: [2], name: "norm1")
  let attention = H3Attention(
    hiddenSize: hiddenSize, sequenceLength: sequenceLength, isRoPEEnabled: true, scaleFactor: 8,
    usesFlashAttention: usesFlashAttention)
  let normed1 = norm1(x).to(.Float16)
  let attentionInputs = (0..<3).map { index in
    let modality = modalities[index]
    let shift = modulations[modality]
    let scale = 1 + modulations[3 + modality]
    return
      (normed1.reshaped(
        [1, lengths[index], hiddenSize], offset: [0, offsets[index], 0],
        strides: [sequenceLength * hiddenSize, hiddenSize, 1]
      ).contiguous()
      .* scale + shift)
  }
  let attentionInput = Functional.concat(
    axis: 1, attentionInputs[0], attentionInputs[1], attentionInputs[2])
  let attentionOutput = attention(attentionInput, rot)
  let gatedAttention = (0..<3).map { index in
    let modality = modalities[index]
    return modulations[6 + modality].to(.Float32)
      .* attentionOutput.reshaped(
        [1, lengths[index], hiddenSize], offset: [0, offsets[index], 0],
        strides: [sequenceLength * hiddenSize, hiddenSize, 1]
      ).contiguous()
  }
  var out =
    x
    + Functional.concat(
      axis: 1, gatedAttention[0], gatedAttention[1], gatedAttention[2])
  let norm2 = RMSNorm(epsilon: 1e-5, axis: [2], name: "norm2")
  let feedForward = H3SwiGLU(hiddenSize: hiddenSize, scaleFactor: scaleFactor)
  let normed2 = norm2(out).to(.Float16)
  let feedForwardInputs = (0..<3).map { index in
    let modality = modalities[index]
    let shift = modulations[9 + modality]
    let scale = 1 + modulations[12 + modality]
    return
      (normed2.reshaped(
        [1, lengths[index], hiddenSize], offset: [0, offsets[index], 0],
        strides: [sequenceLength * hiddenSize, hiddenSize, 1]
      ).contiguous()
      .* scale + shift)
  }
  let feedForwardInput = Functional.concat(
    axis: 1, feedForwardInputs[0], feedForwardInputs[1], feedForwardInputs[2])
  let rawFeedForwardOutput = feedForward(feedForwardInput)
  let gatedFeedForward = (0..<3).map { index in
    let modality = modalities[index]
    return modulations[15 + modality].to(.Float32)
      .* rawFeedForwardOutput.reshaped(
        [1, lengths[index], hiddenSize], offset: [0, offsets[index], 0],
        strides: [sequenceLength * hiddenSize, hiddenSize, 1]
      ).contiguous()
  }
  let feedForwardOutput = Functional.concat(
    axis: 1, gatedFeedForward[0], gatedFeedForward[1], gatedFeedForward[2])
  if let scaleFactor {
    out = out + Float(scaleFactor) * feedForwardOutput
  } else {
    out = out + feedForwardOutput
  }
  return Model([x, rot] + modulations, [out])
}

private func H3TokenRefinerBlock(
  hiddenSize: Int, sequenceLength: Int, usesFlashAttention: FlashAttentionLevel
) -> Model {
  let x = Input()
  let norm1 = RMSNorm(epsilon: 1e-5, axis: [2], name: "refiner_norm1")
  let attention = H3Attention(
    hiddenSize: hiddenSize, sequenceLength: sequenceLength, isRoPEEnabled: false, scaleFactor: 1,
    usesFlashAttention: usesFlashAttention, name: "refiner")
  let attentionInput = norm1(x).to(.Float16)
  let attentionOutput = attention(attentionInput)
  var out = x + attentionOutput
  let norm2 = RMSNorm(epsilon: 1e-5, axis: [2], name: "refiner_norm2")
  let feedForward = H3SwiGLU(hiddenSize: hiddenSize, name: "refiner")
  out = out + feedForward(norm2(out).to(.Float16))
  return Model([x], [out])
}

private func H3TokenRefiner(
  hiddenSize: Int, sequenceLength: Int, usesFlashAttention: FlashAttentionLevel
) -> Model {
  let x = Input()
  var out: Model.IO = x
  for _ in 0..<2 {
    let block = H3TokenRefinerBlock(
      hiddenSize: hiddenSize, sequenceLength: sequenceLength,
      usesFlashAttention: usesFlashAttention)
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

public func MiniMaxH3Fixed(timesteps: Int, hiddenSize: Int, layers: Int) -> Model {
  precondition(timesteps > 0 && hiddenSize > 0 && layers > 0)
  let timestepFrequencies = Input()
  let timeEmbedding = H3TimestepEmbedding(hiddenSize: hiddenSize)
  let activatedTimestep = timeEmbedding(timestepFrequencies).swish()
  var outputs = [Model.IO]()
  for _ in 0..<layers {
    for chunk in 0..<6 {
      for modality in 0..<3 {
        let projected = Dense(
          count: hiddenSize, name: "adaln_\(chunk)_\(modality)"
        )(activatedTimestep)
        let timestep = modality == 2 ? 1 : 0
        outputs.append(
          projected.reshaped(
            [timesteps, 1, hiddenSize], offset: [0, timestep, 0],
            strides: [2 * hiddenSize, hiddenSize, 1]
          ).contiguous())
      }
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
        strides: [2 * hiddenSize, hiddenSize, 1]
      ).contiguous())
    outputs.append(
      outputScales.reshaped(
        [timesteps, 1, hiddenSize], offset: [0, timestep, 0],
        strides: [2 * hiddenSize, hiddenSize, 1]
      ).contiguous())
  }
  precondition(outputs.count == layers * 18 + 4)
  return Model([timestepFrequencies], outputs)
}

public func MiniMaxH3(
  hiddenSize: Int, layers: Int, textLength: Int, audioLength: Int, videoFrames: Int,
  videoHeight: Int, videoWidth: Int, usesFlashAttention: FlashAttentionLevel
) -> Model {
  precondition(hiddenSize > 0 && layers > 0)
  precondition(videoHeight % 2 == 0 && videoWidth % 2 == 0)
  let video = Input()
  let audio = Input()
  let text = Input()
  let rot = Input()
  let fixedConditions = (0..<(layers * 18 + 4)).map { _ in Input() }
  let videoLength = videoFrames * videoHeight / 2 * videoWidth / 2
  let sequenceLength = textLength + audioLength + videoLength

  let xEmbedder = Convolution(
    groups: 1, filters: hiddenSize, filterSize: [2, 2],
    hint: Hint(stride: [2, 2]), format: .OIHW, name: "proj_in")
  let audioInput = Dense(count: hiddenSize, name: "audio_proj_in")
  let textInput = Dense(count: hiddenSize, name: "context_embedder")
  let refiner = H3TokenRefiner(
    hiddenSize: hiddenSize, sequenceLength: textLength,
    usesFlashAttention: usesFlashAttention)
  // The released checkpoint does not fold condition_proj's 0.25 multiplier into its weights.
  let textProjected = 4 * textInput(0.25 * text).to(.Float32)
  let textRows = refiner(textProjected)
  let audioRows = 4 * audioInput(0.25 * audio).to(.Float32)
  let videoRows =
    4 * xEmbedder(0.25 * video).reshaped(.HWC(1, videoLength, hiddenSize)).to(.Float32)
  var out = Functional.concat(axis: 1, textRows, audioRows, videoRows)
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
      hiddenSize: hiddenSize, textLength: textLength, audioLength: audioLength,
      videoLength: videoLength,
      usesFlashAttention: usesFlashAttention,
      scaleFactor: scaleFactor)
    let conditionOffset = layer * 18
    out =
      block(
        [out, rot]
          + Array(
            fixedConditions[
              conditionOffset..<(conditionOffset + 18)
            ]))[0]
  }

  let outputNorm = RMSNorm(epsilon: 1e-5, axis: [2], name: "norm_out")
  let normalized = outputNorm(out)
  let finalConditionOffset = layers * 18
  let videoShift = fixedConditions[finalConditionOffset]
  let videoScale = fixedConditions[finalConditionOffset + 1]
  let audioShift = fixedConditions[finalConditionOffset + 2]
  let audioScale = fixedConditions[finalConditionOffset + 3]
  let audioOutRows =
    normalized.reshaped(
      [1, audioLength, hiddenSize], offset: [0, textLength, 0],
      strides: [sequenceLength * hiddenSize, hiddenSize, 1]
    ).contiguous() .* (1 + audioScale).to(.Float32) + audioShift.to(.Float32)
  let videoOutRows =
    normalized.reshaped(
      [1, videoLength, hiddenSize], offset: [0, textLength + audioLength, 0],
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
    [video, audio, text, rot] + fixedConditions,
    [projectedVideo.to(of: video), projectedAudio.to(of: audio)])
}
