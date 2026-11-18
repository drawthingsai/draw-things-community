import Foundation
import NNC

// Audio input for LTX-2 / LTX-2.3 (audio-to-video conditioning).
//
// Reference implementation: Lightricks LTX-2, packages/ltx-core/src/ltx_core/model/audio_vae
// (ops.py AudioProcessor, audio_vae.py AudioEncoder._normalize_latents) and
// components/patchifiers.py AudioPatchifier.

public enum LTX2AudioEncoding {
  /// Sample rate expected by the LTX-2 audio VAE encoder.
  public static let sampleRate = 16_000
  public static let fftSize = 1_024
  public static let hopLength = 160
  public static let melBins = 64
  /// Width of one patched latent frame: 8 latent channels x 16 frequency bins.
  public static let latentFeatures = 128

  /// Number of mel frames produced for `sampleCount` samples (torch.stft with center=True).
  public static func melFrameCount(sampleCount: Int) -> Int {
    return 1 + sampleCount / hopLength
  }

  /// Cuts `[startTime, startTime + duration)` out of a [channels, samples] waveform, zero-padding
  /// past the end of the file. Returns nil when the start lies beyond the end of the audio.
  public static func segment(
    _ waveform: Tensor<Float>, sampleRate: Int, startTime: Double, duration: Double
  ) -> Tensor<Float>? {
    precondition(waveform.shape.count == 2)
    let channels = waveform.shape[0]
    let totalSamples = waveform.shape[1]
    let startSample = max(0, Int((startTime * Double(sampleRate)).rounded()))
    let segmentSamples = max(1, Int((duration * Double(sampleRate)).rounded()))
    guard startSample < totalSamples else { return nil }
    let copiedSamples = min(segmentSamples, totalSamples - startSample)
    var segment = Tensor<Float>(
      Array(repeating: 0, count: channels * segmentSamples), .CPU, .NC(channels, segmentSamples))
    segment[0..<channels, 0..<copiedSamples] =
      waveform[0..<channels, startSample..<(startSample + copiedSamples)]
    return segment
  }
}

/// Log-mel spectrogram matching `ltx_core.model.audio_vae.ops.AudioProcessor`:
/// torchaudio MelSpectrogram(sample_rate 16000, n_fft 1024, win_length 1024, hop_length 160,
/// f_min 0, f_max 8000, n_mels 64, periodic Hann window, center=True, pad_mode reflect,
/// power 1.0, mel_scale slaney, norm slaney), then log(clamp(x, min: 1e-5)).
///
/// - Parameter waveform: [channels, samples] at 16 kHz. Mono input is duplicated to stereo,
///   since the encoder expects two input channels.
/// - Returns: [1, 2, frames, 64] on CPU, frames = 1 + samples / 160.
public func LTX2AudioMelSpectrogram(waveform: Tensor<Float>) -> Tensor<Float> {
  precondition(waveform.shape.count == 2)
  let inputChannels = waveform.shape[0]
  let sampleCount = waveform.shape[1]
  let nFFT = LTX2AudioEncoding.fftSize
  let hopLength = LTX2AudioEncoding.hopLength
  let melBins = LTX2AudioEncoding.melBins
  let sampleRate = LTX2AudioEncoding.sampleRate
  precondition(sampleCount > nFFT / 2, "Audio segment is too short for reflect padding.")
  let freqBins = nFFT / 2 + 1
  let frames = LTX2AudioEncoding.melFrameCount(sampleCount: sampleCount)
  // Periodic Hann window (torch.hann_window default).
  var window = [Float](repeating: 0, count: nFFT)
  for i in 0..<nFFT {
    window[i] = Float(0.5 * (1 - cos(2 * Double.pi * Double(i) / Double(nFFT))))
  }
  // DFT basis tables.
  var cosTable = [Float](repeating: 0, count: freqBins * nFFT)
  var sinTable = [Float](repeating: 0, count: freqBins * nFFT)
  for k in 0..<freqBins {
    for i in 0..<nFFT {
      let theta = -2 * Double.pi * Double((k * i) % nFFT) / Double(nFFT)
      cosTable[k * nFFT + i] = Float(cos(theta))
      sinTable[k * nFFT + i] = Float(sin(theta))
    }
  }
  // Slaney-scale mel filter bank with slaney normalization (torchaudio.functional.melscale_fbanks).
  func hzToMel(_ f: Double) -> Double {
    if f < 1_000 {
      return 3 * f / 200
    }
    return 15 + 27 * log(f / 1_000) / log(6.4)
  }
  func melToHz(_ m: Double) -> Double {
    if m < 15 {
      return 200 * m / 3
    }
    return 1_000 * exp(log(6.4) * (m - 15) / 27)
  }
  let fMax = Double(sampleRate) / 2
  let melPoints = (0..<(melBins + 2)).map {
    melToHz(hzToMel(fMax) * Double($0) / Double(melBins + 1))
  }
  var melFilters = [Float](repeating: 0, count: melBins * freqBins)
  for m in 0..<melBins {
    let fLow = melPoints[m]
    let fCenter = melPoints[m + 1]
    let fHigh = melPoints[m + 2]
    let norm = 2 / (fHigh - fLow)
    for k in 0..<freqBins {
      let freq = Double(k) * fMax / Double(freqBins - 1)
      let lower = (freq - fLow) / (fCenter - fLow)
      let upper = (fHigh - freq) / (fHigh - fCenter)
      let weight = max(0, min(lower, upper))
      melFilters[m * freqBins + k] = Float(weight * norm)
    }
  }
  var result = Tensor<Float>(.CPU, .NCHW(1, 2, frames, melBins))
  var samples = [Float](repeating: 0, count: sampleCount)
  var frame = [Float](repeating: 0, count: nFFT)
  var magnitude = [Float](repeating: 0, count: freqBins)
  let pad = nFFT / 2
  // Mono input is duplicated to stereo: compute it once and copy.
  let computedChannels = min(inputChannels, 2)
  for channel in 0..<computedChannels {
    for i in 0..<sampleCount {
      samples[i] = waveform[channel, i]
    }
    for t in 0..<frames {
      let start = t * hopLength - pad
      for i in 0..<nFFT {
        var index = start + i
        if index < 0 {
          index = -index
        } else if index >= sampleCount {
          index = 2 * (sampleCount - 1) - index
        }
        frame[i] = samples[index] * window[i]
      }
      for k in 0..<freqBins {
        var re: Float = 0
        var im: Float = 0
        let base = k * nFFT
        for i in 0..<nFFT {
          re += frame[i] * cosTable[base + i]
          im += frame[i] * sinTable[base + i]
        }
        magnitude[k] = (re * re + im * im).squareRoot()
      }
      for m in 0..<melBins {
        var accumulated: Float = 0
        let base = m * freqBins
        for k in 0..<freqBins {
          accumulated += melFilters[base + k] * magnitude[k]
        }
        result[0, channel, t, m] = log(max(accumulated, 1e-5))
      }
    }
  }
  if computedChannels == 1 {
    for t in 0..<frames {
      for m in 0..<melBins {
        result[0, 1, t, m] = result[0, 0, t, m]
      }
    }
  }
  return result
}

/// Encoder half of the LTX-2 / LTX-2.3 audio VAE (mel spectrogram to normalized latents), reading
/// the `audio_encoder` block of the audio-video VAE checkpoint. Decoding goes through the existing
/// audio decoder and vocoder of the video pipeline.
public struct LTX2AudioVAEEncoder {
  public let filePath: String

  public init(filePath: String) {
    self.filePath = filePath
  }

  /// Encodes a mel spectrogram into normalized, patched latents.
  /// - Parameters:
  ///   - mel: [1, 2, melFrames, 64] from `LTX2AudioMelSpectrogram`.
  ///   - latentsMean, latentsStd: the 128 per-feature audio latent statistics
  ///     (`ModelZoo.latentsScalingForModel(_:).audioMean` / `.audioStd`).
  /// - Returns: [1, latentFrames, 128] on CPU, in the layout the LTX-2 transformer and the
  ///   Draw Things audio decoder use (channel-major: index = channel * 16 + frequency).
  public func encode(
    _ mel: Tensor<Float>, latentsMean: [Float], latentsStd: [Float], graph: DynamicGraph
  ) -> Tensor<Float> {
    precondition(mel.shape.count == 4 && mel.shape[1] == 2 && mel.shape[3] == 64)
    precondition(latentsMean.count == 128 && latentsStd.count == 128)
    return graph.withNoGrad {
      let x = graph.variable(mel.toGPU(0))
      let (_, encoder) = LTX2AudioEncoderCausal2D(
        channels: [128, 256, 512], numRepeat: 2, startWidth: mel.shape[3],
        startHeight: mel.shape[2])
      encoder.maxConcurrency = .limit(4)
      encoder.compile(inputs: x)
      graph.openStore(
        filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
      ) {
        $0.read("audio_encoder", model: encoder, codec: [.jit, .externalData])
      }
      // [1, 16, latentFrames, 16]: 8 mean channels followed by 8 log-variance channels.
      let encoded = encoder(inputs: x)[0].as(of: Float.self)
      let latentFrames = encoded.shape[2]
      let frequencyBins = encoded.shape[3]
      precondition(frequencyBins * 8 == LTX2AudioEncoding.latentFeatures)
      // Keep the means, then patchify "b c t f -> b t (c f)".
      let means = encoded[0..<1, 0..<8, 0..<latentFrames, 0..<frequencyBins].copied()
      let patched = means.permuted(0, 2, 1, 3).contiguous().reshaped(
        .HWC(1, latentFrames, LTX2AudioEncoding.latentFeatures))
      // PerChannelStatistics.normalize: (x - mean) / std.
      let negativeMean = graph.variable(
        Tensor<Float>(latentsMean.map { -$0 }, .GPU(0), .HWC(1, 1, latentsMean.count)))
      let inverseStd = graph.variable(
        Tensor<Float>(latentsStd.map { 1 / $0 }, .GPU(0), .HWC(1, 1, latentsStd.count)))
      let normalized = inverseStd .* (patched + negativeMean)
      return normalized.rawValue.toCPU()
    }
  }
}

/// LTX-2.3 per-layer modulation outputs (index modulo 32, final norm at 32 + i) that are computed
/// from the audio timestep or the audio sigma. See `LTX2TransformerBlockFixed` for the order:
/// 3...5, 19...21, 25...27 audio AdaLN; 8, 9 and 13, 14 audio-side cross-attention scale / shift;
/// 10 the audio-to-video gate (driven by the audio sigma in ltx-core); 30, 31 audio prompt AdaLN;
/// 34, 35 audio output norm.
let LTX2_3AudioTimestepModulationIndices: Set<Int> = [
  3, 4, 5, 8, 9, 10, 13, 14, 19, 20, 21, 25, 26, 27, 30, 31, 34, 35,
]

/// Replaces the audio stream input of the LTX-2 transformer with the clean reference audio
/// latents ([1, audioFrames, 128]), so the audio stays fixed while the video is denoised.
func LTX2FrozenAudioInput<FloatType: TensorNumeric & BinaryFloatingPoint>(
  _ audioInput: DynamicGraph.Tensor<FloatType>, reference: DynamicGraph.AnyTensor
) -> DynamicGraph.Tensor<FloatType> {
  let reference = DynamicGraph.Tensor<FloatType>(reference)
  var audioInput = audioInput
  let frames = min(audioInput.shape[1], reference.shape[1])
  let channels = min(audioInput.shape[2], reference.shape[2])
  audioInput[0..<1, 0..<frames, 0..<channels] = reference[0..<1, 0..<frames, 0..<channels]
  return audioInput
}
