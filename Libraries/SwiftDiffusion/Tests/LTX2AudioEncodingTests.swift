import Foundation
import NNC
import XCTest

@testable import Diffusion

final class LTX2AudioEncodingTests: XCTestCase {
  /// 0.25 s of two-tone stereo test signal at 16 kHz (computed in Double, stored as Float).
  private func testWaveform() -> Tensor<Float> {
    let samples = 4_000
    var waveform = Tensor<Float>(.CPU, .NC(2, samples))
    for n in 0..<samples {
      let t = Double(n) / 16_000
      waveform[0, n] = Float(
        0.5 * sin(2 * Double.pi * 440 * t) + 0.25 * sin(2 * Double.pi * 3_000 * t + 0.3))
      waveform[1, n] = Float(
        0.4 * sin(2 * Double.pi * 220 * t + 1.0) + 0.1 * sin(2 * Double.pi * 6_500 * t))
    }
    return waveform
  }

  func testMelSpectrogramMatchesReference() {
    // (channel, frame, mel bin, log-mel). Reference: librosa STFT + slaney mel filters with the
    // settings of ltx-core's AudioProcessor (torchaudio MelSpectrogram, n_fft 1024, hop 160,
    // periodic Hann, center / reflect, power 1, slaney scale and norm), log(clamp(x, 1e-5)).
    // Values near the clamp are left out: they only measure float32 rounding.
    let expected: [(Int, Int, Int, Float)] = [
      (0, 0, 0, -0.92280),
      (0, 0, 3, -0.74385),
      (0, 0, 10, 0.20439),
      (0, 0, 20, -2.65795),
      (0, 0, 35, -3.74465),
      (0, 0, 50, -3.23944),
      (0, 0, 63, -4.33564),
      (0, 1, 0, -1.17421),
      (0, 1, 3, -0.99547),
      (0, 1, 10, -0.04025),
      (0, 1, 20, -2.90908),
      (0, 1, 35, -3.99579),
      (0, 1, 50, -3.49074),
      (0, 1, 63, -4.58694),
      (0, 12, 10, -3.50204),
      (0, 25, 0, -0.95648),
      (0, 25, 3, -0.77453),
      (0, 25, 10, 0.19497),
      (0, 25, 20, -2.56059),
      (0, 25, 35, -4.65334),
      (0, 25, 50, -3.57911),
      (0, 25, 63, -4.64229),
      (1, 0, 0, -1.07641),
      (1, 0, 3, 0.84123),
      (1, 0, 10, -2.63209),
      (1, 0, 20, -4.17299),
      (1, 0, 35, -6.30578),
      (1, 0, 50, -5.70287),
      (1, 0, 63, -3.75457),
      (1, 1, 0, -1.32822),
      (1, 1, 3, 0.70762),
      (1, 1, 10, -2.88314),
      (1, 1, 20, -4.42423),
      (1, 1, 35, -6.55693),
      (1, 1, 50, -5.95416),
      (1, 1, 63, -4.00590),
      (1, 12, 3, 0.23442),
      (1, 25, 0, -0.96384),
      (1, 25, 3, 0.35332),
      (1, 25, 10, -2.46308),
      (1, 25, 20, -3.84492),
      (1, 25, 35, -5.02208),
      (1, 25, 50, -5.28982),
      (1, 25, 63, -4.00170),
    ]
    let mel = LTX2AudioMelSpectrogram(waveform: testWaveform())
    XCTAssertEqual(mel.shape, [1, 2, 26, 64])
    for (channel, frame, bin, value) in expected {
      XCTAssertEqual(
        mel[0, channel, frame, bin], value, accuracy: 1e-3,
        "channel \(channel), frame \(frame), mel bin \(bin)")
    }
  }

  func testMelSpectrogramDuplicatesMonoInput() {
    let stereo = testWaveform()
    var mono = Tensor<Float>(.CPU, .NC(1, 4_000))
    for n in 0..<4_000 {
      mono[0, n] = stereo[0, n]
    }
    let monoMel = LTX2AudioMelSpectrogram(waveform: mono)
    let stereoMel = LTX2AudioMelSpectrogram(waveform: stereo)
    XCTAssertEqual(monoMel.shape, [1, 2, 26, 64])
    for frame in 0..<26 {
      for bin in 0..<64 {
        XCTAssertEqual(monoMel[0, 0, frame, bin], stereoMel[0, 0, frame, bin])
        XCTAssertEqual(monoMel[0, 1, frame, bin], monoMel[0, 0, frame, bin])
      }
    }
  }

  func testSegmentCutsAndZeroPads() {
    var waveform = Tensor<Float>(.CPU, .NC(1, 4))
    for i in 0..<4 {
      waveform[0, i] = Float(i + 1)
    }
    // Starts at sample 2 and asks for 4 samples: two copied, two zero-padded.
    let segment = LTX2AudioEncoding.segment(
      waveform, sampleRate: 2, startTime: 1, duration: 2)
    XCTAssertEqual(segment?.shape, [1, 4])
    XCTAssertEqual(segment?[0, 0], 3)
    XCTAssertEqual(segment?[0, 1], 4)
    XCTAssertEqual(segment?[0, 2], 0)
    XCTAssertEqual(segment?[0, 3], 0)
  }

  func testSegmentRejectsStartPastTheEnd() {
    let waveform = Tensor<Float>(.CPU, .NC(1, 4))
    XCTAssertNil(
      LTX2AudioEncoding.segment(waveform, sampleRate: 2, startTime: 2, duration: 1))
  }
}
