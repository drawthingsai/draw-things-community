import Foundation
import NNC

#if canImport(AVFoundation)
  import AVFoundation
#endif

public enum AudioInputError: Swift.Error, LocalizedError {
  case unsupportedPlatform
  case cannotOpenAudio(String)
  case cannotReadAudio(String)
  case cannotConvertAudio
  case invalidSampleRate(Int)

  public var errorDescription: String? {
    switch self {
    case .unsupportedPlatform:
      return "Audio file conversion requires AVFoundation and is unavailable on this platform."
    case .cannotOpenAudio(let path):
      return "Cannot open audio file at \(path)."
    case .cannotReadAudio(let path):
      return "Cannot read audio file at \(path)."
    case .cannotConvertAudio:
      return "Cannot convert audio to mono PCM."
    case .invalidSampleRate(let sampleRate):
      return "Audio sample rate must be positive, but received \(sampleRate) Hz."
    }
  }
}

public struct AudioInput {
  public let samples: [Float]
  public let sampleRate: Int

  public init(samples: [Float], sampleRate: Int) {
    precondition(!samples.isEmpty)
    precondition(sampleRate > 0)
    self.samples = samples
    self.sampleRate = sampleRate
  }

  public func videoFrameCount(framesPerSecond: Int) -> Int {
    precondition(framesPerSecond > 0)
    return max(1, (samples.count * framesPerSecond + sampleRate - 1) / sampleRate)
  }

  public func waveformTensor(videoFrames: Int, framesPerSecond: Int) -> Tensor<Float> {
    precondition(videoFrames > 0)
    precondition(framesPerSecond > 0)
    var pcm = samples
    let targetSamples = (videoFrames * sampleRate + framesPerSecond - 1) / framesPerSecond
    if pcm.count < targetSamples {
      pcm.append(contentsOf: [Float](repeating: 0, count: targetSamples - pcm.count))
    } else if pcm.count > targetSamples {
      pcm.removeLast(pcm.count - targetSamples)
    }
    var tensor = Tensor<Float>(.CPU, .NC(2, targetSamples))
    for i in 0..<targetSamples {
      tensor[0, i] = pcm[i]
      tensor[1, i] = pcm[i]
    }
    return tensor
  }

  public init(contentsOf path: String, sampleRate: Int) throws {
    guard sampleRate > 0 else {
      throw AudioInputError.invalidSampleRate(sampleRate)
    }
    #if canImport(AVFoundation)
      let url = URL(fileURLWithPath: path)
      guard let file = try? AVAudioFile(forReading: url) else {
        throw AudioInputError.cannotOpenAudio(path)
      }
      guard
        let outputFormat = AVAudioFormat(
          commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate), channels: 1,
          interleaved: false),
        let converter = AVAudioConverter(from: file.processingFormat, to: outputFormat)
      else {
        throw AudioInputError.cannotConvertAudio
      }
      let inputCapacity: AVAudioFrameCount = 65_536
      var samples = [Float]()
      var inputEnded = false
      var readError: AudioInputError? = nil
      let ratio = Double(sampleRate) / file.processingFormat.sampleRate
      conversionLoop: while true {
        guard
          let outputBuffer = AVAudioPCMBuffer(
            pcmFormat: outputFormat,
            frameCapacity: AVAudioFrameCount((Double(inputCapacity) * ratio).rounded(.up) + 16))
        else {
          throw AudioInputError.cannotConvertAudio
        }
        var conversionError: NSError? = nil
        let status = converter.convert(to: outputBuffer, error: &conversionError) {
          packetCount, statusPointer in
          guard !inputEnded else {
            statusPointer.pointee = .endOfStream
            return nil
          }
          let remainingFrames = file.length - file.framePosition
          guard remainingFrames > 0 else {
            statusPointer.pointee = .endOfStream
            inputEnded = true
            return nil
          }
          let remainingCapacity = AVAudioFrameCount(
            min(remainingFrames, AVAudioFramePosition(inputCapacity)))
          guard
            let inputBuffer = AVAudioPCMBuffer(
              pcmFormat: file.processingFormat,
              frameCapacity: max(1, min(packetCount, remainingCapacity)))
          else {
            statusPointer.pointee = .endOfStream
            inputEnded = true
            readError = .cannotReadAudio(path)
            return nil
          }
          do {
            try file.read(into: inputBuffer)
          } catch {
            statusPointer.pointee = .endOfStream
            inputEnded = true
            readError = .cannotReadAudio(path)
            return nil
          }
          if inputBuffer.frameLength == 0 {
            statusPointer.pointee = .endOfStream
            inputEnded = true
            return nil
          }
          statusPointer.pointee = .haveData
          return inputBuffer
        }
        if let channelData = outputBuffer.floatChannelData, outputBuffer.frameLength > 0 {
          samples.append(
            contentsOf: UnsafeBufferPointer(
              start: channelData[0], count: Int(outputBuffer.frameLength)))
        }
        if let readError {
          throw readError
        }
        if let conversionError {
          throw conversionError
        }
        switch status {
        case .haveData, .inputRanDry:
          continue
        case .endOfStream:
          break conversionLoop
        case .error:
          throw AudioInputError.cannotConvertAudio
        @unknown default:
          throw AudioInputError.cannotConvertAudio
        }
      }
      guard !samples.isEmpty else {
        throw AudioInputError.cannotConvertAudio
      }
      self.init(samples: samples, sampleRate: sampleRate)
    #else
      throw AudioInputError.unsupportedPlatform
    #endif
  }
}
