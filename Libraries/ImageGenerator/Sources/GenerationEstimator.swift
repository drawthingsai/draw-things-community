import DataModels
import Foundation
import ImageGenerator

public enum GenerationEstimator {
  // This is a value from 0 to 1, estimating progress made so far.
  public static func estimateProgressValue(
    from estimation: GenerationEstimation, signpost: ImageGeneratorSignpost,
    signposts: Set<ImageGeneratorSignpost>
  ) -> Float {
    var updatedEstimatedTotalDuration: TimeInterval = 0
    for signpost in signposts {
      switch signpost {
      case .textEncoded:
        updatedEstimatedTotalDuration += TimeInterval(estimation.textEncoded)
      case .imageEncoded:
        updatedEstimatedTotalDuration += TimeInterval(estimation.imageEncoded)
      case .sampling(let steps):
        updatedEstimatedTotalDuration += TimeInterval(estimation.samplingStep) * TimeInterval(steps)
      case .imageDecoded:
        updatedEstimatedTotalDuration += TimeInterval(estimation.imageDecoded)
      case .secondPassImageEncoded:
        updatedEstimatedTotalDuration += TimeInterval(estimation.secondPassImageEncoded)
      case .secondPassSampling(let steps):
        updatedEstimatedTotalDuration +=
          TimeInterval(estimation.secondPassSamplingStep) * TimeInterval(steps)
      case .secondPassImageDecoded:
        updatedEstimatedTotalDuration += TimeInterval(estimation.secondPassImageDecoded)
      case .faceRestored:
        updatedEstimatedTotalDuration += TimeInterval(estimation.faceRestored)
      case .imageUpscaled:
        updatedEstimatedTotalDuration += TimeInterval(estimation.imageUpscaled)
      }
    }
    let updatedDurationThisFar: TimeInterval
    switch signpost {
    case .textEncoded:
      updatedDurationThisFar = TimeInterval(estimation.textEncoded)
    case .imageEncoded:
      updatedDurationThisFar = TimeInterval(estimation.textEncoded + estimation.imageEncoded)
    case .sampling(let step):
      updatedDurationThisFar = TimeInterval(
        estimation.textEncoded + (signposts.contains(.imageEncoded) ? estimation.imageEncoded : 0)
          + estimation.samplingStep * Float(step))
    case .imageDecoded:
      let firstStageSampling: Float = signposts.reduce(0) { _, signpost in
        if case .sampling(let steps) = signpost {
          return estimation.samplingStep * Float(steps)
        }
        return 0
      }
      if signposts.contains(.secondPassImageDecoded) || signposts.contains(.faceRestored)
        || signposts.contains(.imageUpscaled)
      {
        updatedDurationThisFar = TimeInterval(
          estimation.textEncoded + (signposts.contains(.imageEncoded) ? estimation.imageEncoded : 0)
            + firstStageSampling + estimation.imageDecoded)
      } else {
        updatedDurationThisFar = updatedEstimatedTotalDuration
      }
    case .secondPassImageEncoded:
      let firstStageSampling: Float = signposts.reduce(0) { _, signpost in
        if case .sampling(let steps) = signpost {
          return estimation.samplingStep * Float(steps)
        }
        return 0
      }
      updatedDurationThisFar = TimeInterval(
        estimation.textEncoded + (signposts.contains(.imageEncoded) ? estimation.imageEncoded : 0)
          + firstStageSampling + estimation.imageDecoded + estimation.secondPassImageEncoded)
    case .secondPassSampling(let step):
      let firstStageSampling: Float = signposts.reduce(0) { _, signpost in
        if case .sampling(let steps) = signpost {
          return estimation.samplingStep * Float(steps)
        }
        return 0
      }
      updatedDurationThisFar = TimeInterval(
        estimation.textEncoded + (signposts.contains(.imageEncoded) ? estimation.imageEncoded : 0)
          + firstStageSampling + estimation.imageDecoded + estimation.secondPassImageEncoded
          + estimation.secondPassSamplingStep * Float(step))
    case .secondPassImageDecoded:
      let firstStageSampling: Float = signposts.reduce(0) { _, signpost in
        if case .sampling(let steps) = signpost {
          return estimation.samplingStep * Float(steps)
        }
        return 0
      }
      if signposts.contains(.faceRestored) || signposts.contains(.imageUpscaled) {
        let secondPassSampling: Float = signposts.reduce(0) { _, signpost in
          if case .secondPassSampling(let steps) = signpost {
            return estimation.secondPassSamplingStep * Float(steps)
          }
          return 0
        }
        updatedDurationThisFar = TimeInterval(
          estimation.textEncoded + (signposts.contains(.imageEncoded) ? estimation.imageEncoded : 0)
            + firstStageSampling + estimation.imageDecoded + estimation.secondPassImageEncoded
            + secondPassSampling + estimation.secondPassImageDecoded)
      } else {
        updatedDurationThisFar = updatedEstimatedTotalDuration
      }
    case .faceRestored:
      let firstStageSampling: Float = signposts.reduce(0) { _, signpost in
        if case .sampling(let steps) = signpost {
          return estimation.samplingStep * Float(steps)
        }
        return 0
      }
      let secondPassSampling: Float = signposts.reduce(0) { _, signpost in
        if case .secondPassSampling(let steps) = signpost {
          return estimation.secondPassSamplingStep * Float(steps)
        }
        return 0
      }
      updatedDurationThisFar = TimeInterval(
        estimation.textEncoded + (signposts.contains(.imageEncoded) ? estimation.imageEncoded : 0)
          + firstStageSampling + estimation.imageDecoded + estimation.secondPassImageEncoded
          + secondPassSampling + estimation.secondPassImageDecoded + estimation.faceRestored)
    case .imageUpscaled:
      updatedDurationThisFar = updatedEstimatedTotalDuration
    }
    return Float(updatedDurationThisFar / updatedEstimatedTotalDuration)
  }
}

public final class ProgressBarPrinter {
  private var hasPrintedBefore = false

  public init() {
  }

  private func terminalWidth() -> Int? {
    var winsize: winsize = winsize()
    let result = ioctl(STDOUT_FILENO, UInt(TIOCGWINSZ), &winsize)
    return result == 0 ? Int(winsize.ws_col) : nil
  }

  public func update(progress: Float) {
    if !hasPrintedBefore {
      print("")
      hasPrintedBefore = true
    }
    let percent = Int(round(progress * 100))
    let template = "Image generation: [] XXX%"
    // Note that in Xcode, terminalWidth() returns 0
    let terminalWidth = max(terminalWidth() ?? 0, template.count + 1)  // Minimum bar size of 1

    let barWidth = terminalWidth - template.count
    let filledWidth = Int(round(Float(barWidth) * progress))
    let filledPart = String(repeating: "█", count: filledWidth)
    let emptyPart = String(repeating: " ", count: barWidth - filledWidth)
    let bar = "[\(filledPart)\(emptyPart)]"

    var string = template
    string = string.replacingOccurrences(of: "[]", with: bar)
    let numberString = String(percent).padding(toLength: 3, withPad: " ", startingAt: 0)
    string = string.replacingOccurrences(of: "XXX", with: numberString)
    let lineClearString = "\u{1B}[1A\u{1B}[K"
    print("\(lineClearString)\(string)")
  }
}
