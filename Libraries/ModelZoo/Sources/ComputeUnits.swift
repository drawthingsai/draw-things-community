import DataModels
import Diffusion
import Foundation

public enum ComputeUnits {
  private static func modelCoefficient(_ modelVersion: ModelVersion) -> Double {
    switch modelVersion {
    case .v1:
      return 1.0
    case .v2:
      return 1.06
    case .sdxlBase:
      return 1.18
    case .kandinsky21:
      return 1.53
    case .sdxlRefiner:
      return 1.18
    case .ssd1b:
      return 0.83
    case .sd3:
      return 1.05
    case .pixart:
      return 0.83
    case .auraflow:
      return 2.05
    case .flux1:
      return 2.6
    case .sd3Large:
      return 2.35
    case .svdI2v:
      return 0.9
    case .wurstchenStageC:
      return 1.18
    case .wurstchenStageB:
      return 1.18
    case .hunyuanVideo:
      return 2.6
    }
  }

  public static func from(_ configuration: GenerationConfiguration) -> Int? {
    guard let model = configuration.model else {
      return nil
    }
    let modelVersion = ModelZoo.versionForModel(model)
    let batchSize: Int
    let numFrames: Int
    let isGuidanceEmbedEnabled =
      ModelZoo.guidanceEmbedForModel(model)
      && (configuration.speedUpWithGuidanceEmbed || modelVersion == .hunyuanVideo)
    let isCfgEnabled =
      (!ModelZoo.isConsistencyModelForModel(model) && !isGuidanceEmbedEnabled)
      || isCfgEnabled(
        textGuidanceScale: configuration.guidanceScale, startFrameCfg: configuration.startFrameCfg,
        version: modelVersion)
    switch modelVersion {
    case .v1, .v2, .kandinsky21, .sdxlBase, .sdxlRefiner, .ssd1b, .wurstchenStageC,
      .wurstchenStageB, .sd3, .pixart, .auraflow, .flux1, .sd3Large:
      batchSize = Int(configuration.batchSize) * (isCfgEnabled ? 2 : 1)
      numFrames = 1
    case .svdI2v, .hunyuanVideo:
      batchSize = isCfgEnabled ? 2 : 1
      numFrames = Int(configuration.numFrames)
    }
    let modelCoefficient = modelCoefficient(modelVersion)
    let root = Double(
      Int(configuration.startWidth) * 64 * Int(configuration.startHeight) * 64
        * Int(configuration.steps) * numFrames)
    let scalingFactor: Double = 0.000001904

    return Int((modelCoefficient * pow(root * scalingFactor, 1.9) * Double(batchSize)).rounded(.up))
  }

  public static func threadhold(for priority: String) -> Int {
    switch priority {
    case "community":
      return 15000  // around 120s
    case "plus":
      return 40000  // around 300s
    default:
      return 15000
    }
  }
}
