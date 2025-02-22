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

  public static func from(
    _ configuration: GenerationConfiguration,
    overrideMapping: [String: ModelZoo.Specification]? = nil
  ) -> Int? {
    guard let model = configuration.model else {
      return nil
    }
    let modelVersion: ModelVersion
    let isGuidanceEmbedEnabled: Bool
    let isConsistencyModel: Bool

    if let overrideMapping = overrideMapping, let specification = overrideMapping[model] {
      modelVersion = specification.version
      isGuidanceEmbedEnabled =
        (specification.guidanceEmbed ?? false)
        && configuration.speedUpWithGuidanceEmbed
      isConsistencyModel = specification.isConsistencyModel ?? false
    } else {
      modelVersion = ModelZoo.versionForModel(model)
      isGuidanceEmbedEnabled =
        ModelZoo.guidanceEmbedForModel(model) && configuration.speedUpWithGuidanceEmbed
      isConsistencyModel = ModelZoo.isConsistencyModelForModel(model)
    }

    let batchSize: Int
    let numFrames: Int

    let isCfgEnabled =
      (!isConsistencyModel && !isGuidanceEmbedEnabled)
      && isCfgEnabled(
        textGuidanceScale: configuration.guidanceScale, startFrameCfg: configuration.startFrameCfg,
        version: modelVersion)
    switch modelVersion {
    case .v1, .v2, .kandinsky21, .sdxlBase, .sdxlRefiner, .ssd1b, .wurstchenStageC,
      .wurstchenStageB, .sd3, .pixart, .auraflow, .flux1, .sd3Large:
      batchSize = max(1, Int(configuration.batchSize)) * (isCfgEnabled ? 2 : 1)
      numFrames = 1
    case .svdI2v:
      batchSize = isCfgEnabled ? 2 : 1
      numFrames = Int(configuration.numFrames)
    case .hunyuanVideo:
      batchSize = isCfgEnabled ? 2 : 1
      numFrames = (Int(configuration.numFrames) - 1) / 4 + 1
    }
    let modelCoefficient = modelCoefficient(modelVersion)
    let root = Double(
      Int(configuration.startWidth) * 64 * Int(configuration.startHeight) * 64 * numFrames)
    let scalingFactor: Double = 0.00000922917

    return Int(
      (modelCoefficient * pow(root * scalingFactor, 1.9) * Double(configuration.steps)
        * Double(max(configuration.strength, 0.05)) * Double(batchSize)).rounded(.up))
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
