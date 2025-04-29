import DataModels
import Diffusion
import Foundation

public enum ComputeUnits {
  private static func modelCoefficient(_ modelVersion: ModelVersion) -> Double {
    switch modelVersion {
    case .v1:
      return 0.5
    case .v2:
      return 0.5294117647
    case .sdxlBase:
      return 0.5882352941
    case .kandinsky21:
      return 0.7647058824
    case .sdxlRefiner:
      return 0.5588235294
    case .ssd1b:
      return 0.4117647059
    case .sd3:
      return 0.5294117647
    case .pixart:
      return 0.4117647059
    case .auraflow:
      return 1.029411765
    case .flux1:
      return 2.588235294
    case .sd3Large:
      return 1.176470588
    case .svdI2v:
      return 0.88 * 0.8
    case .wurstchenStageC:
      return 1.18
    case .wurstchenStageB:
      return 1.18
    case .hunyuanVideo:
      return 3.529411765 * 0.8
    case .wan21_1_3b:
      return 1.176470588 * 0.8
    case .wan21_14b:
      return 2.823529412 * 0.8
    case .hiDreamI1:
      return 2.84465488969
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
    let samplerModifier: SamplerModifier
    let isGuidanceEmbedEnabled: Bool
    let isConsistencyModel: Bool

    if let overrideMapping = overrideMapping, let specification = overrideMapping[model] {
      modelVersion = specification.version
      samplerModifier = specification.modifier ?? .none
      isGuidanceEmbedEnabled =
        (specification.guidanceEmbed ?? false)
        && configuration.speedUpWithGuidanceEmbed
      isConsistencyModel = specification.isConsistencyModel ?? false
    } else {
      modelVersion = ModelZoo.versionForModel(model)
      samplerModifier = ModelZoo.modifierForModel(model)
      isGuidanceEmbedEnabled =
        ModelZoo.guidanceEmbedForModel(model) && configuration.speedUpWithGuidanceEmbed
      isConsistencyModel = ModelZoo.isConsistencyModelForModel(model)
    }

    let batchSize: Int
    let numFrames: Int

    let isCfgEnabled =
      (!isConsistencyModel && !isGuidanceEmbedEnabled)
      && isCfgEnabled(
        textGuidanceScale: configuration.guidanceScale,
        imageGuidanceScale: configuration.imageGuidanceScale,
        startFrameCfg: configuration.startFrameCfg, version: modelVersion, modifier: samplerModifier
      )
    let (cfgChannels, _) = cfgChannelsAndInputChannels(
      channels: 0, conditionShape: nil, isCfgEnabled: isCfgEnabled,
      textGuidanceScale: configuration.guidanceScale,
      imageGuidanceScale: configuration.imageGuidanceScale, version: modelVersion,
      modifier: samplerModifier)
    switch modelVersion {
    case .v1, .v2, .kandinsky21, .sdxlBase, .sdxlRefiner, .ssd1b, .wurstchenStageC,
      .wurstchenStageB, .sd3, .pixart, .auraflow, .flux1, .sd3Large, .hiDreamI1:
      batchSize = max(1, Int(configuration.batchSize)) * cfgChannels
      numFrames = 1
    case .svdI2v:
      batchSize = cfgChannels
      numFrames = Int(configuration.numFrames)
    case .hunyuanVideo, .wan21_1_3b, .wan21_14b:
      batchSize = cfgChannels
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
