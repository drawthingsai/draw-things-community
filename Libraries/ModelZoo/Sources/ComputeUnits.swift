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
    case .wan22_5b:
      return 1.176470588 * 0.8
    case .hiDreamI1:
      return 2.84465488969
    case .qwenImage:
      return 2.84465488969
    }
  }

  public static func from(
    _ configuration: GenerationConfiguration,
    hasImage: Bool, shuffleCount: Int,
    overrideMapping: (
      model: [String: ModelZoo.Specification], lora: [String: LoRAZoo.Specification]
    )? = nil
  ) -> Int? {
    guard let model = configuration.model else {
      return nil
    }
    let modelVersion: ModelVersion
    let samplerModifier: SamplerModifier
    let isGuidanceEmbedEnabled: Bool
    let isConsistencyModel: Bool

    if let overrideMapping = overrideMapping, let specification = overrideMapping.model[model] {
      modelVersion = specification.version
      var modifier = specification.modifier ?? .none
      if modifier == .none {
        for lora in configuration.loras {
          guard let file = lora.file, let specification = overrideMapping.lora[file],
            let loraModifier = specification.modifier
          else { continue }
          if loraModifier != .none {
            modifier = loraModifier
            break
          }
        }
      }
      samplerModifier = modifier
      isGuidanceEmbedEnabled =
        (specification.guidanceEmbed ?? false)
        && configuration.speedUpWithGuidanceEmbed
      isConsistencyModel = specification.isConsistencyModel ?? false
    } else {
      modelVersion = ModelZoo.versionForModel(model)
      var modifier = ModelZoo.modifierForModel(model)
      for lora in configuration.loras {
        guard let file = lora.file else { continue }
        let loraModifier = LoRAZoo.modifierForModel(file)
        if loraModifier != .none {
          modifier = loraModifier
        }
      }
      samplerModifier = modifier
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
    var modelCoefficient = modelCoefficient(modelVersion)
    var root = Double(Int(configuration.startWidth) * 64 * Int(configuration.startHeight) * 64)
    switch modelVersion {
    case .v1, .v2, .kandinsky21, .sdxlBase, .sdxlRefiner, .ssd1b, .wurstchenStageC,
      .wurstchenStageB, .sd3, .pixart, .auraflow, .sd3Large:
      batchSize = max(1, Int(configuration.batchSize)) * cfgChannels
      numFrames = 1
    case .flux1, .qwenImage:
      batchSize = max(1, Int(configuration.batchSize)) * cfgChannels
      numFrames = 1
      if samplerModifier == .kontext || samplerModifier == .qwenimageEditPlus {  // For Kontext, if the reference image is provided, we effectively double the cost at least.
        root = root * Double(1 + (hasImage ? 1 : 0) + shuffleCount)
      }
    case .hiDreamI1:
      batchSize = max(1, Int(configuration.batchSize)) * cfgChannels
      numFrames = 1
      if samplerModifier == .editing {  // For HiDream E1, we extends the width, effectively double the resolution.
        root = root * 2
      }
    case .svdI2v:
      batchSize = cfgChannels
      numFrames = Int(configuration.numFrames)
    case .hunyuanVideo:
      batchSize = cfgChannels
      numFrames = (Int(configuration.numFrames) - 1) / 4 + 1
    case .wan21_1_3b, .wan21_14b, .wan22_5b:
      batchSize = cfgChannels
      numFrames = (Int(configuration.numFrames) - 1) / 4 + 1
      if configuration.causalInferenceEnabled && configuration.causalInference > 0
        && configuration.causalInference + max(0, configuration.causalInferencePad) < numFrames
      {
        // A perfect causal inference would be 1/2 cheaper than non-causal variant. But given causal inference for these models are based on full frames, it is not 1/2, it needs to add batch these extra edges.
        let sequenceLength = root * Double(numFrames)
        let lowerTriangle = sequenceLength * sequenceLength * 0.5
        let upperRidgeLength = root * Double(configuration.causalInference)
        // upperRidgePad is an rectangle.
        let upperRidgePad = upperRidgeLength * root * Double(configuration.causalInferencePad)
        let upperRidge = upperRidgeLength * upperRidgeLength * 0.5 + upperRidgePad
        let totalArea =
          lowerTriangle + upperRidge * Double(configuration.causalInference) / Double(numFrames)
        modelCoefficient = modelCoefficient * (totalArea / (sequenceLength * sequenceLength))
      }
    }
    root = root * Double(numFrames)
    let scalingFactor: Double = 0.00000922917

    return Int(
      (modelCoefficient * pow(root * scalingFactor, 1.9) * Double(configuration.steps)
        * Double(max(configuration.strength, 0.05)) * Double(batchSize)).rounded(.up))
  }

  public static func threshold(for priority: String?) -> Int {
    switch priority {
    case "community":
      return 15000  // around 120s
    case "plus":
      return 40000  // around 300s
    case nil:
      return 15000
    default:
      return 15000
    }
  }

  public static func threshold(
    for priority: String?, computeUnitPolicy: [String: Int]?, expirationTimestamp: Date?
  ) -> Int {
    // Check if we have a valid policy and it's not expired
    let currentTimestamp = Date()
    if let policy = computeUnitPolicy,
      let expiration = expirationTimestamp,
      currentTimestamp < expiration, let priority = priority,
      let policyValue = policy[priority]
    {
      return policyValue
    }

    // Fallback to original default setup
    return threshold(for: priority)
  }
}
