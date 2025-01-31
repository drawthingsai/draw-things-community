import DataModels
import Diffusion
import Foundation
import ModelZoo

public enum ModelCoefficientError: Error {
  case unsupportedModel(String)
  case wipModel(String)
}

public struct ProxyServerUtils {
  public static func modelCoefficient(from modelVersion: ModelVersion) throws -> Double {
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
      throw ModelCoefficientError.unsupportedModel("wurstchenStageC")
    case .wurstchenStageB:
      throw ModelCoefficientError.unsupportedModel("wurstchenStageB")
    case .hunyuanVideo:
      throw ModelCoefficientError.wipModel("hunyuanVideo")
    }
  }

  public static func calculateGenerationCost(
    modelVersion: ModelVersion,
    width: Int,
    height: Int,
    steps: Int,
    batchSize: Int = 1,
    cfgEnabled: Bool = false,
    scalingFactor: Double = 0.000001904
  ) throws -> Double {
    let modelCoefficient = try Self.modelCoefficient(from: modelVersion)

    let baseCalc = Double(width * height * steps)

    let cfgMultiplier = cfgEnabled ? 2.0 : 1.0

    return modelCoefficient * pow(baseCalc * scalingFactor, 1.9) * Double(batchSize) * cfgMultiplier
  }

  public static func calculateGenerationCost(from configuration: GenerationConfiguration) throws
    -> Double
  {
    guard let model = configuration.model else {
      throw ModelCoefficientError.unsupportedModel("empty model name")
    }
    let modelVersion = ModelZoo.versionForModel(model)
    var batchSize = Int(configuration.batchSize)
    var cfgEnabled =
      (configuration.guidanceScale - 1).magnitude > 1e-2
      || (configuration.startFrameCfg - 1).magnitude > 1e-2
    if modelVersion == .svdI2v || modelVersion == .hunyuanVideo {
      batchSize = Int(configuration.batchCount)
      cfgEnabled = (configuration.guidanceScale - 1).magnitude > 1e-2
    }

    return try ProxyServerUtils.calculateGenerationCost(
      modelVersion: modelVersion, width: Int(configuration.startWidth * 64),
      height: Int(configuration.startHeight * 64),
      steps: Int(configuration.steps),
      batchSize: batchSize,
      cfgEnabled: cfgEnabled)
  }

  public static func generationCostThreshold(from priority: String) -> Double {
    switch priority {
    case "community":
      return 15000  // around 120s
    case "plus":
      return 40000  // around 300s
    default:
      return 15000
    }
    return 15000
  }
}
