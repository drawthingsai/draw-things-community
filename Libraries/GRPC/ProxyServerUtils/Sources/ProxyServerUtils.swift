import Diffusion
import Foundation

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
