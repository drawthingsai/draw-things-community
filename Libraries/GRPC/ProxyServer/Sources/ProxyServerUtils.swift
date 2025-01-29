import Foundation

public struct ProxyServerUtils {
  public static func modelCoefficient(from modelName: String) throws -> Double {
    switch modelName {
    case "v1":
      return 1.0
    case "v2":
      return 1.06
    case "sdxl_base_v0.9":
      return 1.18
    case "kandinsky2.1":
      return 1.53
    case "sdxl_refiner_v0.9":
      return 1.18
    case "ssd_1b":
      return 0.83
    case "sd3":
      return 1.05
    case "pixart":
      return 0.83
    case "auraflow":
      return 2.05
    case "flux1":
      return 2.6
    case "sd3_large":
      return 2.35
    default:
      throw NSError(
        domain: "GenerationCostError",
        code: 1,
        userInfo: [NSLocalizedDescriptionKey: "Unknown model: \(modelName)"]
      )
    }
  }

  public static func calculateGenerationCost(
    modelName: String,
    width: Int,
    height: Int,
    steps: Int,
    batchSize: Int = 1,
    cfgEnabled: Bool = false,
    scalingFactor: Double = 0.000001904
  ) throws -> Double {
    let modelCoefficient = try Self.modelCoefficient(from: modelName)

    let baseCalc = Double(width * height * steps)

    let cfgMultiplier = cfgEnabled ? 2.0 : 1.0

    return modelCoefficient * pow(baseCalc * scalingFactor, 1.9) * Double(batchSize) * cfgMultiplier
  }
}
