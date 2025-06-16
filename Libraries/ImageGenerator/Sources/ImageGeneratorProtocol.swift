import DataModels
import Diffusion
import Foundation
import NNC

public enum ImageGeneratorSignpost: Equatable & Hashable {
  case textEncoded
  case imageEncoded
  case controlsGenerated
  case sampling(Int)
  case imageDecoded
  case secondPassImageEncoded
  case secondPassSampling(Int)
  case secondPassImageDecoded
  case faceRestored
  case imageUpscaled
}

public enum ImageGeneratorDeviceType {
  case phone
  case tablet
  case laptop
}

public protocol ImageGenerator {
  func generate(
    _ image: Tensor<FloatType>?, scaleFactor: Int, mask: Tensor<UInt8>?,
    hints: [(ControlHintType, [(AnyTensor, Float)])], text: String, negativeText: String,
    configuration: GenerationConfiguration, fileMapping: [String: String], keywords: [String],
    cancellation: (@escaping () -> Void) -> Void,
    feedback: @escaping (ImageGeneratorSignpost, Set<ImageGeneratorSignpost>, Tensor<FloatType>?)
      -> Bool
  ) throws -> ([Tensor<FloatType>]?, Int)
}

extension ImageGeneratorSignpost {
  public var description: String {
    switch self {
    case .textEncoded:
      return "text_encoded"
    case .imageEncoded:
      return "image_encoded"
    case .controlsGenerated:
      return "controls_generated"
    case .sampling(_):
      return "sampling"
    case .imageDecoded:
      return "image_decoded"
    case .secondPassImageEncoded:
      return "second_pass_image_encoded"
    case .secondPassSampling(_):
      return "second_pass_sampling"
    case .secondPassImageDecoded:
      return "second_pass_image_decoded"
    case .faceRestored:
      return "face_restored"
    case .imageUpscaled:
      return "image_upscaled"
    }
  }
}
