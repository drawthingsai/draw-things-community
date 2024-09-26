import DataModels
import Diffusion
import Foundation
import NNC

public enum ImageGeneratorSignpost: Equatable & Hashable {
  case textEncoded
  case imageEncoded
  case sampling(Int)
  case imageDecoded
  case secondPassImageEncoded
  case secondPassSampling(Int)
  case secondPassImageDecoded
  case faceRestored
  case imageUpscaled
}

public protocol ImageGenerator {

  func generate(
    _ image: Tensor<FloatType>?, scaleFactor: Int, mask: Tensor<UInt8>?,
    hints: [(ControlHintType, [(AnyTensor, Float)])],
    text: String,
    negativeText: String, configuration: GenerationConfiguration,
    feedback: @escaping (ImageGeneratorSignpost, Set<ImageGeneratorSignpost>, Tensor<FloatType>?)
      -> Bool
  ) -> ([Tensor<FloatType>]?, Int)
}