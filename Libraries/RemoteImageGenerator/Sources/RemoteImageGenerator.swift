import DataModels
import Diffusion
import Foundation
import GRPC
import GRPCModels
import GRPCServer
import ImageGenerator
import Logging
import ModelZoo
import NIO
import NNC

public enum RemoteImageGeneratorError: Error {
  case notConnected
}

extension DeviceType {
  init(from type: ImageGeneratorDeviceType) {
    switch type {
    case .phone:
      self = .phone
    case .tablet:
      self = .tablet
    case .laptop:
      self = .laptop
    }
  }
}

public struct RemoteImageGenerator: ImageGenerator {
  private let logger = Logger(label: "com.draw-things.remote-image-generator")
  public let client: ImageGenerationClientWrapper
  public let name: String
  public let deviceType: ImageGeneratorDeviceType
  private var fileExistsCall: UnaryCall<FileListRequest, FileExistenceResponse>? = nil

  public init(
    name: String, deviceType: ImageGeneratorDeviceType, client: ImageGenerationClientWrapper
  ) {
    self.name = name
    self.deviceType = deviceType
    self.client = client
  }

  public func generate(
    _ image: Tensor<FloatType>?, scaleFactor: Int, mask: Tensor<UInt8>?,
    hints: [(ControlHintType, [(AnyTensor, Float)])],
    text: String, negativeText: String, configuration: GenerationConfiguration, keywords: [String],
    cancellation: (@escaping () -> Void) -> Void,
    feedback: @escaping (ImageGeneratorSignpost, Set<ImageGeneratorSignpost>, Tensor<FloatType>?)
      -> Bool
  ) throws -> ([Tensor<FloatType>]?, Int) {
    guard let client = client.client else {
      throw RemoteImageGeneratorError.notConnected
    }
    let metadataOverride = ImageGeneratorUtils.metadataOverride(configuration)
    var overrideProto = MetadataOverride()
    let jsonEncoder = JSONEncoder()
    jsonEncoder.keyEncodingStrategy = .convertToSnakeCase
    overrideProto.models = (try? jsonEncoder.encode(metadataOverride.models)) ?? Data()
    overrideProto.loras = (try? jsonEncoder.encode(metadataOverride.loras)) ?? Data()
    overrideProto.controlNets = (try? jsonEncoder.encode(metadataOverride.controlNets)) ?? Data()
    overrideProto.textualInversions =
      (try? jsonEncoder.encode(
        keywords.compactMap {
          TextualInversionZoo.specificationForModel(
            TextualInversionZoo.modelFromKeyword($0, potentials: []) ?? "")
        })) ?? Data()

    var request = ImageGenerationRequest()
    request.configuration = configuration.toData()
    request.prompt = text
    request.negativePrompt = negativeText
    request.scaleFactor = Int32(scaleFactor)
    request.override = overrideProto
    request.keywords = keywords
    request.user = name
    request.device = DeviceType(from: deviceType)

    if let image = image {
      request.image = image.data(using: [.zip, .fpzip])
    }

    if let mask = mask {
      request.mask = mask.data(using: [.zip, .fpzip])
    }

    for (hintType, hintTensors) in hints {
      if !hintTensors.isEmpty {
        request.hints.append(
          HintProto.with {
            $0.hintType = hintType.rawValue
            $0.tensors = hintTensors.map { tensor, weight in
              return TensorAndWeight.with {
                $0.tensor = ImageGeneratorUtils.convertTensorToData(
                  tensor: tensor, using: [.zip, .fpzip])
                $0.weight = weight
              }
            }
          })
      }
    }

    // Send the request
    // handler is running on event group thread
    let logger = logger
    var tensors = [Tensor<FloatType>]()
    var scaleFactor: Int = 1
    var call: ServerStreamingCall<ImageGenerationRequest, ImageGenerationResponse>? = nil
    let callInstance = client.generateImage(request) { response in
      if !response.generatedImages.isEmpty {
        let imageTensors = response.generatedImages.compactMap { generatedImageData in
          return Tensor<FloatType>(data: generatedImageData, using: [.zip, .fpzip])
        }
        logger.info("Received generated image data")
        tensors.append(contentsOf: imageTensors)
      }
      if response.hasCurrentSignpost {
        let currentSignpost = ImageGeneratorSignpost(
          from: response.currentSignpost)
        let signpostsSet = Set(
          response.signposts.map { signpostProto in
            ImageGeneratorSignpost(from: signpostProto)
          })
        var previewTensor: Tensor<FloatType>? = nil
        if response.hasPreviewImage,
          let tensor = Tensor<FloatType>(
            data: response.previewImage, using: [.zip, .fpzip])
        {
          previewTensor = tensor
        }
        let isGenerating = feedback(currentSignpost, signpostsSet, previewTensor)
        if !isGenerating {
          logger.info("Stream cancel image generating")
          call?.cancel(promise: nil)
        }
      }
      if response.hasScaleFactor {
        scaleFactor = Int(response.scaleFactor)
      }
    }
    cancellation {
      callInstance.cancel(promise: nil)
    }
    call = callInstance

    // This is only for logging purpose, if you want to actually get the result, use the result of wait.
    callInstance.status.whenComplete { result in
      switch result {
      case .success(let status):
        logger.info("Stream completed with status: \(status)")
      case .failure(let error):
        logger.error("Stream failed with error: \(error)")
      }
    }
    let _ = try callInstance.status.wait()
    return (tensors, scaleFactor)
  }
}
