import Crypto
import DataModels
import Dflat
import Diffusion
import Foundation
import GRPC
import GRPCModels
import GRPCServer
import ImageGenerator
import Logging
import ModelZoo
import NIO
import NIOHPACK
import NNC
import OrderedCollections
import ProxyServer
import UserAccount

public enum RemoteImageGeneratorError: Error {
  case notConnected
  case blobStringGenerateFailed
  case jwtTokenGenerateFailed
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
  private let workspace: Workspace
  public init(
    name: String, deviceType: ImageGeneratorDeviceType, client: ImageGenerationClientWrapper,
    workspace: Workspace
  ) {
    self.name = name
    self.deviceType = deviceType
    self.client = client
    self.workspace = workspace
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
    var contents = OrderedDictionary<Data, Data>()

    if let image = image {
      let data = image.data(using: [.zip, .fpzip])
      let hash = Data(SHA256.hash(data: data))
      request.image = hash
      contents[hash] = data
    }

    if let mask = mask {
      let data = mask.data(using: [.zip, .fpzip])
      let hash = Data(SHA256.hash(data: data))
      request.mask = hash
      contents[hash] = data
    }

    // TODO: can check if hints is referenced from configuration to decide whether to send it or not.
    for (hintType, hintTensors) in hints {
      if !hintTensors.isEmpty {
        request.hints.append(
          HintProto.with {
            $0.hintType = hintType.rawValue
            $0.tensors = hintTensors.map { tensor, weight in
              return TensorAndWeight.with {
                let data = ImageGeneratorUtils.convertTensorToData(
                  tensor: tensor, using: [.zip, .fpzip])
                let hash = Data(SHA256.hash(data: data))
                $0.tensor = hash
                contents[hash] = data
                $0.weight = weight
              }
            }
          })
      }
    }
    // If this is txt2img, there is not controlnet, no modifier, not inpainting.
    let modifier: SamplerModifier =
      configuration.model.map {
        ImageGeneratorUtils.modifierForModel($0, LoRAs: configuration.loras.compactMap(\.file))
      } ?? .none
    let isInpainting = ImageGeneratorUtils.isInpainting(for: mask, configuration: configuration)
    if configuration.strength == 1 && configuration.controls.isEmpty && modifier == .none
      && !isInpainting
    {
      // Don't need to send any data. This is a small optimization and this logic can be fragile.
      contents.removeAll()
    }
    var bearToken = ""
    if AccountManager.isSignedUser(workspace: workspace) {
      request.contents = []
      let encodedBlobString = try? request.serializedData().base64EncodedString()
      guard let encodedBlobString = encodedBlobString, encodedBlobString.count > 0 else {
        throw RemoteImageGeneratorError.blobStringGenerateFailed
      }
      let blob = AccountManager.GenerationPayload(blob: encodedBlobString)
      let group = DispatchGroup()

      var error: Error? = nil
      group.enter()
      AccountManager.authenticate(with: workspace, blob: blob) { jwtToken, jwtError in
        bearToken = jwtToken ?? ""
        error = jwtError
        group.leave()
      }
      group.wait()
      guard error == nil, bearToken.count != 0 else {
        throw RemoteImageGeneratorError.jwtTokenGenerateFailed
      }
    }
    request.contents = Array(contents.values)

    // Send the request
    // handler is running on event group thread
    let logger = logger
    var tensors = [Tensor<FloatType>]()
    var scaleFactor: Int = 1
    var call: ServerStreamingCall<ImageGenerationRequest, ImageGenerationResponse>? = nil
    var metadata = HPACKHeaders()
    metadata.add(name: "bear-token", value: bearToken)
    let callOptions = CallOptions(customMetadata: metadata)

    let callInstance = client.generateImage(request, callOptions: callOptions) { response in
      if !response.generatedImages.isEmpty {
        let imageTensors = response.generatedImages.compactMap { generatedImageData in
          if let image = Tensor<FloatType>(data: generatedImageData, using: [.zip, .fpzip]) {
            return Tensor<FloatType>(from: image)
          } else {
            return nil
          }
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
          previewTensor = Tensor<FloatType>(from: tensor)  // This force to convert the tensor into existing type.
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
