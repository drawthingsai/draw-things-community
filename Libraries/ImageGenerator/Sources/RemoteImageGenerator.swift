import DataModels
import Diffusion
import Foundation
import GRPC
import GRPCModels
import GRPCServer
import ImageGenerator
import Logging
import NIO
import NNC

public struct RemoteImageGenerator: ImageGenerator {
  let ip: String
  let port: Int
  let logger = Logger(label: "com.draw-things.remote-image-generator")

  public init(ip: String, port: Int) {
    self.ip = ip
    self.port = port
  }

  public func generate(
    _ image: Tensor<FloatType>?, scaleFactor: Int, mask: Tensor<UInt8>?,
    hints: [(ControlHintType, [(AnyTensor, Float)])],
    text: String,
    negativeText: String, configuration: GenerationConfiguration,
    feedback: @escaping (ImageGeneratorSignpost, Set<ImageGeneratorSignpost>, Tensor<FloatType>?)
      -> Bool
  ) -> ([Tensor<FloatType>]?, Int) {
    var tensors = [Tensor<FloatType>]()
    let group = MultiThreadedEventLoopGroup(numberOfThreads: 1)

    defer {
      try? group.syncShutdownGracefully()
    }

    var grpcConfiguration = GRPCChannelPool.Configuration.with(
      target: .host(ip, port: port),
      transportSecurity: .plaintext,
      eventLoopGroup: group)
    grpcConfiguration.maximumReceiveMessageLength = 16 * 1024 * 1024
    guard let channel = try? GRPCChannelPool.with(configuration: grpcConfiguration) else {
      return (nil, 1)
    }

    defer {
      try? channel.close().wait()
    }

    let client = ImageGeneratingServiceNIOClient(channel: channel)

    var request = ImageGeneratingRequest()
    request.configuration = configuration.toData()
    request.prompt = text
    request.negativePrompt = negativeText
    request.scaleFactor = Int32(scaleFactor)

    if let image = image {
      request.image = image.data(using: [.zip, .fpzip])
    }

    if let mask = mask {
      request.mask = mask.data(using: [.zip, .fpzip])
    }

    for (hintType, hintTensors) in hints {
      print(hintTensors)
      if hintType == .shuffle, !hintTensors.isEmpty {
        var hintMessage = HintProto()
        hintMessage.hintType = hintType.rawValue
        for (tensor, score) in hintTensors {
          var tensorWithScore = TensorWithScore()
          tensorWithScore.tensor = ImageGeneratorUtils.convertTensorToData(
            tensor: tensor, using: [.zip, .fpzip])
          tensorWithScore.score = score
          hintMessage.tensors.append(tensorWithScore)
        }
        request.hints.append(hintMessage)
      } else {

        if let tensor = hintTensors.first {
          var hintMessage = HintProto()
          hintMessage.hintType = hintType.rawValue
          var tensorWithScore = TensorWithScore()
          tensorWithScore.tensor = ImageGeneratorUtils.convertTensorToData(
            tensor: tensor.0, using: [.zip, .fpzip])
          tensorWithScore.score = tensor.1
          hintMessage.tensors = [tensorWithScore]
          request.hints.append(hintMessage)
        }
      }
    }
    var call: ServerStreamingCall<ImageGeneratingRequest, ImageGeneratingResponse>? = nil

    // Send the request
    // handler is running on event group thread
    call = client.generateImage(
      request,
      handler: { response in
        if !response.generatedImages.isEmpty {
          let imageTensors = response.generatedImages.compactMap { generatedImageData in
            return Tensor<FloatType>(data: generatedImageData, using: [.zip, .fpzip])
          }
          logger.info("Received generated image data")
          tensors.append(contentsOf: imageTensors)

        } else if response.hasCurrentSignpost {
          if response.hasCurrentSignpost {
            var step = 1
            if response.hasStep {
              step = Int(response.step)
            }
            let currentSignpost = ImageGeneratorSignpost(
              from: response.currentSignpost, step: step)
            let signpostsSet = Set(
              response.signposts.map { signpostProto in
                ImageGeneratorSignpost(from: signpostProto, step: 1)
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
        }

        if response.progress != 0 {
          logger.info("Progress: \(response.progress * 100)% - \(response.statusMessage)")
        }

      })
    guard let call = call else {
      logger.error("Failed to start stream connection")
      return (nil, 1)
    }

    call.status.whenComplete { result in
      switch result {
      case .success(let status):
        logger.info("Stream completed with status: \(status)")
      case .failure(let error):
        logger.error("Stream failed with error: \(error)")
      }
    }
    do {
      let _ = try call.status.wait()
    } catch {
      logger.error("Failed to receive stream completion: \(error)")
    }
    return (tensors, 1)

  }
}