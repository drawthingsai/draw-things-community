import Atomics
import Crypto
import DataModels
import Diffusion
import Foundation
import GRPC
import GRPCModels
import ImageGenerator
import Logging
import ModelZoo
import NIOCore
import NNC

extension ImageGeneratorSignpost {
  public init(from signpostProto: ImageGenerationSignpostProto) {
    switch signpostProto.signpost {
    case .textEncoded:
      self = .textEncoded
    case .imageEncoded:
      self = .imageEncoded
    case .sampling(let sampling):
      self = .sampling(Int(sampling.step))
    case .imageDecoded:
      self = .imageDecoded
    case .secondPassImageEncoded:
      self = .secondPassImageEncoded
    case .secondPassSampling(let sampling):
      self = .secondPassSampling(Int(sampling.step))
    case .secondPassImageDecoded:
      self = .secondPassImageDecoded
    case .faceRestored:
      self = .faceRestored
    case .imageUpscaled:
      self = .imageUpscaled
    case .none:
      fatalError()
    }
  }
}

extension ImageGenerationSignpostProto {
  public init(from signpost: ImageGeneratorSignpost) {
    self = ImageGenerationSignpostProto.with {
      switch signpost {
      case .textEncoded:
        $0.signpost = .textEncoded(.init())
      case .imageEncoded:
        $0.signpost = .imageEncoded(.init())
      case .sampling(let step):
        $0.signpost = .sampling(
          .with {
            $0.step = Int32(step)
          })
      case .imageDecoded:
        $0.signpost = .imageDecoded(.init())
      case .secondPassImageEncoded:
        $0.signpost = .secondPassImageEncoded(.init())
      case .secondPassSampling(let step):
        $0.signpost = .secondPassSampling(
          .with {
            $0.step = Int32(step)
          })
      case .secondPassImageDecoded:
        $0.signpost = .secondPassImageDecoded(.init())
      case .faceRestored:
        $0.signpost = .faceRestored(.init())
      case .imageUpscaled:
        $0.signpost = .imageUpscaled(.init())
      }
    }
  }
}

extension ImageGeneratorDeviceType {
  public init?(from type: DeviceType) {
    switch type {
    case .phone:
      self = .phone
    case .tablet:
      self = .tablet
    case .laptop:
      self = .laptop
    case .UNRECOGNIZED:
      return nil
    }
  }
}

// Note that all these delegate callbacks will happen on main thread.
public protocol ImageGenerationServiceDelegate: AnyObject {
  func didReceiveGenerationRequest(
    cancelFlag: ManagedAtomic<Bool>, signposts: Set<ImageGeneratorSignpost>, user: String,
    deviceType: ImageGeneratorDeviceType)
  func didUpdateGenerationProgress(
    signpost: ImageGeneratorSignpost, signposts: Set<ImageGeneratorSignpost>)
  func didCompleteGenerationResponse(success: Bool)
}

public class ImageGenerationServiceImpl: ImageGenerationServiceProvider {

  private let queue: DispatchQueue
  private let backupQueue: DispatchQueue
  private let imageGenerator: ImageGenerator

  public weak var delegate: ImageGenerationServiceDelegate? = nil
  public var interceptors: ImageGenerationServiceServerInterceptorFactoryProtocol? = nil
  private let logger = Logger(label: "com.draw-things.image-generation-service")
  public let usesBackupQueue = ManagedAtomic<Bool>(false)

  public init(imageGenerator: ImageGenerator, queue: DispatchQueue, backupQueue: DispatchQueue) {
    self.imageGenerator = imageGenerator
    self.queue = queue
    self.backupQueue = backupQueue
  }
  // Implement the async generateImage method
  public func generateImage(
    request: ImageGenerationRequest,
    context: StreamingResponseCallContext<ImageGenerationResponse>
  ) -> EventLoopFuture<GRPCStatus> {
    // Log the incoming request
    logger.info("Received image processing request with prompt: \(request.prompt)")
    let eventLoop = context.eventLoop
    let promise = eventLoop.makePromise(of: GRPCStatus.self)
    let cancelFlag = ManagedAtomic<Bool>(false)
    context.closeFuture.whenComplete { _ in
      cancelFlag.store(true, ordering: .releasing)
    }
    func isCancelled() -> Bool {
      return cancelFlag.load(ordering: .acquiring)
    }

    let usesBackupQueue = usesBackupQueue.load(ordering: .acquiring)
    let queue = usesBackupQueue ? backupQueue : queue
    queue.async { [weak self] in
      guard let self = self else { return }

      let configuration = GenerationConfiguration.from(data: request.configuration)
      self.logger.info(
        "Received image processing request with decoded configuration: \(configuration), steps:\(configuration.steps)"
      )
      let override = request.override
      let jsonDecoder = JSONDecoder()
      jsonDecoder.keyDecodingStrategy = .convertFromSnakeCase
      let models =
        (try? jsonDecoder.decode(
          [FailableDecodable<ModelZoo.Specification>].self, from: override.models
        ).compactMap({ $0.value })) ?? []
      let loras =
        (try? jsonDecoder.decode(
          [FailableDecodable<LoRAZoo.Specification>].self, from: override.loras
        ).compactMap({ $0.value })) ?? []
      let controlNets =
        (try? jsonDecoder.decode(
          [FailableDecodable<ControlNetZoo.Specification>].self, from: override.controlNets
        ).compactMap({ $0.value })) ?? []
      let textualInversions =
        (try? jsonDecoder.decode(
          [FailableDecodable<TextualInversionZoo.Specification>].self,
          from: override.textualInversions
        ).compactMap({ $0.value })) ?? []
      ModelZoo.overrideMapping = Dictionary(models.map { ($0.file, $0) }) { v, _ in v }
      LoRAZoo.overrideMapping = Dictionary(loras.map { ($0.file, $0) }) { v, _ in v }
      ControlNetZoo.overrideMapping = Dictionary(controlNets.map { ($0.file, $0) }) { v, _ in v }
      TextualInversionZoo.overrideMapping = Dictionary(textualInversions.map { ($0.file, $0) }) {
        v, _ in v
      }
      defer {
        ModelZoo.overrideMapping = [:]
        LoRAZoo.overrideMapping = [:]
        ControlNetZoo.overrideMapping = [:]
        TextualInversionZoo.overrideMapping = [:]
      }
      let progressUpdateHandler:
        (ImageGeneratorSignpost, Set<ImageGeneratorSignpost>, Tensor<FloatType>?) -> Bool = {
          [weak self] signpost, signposts, previewTensor in
          guard let self = self else { return false }

          guard !isCancelled() else {
            self.logger.info(
              "cacncelled image generation for prompt: \(request.prompt)"
            )
            return false
          }

          let update = ImageGenerationResponse.with {
            $0.currentSignpost = ImageGenerationSignpostProto(from: signpost)
            $0.signposts = Array(
              signposts.map { signpost in
                ImageGenerationSignpostProto(from: signpost)
              })

            if let previewTensor = previewTensor {
              $0.previewImage = previewTensor.data(using: [.zip, .fpzip])
            }
          }

          context.sendResponse(update, promise: nil)
          if let delegate = self.delegate {
            DispatchQueue.main.async {
              delegate.didUpdateGenerationProgress(signpost: signpost, signposts: signposts)
            }
          }
          return true
        }

      let image: Tensor<FloatType>? =
        request.hasImage ? Tensor<FloatType>(data: request.image, using: [.zip, .fpzip]) : nil
      let mask: Tensor<UInt8>? =
        request.hasMask ? Tensor<UInt8>(data: request.mask, using: [.zip, .fpzip]) : nil

      var hints = [(ControlHintType, [(AnyTensor, Float)])]()
      for hintProto in request.hints {
        if let hintType = ControlHintType(rawValue: hintProto.hintType) {
          self.logger.info("Created ControlHintType: \(hintType)")
          if hintType == .scribble {
            if let tensorData = hintProto.tensors.first?.tensor,
              let score = hintProto.tensors.first?.weight,
              let hintTensor = Tensor<UInt8>(data: tensorData, using: [.zip, .fpzip])
            {
              hints.append((hintType, [(hintTensor, score)]))
            }
          } else {
            let tensors = hintProto.tensors.compactMap { tensorAndWeight in
              if let tensor = Tensor<FloatType>(
                data: tensorAndWeight.tensor, using: [.zip, .fpzip])
              {
                return (tensor, tensorAndWeight.weight)
              }
              return nil
            }
            hints.append((hintType, tensors))
          }

        } else {
          self.logger.error("Invalid ControlHintType \(hintProto.hintType)")
        }
      }

      let signposts = ImageGeneratorUtils.expectedSignposts(
        image != nil, mask: mask != nil, text: request.prompt, negativeText: request.negativePrompt,
        configuration: configuration, version: ModelZoo.versionForModel(configuration.model ?? ""))
      if let delegate = self.delegate {
        let user = request.user
        let deviceType = ImageGeneratorDeviceType(from: request.device) ?? .laptop
        DispatchQueue.main.async {
          delegate.didReceiveGenerationRequest(
            cancelFlag: cancelFlag, signposts: signposts, user: user, deviceType: deviceType)
        }
      }
      do {
        // Note that the imageGenerator must be local image generator, otherwise it throws.
        let (images, scaleFactor) = try self.imageGenerator.generate(
          image, scaleFactor: Int(request.scaleFactor), mask: mask, hints: hints,
          text: request.prompt, negativeText: request.negativePrompt, configuration: configuration,
          keywords: request.keywords, cancellation: { _ in }, feedback: progressUpdateHandler)

        let imageDatas =
          images?.compactMap { tensor in
            return tensor.data(using: [
              DynamicGraph.Store.Codec.zip, DynamicGraph.Store.Codec.fpzip,
            ])
          } ?? []
        self.logger.info("Image processed")

        let finalResponse = ImageGenerationResponse.with {
          if !imageDatas.isEmpty {
            self.logger.info("Image processed successfully")
            $0.generatedImages.append(contentsOf: imageDatas)
          } else {
            self.logger.error("Image processed failed")
          }
          $0.scaleFactor = Int32(scaleFactor)
        }
        context.sendResponse(finalResponse, promise: nil)
        promise.succeed(.ok)
        context.statusPromise.succeed(.ok)
        let success = imageDatas.isEmpty ? false : true
        if let delegate = self.delegate {
          DispatchQueue.main.async {
            delegate.didCompleteGenerationResponse(success: success)
          }
        }
      } catch (let error) {
        promise.fail(error)
        context.statusPromise.fail(error)
        if let delegate = self.delegate {
          DispatchQueue.main.async {
            delegate.didCompleteGenerationResponse(success: false)
          }
        }
      }
    }
    return promise.futureResult
  }

  public func filesExist(request: FileListRequest, context: any StatusOnlyCallContext)
    -> EventLoopFuture<FileExistenceResponse>
  {
    logger.info("Received request for files exist: \(request.files)")
    var files = [String]()
    var existences = [Bool]()
    for file in request.files {
      let existence = ModelZoo.isModelDownloaded(file)
      files.append(file)
      existences.append(existence)
    }
    let response = FileExistenceResponse.with {
      $0.files = files
      $0.existences = existences
    }
    return context.eventLoop.makeSucceededFuture(response)
  }

  public func echo(request: GRPCModels.EchoRequest, context: any GRPC.StatusOnlyCallContext)
    -> NIOCore.EventLoopFuture<GRPCModels.EchoReply>
  {
    let response = EchoReply.with {
      $0.message = "Hello, \(request.name)!"
    }
    return context.eventLoop.makeSucceededFuture(response)
  }

  public func uploadFile(context: StreamingResponseCallContext<UploadResponse>) -> EventLoopFuture<
    (StreamEvent<FileUploadRequest>) -> Void
  > {
    logger.info("Received uploadFile request")

    var fileHandle: FileHandle?
    var totalBytesReceived: Int64 = 0
    var metadata:
      (file: String, expectedFileSize: Int64, expectedHash: Data, temporaryPath: String)? =
        nil

    // Register a cleanup closure that will be called on disconnection or failure
    context.statusPromise.futureResult.whenFailure { error in
      self.logger.info("Client disconnected or an error occurred: \(error)")
      if let fileHandle = fileHandle {
        do {
          try fileHandle.close()
          self.logger.info("File handle closed successfully.")
        } catch {
          self.logger.error("Failed to close file handle: \(error)")
        }
      }
      // Clean up the partially uploaded file from disk
      if let metadata = metadata {
        self.logger.info("Cleaning up partial file: \(metadata.temporaryPath)")
        try? FileManager.default.removeItem(atPath: metadata.temporaryPath)
      }
    }

    return context.eventLoop.makeSucceededFuture({
      (event: StreamEvent<FileUploadRequest>) -> Void in
      switch event {
      case .message(let uploadRequest):
        switch uploadRequest.request {
        case .initRequest(let initRequest):
          // Process the initial upload request
          do {

            // Create file
            let temporaryPath = ModelZoo.filePathForModelDownloaded(initRequest.filename + ".part")
            metadata = (
              file: initRequest.filename, expectedFileSize: initRequest.totalSize,
              expectedHash: initRequest.sha256, temporaryPath: temporaryPath
            )
            self.logger.info("Init upload for metadata: \(metadata as Any)")
            FileManager.default.createFile(atPath: temporaryPath, contents: nil)
            fileHandle = FileHandle(forWritingAtPath: temporaryPath)
            if fileHandle == nil {
              throw GRPCStatus(code: .internalError, message: "Failed to create file handle")
            }
            var initResponse = UploadResponse()
            initResponse.chunkUploadSuccess = true
            initResponse.filename = initRequest.filename
            initResponse.message = "File uploaded init successfully"
            let _ = context.sendResponse(initResponse)
          } catch {
            context.statusPromise.fail(error)
          }
        case .chunk(let chunk):
          do {
            guard let fileHandle = fileHandle else {
              throw GRPCStatus(code: .internalError, message: "Failed to create file handle")
            }

            self.logger.info(
              "Received chunk \(chunk.filename) \(chunk.content) chunk.offset:\(chunk.offset) totalBytesReceived:\(totalBytesReceived)"
            )

            if chunk.offset != totalBytesReceived {
              throw GRPCStatus(code: .dataLoss, message: "Received chunk with unexpected offset")
            }

            try fileHandle.write(contentsOf: chunk.content)
            totalBytesReceived += Int64(chunk.content.count)
            var chunkResponse = UploadResponse()
            chunkResponse.chunkUploadSuccess = true
            chunkResponse.filename = chunk.filename
            chunkResponse.receivedOffset = totalBytesReceived
            chunkResponse.message = "File uploaded init successfully"
            let _ = context.sendResponse(chunkResponse)
          } catch {
            context.statusPromise.fail(error)
          }
        case .none:
          self.logger.info("Received None uploadRequest \(uploadRequest)")
        }
      case .end:
        do {

          guard let metadata = metadata else {
            throw GRPCStatus(code: .internalError, message: "Missing File metadata")
          }

          guard let fileHandle = fileHandle else {
            throw GRPCStatus(
              code: .internalError, message: "file uploaded end, but Failed to create file handle")
          }
          try fileHandle.close()
          self.logger.info("uploaded filename: \(metadata.file).part")

          guard totalBytesReceived == metadata.expectedFileSize else {
            self.logger.error(
              "uploaded file size does not match expectation totalBytesReceived: \(totalBytesReceived) expectedFileSize:\(metadata.expectedFileSize)"
            )

            throw GRPCStatus(
              code: .invalidArgument,
              message:
                "uploaded file size does not match expectation totalBytesReceived: \(totalBytesReceived) expectedFileSize:\(metadata.expectedFileSize)"
            )
          }
          self.logger.info(
            "uploaded file size totalBytesReceived: \(totalBytesReceived) expectedFileSize:\(metadata.expectedFileSize)"
          )

          if !self.validateUploadedFile(
            atPath: metadata.temporaryPath, filename: metadata.file,
            expectedHash: metadata.expectedHash)
          {
            self.logger.error("File validation failed")
            try? FileManager.default.removeItem(atPath: metadata.temporaryPath)
            throw GRPCStatus(code: .dataLoss, message: "File validation failed")
          }
          self.logger.info("File uploaded successfully")
          try? FileManager.default.removeItem(
            atPath: ModelZoo.filePathForModelDownloaded(metadata.file))
          try? FileManager.default.moveItem(
            atPath: metadata.temporaryPath,
            toPath: ModelZoo.filePathForModelDownloaded(metadata.file))
          context.statusPromise.succeed(.ok)

        } catch {

          context.statusPromise.fail(error)
        }
      }
    })
  }

  private func validateUploadedFile(atPath path: String, filename: String, expectedHash: Data)
    -> Bool
  {
    do {
      let fileData = try Data(contentsOf: URL(fileURLWithPath: path), options: .mappedIfSafe)
      let computedHash = SHA256.hash(data: fileData).withUnsafeBytes {
        return Data(bytes: $0.baseAddress!, count: $0.count)
      }
      self.logger.info("expectedHash: \(expectedHash.map { String(format: "%02x", $0) }.joined())")
      self.logger.info("computedHash: \(computedHash.map { String(format: "%02x", $0) }.joined())")
      if computedHash != expectedHash {
        logger.error(
          "File hash mismatch for \(filename) expectedHash \(computedHash) expectedHash \(expectedHash)"
        )
        return false
      }
    } catch {
      logger.error("Failed to validate file type: \(error.localizedDescription)")
    }

    return true
  }
}