import Atomics
import Crypto
import DataModels
import Diffusion
import Foundation
import GRPC
import GRPCImageServiceModels
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
    cancellation: @escaping () -> Void, signposts: Set<ImageGeneratorSignpost>, user: String,
    deviceType: ImageGeneratorDeviceType)
  func didUpdateGenerationProgress(
    signpost: ImageGeneratorSignpost, signposts: Set<ImageGeneratorSignpost>)
  func didCompleteGenerationResponse(success: Bool)
}

public enum ImageGenerationServiceError: Error {
  case sharedSecret
}

public class ImageGenerationServiceImpl: ImageGenerationServiceProvider {

  private let queue: DispatchQueue
  private let backupQueue: DispatchQueue
  private let imageGenerator: ImageGenerator

  public weak var delegate: ImageGenerationServiceDelegate? = nil
  public var interceptors: ImageGenerationServiceServerInterceptorFactoryProtocol? = nil
  private let logger = Logger(label: "com.draw-things.image-generation-service")
  public let usesBackupQueue = ManagedAtomic<Bool>(false)
  public let responseCompression = ManagedAtomic<Bool>(false)
  public let enableModelBrowsing = ManagedAtomic<Bool>(false)
  public var sharedSecret: String? = nil

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
    logger.info("Received image processing request, begin.")
    let eventLoop = context.eventLoop
    if let sharedSecret = sharedSecret, !sharedSecret.isEmpty {
      guard request.sharedSecret == sharedSecret else {
        return eventLoop.makeFailedFuture(ImageGenerationServiceError.sharedSecret)
      }
    }
    let promise = eventLoop.makePromise(of: GRPCStatus.self)
    let cancelFlag = ManagedAtomic<Bool>(false)
    var cancellation: ProtectedValue<(() -> Void)?> = ProtectedValue(nil)
    func cancel() {
      cancelFlag.store(true, ordering: .releasing)
      cancellation.modify {
        $0?()
        $0 = nil
      }
    }
    context.closeFuture.whenComplete { _ in
      cancel()
    }
    func isCancelled() -> Bool {
      return cancelFlag.load(ordering: .acquiring)
    }

    let usesBackupQueue = usesBackupQueue.load(ordering: .acquiring)
    let responseCompression = responseCompression.load(ordering: .acquiring)
    let queue = usesBackupQueue ? backupQueue : queue
    queue.async { [weak self] in
      guard let self = self else { return }

      let configuration = GenerationConfiguration.from(data: request.configuration)
      self.logger.info(
        "Received image processing request with configuration steps: \(configuration.steps)"
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
              "cacncelled image generation"
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
              let codec: DynamicGraph.Store.Codec = responseCompression ? [.zip, .fpzip] : []
              $0.previewImage = previewTensor.data(using: codec)
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

      var contents = [Data: Data]()
      for content in request.contents {
        let hash = Data(SHA256.hash(data: content))
        contents[hash] = content
      }

      func unwrapData(_ data: Data) -> Data {
        guard data.count == 32 else { return data }
        // If it is 32-byte, that is sha256, unwrap.
        return contents[data] ?? Data()
      }
      // Additional conversion if needed.
      let image: Tensor<FloatType>? =
        request.hasImage
        ? Tensor<FloatType>(data: unwrapData(request.image), using: [.zip, .fpzip]).map {
          Tensor<FloatType>(from: $0)
        } : nil
      let mask: Tensor<UInt8>? =
        request.hasMask
        ? Tensor<UInt8>(data: unwrapData(request.mask), using: [.zip, .fpzip]) : nil

      var hints = [(ControlHintType, [(AnyTensor, Float)])]()
      for hintProto in request.hints {
        if let hintType = ControlHintType(rawValue: hintProto.hintType) {
          self.logger.info("Created ControlHintType: \(hintType)")
          if hintType == .scribble {
            if let tensorData = hintProto.tensors.first?.tensor,
              let score = hintProto.tensors.first?.weight,
              let hintTensor = Tensor<UInt8>(data: unwrapData(tensorData), using: [.zip, .fpzip])
            {
              hints.append((hintType, [(hintTensor, score)]))
            }
          } else {
            let tensors = hintProto.tensors.compactMap { tensorAndWeight in
              if let tensor = Tensor<FloatType>(
                data: unwrapData(tensorAndWeight.tensor), using: [.zip, .fpzip])
              {
                // Additional conversion, if needed.
                return (Tensor<FloatType>(from: tensor), tensorAndWeight.weight)
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
            cancellation: cancel, signposts: signposts, user: user, deviceType: deviceType)
        }
      }
      do {
        // Note that the imageGenerator must be local image generator, otherwise it throws.
        let (images, scaleFactor) = try self.imageGenerator.generate(
          image, scaleFactor: Int(request.scaleFactor), mask: mask, hints: hints,
          text: request.prompt, negativeText: request.negativePrompt, configuration: configuration,
          keywords: request.keywords,
          cancellation: { cancellationBlock in
            cancellation.modify {
              $0 = cancellationBlock
            }
          }, feedback: progressUpdateHandler)

        let codec: DynamicGraph.Store.Codec = responseCompression ? [.zip, .fpzip] : []
        let imageDatas =
          images?.compactMap { tensor in
            return tensor.data(using: codec)
          } ?? []
        self.logger.info("Image processed")

        let finalResponse = ImageGenerationResponse.with {
          if !imageDatas.isEmpty {
            self.logger.info("Image processed successfully")
            $0.generatedImages.append(contentsOf: imageDatas)
          } else {
            if isCancelled() {
              self.logger.info("Image processed cancelled, generated images return nil")
            } else {
              self.logger.error("Image processed failed")
            }
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
    if let sharedSecret = sharedSecret, !sharedSecret.isEmpty {
      guard request.sharedSecret == sharedSecret else {
        return context.eventLoop.makeFailedFuture(ImageGenerationServiceError.sharedSecret)
      }
    }
    var files = [String]()
    var existences = [Bool]()
    var hashes = [Data]()
    let needsToComputeHash = Set<String>(request.filesWithHash)
    for file in request.files {
      let existence = ModelZoo.isModelDownloaded(file)
      files.append(file)
      existences.append(existence)
      if needsToComputeHash.contains(file) {
        let filePath = ModelZoo.filePathForModelDownloaded(file)
        if let fileData = try? Data(
          contentsOf: URL(fileURLWithPath: filePath), options: .mappedIfSafe)
        {
          let computedHash = Data(SHA256.hash(data: fileData))
          hashes.append(computedHash)
        } else {
          hashes.append(Data())
        }
      } else {
        hashes.append(Data())
      }
    }
    let response = FileExistenceResponse.with {
      $0.files = files
      $0.existences = existences
      $0.hashes = hashes
    }
    return context.eventLoop.makeSucceededFuture(response)
  }

  public func echo(
    request: GRPCImageServiceModels.EchoRequest, context: any GRPC.StatusOnlyCallContext
  )
    -> NIOCore.EventLoopFuture<GRPCImageServiceModels.EchoReply>
  {
    let enableModelBrowsing = enableModelBrowsing.load(ordering: .acquiring)
    let response = EchoReply.with {
      logger.info("Received echo from: \(request.name), enableModelBrowsing:\(enableModelBrowsing)")
      if let sharedSecret = sharedSecret, !sharedSecret.isEmpty {
        guard request.sharedSecret == sharedSecret else {
          // Mismatch on shared secret.
          $0.sharedSecretMissing = true
          return
        }
      }
      $0.sharedSecretMissing = false
      $0.message = "HELLO \(request.name)"
      if enableModelBrowsing {
        // Looking for ckpt files.
        let internalFilePath = ModelZoo.internalFilePathForModelDownloaded("")
        let fileManager = FileManager.default
        var fileUrls = [URL]()
        if let urls = try? fileManager.contentsOfDirectory(
          at: URL(fileURLWithPath: internalFilePath), includingPropertiesForKeys: [.fileSizeKey],
          options: [.skipsHiddenFiles, .skipsPackageDescendants, .skipsSubdirectoryDescendants])
        {
          fileUrls.append(contentsOf: urls)
        }
        if let externalUrl = ModelZoo.externalUrl,
          let urls = try? fileManager.contentsOfDirectory(
            at: externalUrl, includingPropertiesForKeys: [.fileSizeKey],
            options: [.skipsHiddenFiles, .skipsPackageDescendants, .skipsSubdirectoryDescendants])
        {
          fileUrls.append(contentsOf: urls)
        }
        // Check if the file ends with ckpt. If it is, this is a file we need to fill.
        $0.files = fileUrls.compactMap {
          guard let values = try? $0.resourceValues(forKeys: [.fileSizeKey]) else { return nil }
          guard let fileSize = values.fileSize, fileSize > 0 else { return nil }
          let file = $0.lastPathComponent
          guard file.lowercased().hasSuffix(".ckpt") else { return nil }
          return file
        }
        // Load all specifications that is available locally into override JSON payload.
        let models = ModelZoo.availableSpecifications.filter {
          return ModelZoo.isModelDownloaded($0)
        }
        let loras = LoRAZoo.availableSpecifications.filter {
          return LoRAZoo.isModelDownloaded($0)
        }
        let controlNets = ControlNetZoo.availableSpecifications.filter {
          return ControlNetZoo.isModelDownloaded($0)
        }
        let textualInversions = TextualInversionZoo.availableSpecifications.filter {
          return TextualInversionZoo.isModelDownloaded($0.file)
        }
        $0.override = MetadataOverride.with {
          let jsonEncoder = JSONEncoder()
          jsonEncoder.keyEncodingStrategy = .convertToSnakeCase
          $0.models = (try? jsonEncoder.encode(models)) ?? Data()
          $0.loras = (try? jsonEncoder.encode(loras)) ?? Data()
          $0.controlNets = (try? jsonEncoder.encode(controlNets)) ?? Data()
          $0.textualInversions = (try? jsonEncoder.encode(textualInversions)) ?? Data()
        }
      }
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

    let sharedSecret = sharedSecret
    return context.eventLoop.makeSucceededFuture({
      (event: StreamEvent<FileUploadRequest>) -> Void in
      switch event {
      case .message(let uploadRequest):
        if let sharedSecret = sharedSecret, !sharedSecret.isEmpty {
          guard uploadRequest.sharedSecret == sharedSecret else {
            context.statusPromise.fail(ImageGenerationServiceError.sharedSecret)
            return
          }
        }
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
      let computedHash = Data(SHA256.hash(data: fileData))
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
