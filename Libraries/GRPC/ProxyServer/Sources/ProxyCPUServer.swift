import BinaryResources
import Crypto
import Foundation
import GRPC
import GRPCControlPanelModels
import GRPCImageServiceModels
import Logging
import ModelZoo
import NIO
import NIOHPACK
import NIOHTTP2
import NIOSSL

#if canImport(FoundationNetworking)
  import FoundationNetworking
#endif

public enum ProxyTaskPriority: Sendable {
  case high
  case low
}

public struct WorkTask {
  let priority: TaskPriority
  let request: ImageGenerationRequest
  let context: StreamingResponseCallContext<ImageGenerationResponse>
  let promise: EventLoopPromise<GRPCStatus>
}

public actor Worker {
  let name: String
  let primaryPriority: ProxyTaskPriority
  public let client: ProxyGPUClientWrapper
  private let logger = Logger(label: "com.draw-things.image-generation-proxy-service")
  enum WorkerError: Error {
    case invalidNioClient
  }
  public init(
    name: String, client: ProxyGPUClientWrapper, primaryPriority: ProxyTaskPriority
  ) {
    self.name = name
    self.client = client
    self.primaryPriority = primaryPriority
  }

  func executeTask(_ task: WorkTask) async {
    self.logger.info(
      "Worker \(name) primaryPriority:\(primaryPriority) starting task  (Priority: \(task.priority))"
    )
    do {
      var call: ServerStreamingCall<ImageGenerationRequest, ImageGenerationResponse>? = nil
      guard let client = client.client else {
        throw WorkerError.invalidNioClient
      }
      let callInstance = client.generateImage(task.request) { response in
        task.context.sendResponse(response).whenComplete { result in
          switch result {
          case .success:
            self.logger.debug("forward response: \(response)")
          case .failure(let error):
            self.logger.error("forward response error \(error)")
            call?.cancel(promise: nil)
            task.promise.fail(error)
          }
        }
      }

      call = callInstance
      task.context.closeFuture.whenComplete { _ in
        callInstance.cancel(promise: nil)
      }

      let status = try await callInstance.status.get()
      task.promise.succeed(status)

    } catch {
      logger.error("forward response error \(error)")

      task.promise.fail(error)
    }
  }
}

actor TaskCoordinator {
  private var highPriorityTasks: [WorkTask] = []
  private var lowPriorityTasks: [WorkTask] = []
  private var pendingRemoveWorkerId = Set<String>()
  private var workers: [String: Worker]

  private let logger: Logger

  // Shared availability stream
  private var workerAvailabilityStream: AsyncStream<Worker>
  private var availabilityContinuation: AsyncStream<Worker>.Continuation

  init(workers: [String: Worker], logger: Logger) {
    self.logger = logger
    self.workers = workers

    let (stream, continuation) = AsyncStream.makeStream(of: Worker.self)
    self.workerAvailabilityStream = stream
    self.availabilityContinuation = continuation
    for (name, worker) in workers {
      availabilityContinuation.yield(worker)
    }
  }

  func nextWorker() async -> Worker? {
    var iterator = workerAvailabilityStream.makeAsyncIterator()

    while let worker = await iterator.next() {
      if workers[worker.name] != nil {
        return worker
      } else {
        logger.info("skip removed worker:\(worker) from workerAvailabilityStream")
      }
    }
    return nil
  }

  func nextTaskForWorker(_ worker: Worker) async -> WorkTask? {
    let isPrimaryHigh = worker.primaryPriority == .high

    // Try primary queue first
    if isPrimaryHigh {
      if let task = highPriorityTasks.first {
        highPriorityTasks.removeFirst()
        return task
      }
      if let task = lowPriorityTasks.first {
        lowPriorityTasks.removeFirst()
        return task
      }
    } else {
      if let task = lowPriorityTasks.first {
        lowPriorityTasks.removeFirst()
        return task
      }
      if let task = highPriorityTasks.first {
        highPriorityTasks.removeFirst()
        return task
      }
    }

    return nil
  }

  func addTask(_ task: WorkTask) {
    if task.priority == .high {
      logger.info("highPriorityTasks append task \(task.priority)")
      highPriorityTasks.append(task)
    } else {
      logger.info("lowPriorityTasks append task \(task.priority)")
      lowPriorityTasks.append(task)
    }
  }

  func addWorkerBackToStream(_ worker: Worker) async {
    guard workers[worker.name] != nil else {
      logger.error("worker:\(worker) is removed, can not be added to worker stream")
      return
    }
    logger.info("add worker:\(worker) back to worker stream")
    availabilityContinuation.yield(worker)
  }

  func addWorker(_ worker: Worker) async {
    guard worker.client.client != nil else {
      logger.error(
        "can add worker:\(worker) to worker TaskCoordinator with invalid nioclient connection")
      return
    }
    workers[worker.name] = worker
    availabilityContinuation.yield(worker)
    logger.info("add worker:\(worker) to worker TaskCoordinator and stream")
  }

  func removeWorkerBasedOnName(_ name: String) async {
    guard let worker = workers[name] else {
      logger.error("failed to find worker based on name \(name)")
      return
    }
    try? worker.client.disconnect()
    workers[worker.name] = nil
    logger.info("remove worker:\(worker) from worker TaskCoordinator")
  }

  deinit {
    for (_, worker) in workers {
      try? worker.client.disconnect()
    }
    availabilityContinuation.finish()
  }
}

public actor ControlConfigs {
  private var throtellPolicy = [String: Int]()
  public private(set) var publicPem: String
  private let logger: Logger
  enum ControlConfigsError: Error {
    case updatePublicKeyFailed(message: String)
  }

  init(throtellPolicy: [String: Int], publicPem: String, logger: Logger) {
    self.throtellPolicy = throtellPolicy
    self.publicPem = publicPem
    self.logger = logger
  }

  func updateThrotellPolicy(newPolicies: [String: Int]) async {
    for (key, value) in newPolicies {
      throtellPolicy[key] = value
    }
  }
  func updatePublicPem() async throws {
    guard let url = URL(string: "https://api.drawthings.ai/key") else {
      self.logger.error("ControlConfigs failed to update Pem, invalid url")
      return
    }
    let (data, response) = try await ControlConfigs.fetchData(from: url)
    if let string = String(data: data, encoding: .utf8) {
      self.publicPem = string
      self.logger.info("ControlConfigs update public Pem as:\(string)")
    } else {
      self.logger.error("ControlConfigs failed to update Pem, response:\(response))")
    }
  }

  static func fetchData(from url: URL) async throws -> (Data, URLResponse) {
    return try await withCheckedThrowingContinuation { continuation in
      let task = URLSession.shared.dataTask(with: url) { data, response, error in
        if let data = data, let response = response {
          continuation.resume(returning: (data, response))
        } else {
          if let error = error {
            continuation.resume(throwing: error)
          } else {
            continuation.resume(throwing: URLError(.badServerResponse))
          }
        }
      }
      task.resume()
    }
  }
}

class ControlPanelService: ControlPanelServiceProvider {
  var interceptors:
    (any GRPCControlPanelModels.ControlPanelServiceServerInterceptorFactoryProtocol)?
  private let taskCoordinator: TaskCoordinator
  private var controlConfigs: ControlConfigs
  private let logger: Logger
  enum ControlPanelError: Error {
    case gpuConnectFailed(message: String)
    case nioClientFailed(message: String)
    case removeGPUFailed(message: String)
  }

  init(taskCoordinator: TaskCoordinator, controlConfigs: ControlConfigs, logger: Logger) {
    self.taskCoordinator = taskCoordinator
    self.controlConfigs = controlConfigs
    self.logger = logger
  }

  func manageGPUServer(
    request: GRPCControlPanelModels.GPUServerRequest, context: any GRPC.StatusOnlyCallContext
  ) -> NIOCore.EventLoopFuture<GRPCControlPanelModels.GPUServerResponse> {
    let promise = context.eventLoop.makePromise(of: GRPCControlPanelModels.GPUServerResponse.self)
    Task {
      let gpuServerName = "\(request.serverConfig.address):\(request.serverConfig.port)"
      if request.operation == .add {
        self.logger.info(
          "Worker connect to server: \(gpuServerName). Worker focus on \(request.serverConfig.isHighPriority ? ProxyTaskPriority.high : ProxyTaskPriority.low) priority Task"
        )
        let client = ProxyGPUClientWrapper(deviceName: gpuServerName)
        do {
          try client.connect(
            host: request.serverConfig.address, port: Int(request.serverConfig.port))
          let result = await client.echo()
          self.logger.info("server: \(gpuServerName). echo \(result)")
          if let _ = client.client, result.0 {
            let worker = Worker(
              name: gpuServerName, client: client,
              primaryPriority: request.serverConfig.isHighPriority ? .high : .low)
            await taskCoordinator.addWorker(worker)
            let response = GPUServerResponse.with {
              $0.message = "added GPU \(gpuServerName) into workers stream"
            }
            promise.succeed(response)
          } else {
            try? client.disconnect()
            promise.fail(
              ControlPanelError.nioClientFailed(
                message: "fail to create nio client for \(gpuServerName)"))
          }
        } catch (let error) {
          promise.fail(
            ControlPanelError.gpuConnectFailed(
              message: "fail to connect GPU \(gpuServerName) error:\(error)"))
        }
      } else {
        await taskCoordinator.removeWorkerBasedOnName(gpuServerName)
        let response = GPUServerResponse.with {
          $0.message = "remove GPU \(gpuServerName) from taskCoordinator"
        }
        promise.succeed(response)
      }

    }

    return promise.futureResult
  }

  func updateModelList(
    request: GRPCControlPanelModels.UpdateModelListRequest, context: any GRPC.StatusOnlyCallContext
  ) -> NIOCore.EventLoopFuture<GRPCControlPanelModels.UpdateModelListResponse> {
    let promise = context.eventLoop.makePromise(
      of: GRPCControlPanelModels.UpdateModelListResponse.self)
    Task {
      let gpuServerName = "\(request.address):\(request.port)"
      self.logger.info(
        "update model list from server: \(gpuServerName)"
      )
      let client = ProxyGPUClientWrapper(deviceName: gpuServerName)
      do {
        try client.connect(
          host: request.address, port: Int(request.port))
        let result = await client.echo()
        self.logger.info("server: \(gpuServerName). echo \(result)")
        if let _ = client.client, result.0 {
          let dedupFiles = Array(Set(result.1))
          let fileList = dedupFiles.joined(separator: "\n")
          let internalFilePath = ModelZoo.internalFilePathForModelDownloaded("model-list")
          try? fileList.write(
            to: URL(fileURLWithPath: internalFilePath),
            atomically: true,
            encoding: .utf8)
          self.logger.info("update model list to file: \(internalFilePath)")

          let response = UpdateModelListResponse.with {
            $0.message = "update model list: \(dedupFiles.count) models from \(gpuServerName)"
            $0.files = dedupFiles
          }
          promise.succeed(response)
        } else {
          try? client.disconnect()
          promise.fail(
            ControlPanelError.nioClientFailed(
              message: "fail to create nio client for \(gpuServerName)"))
        }
      } catch (let error) {
        promise.fail(
          ControlPanelError.gpuConnectFailed(
            message: "fail to connect GPU \(gpuServerName) error:\(error)"))
      }

    }

    return promise.futureResult
  }

  func updateThrottlingConfig(
    request: GRPCControlPanelModels.ThrottlingRequest, context: any GRPC.StatusOnlyCallContext
  ) -> NIOCore.EventLoopFuture<GRPCControlPanelModels.ThrottlingResponse> {
    let promise = context.eventLoop.makePromise(of: GRPCControlPanelModels.ThrottlingResponse.self)
    Task {
      await controlConfigs.updateThrotellPolicy(
        newPolicies: request.limitConfig.mapValues { Int($0) })
      let response = ThrottlingResponse.with {
        $0.message = "Update throttling for \(request.limitConfig)"
      }
      promise.succeed(response)
    }

    return promise.futureResult
  }

  func updatePem(
    request: GRPCControlPanelModels.UpdatePemRequest, context: any GRPC.StatusOnlyCallContext
  ) -> NIOCore.EventLoopFuture<GRPCControlPanelModels.UpdatePemResponse> {
    let promise = context.eventLoop.makePromise(of: GRPCControlPanelModels.UpdatePemResponse.self)
    Task {
      try await controlConfigs.updatePublicPem()
      let pem = await controlConfigs.publicPem
      let response = UpdatePemResponse.with {
        $0.message = "Update proxy pem as:\n \(pem)"
      }
      promise.succeed(response)
    }

    return promise.futureResult
  }
}

class ImageGenerationProxyService: ImageGenerationServiceProvider {
  var interceptors:
    (any GRPCImageServiceModels.ImageGenerationServiceServerInterceptorFactoryProtocol)?

  private let taskCoordinator: TaskCoordinator
  private let logger: Logger
  private var controlConfigs: ControlConfigs

  init(taskCoordinator: TaskCoordinator, controlConfigs: ControlConfigs, logger: Logger) {
    self.taskCoordinator = taskCoordinator
    self.logger = logger
    self.controlConfigs = controlConfigs
  }

  func parseBearer(from string: String) -> String? {
    let components = string.trimmingCharacters(in: .whitespaces).split(separator: " ")
    guard components.count == 2,
      components[0].lowercased() == "bearer"
    else {
      return nil
    }
    return String(components[1])
  }

  func generateImage(
    request: ImageGenerationRequest,
    context: StreamingResponseCallContext<ImageGenerationResponse>
  ) -> EventLoopFuture<GRPCStatus> {
    let headers = context.headers
    guard let authorization = headers.first(name: "authorization"),
      let bearToken = parseBearer(from: authorization)
    else {
      return context.eventLoop.makeFailedFuture(
        GRPCStatus(code: .permissionDenied, message: "Service bear-token is empty")
      )
    }

    var localRequest = request
    localRequest.contents = []
    let encodedBlobString = try? localRequest.serializedData().base64EncodedString()
    let promise = context.eventLoop.makePromise(of: GRPCStatus.self)
    logger.info("Proxy Server enqueue image generating request")
    Task {
      let pem = await controlConfigs.publicPem
      let decoder = try? JWTDecoder(publicKeyPEM: pem)
      let payload = try? decoder?.decode(bearToken)
      guard let fileData = Data(base64Encoded: encodedBlobString ?? "") else {
        logger.info(
          "Proxy Server enqueue image generating request failed, encodedBlobString:\(encodedBlobString) decode to file data failed"
        )
        promise.fail(
          GRPCStatus(code: .permissionDenied, message: "encoded request Blob String failed"))
        return
      }

      let requestHash = Data(SHA256.hash(data: fileData))
      let checksum = requestHash.map({ String(format: "%02x", $0) }).joined()
      guard let payload = payload, payload.checksum == checksum else {
        logger.info(
          "Proxy Server enqueue image generating request failed, payload.blobSHA:\(payload?.checksum ?? "empty"), request blob:\(checksum) "
        )
        logger.info(
          "Proxy Server enqueue image generating request failed, payload:\(payload)"
        )
        promise.fail(
          GRPCStatus(code: .permissionDenied, message: "Service bear-token signature is failed"))
        return
      }
      logger.info("Proxy Server verified request checksum:\(checksum) success")
      let priority = payload.isHighPriority ? TaskPriority.high : TaskPriority.low
      let task = WorkTask(priority: priority, request: request, context: context, promise: promise)
      await taskCoordinator.addTask(task)
      if let worker = await taskCoordinator.nextWorker() {
        if let nextTaskForWorker = await taskCoordinator.nextTaskForWorker(worker) {
          await worker.executeTask(nextTaskForWorker)
          await taskCoordinator.addWorkerBackToStream(worker)
        }
      } else {
        logger.error("worker stream finished, can not get available worker")
      }
    }

    return promise.futureResult
  }

  func filesExist(request: FileListRequest, context: StatusOnlyCallContext) -> EventLoopFuture<
    FileExistenceResponse
  > {
    let response = FileExistenceResponse.with {

      $0.files = [String]()
      $0.existences = [Bool]()
      // by pass files Exist check for uploading
      for file in request.files {
        $0.files.append(file)
        $0.existences.append(true)
      }
    }

    return context.eventLoop.makeSucceededFuture(response)
  }

  func echo(request: EchoRequest, context: StatusOnlyCallContext) -> EventLoopFuture<EchoReply> {
    let response = EchoReply.with {
      logger.info("Proxy Server Received echo from: \(request.name)")
      $0.message = "Hello, \(request.name)!"
      let internalFilePath = ModelZoo.internalFilePathForModelDownloaded("model-list")
      var fileList = [String]()
      if let fileContent = try? String(
        contentsOf: URL(fileURLWithPath: internalFilePath), encoding: .utf8)
      {
        fileList = fileContent.components(separatedBy: .newlines).filter { !$0.isEmpty }
      } else {
        logger.error("Proxy Server file list is nil")
      }
      $0.files = fileList
    }
    return context.eventLoop.makeSucceededFuture(response)
  }

  func uploadFile(context: StreamingResponseCallContext<UploadResponse>) -> EventLoopFuture<
    (StreamEvent<FileUploadRequest>) -> Void
  > {
    context.statusPromise.fail(GRPCStatus(code: .unimplemented, message: "Service not supported"))
    return context.eventLoop.makeFailedFuture(
      GRPCStatus(code: .unimplemented, message: "Service not supported")
    )
  }
}

public class ProxyCPUServer {
  private var imageServer: Server?
  private var controlPanelServer: Server?
  private let workers: [Worker]
  private let logger = Logger(label: "com.draw-things.image-generation-proxy-service")
  private var controlConfigs: ControlConfigs
  private var taskCoordinator: TaskCoordinator

  public init(workers: [Worker], publicPem: String) {
    self.workers = workers
    self.controlConfigs = ControlConfigs(
      throtellPolicy: [
        "request_in_5min": 10, "request_in_10min": 15, "request_in_1hr": 60,
        "request_in_24hr": 1000,
      ], publicPem: publicPem, logger: logger)

    let proxyWorkers = Dictionary(uniqueKeysWithValues: workers.map { ($0.name, $0) })
    self.taskCoordinator = TaskCoordinator(workers: proxyWorkers, logger: logger)
  }

  public func startControlPanel(host: String, port: Int) throws {
    Task {
      print("Control Panel Service starting on \(host):\(port)")
      let group = MultiThreadedEventLoopGroup(numberOfThreads: 1)
      defer {
        try! group.syncShutdownGracefully()
      }
      let controlPanelService = ControlPanelService(
        taskCoordinator: taskCoordinator, controlConfigs: controlConfigs, logger: logger)

      controlPanelServer = try Server.insecure(group: group)
        .withServiceProviders([controlPanelService])
        .bind(host: host, port: port)
        .wait()
      print("Control Panel Service started on \(host):\(port)")

      try controlPanelServer?.onClose.wait()
      print("Leave Control Panel Service, something may be wrong")
    }
  }

  static func certificatesFromPEMFile(_ path: String) throws -> [NIOSSLCertificate] {
    let pemContent = try String(contentsOfFile: path)
    var certificates: [NIOSSLCertificate] = []

    // Split by certificate markers and clean up
    let pemComponents = pemContent.components(separatedBy: "-----BEGIN CERTIFICATE-----")
      .filter { !$0.isEmpty }
      .compactMap { component -> String? in
        guard let endMarkerRange = component.range(of: "-----END CERTIFICATE-----") else {
          return nil
        }
        let certContent = "-----BEGIN CERTIFICATE-----" + component[..<endMarkerRange.upperBound]
        // Ensure the certificate has both markers and proper content between them
        guard certContent.hasSuffix("-----END CERTIFICATE-----") else {
          return nil
        }
        return certContent.trimmingCharacters(in: .whitespacesAndNewlines)
      }

    for pemCert in pemComponents {
      let certData = Array(pemCert.utf8)
      if let cert = try? NIOSSLCertificate(bytes: certData, format: .pem) {
        certificates.append(cert)
      }
    }

    return certificates
  }

  public func start(host: String, port: Int, TLS: Bool, certPath: String, keyPath: String) throws {
    print("ImageGenerationProxyService starting on \(host):\(port)")
    let group = MultiThreadedEventLoopGroup(numberOfThreads: 1)
    let proxyService = ImageGenerationProxyService(
      taskCoordinator: taskCoordinator, controlConfigs: controlConfigs, logger: logger)
    let certificates = try? ProxyCPUServer.certificatesFromPEMFile(certPath)
    let privateKey = try? NIOSSLPrivateKey(file: keyPath, format: .pem)

    if TLS, let certificates = certificates, let privateKey = privateKey {
      let certificateSources = certificates.map { NIOSSLCertificateSource.certificate($0) }
      imageServer = try Server.usingTLS(
        with: GRPCTLSConfiguration.makeServerConfigurationBackedByNIOSSL(
          certificateChain: certificateSources, privateKey: .privateKey(privateKey)),
        on: group
      )
      .withServiceProviders([proxyService])
      .withMaximumReceiveMessageLength(1024 * 1024 * 1024)
      .bind(host: host, port: port).wait()
    } else {

      imageServer = try Server.insecure(group: group)
        .withServiceProviders([proxyService])
        .withMaximumReceiveMessageLength(1024 * 1024 * 1024)
        .bind(host: host, port: port).wait()
    }

    print("Image Generation Proxy Service started on port \(host):\(port)")
    try imageServer?.onClose.wait()
  }
}
