import BinaryResources
import Crypto
import DataModels
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

struct WorkTask {
  var priority: TaskPriority
  var request: ImageGenerationRequest
  var context: StreamingResponseCallContext<ImageGenerationResponse>
  var promise: EventLoopPromise<GRPCStatus>
  var heartbeat: Task<Void, Error>
}

public struct Worker {
  var id: String
  var primaryPriority: ProxyTaskPriority
  public let client: ProxyGPUClientWrapper
  private let logger = Logger(label: "com.draw-things.image-generation-proxy-service")
  enum WorkerError: Error {
    case invalidNioClient
  }
  public init(
    id: String, client: ProxyGPUClientWrapper, primaryPriority: ProxyTaskPriority
  ) {
    self.id = id
    self.client = client
    self.primaryPriority = primaryPriority
  }
}

extension Worker {
  func executeTask(_ task: WorkTask) async {
    logger.info(
      "Worker \(id) primaryPriority:\(primaryPriority) starting task  (Priority: \(task.priority))"
    )
    defer { task.heartbeat.cancel() }
    do {
      var call: ServerStreamingCall<ImageGenerationRequest, ImageGenerationResponse>? = nil
      guard let client = client.client else {
        throw WorkerError.invalidNioClient
      }
      let logger = logger
      let callInstance = client.generateImage(task.request) { response in
        task.context.sendResponse(response).whenComplete { result in
          switch result {
          case .success:
            logger.debug("forward response: \(response)")
          case .failure(let error):
            logger.error("Worker:\(self), forward response error \(error)")
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

actor TaskQueue {
  private var highPriorityTasks: [WorkTask] = []
  private var lowPriorityTasks: [WorkTask] = []
  private var pendingRemoveWorkerId = Set<String>()
  private var workers: [String: Worker]
  private let logger: Logger
  var workerIds: [String] {
    return Array(workers.keys)
  }

  // Shared availability stream
  private let workerAvailabilityStream: AsyncStream<Worker>
  private let availabilityContinuation: AsyncStream<Worker>.Continuation

  init(workers: [Worker], logger: Logger) {
    self.logger = logger
    self.workers = Dictionary(uniqueKeysWithValues: workers.map { ($0.id, $0) })

    (workerAvailabilityStream, availabilityContinuation) = AsyncStream.makeStream(of: Worker.self)
    for worker in workers {
      availabilityContinuation.yield(worker)
    }
  }

  func nextWorker() async -> Worker? {
    for await worker in workerAvailabilityStream {
      if workers[worker.id] != nil {
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

  func returnWorker(_ worker: Worker) async {
    guard workers[worker.id] != nil else {
      logger.error("worker:\(worker) is removed, can not be added to worker stream")
      return
    }
    logger.info("add worker:\(worker) back to worker stream")
    availabilityContinuation.yield(worker)
  }

  func addWorker(_ worker: Worker) async {
    guard worker.client.client != nil else {
      logger.error(
        "can add worker:\(worker) to worker TaskQueue with invalid nioclient connection")
      return
    }
    let alreadyExists = workers[worker.id] != nil
    workers[worker.id] = worker
    guard !alreadyExists else {
      logger.info("worker:\(worker) already exists in workers, skip adding")
      return
    }
    availabilityContinuation.yield(worker)
    logger.info("add worker:\(worker) to worker TaskQueue and stream")
  }

  func removeWorkerById(_ name: String) async {
    guard let worker = workers[name] else {
      logger.error("failed to find worker based on name \(name)")
      return
    }
    try? worker.client.disconnect()
    workers[worker.id] = nil
    logger.info("remove worker:\(worker) from worker TaskQueue")
  }

  deinit {
    for worker in workers.values {
      try? worker.client.disconnect()
    }
    availabilityContinuation.finish()
  }
}

public actor ControlConfigs {
  public private(set) var throttlePolicy = [String: Int]()
  public private(set) var publicKeyPEM: String
  public private(set) var modelListPath: String
  private var nonces = Set<String>()
  private let logger: Logger
  private var nonceSizeLimit: Int
  private var sharedSecret: String?
  enum ControlConfigsError: Error {
    case updatePublicKeyFailed(message: String)
  }

  init(
    throttlePolicy: [String: Int], publicKeyPEM: String, logger: Logger, modelListPath: String,
    nonceSizeLimit: Int
  ) {
    self.throttlePolicy = throttlePolicy
    self.publicKeyPEM = publicKeyPEM
    self.logger = logger
    self.modelListPath = modelListPath
    self.nonceSizeLimit = nonceSizeLimit
  }

  func addProcessedNonce(_ nonce: String) async {
    guard !nonces.contains(nonce) else { return }
    if nonces.count >= nonceSizeLimit {
      if let randomElement = nonces.randomElement() {
        nonces.remove(randomElement)
      }
    }

    nonces.insert(nonce)
    logger.info("ControlConfigs add processed nonce:\(nonce)")
  }

  func isUsedNonce(_ nonce: String) async -> Bool {
    nonces.contains(nonce)
  }

  func isSharedSecretValid(_ sharedSecret: String) async -> Bool {
    guard let configuredSecret = self.sharedSecret else {
      return false
    }
    return configuredSecret == sharedSecret
  }

  func updateThrottlePolicy(newPolicies: [String: Int]) async {
    for (key, value) in newPolicies {
      throttlePolicy[key] = value
    }
  }
  func updatePublicKeyPEM() async throws {
    guard let url = URL(string: "https://api.drawthings.ai/key") else {
      logger.error("ControlConfigs failed to update Pem, invalid url")
      return
    }
    let (data, response) = try await URLSession.shared.data(from: url)
    if let string = String(data: data, encoding: .utf8) {
      publicKeyPEM = string
      logger.info("ControlConfigs update public Pem as:\(string)")
    } else {
      logger.error("ControlConfigs failed to update Pem, response:\(response))")
    }
  }

  func updateSharedSecret() async -> String {
    func generatePassword() -> String {
      // Avoiding confusing characters like 1, l, I, i, 0, O
      let allowedChars = "23456789ABCDEFGHJKMNPQRSTUVWXYZ"
      var password = ""
      // Ensure at least one number and one letter
      let numbers = "23456789"
      let letters = "ABCDEFGHJKMNPQRSTUVWXYZ"
      // Add one random number
      password.append(numbers.randomElement()!)
      // Add one random letter
      password.append(letters.randomElement()!)
      // Fill the remaining 10 characters
      for _ in 0..<10 {
        password.append(allowedChars.randomElement()!)
      }
      // Shuffle the password to randomize position of guaranteed number and letter
      return String(password.shuffled())
    }
    let sharedSecret = generatePassword()
    self.sharedSecret = sharedSecret
    logger.info("ControlConfigs update shared Secret")
    return sharedSecret
  }
}

final class ControlPanelService: ControlPanelServiceProvider {
  var interceptors:
    (any GRPCControlPanelModels.ControlPanelServiceServerInterceptorFactoryProtocol)?
  private let taskQueue: TaskQueue
  private var controlConfigs: ControlConfigs
  private let logger: Logger
  enum ControlPanelError: Error {
    case gpuConnectFailed(message: String)
    case nioClientFailed(message: String)
    case removeGPUFailed(message: String)
  }

  init(taskQueue: TaskQueue, controlConfigs: ControlConfigs, logger: Logger) {
    self.taskQueue = taskQueue
    self.controlConfigs = controlConfigs
    self.logger = logger
  }

  func manageGPUServer(
    request: GRPCControlPanelModels.GPUServerRequest, context: any GRPC.StatusOnlyCallContext
  ) -> NIOCore.EventLoopFuture<GRPCControlPanelModels.GPUServerResponse> {
    let promise = context.eventLoop.makePromise(of: GRPCControlPanelModels.GPUServerResponse.self)
    Task {
      let gpuServerName = "\(request.serverConfig.address):\(request.serverConfig.port)"
      switch request.operation {
      case .add:
        self.logger.info(
          "Worker connecting server: \(gpuServerName). Worker focus on \(request.serverConfig.isHighPriority ? ProxyTaskPriority.high : ProxyTaskPriority.low) priority Task"
        )
        let client = ProxyGPUClientWrapper(deviceName: gpuServerName)
        do {
          try client.connect(
            host: request.serverConfig.address, port: Int(request.serverConfig.port))
          let result = await client.echo()
          self.logger.info("server: \(gpuServerName). echo \(result)")
          if let _ = client.client, result.0 {
            let worker = Worker(
              id: gpuServerName, client: client,
              primaryPriority: request.serverConfig.isHighPriority ? .high : .low)
            await taskQueue.addWorker(worker)
            let workersId = await taskQueue.workerIds
            let response = GPUServerResponse.with {
              $0.message =
                "added GPU \(gpuServerName) into workers stream, current workers:\(workersId)"
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
      case .remove:
        await taskQueue.removeWorkerById(gpuServerName)
        let workersId = await taskQueue.workerIds
        let response = GPUServerResponse.with {
          $0.message =
            "remove GPU \(gpuServerName) from taskCoordinator, current workers:\(workersId)"
        }
        promise.succeed(response)
      case .unspecified, .UNRECOGNIZED(_):
        break
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
      let fileList = request.files.joined(separator: "\n")
      // TODO: The path is problematic.
      let internalFilePath = await controlConfigs.modelListPath
      try? fileList.write(
        to: URL(fileURLWithPath: internalFilePath),
        atomically: true,
        encoding: .utf8)
      self.logger.info(
        "update model list to file: \(internalFilePath) with \(request.files.count) models")

      let response = UpdateModelListResponse.with {
        $0.message = "update model-list file with \(request.files.count) models"
      }
      promise.succeed(response)

    }

    return promise.futureResult
  }

  func updateThrottlingConfig(
    request: GRPCControlPanelModels.ThrottlingRequest, context: any GRPC.StatusOnlyCallContext
  ) -> NIOCore.EventLoopFuture<GRPCControlPanelModels.ThrottlingResponse> {
    let promise = context.eventLoop.makePromise(of: GRPCControlPanelModels.ThrottlingResponse.self)
    Task {
      await controlConfigs.updateThrottlePolicy(
        newPolicies: request.limitConfig.mapValues { Int($0) })
      let currentThrottlePolicies = await controlConfigs.throttlePolicy
      self.logger.info(
        "Update throttling for \(request.limitConfig), current throttling policies are \(currentThrottlePolicies)"
      )
      let response = ThrottlingResponse.with {
        $0.message =
          "Update throttling for \(request.limitConfig), current throttling policies are \(currentThrottlePolicies)"
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
      try await controlConfigs.updatePublicKeyPEM()
      let pem = await controlConfigs.publicKeyPEM
      let response = UpdatePemResponse.with {
        $0.message = "Update proxy pem as:\n \(pem)"
      }
      promise.succeed(response)
    }
    return promise.futureResult
  }

  func updateSharedSecret(request: UpdateSharedSecretRequest, context: StatusOnlyCallContext)
    -> EventLoopFuture<UpdateSharedSecretResponse>
  {
    let promise = context.eventLoop.makePromise(
      of: GRPCControlPanelModels.UpdateSharedSecretResponse.self)
    Task {
      let sharedSecret = await controlConfigs.updateSharedSecret()
      let response = UpdateSharedSecretResponse.with {
        $0.message = "Update proxy shared secret as:\n \(sharedSecret)"
      }
      promise.succeed(response)
    }

    return promise.futureResult
  }
}

final class ImageGenerationProxyService: ImageGenerationServiceProvider {
  var interceptors:
    (any GRPCImageServiceModels.ImageGenerationServiceServerInterceptorFactoryProtocol)?

  private let taskQueue: TaskQueue
  private let logger: Logger
  private var controlConfigs: ControlConfigs
  private var healthCheckTask: Task<Void, Never>?

  init(taskQueue: TaskQueue, controlConfigs: ControlConfigs, logger: Logger, healthCheck: Bool) {
    self.taskQueue = taskQueue
    self.logger = logger
    self.controlConfigs = controlConfigs
    if healthCheck {
      self.startHealthCheck()
    }
  }

  private func startHealthCheck() {
    healthCheckTask = Task { [weak self] in
      while !Task.isCancelled {
        guard let self = self else { break }
        if let worker = await taskQueue.nextWorker() {
          let (success, _) = await worker.client.echo()
          if success {
            logger.info("Health check passed for worker: \(worker.id)")
            await taskQueue.returnWorker(worker)
          } else {
            logger.error("Health check failed for worker: \(worker.id)")
            await taskQueue.removeWorkerById(worker.id)
          }
          try? await Task.sleep(for: .seconds(10))  // wait for 10s
        }
      }
    }
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

    var validationRequest = request
    validationRequest.contents = []
    let encodedBlob = try? validationRequest.serializedData()
    guard let encodedBlob = encodedBlob else {
      return context.eventLoop.makeFailedFuture(
        GRPCStatus(code: .permissionDenied, message: "Cannot encode validation request")
      )
    }
    let promise = context.eventLoop.makePromise(of: GRPCStatus.self)
    logger.info("Proxy Server enqueue image generating request")
    let logger = logger
    Task {
      let pem: String = await controlConfigs.publicKeyPEM
      let decoder = try? JWTDecoder(publicKeyPEM: pem)
      let payload = try? decoder?.decode(bearToken)
      logger.info(
        "Proxy Server enqueue image generating payload:\(payload as Any)"
      )
      let requestHash = Data(SHA256.hash(data: encodedBlob))
      let checksum = requestHash.map({ String(format: "%02x", $0) }).joined()
      guard let payload = payload, payload.checksum == checksum else {
        logger.info(
          "Proxy Server enqueue image generating request failed, payload.blobSHA:\(payload?.checksum ?? "empty"), request blob:\(checksum) "
        )
        logger.info(
          "Proxy Server enqueue image generating request failed, payload:\(payload as Any)"
        )
        promise.fail(
          GRPCStatus(code: .permissionDenied, message: "Service bear-token signature is failed"))
        return
      }
      logger.info("Proxy Server verified request checksum:\(checksum) success")

      guard await !controlConfigs.isUsedNonce(payload.nonce) else {
        logger.error(
          "Proxy Server image generating request failed, \(payload.nonce) is a used nonce"
        )
        promise.fail(
          GRPCStatus(code: .permissionDenied, message: "used nonce"))
        return
      }
      await self.controlConfigs.addProcessedNonce(payload.nonce)
      let throttlePolicies = await controlConfigs.throttlePolicy
      for (key, stat) in payload.stats {
        if let throttlePolicy = throttlePolicies[key], throttlePolicy < stat {
          logger.error(
            "user made \(stat) requests, while policy only allow \(throttlePolicy) for \(key)"
          )
          promise.fail(
            GRPCStatus(
              code: .permissionDenied,
              message: "user failed to pass throttlePolicy, \(key) in \(throttlePolicy)"))
          return
        }
      }
      let priority = taskPriority(from: payload.priority)
      let configuration = GenerationConfiguration.from(data: request.configuration)
      guard let modelName = configuration.model else {
        promise.fail(
          GRPCStatus(
            code: .permissionDenied,
            message: "no valid model name "))
        return
      }
      guard let cost = ComputeUnits.from(configuration) else {
        logger.error(
          "Proxy Server can not calculate cost for configuration \(configuration)"
        )
        promise.fail(
          GRPCStatus(
            code: .permissionDenied,
            message: "Proxy Server can not calculate cost for model \(modelName)"))
        return
      }
      let isSharedSecretValid = await controlConfigs.isSharedSecretValid(request.sharedSecret)
      let costThreshold = ComputeUnits.threadhold(for: payload.priority)
      guard cost < costThreshold || isSharedSecretValid else {
        logger.error(
          "Proxy Server enqueue image generating request failed, cost exceed threshold \(costThreshold)"
        )
        promise.fail(
          GRPCStatus(
            code: .permissionDenied, message: "cost \(cost) exceed threshold \(costThreshold)"))
        return
      }
      // Enqueue task.
      let heartbeat = Task {
        while !Task.isCancelled {
          // Abort the task if we cannot send response any more. Send empty response as heartbeat to keep Cloudflare alive.
          let _ = try await context.sendResponse(ImageGenerationResponse()).get()
          try? await Task.sleep(for: .seconds(20))  // Every 20 seconds send a heartbeat.
        }
      }
      let task = WorkTask(
        priority: priority, request: request, context: context, promise: promise,
        heartbeat: heartbeat)
      await taskQueue.addTask(task)
      if let worker = await taskQueue.nextWorker() {
        // Note that the extracted task may not be the ones we just enqueued.
        if let nextTaskForWorker = await taskQueue.nextTaskForWorker(worker) {
          await worker.executeTask(nextTaskForWorker)
          await taskQueue.returnWorker(worker)
        }
      } else {
        logger.error("worker stream finished, can not get available worker")
        heartbeat.cancel()
      }
    }

    return promise.futureResult
  }

  func taskPriority(from priority: String) -> TaskPriority {
    switch priority {
    case "community":
      return TaskPriority.low
    case "plus":
      return TaskPriority.high
    default:
      return TaskPriority.low
    }
  }

  func filesExist(request: FileListRequest, context: StatusOnlyCallContext) -> EventLoopFuture<
    FileExistenceResponse
  > {
    let promise = context.eventLoop.makePromise(of: FileExistenceResponse.self)
    Task {
      let internalFilePath = await controlConfigs.modelListPath
      let response = FileExistenceResponse.with {
        $0.files = [String]()
        $0.existences = [Bool]()
        var fileList = [String]()
        if let fileContent = try? String(
          contentsOf: URL(fileURLWithPath: internalFilePath), encoding: .utf8)
        {
          fileList = fileContent.components(separatedBy: .newlines).filter { !$0.isEmpty }
        } else {
          logger.error("Proxy Server file list is nil")
        }
        for file in request.files {
          $0.files.append(file)
          let existence = fileList.contains(file)
          $0.existences.append(existence)
        }
      }
      promise.succeed(response)
    }
    return promise.futureResult
  }

  func echo(request: EchoRequest, context: StatusOnlyCallContext) -> EventLoopFuture<EchoReply> {
    let promise = context.eventLoop.makePromise(of: EchoReply.self)
    Task {
      let internalFilePath = await controlConfigs.modelListPath
      let response = EchoReply.with {
        logger.info("Proxy Server Received echo from: \(request.name)")
        $0.message = "Hello, \(request.name)!"
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
      promise.succeed(response)
    }
    return promise.futureResult
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
  private let workers: [Worker]
  private let logger = Logger(label: "com.draw-things.image-generation-proxy-service")
  private var controlConfigs: ControlConfigs
  private var taskQueue: TaskQueue

  public init(workers: [Worker], publicKeyPEM: String, modelListPath: String, nonceSizeLimit: Int) {
    self.workers = workers
    self.controlConfigs = ControlConfigs(
      throttlePolicy: [
        "5_min": 10, "10_min": 15, "1_hour": 60,
        "24_hour": 1000,
      ], publicKeyPEM: publicKeyPEM, logger: logger, modelListPath: modelListPath,
      nonceSizeLimit: nonceSizeLimit)

    self.taskQueue = TaskQueue(workers: workers, logger: logger)
  }

  public func startControlPanel(host: String, port: Int) throws {
    Task {
      logger.info("Control Panel Service starting on \(host):\(port)")
      let group = MultiThreadedEventLoopGroup(numberOfThreads: 1)
      defer {
        try! group.syncShutdownGracefully()
      }
      let controlPanelService = ControlPanelService(
        taskQueue: taskQueue, controlConfigs: controlConfigs, logger: logger)

      let controlPanelServer = try Server.insecure(group: group)
        .withServiceProviders([controlPanelService])
        .bind(host: host, port: port)
        .wait()
      logger.info("Control Panel Service started on \(host):\(port)")

      try controlPanelServer.onClose.wait()
      logger.info("Leave Control Panel Service, something may be wrong")
    }
  }

  private static func certificatesFromPEMFile(_ path: String) throws -> [NIOSSLCertificate] {
    let pemContent = try String(contentsOfFile: path, encoding: .utf8)
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
      guard let cert = try? NIOSSLCertificate(bytes: certData, format: .pem) else {
        continue
      }
      certificates.append(cert)
    }

    return certificates
  }

  public func startAndWait(
    host: String, port: Int, TLS: Bool, certPath: String, keyPath: String, numberOfThreads: Int,
    healthCheck: Bool
  )
    throws
  {
    logger.info("ImageGenerationProxyService starting on \(host):\(port)")
    let group = MultiThreadedEventLoopGroup(numberOfThreads: numberOfThreads)
    let proxyService = ImageGenerationProxyService(
      taskQueue: taskQueue, controlConfigs: controlConfigs, logger: logger, healthCheck: healthCheck
    )
    let imageServer: Server
    if TLS {
      let certificates: [NIOSSLCertificate]
      let privateKey: NIOSSLPrivateKey
      if !certPath.isEmpty && !keyPath.isEmpty {
        certificates = try ProxyCPUServer.certificatesFromPEMFile(certPath)
        privateKey = try NIOSSLPrivateKey(file: keyPath, format: .pem)
      } else {
        certificates = [
          try NIOSSLCertificate(bytes: [UInt8](BinaryResources.server_crt_crt), format: .pem)
        ]
        privateKey = try NIOSSLPrivateKey(
          bytes: [UInt8](BinaryResources.server_key_key), format: .pem)
      }
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

    logger.info("Image Generation Proxy Service started on port \(host):\(port)")
    try imageServer.onClose.wait()
  }
}
