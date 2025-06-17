import Foundation
@preconcurrency import GRPC
import GRPCImageServiceModels
import Logging

public struct ProxyWorker: ProxyWorking {
  public var id: String
  public var primaryPriority: ProxyTaskPriority
  public let client: ProxyGPUClientWrapper
  private let logger = Logger(label: "com.draw-things.image-generation-proxy-service")

  enum ProxyWorkerError: Error {
    case invalidNioClient
  }

  public init(
    id: String, client: ProxyGPUClientWrapper, primaryPriority: ProxyTaskPriority
  ) {
    self.id = id
    self.client = client
    self.primaryPriority = primaryPriority
  }

  public func executeTask(_ task: ProxyWorkTask) async throws {
    logger.info(
      "Worker \(id) primaryPriority:\(primaryPriority) starting task  (Priority: \(task.priority))"
    )
    let taskQueueingTimeMs = Date().timeIntervalSince(task.creationTimestamp) * 1000
    logger.info(
      "Task queueing time: \(taskQueueingTimeMs)ms, (Priority: \(task.priority))"
    )
    defer { task.heartbeat.cancel() }
    do {
      var call: ServerStreamingCall<ImageGenerationRequest, ImageGenerationResponse>? = nil
      guard let client = client.client else {
        logger.error("Worker \(id) task failed: invalid NIO client")
        throw ProxyWorkerError.invalidNioClient
      }
      let logger = logger
      let callInstance = client.generateImage(task.request) { response in
        if !response.generatedImages.isEmpty {
          let totalTimeMs = Date().timeIntervalSince(task.creationTimestamp) * 1000
          logger.info(
            "Task total time: \(totalTimeMs)ms, (Priority: \(task.priority))"
          )
        }
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
      task.context.statusPromise.succeed(status)

      logger.info("Worker \(id) completed task successfully (Priority: \(task.priority))")

    } catch {
      logger.error("Worker \(id) task failed with error: \(error) (Priority: \(task.priority))")
      task.promise.fail(error)
      task.context.statusPromise.fail(error)
      throw error
    }
  }
}
