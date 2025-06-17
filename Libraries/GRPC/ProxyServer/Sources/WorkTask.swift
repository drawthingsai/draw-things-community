import Foundation
import GRPC
import GRPCImageServiceModels
import NIO

public enum ProxyTaskPriority: Sendable {
  case high
  case low
}

public typealias TaskPriority = ProxyTaskPriority

public struct ProxyWorkTask: Sendable {
  public var priority: TaskPriority
  public var request: ImageGenerationRequest
  public var context: StreamingResponseCallContext<ImageGenerationResponse>
  public var promise: EventLoopPromise<GRPCStatus>
  public var heartbeat: Task<Void, Error>
  public var creationTimestamp: Date

  public init(
    priority: TaskPriority,
    request: ImageGenerationRequest,
    context: StreamingResponseCallContext<ImageGenerationResponse>,
    promise: EventLoopPromise<GRPCStatus>,
    heartbeat: Task<Void, Error>,
    creationTimestamp: Date
  ) {
    self.priority = priority
    self.request = request
    self.context = context
    self.promise = promise
    self.heartbeat = heartbeat
    self.creationTimestamp = creationTimestamp
  }
}
