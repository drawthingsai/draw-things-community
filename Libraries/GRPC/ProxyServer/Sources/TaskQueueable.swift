import Foundation
import GRPC
import GRPCImageServiceModels
import Logging
import NIO

/// Protocol defining the interface for managing task queues and workers
public protocol TaskQueueable: Actor {
  /// Returns all worker IDs currently in the queue
  var workerIds: [String] { get async }

  /// Gets the next available worker from the queue
  func nextWorker() async -> ProxyWorking?

  /// Gets the next task for a specific worker based on priority
  func nextTaskForWorker(_ worker: ProxyWorking) async -> ProxyWorkTask?

  /// Adds a task to the appropriate priority queue
  func addTask(_ task: ProxyWorkTask) async

  /// Returns a worker to the availability stream
  func returnWorker(_ worker: ProxyWorking) async

  /// Adds a worker to the queue
  func addWorker(_ worker: ProxyWorking) async

  /// Removes a worker by ID
  func removeWorkerById(_ id: String) async
}
