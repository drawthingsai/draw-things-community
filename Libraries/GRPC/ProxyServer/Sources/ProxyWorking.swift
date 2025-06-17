import Foundation
import GRPC
import GRPCImageServiceModels
import Logging

/// Protocol defining the interface for workers that execute tasks
public protocol ProxyWorking: Sendable {
  /// Unique identifier for the worker
  var id: String { get }

  /// Primary priority that this worker focuses on
  var primaryPriority: ProxyTaskPriority { get }

  /// Client wrapper for communicating with GPU servers
  var client: ProxyGPUClientWrapper { get }

  /// Executes a work task
  func executeTask(_ task: ProxyWorkTask) async throws
}
