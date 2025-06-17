import Foundation
import GRPC
import GRPCImageServiceModels
import Logging

public actor TaskQueue: TaskQueueable {
  private var highPriorityTasks: [ProxyWorkTask] = []
  private var lowPriorityTasks: [ProxyWorkTask] = []
  private var pendingRemoveWorkerId = Set<String>()
  private var workers: [String: ProxyWorking]
  private let logger: Logger

  public var workerIds: [String] {
    return Array(workers.keys)
  }

  // Shared availability stream
  private let workerAvailabilityStream: AsyncStream<ProxyWorking>
  private let availabilityContinuation: AsyncStream<ProxyWorking>.Continuation

  public init(workers: [ProxyWorking], logger: Logger) {
    self.logger = logger
    self.workers = Dictionary(uniqueKeysWithValues: workers.map { ($0.id, $0) })

    (workerAvailabilityStream, availabilityContinuation) = AsyncStream.makeStream(
      of: ProxyWorking.self)
    for worker in workers {
      availabilityContinuation.yield(worker)
    }
  }

  public func nextWorker() async -> ProxyWorking? {
    for await worker in workerAvailabilityStream {
      if workers[worker.id] != nil {
        return worker
      } else {
        logger.info("skip removed worker:\(worker) from workerAvailabilityStream")
      }
    }
    return nil
  }

  public func nextTaskForWorker(_ worker: ProxyWorking) async -> ProxyWorkTask? {
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

  public func addTask(_ task: ProxyWorkTask) async {
    if task.priority == .high {
      logger.info("highPriorityTasks append task \(task.priority)")
      highPriorityTasks.append(task)
    } else {
      logger.info("lowPriorityTasks append task \(task.priority)")
      lowPriorityTasks.append(task)
    }
  }

  public func returnWorker(_ worker: ProxyWorking) async {
    guard workers[worker.id] != nil else {
      logger.error("worker:\(worker) is removed, can not be added to worker stream")
      return
    }
    logger.info("add worker:\(worker) back to worker stream")
    availabilityContinuation.yield(worker)
  }

  public func addWorker(_ worker: ProxyWorking) async {
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

  public func removeWorkerById(_ name: String) async {
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
