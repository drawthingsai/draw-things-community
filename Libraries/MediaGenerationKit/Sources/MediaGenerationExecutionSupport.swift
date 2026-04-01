import Atomics
import Diffusion
import Foundation
import GRPCServer

internal struct MediaGenerationExecutionHintImage {
  let data: Data
  let weight: Float
}

internal struct MediaGenerationExecutionHint {
  let type: ControlHintType
  let images: [MediaGenerationExecutionHintImage]
}

internal struct MediaGenerationExecutionInputs {
  let image: Data?
  let mask: Data?
  let hints: [MediaGenerationExecutionHint]
}

internal final class MediaGenerationCancellationBridge {
  private struct State {
    var cancellation: (() -> Void)?
    var completed = false
  }

  private let cancelled = ManagedAtomic(false)
  private var state = ProtectedValue(State())

  var isCancelled: Bool {
    cancelled.load(ordering: .relaxed)
  }

  func setCancellation(_ cancellation: @escaping () -> Void) {
    var isCompleted = false
    state.modify { state in
      if state.completed {
        isCompleted = true
        return
      }
      state.cancellation = cancellation
    }
    if isCompleted {
      return
    }

    if cancelled.load(ordering: .acquiring) {
      cancellation()
    }
  }

  func cancel() {
    var shouldCancelImmediately = false
    var cancellation: (() -> Void)?
    state.modify { state in
      if state.completed {
        return
      }
      cancellation = state.cancellation
    }

    cancelled.store(true, ordering: .releasing)
    shouldCancelImmediately = cancellation != nil
    if shouldCancelImmediately {
      cancellation?()
    }
  }

  func finish() {
    state.modify { state in
      state.completed = true
      state.cancellation = nil
    }
  }
}

internal final class MediaGenerationAsyncResultBridge<Value>: @unchecked Sendable {
  private let lock = NSLock()
  private var continuation: CheckedContinuation<Value, Error>?
  private var finished = false

  @discardableResult
  func install(_ continuation: CheckedContinuation<Value, Error>) -> Bool {
    lock.lock()
    if finished {
      lock.unlock()
      continuation.resume(throwing: CancellationError())
      return false
    }
    self.continuation = continuation
    lock.unlock()
    return true
  }

  func resume(returning value: Value) {
    resume(with: .success(value))
  }

  func resume(throwing error: Error) {
    resume(with: .failure(error))
  }

  func resume(with result: Result<Value, Error>) {
    let continuation: CheckedContinuation<Value, Error>?
    lock.lock()
    if finished {
      lock.unlock()
      return
    }
    finished = true
    continuation = self.continuation
    self.continuation = nil
    lock.unlock()
    continuation?.resume(with: result)
  }

  func cancel() {
    resume(throwing: CancellationError())
  }
}
