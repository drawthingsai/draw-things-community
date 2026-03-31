import Diffusion
import Foundation

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
  private let lock = NSLock()
  private var cancellation: (() -> Void)?
  private var cancelled = false
  private var completed = false

  var isCancelled: Bool {
    lock.lock()
    defer { lock.unlock() }
    return cancelled
  }

  func setCancellation(_ cancellation: @escaping () -> Void) {
    var shouldCancelImmediately = false
    lock.lock()
    if completed {
      lock.unlock()
      return
    }
    self.cancellation = cancellation
    shouldCancelImmediately = cancelled
    lock.unlock()
    if shouldCancelImmediately {
      cancellation()
    }
  }

  func cancel() {
    let cancellation: (() -> Void)?
    lock.lock()
    if completed {
      lock.unlock()
      return
    }
    cancelled = true
    cancellation = self.cancellation
    lock.unlock()
    cancellation?()
  }

  func finish() {
    lock.lock()
    completed = true
    cancellation = nil
    lock.unlock()
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
