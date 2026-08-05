import Foundation

#if canImport(FoundationNetworking)
  import FoundationNetworking
#endif

/// Minimal HTTP transport used by search and fetch tools.
public protocol HttpTransport {
  /// Performs a request and calls `completion` with response body data plus the HTTP response.
  @discardableResult
  func data(
    for request: URLRequest,
    completion: @escaping (Result<(Data, HTTPURLResponse), Error>) -> Void
  ) -> (() -> Void)?
}

final class HttpRequestCancellationBox: @unchecked Sendable {
  private let lock = NSLock()
  private var cancellation: (() -> Void)?
  private var isCancelled = false

  func setCancellation(_ cancellation: (() -> Void)?) {
    let cancellationToRun: (() -> Void)?
    lock.lock()
    if isCancelled {
      cancellationToRun = cancellation
    } else {
      self.cancellation = cancellation
      cancellationToRun = nil
    }
    lock.unlock()
    cancellationToRun?()
  }

  func cancel() {
    let cancellationToRun: (() -> Void)?
    lock.lock()
    isCancelled = true
    cancellationToRun = cancellation
    cancellation = nil
    lock.unlock()
    cancellationToRun?()
  }
}

extension HttpTransport {
  /// Performs a request with async/await by wrapping the completion-handler API.
  public func data(for request: URLRequest) async throws -> (Data, HTTPURLResponse) {
    let cancellationBox = HttpRequestCancellationBox()
    return try await withTaskCancellationHandler {
      try Task.checkCancellation()
      return try await withCheckedThrowingContinuation { continuation in
        cancellationBox.setCancellation(
          data(for: request) { result in
            continuation.resume(with: result)
          })
      }
    } onCancel: {
      cancellationBox.cancel()
    }
  }
}

/// `URLSession`-backed HTTP transport.
public struct URLSessionHttpTransport: HttpTransport {
  private let session: URLSession

  /// Creates a transport backed by `session`.
  public init(session: URLSession = .shared) {
    self.session = session
  }

  /// Performs a request with `URLSession`.
  @discardableResult
  public func data(
    for request: URLRequest,
    completion: @escaping (Result<(Data, HTTPURLResponse), Error>) -> Void
  ) -> (() -> Void)? {
    let task = session.dataTask(with: request) { data, response, error in
      if let error {
        completion(.failure(error))
        return
      }
      guard let data, let httpResponse = response as? HTTPURLResponse else {
        completion(.failure(WebSearchError.invalidResponse))
        return
      }
      completion(.success((data, httpResponse)))
    }
    task.resume()
    return { task.cancel() }
  }
}
