import CryptoKit
import Foundation

#if canImport(UIKit)
  import UIKit
#endif

public final class ResumableDownloader: NSObject {
  public let remoteUrl: URL
  public let localUrl: URL
  public let sha256: String?

  public init(remoteUrl: URL, localUrl: URL, sha256: String?) {
    self.remoteUrl = remoteUrl
    self.localUrl = localUrl
    self.sha256 = sha256
  }

  deinit {
    #if canImport(UIKit)
      guard let backgroundTask = backgroundTask else { return }
      UIApplication.shared.endBackgroundTask(backgroundTask)
    #endif
  }

  // For now, recreate the session every time we do download to avoid issues related to network condition changes.
  private lazy var session = URLSession(configuration: .default)
  #if canImport(UIKit)
    private var backgroundTask: UIBackgroundTaskIdentifier? = nil
  #endif
  private var task: URLSessionDownloadTask? = nil
  private var handler:
    (
      (
        _ totalBytesWritten: Int64, _ totalBytesExpectedToWrite: Int64, _ isComplete: Bool,
        _ error: Error?
      ) -> Void
    )? =
      nil

  public func resume(
    _ handler: @escaping (
      _ totalBytesWritten: Int64, _ totalBytesExpectedToWrite: Int64, _ isComplete: Bool,
      _ error: Error?
    ) -> Void
  ) {
    let partUrl = localUrl.appendingPathExtension("part")
    var data: Data? = nil
    if FileManager.default.fileExists(atPath: partUrl.path) {
      data = try? Data(contentsOf: partUrl)
    }
    let task: URLSessionDownloadTask
    if let data = data {
      task = session.downloadTask(withResumeData: data)
    } else {
      task = session.downloadTask(with: URLRequest(url: remoteUrl))
    }
    task.delegate = self
    self.handler = handler
    self.task = task
    #if canImport(UIKit)
      backgroundTask = UIApplication.shared.beginBackgroundTask()
    #endif
    task.resume()
  }

  public func cancel() {
    guard let task = task else { return }
    let semaphore = DispatchSemaphore(value: 0)
    // File saving will be handled at error handler in didCompleteWithError
    task.cancel { _ in
      semaphore.signal()
    }
    semaphore.wait()
    self.task = nil
    #if canImport(UIKit)
      if let backgroundTask = backgroundTask {
        UIApplication.shared.endBackgroundTask(backgroundTask)
        self.backgroundTask = nil
      }
    #endif
  }
}

extension ResumableDownloader: URLSessionDownloadDelegate {
  public func urlSession(
    _ session: URLSession, downloadTask: URLSessionDownloadTask,
    didFinishDownloadingTo location: URL
  ) {
    defer {
      #if canImport(UIKit)
        if let backgroundTask = backgroundTask {
          UIApplication.shared.endBackgroundTask(backgroundTask)
          self.backgroundTask = nil
        }
      #endif
    }
    guard let handler = handler else { return }
    /// remove part url as current downloading task finish
    let partUrl = localUrl.appendingPathExtension("part")
    try? FileManager.default.removeItem(at: partUrl)

    let totalBytesWritten = Int64(
      (try? location.resourceValues(forKeys: [.fileSizeKey]))?
        .fileSize ?? 0)
    guard let data = try? Data(contentsOf: location, options: .mappedIfSafe) else {
      handler(totalBytesWritten, totalBytesWritten, false, DownloadError.fileMismatch)
      return
    }
    let digest = SHA256.hash(data: data)
    var hex = ""
    for byte in digest {
      hex += String(format: "%02x", byte)
    }
    guard sha256 == nil || sha256 == hex else {
      handler(totalBytesWritten, totalBytesWritten, false, DownloadError.fileMismatch)
      return
    }
    var localUrl = localUrl
    try? FileManager.default.moveItem(at: location, to: localUrl)
    var resourceValues = URLResourceValues()
    resourceValues.isExcludedFromBackup = true
    try? localUrl.setResourceValues(resourceValues)
    handler(totalBytesWritten, totalBytesWritten, true, nil)
    self.handler = nil
  }

  public func urlSession(
    _ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64,
    totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64
  ) {
    guard let handler = handler else { return }
    handler(totalBytesWritten, totalBytesExpectedToWrite, false, nil)
  }

  public enum DownloadError: Error, CustomStringConvertible {
    case local(Error)
    case server(Int)
    case fileMismatch
    case unexpected
    public var description: String {
      switch self {
      case .local(let error):
        return error.localizedDescription
      case .server(let statusCode):
        return "Server Error Code \(statusCode)"
      case .fileMismatch:
        return "File downloaded doesn't match the known good one"
      case .unexpected:
        return "Unexpected Error"
      }
    }
  }

  private func isTransient(_ e: NSError) -> Bool {
    if e.domain == NSURLErrorDomain {
      let code = URLError.Code(rawValue: e.code)
      switch code {
      case .timedOut, .cannotFindHost, .cannotConnectToHost, .dnsLookupFailed,
        .networkConnectionLost, .notConnectedToInternet:
        return true
      default: break
      }
    }
    if e.domain == NSPOSIXErrorDomain, e.code == 54 || e.code == 32 { return true }  // ECONNRESET/EPIPE
    if e.domain == kCFErrorDomainCFNetwork as String,
      let u = e.userInfo[NSUnderlyingErrorKey] as? NSError,
      u.domain == NSPOSIXErrorDomain, u.code == 54 || u.code == 32
    {
      return true
    }
    return false
  }

  public func urlSession(
    _ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?
  ) {
    defer {
      #if canImport(UIKit)
        if let backgroundTask = backgroundTask {
          UIApplication.shared.endBackgroundTask(backgroundTask)
          self.backgroundTask = nil
        }
      #endif
    }
    guard let handler = handler else { return }
    // Inspect if we have any sever error.
    var err = DownloadError.unexpected
    if let response = task.response, let httpResponse = response as? HTTPURLResponse,
      httpResponse.statusCode >= 400
    {
      err = .server(httpResponse.statusCode)
    } else if let error = error {
      err = .local(error)
    }
    let isTransientError: Bool
    if let nsError = error as? NSError, isTransient(nsError) {
      isTransientError = true
    } else {
      isTransientError = false
    }
    if !isTransientError {
      handler(0, 0, false, err)
      self.handler = nil
    }
    // At the end, cancel the download request.
    if let error = error as? NSError,
      let resumeData = error.userInfo[NSURLSessionDownloadTaskResumeData] as? Data
    {
      let partUrl = localUrl.appendingPathExtension("part")
      do {
        try resumeData.write(to: partUrl)
      } catch {
        // Try to remove, don't care if cannot.
        try? FileManager.default.removeItem(at: partUrl)
      }
    }
    if isTransientError {
      // Restart in 200ms.
      DispatchQueue.main.asyncAfter(deadline: .now() + .milliseconds(200)) {
        self.resume(handler)
      }
    }
  }
}
