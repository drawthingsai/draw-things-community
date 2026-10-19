import Foundation
import os.lock

#if canImport(FoundationNetworking)
  import FoundationNetworking
#endif

/// Recoverable transport failures shared by the browser-backed search providers.
enum WebSearchBrowserFallback {
  static func shouldUseBrowser(after error: Error) -> Bool {
    switch error {
    case WebSearchError.searchBlocked, WebSearchError.unexpectedSearchResponse,
      WebSearchError.bodyDecodingFailed:
      return true
    case WebSearchError.httpStatus(let status, _, _, _, _):
      return status == 403 || status == 408 || status == 429 || (500..<600).contains(status)
    case let error as URLError:
      return error.code == .timedOut || error.code == .networkConnectionLost
    default:
      return false
    }
  }

  /// Whether the provider refused the request outright, as opposed to failing to answer it.
  static func isBlock(_ error: Error) -> Bool {
    switch error {
    case WebSearchError.searchBlocked:
      return true
    case WebSearchError.httpStatus(let status, _, _, _, _):
      return status == 403 || status == 429
    default:
      return false
    }
  }
}

/// After a provider blocks an HTTP search, its searches go to the browser for a while. Retrying
/// HTTP at once lowers the IP's standing further, so the provider challenges sooner. Shared by
/// every search in the process, since providers block the IP, not a request.
final class ChallengeCooldown {
  static let duckDuckGo = ChallengeCooldown()
  static let sogou = ChallengeCooldown()

  private let duration: TimeInterval
  private var lock = os_unfair_lock()
  private var endsAt = Date.distantPast

  init(duration: TimeInterval = 15 * 60) {
    self.duration = duration
  }

  var isActive: Bool {
    os_unfair_lock_lock(&lock)
    defer { os_unfair_lock_unlock(&lock) }
    return Date() < endsAt
  }

  func start() {
    os_unfair_lock_lock(&lock)
    endsAt = Date().addingTimeInterval(duration)
    os_unfair_lock_unlock(&lock)
  }
}
