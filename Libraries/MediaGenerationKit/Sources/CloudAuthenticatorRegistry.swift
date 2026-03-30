import Foundation

internal final class CloudAuthenticatorRegistry {
  static let shared = CloudAuthenticatorRegistry()

  private struct Key: Hashable {
    let apiKey: String
    let baseURL: URL
  }

  private let lock = NSLock()
  private var authenticators: [Key: CloudAuthenticator] = [:]

  private init() {}

  func authenticator(
    apiKey: String,
    baseURL: URL = CloudConfiguration.defaultBaseURL
  ) -> CloudAuthenticator {
    let key = Key(apiKey: apiKey, baseURL: baseURL)
    lock.lock()
    defer { lock.unlock() }
    if let authenticator = authenticators[key] {
      return authenticator
    }
    let authenticator = CloudAuthenticator(
      configuration: CloudConfiguration(
        apiKey: apiKey,
        appCheck: .none,
        baseURL: baseURL,
        tokenRefreshThreshold: 300
      )
    )
    authenticators[key] = authenticator
    return authenticator
  }
}
