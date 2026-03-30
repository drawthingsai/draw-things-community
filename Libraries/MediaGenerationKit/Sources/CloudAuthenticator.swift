import Foundation

// MARK: - Cloud Authenticator

/// Shared short-term token cache keyed externally by `CloudAuthenticatorRegistry`.
///
/// The new MediaGenerationKit cloud path only needs `/sdk/token`; per-request
/// authorization happens in `MediaGenerationCloudAuthentication`.
internal final class CloudAuthenticator {
  internal enum State {
    case idle
    case fetchingToken
    case authenticated(expiresAt: Date?)
    case failed(Error)
  }

  private let apiKey: String
  private let endpoint: URL
  private let tokenRefreshThreshold: TimeInterval
  private let stateHandler: ((State) -> Void)?

  private var appCheck: AppCheckConfiguration
  private var appCheckRevision: UInt64 = 0
  private var cachedShortTermToken: String?
  private var tokenExpiry: Date?
  private let tokenQueue = DispatchQueue(label: "com.drawthings.mediagenerationkit.cloud-auth")

  private var sdkTokenEndpoint: URL {
    endpoint.appendingPathComponent("/sdk/token")
  }

  init(
    configuration: CloudConfiguration,
    stateHandler: ((State) -> Void)? = nil
  ) {
    self.apiKey = configuration.apiKey
    self.endpoint = configuration.baseURL
    self.tokenRefreshThreshold = configuration.tokenRefreshThreshold
    self.appCheck = configuration.appCheck
    self.stateHandler = stateHandler
  }

  func getShortTermToken(
    appCheck: AppCheckConfiguration,
    completion: @escaping (Result<String, Error>) -> Void
  ) {
    tokenQueue.async { [weak self] in
      guard let self = self else {
        completion(.failure(CloudAuthenticatorError.invalidState))
        return
      }
      if self.appCheck != appCheck {
        self.appCheck = appCheck
        self.cachedShortTermToken = nil
        self.tokenExpiry = nil
        self.appCheckRevision &+= 1
        self.stateHandler?(.idle)
      }
      self.getShortTermTokenLocked(completion: completion)
    }
  }

  private func getShortTermTokenLocked(
    completion: @escaping (Result<String, Error>) -> Void
  ) {
    if let token = cachedShortTermToken,
      let expiry = tokenExpiry,
      expiry.timeIntervalSinceNow > tokenRefreshThreshold
    {
      stateHandler?(.authenticated(expiresAt: expiry))
      completion(.success(token))
      return
    }

    let appCheck = self.appCheck
    let revision = appCheckRevision
    stateHandler?(.fetchingToken)
    fetchShortTermToken(appCheck: appCheck, revision: revision, completion: completion)
  }

  private func fetchShortTermToken(
    appCheck: AppCheckConfiguration,
    revision: UInt64,
    completion: @escaping (Result<String, Error>) -> Void
  ) {
    var request = URLRequest(url: sdkTokenEndpoint)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")

    struct TokenRequest: Codable {
      let apiKey: String
      let appCheckType: String
      let appCheckToken: String?
    }

    struct TokenResponse: Codable {
      let shortTermToken: String
      let expiresIn: Int
    }

    let requestBody = TokenRequest(
      apiKey: apiKey,
      appCheckType: appCheck.type,
      appCheckToken: appCheck.token
    )

    guard let body = try? JSONEncoder().encode(requestBody) else {
      let error = CloudAuthenticatorError.encodingFailed
      stateHandler?(.failed(error))
      completion(.failure(error))
      return
    }
    request.httpBody = body

    URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
      guard let self = self else {
        completion(.failure(CloudAuthenticatorError.invalidState))
        return
      }

      if let error = error {
        self.stateHandler?(.failed(error))
        completion(.failure(error))
        return
      }

      guard let httpResponse = response as? HTTPURLResponse else {
        let error = CloudAuthenticatorError.invalidResponse
        self.stateHandler?(.failed(error))
        completion(.failure(error))
        return
      }

      guard httpResponse.statusCode == 200, let data = data else {
        let error = CloudAuthenticatorError.serverError(httpResponse.statusCode)
        self.stateHandler?(.failed(error))
        completion(.failure(error))
        return
      }

      do {
        let tokenResponse = try JSONDecoder().decode(TokenResponse.self, from: data)
        let expiresAt = Date().addingTimeInterval(TimeInterval(tokenResponse.expiresIn))
        self.tokenQueue.async {
          if self.appCheckRevision == revision && self.appCheck == appCheck {
            self.cachedShortTermToken = tokenResponse.shortTermToken
            self.tokenExpiry = expiresAt
            self.stateHandler?(.authenticated(expiresAt: expiresAt))
          }
        }
        completion(.success(tokenResponse.shortTermToken))
      } catch {
        let error = CloudAuthenticatorError.decodingFailed
        self.stateHandler?(.failed(error))
        completion(.failure(error))
      }
    }.resume()
  }
}

// MARK: - Errors

enum CloudAuthenticatorError: Error, LocalizedError {
  case invalidState
  case encodingFailed
  case decodingFailed
  case invalidResponse
  case serverError(Int)

  var errorDescription: String? {
    switch self {
    case .invalidState:
      return "Authenticator is in an invalid state"
    case .encodingFailed:
      return "Failed to encode request"
    case .decodingFailed:
      return "Failed to decode response"
    case .invalidResponse:
      return "Invalid server response"
    case .serverError(let code):
      return "Server error: \(code)"
    }
  }
}
