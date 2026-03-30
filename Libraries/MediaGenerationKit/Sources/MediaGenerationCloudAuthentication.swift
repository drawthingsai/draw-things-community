import Foundation

internal enum MediaGenerationRemoteAuthenticationMode {
  /// No authentication (open server)
  case none
  /// Shared secret password authentication
  case sharedSecret(String)
  /// DrawThings Cloud Compute authentication backed by a shared authenticator registry.
  case cloudCompute(apiKey: String, baseURL: URL, appCheck: AppCheckConfiguration)
}

internal enum MediaGenerationCloudAuthentication {
  private static let paygStatusCacheTTL: TimeInterval = 30
  private static let paygStatusCacheQueue = DispatchQueue(
    label: "com.drawthings.sdk.cloud-auth.payg-cache")
  private static var paygStatusCache: [String: (enabled: Bool, cachedAt: Date)] = [:]

  private struct AuthenticationRequest: Codable {
    let blob: String
    let fromBridge: Bool
    let attestationSupported: Bool
    let assertionPayload: String?
    let originalTransactionId: String?
    let isSandbox: Bool
    let consumableType: String?
    let amount: Double?
  }

  private struct AuthenticationResponse: Codable {
    let gRPCToken: String
  }

  private struct PaygStatusResponse: Codable {
    let paygEnabled: Bool
    let paygEligible: Bool
  }

  static func authenticate(
    shortTermToken: String,
    encodedBlob: String,
    fromBridge: Bool,
    estimatedComputeUnits: Double?,
    baseURL: URL = CloudConfiguration.defaultBaseURL,
    cancellation: (@escaping () -> Void) -> Void
  ) -> String? {

    let group = DispatchGroup()
    group.enter()

    let paygEnabled = fetchPaygEnabled(shortTermToken: shortTermToken, baseURL: baseURL)
    let isPositiveAmount = (estimatedComputeUnits ?? 0) > 0

    var request = URLRequest(url: baseURL.appendingPathComponent("/authenticate"))
    request.httpMethod = "POST"
    request.addValue("application/json", forHTTPHeaderField: "Content-Type")
    request.addValue(shortTermToken, forHTTPHeaderField: "Authorization")

    let requestBody = AuthenticationRequest(
      blob: encodedBlob,
      fromBridge: fromBridge,
      attestationSupported: false,
      assertionPayload: nil,
      originalTransactionId: nil,
      isSandbox: false,
      consumableType: paygEnabled && isPositiveAmount ? "payg" : nil,
      amount: isPositiveAmount ? estimatedComputeUnits : nil
    )

    guard let body = try? JSONEncoder().encode(requestBody) else {
      group.leave()
      return nil
    }
    request.httpBody = body

    var bearerToken: String?

    let task = URLSession.shared.dataTask(with: request) { data, response, error in
      defer { group.leave() }

      guard error == nil,
        let httpResponse = response as? HTTPURLResponse,
        httpResponse.statusCode == 200,
        let data = data,
        let authResponse = try? JSONDecoder().decode(AuthenticationResponse.self, from: data)
      else {
        return
      }

      bearerToken = authResponse.gRPCToken
    }

    cancellation {
      task.cancel()
    }

    task.resume()
    group.wait()

    return bearerToken
  }

  private static func fetchPaygEnabled(
    shortTermToken: String,
    baseURL: URL
  ) -> Bool {
    let now = Date()
    if let cached = paygStatusCacheQueue.sync(
      execute: {
        paygStatusCache[shortTermToken]
      }),
      now.timeIntervalSince(cached.cachedAt) <= paygStatusCacheTTL
    {
      return cached.enabled
    }

    let group = DispatchGroup()
    group.enter()
    var isEnabled = false

    var request = URLRequest(url: baseURL.appendingPathComponent("/billing/stripe/payg"))
    request.httpMethod = "GET"
    request.addValue(shortTermToken, forHTTPHeaderField: "Authorization")

    URLSession.shared.dataTask(with: request) { data, response, _ in
      defer { group.leave() }
      guard let httpResponse = response as? HTTPURLResponse,
        httpResponse.statusCode == 200,
        let data = data,
        let status = try? JSONDecoder().decode(PaygStatusResponse.self, from: data)
      else {
        return
      }
      isEnabled = status.paygEnabled && status.paygEligible
    }.resume()

    group.wait()
    paygStatusCacheQueue.sync {
      paygStatusCache[shortTermToken] = (enabled: isEnabled, cachedAt: now)
    }
    return isEnabled
  }
}
