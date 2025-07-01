import Crypto
import Foundation

public actor ProxyMessageSigner {
  // In-memory key pair
  private var keyPair: (private: P256.Signing.PrivateKey, public: P256.Signing.PublicKey)

  init() {
    // Generate initial key pair in memory
    let privateKey = P256.Signing.PrivateKey()
    let publicKey = privateKey.publicKey
    self.keyPair = (private: privateKey, public: publicKey)
  }

  // MARK: - Public Interface

  /// Sign a message with private key
  public func signMessage(_ message: String) -> Data? {
    let messageData = Data(message.utf8)
    guard let signature = try? keyPair.private.signature(for: messageData) else {
      return nil
    }
    return signature.rawRepresentation
  }

  /// Verify signature with public key
  public func verifySignature(_ signatureData: Data, for message: String) -> Bool {
    let messageData = Data(message.utf8)
    guard let signature = try? P256.Signing.ECDSASignature(rawRepresentation: signatureData) else {
      return false
    }
    return keyPair.public.isValidSignature(signature, for: messageData)
  }

  public func createDirectJWT(_ payload: [String: Any]) -> String? {
    // Create header
    let header = ["alg": "ES256", "typ": "JWT"]
    guard let headerData = try? JSONSerialization.data(withJSONObject: header),
      let headerBase64 = base64URLEncode(headerData)
    else {
      return nil
    }

    guard let payloadData = try? JSONSerialization.data(withJSONObject: payload),
      let payloadBase64 = base64URLEncode(payloadData)
    else {
      return nil
    }

    let headerAndPayload = "\(headerBase64).\(payloadBase64)"
    let messageData = Data(headerAndPayload.utf8)
    guard let signature = try? keyPair.private.signature(for: SHA256.hash(data: messageData)),
      let signatureBase64 = base64URLEncode(signature.rawRepresentation)
    else {
      return nil
    }

    return "\(headerAndPayload).\(signatureBase64)"
  }

  /// Verify and decode a direct JWT token (returns the payload dictionary)
  public func verifyDirectJWT(_ token: String) -> [String: Any]? {
    let parts = token.components(separatedBy: ".")
    guard parts.count == 3 else { return nil }

    let headerAndPayload = "\(parts[0]).\(parts[1])"

    // Decode signature
    guard let signatureData = base64URLDecode(parts[2]),
      let signature = try? P256.Signing.ECDSASignature(rawRepresentation: signatureData)
    else {
      return nil
    }

    // Verify signature
    let messageData = Data(headerAndPayload.utf8)
    guard keyPair.public.isValidSignature(signature, for: SHA256.hash(data: messageData)) else {
      return nil
    }

    // Decode payload to extract the direct payload
    guard let payloadData = base64URLDecode(parts[1]),
      let payloadDict = try? JSONSerialization.jsonObject(with: payloadData) as? [String: Any]
    else {
      return nil
    }

    return payloadDict
  }

  /// Get public key as PEM string
  public func getPublicKey() -> String? {
    return keyPair.public.pemRepresentation
  }

  /// Generate new key pair in memory
  public func reloadKeys() {
    let privateKey = P256.Signing.PrivateKey()
    let publicKey = privateKey.publicKey
    self.keyPair = (private: privateKey, public: publicKey)
  }

  // MARK: - Helper Methods

  private func base64URLEncode(_ data: Data) -> String? {
    let base64 = data.base64EncodedString()
    return
      base64
      .replacingOccurrences(of: "+", with: "-")
      .replacingOccurrences(of: "/", with: "_")
      .replacingOccurrences(of: "=", with: "")
  }

  private func base64URLDecode(_ string: String) -> Data? {
    var base64 =
      string
      .replacingOccurrences(of: "-", with: "+")
      .replacingOccurrences(of: "_", with: "/")

    // Add padding if needed
    while base64.count % 4 != 0 {
      base64 += "="
    }

    return Data(base64Encoded: base64)
  }
}
