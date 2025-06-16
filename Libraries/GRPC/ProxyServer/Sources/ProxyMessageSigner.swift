import Crypto
import Foundation

public actor ProxyMessageSigner {
  private let privateKeyPath: String
  private let publicKeyPath: String

  // Cached keys
  private var cachedPrivateKey: P256.Signing.PrivateKey?
  private var cachedPublicKey: P256.Signing.PublicKey?

  init(privateKeyPath: String, publicKeyPath: String) {
    self.privateKeyPath = privateKeyPath
    self.publicKeyPath = publicKeyPath
  }

  // MARK: - Key Loading

  private func loadPrivateKey() -> P256.Signing.PrivateKey? {
    if let cached = cachedPrivateKey {
      return cached
    }

    guard let pemData = try? String(contentsOfFile: privateKeyPath),
      let privateKey = try? P256.Signing.PrivateKey(pemRepresentation: pemData)
    else {
      return nil
    }

    cachedPrivateKey = privateKey
    return privateKey
  }

  private func loadPublicKey() -> P256.Signing.PublicKey? {
    if let cached = cachedPublicKey {
      return cached
    }

    guard let pemData = try? String(contentsOfFile: publicKeyPath),
      let publicKey = try? P256.Signing.PublicKey(pemRepresentation: pemData)
    else {
      return nil
    }

    cachedPublicKey = publicKey
    return publicKey
  }

  // MARK: - Public Interface

  /// Sign a message with private key
  public func signMessage(_ message: String) -> Data? {
    guard let privateKey = loadPrivateKey() else { return nil }

    let messageData = Data(message.utf8)
    guard let signature = try? privateKey.signature(for: messageData) else {
      return nil
    }

    return signature.rawRepresentation
  }

  /// Verify signature with public key
  public func verifySignature(_ signatureData: Data, for message: String) -> Bool {
    guard let publicKey = loadPublicKey() else { return false }

    let messageData = Data(message.utf8)
    guard let signature = try? P256.Signing.ECDSASignature(rawRepresentation: signatureData) else {
      return false
    }

    return publicKey.isValidSignature(signature, for: messageData)
  }

  /// Get public key
  public func getPublicKey() -> String? {
    guard let publicKey = loadPublicKey() else { return nil }
    return publicKey.pemRepresentation
  }

  /// Reload keys from disk (clears cache)
  public func reloadKeys() {
    cachedPrivateKey = nil
    cachedPublicKey = nil
    // Keys will be loaded fresh on next use
  }
}
