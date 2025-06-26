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

}
