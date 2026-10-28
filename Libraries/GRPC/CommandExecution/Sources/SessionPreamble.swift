import Foundation
import NIOCore

/// What Local Code and Local Code Computer agree on to reach each other.
public enum CommandExecutionConnection {
  /// Raised only for changes that break compatibility.
  public static let protocolVersion: UInt32 = 1
  public static let appGroupIdentifier = "6HP3U6Z7P6.com.liuliu.local-code"
  /// Short, because `sun_path` holds 104 bytes, including the home folder part of the path.
  public static let socketName = "c.sock"
  /// The metadata that carries the session secret, hex encoded, on every call but `Echo`.
  public static let sessionSecretKey = "session-secret"

  public static func hex(_ bytes: [UInt8]) -> String {
    bytes.map { String(format: "%02x", $0) }.joined()
  }
}

/// The bytes Computer writes at the start of each connection, before gRPC starts.
///
/// grpc-swift does not tell a call which connection it arrived on, so the session secret cannot come
/// from a call. A version byte is followed by the secret on the first connection from a Local Code
/// process, and by zeros on any later one.
public enum SessionPreamble {
  public static let version: UInt8 = 1
  public static let secretLength = 32
  public static let length = 1 + secretLength

  public static func encode(secret: [UInt8]?) -> [UInt8] {
    precondition(secret == nil || secret?.count == secretLength)
    return [version] + (secret ?? [UInt8](repeating: 0, count: secretLength))
  }

  /// The secret in a preamble, or nil when it carries none. Throws on another version.
  public static func decode(_ bytes: [UInt8]) throws -> [UInt8]? {
    guard bytes.count == length, bytes[0] == version else { throw Error.unsupported }
    let secret = Array(bytes[1...])
    return secret.allSatisfy { $0 == 0 } ? nil : secret
  }

  public enum Error: Swift.Error {
    case unsupported
    case untrustedServer
    /// Another connection from this Local Code process took the session first.
    case sessionTaken
  }
}

/// The client's session secret, read by each call as it starts.
public final class SessionSecret: @unchecked Sendable {
  private let lock = NSLock()
  private var bytes: [UInt8]?

  public init() {}

  public var value: [UInt8]? {
    lock.lock()
    defer { lock.unlock() }
    return bytes
  }

  /// Keeps a secret from a preamble. A later connection carries none, and the one already kept
  /// stays; a new secret replaces it only after Computer lost its sessions.
  fileprivate func receive(_ secret: [UInt8]?) -> Bool {
    lock.lock()
    defer { lock.unlock() }
    if let secret {
      bytes = secret
    }
    return bytes != nil
  }
}

/// Sits at the front of a client connection's pipeline: checks that the server is Computer before
/// gRPC sees the connection, then takes the preamble off the front of what the server sends.
///
/// When the server fails the check, or the connection brings no secret and none is kept, it closes
/// the connection and reports why through `didFail`.
public final class SessionPreambleHandler: ChannelInboundHandler, RemovableChannelHandler {
  public typealias InboundIn = ByteBuffer
  public typealias InboundOut = ByteBuffer

  private let isTrustedServer: @Sendable (PeerIdentity) -> Bool
  private let secret: SessionSecret
  private let didFail: @Sendable (Swift.Error) -> Void
  private var buffer: ByteBuffer?

  public init(
    secret: SessionSecret, isTrustedServer: @escaping @Sendable (PeerIdentity) -> Bool,
    didFail: @escaping @Sendable (Swift.Error) -> Void
  ) {
    self.secret = secret
    self.isTrustedServer = isTrustedServer
    self.didFail = didFail
  }

  public func channelActive(context: ChannelHandlerContext) {
    // Nothing reaches gRPC, and so nothing is sent, until the server is known to be Computer.
    PeerIdentity.of(context.channel).whenComplete { result in
      if case .success(let identity) = result, self.isTrustedServer(identity) {
        context.fireChannelActive()
      } else {
        self.didFail(SessionPreamble.Error.untrustedServer)
        // gRPC waits forever on a connection that closes before it sees it open. In the same
        // tick, it can send no more than the HTTP/2 preface: calls wait for the server's settings.
        context.fireChannelActive()
        context.close(promise: nil)
      }
    }
  }

  public func channelRead(context: ChannelHandlerContext, data: NIOAny) {
    var bytes = unwrapInboundIn(data)
    if var buffered = buffer {
      buffered.writeBuffer(&bytes)
      bytes = buffered
    }
    guard bytes.readableBytes >= SessionPreamble.length,
      let preamble = bytes.readBytes(length: SessionPreamble.length)
    else {
      buffer = bytes
      return
    }
    buffer = nil
    do {
      guard secret.receive(try SessionPreamble.decode(preamble)) else {
        throw SessionPreamble.Error.sessionTaken
      }
    } catch {
      didFail(error)
      context.close(promise: nil)
      return
    }
    if bytes.readableBytes > 0 {
      context.fireChannelRead(wrapInboundOut(bytes))
    }
    context.pipeline.removeHandler(self, promise: nil)
  }
}
