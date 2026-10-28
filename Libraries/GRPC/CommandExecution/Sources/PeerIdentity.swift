import Darwin
import Foundation
import NIOCore
import Security

/// The process at the other end of a Unix domain socket, as the kernel reports it.
///
/// The identity comes from the socket's audit token, never from a PID, which can be reused.
public struct PeerIdentity: Sendable {
  public let auditToken: audit_token_t

  public init(auditToken: audit_token_t) {
    self.auditToken = auditToken
  }

  /// Reads the peer's audit token from a connected Unix domain socket channel.
  public static func of(_ channel: Channel) -> EventLoopFuture<PeerIdentity> {
    guard let provider = channel as? SocketOptionProvider else {
      return channel.eventLoop.makeFailedFuture(Error.notASocket)
    }
    let future: EventLoopFuture<audit_token_t> = provider.unsafeGetSocketOption(
      level: SocketOptionLevel(SOL_LOCAL), name: SocketOptionName(LOCAL_PEERTOKEN))
    return future.map(PeerIdentity.init(auditToken:))
  }

  public var pid: pid_t {
    pid_t(bitPattern: auditToken.val.5)
  }

  /// When the process started, in microseconds since 1970, or nil when it has exited.
  ///
  /// With the boot and the pid, this tells two processes apart when a pid is reused.
  public var startTime: UInt64? {
    Self.startTime(of: pid)
  }

  public static func startTime(of pid: pid_t) -> UInt64? {
    var info = proc_bsdinfo()
    let size = Int32(MemoryLayout<proc_bsdinfo>.size)
    guard proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, &info, size) == size else { return nil }
    return info.pbi_start_tvsec * 1_000_000 + info.pbi_start_tvusec
  }

  /// Whether the peer's code is validly signed, satisfies the requirement, and, when asked, runs
  /// with the hardened runtime, so no other process can inject a library into it.
  public func satisfies(_ requirement: String, requiresHardenedRuntime: Bool) -> Bool {
    var token = auditToken
    let tokenData = Data(bytes: &token, count: MemoryLayout<audit_token_t>.size)
    var code: SecCode?
    guard
      SecCodeCopyGuestWithAttributes(
        nil, [kSecGuestAttributeAudit: tokenData] as CFDictionary, [], &code) == errSecSuccess,
      let code
    else { return false }
    var secRequirement: SecRequirement?
    guard
      SecRequirementCreateWithString(requirement as CFString, [], &secRequirement)
        == errSecSuccess, let secRequirement,
      SecCodeCheckValidity(code, [], secRequirement) == errSecSuccess
    else { return false }
    guard requiresHardenedRuntime else { return true }
    var staticCode: SecStaticCode?
    var information: CFDictionary?
    guard SecCodeCopyStaticCode(code, [], &staticCode) == errSecSuccess, let staticCode,
      SecCodeCopySigningInformation(
        staticCode, SecCSFlags(rawValue: kSecCSSigningInformation), &information)
        == errSecSuccess,
      let flags = (information as? [String: Any])?[kSecCodeInfoFlags as String] as? UInt32
    else { return false }
    return flags & SecCodeSignatureFlags.runtime.rawValue != 0
  }

  public enum Error: Swift.Error {
    case notASocket
  }
}
