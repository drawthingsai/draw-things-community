//===----------------------------------------------------------------------===//
//
// This source file is part of the SwiftNIO open source project
//
// Copyright (c) 2017-2021 Apple Inc. and the SwiftNIO project authors
// Licensed under Apache License v2.0
//
// See LICENSE.txt for license information
// See CONTRIBUTORS.txt for the list of SwiftNIO project authors
//
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#if canImport(Network)
  import Darwin
  import Foundation
  import NIOCore
  import Network

  @available(OSX 10.14, iOS 12.0, tvOS 12.0, watchOS 6.0, *)
  extension IPv4Address {
    /// Create an `IPv4Address` object from a `sockaddr_in`.
    internal init(fromSockAddr sockAddr: sockaddr_in) {
      var localAddr = sockAddr
      self = withUnsafeBytes(of: &localAddr.sin_addr) {
        precondition($0.count == 4)
        let addrData = Data(bytes: $0.baseAddress!, count: $0.count)
        return IPv4Address(addrData)!
      }
    }
  }

  @available(OSX 10.14, iOS 12.0, tvOS 12.0, watchOS 6.0, *)
  extension IPv6Address {
    internal init(fromSockAddr sockAddr: sockaddr_in6) {
      var localAddr = sockAddr

      // TODO: We should check whether we can reliably pull an interface declaration out
      // here to be more useful.
      self = withUnsafeBytes(of: &localAddr.sin6_addr) {
        precondition($0.count == 16)
        let addrData = Data(bytes: $0.baseAddress!, count: $0.count)
        return IPv6Address(addrData)!
      }
    }
  }

  @available(OSX 10.14, iOS 12.0, tvOS 12.0, watchOS 6.0, *)
  extension NWEndpoint {
    /// Create an `NWEndpoint` value from a NIO `SocketAddress`.
    internal init(fromSocketAddress socketAddress: SocketAddress) {
      switch socketAddress {
      case .unixDomainSocket(let uds):
        var address = uds.address
        let path: String = withUnsafeBytes(of: &address.sun_path) { ptr in
          let ptr = ptr.baseAddress!.bindMemory(to: UInt8.self, capacity: 104)
          return String(cString: ptr)
        }
        self = NWEndpoint.unix(path: path)
      case .v4(let v4Addr):
        let v4Address = IPv4Address(fromSockAddr: v4Addr.address)
        let port = NWEndpoint.Port(rawValue: UInt16(socketAddress.port!))!
        self = NWEndpoint.hostPort(host: .ipv4(v4Address), port: port)
      case .v6(let v6Addr):
        let v6Address = IPv6Address(fromSockAddr: v6Addr.address)
        let port = NWEndpoint.Port(rawValue: UInt16(socketAddress.port!))!
        self = NWEndpoint.hostPort(host: .ipv6(v6Address), port: port)
      }
    }
  }

  // TODO: We'll want to get rid of this when we support returning NWEndpoint directly from
  // the various address-handling functions.
  @available(OSX 10.14, iOS 12.0, tvOS 12.0, watchOS 6.0, *)
  extension SocketAddress {
    internal init(fromNWEndpoint endpoint: NWEndpoint) throws {
      switch endpoint {
      case .hostPort(.ipv4(let host), let port):
        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        addr.sin_port = port.rawValue.bigEndian
        host.rawValue.withUnsafeBytes {
          precondition($0.count == 4)
          memcpy(&addr.sin_addr, $0.baseAddress!, 4)
        }
        self = .init(addr, host: host.debugDescription)
      case .hostPort(.ipv6(let host), let port):
        var addr = sockaddr_in6()
        addr.sin6_family = sa_family_t(AF_INET6)
        addr.sin6_port = port.rawValue.bigEndian
        addr.sin6_len = UInt8(MemoryLayout<sockaddr_in6>.size)
        host.rawValue.withUnsafeBytes {
          precondition($0.count == 16)
          memcpy(&addr.sin6_addr, $0.baseAddress!, 16)
        }
        self = .init(addr, host: host.debugDescription)
      case .unix(let path):
        self = try .init(unixDomainSocketPath: path)
      case .service, .hostPort:
        throw NIOTSErrors.UnableToResolveEndpoint()
      default:
        // We can't use `@unknown default` and explicitly list cases we know about since they
        // would require availability checks within the switch statement (`.url` was added in
        // macOS 10.15).
        throw NIOTSErrors.UnableToResolveEndpoint()
      }
    }
  }

  @available(OSX 10.14, iOS 12.0, tvOS 12.0, watchOS 6.0, *)
  extension SocketAddress {
    /// Change the port on this `SocketAddress` to a new value.
    mutating func newPort(_ port: UInt16) {
      switch self {
      case .v4(let addr):
        var address = addr.address
        address.sin_port = port.bigEndian
        self = SocketAddress(address, host: addr.host)
      case .v6(let addr):
        var address = addr.address
        address.sin6_port = port.bigEndian
        self = SocketAddress(address, host: addr.host)
      case .unixDomainSocket:
        preconditionFailure("Cannot set new port on UDS")
      }
    }
  }
#endif
