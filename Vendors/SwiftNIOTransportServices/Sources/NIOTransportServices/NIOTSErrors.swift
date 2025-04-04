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
  import NIOCore

  /// A tag protocol that can be used to cover all errors thrown by `NIOTransportServices`.
  ///
  /// Users are strongly encouraged not to conform their own types to this protocol.
  @available(OSX 10.14, iOS 12.0, tvOS 12.0, watchOS 6.0, *)
  public protocol NIOTSError: Error, Equatable {}

  @available(OSX 10.14, iOS 12.0, tvOS 12.0, watchOS 6.0, *)
  public enum NIOTSErrors {
    /// ``InvalidChannelStateTransition`` is thrown when a channel has been asked to do something
    /// that is incompatible with its current channel state: e.g. attempting to register an
    /// already registered channel.
    public struct InvalidChannelStateTransition: NIOTSError {}

    /// ``NotPreConfigured`` is thrown when a channel has had `registerAlreadyConfigured`
    /// called on it, but has not had the appropriate underlying network object provided.
    public struct NotPreConfigured: NIOTSError {}

    /// ``UnsupportedSocketOption`` is thrown when an attempt is made to configure a socket option that
    /// is not supported by Network.framework.
    public struct UnsupportedSocketOption: NIOTSError {
      public let optionValue: ChannelOptions.Types.SocketOption

      public static func == (lhs: UnsupportedSocketOption, rhs: UnsupportedSocketOption) -> Bool {
        return lhs.optionValue == rhs.optionValue
      }
    }

    /// ``NoCurrentPath`` is thrown when an attempt is made to request path details from a channel and
    /// that channel has no path available. This can manifest, for example, when asking for remote
    /// or local addresses.
    public struct NoCurrentPath: NIOTSError {}

    /// ``NoCurrentConnection`` is thrown when an attempt is made to request connection details from a channel and
    /// that channel has no connection available.
    public struct NoCurrentConnection: NIOTSError {}

    /// ``InvalidPort`` is thrown when the port passed to a method is not valid.
    public struct InvalidPort: NIOTSError {
      /// The provided port.
      public let port: Int
    }

    /// ``UnableToResolveEndpoint`` is thrown when an attempt is made to resolve a local endpoint, but
    /// insufficient information is available to create it.
    public struct UnableToResolveEndpoint: NIOTSError {}

    /// ``BindTimeout`` is thrown when a timeout set for a `NWListenerBootstrap.bind` call has been exceeded
    /// without successfully binding the address.
    public struct BindTimeout: NIOTSError {
      public var timeout: TimeAmount

      public init(timeout: TimeAmount) {
        self.timeout = timeout
      }
    }

    /// ``InvalidHostname`` is thrown when attempting to connect to an invalid host.
    public struct InvalidHostname: NIOTSError {
      public init() {}
    }
  }
#endif
