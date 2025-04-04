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
  #if swift(>=5.6)
    @preconcurrency import Network
  #else
    import Network
  #endif
  import NIOCore

  /// A tag protocol that can be used to cover all network events emitted by `NIOTransportServices`.
  ///
  /// Users are strongly encouraged not to conform their own types to this protocol.
  @available(OSX 10.14, iOS 12.0, tvOS 12.0, watchOS 6.0, *)
  public protocol NIOTSNetworkEvent: Equatable, NIOPreconcurrencySendable {}

  @available(OSX 10.14, iOS 12.0, tvOS 12.0, watchOS 6.0, *)
  public enum NIOTSNetworkEvents {
    /// ``BetterPathAvailable`` is fired whenever the OS has informed NIO that there is a better
    /// path available to the endpoint that this `Channel` is currently connected to,
    /// e.g. the current connection is using an expensive cellular connection and
    /// a cheaper WiFi connection has become available.
    ///
    /// If you can handle this event, you should make a new connection attempt, and then
    /// transfer your work to that connection before closing this one.
    public struct BetterPathAvailable: NIOTSNetworkEvent {
      /// Create a new ``NIOTSNetworkEvents/BetterPathAvailable`` event.
      public init() {}
    }

    /// ``BetterPathUnavailable`` is fired when the OS has informed NIO that no better path to the
    /// to the remote endpoint than the one currently being used by this `Channel` is
    /// currently available.
    public struct BetterPathUnavailable: NIOTSNetworkEvent {
      /// Create a new ``NIOTSNetworkEvents/BetterPathUnavailable`` event.
      public init() {}
    }

    /// ``PathChanged`` is fired whenever the OS has informed NIO that a new path is in use
    /// for this `Channel`.
    public struct PathChanged: NIOTSNetworkEvent {
      /// The new path for this `Channel`.
      public let newPath: NWPath

      /// Create a new ``NIOTSNetworkEvents/PathChanged`` event.
      public init(newPath: NWPath) {
        self.newPath = newPath
      }
    }

    /// ``ConnectToNWEndpoint`` is fired as an outbound event when NIO would like to ask Network.framework
    /// to handle the connection logic (e.g. its own DNS resolution and happy eyeballs racing).
    public struct ConnectToNWEndpoint: NIOTSNetworkEvent {
      /// The endpoint to which we want to connect.
      public let endpoint: NWEndpoint

      /// Create a new ``NIOTSNetworkEvents/ConnectToNWEndpoint`` event.
      public init(endpoint: NWEndpoint) {
        self.endpoint = endpoint
      }
    }

    /// ``BindToNWEndpoint`` is fired as an outbound event when NIO would like to ask Network.framework
    /// to handle the binding logic (e.g. its own support for bonjour and interface selection).
    public struct BindToNWEndpoint: NIOTSNetworkEvent {
      /// The endpoint to which we want to bind.
      public let endpoint: NWEndpoint

      /// Create a new ``NIOTSNetworkEvents/BindToNWEndpoint`` event.
      public init(endpoint: NWEndpoint) {
        self.endpoint = endpoint
      }
    }

    /// ``WaitingForConnectivity`` is fired when when the OS has informed NIO that it cannot immediately connect
    /// to the remote endpoint, but that it is possible that changes in network conditions may
    /// allow connection in future. This can occur in cases where the route is not currently
    /// satisfiable (e.g. because airplane mode is on, or because the app is forbidden from using cellular)
    /// but where a change in network state may allow the connection.
    public struct WaitingForConnectivity: NIOTSNetworkEvent {
      /// The reason the connection couldn't be established at this time.
      ///
      /// Note that these reasons are _not fatal_: applications are strongly advised not to treat them
      /// as fatal, and instead to use them as information to inform UI decisions.
      public var transientError: NWError

      /// Create a new ``NIOTSNetworkEvents/WaitingForConnectivity`` event.
      public init(transientError: NWError) {
        self.transientError = transientError
      }
    }
  }

  // Network.framework has not adopted `Sendable` yet.
  // We therefore need to import it with `@preconcurrency`.
  // `@preconcurrency` is only available in Swift 5.6.
  #if swift(>=5.6)
    @available(OSX 10.14, iOS 12.0, tvOS 12.0, watchOS 6.0, *)
    extension NIOTSNetworkEvents.BetterPathAvailable: Sendable {}
    @available(OSX 10.14, iOS 12.0, tvOS 12.0, watchOS 6.0, *)
    extension NIOTSNetworkEvents.BetterPathUnavailable: Sendable {}
    @available(OSX 10.14, iOS 12.0, tvOS 12.0, watchOS 6.0, *)
    extension NIOTSNetworkEvents.PathChanged: Sendable {}
    @available(OSX 10.14, iOS 12.0, tvOS 12.0, watchOS 6.0, *)
    extension NIOTSNetworkEvents.ConnectToNWEndpoint: Sendable {}
    @available(OSX 10.14, iOS 12.0, tvOS 12.0, watchOS 6.0, *)
    extension NIOTSNetworkEvents.BindToNWEndpoint: Sendable {}
    @available(OSX 10.14, iOS 12.0, tvOS 12.0, watchOS 6.0, *)
    extension NIOTSNetworkEvents.WaitingForConnectivity: Sendable {}
  #endif

#endif
