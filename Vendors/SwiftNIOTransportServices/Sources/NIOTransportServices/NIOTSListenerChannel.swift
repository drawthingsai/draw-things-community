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
  import Foundation
  import NIOCore
  import NIOFoundationCompat
  import NIOConcurrencyHelpers
  import Dispatch
  import Network
  import Atomics

  @available(OSX 10.14, iOS 12.0, tvOS 12.0, watchOS 6.0, *)
  internal final class NIOTSListenerChannel {
    /// The `ByteBufferAllocator` for this `Channel`.
    public let allocator = ByteBufferAllocator()

    /// An `EventLoopFuture` that will complete when this channel is finally closed.
    public var closeFuture: EventLoopFuture<Void> {
      return self.closePromise.futureResult
    }

    /// The parent `Channel` for this one, if any.
    public let parent: Channel? = nil

    /// The `EventLoop` this `Channel` belongs to.
    internal let tsEventLoop: NIOTSEventLoop

    private var _pipeline: ChannelPipeline! = nil  // this is really a constant (set in .init) but needs `self` to be constructed and therefore a `var`. Do not change as this needs to accessed from arbitrary threads.

    internal let closePromise: EventLoopPromise<Void>

    /// The underlying `NWListener` that this `Channel` wraps. This is only non-nil
    /// after the initial connection attempt has been made.
    private var nwListener: NWListener?

    /// The TCP options for this listener.
    private let tcpOptions: NWProtocolTCP.Options

    /// The TLS options for this listener.
    private let tlsOptions: NWProtocolTLS.Options?

    /// The `DispatchQueue` that socket events for this connection will be dispatched onto.
    private let connectionQueue: DispatchQueue

    /// An `EventLoopPromise` that will be succeeded or failed when a bind attempt succeeds or fails.
    private var bindPromise: EventLoopPromise<Void>?

    /// The state of this connection channel.
    internal var state: ChannelState<ActiveSubstate> = .idle

    /// The kinds of channel activation this channel supports
    internal let supportedActivationType: ActivationType = .bind

    /// The active state, used for safely reporting the channel state across threads.
    internal var isActive0 = ManagedAtomic(false)

    /// Whether a call to NWListener.receive has been made, but the completion
    /// handler has not yet been invoked.
    private var outstandingRead: Bool = false

    /// Whether autoRead is enabled for this channel.
    private var autoRead: Bool = true

    /// The value of SO_REUSEADDR.
    private var reuseAddress = false

    /// The value of SO_REUSEPORT.
    private var reusePort = false

    /// The value of the allowLocalEndpointReuse option.
    private var allowLocalEndpointReuse = false

    /// Whether to enable peer-to-peer connectivity when using Bonjour services.
    private var enablePeerToPeer = false

    /// The event loop group to use for child channels.
    private let childLoopGroup: EventLoopGroup

    /// The QoS to use for child channels.
    private let childChannelQoS: DispatchQoS?

    /// The TCP options to use for child channels.
    private let childTCPOptions: NWProtocolTCP.Options

    /// The TLS options to use for child channels.
    private let childTLSOptions: NWProtocolTLS.Options?

    /// The cache of the local and remote socket addresses. Must be accessed using _addressCacheLock.
    private var _addressCache = AddressCache(local: nil, remote: nil)

    /// A lock that guards the _addressCache.
    private let _addressCacheLock = NIOLock()

    /// Create a `NIOTSListenerChannel` on a given `NIOTSEventLoop`.
    ///
    /// Note that `NIOTSListenerChannel` objects cannot be created on arbitrary loops types.
    internal init(
      eventLoop: NIOTSEventLoop,
      qos: DispatchQoS? = nil,
      tcpOptions: NWProtocolTCP.Options,
      tlsOptions: NWProtocolTLS.Options?,
      childLoopGroup: EventLoopGroup,
      childChannelQoS: DispatchQoS?,
      childTCPOptions: NWProtocolTCP.Options,
      childTLSOptions: NWProtocolTLS.Options?
    ) {
      self.tsEventLoop = eventLoop
      self.closePromise = eventLoop.makePromise()
      self.connectionQueue = eventLoop.channelQueue(
        label: "nio.transportservices.listenerchannel", qos: qos)
      self.tcpOptions = tcpOptions
      self.tlsOptions = tlsOptions
      self.childLoopGroup = childLoopGroup
      self.childChannelQoS = childChannelQoS
      self.childTCPOptions = childTCPOptions
      self.childTLSOptions = childTLSOptions

      // Must come last, as it requires self to be completely initialized.
      self._pipeline = ChannelPipeline(channel: self)
    }

    /// Create a `NIOTSListenerChannel` with an already-established `NWListener`.
    internal convenience init(
      wrapping listener: NWListener,
      on eventLoop: NIOTSEventLoop,
      qos: DispatchQoS? = nil,
      tcpOptions: NWProtocolTCP.Options,
      tlsOptions: NWProtocolTLS.Options?,
      childLoopGroup: EventLoopGroup,
      childChannelQoS: DispatchQoS?,
      childTCPOptions: NWProtocolTCP.Options,
      childTLSOptions: NWProtocolTLS.Options?
    ) {
      self.init(
        eventLoop: eventLoop,
        qos: qos,
        tcpOptions: tcpOptions,
        tlsOptions: tlsOptions,
        childLoopGroup: childLoopGroup,
        childChannelQoS: childChannelQoS,
        childTCPOptions: childTCPOptions,
        childTLSOptions: childTLSOptions
      )
      self.nwListener = listener
    }
  }

  // MARK:- NIOTSListenerChannel implementation of Channel
  @available(OSX 10.14, iOS 12.0, tvOS 12.0, watchOS 6.0, *)
  extension NIOTSListenerChannel: Channel {
    /// The `ChannelPipeline` for this `Channel`.
    public var pipeline: ChannelPipeline {
      return self._pipeline
    }

    /// The local address for this channel.
    public var localAddress: SocketAddress? {
      return self._addressCacheLock.withLock {
        return self._addressCache.local
      }
    }

    /// The remote address for this channel.
    public var remoteAddress: SocketAddress? {
      return self._addressCacheLock.withLock {
        return self._addressCache.remote
      }
    }

    /// Whether this channel is currently writable.
    public var isWritable: Bool {
      // TODO: implement
      return true
    }

    public var _channelCore: ChannelCore {
      return self
    }

    public func setOption<Option: ChannelOption>(_ option: Option, value: Option.Value)
      -> EventLoopFuture<Void>
    {
      if self.eventLoop.inEventLoop {
        return self.eventLoop.makeCompletedFuture(
          Result { try setOption0(option: option, value: value) })
      } else {
        return self.eventLoop.submit { try self.setOption0(option: option, value: value) }
      }
    }

    fileprivate func setOption0<Option: ChannelOption>(option: Option, value: Option.Value) throws {
      self.eventLoop.preconditionInEventLoop()

      guard !self.closed else {
        throw ChannelError.ioOnClosedChannel
      }

      // TODO: Many more channel options, both from NIO and Network.framework.
      switch option {
      case is ChannelOptions.Types.AutoReadOption:
        // AutoRead is currently mandatory for TS listeners.
        if value as! ChannelOptions.Types.AutoReadOption.Value == false {
          throw ChannelError.operationUnsupported
        }
      case let optionValue as ChannelOptions.Types.SocketOption:
        // SO_REUSEADDR and SO_REUSEPORT are handled here.
        switch (optionValue.level, optionValue.name) {
        case (SOL_SOCKET, SO_REUSEADDR):
          self.reuseAddress = (value as! SocketOptionValue) != Int32(0)
        case (SOL_SOCKET, SO_REUSEPORT):
          self.reusePort = (value as! SocketOptionValue) != Int32(0)
        default:
          try self.tcpOptions.applyChannelOption(
            option: optionValue, value: value as! SocketOptionValue)
        }
      case is NIOTSChannelOptions.Types.NIOTSEnablePeerToPeerOption:
        self.enablePeerToPeer =
          value as! NIOTSChannelOptions.Types.NIOTSEnablePeerToPeerOption.Value
      case is NIOTSChannelOptions.Types.NIOTSAllowLocalEndpointReuse:
        self.allowLocalEndpointReuse =
          value as! NIOTSChannelOptions.Types.NIOTSEnablePeerToPeerOption.Value
      default:
        fatalError("option \(option) not supported")
      }
    }

    public func getOption<Option: ChannelOption>(_ option: Option) -> EventLoopFuture<Option.Value>
    {
      if eventLoop.inEventLoop {
        return self.eventLoop.makeCompletedFuture(Result { try getOption0(option: option) })
      } else {
        return eventLoop.submit { try self.getOption0(option: option) }
      }
    }

    fileprivate func getOption0<Option: ChannelOption>(option: Option) throws -> Option.Value {
      self.eventLoop.preconditionInEventLoop()

      guard !self.closed else {
        throw ChannelError.ioOnClosedChannel
      }

      switch option {
      case is ChannelOptions.Types.AutoReadOption:
        return autoRead as! Option.Value
      case let optionValue as ChannelOptions.Types.SocketOption:
        // SO_REUSEADDR and SO_REUSEPORT are handled here.
        switch (optionValue.level, optionValue.name) {
        case (SOL_SOCKET, SO_REUSEADDR):
          return Int32(self.reuseAddress ? 1 : 0) as! Option.Value
        case (SOL_SOCKET, SO_REUSEPORT):
          return Int32(self.reusePort ? 1 : 0) as! Option.Value
        default:
          return try self.tcpOptions.valueFor(socketOption: optionValue) as! Option.Value
        }
      case is NIOTSChannelOptions.Types.NIOTSEnablePeerToPeerOption:
        return self.enablePeerToPeer as! Option.Value
      case is NIOTSChannelOptions.Types.NIOTSAllowLocalEndpointReuse:
        return self.allowLocalEndpointReuse as! Option.Value
      default:
        fatalError("option \(option) not supported")
      }
    }
  }

  // MARK:- NIOTSListenerChannel implementation of StateManagedChannel.
  @available(OSX 10.14, iOS 12.0, tvOS 12.0, watchOS 6.0, *)
  extension NIOTSListenerChannel: StateManagedChannel {
    typealias ActiveSubstate = ListenerActiveSubstate

    /// Listener channels do not have active substates: they are either active or they
    /// are not.
    enum ListenerActiveSubstate: ActiveChannelSubstate {
      case active

      init() {
        self = .active
      }
    }
    internal func alreadyConfigured0(promise: EventLoopPromise<Void>?) {
      guard let listener = nwListener else {
        promise?.fail(NIOTSErrors.NotPreConfigured())
        return
      }

      guard case .setup = listener.state else {
        promise?.fail(NIOTSErrors.NotPreConfigured())
        return
      }
      self.bindPromise = promise
      listener.stateUpdateHandler = self.stateUpdateHandler(newState:)
      listener.newConnectionHandler = self.newConnectionHandler(connection:)
      listener.start(queue: self.connectionQueue)
    }

    public func localAddress0() throws -> SocketAddress {
      guard let listener = self.nwListener else {
        throw ChannelError.ioOnClosedChannel
      }

      guard let localEndpoint = listener.parameters.requiredLocalEndpoint else {
        throw NIOTSErrors.UnableToResolveEndpoint()
      }

      var address = try SocketAddress(fromNWEndpoint: localEndpoint)

      // If we were asked to bind port 0, we need to update that.
      if let port = address.port, port == 0 {
        // We were. Let's ask Network.framework what we got. Nothing is an unacceptable answer.
        guard let actualPort = listener.port else {
          throw NIOTSErrors.UnableToResolveEndpoint()
        }
        address.newPort(actualPort.rawValue)
      }

      return address
    }

    public func remoteAddress0() throws -> SocketAddress {
      throw ChannelError.operationUnsupported
    }

    internal func beginActivating0(to target: NWEndpoint, promise: EventLoopPromise<Void>?) {
      assert(self.bindPromise == nil)
      self.bindPromise = promise

      let parameters = NWParameters(tls: self.tlsOptions, tcp: self.tcpOptions)

      // If we have a target that is not for a Bonjour service, we treat this as a request for
      // a specific local endpoint. That gets configured on the parameters. If this is a bonjour
      // endpoint, we deal with that later, though if it has requested a specific interface we
      // set that now.
      switch target {
      case .hostPort, .unix:
        parameters.requiredLocalEndpoint = target
      case .service(_, _, _, let interface):
        parameters.requiredInterface = interface
      default:
        // We can't use `@unknown default` and explicitly list cases we know about since they
        // would require availability checks within the switch statement (`.url` was added in
        // macOS 10.15).
        ()
      }

      // Network.framework munges REUSEADDR and REUSEPORT together, so we turn this on if we need
      // either or it's been explicitly set.
      parameters.allowLocalEndpointReuse =
        self.reuseAddress || self.reusePort || self.allowLocalEndpointReuse

      parameters.includePeerToPeer = self.enablePeerToPeer

      let listener: NWListener
      do {
        listener = try NWListener(using: parameters)
      } catch {
        self.close0(error: error, mode: .all, promise: nil)
        return
      }

      if case .service(let name, let type, let domain, _) = target {
        // Ok, now we deal with Bonjour.
        listener.service = NWListener.Service(name: name, type: type, domain: domain)
      }

      listener.stateUpdateHandler = self.stateUpdateHandler(newState:)
      listener.newConnectionHandler = self.newConnectionHandler(connection:)

      // Ok, state is ready. Let's go!
      self.nwListener = listener
      listener.start(queue: self.connectionQueue)
    }

    public func write0(_ data: NIOAny, promise: EventLoopPromise<Void>?) {
      promise?.fail(ChannelError.operationUnsupported)
    }

    public func flush0() {
      // Flush is not supported on listening channels.
    }

    /// Perform a read from the network.
    ///
    /// This method has a slightly strange semantic, because we do not allow multiple reads at once. As a result, this
    /// is a *request* to read, and if there is a read already being processed then this method will do nothing.
    public func read0() {
      // AutoRead is currently mandatory, so this method does nothing.
    }

    public func doClose0(error: Error) {
      // Step 1: tell the networking stack (if created) that we're done.
      if let listener = self.nwListener {
        listener.cancel()
      }

      // Step 2: fail any pending bind promise.
      if let pendingBind = self.bindPromise {
        self.bindPromise = nil
        pendingBind.fail(error)
      }
    }

    public func doHalfClose0(error: Error, promise: EventLoopPromise<Void>?) {
      promise?.fail(ChannelError.operationUnsupported)
    }

    public func triggerUserOutboundEvent0(_ event: Any, promise: EventLoopPromise<Void>?) {
      switch event {
      case let x as NIOTSNetworkEvents.BindToNWEndpoint:
        self.bind0(to: x.endpoint, promise: promise)
      default:
        promise?.fail(ChannelError.operationUnsupported)
      }
    }

    public func channelRead0(_ data: NIOAny) {
      self.eventLoop.assertInEventLoop()

      let channel = self.unwrapData(data, as: NIOTSConnectionChannel.self)
      let p: EventLoopPromise<Void> = channel.eventLoop.makePromise()
      channel.eventLoop.execute {
        channel.registerAlreadyConfigured0(promise: p)
        p.futureResult.whenFailure { (_: Error) in
          channel.close(promise: nil)
        }
      }
    }

    public func errorCaught0(error: Error) {
      // Currently we don't do anything with errors that pass through the pipeline
      return
    }

    /// A function that will trigger a socket read if necessary.
    internal func readIfNeeded0() {
      // AutoRead is currently mandatory, so this does nothing.
    }
  }

  // MARK:- Implementations of the callbacks passed to NWListener.
  @available(OSX 10.14, iOS 12.0, tvOS 12.0, watchOS 6.0, *)
  extension NIOTSListenerChannel {
    /// Called by the underlying `NWListener` when its internal state has changed.
    private func stateUpdateHandler(newState: NWListener.State) {
      switch newState {
      case .setup:
        preconditionFailure("Should not be told about this state.")
      case .waiting:
        break
      case .ready:
        // Transitioning to ready means the bind succeeded. Hooray!
        self.bindComplete0()
      case .cancelled:
        // This is the network telling us we're closed. We don't need to actually do anything here
        // other than check our state is ok.
        assert(self.closed)
        self.nwListener = nil
      case .failed(let err):
        // The connection has failed for some reason.
        self.close0(error: err, mode: .all, promise: nil)
      default:
        // This clause is here to help the compiler out: it's otherwise not able to
        // actually validate that the switch is exhaustive. Trust me, it is.
        fatalError("Unreachable")
      }
    }

    /// Called by the underlying `NWListener` when a new connection has been received.
    private func newConnectionHandler(connection: NWConnection) {
      guard self.isActive else {
        return
      }

      let newChannel = NIOTSConnectionChannel(
        wrapping: connection,
        on: self.childLoopGroup.next() as! NIOTSEventLoop,
        parent: self,
        qos: self.childChannelQoS,
        tcpOptions: self.childTCPOptions,
        tlsOptions: self.childTLSOptions)

      self.pipeline.fireChannelRead(NIOAny(newChannel))
      self.pipeline.fireChannelReadComplete()
    }
  }

  // MARK:- Implementations of state management for the channel.
  @available(OSX 10.14, iOS 12.0, tvOS 12.0, watchOS 6.0, *)
  extension NIOTSListenerChannel {
    /// Make the channel active.
    private func bindComplete0() {
      let promise = self.bindPromise
      self.bindPromise = nil

      // Before becoming active, update the cached addresses. Remote is always nil.
      let localAddress = try? self.localAddress0()

      self._addressCacheLock.withLock {
        self._addressCache = AddressCache(local: localAddress, remote: nil)
      }

      self.becomeActive0(promise: promise)
    }
  }

  @available(OSX 10.14, iOS 12.0, tvOS 12.0, watchOS 6.0, *)
  extension NIOTSListenerChannel {
    internal struct SynchronousOptions: NIOSynchronousChannelOptions {
      private let channel: NIOTSListenerChannel

      fileprivate init(channel: NIOTSListenerChannel) {
        self.channel = channel
      }

      public func setOption<Option: ChannelOption>(_ option: Option, value: Option.Value) throws {
        try self.channel.setOption0(option: option, value: value)
      }

      public func getOption<Option: ChannelOption>(_ option: Option) throws -> Option.Value {
        return try self.channel.getOption0(option: option)
      }
    }

    public var syncOptions: NIOSynchronousChannelOptions? {
      return SynchronousOptions(channel: self)
    }
  }

  @available(OSX 10.14, iOS 12.0, tvOS 12.0, watchOS 6.0, *)
  extension NIOTSListenerChannel: @unchecked Sendable {}

#endif
