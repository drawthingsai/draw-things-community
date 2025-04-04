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
  import NIOConcurrencyHelpers
  import NIOFoundationCompat
  import NIOTLS
  import Dispatch
  import Network
  import Security
  import Atomics

  /// Channel options for the connection channel.
  @available(OSX 10.14, iOS 12.0, tvOS 12.0, watchOS 6.0, *)
  private struct ConnectionChannelOptions {
    /// Whether autoRead is enabled for this channel.
    internal var autoRead: Bool = true

    /// Whether we support remote half closure. If not true, remote half closure will
    /// cause connection drops.
    internal var supportRemoteHalfClosure: Bool = false

    /// Whether this channel should wait for the connection to become active.
    internal var waitForActivity: Bool = true
  }

  private typealias PendingWrite = (data: ByteBuffer, promise: EventLoopPromise<Void>?)

  internal struct AddressCache {
    // deliberately lets because they must always be updated together (so forcing `init` is useful).
    let local: SocketAddress?
    let remote: SocketAddress?

    init(local: SocketAddress?, remote: SocketAddress?) {
      self.local = local
      self.remote = remote
    }
  }

  /// A structure that manages backpressure signaling on this channel.
  @available(OSX 10.14, iOS 12.0, tvOS 12.0, watchOS 6.0, *)
  private struct BackpressureManager {
    /// Whether the channel is writable, given the current watermark state.
    ///
    /// This is an atomic only because the channel writability flag needs to be safe to access from multiple
    /// threads. All activity in this structure itself is expected to be thread-safe.
    ///
    /// All code that operates on this atomic uses load/store, not compareAndSwap. This is because we know
    /// that this atomic is only ever written from one thread: the event loop thread. All unsynchronized
    /// access is only reading. As a result, we don't have racing writes, and don't need CAS. This is good,
    /// because in most cases these loads/stores will be free, as the user will never actually check the
    /// channel writability from another thread, meaning this cache line is uncontended. CAS is never free:
    /// it always has some substantial runtime cost over loads/stores.
    let writable = ManagedAtomic(true)

    /// The number of bytes outstanding on the network.
    private var outstandingBytes: Int = 0

    /// The watermarks currently configured by the user.
    private(set) var waterMarks = ChannelOptions.Types.WriteBufferWaterMark(
      low: 32 * 1024, high: 64 * 1024)

    /// Adds `newBytes` to the queue of outstanding bytes, and returns whether this
    /// has caused a writability change.
    ///
    /// - parameters:
    ///     - newBytes: the number of bytes queued to send, but not yet sent.
    /// - returns: Whether the state changed.
    mutating func writabilityChanges(whenQueueingBytes newBytes: Int) -> Bool {
      self.outstandingBytes += newBytes
      if self.outstandingBytes > self.waterMarks.high && self.writable.load(ordering: .relaxed) {
        self.writable.store(false, ordering: .relaxed)
        return true
      }

      return false
    }

    /// Removes `sentBytes` from the queue of outstanding bytes, and returns whether this
    /// has caused a writability change.
    ///
    /// - parameters:
    ///     - newBytes: the number of bytes sent to the network.
    /// - returns: Whether the state changed.
    mutating func writabilityChanges(whenBytesSent sentBytes: Int) -> Bool {
      self.outstandingBytes -= sentBytes
      if self.outstandingBytes < self.waterMarks.low && !self.writable.load(ordering: .relaxed) {
        self.writable.store(true, ordering: .relaxed)
        return true
      }

      return false
    }

    /// Updates the watermarks to `waterMarks`, and returns whether this change has changed the
    /// writability state of the channel.
    ///
    /// - parameters:
    ///     - waterMarks: The new waterMarks to use.
    /// - returns: Whether the state changed.
    mutating func writabilityChanges(
      whenUpdatingWaterMarks waterMarks: ChannelOptions.Types.WriteBufferWaterMark
    ) -> Bool {
      let writable = self.writable.load(ordering: .relaxed)
      self.waterMarks = waterMarks

      if writable && self.outstandingBytes > self.waterMarks.high {
        self.writable.store(false, ordering: .relaxed)
        return true
      } else if !writable && self.outstandingBytes < self.waterMarks.low {
        self.writable.store(true, ordering: .relaxed)
        return true
      }

      return false
    }
  }

  @available(OSX 10.14, iOS 12.0, tvOS 12.0, watchOS 6.0, *)
  internal final class NIOTSConnectionChannel {
    /// The `ByteBufferAllocator` for this `Channel`.
    public let allocator = ByteBufferAllocator()

    /// An `EventLoopFuture` that will complete when this channel is finally closed.
    public var closeFuture: EventLoopFuture<Void> {
      return self.closePromise.futureResult
    }

    /// The parent `Channel` for this one, if any.
    public let parent: Channel?

    /// The `EventLoop` this `Channel` belongs to.
    internal let tsEventLoop: NIOTSEventLoop

    private var _pipeline: ChannelPipeline! = nil  // this is really a constant (set in .init) but needs `self` to be constructed and therefore a `var`. Do not change as this needs to accessed from arbitrary threads.

    internal let closePromise: EventLoopPromise<Void>

    /// The underlying `NWConnection` that this `Channel` wraps. This is only non-nil
    /// after the initial connection attempt has been made.
    private var nwConnection: NWConnection?

    /// The `DispatchQueue` that socket events for this connection will be dispatched onto.
    private let connectionQueue: DispatchQueue

    /// An `EventLoopPromise` that will be succeeded or failed when a connection attempt succeeds or fails.
    private var connectPromise: EventLoopPromise<Void>?

    /// The TCP options for this connection.
    private var tcpOptions: NWProtocolTCP.Options

    /// The TLS options for this connection, if any.
    private var tlsOptions: NWProtocolTLS.Options?

    /// The state of this connection channel.
    internal var state: ChannelState<ActiveSubstate> = .idle

    /// The active state, used for safely reporting the channel state across threads.
    internal var isActive0 = ManagedAtomic(false)

    /// The kinds of channel activation this channel supports
    internal let supportedActivationType: ActivationType = .connect

    /// Whether a call to NWConnection.receive has been made, but the completion
    /// handler has not yet been invoked.
    private var outstandingRead: Bool = false

    /// The options for this channel.
    private var options: ConnectionChannelOptions = ConnectionChannelOptions()

    /// Any pending writes that have yet to be delivered to the network stack.
    private var pendingWrites = CircularBuffer<PendingWrite>(initialCapacity: 8)

    /// An object to keep track of pending writes and manage our backpressure signaling.
    private var backpressureManager = BackpressureManager()

    /// The value of SO_REUSEADDR.
    private var reuseAddress = false

    /// The value of SO_REUSEPORT.
    private var reusePort = false

    /// The value of the allowLocalEndpointReuse option.
    private var allowLocalEndpointReuse = false

    /// Whether to use peer-to-peer connectivity when connecting to Bonjour services.
    private var enablePeerToPeer = false

    /// The cache of the local and remote socket addresses. Must be accessed using _addressCacheLock.
    private var _addressCache = AddressCache(local: nil, remote: nil)

    /// A lock that guards the _addressCache.
    private let _addressCacheLock = NIOLock()

    /// Create a `NIOTSConnectionChannel` on a given `NIOTSEventLoop`.
    ///
    /// Note that `NIOTSConnectionChannel` objects cannot be created on arbitrary loops types.
    internal init(
      eventLoop: NIOTSEventLoop,
      parent: Channel? = nil,
      qos: DispatchQoS? = nil,
      tcpOptions: NWProtocolTCP.Options,
      tlsOptions: NWProtocolTLS.Options?
    ) {
      self.tsEventLoop = eventLoop
      self.closePromise = eventLoop.makePromise()
      self.parent = parent
      self.connectionQueue = eventLoop.channelQueue(
        label: "nio.nioTransportServices.connectionchannel", qos: qos)
      self.tcpOptions = tcpOptions
      self.tlsOptions = tlsOptions

      // Must come last, as it requires self to be completely initialized.
      self._pipeline = ChannelPipeline(channel: self)
    }

    /// Create a `NIOTSConnectionChannel` with an already-established `NWConnection`.
    internal convenience init(
      wrapping connection: NWConnection,
      on eventLoop: NIOTSEventLoop,
      parent: Channel? = nil,
      qos: DispatchQoS? = nil,
      tcpOptions: NWProtocolTCP.Options,
      tlsOptions: NWProtocolTLS.Options?
    ) {
      self.init(
        eventLoop: eventLoop,
        parent: parent,
        qos: qos,
        tcpOptions: tcpOptions,
        tlsOptions: tlsOptions)
      self.nwConnection = connection
    }
  }

  // MARK:- NIOTSConnectionChannel implementation of Channel
  @available(OSX 10.14, iOS 12.0, tvOS 12.0, watchOS 6.0, *)
  extension NIOTSConnectionChannel: Channel {
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
      return self.backpressureManager.writable.load(ordering: .relaxed)
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

    private func setOption0<Option: ChannelOption>(option: Option, value: Option.Value) throws {
      self.eventLoop.preconditionInEventLoop()

      guard !self.closed else {
        throw ChannelError.ioOnClosedChannel
      }

      switch option {
      case _ as ChannelOptions.Types.AutoReadOption:
        self.options.autoRead = value as! Bool
        self.readIfNeeded0()
      case _ as ChannelOptions.Types.AllowRemoteHalfClosureOption:
        self.options.supportRemoteHalfClosure = value as! Bool
      case _ as ChannelOptions.Types.SocketOption:
        let optionValue = option as! ChannelOptions.Types.SocketOption

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
      case _ as ChannelOptions.Types.WriteBufferWaterMarkOption:
        if self.backpressureManager.writabilityChanges(
          whenUpdatingWaterMarks: value as! ChannelOptions.Types.WriteBufferWaterMark)
        {
          self.pipeline.fireChannelWritabilityChanged()
        }
      case _ as NIOTSChannelOptions.Types.NIOTSWaitForActivityOption:
        let newValue = value as! Bool
        self.options.waitForActivity = newValue

        if let state = self.nwConnection?.state, case .waiting(let err) = state, !newValue {
          // We're in waiting now, so we should drop the connection.
          self.close0(error: err, mode: .all, promise: nil)
        }
      case is NIOTSChannelOptions.Types.NIOTSEnablePeerToPeerOption:
        self.enablePeerToPeer =
          value as! NIOTSChannelOptions.Types.NIOTSEnablePeerToPeerOption.Value
      case is NIOTSChannelOptions.Types.NIOTSAllowLocalEndpointReuse:
        self.allowLocalEndpointReuse =
          value as! NIOTSChannelOptions.Types.NIOTSEnablePeerToPeerOption.Value
      default:
        fatalError("option \(type(of: option)).\(option) not supported")
      }
    }

    public func getOption<Option: ChannelOption>(_ option: Option) -> EventLoopFuture<Option.Value>
    {
      if self.eventLoop.inEventLoop {
        return self.eventLoop.makeCompletedFuture(Result { try getOption0(option: option) })
      } else {
        return eventLoop.submit { try self.getOption0(option: option) }
      }
    }

    func getOption0<Option: ChannelOption>(option: Option) throws -> Option.Value {
      self.eventLoop.preconditionInEventLoop()

      guard !self.closed else {
        throw ChannelError.ioOnClosedChannel
      }

      switch option {
      case _ as ChannelOptions.Types.AutoReadOption:
        return self.options.autoRead as! Option.Value
      case _ as ChannelOptions.Types.AllowRemoteHalfClosureOption:
        return self.options.supportRemoteHalfClosure as! Option.Value
      case _ as ChannelOptions.Types.SocketOption:
        let optionValue = option as! ChannelOptions.Types.SocketOption

        // SO_REUSEADDR and SO_REUSEPORT are handled here.
        switch (optionValue.level, optionValue.name) {
        case (SOL_SOCKET, SO_REUSEADDR):
          return Int32(self.reuseAddress ? 1 : 0) as! Option.Value
        case (SOL_SOCKET, SO_REUSEPORT):
          return Int32(self.reusePort ? 1 : 0) as! Option.Value
        default:
          return try self.tcpOptions.valueFor(socketOption: optionValue) as! Option.Value
        }
      case _ as ChannelOptions.Types.WriteBufferWaterMarkOption:
        return self.backpressureManager.waterMarks as! Option.Value
      case _ as NIOTSChannelOptions.Types.NIOTSWaitForActivityOption:
        return self.options.waitForActivity as! Option.Value
      case is NIOTSChannelOptions.Types.NIOTSEnablePeerToPeerOption:
        return self.enablePeerToPeer as! Option.Value
      case is NIOTSChannelOptions.Types.NIOTSAllowLocalEndpointReuse:
        return self.allowLocalEndpointReuse as! Option.Value
      case is NIOTSChannelOptions.Types.NIOTSCurrentPathOption:
        guard let currentPath = self.nwConnection?.currentPath else {
          throw NIOTSErrors.NoCurrentPath()
        }
        return currentPath as! Option.Value
      case is NIOTSChannelOptions.Types.NIOTSMetadataOption:
        let optionValue = option as! NIOTSChannelOptions.Types.NIOTSMetadataOption
        guard let nwConnection = self.nwConnection else {
          throw NIOTSErrors.NoCurrentConnection()
        }
        return nwConnection.metadata(definition: optionValue.definition) as! Option.Value
      default:
        // watchOS 6.0 availability is covered by the @available on this extension.
        if #available(OSX 10.15, iOS 13.0, tvOS 13.0, *) {
          switch option {
          case is NIOTSChannelOptions.Types.NIOTSEstablishmentReportOption:
            guard let nwConnection = self.nwConnection else {
              throw NIOTSErrors.NoCurrentConnection()
            }
            let promise: EventLoopPromise<NWConnection.EstablishmentReport?> =
              eventLoop.makePromise()
            nwConnection.requestEstablishmentReport(queue: connectionQueue) { report in
              promise.succeed(report)
            }
            return promise.futureResult as! Option.Value
          case is NIOTSChannelOptions.Types.NIOTSDataTransferReportOption:
            guard let nwConnection = self.nwConnection else {
              throw NIOTSErrors.NoCurrentConnection()
            }
            return nwConnection.startDataTransferReport() as! Option.Value
          default:
            break
          }
        }

        fatalError("option \(type(of: option)).\(option) not supported")
      }
    }
  }

  // MARK:- NIOTSConnectionChannel implementation of StateManagedChannel.
  @available(OSX 10.14, iOS 12.0, tvOS 12.0, watchOS 6.0, *)
  extension NIOTSConnectionChannel: StateManagedChannel {
    typealias ActiveSubstate = TCPSubstate

    /// A TCP connection may be fully open or partially open. In the fully open state, both
    /// peers may send data. In the partially open states, only one of the two peers may send
    /// data.
    ///
    /// We keep track of this to manage the half-closure state of the TCP connection.
    enum TCPSubstate: ActiveChannelSubstate {
      /// Both peers may send.
      case open

      /// This end of the connection has sent a FIN. We may only receive data.
      case halfClosedLocal

      /// The remote peer has sent a FIN. We may still send data, but cannot expect to
      /// receive more.
      case halfClosedRemote

      /// The channel is "active", but there can be no forward momentum here. The only valid
      /// thing to do in this state is drop the channel.
      case closed

      init() {
        self = .open
      }
    }

    public func localAddress0() throws -> SocketAddress {
      guard let localEndpoint = self.nwConnection?.currentPath?.localEndpoint else {
        throw NIOTSErrors.NoCurrentPath()
      }
      // TODO: Support wider range of address types.
      return try SocketAddress(fromNWEndpoint: localEndpoint)
    }

    public func remoteAddress0() throws -> SocketAddress {
      guard let remoteEndpoint = self.nwConnection?.currentPath?.remoteEndpoint else {
        throw NIOTSErrors.NoCurrentPath()
      }
      // TODO: Support wider range of address types.
      return try SocketAddress(fromNWEndpoint: remoteEndpoint)
    }

    internal func alreadyConfigured0(promise: EventLoopPromise<Void>?) {
      guard let connection = nwConnection else {
        promise?.fail(NIOTSErrors.NotPreConfigured())
        return
      }

      guard case .setup = connection.state else {
        promise?.fail(NIOTSErrors.NotPreConfigured())
        return
      }
      self.connectPromise = promise
      connection.stateUpdateHandler = self.stateUpdateHandler(newState:)
      connection.betterPathUpdateHandler = self.betterPathHandler
      connection.pathUpdateHandler = self.pathChangedHandler(newPath:)
      connection.start(queue: self.connectionQueue)
    }

    internal func beginActivating0(to target: NWEndpoint, promise: EventLoopPromise<Void>?) {
      assert(self.nwConnection == nil)
      assert(self.connectPromise == nil)

      // Before we start, we validate that the target won't cause a crash: see
      // https://github.com/apple/swift-nio/issues/1617.
      if case .hostPort(host: let host, port: _) = target, host == "" {
        // We don't pass the promise in here because we'll actually not complete it. We complete it manually ourselves.
        self.close0(error: NIOTSErrors.InvalidHostname(), mode: .all, promise: nil)
        promise?.fail(NIOTSErrors.InvalidHostname())
        return
      }

      self.connectPromise = promise

      let parameters = NWParameters(tls: self.tlsOptions, tcp: self.tcpOptions)

      // Network.framework munges REUSEADDR and REUSEPORT together, so we turn this on if we need
      // either or it's been explicitly set.
      parameters.allowLocalEndpointReuse =
        self.reuseAddress || self.reusePort || self.allowLocalEndpointReuse

      parameters.includePeerToPeer = self.enablePeerToPeer

      let connection = NWConnection(to: target, using: parameters)
      connection.stateUpdateHandler = self.stateUpdateHandler(newState:)
      connection.betterPathUpdateHandler = self.betterPathHandler
      connection.pathUpdateHandler = self.pathChangedHandler(newPath:)

      // Ok, state is ready. Let's go!
      self.nwConnection = connection
      connection.start(queue: self.connectionQueue)
    }

    public func write0(_ data: NIOAny, promise: EventLoopPromise<Void>?) {
      guard self.isActive else {
        promise?.fail(ChannelError.ioOnClosedChannel)
        return
      }

      // TODO: We would ideally support all of IOData here, gotta work out how to do that without HOL blocking
      // all writes terribly.
      // My best guess at this time is that Data(contentsOf:) may mmap the file in question, which would let us
      // at least only block the network stack itself rather than our thread. I'm not certain though, especially
      // on Linux. Should investigate.
      let data = self.unwrapData(data, as: ByteBuffer.self)
      self.pendingWrites.append((data, promise))

      /// This may cause our writability state to change.
      if self.backpressureManager.writabilityChanges(whenQueueingBytes: data.readableBytes) {
        self.pipeline.fireChannelWritabilityChanged()
      }
    }

    public func flush0() {
      guard self.isActive else {
        return
      }

      guard let conn = self.nwConnection else {
        preconditionFailure("nwconnection cannot be nil while channel is active")
      }

      func completionCallback(promise: EventLoopPromise<Void>?, sentBytes: Int) -> (
        (NWError?) -> Void
      ) {
        return { error in
          if let error = error {
            promise?.fail(error)
          } else {
            promise?.succeed(())
          }

          if self.backpressureManager.writabilityChanges(whenBytesSent: sentBytes) {
            self.pipeline.fireChannelWritabilityChanged()
          }
        }
      }

      conn.batch {
        while self.pendingWrites.count > 0 {
          let write = self.pendingWrites.removeFirst()
          let buffer = write.data
          let content = buffer.getData(at: buffer.readerIndex, length: buffer.readableBytes)
          conn.send(
            content: content,
            completion: .contentProcessed(
              completionCallback(promise: write.promise, sentBytes: buffer.readableBytes)))
        }
      }
    }

    /// Perform a read from the network.
    ///
    /// This method has a slightly strange semantic, because we do not allow multiple reads at once. As a result, this
    /// is a *request* to read, and if there is a read already being processed then this method will do nothing.
    public func read0() {
      guard self.inboundStreamOpen && !self.outstandingRead else {
        return
      }

      guard let conn = self.nwConnection else {
        preconditionFailure("Connection should not be nil")
      }

      // TODO: Can we do something sensible with these numbers?
      self.outstandingRead = true
      conn.receive(
        minimumIncompleteLength: 1, maximumLength: 8192,
        completion: self.dataReceivedHandler(content:context:isComplete:error:))
    }

    public func doClose0(error: Error) {
      guard let conn = self.nwConnection else {
        // We don't have a connection to close here, so we're actually done. Our old state
        // was idle.
        assert(self.pendingWrites.count == 0)
        return
      }

      // Step 1 is to tell the network stack we're done.
      // TODO: Does this drop the connection fully, or can we keep receiving data? Must investigate.
      conn.cancel()

      // Step 2 is to fail all outstanding writes.
      self.dropOutstandingWrites(error: error)

      // Step 3 is to cancel a pending connect promise, if any.
      if let pendingConnect = self.connectPromise {
        self.connectPromise = nil
        pendingConnect.fail(error)
      }
    }

    public func doHalfClose0(error: Error, promise: EventLoopPromise<Void>?) {
      guard let conn = self.nwConnection else {
        // We don't have a connection to half close, so fail the promise.
        promise?.fail(ChannelError.ioOnClosedChannel)
        return
      }

      do {
        try self.state.closeOutput()
      } catch ChannelError.outputClosed {
        // Here we *only* fail the promise, no need to blow up the connection.
        promise?.fail(ChannelError.outputClosed)
        return
      } catch {
        // For any other error, this is fatal.
        self.close0(error: error, mode: .all, promise: promise)
        return
      }

      func completionCallback(for promise: EventLoopPromise<Void>?) -> ((NWError?) -> Void) {
        return { error in
          if let error = error {
            promise?.fail(error)
          } else {
            promise?.succeed(())
          }
        }
      }

      // It should not be possible to have a pending connect promise while we're doing half-closure.
      assert(self.connectPromise == nil)

      // Step 1 is to tell the network stack we're done.
      conn.send(
        content: nil, contentContext: .finalMessage,
        completion: .contentProcessed(completionCallback(for: promise)))

      // Step 2 is to fail all outstanding writes.
      self.dropOutstandingWrites(error: error)
    }

    public func triggerUserOutboundEvent0(_ event: Any, promise: EventLoopPromise<Void>?) {
      switch event {
      case let x as NIOTSNetworkEvents.ConnectToNWEndpoint:
        self.connect0(to: x.endpoint, promise: promise)
      default:
        promise?.fail(ChannelError.operationUnsupported)
      }
    }

    public func channelRead0(_ data: NIOAny) {
      // drop the data, do nothing
      return
    }

    public func errorCaught0(error: Error) {
      // Currently we don't do anything with errors that pass through the pipeline
      return
    }

    /// A function that will trigger a socket read if necessary.
    internal func readIfNeeded0() {
      if self.options.autoRead {
        self.pipeline.read()
      }
    }
  }

  // MARK:- Implementations of the callbacks passed to NWConnection.
  @available(OSX 10.14, iOS 12.0, tvOS 12.0, watchOS 6.0, *)
  extension NIOTSConnectionChannel {
    /// Called by the underlying `NWConnection` when its internal state has changed.
    private func stateUpdateHandler(newState: NWConnection.State) {
      switch newState {
      case .setup:
        preconditionFailure("Should not be told about this state.")
      case .waiting(let err):
        if case .activating = self.state, self.options.waitForActivity {
          // This means the connection cannot currently be completed. We should notify the pipeline
          // here, or support this with a channel option or something, but for now for the sake of
          // demos we will just allow ourselves into this stage.
          self.pipeline.fireUserInboundEventTriggered(
            NIOTSNetworkEvents.WaitingForConnectivity(transientError: err))
          break
        }

        // In this state we've transitioned into waiting, presumably from active or closing. In this
        // version of NIO this is an error, but we should aim to support this at some stage.
        self.close0(error: err, mode: .all, promise: nil)
      case .preparing:
        // This just means connections are being actively established. We have no specific action
        // here.
        break
      case .ready:
        // Transitioning to ready means the connection was succeeded. Hooray!
        self.connectionComplete0()
      case .cancelled:
        // This is the network telling us we're closed. We don't need to actually do anything here
        // other than check our state is ok.
        assert(self.closed)
        self.nwConnection = nil
      case .failed(let err):
        // The connection has failed for some reason.
        self.close0(error: err, mode: .all, promise: nil)
      default:
        // This clause is here to help the compiler out: it's otherwise not able to
        // actually validate that the switch is exhaustive. Trust me, it is.
        fatalError("Unreachable")
      }
    }

    /// Called by the underlying `NWConnection` when a network receive has completed.
    ///
    /// The state matrix here is large. If `content` is non-nil, some data was received: we need to send it down the pipeline
    /// and call channelReadComplete. This may be nil, in which case we expect either `isComplete` to be `true` or `error`
    /// to be non-nil. `isComplete` indicates half-closure on the read side of a connection. `error` is set if the receive
    /// did not complete due to an error, though there may still be some data.
    private func dataReceivedHandler(
      content: Data?, context: NWConnection.ContentContext?, isComplete: Bool, error: NWError?
    ) {
      precondition(self.outstandingRead)
      self.outstandingRead = false

      guard self.isActive else {
        // If we're already not active, we aren't going to process any of this: it's likely the result of an extra
        // read somewhere along the line.
        return
      }

      // First things first, if there's data we need to deliver it.
      if let content = content {
        // It would be nice if we didn't have to do this copy, but I'm not sure how to avoid it with the current Data
        // APIs.
        var buffer = self.allocator.buffer(capacity: content.count)
        buffer.writeBytes(content)
        self.pipeline.fireChannelRead(NIOAny(buffer))
        self.pipeline.fireChannelReadComplete()
      }

      // Next, we want to check if there's an error. If there is, we're going to deliver it, and then close the connection with
      // it. Otherwise, we're going to check if we read EOF, and if we did we'll close with that instead.
      if let error = error {
        self.pipeline.fireErrorCaught(error)
        self.close0(error: error, mode: .all, promise: nil)
      } else if isComplete {
        self.didReadEOF()
      }

      // Last, issue a new read automatically if we need to.
      self.readIfNeeded0()
    }

    /// Called by the underlying `NWConnection` when a better path for this connection is available.
    ///
    /// Notifies the channel pipeline of the new option.
    private func betterPathHandler(available: Bool) {
      if available {
        self.pipeline.fireUserInboundEventTriggered(NIOTSNetworkEvents.BetterPathAvailable())
      } else {
        self.pipeline.fireUserInboundEventTriggered(NIOTSNetworkEvents.BetterPathUnavailable())
      }
    }

    /// Called by the underlying `NWConnection` when this connection changes its network path.
    ///
    /// Notifies the channel pipeline of the new path.
    private func pathChangedHandler(newPath path: NWPath) {
      self.pipeline.fireUserInboundEventTriggered(NIOTSNetworkEvents.PathChanged(newPath: path))
    }
  }

  // MARK:- Implementations of state management for the channel.
  @available(OSX 10.14, iOS 12.0, tvOS 12.0, watchOS 6.0, *)
  extension NIOTSConnectionChannel {
    /// Whether the inbound side of the connection is still open.
    private var inboundStreamOpen: Bool {
      switch self.state {
      case .active(.open), .active(.halfClosedLocal):
        return true
      case .idle, .registered, .activating, .active, .inactive:
        return false
      }
    }

    /// Make the channel active.
    private func connectionComplete0() {
      let promise = self.connectPromise
      self.connectPromise = nil

      // Before becoming active, update the cached addresses.
      let localAddress = try? self.localAddress0()
      let remoteAddress = try? self.remoteAddress0()

      self._addressCacheLock.withLock {
        self._addressCache = AddressCache(local: localAddress, remote: remoteAddress)
      }

      self.becomeActive0(promise: promise)

      if let metadata = self.nwConnection?.metadata(definition: NWProtocolTLS.definition)
        as? NWProtocolTLS.Metadata
      {
        // This is a TLS connection, we may need to fire some other events.
        let securityMetadata = metadata.securityProtocolMetadata

        // The pointer returned by `sec_protocol_metadata_get_negotiated_protocol` is presumably owned by it, so we need
        // to confirm it's still alive while we copy the data out.
        let negotiatedProtocol = withExtendedLifetime(securityMetadata) {
          sec_protocol_metadata_get_negotiated_protocol(metadata.securityProtocolMetadata).map {
            String(cString: $0)
          }
        }

        self.pipeline.fireUserInboundEventTriggered(
          TLSUserEvent.handshakeCompleted(negotiatedProtocol: negotiatedProtocol))
      }
    }

    /// Drop all outstanding writes. Must only be called in the inactive
    /// state.
    private func dropOutstandingWrites(error: Error) {
      while self.pendingWrites.count > 0 {
        self.pendingWrites.removeFirst().promise?.fail(error)
      }
    }

    /// Handle a read EOF.
    ///
    /// If the user has indicated they support half-closure, we will emit the standard half-closure
    /// event. If they have not, we upgrade this to regular closure.
    private func didReadEOF() {
      if self.options.supportRemoteHalfClosure {
        // This is a half-closure, but the connection is still valid.
        do {
          try self.state.closeInput()
        } catch {
          return self.close0(error: error, mode: .all, promise: nil)
        }

        self.pipeline.fireUserInboundEventTriggered(ChannelEvent.inputClosed)
      } else {
        self.close0(error: ChannelError.eof, mode: .all, promise: nil)
      }
    }
  }

  // MARK:- Managing TCP substate.
  @available(OSX 10.14, iOS 12.0, tvOS 12.0, watchOS 6.0, *)
  extension ChannelState where ActiveSubstate == NIOTSConnectionChannel.TCPSubstate {
    /// Close the input side of the TCP state machine.
    fileprivate mutating func closeInput() throws {
      switch self {
      case .active(.open):
        self = .active(.halfClosedRemote)
      case .active(.halfClosedLocal):
        self = .active(.closed)
      case .idle, .registered, .activating, .active(.halfClosedRemote), .active(.closed), .inactive:
        throw NIOTSErrors.InvalidChannelStateTransition()
      }
    }

    /// Close the output side of the TCP state machine.
    fileprivate mutating func closeOutput() throws {
      switch self {
      case .active(.open):
        self = .active(.halfClosedLocal)
      case .active(.halfClosedRemote):
        self = .active(.closed)
      case .active(.halfClosedLocal), .active(.closed):
        // This is a special case for closing the output, as it's user-controlled. If they already
        // closed it, we want to throw a special error to tell them.
        throw ChannelError.outputClosed
      case .idle, .registered, .activating, .inactive:
        throw NIOTSErrors.InvalidChannelStateTransition()
      }
    }
  }

  @available(OSX 10.14, iOS 12.0, tvOS 12.0, watchOS 6.0, *)
  extension NIOTSConnectionChannel {
    internal struct SynchronousOptions: NIOSynchronousChannelOptions {
      private let channel: NIOTSConnectionChannel

      fileprivate init(channel: NIOTSConnectionChannel) {
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

  public struct NIOTSConnectionNotInitialized: Error, Hashable {
    public init() {}
  }

  public struct NIOTSChannelIsNotANIOTSConnectionChannel: Error, Hashable {
    public init() {}
  }

  @available(OSX 10.14, iOS 12.0, tvOS 12.0, watchOS 6.0, *)
  extension NIOTSConnectionChannel {
    fileprivate func metadata(definition: NWProtocolDefinition) throws -> NWProtocolMetadata? {
      guard let nwConnection = self.nwConnection else {
        throw NIOTSConnectionNotInitialized()
      }
      return nwConnection.metadata(definition: definition)
    }
  }

  @available(OSX 10.14, iOS 12.0, tvOS 12.0, watchOS 6.0, *)
  extension Channel {
    /// Retrieves the metadata for a specific protocol from the underlying ``NWConnection``
    /// - Throws: If `self` isn't a `NIOTS` channel with a `NWConnection` this method will throw
    /// ``NIOTSChannelIsNotATransportServicesChannel`` or ``NIOTSConnectionNotInitialized``.
    public func getMetadata(definition: NWProtocolDefinition) -> EventLoopFuture<
      NWProtocolMetadata?
    > {
      guard let channel = self as? NIOTSConnectionChannel else {
        return self.eventLoop.makeFailedFuture(NIOTSChannelIsNotANIOTSConnectionChannel())
      }
      if self.eventLoop.inEventLoop {
        return self.eventLoop.makeCompletedFuture {
          try channel.metadata(definition: definition)
        }
      } else {
        return self.eventLoop.submit {
          try channel.metadata(definition: definition)
        }
      }
    }

    /// Retrieves the metadata for a specific protocol from the underlying ``NWConnection``
    /// - Precondition: Must be called on the `EventLoop` the `Channel` is running on.
    /// - Throws: If `self` isn't a `NIOTS` channel with a `NWConnection` this method will throw
    /// ``NIOTSChannelIsNotATransportServicesChannel`` or ``NIOTSConnectionNotInitialized``.
    public func getMetadataSync(
      definition: NWProtocolDefinition,
      file: StaticString = #fileID,
      line: UInt = #line
    ) throws -> NWProtocolMetadata? {
      self.eventLoop.preconditionInEventLoop(file: file, line: line)
      guard let channel = self as? NIOTSConnectionChannel else {
        throw NIOTSChannelIsNotANIOTSConnectionChannel()
      }
      return try channel.metadata(definition: definition)
    }
  }

#endif
