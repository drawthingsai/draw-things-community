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

  /// An object that conforms to this protocol represents the substate of a channel in the
  /// active state. This can be used to provide more fine-grained tracking of states
  /// within the active state of a channel. Example uses include for tracking TCP half-closure
  /// state in a TCP stream channel.
  @available(OSX 10.14, iOS 12.0, tvOS 12.0, watchOS 6.0, *)
  internal protocol ActiveChannelSubstate {
    /// Create the substate in its default initial state.
    init()
  }

  /// A state machine enum that tracks the state of the connection channel.
  @available(OSX 10.14, iOS 12.0, tvOS 12.0, watchOS 6.0, *)
  internal enum ChannelState<ActiveSubstate: ActiveChannelSubstate> {
    case idle
    case registered
    case activating
    case active(ActiveSubstate)
    case inactive

    /// Unlike every other one of these methods, this one has a side-effect. This is because
    /// it's impossible to correctly be in the reigstered state without verifying that
    /// registration has occurred.
    fileprivate mutating func register(eventLoop: NIOTSEventLoop, channel: Channel) throws {
      guard case .idle = self else {
        throw NIOTSErrors.InvalidChannelStateTransition()
      }
      try eventLoop.register(channel)
      self = .registered
    }

    fileprivate mutating func beginActivating() throws {
      guard case .registered = self else {
        throw NIOTSErrors.InvalidChannelStateTransition()
      }
      self = .activating
    }

    fileprivate mutating func becomeActive() throws {
      guard case .activating = self else {
        throw NIOTSErrors.InvalidChannelStateTransition()
      }
      self = .active(ActiveSubstate())
    }

    fileprivate mutating func becomeInactive() throws -> ChannelState {
      let oldState = self

      switch self {
      case .idle, .registered, .activating, .active:
        self = .inactive
      case .inactive:
        // In this state we're already closed.
        throw ChannelError.alreadyClosed
      }

      return oldState
    }
  }

  /// The kinds of activation that a channel may support.
  @available(OSX 10.14, iOS 12.0, tvOS 12.0, watchOS 6.0, *)
  internal enum ActivationType {
    case connect
    case bind
  }

  /// A protocol for `Channel` implementations with a simple Network.framework
  /// state management layer.
  ///
  /// This protocol provides default hooks for managing state appropriately for a
  /// given channel. It also provides some default implementations of `Channel` methods
  /// for simple behaviours.
  @available(OSX 10.14, iOS 12.0, tvOS 12.0, watchOS 6.0, *)
  internal protocol StateManagedChannel: Channel, ChannelCore {
    associatedtype ActiveSubstate: ActiveChannelSubstate

    var state: ChannelState<ActiveSubstate> { get set }

    var isActive0: ManagedAtomic<Bool> { get set }

    var tsEventLoop: NIOTSEventLoop { get }

    var closePromise: EventLoopPromise<Void> { get }

    var supportedActivationType: ActivationType { get }

    func beginActivating0(to: NWEndpoint, promise: EventLoopPromise<Void>?)

    func becomeActive0(promise: EventLoopPromise<Void>?)

    func alreadyConfigured0(promise: EventLoopPromise<Void>?)

    func doClose0(error: Error)

    func doHalfClose0(error: Error, promise: EventLoopPromise<Void>?)

    func readIfNeeded0()
  }

  @available(OSX 10.14, iOS 12.0, tvOS 12.0, watchOS 6.0, *)
  extension StateManagedChannel {
    public var eventLoop: EventLoop {
      return self.tsEventLoop
    }

    /// Whether this channel is currently active.
    public var isActive: Bool {
      return self.isActive0.load(ordering: .relaxed)
    }

    /// Whether this channel is currently closed. This is not necessary for the public
    /// API, it's just a convenient helper.
    internal var closed: Bool {
      switch self.state {
      case .inactive:
        return true
      case .idle, .registered, .activating, .active:
        return false
      }
    }

    public func register0(promise: EventLoopPromise<Void>?) {
      // TODO: does this need to do anything more than this?
      do {
        try self.state.register(eventLoop: self.tsEventLoop, channel: self)
        self.pipeline.fireChannelRegistered()
        promise?.succeed(())
      } catch {
        promise?.fail(error)
        self.close0(error: error, mode: .all, promise: nil)
      }
    }

    public func registerAlreadyConfigured0(promise: EventLoopPromise<Void>?) {
      do {
        try self.state.register(eventLoop: self.tsEventLoop, channel: self)
        self.pipeline.fireChannelRegistered()
        try self.state.beginActivating()
      } catch {
        promise?.fail(error)
        self.close0(error: error, mode: .all, promise: nil)
        return
      }

      // Ok, we are registered and ready to begin activating. Tell the channel: it must
      // call becomeActive0 directly.
      self.alreadyConfigured0(promise: promise)
    }

    public func connect0(to address: SocketAddress, promise: EventLoopPromise<Void>?) {
      self.activateWithType(
        type: .connect, to: NWEndpoint(fromSocketAddress: address), promise: promise)
    }

    public func connect0(to endpoint: NWEndpoint, promise: EventLoopPromise<Void>?) {
      self.activateWithType(type: .connect, to: endpoint, promise: promise)
    }

    public func bind0(to address: SocketAddress, promise: EventLoopPromise<Void>?) {
      self.activateWithType(
        type: .bind, to: NWEndpoint(fromSocketAddress: address), promise: promise)
    }

    public func bind0(to endpoint: NWEndpoint, promise: EventLoopPromise<Void>?) {
      self.activateWithType(type: .bind, to: endpoint, promise: promise)
    }

    public func close0(error: Error, mode: CloseMode, promise: EventLoopPromise<Void>?) {
      switch mode {
      case .all:
        let oldState: ChannelState<ActiveSubstate>
        do {
          oldState = try self.state.becomeInactive()
        } catch let thrownError {
          promise?.fail(thrownError)
          return
        }

        self.isActive0.store(false, ordering: .relaxed)

        self.doClose0(error: error)

        switch oldState {
        case .active:
          self.pipeline.fireChannelInactive()
          fallthrough
        case .registered, .activating:
          self.tsEventLoop.deregister(self)
          self.pipeline.fireChannelUnregistered()
        case .idle:
          // If this was already idle we don't have anything to do.
          break
        case .inactive:
          preconditionFailure("Should be prevented by state machine")
        }

        // Next we fire the promise passed to this method.
        promise?.succeed(())

        // Now we schedule our final cleanup. We need to keep the channel pipeline alive for at least one more event
        // loop tick, as more work might be using it.
        self.eventLoop.execute {
          self.removeHandlers(pipeline: self.pipeline)
          self.closePromise.succeed(())
        }

      case .input:
        promise?.fail(ChannelError.operationUnsupported)

      case .output:
        self.doHalfClose0(error: error, promise: promise)
      }
    }

    public func becomeActive0(promise: EventLoopPromise<Void>?) {
      // Here we crash if we cannot transition our state. That's because my understanding is that we
      // should not be able to hit this.
      do {
        try self.state.becomeActive()
      } catch {
        self.close0(error: error, mode: .all, promise: promise)
        return
      }

      self.isActive0.store(true, ordering: .relaxed)
      promise?.succeed(())
      self.pipeline.fireChannelActive()
      self.readIfNeeded0()
    }

    /// A helper to handle the fact that activation is mostly common across connect and bind, and that both are
    /// not supported by a single channel type.
    private func activateWithType(
      type: ActivationType, to endpoint: NWEndpoint, promise: EventLoopPromise<Void>?
    ) {
      guard type == self.supportedActivationType else {
        promise?.fail(ChannelError.operationUnsupported)
        return
      }

      do {
        try self.state.beginActivating()
      } catch {
        promise?.fail(error)
        return
      }

      self.beginActivating0(to: endpoint, promise: promise)
    }
  }
#endif
