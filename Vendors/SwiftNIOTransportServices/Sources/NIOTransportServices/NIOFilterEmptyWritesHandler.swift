//===----------------------------------------------------------------------===//
//
// This source file is part of the SwiftNIO open source project
//
// Copyright (c) 2020-2021 Apple Inc. and the SwiftNIO project authors
// Licensed under Apache License v2.0
//
// See LICENSE.txt for license information
// See CONTRIBUTORS.txt for the list of SwiftNIO project authors
//
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

import NIOCore

/// A `ChannelHandler` that checks for outbound writes of zero length, which are then dropped. This is
/// due to a bug in `Network Framework`, where zero byte TCP writes lead to stalled connections.
/// Write promises are confirmed in the correct order.
public final class NIOFilterEmptyWritesHandler: ChannelDuplexHandler {
  public typealias InboundIn = ByteBuffer
  public typealias InboundOut = ByteBuffer
  public typealias OutboundIn = ByteBuffer
  public typealias OutboundOut = ByteBuffer

  fileprivate enum ChannelState: Equatable {
    case notActiveYet
    case open
    case closedFromLocal
    case closedFromRemote
    case error
  }

  private var state: ChannelState = .notActiveYet
  private var prefixEmptyWritePromise: Optional<EventLoopPromise<Void>>
  private var lastWritePromise: Optional<EventLoopPromise<Void>>

  public init() {
    self.prefixEmptyWritePromise = nil
    self.lastWritePromise = nil
  }

  public func write(context: ChannelHandlerContext, data: NIOAny, promise: EventLoopPromise<Void>?)
  {
    switch self.state {
    case .open:
      let buffer = self.unwrapOutboundIn(data)
      if buffer.readableBytes > 0 {
        self.lastWritePromise = promise ?? context.eventLoop.makePromise()
        context.write(data, promise: self.lastWritePromise)
      } else {
        /*
                 Empty writes need to be handled individually depending on:
                 A) Empty write occurring before any non-empty write needs a
                 separate promise to cascade from (prefix)
                 B) Non-empty writes carry a promise, that subsequent empty
                 writes can cascade from
                 */
        switch (self.prefixEmptyWritePromise, self.lastWritePromise, promise) {
        case (_, _, .none): ()
        case (.none, .none, .some(let promise)):
          self.prefixEmptyWritePromise = promise
        case (_, .some(let lastWritePromise), .some(let promise)):
          lastWritePromise.futureResult.cascade(to: promise)
        case (.some(let prefixEmptyWritePromise), .none, .some(let promise)):
          prefixEmptyWritePromise.futureResult.cascade(to: promise)
        }
      }
    case .closedFromLocal, .closedFromRemote, .error:
      // Since channel is closed, Network Framework bug is not triggered for empty writes
      context.write(data, promise: promise)
    case .notActiveYet:
      preconditionFailure()
    }
  }

  public func flush(context: ChannelHandlerContext) {
    self.lastWritePromise = nil
    if let prefixEmptyWritePromise = self.prefixEmptyWritePromise {
      self.prefixEmptyWritePromise = nil
      prefixEmptyWritePromise.succeed(())
    }

    context.flush()
  }
}

#if swift(>=5.6)
  @available(*, unavailable)
  extension NIOFilterEmptyWritesHandler: Sendable {}
#endif

// Connection state management
extension NIOFilterEmptyWritesHandler {
  public func channelActive(context: ChannelHandlerContext) {
    switch self.state {
    case .notActiveYet:
      self.state = .open
      context.fireChannelActive()
    case .closedFromLocal:
      // Closed before we activated, not a problem.
      assert(self.prefixEmptyWritePromise == nil)
    case .open, .closedFromRemote, .error:
      preconditionFailure()
    }
  }

  public func channelInactive(context: ChannelHandlerContext) {
    let save = self.prefixEmptyWritePromise
    self.prefixEmptyWritePromise = nil
    self.lastWritePromise = nil

    switch self.state {
    case .open:
      self.state = .closedFromRemote
      save?.fail(ChannelError.eof)
    case .closedFromLocal, .closedFromRemote, .error:
      assert(save == nil)
    case .notActiveYet:
      preconditionFailure()
    }
    context.fireChannelInactive()
  }

  public func close(
    context: ChannelHandlerContext, mode: CloseMode, promise: EventLoopPromise<Void>?
  ) {
    let save = self.prefixEmptyWritePromise
    self.prefixEmptyWritePromise = nil
    self.lastWritePromise = nil

    switch (mode, self.state) {
    case (.all, .open),
      (.output, .open),

      // We allow closure in .notActiveYet because it is possible to close before the channelActive fires.
      (.all, .notActiveYet),
      (.output, .notActiveYet):
      self.state = .closedFromLocal
      save?.fail(ChannelError.outputClosed)
    case (.all, .closedFromLocal),
      (.output, .closedFromLocal),
      (.all, .closedFromRemote),
      (.output, .closedFromRemote),
      (.all, .error),
      (.output, .error):
      assert(save == nil)
    case (.input, _):
      save?.fail(ChannelError.operationUnsupported)
    }

    context.close(mode: mode, promise: promise)
  }

  public func errorCaught(context: ChannelHandlerContext, error: Error) {
    let save = self.prefixEmptyWritePromise
    self.prefixEmptyWritePromise = nil
    self.lastWritePromise = nil

    switch self.state {
    case .open:
      self.state = .error
      save?.fail(error)
    case .closedFromLocal, .closedFromRemote, .error:
      assert(save == nil)
    case .notActiveYet:
      preconditionFailure()
    }

    context.fireErrorCaught(error)
  }

  public func handlerAdded(context: ChannelHandlerContext) {
    assert(self.state == .notActiveYet)
    if context.channel.isActive {
      self.state = .open
    }
  }
}
