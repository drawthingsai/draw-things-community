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

#if canImport(Network)
  import XCTest
  import NIOCore
  import NIOEmbedded
  import NIOTransportServices

  class NIOFilterEmptyWritesHandlerTests: XCTestCase {
    var allocator: ByteBufferAllocator!
    var channel: EmbeddedChannel!
    var eventLoop: EmbeddedEventLoop!

    override func setUp() {
      self.channel = EmbeddedChannel(handler: NIOFilterEmptyWritesHandler())
      XCTAssertNoThrow(try self.channel.connect(to: .init(ipAddress: "1.1.1.1", port: 1)).wait())
      self.allocator = self.channel.allocator
      let eventLoop = self.channel.eventLoop as! EmbeddedEventLoop
      self.eventLoop = eventLoop
    }

    override func tearDown() {
      XCTAssertNoThrow(
        XCTAssertTrue(try self.channel.finish().isClean)
      )
      self.channel = nil
      self.eventLoop = nil
    }

    func testEmptyWritePromise() {
      let emptyWrite = self.allocator.buffer(capacity: 0)
      let emptyWritePromise = self.eventLoop.makePromise(of: Void.self)
      self.channel.write(NIOAny(emptyWrite), promise: emptyWritePromise)
      self.channel.flush()
      XCTAssertNoThrow(
        try emptyWritePromise.futureResult.wait()
      )
      XCTAssertNoThrow(
        XCTAssertNil(try channel.readOutbound(as: ByteBuffer.self))
      )
    }

    func testEmptyWritesNoWriteThrough() {
      class OutboundTestHandler: ChannelOutboundHandler {
        typealias OutboundIn = ByteBuffer
        typealias OutboundOut = ByteBuffer

        func write(
          context: ChannelHandlerContext,
          data: NIOAny,
          promise: EventLoopPromise<Void>?
        ) {
          XCTFail()
          context.write(data, promise: promise)
        }
      }
      XCTAssertNoThrow(
        try self.channel.pipeline.addHandler(
          OutboundTestHandler(),
          position: .first
        ).wait()
      )
      let emptyWrite = self.allocator.buffer(capacity: 0)
      let thenEmptyWrite = self.allocator.buffer(capacity: 0)
      let thenEmptyWritePromise = self.eventLoop.makePromise(of: Void.self)
      self.channel.write(NIOAny(emptyWrite), promise: nil)
      self.channel.write(
        NIOAny(thenEmptyWrite),
        promise: thenEmptyWritePromise)
      self.channel.flush()
      XCTAssertNoThrow(try thenEmptyWritePromise.futureResult.wait())
      XCTAssertNoThrow(
        XCTAssertNil(try self.channel.readOutbound(as: ByteBuffer.self))
      )
    }

    func testSomeWriteThenEmptyWritePromiseCascade() {
      let someWrite = self.allocator.bufferFor(string: "non empty")
      let someWritePromise = self.eventLoop.makePromise(of: Void.self)
      let thenEmptyWrite = self.allocator.buffer(capacity: 0)
      let thenEmptyWritePromise = self.eventLoop.makePromise(of: Void.self)
      enum CheckOrder {
        case noWrite
        case someWrite
        case thenEmptyWrite
      }
      var checkOrder = CheckOrder.noWrite
      someWritePromise.futureResult.whenSuccess {
        XCTAssertEqual(checkOrder, .noWrite)
        checkOrder = .someWrite
      }
      thenEmptyWritePromise.futureResult.whenSuccess {
        XCTAssertEqual(checkOrder, .someWrite)
        checkOrder = .thenEmptyWrite
      }
      self.channel.write(
        NIOAny(someWrite),
        promise: someWritePromise)
      self.channel.write(
        NIOAny(thenEmptyWrite),
        promise: thenEmptyWritePromise)
      self.channel.flush()
      XCTAssertNoThrow(try thenEmptyWritePromise.futureResult.wait())
      XCTAssertNoThrow(
        XCTAssertNotNil(try self.channel.readOutbound(as: ByteBuffer.self))
      )
      XCTAssertNoThrow(
        XCTAssertNil(try self.channel.readOutbound(as: ByteBuffer.self))
      )
      XCTAssertEqual(checkOrder, .thenEmptyWrite)
    }

    func testEmptyWriteTwicePromiseCascade() {
      let emptyWrite = self.allocator.buffer(capacity: 0)
      let emptyWritePromise = self.eventLoop.makePromise(of: Void.self)
      let thenEmptyWrite = self.allocator.buffer(capacity: 0)
      let thenEmptyWritePromise = self.eventLoop.makePromise(of: Void.self)
      enum CheckOrder {
        case noWrite
        case emptyWrite
        case thenEmptyWrite
      }
      var checkOrder = CheckOrder.noWrite
      emptyWritePromise.futureResult.whenSuccess {
        XCTAssertEqual(checkOrder, .noWrite)
        checkOrder = .emptyWrite
      }
      thenEmptyWritePromise.futureResult.whenSuccess {
        XCTAssertEqual(checkOrder, .emptyWrite)
        checkOrder = .thenEmptyWrite
      }
      self.channel.write(
        NIOAny(emptyWrite),
        promise: emptyWritePromise)
      self.channel.write(
        NIOAny(thenEmptyWrite),
        promise: thenEmptyWritePromise)
      self.channel.flush()
      XCTAssertNoThrow(try thenEmptyWritePromise.futureResult.wait())
      XCTAssertNoThrow(
        XCTAssertNil(try self.channel.readOutbound(as: ByteBuffer.self))
      )
      XCTAssertEqual(checkOrder, .thenEmptyWrite)
    }

    func testEmptyWriteThenSomeWriteThenEmptyWritePromiseCascade() {
      let emptyWrite = self.allocator.buffer(capacity: 0)
      let emptyWritePromise = self.eventLoop.makePromise(of: Void.self)
      let thenSomeWrite = self.allocator.bufferFor(string: "non-empty")
      let thenSomeWritePromise = self.eventLoop.makePromise(of: Void.self)
      let thenEmptyWrite = self.allocator.buffer(capacity: 0)
      let thenEmptyWritePromise = self.eventLoop.makePromise(of: Void.self)
      enum CheckOrder {
        case noWrite
        case emptyWrite
        case thenSomeWrite
        case thenEmptyWrite
      }
      var checkOrder = CheckOrder.noWrite
      emptyWritePromise.futureResult.whenSuccess {
        XCTAssertEqual(checkOrder, .noWrite)
        checkOrder = .emptyWrite
      }
      thenSomeWritePromise.futureResult.whenSuccess {
        XCTAssertEqual(checkOrder, .emptyWrite)
        checkOrder = .thenSomeWrite
      }
      thenEmptyWritePromise.futureResult.whenSuccess {
        XCTAssertEqual(checkOrder, .thenSomeWrite)
        checkOrder = .thenEmptyWrite
      }
      self.channel.write(NIOAny(emptyWrite), promise: emptyWritePromise)
      self.channel.write(NIOAny(thenSomeWrite), promise: thenSomeWritePromise)
      self.channel.write(NIOAny(thenEmptyWrite), promise: thenEmptyWritePromise)
      self.channel.flush()
      XCTAssertNoThrow(try thenEmptyWritePromise.futureResult.wait())
      XCTAssertNoThrow(
        XCTAssertNotNil(try self.channel.readOutbound(as: ByteBuffer.self))
      )
      XCTAssertNoThrow(
        XCTAssertNil(try self.channel.readOutbound(as: ByteBuffer.self))
      )
      XCTAssertEqual(checkOrder, .thenEmptyWrite)
    }

    func testSomeWriteWithNilPromiseThenEmptyWriteWithNilPromiseThenSomeWrite() {
      let someWrite = self.allocator.bufferFor(string: "non empty")
      let thenEmptyWrite = self.allocator.buffer(capacity: 0)
      let thenSomeWrite = self.allocator.bufferFor(string: "then some")
      let thenSomeWritePromise = self.eventLoop.makePromise(of: Void.self)
      self.channel.write(NIOAny(someWrite), promise: nil)
      self.channel.write(NIOAny(thenEmptyWrite), promise: nil)
      self.channel.write(NIOAny(thenSomeWrite), promise: thenSomeWritePromise)
      self.channel.flush()
      XCTAssertNoThrow(try thenSomeWritePromise.futureResult.wait())
      var someWriteOutput: ByteBuffer?
      XCTAssertNoThrow(
        someWriteOutput = try self.channel.readOutbound()
      )
      XCTAssertEqual(someWriteOutput, someWrite)
      var thenSomeWriteOutput: ByteBuffer?
      XCTAssertNoThrow(
        thenSomeWriteOutput = try self.channel.readOutbound()
      )
      XCTAssertEqual(thenSomeWriteOutput, thenSomeWrite)
      XCTAssertNoThrow(
        XCTAssertNil(try self.channel.readOutbound(as: ByteBuffer.self))
      )
    }

    func testSomeWriteAndFlushThenSomeWriteAndFlush() {
      let someWrite = self.allocator.bufferFor(string: "non empty")
      var someWritePromise: EventLoopPromise<Void>! = self.eventLoop.makePromise()
      self.channel.write(NIOAny(someWrite), promise: someWritePromise)
      self.channel.flush()
      XCTAssertNoThrow(
        try someWritePromise.futureResult.wait()
      )
      XCTAssertNoThrow(
        XCTAssertNotNil(try self.channel.readOutbound(as: ByteBuffer.self))
      )
      someWritePromise = nil
      let thenSomeWrite = self.allocator.bufferFor(string: "then some")
      var thenSomeWritePromise: EventLoopPromise<Void>! = self.eventLoop.makePromise()
      self.channel.write(NIOAny(thenSomeWrite), promise: thenSomeWritePromise)
      self.channel.flush()
      XCTAssertNoThrow(
        try thenSomeWritePromise.futureResult.wait()
      )
      XCTAssertNoThrow(
        XCTAssertNotNil(try self.channel.readOutbound(as: ByteBuffer.self))
      )
      thenSomeWritePromise = nil
    }

    func testEmptyWriteAndFlushThenEmptyWriteAndFlush() {
      let emptyWrite = self.allocator.buffer(capacity: 0)
      var emptyWritePromise: EventLoopPromise<Void>! = self.eventLoop.makePromise()
      self.channel.write(NIOAny(emptyWrite), promise: emptyWritePromise)
      self.channel.flush()
      XCTAssertNoThrow(
        try emptyWritePromise.futureResult.wait()
      )
      XCTAssertNoThrow(
        XCTAssertNil(try self.channel.readOutbound(as: ByteBuffer.self))
      )
      emptyWritePromise = nil
      let thenEmptyWrite = self.allocator.buffer(capacity: 0)
      var thenEmptyWritePromise: EventLoopPromise<Void>! = self.eventLoop.makePromise()
      self.channel.write(NIOAny(thenEmptyWrite), promise: thenEmptyWritePromise)
      self.channel.flush()
      XCTAssertNoThrow(
        try thenEmptyWritePromise.futureResult.wait()
      )
      XCTAssertNoThrow(
        XCTAssertNil(try self.channel.readOutbound(as: ByteBuffer.self))
      )
      thenEmptyWritePromise = nil
    }

    func testCloseOnConnect() {
      /// This test reproduces https://github.com/grpc/grpc-swift/issues/1014. It does so by testing
      /// that a precondition _does not fire_. This test makes no assertions, it just shouldn't crash.
      let channel = EmbeddedChannel(handler: NIOFilterEmptyWritesHandler())
      let promise = channel.eventLoop.makePromise(of: Void.self)
      let future: EventLoopFuture<Void> = promise.futureResult.flatMap {
        XCTAssertTrue(channel.isActive)
        return channel.close()
      }
      channel.connect(to: try! .init(ipAddress: "1.1.1.1", port: 1), promise: promise)
      XCTAssertNoThrow(try future.wait())
      XCTAssertFalse(channel.isActive)
    }
  }
#endif
