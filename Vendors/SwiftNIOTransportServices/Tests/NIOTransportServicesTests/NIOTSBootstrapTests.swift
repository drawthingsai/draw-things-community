//===----------------------------------------------------------------------===//
//
// This source file is part of the SwiftNIO open source project
//
// Copyright (c) 2019-2021 Apple Inc. and the SwiftNIO project authors
// Licensed under Apache License v2.0
//
// See LICENSE.txt for license information
// See CONTRIBUTORS.txt for the list of SwiftNIO project authors
//
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#if canImport(Network)
  import Atomics
  import XCTest
  import Network
  import NIOCore
  import NIOEmbedded
  import NIOTransportServices
  import NIOConcurrencyHelpers
  import Foundation

  @available(OSX 10.14, iOS 12.0, tvOS 12.0, *)
  final class NIOTSBootstrapTests: XCTestCase {
    var groupBag: [NIOTSEventLoopGroup]? = nil  // protected by `self.lock`
    let lock = NIOLock()

    override func setUp() {
      self.lock.withLock {
        XCTAssertNil(self.groupBag)
        self.groupBag = []
      }
    }

    override func tearDown() {
      XCTAssertNoThrow(
        try self.lock.withLock {
          guard let groupBag = self.groupBag else {
            XCTFail()
            return
          }
          XCTAssertNoThrow(
            try groupBag.forEach {
              XCTAssertNoThrow(try $0.syncShutdownGracefully())
            })
          self.groupBag = nil
        })
    }

    func freshEventLoop() -> EventLoop {
      let group: NIOTSEventLoopGroup = .init(loopCount: 1, defaultQoS: .default)
      self.lock.withLock {
        self.groupBag!.append(group)
      }
      return group.next()
    }

    func testBootstrapsTolerateFuturesFromDifferentEventLoopsReturnedInInitializers() throws {
      let childChannelDone = self.freshEventLoop().makePromise(of: Void.self)
      let serverChannelDone = self.freshEventLoop().makePromise(of: Void.self)
      let serverChannel = try assertNoThrowWithValue(
        NIOTSListenerBootstrap(group: self.freshEventLoop())
          .childChannelInitializer { channel in
            channel.eventLoop.preconditionInEventLoop()
            defer {
              childChannelDone.succeed(())
            }
            return self.freshEventLoop().makeSucceededFuture(())
          }
          .serverChannelInitializer { channel in
            channel.eventLoop.preconditionInEventLoop()
            defer {
              serverChannelDone.succeed(())
            }
            return self.freshEventLoop().makeSucceededFuture(())
          }
          .bind(host: "127.0.0.1", port: 0)
          .wait())
      defer {
        XCTAssertNoThrow(try serverChannel.close().wait())
      }

      let client = try assertNoThrowWithValue(
        NIOTSConnectionBootstrap(group: self.freshEventLoop())
          .channelInitializer { channel in
            channel.eventLoop.preconditionInEventLoop()
            return self.freshEventLoop().makeSucceededFuture(())
          }
          .connect(to: serverChannel.localAddress!)
          .wait())
      defer {
        XCTAssertNoThrow(try client.syncCloseAcceptingAlreadyClosed())
      }
      XCTAssertNoThrow(try childChannelDone.futureResult.wait())
      XCTAssertNoThrow(try serverChannelDone.futureResult.wait())
    }

    func testUniveralBootstrapWorks() {
      final class TellMeIfConnectionIsTLSHandler: ChannelInboundHandler {
        typealias InboundIn = ByteBuffer
        typealias OutboundOut = ByteBuffer

        private let isTLS: EventLoopPromise<Bool>
        private var buffer: ByteBuffer?

        init(isTLS: EventLoopPromise<Bool>) {
          self.isTLS = isTLS
        }

        func channelRead(context: ChannelHandlerContext, data: NIOAny) {
          var buffer = self.unwrapInboundIn(data)

          if self.buffer == nil {
            self.buffer = buffer
          } else {
            self.buffer!.writeBuffer(&buffer)
          }

          switch self.buffer!.readBytes(length: 2) {
          case .some([0x16, 0x03]):  // TLS ClientHello always starts with 0x16, 0x03
            self.isTLS.succeed(true)
            context.channel.close(promise: nil)
          case .some(_):
            self.isTLS.succeed(false)
            context.channel.close(promise: nil)
          case .none:
            // not enough data
            ()
          }
        }
      }
      let group = NIOTSEventLoopGroup()
      func makeServer(isTLS: EventLoopPromise<Bool>) throws -> Channel {
        let numberOfConnections = ManagedAtomic(0)
        return try NIOTSListenerBootstrap(group: group)
          .childChannelInitializer { channel in
            XCTAssertEqual(0, numberOfConnections.loadThenWrappingIncrement(ordering: .relaxed))
            return channel.pipeline.addHandler(TellMeIfConnectionIsTLSHandler(isTLS: isTLS))
          }
          .bind(host: "127.0.0.1", port: 0)
          .wait()
      }

      let isTLSConnection1 = group.next().makePromise(of: Bool.self)
      let isTLSConnection2 = group.next().makePromise(of: Bool.self)

      var maybeServer1: Channel? = nil
      var maybeServer2: Channel? = nil

      XCTAssertNoThrow(maybeServer1 = try makeServer(isTLS: isTLSConnection1))
      XCTAssertNoThrow(maybeServer2 = try makeServer(isTLS: isTLSConnection2))

      guard let server1 = maybeServer1, let server2 = maybeServer2 else {
        XCTFail("can't make servers")
        return
      }
      defer {
        XCTAssertNoThrow(try server1.close().wait())
        XCTAssertNoThrow(try server2.close().wait())
      }

      let tlsOptions = NWProtocolTLS.Options()
      let bootstrap = NIOClientTCPBootstrap(
        NIOTSConnectionBootstrap(group: group),
        tls: NIOTSClientTLSProvider(tlsOptions: tlsOptions))
      let tlsBootstrap = NIOClientTCPBootstrap(
        NIOTSConnectionBootstrap(group: group),
        tls: NIOTSClientTLSProvider()
      )
      .enableTLS()

      var buffer = server1.allocator.buffer(capacity: 2)
      buffer.writeString("NO")

      var maybeClient1: Channel? = nil
      XCTAssertNoThrow(maybeClient1 = try bootstrap.connect(to: server1.localAddress!).wait())
      guard let client1 = maybeClient1 else {
        XCTFail("can't connect to server1")
        return
      }
      XCTAssertNotNil(
        try client1.getMetadata(definition: NWProtocolTCP.definition).wait()
          as? NWProtocolTCP.Metadata)
      XCTAssertNoThrow(try client1.writeAndFlush(buffer).wait())

      // The TLS connection won't actually succeed but it takes Network.framework a while to tell us, we don't
      // actually care because we're only interested in the first 2 bytes which we're waiting for below.
      tlsBootstrap.connect(to: server2.localAddress!).whenSuccess { channel in
        XCTFail("TLS connection succeeded but really shouldn't have: \(channel)")
        channel.writeAndFlush(buffer, promise: nil)
      }

      XCTAssertNoThrow(XCTAssertFalse(try isTLSConnection1.futureResult.wait()))
      XCTAssertNoThrow(XCTAssertTrue(try isTLSConnection2.futureResult.wait()))
    }

    func testNIOTSConnectionBootstrapValidatesWorkingELGsCorrectly() {
      let elg = NIOTSEventLoopGroup()
      defer {
        XCTAssertNoThrow(try elg.syncShutdownGracefully())
      }
      let el = elg.next()

      XCTAssertNotNil(NIOTSConnectionBootstrap(validatingGroup: elg))
      XCTAssertNotNil(NIOTSConnectionBootstrap(validatingGroup: el))
    }

    func testNIOTSConnectionBootstrapRejectsNotWorkingELGsCorrectly() {
      let elg = EmbeddedEventLoop()
      defer {
        XCTAssertNoThrow(try elg.syncShutdownGracefully())
      }
      let el = elg.next()

      XCTAssertNil(NIOTSConnectionBootstrap(validatingGroup: elg))
      XCTAssertNil(NIOTSConnectionBootstrap(validatingGroup: el))
    }

    func testNIOTSListenerBootstrapValidatesWorkingELGsCorrectly() {
      let elg = NIOTSEventLoopGroup()
      defer {
        XCTAssertNoThrow(try elg.syncShutdownGracefully())
      }
      let el = elg.next()

      XCTAssertNotNil(NIOTSListenerBootstrap(validatingGroup: elg))
      XCTAssertNotNil(NIOTSListenerBootstrap(validatingGroup: el))
      XCTAssertNotNil(NIOTSListenerBootstrap(validatingGroup: elg, childGroup: elg))
      XCTAssertNotNil(NIOTSListenerBootstrap(validatingGroup: el, childGroup: el))
    }

    func testNIOTSListenerBootstrapRejectsNotWorkingELGsCorrectly() {
      let correctELG = NIOTSEventLoopGroup()
      defer {
        XCTAssertNoThrow(try correctELG.syncShutdownGracefully())
      }

      let wrongELG = EmbeddedEventLoop()
      defer {
        XCTAssertNoThrow(try wrongELG.syncShutdownGracefully())
      }
      let wrongEL = wrongELG.next()
      let correctEL = correctELG.next()

      // both wrong
      XCTAssertNil(NIOTSListenerBootstrap(validatingGroup: wrongELG))
      XCTAssertNil(NIOTSListenerBootstrap(validatingGroup: wrongEL))
      XCTAssertNil(NIOTSListenerBootstrap(validatingGroup: wrongELG, childGroup: wrongELG))
      XCTAssertNil(NIOTSListenerBootstrap(validatingGroup: wrongEL, childGroup: wrongEL))

      // group correct, child group wrong
      XCTAssertNil(NIOTSListenerBootstrap(validatingGroup: correctELG, childGroup: wrongELG))
      XCTAssertNil(NIOTSListenerBootstrap(validatingGroup: correctEL, childGroup: wrongEL))

      // group wrong, child group correct
      XCTAssertNil(NIOTSListenerBootstrap(validatingGroup: wrongELG, childGroup: correctELG))
      XCTAssertNil(NIOTSListenerBootstrap(validatingGroup: wrongEL, childGroup: correctEL))
    }

    func testEndpointReuseShortcutOption() throws {
      let group = NIOTSEventLoopGroup()
      let listenerChannel = try NIOTSListenerBootstrap(group: group)
        .bind(host: "127.0.0.1", port: 0)
        .wait()

      let bootstrap = NIOClientTCPBootstrap(
        NIOTSConnectionBootstrap(group: group),
        tls: NIOInsecureNoTLS()
      )
      .channelConvenienceOptions([.allowLocalEndpointReuse])
      let client = try bootstrap.connect(to: listenerChannel.localAddress!).wait()
      let optionValue = try client.getOption(NIOTSChannelOptions.allowLocalEndpointReuse).wait()
      try client.close().wait()

      XCTAssertEqual(optionValue, true)
    }

    func testShorthandOptionsAreEquivalent() throws {
      func setAndGetOption<Option>(
        option: Option,
        _ applyOptions: (NIOClientTCPBootstrap) -> NIOClientTCPBootstrap
      )
        throws -> Option.Value where Option: ChannelOption
      {
        let group = NIOTSEventLoopGroup()
        let listenerChannel = try NIOTSListenerBootstrap(group: group)
          .bind(host: "127.0.0.1", port: 0)
          .wait()

        let bootstrap = applyOptions(
          NIOClientTCPBootstrap(
            NIOTSConnectionBootstrap(group: group),
            tls: NIOInsecureNoTLS()))
        let client = try bootstrap.connect(to: listenerChannel.localAddress!).wait()
        let optionRead = try client.getOption(option).wait()
        try client.close().wait()
        return optionRead
      }

      func checkOptionEquivalence<Option>(
        longOption: Option, setValue: Option.Value,
        shortOption: ChannelOptions.TCPConvenienceOption
      ) throws
      where Option: ChannelOption, Option.Value: Equatable {
        let longSetValue = try setAndGetOption(option: longOption) { bs in
          bs.channelOption(longOption, value: setValue)
        }
        let shortSetValue = try setAndGetOption(option: longOption) { bs in
          bs.channelConvenienceOptions([shortOption])
        }
        let unsetValue = try setAndGetOption(option: longOption) { $0 }

        XCTAssertEqual(longSetValue, shortSetValue)
        XCTAssertNotEqual(longSetValue, unsetValue)
      }

      try checkOptionEquivalence(
        longOption: NIOTSChannelOptions.allowLocalEndpointReuse,
        setValue: true,
        shortOption: .allowLocalEndpointReuse)
      try checkOptionEquivalence(
        longOption: ChannelOptions.allowRemoteHalfClosure,
        setValue: true,
        shortOption: .allowRemoteHalfClosure)
      try checkOptionEquivalence(
        longOption: ChannelOptions.autoRead,
        setValue: false,
        shortOption: .disableAutoRead)
    }

    func testBootstrapsErrorGracefullyOnOutOfBandPorts() throws {
      let invalidPortNumbers = [-1, 65536]

      let group = NIOTSEventLoopGroup()
      defer {
        try! group.syncShutdownGracefully()
      }

      let listenerBootstrap = NIOTSListenerBootstrap(group: group)
      let connectionBootstrap = NIOTSConnectionBootstrap(group: group)

      for invalidPort in invalidPortNumbers {
        var listenerChannel: Channel?
        var connectionChannel: Channel?

        XCTAssertThrowsError(
          listenerChannel = try listenerBootstrap.bind(host: "localhost", port: invalidPort).wait()
        ) { error in
          XCTAssertNotNil(error as? NIOTSErrors.InvalidPort)
        }
        XCTAssertThrowsError(
          connectionChannel = try connectionBootstrap.connect(host: "localhost", port: invalidPort)
            .wait()
        ) { error in
          XCTAssertNotNil(error as? NIOTSErrors.InvalidPort)
        }

        try? listenerChannel?.close().wait()
        try? connectionChannel?.close().wait()
      }
    }
  }

  extension Channel {
    func syncCloseAcceptingAlreadyClosed() throws {
      do {
        try self.close().wait()
      } catch ChannelError.alreadyClosed {
        /* we're happy with this one */
      } catch let e {
        throw e
      }
    }
  }

#endif
