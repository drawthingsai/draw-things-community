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
  import XCTest
  import Network
  import NIOCore
  import NIOTransportServices

  @available(OSX 10.14, iOS 12.0, tvOS 12.0, *)
  final class BindRecordingHandler: ChannelOutboundHandler {
    typealias OutboundIn = Any
    typealias OutboundOut = Any

    var bindTargets: [SocketAddress] = []
    var endpointTargets: [NWEndpoint] = []

    func bind(
      context: ChannelHandlerContext, to address: SocketAddress, promise: EventLoopPromise<Void>?
    ) {
      self.bindTargets.append(address)
      context.bind(to: address, promise: promise)
    }

    func triggerUserOutboundEvent(
      context: ChannelHandlerContext, event: Any, promise: EventLoopPromise<Void>?
    ) {
      switch event {
      case let evt as NIOTSNetworkEvents.BindToNWEndpoint:
        self.endpointTargets.append(evt.endpoint)
      default:
        break
      }
      context.triggerUserOutboundEvent(event, promise: promise)
    }
  }

  @available(OSX 10.14, iOS 12.0, tvOS 12.0, *)
  class NIOTSListenerChannelTests: XCTestCase {
    private var group: NIOTSEventLoopGroup!

    override func setUp() {
      self.group = NIOTSEventLoopGroup(loopCount: 1)
    }

    override func tearDown() {
      XCTAssertNoThrow(try self.group.syncShutdownGracefully())
    }

    func testBindingToSocketAddressTraversesPipeline() throws {
      let bindRecordingHandler = BindRecordingHandler()
      let target = try SocketAddress.makeAddressResolvingHost("localhost", port: 0)
      let bindBootstrap = NIOTSListenerBootstrap(group: self.group)
        .serverChannelInitializer { channel in channel.pipeline.addHandler(bindRecordingHandler) }

      XCTAssertEqual(bindRecordingHandler.bindTargets, [])
      XCTAssertEqual(bindRecordingHandler.endpointTargets, [])

      let listener = try bindBootstrap.bind(to: target).wait()
      defer {
        XCTAssertNoThrow(try listener.close().wait())
      }

      try self.group.next().submit {
        XCTAssertEqual(bindRecordingHandler.bindTargets, [target])
        XCTAssertEqual(bindRecordingHandler.endpointTargets, [])
      }.wait()
    }

    func testConnectingToHostPortTraversesPipeline() throws {
      let bindRecordingHandler = BindRecordingHandler()
      let bindBootstrap = NIOTSListenerBootstrap(group: self.group)
        .serverChannelInitializer { channel in channel.pipeline.addHandler(bindRecordingHandler) }

      XCTAssertEqual(bindRecordingHandler.bindTargets, [])
      XCTAssertEqual(bindRecordingHandler.endpointTargets, [])

      let listener = try bindBootstrap.bind(host: "localhost", port: 0).wait()
      defer {
        XCTAssertNoThrow(try listener.close().wait())
      }

      try self.group.next().submit {
        XCTAssertEqual(
          bindRecordingHandler.bindTargets,
          [try SocketAddress.makeAddressResolvingHost("localhost", port: 0)])
        XCTAssertEqual(bindRecordingHandler.endpointTargets, [])
      }.wait()
    }

    func testConnectingToEndpointSkipsPipeline() throws {
      let endpoint = NWEndpoint.hostPort(host: .ipv4(.loopback), port: .any)
      let bindRecordingHandler = BindRecordingHandler()
      let bindBootstrap = NIOTSListenerBootstrap(group: self.group)
        .serverChannelInitializer { channel in channel.pipeline.addHandler(bindRecordingHandler) }

      XCTAssertEqual(bindRecordingHandler.bindTargets, [])
      XCTAssertEqual(bindRecordingHandler.endpointTargets, [])

      let listener = try bindBootstrap.bind(endpoint: endpoint).wait()
      defer {
        XCTAssertNoThrow(try listener.close().wait())
      }

      try self.group.next().submit {
        XCTAssertEqual(bindRecordingHandler.bindTargets, [])
        XCTAssertEqual(bindRecordingHandler.endpointTargets, [endpoint])
      }.wait()
    }

    func testSettingGettingReuseaddr() throws {
      let listener = try NIOTSListenerBootstrap(group: self.group).bind(host: "localhost", port: 0)
        .wait()
      defer {
        XCTAssertNoThrow(try listener.close().wait())
      }

      XCTAssertEqual(
        0, try listener.getOption(ChannelOptions.socket(SOL_SOCKET, SO_REUSEADDR)).wait())
      XCTAssertNoThrow(
        try listener.setOption(ChannelOptions.socket(SOL_SOCKET, SO_REUSEADDR), value: 5).wait())
      XCTAssertEqual(
        1, try listener.getOption(ChannelOptions.socket(SOL_SOCKET, SO_REUSEADDR)).wait())
      XCTAssertNoThrow(
        try listener.setOption(ChannelOptions.socket(SOL_SOCKET, SO_REUSEADDR), value: 0).wait())
      XCTAssertEqual(
        0, try listener.getOption(ChannelOptions.socket(SOL_SOCKET, SO_REUSEADDR)).wait())
    }

    func testSettingGettingReuseport() throws {
      let listener = try NIOTSListenerBootstrap(group: self.group).bind(host: "localhost", port: 0)
        .wait()
      defer {
        XCTAssertNoThrow(try listener.close().wait())
      }

      XCTAssertEqual(
        0, try listener.getOption(ChannelOptions.socket(SOL_SOCKET, SO_REUSEPORT)).wait())
      XCTAssertNoThrow(
        try listener.setOption(ChannelOptions.socket(SOL_SOCKET, SO_REUSEPORT), value: 5).wait())
      XCTAssertEqual(
        1, try listener.getOption(ChannelOptions.socket(SOL_SOCKET, SO_REUSEPORT)).wait())
      XCTAssertNoThrow(
        try listener.setOption(ChannelOptions.socket(SOL_SOCKET, SO_REUSEPORT), value: 0).wait())
      XCTAssertEqual(
        0, try listener.getOption(ChannelOptions.socket(SOL_SOCKET, SO_REUSEPORT)).wait())
    }

    func testErrorsInChannelSetupAreFine() throws {
      struct MyError: Error {}

      let listenerFuture = NIOTSListenerBootstrap(group: self.group)
        .serverChannelInitializer { channel in channel.eventLoop.makeFailedFuture(MyError()) }
        .bind(host: "localhost", port: 0)

      do {
        let listener = try listenerFuture.wait()
        XCTAssertNoThrow(try listener.close().wait())
        XCTFail("Did not throw")
      } catch is MyError {
        // fine
      } catch {
        XCTFail("Unexpected error")
      }
    }

    func testCanSafelyInvokeChannelsAcrossThreads() throws {
      // This is a test that aims to trigger TSAN violations.
      let childGroup = NIOTSEventLoopGroup(loopCount: 2)
      let childChannelPromise: EventLoopPromise<Channel> = childGroup.next().makePromise()
      let activePromise: EventLoopPromise<Void> = childGroup.next().makePromise()

      let listener = try NIOTSListenerBootstrap(group: self.group, childGroup: childGroup)
        .childChannelInitializer { channel in
          childChannelPromise.succeed(channel)
          return channel.pipeline.addHandler(PromiseOnActiveHandler(activePromise))
        }.bind(host: "localhost", port: 0).wait()
      defer {
        XCTAssertNoThrow(try listener.close().wait())
      }

      // Connect to the listener.
      let channel = try NIOTSConnectionBootstrap(group: self.group)
        .connect(to: listener.localAddress!).wait()

      // Wait for the child channel to become active.
      let childChannel = try childChannelPromise.futureResult.wait()
      XCTAssertNoThrow(try activePromise.futureResult.wait())

      // Now close the child channel.
      XCTAssertNoThrow(try childChannel.close().wait())
      XCTAssertNoThrow(try channel.closeFuture.wait())
    }

    func testBindingChannelsOnShutdownEventLoopsFails() throws {
      let temporaryGroup = NIOTSEventLoopGroup()
      XCTAssertNoThrow(try temporaryGroup.syncShutdownGracefully())

      let bootstrap = NIOTSListenerBootstrap(group: temporaryGroup)

      do {
        _ = try bootstrap.bind(host: "localhost", port: 0).wait()
      } catch EventLoopError.shutdown {
        // Expected
      } catch {
        XCTFail("Unexpected error: \(error)")
      }
    }

    func testCanObserveValueOfEnablePeerToPeer() throws {
      let listener = try NIOTSListenerBootstrap(group: self.group)
        .serverChannelInitializer { channel in
          return channel.getOption(NIOTSChannelOptions.enablePeerToPeer).map { value in
            XCTAssertFalse(value)
          }.flatMap {
            channel.setOption(NIOTSChannelOptions.enablePeerToPeer, value: true)
          }.flatMap {
            channel.getOption(NIOTSChannelOptions.enablePeerToPeer)
          }.map { value in
            XCTAssertTrue(value)
          }
        }
        .bind(host: "localhost", port: 0).wait()
      do {
        XCTAssertNoThrow(try listener.close().wait())
      }
    }

    func testChannelEmitsChannels() throws {
      class ChannelReceiver: ChannelInboundHandler {
        typealias InboundIn = Channel
        typealias InboundOut = Channel

        private let promise: EventLoopPromise<Channel>

        init(_ promise: EventLoopPromise<Channel>) {
          self.promise = promise
        }

        func channelRead(context: ChannelHandlerContext, data: NIOAny) {
          let channel = self.unwrapInboundIn(data)
          self.promise.succeed(channel)
          context.fireChannelRead(data)
        }
      }

      let channelPromise = self.group.next().makePromise(of: Channel.self)
      let listener = try NIOTSListenerBootstrap(group: self.group)
        .serverChannelInitializer { channel in
          channel.pipeline.addHandler(ChannelReceiver(channelPromise))
        }
        .bind(host: "localhost", port: 0).wait()
      defer {
        XCTAssertNoThrow(try listener.close().wait())
      }

      let connection = try NIOTSConnectionBootstrap(group: self.group).connect(
        to: listener.localAddress!
      ).wait()
      defer {
        XCTAssertNoThrow(try connection.close().wait())
      }

      // We must wait for channel active here, or the socket addresses won't be set.
      let promisedChannel = try channelPromise.futureResult.flatMap {
        (channel) -> EventLoopFuture<Channel> in
        let promiseChannelActive = channel.eventLoop.makePromise(of: Channel.self)
        _ = channel.pipeline.addHandler(WaitForActiveHandler(promiseChannelActive))
        return promiseChannelActive.futureResult
      }.wait()

      XCTAssertEqual(promisedChannel.remoteAddress, connection.localAddress)
      XCTAssertEqual(promisedChannel.localAddress, connection.remoteAddress)
    }

    func testBindTimeout() throws {
      // Testing the bind timeout is damn fiddly, because I don't know a reliable way to force it
      // to happen. The best approach I can think of is to set the timeout to "now".
      // If you see this test fail, verify that it isn't a simple timing issue first.
      let listener = NIOTSListenerBootstrap(group: self.group)
        .bindTimeout(.nanoseconds(0))

      do {
        let channel = try listener.bind(host: "localhost", port: 0).wait()
        XCTAssertNoThrow(try channel.close().wait())
        XCTFail("Did not throw")
      } catch {
        XCTAssertEqual(
          error as? NIOTSErrors.BindTimeout, NIOTSErrors.BindTimeout(timeout: .nanoseconds(0)),
          "unexpected error: \(error)")
      }
    }

    func testLoadingAddressesInMultipleQueues() throws {
      let listener = try NIOTSListenerBootstrap(group: self.group)
        .bind(host: "localhost", port: 0).wait()
      defer {
        XCTAssertNoThrow(try listener.close().wait())
      }

      let ourSyncQueue = DispatchQueue(label: "ourSyncQueue")

      let workFuture = NIOTSConnectionBootstrap(group: self.group).connect(
        to: listener.localAddress!
      ).map { channel -> Channel in
        XCTAssertTrue(listener.eventLoop.inEventLoop)

        ourSyncQueue.sync {
          XCTAssertFalse(listener.eventLoop.inEventLoop)

          // These will crash before we apply our fix.
          XCTAssertNotNil(listener.localAddress)
          XCTAssertNil(listener.remoteAddress)
        }

        return channel
      }.flatMap { $0.close() }

      XCTAssertNoThrow(try workFuture.wait())
    }
  }
#endif
