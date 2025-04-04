//===----------------------------------------------------------------------===//
//
// This source file is part of the SwiftNIO open source project
//
// Copyright (c) 2022 Apple Inc. and the SwiftNIO project authors
// Licensed under Apache License v2.0
//
// See LICENSE.txt for license information
// See CONTRIBUTORS.txt for the list of SwiftNIO project authors
//
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#if canImport(Network)
  import Network
  import NIOCore
  import XCTest
  @testable import NIOTransportServices

  @available(OSX 10.14, iOS 12.0, tvOS 12.0, watchOS 6.0, *)
  final class NIOTSChannelMetadataTests: XCTestCase {
    func testThrowsIfCalledOnWrongChannel() throws {
      let eventLoopGroup = NIOTSEventLoopGroup()
      defer { XCTAssertNoThrow(try eventLoopGroup.syncShutdownGracefully()) }
      let listenerBootsrap = NIOTSListenerBootstrap(group: eventLoopGroup)
      let listenerChannel = try listenerBootsrap.bind(host: "localhost", port: 0).wait()
      defer { XCTAssertNoThrow(try listenerChannel.close().wait()) }

      XCTAssertThrowsError(
        try listenerChannel.getMetadata(definition: NWProtocolTLS.definition).wait()
      ) { error in
        XCTAssertTrue(
          error is NIOTSChannelIsNotANIOTSConnectionChannel, "unexpected error \(error)")
      }
      try! listenerChannel.eventLoop.submit {
        XCTAssertThrowsError(
          try listenerChannel.getMetadataSync(definition: NWProtocolTLS.definition)
        ) { error in
          XCTAssertTrue(
            error is NIOTSChannelIsNotANIOTSConnectionChannel, "unexpected error \(error)")
        }
      }.wait()

    }
    func testThowsIfCalledOnANonInitializedChannel() {
      let eventLoopGroup = NIOTSEventLoopGroup()
      defer { XCTAssertNoThrow(try eventLoopGroup.syncShutdownGracefully()) }
      let channel = NIOTSConnectionChannel(
        eventLoop: eventLoopGroup.next() as! NIOTSEventLoop, tcpOptions: .init(),
        tlsOptions: .init())
      XCTAssertThrowsError(try channel.getMetadata(definition: NWProtocolTLS.definition).wait()) {
        error in
        XCTAssertTrue(error is NIOTSConnectionNotInitialized, "unexpected error \(error)")
      }
      try! channel.eventLoop.submit {
        XCTAssertThrowsError(try channel.getMetadataSync(definition: NWProtocolTLS.definition)) {
          error in
          XCTAssertTrue(error is NIOTSConnectionNotInitialized, "unexpected error \(error)")
        }
      }.wait()
    }
  }
#endif
