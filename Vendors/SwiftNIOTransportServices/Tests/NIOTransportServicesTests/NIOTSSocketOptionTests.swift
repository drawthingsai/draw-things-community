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
  import NIOCore
  import Network
  @testable import NIOTransportServices

  @available(OSX 10.14, iOS 12.0, tvOS 12.0, *)
  class NIOTSSocketOptionTests: XCTestCase {
    private var options: NWProtocolTCP.Options!

    override func setUp() {
      self.options = NWProtocolTCP.Options()
    }

    override func tearDown() {
      self.options = nil
    }

    private func assertProperty<T: Equatable>(
      called path: KeyPath<NWProtocolTCP.Options, T>,
      correspondsTo socketOption: ChannelOptions.Types.SocketOption,
      defaultsTo defaultValue: T,
      and defaultSocketOptionValue: SocketOptionValue,
      canBeSetTo unusualValue: SocketOptionValue,
      whichLeadsTo newInnerValue: T
    ) throws {
      // Confirm the default is right.
      let actualDefaultSocketOptionValue = try self.options.valueFor(socketOption: socketOption)
      XCTAssertEqual(self.options[keyPath: path], defaultValue)
      XCTAssertEqual(actualDefaultSocketOptionValue, defaultSocketOptionValue)

      // Confirm that we can set this to a new value, and that it leads to the right outcome.
      try self.options.applyChannelOption(option: socketOption, value: unusualValue)
      XCTAssertEqual(self.options[keyPath: path], newInnerValue)
      XCTAssertEqual(try self.options.valueFor(socketOption: socketOption), unusualValue)

      // And confirm that we can set it back to the default.
      try self.options.applyChannelOption(
        option: socketOption, value: actualDefaultSocketOptionValue)
      XCTAssertEqual(self.options[keyPath: path], defaultValue)
      XCTAssertEqual(actualDefaultSocketOptionValue, defaultSocketOptionValue)
    }

    func testReadingAndSettingNoDelay() throws {
      try self.assertProperty(
        called: \NWProtocolTCP.Options.noDelay,
        correspondsTo: ChannelOptions.Types.SocketOption(level: IPPROTO_TCP, name: TCP_NODELAY),
        defaultsTo: false, and: 0,
        canBeSetTo: 1, whichLeadsTo: true)
    }

    func testReadingAndSettingNoPush() throws {
      try self.assertProperty(
        called: \NWProtocolTCP.Options.noPush,
        correspondsTo: ChannelOptions.Types.SocketOption(level: IPPROTO_TCP, name: TCP_NOPUSH),
        defaultsTo: false, and: 0,
        canBeSetTo: 1, whichLeadsTo: true)
    }

    func testReadingAndSettingNoOpt() throws {
      try self.assertProperty(
        called: \NWProtocolTCP.Options.noOptions,
        correspondsTo: ChannelOptions.Types.SocketOption(level: IPPROTO_TCP, name: TCP_NOOPT),
        defaultsTo: false, and: 0,
        canBeSetTo: 1, whichLeadsTo: true)
    }

    func testReadingAndSettingKeepaliveCount() throws {
      try self.assertProperty(
        called: \NWProtocolTCP.Options.keepaliveCount,
        correspondsTo: ChannelOptions.Types.SocketOption(level: IPPROTO_TCP, name: TCP_KEEPCNT),
        defaultsTo: 0, and: 0,
        canBeSetTo: 5, whichLeadsTo: 5)
    }

    func testReadingAndSettingKeepaliveIdle() throws {
      try self.assertProperty(
        called: \NWProtocolTCP.Options.keepaliveIdle,
        correspondsTo: ChannelOptions.Types.SocketOption(level: IPPROTO_TCP, name: TCP_KEEPALIVE),
        defaultsTo: 0, and: 0,
        canBeSetTo: 5, whichLeadsTo: 5)
    }

    func testReadingAndSettingKeepaliveInterval() throws {
      try self.assertProperty(
        called: \NWProtocolTCP.Options.keepaliveInterval,
        correspondsTo: ChannelOptions.Types.SocketOption(level: IPPROTO_TCP, name: TCP_KEEPINTVL),
        defaultsTo: 0, and: 0,
        canBeSetTo: 5, whichLeadsTo: 5)
    }

    func testReadingAndSettingMaxSeg() throws {
      try self.assertProperty(
        called: \NWProtocolTCP.Options.maximumSegmentSize,
        correspondsTo: ChannelOptions.Types.SocketOption(level: IPPROTO_TCP, name: TCP_MAXSEG),
        defaultsTo: 0, and: 0,
        canBeSetTo: 5, whichLeadsTo: 5)
    }

    func testReadingAndSettingConnectTimeout() throws {
      try self.assertProperty(
        called: \NWProtocolTCP.Options.connectionTimeout,
        correspondsTo: ChannelOptions.Types.SocketOption(
          level: IPPROTO_TCP, name: TCP_CONNECTIONTIMEOUT),
        defaultsTo: 0, and: 0,
        canBeSetTo: 5, whichLeadsTo: 5)
    }

    func testReadingAndSettingConnectDropTime() throws {
      try self.assertProperty(
        called: \NWProtocolTCP.Options.connectionDropTime,
        correspondsTo: ChannelOptions.Types.SocketOption(
          level: IPPROTO_TCP, name: TCP_RXT_CONNDROPTIME),
        defaultsTo: 0, and: 0,
        canBeSetTo: 5, whichLeadsTo: 5)
    }

    func testReadingAndSettingFinDrop() throws {
      try self.assertProperty(
        called: \NWProtocolTCP.Options.retransmitFinDrop,
        correspondsTo: ChannelOptions.Types.SocketOption(level: IPPROTO_TCP, name: TCP_RXT_FINDROP),
        defaultsTo: false, and: 0,
        canBeSetTo: 1, whichLeadsTo: true)
    }

    func testReadingAndSettingAckStretching() throws {
      try self.assertProperty(
        called: \NWProtocolTCP.Options.disableAckStretching,
        correspondsTo: ChannelOptions.Types.SocketOption(
          level: IPPROTO_TCP, name: TCP_SENDMOREACKS),
        defaultsTo: false, and: 0,
        canBeSetTo: 1, whichLeadsTo: true)
    }

    func testReadingAndSettingKeepalive() throws {
      try self.assertProperty(
        called: \NWProtocolTCP.Options.enableKeepalive,
        correspondsTo: ChannelOptions.Types.SocketOption(level: SOL_SOCKET, name: SO_KEEPALIVE),
        defaultsTo: false, and: 0,
        canBeSetTo: 1, whichLeadsTo: true)
    }

    func testWritingNonexistentSocketOption() {
      let option = ChannelOptions.Types.SocketOption(level: Int32.max, name: Int32.max)

      do {
        try self.options.applyChannelOption(option: option, value: 0)
      } catch let err as NIOTSErrors.UnsupportedSocketOption {
        XCTAssertEqual(err.optionValue.level, Int32.max)
        XCTAssertEqual(err.optionValue.name, Int32.max)
      } catch {
        XCTFail("Unexpected error \(error)")
      }
    }

    func testReadingNonexistentSocketOption() {
      let option = ChannelOptions.Types.SocketOption(level: Int32.max, name: Int32.max)

      do {
        _ = try self.options.valueFor(socketOption: option)
      } catch let err as NIOTSErrors.UnsupportedSocketOption {
        XCTAssertEqual(err.optionValue.level, Int32.max)
        XCTAssertEqual(err.optionValue.name, Int32.max)
      } catch {
        XCTFail("Unexpected error \(error)")
      }
    }
  }
#endif
