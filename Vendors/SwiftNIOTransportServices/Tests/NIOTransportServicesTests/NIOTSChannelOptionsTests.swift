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
  import NIOConcurrencyHelpers
  import NIOTransportServices
  import Network

  @available(OSX 10.14, iOS 12.0, tvOS 12.0, watchOS 6.0, *)
  class NIOTSChannelOptionsTests: XCTestCase {
    private var group: NIOTSEventLoopGroup!

    override func setUp() {
      self.group = NIOTSEventLoopGroup()
    }

    override func tearDown() {
      XCTAssertNoThrow(try self.group.syncShutdownGracefully())
    }

    func testCurrentPath() throws {
      let listener = try NIOTSListenerBootstrap(group: self.group)
        .bind(host: "localhost", port: 0).wait()
      defer {
        XCTAssertNoThrow(try listener.close().wait())
      }

      let connection = try NIOTSConnectionBootstrap(group: self.group)
        .connect(to: listener.localAddress!)
        .wait()
      defer {
        XCTAssertNoThrow(try connection.close().wait())
      }

      let currentPath = try connection.getOption(NIOTSChannelOptions.currentPath).wait()
      XCTAssertEqual(currentPath.status, NWPath.Status.satisfied)
    }

    func testMetadata() throws {
      let listener = try NIOTSListenerBootstrap(group: self.group)
        .bind(host: "localhost", port: 0).wait()
      defer {
        XCTAssertNoThrow(try listener.close().wait())
      }

      let connection = try NIOTSConnectionBootstrap(group: self.group)
        .connect(to: listener.localAddress!)
        .wait()
      defer {
        XCTAssertNoThrow(try connection.close().wait())
      }

      let metadata =
        try connection.getOption(NIOTSChannelOptions.metadata(NWProtocolTCP.definition)).wait()
        as! NWProtocolTCP.Metadata
      XCTAssertEqual(metadata.availableReceiveBuffer, 0)
    }

    @available(OSX 10.15, iOS 13.0, tvOS 13.0, watchOS 6.0, *)
    func testEstablishmentReport() throws {
      let listener = try NIOTSListenerBootstrap(group: self.group)
        .bind(host: "localhost", port: 0).wait()
      defer {
        XCTAssertNoThrow(try listener.close().wait())
      }

      let connection = try NIOTSConnectionBootstrap(group: self.group)
        .connect(to: listener.localAddress!)
        .wait()
      defer {
        XCTAssertNoThrow(try connection.close().wait())
      }

      let reportFuture = try connection.getOption(NIOTSChannelOptions.establishmentReport).wait()
      let establishmentReport = try reportFuture.wait()

      XCTAssertEqual(establishmentReport!.resolutions.count, 0)
    }

    @available(OSX 10.15, iOS 13.0, tvOS 13.0, watchOS 6.0, *)
    func testDataTransferReport() throws {
      let syncQueue = DispatchQueue(label: "syncQueue")
      let collectGroup = DispatchGroup()

      let listener = try NIOTSListenerBootstrap(group: self.group)
        .bind(host: "localhost", port: 0).wait()
      defer {
        XCTAssertNoThrow(try listener.close().wait())
      }

      let connection = try NIOTSConnectionBootstrap(group: self.group)
        .connect(to: listener.localAddress!)
        .wait()
      defer {
        XCTAssertNoThrow(try connection.close().wait())
      }

      let pendingReport = try connection.getOption(NIOTSChannelOptions.dataTransferReport).wait()

      collectGroup.enter()
      pendingReport.collect(queue: syncQueue) { report in
        XCTAssertEqual(report.pathReports.count, 1)
        collectGroup.leave()
      }

      collectGroup.wait()
    }

  }
#endif
