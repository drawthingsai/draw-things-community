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
import XCTest

final class NIOTSWorkaroundTests: XCTestCase {
  func testWorkaround5695() {
    // This test works around https://github.com/apple/swift-package-manager/issues/5695 by just
    // doing nothing.
    XCTAssertTrue(true)
  }
}
