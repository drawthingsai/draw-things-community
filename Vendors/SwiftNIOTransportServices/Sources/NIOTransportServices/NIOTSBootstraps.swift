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
  import NIOCore

  /// Shared functionality across NIOTS bootstraps.
  internal enum NIOTSBootstraps {
    @available(OSX 10.14, iOS 12.0, tvOS 12.0, watchOS 6.0, *)
    internal static func isCompatible(group: EventLoopGroup) -> Bool {
      return group is NIOTSEventLoop || group is NIOTSEventLoopGroup
    }
  }

#endif
