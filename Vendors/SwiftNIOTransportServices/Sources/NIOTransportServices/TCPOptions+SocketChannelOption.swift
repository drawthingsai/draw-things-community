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
  import Network

  @available(OSX 10.14, iOS 12.0, tvOS 12.0, watchOS 6.0, *)
  extension NWProtocolTCP.Options {
    /// Apply a given channel `SocketOption` to this protocol options state.
    func applyChannelOption(option: ChannelOptions.Types.SocketOption, value: SocketOptionValue)
      throws
    {
      switch (option.level, option.name) {
      case (IPPROTO_TCP, TCP_NODELAY):
        self.noDelay = value != 0
      case (IPPROTO_TCP, TCP_NOPUSH):
        self.noPush = value != 0
      case (IPPROTO_TCP, TCP_NOOPT):
        self.noOptions = value != 0
      case (IPPROTO_TCP, TCP_KEEPCNT):
        self.keepaliveCount = Int(value)
      case (IPPROTO_TCP, TCP_KEEPALIVE):
        self.keepaliveIdle = Int(value)
      case (IPPROTO_TCP, TCP_KEEPINTVL):
        self.keepaliveInterval = Int(value)
      case (IPPROTO_TCP, TCP_MAXSEG):
        self.maximumSegmentSize = Int(value)
      case (IPPROTO_TCP, TCP_CONNECTIONTIMEOUT):
        self.connectionTimeout = Int(value)
      case (IPPROTO_TCP, TCP_RXT_CONNDROPTIME):
        self.connectionDropTime = Int(value)
      case (IPPROTO_TCP, TCP_RXT_FINDROP):
        self.retransmitFinDrop = value != 0
      case (IPPROTO_TCP, TCP_SENDMOREACKS):
        self.disableAckStretching = value != 0
      case (SOL_SOCKET, SO_KEEPALIVE):
        self.enableKeepalive = value != 0
      default:
        throw NIOTSErrors.UnsupportedSocketOption(optionValue: option)
      }
    }

    /// Obtain the given `SocketOption` value for this protocol options state.
    func valueFor(socketOption option: ChannelOptions.Types.SocketOption) throws
      -> SocketOptionValue
    {
      switch (option.level, option.name) {
      case (IPPROTO_TCP, TCP_NODELAY):
        return self.noDelay ? 1 : 0
      case (IPPROTO_TCP, TCP_NOPUSH):
        return self.noPush ? 1 : 0
      case (IPPROTO_TCP, TCP_NOOPT):
        return self.noOptions ? 1 : 0
      case (IPPROTO_TCP, TCP_KEEPCNT):
        return Int32(self.keepaliveCount)
      case (IPPROTO_TCP, TCP_KEEPALIVE):
        return Int32(self.keepaliveIdle)
      case (IPPROTO_TCP, TCP_KEEPINTVL):
        return Int32(self.keepaliveInterval)
      case (IPPROTO_TCP, TCP_MAXSEG):
        return Int32(self.maximumSegmentSize)
      case (IPPROTO_TCP, TCP_CONNECTIONTIMEOUT):
        return Int32(self.connectionTimeout)
      case (IPPROTO_TCP, TCP_RXT_CONNDROPTIME):
        return Int32(self.connectionDropTime)
      case (IPPROTO_TCP, TCP_RXT_FINDROP):
        return self.retransmitFinDrop ? 1 : 0
      case (IPPROTO_TCP, TCP_SENDMOREACKS):
        return self.disableAckStretching ? 1 : 0
      case (SOL_SOCKET, SO_KEEPALIVE):
        return self.enableKeepalive ? 1 : 0
      default:
        throw NIOTSErrors.UnsupportedSocketOption(optionValue: option)
      }
    }
  }
#endif
