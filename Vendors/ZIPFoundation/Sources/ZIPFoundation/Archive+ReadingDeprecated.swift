//
//  Archive+ReadingDeprecated.swift
//  ZIPFoundation
//
//  Copyright © 2017-2021 Thomas Zoechling, https://www.peakstep.com and the ZIP Foundation project authors.
//  Released under the MIT License.
//
//  See https://github.com/weichsel/ZIPFoundation/blob/master/LICENSE for license information.
//

import Foundation

extension Archive {

  @available(
    *, deprecated,
    message: "Please use `Int` for `bufferSize`."
  )
  public func extract(
    _ entry: Entry, to url: URL, bufferSize: UInt32, skipCRC32: Bool = false,
    progress: Progress? = nil
  ) throws -> CRC32 {
    try self.extract(
      entry, to: url, bufferSize: Int(bufferSize), skipCRC32: skipCRC32, progress: progress)
  }

  @available(
    *, deprecated,
    message: "Please use `Int` for `bufferSize`."
  )
  public func extract(
    _ entry: Entry, bufferSize: UInt32, skipCRC32: Bool = false,
    progress: Progress? = nil, consumer: Consumer
  ) throws -> CRC32 {
    try self.extract(
      entry, bufferSize: Int(bufferSize), skipCRC32: skipCRC32,
      progress: progress, consumer: consumer)
  }
}
