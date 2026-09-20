import Foundation
import NNC

#if canImport(Darwin)
  import Darwin
#elseif canImport(Glibc)
  import Glibc
#endif

/// Hash text n-grams on CPU using the exact exported tokenizer map and integer
/// constants. `previousTokens` holds up to the preceding three original token
/// IDs; keep the returned history with the KV state for continuation/rollback.
/// The returned tensor is [tokens, Engram layers, n-gram/head columns].
public func DeepSeek4_1EngramHash(
  tokens: [Int32], previousTokens: [Int32] = [], constants: DeepSeek4_1EngramHashConstants,
  configuration: DeepSeek4_1ModelConfiguration = .deepSeekV4_1Flash
) throws -> (hashes: Tensor<Int32>, history: [Int32]) {
  let ngram = configuration.engramMaxNgramSize
  let layers = configuration.engramLayers.count
  let heads = configuration.engramHeads
  let columns = (ngram - 1) * heads
  try constants.validate(configuration: configuration)
  guard !tokens.isEmpty else { throw DeepSeek4_1EngramError.invalidHashConstants }
  guard previousTokens.count < ngram else { throw DeepSeek4_1EngramError.invalidHistory }
  let combined = previousTokens + tokens
  let compressed: [UInt64] = try combined.map { token in
    guard token >= 0 && Int(token) < constants.tokenMap.count else {
      throw DeepSeek4_1EngramError.invalidToken(token)
    }
    return UInt64(constants.tokenMap[Int(token)])
  }
  let pad = UInt64(constants.tokenMap[configuration.engramPadID])
  var values = [Int32](repeating: 0, count: tokens.count * layers * columns)
  for token in 0..<tokens.count {
    let position = previousTokens.count + token
    for layer in 0..<layers {
      var rolling = compressed[position] * constants.multipliers[layer][0]
      for lookback in 1..<ngram {
        let previous = position >= lookback ? compressed[position - lookback] : pad
        rolling ^= previous * constants.multipliers[layer][lookback]
        for head in 0..<heads {
          let column = (lookback - 1) * heads + head
          let row =
            rolling % constants.primes[layer][lookback - 1][head] + constants.offsets[layer][column]
          values[(token * layers + layer) * columns + column] = Int32(row)
        }
      }
    }
  }
  return (
    Tensor(values, .CPU, .HWC(tokens.count, layers, columns)), Array(combined.suffix(ngram - 1))
  )
}

/// Reads only requested FP8 rows and their E8M0/32 scales. Regions can point to
/// original safetensors payloads or copied packed table files. No full-table
/// mapping or BF16 expansion is required. Tables are immutable for this reader's
/// lifetime; a bounded cache retains decoded rows across prefill and decode calls.
public final class DeepSeek4_1EngramTable {
  public let descriptor: DeepSeek4_1EngramTableDescriptor
  private let values: FileHandle
  private let scales: FileHandle
  private var cachedRows = [Int32: [BFloat16]]()
  private var cacheOrder = [Int32]()
  private var nextCacheIndex = 0
  private let cacheCapacity: Int

  public init(
    directory: URL, descriptor: DeepSeek4_1EngramTableDescriptor, cacheCapacity: Int = 16_384
  ) throws {
    guard descriptor.rows > 0 && descriptor.rows <= Int(Int32.max),
      descriptor.width == 256, cacheCapacity > 0,
      descriptor.values.offset <= UInt64(Int64.max), descriptor.scales.offset <= UInt64(Int64.max)
    else { throw DeepSeek4_1EngramError.invalidTableDescriptor }
    self.descriptor = descriptor
    self.cacheCapacity = cacheCapacity
    values = try FileHandle(
      forReadingFrom: URL(fileURLWithPath: descriptor.values.file, relativeTo: directory))
    scales = try FileHandle(
      forReadingFrom: URL(fileURLWithPath: descriptor.scales.file, relativeTo: directory))
    let valuesSize = try values.seekToEnd()
    let scalesSize = try scales.seekToEnd()
    guard descriptor.values.offset <= valuesSize, descriptor.scales.offset <= scalesSize,
      UInt64(descriptor.rows) * UInt64(descriptor.width) <= valuesSize - descriptor.values.offset,
      UInt64(descriptor.rows) * UInt64(descriptor.width / 32) <= scalesSize
        - descriptor.scales.offset
    else { throw DeepSeek4_1EngramError.invalidTableDescriptor }
  }

  deinit {
    try? values.close()
    try? scales.close()
  }

  private func read(_ file: FileHandle, offset: UInt64, count: Int) throws -> [UInt8] {
    var bytes = [UInt8](repeating: 0, count: count)
    try bytes.withUnsafeMutableBytes { buffer in
      var readCount = 0
      while readCount < count {
        let result = pread(
          file.fileDescriptor, buffer.baseAddress!.advanced(by: readCount),
          count - readCount, off_t(offset + UInt64(readCount)))
        if result < 0 && errno == EINTR { continue }
        guard result > 0 else {
          throw DeepSeek4_1EngramError.tableReadFailed(result < 0 ? errno : 0)
        }
        readCount += result
      }
    }
    return bytes
  }

  public func read(rows: [Int32]) throws -> Tensor<BFloat16> {
    guard !rows.isEmpty else { throw DeepSeek4_1EngramError.invalidTableDescriptor }
    let width = descriptor.width
    var decoded = [Int32: [BFloat16]]()
    for row in rows {
      guard row >= 0 && Int(row) < descriptor.rows else {
        throw DeepSeek4_1EngramError.invalidRow(row)
      }
      if decoded[row] != nil { continue }
      if let cached = cachedRows[row] {
        decoded[row] = cached
        continue
      }
      let packed = try read(
        values, offset: descriptor.values.offset + UInt64(row) * UInt64(width), count: width)
      let scaleBytes = try read(
        scales, offset: descriptor.scales.offset + UInt64(row) * UInt64(width / 32),
        count: width / 32)
      var output = [BFloat16](repeating: 0, count: width)
      for column in 0..<width {
        let byte = packed[column]
        let exponent = Int((byte >> 3) & 15)
        let mantissa = Int(byte & 7)
        let scale = scaleBytes[column / 32]
        guard !(exponent == 15 && mantissa == 7), scale != 255 else {
          throw DeepSeek4_1EngramError.invalidTableValue(row)
        }
        let value = Float(
          sign: byte < 128 ? .plus : .minus,
          exponent: (exponent == 0 ? -6 : exponent - 7) + Int(scale) - 127,
          significand: (exponent == 0 ? 0 : 1) + Float(mantissa) / 8)
        let rounded = BFloat16(value)
        guard rounded.floatValue.isFinite else {
          throw DeepSeek4_1EngramError.invalidTableValue(row)
        }
        output[column] = rounded
      }
      decoded[row] = output
      if cacheOrder.count == cacheCapacity {
        cachedRows.removeValue(forKey: cacheOrder[nextCacheIndex])
        cacheOrder[nextCacheIndex] = row
        nextCacheIndex = (nextCacheIndex + 1) % cacheCapacity
      } else {
        cacheOrder.append(row)
      }
      cachedRows[row] = output
    }
    return Tensor(rows.flatMap { decoded[$0]! }, .CPU, .WC(rows.count, width))
  }
}
