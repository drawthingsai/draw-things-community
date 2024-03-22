import C_ccv
import Fickling
import Foundation
import NNC

public final class SafeTensors {
  private var data: Data
  private let bufferStart: Int
  public let states: [String: TensorDescriptor]
  public init?(url: URL) {
    guard let data = try? Data(contentsOf: url, options: .mappedIfSafe) else { return nil }
    guard data.count >= 8 else { return nil }
    let headerSize = data.withUnsafeBytes { $0.load(as: UInt64.self) }
    // It doesn't make sense for my use-case has more than 10MiB header.
    guard headerSize > 0 && headerSize < data.count + 8 && headerSize < 1_024 * 1_024 * 10 else {
      return nil
    }
    guard
      let jsonDict = try? JSONSerialization.jsonObject(
        with: data[8..<(8 + headerSize)], options: .topLevelDictionaryAssumed) as? [String: Any]
    else { return nil }
    var states = [String: TensorDescriptor]()
    for (key, value) in jsonDict {
      guard let value = value as? [String: Any], let offsets = value["data_offsets"] as? [Int],
        let dtype = (value["dtype"] as? String)?.lowercased(), var shape = value["shape"] as? [Int],
        offsets.count == 2 && shape.count >= 0
      else { continue }
      let offsetStart = offsets[0]
      let offsetEnd = offsets[1]
      guard offsetEnd > offsetStart && offsetEnd <= data.count else {
        continue
      }
      if shape.count == 0 {
        shape = [1]
      }
      guard !(shape.contains { $0 <= 0 }) else {
        continue
      }
      guard
        dtype == "f32" || dtype == "f16" || dtype == "float16" || dtype == "float32"
          || dtype == "float" || dtype == "half" || dtype == "float64" || dtype == "f64"
          || dtype == "double" || dtype == "bf16" || dtype == "bfloat16"
      else {
        continue
      }
      let BF16 = (dtype == "bf16" || dtype == "bfloat16")
      let dataType: DataType
      if dtype == "f32" || dtype == "float32" || dtype == "float" {
        dataType = .Float32
      } else if dtype == "f64" || dtype == "float64" || dtype == "double" {
        dataType = .Float64
      } else {
        dataType = .Float16
      }
      var strides = [Int]()
      var v = 1
      for i in stride(from: shape.count - 1, through: 0, by: -1) {
        strides.append(v)
        v *= shape[i]
      }
      strides.reverse()
      let tensorDescriptor = TensorDescriptor(
        storage: Storage(name: key, size: offsetEnd - offsetStart, dataType: dataType, BF16: BF16),
        storageOffset: offsetStart, shape: shape, strides: strides)
      states[key] = tensorDescriptor
    }
    self.data = data
    self.states = states
    bufferStart = 8 + Int(headerSize)
  }
}

extension SafeTensors: TensorArchive {
  public func with<T>(_ tensorDescriptor: TensorDescriptor, block: (AnyTensor) throws -> T) throws
    -> T
  {
    // Don't subrange data, otherwise it will materialize the data into memory. Accessing the underlying
    // bytes directly, this way, it is just the mmap bytes, and we won't cause spike in memory usage.
    return try data.withUnsafeMutableBytes {
      guard let address = $0.baseAddress else { throw InflateError.dataNoBaseAddress }
      if Interpreter.inflateInterrupter?() ?? false {
        throw InflateError.interrupted
      }
      let tensor: AnyTensor
      #if !((os(macOS) || (os(iOS) && targetEnvironment(macCatalyst))) && (arch(i386) || arch(x86_64)))
        if tensorDescriptor.storage.dataType == .Float16 {
          if tensorDescriptor.storage.BF16 {
            let count = tensorDescriptor.strides[0] * tensorDescriptor.shape[0]
            let u16 = UnsafeMutablePointer<UInt16>.allocate(capacity: count * 2)
            let bf16 = (address + bufferStart + tensorDescriptor.storageOffset).assumingMemoryBound(
              to: UInt16.self)
            for i in 0..<count {
              u16[i * 2] = 0
              u16[i * 2 + 1] = bf16[i]
            }
            tensor = Tensor<Float>(
              .CPU, format: .NCHW, shape: TensorShape(tensorDescriptor.shape),
              unsafeMutablePointer: UnsafeMutableRawPointer(u16).assumingMemoryBound(
                to: Float.self), bindLifetimeOf: self
            ).copied()
            u16.deallocate()
          } else {
            tensor = Tensor<Float16>(
              .CPU, format: .NCHW, shape: TensorShape(tensorDescriptor.shape),
              unsafeMutablePointer: (address + bufferStart + tensorDescriptor.storageOffset)
                .assumingMemoryBound(
                  to: Float16.self), bindLifetimeOf: self
            ).copied()
          }
        } else if tensorDescriptor.storage.dataType == .Float64 {
          tensor = Tensor<Double>(
            .CPU, format: .NCHW, shape: TensorShape(tensorDescriptor.shape),
            unsafeMutablePointer: (address + bufferStart + tensorDescriptor.storageOffset)
              .assumingMemoryBound(
                to: Double.self), bindLifetimeOf: self
          )
        } else {
          tensor = Tensor<Float>(
            .CPU, format: .NCHW, shape: TensorShape(tensorDescriptor.shape),
            unsafeMutablePointer: (address + bufferStart + tensorDescriptor.storageOffset)
              .assumingMemoryBound(
                to: Float.self), bindLifetimeOf: self
          )
        }
      #else
        if tensorDescriptor.storage.dataType == .Float16 {
          let count = tensorDescriptor.strides[0] * tensorDescriptor.shape[0]
          let u16 = UnsafeMutablePointer<UInt16>.allocate(capacity: count * 2)
          let f16 = (address + bufferStart + tensorDescriptor.storageOffset).assumingMemoryBound(
            to: UInt16.self)
          if tensorDescriptor.storage.BF16 {
            for i in 0..<count {
              u16[i * 2] = 0
              u16[i * 2 + 1] = f16[i]
            }
          } else {
            ccv_half_precision_to_float(
              f16, UnsafeMutableRawPointer(u16).assumingMemoryBound(to: Float.self), count)
          }
          tensor = Tensor<Float>(
            .CPU, format: .NCHW, shape: TensorShape(tensorDescriptor.shape),
            unsafeMutablePointer: UnsafeMutableRawPointer(u16).assumingMemoryBound(
              to: Float.self), bindLifetimeOf: self
          ).copied()
          u16.deallocate()
        } else if tensorDescriptor.storage.dataType == .Float64 {
          tensor = Tensor<Double>(
            .CPU, format: .NCHW, shape: TensorShape(tensorDescriptor.shape),
            unsafeMutablePointer: (address + bufferStart + tensorDescriptor.storageOffset)
              .assumingMemoryBound(
                to: Double.self), bindLifetimeOf: self
          )
        } else {
          tensor = Tensor<Float>(
            .CPU, format: .NCHW, shape: TensorShape(tensorDescriptor.shape),
            unsafeMutablePointer: (address + bufferStart + tensorDescriptor.storageOffset)
              .assumingMemoryBound(
                to: Float.self), bindLifetimeOf: self
          ).copied()
        }
      #endif
      return try block(tensor)
    }
  }
}
