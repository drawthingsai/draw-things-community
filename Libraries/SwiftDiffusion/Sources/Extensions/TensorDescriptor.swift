import Fickling
import Foundation
import NNC
import ZIPFoundation

public enum InflateError: Error {
  case tensorNotFound
  case dataNoBaseAddress
  case interrupted
}

public struct Storage {
  var name: String
  var size: Int
  var dataType: DataType
  var BF16: Bool
}

public struct TensorDescriptor {
  public var storage: Storage
  public var storageOffset: Int
  public var shape: [Int]
  public var strides: [Int]
}

extension TensorDescriptor {
  func inflate<T: TensorNumeric>(from archive: TensorArchive, of type: T.Type) throws -> Tensor<T> {
    return try archive.with(self) {
      Tensor<T>(from: $0).copied()
    }
  }
}
