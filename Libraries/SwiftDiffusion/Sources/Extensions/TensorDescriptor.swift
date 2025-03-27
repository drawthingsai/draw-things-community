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
  var FP8: Bool
}

public struct TensorDescriptor {
  public var storage: Storage
  public var storageOffset: Int
  public var shape: [Int]
  public var strides: [Int]
}

extension TensorDescriptor {
  public func inflate<T: TensorNumeric>(from archive: TensorArchive, of type: T.Type) throws
    -> Tensor<T>
  {
    return try archive.with(self) {
      Tensor<T>(from: $0).copied()
    }
  }
}

extension ModelWeightElement {
  public func write<FloatType: BinaryFloatingPoint & TensorNumeric>(
    graph: DynamicGraph, to store: DynamicGraph.Store, tensor: Tensor<FloatType>, format: Format,
    isDiagonalUp: Bool, isDiagonalDown: Bool, renamer: (String) -> String
  ) {
    var tensor = tensor
    if scale != 1 {
      // Scale the tensor if needed.
      tensor = graph.withNoGrad {
        Tensor<FloatType>(
          from: (scale * graph.variable(Tensor<Float>(from: tensor).toCPU())).rawValue)
      }
    }
    switch format {
    case .O:
      let squeezedDim = tensor.shape.compactMap { $0 == 1 ? nil : $0 }
      tensor = tensor.reshaped(format: tensor.format, shape: TensorShape(squeezedDim))
      let shape = tensor.shape
      if self.count > 1 {
        switch self.format {
        case .O:
          let count = shape[0] / self.count
          if isDiagonalUp {
            let jCount = shape[1] / self.count
            if let offsets = offsets {
              for (i, name) in self.enumerated() {
                store.write(
                  renamer(name),
                  tensor: tensor[
                    (offsets[i])..<(i < offsets.count - 1 ? offsets[i + 1] : shape[0]),
                    (i * jCount)..<((i + 1) * jCount)
                  ].copied())
              }
            } else {
              for (i, name) in self.enumerated() {
                store.write(
                  renamer(name),
                  tensor: tensor[
                    (i * count)..<((i + 1) * count), (i * jCount)..<((i + 1) * jCount)
                  ].copied())
              }
            }
          } else {
            if let offsets = offsets {
              for (i, name) in self.enumerated() {
                if shape.count > 1 {
                  store.write(
                    renamer(name),
                    tensor: tensor[
                      (offsets[i])..<(i < offsets.count - 1 ? offsets[i + 1] : shape[0]),
                      0..<shape[1]
                    ].copied())
                } else {
                  store.write(
                    renamer(name),
                    tensor: tensor[
                      (offsets[i])..<(i < offsets.count - 1 ? offsets[i + 1] : shape[0])
                    ].copied())
                }
              }
            } else {
              for (i, name) in self.enumerated() {
                if shape.count > 1 {
                  store.write(
                    renamer(name),
                    tensor: tensor[(i * count)..<((i + 1) * count), 0..<shape[1]]
                      .copied())
                } else {
                  store.write(
                    renamer(name),
                    tensor: tensor[(i * count)..<((i + 1) * count)]
                      .copied())
                }
              }
            }
          }
        case .I:
          if isDiagonalDown {
            let jCount = shape[1] / self.count
            for (i, name) in self.enumerated() {
              store.write(
                renamer(name),
                tensor: tensor[0..<shape[0], (i * jCount)..<((i + 1) * jCount)].copied())
            }
          } else {
            for name in self {
              store.write(renamer(name), tensor: tensor)
            }
          }
        }
      } else {
        for name in self {
          store.write(renamer(name), tensor: tensor)
        }
      }
    case .I:
      if self.count > 1 {
        let shape = tensor.shape
        switch self.format {
        case .I:
          if isDiagonalDown {
            let iCount = shape[0] / self.count
            if let offsets = offsets {
              for (i, name) in self.enumerated() {
                store.write(
                  renamer(name),
                  tensor: tensor[
                    (i * iCount)..<((i + 1) * iCount),
                    offsets[i]..<(i < offsets.count - 1 ? offsets[i + 1] : shape[1])
                  ].copied())
              }
            } else {
              let count = shape[1] / self.count
              for (i, name) in self.enumerated() {
                store.write(
                  renamer(name),
                  tensor: tensor[(i * iCount)..<((i + 1) * iCount), (i * count)..<((i + 1) * count)]
                    .copied())
              }
            }
          } else {
            if let offsets = offsets {
              for (i, name) in self.enumerated() {
                if shape.count > 1 {
                  store.write(
                    renamer(name),
                    tensor: tensor[
                      0..<shape[0],
                      offsets[i]..<(i < offsets.count - 1 ? offsets[i + 1] : shape[1])
                    ].copied())
                } else {
                  store.write(
                    renamer(name),
                    tensor: tensor[
                      offsets[i]..<(i < offsets.count - 1 ? offsets[i + 1] : shape[0])
                    ].copied())
                }
              }
            } else {
              let count = shape.count > 1 ? shape[1] / self.count : shape[0] / self.count
              for (i, name) in self.enumerated() {
                if shape.count > 1 {
                  store.write(
                    renamer(name),
                    tensor: tensor[0..<shape[0], (i * count)..<((i + 1) * count)].copied())
                } else {
                  store.write(
                    renamer(name),
                    tensor: tensor[(i * count)..<((i + 1) * count)].copied())
                }
              }
            }
          }
        case .O:
          if isDiagonalUp {
            let count = shape[0] / self.count
            if shape.count == 2 {
              for (i, name) in self.enumerated() {
                store.write(
                  renamer(name),
                  tensor: tensor[(i * count)..<((i + 1) * count), 0..<shape[1]].copied())
              }
            } else if shape.count == 4 {
              for (i, name) in self.enumerated() {
                store.write(
                  renamer(name),
                  tensor: tensor[
                    (i * count)..<((i + 1) * count), 0..<shape[1], 0..<shape[2],
                    0..<shape[3]
                  ].copied())
              }
            } else {
              for (i, name) in self.enumerated() {
                store.write(
                  renamer(name),
                  tensor: tensor[(i * count)..<((i + 1) * count)].copied())
              }
            }
          } else {
            for name in self {
              store.write(renamer(name), tensor: tensor)
            }
          }
        }
      } else {
        for name in self {
          store.write(renamer(name), tensor: tensor)
        }
      }
    }
  }
}
