import NNC

public protocol TensorArchive {
  func with<T>(_: TensorDescriptor, block: (AnyTensor) throws -> T) throws -> T
}
