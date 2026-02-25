public enum ModelWeightFormat {
  case diffusers
  case generativeModels
}

public struct ModelWeightElement: RandomAccessCollection, ExpressibleByArrayLiteral {
  public enum Format {
    case O
    case I
  }
  public let format: Format
  public let offsets: [Int]?
  public let scale: Float
  public let index: Int
  public let isBF16: Bool
  private let underlyingArray: [String]

  public typealias Element = String
  public typealias Index = Int
  public typealias Indices = Range<Index>
  public typealias SubSequence = Array<Element>.SubSequence
  public var endIndex: Index { underlyingArray.endIndex }
  public var indices: Indices { underlyingArray.indices }
  public var startIndex: Index { underlyingArray.startIndex }
  public func formIndex(after i: inout Index) { underlyingArray.formIndex(after: &i) }
  public func formIndex(before i: inout Index) { underlyingArray.formIndex(before: &i) }
  public subscript(position: Index) -> Element { underlyingArray[position] }
  public subscript(x: Indices) -> SubSequence { underlyingArray[x] }

  public init(
    _ array: [Element], format: Format = .O, offsets: [Int]? = nil, scale: Float = 1,
    index: Int = 0, isBF16: Bool = false
  ) {
    self.underlyingArray = array
    self.format = format
    self.offsets = offsets
    self.scale = scale
    self.index = index
    self.isBF16 = isBF16
  }

  public init(arrayLiteral elements: Element...) {
    self.underlyingArray = elements
    self.format = .O
    self.offsets = nil
    self.scale = 1
    self.index = 0
    self.isBF16 = false
  }
}

public typealias ModelWeightMapping = [String: ModelWeightElement]

public typealias ModelWeightMapper = (ModelWeightFormat) -> ModelWeightMapping
