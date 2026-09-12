import Foundation

public struct DeepSeek4_1ModelConfiguration: Codable, Sendable {
  public var vocabularySize = 129_280
  public var hiddenSize = 5_120
  public var layers = 40
  public var hcCount = 4
  public var hcSinkhornIterations = 20
  public var hcEpsilon: Float = 1e-6
  public var normEpsilon: Float = 1e-20
  public var attentionHeads = 64
  public var attentionHeadDim = 512
  public var rotaryDim = 64
  public var rawWindow = 128
  public var expertCount = 384
  public var routedExperts = 6
  public var expertIntermediateSize = 2_304
  public var sharedIntermediateSize = 2_304
  public var expertResidentSlots = Array(repeating: 384, count: 40)
  public var attentionOutputGroups = 8
  public var attentionLowRank = 1_024
  public var queryLowRank = 1_280
  public var indexerHeads = 32
  public var indexerHeadDim = 128
  public var indexerTopK = 512
  public var ropeTheta: Double = 10_000
  public var compressedRopeTheta: Double = 160_000
  public var ropeScaleFactor: Double = 16
  public var ropeOriginalContext = 65_536
  public var ropeYarnBetaFast: Double = 32
  public var ropeYarnBetaSlow: Double = 1
  public var kvSourceLayers = [2, 8, 14, 20]
  public var indexSourceLayers = [2, 8, 14, 20, 24, 28, 32, 36]
  public var compressionRatios =
    [0, 0] + Array(repeating: 2, count: 18) + Array(repeating: 1, count: 20)
  public var candidateSourceLayer = 20
  public var candidateBlockSize = 8
  public var candidateTopKBlocks = 2_048
  public var engramLayers = [1, 14]
  public var engramRows = [384_006_168, 384_016_682]
  public var engramHeads = 8
  public var engramHeadDim = 256
  public var engramMaxNgramSize = 4
  public var engramPadID = 2

  public init() {}

  public static let deepSeekV4_1Flash = DeepSeek4_1ModelConfiguration()
  public var hcMixDim: Int { (2 + hcCount) * hcCount }
  public var attentionOutputLowDim: Int { attentionOutputGroups * attentionLowRank }
  public var engramEmbeddingSize: Int { (engramMaxNgramSize - 1) * engramHeads * engramHeadDim }

  public func kvSource(layerIndex: Int) -> Int? {
    precondition(layerIndex >= 0 && layerIndex < layers)
    return kvSourceLayers.last { $0 <= layerIndex }
  }

  public func indexSource(layerIndex: Int) -> Int? {
    precondition(layerIndex >= 0 && layerIndex < layers)
    return indexSourceLayers.last { $0 <= layerIndex }
  }
}

public struct DeepSeek4_1EngramHashConstants: Codable, Sendable {
  public var tokenMap: [Int32]
  public var multipliers: [[UInt64]]
  public var primes: [[[UInt64]]]
  public var offsets: [[UInt64]]

  public func validate(
    configuration: DeepSeek4_1ModelConfiguration = .deepSeekV4_1Flash
  ) throws {
    let ngram = configuration.engramMaxNgramSize
    let layers = configuration.engramLayers.count
    let heads = configuration.engramHeads
    let columns = (ngram - 1) * heads
    guard ngram > 1, heads > 0, layers > 0,
      tokenMap.count == configuration.vocabularySize,
      configuration.engramPadID >= 0 && configuration.engramPadID < tokenMap.count,
      tokenMap.allSatisfy({ $0 >= 0 }),
      multipliers.count == layers, primes.count == layers,
      offsets.count == layers, configuration.engramRows.count == layers
    else { throw DeepSeek4_1EngramError.invalidHashConstants }
    let maximumToken = UInt64(tokenMap.max()!)
    for layer in 0..<layers {
      guard multipliers[layer].count == ngram,
        primes[layer].count == ngram - 1, offsets[layer].count == columns,
        multipliers[layer].allSatisfy({
          $0 > 0 && $0 <= UInt64(Int64.max) / max(maximumToken, 1)
        })
      else { throw DeepSeek4_1EngramError.invalidHashConstants }
      var offset: UInt64 = 0
      for size in 0..<(ngram - 1) {
        guard primes[layer][size].count == heads else {
          throw DeepSeek4_1EngramError.invalidHashConstants
        }
        for head in 0..<heads {
          let prime = primes[layer][size][head]
          guard prime > 0, prime <= UInt64(Int32.max) - offset,
            offsets[layer][size * heads + head] == offset
          else { throw DeepSeek4_1EngramError.invalidHashConstants }
          offset += prime
        }
      }
      guard offset == configuration.engramRows[layer] else {
        throw DeepSeek4_1EngramError.invalidHashConstants
      }
    }
  }

  enum CodingKeys: String, CodingKey {
    case tokenMap = "token_map"
    case multipliers, primes, offsets
  }
}

public enum DeepSeek4_1EngramError: Error {
  case invalidHashConstants
  case invalidToken(Int32)
  case invalidHistory
  case invalidTableDescriptor
  case invalidRow(Int32)
  case invalidTableValue(Int32)
  case tableReadFailed(Int32)
}

public struct DeepSeek4_1EngramTableDescriptor: Codable, Sendable {
  public struct Region: Codable, Sendable {
    public var file: String
    public var offset: UInt64

    public init(file: String, offset: UInt64) {
      self.file = file
      self.offset = offset
    }
  }

  public var rows: Int
  public var width: Int
  public var values: Region
  public var scales: Region

  public init(rows: Int, width: Int = 256, values: Region, scales: Region) {
    self.rows = rows
    self.width = width
    self.values = values
    self.scales = scales
  }
}
