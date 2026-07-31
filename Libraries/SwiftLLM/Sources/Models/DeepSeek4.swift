import Foundation
import NNC

public let DeepSeek4StoreLoadCodec: DynamicGraph.Store.Codec = [
  .jit, .i8x(.iq2xxs), .ezm7, .externalData(.wholeFile),
]

private let DeepSeek4Q8DenseFlags: Functional.GEMMFlag = [.Int8, .Float32]

public enum DeepSeek4AttentionKind: Sendable, Equatable {
  case raw
  case compressed(compressionRatio: Int)
  case indexed(compressionRatio: Int)
}

public enum DeepSeek4RouterKind: Sendable, Equatable {
  case standard
  case tokenHash
}

public struct DeepSeek4ModelConfiguration: Sendable {
  public var vocabularySize: Int
  public var hiddenSize: Int
  public var layers: Int
  public var hcCount: Int
  public var attentionHeads: Int
  public var attentionHeadDim: Int
  public var rotaryDim: Int
  public var rawWindow: Int
  public var expertCount: Int
  public var routedExperts: Int
  public var expertIntermediateSize: Int
  public var sharedIntermediateSize: Int
  public var attentionOutputGroups: Int
  public var attentionLowRank: Int
  public var queryLowRank: Int
  public var indexerHeads: Int
  public var indexerHeadDim: Int
  public var indexerTopK: Int
  public var ropeTheta: Double
  public var ropeScaleFactor: Double
  public var ropeOriginalContext: Int
  public var ropeYarnBetaFast: Double
  public var ropeYarnBetaSlow: Double
  public var layerAttentionKinds: [DeepSeek4AttentionKind]
  public var layerRouterKinds: [DeepSeek4RouterKind]
  public var enableFP8KVRoundTrip: Bool

  public init(
    vocabularySize: Int, hiddenSize: Int, layers: Int, hcCount: Int,
    attentionHeads: Int, attentionHeadDim: Int, rotaryDim: Int, rawWindow: Int,
    expertCount: Int, routedExperts: Int, expertIntermediateSize: Int,
    sharedIntermediateSize: Int, attentionOutputGroups: Int, attentionLowRank: Int,
    queryLowRank: Int, indexerHeads: Int, indexerHeadDim: Int, indexerTopK: Int,
    ropeTheta: Double, ropeScaleFactor: Double, ropeOriginalContext: Int,
    ropeYarnBetaFast: Double, ropeYarnBetaSlow: Double,
    layerAttentionKinds: [DeepSeek4AttentionKind],
    layerRouterKinds: [DeepSeek4RouterKind],
    enableFP8KVRoundTrip: Bool = true
  ) {
    precondition(layerAttentionKinds.count == layers)
    precondition(layerRouterKinds.count == layers)
    self.vocabularySize = vocabularySize
    self.hiddenSize = hiddenSize
    self.layers = layers
    self.hcCount = hcCount
    self.attentionHeads = attentionHeads
    self.attentionHeadDim = attentionHeadDim
    self.rotaryDim = rotaryDim
    self.rawWindow = rawWindow
    self.expertCount = expertCount
    self.routedExperts = routedExperts
    self.expertIntermediateSize = expertIntermediateSize
    self.sharedIntermediateSize = sharedIntermediateSize
    self.attentionOutputGroups = attentionOutputGroups
    self.attentionLowRank = attentionLowRank
    self.queryLowRank = queryLowRank
    self.indexerHeads = indexerHeads
    self.indexerHeadDim = indexerHeadDim
    self.indexerTopK = indexerTopK
    self.ropeTheta = ropeTheta
    self.ropeScaleFactor = ropeScaleFactor
    self.ropeOriginalContext = ropeOriginalContext
    self.ropeYarnBetaFast = ropeYarnBetaFast
    self.ropeYarnBetaSlow = ropeYarnBetaSlow
    self.layerAttentionKinds = layerAttentionKinds
    self.layerRouterKinds = layerRouterKinds
    self.enableFP8KVRoundTrip = enableFP8KVRoundTrip
  }

  public var attentionOutputLowDim: Int { attentionOutputGroups * attentionLowRank }
  public var hcMixDim: Int { 2 * hcCount + hcCount * hcCount }

  public func attentionKind(layerIndex: Int) -> DeepSeek4AttentionKind {
    precondition(layerIndex >= 0 && layerIndex < layers)
    return layerAttentionKinds[layerIndex]
  }

  public func routerKind(layerIndex: Int) -> DeepSeek4RouterKind {
    precondition(layerIndex >= 0 && layerIndex < layers)
    return layerRouterKinds[layerIndex]
  }

  /// Compression ratios used by the configured attention layers.
  public var compressionRatios: [Int] {
    var ratios = Set<Int>()
    for attentionKind in layerAttentionKinds {
      switch attentionKind {
      case .raw:
        break
      case .compressed(let compressionRatio), .indexed(let compressionRatio):
        ratios.insert(compressionRatio)
      }
    }
    return ratios.sorted()
  }
}

extension DeepSeek4ModelConfiguration {
  public static let deepSeekV4Flash = DeepSeek4ModelConfiguration(
    vocabularySize: 129_280,
    hiddenSize: 4_096,
    layers: 43,
    hcCount: 4,
    attentionHeads: 64,
    attentionHeadDim: 512,
    rotaryDim: 64,
    rawWindow: 128,
    expertCount: 256,
    routedExperts: 6,
    expertIntermediateSize: 2_048,
    sharedIntermediateSize: 2_048,
    attentionOutputGroups: 8,
    attentionLowRank: 1_024,
    queryLowRank: 1_024,
    indexerHeads: 64,
    indexerHeadDim: 128,
    indexerTopK: 512,
    ropeTheta: 160_000,
    ropeScaleFactor: 16,
    ropeOriginalContext: 65_536,
    ropeYarnBetaFast: 32,
    ropeYarnBetaSlow: 1,
    layerAttentionKinds: [
      .raw, .raw, .indexed(compressionRatio: 4), .compressed(compressionRatio: 128),
    ]
      + (4..<43).map {
        $0.isMultiple(of: 2) ? .indexed(compressionRatio: 4) : .compressed(compressionRatio: 128)
      },
    layerRouterKinds: (0..<43).map { $0 < 3 ? .tokenHash : .standard })
}

/// Describes the compressor rows and retained state for one continuation step.
public struct DeepSeek4CompressionPlan: Sendable, Equatable {
  public let compressionRatio: Int
  public let existingRowCount: Int
  public let totalRowCount: Int
  public let emittedRowCount: Int
  public let compressorTokenOffset: Int
  public let compressorTokenCount: Int
  public let compressorOutputRowOffset: Int
  public let stateCount: Int
  public let nextStateCount: Int
  public let nextStateOffset: Int

  public init(cachedTokenLength: Int, tokenLength: Int, compressionRatio: Int) {
    precondition(cachedTokenLength >= 0)
    precondition(tokenLength > 0)
    precondition(compressionRatio > 0)
    let existingRowCount = cachedTokenLength / compressionRatio
    let cachedRemainder = cachedTokenLength % compressionRatio
    let keepsPreviousWindow = compressionRatio == 4 && existingRowCount > 0
    let stateCount = cachedRemainder + (keepsPreviousWindow ? compressionRatio : 0)
    let compressorTokenOffset = cachedTokenLength - stateCount
    let totalTokenLength = cachedTokenLength + tokenLength
    let totalRowCount = totalTokenLength / compressionRatio
    let emittedRowCount = totalRowCount - existingRowCount
    let compressorOutputRowOffset =
      compressionRatio == 4 && existingRowCount > 0 && emittedRowCount > 0 ? 1 : 0
    let compressorTokenCount =
      (compressorOutputRowOffset + emittedRowCount) * compressionRatio
    let nextRemainder = totalTokenLength % compressionRatio
    let nextStateCount =
      nextRemainder
      + (compressionRatio == 4 && totalRowCount > 0 ? compressionRatio : 0)
    let nextStateTokenOffset =
      compressionRatio == 4 && totalRowCount > 0
      ? (totalRowCount - 1) * compressionRatio
      : totalRowCount * compressionRatio
    let nextStateOffset = nextStateTokenOffset - compressorTokenOffset
    precondition(nextStateOffset >= 0)
    precondition(nextStateOffset + nextStateCount <= stateCount + tokenLength)
    self.compressionRatio = compressionRatio
    self.existingRowCount = existingRowCount
    self.totalRowCount = totalRowCount
    self.emittedRowCount = emittedRowCount
    self.compressorTokenOffset = compressorTokenOffset
    self.compressorTokenCount = compressorTokenCount
    self.compressorOutputRowOffset = compressorOutputRowOffset
    self.stateCount = stateCount
    self.nextStateCount = nextStateCount
    self.nextStateOffset = nextStateOffset
  }

  public var stateCapacity: Int {
    DeepSeek4CompressorStateCapacity(compressionRatio: compressionRatio)
  }
}

public func DeepSeek4CompressorStateCapacity(compressionRatio: Int) -> Int {
  precondition(compressionRatio > 0)
  return compressionRatio == 4 ? compressionRatio * 2 - 1 : compressionRatio - 1
}

private func DeepSeek4StoreUsesBareTensorName(_ name: String) -> Bool {
  switch name {
  case "embed.hc_broadcast", "hc_head_scale", "hc_head_base":
    return true
  default:
    return name.hasSuffix(".hc_attn_scale")
      || name.hasSuffix(".hc_attn_base")
      || name.hasSuffix(".hc_ffn_scale")
      || name.hasSuffix(".hc_ffn_base")
      || name.hasSuffix(".attn.attn_sink")
      || name.hasSuffix(".attn.wo_a.group_ids")
      || name.hasSuffix(".compressor.ape")
      || name.hasSuffix(".ffn.gate.bias")
      || name.hasSuffix(".ffn.gate.tid2eid")
  }
}

public func DeepSeek4StoreReader(
  storeKey: String = "text_model",
  configuration _: DeepSeek4ModelConfiguration = .deepSeekV4Flash
) -> (String, DataType, TensorFormat, TensorShape) -> DynamicGraph.Store.ModelReaderResult {
  let keyPrefix = "__\(storeKey)__["
  return { name, _, _, _ in
    guard name.hasPrefix(keyPrefix), name.hasSuffix("]") else {
      return .fail
    }
    var tensorName = String(name.dropFirst(keyPrefix.count).dropLast())
    guard tensorName.hasPrefix("t-") else {
      return .fail
    }
    tensorName.removeFirst(2)
    if let range = tensorName.range(of: #"-\d+-\d+$"#, options: .regularExpression) {
      tensorName.removeSubrange(range)
    }
    let storeTensorName =
      DeepSeek4StoreUsesBareTensorName(tensorName)
      ? tensorName : "\(tensorName).weight"
    return .continue("__\(storeKey)__[\(storeTensorName)]")
  }
}

public func DeepSeek4PairToToken(
  tokenLength: Int, configuration: DeepSeek4ModelConfiguration = .deepSeekV4Flash
) -> Tensor<Int32> {
  var tokenIndices = [Int32]()
  tokenIndices.reserveCapacity(tokenLength * configuration.routedExperts)
  for tokenIndex in 0..<tokenLength {
    tokenIndices.append(
      contentsOf: repeatElement(
        Int32(tokenIndex), count: configuration.routedExperts))
  }
  return Tensor<Int32>(
    tokenIndices, .CPU, .C(tokenLength * configuration.routedExperts))
}

private func DeepSeek4RopeYarnRamp(low: Double, high: Double, index: Int) -> Double {
  let y = (Double(index / 2) - low) / max(0.001, high - low)
  return 1.0 - min(1.0, max(0.0, y))
}

private func DeepSeek4RopeYarnCorrDim(
  nDims: Int, originalContext: Int, nRot: Double, base: Double
) -> Double {
  return Double(nDims) * log(Double(originalContext) / (nRot * 2.0 * Double.pi))
    / (2.0 * log(base))
}

private func DeepSeek4RopeYarnCorrDims(
  nDims: Int, originalContext: Int, base: Double, betaFast: Double, betaSlow: Double
) -> (Double, Double) {
  let start = floor(
    DeepSeek4RopeYarnCorrDim(
      nDims: nDims, originalContext: originalContext, nRot: betaFast, base: base))
  let end = ceil(
    DeepSeek4RopeYarnCorrDim(
      nDims: nDims, originalContext: originalContext, nRot: betaSlow, base: base))
  return (max(0.0, start), min(Double(nDims - 1), end))
}

public func DeepSeek4RotaryEmbedding<FloatType: TensorNumeric & BinaryFloatingPoint>(
  sequenceLength: Int, cachedTokenLength: Int = 0,
  compressed: Bool = false,
  configuration: DeepSeek4ModelConfiguration = .deepSeekV4Flash,
  of dataType: FloatType.Type = FloatType.self
) -> Tensor<FloatType> {
  let headDim = configuration.attentionHeadDim
  let nRot = configuration.rotaryDim
  let nNope = headDim - nRot
  let freqBase = compressed ? configuration.ropeTheta : 10_000
  let freqScale = compressed ? 1.0 / configuration.ropeScaleFactor : 1.0
  let extFactor = compressed ? 1.0 : 0.0
  let attnFactor = extFactor != 0 ? 1.0 / (1.0 + 0.1 * log(1.0 / freqScale)) : 1.0
  let corr = DeepSeek4RopeYarnCorrDims(
    nDims: nRot, originalContext: configuration.ropeOriginalContext,
    base: freqBase, betaFast: configuration.ropeYarnBetaFast,
    betaSlow: configuration.ropeYarnBetaSlow)
  var rotary = Tensor<FloatType>(
    Array(repeating: FloatType.zero, count: sequenceLength * headDim), .CPU,
    .NHWC(1, sequenceLength, 1, headDim))
  for row in 0..<sequenceLength {
    for i in stride(from: 0, to: nNope, by: 2) {
      rotary[0, row, 0, i] = 1
      rotary[0, row, 0, i + 1] = 0
    }
    let position = Double(cachedTokenLength + row)
    for i in stride(from: 0, to: nRot, by: 2) {
      let freq = pow(freqBase, -Double(i) / Double(nRot))
      let thetaExtrap = position * freq
      let thetaInterp = freqScale * thetaExtrap
      let rampMix = DeepSeek4RopeYarnRamp(low: corr.0, high: corr.1, index: i) * extFactor
      let theta = thetaInterp * (1.0 - rampMix) + thetaExtrap * rampMix
      let mscale = extFactor != 0 ? attnFactor * (1.0 + 0.1 * log(1.0 / freqScale)) : attnFactor
      let offset = nNope + i
      rotary[0, row, 0, offset] = FloatType(cos(theta) * mscale)
      rotary[0, row, 0, offset + 1] = FloatType(sin(theta) * mscale)
    }
  }
  return rotary
}

private func DeepSeek4RotaryRowsForHeadDim(
  _ rotary: Model.IO, rowCount: Int, headDim: Int,
  configuration: DeepSeek4ModelConfiguration
) -> Model.IO {
  let fullHeadDim = configuration.attentionHeadDim
  if headDim == fullHeadDim {
    return rotary.reshaped([rowCount, fullHeadDim])
  }
  let nNope = headDim - configuration.rotaryDim
  let prefix = rotary.reshaped([rowCount, nNope], offset: [0, 0], strides: [fullHeadDim, 1])
  let tail = rotary.reshaped(
    [rowCount, configuration.rotaryDim],
    offset: [0, rowCount > 0 ? fullHeadDim - configuration.rotaryDim : 0],
    strides: [fullHeadDim, 1]
  ).copied()
  return Concat(axis: 1)([prefix, tail])
}

private func DeepSeek4RopeTailRows(
  _ x: Model.IO, rotary: Model.IO, rowCount: Int, heads: Int, headDim: Int,
  inverse: Bool, configuration: DeepSeek4ModelConfiguration
) -> Model.IO {
  let rotaryRows = DeepSeek4RotaryRowsForHeadDim(
    rotary, rowCount: rowCount, headDim: headDim, configuration: configuration)
  return Cmul(conjugate: inverse)(
    x.reshaped([rowCount, heads, headDim]),
    rotaryRows.reshaped([rowCount, 1, headDim])
  ).reshaped([rowCount, heads * headDim])
}

private func DeepSeek4CompressedRotaryRows(
  _ rotary: Model.IO, tokenLength: Int, windowCount: Int, compressionRatio: Int,
  headDim: Int, configuration: DeepSeek4ModelConfiguration
) -> Model.IO {
  precondition(windowCount * compressionRatio <= tokenLength)
  let fullHeadDim = configuration.attentionHeadDim
  let fullRows = rotary.reshaped(
    [windowCount, fullHeadDim], offset: [0, 0],
    strides: [compressionRatio * fullHeadDim, 1]
  ).copied()
  return DeepSeek4RotaryRowsForHeadDim(
    fullRows, rowCount: windowCount, headDim: headDim, configuration: configuration
  ).contiguous()
}

private func DeepSeek4HCSplitWeightedSum(
  mix: Model.IO, scale: ModelIOConvertible, base: ModelIOConvertible,
  residualHC: Model.IO, tokenLength: Int,
  configuration: DeepSeek4ModelConfiguration
) -> (post: Model.IO, comb: Model.IO, weighted: Model.IO) {
  let hc = configuration.hcCount
  let mixDim = configuration.hcMixDim
  let outputs = HyperConnection(
    count: hc, sinkhornIterations: 20, epsilon: 1.0e-6, operation: .splitWeightedSum
  )(
    mix.reshaped([tokenLength, mixDim]), scale, base,
    residualHC.reshaped([tokenLength, hc, configuration.hiddenSize]))
  return (
    outputs[0].reshaped([tokenLength, hc]),
    outputs[1].reshaped([tokenLength, hc * hc]),
    outputs[2].reshaped([tokenLength, configuration.hiddenSize])
  )
}

private func DeepSeek4HCExpand(
  block: Model.IO, residualHC: Model.IO, post: Model.IO, comb: Model.IO,
  tokenLength: Int, configuration: DeepSeek4ModelConfiguration
) -> Model.IO {
  let hc = configuration.hcCount
  let hidden = configuration.hiddenSize
  let block = block.to(of: residualHC)
  return HyperConnection(count: hc, operation: .expand)(
    block.reshaped([tokenLength, hidden]), residualHC.reshaped([tokenLength, hc, hidden]),
    post.reshaped([tokenLength, hc]), comb.reshaped([tokenLength, hc, hc]))[0]
}

private func DeepSeek4AttentionProjection(
  prefix: String, x: Model.IO, rotary: Model.IO, tokenLength: Int,
  configuration: DeepSeek4ModelConfiguration
) -> (query: Model.IO, keyValue: Model.IO, queryRank: Model.IO) {
  let headDim = configuration.attentionHeadDim
  let heads = configuration.attentionHeads
  let wqA = Dense(
    count: configuration.queryLowRank, noBias: true, flags: DeepSeek4Q8DenseFlags,
    name: "\(prefix).wq_a")
  let wqB = Dense(
    count: heads * headDim, noBias: true, flags: DeepSeek4Q8DenseFlags,
    name: "\(prefix).wq_b")
  let wkv = Dense(
    count: headDim, noBias: true, flags: DeepSeek4Q8DenseFlags, name: "\(prefix).wkv")
  let q8Input = x.reshaped([tokenLength, configuration.hiddenSize]).to(.Float16)

  let queryLowRankRaw = wqA(q8Input).to(.Float32)
    .reshaped([tokenLength, configuration.queryLowRank])
  let queryRank = RMSNorm(epsilon: 1.0e-6, axis: [1], name: "\(prefix).q_norm")(
    queryLowRankRaw
  ).reshaped([tokenLength, configuration.queryLowRank])
  let qDense = wqB(queryRank.to(.Float16)).to(.Float32)
    .reshaped([tokenLength, heads, headDim])
  let qNorm = RMSNorm(epsilon: 1.0e-6, axis: [2], elementwiseAffine: false)(
    qDense
  ).reshaped([tokenLength, heads * headDim])
  let query = DeepSeek4RopeTailRows(
    qNorm, rotary: rotary, rowCount: tokenLength, heads: heads, headDim: headDim,
    inverse: false, configuration: configuration)

  let kvRaw = wkv(q8Input).to(.Float32).reshaped([tokenLength, headDim])
  let kvNorm = RMSNorm(epsilon: 1.0e-6, axis: [1], name: "\(prefix).kv_norm")(
    kvRaw
  ).reshaped([tokenLength, headDim])
  let keyValue = DeepSeek4RopeTailRows(
    kvNorm, rotary: rotary, rowCount: tokenLength, heads: 1, headDim: headDim,
    inverse: false, configuration: configuration)
  return (query, keyValue, queryRank)
}

private func DeepSeek4AttentionOutput(
  prefix: String, heads: Model.IO, rotary: Model.IO, attentionOutputGroupCounts: Model.IO,
  tokenLength: Int,
  configuration: DeepSeek4ModelConfiguration
) -> Model.IO {
  let headDim = configuration.attentionHeadDim
  let outGroups = configuration.attentionOutputGroups
  let headsPerGroup = configuration.attentionHeads / outGroups
  let groupDim = headsPerGroup * headDim
  let headsBack = DeepSeek4RopeTailRows(
    heads, rotary: rotary, rowCount: tokenLength, heads: configuration.attentionHeads,
    headDim: headDim, inverse: true, configuration: configuration)
  let groupIDs = Parameter<Int32>(
    .GPU(0), .C(outGroups), trainable: false, name: "\(prefix).wo_a.group_ids")
  let woA = SegmentedDense(
    segments: outGroups, count: configuration.attentionLowRank, noBias: true,
    name: "\(prefix).wo_a")
  let groupMajorHeads = headsBack.reshaped([tokenLength, outGroups, groupDim])
    .transposed(0, 1)
    .contiguous()
    .reshaped([tokenLength * outGroups, groupDim])
    .to(.Float16)
  let low = woA(groupMajorHeads, groupIDs, attentionOutputGroupCounts.reshaped([outGroups]))
    .reshaped([outGroups, tokenLength, configuration.attentionLowRank])
    .transposed(0, 1)
    .contiguous()
    .reshaped([tokenLength, configuration.attentionOutputLowDim])
  let woB = Dense(count: configuration.hiddenSize, noBias: true, name: "\(prefix).wo_b")
  return woB(low).to(.Float32).reshaped([tokenLength, configuration.hiddenSize])
}

private func DeepSeek4RawAttention(
  query: Model.IO, rawKV: Model.IO, sinks: ModelIOConvertible, queryLength: Int,
  rawRowCount: Int,
  configuration: DeepSeek4ModelConfiguration
) -> Model.IO {
  let headDim = configuration.attentionHeadDim
  let attentionQuery = query.to(.Float16)
  let kv = rawKV.to(.Float16)
  let attention = ScaledDotProductAttention(
    scale: 1.0 / Float(headDim).squareRoot(),
    isCausal: true,
    hasAttentionSinks: true,
    slidingWindow: configuration.rawWindow,
    name: "DeepSeek4RawAttention")
  return attention(
    attentionQuery.reshaped(.NHWC(1, queryLength, configuration.attentionHeads, headDim)),
    kv.reshaped(.NHWC(1, rawRowCount, 1, headDim)),
    kv.reshaped(.NHWC(1, rawRowCount, 1, headDim)),
    sinks.reshaped(.NHWC(1, 1, configuration.attentionHeads, 1)).to(of: attentionQuery)
  ).reshaped([queryLength, configuration.attentionHeads * headDim])
}

private func DeepSeek4RawCachedAttention(
  query: Model.IO, currentRawKV: Model.IO, sinks: ModelIOConvertible, cache: Model.IO,
  cachedTokenLength: Int, tokenLength: Int,
  configuration: DeepSeek4ModelConfiguration
) -> Model.IO {
  let queryInput = Input()
  let currentRawKVInput = Input()
  let sinksInput = Input()
  let cacheInput = Input()
  let cacheOutput = DeepSeek4CacheAppend(
    currentRawKVInput, to: cacheInput, rowOffset: cachedTokenLength,
    rowCount: tokenLength, width: configuration.attentionHeadDim)
  let totalTokenLength = cachedTokenLength + tokenLength
  let attentionKeyValue = cacheInput.reshaped(
    [totalTokenLength, configuration.attentionHeadDim])
  let heads = DeepSeek4RawAttention(
    query: queryInput, rawKV: attentionKeyValue, sinks: sinksInput,
    queryLength: tokenLength, rawRowCount: totalTokenLength,
    configuration: configuration)
  heads.add(dependencies: [cacheOutput])
  return Model(
    [queryInput, currentRawKVInput, sinksInput, cacheInput], [heads]
  )(query, currentRawKV, sinks, cache)
}

private func DeepSeek4SparseIndexedAttention(
  query: Model.IO, rawKV: Model.IO, compressedKV: Model.IO,
  selectedCompressedRows: Model.IO,
  sinks: ModelIOConvertible, queryLength: Int, rawRowCount: Int, compressedRows: Int,
  selectedRowCount: Int,
  configuration: DeepSeek4ModelConfiguration
) -> Model.IO {
  let headDim = configuration.attentionHeadDim
  let attentionQuery = query.to(.Float16)
  let rawAttentionKV = rawKV.to(.Float16)
  let compressedAttentionKV = compressedKV.to(.Float16)
  let compressedAttentionKVInput =
    compressedRows == 0
    ? compressedAttentionKV.reshaped([0])
    : compressedAttentionKV.reshaped([1, compressedRows, 1, headDim], format: .NHWC)
  let attention = SparseIndexedAttention(
    scale: 1.0 / Float(headDim).squareRoot(),
    isCausal: true, hasAttentionSinks: true,
    slidingWindow: configuration.rawWindow)
  let attentionQueryInput = attentionQuery.reshaped(
    .NHWC(1, queryLength, configuration.attentionHeads, headDim))
  let rawAttentionKVInput = rawAttentionKV.reshaped(.NHWC(1, rawRowCount, 1, headDim))
  let sinksInput = sinks.reshaped(
    .NHWC(1, 1, configuration.attentionHeads, 1)
  ).to(of: attentionQuery)
  let heads = attention(
    attentionQueryInput, rawAttentionKVInput, rawAttentionKVInput,
    compressedAttentionKVInput, compressedAttentionKVInput,
    selectedCompressedRows.reshaped(
      [queryLength, selectedRowCount], format: .NHWC),
    sinksInput)
  return heads.reshaped([queryLength, configuration.attentionHeads * headDim])
}

private func DeepSeek4CausalCompressedIndicesInput(
  layerIndex: Int, configuration: DeepSeek4ModelConfiguration,
  inputs: inout [Input], indices: inout [Int: Input]
) -> Input? {
  switch configuration.attentionKind(layerIndex: layerIndex) {
  case .compressed(let compressionRatio):
    if let existing = indices[compressionRatio] {
      return existing
    }
    let input = Input()
    indices[compressionRatio] = input
    inputs.append(input)
    return input
  case .raw, .indexed:
    return nil
  }
}

private struct DeepSeek4LayerCacheInputs {
  let rawKeyValue: Input
  let compressedKeyValue: Input?
  let compressorState: Input?
  let nextCompressorState: Input?
  let indexerKeyValue: Input?
}

private struct DeepSeek4PreparedCompressorInput {
  let input: Model.IO
  let stateUpdate: Model.IO
  let plan: DeepSeek4CompressionPlan
}

private func DeepSeek4CompressorStateBarrier(
  _ value: Model.IO, stateUpdate: Model.IO
) -> Model.IO {
  // Keep the ping-pong state write in the dataflow before the caller swaps the buffers.
  let stateDependency = stateUpdate.reshaped([1], offset: [0], strides: [1])
    .to(of: value)
  return value + stateDependency * 0
}

private func DeepSeek4CacheAppend(
  _ value: Model.IO, to cache: Model.IO, rowOffset: Int, rowCount: Int, width: Int
) -> Model.IO {
  let destination = cache.reshaped(
    [rowCount, width], offset: [rowOffset, 0], strides: [width, 1])
  return value.reshaped([rowCount, width]).moved(to: destination)
}

private func DeepSeek4ComposeCache(
  _ current: Model.IO, cache: Model.IO, existingRowCount: Int,
  currentRowCount: Int, width: Int
) -> Model.IO {
  let currentInput = Input()
  let cacheInput = Input()
  let output = DeepSeek4CacheAppend(
    currentInput, to: cacheInput, rowOffset: existingRowCount,
    rowCount: currentRowCount, width: width)
  let composed = cacheInput.reshaped(
    [existingRowCount + currentRowCount, width],
    offset: [0, 0], strides: [width, 1])
  composed.add(dependencies: [output])
  return Model([currentInput, cacheInput], [composed])(current, cache)
}

private func DeepSeek4UpdateFixedCapacityCache(
  _ current: Model.IO, cache: Model.IO, rowCount: Int, capacity: Int, width: Int
) -> Model.IO {
  let currentInput = Input()
  let cacheInput = Input()
  let output = DeepSeek4CacheAppend(
    currentInput, to: cacheInput, rowOffset: 0, rowCount: rowCount, width: width)
  let updatedCache = cacheInput.reshaped(
    [capacity, width], offset: [0, 0], strides: [width, 1])
  updatedCache.add(dependencies: [output])
  return Model([currentInput, cacheInput], [updatedCache])(current, cache)
}

private func DeepSeek4PrepareCompressorInput(
  _ x: Model.IO, state: Model.IO, nextStateCache: Model.IO,
  cachedTokenLength: Int, tokenLength: Int,
  compressionRatio: Int, configuration: DeepSeek4ModelConfiguration
) -> DeepSeek4PreparedCompressorInput {
  let plan = DeepSeek4CompressionPlan(
    cachedTokenLength: cachedTokenLength, tokenLength: tokenLength,
    compressionRatio: compressionRatio)
  let hidden = configuration.hiddenSize
  let stateRows = state.reshaped(
    [plan.stateCount, hidden], offset: [0, 0], strides: [hidden, 1])
  let combined = Concat(axis: 0)([
    stateRows,
    x.reshaped([tokenLength, hidden]),
  ])
  let compressorInput = combined.reshaped(
    [plan.compressorTokenCount, hidden], offset: [0, 0], strides: [hidden, 1])
  let nextState = combined.reshaped(
    [plan.nextStateCount, hidden], offset: [plan.nextStateOffset, 0],
    strides: [hidden, 1])
  let updatedState = DeepSeek4UpdateFixedCapacityCache(
    nextState.to(of: nextStateCache), cache: nextStateCache,
    rowCount: plan.nextStateCount, capacity: plan.stateCapacity, width: hidden)
  return DeepSeek4PreparedCompressorInput(
    input: compressorInput, stateUpdate: updatedState, plan: plan)
}

private func DeepSeek4FP8KVRoundTrip(
  _ x: Model.IO, rowCount: Int, headDim: Int, rotaryDim: Int, enabled: Bool
) -> Model.IO {
  guard enabled else { return x.reshaped([rowCount, headDim]) }
  let nNope = headDim - rotaryDim
  precondition(nNope >= 0 && nNope % 64 == 0)
  guard nNope > 0 else { return x }
  return ConformDataFormat(.FP8E4M3, preservedTail: rotaryDim)(
    x.reshaped([rowCount, headDim]).to(.Float32).copied()
  ).reshaped([rowCount, headDim])
}

private func DeepSeek4Ratio4RollingPool(
  kvProjected: Model.IO, scoreProjected: Model.IO, ape: ModelIOConvertible,
  windowCount: Int, headDim: Int
) -> Model.IO {
  let compressionRatio = 4
  let width = 2 * headDim
  let rowWidth = width * compressionRatio
  let score =
    scoreProjected.reshaped([windowCount, compressionRatio, width])
    + ape.reshaped([1, compressionRatio, width]).to(of: scoreProjected)
  let primaryKV = kvProjected.reshaped(
    [windowCount, compressionRatio, headDim], offset: [0, 0, 0],
    strides: [rowWidth, width, 1]
  ).transposed(1, 2).contiguous()
  let primaryScore = score.reshaped(
    [windowCount, compressionRatio, headDim], offset: [0, 0, 0],
    strides: [rowWidth, width, 1]
  ).transposed(1, 2).contiguous()
  let companionKV = kvProjected.reshaped(
    [windowCount, compressionRatio, headDim],
    offset: [0, 0, windowCount > 0 ? headDim : 0],
    strides: [rowWidth, width, 1]
  ).transposed(1, 2).contiguous()
  let companionScore = score.reshaped(
    [windowCount, compressionRatio, headDim],
    offset: [0, 0, windowCount > 0 ? headDim : 0],
    strides: [rowWidth, width, 1]
  ).transposed(1, 2).contiguous()
  let zeroKV =
    ape.reshaped(
      [1, compressionRatio, headDim], offset: [0, 0, 0],
      strides: [rowWidth, width, 1]
    ).transposed(1, 2).contiguous() * 0
  let negInfScore = zeroKV - 1.0e4
  let prefixWindowCount = min(windowCount, 1)
  let previousWindowCount = max(windowCount - 1, 0)
  let zeroPrefix = zeroKV.reshaped(
    [prefixWindowCount, headDim, compressionRatio], offset: [0, 0, 0],
    strides: [headDim * compressionRatio, compressionRatio, 1])
  let negInfPrefix = negInfScore.reshaped(
    [prefixWindowCount, headDim, compressionRatio], offset: [0, 0, 0],
    strides: [headDim * compressionRatio, compressionRatio, 1])
  let previousPrimaryKV = primaryKV.reshaped(
    [previousWindowCount, headDim, compressionRatio], offset: [0, 0, 0],
    strides: [headDim * compressionRatio, compressionRatio, 1])
  let previousPrimaryScore = primaryScore.reshaped(
    [previousWindowCount, headDim, compressionRatio], offset: [0, 0, 0],
    strides: [headDim * compressionRatio, compressionRatio, 1])
  let previousKV = Concat(axis: 0)([zeroPrefix, previousPrimaryKV])
  let previousScore = Concat(axis: 0)([negInfPrefix, previousPrimaryScore])
  let rows = windowCount * headDim
  let paddedKV = Concat(axis: 1)([
    previousKV.reshaped([rows, compressionRatio]),
    companionKV.reshaped([rows, compressionRatio]),
  ]).reshaped([windowCount, headDim, 2 * compressionRatio])
  let paddedScore = Concat(axis: 1)([
    previousScore.reshaped([rows, compressionRatio]),
    companionScore.reshaped([rows, compressionRatio]),
  ]).reshaped([windowCount, headDim, 2 * compressionRatio])
  let weights = paddedScore.reshaped([rows, 2 * compressionRatio])
    .softmax()
    .reshaped([windowCount, headDim, 2 * compressionRatio])
  return (weights .* paddedKV).reduced(.sum, axis: [2]).reshaped([windowCount, headDim])
}

private func DeepSeek4Compressor<FloatType: TensorNumeric>(
  prefix: String, x: Model.IO, rotary: Model.IO, tokenLength: Int, compressionRatio: Int,
  headDim: Int, emitIndexerWHT: Bool, configuration: DeepSeek4ModelConfiguration,
  of dataType: FloatType.Type
) -> Model.IO {
  let windowCount = tokenLength / compressionRatio
  let tokenRows = windowCount * compressionRatio
  let width = (compressionRatio == 4 ? 2 : 1) * headDim
  let kv = Dense(count: width, noBias: true, name: "\(prefix).wkv")
  let gate = Dense(count: width, noBias: true, name: "\(prefix).wgate")
  let ape = Parameter<FloatType>(.GPU(0), .NC(compressionRatio, width), name: "\(prefix).ape")
  let xWindow = x.reshaped(
    [tokenRows, configuration.hiddenSize], offset: [0, 0],
    strides: [configuration.hiddenSize, 1])
  let kvProjected = kv(xWindow)
  let scoreProjected = gate(xWindow)
  let pooled: Model.IO
  if compressionRatio == 4 {
    pooled = DeepSeek4Ratio4RollingPool(
      kvProjected: kvProjected, scoreProjected: scoreProjected, ape: ape,
      windowCount: windowCount, headDim: headDim)
  } else {
    let kvRows = kvProjected.reshaped([windowCount, compressionRatio, headDim])
    let scores =
      scoreProjected.reshaped([windowCount, compressionRatio, headDim])
      + ape.reshaped([1, compressionRatio, headDim]).to(of: scoreProjected)
    let weights = scores.transposed(1, 2)
      .reshaped([windowCount * headDim, compressionRatio])
      .softmax()
      .reshaped([windowCount, headDim, compressionRatio])
      .transposed(1, 2)
    pooled = (weights .* kvRows).reduced(.sum, axis: [1]).reshaped([windowCount, headDim])
  }
  let normed = RMSNorm(epsilon: 1.0e-6, axis: [1], name: "\(prefix).norm")(
    pooled.reshaped([windowCount, headDim])
  )
  let rotaryRows = DeepSeek4CompressedRotaryRows(
    rotary, tokenLength: tokenLength, windowCount: windowCount,
    compressionRatio: compressionRatio, headDim: headDim, configuration: configuration)
  let compressed = Functional.cmul(
    left: normed,
    right: rotaryRows.reshaped([windowCount, headDim]))
  if emitIndexerWHT {
    return DeepSeek4IndexerWHT(
      compressed, rowCount: windowCount, width: headDim
    ).reshaped([windowCount, headDim])
  } else {
    return compressed.reshaped([windowCount, headDim])
  }
}

private func DeepSeek4IndexerWHT(_ x: Model.IO, rowCount: Int, width: Int) -> Model.IO {
  return WalshHadamardTransform(scale: 1.0 / Float(width).squareRoot())(
    x.reshaped([rowCount, width]).to(.Float32))
}

private func DeepSeek4IndexerQAT<FloatType: TensorNumeric>(
  prefix: String, _ x: Model.IO, rowCount: Int,
  configuration: DeepSeek4ModelConfiguration, of dataType: FloatType.Type
) -> Model.IO {
  let width = configuration.indexerHeadDim
  return DeepSeek4IndexerWHT(x, rowCount: rowCount, width: width)
}

private func DeepSeek4IndexerSelection<FloatType: TensorNumeric>(
  prefix: String, queryRank: Model.IO, attnNorm: Model.IO, rotary: Model.IO,
  indexerKV: Model.IO, tokenLength: Int, compressionRatio: Int, compressedRows: Int,
  configuration: DeepSeek4ModelConfiguration, of dataType: FloatType.Type
) -> Model.IO {
  let indexerWqB = Dense(
    count: configuration.indexerHeads * configuration.indexerHeadDim, noBias: true,
    name: "\(prefix).indexer.wq_b")
  let indexerWeightsProj = Dense(
    count: configuration.indexerHeads, noBias: true, name: "\(prefix).indexer.weights_proj")
  let indexDense = indexerWqB(queryRank)
    .reshaped([tokenLength, configuration.indexerHeads * configuration.indexerHeadDim])
  let indexRope = DeepSeek4RopeTailRows(
    indexDense, rotary: rotary, rowCount: tokenLength, heads: configuration.indexerHeads,
    headDim: configuration.indexerHeadDim, inverse: false, configuration: configuration)
  let indexRows = indexRope.reshaped([
    tokenLength, configuration.indexerHeads, configuration.indexerHeadDim,
  ]).contiguous().reshaped([
    tokenLength * configuration.indexerHeads, configuration.indexerHeadDim,
  ])
  let indexQ = DeepSeek4IndexerQAT(
    prefix: prefix, indexRows, rowCount: tokenLength * configuration.indexerHeads,
    configuration: configuration, of: dataType)
  let indexWeights =
    indexerWeightsProj(attnNorm.reshaped([tokenLength, configuration.hiddenSize]))
    .reshaped([tokenLength, configuration.indexerHeads])
    * (1.0 / Float(configuration.indexerHeadDim).squareRoot()
      / Float(configuration.indexerHeads).squareRoot())
  return ScaledDotProductArgPartition(
    kth: configuration.indexerTopK,
    scale: 1,
    isCausal: true,
    compressionRatio: compressionRatio,
    name: "\(prefix).indexer.sdpap")(
      indexQ.reshaped([tokenLength, configuration.indexerHeads, configuration.indexerHeadDim]),
      indexerKV.reshaped([compressedRows, configuration.indexerHeadDim]),
      indexWeights.reshaped([tokenLength, configuration.indexerHeads])
    ).reshaped([tokenLength, configuration.indexerTopK])
}

private func DeepSeek4SharedFFN(
  prefix: String, x: Model.IO, tokenLength: Int,
  configuration: DeepSeek4ModelConfiguration
) -> Model.IO {
  let gate = Dense(
    count: configuration.sharedIntermediateSize, noBias: true,
    name: "\(prefix).shared_experts.w1")
  let up = Dense(
    count: configuration.sharedIntermediateSize, noBias: true,
    name: "\(prefix).shared_experts.w3")
  let down = Dense(
    count: configuration.hiddenSize, noBias: true, name: "\(prefix).shared_experts.w2")
  let mid = Functional.swishMul(
    value: up(x).clamped((-10.0)...10.0), gate: gate(x).clamped(...10.0))
  return down(mid).reshaped([
    tokenLength, configuration.hiddenSize,
  ])
}

private func DeepSeek4NormalizeRouterWeights(
  _ selectedProbs: Model.IO, tokenLength: Int, configuration: DeepSeek4ModelConfiguration
) -> Model.IO {
  return
    (selectedProbs
    .* selectedProbs.reduced(.sum, axis: [1]).reshaped([tokenLength, 1])
    .reciprocal()) * 1.5
}

private func DeepSeek4RoutedMoE<FloatType: TensorNumeric>(
  prefix: String, x: Model.IO, routerInput: Model.IO, selectedExpertOverride: Model.IO?,
  selectedProbabilityIndexOverride: Model.IO?, pairToToken: Model.IO, tokenLength: Int,
  routerKind: DeepSeek4RouterKind, configuration: DeepSeek4ModelConfiguration,
  of dataType: FloatType.Type
) -> Model.IO {
  let router = Dense(count: configuration.expertCount, noBias: true, name: "\(prefix).gate")
  let routerBias = Parameter<FloatType>(
    .GPU(0), .C(configuration.expertCount), name: "\(prefix).gate.bias")
  let logits = router(routerInput.reshaped([tokenLength, configuration.hiddenSize]).to(.Float32))
  let probs = logits.softplus().squareRoot().reshaped([tokenLength, configuration.expertCount])
  let selected: Model.IO
  let routerWeights: Model.IO
  switch routerKind {
  case .standard:
    let route = (probs + routerBias.reshaped([1, configuration.expertCount]).to(of: probs))
      .partitioned(kth: configuration.routedExperts, axis: 1, descending: true)
    // This follows the HiDream partitioned-router shape: route[0] is the selected score tensor
    // and route[1] is the selected expert id tensor.
    let selectedScores = route[0].reshaped([tokenLength, configuration.routedExperts])
    selected = route[1].reshaped([tokenLength, configuration.routedExperts])
    let selectedBias = IndexSelect()(
      routerBias.reshaped([configuration.expertCount]).to(of: selectedScores),
      selected.reshaped([tokenLength * configuration.routedExperts])
    ).reshaped([tokenLength, configuration.routedExperts])
    let selectedProbs = selectedScores - selectedBias
    routerWeights = DeepSeek4NormalizeRouterWeights(
      selectedProbs, tokenLength: tokenLength, configuration: configuration)
  case .tokenHash:
    precondition(selectedExpertOverride != nil)
    precondition(selectedProbabilityIndexOverride != nil)
    selected = selectedExpertOverride!.reshaped([tokenLength, configuration.routedExperts])
    let selectedProbs = IndexSelect()(
      probs.reshaped([tokenLength * configuration.expertCount]),
      selectedProbabilityIndexOverride!.reshaped([tokenLength * configuration.routedExperts])
    ).reshaped([tokenLength, configuration.routedExperts])
    routerWeights = DeepSeek4NormalizeRouterWeights(
      selectedProbs, tokenLength: tokenLength, configuration: configuration)
  }
  let pairs = tokenLength * configuration.routedExperts
  let selectedFlat = selected.reshaped([pairs])
  let sorted = selectedFlat.sorted(axis: 0, descending: false)
  let sortedExperts = sorted[0]
  let sortIndices = sorted[1]
  let sortedWeights = IndexSelect()(routerWeights.reshaped([pairs]), sortIndices)
  let sortedTokenIndices = IndexSelect()(pairToToken.reshaped([pairs]), sortIndices)
  let gathered = IndexSelect()(
    x.reshaped([tokenLength, configuration.hiddenSize]), sortedTokenIndices)
  let groupedExpertIds = sortedExperts.uniqueConsecutive(count: configuration.expertCount)
  let gate = SegmentedDense(
    segments: configuration.expertCount, count: configuration.expertIntermediateSize,
    noBias: true, name: "\(prefix).experts.w1")
  let up = SegmentedDense(
    segments: configuration.expertCount, count: configuration.expertIntermediateSize,
    noBias: true, name: "\(prefix).experts.w3")
  let down = SegmentedDense(
    segments: configuration.expertCount, count: configuration.hiddenSize,
    noBias: true, name: "\(prefix).experts.w2")
  let sortedGate = gate(gathered, groupedExpertIds).clamped(...10.0)
  let sortedUp = up(gathered, groupedExpertIds).clamped((-10.0)...10.0)
  let hidden =
    Functional.swishMul(value: sortedUp, gate: sortedGate)
    .* sortedWeights.reshaped([pairs, 1]).to(of: sortedGate)
  let sortedOut = down(hidden, groupedExpertIds)
  let out = Functional.scatterAdd(
    count: tokenLength, sortedOut, index: sortedTokenIndices
  ).reshaped([tokenLength, configuration.hiddenSize])
  return out
}

private func DeepSeek4Embedding<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, tokens: Model.IO, tokenLength: Int,
  configuration: DeepSeek4ModelConfiguration
) -> Model.IO {
  let embed = Embedding(
    FloatType.self, vocabularySize: configuration.vocabularySize,
    embeddingSize: configuration.hiddenSize,
    name: "embed")
  let tokenEmbedding = embed(tokens).reshaped([tokenLength, 1, configuration.hiddenSize])
  let hcBroadcast = Parameter<FloatType>(
    .GPU(0), .HWC(1, configuration.hcCount, 1), trainable: false,
    name: "embed.hc_broadcast")
  return (tokenEmbedding .* hcBroadcast.reshaped([1, configuration.hcCount, 1]))
    .reshaped([tokenLength, configuration.hcCount, configuration.hiddenSize])
}

private func DeepSeek4Layer<FloatType: TensorNumeric>(
  prefix: String, layerIndex: Int, residualHC: Model.IO, rotary: Model.IO,
  compressorRotary: Model.IO?,
  attentionOutputGroupCounts: Model.IO, causalCompressedIndices: Model.IO?,
  selectedExpertOverride: Model.IO?,
  selectedProbabilityIndexOverride: Model.IO?, pairToToken: Model.IO, tokenLength: Int,
  cachedTokenLength: Int, cacheInputs: DeepSeek4LayerCacheInputs?,
  configuration: DeepSeek4ModelConfiguration, of dataType: FloatType.Type
) -> Model.IO {
  let attentionKind = configuration.attentionKind(layerIndex: layerIndex)
  let routerKind = configuration.routerKind(layerIndex: layerIndex)
  let hc = configuration.hcCount
  let hidden = configuration.hiddenSize
  let hcDim = hc * hidden
  let mixDim = configuration.hcMixDim

  let attnHC = Dense(count: mixDim, noBias: true, name: "\(prefix).hc_attn_fn")
  let attnScale = Parameter<Float>(.GPU(0), .C(3), name: "\(prefix).hc_attn_scale")
  let attnBase = Parameter<Float>(.GPU(0), .C(mixDim), name: "\(prefix).hc_attn_base")
  let sinks = Parameter<Float>(
    .GPU(0), .C(configuration.attentionHeads), name: "\(prefix).attn.attn_sink")

  let ffnHC = Dense(count: mixDim, noBias: true, name: "\(prefix).hc_ffn_fn")
  let ffnScale = Parameter<Float>(.GPU(0), .C(3), name: "\(prefix).hc_ffn_scale")
  let ffnBase = Parameter<Float>(.GPU(0), .C(mixDim), name: "\(prefix).hc_ffn_base")

  let attnFlat = RMSNorm(epsilon: 1.0e-6, axis: [1], elementwiseAffine: false)(
    residualHC.reshaped([tokenLength, hcDim])
  ).reshaped([tokenLength, hcDim])
  let attnMix = attnHC(attnFlat).reshaped([tokenLength, mixDim])
  let attnParts = DeepSeek4HCSplitWeightedSum(
    mix: attnMix, scale: attnScale, base: attnBase, residualHC: residualHC,
    tokenLength: tokenLength,
    configuration: configuration)
  let attnNorm = RMSNorm(epsilon: 1.0e-6, axis: [1], name: "\(prefix).attn_norm")(
    attnParts.weighted.reshaped([tokenLength, hidden])
  ).reshaped([tokenLength, hidden])
  let attnBranch = attnNorm.to(FloatType.dataType)

  let projection = DeepSeek4AttentionProjection(
    prefix: "\(prefix).attn", x: attnBranch, rotary: rotary, tokenLength: tokenLength,
    configuration: configuration)
  let kvRope = projection.keyValue.reshaped([tokenLength, configuration.attentionHeadDim])
  let rawKV = DeepSeek4FP8KVRoundTrip(
    kvRope, rowCount: tokenLength, headDim: configuration.attentionHeadDim,
    rotaryDim: configuration.rotaryDim, enabled: configuration.enableFP8KVRoundTrip
  ).reshaped([tokenLength, configuration.attentionHeadDim])
  let rawRows: Int
  let attentionRawKV: Model.IO
  if let cacheInputs = cacheInputs {
    let totalRawRows = cachedTokenLength + tokenLength
    switch attentionKind {
    case .raw:
      rawRows = totalRawRows
      attentionRawKV = cacheInputs.rawKeyValue
    case .compressed, .indexed:
      let composed = DeepSeek4ComposeCache(
        rawKV.to(.Float16), cache: cacheInputs.rawKeyValue,
        existingRowCount: cachedTokenLength, currentRowCount: tokenLength,
        width: configuration.attentionHeadDim)
      rawRows = totalRawRows
      attentionRawKV = composed.reshaped(
        [rawRows, configuration.attentionHeadDim])
    }
  } else {
    rawRows = tokenLength
    attentionRawKV = rawKV
  }

  let preparedCompressor: DeepSeek4PreparedCompressorInput?
  switch attentionKind {
  case .raw:
    preparedCompressor = nil
  case .compressed(let compressionRatio), .indexed(let compressionRatio):
    if let cacheInputs = cacheInputs {
      guard let compressorState = cacheInputs.compressorState,
        let nextCompressorState = cacheInputs.nextCompressorState
      else {
        preconditionFailure("Compressed attention requires compressor state caches.")
      }
      let prepared = DeepSeek4PrepareCompressorInput(
        attnBranch, state: compressorState,
        nextStateCache: nextCompressorState,
        cachedTokenLength: cachedTokenLength, tokenLength: tokenLength,
        compressionRatio: compressionRatio, configuration: configuration)
      preparedCompressor = prepared
    } else {
      preparedCompressor = nil
    }
  }

  func emitCompressedRows(
    prefix: String, compressionRatio: Int, headDim: Int, emitIndexerWHT: Bool
  ) -> (
    rows: Model.IO, existingRowCount: Int, totalRowCount: Int, emittedRowCount: Int
  ) {
    if let preparedCompressor = preparedCompressor {
      guard let compressorRotary = compressorRotary else {
        preconditionFailure("Cached compression requires rotary inputs.")
      }
      let plan = preparedCompressor.plan
      let sourceRows = DeepSeek4Compressor(
        prefix: prefix, x: preparedCompressor.input, rotary: compressorRotary,
        tokenLength: plan.compressorTokenCount, compressionRatio: compressionRatio,
        headDim: headDim, emitIndexerWHT: emitIndexerWHT,
        configuration: configuration, of: dataType)
      let emittedRows = sourceRows.reshaped(
        [plan.emittedRowCount, headDim],
        offset: [plan.compressorOutputRowOffset, 0],
        strides: [headDim, 1])
      return (
        emittedRows, plan.existingRowCount, plan.totalRowCount, plan.emittedRowCount
      )
    }
    let rowCount = tokenLength / compressionRatio
    return (
      DeepSeek4Compressor(
        prefix: prefix, x: attnBranch, rotary: rotary,
        tokenLength: tokenLength, compressionRatio: compressionRatio,
        headDim: headDim, emitIndexerWHT: emitIndexerWHT,
        configuration: configuration, of: dataType),
      0, rowCount, rowCount
    )
  }

  var heads: Model.IO
  switch attentionKind {
  case .raw:
    if let cacheInputs = cacheInputs {
      heads = DeepSeek4RawCachedAttention(
        query: projection.query, currentRawKV: rawKV.to(.Float16), sinks: sinks,
        cache: cacheInputs.rawKeyValue, cachedTokenLength: cachedTokenLength,
        tokenLength: tokenLength, configuration: configuration)
    } else {
      heads = DeepSeek4RawAttention(
        query: projection.query, rawKV: attentionRawKV, sinks: sinks,
        queryLength: tokenLength, rawRowCount: rawRows,
        configuration: configuration)
    }
  case .compressed(let compressionRatio):
    let emitted = emitCompressedRows(
      prefix: "\(prefix).attn.compressor", compressionRatio: compressionRatio,
      headDim: configuration.attentionHeadDim, emitIndexerWHT: false)
    let emittedCompressed = DeepSeek4FP8KVRoundTrip(
      emitted.rows, rowCount: emitted.emittedRowCount,
      headDim: configuration.attentionHeadDim,
      rotaryDim: configuration.rotaryDim, enabled: configuration.enableFP8KVRoundTrip)
    let attentionCompressed: Model.IO
    if let cacheInputs = cacheInputs {
      guard let compressedKeyValue = cacheInputs.compressedKeyValue else {
        preconditionFailure("Compressed attention requires a compressed KV cache.")
      }
      let composed = DeepSeek4ComposeCache(
        emittedCompressed.to(.Float16), cache: compressedKeyValue,
        existingRowCount: emitted.existingRowCount,
        currentRowCount: emitted.emittedRowCount,
        width: configuration.attentionHeadDim)
      attentionCompressed = composed
    } else {
      attentionCompressed = emittedCompressed
    }
    guard let causalCompressedIndices = causalCompressedIndices else {
      preconditionFailure("Compressed attention requires causal compressed indices.")
    }
    heads = DeepSeek4SparseIndexedAttention(
      query: projection.query, rawKV: attentionRawKV, compressedKV: attentionCompressed,
      selectedCompressedRows: causalCompressedIndices, sinks: sinks,
      queryLength: tokenLength, rawRowCount: rawRows,
      compressedRows: emitted.totalRowCount,
      selectedRowCount: max(emitted.totalRowCount, 1),
      configuration: configuration)
  case .indexed(let compressionRatio):
    let emitted = emitCompressedRows(
      prefix: "\(prefix).attn.compressor", compressionRatio: compressionRatio,
      headDim: configuration.attentionHeadDim, emitIndexerWHT: false)
    let emittedCompressed = DeepSeek4FP8KVRoundTrip(
      emitted.rows, rowCount: emitted.emittedRowCount,
      headDim: configuration.attentionHeadDim,
      rotaryDim: configuration.rotaryDim, enabled: configuration.enableFP8KVRoundTrip)
    let emittedIndexer = emitCompressedRows(
      prefix: "\(prefix).attn.indexer.compressor", compressionRatio: compressionRatio,
      headDim: configuration.indexerHeadDim, emitIndexerWHT: true)
    let attentionCompressed: Model.IO
    let attentionIndexer: Model.IO
    if let cacheInputs = cacheInputs {
      guard let compressedKeyValue = cacheInputs.compressedKeyValue,
        let indexerKeyValue = cacheInputs.indexerKeyValue
      else {
        preconditionFailure("Indexed attention requires compressed and indexer KV caches.")
      }
      let compressedComposition = DeepSeek4ComposeCache(
        emittedCompressed.to(.Float16), cache: compressedKeyValue,
        existingRowCount: emitted.existingRowCount,
        currentRowCount: emitted.emittedRowCount,
        width: configuration.attentionHeadDim)
      let indexerComposition = DeepSeek4ComposeCache(
        emittedIndexer.rows.to(.Float32), cache: indexerKeyValue,
        existingRowCount: emittedIndexer.existingRowCount,
        currentRowCount: emittedIndexer.emittedRowCount,
        width: configuration.indexerHeadDim)
      attentionCompressed = compressedComposition
      attentionIndexer = indexerComposition
    } else {
      attentionCompressed = emittedCompressed
      attentionIndexer = emittedIndexer.rows
    }
    let selectedRows = DeepSeek4IndexerSelection(
      prefix: "\(prefix).attn", queryRank: projection.queryRank, attnNorm: attnBranch,
      rotary: rotary, indexerKV: attentionIndexer, tokenLength: tokenLength,
      compressionRatio: compressionRatio, compressedRows: emittedIndexer.totalRowCount,
      configuration: configuration, of: dataType)
    heads = DeepSeek4SparseIndexedAttention(
      query: projection.query, rawKV: attentionRawKV, compressedKV: attentionCompressed,
      selectedCompressedRows: selectedRows, sinks: sinks,
      queryLength: tokenLength, rawRowCount: rawRows,
      compressedRows: emitted.totalRowCount,
      selectedRowCount: configuration.indexerTopK,
      configuration: configuration)
  }
  if let preparedCompressor = preparedCompressor {
    heads = DeepSeek4CompressorStateBarrier(
      heads, stateUpdate: preparedCompressor.stateUpdate)
  }
  let attnOutput = DeepSeek4AttentionOutput(
    prefix: "\(prefix).attn", heads: heads, rotary: rotary,
    attentionOutputGroupCounts: attentionOutputGroupCounts, tokenLength: tokenLength,
    configuration: configuration)
  let afterAttn = DeepSeek4HCExpand(
    block: attnOutput, residualHC: residualHC, post: attnParts.post, comb: attnParts.comb,
    tokenLength: tokenLength, configuration: configuration)

  let ffnFlat = RMSNorm(epsilon: 1.0e-6, axis: [1], elementwiseAffine: false)(
    afterAttn.reshaped([tokenLength, hcDim])
  ).reshaped([tokenLength, hcDim])
  let ffnMix = ffnHC(ffnFlat).reshaped([tokenLength, mixDim])
  let ffnParts = DeepSeek4HCSplitWeightedSum(
    mix: ffnMix, scale: ffnScale, base: ffnBase, residualHC: afterAttn,
    tokenLength: tokenLength,
    configuration: configuration)
  let ffnNorm = RMSNorm(epsilon: 1.0e-6, axis: [1], name: "\(prefix).ffn_norm")(
    ffnParts.weighted.reshaped([tokenLength, hidden])
  ).reshaped([tokenLength, hidden])
  let ffnBranch = ffnNorm

  let routed = DeepSeek4RoutedMoE(
    prefix: "\(prefix).ffn", x: ffnBranch, routerInput: ffnNorm,
    selectedExpertOverride: selectedExpertOverride,
    selectedProbabilityIndexOverride: selectedProbabilityIndexOverride, pairToToken: pairToToken,
    tokenLength: tokenLength, routerKind: routerKind, configuration: configuration, of: dataType)
  let shared = DeepSeek4SharedFFN(
    prefix: "\(prefix).ffn", x: ffnBranch, tokenLength: tokenLength, configuration: configuration)
  let ffnBlock = routed + shared
  return DeepSeek4HCExpand(
    block: ffnBlock, residualHC: afterAttn, post: ffnParts.post,
    comb: ffnParts.comb, tokenLength: tokenLength, configuration: configuration)
}

private func DeepSeek4OutputHead<FloatType: TensorNumeric>(
  x: Model.IO, tokenLength: Int, configuration: DeepSeek4ModelConfiguration,
  includeLogits: Bool, lastTokenOnly: Bool, of dataType: FloatType.Type
) -> Model.IO {
  let hc = configuration.hcCount
  let hidden = configuration.hiddenSize
  let hcDim = hc * hidden
  let outputTokenCount = lastTokenOnly ? 1 : tokenLength
  let flatInput = x.reshaped([tokenLength, hcDim])
  let outputFlatInput =
    lastTokenOnly
    ? flatInput.reshaped([1, hcDim], offset: [tokenLength - 1, 0], strides: [hcDim, 1])
      .copied()
    : flatInput
  let outputInput = outputFlatInput.reshaped([outputTokenCount, hc, hidden])
  let hcFn = Dense(count: hc, noBias: true, name: "hc_head_fn")
  let hcScale = Parameter<Float>(.GPU(0), .C(1), name: "hc_head_scale")
  let hcBase = Parameter<Float>(.GPU(0), .C(hc), name: "hc_head_base")
  let flat = RMSNorm(epsilon: 1.0e-6, axis: [1], elementwiseAffine: false)(
    outputFlatInput
  ).reshaped([outputTokenCount, hcDim])
  let mix = hcFn(flat).reshaped([outputTokenCount, hc])
  let weights =
    (mix .* hcScale.reshaped([1, 1]).to(of: mix) + hcBase.reshaped([1, hc]).to(of: mix))
    .sigmoid() + 1.0e-6
  let hiddenState = (outputInput .* weights.reshaped([outputTokenCount, hc, 1]))
    .reduced(.sum, axis: [1])
    .reshaped([outputTokenCount, hidden])
  let norm = RMSNorm(epsilon: 1.0e-6, axis: [1], name: "norm")
  var out = norm(hiddenState)
  if includeLogits {
    let head = Dense(count: configuration.vocabularySize, noBias: true, name: "head")
    out = head(out)
  }
  return lastTokenOnly
    ? out.reshaped([includeLogits ? configuration.vocabularySize : hidden])
    : out
}

private func DeepSeek4Prefix<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, tokenLength: Int, cachedTokenLength: Int,
  configuration: DeepSeek4ModelConfiguration, useKVCache: Bool
) -> (inputs: [Input], hidden: Model.IO) {
  precondition(tokenLength > 0)
  precondition(
    cachedTokenLength == 0 || useKVCache,
    "DeepSeek4 continuation requires KV-cache inputs.")
  let tokens = Input()
  let rawRotary = Input()
  let usesCompressedRotary = configuration.layerAttentionKinds.contains {
    if case .raw = $0 { return false }
    return true
  }
  let compressedRotary = usesCompressedRotary ? Input() : nil
  var inputs: [Input] = [tokens, rawRotary]
  if let compressedRotary = compressedRotary {
    inputs.append(compressedRotary)
  }
  var compressorRotaries = [Int: Input]()
  if useKVCache {
    for compressionRatio in configuration.compressionRatios {
      let compressorRotary = Input()
      compressorRotaries[compressionRatio] = compressorRotary
      inputs.append(compressorRotary)
    }
  }
  let attentionOutputGroupCounts = Input()
  inputs.append(attentionOutputGroupCounts)
  let pairToToken = Input()
  inputs.append(pairToToken)
  var out = DeepSeek4Embedding(
    Float16.self, tokens: tokens, tokenLength: tokenLength, configuration: configuration
  ).to(.Float32)
  var causalCompressedIndices = [Int: Input]()

  for layerIndex in 0..<configuration.layers {
    let prefix = "layers.\(layerIndex)"
    let attentionKind = configuration.attentionKind(layerIndex: layerIndex)
    let layerRotary: Input
    switch attentionKind {
    case .raw:
      layerRotary = rawRotary
    case .compressed, .indexed:
      guard let compressedRotary = compressedRotary else {
        preconditionFailure("Compressed attention requires rotary inputs.")
      }
      layerRotary = compressedRotary
    }
    let layerCausalCompressedIndices = DeepSeek4CausalCompressedIndicesInput(
      layerIndex: layerIndex, configuration: configuration,
      inputs: &inputs, indices: &causalCompressedIndices)
    let layerCompressorRotary: Input?
    switch attentionKind {
    case .raw:
      layerCompressorRotary = nil
    case .compressed(let compressionRatio), .indexed(let compressionRatio):
      layerCompressorRotary = compressorRotaries[compressionRatio]
    }
    let layerCacheInputs: DeepSeek4LayerCacheInputs?
    if useKVCache {
      let rawKeyValue = Input()
      inputs.append(rawKeyValue)
      switch attentionKind {
      case .raw:
        layerCacheInputs = DeepSeek4LayerCacheInputs(
          rawKeyValue: rawKeyValue, compressedKeyValue: nil,
          compressorState: nil, nextCompressorState: nil,
          indexerKeyValue: nil)
      case .compressed:
        let compressedKeyValue = Input()
        let compressorState = Input()
        let nextCompressorState = Input()
        inputs.append(contentsOf: [
          compressedKeyValue, compressorState, nextCompressorState,
        ])
        layerCacheInputs = DeepSeek4LayerCacheInputs(
          rawKeyValue: rawKeyValue, compressedKeyValue: compressedKeyValue,
          compressorState: compressorState,
          nextCompressorState: nextCompressorState, indexerKeyValue: nil)
      case .indexed:
        let compressedKeyValue = Input()
        let compressorState = Input()
        let nextCompressorState = Input()
        let indexerKeyValue = Input()
        inputs.append(contentsOf: [
          compressedKeyValue, compressorState, nextCompressorState,
          indexerKeyValue,
        ])
        layerCacheInputs = DeepSeek4LayerCacheInputs(
          rawKeyValue: rawKeyValue, compressedKeyValue: compressedKeyValue,
          compressorState: compressorState,
          nextCompressorState: nextCompressorState,
          indexerKeyValue: indexerKeyValue)
      }
    } else {
      layerCacheInputs = nil
    }
    let selectedExperts: Input?
    let selectedProbabilityIndices: Input?
    if configuration.routerKind(layerIndex: layerIndex) == .tokenHash {
      let selectedExpertsInput = Input()
      let selectedProbabilityIndicesInput = Input()
      selectedExperts = selectedExpertsInput
      selectedProbabilityIndices = selectedProbabilityIndicesInput
      inputs.append(selectedExpertsInput)
      inputs.append(selectedProbabilityIndicesInput)
    } else {
      selectedExperts = nil
      selectedProbabilityIndices = nil
    }
    let layer = DeepSeek4Layer(
      prefix: prefix, layerIndex: layerIndex, residualHC: out, rotary: layerRotary,
      compressorRotary: layerCompressorRotary,
      attentionOutputGroupCounts: attentionOutputGroupCounts,
      causalCompressedIndices: layerCausalCompressedIndices,
      selectedExpertOverride: selectedExperts,
      selectedProbabilityIndexOverride: selectedProbabilityIndices, pairToToken: pairToToken,
      tokenLength: tokenLength, cachedTokenLength: cachedTokenLength,
      cacheInputs: layerCacheInputs, configuration: configuration, of: dataType)
    out = layer.to(.Float32).copied()
  }

  return (inputs, out)
}

public func DeepSeek4CausalLM<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, tokenLength: Int, cachedTokenLength: Int = 0,
  configuration: DeepSeek4ModelConfiguration = .deepSeekV4Flash,
  includeLogits: Bool = true, lastTokenOnly: Bool = false,
  useKVCache: Bool = false
) -> Model {
  let prefix = DeepSeek4Prefix(
    dataType, tokenLength: tokenLength, cachedTokenLength: cachedTokenLength,
    configuration: configuration, useKVCache: useKVCache)
  let output = DeepSeek4OutputHead(
    x: prefix.hidden, tokenLength: tokenLength, configuration: configuration,
    includeLogits: includeLogits, lastTokenOnly: lastTokenOnly, of: dataType
  ).to(FloatType.dataType).copied()
  return Model(prefix.inputs, [output])
}

public func DeepSeek4PrefixHiddenState<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, tokenLength: Int, cachedTokenLength: Int = 0,
  configuration: DeepSeek4ModelConfiguration = .deepSeekV4Flash,
  useKVCache: Bool = false
) -> Model {
  let prefix = DeepSeek4Prefix(
    dataType, tokenLength: tokenLength, cachedTokenLength: cachedTokenLength,
    configuration: configuration, useKVCache: useKVCache)
  return Model(prefix.inputs, [prefix.hidden])
}
