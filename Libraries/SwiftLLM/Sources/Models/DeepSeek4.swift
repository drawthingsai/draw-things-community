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
    ] + (4..<43).map { $0.isMultiple(of: 2) ? .indexed(compressionRatio: 4) : .compressed(compressionRatio: 128) },
    layerRouterKinds: (0..<43).map { $0 < 3 ? .tokenHash : .standard })
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
    let storeTensorName = DeepSeek4StoreUsesBareTensorName(tensorName)
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
    offset: [0, fullHeadDim - configuration.rotaryDim],
    strides: [fullHeadDim, 1]).copied()
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
  let rotaryRows = rotary.reshaped([tokenLength, fullHeadDim])
  let fullRows = rotaryRows.reshaped(
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
  query: Model.IO, rawKV: Model.IO, sinks: ModelIOConvertible, tokenLength: Int,
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
    attentionQuery.reshaped(.NHWC(1, tokenLength, configuration.attentionHeads, headDim)),
    kv.reshaped(.NHWC(1, tokenLength, 1, headDim)),
    kv.reshaped(.NHWC(1, tokenLength, 1, headDim)),
    sinks.reshaped(.NHWC(1, 1, configuration.attentionHeads, 1)).to(of: attentionQuery)
  ).reshaped([tokenLength, configuration.attentionHeads * headDim])
}

private func DeepSeek4SparseIndexedAttention(
  query: Model.IO, rawKV: Model.IO, compressedKV: Model.IO, selectedCompressedRows: Model.IO,
  sinks: ModelIOConvertible, tokenLength: Int, compressedRows: Int, selectedRowCount: Int,
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
  return attention(
    attentionQuery.reshaped(
      .NHWC(1, tokenLength, configuration.attentionHeads, headDim)),
    rawAttentionKV.reshaped(.NHWC(1, tokenLength, 1, headDim)),
    rawAttentionKV.reshaped(.NHWC(1, tokenLength, 1, headDim)),
    compressedAttentionKVInput,
    compressedAttentionKVInput,
    selectedCompressedRows.reshaped(
      [tokenLength, selectedRowCount], format: .NHWC),
    sinks.reshaped(.NHWC(1, 1, configuration.attentionHeads, 1)).to(of: attentionQuery)
  ).reshaped([tokenLength, configuration.attentionHeads * headDim])
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
  let kv = kvProjected.reshaped([windowCount, compressionRatio, width])
  let score = scoreProjected.reshaped([windowCount, compressionRatio, width])
    + ape.reshaped([1, compressionRatio, width]).to(of: scoreProjected)
  let primaryKV = kv.reshaped(
    [windowCount, compressionRatio, headDim], offset: [0, 0, 0],
    strides: [rowWidth, width, 1]
  ).transposed(1, 2).contiguous()
  let primaryScore = score.reshaped(
    [windowCount, compressionRatio, headDim], offset: [0, 0, 0],
    strides: [rowWidth, width, 1]
  ).transposed(1, 2).contiguous()
  let companionKV = kv.reshaped(
    [windowCount, compressionRatio, headDim], offset: [0, 0, headDim],
    strides: [rowWidth, width, 1]
  ).transposed(1, 2).contiguous()
  let companionScore = score.reshaped(
    [windowCount, compressionRatio, headDim], offset: [0, 0, headDim],
    strides: [rowWidth, width, 1]
  ).transposed(1, 2).contiguous()
  let zeroKV = ape.reshaped([1, compressionRatio, width]).reshaped(
    [1, compressionRatio, headDim], offset: [0, 0, 0],
    strides: [rowWidth, width, 1]
  ).transposed(1, 2).contiguous() * 0
  let negInfScore = zeroKV - 1.0e4
  let previousKV = Concat(axis: 0)([zeroKV, primaryKV])
  let previousScore = Concat(axis: 0)([negInfScore, primaryScore])
  let currentKV = Concat(axis: 0)([companionKV, zeroKV])
  let currentScore = Concat(axis: 0)([companionScore, negInfScore])
  let paddedKV = Concat(axis: 2)([previousKV, currentKV])
  let paddedScore = Concat(axis: 2)([previousScore, currentScore])
  let pooledWidth = 2 * compressionRatio
  let allKV = paddedKV.reshaped(
    [windowCount, headDim, pooledWidth], offset: [0, 0, 0],
    strides: [headDim * pooledWidth, pooledWidth, 1])
  let allScore = paddedScore.reshaped(
    [windowCount, headDim, pooledWidth], offset: [0, 0, 0],
    strides: [headDim * pooledWidth, pooledWidth, 1])
  let rows = windowCount * headDim
  let weights = allScore.reshaped([rows, pooledWidth])
    .softmax()
    .reshaped([windowCount, headDim, pooledWidth])
  return (weights .* allKV).reduced(.sum, axis: [2]).reshaped([windowCount, headDim])
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
  let kvProjected = kv(xWindow).reshaped([tokenRows, width])
  let scoreProjected = gate(xWindow).reshaped([tokenRows, width])
  let pooled: Model.IO
  if compressionRatio == 4 {
    pooled = DeepSeek4Ratio4RollingPool(
      kvProjected: kvProjected, scoreProjected: scoreProjected, ape: ape,
      windowCount: windowCount, headDim: headDim)
  } else {
    let kvRows = kvProjected.reshaped([windowCount, compressionRatio, headDim])
    let scores = scoreProjected.reshaped([windowCount, compressionRatio, headDim])
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
    left: normed.reshaped([windowCount, headDim]),
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
  let indexWeights = indexerWeightsProj(attnNorm.reshaped([tokenLength, configuration.hiddenSize]))
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
  return (selectedProbs .* selectedProbs.reduced(.sum, axis: [1]).reshaped([tokenLength, 1])
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
  let gathered = IndexSelect()(x.reshaped([tokenLength, configuration.hiddenSize]), sortedTokenIndices)
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
  let hidden = Functional.swishMul(value: sortedUp, gate: sortedGate)
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
    FloatType.self, vocabularySize: configuration.vocabularySize, embeddingSize: configuration.hiddenSize,
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
  attentionOutputGroupCounts: Model.IO, causalCompressedIndices: Model.IO?,
  selectedExpertOverride: Model.IO?,
  selectedProbabilityIndexOverride: Model.IO?, pairToToken: Model.IO, tokenLength: Int,
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
  let sinks = Parameter<Float>(.GPU(0), .C(configuration.attentionHeads), name: "\(prefix).attn.attn_sink")

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
  let heads: Model.IO
  switch attentionKind {
  case .raw:
    heads = DeepSeek4RawAttention(
      query: projection.query, rawKV: rawKV, sinks: sinks, tokenLength: tokenLength,
      configuration: configuration)
  case .compressed(let compressionRatio):
    let compressedRows = tokenLength / compressionRatio
    let compressed = DeepSeek4FP8KVRoundTrip(
      DeepSeek4Compressor(
        prefix: "\(prefix).attn.compressor", x: attnBranch, rotary: rotary,
        tokenLength: tokenLength, compressionRatio: compressionRatio,
        headDim: configuration.attentionHeadDim, emitIndexerWHT: false,
        configuration: configuration, of: dataType),
      rowCount: compressedRows, headDim: configuration.attentionHeadDim,
      rotaryDim: configuration.rotaryDim, enabled: configuration.enableFP8KVRoundTrip)
    precondition(causalCompressedIndices != nil)
    heads = DeepSeek4SparseIndexedAttention(
      query: projection.query, rawKV: rawKV, compressedKV: compressed,
      selectedCompressedRows: causalCompressedIndices!, sinks: sinks, tokenLength: tokenLength,
      compressedRows: compressedRows, selectedRowCount: max(compressedRows, 1),
      configuration: configuration)
  case .indexed(let compressionRatio):
    let compressedRows = tokenLength / compressionRatio
    let compressed = DeepSeek4FP8KVRoundTrip(
      DeepSeek4Compressor(
        prefix: "\(prefix).attn.compressor", x: attnBranch, rotary: rotary,
        tokenLength: tokenLength, compressionRatio: compressionRatio,
        headDim: configuration.attentionHeadDim, emitIndexerWHT: false,
        configuration: configuration, of: dataType),
      rowCount: compressedRows, headDim: configuration.attentionHeadDim,
      rotaryDim: configuration.rotaryDim, enabled: configuration.enableFP8KVRoundTrip)
    let indexer = DeepSeek4Compressor(
      prefix: "\(prefix).attn.indexer.compressor", x: attnBranch, rotary: rotary,
      tokenLength: tokenLength, compressionRatio: compressionRatio,
      headDim: configuration.indexerHeadDim, emitIndexerWHT: true,
      configuration: configuration, of: dataType)
    let selectedRows = DeepSeek4IndexerSelection(
      prefix: "\(prefix).attn", queryRank: projection.queryRank, attnNorm: attnBranch,
      rotary: rotary, indexerKV: indexer, tokenLength: tokenLength,
      compressionRatio: compressionRatio, compressedRows: compressedRows,
      configuration: configuration, of: dataType)
    heads = DeepSeek4SparseIndexedAttention(
      query: projection.query, rawKV: rawKV, compressedKV: compressed,
      selectedCompressedRows: selectedRows, sinks: sinks, tokenLength: tokenLength,
      compressedRows: compressedRows, selectedRowCount: configuration.indexerTopK,
      configuration: configuration)
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
  let weights = (mix .* hcScale.reshaped([1, 1]).to(of: mix) + hcBase.reshaped([1, hc]).to(of: mix))
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
  configuration: DeepSeek4ModelConfiguration
) -> (inputs: [Input], hidden: Model.IO) {
  precondition(tokenLength > 0)
  precondition(cachedTokenLength == 0, "The initial DeepSeek4 prefill draft handles fresh prefill only.")
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
      layerRotary = compressedRotary!
    }
    let layerCausalCompressedIndices = DeepSeek4CausalCompressedIndicesInput(
      layerIndex: layerIndex, configuration: configuration,
      inputs: &inputs, indices: &causalCompressedIndices)
    let selectedExperts: Input?
    let selectedProbabilityIndices: Input?
    if configuration.routerKind(layerIndex: layerIndex) == .tokenHash {
      selectedExperts = Input()
      selectedProbabilityIndices = Input()
      inputs.append(selectedExperts!)
      inputs.append(selectedProbabilityIndices!)
    } else {
      selectedExperts = nil
      selectedProbabilityIndices = nil
    }
    let layer = DeepSeek4Layer(
      prefix: prefix, layerIndex: layerIndex, residualHC: out, rotary: layerRotary,
      attentionOutputGroupCounts: attentionOutputGroupCounts,
      causalCompressedIndices: layerCausalCompressedIndices,
      selectedExpertOverride: selectedExperts,
      selectedProbabilityIndexOverride: selectedProbabilityIndices, pairToToken: pairToToken,
      tokenLength: tokenLength, configuration: configuration, of: dataType)
    out = layer.to(.Float32).copied()
  }

  return (inputs, out)
}

public func DeepSeek4CausalLM<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, tokenLength: Int, cachedTokenLength: Int = 0,
  configuration: DeepSeek4ModelConfiguration = .deepSeekV4Flash,
  includeLogits: Bool = true, lastTokenOnly: Bool = false
) -> Model {
  let prefix = DeepSeek4Prefix(
    dataType, tokenLength: tokenLength, cachedTokenLength: cachedTokenLength,
    configuration: configuration)
  let output = DeepSeek4OutputHead(
    x: prefix.hidden, tokenLength: tokenLength, configuration: configuration,
    includeLogits: includeLogits, lastTokenOnly: lastTokenOnly, of: dataType
  ).to(FloatType.dataType).copied()
  return Model(prefix.inputs, [output])
}

public func DeepSeek4PrefixHiddenState<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, tokenLength: Int, cachedTokenLength: Int = 0,
  configuration: DeepSeek4ModelConfiguration = .deepSeekV4Flash
) -> Model {
  let prefix = DeepSeek4Prefix(
    dataType, tokenLength: tokenLength, cachedTokenLength: cachedTokenLength,
    configuration: configuration)
  return Model(prefix.inputs, [prefix.hidden])
}
