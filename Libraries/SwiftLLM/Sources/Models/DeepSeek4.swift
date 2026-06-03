import Foundation
import NNC

public let DeepSeek4StoreLoadCodec: DynamicGraph.Store.Codec = [.jit, .i8x, .ezm7]

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

  public init(
    vocabularySize: Int, hiddenSize: Int, layers: Int, hcCount: Int,
    attentionHeads: Int, attentionHeadDim: Int, rotaryDim: Int, rawWindow: Int,
    expertCount: Int, routedExperts: Int, expertIntermediateSize: Int,
    sharedIntermediateSize: Int, attentionOutputGroups: Int, attentionLowRank: Int,
    queryLowRank: Int, indexerHeads: Int, indexerHeadDim: Int, indexerTopK: Int,
    ropeTheta: Double, ropeScaleFactor: Double, ropeOriginalContext: Int,
    ropeYarnBetaFast: Double, ropeYarnBetaSlow: Double,
    layerAttentionKinds: [DeepSeek4AttentionKind],
    layerRouterKinds: [DeepSeek4RouterKind]
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
      || name.hasSuffix(".attn.wo_a.group_counts")
      || name.hasSuffix(".attn.wo_a.group_ids")
      || name.hasSuffix(".attn.fp4.scale_candidates")
      || name.hasSuffix(".attn.fp4.e2m1_candidates")
      || name.hasSuffix(".compressor.ape")
      || name.hasSuffix(".ffn.gate.bias")
      || name.hasSuffix(".ffn.gate.tid2eid")
  }
}

public func DeepSeek4StoreReader(
  storeKey: String = "deepseek4"
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
  configuration: DeepSeek4ModelConfiguration = .deepSeekV4Flash,
  of dataType: FloatType.Type = FloatType.self
) -> Tensor<FloatType> {
  let headDim = configuration.attentionHeadDim
  let nRot = configuration.rotaryDim
  let nNope = headDim - nRot
  let freqScale = 1.0 / configuration.ropeScaleFactor
  let extFactor = 1.0
  let attnFactor = 1.0 / (1.0 + 0.1 * log(1.0 / freqScale))
  let corr = DeepSeek4RopeYarnCorrDims(
    nDims: nRot, originalContext: configuration.ropeOriginalContext,
    base: configuration.ropeTheta, betaFast: configuration.ropeYarnBetaFast,
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
      let freq = pow(configuration.ropeTheta, -Double(i) / Double(nRot))
      let thetaExtrap = position * freq
      let thetaInterp = freqScale * thetaExtrap
      let rampMix = DeepSeek4RopeYarnRamp(low: corr.0, high: corr.1, index: i) * extFactor
      let theta = thetaInterp * (1.0 - rampMix) + thetaExtrap * rampMix
      let mscale = attnFactor * (1.0 + 0.1 * log(1.0 / freqScale))
      let offset = nNope + i
      rotary[0, row, 0, offset] = FloatType(cos(theta) * mscale)
      rotary[0, row, 0, offset + 1] = FloatType(sin(theta) * mscale)
    }
  }
  return rotary
}

private func DeepSeek4ClampedSwiGLU(gate: Model.IO, up: Model.IO) -> Model.IO {
  return gate.clamped(...10.0).swish() .* up.clamped((-10.0)...10.0)
}

private func DeepSeek4RotaryConjugateRows(
  _ rotary: Model.IO, rowCount: Int, headDim: Int
) -> Model.IO {
  let pairCount = headDim / 2
  let pairs = rotary.reshaped([rowCount, pairCount, 2])
  let real = pairs.reshaped(
    [rowCount, pairCount, 1], offset: [0, 0, 0], strides: [headDim, 2, 1])
  let imag = pairs.reshaped(
    [rowCount, pairCount, 1], offset: [0, 0, 1], strides: [headDim, 2, 1]) * -1.0
  return Concat(axis: 2)([real, imag]).reshaped([rowCount, headDim])
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
  let weights = inverse
    ? DeepSeek4RotaryConjugateRows(rotaryRows, rowCount: rowCount, headDim: headDim)
    : rotaryRows
  return Functional.cmul(
    left: x.reshaped([rowCount, heads, headDim]),
    right: weights.reshaped([rowCount, 1, headDim])
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
    strides: [compressionRatio * fullHeadDim, 1])
  return DeepSeek4RotaryRowsForHeadDim(
    fullRows, rowCount: windowCount, headDim: headDim, configuration: configuration
  ).contiguous()
}

private func DeepSeek4SinkhornNormalize(
  _ comb: Model.IO, tokenLength: Int, hc: Int, remainingIterations: Int
) -> Model.IO {
  guard remainingIterations > 0 else {
    return comb
  }
  let rowNormalized = comb
    .* (comb.reduced(.sum, axis: [2]).reshaped([tokenLength, hc, 1]) + 1.0e-6).reciprocal()
  let columnNormalized = rowNormalized
    .* (rowNormalized.reduced(.sum, axis: [1]).reshaped([tokenLength, 1, hc]) + 1.0e-6)
      .reciprocal()
  return DeepSeek4SinkhornNormalize(
    columnNormalized, tokenLength: tokenLength, hc: hc,
    remainingIterations: remainingIterations - 1)
}

private func DeepSeek4HCSplit(
  mix: Model.IO, scale: ModelIOConvertible, base: ModelIOConvertible, tokenLength: Int,
  configuration: DeepSeek4ModelConfiguration
) -> (pre: Model.IO, post: Model.IO, comb: Model.IO) {
  let hc = configuration.hcCount
  let mixDim = configuration.hcMixDim
  let preScale = scale.reshaped([1], offset: [0], strides: [1]).reshaped([1, 1]).to(of: mix)
  let postScale = scale.reshaped([1], offset: [1], strides: [1]).reshaped([1, 1]).to(of: mix)
  let combScale = scale.reshaped([1], offset: [2], strides: [1]).reshaped([1, 1]).to(of: mix)
  let preBase = base.reshaped([hc], offset: [0], strides: [1]).reshaped([1, hc]).to(of: mix)
  let postBase = base.reshaped([hc], offset: [hc], strides: [1]).reshaped([1, hc]).to(of: mix)
  let combBase = base.reshaped([hc * hc], offset: [2 * hc], strides: [1])
    .reshaped([1, hc * hc])
    .to(of: mix)
  let pre = (mix.reshaped([tokenLength, hc], offset: [0, 0], strides: [mixDim, 1])
    .* preScale
    + preBase).sigmoid() + 1.0e-6
  let post = 2.0 * (mix.reshaped([tokenLength, hc], offset: [0, hc], strides: [mixDim, 1])
    .* postScale
    + postBase).sigmoid()
  var comb = (mix.reshaped([tokenLength, hc * hc], offset: [0, 2 * hc], strides: [mixDim, 1])
    .* combScale
    + combBase)
    .reshaped([tokenLength, hc, hc])
  comb = comb.softmax() + 1.0e-6
  comb = comb .* (comb.reduced(.sum, axis: [1]).reshaped([tokenLength, 1, hc]) + 1.0e-6).reciprocal()
  comb = DeepSeek4SinkhornNormalize(
    comb, tokenLength: tokenLength, hc: hc, remainingIterations: 19)
  return (pre, post, comb.reshaped([tokenLength, hc * hc]))
}

private func DeepSeek4HCWeightedSum(
  _ residualHC: Model.IO, weights: Model.IO, tokenLength: Int,
  configuration: DeepSeek4ModelConfiguration
) -> Model.IO {
  return (residualHC.reshaped([tokenLength, configuration.hcCount, configuration.hiddenSize])
    .* weights.reshaped([tokenLength, configuration.hcCount, 1]))
    .reduced(.sum, axis: [1])
}

private func DeepSeek4HCExpand(
  block: Model.IO, residualHC: Model.IO, post: Model.IO, comb: Model.IO,
  tokenLength: Int, configuration: DeepSeek4ModelConfiguration
) -> Model.IO {
  let hc = configuration.hcCount
  let hidden = configuration.hiddenSize
  let block = block.to(of: residualHC)
  return block.reshaped([tokenLength, 1, hidden]) .* post.reshaped([tokenLength, hc, 1])
    + Matmul(transposeA: (1, 2))(
      comb.reshaped([tokenLength, hc, hc]),
      residualHC.reshaped([tokenLength, hc, hidden]))
}

private func DeepSeek4AttentionProjection(
  prefix: String, x: Model.IO, rotary: Model.IO, tokenLength: Int,
  configuration: DeepSeek4ModelConfiguration
) -> (query: Model.IO, keyValue: Model.IO, queryRank: Model.IO) {
  let headDim = configuration.attentionHeadDim
  let heads = configuration.attentionHeads
  let wqA = Dense(count: configuration.queryLowRank, noBias: true, name: "\(prefix).wq_a")
  let wqB = Dense(count: heads * headDim, noBias: true, name: "\(prefix).wq_b")
  let wkv = Dense(count: headDim, noBias: true, name: "\(prefix).wkv")

  let queryRank = RMSNorm(epsilon: 1.0e-6, axis: [1], name: "\(prefix).q_norm")(
    wqA(x.reshaped([tokenLength, configuration.hiddenSize]))
  ).reshaped([tokenLength, configuration.queryLowRank])
  let qDense = wqB(queryRank).reshaped([tokenLength, heads, headDim])
  let qNorm = RMSNorm(epsilon: 1.0e-6, axis: [2], elementwiseAffine: false)(
    qDense
  ).reshaped([tokenLength, heads * headDim])
  let query = DeepSeek4RopeTailRows(
    qNorm, rotary: rotary, rowCount: tokenLength, heads: heads, headDim: headDim,
    inverse: false, configuration: configuration)

  let kvRaw = wkv(x.reshaped([tokenLength, configuration.hiddenSize])).reshaped([tokenLength, headDim])
  let kvNorm = RMSNorm(epsilon: 1.0e-6, axis: [1], name: "\(prefix).kv_norm")(
    kvRaw
  ).reshaped([tokenLength, headDim])
  let keyValue = DeepSeek4RopeTailRows(
    kvNorm, rotary: rotary, rowCount: tokenLength, heads: 1, headDim: headDim,
    inverse: false, configuration: configuration)
  return (query, keyValue, queryRank)
}

private func DeepSeek4AttentionOutput(
  prefix: String, heads: Model.IO, rotary: Model.IO, tokenLength: Int,
  configuration: DeepSeek4ModelConfiguration
) -> Model.IO {
  let headDim = configuration.attentionHeadDim
  let outGroups = configuration.attentionOutputGroups
  let headsPerGroup = configuration.attentionHeads / outGroups
  let groupDim = headsPerGroup * headDim
  let lowRank = configuration.attentionLowRank
  let headsBack = DeepSeek4RopeTailRows(
    heads, rotary: rotary, rowCount: tokenLength, heads: configuration.attentionHeads,
    headDim: headDim, inverse: true, configuration: configuration)
  let grouped = headsBack.reshaped([tokenLength, outGroups, groupDim])
  let groupMajor = grouped.transposed(0, 1).contiguous().reshaped([outGroups * tokenLength, groupDim])
  let groupIDs = Parameter<Int32>(
    .GPU(0), .C(outGroups), trainable: false, name: "\(prefix).wo_a.group_ids")
  let groupCounts = Parameter<Int32>(
    .GPU(0), .C(outGroups), trainable: false, name: "\(prefix).wo_a.group_counts")
  let woA = SegmentedDense(
    segments: outGroups, count: lowRank, noBias: true, name: "\(prefix).wo_a")
  let lowGrouped = woA(groupMajor, groupIDs, groupCounts)
    .reshaped([outGroups, tokenLength, lowRank])
  let low = lowGrouped.transposed(0, 1).contiguous()
    .reshaped([tokenLength, configuration.attentionOutputLowDim])
  let woB = Dense(count: configuration.hiddenSize, noBias: true, name: "\(prefix).wo_b")
  return woB(low).reshaped([tokenLength, configuration.hiddenSize])
}

private func DeepSeek4RawAttention(
  query: Model.IO, rawKV: Model.IO, sinks: ModelIOConvertible, tokenLength: Int,
  configuration: DeepSeek4ModelConfiguration
) -> Model.IO {
  let headDim = configuration.attentionHeadDim
  let attention = ScaledDotProductAttention(
    scale: 1.0 / Float(headDim).squareRoot(),
    isCausal: true,
    hasAttentionSinks: true,
    slidingWindow: configuration.rawWindow,
    name: "DeepSeek4RawAttention")
  return attention(
    query.reshaped(.NHWC(1, tokenLength, configuration.attentionHeads, headDim)),
    rawKV.reshaped(.NHWC(1, tokenLength, 1, headDim)),
    rawKV.reshaped(.NHWC(1, tokenLength, 1, headDim)),
    sinks.reshaped(.NHWC(1, 1, configuration.attentionHeads, 1))
  ).reshaped([tokenLength, configuration.attentionHeads * headDim])
}

private func DeepSeek4CompressedAttention(
  query: Model.IO, rawKV: Model.IO, compressedKV: Model.IO, mask: Model.IO,
  sinks: ModelIOConvertible,
  tokenLength: Int, compressedRows: Int, configuration: DeepSeek4ModelConfiguration
) -> Model.IO {
  let headDim = configuration.attentionHeadDim
  let keyValueRows = tokenLength + compressedRows
  let kv = Functional.concat(
    axis: 0,
    rawKV.reshaped([tokenLength, headDim]),
    compressedKV.reshaped([compressedRows, headDim]))
  let attention = ScaledDotProductAttention(
    scale: 1.0 / Float(headDim).squareRoot(),
    hasAttentionMask: true, hasAttentionSinks: true)
  return attention(
    query.reshaped(.NHWC(1, tokenLength, configuration.attentionHeads, headDim)),
    kv.reshaped(.NHWC(1, keyValueRows, 1, headDim)),
    kv.reshaped(.NHWC(1, keyValueRows, 1, headDim)),
    mask.reshaped(.NHWC(1, 1, tokenLength, keyValueRows)),
    sinks.reshaped(.NHWC(1, 1, configuration.attentionHeads, 1))
  ).reshaped([tokenLength, configuration.attentionHeads * headDim])
}

private func DeepSeek4SparseIndexedAttention(
  query: Model.IO, rawKV: Model.IO, compressedKV: Model.IO, selectedCompressedRows: Model.IO,
  sinks: ModelIOConvertible, tokenLength: Int, compressedRows: Int,
  configuration: DeepSeek4ModelConfiguration
) -> Model.IO {
  let headDim = configuration.attentionHeadDim
  let attention = SparseIndexedAttention(
    scale: 1.0 / Float(headDim).squareRoot(),
    isCausal: true, hasAttentionSinks: true)
  return attention(
    query.reshaped([tokenLength, configuration.attentionHeads, headDim]),
    rawKV.reshaped([tokenLength, headDim]),
    rawKV.reshaped([tokenLength, headDim]),
    compressedKV.reshaped([compressedRows, headDim]),
    compressedKV.reshaped([compressedRows, headDim]),
    selectedCompressedRows.reshaped([tokenLength, configuration.indexerTopK]),
    sinks.reshaped([configuration.attentionHeads])
  ).reshaped([tokenLength, configuration.attentionHeads * headDim])
}

private func DeepSeek4Ratio4RollingPool(
  kvProjected: Model.IO, scoreProjected: Model.IO, ape: ModelIOConvertible,
  windowCount: Int, headDim: Int
) -> Model.IO {
  let compressionRatio = 4
  let width = 2 * headDim
  let rowWidth = width * compressionRatio
  let blockWidth = headDim * compressionRatio
  let kv = kvProjected.reshaped([windowCount, compressionRatio, width])
  let score = scoreProjected.reshaped([windowCount, compressionRatio, width])
    + ape.reshaped([1, compressionRatio, width])
  let kvTransposed = kv.transposed(1, 2)
  let scoreTransposed = score.transposed(1, 2)
  let primaryKV = kvTransposed.reshaped(
    [windowCount, blockWidth], offset: [0, 0],
    strides: [rowWidth, 1]
  ).contiguous().reshaped([windowCount, headDim, compressionRatio])
  let primaryScore = scoreTransposed.reshaped(
    [windowCount, blockWidth], offset: [0, 0],
    strides: [rowWidth, 1]
  ).contiguous().reshaped([windowCount, headDim, compressionRatio])
  let companionKV = kvTransposed.reshaped(
    [windowCount, blockWidth], offset: [0, blockWidth],
    strides: [rowWidth, 1]
  ).contiguous().reshaped([windowCount, headDim, compressionRatio])
  let companionScore = scoreTransposed.reshaped(
    [windowCount, blockWidth], offset: [0, blockWidth],
    strides: [rowWidth, 1]
  ).contiguous().reshaped([windowCount, headDim, compressionRatio])
  let zeroKV = primaryKV.reshaped(
    [1, headDim, compressionRatio], offset: [0, 0, 0],
    strides: [headDim * compressionRatio, compressionRatio, 1]) * 0
  let negInfScore = primaryScore.reshaped(
    [1, headDim, compressionRatio], offset: [0, 0, 0],
    strides: [headDim * compressionRatio, compressionRatio, 1]) * 0 - 1.0e4
  let previousKV: Model.IO
  let previousScore: Model.IO
  if windowCount == 1 {
    previousKV = zeroKV
    previousScore = negInfScore
  } else {
    previousKV = Concat(axis: 0)([
      zeroKV,
      primaryKV.reshaped(
        [windowCount - 1, headDim, compressionRatio], offset: [0, 0, 0],
        strides: [headDim * compressionRatio, compressionRatio, 1]),
    ])
    previousScore = Concat(axis: 0)([
      negInfScore,
      primaryScore.reshaped(
        [windowCount - 1, headDim, compressionRatio], offset: [0, 0, 0],
        strides: [headDim * compressionRatio, compressionRatio, 1]),
    ])
  }
  let rows = windowCount * headDim
  let allKV = Concat(axis: 1)([
    previousKV.reshaped([rows, compressionRatio]),
    companionKV.reshaped([rows, compressionRatio]),
  ]).reshaped([windowCount, headDim, 2 * compressionRatio])
  let allScore = Concat(axis: 1)([
    previousScore.reshaped([rows, compressionRatio]),
    companionScore.reshaped([rows, compressionRatio]),
  ]).reshaped([windowCount, headDim, 2 * compressionRatio])
  let weights = allScore.reshaped([rows, 2 * compressionRatio])
    .softmax()
    .reshaped([windowCount, headDim, 2 * compressionRatio])
  return (weights .* allKV).reduced(.sum, axis: [2]).reshaped([windowCount, headDim])
}

private func DeepSeek4Compressor<FloatType: TensorNumeric>(
  prefix: String, x: Model.IO, rotary: Model.IO, tokenLength: Int, compressionRatio: Int,
  headDim: Int, emitIndexerWHT: Bool, configuration: DeepSeek4ModelConfiguration,
  of dataType: FloatType.Type
) -> Model.IO {
  let windowCount = tokenLength / compressionRatio
  precondition(windowCount > 0)
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
      + ape.reshaped([1, compressionRatio, headDim])
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
    return WalshHadamardTransform(scale: 1.0 / Float(headDim).squareRoot())(
      compressed.reshaped([windowCount, headDim])
    ).reshaped([windowCount, headDim])
  }
  return compressed.reshaped([windowCount, headDim])
}

private func DeepSeek4IndexerQAT<FloatType: TensorNumeric>(
  prefix: String, _ x: Model.IO, rowCount: Int,
  configuration: DeepSeek4ModelConfiguration, of dataType: FloatType.Type
) -> Model.IO {
  let width = configuration.indexerHeadDim
  let scaleCount = 191
  let e2m1Count = 8
  let scaleCandidates = Parameter<FloatType>(
    .GPU(0), .C(scaleCount), name: "\(prefix).fp4.scale_candidates")
  let e2m1Candidates = Parameter<FloatType>(
    .GPU(0), .C(e2m1Count), name: "\(prefix).fp4.e2m1_candidates")
  let rotated = WalshHadamardTransform(scale: 1.0 / Float(width).squareRoot())(
    x.reshaped([rowCount, width]))
  let blocks = rotated.reshaped([rowCount, 4, 32])
  let absBlocks = blocks.copied().ReLU() + (blocks.copied() * -1.0).ReLU()
  let amax = absBlocks.reduced(.max, axis: [2]).clamped(1.0e-6...)
  let threshold = (amax * (1.0 / 6.0)).reshaped([rowCount, 4, 1])
  let scaleGrid = scaleCandidates.copied().reshaped([1, 1, scaleCount])
  let scalePenalty = (threshold .* scaleGrid.reciprocal() - 1.0).ReLU() * 1.0e4
  let scaleIndex = (scaleGrid + scalePenalty).argmin(axis: 2).reshaped([rowCount * 4])
  let scale = IndexSelect()(scaleCandidates.copied().reshaped([scaleCount]), scaleIndex)
    .reshaped([rowCount, 4, 1])
  let normalized = (blocks.copied() .* scale.copied().reciprocal()).clamped((-6.0)...6.0)
  let absNormalized = normalized.copied().ReLU() + (normalized.copied() * -1.0).ReLU()
  let valueGrid = e2m1Candidates.copied().reshaped([1, 1, 1, e2m1Count])
  let valueDiff = absNormalized.copied().reshaped([rowCount, 4, 32, 1]) - valueGrid
  let absValueDiff = valueDiff.copied().ReLU() + (valueDiff.copied() * -1.0).ReLU()
  let valueIndex = absValueDiff.argmin(axis: 3).reshaped([rowCount * width])
  let dequantAbs = IndexSelect()(e2m1Candidates.copied().reshaped([e2m1Count]), valueIndex)
    .reshaped([rowCount, 4, 32])
  let signDenom = absNormalized.copied() + 1.0e-6
  let sign = normalized.copied() .* signDenom.reciprocal()
  return (dequantAbs .* sign .* scale).reshaped([rowCount, width])
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
  let mid = DeepSeek4ClampedSwiGLU(gate: gate(x), up: up(x))
  return down(mid).reshaped([tokenLength, configuration.hiddenSize])
}

private func DeepSeek4RepeatedMoEInput(
  _ x: Model.IO, routerWeights: Model.IO, tokenLength: Int,
  configuration: DeepSeek4ModelConfiguration
) -> Model.IO {
  let broadcastShape = (routerWeights.reshaped([tokenLength, configuration.routedExperts, 1]) * 0)
    + 1
  return (x.reshaped([tokenLength, 1, configuration.hiddenSize]) .* broadcastShape).reshaped([
    tokenLength * configuration.routedExperts, configuration.hiddenSize,
  ])
}

private func DeepSeek4NormalizeRouterWeights(
  _ selectedProbs: Model.IO, tokenLength: Int, configuration: DeepSeek4ModelConfiguration
) -> Model.IO {
  return (selectedProbs .* selectedProbs.reduced(.sum, axis: [1]).reshaped([tokenLength, 1])
    .reciprocal()) * 1.5
}

private func DeepSeek4RoutedMoE<FloatType: TensorNumeric>(
  prefix: String, x: Model.IO, selectedExpertOverride: Model.IO?,
  selectedProbabilityIndexOverride: Model.IO?, tokenLength: Int, routerKind: DeepSeek4RouterKind,
  configuration: DeepSeek4ModelConfiguration,
  of dataType: FloatType.Type
) -> (out: Model.IO, selected: Model.IO) {
  let router = Dense(count: configuration.expertCount, noBias: true, name: "\(prefix).gate")
  let routerBias = Parameter<FloatType>(
    .GPU(0), .C(configuration.expertCount), name: "\(prefix).gate.bias")
  let logits = router(x.reshaped([tokenLength, configuration.hiddenSize]))
  let probs = logits.softplus().squareRoot().reshaped([tokenLength, configuration.expertCount])
  let selected: Model.IO
  let routerWeights: Model.IO
  switch routerKind {
  case .standard:
    let route = (probs + routerBias.reshaped([1, configuration.expertCount]))
      .partitioned(kth: configuration.routedExperts, axis: 1, descending: true)
    // This follows the HiDream partitioned-router shape: route[0] is the selected score tensor
    // and route[1] is the selected expert id tensor.
    let selectedScores = route[0].reshaped([tokenLength, configuration.routedExperts])
    selected = route[1].reshaped([tokenLength, configuration.routedExperts])
    routerWeights = DeepSeek4NormalizeRouterWeights(
      selectedScores, tokenLength: tokenLength, configuration: configuration)
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
  let repeatedInput = DeepSeek4RepeatedMoEInput(
    x, routerWeights: routerWeights, tokenLength: tokenLength, configuration: configuration)
  let gathered = IndexSelect()(repeatedInput, sortIndices)
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
  let hidden = DeepSeek4ClampedSwiGLU(
    gate: gate(gathered, groupedExpertIds), up: up(gathered, groupedExpertIds))
    .* sortedWeights.reshaped([pairs, 1])
  let sortedOut = down(hidden, groupedExpertIds)
  let pairOut = Functional.scatterAdd(count: pairs, sortedOut, index: sortIndices)
  let out = pairOut.reshaped([tokenLength, configuration.routedExperts, configuration.hiddenSize])
    .reduced(.sum, axis: [1])
    .reshaped([tokenLength, configuration.hiddenSize])
  return (out, selected)
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
  prefix: String, layerIndex: Int, residualHC: Model.IO, tokens: Model.IO, rotary: Model.IO,
  compressedAttentionMask: Model.IO?, selectedExpertOverride: Model.IO?,
  selectedProbabilityIndexOverride: Model.IO?, tokenLength: Int,
  configuration: DeepSeek4ModelConfiguration, of dataType: FloatType.Type
) -> (
  out: Model.IO, rawKV: Model.IO, compressedKV: Model.IO?, indexerKV: Model.IO?,
  selectedExperts: Model.IO, debug: [Model.IO]
) {
  let attentionKind = configuration.attentionKind(layerIndex: layerIndex)
  let routerKind = configuration.routerKind(layerIndex: layerIndex)
  let hc = configuration.hcCount
  let hidden = configuration.hiddenSize
  let hcDim = hc * hidden
  let mixDim = configuration.hcMixDim

  let attnHC = Dense(count: mixDim, noBias: true, name: "\(prefix).hc_attn_fn")
  let attnScale = Parameter<FloatType>(.GPU(0), .C(3), name: "\(prefix).hc_attn_scale")
  let attnBase = Parameter<FloatType>(.GPU(0), .C(mixDim), name: "\(prefix).hc_attn_base")
  let sinks = Parameter<FloatType>(.GPU(0), .C(configuration.attentionHeads), name: "\(prefix).attn.attn_sink")

  let ffnHC = Dense(count: mixDim, noBias: true, name: "\(prefix).hc_ffn_fn")
  let ffnScale = Parameter<FloatType>(.GPU(0), .C(3), name: "\(prefix).hc_ffn_scale")
  let ffnBase = Parameter<FloatType>(.GPU(0), .C(mixDim), name: "\(prefix).hc_ffn_base")

  let attnFlat = RMSNorm(epsilon: 1.0e-6, axis: [1], elementwiseAffine: false)(
    residualHC.reshaped([tokenLength, hcDim])
  ).reshaped([tokenLength, hcDim])
  let attnMix = attnHC(attnFlat).reshaped([tokenLength, mixDim])
  let attnParts = DeepSeek4HCSplit(
    mix: attnMix, scale: attnScale, base: attnBase, tokenLength: tokenLength,
    configuration: configuration)
  let attnPlain = DeepSeek4HCWeightedSum(
    residualHC, weights: attnParts.pre, tokenLength: tokenLength, configuration: configuration)
  let attnNorm = RMSNorm(epsilon: 1.0e-6, axis: [1], name: "\(prefix).attn_norm")(
    attnPlain.reshaped([tokenLength, hidden])
  ).reshaped([tokenLength, hidden])
  let attnBranch = attnNorm.to(FloatType.dataType)

  let projection = DeepSeek4AttentionProjection(
    prefix: "\(prefix).attn", x: attnBranch, rotary: rotary, tokenLength: tokenLength,
    configuration: configuration)
  let rawKV = projection.keyValue.reshaped([tokenLength, configuration.attentionHeadDim])
  let compressedKV: Model.IO?
  let indexerKV: Model.IO?
  let heads: Model.IO
  switch attentionKind {
  case .raw:
    compressedKV = nil
    indexerKV = nil
    heads = DeepSeek4RawAttention(
      query: projection.query, rawKV: rawKV, sinks: sinks, tokenLength: tokenLength,
      configuration: configuration)
  case .compressed(let compressionRatio):
    let compressed = DeepSeek4Compressor(
      prefix: "\(prefix).attn.compressor", x: attnBranch, rotary: rotary,
      tokenLength: tokenLength, compressionRatio: compressionRatio,
      headDim: configuration.attentionHeadDim, emitIndexerWHT: false,
      configuration: configuration, of: dataType)
    let compressedRows = tokenLength / compressionRatio
    precondition(compressedAttentionMask != nil)
    compressedKV = compressed
    indexerKV = nil
    heads = DeepSeek4CompressedAttention(
      query: projection.query, rawKV: rawKV, compressedKV: compressed,
      mask: compressedAttentionMask!, sinks: sinks, tokenLength: tokenLength,
      compressedRows: compressedRows, configuration: configuration)
  case .indexed(let compressionRatio):
    let compressed = DeepSeek4Compressor(
      prefix: "\(prefix).attn.compressor", x: attnBranch, rotary: rotary,
      tokenLength: tokenLength, compressionRatio: compressionRatio,
      headDim: configuration.attentionHeadDim, emitIndexerWHT: false,
      configuration: configuration, of: dataType)
    let indexer = DeepSeek4Compressor(
      prefix: "\(prefix).attn.indexer.compressor", x: attnBranch, rotary: rotary,
      tokenLength: tokenLength, compressionRatio: compressionRatio,
      headDim: configuration.indexerHeadDim, emitIndexerWHT: true,
      configuration: configuration, of: dataType)
    let compressedRows = tokenLength / compressionRatio
    let selectedRows = DeepSeek4IndexerSelection(
      prefix: "\(prefix).attn", queryRank: projection.queryRank, attnNorm: attnBranch,
      rotary: rotary, indexerKV: indexer, tokenLength: tokenLength,
      compressionRatio: compressionRatio, compressedRows: compressedRows,
      configuration: configuration, of: dataType)
    compressedKV = compressed
    indexerKV = indexer
    heads = DeepSeek4SparseIndexedAttention(
      query: projection.query, rawKV: rawKV, compressedKV: compressed,
      selectedCompressedRows: selectedRows, sinks: sinks, tokenLength: tokenLength,
      compressedRows: compressedRows, configuration: configuration)
  }
  let attnOut = DeepSeek4AttentionOutput(
    prefix: "\(prefix).attn", heads: heads, rotary: rotary, tokenLength: tokenLength,
    configuration: configuration)
  let afterAttn = DeepSeek4HCExpand(
    block: attnOut, residualHC: residualHC, post: attnParts.post, comb: attnParts.comb,
    tokenLength: tokenLength, configuration: configuration)

  let ffnFlat = RMSNorm(epsilon: 1.0e-6, axis: [1], elementwiseAffine: false)(
    afterAttn.reshaped([tokenLength, hcDim])
  ).reshaped([tokenLength, hcDim])
  let ffnMix = ffnHC(ffnFlat).reshaped([tokenLength, mixDim])
  let ffnParts = DeepSeek4HCSplit(
    mix: ffnMix, scale: ffnScale, base: ffnBase, tokenLength: tokenLength,
    configuration: configuration)
  let ffnPlain = DeepSeek4HCWeightedSum(
    afterAttn, weights: ffnParts.pre, tokenLength: tokenLength, configuration: configuration)
  let ffnNorm = RMSNorm(epsilon: 1.0e-6, axis: [1], name: "\(prefix).ffn_norm")(
    ffnPlain.reshaped([tokenLength, hidden])
  ).reshaped([tokenLength, hidden])
  let ffnBranch = ffnNorm.to(FloatType.dataType)

  let routed = DeepSeek4RoutedMoE(
    prefix: "\(prefix).ffn", x: ffnBranch, selectedExpertOverride: selectedExpertOverride,
    selectedProbabilityIndexOverride: selectedProbabilityIndexOverride, tokenLength: tokenLength,
    routerKind: routerKind, configuration: configuration, of: dataType)
  let shared = DeepSeek4SharedFFN(
    prefix: "\(prefix).ffn", x: ffnBranch, tokenLength: tokenLength, configuration: configuration)
  let ffnBlock = routed.out + shared
  let nextHC = DeepSeek4HCExpand(
    block: ffnBlock, residualHC: afterAttn, post: ffnParts.post,
    comb: ffnParts.comb, tokenLength: tokenLength, configuration: configuration)
  let debug: [Model.IO] = [
    attnFlat, attnMix, attnParts.pre, attnParts.post, attnParts.comb, attnPlain, attnNorm,
    projection.query, projection.queryRank, heads, attnOut, afterAttn, ffnFlat, ffnMix,
    ffnParts.pre, ffnParts.post, ffnParts.comb, ffnPlain, ffnNorm, routed.out, shared,
    ffnBlock,
  ]
  return (nextHC, rawKV, compressedKV, indexerKV, routed.selected, debug)
}

private func DeepSeek4OutputHead<FloatType: TensorNumeric>(
  x: Model.IO, tokenLength: Int, configuration: DeepSeek4ModelConfiguration,
  includeLogits: Bool, of dataType: FloatType.Type
) -> Model.IO {
  let hc = configuration.hcCount
  let hidden = configuration.hiddenSize
  let hcDim = hc * hidden
  let hcFn = Dense(count: hc, noBias: true, name: "hc_head_fn")
  let hcScale = Parameter<FloatType>(.GPU(0), .C(1), name: "hc_head_scale")
  let hcBase = Parameter<FloatType>(.GPU(0), .C(hc), name: "hc_head_base")
  let mix = hcFn(x.reshaped([tokenLength, hcDim])).reshaped([tokenLength, hc])
  let weights = (mix .* hcScale.reshaped([1, 1]).to(of: mix) + hcBase.reshaped([1, hc]).to(of: mix))
    .sigmoid() + 1.0e-6
  let hiddenState = (x.reshaped([tokenLength, hc, hidden]) .* weights.reshaped([tokenLength, hc, 1]))
    .reduced(.sum, axis: [1])
    .reshaped([tokenLength, hidden])
  let norm = RMSNorm(epsilon: 1.0e-6, axis: [1], name: "norm")
  var out = norm(hiddenState)
  if includeLogits {
    let head = Dense(count: configuration.vocabularySize, noBias: true, name: "head")
    out = head(out)
  }
  return out
}

public func DeepSeek4CausalLM<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, tokenLength: Int, cachedTokenLength: Int = 0,
  configuration: DeepSeek4ModelConfiguration = .deepSeekV4Flash,
  includeLogits: Bool = true, outputCacheStates: Bool = false,
  outputHiddenStates: Bool = false, outputRouterSelections: Bool = false,
  debugLayerIndex: Int? = nil
) -> Model {
  precondition(tokenLength > 0)
  precondition(cachedTokenLength == 0, "The initial DeepSeek4 prefill draft handles fresh prefill only.")
  precondition(
    tokenLength >= 128,
    "The initial DeepSeek4 prefill draft expects enough tokens to exercise ratio-128 compression.")
  let tokens = Input()
  let rotary = Input()
  var inputs: [Input] = [tokens, rotary]
  var outputs = [Model.IO]()
  var cacheOutputs = [Model.IO]()
  var debugOutputs = [Model.IO]()
  var routerOutputs = [Model.IO]()
  var out = DeepSeek4Embedding(
    Float.self, tokens: tokens, tokenLength: tokenLength, configuration: configuration
  ).to(.Float32)

  for layerIndex in 0..<configuration.layers {
    let prefix = "layers.\(layerIndex)"
    let mask: Input?
    if case .compressed = configuration.attentionKind(layerIndex: layerIndex) {
      mask = Input()
      inputs.append(mask!)
    } else {
      mask = nil
    }
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
      prefix: prefix, layerIndex: layerIndex, residualHC: out, tokens: tokens, rotary: rotary,
      compressedAttentionMask: mask, selectedExpertOverride: selectedExperts,
      selectedProbabilityIndexOverride: selectedProbabilityIndices, tokenLength: tokenLength,
      configuration: configuration, of: dataType)
    out = layer.out.to(.Float32).copied()
    if outputHiddenStates {
      outputs.append(out.to(FloatType.dataType).copied())
    }
    if outputCacheStates {
      cacheOutputs.append(layer.rawKV.copied())
      if let compressedKV = layer.compressedKV {
        cacheOutputs.append(compressedKV.copied())
      }
      if let indexerKV = layer.indexerKV {
        cacheOutputs.append(indexerKV.copied())
      }
    }
    if outputRouterSelections {
      routerOutputs.append(layer.selectedExperts)
    }
    if debugLayerIndex == layerIndex {
      debugOutputs.append(contentsOf: layer.debug.map { $0.to(FloatType.dataType) })
    }
  }

  outputs.append(
    DeepSeek4OutputHead(
      x: out, tokenLength: tokenLength, configuration: configuration,
      includeLogits: includeLogits, of: dataType
    ).to(FloatType.dataType).copied())
  outputs.append(contentsOf: cacheOutputs)
  outputs.append(contentsOf: debugOutputs)
  outputs.append(contentsOf: routerOutputs)
  return Model(inputs, outputs)
}
