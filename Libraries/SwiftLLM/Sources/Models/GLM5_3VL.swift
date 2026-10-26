import Foundation
import NNC

/// GLM-5.3-Flash vision tower: a 24-block ViT with per-head Q/K RMSNorm and axial 2D
/// RoPE, a 2x2 convolutional downsample to the text width, and a clamped SwiGLU merger.
/// Its outputs replace the embeddings of `<|image|>` placeholders; the NoPE text model
/// needs no multimodal positions.
public struct GLM5_3VisionConfiguration: Codable, Sendable {
  public var hiddenSize = 1_024
  public var heads = 16
  public var layers = 24
  public var intermediateSize = 4_096
  public var patchSize = 14
  public var temporalPatchSize = 2
  public var spatialMergeSize = 2
  public var outputHiddenSize = 4_096
  public var projectionIntermediateSize = 10_240
  public var normEpsilon: Float = 1e-5
  public var swigluLimit: Float = 10
  public var ropeTheta: Double = 10_000
  public var imageMean: [Float] = [0.48145466, 0.4578275, 0.40821073]
  public var imageStd: [Float] = [0.26862954, 0.26130258, 0.27577711]
  public var minimumImageTokens = 16
  public var maximumImageTokens = 8_000
  public var imageTokenID: Int32 = 154_854
  public var imageStartTokenID: Int32 = 154_830
  public var imageEndTokenID: Int32 = 154_831

  public init() {}

  public static let glm5_3Flash = GLM5_3VisionConfiguration()

  public var headDim: Int { hiddenSize / heads }
  public var patchVectorSize: Int { 3 * temporalPatchSize * patchSize * patchSize }
  public var mergedHiddenSize: Int { hiddenSize * spatialMergeSize * spatialMergeSize }
}

/// Merged image tokens (one `<|image|>` placeholder each) for a patch grid.
public func GLM5_3VisionTokenCount(
  gridThw: (t: Int, h: Int, w: Int), configuration: GLM5_3VisionConfiguration = .glm5_3Flash
) -> Int {
  gridThw.t * gridThw.h * gridThw.w
    / (configuration.spatialMergeSize * configuration.spatialMergeSize)
}

/// The reference `smart_resize` canvas and the aspect-preserving content size placed in
/// its top-left corner; the rest of the canvas is zero padding.
public func GLM5_3VisionSize(
  height: Int, width: Int, configuration: GLM5_3VisionConfiguration = .glm5_3Flash
) -> (canvas: (height: Int, width: Int), content: (height: Int, width: Int)) {
  let factor = configuration.patchSize * configuration.spatialMergeSize
  let temporal = configuration.temporalPatchSize
  let pixelsPerToken = temporal * factor * factor
  let minimumPixels = configuration.minimumImageTokens * pixelsPerToken
  let maximumPixels = configuration.maximumImageTokens * pixelsPerToken
  func align(_ value: Int) -> Int { (value + factor - 1) / factor * factor }
  // An image is one frame, rounded up to the temporal patch.
  let frames = temporal
  var alignedHeight = align(height)
  var alignedWidth = align(width)
  if frames * alignedHeight * alignedWidth < minimumPixels {
    let scale = (Double(minimumPixels) / Double(frames * height * width)).squareRoot()
    alignedHeight = align(max(1, Int((Double(height) * scale).rounded(.up))))
    alignedWidth = align(max(1, Int((Double(width) * scale).rounded(.up))))
  }
  if frames * alignedHeight * alignedWidth > maximumPixels {
    var low = 1
    var high = height
    alignedHeight = factor
    alignedWidth = factor
    while low <= high {
      let contentHeight = (low + high) / 2
      let contentWidth = max(1, width * contentHeight / height)
      let candidateHeight = align(contentHeight)
      let candidateWidth = align(contentWidth)
      if frames * candidateHeight * candidateWidth <= maximumPixels {
        alignedHeight = candidateHeight
        alignedWidth = candidateWidth
        low = contentHeight + 1
      } else {
        high = contentHeight - 1
      }
    }
  }
  var scale = min(Double(alignedHeight) / Double(height), Double(alignedWidth) / Double(width))
  if temporal * height * width >= pixelsPerToken * configuration.minimumImageTokens {
    scale = min(1, scale)
  }
  let contentHeight = max(1, min(alignedHeight, Int((Double(height) * scale).rounded(.down))))
  let contentWidth = max(1, min(alignedWidth, Int((Double(width) * scale).rounded(.down))))
  return ((alignedHeight, alignedWidth), (contentHeight, contentWidth))
}

/// Fixed-point weights for one axis of PyTorch's antialiased bicubic resize of 8-bit
/// images: Keys cubic convolution (a = -0.5), widened by the downscale factor and
/// normalized per output, with the most fractional bits (at most 22) that keep every weight
/// within int16.
private func GLM5_3VisionResampleWeights(inputSize: Int, outputSize: Int) -> (
  starts: [Int], weights: [[Int]], precision: Int
) {
  func cubic(_ x: Double) -> Double {
    let a = -0.5
    let x = abs(x)
    if x < 1 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
    if x < 2 { return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a }
    return 0
  }
  let scale = Double(inputSize) / Double(outputSize)
  let support = scale >= 1 ? 2 * scale : 2
  let inverseScale = scale >= 1 ? 1 / scale : 1
  var starts = [Int]()
  var rows = [[Double]]()
  for i in 0..<outputSize {
    let center = scale * (Double(i) + 0.5)
    // Int(_:) truncates toward zero, as the C casts do.
    let start = max(Int(center - support + 0.5), 0)
    let end = min(Int(center + support + 0.5), inputSize)
    var weights = (start..<end).map { cubic((Double($0) - center + 0.5) * inverseScale) }
    let total = weights.reduce(0, +)
    if total != 0 { weights = weights.map { $0 / total } }
    starts.append(start)
    rows.append(weights)
  }
  let maximum = rows.joined().max() ?? 0
  var precision = 0
  while precision < 22 && Int(0.5 + maximum * Double(1 << (precision + 1))) < 1 << 15 {
    precision += 1
  }
  let one = Double(1 << precision)
  return (starts, rows.map { $0.map { Int($0 * one + ($0 < 0 ? -0.5 : 0.5)) } }, precision)
}

/// Resamples the middle axis of `[outer, inputSize, inner]` 8-bit values to `outputSize`.
private func GLM5_3VisionResample(
  _ source: [UInt8], outer: Int, inputSize: Int, outputSize: Int, inner: Int
) -> [UInt8] {
  let (starts, weights, precision) = GLM5_3VisionResampleWeights(
    inputSize: inputSize, outputSize: outputSize)
  var output = [UInt8](repeating: 0, count: outer * outputSize * inner)
  var sums = [Int](repeating: 0, count: inner)
  for o in 0..<outer {
    for i in 0..<outputSize {
      for k in 0..<inner { sums[k] = 1 << (precision - 1) }
      for (j, weight) in weights[i].enumerated() {
        let base = (o * inputSize + starts[i] + j) * inner
        for k in 0..<inner { sums[k] += Int(source[base + k]) * weight }
      }
      let base = (o * outputSize + i) * inner
      for k in 0..<inner { output[base + k] = UInt8(clamping: sums[k] >> precision) }
    }
  }
  return output
}

/// Resizes RGB `[1, height, width, 3]` values in `[0, 1]` as the reference processor does:
/// torchvision runs PyTorch's antialiased bicubic on the 8-bit image, the width pass
/// rounded to 8 bits before the height pass. Then pads to the canvas, normalizes (padding
/// becomes `-mean / std`, as in the reference) and emits block-major patches
/// `[h * w, 3 * temporal * patch * patch]` with the frame repeated temporally.
public func GLM5_3VisionPreprocess(
  _ input: Tensor<Float>, configuration: GLM5_3VisionConfiguration = .glm5_3Flash
) -> (patches: Tensor<Float>, grid: (t: Int, h: Int, w: Int)) {
  let shape = input.shape
  precondition(shape.count == 4 && shape[0] == 1 && shape[3] == 3)
  let (canvas, content) = GLM5_3VisionSize(
    height: shape[1], width: shape[2], configuration: configuration)
  let resized: Tensor<Float>
  if content.height == shape[1] && content.width == shape[2] {
    resized = input.reshaped(.NHWC(1, shape[1], shape[2], 3)).copied()
  } else {
    var pixels = input.reshaped(.NHWC(1, shape[1], shape[2], 3)).copied().withUnsafeBytes {
      $0.bindMemory(to: Float.self).map { UInt8(clamping: Int(($0 * 255).rounded())) }
    }
    if content.width != shape[2] {
      pixels = GLM5_3VisionResample(
        pixels, outer: shape[1], inputSize: shape[2], outputSize: content.width, inner: 3)
    }
    if content.height != shape[1] {
      pixels = GLM5_3VisionResample(
        pixels, outer: 1, inputSize: shape[1], outputSize: content.height,
        inner: content.width * 3)
    }
    resized = Tensor<Float>(
      pixels.map { Float($0) / 255 }, .CPU, .NHWC(1, content.height, content.width, 3))
  }
  let patch = configuration.patchSize
  let temporal = configuration.temporalPatchSize
  let merge = configuration.spatialMergeSize
  let grid = (t: 1, h: canvas.height / patch, w: canvas.width / patch)
  let mean = configuration.imageMean
  let std = configuration.imageStd
  var patches = Tensor<Float>(.CPU, .WC(grid.h * grid.w, configuration.patchVectorSize))
  patches.withUnsafeMutableBytes { patchBuffer in
    let destination = patchBuffer.baseAddress!.assumingMemoryBound(to: Float.self)
    resized.withUnsafeBytes { resizedBuffer in
      let pixels = resizedBuffer.baseAddress!.assumingMemoryBound(to: Float.self)
      var row = 0
      for blockY in 0..<(grid.h / merge) {
        for blockX in 0..<(grid.w / merge) {
          for intraY in 0..<merge {
            for intraX in 0..<merge {
              let baseY = (blockY * merge + intraY) * patch
              let baseX = (blockX * merge + intraX) * patch
              var column = row * configuration.patchVectorSize
              for channel in 0..<3 {
                for _ in 0..<temporal {
                  for py in 0..<patch {
                    for px in 0..<patch {
                      let y = baseY + py
                      let x = baseX + px
                      let value =
                        y < content.height && x < content.width
                        ? pixels[(y * content.width + x) * 3 + channel] : 0
                      destination[column] = (value - mean[channel]) / std[channel]
                      column += 1
                    }
                  }
                }
              }
              row += 1
            }
          }
        }
      }
    }
  }
  return (patches, grid)
}

/// Axial rotary table `[1, tokens, 1, headDim]` as adjacent cos/sin pairs. Pair k uses the
/// row position (k < headDim / 4) or column position; the converter permutes Q/K so the
/// reference's rotate-half pairs (k, k + headDim / 2) become adjacent.
public func GLM5_3VisionRotaryEmbedding<FloatType: TensorNumeric & BinaryFloatingPoint>(
  gridThw: [(t: Int, h: Int, w: Int)], configuration: GLM5_3VisionConfiguration = .glm5_3Flash,
  of dataType: FloatType.Type = FloatType.self
) -> Tensor<FloatType> {
  let tokenLength = gridThw.reduce(0) { $0 + $1.t * $1.h * $1.w }
  let headDim = configuration.headDim
  let pairs = headDim / 2
  let axisFrequencies = pairs / 2
  let merge = configuration.spatialMergeSize
  var rotary = Tensor<FloatType>(.CPU, .NHWC(1, tokenLength, 1, headDim))
  var token = 0
  for grid in gridThw {
    for _ in 0..<grid.t {
      // Block row, block column, intra row, intra column, as the patches.
      for blockY in 0..<(grid.h / merge) {
        for blockX in 0..<(grid.w / merge) {
          for intraY in 0..<merge {
            for intraX in 0..<merge {
              for k in 0..<pairs {
                let position =
                  k < axisFrequencies ? blockY * merge + intraY : blockX * merge + intraX
                let frequency = k % axisFrequencies
                let theta =
                  Double(position)
                  / pow(configuration.ropeTheta, Double(2 * frequency) / Double(pairs))
                rotary[0, token, 0, 2 * k] = FloatType(cos(theta))
                rotary[0, token, 0, 2 * k + 1] = FloatType(sin(theta))
              }
              token += 1
            }
          }
        }
      }
    }
  }
  return rotary
}

/// Cumulative patch offsets per image, for variable-length attention.
public func GLM5_3VisionSequenceOffsets(gridThw: [(t: Int, h: Int, w: Int)]) -> (
  offsets: Tensor<Int32>, maxSequenceLength: Int
) {
  var offsets: [Int32] = [0]
  var maxSequenceLength = 0
  for grid in gridThw {
    for _ in 0..<grid.t {
      offsets.append(offsets.last! + Int32(grid.h * grid.w))
      maxSequenceLength = max(maxSequenceLength, grid.h * grid.w)
    }
  }
  return (
    Tensor<Int32>(offsets, kind: .CPU, format: .NHWC, shape: [offsets.count]), maxSequenceLength
  )
}

private func GLM5_3VisionBlock(
  prefix: String, x: Model.IO, rotary: Model.IO, sequenceOffsets: Model.IO, tokenLength: Int,
  maxSequenceLength: Int, configuration: GLM5_3VisionConfiguration
) -> Model.IO {
  let hidden = configuration.hiddenSize
  let heads = configuration.heads
  let headDim = configuration.headDim
  let normalized = RMSNorm(
    epsilon: configuration.normEpsilon, axis: [1], name: "\(prefix).norm1")(x)
  let qkv = Dense(count: 3 * hidden, name: "\(prefix).attn.qkv")(normalized)
  func split(_ index: Int) -> Model.IO {
    qkv.reshaped(
      .NHWC(1, tokenLength, heads, headDim), offset: [0, 0, 0, index * hidden],
      strides: [tokenLength * 3 * hidden, 3 * hidden, headDim, 1]
    ).contiguous()
  }
  let queries = Functional.cmul(
    left: RMSNorm(
      epsilon: configuration.normEpsilon, axis: [3], name: "\(prefix).attn.q_norm")(split(0)),
    right: rotary)
  let keys = Functional.cmul(
    left: RMSNorm(
      epsilon: configuration.normEpsilon, axis: [3], name: "\(prefix).attn.k_norm")(split(1)),
    right: rotary)
  let attention = ScaledDotProductAttention(
    scale: 1 / Float(headDim).squareRoot(), isVariableLength: true,
    maxSequenceLength: (query: maxSequenceLength, keyValue: maxSequenceLength))(
      [queries, keys, split(2), sequenceOffsets, sequenceOffsets]
    ).reshaped([tokenLength, hidden])
  let out = x + Dense(count: hidden, name: "\(prefix).attn.proj")(attention)
  let mlpInput = RMSNorm(
    epsilon: configuration.normEpsilon, axis: [1], name: "\(prefix).norm2")(out)
  let gate = Dense(count: configuration.intermediateSize, name: "\(prefix).mlp.gate_proj")(
    mlpInput)
  let up = Dense(count: configuration.intermediateSize, name: "\(prefix).mlp.up_proj")(mlpInput)
  return out
    + Dense(count: hidden, name: "\(prefix).mlp.down_proj")(
      Functional.swishMul(value: up, gate: gate, clamp: configuration.swigluLimit))
}

/// Inputs: block-major patches `[tokens, patchVectorSize]`, the rotary table and the
/// sequence offsets. Output: image embeddings `[tokens / 4, outputHiddenSize]` in the
/// placeholder order of the text prompt.
public func GLM5_3VisionTransformer<T: TensorNumeric>(
  _ dataType: T.Type, gridThw: [(t: Int, h: Int, w: Int)],
  configuration: GLM5_3VisionConfiguration = .glm5_3Flash
) -> Model {
  let tokenLength = gridThw.reduce(0) { $0 + $1.t * $1.h * $1.w }
  let maxSequenceLength = GLM5_3VisionSequenceOffsets(gridThw: gridThw).maxSequenceLength
  let merge = configuration.spatialMergeSize * configuration.spatialMergeSize
  let patches = Input()
  let rotary = Input()
  let sequenceOffsets = Input()
  var out = Dense(count: configuration.hiddenSize, name: "visual.patch_embed")(patches)
  for i in 0..<configuration.layers {
    out = GLM5_3VisionBlock(
      prefix: "visual.blocks.\(i)", x: out, rotary: rotary, sequenceOffsets: sequenceOffsets,
      tokenLength: tokenLength, maxSequenceLength: maxSequenceLength,
      configuration: configuration)
  }
  out = RMSNorm(
    epsilon: configuration.normEpsilon, axis: [1], name: "visual.post_layernorm")(out)
  // The 2x2 stride-2 convolution reads each window's four block-major patches; the
  // converter orders its weight as (row, column, channel).
  let downsampled = Dense(count: configuration.outputHiddenSize, name: "visual.downsample")(
    out.reshaped([tokenLength / merge, configuration.mergedHiddenSize]))
  let projected = LayerNorm(
    epsilon: 1e-5, axis: [1], name: "visual.merger.post_projection_norm")(
      Dense(count: configuration.outputHiddenSize, noBias: true, name: "visual.merger.proj")(
        downsampled)
    ).GELU()
  let gate = Dense(
    count: configuration.projectionIntermediateSize, noBias: true,
    name: "visual.merger.gate_proj")(projected)
  let up = Dense(
    count: configuration.projectionIntermediateSize, noBias: true, name: "visual.merger.up_proj")(
      projected)
  let merged = Dense(
    count: configuration.outputHiddenSize, noBias: true, name: "visual.merger.down_proj")(
      Functional.swishMul(value: up, gate: gate, clamp: configuration.swigluLimit))
  return Model([patches, rotary, sequenceOffsets], [merged])
}
