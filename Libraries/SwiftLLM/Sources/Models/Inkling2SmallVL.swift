import Foundation
import NNC

/// Inkling 2 Small vision tower: a 23-layer pre-norm ViT over 20-pixel patches with a class token,
/// a learned 32x32 position table interpolated to each image, interleaved axial 2D RoPE and layer
/// scales; a GELU projector turns each 2x2 block of patches into one text-model token. Its outputs
/// replace the embeddings of the image placeholders; the text model needs no positions for them.
public struct Inkling2SmallVisionConfiguration: Codable, Sendable {
  public var hiddenSize = 1_024
  public var heads = 16
  public var layers = 23
  public var intermediateSize = 4_096
  public var patchSize = 20
  public var spatialMergeSize = 2
  public var positionGridSize = 32
  public var projectionHiddenSize = 4_096
  public var outputHiddenSize = 4_608
  public var normEpsilon: Float = 1e-5
  public var ropeTheta: Double = 10_000
  public var imageTokenID: Int32 = 200_054

  public init() {}

  public static let inkling2Small = Inkling2SmallVisionConfiguration()

  public var headDim: Int { hiddenSize / heads }
  public var patchVectorSize: Int { patchSize * patchSize * 3 }
  /// Side in pixels of the square one image token covers.
  public var tokenSize: Int { patchSize * spatialMergeSize }
}

/// Image tokens down and across: the reference pads the image, unresized, to whole tokens.
public func Inkling2SmallVisionGrid(
  height: Int, width: Int, configuration: Inkling2SmallVisionConfiguration = .inkling2Small
) -> (height: Int, width: Int) {
  let size = configuration.tokenSize
  return ((height + size - 1) / size, (width + size - 1) / size)
}

/// Pads RGB `[1, height, width, 3]` values in `[0, 1]` to whole tokens with the reference's fill
/// (-1 on the 0-255 scale), normalizes them to `[-1, 1]` and emits block-major patches
/// `[4 * tokens + 1, patchVectorSize]` in (pixel row, pixel column, channel) order; the last row is
/// the class token's zero patch. The patches keep the input's float type, which stays generic:
/// this library also builds for Intel Macs, where Float16 is unavailable.
public func Inkling2SmallVisionPreprocess<FloatType: TensorNumeric & BinaryFloatingPoint>(
  _ input: Tensor<FloatType>, configuration: Inkling2SmallVisionConfiguration = .inkling2Small
) -> (patches: Tensor<FloatType>, grid: (height: Int, width: Int)) {
  let shape = input.shape
  precondition(shape.count == 4 && shape[0] == 1 && shape[3] == 3)
  let (height, width) = (shape[1], shape[2])
  let grid = Inkling2SmallVisionGrid(height: height, width: width, configuration: configuration)
  let patch = configuration.patchSize
  let merge = configuration.spatialMergeSize
  let fill: FloatType = (-1 / 255 - 0.5) / 0.5
  var patches = Tensor<FloatType>(
    .CPU, .WC(grid.height * grid.width * merge * merge + 1, configuration.patchVectorSize))
  patches.withUnsafeMutableBytes { patchBuffer in
    let destination = patchBuffer.baseAddress!.assumingMemoryBound(to: FloatType.self)
    input.withUnsafeBytes { inputBuffer in
      let pixels = inputBuffer.baseAddress!.assumingMemoryBound(to: FloatType.self)
      var index = 0
      for tokenY in 0..<grid.height {
        for tokenX in 0..<grid.width {
          for intraY in 0..<merge {
            for intraX in 0..<merge {
              for py in 0..<patch {
                for px in 0..<patch {
                  let y = (tokenY * merge + intraY) * patch + py
                  let x = (tokenX * merge + intraX) * patch + px
                  for channel in 0..<3 {
                    destination[index] =
                      y < height && x < width
                      ? (pixels[(y * width + x) * 3 + channel] - 0.5) / 0.5 : fill
                    index += 1
                  }
                }
              }
            }
          }
        }
      }
      for _ in 0..<configuration.patchVectorSize {
        destination[index] = 0
        index += 1
      }
    }
  }
  return (patches, grid)
}

/// Bilinear interpolation (align_corners false, edges clamped) of the position table
/// `[positionGridSize^2 + 1, hidden]` for the patches in block-major order: the four table rows of
/// each patch `[4 * rows]` and their weights `[rows, 4, 1]`. The class token, last, takes the
/// table's last row.
public func Inkling2SmallVisionPositionInterpolation<
  FloatType: TensorNumeric & BinaryFloatingPoint
>(
  grid: (height: Int, width: Int),
  configuration: Inkling2SmallVisionConfiguration = .inkling2Small,
  of dataType: FloatType.Type = FloatType.self
) -> (indices: Tensor<Int32>, weights: Tensor<FloatType>) {
  let side = configuration.positionGridSize
  let merge = configuration.spatialMergeSize
  let height = grid.height * merge
  let width = grid.width * merge
  // For each patch row, then each patch column: the table lines below and above its source
  // position, and the weight of the one above, as the reference computes them.
  var lines = [(lower: Int, upper: Int, fraction: Float)]()
  lines.reserveCapacity(height + width)
  for i in 0..<(height + width) {
    let (index, size) = i < height ? (i, height) : (i - height, width)
    let source = (Float(index) + 0.5) * Float(side) / Float(size) - 0.5
    let floor = source.rounded(.down)
    lines.append((max(Int(floor), 0), min(Int(floor) + 1, side - 1), source - floor))
  }
  let count = grid.height * grid.width * merge * merge + 1
  var indices = [Int32]()
  var weights = [FloatType]()
  indices.reserveCapacity(4 * count)
  weights.reserveCapacity(4 * count)
  for blockY in 0..<grid.height {
    for blockX in 0..<grid.width {
      for intraY in 0..<merge {
        let row = lines[blockY * merge + intraY]
        for intraX in 0..<merge {
          let column = lines[height + blockX * merge + intraX]
          indices.append(Int32(row.lower * side + column.lower))
          indices.append(Int32(row.lower * side + column.upper))
          indices.append(Int32(row.upper * side + column.lower))
          indices.append(Int32(row.upper * side + column.upper))
          weights.append(FloatType((1 - row.fraction) * (1 - column.fraction)))
          weights.append(FloatType((1 - row.fraction) * column.fraction))
          weights.append(FloatType(row.fraction * (1 - column.fraction)))
          weights.append(FloatType(row.fraction * column.fraction))
        }
      }
    }
  }
  for i in 0..<4 {
    indices.append(Int32(side * side))
    weights.append(i == 0 ? 1 : 0)
  }
  return (Tensor(indices, .CPU, .C(4 * count)), Tensor(weights, .CPU, .HWC(count, 4, 1)))
}

/// Axial rotary table `[1, 4 * tokens + 1, 1, headDim]` as adjacent cos/sin pairs, the reference's
/// interleaved rotation: the first half of a patch's pairs rotate by its column, the second half by
/// its row, both counted from 1; the class token, last, sits at 0.
public func Inkling2SmallVisionRotaryEmbedding<FloatType: TensorNumeric & BinaryFloatingPoint>(
  grid: (height: Int, width: Int),
  configuration: Inkling2SmallVisionConfiguration = .inkling2Small,
  of dataType: FloatType.Type = FloatType.self
) -> Tensor<FloatType> {
  let half = configuration.headDim / 2
  let merge = configuration.spatialMergeSize
  let inverseFrequencies = (0..<(half / 2)).map {
    1 / pow(configuration.ropeTheta, Double(2 * $0) / Double(half))
  }
  // The cos/sin pairs of each axis position, 0 through the longer side.
  let longest = max(grid.height, grid.width) * merge
  var axis = [FloatType]()
  axis.reserveCapacity((longest + 1) * half)
  for position in 0...longest {
    for inverseFrequency in inverseFrequencies {
      let theta = Double(position) * inverseFrequency
      axis.append(FloatType(cos(theta)))
      axis.append(FloatType(sin(theta)))
    }
  }
  let count = grid.height * grid.width * merge * merge + 1
  var rotary = [FloatType]()
  rotary.reserveCapacity(count * 2 * half)
  for blockY in 0..<grid.height {
    for blockX in 0..<grid.width {
      for intraY in 0..<merge {
        let y = blockY * merge + intraY + 1
        for intraX in 0..<merge {
          let x = blockX * merge + intraX + 1
          rotary += axis[(x * half)..<((x + 1) * half)]
          rotary += axis[(y * half)..<((y + 1) * half)]
        }
      }
    }
  }
  rotary += axis[0..<half]
  rotary += axis[0..<half]
  return Tensor(rotary, .CPU, .NHWC(1, count, 1, 2 * half))
}

/// Inputs: one image's patches `[4 * tokens + 1, patchVectorSize]`, the position table rows and
/// weights from Inkling2SmallVisionPositionInterpolation, and the rotary table. Output: image embeddings `[tokens, outputHiddenSize]` in row-major token
/// order, the placeholder order of the prompt.
public func Inkling2SmallVisionTransformer<T: TensorNumeric>(
  _ dataType: T.Type, grid: (height: Int, width: Int),
  configuration: Inkling2SmallVisionConfiguration = .inkling2Small
) -> Model {
  let h = configuration.hiddenSize
  let heads = configuration.heads
  let headDim = configuration.headDim
  let merge = configuration.spatialMergeSize * configuration.spatialMergeSize
  let tokens = grid.height * grid.width
  let rows = tokens * merge + 1
  let patches = Input()
  let positionIndices = Input()
  let positionWeights = Input()
  let rotary = Input()
  // The class token is a zero patch, which the bias-free projection keeps zero, and takes the
  // table's last row: the class and class position embeddings. Attention is bidirectional, so it
  // can follow the patches, whose rows then make the merged tokens in place.
  let positions = Embedding(
    T.self, vocabularySize: configuration.positionGridSize * configuration.positionGridSize + 1,
    embeddingSize: h, name: "visual.backbone.embeddings.position_embedding")(positionIndices)
  var x = LayerNorm(epsilon: configuration.normEpsilon, axis: [1], name: "visual.backbone.ln_pre")(
    Dense(count: h, noBias: true, name: "visual.backbone.embeddings.patch_embedding")(patches)
      + (positions.reshaped([rows, 4, h]) .* positionWeights).reduced(.sum, axis: [1]).reshaped(
        [rows, h]))
  for i in 0..<configuration.layers {
    let prefix = "visual.backbone.layers.\(i)"
    let normalized = LayerNorm(
      epsilon: configuration.normEpsilon, axis: [1], name: "\(prefix).ln_1")(x)
    let attention = ScaledDotProductAttention(scale: 1 / Float(headDim).squareRoot())(
      Functional.cmul(
        left: Dense(count: h, name: "\(prefix).self_attn.q_proj")(normalized).reshaped([
          1, rows, heads, headDim,
        ]), right: rotary),
      Functional.cmul(
        left: Dense(count: h, name: "\(prefix).self_attn.k_proj")(normalized).reshaped([
          1, rows, heads, headDim,
        ]), right: rotary),
      Dense(count: h, name: "\(prefix).self_attn.v_proj")(normalized).reshaped([
        1, rows, heads, headDim,
      ])
    ).reshaped([rows, h])
    // The converter folds the layer scales into out_proj and c_proj.
    x = x + Dense(count: h, name: "\(prefix).self_attn.out_proj")(attention)
    x =
      x
      + Dense(count: h, name: "\(prefix).mlp.c_proj")(
        Dense(count: configuration.intermediateSize, name: "\(prefix).mlp.c_fc")(
          LayerNorm(epsilon: configuration.normEpsilon, axis: [1], name: "\(prefix).ln_2")(x)
        ).GELU())
  }
  // Four block-major patches make one token.
  let merged = x.reshaped([tokens, merge * h], offset: [0, 0], strides: [merge * h, 1])
  return Model(
    [patches, positionIndices, positionWeights, rotary],
    [
      Dense(count: configuration.outputHiddenSize, noBias: true, name: "visual.proj_out")(
        Dense(count: configuration.projectionHiddenSize, noBias: true, name: "visual.proj_in")(
          merged
        ).GELU())
    ])
}
