import Foundation

// Counts only Dense (GEMM), SDPA-equivalent attention, and Convolution instructions.
// Ignores LoRA, norms, elementwise ops, reshapes, concat/slice, and adapter/control injections.
public func Flux2InstructionCount(
  batchSize: Int, tokenLength: Int, referenceSequenceLength: Int, height: Int, width: Int,
  channels: Int, layers: (Int, Int)
) -> Int {
  precondition(height % 2 == 0 && width % 2 == 0)
  precondition(channels % 128 == 0)

  let imageHeight = height / 2
  let imageWidth = width / 2
  let imageSequenceLength = imageHeight * imageWidth
  let imageAndReferenceSequenceLength = imageSequenceLength + referenceSequenceLength
  let totalSequenceLength = tokenLength + imageAndReferenceSequenceLength
  let heads = channels / 128
  let headDimension = 128
  var total = 0

  // x_embedder patchify conv: latent 32ch -> channels with 2x2 stride 2.
  total += ConvolutionInstructionCount(
    batchSize: batchSize, outHeight: imageHeight, outWidth: imageWidth, outChannels: channels,
    kernelHeight: 2, kernelWidth: 2, inputChannels: 32)

  let rowsContext = batchSize * tokenLength
  let rowsImage = batchSize * imageAndReferenceSequenceLength

  for _ in 0..<layers.0 {
    // JointTransformerBlock: text + image streams, SwiGLU FFN with intermediate=3*channels.
    total += 4 * DenseInstructionCount(rows: rowsContext, input: channels, output: channels)
    total += 2 * DenseInstructionCount(rows: rowsContext, input: channels, output: channels * 3)
    total += DenseInstructionCount(rows: rowsContext, input: channels * 3, output: channels)

    total += 4 * DenseInstructionCount(rows: rowsImage, input: channels, output: channels)
    total += 2 * DenseInstructionCount(rows: rowsImage, input: channels, output: channels * 3)
    total += DenseInstructionCount(rows: rowsImage, input: channels * 3, output: channels)

    total += ScaledDotProductAttentionInstructionCount(
      batchSize: batchSize, heads: heads, headDimension: headDimension,
      sequenceDimensionA: totalSequenceLength, sequenceDimensionB: totalSequenceLength)
  }

  for i in 0..<layers.1 {
    let isLast = (i == layers.1 - 1)
    let projectedSequenceLength = isLast ? imageSequenceLength : totalSequenceLength
    let rowsProjected = batchSize * projectedSequenceLength
    let rowsTotal = batchSize * totalSequenceLength

    total += 3 * DenseInstructionCount(rows: rowsTotal, input: channels, output: channels)  // qkv
    total += DenseInstructionCount(rows: rowsProjected, input: channels, output: channels)  // x_o
    total += 2 * DenseInstructionCount(rows: rowsProjected, input: channels, output: channels * 3)  // x_w1/x_w3
    total += DenseInstructionCount(rows: rowsProjected, input: channels * 3, output: channels)  // x_w2
    total += ScaledDotProductAttentionInstructionCount(
      batchSize: batchSize, heads: heads, headDimension: headDimension,
      sequenceDimensionA: totalSequenceLength, sequenceDimensionB: totalSequenceLength)
  }

  total += DenseInstructionCount(
    rows: batchSize * imageSequenceLength, input: channels, output: 2 * 2 * 32)

  return total
}

// `Flux2Fixed(...)` itself does not carry enough shape information (timesteps/tokenLength/batch),
// so those compile-shape inputs are passed explicitly here.
public func Flux2FixedInstructionCount(
  contextBatchSize: Int, tokenLength: Int, timestepCount: Int, referenceSequenceLength: Int,
  channels: Int, guidanceEmbed: Bool
) -> Int {
  precondition(channels % 128 == 0)
  precondition(contextBatchSize >= 0 && tokenLength >= 0 && timestepCount >= 0)
  var total = 0

  if referenceSequenceLength > 0 {
    // x_embedder patchify conv on reference images (post-patchify token count aggregated).
    total += 128 * channels * referenceSequenceLength  // 2*2*32 * channels per token
  }

  // context_embedder on text tokens.
  total += DenseInstructionCount(
    rows: contextBatchSize * tokenLength, input: 4096, output: channels)

  // t embedder MLP (256 -> c -> c), one row per timestep.
  total += DenseInstructionCount(rows: timestepCount, input: 256, output: channels)
  total += DenseInstructionCount(rows: timestepCount, input: channels, output: channels)

  if guidanceEmbed {
    total += DenseInstructionCount(rows: timestepCount, input: 256, output: channels)
    total += DenseInstructionCount(rows: timestepCount, input: channels, output: channels)
  }

  // Modulation heads from `vec` (timestepCount rows).
  // xAdaLNs (6) + contextAdaLNs (6) + singleAdaLNs (3) + final (scale,shift) = 17 Dense.
  total += 17 * DenseInstructionCount(rows: timestepCount, input: channels, output: channels)

  return total
}
