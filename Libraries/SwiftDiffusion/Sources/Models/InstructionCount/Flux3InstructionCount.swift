import Foundation

// Counts only Dense (GEMM) and SDPA-equivalent attention instructions.
// Ignores LoRA, norms, elementwise ops, reshapes, concat/slice, and adapter/control injections.
public func Flux3InstructionCount(
  batchSize: Int, tokenLength: Int, referenceSequenceLength: Int, height: Int, width: Int,
  channels: Int, layers: (Int, Int)
) -> Int {
  precondition(height % 2 == 0 && width % 2 == 0)
  precondition(channels % 128 == 0)

  let imageSequenceLength = (height / 2) * (width / 2)
  let sequenceLength = imageSequenceLength + referenceSequenceLength + tokenLength
  let heads = channels / 128
  let headDimension = 128
  var total = 0

  // x_embedder on patchified latents: 2*2*32 -> channels.
  total += DenseInstructionCount(
    rows: batchSize * imageSequenceLength, input: 2 * 2 * 32, output: channels)

  // SingleTransformerBlock: qkv, out projection, SwiGLU FFN with intermediate=3*channels.
  func blockInstructionCount(length: Int, projectedLength: Int) -> Int {
    let rows = batchSize * length
    let rowsProjected = batchSize * projectedLength
    var total = 3 * DenseInstructionCount(rows: rows, input: channels, output: channels)
    total += DenseInstructionCount(rows: rowsProjected, input: channels, output: channels)
    total += 2 * DenseInstructionCount(rows: rowsProjected, input: channels, output: channels * 3)
    total += DenseInstructionCount(rows: rowsProjected, input: channels * 3, output: channels)
    total += ScaledDotProductAttentionInstructionCount(
      batchSize: batchSize, heads: heads, headDimension: headDimension,
      sequenceDimensionA: length, sequenceDimensionB: length)
    return total
  }

  for _ in 0..<layers.0 {
    // Independent refiners for text, image and reference.
    total += blockInstructionCount(length: tokenLength, projectedLength: tokenLength)
    total += blockInstructionCount(
      length: imageSequenceLength, projectedLength: imageSequenceLength)
    if referenceSequenceLength > 0 {
      total += blockInstructionCount(
        length: referenceSequenceLength, projectedLength: referenceSequenceLength)
    }
  }

  for i in 0..<layers.1 {
    let isLast = (i == layers.1 - 1)
    total += blockInstructionCount(
      length: sequenceLength, projectedLength: isLast ? imageSequenceLength : sequenceLength)
  }

  total += DenseInstructionCount(
    rows: batchSize * imageSequenceLength, input: channels, output: 2 * 2 * 32)

  return total
}

// `Flux3Fixed(...)` itself does not carry enough shape information (timesteps/tokenLength/batch),
// so those compile-shape inputs are passed explicitly here.
public func Flux3FixedInstructionCount(
  contextBatchSize: Int, tokenLength: Int, timestepCount: Int, referenceSequenceLength: Int,
  channels: Int
) -> Int {
  precondition(channels % 128 == 0)
  precondition(contextBatchSize >= 0 && tokenLength >= 0 && timestepCount >= 0)
  var total = 0

  if referenceSequenceLength > 0 {
    // ref_embedder on patchified reference latents.
    total += DenseInstructionCount(
      rows: referenceSequenceLength, input: 2 * 2 * 32, output: channels)
  }

  // context_embedder on the 8 concatenated Qwen3-VL hidden states.
  total += DenseInstructionCount(
    rows: contextBatchSize * tokenLength, input: 20_480, output: channels)

  // t embedder MLP (256 -> c -> c), one row per timestep plus timestep 0.
  total += DenseInstructionCount(rows: timestepCount + 1, input: 256, output: channels)
  total += DenseInstructionCount(rows: timestepCount + 1, input: channels, output: channels)

  // next_time embedder MLP (256 -> c -> c), one row per timestep.
  total += DenseInstructionCount(rows: timestepCount, input: 256, output: channels)
  total += DenseInstructionCount(rows: timestepCount, input: channels, output: channels)

  // Modulation heads: (4 + 2 with reference) * 3 chunks, plus final (shift, scale).
  let modulationCount = referenceSequenceLength > 0 ? 6 : 4
  total +=
    modulationCount
    * DenseInstructionCount(rows: timestepCount, input: channels, output: channels * 3)
  total += DenseInstructionCount(rows: timestepCount, input: channels, output: channels * 2)

  return total
}
