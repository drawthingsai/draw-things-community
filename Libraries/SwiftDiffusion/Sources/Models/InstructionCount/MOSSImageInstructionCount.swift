import Foundation

// Counts only Dense (GEMM) and SDPA-equivalent attention instructions.
// Ignores norms, elementwise ops, reshapes and concat/slice. Skipped negative-branch blocks are
// counted as regular blocks.
public func MOSSImageInstructionCount(
  batchSize: Int, tokenLength: Int, height: Int, width: Int, channels: Int, layers: Int
) -> Int {
  precondition(height % 2 == 0 && width % 2 == 0)
  precondition(channels % 128 == 0)

  let imageSequenceLength = (height / 2) * (width / 2)
  let sequenceLength = imageSequenceLength + tokenLength
  var total = 0

  // x_embedder on patchified latents.
  total += DenseInstructionCount(
    rows: batchSize * imageSequenceLength, input: 2 * 2 * 32, output: channels)

  for i in 0..<layers {
    let isLast = (i == layers - 1)
    total += MOSSImageBlockInstructionCount(
      batchSize: batchSize, length: sequenceLength,
      projectedLength: isLast ? imageSequenceLength : sequenceLength, channels: channels)
  }

  total += DenseInstructionCount(
    rows: batchSize * imageSequenceLength, input: channels, output: 2 * 2 * 32)

  return total
}

// `MOSSImageFixed(...)` carries the text refiner, the time MLP and the modulations.
public func MOSSImageFixedInstructionCount(
  batchSize: Int, tokenLength: Int, timesteps: Int, channels: Int
) -> Int {
  precondition(timesteps >= 0)
  // context_embedder on Qwen 3.5 hidden states, then the text refiner.
  var total = DenseInstructionCount(rows: batchSize * tokenLength, input: 4_096, output: channels)
  for _ in 0..<2 {
    total += MOSSImageBlockInstructionCount(
      batchSize: batchSize, length: tokenLength, projectedLength: tokenLength, channels: channels)
  }
  total += DenseInstructionCount(rows: timesteps, input: 256, output: channels)
  total += DenseInstructionCount(rows: timesteps, input: channels, output: channels)
  // 6 shared modulations and the head shift / scale.
  total += 8 * DenseInstructionCount(rows: timesteps, input: channels, output: channels)
  return total
}

// qkv, out projection and SiTU FFN, shared by the text refiner and the main blocks.
private func MOSSImageBlockInstructionCount(
  batchSize: Int, length: Int, projectedLength: Int, channels: Int
) -> Int {
  let rows = batchSize * length
  let rowsProjected = batchSize * projectedLength
  var total = 3 * DenseInstructionCount(rows: rows, input: channels, output: channels)
  total += DenseInstructionCount(rows: rowsProjected, input: channels, output: channels)
  total += 2 * DenseInstructionCount(rows: rowsProjected, input: channels, output: 8_192)
  total += DenseInstructionCount(rows: rowsProjected, input: 8_192, output: channels)
  total += ScaledDotProductAttentionInstructionCount(
    batchSize: batchSize, heads: channels / 128, headDimension: 128,
    sequenceDimensionA: length, sequenceDimensionB: length)
  return total
}
