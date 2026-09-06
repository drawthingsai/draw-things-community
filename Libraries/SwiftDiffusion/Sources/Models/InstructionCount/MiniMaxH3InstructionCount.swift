import Foundation

private func MiniMaxH3AttentionInstructionCount(rows: Int) -> Int {
  let hiddenSize = 5_376
  let heads = 56
  let headDimension = 128
  let attentionSize = heads * headDimension
  return 3 * DenseInstructionCount(rows: rows, input: hiddenSize, output: attentionSize)
    + ScaledDotProductAttentionInstructionCount(
      batchSize: 1, heads: heads, headDimension: headDimension,
      sequenceDimensionA: rows, sequenceDimensionB: rows)
    + DenseInstructionCount(rows: rows, input: attentionSize, output: hiddenSize)
}

private func MiniMaxH3FeedForwardInstructionCount(rows: Int) -> Int {
  let hiddenSize = 5_376
  let intermediateSize = 14_336
  return 2 * DenseInstructionCount(rows: rows, input: hiddenSize, output: intermediateSize)
    + DenseInstructionCount(rows: rows, input: intermediateSize, output: hiddenSize)
}

public func MiniMaxH3InstructionCount(
  videoTime: Int, videoHeight: Int, videoWidth: Int, audioLength: Int, textLength: Int,
  layers: Int = 50
) -> Int {
  precondition(videoTime > 0 && videoHeight > 0 && videoWidth > 0)
  precondition(videoHeight.isMultiple(of: 2) && videoWidth.isMultiple(of: 2))
  precondition(audioLength > 0 && textLength > 0 && layers > 0)
  let hiddenSize = 5_376
  let videoRows = videoTime * (videoHeight / 2) * (videoWidth / 2)
  let rows = textLength + audioLength + videoRows

  var total = 0
  total += DenseInstructionCount(rows: videoRows, input: 96, output: hiddenSize)
  total += DenseInstructionCount(rows: audioLength, input: 32, output: hiddenSize)
  total += DenseInstructionCount(rows: textLength, input: 5_120, output: hiddenSize)
  for _ in 0..<2 {
    total += MiniMaxH3AttentionInstructionCount(rows: textLength)
    total += MiniMaxH3FeedForwardInstructionCount(rows: textLength)
  }
  for _ in 0..<layers {
    total += MiniMaxH3AttentionInstructionCount(rows: rows)
    total += MiniMaxH3FeedForwardInstructionCount(rows: rows)
  }
  total += DenseInstructionCount(rows: videoRows, input: hiddenSize, output: 96)
  total += DenseInstructionCount(rows: audioLength, input: hiddenSize, output: 32)
  return total
}

public func MiniMaxH3FixedInstructionCount(timesteps: Int, hiddenSize: Int, layers: Int) -> Int {
  precondition(timesteps > 0 && hiddenSize > 0 && layers > 0)
  // Each projection runs on both video and audio timesteps before the modality slice.
  let rows = timesteps * 2
  var total = DenseInstructionCount(rows: rows, input: 256, output: hiddenSize)
  total += DenseInstructionCount(rows: rows, input: hiddenSize, output: 2_688)
  total += (layers * 18 + 2) * DenseInstructionCount(rows: rows, input: 2_688, output: hiddenSize)
  return total
}
