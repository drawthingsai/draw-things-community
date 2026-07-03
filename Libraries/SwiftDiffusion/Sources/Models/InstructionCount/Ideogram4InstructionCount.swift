import Foundation

private func Ideogram4TransformerBlockInstructionCount(
  batchSize: Int, sequenceLength: Int, channels: Int = 4_608, headDimension: Int = 256,
  intermediateSize: Int = 12_288, modulationSize: Int = 512
) -> Int {
  precondition(channels % headDimension == 0)
  let rows = batchSize * sequenceLength
  var total = 0

  total += 4 * DenseInstructionCount(rows: batchSize, input: modulationSize, output: channels)
  total += 4 * DenseInstructionCount(rows: rows, input: channels, output: channels)
  total += ScaledDotProductAttentionInstructionCount(
    batchSize: batchSize, heads: channels / headDimension, headDimension: headDimension,
    sequenceDimensionA: sequenceLength, sequenceDimensionB: sequenceLength)
  total += 2 * DenseInstructionCount(rows: rows, input: channels, output: intermediateSize)
  total += DenseInstructionCount(rows: rows, input: intermediateSize, output: channels)

  return total
}

public func Ideogram4InstructionCount(
  batchSize: Int, height: Int, width: Int, textLength: Int,
  textInputChannels: Int = 4_096 * 13, channels: Int = 4_608, layers: Int = 34,
  intermediateSize: Int = 12_288
) -> Int {
  precondition(height % 2 == 0 && width % 2 == 0)
  let patchHeight = height / 2
  let patchWidth = width / 2
  let imageLength = patchHeight * patchWidth
  let totalLength = imageLength + textLength
  var total = 0

  total += DenseInstructionCount(rows: batchSize * imageLength, input: 128, output: channels)
  total += DenseInstructionCount(
    rows: batchSize * textLength, input: textInputChannels, output: channels)
  total += DenseInstructionCount(rows: batchSize, input: channels, output: channels)
  total += DenseInstructionCount(rows: batchSize, input: channels, output: channels)
  total += DenseInstructionCount(rows: batchSize, input: channels, output: 512)

  for _ in 0..<layers {
    total += Ideogram4TransformerBlockInstructionCount(
      batchSize: batchSize, sequenceLength: totalLength, channels: channels,
      intermediateSize: intermediateSize)
  }

  total += DenseInstructionCount(rows: batchSize, input: 512, output: channels)
  total += DenseInstructionCount(rows: batchSize * totalLength, input: channels, output: 128)
  return total
}
