import Foundation
import NNC

private func qwenVLMRoPEAngles(
  position: (Int, Int, Int), headDim: Int, theta: Double = 5_000_000
) -> [Double] {
  let half = headDim / 2
  var angles = [Double](repeating: 0, count: half)
  for i in 0..<half {
    angles[i] = Double(position.0) / pow(theta, Double(i * 2) / Double(headDim))
  }
  let positions = [position.0, position.1, position.2]
  let sections = [24, 20, 20]
  for axis in 1...2 {
    let length = sections[axis] * 3
    var i = axis
    while i < length {
      angles[i] = Double(positions[axis]) / pow(theta, Double(i * 2) / Double(headDim))
      i += 3
    }
  }
  return angles
}

public func QwenVLRotaryEmbedding<FloatType: TensorNumeric & BinaryFloatingPoint>(
  sequenceLength: Int, of dataType: FloatType.Type = FloatType.self
) -> Tensor<FloatType> {
  let headDim = 128
  let half = headDim / 2
  var rotary = Tensor<FloatType>(.CPU, .NHWC(1, sequenceLength, 1, headDim))
  for i in 0..<sequenceLength {
    let angles = qwenVLMRoPEAngles(position: (i, i, i), headDim: headDim)
    for k in 0..<half {
      rotary[0, i, 0, k * 2] = FloatType(cos(angles[k]))
      rotary[0, i, 0, k * 2 + 1] = FloatType(sin(angles[k]))
    }
  }
  return rotary
}
