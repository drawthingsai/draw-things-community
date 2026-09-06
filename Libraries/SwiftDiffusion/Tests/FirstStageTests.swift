import Diffusion
import Foundation
import NNC
import XCTest

final class FirstStageTests: XCTestCase {
  // Parameter-free models exercise the public FirstStage path without checkpoints.
  private func stage(version: ModelVersion, tilePixels: Int, tiled: Bool) -> FirstStage<Float16> {
    let tiling = TiledConfiguration(
      isEnabled: tiled,
      tileSize: .init(width: tilePixels / 64, height: tilePixels / 64), tileOverlap: 0)
    return FirstStage<Float16>(
      filePath: "", version: version,
      latentsScaling: (nil, nil, 1, nil, nil, nil),
      highPrecisionKeysAndValues: false, highPrecisionFallback: false,
      tiledDecoding: tiling, tiledDiffusion: tiling, externalOnDemand: false,
      usesFlashAttention: false, alternativeFilePath: nil, alternativeDecoderVersion: nil,
      deviceProperties: DeviceProperties(
        isFreadPreferred: true, memoryCapacity: .high, isNHWCPreferred: true,
        cacheUri: URL(fileURLWithPath: NSTemporaryDirectory()), isPartialOffloadPreferred: false))
  }

  func testHunyuanSpatialTilesPreserveShortClipDepth() {
    let graph = DynamicGraph()
    graph.withNoGrad {
      for frames in [1, 2, 14, 15, 16] {
        let expectedDepth = min(frames, 15)
        let input = Input()
        let mask = Input()
        var inputDepths = [Int]()
        let checked = input.debug { tensors, _ in
          inputDepths.append(tensors[0]!.shape[0])
        }
        let pixels = Upsample(.nearest, widthScale: 8, heightScale: 8)(
          checked + ReduceMax(axis: [0, 1, 2, 3])(mask))
        let decoder = Model(
          [input, mask],
          [
            Pad(
              .replicate, begin: [0, 0, 0, 0], end: [(expectedDepth - 1) * 3, 0, 0, 0])(
                pixels)
          ])
        let latent = graph.variable(.GPU(0), .NHWC(frames, 64, 64, 3), of: Float16.self)
        latent.full(0.25)
        let result = stage(version: .hunyuanVideo, tilePixels: 256, tiled: true).decode(
          latent, decoder: decoder, cancellation: { _ in }
        ).0.rawValue.toCPU()
        graph.joined()
        XCTAssertFalse(inputDepths.isEmpty)
        XCTAssertTrue(inputDepths.allSatisfy { $0 == expectedDepth }, "\(frames): \(inputDepths)")
        XCTAssertEqual(Array(result.shape), [(frames - 1) * 4 + 1, 512, 512, 3])
        XCTAssertEqual(Float(result[0, 0, 0, 0]), 0.25, accuracy: 0.001)
        XCTAssertEqual(Float(result[result.shape[0] - 1, 511, 511, 2]), 0.25, accuracy: 0.001)
      }
    }
  }

  func testMiniMaxH3EncoderNormalizesPixelsBeforeTiling() {
    let graph = DynamicGraph()
    graph.withNoGrad {
      var pixels = Tensor<Float16>(.CPU, .NHWC(1, 128, 128, 3))
      let values: [Float16] = [-1, 0, 1]
      for y in 0..<128 {
        for x in 0..<128 {
          for channel in 0..<3 {
            pixels[0, y, x, channel] = values[channel]
          }
        }
      }
      let mean: [Float] = [0.485, 0.456, 0.406]
      let std: [Float] = [0.229, 0.224, 0.225]
      for tiled in [false, true] {
        let input = Input()
        let pooled = AveragePool(filterSize: [16, 16], hint: Hint(stride: [16, 16]))(input)
        let encoder = Model([input], [Concat(axis: 3)(Array(repeating: pooled, count: 16))])
        let firstStage = stage(version: .minimaxH3, tilePixels: 64, tiled: tiled)
        // Reuse the encoder as well: normalization must happen on every encode call.
        for _ in 0..<2 {
          let result = firstStage.encode(
            graph.variable(pixels.toGPU(0)), encoder: encoder, cancellation: { _ in }
          ).0.rawValue.toCPU()
          XCTAssertEqual(Array(result.shape), [1, 8, 8, 48])
          guard Array(result.shape) == [1, 8, 8, 48] else { continue }
          for y in 0..<8 {
            for x in 0..<8 {
              for channel in 0..<48 {
                let c = channel % 3
                let expected = ((Float(values[c]) + 1) * 0.5 - mean[c]) / std[c]
                XCTAssertEqual(Float(result[0, y, x, channel]), expected, accuracy: 0.01)
              }
            }
          }
        }
      }
    }
  }
}
