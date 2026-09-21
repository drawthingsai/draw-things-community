import Foundation
import NNC
import XCTest

@testable import Diffusion

final class QwenImage2_1Tests: XCTestCase {
  func testNHWCVAEParity() throws {
    guard let path = ProcessInfo.processInfo.environment["QWEN_IMAGE_2_1_VAE_CHECKPOINT"] else {
      throw XCTSkip("Set QWEN_IMAGE_2_1_VAE_CHECKPOINT to the converted f16 VAE checkpoint")
    }
    for (encoder, useFloat16, flash) in [
      (true, false, false), (false, false, false), (true, true, false), (false, true, false),
      (true, false, true), (false, false, true), (true, true, true), (false, true, true),
    ] {
      let graph = DynamicGraph()
      graph.withNoGrad {
        let height = encoder ? 32 : 2
        let width = encoder ? 48 : 3
        let channels = encoder ? 4 : 64
        var input = Tensor<Float>(.CPU, .NHWC(1, height, width, channels))
        for y in 0..<height {
          for x in 0..<width {
            for c in 0..<channels {
              input[0, y, x, c] = sin(Float(y * width * channels + x * channels + c) * 0.03)
            }
          }
        }
        let model =
          encoder
          ? QwenImage2_1Encoder(
            channels: [96, 192, 384, 768, 768], height: height, width: width,
            usesFlashAttention: flash)
          : QwenImage2_1Decoder(
            channels: [1152, 1152, 576, 288, 144], height: height, width: width,
            usesFlashAttention: flash)
        let baseline = QwenImage2_1VAEReference(encoder: encoder, height: height, width: width).0
        let x: DynamicGraph.AnyTensor =
          useFloat16
          ? graph.variable(Tensor<Float16>(from: input).toGPU(0)) : graph.variable(input.toGPU(0))
        model.maxConcurrency = .limit(1)
        baseline.maxConcurrency = .limit(1)
        model.compile(inputs: x)
        baseline.compile(inputs: x)
        graph.openStore(
          path, flags: .readOnly, externalStore: TensorData.externalStore(filePath: path)
        ) {
          try! $0.read(
            encoder ? "encoder" : "decoder", model: model, strict: true,
            codec: [.externalData(.mmap)])
          try! $0.read(
            encoder ? "encoder" : "decoder", model: baseline, strict: true,
            codec: [.externalData(.mmap)])
        }
        let actualOutput = model(inputs: x)[0]
        let expectedOutput = baseline(inputs: x)[0]
        let actual =
          useFloat16
          ? Tensor<Float>(from: actualOutput.as(of: Float16.self).rawValue.toCPU())
          : actualOutput.as(of: Float.self).rawValue.toCPU()
        let expected =
          useFloat16
          ? Tensor<Float>(from: expectedOutput.as(of: Float16.self).rawValue.toCPU())
          : expectedOutput.as(of: Float.self).rawValue.toCPU()
        XCTAssertEqual(Array(actual.shape), Array(expected.shape))
        var squaredError: Double = 0
        var squaredSignal: Double = 0
        var maximum: Float = 0
        for y in 0..<actual.shape[1] {
          for x in 0..<actual.shape[2] {
            for c in 0..<actual.shape[3] {
              let a = actual[0, y, x, c]
              let b = expected[0, y, x, c]
              XCTAssertTrue(a.isFinite)
              maximum = max(maximum, abs(a - b))
              squaredError += Double((a - b) * (a - b))
              squaredSignal += Double(b * b)
            }
          }
        }
        let relative = sqrt(squaredError / max(squaredSignal, 1e-12))
        print(
          "Qwen2.1 VAE parity encoder=\(encoder) fp16=\(useFloat16) flash=\(flash): relative RMS=\(relative), max=\(maximum)"
        )
        XCTAssertLessThan(relative, useFloat16 ? 0.01 : 0.001)
        XCTAssertLessThan(maximum, useFloat16 ? 0.1 : 0.01)
      }
    }
  }

  func testCachedPrefixMatchesFullSequence() throws {
    guard let path = ProcessInfo.processInfo.environment["QWEN_IMAGE_2_1_CHECKPOINT"] else {
      throw XCTSkip("Set QWEN_IMAGE_2_1_CHECKPOINT to the converted f16 DiT checkpoint")
    }
    for flashAttention in [FlashAttentionLevel.none, .scale1, .scaleMerged, .quantized] {
      let flash = flashAttention != .none
      for references in [false, true] {
        let graph = DynamicGraph()
        graph.withNoGrad {
          let textLength = 8
          let referenceLength = references ? 8 : 0
          let prefixLength = textLength + referenceLength
          let targetLength = 4
          let count = prefixLength + targetLength
          let batchSize = 2
          // Adjacent image blocks must not see one another's future keys.
          let prefix: [QwenImage2_1Segment] =
            references
            ? [
              .init(image: false, length: 3, sourceOffset: 0),
              .init(image: true, length: 4, sourceOffset: 0, height: 2, width: 2),
              .init(image: true, length: 4, sourceOffset: 4, height: 2, width: 2),
              .init(image: false, length: 5, sourceOffset: 3),
            ] : [.init(image: false, length: textLength, sourceOffset: 0)]
          let target = QwenImage2_1Segment(
            image: true, length: targetLength,
            sourceOffset: referenceLength, height: 2, width: 2)
          let segments = prefix + [target]
          var indices = [Int32]()
          for segment in segments {
            let offset = segment.sourceOffset + (segment.image ? textLength : 0)
            indices += (offset..<(offset + segment.length)).map(Int32.init)
          }
          var context = Tensor<Float16>(.CPU, .HWC(batchSize, textLength, 4096))
          var reference = Tensor<Float16>(.CPU, .WC(max(1, referenceLength), 64))
          var image = Tensor<Float16>(.CPU, .NHWC(batchSize, 2, 2, 64))
          for batch in 0..<batchSize {
            for row in 0..<textLength {
              for col in 0..<4096 {
                context[batch, row, col] = Float16(
                  sin(Float(1 + batch * 13 + row * 4096 + col) * 0.013))
              }
            }
            for y in 0..<2 {
              for x in 0..<2 {
                for col in 0..<64 {
                  image[batch, y, x, col] = Float16(
                    cos(Float(batch * 256 + y * 128 + x * 64 + col) * 0.07))
                }
              }
            }
          }
          for row in 0..<max(1, referenceLength) {
            for col in 0..<64 {
              reference[row, col] = Float16(sin(Float(row * 64 + col) * 0.03))
            }
          }
          var rotary = Tensor<Float>(.CPU, .NHWC(batchSize, count, 1, 128))
          var prefixRotary = Tensor<Float>(.CPU, .NHWC(batchSize, prefixLength, 1, 128))
          var targetRotary = Tensor<Float>(.CPU, .NHWC(batchSize, targetLength, 1, 128))
          var mask = Tensor<Float16>(.CPU, .NHWC(batchSize, 1, count, count))
          let layout = QwenImage2_1AttentionMask(segments)
          for batch in 0..<batchSize {
            let padding = batch == 0 ? 2 : 0
            var actual = prefix
            let last = actual.count - 1
            actual[last] = .init(
              image: false, length: actual[last].length - padding,
              sourceOffset: actual[last].sourceOffset)
            let positions = QwenImage2_1RotaryEmbedding(actual + [target])
            for row in 0..<count {
              let positionRow =
                row >= prefixLength ? row - padding : min(row, prefixLength - padding - 1)
              for col in 0..<128 {
                let value = positions[0, positionRow, 0, col]
                rotary[batch, row, 0, col] = value
                if row < prefixLength {
                  prefixRotary[batch, row, 0, col] = value
                } else {
                  targetRotary[batch, row - prefixLength, 0, col] = value
                }
              }
              for col in 0..<count {
                let padded = col >= prefixLength - padding && col < prefixLength
                mask[batch, 0, row, col] =
                  padded || !layout[0, 0, row, col].isFinite ? -Float16.greatestFiniteMagnitude : 0

              }
            }
          }
          let timesteps: [Float] = [0.9, 0.3]
          var timeTable = Tensor<Float16>(.CPU, .WC(3, 256))
          for (i, t) in (timesteps + [0]).enumerated() {
            let value = timeEmbedding(
              timestep: t * 1000, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000)
            for col in 0..<256 { timeTable[i, col] = Float16(value[0, col]) }
          }
          let contextInput = graph.variable(context.toGPU(0))
          let imageInput = graph.variable(image.toGPU(0))
          let referenceInput = graph.variable(reference.toGPU(0))
          let (fixedMapper, fixed) = QwenImage2_1Fixed(
            Float16.self, batchSize: batchSize,
            textLength: textLength, referenceLength: referenceLength, timesteps: timesteps.count,
            channels: 4096, layers: 32, segments: prefix.map { ($0.length, $0.image) },
            usesFlashAttention: flashAttention)
          let fixedInputs: [DynamicGraph.AnyTensor] =
            [
              contextInput, graph.variable(timeTable.toGPU(0)),
              graph.variable(prefixRotary.toGPU(0)),
            ]
            + (flash
              ? []
              : [graph.variable(Tensor<Float16>(from: QwenImage2_1AttentionMask(prefix)).toGPU(0))])
            + (references ? [referenceInput] : [])
          fixed.compile(inputs: fixedInputs)
          graph.openStore(
            path, flags: .readOnly, externalStore: TensorData.externalStore(filePath: path)
          ) {
            try! $0.read("dit", model: fixed, strict: true, codec: [.externalData(.mmap)])
          }
          let cached = fixed(inputs: fixedInputs[0], Array(fixedInputs.dropFirst()))
          XCTAssertEqual(cached.count, 69)
          let denoisers = (0..<batchSize).map { batch in
            QwenImage2_1(
              Float16.self, batchSize: 1, height: 2, width: 2,
              prefixLength: prefixLength - (batch == 0 ? 2 : 0),
              channels: 4096, layers: 32, usesFlashAttention: flashAttention)
          }
          let (baselineMapper, baseline) = QwenImage2_1Unsplit(
            Float16.self, batchSize: batchSize,
            height: 2, width: 2, textLength: textLength, referenceLength: referenceLength,
            usesFlashAttention: flash)
          for (step, t) in timesteps.enumerated() {
            let inputs: [[DynamicGraph.AnyTensor]] = (0..<batchSize).map { batch in
              let length = prefixLength - (batch == 0 ? 2 : 0)
              let image = imageInput[batch..<(batch + 1), 0..<2, 0..<2, 0..<64].copied()
              let rot = graph.variable(targetRotary.toGPU(0))[
                batch..<(batch + 1), 0..<targetLength, 0..<1, 0..<128
              ].copied()
              let modulation: [DynamicGraph.AnyTensor] = cached.prefix(5).map {
                $0.as(of: Float16.self)[step..<(step + 1), 0..<4096].copied()
              }
              let kvs: [DynamicGraph.AnyTensor] = cached.dropFirst(5).map {
                $0.as(of: Float16.self)[batch..<(batch + 1), 0..<length, 0..<32, 0..<128].copied()
              }
              return [image, rot] + modulation + kvs
            }
            let baselineInputs: [DynamicGraph.AnyTensor] =
              [
                imageInput,
                graph.variable(Tensor<Float16>(from: QwenImage2_1UnsplitTimeEmbedding(t)).toGPU(0)),
                contextInput, graph.variable(rotary.toGPU(0)), graph.variable(mask.toGPU(0)),
                graph.variable(Tensor<Int32>(indices, .CPU, .C(count)).toGPU(0)),
              ]
              + (references ? [referenceInput] : [])
            if step == 0 {
              for batch in 0..<batchSize { denoisers[batch].1.compile(inputs: inputs[batch]) }
              baseline.compile(inputs: baselineInputs)
              let expectedMapping = baselineMapper(.diffusers)
              let fixedMapping = fixedMapper(.diffusers)
              let denoiserMapping = denoisers[0].0(.diffusers)
              XCTAssertEqual(
                Set(fixedMapping.keys).union(denoiserMapping.keys), Set(expectedMapping.keys))
              for mapping in [fixedMapping, denoiserMapping] {
                for (key, names) in mapping {
                  XCTAssertEqual(Array(names), Array(expectedMapping[key]!), key)
                }
              }
              graph.openStore(
                path, flags: .readOnly, externalStore: TensorData.externalStore(filePath: path)
              ) {
                for (_, denoiser) in denoisers {
                  try! $0.read("dit", model: denoiser, strict: true, codec: [.externalData(.mmap)])
                }
                try! $0.read("dit", model: baseline, strict: true, codec: [.externalData(.mmap)])
              }
            }
            let results = (0..<batchSize).map { batch in
              denoisers[batch].1(inputs: inputs[batch][0], Array(inputs[batch].dropFirst()))[0]
                .as(of: Float16.self).rawValue.toCPU()
            }
            let expected = baseline(inputs: baselineInputs[0], Array(baselineInputs.dropFirst()))[0]
              .as(of: Float16.self).rawValue.toCPU()
            var squaredError: Double = 0
            var squaredSignal: Double = 0
            var maximum: Float = 0
            var finite = true
            for batch in 0..<batchSize {
              for y in 0..<2 {
                for x in 0..<2 {
                  for col in 0..<64 {
                    let a = Float(results[batch][0, y, x, col])
                    let b = Float(expected[batch, y, x, col])
                    finite = finite && a.isFinite && b.isFinite
                    maximum = max(maximum, abs(a - b))
                    squaredError += Double((a - b) * (a - b))
                    squaredSignal += Double(b * b)
                  }
                }
              }
            }
            XCTAssertTrue(finite)
            let relative = sqrt(squaredError / max(squaredSignal, 1e-12))
            print(
              "Qwen2.1 parity flash=\(flashAttention) refs=\(references) t=\(t): relative RMS=\(relative), max=\(maximum)"
            )
            // INT8 attention uses a wider error budget than FP16 attention.
            XCTAssertLessThan(relative, flashAttention == .quantized ? 0.1 : 0.01)
            XCTAssertLessThan(maximum, flashAttention == .quantized ? 1 : 0.1)
          }
        }
      }
    }
  }
}
