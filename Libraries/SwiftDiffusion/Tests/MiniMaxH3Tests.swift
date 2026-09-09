import BinaryResources
import Diffusion
import Foundation
import NNC
import Tokenizer
import XCTest

final class MiniMaxH3Tests: XCTestCase {
  func testFirstPicturePrefixMatchesRotaryLayout() {
    let tokenizer = TiktokenTokenizer(
      vocabulary: BinaryResources.vocab_qwen3_json, merges: BinaryResources.merges_qwen3_txt,
      specialTokens: [
        "<|endoftext|>": 151_643, "<|vision_start|>": 151_652,
        "<|vision_end|>": 151_653, "<|image_pad|>": 151_655,
      ])
    let tokens = tokenizer.tokenize(
      text: "<Picture 1>: <|vision_start|><|image_pad|><|vision_end|>Hello"
    ).0
    XCTAssertEqual(tokens.firstIndex(of: 151_652), 6)
    XCTAssertEqual(tokens.firstIndex(of: 151_655), 7)
  }

  func testGroupedRotaryPreservesEveryTokenPosition() {
    for frames in [1, 2, 7] {
      for (height, width) in [(4, 6), (6, 4)] {
        for textLength in [14, 20] {
          let original = MiniMaxH3RotaryEmbedding(
            textLength: textLength, audioLength: 4, videoFrames: frames,
            videoHeight: height, videoWidth: width,
            referenceImages: [(height, width, Float(textLength))])
          let grouped = MiniMaxH3RotaryEmbedding(
            textLength: textLength, audioLength: 4, videoFrames: frames,
            videoHeight: height, videoWidth: width,
            referenceImages: [(height, width, Float(textLength))], visionTokenRanges: [6..<12])
          let videoOffset = textLength + height / 2 * (width / 2) + 4
          let order =
            Array(0..<6) + Array(12..<textLength)
            + Array(textLength..<videoOffset) + Array(6..<12)
            + Array(videoOffset..<original.shape[1])
          XCTAssertEqual(order.count, grouped.shape[1])
          for (row, originalRow) in order.enumerated() {
            for channel in 0..<128 {
              XCTAssertEqual(grouped[0, row, 0, channel], original[0, originalRow, 0, channel])
            }
          }
        }
      }
    }
  }

  func testQwenVLImageRotaryUsesInterleavedAxes() {
    let plain = QwenVLRotaryEmbedding(sequenceLength: 12, of: Float.self)
    let image = QwenVLRotaryEmbedding(
      sequenceLength: 12, images: [(start: 3, height: 2, width: 3)], of: Float.self)
    for row in 0..<12 {
      for k in 0..<64 {
        let position: Int
        if row < 3 {
          position = row
        } else if row < 9 {
          if k % 3 == 1 && k < 60 {
            position = 3 + (row - 3) / 3
          } else if k % 3 == 2 && k < 60 {
            position = 3 + (row - 3) % 3
          } else {
            position = 3
          }
        } else {
          position = row - 3
        }
        XCTAssertEqual(image[0, row, 0, k * 2], plain[0, position, 0, k * 2])
        XCTAssertEqual(image[0, row, 0, k * 2 + 1], plain[0, position, 0, k * 2 + 1])
      }
    }
  }

  func testQwenDeepStackReusesTextWeightsAndInjectsAfterBlock() throws {
    let graph = DynamicGraph()
    let path = FileManager.default.temporaryDirectory.appendingPathComponent(
      "qwen-deepstack-\(UUID().uuidString).ckpt"
    ).path
    defer { try? FileManager.default.removeItem(atPath: path) }
    try graph.withNoGrad {
      let tokens = graph.variable(Tensor<Int32>([1, 2, 3, 4], .CPU, .C(4)).toGPU(0))
      let rotary = graph.variable(
        QwenVLRotaryEmbedding(sequenceLength: 4, of: Float16.self).toGPU(0))
      let causalMask = graph.variable(.GPU(0), .NHWC(1, 1, 4, 4), of: Float16.self)
      causalMask.full(0)
      let plain = Qwen3(
        Float16.self, vocabularySize: 8, width: 16, tokenLength: 4, layers: 3, MLP: 32,
        heads: 8, outputHiddenStates: [0, 1, 2], noFinalNormalizedOutput: true,
        batchSize: 1, usesFlashAttention: true)
      let expected = plain(inputs: tokens, rotary, causalMask).map {
        $0.as(of: Float16.self).rawValue.toCPU()
      }
      graph.openStore(path) { $0.write("text_model", model: plain) }
      let injectedModel = Qwen3(
        Float16.self, vocabularySize: 8, width: 16, tokenLength: 4, layers: 3, MLP: 32,
        heads: 8, outputHiddenStates: [0, 1, 2], noFinalNormalizedOutput: true,
        batchSize: 1, usesFlashAttention: true, injectEmbeddings: true, deepStackLayers: 3)
      let mask = graph.variable(.GPU(0), .WC(4, 1), of: Float16.self)
      mask.full(1)
      let zero = graph.variable(.GPU(0), .WC(4, 16), of: Float16.self)
      zero.full(0)
      let deepStack = graph.variable(like: zero)
      deepStack.full(0)
      let inputs: [DynamicGraph.AnyTensor] = [
        rotary, causalMask, mask, zero, deepStack, zero, zero,
      ]
      injectedModel.compile(inputs: [tokens] + inputs)
      try graph.openStore(path, flags: .readOnly) {
        try $0.read("text_model", model: injectedModel, strict: true)
      }
      let unchanged = injectedModel(inputs: tokens, inputs).map {
        $0.as(of: Float16.self).rawValue.toCPU()
      }
      deepStack[2..<3, 0..<16].full(0.25)
      let actual = injectedModel(inputs: tokens, inputs).map {
        $0.as(of: Float16.self).rawValue.toCPU()
      }
      for layer in 0..<3 {
        for row in 0..<4 {
          for channel in 0..<16 {
            XCTAssertEqual(unchanged[layer][row, channel], expected[layer][row, channel])
            XCTAssertTrue(actual[layer][row, channel].isFinite)
            if layer == 0 {
              XCTAssertEqual(
                Float(actual[layer][row, channel]),
                Float(expected[layer][row, channel]) + (row == 2 ? 0.25 : 0), accuracy: 0.002)
            }
          }
        }
      }
    }
  }

  func testMultipleVisionRangesPreserveOriginalContextPositions() {
    let ranges = [6..<12, 18..<28]
    let plain = MiniMaxH3RotaryEmbedding(textLength: 32)
    let packed = MiniMaxH3RotaryEmbedding(textLength: 32, visionTokenRanges: ranges)
    let order =
      (0..<32).filter { row in !ranges.contains { $0.contains(row) } }
      + ranges.flatMap { Array($0) }
    for (row, original) in order.enumerated() {
      for channel in 0..<128 {
        XCTAssertEqual(packed[0, row, 0, channel], plain[0, original, 0, channel])
      }
    }
  }

  func testReferenceImagePositionsAndIndependentSpatialGrids() {
    let frames = 7
    let textLength = 20
    let lastPosition = Float(textLength) + Float(22 - 1) * 5 / 3
    let endpoints = MiniMaxH3RotaryEmbedding(
      textLength: textLength, audioLength: 4, videoFrames: frames,
      videoHeight: 4, videoWidth: 6,
      referenceImages: [(4, 6, Float(textLength)), (4, 6, lastPosition)])
    let plain = MiniMaxH3RotaryEmbedding(
      textLength: textLength, audioLength: 4, videoFrames: frames,
      videoHeight: 4, videoWidth: 6)
    for row in 0..<(4 + frames * 6) {
      for channel in 0..<128 {
        XCTAssertEqual(
          endpoints[0, textLength + 12 + row, 0, channel], plain[0, textLength + row, 0, channel])
      }
    }
    XCTAssertEqual(endpoints[0, textLength + 6, 0, 0], cos(lastPosition), accuracy: 1e-6)
    let references = MiniMaxH3RotaryEmbedding(
      textLength: textLength, audioLength: 4, videoFrames: frames,
      videoHeight: 4, videoWidth: 6,
      referenceImages: [(4, 6, 20), (8, 4, 21)], videoPosition: 22)
    let second = MiniMaxH3RotaryEmbedding(
      textLength: 0, videoFrames: 1, videoHeight: 8, videoWidth: 4, videoPosition: 21)
    for row in 0..<8 {
      for channel in 0..<128 {
        XCTAssertEqual(references[0, textLength + 6 + row, 0, channel], second[0, row, 0, channel])
      }
    }
    XCTAssertEqual(references[0, textLength + 14 + 4, 0, 0], cos(Float(22)), accuracy: 1e-6)
  }

  func testQwenVLMultipleImagesAdvanceByCompressedGridSize() {
    let rotary = QwenVLRotaryEmbedding(
      sequenceLength: 24, images: [(3, 2, 3), (12, 3, 2)], of: Float.self)
    let plain = QwenVLRotaryEmbedding(sequenceLength: 24, of: Float.self)
    for row in 0..<24 {
      for k in 0..<64 {
        let position: Int
        if (3..<9).contains(row) {
          position =
            3 + (k % 3 == 1 && k < 60 ? (row - 3) / 3 : (k % 3 == 2 && k < 60 ? (row - 3) % 3 : 0))
        } else if (12..<18).contains(row) {
          position =
            9
            + (k % 3 == 1 && k < 60 ? (row - 12) / 2 : (k % 3 == 2 && k < 60 ? (row - 12) % 2 : 0))
        } else {
          position = row - (row >= 9 ? 3 : 0) - (row >= 18 ? 3 : 0)
        }
        for component in 0..<2 {
          XCTAssertEqual(
            rotary[0, row, 0, 2 * k + component], plain[0, position, 0, 2 * k + component])
        }
      }
    }
  }

  func testFirstFrameRotaryPreservesTargetPositions() {
    for frames in [1, 2, 7] {
      for (height, width) in [(4, 6), (6, 4)] {
        let textLength = 3
        let audioLength = 4
        let frameLength = height / 2 * (width / 2)
        let plain = MiniMaxH3RotaryEmbedding(
          textLength: textLength, audioLength: audioLength, videoFrames: frames,
          videoHeight: height, videoWidth: width)
        let conditioned = MiniMaxH3RotaryEmbedding(
          textLength: textLength, audioLength: audioLength, videoFrames: frames,
          videoHeight: height, videoWidth: width,
          referenceImages: [(height, width, Float(textLength))])
        XCTAssertEqual(conditioned.shape[1], plain.shape[1] + frameLength)
        for row in 0..<plain.shape[1] {
          let targetRow = row < textLength ? row : row + frameLength
          for channel in 0..<128 {
            XCTAssertEqual(plain[0, row, 0, channel], conditioned[0, targetRow, 0, channel])
          }
        }
        for row in 0..<frameLength {
          for channel in 0..<128 {
            XCTAssertEqual(
              conditioned[0, textLength + row, 0, channel],
              plain[0, textLength + audioLength + row, 0, channel])
          }
        }
      }
    }
  }

  func testExtractConditionsKeepsKeyframeIndependentOfStepAndCFG() {
    let graph = DynamicGraph()
    graph.withNoGrad {
      let text = graph.variable(.CPU, .HWC(2, 3, 8), of: Float.self)
      let rotary = graph.variable(.CPU, .NHWC(2, 25, 1, 128), of: Float.self)
      let firstFrame = graph.variable(.CPU, .NHWC(1, 2, 3, 8), of: Float.self)
      firstFrame.full(0.25)
      let modulation = graph.variable(.CPU, .HWC(3, 1, 8), of: Float.self)
      for step in 0..<3 { modulation[step..<(step + 1), 0..<1, 0..<8].full(Float(step)) }
      for referenceCount in 0...3 {
        let prefix: [DynamicGraph.AnyTensor] =
          [text, rotary] + Array(repeating: firstFrame, count: referenceCount)
        let count = 50 * (referenceCount > 0 ? 24 : 18) + 4
        let conditions = prefix + Array(repeating: modulation, count: count)
        for step in 0..<3 {
          let extracted = UNetExtractConditions(
            of: Float.self, graph: graph, index: step, batchSize: 4,
            tokenLengthUncond: 2, tokenLengthCond: 3, conditions: conditions,
            referenceImageCount: referenceCount, version: .minimaxH3, modifier: .fl2va,
            isCfgEnabled: true)
          XCTAssertEqual(extracted.count, prefix.count + count)
          XCTAssertEqual(Array(extracted[0].shape), [2, 3, 8])
          if referenceCount > 0 {
            XCTAssertEqual(Array(extracted[2].shape), [1, 2, 3, 8])
            XCTAssertEqual(DynamicGraph.Tensor<Float>(extracted[2]).rawValue[0, 1, 2, 7], 0.25)
          }
          for index in [prefix.count, extracted.count - 1] {
            let value = DynamicGraph.Tensor<Float>(extracted[index]).rawValue
            XCTAssertEqual(Array(value.shape), [1, 8])
            XCTAssertEqual(value[0, 7], Float(step))
          }
        }
      }
    }
  }

  func testFixedKeyframeModulationReusesVideoWeights() throws {
    let hiddenSize = 128
    let graph = DynamicGraph()
    let path = FileManager.default.temporaryDirectory.appendingPathComponent(
      "h3-fixed-\(UUID().uuidString).ckpt"
    ).path
    defer { try? FileManager.default.removeItem(atPath: path) }
    try graph.withNoGrad {
      let frequencies = graph.variable(.GPU(0), .HWC(2, 2, 256), of: Float.self)
      frequencies.randn()
      let text = graph.variable(.GPU(0), .HWC(1, 4, 5120), of: Float16.self)
      text.full(0.1)
      var conditionedFrequencies = graph.variable(.GPU(0), .HWC(2, 3, 256), of: Float.self)
      conditionedFrequencies[0..<2, 0..<2, 0..<256] = frequencies
      // At equal timesteps the new keyframe modulation must equal video modulation exactly.
      conditionedFrequencies[0..<2, 2..<3, 0..<256] = frequencies[0..<2, 0..<1, 0..<256]
      let firstFrame = graph.variable(.GPU(0), .NHWC(1, 4, 6, 24), of: Float16.self)
      let lastFrame = graph.variable(.GPU(0), .NHWC(1, 8, 4, 24), of: Float16.self)
      firstFrame.randn()
      lastFrame.randn()
      let conditioned = MiniMaxH3Fixed(
        timesteps: 2, hiddenSize: hiddenSize, layers: 2, textLength: (0, 4),
        usesFlashAttention: .scale1,
        referenceImageCount: 2)
      let outputs = conditioned(inputs: text, conditionedFrequencies, firstFrame, lastFrame)
      XCTAssertEqual(Array(outputs[1].shape), [1, 2, 3, hiddenSize])
      XCTAssertEqual(Array(outputs[2].shape), [1, 4, 2, hiddenSize])
      let actual = outputs.dropFirst(3).map { $0.as(of: Float.self).toCPU().rawValue }
      graph.openStore(path) { $0.write("dit", model: conditioned) }
      let plain = MiniMaxH3Fixed(
        timesteps: 2, hiddenSize: hiddenSize, layers: 2, textLength: (0, 4),
        usesFlashAttention: .scale1)
      plain.compile(inputs: text, frequencies)
      try graph.openStore(path, flags: .readOnly) {
        try $0.read("dit", model: plain, strict: true)
      }.get()
      let expected = plain(inputs: text, frequencies).dropFirst().map {
        $0.as(of: Float.self).toCPU().rawValue
      }
      // The cached projection uses the original denoiser weight names, including its bias.
      for (index, reference) in [firstFrame, lastFrame].enumerated() {
        let image = Input()
        let projection = Model(
          [image],
          [
            Convolution(
              groups: 1, filters: hiddenSize, filterSize: [2, 2], hint: Hint(stride: [2, 2]),
              format: .OIHW, name: "proj_in")(image).to(.Float32)
          ])
        projection.compile(inputs: reference)
        try graph.openStore(path, flags: .readOnly) {
          try $0.read("dit", model: projection, strict: true)
        }.get()
        let projected = projection(inputs: reference)[0].as(of: Float.self).toCPU().rawValue
        let count = projected.shape.reduce(1, *)
        let cached = outputs[index + 1].as(of: Float16.self).toCPU().rawValue.reshaped(.C(count))
        let direct = projected.reshaped(.C(count))
        for i in 0..<count { XCTAssertEqual(Float(cached[i]), direct[i]) }
        let tile = reference[0..<1, 2..<4, 0..<4, 0..<24].copied()
        let tileInput = Input()
        let tileModel = Model(
          [tileInput],
          [
            Convolution(
              groups: 1, filters: hiddenSize, filterSize: [2, 2], hint: Hint(stride: [2, 2]),
              format: .OIHW, name: "proj_in")(tileInput).to(.Float32)
          ])
        tileModel.compile(inputs: tile)
        try graph.openStore(path, flags: .readOnly) {
          try $0.read("dit", model: tileModel, strict: true)
        }.get()
        let tileProjection = tileModel(inputs: tile)[0].as(of: Float.self).toCPU().rawValue
        for x in 0..<2 {
          for channel in 0..<hiddenSize {
            XCTAssertEqual(tileProjection[0, 0, x, channel], projected[0, 1, x, channel])
          }
        }
      }
      XCTAssertEqual(actual.count, 2 * 24 + 4)
      for layer in 0..<2 {
        for chunk in 0..<6 {
          for modality in 0..<4 {
            let value = actual[layer * 24 + chunk * 4 + modality]
            let reference = expected[layer * 18 + chunk * 3 + (modality == 3 ? 0 : modality)]
            for step in 0..<2 {
              for channel in 0..<hiddenSize {
                XCTAssertEqual(value[step, 0, channel], reference[step, 0, channel], accuracy: 1e-5)
              }
            }
          }
        }
      }
      for index in 0..<4 {
        for step in 0..<2 {
          for channel in 0..<hiddenSize {
            XCTAssertEqual(
              actual[48 + index][step, 0, channel], expected[36 + index][step, 0, channel],
              accuracy: 1e-5)
          }
        }
      }
    }
  }

  func testFixedRefinerSlicesCFGAndVisionWithoutCrossBranchAttention() throws {
    let flags = DynamicGraph.flags
    DynamicGraph.flags.insert(.disableMFAAppleNeuralEngine)
    defer { DynamicGraph.flags = flags }
    DynamicGraph.setSeed(42)
    let watermark = DynamicGraph.queueWatermark
    DynamicGraph.queueWatermark = 1
    defer { DynamicGraph.queueWatermark = watermark }
    let graph = DynamicGraph()
    let path = FileManager.default.temporaryDirectory.appendingPathComponent(
      "h3-refiner-\(UUID().uuidString).ckpt"
    ).path
    defer { try? FileManager.default.removeItem(atPath: path) }
    try graph.withNoGrad {
      let hiddenSize = 128
      let frequencies = graph.variable(.GPU(0), .HWC(2, 2, 256), of: Float16.self)
      frequencies.full(0.1)
      for lengths in [(3, 5), (5, 3)] {
        for visionLength in [0, 2] {
          let paddedLength = max(lengths.0, lengths.1)
          var text = graph.variable(
            .GPU(0), .HWC(2, paddedLength + visionLength, 5120), of: Float16.self)
          text.randn()
          let fixed = MiniMaxH3Fixed(
            timesteps: 2, hiddenSize: hiddenSize, layers: 1,
            textLength: (lengths.0 + visionLength, lengths.1 + visionLength),
            usesFlashAttention: .scale1, visionLength: visionLength)
          let outputs = fixed(inputs: text, frequencies)
          let batched = outputs[0].as(of: Float.self).toCPU().rawValue
          XCTAssertEqual(Array(batched.shape), [2, paddedLength + visionLength, hiddenSize])
          XCTAssertEqual(outputs.count, 23)
          graph.openStore(path) { $0.write("dit", model: fixed) }
          for (batch, length) in [lengths.0, lengths.1].enumerated() {
            var branch = text[batch..<(batch + 1), 0..<length, 0..<5120].copied()
            if visionLength > 0 {
              branch = Functional.concat(
                axis: 1, branch,
                text[
                  batch..<(batch + 1), paddedLength..<(paddedLength + visionLength), 0..<5120
                ].copied())
            }
            let single = MiniMaxH3Fixed(
              timesteps: 2, hiddenSize: hiddenSize, layers: 1,
              textLength: (0, length + visionLength),
              usesFlashAttention: .scale1, visionLength: visionLength)
            single.compile(inputs: branch, frequencies)
            try graph.openStore(path, flags: .readOnly) {
              try $0.read("dit", model: single, strict: true)
            }.get()
            let expected = single(inputs: branch, frequencies)[0].as(of: Float.self).toCPU()
              .rawValue
            var error = 0.0
            var magnitude = 0.0
            for row in 0..<(length + visionLength) {
              let actualRow = row < length ? row : paddedLength + row - length
              for channel in 0..<hiddenSize {
                let a = Double(batched[batch, actualRow, channel])
                let b = Double(expected[0, row, channel])
                XCTAssertTrue(a.isFinite && b.isFinite)
                error += (a - b) * (a - b)
                magnitude += b * b
              }
            }
            // Packed and single-branch GEMMs have different row counts. Compare numerically;
            // padding and cross-branch independence below must still be bit-exact.
            XCTAssertLessThan(sqrt(error / max(magnitude, 1e-30)), 0.003)
            for row in length..<paddedLength {
              for channel in 0..<hiddenSize { XCTAssertEqual(batched[batch, row, channel], 0) }
            }
          }
          // Padding is not part of either refiner attention segment.
          let padding = graph.variable(
            .GPU(0), .HWC(1, paddedLength - min(lengths.0, lengths.1), 5120), of: Float16.self)
          padding.full(100)
          let shorter = lengths.0 < lengths.1 ? 0 : 1
          text[shorter..<(shorter + 1), min(lengths.0, lengths.1)..<paddedLength, 0..<5120] =
            padding
          let changedPadding = fixed(inputs: text, frequencies)[0].as(of: Float.self).toCPU()
            .rawValue
          for batch in 0..<2 {
            for row in 0..<(paddedLength + visionLength) {
              for channel in 0..<hiddenSize {
                XCTAssertEqual(changedPadding[batch, row, channel], batched[batch, row, channel])
              }
            }
          }
          // Changing one entire branch cannot affect the other branch (including vision rows).
          for changedBatch in 0..<2 {
            let changed = text.copied()
            changed[changedBatch..<(changedBatch + 1), 0..<(paddedLength + visionLength), 0..<5120]
              .full(1)
            let result = fixed(inputs: changed, frequencies)[0].as(of: Float.self).toCPU().rawValue
            let unchangedBatch = 1 - changedBatch
            for row in 0..<(paddedLength + visionLength) {
              for channel in 0..<hiddenSize {
                XCTAssertEqual(
                  result[unchangedBatch, row, channel], batched[unchangedBatch, row, channel])
              }
            }
          }
        }
      }
    }
  }

  func testDenoiserUsesKeyframeWithoutReturningConditionRows() throws {
    DynamicGraph.setSeed(42)
    let watermark = DynamicGraph.queueWatermark
    DynamicGraph.queueWatermark = 1
    defer { DynamicGraph.queueWatermark = watermark }
    let graph = DynamicGraph()
    let path = FileManager.default.temporaryDirectory.appendingPathComponent(
      "h3-denoiser-\(UUID().uuidString).ckpt"
    ).path
    defer { try? FileManager.default.removeItem(atPath: path) }
    try graph.withNoGrad {
      let video = graph.variable(.GPU(0), .NHWC(2, 4, 6, 24), of: Float16.self)
      let audio = graph.variable(.GPU(0), .HWC(1, 4, 32), of: Float16.self)
      let text = graph.variable(.GPU(0), .HWC(1, 8, 8), of: Float.self)
      let firstFrame = graph.variable(.GPU(0), .NHWC(1, 2, 3, 8), of: Float16.self)
      video.full(0.1)
      audio.full(0.2)
      text.full(0.3)
      firstFrame.full(0.4)
      let rotary = graph.variable(
        Tensor<Float16>(
          from: MiniMaxH3RotaryEmbedding(
            textLength: 8, audioLength: 4, videoFrames: 2, videoHeight: 4, videoWidth: 6,
            referenceImages: [(4, 6, 8)])
        ).toGPU(0))
      let modulation = graph.variable(.GPU(0), .WC(1, 8), of: Float16.self)
      modulation.full(0.1)
      let model = MiniMaxH3(
        hiddenSize: 8, layers: 1, textLength: 8, audioLength: 4, videoFrames: 2,
        videoHeight: 4, videoWidth: 6, usesFlashAttention: .scale1, referenceImageSizes: [(4, 6)])
      let inputs: [DynamicGraph.AnyTensor] =
        [audio, text, rotary, firstFrame] + Array(repeating: modulation, count: 28)
      let plainRotary = graph.variable(
        Tensor<Float16>(
          from: MiniMaxH3RotaryEmbedding(
            textLength: 8, audioLength: 4, videoFrames: 2, videoHeight: 4, videoWidth: 6)
        ).toGPU(0))
      let plain = MiniMaxH3(
        hiddenSize: 8, layers: 1, textLength: 8, audioLength: 4, videoFrames: 2,
        videoHeight: 4, videoWidth: 6, usesFlashAttention: .scale1)
      _ = plain(
        inputs: video, [audio, text, plainRotary] + Array(repeating: modulation, count: 22))
      graph.openStore(path) { $0.write("dit", model: plain) }
      model.compile(inputs: [video] + inputs)
      try graph.openStore(path, flags: .readOnly) {
        try $0.read("dit", model: model, strict: true)
      }
      let first = model(inputs: video, inputs).map { $0.as(of: Float16.self).rawValue.toCPU() }
      let withVision = MiniMaxH3(
        hiddenSize: 8, layers: 1, textLength: 8, audioLength: 4, videoFrames: 2,
        videoHeight: 4, videoWidth: 6, usesFlashAttention: .scale1, referenceImageSizes: [(4, 6)],
        visionLength: 1)
      let groupedRotary = graph.variable(
        Tensor<Float16>(
          from: MiniMaxH3RotaryEmbedding(
            textLength: 8, audioLength: 4, videoFrames: 2, videoHeight: 4, videoWidth: 6,
            referenceImages: [(4, 6, 8)], visionTokenRanges: [6..<7])
        ).toGPU(0))
      let visionInputs: [DynamicGraph.AnyTensor] =
        [
          audio,
          Functional.concat(
            axis: 1, text[0..<1, 0..<6, 0..<8].copied(),
            text[0..<1, 7..<8, 0..<8].copied(),
            text[0..<1, 6..<7, 0..<8].copied()), groupedRotary, firstFrame,
        ] + Array(repeating: modulation, count: 28)
      withVision.compile(inputs: [video] + visionInputs)
      try graph.openStore(path, flags: .readOnly) {
        try $0.read("dit", model: withVision, strict: true)
      }
      let split = withVision(inputs: video, visionInputs).map {
        $0.as(of: Float16.self).rawValue.toCPU()
      }
      for output in 0..<2 {
        let count = first[output].shape.reduce(1, *)
        let a = first[output].reshaped(.C(count))
        let b = split[output].reshaped(.C(count))
        for i in 0..<count { XCTAssertEqual(Float(a[i]), Float(b[i]), accuracy: 0.002) }
      }
      firstFrame.full(-0.4)
      let second = model(inputs: video, inputs).map { $0.as(of: Float16.self).rawValue.toCPU() }
      XCTAssertEqual(Array(first[0].shape), [2, 4, 6, 24])
      XCTAssertEqual(Array(first[1].shape), [1, 4, 32])
      var difference: Float = 0
      for index in 0..<2 {
        let count = first[index].shape.reduce(1, *)
        let a = first[index].reshaped(.C(count))
        let b = second[index].reshaped(.C(count))
        for j in 0..<count {
          XCTAssertTrue(a[j].isFinite && b[j].isFinite)
          difference += abs(Float(a[j]) - Float(b[j]))
        }
      }
      XCTAssertGreaterThan(difference, 0.001)
      let reference = graph.variable(.GPU(0), .NHWC(1, 4, 2, 8), of: Float16.self)
      reference.full(0.2)
      let multipleRotary = graph.variable(
        Tensor<Float16>(
          from: MiniMaxH3RotaryEmbedding(
            textLength: 8, audioLength: 4, videoFrames: 2, videoHeight: 4, videoWidth: 6,
            referenceImages: [(4, 6, 8), (8, 4, 9)], videoPosition: 10)
        ).toGPU(0))
      let multiple = MiniMaxH3(
        hiddenSize: 8, layers: 1, textLength: 8, audioLength: 4, videoFrames: 2,
        videoHeight: 4, videoWidth: 6, usesFlashAttention: .scale1,
        referenceImageSizes: [(4, 6), (8, 4)])
      let multipleInputs: [DynamicGraph.AnyTensor] =
        [audio, text, multipleRotary, firstFrame, reference]
        + Array(repeating: modulation, count: 28)
      multiple.compile(inputs: [video] + multipleInputs)
      try graph.openStore(path, flags: .readOnly) {
        try $0.read("dit", model: multiple, strict: true)
      }
      let outputs = multiple(inputs: video, multipleInputs)
      XCTAssertEqual(Array(outputs[0].shape), [2, 4, 6, 24])
      XCTAssertEqual(Array(outputs[1].shape), [1, 4, 32])
      for output in outputs {
        let value = output.as(of: Float16.self).rawValue.toCPU()
        let count = value.shape.reduce(1, *)
        let flat = value.reshaped(.C(count))
        for i in 0..<count { XCTAssertTrue(flat[i].isFinite) }
      }
    }
  }
}
