import LLM
import NNC
import XCTest

final class GLM5_3VisionTests: XCTestCase {
  func testVisionSizeMatchesReferenceProcessor() {
    // (height, width) -> canvas grid (rows, columns) and content size, from transformers'
    // Glm5NextImageProcessor (smart_resize, then the aspect-preserving content size).
    let cases: [(input: (Int, Int), grid: (Int, Int), content: (Int, Int))] = [
      ((512, 512), (38, 38), (512, 512)),
      ((520, 1_200), (38, 86), (520, 1_200)),
      ((512, 1_536), (38, 110), (512, 1_536)),
      ((1_536, 512), (110, 38), (1_536, 512)),
      ((40, 40), (8, 8), (112, 112)),
      ((2_040, 3_000), (146, 216), (2_040, 3_000)),
      ((3_300, 4_100), (160, 200), (2_240, 2_783)),
      ((2_800, 4_200), (146, 218), (2_034, 3_052)),
      ((1, 1), (8, 8), (112, 112)),
      ((1, 100_000), (2, 7_144), (1, 100_000)),
      ((100_000, 1), (7_144, 2), (100_000, 1)),
      ((2_160, 3_840), (134, 238), (1_874, 3_332)),
      ((3_840, 2_160), (238, 134), (3_332, 1_874)),
      ((3_024, 4_032), (154, 206), (2_156, 2_874)),
      ((100, 3_000), (8, 216), (100, 3_000)),
      ((7, 13), (6, 12), (84, 156)),
      ((123, 457), (10, 34), (123, 457)),
    ]
    let configuration = GLM5_3VisionConfiguration.glm5_3Flash
    for (input, grid, content) in cases {
      let size = GLM5_3VisionSize(height: input.0, width: input.1)
      XCTAssertEqual(size.canvas.height / configuration.patchSize, grid.0, "\(input)")
      XCTAssertEqual(size.canvas.width / configuration.patchSize, grid.1, "\(input)")
      XCTAssertEqual(size.content.height, content.0, "\(input)")
      XCTAssertEqual(size.content.width, content.1, "\(input)")
      XCTAssertLessThanOrEqual(
        GLM5_3VisionTokenCount(gridThw: (1, grid.0, grid.1)), configuration.maximumImageTokens,
        "\(input)")
    }
  }

  func testVisionPreprocessResizesAsReferenceProcessor() {
    // 8-bit patterns upscaled (40x40 to 112x112) and downscaled (a 64-token budget), with
    // grids and level checksums from transformers' Glm5NextImageProcessor, which resizes
    // the 8-bit image with torchvision's antialiased bicubic.
    let cases: [(size: (Int, Int), maximumImageTokens: Int, grid: (Int, Int), checksum: Int)] = [
      ((40, 40), 8_000, (8, 8), 1_209_649_208),
      ((200, 300), 64, (12, 18), 4_079_026_494),
      ((300, 200), 64, (20, 12), 4_080_781_125),
    ]
    for (size, maximumImageTokens, grid, checksum) in cases {
      var configuration = GLM5_3VisionConfiguration.glm5_3Flash
      configuration.maximumImageTokens = maximumImageTokens
      let (height, width) = size
      var levels = [Float]()
      for y in 0..<height {
        for x in 0..<width {
          for c in 0..<3 {
            levels.append(Float((x * 7 + y * 13 + c * 29 + (x * y) % 17) % 256) / 255)
          }
        }
      }
      let (patches, patchGrid) = GLM5_3VisionPreprocess(
        Tensor<Float>(levels, .CPU, .NHWC(1, height, width, 3)), configuration: configuration)
      XCTAssertEqual(patchGrid.h, grid.0, "\(size)")
      XCTAssertEqual(patchGrid.w, grid.1, "\(size)")
      // Undo the normalization to recover each value's 8-bit level.
      let columns = configuration.patchVectorSize
      let values = patches.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
      var sum = 0
      for (index, value) in values.enumerated() {
        let channel = (index % columns) / (columns / 3)
        let level =
          (Double(value) * Double(configuration.imageStd[channel])
            + Double(configuration.imageMean[channel])) * 255
        sum += Int(level.rounded()) * (index % 251 + 1)
      }
      XCTAssertEqual(sum, checksum, "\(size)")
    }
  }

  func testVisionTransformerFollowsGridGeometryAcrossImages() {
    var configuration = GLM5_3VisionConfiguration.glm5_3Flash
    configuration.layers = 1
    configuration.hiddenSize = 16
    configuration.heads = 2
    configuration.intermediateSize = 32
    configuration.patchSize = 2
    configuration.outputHiddenSize = 8
    configuration.projectionIntermediateSize = 16
    // 4x12, 12x4, 6x8 and 8x6 have 48 patches each, so one graph serves them all and only
    // the rotary table carries the geometry; 4x4 rebuilds the graph.
    let grids = [
      (t: 1, h: 4, w: 12), (t: 1, h: 12, w: 4), (t: 1, h: 6, w: 8), (t: 1, h: 4, w: 4),
      (t: 1, h: 8, w: 6), (t: 1, h: 4, w: 12),
    ]
    let graph = DynamicGraph()
    graph.withNoGrad {
      let model = ModelBuilder { (grid: (t: Int, h: Int, w: Int), _) in
        GLM5_3VisionTransformer(Float16.self, gridThw: [grid], configuration: configuration)
      }
      for (index, grid) in grids.enumerated() {
        let count = grid.h * grid.w * configuration.patchVectorSize
        let patches = (0..<count).map { Float16(Float(($0 * 7_919 + index) % 1_001) / 500 - 1) }
        let inputs: [DynamicGraph.AnyTensor] = [
          graph.variable(
            Tensor<Float16>(
              patches, .CPU, .WC(grid.h * grid.w, configuration.patchVectorSize)
            ).toGPU(0)),
          graph.variable(
            GLM5_3VisionRotaryEmbedding(
              gridThw: [grid], configuration: configuration, of: Float16.self
            ).toGPU(0)),
          graph.variable(GLM5_3VisionSequenceOffsets(gridThw: [grid]).offsets.toGPU(0)),
        ]
        // As in GLM5_3Generate, compile explicitly only for the first image.
        if index == 0 { model.compile(grid, inputs: inputs) }
        let actual = model(grid, inputs: inputs[0], Array(inputs.dropFirst()))[0]
          .as(of: Float16.self).toCPU()
        let reference = GLM5_3VisionTransformer(
          Float16.self, gridThw: [grid], configuration: configuration)
        reference.compile(inputs: inputs)
        reference.parameters.copy(from: model.parameters)
        let expected = reference(inputs: inputs[0], Array(inputs.dropFirst()))[0]
          .as(of: Float16.self).toCPU()
        graph.joined()
        let rows = GLM5_3VisionTokenCount(gridThw: grid, configuration: configuration)
        XCTAssertEqual(actual.shape, [rows, configuration.outputHiddenSize], "grid \(grid)")
        XCTAssertEqual(expected.shape, [rows, configuration.outputHiddenSize], "grid \(grid)")
        guard actual.shape == expected.shape else { continue }
        var maximumDifference: Float = 0
        for row in 0..<rows {
          for channel in 0..<configuration.outputHiddenSize {
            maximumDifference = max(
              maximumDifference, abs(Float(actual[row, channel]) - Float(expected[row, channel])))
          }
        }
        XCTAssertLessThan(maximumDifference, 1e-3, "grid \(grid)")
      }
    }
  }

  func testEmbeddingPlacesImageRowsAfterTokensAcrossChunks() {
    var configuration = GLM5_3ModelConfiguration.glm5_3Flash
    configuration.hiddenSize = 4
    configuration.vocabularySize = 8
    configuration.hcCount = 2
    let graph = DynamicGraph()
    graph.withNoGrad {
      let embedding = ModelBuilder<Int> { count, _ in
        GLM5_3Embedding(Float16.self, tokenLength: count, configuration: configuration)
      }
      let empty = graph.variable(.GPU(0), format: .NHWC, shape: [0], of: Float16.self)
      let headroom = graph.variable(
        Tensor<Int32>(Array(repeating: 0, count: 6), kind: .CPU, format: .NHWC, shape: [6])
          .toGPU(0))
      embedding.compile(6, inputs: headroom, empty)
      embedding.parameters[0].copy(
        from: Tensor<Float16>((0..<32).map { Float16($0) }, .CPU, .WC(8, 4)).toGPU(0))
      // Text only, image only, text followed by image rows (as GLM5_3Generate chunks), and
      // single rows of each.
      for lengths in [(6, 0), (0, 5), (2, 3), (0, 1), (1, 0), (4, 2)] {
        let count = lengths.0 + lengths.1
        let ids = (0..<lengths.0).map { Int32($0 + 1) }
        let tokens =
          lengths.0 == 0
          ? graph.variable(.GPU(0), format: .NHWC, shape: [0], of: Int32.self)
          : graph.variable(
            Tensor<Int32>(ids, kind: .CPU, format: .NHWC, shape: [lengths.0]).toGPU(0))
        let injected =
          lengths.1 == 0
          ? empty
          : graph.variable(
            Tensor<Float16>(
              (0..<(lengths.1 * 4)).map { Float16(100 + $0) }, .CPU, .WC(lengths.1, 4)
            ).toGPU(0))
        let output = embedding(count, inputs: tokens, injected)[0].as(of: Float.self).toCPU()
        graph.joined()
        XCTAssertEqual(output.shape, [count, 2, 4], "\(lengths)")
        guard output.shape == [count, 2, 4] else { continue }
        for row in 0..<count {
          for channel in 0..<4 {
            let expected =
              row < lengths.0
              ? Float(Int(ids[row]) * 4 + channel) : Float(100 + (row - lengths.0) * 4 + channel)
            for stream in 0..<2 {
              XCTAssertEqual(output[row, stream, channel], expected, "\(lengths) row \(row)")
            }
          }
        }
      }
    }
  }
}
