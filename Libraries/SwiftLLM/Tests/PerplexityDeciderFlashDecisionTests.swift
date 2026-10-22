import Foundation
import LLM
import NNC
import XCTest

final class PerplexityDeciderFlashDecisionTests: XCTestCase {
  // One linear-attention layer, then one full-attention layer.
  private let configuration = Qwen3_5ModelConfiguration(
    vocabularySize: 32, hiddenSize: 128, intermediateSize: 256, layers: 2,
    fullAttentionInterval: 2, attentionHeads: 2, keyValueHeads: 1,
    attentionHeadDim: 64, rotaryDim: 64, mropeSection: (11, 11, 10),
    ropeTheta: 10_000, linearNumKeyHeads: 1, linearNumValueHeads: 1,
    linearKeyHeadDim: 64, linearValueHeadDim: 64, linearConvKernel: 4)
  private var path: String!

  override func setUpWithError() throws {
    path =
      FileManager.default.temporaryDirectory.appendingPathComponent(
        "pplx-decider-flash-\(UUID().uuidString).ckpt"
      ).path
    // A tiny randomly initialized checkpoint exercises the real Metal execution
    // and compilation paths without downloading model weights.
    let graph = DynamicGraph()
    graph.withNoGrad {
      let tokens = graph.variable(Tensor<Int32>([1], .CPU, .C(1)).toGPU(0))
      let markers = graph.variable(Tensor<Int32>([0], .CPU, .C(1)).toGPU(0))
      let rotary = graph.variable(
        Qwen3_5RotaryEmbedding(
          sequenceLength: 1, configuration: configuration, of: Float16.self
        ).toGPU(0))
      let convState = graph.variable(
        .GPU(0), .NHWC(1, configuration.linearConvKernel - 1, 1, configuration.linearConvDim),
        of: Float16.self)
      let recurrentState = graph.variable(.GPU(0), .NHWC(1, 1, 64, 64), of: Float.self)
      convState.full(0)
      recurrentState.full(0)
      let model = PerplexityDeciderFlash(Float16.self, tokenLength: 1, configuration: configuration)
      _ = model(inputs: tokens, markers, rotary, convState, recurrentState)[0].as(of: Float16.self)
        .toCPU()
      graph.openStore(path) { $0.write("decider_model", model: model) }
    }
  }

  override func tearDownWithError() throws {
    try FileManager.default.removeItem(atPath: path)
  }

  func testRepeatedRequestsAreIdentical() throws {
    let decision = PerplexityDeciderFlashDecision<Float16>(
      filePath: path, configuration: configuration)
    let first = try decision.logits(tokenIds: [1, 2, 3, 4, 5], markers: [2, 4])
    _ = try decision.logits(tokenIds: [6, 7, 8], markers: [1])
    let second = try decision.logits(tokenIds: [1, 2, 3, 4, 5], markers: [2, 4])
    XCTAssertEqual(first.count, 2)
    XCTAssertEqual(first[0].count, PerplexityDeciderFlashDecision<Float16>.optionCount)
    XCTAssertTrue(first.joined().allSatisfy { $0.isFinite })
    XCTAssertEqual(first, second)
  }

  func testMarkersReadLaterTokens() throws {
    let decision = PerplexityDeciderFlashDecision<Float16>(
      filePath: path, configuration: configuration)
    let original = try decision.logits(tokenIds: [1, 2, 3, 4, 5], markers: [1])
    let changedAfterMarker = try decision.logits(tokenIds: [1, 2, 3, 4, 9], markers: [1])
    XCTAssertNotEqual(original, changedAfterMarker)
  }
}
