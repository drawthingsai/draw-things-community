import Foundation
import LLM
import NNC
import XCTest

final class Qwen3_5TextGenerationTests: XCTestCase {
  private let configuration = Qwen3_5ModelConfiguration(
    vocabularySize: 32, hiddenSize: 128, intermediateSize: 256, layers: 1,
    fullAttentionInterval: 1, attentionHeads: 2, keyValueHeads: 1,
    attentionHeadDim: 64, rotaryDim: 64, mropeSection: (11, 11, 10),
    ropeTheta: 10_000, linearNumKeyHeads: 1, linearNumValueHeads: 1,
    linearKeyHeadDim: 64, linearValueHeadDim: 64, linearConvKernel: 4)
  private var path: String!

  override func setUpWithError() throws {
    path =
      FileManager.default.temporaryDirectory.appendingPathComponent(
        "qwen-generation-\(UUID().uuidString).ckpt"
      ).path
    // A tiny randomly initialized checkpoint exercises the real Metal execution
    // and compilation paths without downloading model weights.
    let graph = DynamicGraph()
    try graph.withNoGrad {
      let tokens = graph.variable(Tensor<Int32>([1], .CPU, .C(1)).toGPU(0))
      let rotary = graph.variable(
        Qwen3_5RotaryEmbedding(
          sequenceLength: 1, configuration: configuration, of: Float16.self
        ).toGPU(0))
      let k = graph.variable(.GPU(0), .NHWC(1, 1, 1, 64), of: Float16.self)
      let v = graph.variable(like: k)
      k.full(0)
      v.full(0)
      let model = Qwen3_5CausalLM(
        Float16.self, tokenLength: 1, configuration: configuration)
      _ = model(inputs: tokens, rotary, k, v)[0].as(of: Float16.self).toCPU()
      graph.openStore(path) { $0.write("text_model", model: model) }

      // Metal's random initializer supports FP16. Convert the saved drafter
      // weights on CPU to match the runtime's BF16 checkpoint contract.
      let mtp = Qwen3_5MTP(
        Float16.self, Float16.self, configuration: configuration, batchSize: 1,
        tokenLength: 1, cachedTokenLength: 0, lastNumberOfTokens: 1, tieEmbedding: false)
      let hidden = graph.variable(.GPU(0), .NC(1, 128), of: Float16.self)
      let mtpK = graph.variable(.GPU(0), .NHWC(1, 1, 1, 64), of: Float16.self)
      let mtpV = graph.variable(like: mtpK)
      hidden.full(0)
      mtpK.full(0)
      mtpV.full(0)
      _ = mtp(inputs: tokens, hidden, rotary, mtpK, mtpV)[1].as(of: Float16.self).toCPU()
      graph.openStore(path) { $0.write("text_model", model: mtp) }
      try graph.openStore(path) { store in
        let keys = store.keys.filter { $0.contains("mtp.") }
        XCTAssertFalse(keys.isEmpty)
        for key in keys {
          let tensor = try XCTUnwrap(store.read(key, kind: .CPU))
          try store.write(key, tensor: Tensor<BFloat16>(from: tensor), strict: true)
        }
      }
    }
  }

  override func tearDownWithError() throws {
    try FileManager.default.removeItem(atPath: path)
  }

  private var generator: Qwen3_5TextGeneration<Float16> {
    Qwen3_5TextGeneration(
      filePath: path, configuration: configuration, eosTokenIds: [], tieEmbedding: false)
  }

  func testSingleTokenPrefillAndDecode() throws {
    for prompt: [Int32] in [[1], [1, 2]] {
      var timing: Qwen3_5GenerationTiming?
      let tokens = try generator.generate(
        graph: DynamicGraph(), promptTokenIds: prompt, maxTokens: 4,
        partialHandler: { _ in true }, timingHandler: { timing = $0 })
      XCTAssertEqual(tokens.count, 4)
      XCTAssertTrue(tokens.allSatisfy { (0..<32).contains($0) })
      XCTAssertEqual(timing?.decodeLoopTokens, 3)
    }
  }

  func testGenerationStopsAtPartialHandlerBoundary() throws {
    let tokens = try generator.generate(
      graph: DynamicGraph(), promptTokenIds: [1], maxTokens: 8,
      partialHandler: { $0.count < 2 })
    XCTAssertEqual(tokens.count, 2)
  }

  func testMTPGenerationAndDraftAcceptance() throws {
    let result = try generator.generateWithMTPDrafting(
      graph: DynamicGraph(), promptTokenIds: [1], maxTokens: 4)
    XCTAssertEqual(result.generatedTokenIds.count, 4)
    XCTAssertTrue(result.generatedTokenIds.allSatisfy { (0..<32).contains($0) })
    let acceptance = try generator.measureMTPDraftAcceptance(
      graph: DynamicGraph(), promptTokenIds: [1], startCount: 2, maxDraftTokens: 2)
    XCTAssertEqual(acceptance.startCount, 2)
    XCTAssertEqual(acceptance.acceptedLengthCounts.reduce(0, +), 2)
    XCTAssertTrue(acceptance.acceptedPrefixRates.allSatisfy { (0...1).contains($0) })
  }
}
