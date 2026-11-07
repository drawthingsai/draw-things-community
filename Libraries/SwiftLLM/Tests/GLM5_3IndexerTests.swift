import LLM
import NNC
import XCTest

final class GLM5_3IndexerTests: XCTestCase {
  func testDecodeWithoutCompletingAPool() {
    var config = GLM5_3ModelConfiguration.glm5_3Flash
    config.hiddenSize = 8
    config.queryLowRank = 8
    config.indexerHeads = 2
    config.indexerHeadDim = 4
    config.indexerTopK = 8
    let graph = DynamicGraph()
    graph.withNoGrad {
      let model = ModelBuilder { (cached: Int, _) in
        let x = Input()
        let query = Input()
        let indexer = Input()
        let pooled = Input()
        return Model(
          [x, query, indexer, pooled],
          [
            GLM5_3Indexer(
              Float16.self, prefix: "indexer", x: x, queryResidual: query, indexerCache: indexer,
              pooledCache: pooled, tokenLength: 1, cachedTokenLength: cached, configuration: config)
          ])
      }
      let x = graph.variable(.GPU(0), .WC(1, 8), of: Float16.self)
      x.full(1)
      let indexer = graph.variable(.GPU(0), .NHWC(1, 8, 1, 8), of: Float16.self)
      indexer.full(0)
      let pooled = graph.variable(.GPU(0), .WC(2, 4), of: Float16.self)
      pooled.full(0)
      // Compile with a complete pool so every parameter exists, then cover each remainder.
      for cached in [3, 0, 1, 2, 3, 4, 5, 6, 7] {
        let output = model(
          cached, inputs: x, x,
          indexer.reshaped(.NHWC(1, cached + 1, 1, 8)), pooled)[0].as(of: Int32.self).toCPU()
          .rawValue
        XCTAssertEqual(output.shape[0], 1)
        XCTAssertEqual(output.shape[1], max(1, (cached + 1) / 4))
        if cached < 3 { XCTAssertEqual(output[0, 0], -1) }
      }
    }
  }
}
