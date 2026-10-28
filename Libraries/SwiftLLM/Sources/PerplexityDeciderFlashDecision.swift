import Foundation
import NNC

/// Runs pplx-decider-flash on tokenized requests. The checkpoint is read on the first request and stays loaded.
/// Not thread-safe: callers serialize requests. Only `cancel()` may be called from another thread.
public final class PerplexityDeciderFlashDecision<FloatType: TensorNumeric & BinaryFloatingPoint> {
  /// The calibrated temperature saved with pplx-decider-flash-v1.2 (`decision_config.json`).
  public static var temperature: Double { 1.5539092221462472 }
  /// Answer codes the readout scores. A question with n options uses the first n.
  public static var optionCount: Int { 255 }

  public let filePath: String
  public let configuration: Qwen3_5ModelConfiguration
  private let graph: DynamicGraph
  private let model: ModelBuilder<Int>
  private var states: [DynamicGraph.AnyTensor]? = nil

  public init(
    filePath: String, configuration: Qwen3_5ModelConfiguration = .qwen3_5_4B,
    graph: DynamicGraph = DynamicGraph()
  ) {
    self.filePath = filePath
    self.configuration = configuration
    self.graph = graph
    model = ModelBuilder { tokenLength, _ in
      PerplexityDeciderFlash(
        FloatType.self, tokenLength: tokenLength, configuration: configuration,
        optionCount: Self.optionCount)
    }
    model.maxConcurrency = .limit(4)
  }

  /// Readout logits for one tokenized request: one row of `optionCount` logits per marker position, in order.
  public func logits(tokenIds: [Int32], markers: [Int]) throws -> [[Float]] {
    precondition(!tokenIds.isEmpty && !markers.isEmpty)
    precondition(markers.allSatisfy { $0 >= 0 && $0 < tokenIds.count })
    // Requests come in every length; GEMM kernels specialized on M would compile again for each one.
    let flags = DynamicGraph.flags
    defer {
      DynamicGraph.flags = flags
    }
    DynamicGraph.flags.insert(.disableMFAGEMMSpecializingM)
    return try graph.withNoGrad {
      let tokens = graph.variable(Tensor<Int32>(tokenIds, .CPU, .C(tokenIds.count)).toGPU(0))
      let markerTensor = graph.variable(
        Tensor<Int32>(markers.map { Int32($0) }, .CPU, .C(markers.count)).toGPU(0))
      let rotary = graph.variable(
        Qwen3_5RotaryEmbedding(
          sequenceLength: tokenIds.count, configuration: configuration, of: FloatType.self
        ).toGPU(0))
      if states == nil {
        // Linear-attention layers start every request from zero states. The model only reads them.
        var zeroStates = [DynamicGraph.AnyTensor]()
        for layerIndex in 0..<configuration.layers
        where configuration.isLinearAttentionLayer(layerIndex) {
          let convState = graph.variable(
            .GPU(0),
            .NHWC(1, configuration.linearConvKernel - 1, 1, configuration.linearConvDim),
            of: FloatType.self)
          convState.full(0)
          let recurrentState = graph.variable(
            .GPU(0),
            .NHWC(
              1, configuration.linearNumValueHeads, configuration.linearValueHeadDim,
              configuration.linearKeyHeadDim), of: Float.self)
          recurrentState.full(0)
          zeroStates.append(contentsOf: [convState, recurrentState])
        }
        try graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) { store in
          model.compile(
            tokenIds.count, inputs: [tokens, markerTensor, rotary] + zeroStates)
          try store.read(
            "decider_model", model: model, strict: true, codec: [.jit, .i8x, .ezm7, .externalData])
        }
        states = zeroStates
      }
      let output = model(tokenIds.count, inputs: tokens, [markerTensor, rotary] + states!)[0].as(
        of: FloatType.self
      ).toCPU()
      return (0..<markers.count).map { row in
        (0..<Self.optionCount).map { Float(output[row, $0]) }
      }
    }
  }

  /// Asks the request being evaluated to stop, from another thread. A stopped request returns garbage logits, so a
  /// caller that cancels discards them. It does not always stop, and a request whose evaluation has not started yet
  /// runs as usual.
  public func cancel() {
    model.cancel()
  }

  /// Option probabilities of a question with `count` options, from its readout logits.
  public static func probabilities(
    logits: [Float], count: Int, temperature: Double = temperature
  ) -> [Double] {
    precondition(count > 0 && count <= logits.count)
    let scaled = logits[0..<count].map { Double($0) / temperature }
    let maximum = scaled.max()!
    let exponentials = scaled.map { exp($0 - maximum) }
    let total = exponentials.reduce(0, +)
    return exponentials.map { $0 / total }
  }
}
