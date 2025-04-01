import NNC

public struct TeaCacheConfiguration {
  public var coefficients: (Float, Float, Float, Float, Float)
  public var steps: ClosedRange<Int>
  public var threshold: Float
  public init(
    coefficients: (Float, Float, Float, Float, Float), steps: ClosedRange<Int>, threshold: Float
  ) {
    self.coefficients = coefficients
    self.steps = steps
    self.threshold = threshold
  }
}

final class TeaCache<FloatType: TensorNumeric & BinaryFloatingPoint> {
  private let modelVersion: ModelVersion
  private let coefficients: (Float, Float, Float, Float, Float)
  private let threshold: Float
  private let steps: ClosedRange<Int>
  private let reducedModel: Model
  private var lastTs: [Int: [DynamicGraph.AnyTensor]]
  private var accumulatedRelL1Distances: [Int: Float]
  private var lastResiduals: [Int: DynamicGraph.AnyTensor]
  private var parameterShared: Bool

  public init(
    modelVersion: ModelVersion, coefficients: (Float, Float, Float, Float, Float), threshold: Float,
    steps: ClosedRange<Int>, reducedModel: Model
  ) {
    self.modelVersion = modelVersion
    self.coefficients = coefficients
    self.threshold = threshold
    self.steps = steps
    self.reducedModel = reducedModel
    lastTs = [Int: [DynamicGraph.AnyTensor]]()
    accumulatedRelL1Distances = [Int: Float]()
    lastResiduals = [Int: DynamicGraph.Tensor<FloatType>]()
    parameterShared = false
  }

  public func shouldUseCacheForTimeEmbedding(_ t: [DynamicGraph.AnyTensor], step: Int, marker: Int)
    -> Bool
  {
    guard let lastT = lastTs[marker], steps.contains(step) else {
      lastTs[marker] = t
      accumulatedRelL1Distances[marker] = 0
      return false
    }
    var totalR1: Float = 0
    var totalR2: Float = 0
    for (t, lastT) in zip(t, lastT) {
      let tf32 = t.as(of: Float.self)
      let lastTf32 = lastT.as(of: Float.self)
      let r1 = Functional.abs(tf32 - lastTf32).reduced(.mean, axis: [0, 1, 2]).rawValue.toCPU()
      let r2 = Functional.abs(lastTf32).reduced(.mean, axis: [0, 1, 2]).rawValue.toCPU()
      totalR1 += r1[0, 0, 0]
      totalR2 += r2[0, 0, 0]
    }
    let r = totalR1 / totalR2
    let dist =
      coefficients.0 * r * r * r * r + coefficients.1 * r * r * r + coefficients.2 * r * r
      + coefficients.3 * r + coefficients.4
    var accumulatedRelL1Distance = accumulatedRelL1Distances[marker] ?? 0
    accumulatedRelL1Distance += dist
    var shouldUseCache = true
    if accumulatedRelL1Distance >= threshold {
      accumulatedRelL1Distance = 0
      shouldUseCache = false
    }
    accumulatedRelL1Distances[marker] = accumulatedRelL1Distance
    lastTs[marker] = t
    return shouldUseCache
  }

  public func compile(model: ModelBuilderOrModel, inputs: [DynamicGraph.AnyTensor]) {
    switch modelVersion {
    case .v1, .v2, .kandinsky21, .sdxlBase, .sdxlRefiner, .ssd1b, .svdI2v, .wurstchenStageC,
      .wurstchenStageB, .sd3, .pixart, .auraflow, .flux1, .sd3Large, .hunyuanVideo:
      fatalError()
    case .wan21_1_3b, .wan21_14b:
      reducedModel.compile(inputs: [
        inputs[0], inputs[0], inputs[inputs.count - 2], inputs[inputs.count - 1],
      ])
    }
  }

  public func cache(outputs: [DynamicGraph.AnyTensor], marker: Int) {
    lastResiduals[marker] = outputs[outputs.count - 1]
  }

  public func callAsFunction(
    model: ModelBuilderOrModel, inputs firstInput: DynamicGraph.Tensor<FloatType>,
    _ restInputs: [DynamicGraph.AnyTensor], marker: Int
  ) -> DynamicGraph.Tensor<FloatType>? {
    guard let lastResidual = lastResiduals[marker] else {
      return nil
    }
    if !parameterShared {
      switch model {
      case .modelBuilder(_):
        fatalError()
      case .model(let model):
        reducedModel.parameters.share(from: model.parameters) { reducedModelName, _ in
          return .continue(reducedModelName)
        }
      }
      parameterShared = true
    }
    return reducedModel(
      inputs: firstInput,
      [lastResidual, restInputs[restInputs.count - 2], restInputs[restInputs.count - 1]])[0].as(
        of: FloatType.self)
  }
}
