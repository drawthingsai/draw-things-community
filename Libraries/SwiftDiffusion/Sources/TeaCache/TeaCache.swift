import NNC

public struct TeaCacheConfiguration {
  public var coefficients: (Float, Float, Float, Float, Float)
  public var steps: ClosedRange<Int>
  public var threshold: Float
  public var maxSkipSteps: Int
  public init(
    coefficients: (Float, Float, Float, Float, Float), steps: ClosedRange<Int>, threshold: Float,
    maxSkipSteps: Int
  ) {
    self.coefficients = coefficients
    self.steps = steps
    self.threshold = threshold
    self.maxSkipSteps = maxSkipSteps
  }
}

final class TeaCache<FloatType: TensorNumeric & BinaryFloatingPoint> {
  private let modelVersion: ModelVersion
  private let coefficients: (Float, Float, Float, Float, Float)
  private let threshold: Float
  private let steps: ClosedRange<Int>
  private let maxSkipSteps: Int
  private let reducedModel: Model
  private let inferModel: Model?
  private let referenceImageCount: Int
  private var lastTs: [Int: [DynamicGraph.AnyTensor]]
  private var accumulatedRelL1Distances: [Int: Float]
  private var lastResiduals: [Int: DynamicGraph.AnyTensor]
  private var skipSteps: [Int: Int]
  private var reducedModelParameterShared: Bool
  private var inferModelParameterShared: Bool

  public init(
    modelVersion: ModelVersion, coefficients: (Float, Float, Float, Float, Float), threshold: Float,
    steps: ClosedRange<Int>, maxSkipSteps: Int, reducedModel: Model, inferModel: Model? = nil,
    referenceImageCount: Int = 0
  ) {
    self.modelVersion = modelVersion
    self.coefficients = coefficients
    self.threshold = threshold
    self.steps = steps
    self.maxSkipSteps = max(1, maxSkipSteps)
    self.reducedModel = reducedModel
    self.inferModel = inferModel
    self.referenceImageCount = referenceImageCount
    lastTs = [Int: [DynamicGraph.AnyTensor]]()
    accumulatedRelL1Distances = [Int: Float]()
    lastResiduals = [Int: DynamicGraph.Tensor<FloatType>]()
    reducedModelParameterShared = false
    inferModelParameterShared = false
    skipSteps = [Int: Int]()
  }

  public func shouldUseCacheForTimeEmbedding<T: TensorNumeric & BinaryFloatingPoint>(
    _ t: [DynamicGraph.AnyTensor], model: ModelBuilderOrModel, step: Int, marker: Int,
    of: T.Type = T.self
  )
    -> Bool
  {
    var t = t
    if let inferModel = inferModel {
      if !inferModelParameterShared {
        switch model {
        case .modelBuilder(let model):
          inferModel.parameters.share(from: model.parameters) { inferModelName, _ in
            return .continue(inferModelName)
          }
        case .model(let model):
          inferModel.parameters.share(from: model.parameters) { inferModelName, _ in
            return .continue(inferModelName)
          }
        }
        inferModelParameterShared = true
      }
      switch modelVersion {
      case .v1, .v2, .kandinsky21, .sdxlBase, .sdxlRefiner, .ssd1b, .svdI2v, .wurstchenStageC,
        .wurstchenStageB, .sd3, .pixart, .auraflow, .sd3Large, .wan21_14b, .wan21_1_3b, .hiDreamI1:
        fatalError()
      case .hunyuanVideo:
        t = [inferModel(inputs: t[0], Array(t[(4 + 6)..<(6 + 6)]))[0]]  // context chunks is before x chunks. We need x chunks.
      case .flux1:
        if referenceImageCount > 0 {
          t = [inferModel(inputs: t[0], [t[2]] + Array(t[(4 + 6)..<(6 + 6)]))[0]]  // context chunks is before x chunks. We need x chunks.
        } else {
          t = [inferModel(inputs: t[0], Array(t[(3 + 6)..<(5 + 6)]))[0]]  // context chunks is before x chunks. We need x chunks.
        }
      }
    }
    guard let lastT = lastTs[marker], steps.contains(step),
      skipSteps[marker, default: 0] < maxSkipSteps
    else {
      lastTs[marker] = t
      accumulatedRelL1Distances[marker] = 0
      skipSteps[marker] = 0
      return false
    }
    var totalR1: Float = 0
    var totalR2: Float = 0
    for (t, lastT) in zip(t, lastT) {
      let tf32 = t.as(of: T.self)
      let lastTf32 = lastT.as(of: T.self)
      let shape = tf32.shape
      let r1 = Functional.abs(tf32 - lastTf32).reduced(.mean, axis: Array(0..<shape.count)).rawValue
        .toCPU()
      let r2 = Functional.abs(lastTf32).reduced(.mean, axis: Array(0..<shape.count)).rawValue
        .toCPU()
      totalR1 += Float(r1.reshaped(.C(1))[0])
      totalR2 += Float(r2.reshaped(.C(1))[0])
    }
    let r = totalR1 / totalR2
    let dist =
      coefficients.0 * r * r * r * r + coefficients.1 * r * r * r + coefficients.2 * r * r
      + coefficients.3 * r + coefficients.4
    var accumulatedRelL1Distance = accumulatedRelL1Distances[marker] ?? 0
    print(dist)
    accumulatedRelL1Distance += dist
    var shouldUseCache = true
    if accumulatedRelL1Distance >= threshold {
      accumulatedRelL1Distance = 0
      shouldUseCache = false
      skipSteps[marker] = 0  // Reset skip steps in this case.
    }
    accumulatedRelL1Distances[marker] = accumulatedRelL1Distance
    lastTs[marker] = t
    return shouldUseCache
  }

  public func compile(model: ModelBuilderOrModel, inputs: [DynamicGraph.AnyTensor]) {
    switch modelVersion {
    case .v1, .v2, .kandinsky21, .sdxlBase, .sdxlRefiner, .ssd1b, .svdI2v, .wurstchenStageC,
      .wurstchenStageB, .sd3, .pixart, .auraflow, .sd3Large:
      fatalError()
    case .hunyuanVideo:
      if let inferModel = inferModel {
        inferModel.compile(inputs: [inputs[0]] + Array(inputs[(4 + 6)..<(6 + 6)]))
      }
      reducedModel.compile(inputs: [
        inputs[0], inputs[0], inputs[inputs.count - 2], inputs[inputs.count - 1],
      ])
    case .flux1:
      if let inferModel = inferModel {
        if referenceImageCount > 0 {
          inferModel.compile(inputs: [inputs[0], inputs[2]] + Array(inputs[(4 + 6)..<(6 + 6)]))
        } else {
          inferModel.compile(inputs: [inputs[0]] + Array(inputs[(3 + 6)..<(5 + 6)]))
        }
      }
      let shift: DynamicGraph.AnyTensor = inputs[
        (referenceImageCount > 0 ? 4 : 3) + 19 * 12 + 38 * 3]
      let scale: DynamicGraph.AnyTensor = inputs[
        (referenceImageCount > 0 ? 4 : 3) + 19 * 12 + 38 * 3 + 1]
      reducedModel.compile(inputs: [
        inputs[0], inputs[0], shift, scale,
      ])
    case .hiDreamI1:
      reducedModel.compile(inputs: [
        inputs[0], inputs[0], inputs[inputs.count - 2], inputs[inputs.count - 1],
      ])
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
    if !reducedModelParameterShared {
      switch model {
      case .modelBuilder(let model):
        reducedModel.parameters.share(from: model.parameters) { reducedModelName, _ in
          return .continue(reducedModelName)
        }
      case .model(let model):
        reducedModel.parameters.share(from: model.parameters) { reducedModelName, _ in
          return .continue(reducedModelName)
        }
      }
      reducedModelParameterShared = true
    }
    let shift: DynamicGraph.AnyTensor
    let scale: DynamicGraph.AnyTensor
    switch modelVersion {
    case .v1, .v2, .kandinsky21, .sdxlBase, .sdxlRefiner, .ssd1b, .svdI2v, .wurstchenStageC,
      .wurstchenStageB, .sd3, .pixart, .auraflow, .sd3Large, .hunyuanVideo, .wan21_1_3b, .wan21_14b,
      .hiDreamI1:
      shift = restInputs[restInputs.count - 2]
      scale = restInputs[restInputs.count - 1]
    case .flux1:
      shift = restInputs[(referenceImageCount > 0 ? 3 : 2) + 19 * 12 + 38 * 3]
      scale = restInputs[(referenceImageCount > 0 ? 3 : 2) + 19 * 12 + 38 * 3 + 1]
    }
    // Running the reduced model, this is skipped.
    skipSteps[marker, default: 0] += 1
    return reducedModel(
      inputs: firstInput, lastResidual, shift, scale)[0].as(
        of: FloatType.self)
  }
}
