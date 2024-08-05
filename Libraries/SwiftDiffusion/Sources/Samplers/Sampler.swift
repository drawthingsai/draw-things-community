import Foundation
import NNC

public enum ModelVersion: String, Codable {
  case v1 = "v1"
  case v2 = "v2"
  case kandinsky21 = "kandinsky2.1"
  case sdxlBase = "sdxl_base_v0.9"
  case sdxlRefiner = "sdxl_refiner_v0.9"
  case ssd1b = "ssd_1b"
  case svdI2v = "svd_i2v"
  case wurstchenStageC = "wurstchen_v3.0_stage_c"
  case wurstchenStageB = "wurstchen_v3.0_stage_b"
  case sd3 = "sd3"
  case pixart = "pixart"
  case auraflow = "auraflow"
  case flux1 = "flux1"
}

public enum TextEncoderVersion: String, Codable {
  case chatglm3_6b = "chatglm3_6b"
}

public enum ImageEncoderVersion: String, Codable {
  case clipL14_336 = "clip_l14_336"
  case openClipH14 = "open_clip_h14"
}

public enum AlternativeDecoderVersion: String, Codable {
  case transparent = "transparent"
}

public enum SamplerModifier: String, Codable {
  case none = "none"
  case inpainting = "inpainting"
  case depth = "depth"
  case editing = "editing"
}

public struct LoRAConfiguration: Equatable {
  public var file: String
  public var weight: Float
  public var version: ModelVersion
  public var isLoHa: Bool
  public var modifier: SamplerModifier
  public init(
    file: String, weight: Float, version: ModelVersion, isLoHa: Bool, modifier: SamplerModifier
  ) {
    self.file = file
    self.weight = weight
    self.version = version
    self.isLoHa = isLoHa
    self.modifier = modifier
  }
}

public struct SamplerOutput<FloatType: TensorNumeric & BinaryFloatingPoint, UNet: UNetProtocol> {
  public var x: DynamicGraph.Tensor<FloatType>
  public var unets: [UNet?]
  init(x: DynamicGraph.Tensor<FloatType>, unets: [UNet?]) {
    self.x = x
    self.unets = unets
  }
}

public func isNaN<T: TensorNumeric & BinaryFloatingPoint>(_ x: Tensor<T>) -> Bool {
  let shape = x.shape
  for b in 0..<shape[0] {
    for i in 0..<shape[1] {
      for j in 0..<shape[2] {
        for k in 0..<shape[3] {
          if x[b, i, j, k].isNaN {
            return true
          }
        }
      }
    }
  }
  return false
}

func clipDenoised<T: TensorNumeric & BinaryFloatingPoint>(_ x: DynamicGraph.Tensor<T>)
  -> DynamicGraph.Tensor<T>
{
  let x = x.clamped(-2...2)
  let shape = x.shape
  let y = x.rawValue.toCPU()
  var values = [Float]()
  for i in 0..<shape[1] {
    for j in 0..<shape[2] {
      for k in 0..<shape[3] {
        values.append(abs(Float(y[0, i, j, k])))
      }
    }
  }
  values.sort()
  let s = max(values[Int((Float(values.count - 1) * 0.995).rounded(.down))], 1)
  return (1.0 / s) * x.clamped(-s...s)
}

func Blur<T: TensorNumeric & BinaryFloatingPoint>(
  filters: Int, sigma: Float, size: Int, input: DynamicGraph.Tensor<T>
) -> Model {
  let model = Convolution(
    groups: filters, filters: filters, filterSize: [size, size], noBias: true,
    hint: Hint(
      stride: [1, 1], border: Hint.Border(begin: [size / 2, size / 2], end: [size / 2, size / 2])),
    format: .OIHW)
  model.compile(inputs: input)
  var weight = Tensor<T>(.CPU, .NCHW(filters, 1, size, size))
  var sum: Float = 0
  for i in 0..<size {
    let y = Float(i - size / 2)
    let y_gauss = expf(-y * y / (2 * sigma * sigma))
    for j in 0..<size {
      let x = Float(i - size / 2)
      let x_gauss = expf(-x * x / (2 * sigma * sigma))
      let val = x_gauss * y_gauss
      sum += val
      weight[0, 0, i, j] = T(val)
    }
  }
  for i in 0..<size {
    for j in 0..<size {
      let val = T(Float(weight[0, 0, i, j]) / sum)
      for k in 0..<filters {
        weight[k, 0, i, j] = val
      }
    }
  }
  model.weight.copy(from: weight)
  return model
}

func isCfgEnabled(textGuidanceScale: Float, startFrameCfg: Float, version: ModelVersion) -> Bool {
  guard version == .svdI2v else {
    return (textGuidanceScale - 1).magnitude > 1e-2
  }
  return (textGuidanceScale - 1).magnitude > 1e-2 || (startFrameCfg - 1).magnitude > 1e-2
}

public enum SamplerError: Error {
  case cancelled
  case isNaN
}

public struct Refiner: Equatable {
  public var start: Float
  public var filePath: String
  public var externalOnDemand: Bool
  public var version: ModelVersion
  public var is8BitModel: Bool
  public var isConsistencyModel: Bool
  // We probably need to copy all the rest in sampler over, but for now, we will just ignore.
  public init(
    start: Float, filePath: String, externalOnDemand: Bool, version: ModelVersion,
    is8BitModel: Bool, isConsistencyModel: Bool
  ) {
    self.start = start
    self.filePath = filePath
    self.externalOnDemand = externalOnDemand
    self.version = version
    self.is8BitModel = is8BitModel
    self.isConsistencyModel = isConsistencyModel
  }
}

public struct Sampling {
  public var steps: Int
  public var shift: Double
  public init(steps: Int, shift: Double = 1.0) {
    self.steps = steps
    self.shift = shift
  }
}

public protocol Sampler<FloatType, UNet> {
  associatedtype FloatType: TensorNumeric & BinaryFloatingPoint
  associatedtype UNet: UNetProtocol where UNet.FloatType == FloatType

  var filePath: String { get }
  var modifier: SamplerModifier { get }
  var version: ModelVersion { get }
  var upcastAttention: Bool { get }
  var usesFlashAttention: Bool { get }
  var externalOnDemand: Bool { get }
  var injectControls: Bool { get }
  var injectT2IAdapters: Bool { get }
  var injectIPAdapterLengths: [Int] { get }
  var lora: [LoRAConfiguration] { get }
  var tiledDiffusion: TiledConfiguration { get }
  var guidanceEmbed: Bool { get }

  func sample(
    _ x_T: DynamicGraph.Tensor<FloatType>, unets: [UNet?], sample: DynamicGraph.Tensor<FloatType>?,
    maskedImage: DynamicGraph.Tensor<FloatType>?, depthImage: DynamicGraph.Tensor<FloatType>?,
    mask: DynamicGraph.Tensor<FloatType>?, negMask: DynamicGraph.Tensor<FloatType>?,
    conditioning c: [DynamicGraph.Tensor<FloatType>], tokenLengthUncond: Int, tokenLengthCond: Int,
    extraProjection: DynamicGraph.Tensor<FloatType>?,
    injectedControls: [(
      model: ControlModel<FloatType>, hints: [([DynamicGraph.Tensor<FloatType>], Float)]
    )],
    textGuidanceScale: Float, imageGuidanceScale: Float,
    startStep: (integral: Int, fractional: Float), endStep: (integral: Int, fractional: Float),
    originalSize: (width: Int, height: Int), cropTopLeft: (top: Int, left: Int),
    targetSize: (width: Int, height: Int), aestheticScore: Float,
    negativeOriginalSize: (width: Int, height: Int), negativeAestheticScore: Float,
    zeroNegativePrompt: Bool, refiner: Refiner?, fpsId: Int, motionBucketId: Int, condAug: Float,
    startFrameCfg: Float, sharpness: Float, sampling: Sampling,
    feedback: (Int, Tensor<FloatType>?) -> Bool
  ) -> Result<SamplerOutput<FloatType, UNet>, Error>

  // For most, this is straightforward, just multiple the steps by the strength. But for Euler A
  // or DPM++ 2M Karras, it is not as obvious. Particularly for Karras' schedule, the best we
  // can do is to find the closest timestep and startStep, and alphaCumprod, and go from there.
  func timestep(for strength: Float, sampling: Sampling) -> (
    timestep: Float, startStep: Float, roundedDownStartStep: Int, roundedUpStartStep: Int
  )

  // Whether we need to scale the noise at a particular intermediate step.
  func noiseScaleFactor(at step: Float, sampling: Sampling) -> Float

  // Whether we need to scale the sample at a particular intermediate step.
  func sampleScaleFactor(at step: Float, sampling: Sampling) -> Float
}
