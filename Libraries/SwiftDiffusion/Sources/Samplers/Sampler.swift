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
  case sd3Large = "sd3_large"
  case hunyuanVideo = "hunyuan_video"
  case wan21_1_3b = "wan_v2.1_1.3b"
  case wan21_14b = "wan_v2.1_14b"
  case hiDreamI1 = "hidream_i1"
  case qwenImage = "qwen_image"
}

public enum TextEncoderVersion: String, Codable {
  case chatglm3_6b = "chatglm3_6b"
}

public enum ImageEncoderVersion: String, Codable {
  case clipL14_336 = "clip_l14_336"
  case openClipH14 = "open_clip_h14"
  case eva02L14_336 = "eva02_l14_336"
  case siglipL27_384 = "siglip_l27_384"
  case siglip2L27_512 = "siglip2_l27_512"
}

public enum AlternativeDecoderVersion: String, Codable {
  case transparent = "transparent"
}

public enum SamplerModifier: String, Codable {
  case none = "none"
  case inpainting = "inpainting"
  case depth = "depth"
  case editing = "editing"
  case double = "double"
  case canny = "canny"
  case kontext = "kontext"
}

public struct LoRAConfiguration: Equatable {
  public enum Mode: String, Codable {
    case all
    case base
    case refiner
  }
  public var file: String
  public var weight: Float
  public var version: ModelVersion
  public var isLoHa: Bool
  public var modifier: SamplerModifier
  public var mode: Mode
  public init(
    file: String, weight: Float, version: ModelVersion, isLoHa: Bool, modifier: SamplerModifier,
    mode: Mode
  ) {
    self.file = file
    self.weight = weight
    self.version = version
    self.isLoHa = isLoHa
    self.modifier = modifier
    self.mode = mode
  }
}

public struct DeviceProperties {
  public var isFreadPreferred: Bool
  public var memoryCapacity: MemoryCapacity
  public var isNHWCPreferred: Bool
  public init(isFreadPreferred: Bool, memoryCapacity: MemoryCapacity, isNHWCPreferred: Bool) {
    self.isFreadPreferred = isFreadPreferred
    self.memoryCapacity = memoryCapacity
    self.isNHWCPreferred = isNHWCPreferred
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

@inlinable
public func isNaN<T: TensorNumeric & BinaryFloatingPoint>(_ x: Tensor<T>) -> Bool {
  return x.isNaN
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

public enum SamplerError: Error {
  case cancelled([(any UNetProtocol)?])
  case isNaN
}

public struct Refiner: Equatable {
  public var start: Float
  public var filePath: String
  public var externalOnDemand: Bool
  public var version: ModelVersion
  public var isQuantizedModel: Bool
  public var isConsistencyModel: Bool
  public var qkNorm: Bool
  public var dualAttentionLayers: [Int]
  public var distilledGuidanceLayers: Int
  public var upcastAttention: Bool
  public var builtinLora: Bool
  public var isBF16: Bool
  // We probably need to copy all the rest in sampler over, but for now, we will just ignore.
  public init(
    start: Float, filePath: String, externalOnDemand: Bool, version: ModelVersion,
    isQuantizedModel: Bool, isConsistencyModel: Bool, qkNorm: Bool, dualAttentionLayers: [Int],
    distilledGuidanceLayers: Int, upcastAttention: Bool, builtinLora: Bool, isBF16: Bool
  ) {
    self.start = start
    self.filePath = filePath
    self.externalOnDemand = externalOnDemand
    self.version = version
    self.isQuantizedModel = isQuantizedModel
    self.isConsistencyModel = isConsistencyModel
    self.qkNorm = qkNorm
    self.dualAttentionLayers = dualAttentionLayers
    self.distilledGuidanceLayers = distilledGuidanceLayers
    self.upcastAttention = upcastAttention
    self.builtinLora = builtinLora
    self.isBF16 = isBF16
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
  var injectAttentionKV: Bool { get }
  var lora: [LoRAConfiguration] { get }
  var tiledDiffusion: TiledConfiguration { get }
  var isGuidanceEmbedEnabled: Bool { get }

  func sample(
    _ x_T: DynamicGraph.Tensor<FloatType>, unets: [UNet?], sample: DynamicGraph.Tensor<FloatType>?,
    conditionImage: DynamicGraph.Tensor<FloatType>?,
    referenceImages: [DynamicGraph.Tensor<FloatType>],
    mask: DynamicGraph.Tensor<FloatType>?, negMask: DynamicGraph.Tensor<FloatType>?,
    conditioning c: [DynamicGraph.Tensor<FloatType>], tokenLengthUncond: Int, tokenLengthCond: Int,
    extraProjection: DynamicGraph.Tensor<FloatType>?,
    injectedControls: [(
      model: ControlModel<FloatType>, hints: [([DynamicGraph.Tensor<FloatType>], Float)]
    )],
    textGuidanceScale: Float, imageGuidanceScale: Float, guidanceEmbed: Float,
    startStep: (integral: Int, fractional: Float), endStep: (integral: Int, fractional: Float),
    originalSize: (width: Int, height: Int), cropTopLeft: (top: Int, left: Int),
    targetSize: (width: Int, height: Int), aestheticScore: Float,
    negativeOriginalSize: (width: Int, height: Int), negativeAestheticScore: Float,
    zeroNegativePrompt: Bool, refiner: Refiner?, fpsId: Int, motionBucketId: Int, condAug: Float,
    startFrameCfg: Float, sharpness: Float, sampling: Sampling,
    cancellation: (@escaping () -> Void) -> Void, feedback: (Int, Tensor<FloatType>?) -> Bool
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

// Cfg related shared functions.

public func isCfgEnabled(
  textGuidanceScale: Float, imageGuidanceScale: Float, startFrameCfg: Float, version: ModelVersion,
  modifier: SamplerModifier
)
  -> Bool
{
  guard version != .svdI2v else {
    return (textGuidanceScale - 1).magnitude > 1e-3 || (startFrameCfg - 1).magnitude > 1e-3
  }
  guard modifier != .editing else {
    // If both are 1, no cfg.
    if (textGuidanceScale - 1).magnitude <= 1e-3 && (imageGuidanceScale - 1).magnitude <= 1e-3 {
      return false
    }
    return true
  }
  return (textGuidanceScale - 1).magnitude > 1e-3
}

public func cfgChannelsAndInputChannels(
  channels: Int, conditionShape: TensorShape?, isCfgEnabled: Bool, textGuidanceScale: Float,
  imageGuidanceScale: Float, version: ModelVersion, modifier: SamplerModifier
) -> (Int, Int) {
  let cfgChannels: Int
  let inChannels: Int
  if version == .svdI2v {
    cfgChannels = 1
    inChannels = channels * 2
  } else {
    switch modifier {
    case .inpainting, .depth, .canny:
      cfgChannels = isCfgEnabled ? 2 : 1
      inChannels = channels + (conditionShape?[3] ?? 0)
    case .editing:
      if isCfgEnabled {
        // If they are the same, it degrades to guidance * (etCond - etAllUncond) + etAllUncond
        if (textGuidanceScale - imageGuidanceScale).magnitude <= 1e-3 {
          cfgChannels = 2
        } else {
          cfgChannels = 3
        }
      } else {
        cfgChannels = 1
      }
      inChannels = channels * 2
    case .double:
      cfgChannels = isCfgEnabled ? 2 : 1
      inChannels = channels * 2
    case .none, .kontext:
      cfgChannels = isCfgEnabled ? 2 : 1
      inChannels = channels
    }
  }
  return (cfgChannels, inChannels)
}

func updateCfgInputAndConditions<FloatType: TensorNumeric & BinaryFloatingPoint>(
  xIn: inout DynamicGraph.Tensor<FloatType>, conditions c: inout [DynamicGraph.Tensor<FloatType>],
  conditionImage: DynamicGraph.Tensor<FloatType>?, batchSize: Int, startHeight: Int,
  startWidth: Int, channels: Int, isCfgEnabled: Bool, textGuidanceScale: Float,
  modifier: SamplerModifier
) {
  let graph = xIn.graph
  switch modifier {
  case .inpainting, .depth, .canny:
    if let conditionImage = conditionImage {
      let shape = conditionImage.shape
      for i in stride(from: 0, to: batchSize, by: shape[0]) {
        xIn[
          i..<(i + shape[0]), 0..<startHeight, 0..<startWidth, channels..<(channels + shape[3])] =
          conditionImage
        if isCfgEnabled {
          xIn[
            (batchSize + i)..<(batchSize + i + shape[0]), 0..<startHeight, 0..<startWidth,
            channels..<(channels + shape[3])] = conditionImage
        }
      }
    }
  case .editing:
    let cfgChannels = xIn.shape[0] / batchSize
    let maskedImage = conditionImage!
    for i in 0..<batchSize {
      if isCfgEnabled {
        xIn[
          (batchSize + i)..<(batchSize + i + 1), 0..<startHeight, 0..<startWidth,
          channels..<(channels * 2)] = maskedImage
        if cfgChannels == 2 {
          // In place of etUncond, now it is etAllUncond.
          xIn[i..<(i + 1), 0..<startHeight, 0..<startWidth, channels..<(channels * 2)].full(0)
        } else {
          xIn[i..<(i + 1), 0..<startHeight, 0..<startWidth, channels..<(channels * 2)] = maskedImage
          xIn[
            (batchSize * 2 + i)..<(batchSize * 2 + i + 1), 0..<startHeight, 0..<startWidth,
            channels..<(channels * 2)
          ].full(0)
        }
      } else {
        xIn[i..<(i + 1), 0..<startHeight, 0..<startWidth, channels..<(channels * 2)] = maskedImage
      }
    }
    if isCfgEnabled {
      if cfgChannels == 2 {
        // Do nothing.
      } else {
        c = c.map {
          let oldC = $0
          let shape = oldC.shape
          if shape.count == 2 {
            var c = graph.variable(
              .GPU(0), .WC(3 * batchSize, oldC.shape[1]), of: FloatType.self)
            // Expanding c.
            c[0..<(batchSize * 2), 0..<oldC.shape[1]] = oldC
            c[(batchSize * 2)..<(batchSize * 3), 0..<oldC.shape[1]] =
              oldC[0..<batchSize, 0..<oldC.shape[1]]
            return c
          } else {
            var c = graph.variable(
              .GPU(0), .HWC(3 * batchSize, oldC.shape[1], oldC.shape[2]), of: FloatType.self)
            // Expanding c.
            c[0..<(batchSize * 2), 0..<oldC.shape[1], 0..<oldC.shape[2]] = oldC
            c[(batchSize * 2)..<(batchSize * 3), 0..<oldC.shape[1], 0..<oldC.shape[2]] =
              oldC[0..<batchSize, 0..<oldC.shape[1], 0..<oldC.shape[2]]
            return c
          }
        }
      }
    }
  case .double:
    let maskedImage = conditionImage!
    let maskedImageChannels = maskedImage.shape[3]
    for i in 0..<batchSize {
      xIn[
        i..<(i + 1), 0..<startHeight, 0..<startWidth, channels..<(channels + maskedImageChannels)] =
        maskedImage
      if isCfgEnabled {
        xIn[
          (batchSize + i)..<(batchSize + i + 1), 0..<startHeight, 0..<startWidth,
          channels..<(channels + maskedImageChannels)] =
          maskedImage
      }
    }
  case .none, .kontext:
    break
  }
}

public struct CfgZeroStarConfiguration {
  public var isEnabled: Bool
  public var zeroInitSteps: Int
  public init(isEnabled: Bool, zeroInitSteps: Int) {
    self.isEnabled = isEnabled
    self.zeroInitSteps = zeroInitSteps
  }
}

func isBatchEnabled(_ version: ModelVersion) -> Bool {
  switch version {
  case .auraflow, .flux1, .hiDreamI1, .kandinsky21, .pixart, .sd3, .sd3Large, .sdxlBase,
    .sdxlRefiner, .ssd1b, .v1, .v2, .wurstchenStageB, .wurstchenStageC, .qwenImage:
    return true
  case .hunyuanVideo, .svdI2v, .wan21_14b, .wan21_1_3b:
    return false
  }
}

func applyCfg<FloatType: TensorNumeric & BinaryFloatingPoint>(
  etOut: DynamicGraph.Tensor<FloatType>, blur: Model?, batchSize: Int, startHeight: Int,
  startWidth: Int, channels: Int, isCfgEnabled: Bool, textGuidanceScale: Float,
  imageGuidanceScale: Float, alpha: Float, modifier: SamplerModifier, step: Int,
  isBatchEnabled: Bool, cfgZeroStar: CfgZeroStarConfiguration
) -> DynamicGraph.Tensor<FloatType> {
  let et: DynamicGraph.Tensor<FloatType>
  if isCfgEnabled {
    var etUncond = etOut[0..<batchSize, 0..<startHeight, 0..<startWidth, 0..<channels].copied()
    var etCond = etOut[batchSize..<(batchSize * 2), 0..<startHeight, 0..<startWidth, 0..<channels]
      .copied()
    if let blur = blur {
      let etCondDegraded = blur(inputs: etCond)[0].as(of: FloatType.self)
      etCond = Functional.add(
        left: etCondDegraded, right: etCond, leftScalar: alpha, rightScalar: 1 - alpha)
    }
    if modifier == .editing {
      let cfgChannels = etOut.shape[0] / batchSize
      if cfgChannels == 2 {
        if cfgZeroStar.isEnabled {
          let dotProduct: DynamicGraph.Tensor<Float>
          let squaredSum: DynamicGraph.Tensor<Float>
          let etUncondF32 = DynamicGraph.Tensor<Float>(from: etUncond)
          let etCondF32 = DynamicGraph.Tensor<Float>(from: etCond)
          if isBatchEnabled {
            dotProduct = (etUncondF32 .* etCondF32).reduced(
              .sum, axis: [0, 1, 2, 3])
            squaredSum =
              (etUncondF32 .* etUncondF32).reduced(
                .sum, axis: [0, 1, 2, 3]) + 1e-8
          } else {
            dotProduct = (etUncondF32 .* etCondF32).reduced(
              .sum, axis: [1, 2, 3])
            squaredSum =
              (etUncondF32 .* etUncondF32).reduced(.sum, axis: [1, 2, 3])
              + 1e-8
          }
          let stStar = dotProduct ./ squaredSum
          etUncond = DynamicGraph.Tensor<FloatType>(from: etUncondF32 .* stStar)
        }
        // This handles the case text guidance scale == 0 and text guidance == image guidance.
        et = etUncond + imageGuidanceScale * (etCond - etUncond)
      } else {
        let etAllUncond =
          etOut[
            (batchSize * 2)..<(batchSize * 3), 0..<startHeight, 0..<startWidth, 0..<channels
          ].copied()
        if cfgZeroStar.isEnabled {
          let etUncondF32 = DynamicGraph.Tensor<Float>(from: etUncond)
          let etCondF32 = DynamicGraph.Tensor<Float>(from: etCond)
          var etAllUncondF32 = DynamicGraph.Tensor<Float>(from: etAllUncond)
          let dotProduct = (etAllUncondF32 .* etUncondF32).reduced(.sum, axis: [1, 2, 3])
          let squaredSum = (etAllUncondF32 .* etAllUncondF32).reduced(
            .sum, axis: [1, 2, 3])
          let stStar = dotProduct ./ squaredSum
          etAllUncondF32 = etAllUncondF32 .* stStar
          et = DynamicGraph.Tensor<FloatType>(
            from: etAllUncondF32 + textGuidanceScale * (etCondF32 - etUncondF32)
              + imageGuidanceScale * (etUncondF32 - etAllUncondF32))
        } else {
          et =
            etAllUncond + textGuidanceScale * (etCond - etUncond) + imageGuidanceScale
            * (etUncond - etAllUncond)
        }
        /*
        let etUncondF32 = DynamicGraph.Tensor<Float>(from: etUncond)
        let etCondF32 = DynamicGraph.Tensor<Float>(from: etCond)
        let etAllUncondF32 =
        DynamicGraph.Tensor<Float>(from: etOut[
            (batchSize * 2)..<(batchSize * 3), 0..<startHeight, 0..<startWidth, 0..<channels
          ].copied())
        let predText_ = etUncondF32 + textGuidanceScale * (etCondF32 - etUncondF32)
        let normFullCondF32 = etCondF32.reduced(.norm2, axis: [1, 2, 3])
        let normPredText = predText_.reduced(.norm2, axis: [1, 2, 3])
        let scale = (normFullCondF32 ./ normPredText + 1e-8).clamped(0...1)
        let predText = scale .* predText_
        et = DynamicGraph.Tensor<FloatType>(from: etAllUncondF32 + imageGuidanceScale * (predText - etAllUncondF32))
          */
      }
    } else {
      if cfgZeroStar.isEnabled {
        let dotProduct: DynamicGraph.Tensor<Float>
        let squaredSum: DynamicGraph.Tensor<Float>
        let etUncondF32 = DynamicGraph.Tensor<Float>(from: etUncond)
        let etCondF32 = DynamicGraph.Tensor<Float>(from: etCond)
        if isBatchEnabled {
          dotProduct = (etUncondF32 .* etCondF32).reduced(
            .sum, axis: [0, 1, 2, 3])
          squaredSum =
            (etUncondF32 .* etUncondF32).reduced(
              .sum, axis: [0, 1, 2, 3]) + 1e-8
        } else {
          dotProduct = (etUncondF32 .* etCondF32).reduced(
            .sum, axis: [1, 2, 3])
          squaredSum =
            (etUncondF32 .* etUncondF32).reduced(.sum, axis: [1, 2, 3])
            + 1e-8
        }
        let stStar = dotProduct ./ squaredSum
        etUncond = DynamicGraph.Tensor<FloatType>(from: etUncondF32 .* stStar)
      }
      et = etUncond + textGuidanceScale * (etCond - etUncond)
    }
  } else {
    var etOut = etOut
    if channels < etOut.shape[3] {
      etOut = etOut[0..<batchSize, 0..<startHeight, 0..<startWidth, 0..<channels].copied()
    }
    if let blur = blur {
      let etOutDegraded = blur(inputs: etOut)[0].as(of: FloatType.self)
      etOut = Functional.add(
        left: etOutDegraded, right: etOut, leftScalar: alpha, rightScalar: 1 - alpha)
    }
    et = etOut
  }
  if cfgZeroStar.isEnabled, step < cfgZeroStar.zeroInitSteps {
    // Predicting zero.
    et.full(0)
  }
  return et
}
