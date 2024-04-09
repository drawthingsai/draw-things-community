import DataModels
import Foundation
import NNC

public final class JSLoRA: Codable {
  let file: String?
  let weight: Float32
  public init(lora: LoRA) {
    file = lora.file
    weight = lora.weight
  }

  public func createLora() -> LoRA {
    return DataModels.LoRA(file: file, weight: weight)
  }
}

public final class JSControl: Codable {
  let file: String?
  let weight: Float32
  let guidanceStart: Float32
  let guidanceEnd: Float32
  let noPrompt: Bool
  let globalAveragePooling: Bool
  let downSamplingRate: Float32
  let controlImportance: String

  public init(control: DataModels.Control) {
    file = control.file
    weight = control.weight
    guidanceStart = control.guidanceStart
    guidanceEnd = control.guidanceEnd
    noPrompt = control.noPrompt
    globalAveragePooling = control.globalAveragePooling
    downSamplingRate = control.downSamplingRate
    switch control.controlMode {
    case .balanced:
      controlImportance = "balanced"
    case .prompt:
      controlImportance = "prompt"
    case .control:
      controlImportance = "control"
    }
  }

  public func createControl() -> DataModels.Control {
    let controlMode: DataModels.ControlMode = {
      switch controlImportance.lowercased() {
      case "balanced":
        return .balanced
      case "prompt":
        return .prompt
      case "control":
        return .control
      default:
        return .balanced
      }
    }()
    return Control(
      file: file, weight: weight, guidanceStart: guidanceStart, guidanceEnd: guidanceEnd,
      noPrompt: noPrompt, globalAveragePooling: globalAveragePooling,
      downSamplingRate: downSamplingRate, controlMode: controlMode)
  }
}

// TODO: when we decode from JSON, should we catch any unknown keys, e.g. from a misspelling like:
// configuration.startWidth = 2;
public final class JSGenerationConfiguration: Codable {
  let id: Int64
  var width: UInt32
  var height: UInt32
  let seed: Int64
  let steps: UInt32
  let guidanceScale: Float32
  let strength: Float32
  let model: String?
  let sampler: Int8
  let hiresFix: Bool
  var hiresFixWidth: UInt32
  var hiresFixHeight: UInt32
  let hiresFixStrength: Float32
  let tiledDecoding: Bool
  var decodingTileWidth: UInt32
  var decodingTileHeight: UInt32
  var decodingTileOverlap: UInt32
  let upscaler: String?
  let imageGuidanceScale: Float32
  let seedMode: Int8
  let clipSkip: UInt32
  let controls: [JSControl]
  let loras: [JSLoRA]
  let maskBlur: Float32
  let maskBlurOutset: Int32
  let sharpness: Float32
  let faceRestoration: String?
  let clipWeight: Float32
  let negativePromptForImagePrior: Bool
  let imagePriorSteps: UInt32
  let refinerModel: String?
  let originalImageHeight: UInt32
  let originalImageWidth: UInt32
  let cropTop: Int32
  let cropLeft: Int32
  let targetImageHeight: UInt32
  let targetImageWidth: UInt32
  let aestheticScore: Float32
  let negativeAestheticScore: Float32
  let zeroNegativePrompt: Bool
  let refinerStart: Float32
  let negativeOriginalImageHeight: UInt32
  let negativeOriginalImageWidth: UInt32
  let batchCount: UInt32
  let batchSize: UInt32
  let numFrames: UInt32
  let fps: UInt32
  let motionScale: UInt32
  let guidingFrameNoise: Float32
  let startFrameGuidance: Float32
  let shift: Float32
  let stage2Steps: UInt32
  let stage2Guidance: Float32
  let stage2Shift: Float32
  let stochasticSamplingGamma: Float32
  let preserveOriginalAfterInpaint: Bool

  public init(configuration: GenerationConfiguration) {
    id = configuration.id
    width = UInt32(configuration.startWidth) * 64
    height = UInt32(configuration.startHeight) * 64
    seed = Int64(configuration.seed)
    steps = configuration.steps
    guidanceScale = configuration.guidanceScale
    strength = configuration.strength
    model = configuration.model
    sampler = configuration.sampler.rawValue
    hiresFix = configuration.hiresFix
    hiresFixWidth = UInt32(configuration.hiresFixStartWidth) * 64
    hiresFixHeight = UInt32(configuration.hiresFixStartHeight) * 64
    hiresFixStrength = configuration.hiresFixStrength
    tiledDecoding = configuration.tiledDecoding
    decodingTileWidth = UInt32(configuration.decodingTileWidth) * 64
    decodingTileHeight = UInt32(configuration.decodingTileHeight) * 64
    decodingTileOverlap = UInt32(configuration.decodingTileOverlap) * 64
    upscaler = configuration.upscaler
    imageGuidanceScale = configuration.imageGuidanceScale
    seedMode = configuration.seedMode.rawValue
    clipSkip = configuration.clipSkip
    controls = configuration.controls.map { JSControl(control: $0) }
    loras = configuration.loras.map { JSLoRA(lora: $0) }
    maskBlur = configuration.maskBlur
    maskBlurOutset = configuration.maskBlurOutset
    sharpness = configuration.sharpness
    faceRestoration = configuration.faceRestoration
    clipWeight = configuration.clipWeight
    negativePromptForImagePrior = configuration.negativePromptForImagePrior
    imagePriorSteps = configuration.imagePriorSteps
    refinerModel = configuration.refinerModel
    originalImageHeight = configuration.originalImageHeight
    originalImageWidth = configuration.originalImageWidth
    cropTop = configuration.cropTop
    cropLeft = configuration.cropLeft
    targetImageHeight = configuration.targetImageHeight
    targetImageWidth = configuration.targetImageWidth
    aestheticScore = configuration.aestheticScore
    negativeAestheticScore = configuration.negativeAestheticScore
    zeroNegativePrompt = configuration.zeroNegativePrompt
    refinerStart = configuration.refinerStart
    negativeOriginalImageHeight = configuration.negativeOriginalImageHeight
    negativeOriginalImageWidth = configuration.negativeOriginalImageWidth
    batchCount = configuration.batchCount
    batchSize = configuration.batchSize
    numFrames = configuration.numFrames
    fps = configuration.fpsId
    motionScale = configuration.motionBucketId
    guidingFrameNoise = configuration.condAug
    startFrameGuidance = configuration.startFrameCfg
    shift = configuration.shift
    stage2Guidance = configuration.stage2Cfg
    stage2Shift = configuration.stage2Shift
    stage2Steps = configuration.stage2Steps
    stochasticSamplingGamma = configuration.stochasticSamplingGamma
    preserveOriginalAfterInpaint = configuration.preserveOriginalAfterInpaint
  }

  public func createGenerationConfiguration() -> GenerationConfiguration {
    let loras: [DataModels.LoRA] = loras.map { $0.createLora() }
    let controls: [DataModels.Control] = controls.map { $0.createControl() }
    return GenerationConfiguration(
      id: id, startWidth: UInt16(width / 64), startHeight: UInt16(height / 64),
      seed: seed >= 0 ? UInt32(seed) : UInt32.random(in: UInt32.min...UInt32.max), steps: steps,
      guidanceScale: guidanceScale, strength: strength, model: model,
      sampler: SamplerType(rawValue: sampler)!, batchCount: max(batchCount, 1),
      batchSize: min(max(batchSize, 1), 4),
      hiresFix: hiresFix, hiresFixStartWidth: UInt16(hiresFixWidth / 64),
      hiresFixStartHeight: UInt16(hiresFixHeight / 64), hiresFixStrength: hiresFixStrength,
      upscaler: upscaler, imageGuidanceScale: imageGuidanceScale,
      seedMode: SeedMode(rawValue: seedMode)!, clipSkip: clipSkip, controls: controls, loras: loras,
      maskBlur: maskBlur, faceRestoration: faceRestoration, clipWeight: clipWeight,
      negativePromptForImagePrior: negativePromptForImagePrior, imagePriorSteps: imagePriorSteps,
      refinerModel: refinerModel, originalImageHeight: originalImageHeight,
      originalImageWidth: originalImageWidth, cropTop: cropTop, cropLeft: cropLeft,
      targetImageHeight: targetImageHeight, targetImageWidth: targetImageWidth,
      aestheticScore: aestheticScore, negativeAestheticScore: negativeAestheticScore,
      zeroNegativePrompt: zeroNegativePrompt, refinerStart: refinerStart,
      negativeOriginalImageHeight: negativeOriginalImageHeight,
      negativeOriginalImageWidth: negativeOriginalImageWidth,
      fpsId: fps, motionBucketId: motionScale, condAug: guidingFrameNoise,
      startFrameCfg: startFrameGuidance, numFrames: numFrames, maskBlurOutset: maskBlurOutset,
      sharpness: sharpness, shift: shift, stage2Steps: stage2Steps, stage2Cfg: stage2Guidance,
      stage2Shift: stage2Shift, tiledDecoding: tiledDecoding,
      decodingTileWidth: UInt16(decodingTileWidth / 64),
      decodingTileHeight: UInt16(decodingTileHeight / 64),
      decodingTileOverlap: UInt16(decodingTileOverlap / 64),
      stochasticSamplingGamma: stochasticSamplingGamma,
      preserveOriginalAfterInpaint: preserveOriginalAfterInpaint
    )
  }
}

final class JSPoint: Codable {
  let x: CGFloat
  let y: CGFloat

  init(point: CGPoint) {
    self.x = point.x
    self.y = point.y
  }

  func createCGPoint() -> CGPoint {
    return CGPoint(x: x, y: y)
  }
}

final class JSLandmark: Codable {
  let rect: JSRect
  let normalizedPoints: [JSPoint]

  init(rect: JSRect, points: [JSPoint]) {
    self.rect = rect
    self.normalizedPoints = points
  }
}

final class JSRect: Codable {
  let origin: JSPoint
  let size: JSSize

  init(rect: CGRect) {
    self.origin = JSPoint(point: rect.origin)
    self.size = JSSize(size: rect.size)
  }
  init(lowerLeftRect: CGRect) {
    self.origin = JSPoint(
      point: CGPoint(
        x: lowerLeftRect.origin.x, y: lowerLeftRect.origin.y - lowerLeftRect.size.height))
    self.size = JSSize(size: lowerLeftRect.size)
  }
  func createCGRect() -> CGRect {
    return CGRect(origin: origin.createCGPoint(), size: size.createCGSize())
  }
}

final class JSSize: Codable {
  let width: CGFloat
  let height: CGFloat

  init(size: CGSize) {
    self.width = size.width
    self.height = size.height
  }

  func createCGSize() -> CGSize {
    return CGSize(width: width, height: height)
  }
}

final class MaskManager {
  // Start actual handles at 1, because 0 indicates that a null mask handle was passed from JS
  let nullMask = JSMask(handle: 0)
  private var nextMaskHandle: Int = 1
  private var maskHandleToMask: [Int: Tensor<UInt8>] = [:]

  func mask(forJSMask jsMask: JSMask) -> Tensor<UInt8>? {
    return maskHandleToMask[jsMask.handle]
  }

  func setMask(_ mask: Tensor<UInt8>, forJSMask jsMask: JSMask) {
    maskHandleToMask[jsMask.handle] = mask
  }

  func createNewMask() -> JSMask {
    let mask = JSMask(handle: nextMaskHandle)
    nextMaskHandle += 1
    return mask
  }
}

final class JSMask: Codable {
  let handle: Int

  fileprivate init(handle: Int) {
    self.handle = handle
  }
}
