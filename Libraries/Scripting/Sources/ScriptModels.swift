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
  let inputOverride: String
  let targetBlocks: [String]

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
    targetBlocks = control.targetBlocks
    inputOverride = control.inputOverride.description.lowercased()
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
    let inputOverride: ControlInputType = {
      for inputType in ControlInputType.allCases {
        if self.inputOverride.lowercased() == inputType.description.lowercased() {
          return inputType
        }
      }
      return .unspecified
    }()
    return Control(
      file: file, weight: weight, guidanceStart: guidanceStart, guidanceEnd: guidanceEnd,
      noPrompt: noPrompt, globalAveragePooling: globalAveragePooling,
      downSamplingRate: downSamplingRate, controlMode: controlMode, targetBlocks: targetBlocks,
      inputOverride: inputOverride)
  }
}

public final class JSGenerationConfiguration: Codable {
  public let id: Int64
  public var width: UInt32
  public var height: UInt32
  public let seed: Int64
  public let steps: UInt32
  public let guidanceScale: Float32
  public let strength: Float32
  public let model: String?
  public let sampler: Int8
  public let hiresFix: Bool
  public var hiresFixWidth: UInt32
  public var hiresFixHeight: UInt32
  public let hiresFixStrength: Float32
  public let tiledDecoding: Bool
  public var decodingTileWidth: UInt32
  public var decodingTileHeight: UInt32
  public var decodingTileOverlap: UInt32
  public let tiledDiffusion: Bool
  public var diffusionTileWidth: UInt32
  public var diffusionTileHeight: UInt32
  public var diffusionTileOverlap: UInt32
  public let upscaler: String?
  public var upscalerScaleFactor: UInt8
  public let imageGuidanceScale: Float32
  public let seedMode: Int8
  public let clipSkip: UInt32
  public let controls: [JSControl]
  public let loras: [JSLoRA]
  public let maskBlur: Float32
  public let maskBlurOutset: Int32
  public let sharpness: Float32
  public let faceRestoration: String?
  public let clipWeight: Float32
  public let negativePromptForImagePrior: Bool
  public let imagePriorSteps: UInt32
  public let refinerModel: String?
  public let originalImageHeight: UInt32
  public let originalImageWidth: UInt32
  public let cropTop: Int32
  public let cropLeft: Int32
  public let targetImageHeight: UInt32
  public let targetImageWidth: UInt32
  public let aestheticScore: Float32
  public let negativeAestheticScore: Float32
  public let zeroNegativePrompt: Bool
  public let refinerStart: Float32
  public let negativeOriginalImageHeight: UInt32
  public let negativeOriginalImageWidth: UInt32
  public let batchCount: UInt32
  public let batchSize: UInt32
  public let numFrames: UInt32
  public let fps: UInt32
  public let motionScale: UInt32
  public let guidingFrameNoise: Float32
  public let startFrameGuidance: Float32
  public let shift: Float32
  public let stage2Steps: UInt32
  public let stage2Guidance: Float32
  public let stage2Shift: Float32
  public let stochasticSamplingGamma: Float32
  public let preserveOriginalAfterInpaint: Bool
  public let t5TextEncoder: Bool
  public let separateClipL: Bool
  public let clipLText: String?
  public let separateOpenClipG: Bool
  public let openClipGText: String?
  public let speedUpWithGuidanceEmbed: Bool
  public let guidanceEmbed: Float32
  public let resolutionDependentShift: Bool
  public let teaCache: Bool
  public let teaCacheStart: Int32
  public let teaCacheEnd: Int32
  public let teaCacheThreshold: Float32
  public let separateT5: Bool
  public let t5Text: String?

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
    tiledDiffusion = configuration.tiledDiffusion
    diffusionTileWidth = UInt32(configuration.diffusionTileWidth) * 64
    diffusionTileHeight = UInt32(configuration.diffusionTileHeight) * 64
    diffusionTileOverlap = UInt32(configuration.diffusionTileOverlap) * 64
    upscaler = configuration.upscaler
    upscalerScaleFactor = configuration.upscalerScaleFactor
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
    t5TextEncoder = configuration.t5TextEncoder
    separateClipL = configuration.separateClipL
    clipLText = configuration.clipLText
    separateOpenClipG = configuration.separateOpenClipG
    openClipGText = configuration.openClipGText
    speedUpWithGuidanceEmbed = configuration.speedUpWithGuidanceEmbed
    guidanceEmbed = configuration.guidanceEmbed
    resolutionDependentShift = configuration.resolutionDependentShift
    teaCacheStart = configuration.teaCacheStart
    teaCacheEnd = configuration.teaCacheEnd
    teaCacheThreshold = configuration.teaCacheThreshold
    teaCache = configuration.teaCache
    separateT5 = configuration.separateT5
    t5Text = configuration.t5Text
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
      preserveOriginalAfterInpaint: preserveOriginalAfterInpaint,
      tiledDiffusion: tiledDiffusion,
      diffusionTileWidth: UInt16(diffusionTileWidth / 64),
      diffusionTileHeight: UInt16(diffusionTileHeight / 64),
      diffusionTileOverlap: UInt16(diffusionTileOverlap / 64),
      upscalerScaleFactor: upscalerScaleFactor,
      t5TextEncoder: t5TextEncoder,
      separateClipL: separateClipL,
      clipLText: clipLText,
      separateOpenClipG: separateOpenClipG,
      openClipGText: openClipGText,
      speedUpWithGuidanceEmbed: speedUpWithGuidanceEmbed,
      guidanceEmbed: guidanceEmbed,
      resolutionDependentShift: resolutionDependentShift,
      teaCacheStart: teaCacheStart,
      teaCacheEnd: teaCacheEnd,
      teaCacheThreshold: teaCacheThreshold,
      teaCache: teaCache,
      separateT5: separateT5,
      t5Text: t5Text
    )
  }
}

public final class JSPoint: Codable {
  public let x: CGFloat
  public let y: CGFloat

  public init(point: CGPoint) {
    self.x = point.x
    self.y = point.y
  }

  public func createCGPoint() -> CGPoint {
    return CGPoint(x: x, y: y)
  }
}

public final class JSLandmark: Codable {
  public let rect: JSRect
  public let normalizedPoints: [JSPoint]

  public init(rect: JSRect, points: [JSPoint]) {
    self.rect = rect
    self.normalizedPoints = points
  }
}

public final class JSRect: Codable {
  public let origin: JSPoint
  public let size: JSSize

  public init(rect: CGRect) {
    self.origin = JSPoint(point: rect.origin)
    self.size = JSSize(size: rect.size)
  }
  public init(lowerLeftRect: CGRect) {
    self.origin = JSPoint(
      point: CGPoint(
        x: lowerLeftRect.origin.x, y: lowerLeftRect.origin.y - lowerLeftRect.size.height))
    self.size = JSSize(size: lowerLeftRect.size)
  }
  public func createCGRect() -> CGRect {
    return CGRect(origin: origin.createCGPoint(), size: size.createCGSize())
  }
}

public final class JSSize: Codable {
  public let width: CGFloat
  public let height: CGFloat

  public init(size: CGSize) {
    self.width = size.width
    self.height = size.height
  }

  public func createCGSize() -> CGSize {
    return CGSize(width: width, height: height)
  }
}
