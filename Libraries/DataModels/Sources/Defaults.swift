import Dflat

extension GenerationConfiguration {
  public static var `default`: GenerationConfiguration {
    let defaultScale = DeviceCapability.default
    return GenerationConfiguration(
      id: 0, startWidth: defaultScale.widthScale, startHeight: defaultScale.heightScale,
      seed: UInt32.random(in: UInt32.min...UInt32.max),
      steps: 30, guidanceScale: 7.5, strength: 1.0, model: "sd_v1.5_f16.ckpt",
      sampler: .dPMPP2MKarras, hiresFixStartWidth: 7, hiresFixStartHeight: 7, hiresFixStrength: 0.7,
      imageGuidanceScale: 1.5,
      seedMode: .scaleAlike, clipSkip: 1, maskBlur: 1.5, clipWeight: 1, aestheticScore: 6,
      negativeAestheticScore: 2.5, refinerStart: 0.85, fpsId: 5, motionBucketId: 127, condAug: 0.02,
      startFrameCfg: 1.0, numFrames: 14, maskBlurOutset: 0, sharpness: 0)
  }
}

extension GenerationEstimation {
  public static var `default`: GenerationEstimation {
    // TODO: This needs to be based on the device model. It also depends on input width / height. To some extents, the sampler.
    return GenerationEstimation(
      textEncoded: 0.56, imageEncoded: 1, samplingStep: 2.05, imageDecoded: 1.9,
      secondPassImageEncoded: 1, secondPassSamplingStep: 2.05, secondPassImageDecoded: 1.9,
      imageUpscaled: 5.5, faceRestored: 2)
  }
}

extension SamplerType: CustomStringConvertible {
  public var description: String {
    switch self {
    case .PLMS:
      return "PLMS"
    case .DDIM:
      return "DDIM"
    case .eulerA:
      return "Euler Ancestral"
    case .dPMPP2MKarras:
      return "DPM++ 2M Karras"
    case .dPMPPSDEKarras:
      return "DPM++ SDE Karras"
    case .uniPC:
      return "UniPC"
    case .LCM:
      return "LCM"
    case .eulerASubstep:
      return "Euler A Substep"
    case .dPMPPSDESubstep:
      return "DPM++ SDE Substep"
    case .TCD:
      return "TCD"
    case .eulerATrailing:
      return "Euler A Trailing"
    case .dPMPPSDETrailing:
      return "DPM++ SDE Trailing"
    case .DPMPP2MAYS:
      return "DPM++ 2M AYS"
    case .eulerAAYS:
      return "Euler A AYS"
    case .DPMPPSDEAYS:
      return "DPM++ SDE AYS"
    case .dPMPP2MTrailing:
      return "DPM++ 2M Trailing"
    case .dDIMTrailing:
      return "DDIM Trailing"
    }
  }

  public init(from rawString: String) {
    let sampler = rawString.lowercased()
    if sampler.contains("dpm") {
      if sampler.contains("sde") {
        if sampler.contains("substep") {
          self = SamplerType.dPMPPSDESubstep
        } else if sampler.contains("trailing") {
          self = SamplerType.dPMPPSDETrailing
        } else if sampler.contains("ays") {
          self = SamplerType.DPMPPSDEAYS
        } else {
          self = SamplerType.dPMPPSDEKarras
        }
      } else {
        if sampler.contains("ays") {
          self = SamplerType.DPMPP2MAYS
        } else if sampler.contains("trailing") {
          self = SamplerType.dPMPP2MTrailing
        } else {
          self = SamplerType.dPMPP2MKarras
        }
      }
    } else if sampler.contains("euler") {
      if sampler.contains("substep") {
        self = SamplerType.eulerASubstep
      } else if sampler.contains("trailing") {
        self = SamplerType.eulerATrailing
      } else if sampler.contains("ays") {
        self = SamplerType.eulerAAYS
      } else {
        self = SamplerType.eulerA
      }
    } else if sampler.contains("unipc") {
      self = SamplerType.uniPC
    } else if sampler.contains("plms") {
      self = SamplerType.PLMS
    } else if sampler.contains("ddim") {
      if sampler.contains("trailing") {
        self = SamplerType.dDIMTrailing
      } else {
        self = SamplerType.DDIM
      }
    } else if sampler.contains("lcm") {
      self = SamplerType.LCM
    } else if sampler.contains("tcd") {
      self = SamplerType.TCD
    } else {
      self = SamplerType.dPMPP2MKarras
    }
  }
}

extension SeedMode {
  public init(from rawString: String) {
    switch rawString {
    case "Legacy":
      self = .legacy
    case "Scale Alike":
      self = .scaleAlike
    case "Torch CPU Compatible":
      self = .torchCpuCompatible
    case "NVIDIA GPU Compatible":
      self = .nvidiaGpuCompatible
    default:
      self = .scaleAlike
    }
  }
}

extension LoRATrainingConfiguration {
  public static var `default`: LoRATrainingConfiguration {
    return LoRATrainingConfiguration(
      id: 0, startWidth: 8, startHeight: 8, seed: UInt32.random(in: UInt32.min...UInt32.max),
      trainingSteps: 2000, baseModel: nil, networkDim: 32, networkScale: 1, unetLearningRate: 1e-4,
      saveEveryNSteps: 250, warmupSteps: 20, gradientAccumulationSteps: 4, cotrainTextModel: false,
      textModelLearningRate: 4e-5, clipSkip: 1, noiseOffset: 0.05, denoisingStart: 0,
      denoisingEnd: 1, autoCaptioning: true, cotrainCustomEmbedding: false,
      customEmbeddingLearningRate: 0.05, customEmbeddingLength: 4, stopEmbeddingTrainingAtStep: 500)
  }
}
