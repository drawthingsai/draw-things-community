import C_ccv
import DataModels
import Dflat
import Diffusion
import DiffusionPreprocessors
import DiffusionUNetWrapper
import Foundation
import ImageGenerator
import ModelZoo
import NNC
import Tokenizer
import Upscaler
import WeightsCache

#if !os(Linux)
  import DiffusionCoreML
  import FaceRestorer
#endif

public func xorshift(_ a: UInt32) -> UInt32 {
  var x = a == 0 ? 0xbad_5eed : a
  x ^= x &<< 13
  x ^= x &>> 17
  x ^= x &<< 5
  return x
}

public struct LocalImageGenerator: ImageGenerator {
  public let modelPreloader: ModelPreloader
  public var tokenizerV1: TextualInversionAttentionCLIPTokenizer
  public var tokenizerV2: TextualInversionAttentionCLIPTokenizer
  public var tokenizerXL: TextualInversionAttentionCLIPTokenizer
  public var tokenizerKandinsky: SentencePieceTokenizer
  public var tokenizerT5: SentencePieceTokenizer
  public var tokenizerPileT5: SentencePieceTokenizer
  public var tokenizerChatGLM3: SentencePieceTokenizer
  public var tokenizerLlama3: TiktokenTokenizer
  public var tokenizerUMT5: SentencePieceTokenizer
  private let queue: DispatchQueue
  private let weightsCache: WeightsCache
  public init(
    queue: DispatchQueue, configurations: FetchedResult<GenerationConfiguration>,
    workspace: Workspace, tokenizerV1: TextualInversionAttentionCLIPTokenizer,
    tokenizerV2: TextualInversionAttentionCLIPTokenizer,
    tokenizerXL: TextualInversionAttentionCLIPTokenizer,
    tokenizerKandinsky: SentencePieceTokenizer,
    tokenizerT5: SentencePieceTokenizer,
    tokenizerPileT5: SentencePieceTokenizer,
    tokenizerChatGLM3: SentencePieceTokenizer,
    tokenizerLlama3: TiktokenTokenizer,
    tokenizerUMT5: SentencePieceTokenizer
  ) {
    self.queue = queue
    self.tokenizerV1 = tokenizerV1
    self.tokenizerV2 = tokenizerV2
    self.tokenizerXL = tokenizerXL
    self.tokenizerKandinsky = tokenizerKandinsky
    self.tokenizerT5 = tokenizerT5
    self.tokenizerPileT5 = tokenizerPileT5
    self.tokenizerChatGLM3 = tokenizerChatGLM3
    self.tokenizerLlama3 = tokenizerLlama3
    self.tokenizerUMT5 = tokenizerUMT5
    weightsCache = WeightsCache(
      maxTotalCacheSize: DeviceCapability.maxTotalWeightsCacheSize,
      memorySubsystem: DeviceCapability.isUMA ? .UMA : .dGPU)
    modelPreloader = ModelPreloader(
      queue: queue, weightsCache: weightsCache, configurations: configurations, workspace: workspace
    )
  }
}

extension LocalImageGenerator {
  public static func sampler<FloatType: TensorNumeric & BinaryFloatingPoint>(
    from type: SamplerType, isCfgEnabled: Bool, filePath: String, modifier: SamplerModifier,
    version: ModelVersion, qkNorm: Bool, dualAttentionLayers: [Int],
    distilledGuidanceLayers: Int,
    usesFlashAttention: Bool, objective: Denoiser.Objective,
    upcastAttention: Bool, externalOnDemand: Bool, injectControls: Bool, injectT2IAdapters: Bool,
    injectAttentionKV: Bool,
    injectIPAdapterLengths: [Int], lora: [LoRAConfiguration], isGuidanceEmbedEnabled: Bool,
    isQuantizedModel: Bool, canRunLoRASeparately: Bool, stochasticSamplingGamma: Float,
    conditioning: Denoiser.Conditioning, parameterization: Denoiser.Parameterization,
    tiledDiffusion: TiledConfiguration, teaCache: TeaCacheConfiguration, causalInference: Int,
    weightsCache: WeightsCache, of: FloatType.Type
  ) -> any Sampler<FloatType, UNetWrapper<FloatType>> {
    let manualSubsteps: (Int) -> [Int] = {
      switch $0 {
      case 1:
        return [999, 0]
      case 2:
        return [999, 899, 0]
      case 3:
        return [999, 899, 799, 0]
      case 4:
        return [999, 899, 799, 699, 0]
      case 5:
        return [999, 899, 799, 699, 599, 0]
      case 6:
        return [999, 899, 799, 699, 599, 499, 0]
      case 7:
        return [999, 899, 799, 699, 599, 499, 399, 0]
      case 8:
        return [999, 899, 799, 699, 599, 499, 399, 299, 0]
      case 9:
        return [999, 899, 799, 699, 599, 499, 399, 299, 199, 0]
      default:
        return []
      }
    }
    let samplingTimesteps: [Int]
    let samplingSigmas: [Double]
    switch version {
    case .v1, .v2:
      samplingTimesteps = [999, 850, 736, 645, 545, 455, 343, 233, 124, 24, 0]
      samplingSigmas = []
    case .sdxlBase, .sdxlRefiner, .ssd1b:
      switch parameterization {
      case .rf(_), .edm(_):
        samplingTimesteps = []
      case .ddpm(_):
        samplingTimesteps = [999, 845, 730, 587, 443, 310, 193, 116, 53, 13, 0]
      }
      samplingSigmas = []
    case .svdI2v:
      samplingTimesteps = []
      samplingSigmas = [
        700.00, 54.5, 15.886, 7.977, 4.248, 1.789, 0.981, 0.403, 0.173, 0.034, 0.002,
      ]
    case .sd3, .sd3Large, .pixart, .auraflow, .flux1, .kandinsky21, .wurstchenStageB,
      .wurstchenStageC, .hunyuanVideo, .wan21_1_3b, .wan21_14b, .hiDreamI1:
      samplingTimesteps = []
      samplingSigmas = []
    }
    let deviceProperties = DeviceCapability.deviceProperties
    guard version != .wurstchenStageC && version != .wurstchenStageB else {
      switch type {
      case .dPMPP2MKarras, .DPMPP2MAYS, .dPMPP2MTrailing:
        return DPMPP2MSampler<FloatType, UNetWrapper<FloatType>, Denoiser.CosineDiscretization>(
          filePath: filePath, modifier: modifier, version: version, qkNorm: qkNorm,
          dualAttentionLayers: dualAttentionLayers,
          distilledGuidanceLayers: distilledGuidanceLayers,
          usesFlashAttention: usesFlashAttention,
          upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
          injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
          injectAttentionKV: injectAttentionKV,
          injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
          classifierFreeGuidance: isCfgEnabled, isGuidanceEmbedEnabled: isGuidanceEmbedEnabled,
          isQuantizedModel: isQuantizedModel,
          canRunLoRASeparately: canRunLoRASeparately,
          deviceProperties: deviceProperties, conditioning: conditioning,
          tiledDiffusion: tiledDiffusion, teaCache: teaCache, causalInference: causalInference,
          discretization: Denoiser.CosineDiscretization(parameterization, objective: objective),
          weightsCache: weightsCache)
      case .eulerA, .eulerASubstep, .eulerATrailing, .eulerAAYS:
        return EulerASampler<FloatType, UNetWrapper<FloatType>, Denoiser.CosineDiscretization>(
          filePath: filePath, modifier: modifier, version: version, qkNorm: qkNorm,
          dualAttentionLayers: dualAttentionLayers,
          distilledGuidanceLayers: distilledGuidanceLayers,
          usesFlashAttention: usesFlashAttention,
          upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
          injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
          injectAttentionKV: injectAttentionKV,
          injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
          classifierFreeGuidance: isCfgEnabled, isGuidanceEmbedEnabled: isGuidanceEmbedEnabled,
          isQuantizedModel: isQuantizedModel,
          canRunLoRASeparately: canRunLoRASeparately,
          deviceProperties: deviceProperties, conditioning: conditioning,
          tiledDiffusion: tiledDiffusion, teaCache: teaCache, causalInference: causalInference,
          discretization: Denoiser.CosineDiscretization(parameterization, objective: objective),
          weightsCache: weightsCache)
      case .DDIM, .dDIMTrailing:
        return DDIMSampler<FloatType, UNetWrapper<FloatType>, Denoiser.CosineDiscretization>(
          filePath: filePath, modifier: modifier, version: version, qkNorm: qkNorm,
          dualAttentionLayers: dualAttentionLayers,
          distilledGuidanceLayers: distilledGuidanceLayers,
          usesFlashAttention: usesFlashAttention,
          upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
          injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
          injectAttentionKV: injectAttentionKV,
          injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
          classifierFreeGuidance: isCfgEnabled, isGuidanceEmbedEnabled: isGuidanceEmbedEnabled,
          isQuantizedModel: isQuantizedModel,
          canRunLoRASeparately: canRunLoRASeparately,
          deviceProperties: deviceProperties, conditioning: conditioning,
          tiledDiffusion: tiledDiffusion, teaCache: teaCache, causalInference: causalInference,
          discretization: Denoiser.CosineDiscretization(parameterization, objective: objective),
          weightsCache: weightsCache)
      case .PLMS:
        return PLMSSampler<FloatType, UNetWrapper<FloatType>, Denoiser.CosineDiscretization>(
          filePath: filePath, modifier: modifier, version: version, qkNorm: qkNorm,
          dualAttentionLayers: dualAttentionLayers,
          distilledGuidanceLayers: distilledGuidanceLayers,
          usesFlashAttention: usesFlashAttention,
          upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
          injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
          injectAttentionKV: injectAttentionKV,
          injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
          classifierFreeGuidance: isCfgEnabled, isGuidanceEmbedEnabled: isGuidanceEmbedEnabled,
          isQuantizedModel: isQuantizedModel,
          canRunLoRASeparately: canRunLoRASeparately,
          deviceProperties: deviceProperties, conditioning: conditioning,
          tiledDiffusion: tiledDiffusion, teaCache: teaCache, causalInference: causalInference,
          discretization: Denoiser.CosineDiscretization(parameterization, objective: objective),
          weightsCache: weightsCache)
      case .dPMPPSDEKarras, .dPMPPSDESubstep, .dPMPPSDETrailing, .DPMPPSDEAYS:
        return DPMPPSDESampler<FloatType, UNetWrapper<FloatType>, Denoiser.CosineDiscretization>(
          filePath: filePath, modifier: modifier, version: version, qkNorm: qkNorm,
          dualAttentionLayers: dualAttentionLayers,
          distilledGuidanceLayers: distilledGuidanceLayers,
          usesFlashAttention: usesFlashAttention,
          upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
          injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
          injectAttentionKV: injectAttentionKV,
          injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
          classifierFreeGuidance: isCfgEnabled, isGuidanceEmbedEnabled: isGuidanceEmbedEnabled,
          isQuantizedModel: isQuantizedModel,
          canRunLoRASeparately: canRunLoRASeparately,
          deviceProperties: deviceProperties, conditioning: conditioning,
          tiledDiffusion: tiledDiffusion, teaCache: teaCache, causalInference: causalInference,
          discretization: Denoiser.CosineDiscretization(parameterization, objective: objective),
          weightsCache: weightsCache)
      case .uniPC:
        return UniPCSampler<FloatType, UNetWrapper<FloatType>, Denoiser.CosineDiscretization>(
          filePath: filePath, modifier: modifier, version: version, qkNorm: qkNorm,
          dualAttentionLayers: dualAttentionLayers,
          distilledGuidanceLayers: distilledGuidanceLayers,
          usesFlashAttention: usesFlashAttention,
          upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
          injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
          injectAttentionKV: injectAttentionKV,
          injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
          classifierFreeGuidance: isCfgEnabled, isGuidanceEmbedEnabled: isGuidanceEmbedEnabled,
          isQuantizedModel: isQuantizedModel,
          canRunLoRASeparately: canRunLoRASeparately,
          deviceProperties: deviceProperties, conditioning: conditioning,
          tiledDiffusion: tiledDiffusion, teaCache: teaCache, causalInference: causalInference,
          discretization: Denoiser.CosineDiscretization(parameterization, objective: objective),
          weightsCache: weightsCache)
      case .LCM:
        return LCMSampler<FloatType, UNetWrapper<FloatType>, Denoiser.CosineDiscretization>(
          filePath: filePath, modifier: modifier, version: version, qkNorm: qkNorm,
          dualAttentionLayers: dualAttentionLayers,
          distilledGuidanceLayers: distilledGuidanceLayers,
          usesFlashAttention: usesFlashAttention,
          upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
          injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
          injectAttentionKV: injectAttentionKV,
          injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
          isGuidanceEmbedEnabled: isGuidanceEmbedEnabled, isQuantizedModel: isQuantizedModel,
          canRunLoRASeparately: canRunLoRASeparately,
          deviceProperties: deviceProperties,
          conditioning: conditioning, tiledDiffusion: tiledDiffusion, teaCache: teaCache,
          causalInference: causalInference,
          discretization: Denoiser.CosineDiscretization(parameterization, objective: objective),
          weightsCache: weightsCache)
      case .TCD:
        return TCDSampler<FloatType, UNetWrapper<FloatType>, Denoiser.CosineDiscretization>(
          filePath: filePath, modifier: modifier, version: version, qkNorm: qkNorm,
          dualAttentionLayers: dualAttentionLayers,
          distilledGuidanceLayers: distilledGuidanceLayers,
          usesFlashAttention: usesFlashAttention,
          upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
          injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
          injectAttentionKV: injectAttentionKV,
          injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
          isGuidanceEmbedEnabled: isGuidanceEmbedEnabled, isQuantizedModel: isQuantizedModel,
          canRunLoRASeparately: canRunLoRASeparately,
          deviceProperties: deviceProperties,
          stochasticSamplingGamma: stochasticSamplingGamma,
          conditioning: conditioning, tiledDiffusion: tiledDiffusion, teaCache: teaCache,
          causalInference: causalInference,
          discretization: Denoiser.CosineDiscretization(parameterization, objective: objective),
          weightsCache: weightsCache)
      }
    }
    switch type {
    case .dPMPP2MKarras:
      return DPMPP2MSampler<FloatType, UNetWrapper<FloatType>, Denoiser.KarrasDiscretization>(
        filePath: filePath, modifier: modifier, version: version, qkNorm: qkNorm,
        dualAttentionLayers: dualAttentionLayers, distilledGuidanceLayers: distilledGuidanceLayers,
        usesFlashAttention: usesFlashAttention,
        upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
        injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
        injectAttentionKV: injectAttentionKV,
        injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
        classifierFreeGuidance: isCfgEnabled, isGuidanceEmbedEnabled: isGuidanceEmbedEnabled,
        isQuantizedModel: isQuantizedModel,
        canRunLoRASeparately: canRunLoRASeparately, deviceProperties: deviceProperties,
        conditioning: conditioning,
        tiledDiffusion: tiledDiffusion, teaCache: teaCache, causalInference: causalInference,
        discretization: Denoiser.KarrasDiscretization(parameterization, objective: objective),
        weightsCache: weightsCache)
    case .DPMPP2MAYS:
      if samplingTimesteps.isEmpty && samplingSigmas.isEmpty {
        return DPMPP2MSampler<
          FloatType, UNetWrapper<FloatType>, Denoiser.LinearDiscretization
        >(
          filePath: filePath, modifier: modifier, version: version, qkNorm: qkNorm,
          dualAttentionLayers: dualAttentionLayers,
          distilledGuidanceLayers: distilledGuidanceLayers,
          usesFlashAttention: usesFlashAttention,
          upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
          injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
          injectAttentionKV: injectAttentionKV,
          injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
          classifierFreeGuidance: isCfgEnabled, isGuidanceEmbedEnabled: isGuidanceEmbedEnabled,
          isQuantizedModel: isQuantizedModel,
          canRunLoRASeparately: canRunLoRASeparately,
          deviceProperties: deviceProperties, conditioning: conditioning,
          tiledDiffusion: tiledDiffusion, teaCache: teaCache, causalInference: causalInference,
          discretization: Denoiser.LinearDiscretization(
            parameterization, objective: objective, timestepSpacing: .trailing),
          weightsCache: weightsCache)
      } else if samplingTimesteps.isEmpty {
        return DPMPP2MSampler<
          FloatType, UNetWrapper<FloatType>, Denoiser.AYSLogLinearInterpolatedKarrasDiscretization
        >(
          filePath: filePath, modifier: modifier, version: version, qkNorm: qkNorm,
          dualAttentionLayers: dualAttentionLayers,
          distilledGuidanceLayers: distilledGuidanceLayers,
          usesFlashAttention: usesFlashAttention,
          upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
          injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
          injectAttentionKV: injectAttentionKV,
          injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
          classifierFreeGuidance: isCfgEnabled, isGuidanceEmbedEnabled: isGuidanceEmbedEnabled,
          isQuantizedModel: isQuantizedModel,
          canRunLoRASeparately: canRunLoRASeparately,
          deviceProperties: deviceProperties, conditioning: conditioning,
          tiledDiffusion: tiledDiffusion, teaCache: teaCache, causalInference: causalInference,
          discretization: Denoiser.AYSLogLinearInterpolatedKarrasDiscretization(
            parameterization, objective: objective, samplingSigmas: samplingSigmas),
          weightsCache: weightsCache)
      } else {
        return DPMPP2MSampler<
          FloatType, UNetWrapper<FloatType>, Denoiser.AYSLogLinearInterpolatedTimestepDiscretization
        >(
          filePath: filePath, modifier: modifier, version: version, qkNorm: qkNorm,
          dualAttentionLayers: dualAttentionLayers,
          distilledGuidanceLayers: distilledGuidanceLayers,
          usesFlashAttention: usesFlashAttention,
          upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
          injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
          injectAttentionKV: injectAttentionKV,
          injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
          classifierFreeGuidance: isCfgEnabled, isGuidanceEmbedEnabled: isGuidanceEmbedEnabled,
          isQuantizedModel: isQuantizedModel,
          canRunLoRASeparately: canRunLoRASeparately,
          deviceProperties: deviceProperties, conditioning: conditioning,
          tiledDiffusion: tiledDiffusion, teaCache: teaCache, causalInference: causalInference,
          discretization: Denoiser.AYSLogLinearInterpolatedTimestepDiscretization(
            parameterization, objective: objective, samplingTimesteps: samplingTimesteps),
          weightsCache: weightsCache)
      }
    case .dPMPP2MTrailing:
      return DPMPP2MSampler<FloatType, UNetWrapper<FloatType>, Denoiser.LinearDiscretization>(
        filePath: filePath, modifier: modifier, version: version, qkNorm: qkNorm,
        dualAttentionLayers: dualAttentionLayers, distilledGuidanceLayers: distilledGuidanceLayers,
        usesFlashAttention: usesFlashAttention,
        upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
        injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
        injectAttentionKV: injectAttentionKV,
        injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
        classifierFreeGuidance: isCfgEnabled, isGuidanceEmbedEnabled: isGuidanceEmbedEnabled,
        isQuantizedModel: isQuantizedModel,
        canRunLoRASeparately: canRunLoRASeparately, deviceProperties: deviceProperties,
        conditioning: conditioning,
        tiledDiffusion: tiledDiffusion, teaCache: teaCache, causalInference: causalInference,
        discretization: Denoiser.LinearDiscretization(
          parameterization, objective: objective, timestepSpacing: .trailing),
        weightsCache: weightsCache)
    case .eulerA:
      return EulerASampler<FloatType, UNetWrapper<FloatType>, Denoiser.LinearDiscretization>(
        filePath: filePath, modifier: modifier, version: version, qkNorm: qkNorm,
        dualAttentionLayers: dualAttentionLayers, distilledGuidanceLayers: distilledGuidanceLayers,
        usesFlashAttention: usesFlashAttention,
        upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
        injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
        injectAttentionKV: injectAttentionKV,
        injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
        classifierFreeGuidance: isCfgEnabled, isGuidanceEmbedEnabled: isGuidanceEmbedEnabled,
        isQuantizedModel: isQuantizedModel,
        canRunLoRASeparately: canRunLoRASeparately, deviceProperties: deviceProperties,
        conditioning: conditioning,
        tiledDiffusion: tiledDiffusion, teaCache: teaCache, causalInference: causalInference,
        discretization: Denoiser.LinearDiscretization(parameterization, objective: objective),
        weightsCache: weightsCache)
    case .eulerATrailing:
      return EulerASampler<FloatType, UNetWrapper<FloatType>, Denoiser.LinearDiscretization>(
        filePath: filePath, modifier: modifier, version: version, qkNorm: qkNorm,
        dualAttentionLayers: dualAttentionLayers, distilledGuidanceLayers: distilledGuidanceLayers,
        usesFlashAttention: usesFlashAttention,
        upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
        injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
        injectAttentionKV: injectAttentionKV,
        injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
        classifierFreeGuidance: isCfgEnabled, isGuidanceEmbedEnabled: isGuidanceEmbedEnabled,
        isQuantizedModel: isQuantizedModel,
        canRunLoRASeparately: canRunLoRASeparately, deviceProperties: deviceProperties,
        conditioning: conditioning,
        tiledDiffusion: tiledDiffusion, teaCache: teaCache, causalInference: causalInference,
        discretization: Denoiser.LinearDiscretization(
          parameterization, objective: objective, timestepSpacing: .trailing),
        weightsCache: weightsCache)
    case .eulerAAYS:
      if samplingTimesteps.isEmpty && samplingSigmas.isEmpty {
        return EulerASampler<
          FloatType, UNetWrapper<FloatType>, Denoiser.LinearDiscretization
        >(
          filePath: filePath, modifier: modifier, version: version, qkNorm: qkNorm,
          dualAttentionLayers: dualAttentionLayers,
          distilledGuidanceLayers: distilledGuidanceLayers,
          usesFlashAttention: usesFlashAttention,
          upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
          injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
          injectAttentionKV: injectAttentionKV,
          injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
          classifierFreeGuidance: isCfgEnabled, isGuidanceEmbedEnabled: isGuidanceEmbedEnabled,
          isQuantizedModel: isQuantizedModel,
          canRunLoRASeparately: canRunLoRASeparately,
          deviceProperties: deviceProperties, conditioning: conditioning,
          tiledDiffusion: tiledDiffusion, teaCache: teaCache, causalInference: causalInference,
          discretization: Denoiser.LinearDiscretization(
            parameterization, objective: objective, timestepSpacing: .trailing),
          weightsCache: weightsCache)
      } else if samplingTimesteps.isEmpty {
        return EulerASampler<
          FloatType, UNetWrapper<FloatType>, Denoiser.AYSLogLinearInterpolatedKarrasDiscretization
        >(
          filePath: filePath, modifier: modifier, version: version, qkNorm: qkNorm,
          dualAttentionLayers: dualAttentionLayers,
          distilledGuidanceLayers: distilledGuidanceLayers,
          usesFlashAttention: usesFlashAttention,
          upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
          injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
          injectAttentionKV: injectAttentionKV,
          injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
          classifierFreeGuidance: isCfgEnabled, isGuidanceEmbedEnabled: isGuidanceEmbedEnabled,
          isQuantizedModel: isQuantizedModel,
          canRunLoRASeparately: canRunLoRASeparately,
          deviceProperties: deviceProperties, conditioning: conditioning,
          tiledDiffusion: tiledDiffusion, teaCache: teaCache, causalInference: causalInference,
          discretization: Denoiser.AYSLogLinearInterpolatedKarrasDiscretization(
            parameterization, objective: objective, samplingSigmas: samplingSigmas),
          weightsCache: weightsCache)
      } else {
        return EulerASampler<
          FloatType, UNetWrapper<FloatType>, Denoiser.AYSLogLinearInterpolatedTimestepDiscretization
        >(
          filePath: filePath, modifier: modifier, version: version, qkNorm: qkNorm,
          dualAttentionLayers: dualAttentionLayers,
          distilledGuidanceLayers: distilledGuidanceLayers,
          usesFlashAttention: usesFlashAttention,
          upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
          injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
          injectAttentionKV: injectAttentionKV,
          injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
          classifierFreeGuidance: isCfgEnabled, isGuidanceEmbedEnabled: isGuidanceEmbedEnabled,
          isQuantizedModel: isQuantizedModel,
          canRunLoRASeparately: canRunLoRASeparately,
          deviceProperties: deviceProperties, conditioning: conditioning,
          tiledDiffusion: tiledDiffusion, teaCache: teaCache, causalInference: causalInference,
          discretization: Denoiser.AYSLogLinearInterpolatedTimestepDiscretization(
            parameterization, objective: objective, samplingTimesteps: samplingTimesteps),
          weightsCache: weightsCache)
      }
    case .DDIM:
      return DDIMSampler<FloatType, UNetWrapper<FloatType>, Denoiser.LinearDiscretization>(
        filePath: filePath, modifier: modifier, version: version, qkNorm: qkNorm,
        dualAttentionLayers: dualAttentionLayers, distilledGuidanceLayers: distilledGuidanceLayers,
        usesFlashAttention: usesFlashAttention,
        upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
        injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
        injectAttentionKV: injectAttentionKV,
        injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
        classifierFreeGuidance: isCfgEnabled, isGuidanceEmbedEnabled: isGuidanceEmbedEnabled,
        isQuantizedModel: isQuantizedModel,
        canRunLoRASeparately: canRunLoRASeparately, deviceProperties: deviceProperties,
        conditioning: conditioning,
        tiledDiffusion: tiledDiffusion, teaCache: teaCache, causalInference: causalInference,
        discretization: Denoiser.LinearDiscretization(
          parameterization, objective: objective, timestepSpacing: .leading),
        weightsCache: weightsCache)
    case .dDIMTrailing:
      return DDIMSampler<FloatType, UNetWrapper<FloatType>, Denoiser.LinearDiscretization>(
        filePath: filePath, modifier: modifier, version: version, qkNorm: qkNorm,
        dualAttentionLayers: dualAttentionLayers, distilledGuidanceLayers: distilledGuidanceLayers,
        usesFlashAttention: usesFlashAttention,
        upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
        injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
        injectAttentionKV: injectAttentionKV,
        injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
        classifierFreeGuidance: isCfgEnabled, isGuidanceEmbedEnabled: isGuidanceEmbedEnabled,
        isQuantizedModel: isQuantizedModel,
        canRunLoRASeparately: canRunLoRASeparately, deviceProperties: deviceProperties,
        conditioning: conditioning,
        tiledDiffusion: tiledDiffusion, teaCache: teaCache, causalInference: causalInference,
        discretization: Denoiser.LinearDiscretization(
          parameterization, objective: objective, timestepSpacing: .trailing),
        weightsCache: weightsCache)
    case .PLMS:
      return PLMSSampler<FloatType, UNetWrapper<FloatType>, Denoiser.LinearDiscretization>(
        filePath: filePath, modifier: modifier, version: version, qkNorm: qkNorm,
        dualAttentionLayers: dualAttentionLayers, distilledGuidanceLayers: distilledGuidanceLayers,
        usesFlashAttention: usesFlashAttention,
        upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
        injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
        injectAttentionKV: injectAttentionKV,
        injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
        classifierFreeGuidance: isCfgEnabled, isGuidanceEmbedEnabled: isGuidanceEmbedEnabled,
        isQuantizedModel: isQuantizedModel,
        canRunLoRASeparately: canRunLoRASeparately, deviceProperties: deviceProperties,
        conditioning: conditioning,
        tiledDiffusion: tiledDiffusion, teaCache: teaCache, causalInference: causalInference,
        discretization: Denoiser.LinearDiscretization(
          parameterization, objective: objective, timestepSpacing: .leading),
        weightsCache: weightsCache)
    case .dPMPPSDEKarras:
      return DPMPPSDESampler<FloatType, UNetWrapper<FloatType>, Denoiser.KarrasDiscretization>(
        filePath: filePath, modifier: modifier, version: version, qkNorm: qkNorm,
        dualAttentionLayers: dualAttentionLayers, distilledGuidanceLayers: distilledGuidanceLayers,
        usesFlashAttention: usesFlashAttention,
        upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
        injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
        injectAttentionKV: injectAttentionKV,
        injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
        classifierFreeGuidance: isCfgEnabled, isGuidanceEmbedEnabled: isGuidanceEmbedEnabled,
        isQuantizedModel: isQuantizedModel,
        canRunLoRASeparately: canRunLoRASeparately, deviceProperties: deviceProperties,
        conditioning: conditioning,
        tiledDiffusion: tiledDiffusion, teaCache: teaCache, causalInference: causalInference,
        discretization: Denoiser.KarrasDiscretization(parameterization, objective: objective),
        weightsCache: weightsCache)
    case .dPMPPSDETrailing:
      return DPMPPSDESampler<FloatType, UNetWrapper<FloatType>, Denoiser.LinearDiscretization>(
        filePath: filePath, modifier: modifier, version: version, qkNorm: qkNorm,
        dualAttentionLayers: dualAttentionLayers, distilledGuidanceLayers: distilledGuidanceLayers,
        usesFlashAttention: usesFlashAttention,
        upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
        injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
        injectAttentionKV: injectAttentionKV,
        injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
        classifierFreeGuidance: isCfgEnabled, isGuidanceEmbedEnabled: isGuidanceEmbedEnabled,
        isQuantizedModel: isQuantizedModel,
        canRunLoRASeparately: canRunLoRASeparately, deviceProperties: deviceProperties,
        conditioning: conditioning,
        tiledDiffusion: tiledDiffusion, teaCache: teaCache, causalInference: causalInference,
        discretization: Denoiser.LinearDiscretization(
          parameterization, objective: objective, timestepSpacing: .trailing),
        weightsCache: weightsCache)
    case .DPMPPSDEAYS:
      if samplingTimesteps.isEmpty && samplingSigmas.isEmpty {
        return DPMPPSDESampler<
          FloatType, UNetWrapper<FloatType>, Denoiser.LinearDiscretization
        >(
          filePath: filePath, modifier: modifier, version: version, qkNorm: qkNorm,
          dualAttentionLayers: dualAttentionLayers,
          distilledGuidanceLayers: distilledGuidanceLayers,
          usesFlashAttention: usesFlashAttention,
          upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
          injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
          injectAttentionKV: injectAttentionKV,
          injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
          classifierFreeGuidance: isCfgEnabled, isGuidanceEmbedEnabled: isGuidanceEmbedEnabled,
          isQuantizedModel: isQuantizedModel,
          canRunLoRASeparately: canRunLoRASeparately,
          deviceProperties: deviceProperties, conditioning: conditioning,
          tiledDiffusion: tiledDiffusion, teaCache: teaCache, causalInference: causalInference,
          discretization: Denoiser.LinearDiscretization(
            parameterization, objective: objective, timestepSpacing: .trailing),
          weightsCache: weightsCache)
      } else if samplingTimesteps.isEmpty {
        return DPMPPSDESampler<
          FloatType, UNetWrapper<FloatType>, Denoiser.AYSLogLinearInterpolatedKarrasDiscretization
        >(
          filePath: filePath, modifier: modifier, version: version, qkNorm: qkNorm,
          dualAttentionLayers: dualAttentionLayers,
          distilledGuidanceLayers: distilledGuidanceLayers,
          usesFlashAttention: usesFlashAttention,
          upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
          injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
          injectAttentionKV: injectAttentionKV,
          injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
          classifierFreeGuidance: isCfgEnabled, isGuidanceEmbedEnabled: isGuidanceEmbedEnabled,
          isQuantizedModel: isQuantizedModel,
          canRunLoRASeparately: canRunLoRASeparately,
          deviceProperties: deviceProperties, conditioning: conditioning,
          tiledDiffusion: tiledDiffusion, teaCache: teaCache, causalInference: causalInference,
          discretization: Denoiser.AYSLogLinearInterpolatedKarrasDiscretization(
            parameterization, objective: objective, samplingSigmas: samplingSigmas),
          weightsCache: weightsCache)
      } else {
        return DPMPPSDESampler<
          FloatType, UNetWrapper<FloatType>, Denoiser.AYSLogLinearInterpolatedTimestepDiscretization
        >(
          filePath: filePath, modifier: modifier, version: version, qkNorm: qkNorm,
          dualAttentionLayers: dualAttentionLayers,
          distilledGuidanceLayers: distilledGuidanceLayers,
          usesFlashAttention: usesFlashAttention,
          upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
          injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
          injectAttentionKV: injectAttentionKV,
          injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
          classifierFreeGuidance: isCfgEnabled, isGuidanceEmbedEnabled: isGuidanceEmbedEnabled,
          isQuantizedModel: isQuantizedModel,
          canRunLoRASeparately: canRunLoRASeparately,
          deviceProperties: deviceProperties, conditioning: conditioning,
          tiledDiffusion: tiledDiffusion, teaCache: teaCache, causalInference: causalInference,
          discretization: Denoiser.AYSLogLinearInterpolatedTimestepDiscretization(
            parameterization, objective: objective, samplingTimesteps: samplingTimesteps),
          weightsCache: weightsCache)
      }
    case .uniPC:
      return UniPCSampler<FloatType, UNetWrapper<FloatType>, Denoiser.LinearDiscretization>(
        filePath: filePath, modifier: modifier, version: version, qkNorm: qkNorm,
        dualAttentionLayers: dualAttentionLayers, distilledGuidanceLayers: distilledGuidanceLayers,
        usesFlashAttention: usesFlashAttention,
        upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
        injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
        injectAttentionKV: injectAttentionKV,
        injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
        classifierFreeGuidance: isCfgEnabled, isGuidanceEmbedEnabled: isGuidanceEmbedEnabled,
        isQuantizedModel: isQuantizedModel,
        canRunLoRASeparately: canRunLoRASeparately, deviceProperties: deviceProperties,
        conditioning: conditioning,
        tiledDiffusion: tiledDiffusion, teaCache: teaCache, causalInference: causalInference,
        discretization: Denoiser.LinearDiscretization(parameterization, objective: objective),
        weightsCache: weightsCache)
    case .LCM:
      return LCMSampler<FloatType, UNetWrapper<FloatType>, Denoiser.LinearDiscretization>(
        filePath: filePath, modifier: modifier, version: version, qkNorm: qkNorm,
        dualAttentionLayers: dualAttentionLayers, distilledGuidanceLayers: distilledGuidanceLayers,
        usesFlashAttention: usesFlashAttention,
        upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
        injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
        injectAttentionKV: injectAttentionKV,
        injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
        isGuidanceEmbedEnabled: isGuidanceEmbedEnabled, isQuantizedModel: isQuantizedModel,
        canRunLoRASeparately: canRunLoRASeparately, deviceProperties: deviceProperties,
        conditioning: conditioning, tiledDiffusion: tiledDiffusion, teaCache: teaCache,
        causalInference: causalInference,
        discretization: Denoiser.LinearDiscretization(parameterization, objective: objective),
        weightsCache: weightsCache)
    case .TCD:
      return TCDSampler<FloatType, UNetWrapper<FloatType>, Denoiser.LinearDiscretization>(
        filePath: filePath, modifier: modifier, version: version, qkNorm: qkNorm,
        dualAttentionLayers: dualAttentionLayers, distilledGuidanceLayers: distilledGuidanceLayers,
        usesFlashAttention: usesFlashAttention,
        upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
        injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
        injectAttentionKV: injectAttentionKV,
        injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
        isGuidanceEmbedEnabled: isGuidanceEmbedEnabled, isQuantizedModel: isQuantizedModel,
        canRunLoRASeparately: canRunLoRASeparately, deviceProperties: deviceProperties,
        stochasticSamplingGamma: stochasticSamplingGamma,
        conditioning: conditioning, tiledDiffusion: tiledDiffusion, teaCache: teaCache,
        causalInference: causalInference,
        discretization: Denoiser.LinearDiscretization(parameterization, objective: objective),
        weightsCache: weightsCache)
    case .eulerASubstep:
      return EulerASampler<FloatType, UNetWrapper<FloatType>, Denoiser.LinearManualDiscretization>(
        filePath: filePath, modifier: modifier, version: version, qkNorm: qkNorm,
        dualAttentionLayers: dualAttentionLayers, distilledGuidanceLayers: distilledGuidanceLayers,
        usesFlashAttention: usesFlashAttention,
        upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
        injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
        injectAttentionKV: injectAttentionKV,
        injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
        classifierFreeGuidance: false, isGuidanceEmbedEnabled: isGuidanceEmbedEnabled,
        isQuantizedModel: isQuantizedModel,
        canRunLoRASeparately: canRunLoRASeparately, deviceProperties: deviceProperties,
        conditioning: conditioning,
        tiledDiffusion: tiledDiffusion, teaCache: teaCache, causalInference: causalInference,
        discretization: Denoiser.LinearManualDiscretization(
          parameterization, objective: objective, manual: manualSubsteps),
        weightsCache: weightsCache)
    case .dPMPPSDESubstep:
      return DPMPPSDESampler<
        FloatType, UNetWrapper<FloatType>, Denoiser.LinearManualDiscretization
      >(
        filePath: filePath, modifier: modifier, version: version, qkNorm: qkNorm,
        dualAttentionLayers: dualAttentionLayers, distilledGuidanceLayers: distilledGuidanceLayers,
        usesFlashAttention: usesFlashAttention,
        upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
        injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
        injectAttentionKV: injectAttentionKV,
        injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
        classifierFreeGuidance: false, isGuidanceEmbedEnabled: isGuidanceEmbedEnabled,
        isQuantizedModel: isQuantizedModel,
        canRunLoRASeparately: canRunLoRASeparately, deviceProperties: deviceProperties,
        conditioning: conditioning,
        tiledDiffusion: tiledDiffusion, teaCache: teaCache, causalInference: causalInference,
        discretization: Denoiser.LinearManualDiscretization(
          parameterization, objective: objective, manual: manualSubsteps),
        weightsCache: weightsCache)
    }
  }
}

extension LocalImageGenerator {

  public func startGenerating() {
    modelPreloader.startGenerating()
  }
  public func stopGenerating() {
    modelPreloader.stopGenerating()
  }
  public func generate(
    _ image: Tensor<FloatType>?, scaleFactor: Int, mask: Tensor<UInt8>?,
    hints: [(ControlHintType, [(AnyTensor, Float)])],
    text: String, negativeText: String, configuration: GenerationConfiguration,
    fileMapping: [String: String], keywords: [String], cancellation: (@escaping () -> Void) -> Void,
    feedback: @escaping (ImageGeneratorSignpost, Set<ImageGeneratorSignpost>, Tensor<FloatType>?) ->
      Bool
  ) -> ([Tensor<FloatType>]?, Int) {

    let depth =
      (hints.first {
        $0.0 == .depth
      })?.1.first?.0 as? Tensor<FloatType>

    let custom: Tensor<FloatType>? =
      (hints.first {
        $0.0 == .custom
      })?.1.first?.0 as? Tensor<FloatType>

    let shuffles: [(Tensor<FloatType>, Float)] =
      hints.first(where: { $0.0 == .shuffle })?.1 as? [(Tensor<FloatType>, Float)] ?? []

    let poses: [(Tensor<FloatType>, Float)] =
      hints.first(where: { $0.0 == .pose })?.1 as? [(Tensor<FloatType>, Float)] ?? []

    let hints: [ControlHintType: AnyTensor] = hints.reduce(into: [:]) { dict, hint in
      if hint.0 != .depth, hint.0 != .custom, hint.0 != .shuffle, hint.0 != .pose,
        let tensor = hint.1.first?.0
      {
        dict[hint.0] = tensor
      }
    }

    let file =
      (configuration.model.flatMap {
        ModelZoo.isModelDownloaded($0) ? $0 : nil
      }) ?? ModelZoo.defaultSpecification.file
    let denoiserParameterization: Denoiser.Parameterization
    switch ModelZoo.noiseDiscretizationForModel(file) {
    case .edm(let edm):
      denoiserParameterization = .edm(edm)
    case .ddpm(let ddpm):
      denoiserParameterization = .ddpm(ddpm)
    case .rf(let rf):
      denoiserParameterization = .rf(rf)
    }
    let shift: Double
    if ModelZoo.isResolutionDependentShiftAvailable(
      ModelZoo.versionForModel(file), isConsistencyModel: ModelZoo.isConsistencyModelForModel(file)),
      configuration.resolutionDependentShift
    {
      let tiledWidth =
        configuration.tiledDiffusion
        ? min(configuration.diffusionTileWidth, configuration.startWidth) : configuration.startWidth
      let tiledHeight =
        configuration.tiledDiffusion
        ? min(configuration.diffusionTileHeight, configuration.startHeight)
        : configuration.startHeight
      shift = ModelZoo.shiftFor((width: tiledWidth, height: tiledHeight))
    } else {
      shift = Double(configuration.shift)
    }
    let sampling = Sampling(steps: Int(configuration.steps), shift: shift)
    guard let image = image else {
      return generateTextOnly(
        nil, scaleFactor: scaleFactor, depth: depth, hints: hints, custom: custom,
        shuffles: shuffles, poses: poses,
        text: text, negativeText: negativeText, configuration: configuration,
        denoiserParameterization: denoiserParameterization, sampling: sampling,
        cancellation: cancellation, feedback: feedback)
    }
    guard let mask = mask else {
      return generateImageOnly(
        image, scaleFactor: scaleFactor, depth: depth, hints: hints, custom: custom,
        shuffles: shuffles, poses: poses, text: text,
        negativeText: negativeText, configuration: configuration,
        denoiserParameterization: denoiserParameterization, sampling: sampling,
        cancellation: cancellation, feedback: feedback)
    }
    return generateImageWithMask(
      image, scaleFactor: scaleFactor, mask: mask, depth: depth, hints: hints, custom: custom,
      shuffles: shuffles, poses: poses,
      text: text, negativeText: negativeText, configuration: configuration,
      denoiserParameterization: denoiserParameterization, sampling: sampling,
      cancellation: cancellation, feedback: feedback)
  }
}

extension LocalImageGenerator {
  private func kandinskyTokenize(
    graph: DynamicGraph, text: String, negativeText: String, negativePromptForImagePrior: Bool
  ) -> (
    [DynamicGraph.Tensor<Int32>], [DynamicGraph.Tensor<Int32>], [Int], [Int]
  ) {
    let (_, tokens, _, _, _) = tokenizerKandinsky.tokenize(
      text: text, truncation: true, maxLength: 77)
    let (_, unconditionalTokens, _, _, _) = tokenizerKandinsky.tokenize(
      text: negativePromptForImagePrior ? "" : negativeText, truncation: true, maxLength: 77)
    let (_, CLIPTokens, _, _, lengthsOfCond) = tokenizerV1.tokenize(
      text: text, truncation: true, maxLength: 77, paddingToken: 0)
    let (_, unconditionalCLIPTokens, _, _, lengthsOfUncond) = tokenizerV1.tokenize(
      text: negativeText, truncation: true, maxLength: 77, paddingToken: 0)
    let (_, zeroCLIPTokens, _, _, _) = tokenizerV1.tokenize(
      text: "", truncation: true, maxLength: 77, paddingToken: 0)
    let tokensTensor = graph.variable(.CPU, format: .NHWC, shape: [2 * 77], of: Int32.self)
    let positionTensor = graph.variable(.CPU, format: .NHWC, shape: [2 * 77], of: Int32.self)
    let CLIPTokensTensor = graph.variable(.CPU, format: .NHWC, shape: [3 * 77], of: Int32.self)
    let CLIPPositionTensor = graph.variable(.CPU, format: .NHWC, shape: [3 * 77], of: Int32.self)
    var unconditionalTokenLength: Int? = nil
    var tokenLength: Int? = nil
    for i in 0..<77 {
      tokensTensor[i] = unconditionalTokens[i]
      tokensTensor[i + 77] = tokens[i]
      positionTensor[i] = Int32(i + 2)
      positionTensor[i + 77] = Int32(i + 2)
      if unconditionalTokens[i] == 2 && unconditionalTokenLength == nil {
        unconditionalTokenLength = i + 1
      }
      if tokens[i] == 2 && tokenLength == nil {
        tokenLength = i + 1
      }
      CLIPTokensTensor[i] = unconditionalCLIPTokens[i]
      CLIPTokensTensor[i + 77] = CLIPTokens[i]
      CLIPTokensTensor[i + 77 * 2] = zeroCLIPTokens[i]
      CLIPPositionTensor[i] = Int32(i)
      CLIPPositionTensor[i + 77] = Int32(i)
      CLIPPositionTensor[i + 77 * 2] = Int32(i)
    }
    if let unconditionalTokenLength = unconditionalTokenLength {
      for i in unconditionalTokenLength..<77 {
        positionTensor[i] = 1
      }
    }
    if let tokenLength = tokenLength {
      for i in tokenLength..<77 {
        positionTensor[i + 77] = 1
      }
    }
    return (
      [tokensTensor, CLIPTokensTensor], [positionTensor, CLIPPositionTensor], lengthsOfUncond,
      lengthsOfCond
    )
  }
  private func tokenize(
    graph: DynamicGraph, tokenizer: Tokenizer & TextualInversionPoweredTokenizer,
    text: String, negativeText: String, paddingToken: Int32?, conditionalLength: Int,
    modifier: TextualInversionZoo.Modifier, potentials: [String], startLength: Int = 1,
    endLength: Int = 1, maxLength: Int = 77, paddingLength: Int = 77, minPadding: Int = 0
  ) -> (
    //    tokensTensors, positionTensors, embedMask, injectedEmbeddings, unconditionalAttentionWeights,
    //    attentionWeights, hasNonOneWeights, tokenLengthUncond, tokenLengthCond, lengthsOfUncond,
    //    lengthsOfCond
    [DynamicGraph.Tensor<Int32>], [DynamicGraph.Tensor<Int32>], [DynamicGraph.Tensor<FloatType>],
    [DynamicGraph.Tensor<FloatType>], [Float], [Float], Bool, Int, Int, [Int], [Int]
  ) {
    let paddingLength = max(maxLength, paddingLength)
    var (_, unconditionalTokens, unconditionalAttentionWeights, _, lengthsOfUncond) =
      tokenizer.tokenize(
        text: negativeText, truncation: false, maxLength: paddingLength, paddingToken: paddingToken)
    var (_, tokens, attentionWeights, _, lengthsOfCond) = tokenizer.tokenize(
      text: text, truncation: false, maxLength: paddingLength, paddingToken: paddingToken)
    var unconditionalTokensCount = unconditionalTokens.count
    // If textual inversion is multivector, add the count.
    for token in unconditionalTokens {
      if tokenizer.isTextualInversion(token), let keyword = tokenizer.textualInversion(for: token),
        let file = TextualInversionZoo.modelFromKeyword(keyword, potentials: potentials)
      {
        let length = TextualInversionZoo.tokenLengthForModel(file)
        unconditionalTokensCount += length - 1
      }
    }
    if unconditionalTokensCount > unconditionalTokens.count {
      // Check if we have enough padding tokens to remove.
      let paddingToken = paddingToken ?? tokenizer.endToken
      let oldUnconditionalTokensCount = unconditionalTokensCount
      for token in unconditionalTokens.reversed() {
        if token != paddingToken {
          if oldUnconditionalTokensCount > unconditionalTokensCount
            && paddingToken == tokenizer.endToken
          {
            // If paddingToken is endToken, we might removed the endToken in this process,
            // This is to recognize that situation and add back the endToken.
            unconditionalTokensCount += 1
          }
          break
        }
        unconditionalTokensCount -= 1
        if unconditionalTokensCount == unconditionalTokens.count {
          break
        }
      }
    }
    var tokensCount = tokens.count
    // If textual inversion is multivector, add the count.
    for token in tokens {
      if tokenizer.isTextualInversion(token), let keyword = tokenizer.textualInversion(for: token),
        let file = TextualInversionZoo.modelFromKeyword(keyword, potentials: potentials)
      {
        let length = TextualInversionZoo.tokenLengthForModel(file)
        tokensCount += length - 1
      }
    }
    if tokensCount > tokens.count {
      // Check if we have enough padding tokens to remove.
      let paddingToken = paddingToken ?? tokenizer.endToken
      let oldTokensCount = tokensCount
      for token in tokens.reversed() {
        if token != paddingToken {
          if oldTokensCount > tokensCount && paddingToken == tokenizer.endToken {
            // If paddingToken is endToken, we might removed the endToken in this process,
            // This is to recognize that situation and add back the endToken.
            tokensCount += 1
          }
          break
        }
        tokensCount -= 1
        if tokensCount == tokens.count {
          break
        }
      }
    }
    let hasNonOneWeights =
      (attentionWeights.contains { $0 != 1 })
      || (unconditionalAttentionWeights.contains { $0 != 1 })
    var tokenLength = max(unconditionalTokensCount, tokensCount)
    if tokenLength + minPadding > paddingLength {
      tokenLength += minPadding
      unconditionalTokensCount += minPadding
      tokensCount += minPadding
      (_, unconditionalTokens, unconditionalAttentionWeights, _, lengthsOfUncond) =
        tokenizer.tokenize(
          text: negativeText, truncation: true, maxLength: tokenLength, paddingToken: paddingToken)
      (_, tokens, attentionWeights, _, lengthsOfCond) = tokenizer.tokenize(
        text: text, truncation: true, maxLength: tokenLength, paddingToken: paddingToken)
    }
    // shift the token around to include the textual inversion, also record which textual inversion at which range.
    var unconditionalTextualInversionRanges = [(String, Range<Int>)]()
    var newUnconditionalTokens = [Int32]()
    var newUnconditionalWeights = [Float]()
    var j = 0
    var prefixLength = 0
    for (i, token) in unconditionalTokens.enumerated() {
      if j < lengthsOfUncond.count && i - 1 >= lengthsOfUncond[j] + prefixLength {
        prefixLength += lengthsOfUncond[j]
        j += 1
      }
      if tokenizer.isTextualInversion(token) {
        if let keyword = tokenizer.textualInversion(for: token),
          let file = TextualInversionZoo.modelFromKeyword(keyword, potentials: potentials)
        {
          let tokenLength = TextualInversionZoo.tokenLengthForModel(file)
          unconditionalTextualInversionRanges.append(
            (file, newUnconditionalTokens.count..<(newUnconditionalTokens.count + tokenLength)))
          for _ in 0..<tokenLength {
            newUnconditionalTokens.append(tokenizer.unknownToken)
            newUnconditionalWeights.append(unconditionalAttentionWeights[i])
          }
          lengthsOfUncond[j] += tokenLength - 1
        } else {
          newUnconditionalTokens.append(tokenizer.unknownToken)
          newUnconditionalWeights.append(unconditionalAttentionWeights[i])
        }
      } else {
        newUnconditionalTokens.append(token)
        newUnconditionalWeights.append(unconditionalAttentionWeights[i])
      }
    }
    if newUnconditionalTokens.count > unconditionalTokens.count {
      newUnconditionalTokens.removeLast(newUnconditionalTokens.count - unconditionalTokens.count)
      newUnconditionalWeights.removeLast(
        newUnconditionalWeights.count - unconditionalAttentionWeights.count)
    }
    let totalLengthOfUncond = lengthsOfUncond.reduce(0, +)
    if totalLengthOfUncond + endLength + startLength > tokenLength {
      lengthsOfUncond[lengthsOfUncond.count - 1] -=
        totalLengthOfUncond + endLength + startLength - tokenLength
    }
    unconditionalTokens = newUnconditionalTokens
    unconditionalAttentionWeights = newUnconditionalWeights
    var textualInversionRanges = [(String, Range<Int>)]()
    var newTokens = [Int32]()
    var newWeights = [Float]()
    j = 0
    prefixLength = 0
    for (i, token) in tokens.enumerated() {
      if j < lengthsOfCond.count && i - 1 >= lengthsOfCond[j] + prefixLength {
        prefixLength += lengthsOfCond[j]
        j += 1
      }
      if tokenizer.isTextualInversion(token) {
        if let keyword = tokenizer.textualInversion(for: token),
          let file = TextualInversionZoo.modelFromKeyword(keyword, potentials: potentials)
        {
          let tokenLength = TextualInversionZoo.tokenLengthForModel(file)
          textualInversionRanges.append((file, newTokens.count..<(newTokens.count + tokenLength)))
          for _ in 0..<tokenLength {
            newTokens.append(tokenizer.unknownToken)
            newWeights.append(attentionWeights[i])
          }
          lengthsOfCond[j] += tokenLength - 1
        } else {
          newTokens.append(tokenizer.unknownToken)
          newWeights.append(attentionWeights[i])
        }
      } else {
        newTokens.append(token)
        newWeights.append(attentionWeights[i])
      }
    }
    if newTokens.count > tokens.count {
      newTokens.removeLast(newTokens.count - tokens.count)
      newUnconditionalWeights.removeLast(newWeights.count - attentionWeights.count)
    }
    let totalLengthOfCond = lengthsOfCond.reduce(0, +)
    if totalLengthOfCond + endLength + startLength > tokenLength {
      lengthsOfCond[lengthsOfCond.count - 1] -=
        totalLengthOfCond + endLength + startLength - tokenLength
    }
    tokens = newTokens
    attentionWeights = newWeights
    let tokensTensor = graph.variable(.CPU, format: .NHWC, shape: [2 * tokenLength], of: Int32.self)
    let positionTensor = graph.variable(
      .CPU, format: .NHWC, shape: [2 * tokenLength], of: Int32.self)
    for i in 0..<tokenLength {
      tokensTensor[i] = unconditionalTokens[i]
      tokensTensor[i + tokenLength] = tokens[i]
    }
    // position index computation is a bit more involved if there is line-break.
    if tokenLength > 0 {
      positionTensor[0] = 0
      positionTensor[tokenLength - 1] = Int32(maxLength - 1)
      positionTensor[tokenLength] = 0
      positionTensor[tokenLength * 2 - 1] = Int32(maxLength - 1)
    }
    // For everything else, we will go through lengths of each, and assigning accordingly.
    j = startLength
    var maxPosition = 0
    prefixLength = startLength
    for length in lengthsOfUncond {
      for i in 0..<length {
        var position =
          length <= (maxLength - 2)
          ? i + 1 : Int(((Float(i) + 0.5) * Float(maxLength - 2) / Float(length) + 0.5).rounded())
        position = min(max(position, 1), maxLength - 2)
        positionTensor[j] = Int32(position)
        j += 1
      }
      maxPosition = max(length, maxPosition)
      prefixLength += length
    }
    var tokenLengthUncond = unconditionalTokensCount
    // We shouldn't have anything to fill between maxPosition and tokenLength - 1 if we are longer than paddingLength.
    if prefixLength < tokenLength - 1 {
      if maxPosition + endLength + startLength > paddingLength {  // If it is paddingLength, we can go to later to find i
        tokenLengthUncond = prefixLength + 1
      }
      var position = maxPosition + startLength
      for i in prefixLength..<(tokenLength - 1) {
        positionTensor[i] = Int32(min(position, maxLength - 1))
        position += 1
        if position == paddingLength {
          tokenLengthUncond = i + 1
        }
      }
    }
    j = tokenLength + startLength
    maxPosition = 0
    prefixLength = startLength
    for length in lengthsOfCond {
      for i in 0..<length {
        var position =
          length <= (maxLength - 2)
          ? i + 1 : Int(((Float(i) + 0.5) * Float(maxLength - 2) / Float(length) + 0.5).rounded())
        position = min(max(position, 1), maxLength - 2)
        positionTensor[j] = Int32(position)
        j += 1
      }
      maxPosition = max(length, maxPosition)
      prefixLength += length
    }
    var tokenLengthCond = tokensCount
    // We shouldn't have anything to fill between maxPosition and tokenLength - 1 if we are longer than paddingLength.
    if prefixLength < tokenLength - 1 {
      if maxPosition + endLength + startLength > paddingLength {  // If it is paddingLength, we can go to later to find i
        tokenLengthCond = prefixLength + 1
      }
      var position = maxPosition + startLength
      for i in prefixLength..<(tokenLength - 1) {
        positionTensor[tokenLength + i] = Int32(min(position, maxLength - 1))
        position += 1
        if position == paddingLength {
          tokenLengthCond = i + 1
        }
      }
    }
    // Compute mask and embeddings.
    let embedMask: [DynamicGraph.Tensor<FloatType>]
    let injectedEmbeddings: [DynamicGraph.Tensor<FloatType>]
    if unconditionalTextualInversionRanges.count > 0 || textualInversionRanges.count > 0 {
      let mask = graph.variable(.CPU, .WC(2 * tokenLength, 1), of: FloatType.self)
      mask.full(1)
      var embeddings = graph.variable(
        .CPU, .WC(2 * tokenLength, conditionalLength), of: FloatType.self)
      embeddings.full(0)
      for (file, range) in unconditionalTextualInversionRanges {
        mask[range, 0..<1].full(0)
        if let embedding = TextualInversionZoo.embeddingForModel(
          file, graph: graph, modifier: modifier, of: FloatType.self)
        {
          embeddings[range, 0..<conditionalLength] = graph.constant(embedding)
        }
      }
      for (file, range) in textualInversionRanges {
        let shiftedRange = (tokenLength + range.lowerBound)..<(tokenLength + range.upperBound)
        mask[shiftedRange, 0..<1].full(0)
        if let embedding = TextualInversionZoo.embeddingForModel(
          file, graph: graph, modifier: modifier, of: FloatType.self)
        {
          embeddings[shiftedRange, 0..<conditionalLength] = graph.constant(embedding)
        }
      }
      embedMask = [mask]
      injectedEmbeddings = [embeddings]
    } else {
      embedMask = []
      injectedEmbeddings = []
    }
    precondition(tokenLength == tokenLengthUncond || tokenLength == tokenLengthCond)
    return (
      [tokensTensor], [positionTensor], embedMask, injectedEmbeddings,
      unconditionalAttentionWeights, attentionWeights, hasNonOneWeights, tokenLengthUncond,
      tokenLengthCond, lengthsOfUncond, lengthsOfCond
    )
  }

  private func tokenize(
    graph: DynamicGraph, modelVersion: ModelVersion, textEncoderVersion: TextEncoderVersion?,
    paddedTextEncodingLength: Int, text: String, negativeText: String,
    negativePromptForImagePrior: Bool, potentials: [String], T5TextEncoder: Bool, clipL: String?,
    openClipG: String?, t5: String?
  ) -> (
    //  return:
    //  tokensTensors, positionTensors, embedMask, injectedEmbeddings, unconditionalAttentionWeights,
    //  attentionWeights, hasNonOneWeights, tokenLengthUncond, tokenLengthCond, lengthsOfUncond,
    //  lengthsOfCond
    [DynamicGraph.Tensor<Int32>], [DynamicGraph.Tensor<Int32>], [DynamicGraph.Tensor<FloatType>],
    [DynamicGraph.Tensor<FloatType>], [Float], [Float], Bool, Int, Int, [Int], [Int]
  ) {
    switch modelVersion {
    case .v1:
      return tokenize(
        graph: graph, tokenizer: tokenizerV1, text: text, negativeText: negativeText,
        paddingToken: nil, conditionalLength: 768, modifier: .clipL, potentials: potentials)
    case .v2, .svdI2v:
      return tokenize(
        graph: graph, tokenizer: tokenizerV2, text: text, negativeText: negativeText,
        paddingToken: 0, conditionalLength: 1024, modifier: .clipL, potentials: potentials)
    case .kandinsky21:
      let (tokenTensors, positionTensors, lengthsOfUncond, lengthsOfCond) = kandinskyTokenize(
        graph: graph, text: text, negativeText: negativeText,
        negativePromptForImagePrior: negativePromptForImagePrior)
      return (
        tokenTensors, positionTensors, [], [], [Float](repeating: 1, count: 77),
        [Float](repeating: 1, count: 77), false, 77, 77, lengthsOfUncond, lengthsOfCond
      )
    case .hunyuanVideo:
      let tokenizerV1 = tokenizerV1
      var result = tokenize(
        graph: graph, tokenizer: tokenizerV1, text: clipL ?? text, negativeText: negativeText,
        paddingToken: nil, conditionalLength: 768, modifier: .clipL, potentials: potentials)
      assert(result.7 >= 77 && result.8 >= 77)
      let promptWithTemplate =
        "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: 1. The main content and theme of the video.2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects.3. Actions, events, behaviors temporal relationships, physical movement changes of the objects.4. background environment, light, style and atmosphere.5. camera angles, movements, and transitions used in the video:<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n\(text)<|eot_id|>"
      let negativePromptWithTemplate =
        "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: 1. The main content and theme of the video.2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects.3. Actions, events, behaviors temporal relationships, physical movement changes of the objects.4. background environment, light, style and atmosphere.5. camera angles, movements, and transitions used in the video:<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n\(negativeText)<|eot_id|>"
      let (
        llama3Tokens, _, _, _, _, _, _, tokenLengthsUncond, tokenLengthsCond, _, _
      ) = tokenize(
        graph: graph, tokenizer: tokenizerLlama3, text: promptWithTemplate,
        negativeText: negativePromptWithTemplate,
        paddingToken: nil, conditionalLength: 4096, modifier: .llama3, potentials: potentials,
        startLength: 0, endLength: 0, maxLength: 0, paddingLength: 0)
      result.0 = llama3Tokens + result.0
      result.7 = tokenLengthsUncond - 95  // Remove the leading template.
      result.8 = tokenLengthsCond - 95
      return result
    case .wan21_14b, .wan21_1_3b:
      return tokenize(
        graph: graph, tokenizer: tokenizerUMT5, text: text, negativeText: negativeText,
        paddingToken: 0, conditionalLength: 4096, modifier: .t5xxl, potentials: potentials,
        startLength: 0, maxLength: 0, paddingLength: 0)
    case .hiDreamI1:
      var tokenizerV1 = tokenizerV1
      tokenizerV1.textualInversions = []
      var result = tokenize(
        graph: graph, tokenizer: tokenizerV1, text: clipL ?? text, negativeText: negativeText,
        paddingToken: nil, conditionalLength: 768, modifier: .clipL, potentials: potentials,
        maxLength: 248, paddingLength: 248)
      var tokenizerV2 = tokenizerXL
      tokenizerV2.textualInversions = []
      let (
        tokens, positions, embedMask, injectedEmbeddings, _, _, _, _, _, _, _
      ) = tokenize(
        graph: graph, tokenizer: tokenizerV2, text: openClipG ?? text, negativeText: negativeText,
        paddingToken: 0, conditionalLength: 1280, modifier: .clipG, potentials: potentials,
        maxLength: 218, paddingLength: 218)
      result.0 = result.0 + tokens
      result.1 = result.1 + positions
      result.2 = result.2 + embedMask
      result.3 = result.3 + injectedEmbeddings
      let (
        t5Tokens, _, t5EmbedMask, t5InjectedEmbeddings, _, _, _, _, _,
        t5LengthsOfUncond, t5LengthsOfCond
      ) = tokenize(
        graph: graph, tokenizer: tokenizerT5, text: t5 ?? text, negativeText: negativeText,
        paddingToken: nil, conditionalLength: 4096, modifier: .t5xxl, potentials: potentials,
        maxLength: paddedTextEncodingLength, paddingLength: paddedTextEncodingLength)
      result.0 = result.0 + t5Tokens
      result.2 = result.2 + t5EmbedMask
      result.3 = result.3 + t5InjectedEmbeddings
      let (
        llama3Tokens, _, _, _, _, _, _, tokenLengthsUncond, tokenLengthsCond, llama3LengthsOfUncond,
        llama3LengthsOfCond
      ) = tokenize(
        graph: graph, tokenizer: tokenizerLlama3, text: text, negativeText: negativeText,
        paddingToken: 128009, conditionalLength: 4096, modifier: .llama3, potentials: potentials,
        startLength: 0, endLength: 0, maxLength: paddedTextEncodingLength,
        paddingLength: paddedTextEncodingLength)
      result.0 = result.0 + llama3Tokens
      result.7 = tokenLengthsUncond
      result.8 = tokenLengthsCond
      result.9 = [t5LengthsOfUncond[0] + 1, llama3LengthsOfUncond[0]]
      result.10 = [t5LengthsOfCond[0] + 1, llama3LengthsOfCond[0]]
      return result
    case .wurstchenStageC, .wurstchenStageB:
      // The difference between this and SDXL: paddingToken is no long '!' (indexed by 0) but unknown.
      return tokenize(
        graph: graph, tokenizer: tokenizerXL, text: text, negativeText: negativeText,
        paddingToken: nil, conditionalLength: 1280, modifier: .clipG, potentials: potentials)
    case .sdxlBase, .sdxlRefiner, .ssd1b:
      switch textEncoderVersion {
      case .chatglm3_6b:
        var result = tokenize(
          graph: graph, tokenizer: tokenizerChatGLM3, text: text, negativeText: negativeText,
          paddingToken: nil, conditionalLength: 4096, modifier: .chatglm3_6b,
          potentials: potentials, startLength: 0, endLength: 0, maxLength: 0, paddingLength: 0)
        result.7 = max(256, result.7 + 2)
        result.8 = max(256, result.8 + 2)
        return result
      case nil:
        let tokenizerV2 = tokenizerXL
        var tokenizerV1 = tokenizerV1
        tokenizerV1.textualInversions = tokenizerV2.textualInversions
        var result = tokenize(
          graph: graph, tokenizer: tokenizerV2, text: text, negativeText: negativeText,
          paddingToken: 0, conditionalLength: 1280, modifier: .clipG, potentials: potentials)
        let (tokens, _, embedMask, injectedEmbeddings, _, _, _, _, _, _, _) = tokenize(
          graph: graph, tokenizer: tokenizerV1, text: text, negativeText: negativeText,
          paddingToken: nil, conditionalLength: 768, modifier: .clipL, potentials: potentials)
        result.0 = tokens + result.0
        result.2 = embedMask + result.2
        result.3 = injectedEmbeddings + result.3
        return result
      }
    case .pixart:
      return tokenize(
        graph: graph, tokenizer: tokenizerT5, text: text, negativeText: negativeText,
        paddingToken: nil, conditionalLength: 4096, modifier: .t5xxl, potentials: potentials,
        startLength: 0, maxLength: 0, paddingLength: 0)
    case .auraflow:
      return tokenize(
        graph: graph, tokenizer: tokenizerPileT5, text: text, negativeText: negativeText,
        paddingToken: 1, conditionalLength: 2048, modifier: .pilet5xl, potentials: potentials,
        startLength: 0, maxLength: 0, paddingLength: 0)
    case .flux1:
      let tokenizerV1 = tokenizerV1
      var result = tokenize(
        graph: graph, tokenizer: tokenizerV1, text: clipL ?? text, negativeText: negativeText,
        paddingToken: nil, conditionalLength: 768, modifier: .clipL, potentials: potentials)
      assert(result.7 >= 77 && result.8 >= 77)
      let (
        t5Tokens, _, t5EmbedMask, t5InjectedEmbeddings, _, _, _, tokenLengthUncond, tokenLengthCond,
        _, _
      ) = tokenize(
        graph: graph, tokenizer: tokenizerT5, text: text, negativeText: negativeText,
        paddingToken: nil, conditionalLength: 4096, modifier: .t5xxl, potentials: potentials,
        maxLength: paddedTextEncodingLength, paddingLength: paddedTextEncodingLength,
        minPadding: paddedTextEncodingLength == 0 ? 1 : 0)
      result.0 = t5Tokens + result.0
      result.2 = t5EmbedMask + result.2
      result.3 = t5InjectedEmbeddings + result.3
      result.7 = tokenLengthUncond
      result.8 = tokenLengthCond
      return result
    case .sd3:
      let tokenizerV2 = tokenizerXL
      var tokenizerV1 = tokenizerV1
      tokenizerV1.textualInversions = tokenizerV2.textualInversions
      var result = tokenize(
        graph: graph, tokenizer: tokenizerV2, text: openClipG ?? text, negativeText: negativeText,
        paddingToken: 0, conditionalLength: 1280, modifier: .clipG, potentials: potentials)
      assert(result.7 >= 77 && result.8 >= 77)
      let (
        tokens, _, embedMask, injectedEmbeddings, _, _, _, tokenLengthUncond, tokenLengthCond, _, _
      ) = tokenize(
        graph: graph, tokenizer: tokenizerV1, text: clipL ?? text, negativeText: negativeText,
        paddingToken: nil, conditionalLength: 768, modifier: .clipL, potentials: potentials,
        paddingLength: max(result.7, result.8))
      result.0 = tokens + result.0
      result.2 = embedMask + result.2
      result.3 = injectedEmbeddings + result.3
      if max(result.7, result.8) < max(tokenLengthUncond, tokenLengthCond) {
        // We need to redo this for initial result from OpenCLIP G to make sure they are aligned.
        result = tokenize(
          graph: graph, tokenizer: tokenizerV2, text: openClipG ?? text, negativeText: negativeText,
          paddingToken: 0, conditionalLength: 1280, modifier: .clipG, potentials: potentials,
          paddingLength: max(tokenLengthUncond, tokenLengthCond))
        result.0 = tokens + result.0
        result.2 = embedMask + result.2
        result.3 = injectedEmbeddings + result.3
      }
      result.7 = tokenLengthUncond
      result.8 = tokenLengthCond
      if T5TextEncoder {
        let (t5Tokens, _, t5EmbedMask, t5InjectedEmbeddings, _, _, _, _, _, _, _) = tokenize(
          graph: graph, tokenizer: tokenizerT5, text: text, negativeText: negativeText,
          paddingToken: nil, conditionalLength: 4096, modifier: .t5xxl, potentials: potentials)
        result.0 = result.0 + t5Tokens
        result.2 = result.2 + t5EmbedMask
        result.3 = result.3 + t5InjectedEmbeddings
        // tokenLengthUncond / tokenLengthCond are used by causalAttentionMask, hence used by CLIP, not by T5. No need to update.
      }
      return result
    case .sd3Large:
      let tokenizerV2 = tokenizerXL
      var tokenizerV1 = tokenizerV1
      tokenizerV1.textualInversions = tokenizerV2.textualInversions
      var result = tokenize(
        graph: graph, tokenizer: tokenizerV2, text: openClipG ?? text, negativeText: negativeText,
        paddingToken: 0, conditionalLength: 1280, modifier: .clipG, potentials: potentials)
      assert(result.7 >= 77 && result.8 >= 77)
      let (
        tokens, _, embedMask, injectedEmbeddings, _, _, _, tokenLengthUncond, tokenLengthCond, _, _
      ) = tokenize(
        graph: graph, tokenizer: tokenizerV1, text: clipL ?? text, negativeText: negativeText,
        paddingToken: nil, conditionalLength: 768, modifier: .clipL, potentials: potentials,
        paddingLength: max(result.7, result.8))
      result.0 = tokens + result.0
      result.2 = embedMask + result.2
      result.3 = injectedEmbeddings + result.3
      if max(result.7, result.8) < max(tokenLengthUncond, tokenLengthCond) {
        // We need to redo this for initial result from OpenCLIP G to make sure they are aligned.
        result = tokenize(
          graph: graph, tokenizer: tokenizerV2, text: openClipG ?? text, negativeText: negativeText,
          paddingToken: 0, conditionalLength: 1280, modifier: .clipG, potentials: potentials,
          paddingLength: max(tokenLengthUncond, tokenLengthCond))
        result.0 = tokens + result.0
        result.2 = embedMask + result.2
        result.3 = injectedEmbeddings + result.3
      }
      result.7 = tokenLengthUncond
      result.8 = tokenLengthCond
      if T5TextEncoder {
        let (t5Tokens, _, t5EmbedMask, t5InjectedEmbeddings, _, _, _, _, _, _, _) = tokenize(
          graph: graph, tokenizer: tokenizerT5, text: text, negativeText: negativeText,
          paddingToken: nil, conditionalLength: 4096, modifier: .t5xxl, potentials: potentials,
          maxLength: 256)
        result.0 = result.0 + t5Tokens
        result.2 = result.2 + t5EmbedMask
        result.3 = result.3 + t5InjectedEmbeddings
        // tokenLengthUncond / tokenLengthCond are used by causalAttentionMask, hence used by CLIP, not by T5. No need to update.
      }
      return result
    }
  }

  private func upscaleImageAndToCPU(
    _ image: DynamicGraph.Tensor<FloatType>, configuration: GenerationConfiguration
  ) -> (Tensor<FloatType>, Int) {
    guard let upscaler = configuration.upscaler, UpscalerZoo.isModelDownloaded(upscaler) else {
      return (image.rawValue.toCPU(), 1)
    }
    let upscalerFilePath = UpscalerZoo.filePathForModelDownloaded(upscaler)
    let nativeScaleFactor = UpscalerZoo.scaleFactorForModel(upscaler)
    let forcedScaleFactor: UpscaleFactor
    switch configuration.upscalerScaleFactor {
    case 2:
      forcedScaleFactor = .x2
    case 4:
      forcedScaleFactor = .x4
    default:
      forcedScaleFactor = nativeScaleFactor
    }
    let numberOfBlocks = UpscalerZoo.numberOfBlocksForModel(upscaler)
    let realESRGANer = RealESRGANer<FloatType>(
      filePath: upscalerFilePath, nativeScaleFactor: nativeScaleFactor,
      forcedScaleFactor: forcedScaleFactor, numberOfBlocks: numberOfBlocks,
      isNHWCPreferred: DeviceCapability.isNHWCPreferred,
      tileSize: DeviceCapability.RealESRGANerTileSize)
    let shape = image.shape
    if shape[3] > 3 {
      let graph = image.graph
      let result = realESRGANer.upscale(
        image[0..<shape[0], 0..<shape[1], 0..<shape[2], (shape[3] - 3)..<shape[3]].copied()
      ).0
      let upscaledShape = result.shape
      var original = graph.variable(
        .GPU(0), .NHWC(shape[0], upscaledShape[1], upscaledShape[2], shape[3]), of: FloatType.self)
      original[
        0..<shape[0], 0..<upscaledShape[1], 0..<upscaledShape[2], (shape[3] - 3)..<shape[3]] =
        result.toGPU(
          0)
      // Retain the alpha channel.
      original[0..<shape[0], 0..<upscaledShape[1], 0..<upscaledShape[2], 0..<(shape[3] - 3)] =
        Upsample(
          .bilinear, widthScale: Float(upscaledShape[2]) / Float(shape[2]),
          heightScale: Float(upscaledShape[1]) / Float(shape[1]))(
          image[0..<shape[0], 0..<shape[1], 0..<shape[2], 0..<(shape[3] - 3)].copied())
      return (original.rawValue.toCPU().copied(), forcedScaleFactor.rawValue)
    } else {
      return (
        realESRGANer.upscale(image).0.rawValue.copied(),
        forcedScaleFactor.rawValue
      )
    }
  }

  private func downscaleImageAndToGPU(_ image: DynamicGraph.Tensor<FloatType>, scaleFactor: Int)
    -> DynamicGraph.Tensor<FloatType>
  {
    return RealESRGANer.downscale(image, scaleFactor: scaleFactor)
  }

  private func downscaleImage(_ image: Tensor<FloatType>, scaleFactor: Int)
    -> Tensor<FloatType>
  {
    guard scaleFactor > 1 else { return image }
    let graph = DynamicGraph()
    return RealESRGANer.downscale(graph.variable(image), scaleFactor: scaleFactor).rawValue.toCPU()
  }

  private func upscaleImages(_ images: [Tensor<FloatType>], configuration: GenerationConfiguration)
    -> ([Tensor<FloatType>], Int)
  {
    guard let upscaler = configuration.upscaler, UpscalerZoo.isModelDownloaded(upscaler) else {
      return (images, 1)
    }
    let upscalerFilePath = UpscalerZoo.filePathForModelDownloaded(upscaler)
    let nativeScaleFactor = UpscalerZoo.scaleFactorForModel(upscaler)
    let forcedScaleFactor: UpscaleFactor
    switch configuration.upscalerScaleFactor {
    case 2:
      forcedScaleFactor = .x2
    case 4:
      forcedScaleFactor = .x4
    default:
      forcedScaleFactor = nativeScaleFactor
    }
    let numberOfBlocks = UpscalerZoo.numberOfBlocksForModel(upscaler)
    let graph = DynamicGraph()
    return graph.withNoGrad {
      let realESRGANer = RealESRGANer<FloatType>(
        filePath: upscalerFilePath, nativeScaleFactor: nativeScaleFactor,
        forcedScaleFactor: forcedScaleFactor, numberOfBlocks: numberOfBlocks,
        isNHWCPreferred: DeviceCapability.isNHWCPreferred,
        tileSize: DeviceCapability.RealESRGANerTileSize)
      var rrdbnet: Model? = nil
      var results = [Tensor<FloatType>]()
      for image in images {
        let shape = image.shape
        if shape[3] > 3 {
          let image = graph.variable(image).toGPU()
          let (result, net) = realESRGANer.upscale(
            image[0..<shape[1], 0..<shape[2], (shape[3] - 3)..<shape[3]].copied(), rrdbnet: rrdbnet)
          let upscaledShape = result.shape
          var original = graph.variable(
            .GPU(0), .NHWC(shape[0], upscaledShape[1], upscaledShape[2], shape[3]),
            of: FloatType.self)
          original[
            0..<shape[0], 0..<upscaledShape[1], 0..<upscaledShape[2], (shape[3] - 3)..<shape[3]] =
            result.toGPU(
              0)
          // Retain the alpha channel.
          original[0..<shape[0], 0..<upscaledShape[1], 0..<upscaledShape[2], 0..<(shape[3] - 3)] =
            Upsample(
              .bilinear, widthScale: Float(upscaledShape[2]) / Float(shape[2]),
              heightScale: Float(upscaledShape[1]) / Float(shape[1]))(
              image[0..<shape[0], 0..<shape[1], 0..<shape[2], 0..<(shape[3] - 3)].copied())
          rrdbnet = net
          results.append(original.rawValue.toCPU())
        } else {
          let image = graph.variable(image).toGPU()
          let (result, net) = realESRGANer.upscale(image, rrdbnet: rrdbnet)
          rrdbnet = net
          results.append(result.rawValue.copied())
        }
      }
      return (results, forcedScaleFactor.rawValue)
    }
  }

  private func faceRestoreImage(
    _ image: DynamicGraph.Tensor<FloatType>, configuration: GenerationConfiguration
  ) -> DynamicGraph.Tensor<FloatType> {
    #if !os(Linux)

      guard let faceRestoration = configuration.faceRestoration,
        EverythingZoo.isModelDownloaded(faceRestoration)
          && EverythingZoo.isModelDownloaded(EverythingZoo.parsenetForModel(faceRestoration))
      else { return image }
      let parsenet = EverythingZoo.parsenetForModel(faceRestoration)
      let filePath = EverythingZoo.filePathForModelDownloaded(faceRestoration)
      let parseFilePath = EverythingZoo.filePathForModelDownloaded(parsenet)
      let faceRestorer = FaceRestorer<FloatType>(filePath: filePath, parseFilePath: parseFilePath)
      let shape = image.shape
      if shape[3] > 3 {
        let graph = image.graph
        var original = graph.variable(like: image)
        let (result, _, _, _) = faceRestorer.enhance(
          image[0..<shape[0], 0..<shape[1], 0..<shape[2], (shape[3] - 3)..<shape[3]].copied())
        // Copy back so we retain the other channels.
        original[0..<shape[0], 0..<shape[1], 0..<shape[2], (shape[3] - 3)..<shape[3]] = result
        original[0..<shape[0], 0..<shape[1], 0..<shape[2], 0..<(shape[3] - 3)] =
          image[0..<shape[0], 0..<shape[1], 0..<shape[2], 0..<(shape[3] - 3)]
        return original
      } else {
        let (result, _, _, _) = faceRestorer.enhance(image)
        return result
      }
    #else
      return image
    #endif
  }

  private func faceRestoreImages(
    _ images: [Tensor<FloatType>], configuration: GenerationConfiguration
  ) -> [Tensor<FloatType>] {
    #if !os(Linux)
      guard let faceRestoration = configuration.faceRestoration,
        EverythingZoo.isModelDownloaded(faceRestoration)
          && EverythingZoo.isModelDownloaded(EverythingZoo.parsenetForModel(faceRestoration))
      else { return images }
      let parsenet = EverythingZoo.parsenetForModel(faceRestoration)
      let filePath = EverythingZoo.filePathForModelDownloaded(faceRestoration)
      let parseFilePath = EverythingZoo.filePathForModelDownloaded(parsenet)
      let graph = DynamicGraph()
      return graph.withNoGrad {
        let faceRestorer = FaceRestorer<FloatType>(filePath: filePath, parseFilePath: parseFilePath)
        var restoreFormer: Model? = nil
        var embedding: DynamicGraph.Tensor<FloatType>? = nil
        var parsenet: Model? = nil
        var results = [Tensor<FloatType>]()
        for image in images {
          let shape = image.shape
          if shape[3] > 3 {
            var original = graph.variable(image).toGPU(0)
            let (result, net1, embedding1, net2) = faceRestorer.enhance(
              original[0..<shape[0], 0..<shape[1], 0..<shape[2], (shape[3] - 3)..<shape[3]]
                .copied(),
              restoreFormer: restoreFormer, embedding: embedding, parsenet: parsenet)
            restoreFormer = net1
            embedding = embedding1
            parsenet = net2
            // Copy back so we retain the other channels.
            original[0..<shape[0], 0..<shape[1], 0..<shape[2], (shape[3] - 3)..<shape[3]] = result
            results.append(original.toCPU().rawValue.copied())
          } else {
            let (result, net1, embedding1, net2) = faceRestorer.enhance(
              graph.variable(image).toGPU(),
              restoreFormer: restoreFormer, embedding: embedding, parsenet: parsenet)
            restoreFormer = net1
            embedding = embedding1
            parsenet = net2
            results.append(result.toCPU().rawValue.copied())
          }
        }
        return results
      }
    #else
      return images
    #endif
  }

  @inline(__always)
  private func copy(
    _ noise: [Float], inputOffset: Int, output: inout [Float], outputOffset: Int, startWidth: Int,
    channels: Int
  ) {
    for y in 0..<8 {
      for x in 0..<8 {
        for c in 0..<channels {
          output[outputOffset + y * startWidth * channels + x * channels + c] =
            noise[inputOffset + y * 8 * channels + x * channels + c]
        }
      }
    }
  }

  private func nestSquares(_ noise: [Float], startWidth: Int, startHeight: Int, channels: Int)
    -> [Float]
  {
    guard startWidth % 8 == 0 && startHeight % 8 == 0 else { return noise }
    // Treat them as a 8x8x4 block, and layout them from center.
    let width = startWidth / 8
    let height = startHeight / 8
    let minSize = (min(width, height) / 2) * 2
    guard minSize >= 2 else { return noise }
    let maxSize = ((max(width, height) + 1) / 2) * 2
    var output = [Float](repeating: 0, count: startWidth * startHeight * channels)
    var inputOffset = 0
    for k in 0..<(maxSize / 2) {  // At the center, it is a 4x4.
      let length = (k + 1) * 2
      let kx = (width + 1) / 2 - k - 1
      let ky = (height + 1) / 2 - k - 1
      let lr = ky * startWidth * 8 * channels + kx * 8 * channels
      if ky >= 0 && ky < height {
        for i in 0..<(length - 1) {
          guard i + kx >= 0 && i + kx < width else { continue }
          let outputOffset = lr + i * 8 * channels
          copy(
            noise, inputOffset: inputOffset, output: &output, outputOffset: outputOffset,
            startWidth: startWidth, channels: channels)
          inputOffset += 8 * 8 * channels
        }
      }
      if length - 1 + kx >= 0 && length - 1 + kx < width {
        for i in 0..<(length - 1) {
          guard i + ky >= 0 && i + ky < height else { continue }
          let outputOffset = lr + (length - 1) * 8 * channels + i * startWidth * 8 * channels
          copy(
            noise, inputOffset: inputOffset, output: &output, outputOffset: outputOffset,
            startWidth: startWidth, channels: channels)
          inputOffset += 8 * 8 * channels
        }
      }
      if length - 1 + ky >= 0 && length - 1 + ky < height {
        for i in 0..<(length - 1) {
          guard length - 1 - i + kx >= 0 && length - 1 - i + kx < width else { continue }
          let outputOffset =
            lr + (length - 1) * startWidth * 8 * channels + (length - 1 - i) * 8 * channels
          copy(
            noise, inputOffset: inputOffset, output: &output, outputOffset: outputOffset,
            startWidth: startWidth, channels: channels)
          inputOffset += 8 * 8 * channels
        }
      }
      if kx >= 0 && kx < width {
        for i in 0..<(length - 1) {
          guard length - 1 - i + ky >= 0 && length - 1 - i + ky < height else { continue }
          let outputOffset = lr + (length - 1 - i) * startWidth * 8 * channels
          copy(
            noise, inputOffset: inputOffset, output: &output, outputOffset: outputOffset,
            startWidth: startWidth, channels: channels)
          inputOffset += 8 * 8 * channels
        }
      }
    }
    return output
  }

  private func randomLatentNoise(
    graph: DynamicGraph, batchSize: Int, startHeight: Int, startWidth: Int, channels: Int,
    seed: UInt32,
    seedMode: SeedMode
  ) -> DynamicGraph.Tensor<FloatType> {
    let x_T: DynamicGraph.Tensor<FloatType>
    switch seedMode {
    case .legacy:
      x_T = graph.variable(
        .GPU(0), .NHWC(batchSize, startHeight, startWidth, channels), of: FloatType.self)
      x_T.randn(std: 1, mean: 0)
    case .torchCpuCompatible:
      var torchRandomSource = TorchRandomSource(seed: seed)
      let noise = torchRandomSource.normalArray(
        count: batchSize * startHeight * startWidth * channels)
      let noiseTensor = Tensor<FloatType>(
        from: Tensor<Float>(noise, .CPU, .NHWC(batchSize, channels, startHeight, startWidth))
          .permuted(
            0, 2, 3, 1
          ).copied())
      x_T = graph.variable(noiseTensor.toGPU(0))
    case .scaleAlike:
      var noiseTensor = Tensor<FloatType>(.CPU, .NHWC(batchSize, startHeight, startWidth, channels))
      var seeds = [UInt32]()
      seeds.append(seed)
      if batchSize > 1 {
        var seed = seed
        for _ in 1..<batchSize {
          seed = xorshift(seed)
          seeds.append(seed)
        }
      }
      for (i, seed) in seeds.enumerated() {
        var torchRandomSource = TorchRandomSource(seed: seed)
        let noise = torchRandomSource.normalArray(count: startHeight * startWidth * channels)
        let nestSquaresNoise = nestSquares(
          noise, startWidth: startWidth, startHeight: startHeight, channels: channels)
        let noiseSlice = Tensor<FloatType>(
          from: Tensor<Float>(nestSquaresNoise, .CPU, .NHWC(1, startHeight, startWidth, channels)))
        noiseTensor[i..<(i + 1), 0..<startHeight, 0..<startWidth, 0..<channels] = noiseSlice
      }
      x_T = graph.variable(noiseTensor.toGPU(0))
    case .nvidiaGpuCompatible:
      var nvRandomSource = NVRandomSource(seed: UInt64(seed))
      let noise = nvRandomSource.normalArray(count: batchSize * startHeight * startWidth * channels)
      let noiseTensor = Tensor<FloatType>(
        from: Tensor<Float>(noise, .CPU, .NHWC(batchSize, channels, startHeight, startWidth))
          .permuted(
            0, 2, 3, 1
          ).copied())
      x_T = graph.variable(noiseTensor.toGPU(0))
    }
    return x_T
  }

  private func ipAdapterRGB(
    shuffles: [(Tensor<FloatType>, Float)], imageEncoderVersion: ImageEncoderVersion,
    graph: DynamicGraph
  ) -> [(
    DynamicGraph.Tensor<FloatType>, Float
  )] {
    var rgbResults = [(DynamicGraph.Tensor<FloatType>, Float)]()
    // IP-Adapter requires image to be normalized to the format CLIP model requires.
    let mean = graph.variable(
      Tensor<FloatType>(
        [
          FloatType(2 * 0.48145466 - 1), FloatType(2 * 0.4578275 - 1),
          FloatType(2 * 0.40821073 - 1),
        ], .GPU(0), .NHWC(1, 1, 1, 3)))
    let invStd = graph.variable(
      Tensor<FloatType>(
        [
          FloatType(0.5 / 0.26862954), FloatType(0.5 / 0.26130258), FloatType(0.5 / 0.27577711),
        ],
        .GPU(0), .NHWC(1, 1, 1, 3)))
    for (shuffle, strength) in shuffles {
      let input = graph.variable(Tensor<FloatType>(shuffle).toGPU(0))
      let inputHeight = input.shape[1]
      let inputWidth = input.shape[2]
      precondition(input.shape[3] == 3)
      let imageSize: Int
      switch imageEncoderVersion {
      case .siglipL27_384:
        imageSize = 378
      case .siglip2L27_512:
        imageSize = 512
      case .clipL14_336, .eva02L14_336:
        imageSize = 336
      case .openClipH14:
        imageSize = 224
      }
      switch imageEncoderVersion {
      case .siglipL27_384, .siglip2L27_512:
        // siglip normalizes with simply 0.5, 0.5, hence in the range of -1, 1.
        if inputHeight != imageSize || inputWidth != imageSize {
          rgbResults.append(
            (
              Upsample(
                .bilinear, widthScale: Float(imageSize) / Float(inputWidth),
                heightScale: Float(imageSize) / Float(inputHeight))(input),
              strength
            ))
        } else {
          rgbResults.append((input, strength))
        }
      case .clipL14_336, .eva02L14_336, .openClipH14:
        if inputHeight != imageSize || inputWidth != imageSize {
          rgbResults.append(
            (
              (Upsample(
                .bilinear, widthScale: Float(imageSize) / Float(inputWidth),
                heightScale: Float(imageSize) / Float(inputHeight))(input) - mean) .* invStd,
              strength
            ))
        } else {
          rgbResults.append(((input - mean) .* invStd, strength))
        }
      }
    }
    return rgbResults
  }

  private func shuffleRGB(
    shuffles: [(Tensor<FloatType>, Float)], graph: DynamicGraph, startHeight: Int, startWidth: Int,
    adjustRGB: Bool, aspectFit: Bool
  ) -> [(DynamicGraph.Tensor<FloatType>, Float)] {
    var rgbResults = [(DynamicGraph.Tensor<FloatType>, Float)]()
    for (shuffle, strength) in shuffles {
      let input = graph.variable(Tensor<FloatType>(shuffle).toGPU(0))
      let inputHeight = input.shape[1]
      let inputWidth = input.shape[2]
      precondition(input.shape[3] == 3)
      if inputHeight != startHeight * 8 || inputWidth != startWidth * 8 {
        if aspectFit {
          let scale = min(
            Float(startWidth * 8) / Float(inputWidth), Float(startHeight * 8) / Float(inputHeight))
          let graph = input.graph
          let shape = input.shape
          var resampled = Upsample(.bilinear, widthScale: scale, heightScale: scale)(input)
          resampled = adjustRGB ? 0.5 * (resampled + 1) : resampled
          var adjusted = graph.variable(
            .GPU(0), .NHWC(shape[0], startHeight * 8, startWidth * 8, 3), of: FloatType.self)
          adjusted.full(1)
          let yOffset = (startHeight * 8 - resampled.shape[1]) / 2
          let xOffset = (startWidth * 8 - resampled.shape[2]) / 2
          adjusted[
            0..<shape[0], yOffset..<(yOffset + resampled.shape[1]),
            xOffset..<(xOffset + resampled.shape[2]), 0..<3] = resampled
          rgbResults.append((adjusted, strength))
        } else {
          let resampled = Upsample(
            .bilinear, widthScale: Float(startWidth * 8) / Float(inputWidth),
            heightScale: Float(startHeight * 8) / Float(inputHeight))(input)
          rgbResults.append((adjustRGB ? 0.5 * (resampled + 1) : resampled, strength))
        }
      } else {
        rgbResults.append((adjustRGB ? 0.5 * (input + 1) : input, strength))
      }
    }
    return rgbResults
  }

  private func generateInjectedTextEmbeddings(
    batchSize: Int, startHeight: Int, startWidth: Int, image: Tensor<FloatType>?,
    graph: DynamicGraph, hints: [ControlHintType: AnyTensor],
    custom: Tensor<FloatType>?, shuffles: [(Tensor<FloatType>, Float)], pose: Tensor<FloatType>?,
    controls: [Control], version: ModelVersion, tiledDiffusion: TiledConfiguration,
    usesFlashAttention: Bool, externalOnDemand: Bool,
    cancellation: (@escaping () -> Void) -> Void
  ) -> [(model: ControlModel<FloatType>, hints: [([DynamicGraph.Tensor<FloatType>], Float)])] {
    return controls.enumerated().compactMap {
      index, control -> (
        model: ControlModel<FloatType>, hints: [([DynamicGraph.Tensor<FloatType>], Float)]
      )?
      in
      guard let file = control.file, let specification = ControlNetZoo.specificationForModel(file),
        ControlNetZoo.isModelDownloaded(specification)
      else { return nil }
      guard ControlNetZoo.versionForModel(file) == version else { return nil }
      guard control.weight > 0 else { return nil }
      guard
        let modifier = ControlNetZoo.modifierForModel(file)
          ?? ControlHintType(from: control.inputOverride)
      else { return nil }
      let type = ControlNetZoo.typeForModel(file)
      let imageEncoderVersion = ControlNetZoo.imageEncoderVersionForModel(file)
      var filePaths = [ControlNetZoo.filePathForModelDownloaded(file)]
      if let imageEncoder = ControlNetZoo.imageEncoderForModel(file) {
        filePaths.append(ControlNetZoo.filePathForModelDownloaded(imageEncoder))
      }
      // We don't adjust RGB range if it is a ControlNet not trained for SDXL.
      let controlModel = ControlModel<FloatType>(
        filePaths: filePaths, type: type, modifier: modifier,
        externalOnDemand: externalOnDemand, version: version,
        tiledDiffusion: tiledDiffusion, usesFlashAttention: usesFlashAttention,
        startStep: 0, endStep: 0, controlMode: .prompt,
        globalAveragePooling: false, transformerBlocks: [],
        targetBlocks: [], imageEncoderVersion: imageEncoderVersion,
        deviceProperties: DeviceCapability.deviceProperties, ipAdapterConfig: nil, firstStage: nil)
      switch type {
      case .controlnet, .controlnetunion, .controlnetlora, .injectKV, .ipadapterplus,
        .ipadapterfull, .redux, .ipadapterfaceidplus, .pulid, .t2iadapter:
        return nil
      case .llava:
        var shuffles = shuffles
        if shuffles.isEmpty {
          guard let custom = custom ?? image else { return nil }
          shuffles = [(custom, 1)]
        }
        let rgbs = ipAdapterRGB(
          shuffles: shuffles, imageEncoderVersion: imageEncoderVersion, graph: graph)
        let hints: [([DynamicGraph.Tensor<FloatType>], Float)] = controlModel.hint(
          batchSize: batchSize, startHeight: startHeight, startWidth: startWidth,
          image: nil, graph: graph,
          inputs: rgbs.map { (hint: $0.0, weight: $0.1 * control.weight) },
          cancellation: cancellation
        ).map { ($0, 1) }
        return (model: controlModel, hints: hints)
      }
    }
  }

  private func generateInjectedControls(
    graph: DynamicGraph, batchSize: Int, startHeight: Int, startWidth: Int,
    image: DynamicGraph.Tensor<FloatType>?,
    depth: DynamicGraph.Tensor<FloatType>?, hints: [ControlHintType: AnyTensor],
    custom: Tensor<FloatType>?, shuffles: [(Tensor<FloatType>, Float)], pose: Tensor<FloatType>?,
    mask: Tensor<FloatType>?,
    controls: [Control], version: ModelVersion, tiledDiffusion: TiledConfiguration,
    usesFlashAttention: Bool, externalOnDemand: Bool, steps: Int, firstStage: FirstStage<FloatType>,
    cancellation: (@escaping () -> Void) -> Void
  ) -> [(model: ControlModel<FloatType>, hints: [([DynamicGraph.Tensor<FloatType>], Float)])] {
    return controls.enumerated().compactMap {
      index, control -> (
        model: ControlModel<FloatType>, hints: [([DynamicGraph.Tensor<FloatType>], Float)]
      )?
      in
      guard let file = control.file, let specification = ControlNetZoo.specificationForModel(file),
        ControlNetZoo.isModelDownloaded(specification)
      else { return nil }
      guard ControlNetZoo.versionForModel(file) == version else { return nil }
      guard control.weight > 0 else { return nil }
      guard
        let modifier = ControlNetZoo.modifierForModel(file)
          ?? ControlHintType(from: control.inputOverride)
      else { return nil }
      let startStep = Int(floor(Float(steps - 1) * control.guidanceStart + 0.5))
      let endStep = Int(ceil(Float(steps - 1) * control.guidanceEnd + 0.5))
      let type = ControlNetZoo.typeForModel(file)
      let isPreprocessorDownloaded = ControlNetZoo.preprocessorForModel(file).map {
        ControlNetZoo.isModelDownloaded($0)
      }
      let controlMode: Diffusion.ControlMode =
        (type == .controlnet || type == .controlnetlora) && modifier == .shuffle
        ? .control
        : {
          switch control.controlMode {
          case .balanced:
            return .balanced
          case .prompt:
            return .prompt
          case .control:
            return .control
          }
        }()
      let globalAveragePooling = modifier == .shuffle ? control.globalAveragePooling : false
      let downSamplingRate = max(control.downSamplingRate, 1)
      let transformerBlocks = ControlNetZoo.transformerBlocksForModel(file)
      let imageEncoderVersion = ControlNetZoo.imageEncoderVersionForModel(file)
      let ipAdapterConfig = ControlNetZoo.IPAdapterConfigForModel(file)
      var filePaths = [ControlNetZoo.filePathForModelDownloaded(file)]
      if let imageEncoder = ControlNetZoo.imageEncoderForModel(file) {
        filePaths.append(ControlNetZoo.filePathForModelDownloaded(imageEncoder))
      }
      if let autoencoder = ControlNetZoo.autoencoderForModel(file) {
        filePaths.append(ControlNetZoo.filePathForModelDownloaded(autoencoder))
      }
      if let preprocessor = ControlNetZoo.preprocessorForModel(file) {
        filePaths.append(ControlNetZoo.filePathForModelDownloaded(preprocessor))
      }
      // We don't adjust RGB range if it is a ControlNet not trained for SDXL.
      let adjustRGB = (version != .flux1 && version != .wan21_14b && version != .wan21_1_3b)
      let aspectFit = (version == .wan21_14b || version == .wan21_1_3b)
      let controlModel = ControlModel<FloatType>(
        filePaths: filePaths, type: type, modifier: modifier,
        externalOnDemand: externalOnDemand, version: version,
        tiledDiffusion: tiledDiffusion, usesFlashAttention: usesFlashAttention,
        startStep: startStep, endStep: endStep, controlMode: controlMode,
        globalAveragePooling: globalAveragePooling, transformerBlocks: transformerBlocks,
        targetBlocks: control.targetBlocks, imageEncoderVersion: imageEncoderVersion,
        deviceProperties: DeviceCapability.deviceProperties, ipAdapterConfig: ipAdapterConfig,
        firstStage: firstStage)
      func customRGB(_ convert: Bool) -> DynamicGraph.Tensor<FloatType>? {
        custom.map({
          let input = graph.variable(Tensor<FloatType>($0).toGPU(0))
          let inputHeight = input.shape[1]
          let inputWidth = input.shape[2]
          precondition(input.shape[3] == 3)
          guard convert else {
            guard inputHeight != startHeight * 8 || inputWidth != startWidth * 8 else {
              return input
            }
            return Upsample(
              .bilinear, widthScale: Float(startWidth * 8) / Float(inputWidth),
              heightScale: Float(startHeight * 8) / Float(inputHeight))(input)
          }
          guard inputHeight != startHeight * 8 || inputWidth != startWidth * 8 else {
            return adjustRGB ? 0.5 * (input + 1) : input
          }
          if aspectFit {
            let scale = min(
              Float(startWidth * 8) / Float(inputWidth), Float(startHeight * 8) / Float(inputHeight)
            )
            let graph = input.graph
            let shape = input.shape
            var resampled = Upsample(.bilinear, widthScale: scale, heightScale: scale)(input)
            resampled = adjustRGB ? 0.5 * (resampled + 1) : resampled
            var adjusted = graph.variable(
              .GPU(0), .NHWC(shape[0], startHeight * 8, startWidth * 8, 3), of: FloatType.self)
            adjusted.full(1)
            let yOffset = (startHeight * 8 - resampled.shape[1]) / 2
            let xOffset = (startWidth * 8 - resampled.shape[2]) / 2
            adjusted[
              0..<shape[0], yOffset..<(yOffset + resampled.shape[1]),
              xOffset..<(xOffset + resampled.shape[2]), 0..<3] = resampled
            return adjusted
          } else {
            let resampled = Upsample(
              .bilinear, widthScale: Float(startWidth * 8) / Float(inputWidth),
              heightScale: Float(startHeight * 8) / Float(inputHeight))(input)
            return adjustRGB ? 0.5 * (resampled + 1) : resampled
          }
        })
      }
      switch type {
      case .controlnet, .controlnetunion, .controlnetlora:
        switch modifier {
        case .canny:
          guard let image = image else {
            guard let rgb = customRGB(true) else { return nil }
            let hints: [([DynamicGraph.Tensor<FloatType>], Float)] = controlModel.hint(
              batchSize: batchSize, startHeight: startHeight, startWidth: startWidth,
              image: image, graph: graph,
              inputs: [
                (hint: rgb, weight: 1)
              ], cancellation: cancellation
            ).map { ($0, control.weight) }
            return (model: controlModel, hints: hints)
          }
          let canny = graph.variable(
            ControlModel<FloatType>.canny(image.rawValue.toCPU(), adjustRGB: adjustRGB).toGPU(0))
          let hints: [([DynamicGraph.Tensor<FloatType>], Float)] = controlModel.hint(
            batchSize: batchSize, startHeight: startHeight, startWidth: startWidth,
            image: image, graph: graph,
            inputs: [
              (hint: canny, weight: 1)
            ], cancellation: cancellation
          ).map { ($0, control.weight) }
          return (model: controlModel, hints: hints)
        case .softedge:
          let isPreprocessorDownloaded =
            isPreprocessorDownloaded
            ?? ControlNetZoo.isModelDownloaded(ImageGeneratorUtils.defaultSoftEdgePreprocessor)
          guard isPreprocessorDownloaded, let image = image else {
            guard let rgb = customRGB(true) else { return nil }
            let hints: [([DynamicGraph.Tensor<FloatType>], Float)] = controlModel.hint(
              batchSize: batchSize, startHeight: startHeight, startWidth: startWidth,
              image: image, graph: graph,
              inputs: [
                (hint: rgb, weight: 1)
              ], cancellation: cancellation
            ).map { ($0, control.weight) }
            return (model: controlModel, hints: hints)
          }
          var softedge = graph.variable(image.rawValue.toGPU(0))
          let shape = softedge.shape
          let preprocessor =
            ControlNetZoo.preprocessorForModel(file)
            ?? ImageGeneratorUtils.defaultSoftEdgePreprocessor
          let preprocessed = ControlModel<FloatType>.hed(
            softedge,
            modelFilePath: ControlNetZoo.filePathForModelDownloaded(preprocessor))
          var softedgeRGB = graph.variable(
            .GPU(0), .NHWC(shape[0], shape[1], shape[2], 3), of: FloatType.self)
          softedgeRGB[0..<shape[0], 0..<shape[1], 0..<shape[2], 0..<1] = preprocessed
          softedgeRGB[0..<shape[0], 0..<shape[1], 0..<shape[2], 1..<2] = preprocessed
          softedgeRGB[0..<shape[0], 0..<shape[1], 0..<shape[2], 2..<3] = preprocessed
          if adjustRGB {
            // HED output is 0 to 1.
            softedge = softedgeRGB
          } else {
            softedge = softedgeRGB * 2 - 1
          }
          let hints: [([DynamicGraph.Tensor<FloatType>], Float)] = controlModel.hint(
            batchSize: batchSize, startHeight: startHeight, startWidth: startWidth,
            image: image, graph: graph,
            inputs: [
              (hint: softedge, weight: 1)
            ], cancellation: cancellation
          ).map { ($0, control.weight) }
          return (model: controlModel, hints: hints)
        case .mlsd:
          guard let image = image else {
            guard let rgb = customRGB(true) else { return nil }
            let hints: [([DynamicGraph.Tensor<FloatType>], Float)] = controlModel.hint(
              batchSize: batchSize, startHeight: startHeight, startWidth: startWidth,
              image: image, graph: graph,
              inputs: [
                (hint: rgb, weight: 1)
              ], cancellation: cancellation
            ).map { ($0, control.weight) }
            return (model: controlModel, hints: hints)
          }
          let imageTensor = image.rawValue.toCPU()
          guard
            let mlsdTensor = ControlModel<FloatType>.mlsd(graph.variable(imageTensor.toGPU(0)))
          else {
            return nil
          }
          let hints: [([DynamicGraph.Tensor<FloatType>], Float)] = controlModel.hint(
            batchSize: batchSize, startHeight: startHeight, startWidth: startWidth,
            image: image, graph: graph,
            inputs: [
              (hint: mlsdTensor.toGPU(0), weight: 1)
            ], cancellation: cancellation
          ).map { ($0, control.weight) }
          return (model: controlModel, hints: hints)
        case .depth:
          guard
            var depth = depth
              ?? (image.flatMap {
                guard ImageGeneratorUtils.isDepthModelAvailable else { return nil }
                return graph.variable(
                  ImageGeneratorUtils.extractDepthMap(
                    $0, imageWidth: startWidth * 8, imageHeight: startHeight * 8,
                    usesFlashAttention: usesFlashAttention
                  )
                  .toGPU(0))
              })
          else { return nil }
          // ControlNet input is always RGB at 0~1 range.
          let shape = depth.shape
          precondition(shape[3] == 1)
          if adjustRGB {
            depth = 0.5 * (depth + 1)
          }
          var depthRGB = graph.variable(
            .GPU(0), .NHWC(shape[0], shape[1], shape[2], 3), of: FloatType.self)
          depthRGB[0..<shape[0], 0..<shape[1], 0..<shape[2], 0..<1] = depth
          depthRGB[0..<shape[0], 0..<shape[1], 0..<shape[2], 1..<2] = depth
          depthRGB[0..<shape[0], 0..<shape[1], 0..<shape[2], 2..<3] = depth
          let hints: [([DynamicGraph.Tensor<FloatType>], Float)] = controlModel.hint(
            batchSize: batchSize, startHeight: startHeight, startWidth: startWidth,
            image: image, graph: graph,
            inputs: [
              (hint: depthRGB, weight: 1)
            ], cancellation: cancellation
          ).map { ($0, control.weight) }
          return (model: controlModel, hints: hints)
        case .scribble:
          let isPreprocessorDownloaded =
            isPreprocessorDownloaded
            ?? ControlNetZoo.isModelDownloaded(ImageGeneratorUtils.defaultSoftEdgePreprocessor)
          var scribble: DynamicGraph.Tensor<FloatType>? = nil
          if let rgb = hints[.scribble].map({
            let input = graph.variable(Tensor<FloatType>($0).toGPU(0))
            let inputHeight = input.shape[1]
            let inputWidth = input.shape[2]
            guard inputHeight != startHeight * 8 || inputWidth != startWidth * 8 else {
              return input
            }
            return Upsample(
              .bilinear, widthScale: Float(startWidth * 8) / Float(inputWidth),
              heightScale: Float(startHeight * 8) / Float(inputHeight))(input)
          }) {
            if adjustRGB {
              // rgb output is 0 to 1.
              scribble = rgb
            } else {
              scribble = rgb * 2 - 1
            }
          } else if isPreprocessorDownloaded, let image = image {
            let rawImage = graph.variable(image.rawValue.toGPU(0))
            let shape = rawImage.shape
            let preprocessor =
              ControlNetZoo.preprocessorForModel(file)
              ?? ImageGeneratorUtils.defaultSoftEdgePreprocessor
            let preprocessed = ControlModel<FloatType>.hed(
              rawImage,
              modelFilePath: ControlNetZoo.filePathForModelDownloaded(preprocessor))
            var scribbleRGB = graph.variable(
              .GPU(0), .NHWC(shape[0], shape[1], shape[2], 3), of: FloatType.self)
            scribbleRGB[0..<shape[0], 0..<shape[1], 0..<shape[2], 0..<1] = preprocessed
            scribbleRGB[0..<shape[0], 0..<shape[1], 0..<shape[2], 1..<2] = preprocessed
            scribbleRGB[0..<shape[0], 0..<shape[1], 0..<shape[2], 2..<3] = preprocessed
            if adjustRGB {
              // HED output is 0 to 1.
              scribble = scribbleRGB
            } else {
              scribble = scribbleRGB * 2 - 1
            }
          }
          guard let scribble = scribble else {
            return nil
          }
          let hints: [([DynamicGraph.Tensor<FloatType>], Float)] = controlModel.hint(
            batchSize: batchSize, startHeight: startHeight, startWidth: startWidth,
            image: image, graph: graph,
            inputs: [
              (hint: scribble, weight: 1)
            ], cancellation: cancellation
          ).map { ($0, control.weight) }
          return (model: controlModel, hints: hints)
        case .pose:
          guard let pose = pose else {
            guard let rgb = customRGB(true) else { return nil }
            let hints: [([DynamicGraph.Tensor<FloatType>], Float)] = controlModel.hint(
              batchSize: batchSize, startHeight: startHeight, startWidth: startWidth,
              image: image, graph: graph,
              inputs: [
                (hint: rgb, weight: 1)
              ], cancellation: cancellation
            ).map { ($0, control.weight) }
            return (model: controlModel, hints: hints)
          }

          let hints: [([DynamicGraph.Tensor<FloatType>], Float)] = controlModel.hint(
            batchSize: batchSize, startHeight: startHeight, startWidth: startWidth,
            image: image, graph: graph,
            inputs: [
              (hint: graph.variable(pose.toGPU(0)), weight: 1)
            ], cancellation: cancellation
          ).map { ($0, control.weight) }
          return (model: controlModel, hints: hints)
        case .lineart:
          guard var rgb = customRGB(false) else { return nil }
          rgb = 0.5 * (1 - rgb)
          let hints: [([DynamicGraph.Tensor<FloatType>], Float)] = controlModel.hint(
            batchSize: batchSize, startHeight: startHeight, startWidth: startWidth,
            image: image, graph: graph,
            inputs: [
              (hint: rgb, weight: 1)
            ], cancellation: cancellation
          ).map { ($0, control.weight) }
          return (model: controlModel, hints: hints)
        case .normalbae, .seg, .custom:
          guard let rgb = customRGB(true) else { return nil }
          let hints: [([DynamicGraph.Tensor<FloatType>], Float)] = controlModel.hint(
            batchSize: batchSize, startHeight: startHeight, startWidth: startWidth,
            image: image, graph: graph,
            inputs: [
              (hint: rgb, weight: 1)
            ], cancellation: cancellation
          ).map { ($0, control.weight) }
          return (model: controlModel, hints: hints)
        case .shuffle:
          let isVACE = (version == .wan21_1_3b || version == .wan21_14b)
          guard !shuffles.isEmpty else {
            guard let rgb = customRGB(true) else {
              if isVACE {
                let hints: [([DynamicGraph.Tensor<FloatType>], Float)] = controlModel.hint(
                  batchSize: batchSize, startHeight: startHeight, startWidth: startWidth,
                  image: image, graph: graph,
                  inputs: [], cancellation: cancellation
                ).map { ($0, control.weight) }
                return (model: controlModel, hints: hints)
              }
              return nil
            }
            let hints: [([DynamicGraph.Tensor<FloatType>], Float)] = controlModel.hint(
              batchSize: batchSize, startHeight: startHeight, startWidth: startWidth,
              image: image, graph: graph,
              inputs: [
                (hint: rgb, weight: 1)
              ], cancellation: cancellation
            ).map { ($0, control.weight) }
            return (model: controlModel, hints: hints)
          }
          let rgbs = shuffleRGB(
            shuffles: shuffles, graph: graph, startHeight: startHeight, startWidth: startWidth,
            adjustRGB: adjustRGB, aspectFit: aspectFit)
          let hints: [([DynamicGraph.Tensor<FloatType>], Float)] = zip(
            rgbs,
            controlModel.hint(
              batchSize: batchSize, startHeight: startHeight, startWidth: startWidth,
              image: image, graph: graph, inputs: rgbs.map { (hint: $0.0, weight: 1) },
              cancellation: cancellation
            )
          ).map { ($0.1, isVACE ? control.weight : $0.0.1 * control.weight) }
          return (model: controlModel, hints: hints)
        case .inpaint:
          guard var input = image else { return nil }
          let inputHeight = input.shape[1]
          let inputWidth = input.shape[2]
          precondition(input.shape[3] == 3)
          if inputHeight != startHeight * 8 || inputWidth != startWidth * 8 {
            input = Upsample(
              .bilinear, widthScale: Float(startWidth * 8) / Float(inputWidth),
              heightScale: Float(startHeight * 8) / Float(inputHeight))(input)
          }
          if adjustRGB {
            input = 0.5 * (input + 1)
          }
          // If mask exists, apply mask, by making the mask area (0) to -1.
          if var mask = mask {
            // Making it either 0 or 1.
            let shape = mask.shape
            precondition(shape[0] == 1)
            precondition(shape[3] == 1)
            for y in 0..<shape[1] {
              for x in 0..<shape[2] {
                mask[0, y, x, 0] = mask[0, y, x, 0] == 1 ? 1 : 0
              }
            }
            var inputMask = graph.variable(mask.toGPU(0))
            if shape[1] != startHeight * 8 || shape[2] != startWidth * 8 {
              inputMask = Upsample(
                .nearest, widthScale: Float(startWidth * 8) / Float(shape[2]),
                heightScale: Float(startHeight * 8) / Float(shape[1]))(inputMask)
            }
            if type == .controlnetunion && version != .v1 {
              input = input .* inputMask  // For Xinsir control union, there is no explicit -1.
            } else {
              input = input .* inputMask + (inputMask - 1)
            }
          }
          var hints: [([DynamicGraph.Tensor<FloatType>], Float)] = controlModel.hint(
            batchSize: batchSize, startHeight: startHeight, startWidth: startWidth,
            image: image, graph: graph,
            inputs: [
              (hint: input, weight: 1)
            ], cancellation: cancellation
          ).map { ($0, control.weight) }
          if version == .flux1 {
            hints = hints.map {
              return (
                $0.0.map {
                  guard let mask = mask else {
                    return $0
                  }
                  let shape = $0.shape
                  let maskShape = mask.shape
                  var inputMask = graph.variable(mask.toGPU(0))
                  inputMask = Upsample(
                    .bilinear, widthScale: Float(shape[2]) / Float(maskShape[2]),
                    heightScale: Float(shape[1]) / Float(maskShape[1]))(inputs: inputMask)[0].as(
                      of: FloatType.self)
                  return Functional.concat(axis: 3, $0, inputMask)
                }, $0.1
              )
            }
          }
          return (model: controlModel, hints: hints)
        case .ip2p:
          guard var input = image else { return nil }
          let inputHeight = input.shape[1]
          let inputWidth = input.shape[2]
          precondition(input.shape[3] == 3)
          if inputHeight != startHeight * 8 || inputWidth != startWidth * 8 {
            input = Upsample(
              .bilinear, widthScale: Float(startWidth * 8) / Float(inputWidth),
              heightScale: Float(startHeight * 8) / Float(inputHeight))(input)
          }
          if adjustRGB {
            input = 0.5 * (input + 1)
          }
          let hints: [([DynamicGraph.Tensor<FloatType>], Float)] = controlModel.hint(
            batchSize: batchSize, startHeight: startHeight, startWidth: startWidth,
            image: image, graph: graph,
            inputs: [
              (hint: input, weight: 1)
            ], cancellation: cancellation
          ).map { ($0, control.weight) }
          return (model: controlModel, hints: hints)
        case .tile, .blur, .gray, .lowquality:
          // Prefer custom for tile.
          if let rgb = customRGB(true) {
            let hint = controlModel.hint(
              batchSize: batchSize, startHeight: startHeight, startWidth: startWidth,
              image: image, graph: graph, inputs: [(hint: rgb, weight: 1)],
              cancellation: cancellation)[0]
            return (model: controlModel, hints: [(hint, control.weight)])
          }
          guard var input = image else { return nil }
          let inputHeight = input.shape[1]
          let inputWidth = input.shape[2]
          precondition(input.shape[3] == 3)
          if inputHeight != startHeight * 8 || inputWidth != startWidth * 8 {
            input = Upsample(
              .bilinear, widthScale: Float(startWidth * 8) / Float(inputWidth),
              heightScale: Float(startHeight * 8) / Float(inputHeight))(input)
          }
          if adjustRGB {
            input = 0.5 * (input + 1)
          }
          if downSamplingRate > 1.1 {
            let inputHeight = input.shape[1]
            let inputWidth = input.shape[2]
            let tensor = Tensor<Float>(
              from: input.rawValue.reshaped(.HWC(inputHeight, inputWidth, 3)).toCPU())
            var b: UnsafeMutablePointer<ccv_dense_matrix_t>? = nil
            ccv_resample(
              UnsafeMutableRawPointer(tensor.cTensor).assumingMemoryBound(
                to: ccv_dense_matrix_t.self), &b, 0, 1 / Double(downSamplingRate),
              1 / Double(downSamplingRate), Int32(CCV_INTER_AREA | CCV_INTER_CUBIC))
            var c = ccv_dense_matrix_new(
              Int32(inputHeight), Int32(inputWidth), Int32(CCV_32F | CCV_C3), nil, 0)
            ccv_resample(
              b, &c, 0, Double(downSamplingRate), Double(downSamplingRate),
              Int32(CCV_INTER_AREA | CCV_INTER_CUBIC))
            ccv_matrix_free(b)
            input = graph.variable(
              Tensor<FloatType>(
                from: Tensor<Float>(
                  .CPU, format: .NHWC, shape: [1, inputHeight, inputWidth, 3],
                  unsafeMutablePointer: c!.pointee.data.f32, bindLifetimeOf: c!
                ).toGPU()))
            ccv_matrix_free(c)
          }
          let hints: [([DynamicGraph.Tensor<FloatType>], Float)] = controlModel.hint(
            batchSize: batchSize, startHeight: startHeight, startWidth: startWidth,
            image: image, graph: graph,
            inputs: [
              (hint: input, weight: 1)
            ], cancellation: cancellation
          ).map { ($0, control.weight) }
          return (model: controlModel, hints: hints)
        case .color:
          return nil  // Not supported at the moment.
        }
      case .injectKV:
        guard !shuffles.isEmpty else { return nil }
        let hints: [([DynamicGraph.Tensor<FloatType>], Float)] = controlModel.hint(
          batchSize: batchSize, startHeight: startHeight, startWidth: startWidth,
          image: image, graph: graph,
          inputs: shuffles.map {
            (hint: graph.variable($0.0).toGPU(0), weight: $0.1 * control.weight)
          }, cancellation: cancellation
        ).map { ($0, 1) }
        return (model: controlModel, hints: hints)
      case .llava:
        return nil
      case .ipadapterplus, .ipadapterfull, .redux:
        var shuffles = shuffles
        if shuffles.isEmpty {
          guard let custom = custom else { return nil }
          shuffles = [(custom, 1)]
        }
        let rgbs = ipAdapterRGB(
          shuffles: shuffles, imageEncoderVersion: imageEncoderVersion, graph: graph)
        let hints: [([DynamicGraph.Tensor<FloatType>], Float)] = controlModel.hint(
          batchSize: batchSize, startHeight: startHeight, startWidth: startWidth,
          image: image, graph: graph,
          inputs: rgbs.map { (hint: $0.0, weight: $0.1 * control.weight) },
          cancellation: cancellation
        ).map { ($0, 1) }
        return (model: controlModel, hints: hints)
      case .ipadapterfaceidplus, .pulid:
        var shuffles = shuffles
        if shuffles.isEmpty {
          guard let custom = custom else { return nil }
          shuffles = [(custom, 1)]
        }
        let rgbs = shuffles.map { (graph.variable($0.0), $0.1) }
        let hints: [([DynamicGraph.Tensor<FloatType>], Float)] = controlModel.hint(
          batchSize: batchSize, startHeight: startHeight, startWidth: startWidth,
          image: image, graph: graph,
          inputs: (rgbs.map { (hint: $0.0, weight: $0.1 * control.weight) }),
          cancellation: cancellation
        ).map { ($0, 1) }
        return (model: controlModel, hints: hints)
      case .t2iadapter:
        switch modifier {
        case .canny:
          guard let image = image else {
            guard let rgb = customRGB(true) else { return nil }
            let shape = rgb.shape
            let input = rgb[0..<shape[0], 0..<shape[1], 0..<shape[2], 0..<1].copied().reshaped(
              format: .NHWC, shape: [shape[0], startHeight, 8, startWidth, 8]
            ).permuted(0, 1, 3, 2, 4).copied().reshaped(
              .NHWC(shape[0], startHeight, startWidth, 64))
            let hint = controlModel.hint(
              batchSize: batchSize, startHeight: startHeight, startWidth: startWidth,
              image: image, graph: graph, inputs: [(hint: input, weight: control.weight)],
              cancellation: cancellation)[0]
            return (model: controlModel, hints: [(hint, 1)])
          }
          let canny = graph.variable(
            ControlModel<FloatType>.canny(image.rawValue.toCPU(), adjustRGB: adjustRGB).toGPU(0))
          let shape = canny.shape
          let input = canny[0..<shape[0], 0..<shape[1], 0..<shape[2], 0..<1].copied().reshaped(
            format: .NHWC, shape: [shape[0], startHeight, 8, startWidth, 8]
          ).permuted(0, 1, 3, 2, 4).copied().reshaped(
            .NHWC(shape[0], startHeight, startWidth, 64))
          let hints: [([DynamicGraph.Tensor<FloatType>], Float)] = controlModel.hint(
            batchSize: batchSize, startHeight: startHeight, startWidth: startWidth,
            image: image, graph: graph,
            inputs: [
              (hint: input, weight: control.weight)
            ], cancellation: cancellation
          ).map { ($0, 1) }
          return (model: controlModel, hints: hints)
        case .depth:
          guard var depth = depth else { return nil }
          // ControlNet input is always RGB at 0~1 range.
          let shape = depth.shape
          precondition(shape[3] == 1)
          if adjustRGB {
            depth = 0.5 * (depth + 1)
          }
          var depthRGB = graph.variable(
            .GPU(0), .NHWC(shape[0], shape[1], shape[2], 3), of: FloatType.self)
          depthRGB[0..<shape[0], 0..<shape[1], 0..<shape[2], 0..<1] = depth
          depthRGB[0..<shape[0], 0..<shape[1], 0..<shape[2], 1..<2] = depth
          depthRGB[0..<shape[0], 0..<shape[1], 0..<shape[2], 2..<3] = depth
          let input = depthRGB.reshaped(
            format: .NHWC, shape: [shape[0], startHeight, 8, startWidth, 8, 3]
          ).permuted(0, 1, 3, 5, 2, 4).copied().reshaped(
            .NHWC(shape[0], startHeight, startWidth, 64 * 3))
          let hints: [([DynamicGraph.Tensor<FloatType>], Float)] = controlModel.hint(
            batchSize: batchSize, startHeight: startHeight, startWidth: startWidth,
            image: image, graph: graph,
            inputs: [
              (hint: input, weight: control.weight)
            ], cancellation: cancellation
          ).map { ($0, 1) }
          return (model: controlModel, hints: hints)
        case .scribble:
          guard
            let rgb = hints[.scribble].map({
              let input = graph.variable(Tensor<FloatType>($0).toGPU(0))
              let inputHeight = input.shape[1]
              let inputWidth = input.shape[2]
              guard inputHeight != startHeight * 8 || inputWidth != startWidth * 8 else {
                return input
              }
              return Upsample(
                .bilinear, widthScale: Float(startWidth * 8) / Float(inputWidth),
                heightScale: Float(startHeight * 8) / Float(inputHeight))(input)
            })
          else { return nil }
          let shape = rgb.shape
          let input = rgb[0..<shape[0], 0..<shape[1], 0..<shape[2], 0..<1].copied().reshaped(
            format: .NHWC, shape: TensorShape(arrayLiteral: shape[0], startHeight, 8, startWidth, 8)
          ).permuted(0, 1, 3, 2, 4).copied().reshaped(
            .NHWC(shape[0], startHeight, startWidth, 64))
          let hints: [([DynamicGraph.Tensor<FloatType>], Float)] = controlModel.hint(
            batchSize: batchSize, startHeight: startHeight, startWidth: startWidth,
            image: image, graph: graph,
            inputs: [
              (hint: input, weight: control.weight)
            ], cancellation: cancellation
          ).map { ($0, 1) }
          return (model: controlModel, hints: hints)
        case .pose:
          guard let pose = pose else {
            guard let rgb = customRGB(true) else { return nil }
            let shape = rgb.shape
            let input = rgb.reshaped(
              format: .NHWC, shape: [shape[0], startHeight, 8, startWidth, 8, 3]
            ).permuted(0, 1, 3, 5, 2, 4).copied().reshaped(
              .NHWC(shape[0], startHeight, startWidth, 64 * 3))
            let hints: [([DynamicGraph.Tensor<FloatType>], Float)] = controlModel.hint(
              batchSize: batchSize, startHeight: startHeight, startWidth: startWidth,
              image: image, graph: graph,
              inputs: [
                (hint: input, weight: control.weight)
              ], cancellation: cancellation
            ).map { ($0, 1) }
            return (model: controlModel, hints: hints)
          }
          let shape = pose.shape
          let input = graph.variable(pose.toGPU(0)).reshaped(
            format: .NHWC, shape: [shape[0], startHeight, 8, startWidth, 8, 3]
          ).permuted(0, 1, 3, 5, 2, 4).copied().reshaped(
            .NHWC(shape[0], startHeight, startWidth, 64 * 3))
          let hints: [([DynamicGraph.Tensor<FloatType>], Float)] = controlModel.hint(
            batchSize: batchSize, startHeight: startHeight, startWidth: startWidth,
            image: image, graph: graph,
            inputs: [
              (hint: input, weight: control.weight)
            ], cancellation: cancellation
          ).map { ($0, 1) }
          return (model: controlModel, hints: hints)
        case .color:
          let sourceColor: DynamicGraph.Tensor<FloatType>? =
            hints[.color].map({ graph.variable(Tensor<FloatType>($0).toGPU(0)) }) ?? image
          guard
            var rgb = sourceColor.map({
              var input = $0
              let inputHeight = input.shape[1]
              let inputWidth = input.shape[2]
              if (inputHeight != startHeight * 8 || inputWidth != startWidth * 8)
                && (inputHeight != startHeight / 8 || inputWidth != startWidth / 8)
              {
                input = Upsample(
                  .bilinear, widthScale: Float(startWidth * 8) / Float(inputWidth),
                  heightScale: Float(startHeight * 8) / Float(inputHeight))(input)
              }
              let colorPalette: DynamicGraph.Tensor<FloatType>
              if inputHeight != startHeight / 8 || inputWidth != startWidth / 8 {
                colorPalette = AveragePool(filterSize: [64, 64], hint: Hint(stride: [64, 64]))(
                  input)
              } else {
                colorPalette = input  // It is already in color palette format.
              }
              return Upsample(.nearest, widthScale: 64, heightScale: 64)(colorPalette)
            })
          else { return nil }
          let shape = rgb.shape
          precondition(shape[3] == 3)
          if adjustRGB {
            rgb = 0.5 * (rgb + 1)
          }
          let input = rgb.reshaped(
            format: .NHWC, shape: [shape[0], startHeight, 8, startWidth, 8, 3]
          ).permuted(0, 1, 3, 5, 2, 4).copied().reshaped(
            .NHWC(shape[0], startHeight, startWidth, 64 * 3))
          let hints: [([DynamicGraph.Tensor<FloatType>], Float)] = controlModel.hint(
            batchSize: batchSize, startHeight: startHeight, startWidth: startWidth,
            image: image, graph: graph,
            inputs: [
              (hint: input, weight: control.weight)
            ], cancellation: cancellation
          ).map { ($0, 1) }
          return (model: controlModel, hints: hints)
        case .normalbae, .lineart, .softedge, .seg, .inpaint, .ip2p, .shuffle, .mlsd, .tile,
          .custom, .blur, .gray, .lowquality:
          return nil  // Not supported at the moment.
        }
      }
    }
  }

  private func stageCLatentsSize(_ configuration: GenerationConfiguration) -> (
    width: Int, height: Int
  ) {
    let firstPassStartWidth =
      configuration.hiresFix && configuration.hiresFixStartWidth > 0
      ? Int(configuration.hiresFixStartWidth) * 2
      : Int((Double((Int(configuration.startWidth)) * 3) / 2).rounded(.up))
    let firstPassStartHeight =
      configuration.hiresFix && configuration.hiresFixStartHeight > 0
      ? Int(configuration.hiresFixStartHeight) * 2
      : Int((Double((Int(configuration.startHeight)) * 3) / 2).rounded(.up))
    return (firstPassStartWidth, firstPassStartHeight)
  }

  private func repeatConditionsToMatchBatchSize(
    c: [DynamicGraph.Tensor<FloatType>], extraProjection: DynamicGraph.Tensor<FloatType>?,
    unconditionalAttentionWeights: [Float], attentionWeights: [Float], version: ModelVersion,
    tokenLength: Int, batchSize: Int, hasNonOneWeights: Bool
  ) -> ([DynamicGraph.Tensor<FloatType>], DynamicGraph.Tensor<FloatType>?) {
    var c = c
    var extraProjection = extraProjection
    if hasNonOneWeights {
      // Need to scale c according to these weights. C has two parts, the unconditional and conditional.
      // We also want to do z-score type of scaling, so we need to compute mean of both parts separately.
      // This is only supported for SD v1, Stable Cascade, SDXL, SD3.
      switch version {
      case .ssd1b, .svdI2v, .v1, .v2, .sdxlBase, .sdxlRefiner, .wurstchenStageB, .wurstchenStageC,
        .sd3, .sd3Large:
        c = c.map { c in
          guard tokenLength == c.shape[1], c.shape.count == 3 else { return c }
          let conditionalLength = c.shape[2]
          // 768, 1280 is the length of CLIP-L and OpenCLIP-G
          guard conditionalLength <= 1280 else { return c }
          var c = c
          let graph = c.graph
          if c.shape[0] >= 2 {
            let cc = c[1..<2, 0..<tokenLength, 0..<conditionalLength]
            let cmean = cc.reduced(.mean, axis: [1, 2])
            let cw = graph.variable(
              Tensor<FloatType>(from: Tensor(attentionWeights, .CPU, .HWC(1, tokenLength, 1)))
                .toGPU(
                  0))
            c[1..<2, 0..<tokenLength, 0..<conditionalLength] = (cc - cmean) .* cw + cmean
            let uc = c[0..<1, 0..<tokenLength, 0..<conditionalLength]
            let umean = uc.reduced(.mean, axis: [1, 2])
            let uw = graph.variable(
              Tensor<FloatType>(
                from: Tensor(unconditionalAttentionWeights, .CPU, .HWC(1, tokenLength, 1))
              )
              .toGPU(0))
            // Keep the mean unchanged while scale it.
            c[0..<1, 0..<tokenLength, 0..<conditionalLength] = (uc - umean) .* uw + umean
          } else {
            let cc = c[0..<1, 0..<tokenLength, 0..<conditionalLength]
            let cmean = cc.reduced(.mean, axis: [1, 2])
            let cw = graph.variable(
              Tensor<FloatType>(from: Tensor(attentionWeights, .CPU, .HWC(1, tokenLength, 1)))
                .toGPU(
                  0))
            c[0..<1, 0..<tokenLength, 0..<conditionalLength] = (cc - cmean) .* cw + cmean
          }
          return c
        }
      case .hiDreamI1:
        // While HiDream uses CLIP, these are not meaningful because it only uses the pooling vector from CLIP.
        break
      case .wan21_1_3b, .wan21_14b:
        fatalError()
      case .auraflow, .flux1, .kandinsky21, .pixart, .hunyuanVideo:
        break
      }
    }
    if batchSize > 1 {
      c = c.map { c in
        var c = c
        let oldC = c
        let graph = c.graph
        let shape = c.shape
        let cBatchSize = batchSize * shape[0]
        if shape.count == 3 {
          c = graph.variable(
            .GPU(0), .HWC(cBatchSize, shape[1], shape[2]), of: FloatType.self)
          for i in 0..<batchSize {
            for j in 0..<shape[0] {
              c[(batchSize * j + i)..<(batchSize * j + i + 1), 0..<shape[1], 0..<shape[2]] =
                oldC[j..<(j + 1), 0..<shape[1], 0..<shape[2]]
            }
          }
        } else if shape.count == 2 {
          c = graph.variable(
            .GPU(0), .WC(cBatchSize, shape[1]), of: FloatType.self)
          for i in 0..<batchSize {
            for j in 0..<shape[0] {
              c[(batchSize * j + i)..<(batchSize * j + i + 1), 0..<shape[1]] =
                oldC[j..<(j + 1), 0..<shape[1]]
            }
          }
        }
        return c
      }
      if let oldProj = extraProjection {
        let shape = oldProj.shape
        let graph = oldProj.graph
        let cBatchSize = batchSize * shape[0]
        var xfProj = graph.variable(
          .GPU(0), .HWC(cBatchSize, shape[1], shape[2]), of: FloatType.self)
        for i in 0..<batchSize {
          for j in 0..<shape[0] {
            xfProj[(batchSize * j + i)..<(batchSize * j + i + 1), 0..<shape[1], 0..<shape[2]] =
              oldProj[j..<(j + 1), 0..<shape[1], 0..<shape[2]]
          }
        }
        extraProjection = xfProj
      }
    }
    return (c, extraProjection)
  }

  private func encodeImageCond(
    image: DynamicGraph.Tensor<FloatType>?, depth: DynamicGraph.Tensor<FloatType>?,
    custom: DynamicGraph.Tensor<FloatType>?, modifier: SamplerModifier, version: ModelVersion,
    firstStage: FirstStage<FloatType>, usesFlashAttention: Bool
  ) -> DynamicGraph.Tensor<FloatType>? {
    switch modifier {
    case .depth:
      guard
        let depth = depth
          ?? (image.flatMap {
            guard ImageGeneratorUtils.isDepthModelAvailable else { return nil }
            let graph = $0.graph
            let shape = $0.shape
            return graph.variable(
              ImageGeneratorUtils.extractDepthMap(
                $0, imageWidth: shape[2], imageHeight: shape[1],
                usesFlashAttention: usesFlashAttention
              )
              .toGPU(0))
          })
      else { return nil }
      switch version {
      case .v1, .v2, .auraflow, .kandinsky21, .pixart, .sd3, .sd3Large, .sdxlBase, .sdxlRefiner,
        .ssd1b, .svdI2v, .wurstchenStageB, .wurstchenStageC:
        return Functional.averagePool(depth, filterSize: [8, 8], hint: Hint(stride: [8, 8]))
      case .flux1:
        let graph = depth.graph
        let shape = depth.shape
        var depthRGB = graph.variable(
          .GPU(0), .NHWC(shape[0], shape[1], shape[2], 3), of: FloatType.self)
        depthRGB[0..<shape[0], 0..<shape[1], 0..<shape[2], 0..<1] = depth
        depthRGB[0..<shape[0], 0..<shape[1], 0..<shape[2], 1..<2] = depth
        depthRGB[0..<shape[0], 0..<shape[1], 0..<shape[2], 2..<3] = depth
        let encodedDepth = firstStage.encode(depthRGB, encoder: nil, cancellation: { _ in }).0
        let encodedShape = encodedDepth.shape
        return firstStage.scale(
          encodedDepth[0..<1, 0..<encodedShape[1], 0..<encodedShape[2], 0..<16].copied())
      case .hiDreamI1:
        fatalError()
      case .wan21_1_3b, .wan21_14b:
        fatalError()
      case .hunyuanVideo:
        fatalError()
      }
    case .canny:
      switch version {
      case .v1, .v2, .auraflow, .kandinsky21, .pixart, .sd3, .sd3Large, .sdxlBase, .sdxlRefiner,
        .ssd1b, .svdI2v, .wurstchenStageB, .wurstchenStageC, .hunyuanVideo, .wan21_1_3b, .wan21_14b,
        .hiDreamI1:
        return nil
      case .flux1:
        guard
          let cannyRGB = {
            guard let image = image else {
              return custom
            }
            let graph = image.graph
            return graph.variable(
              ControlModel<FloatType>.canny(image.rawValue.toCPU(), adjustRGB: false).toGPU(0))
          }()
        else { return nil }
        let encodedCanny = firstStage.encode(cannyRGB, encoder: nil, cancellation: { _ in }).0
        let encodedShape = encodedCanny.shape
        return firstStage.scale(
          encodedCanny[0..<1, 0..<encodedShape[1], 0..<encodedShape[2], 0..<16].copied())
      }
    case .double, .editing, .inpainting, .none:
      return nil
    }
  }

  private func isI2v(version: ModelVersion, modifier: SamplerModifier) -> Bool {
    switch version {
    case .v1, .v2, .auraflow, .kandinsky21, .pixart, .sd3, .sd3Large, .sdxlBase, .sdxlRefiner,
      .ssd1b, .wurstchenStageB, .wurstchenStageC, .flux1, .hiDreamI1:
      return false
    case .svdI2v:
      return true
    case .wan21_14b, .wan21_1_3b, .hunyuanVideo:
      return modifier == .inpainting
    }
  }

  private func expandImageForEncoding(
    batchSize: (Int, Int), version: ModelVersion, modifier: SamplerModifier,
    image: DynamicGraph.Tensor<FloatType>
  ) -> (Int, DynamicGraph.Tensor<FloatType>, DynamicGraph.Tensor<FloatType>?) {
    switch version {
    case .v1, .v2, .auraflow, .kandinsky21, .pixart, .sd3, .sd3Large, .sdxlBase, .sdxlRefiner,
      .ssd1b, .svdI2v, .wurstchenStageB, .wurstchenStageC, .hunyuanVideo, .flux1, .hiDreamI1:
      return (1, image, nil)
    case .wan21_14b, .wan21_1_3b:
      let shape = image.shape
      let batchSize = batchSize.0 - batchSize.1
      guard shape[0] < (batchSize - 1) * 4 + 1 else {
        let copied = image[0..<(batchSize - 1) * 4 + 1, 0..<shape[1], 0..<shape[2], 0..<shape[3]]
          .copied()
        if modifier == .inpainting {
          return (batchSize, copied, copied)
        } else {
          return (batchSize, copied, nil)
        }
      }
      let graph = image.graph
      let decodedSize = (batchSize - 1) * 4 + 1
      var repeatedImage = graph.variable(
        .GPU(0), .NHWC(decodedSize, shape[1], shape[2], shape[3]), of: FloatType.self)
      // Replicate images throughout.
      for i in stride(from: 0, to: decodedSize, by: shape[0]) {
        repeatedImage[
          i..<min(i + shape[0], decodedSize), 0..<shape[1], 0..<shape[2], 0..<shape[3]] = image[
            0..<min(shape[0], decodedSize - i), 0..<shape[1], 0..<shape[2], 0..<shape[3]
          ].copied()
      }
      guard modifier == .inpainting else {
        return (batchSize, repeatedImage, nil)
      }
      var expandedImage = graph.variable(
        .GPU(0), .NHWC(decodedSize, shape[1], shape[2], shape[3]), of: FloatType.self)
      expandedImage.full(0)
      expandedImage[0..<shape[0], 0..<shape[1], 0..<shape[2], 0..<shape[3]] = image
      return (batchSize, expandedImage, repeatedImage)
    }
  }

  private func injectVACEFrames(
    batchSize: (Int, Int), version: ModelVersion, image: DynamicGraph.Tensor<FloatType>,
    injectedControls: [(
      model: ControlModel<FloatType>, hints: [([DynamicGraph.Tensor<FloatType>], Float)]
    )]
  ) -> DynamicGraph.Tensor<FloatType> {
    guard version == .wan21_14b || version == .wan21_1_3b else { return image }
    let shape = image.shape
    guard shape[0] < batchSize.0 else { return image }
    guard
      let hint = injectedControls.first(where: {
        $0.model.type == .controlnet && $0.model.version == version
      })?.hints.first?.0.first
    else { return image }
    var newShape = shape
    newShape[0] = batchSize.0
    var injectedImage = image.graph.variable(
      .GPU(0), format: .NHWC, shape: newShape, of: FloatType.self)
    injectedImage[
      (batchSize.0 - shape[0])..<batchSize.0, 0..<shape[1], 0..<shape[2], 0..<shape[3]] = image
    let maskChannels = shape[3] - 16
    if maskChannels > 0 {
      injectedImage[0..<(batchSize.0 - shape[0]), 0..<shape[1], 0..<shape[2], 0..<maskChannels]
        .full(1)
    }
    injectedImage[
      0..<(batchSize.0 - shape[0]), 0..<shape[1], 0..<shape[2], maskChannels..<(maskChannels + 16)] =
      hint[
        0..<(batchSize.0 - shape[0]), 0..<shape[1], 0..<shape[2], 0..<16
      ].copied()
    return injectedImage
  }

  private func injectReferenceFrames(
    batchSize: Int, version: ModelVersion, canInjectControls: Bool, shuffleCount: Int,
    hasCustom: Bool
  ) -> (Int, Int) {
    switch version {
    case .wan21_14b, .wan21_1_3b:
      let referenceFrames =
        (canInjectControls ? (shuffleCount > 0 ? shuffleCount : (hasCustom ? 1 : 0)) : 0)
      return (
        batchSize + referenceFrames, referenceFrames
      )
    case .hunyuanVideo, .auraflow, .flux1, .hiDreamI1, .kandinsky21, .pixart, .sd3, .sd3Large,
      .sdxlBase, .sdxlRefiner, .ssd1b, .svdI2v, .v1, .v2, .wurstchenStageB, .wurstchenStageC:
      return (batchSize, 0)
    }
  }

  private func concatMaskWithMaskedImage(
    hasImage: Bool,
    batchSize: (Int, Int), version: ModelVersion, encodedImage: DynamicGraph.Tensor<FloatType>,
    encodedMask: DynamicGraph.Tensor<FloatType>, imageNegMask: DynamicGraph.Tensor<FloatType>?
  ) -> DynamicGraph.Tensor<FloatType> {
    let graph = encodedImage.graph
    switch version {
    case .v1, .v2, .auraflow, .kandinsky21, .pixart, .sd3, .sd3Large, .sdxlBase, .sdxlRefiner,
      .ssd1b, .svdI2v, .wurstchenStageB, .wurstchenStageC:
      let shape = encodedImage.shape
      let maskShape = encodedMask.shape
      var result = graph.variable(
        encodedImage.kind, format: encodedImage.format,
        shape: [shape[0], shape[1], shape[2], shape[3] + maskShape[3]], of: FloatType.self)
      result[0..<shape[0], 0..<shape[1], 0..<shape[2], 0..<maskShape[3]] = encodedMask
      result[0..<shape[0], 0..<shape[1], 0..<shape[2], maskShape[3]..<(maskShape[3] + shape[3])] =
        encodedImage
      return result
    case .hunyuanVideo:
      // We don't handle mask yet, or we need to figure out for new variant of Hunyuan, how to handle mask.
      let shape = encodedImage.shape
      var result = graph.variable(
        encodedImage.kind, format: encodedImage.format,
        shape: [batchSize.0, shape[1], shape[2], shape[3]], of: FloatType.self)
      result.full(0)
      result[0..<min(shape[0], batchSize.0), 0..<shape[1], 0..<shape[2], 0..<shape[3]] =
        encodedImage[0..<min(shape[0], batchSize.0), 0..<shape[1], 0..<shape[2], 0..<shape[3]]
      return result
    case .wan21_1_3b, .wan21_14b:
      // For this mask, it contains 4 channels, each channel represent a frame (4x compression). First frame will use all 4 channels.
      let shape = encodedImage.shape
      var result = graph.variable(
        encodedImage.kind, format: encodedImage.format,
        shape: [shape[0], shape[1], shape[2], 4 + shape[3]], of: FloatType.self)
      result.full(0)
      result[0..<shape[0], 0..<shape[1], 0..<shape[2], 4..<(4 + shape[3])] = encodedImage
      if hasImage {
        let firstFrames = min(max(1, shape[0] - (batchSize.0 - batchSize.1) + 1), shape[0])
        result[0..<firstFrames, 0..<shape[1], 0..<shape[2], 0..<4].full(1)
      }
      return result
    case .hiDreamI1:
      fatalError()
    case .flux1:
      let shape = encodedImage.shape
      var result = graph.variable(
        encodedImage.kind, format: encodedImage.format,
        shape: [shape[0], shape[1], shape[2], 64 + shape[3]], of: FloatType.self)
      result[0..<shape[0], 0..<shape[1], 0..<shape[2], 0..<shape[3]] = encodedImage
      guard let imageNegMask = imageNegMask else {
        for i in 0..<64 {
          result[0..<shape[0], 0..<shape[1], 0..<shape[2], (i + shape[3])..<(i + shape[3] + 1)] =
            encodedMask
        }
        return result
      }
      let imageShuffledMask = (1 - imageNegMask).reshaped(
        format: .NHWC, shape: [shape[0], shape[1], 8, shape[2], 8]
      ).permuted(0, 1, 3, 2, 4).contiguous().reshaped(.NHWC(shape[0], shape[1], shape[2], 64))
      result[0..<shape[0], 0..<shape[1], 0..<shape[2], shape[3]..<(shape[3] + 64)] =
        imageShuffledMask
      return result
    }
  }

  private func generateTextOnly(
    _ image: Tensor<FloatType>?, scaleFactor imageScaleFactor: Int,
    depth: Tensor<FloatType>?, hints: [ControlHintType: AnyTensor], custom: Tensor<FloatType>?,
    shuffles: [(Tensor<FloatType>, Float)], poses: [(Tensor<FloatType>, Float)],
    text: String, negativeText: String, configuration: GenerationConfiguration,
    denoiserParameterization: Denoiser.Parameterization, sampling: Sampling,
    cancellation: (@escaping () -> Void) -> Void,
    feedback: @escaping (ImageGeneratorSignpost, Set<ImageGeneratorSignpost>, Tensor<FloatType>?)
      -> Bool
  ) -> ([Tensor<FloatType>]?, Int) {
    let coreMLGuard = modelPreloader.beginCoreMLGuard()
    defer {
      if coreMLGuard {
        modelPreloader.endCoreMLGuard()
      }
    }
    let mfaGuard = modelPreloader.beginMFAGuard()
    defer {
      if mfaGuard {
        modelPreloader.endMFAGuard()
      }
    }
    let file =
      (configuration.model.flatMap {
        ModelZoo.isModelDownloaded($0) ? $0 : nil
      }) ?? ModelZoo.defaultSpecification.file
    let modifier = ImageGeneratorUtils.modifierForModel(
      file, LoRAs: configuration.loras.compactMap(\.file))
    let modelVersion = ModelZoo.versionForModel(file)
    let textEncoderVersion = ModelZoo.textEncoderVersionForModel(file)
    // generateTextOnly cannot handle I2v model.
    guard modelVersion != .svdI2v else {
      return (nil, 1)
    }
    let modelObjective = ModelZoo.objectiveForModel(file)
    let modelUpcastAttention = ModelZoo.isUpcastAttentionForModel(file)
    var textEncoderFiles: [String] =
      [
        ModelZoo.textEncoderForModel(file).flatMap {
          ModelZoo.isModelDownloaded($0) ? $0 : nil
        } ?? ImageGeneratorUtils.defaultTextEncoder
      ]
      + ModelZoo.CLIPEncodersForModel(file).compactMap { ModelZoo.isModelDownloaded($0) ? $0 : nil }
    textEncoderFiles +=
      ((ModelZoo.T5EncoderForModel(file).flatMap { ModelZoo.isModelDownloaded($0) ? $0 : nil }).map
      { [$0] } ?? [])
    let diffusionMappingFile = ModelZoo.diffusionMappingForModel(file).flatMap {
      ModelZoo.isModelDownloaded($0) ? $0 : nil
    }
    let fpsId = Int(configuration.fpsId)
    let motionBucketId = Int(configuration.motionBucketId)
    let condAug = configuration.condAug
    let startFrameCfg = configuration.startFrameCfg
    let clipSkip = Int(configuration.clipSkip)
    let autoencoderFile =
      ModelZoo.autoencoderForModel(file).flatMap {
        ModelZoo.isModelDownloaded($0) ? $0 : nil
      } ?? ImageGeneratorUtils.defaultAutoencoder
    // Always enable for Hunyuan.
    let isGuidanceEmbedEnabled =
      ModelZoo.guidanceEmbedForModel(file) && configuration.speedUpWithGuidanceEmbed
    var isCfgEnabled = !ModelZoo.isConsistencyModelForModel(file) && !isGuidanceEmbedEnabled
    let latentsScaling = ModelZoo.latentsScalingForModel(file)
    let paddedTextEncodingLength = ModelZoo.paddedTextEncodingLengthForModel(file)
    let conditioning = ModelZoo.conditioningForModel(file)
    let refinerVersion: ModelVersion? = configuration.refinerModel.flatMap {
      guard $0 != file, ModelZoo.isModelDownloaded($0) else { return nil }
      let version = ModelZoo.versionForModel($0)
      guard
        version == modelVersion
          || ([.sdxlBase, .sdxlRefiner, .ssd1b].contains(version)
            && [.sdxlBase, .sdxlRefiner, .ssd1b].contains(modelVersion))
      else { return nil }
      return ModelZoo.versionForModel($0)
    }
    let tiledDecoding = TiledConfiguration(
      isEnabled: configuration.tiledDecoding,
      tileSize: .init(
        width: Int(configuration.decodingTileWidth), height: Int(configuration.decodingTileHeight)),
      tileOverlap: Int(configuration.decodingTileOverlap))
    let tiledDiffusion = TiledConfiguration(
      isEnabled: configuration.tiledDiffusion,
      tileSize: .init(
        width: Int(configuration.diffusionTileWidth), height: Int(configuration.diffusionTileHeight)
      ), tileOverlap: Int(configuration.diffusionTileOverlap))
    var alternativeDecoderFilePath: String? = nil
    var alternativeDecoderVersion: AlternativeDecoderVersion? = nil
    let lora: [LoRAConfiguration] =
      (ModelZoo.builtinLoRAForModel(file)
        ? [
          LoRAConfiguration(
            file: ModelZoo.filePathForModelDownloaded(file), weight: 1, version: modelVersion,
            isLoHa: false, modifier: .none)
        ] : [])
      + configuration.loras.compactMap {
        guard let file = $0.file else { return nil }
        let loraVersion = LoRAZoo.versionForModel(file)
        guard LoRAZoo.isModelDownloaded(file),
          modelVersion == loraVersion || refinerVersion == loraVersion
            || (modelVersion == .kandinsky21 && loraVersion == .v1)
        else { return nil }
        if LoRAZoo.isConsistencyModelForModel(file) {
          isCfgEnabled = false
        }
        if let alternativeDecoder = LoRAZoo.alternativeDecoderForModel(file) {
          alternativeDecoderFilePath = LoRAZoo.filePathForModelDownloaded(alternativeDecoder.0)
          alternativeDecoderVersion = alternativeDecoder.1
        }
        return LoRAConfiguration(
          file: LoRAZoo.filePathForModelDownloaded(file), weight: $0.weight, version: loraVersion,
          isLoHa: LoRAZoo.isLoHaForModel(file), modifier: LoRAZoo.modifierForModel(file))
      }
    if modelVersion == .v2 || modelVersion == .sdxlBase || modelVersion == .sdxlRefiner
      || modelVersion == .ssd1b || modelVersion == .svdI2v || modelVersion == .wurstchenStageC
      || modelVersion == .sd3 || modelVersion == .pixart
    {
      DynamicGraph.flags = .disableMixedMPSGEMM
    }
    if !DeviceCapability.isMemoryMapBufferSupported {
      DynamicGraph.flags.insert(.disableMmapMTLBuffer)
    }
    let isMFAEnabled = DeviceCapability.isMFAEnabled.load(ordering: .acquiring)
    if !isMFAEnabled {
      DynamicGraph.flags.insert(.disableMetalFlashAttention)
    } else {
      DynamicGraph.flags.remove(.disableMetalFlashAttention)
      if !DeviceCapability.isMFAGEMMFaster {
        DynamicGraph.flags.insert(.disableMFAGEMM)
      }
    }
    var signposts = Set<ImageGeneratorSignpost>([
      .textEncoded, .sampling(sampling.steps), .imageDecoded,
    ])
    if modifier == .inpainting || modifier == .editing || modifier == .double {
      signposts.insert(.imageEncoded)
    }
    if let faceRestoration = configuration.faceRestoration,
      EverythingZoo.isModelDownloaded(faceRestoration)
        && EverythingZoo.isModelDownloaded(EverythingZoo.parsenetForModel(faceRestoration))
    {
      signposts.insert(.faceRestored)
    }
    if let upscaler = configuration.upscaler, UpscalerZoo.isModelDownloaded(upscaler) {
      signposts.insert(.imageUpscaled)
    }
    var hasHints = Set(hints.keys)
    if !poses.isEmpty {
      hasHints.insert(.pose)
    }
    let hasImage = image != nil
    let (
      canInjectControls, canInjectT2IAdapters, canInjectAttentionKVs, _, injectIPAdapterLengths,
      canInjectedControls
    ) =
      ImageGeneratorUtils.canInjectControls(
        hasImage: hasImage, hasDepth: depth != nil, hasHints: hasHints,
        hasCustom: custom != nil,
        shuffleCount: shuffles.count, controls: configuration.controls,
        version: modelVersion)
    let isQuantizedModel = ModelZoo.isQuantizedModel(file)
    let (qkNorm, dualAttentionLayers, distilledGuidanceLayers) =
      ModelZoo.MMDiTForModel(file).map {
        return ($0.qkNorm, $0.dualAttentionLayers, $0.distilledGuidanceLayers ?? 0)
      } ?? (false, [], 0)
    let is8BitModel = ModelZoo.is8BitModel(file)
    let canRunLoRASeparately = modelPreloader.canRunLoRASeparately
    let externalOnDemand = modelPreloader.externalOnDemand(
      version: modelVersion,
      scale: DeviceCapability.Scale(
        widthScale: configuration.startWidth, heightScale: configuration.startHeight),
      variant: .unet, injectedControls: 0,
      is8BitModel: is8BitModel && (canRunLoRASeparately || lora.isEmpty))
    let textEncoderExternalOnDemand = modelPreloader.externalOnDemand(
      version: modelVersion,
      scale: DeviceCapability.Scale(
        widthScale: configuration.startWidth, heightScale: configuration.startHeight),
      variant: .textEncoder, injectedControls: 0)
    let vaeExternalOnDemand = modelPreloader.externalOnDemand(
      version: modelVersion,
      scale: DeviceCapability.Scale(
        widthScale: configuration.startWidth, heightScale: configuration.startHeight),
      variant: .autoencoder, injectedControls: 0)
    let dmExternalOnDemand = modelPreloader.externalOnDemand(
      version: modelVersion,
      scale: DeviceCapability.Scale(
        widthScale: configuration.startWidth, heightScale: configuration.startHeight),
      variant: .diffusionMapping, injectedControls: 0)
    let batchSize =
      ImageGeneratorUtils.isVideoModel(modelVersion) ? 1 : Int(configuration.batchSize)
    precondition(batchSize > 0)
    let textGuidanceScale = configuration.guidanceScale
    let imageGuidanceScale = configuration.imageGuidanceScale
    let guidanceEmbed = configuration.guidanceEmbed
    let originalSize =
      configuration.originalImageWidth == 0 || configuration.originalImageHeight == 0
      ? (width: Int(configuration.startWidth) * 64, height: Int(configuration.startHeight) * 64)
      : (
        width: Int(configuration.originalImageWidth), height: Int(configuration.originalImageHeight)
      )
    let cropTopLeft = (top: Int(configuration.cropTop), left: Int(configuration.cropLeft))
    let targetSize =
      configuration.targetImageWidth == 0 || configuration.targetImageHeight == 0
      ? (width: Int(configuration.startWidth) * 64, height: Int(configuration.startHeight) * 64)
      : (width: Int(configuration.targetImageWidth), height: Int(configuration.targetImageHeight))
    let negativeOriginalSize =
      configuration.negativeOriginalImageWidth == 0
        || configuration.negativeOriginalImageHeight == 0
      ? originalSize
      : (
        width: Int(configuration.negativeOriginalImageWidth),
        height: Int(configuration.negativeOriginalImageHeight)
      )
    let aestheticScore = configuration.aestheticScore
    let negativeAestheticScore = configuration.negativeAestheticScore
    let zeroNegativePrompt = configuration.zeroNegativePrompt
    let sharpness = configuration.sharpness
    let refiner: Refiner? = configuration.refinerModel.flatMap {
      guard $0 != file, ModelZoo.isModelDownloaded($0) else { return nil }
      let version = ModelZoo.versionForModel($0)
      guard
        version == modelVersion
          || ([.sdxlBase, .sdxlRefiner, .ssd1b].contains(version)
            && [.sdxlBase, .sdxlRefiner, .ssd1b].contains(modelVersion))
      else { return nil }
      return Refiner(
        start: configuration.refinerStart, filePath: ModelZoo.filePathForModelDownloaded($0),
        externalOnDemand: externalOnDemand, version: ModelZoo.versionForModel($0),
        isQuantizedModel: ModelZoo.isQuantizedModel($0),
        isConsistencyModel: ModelZoo.isConsistencyModelForModel($0))
    }
    let hiresFixStrength = configuration.hiresFixStrength
    let queueWatermark = DynamicGraph.queueWatermark
    if (canInjectControls && modelVersion == .v2) && !DeviceCapability.isMaxPerformance {
      DynamicGraph.queueWatermark = 8
    }
    defer {
      if (canInjectControls && modelVersion == .v2) && !DeviceCapability.isMaxPerformance {
        DynamicGraph.queueWatermark = queueWatermark
      }
    }
    let teaCache =
      ModelZoo.teaCacheCoefficientsForModel(file).map {
        var teaCacheEnd =
          configuration.teaCacheEnd < 0
          ? Int(configuration.steps) + 1 + Int(configuration.teaCacheEnd)
          : Int(configuration.teaCacheEnd)
        let teaCacheStart = min(max(Int(configuration.teaCacheStart), 0), Int(configuration.steps))
        teaCacheEnd = min(max(max(teaCacheStart, teaCacheEnd), 0), Int(configuration.steps))
        return TeaCacheConfiguration(
          coefficients: $0,
          steps: min(teaCacheStart, teaCacheEnd)...max(teaCacheStart, teaCacheEnd),
          threshold: configuration.teaCache ? configuration.teaCacheThreshold : 0,
          maxSkipSteps: Int(configuration.teaCacheMaxSkipSteps))
      }
      ?? TeaCacheConfiguration(
        coefficients: (0, 0, 0, 0, 0), steps: 0...0, threshold: 0, maxSkipSteps: 0)
    let causalInference =
      configuration.causalInferenceEnabled ? Int(configuration.causalInference) : 0
    let sampler = LocalImageGenerator.sampler(
      from: configuration.sampler, isCfgEnabled: isCfgEnabled,
      filePath: ModelZoo.filePathForModelDownloaded(file), modifier: modifier,
      version: modelVersion, qkNorm: qkNorm, dualAttentionLayers: dualAttentionLayers,
      distilledGuidanceLayers: distilledGuidanceLayers, usesFlashAttention: isMFAEnabled,
      objective: modelObjective,
      upcastAttention: modelUpcastAttention,
      externalOnDemand: externalOnDemand, injectControls: canInjectControls,
      injectT2IAdapters: canInjectT2IAdapters, injectAttentionKV: canInjectAttentionKVs,
      injectIPAdapterLengths: injectIPAdapterLengths,
      lora: lora, isGuidanceEmbedEnabled: isGuidanceEmbedEnabled,
      isQuantizedModel: isQuantizedModel,
      canRunLoRASeparately: canRunLoRASeparately,
      stochasticSamplingGamma: configuration.stochasticSamplingGamma,
      conditioning: conditioning, parameterization: denoiserParameterization,
      tiledDiffusion: tiledDiffusion, teaCache: teaCache, causalInference: causalInference,
      weightsCache: weightsCache,
      of: FloatType.self)
    let initTimestep = sampler.timestep(for: hiresFixStrength, sampling: sampling)
    let hiresFixEnabled =
      configuration.hiresFix && initTimestep.startStep > 0 && configuration.hiresFixStartWidth > 0
      && configuration.hiresFixStartHeight > 0
      && configuration.hiresFixStartWidth < configuration.startWidth
      && configuration.hiresFixStartHeight < configuration.startHeight
    let firstPassStartWidth: Int
    let firstPassStartHeight: Int
    let firstPassChannels: Int
    if modelVersion == .wurstchenStageC {
      (firstPassStartWidth, firstPassStartHeight) = stageCLatentsSize(configuration)
      firstPassChannels = 16
    } else {
      firstPassStartWidth =
        (hiresFixEnabled ? Int(configuration.hiresFixStartWidth) : Int(configuration.startWidth))
        * 8
      firstPassStartHeight =
        (hiresFixEnabled ? Int(configuration.hiresFixStartHeight) : Int(configuration.startHeight))
        * 8
      switch modelVersion {
      case .wurstchenStageC, .sd3, .sd3Large, .flux1, .hunyuanVideo, .wan21_1_3b, .wan21_14b,
        .hiDreamI1:
        firstPassChannels = 16
      case .auraflow, .kandinsky21, .pixart, .sdxlBase, .sdxlRefiner, .ssd1b, .svdI2v, .v1, .v2,
        .wurstchenStageB:
        firstPassChannels = 4
      }
    }
    let firstPassScale = DeviceCapability.Scale(
      widthScale: hiresFixEnabled ? configuration.hiresFixStartWidth : configuration.startWidth,
      heightScale: hiresFixEnabled ? configuration.hiresFixStartHeight : configuration.startHeight)
    let isHighPrecisionVAEFallbackEnabled = DeviceCapability.isHighPrecisionVAEFallbackEnabled(
      scale: firstPassScale)
    let controlExternalOnDemand = modelPreloader.externalOnDemand(
      version: modelVersion, scale: firstPassScale, variant: .control,
      injectedControls: canInjectedControls, suffix: "ctrl")
    let imageScale = DeviceCapability.Scale(
      widthScale: configuration.startWidth, heightScale: configuration.startHeight)
    if hiresFixEnabled || modelVersion == .wurstchenStageC {
      signposts.insert(.secondPassImageEncoded)
      if modelVersion == .wurstchenStageC {
        signposts.insert(.secondPassSampling(Int(configuration.stage2Steps)))
      } else {
        signposts.insert(
          .secondPassSampling(sampling.steps - initTimestep.roundedDownStartStep))
      }
      signposts.insert(.secondPassImageDecoded)
    }
    let highPrecisionForAutoencoder = ModelZoo.isHighPrecisionAutoencoderForModel(file)
    let graph = DynamicGraph()
    if externalOnDemand
      || externalOnDemandPartially(
        version: modelVersion, memoryCapacity: DeviceCapability.memoryCapacity,
        externalOnDemand: externalOnDemand)
    {
      TensorData.makeExternalData(for: ModelZoo.filePathForModelDownloaded(file), graph: graph)
      for stageModel in ModelZoo.stageModelsForModel(file) {
        TensorData.makeExternalData(
          for: ModelZoo.filePathForModelDownloaded(stageModel), graph: graph)
      }
    }
    if textEncoderExternalOnDemand {
      TensorData.makeExternalData(
        for: ModelZoo.filePathForModelDownloaded(textEncoderFiles[0]), graph: graph)
    }
    if controlExternalOnDemand {
      for file in Set(configuration.controls.compactMap { $0.file }) {
        TensorData.makeExternalData(for: ModelZoo.filePathForModelDownloaded(file), graph: graph)
      }
    }
    if vaeExternalOnDemand {
      TensorData.makeExternalData(
        for: ModelZoo.filePathForModelDownloaded(autoencoderFile), graph: graph)
    }
    if dmExternalOnDemand, let diffusionMappingFile = diffusionMappingFile {
      TensorData.makeExternalData(
        for: ModelZoo.filePathForModelDownloaded(diffusionMappingFile), graph: graph)
    }
    let potentials = lora.map { ($0.file as NSString).lastPathComponent }
    let (
      tokensTensors, positionTensors, embedMask, injectedEmbeddings, unconditionalAttentionWeights,
      attentionWeights, hasNonOneWeights, tokenLengthUncond, tokenLengthCond, lengthsOfUncond,
      lengthsOfCond
    ) = tokenize(
      graph: graph, modelVersion: modelVersion, textEncoderVersion: textEncoderVersion,
      paddedTextEncodingLength: paddedTextEncodingLength, text: text, negativeText: negativeText,
      negativePromptForImagePrior: configuration.negativePromptForImagePrior,
      potentials: potentials, T5TextEncoder: configuration.t5TextEncoder,
      clipL: configuration.separateClipL ? (configuration.clipLText ?? "") : nil,
      openClipG: configuration.separateOpenClipG ? (configuration.openClipGText ?? "") : nil,
      t5: configuration.separateT5 ? (configuration.t5Text ?? "") : nil
    )
    return graph.withNoGrad {
      let injectedTextEmbeddings = generateInjectedTextEmbeddings(
        batchSize: batchSize, startHeight: firstPassStartHeight, startWidth: firstPassStartWidth,
        image: image,
        graph: graph, hints: hints, custom: custom, shuffles: shuffles, pose: poses.first?.0,
        controls: configuration.controls,
        version: modelVersion, tiledDiffusion: tiledDiffusion, usesFlashAttention: isMFAEnabled,
        externalOnDemand: controlExternalOnDemand, cancellation: cancellation)
      let (tokenLengthUncond, tokenLengthCond) = ControlModel<FloatType>.modifyTextEmbeddings(
        tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
        injecteds: injectedTextEmbeddings)
      let tokenLength = max(tokenLengthUncond, tokenLengthCond)
      let textEncoder = TextEncoder<FloatType>(
        filePaths: textEncoderFiles.map { ModelZoo.filePathForModelDownloaded($0) },
        version: modelVersion, textEncoderVersion: textEncoderVersion,
        isCfgEnabled: isCfgEnabled,
        usesFlashAttention: isMFAEnabled && DeviceCapability.isMFACausalAttentionMaskSupported,
        injectEmbeddings: !injectedEmbeddings.isEmpty,
        externalOnDemand: textEncoderExternalOnDemand,
        deviceProperties: DeviceCapability.deviceProperties, weightsCache: weightsCache,
        maxLength: tokenLength, clipSkip: clipSkip, lora: lora)
      let image = image.map {
        downscaleImageAndToGPU(graph.variable($0), scaleFactor: imageScaleFactor)
      }
      let textEncodings = modelPreloader.consumeTextModels(
        textEncoder.encode(
          tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
          tokens: tokensTensors, positions: positionTensors, mask: embedMask,
          injectedEmbeddings: injectedEmbeddings, image: image.map { [$0] } ?? [],
          lengthsOfUncond: lengthsOfUncond,
          lengthsOfCond: lengthsOfCond, injectedTextEmbeddings: injectedTextEmbeddings,
          textModels: modelPreloader.retrieveTextModels(textEncoder: textEncoder)),
        textEncoder: textEncoder)
      var c: [DynamicGraph.Tensor<FloatType>]
      var extraProjection: DynamicGraph.Tensor<FloatType>?
      DynamicGraph.setSeed(configuration.seed)
      if modelVersion == .kandinsky21, let diffusionMappingFile = diffusionMappingFile {
        let diffusionMapping = DiffusionMapping<FloatType>(
          filePath: ModelZoo.filePathForModelDownloaded(diffusionMappingFile),
          usesFlashAttention: isMFAEnabled && DeviceCapability.isMFACausalAttentionMaskSupported,
          steps: Int(configuration.imagePriorSteps),
          negativePromptForImagePrior: configuration.negativePromptForImagePrior,
          CLIPWeight: configuration.clipWeight, externalOnDemand: dmExternalOnDemand)
        let imageEmb = diffusionMapping.sample(
          textEncoding: textEncodings[2], textEmbedding: textEncodings[3], tokens: tokensTensors[1])
        let kandinskyEmbedding = KandinskyEmbedding<FloatType>(
          filePath: ModelZoo.filePathForModelDownloaded(file))
        let (xfProj, xfOut) = kandinskyEmbedding.encode(
          textEncoding: textEncodings[0], textEmbedding: textEncodings[1], imageEmbedding: imageEmb)
        extraProjection = xfProj
        c = [xfOut]
      } else {
        extraProjection = nil
        c = textEncodings
      }
      (c, extraProjection) = repeatConditionsToMatchBatchSize(
        c: c, extraProjection: extraProjection,
        unconditionalAttentionWeights: unconditionalAttentionWeights,
        attentionWeights: attentionWeights, version: modelVersion, tokenLength: tokenLength,
        batchSize: batchSize, hasNonOneWeights: hasNonOneWeights)
      guard feedback(.textEncoded, signposts, nil) else { return (nil, 1) }
      var maskedImage: DynamicGraph.Tensor<FloatType>? = nil
      var mask: DynamicGraph.Tensor<FloatType>? = nil
      var firstPassImage: DynamicGraph.Tensor<FloatType>? = nil
      if modifier == .inpainting || modifier == .editing || modifier == .double
        || modifier == .depth || modifier == .canny || canInjectControls
        || canInjectT2IAdapters || !injectIPAdapterLengths.isEmpty
      {
        // TODO: This needs to be properly handled for Wurstchen (i.e. using EfficientNet to encode image).
        firstPassImage =
          (image.map {
            let imageHeight = $0.shape[1]
            let imageWidth = $0.shape[2]
            if imageHeight == firstPassStartHeight * 8 && imageWidth == firstPassStartWidth * 8 {
              return $0
            }
            return Upsample(
              .bilinear, widthScale: Float(firstPassStartWidth * 8) / Float(imageWidth),
              heightScale: Float(firstPassStartHeight * 8) / Float(imageHeight))($0)
          })
      }
      var firstStage = FirstStage<FloatType>(
        filePath: ModelZoo.filePathForModelDownloaded(autoencoderFile), version: modelVersion,
        latentsScaling: latentsScaling, highPrecisionKeysAndValues: highPrecisionForAutoencoder,
        highPrecisionFallback: isHighPrecisionVAEFallbackEnabled,
        tiledDecoding: tiledDecoding, tiledDiffusion: tiledDiffusion,
        externalOnDemand: vaeExternalOnDemand, alternativeUsesFlashAttention: isMFAEnabled,
        alternativeFilePath: alternativeDecoderFilePath,
        alternativeDecoderVersion: alternativeDecoderVersion,
        deviceProperties: DeviceCapability.deviceProperties)
      var batchSize = (batchSize, 0)
      switch modelVersion {
      case .svdI2v:
        batchSize = (Int(configuration.numFrames), 0)
      case .hunyuanVideo, .wan21_1_3b, .wan21_14b:
        batchSize = injectReferenceFrames(
          batchSize: ((Int(configuration.numFrames) - 1) / 4) + 1, version: modelVersion,
          canInjectControls: canInjectControls, shuffleCount: shuffles.count,
          hasCustom: custom != nil)
      case .auraflow, .flux1, .kandinsky21, .pixart, .sd3, .sd3Large, .sdxlBase, .sdxlRefiner,
        .ssd1b, .v1, .v2, .wurstchenStageB, .wurstchenStageC, .hiDreamI1:
        break
      }
      if modifier == .inpainting || modifier == .editing || modifier == .double {
        // Only apply the image fill logic (for image encoding purpose) when it is inpainting or editing.
        var firstPassImage =
          firstPassImage
          ?? {
            let image = graph.variable(
              .GPU(0), .NHWC(1, firstPassStartHeight * 8, firstPassStartWidth * 8, 3),
              of: FloatType.self)
            image.full(0)
            return image
          }()
        let imageSize: Int
        (imageSize, firstPassImage, _) = expandImageForEncoding(
          batchSize: batchSize, version: modelVersion, modifier: modifier, image: firstPassImage)
        let encodedImage = modelPreloader.consumeFirstStageEncode(
          firstStage.encode(
            firstPassImage,
            encoder: modelPreloader.retrieveFirstStageEncoder(
              firstStage: firstStage, scale: firstPassScale), cancellation: cancellation),
          firstStage: firstStage,
          scale: firstPassScale)
        if modifier == .inpainting {
          maskedImage = firstStage.scale(
            encodedImage[
              0..<imageSize, 0..<firstPassStartHeight, 0..<firstPassStartWidth,
              0..<firstPassChannels
            ].copied())
          mask = graph.variable(
            .GPU(0), .NHWC(1, firstPassStartHeight, firstPassStartWidth, 1), of: FloatType.self)
          mask?.full(1)
          maskedImage = concatMaskWithMaskedImage(
            hasImage: hasImage, batchSize: batchSize, version: modelVersion,
            encodedImage: maskedImage!, encodedMask: mask!, imageNegMask: nil
          )
        } else if modifier == .editing {
          if modelVersion == .v1 {
            maskedImage = encodedImage[
              0..<imageSize, 0..<firstPassStartHeight, 0..<firstPassStartWidth,
              0..<firstPassChannels
            ]
            .copied()
          } else {
            maskedImage = firstStage.scale(
              encodedImage[
                0..<imageSize, 0..<firstPassStartHeight, 0..<firstPassStartWidth,
                0..<firstPassChannels
              ]
              .copied())
          }
        } else {
          maskedImage = encodedImage[
            0..<imageSize, 0..<firstPassStartHeight, 0..<firstPassStartWidth, 0..<firstPassChannels
          ].copied()
        }
        guard feedback(.imageEncoded, signposts, nil) else { return (nil, 1) }
      }
      let x_T = randomLatentNoise(
        graph: graph, batchSize: batchSize.0, startHeight: firstPassStartHeight,
        startWidth: firstPassStartWidth, channels: firstPassChannels, seed: configuration.seed,
        seedMode: configuration.seedMode)
      let depthImage = depth.map { graph.variable($0.toGPU(0)) }
      let firstPassDepthImage = depthImage.map {
        let depthHeight = $0.shape[1]
        let depthWidth = $0.shape[2]
        guard depthHeight != firstPassStartHeight * 8 || depthWidth != firstPassStartWidth * 8
        else {
          return $0
        }
        return Upsample(
          .bilinear, widthScale: Float(firstPassStartHeight * 8) / Float(depthHeight),
          heightScale: Float(firstPassStartWidth * 8) / Float(depthWidth))($0)
      }
      let customImage = custom.map { graph.variable($0.toGPU(0)) }
      let firstPassCustomImage = customImage.map {
        let customHeight = $0.shape[1]
        let customWidth = $0.shape[2]
        guard customHeight != firstPassStartHeight * 8 || customWidth != firstPassStartWidth * 8
        else {
          return $0
        }
        return Upsample(
          .bilinear, widthScale: Float(firstPassStartHeight * 8) / Float(customHeight),
          heightScale: Float(firstPassStartWidth * 8) / Float(customWidth))($0)
      }
      let firstPassImageCond = encodeImageCond(
        image: firstPassImage, depth: firstPassDepthImage, custom: firstPassCustomImage,
        modifier: modifier, version: modelVersion, firstStage: firstStage,
        usesFlashAttention: isMFAEnabled)

      let injectedControls = generateInjectedControls(
        graph: graph, batchSize: batchSize.0, startHeight: firstPassStartHeight,
        startWidth: firstPassStartWidth,
        image: firstPassImage, depth: firstPassDepthImage, hints: hints, custom: custom,
        shuffles: shuffles, pose: poses.first?.0, mask: nil, controls: configuration.controls,
        version: modelVersion, tiledDiffusion: tiledDiffusion, usesFlashAttention: isMFAEnabled,
        externalOnDemand: controlExternalOnDemand, steps: sampling.steps, firstStage: firstStage,
        cancellation: cancellation)
      guard feedback(.controlsGenerated, signposts, nil) else { return (nil, 1) }

      if let image = maskedImage {
        maskedImage = injectVACEFrames(
          batchSize: batchSize, version: modelVersion, image: image,
          injectedControls: injectedControls)
      }
      guard
        let x =
          try? modelPreloader.consumeUNet(
            sampler.sample(
              x_T,
              unets: modelPreloader.retrieveUNet(
                sampler: sampler, scale: firstPassScale, tokenLengthUncond: tokenLengthUncond,
                tokenLengthCond: tokenLengthCond), sample: nil,
              conditionImage: maskedImage ?? firstPassImageCond,
              mask: mask, negMask: nil, conditioning: c, tokenLengthUncond: tokenLengthUncond,
              tokenLengthCond: tokenLengthCond, extraProjection: extraProjection,
              injectedControls: injectedControls, textGuidanceScale: textGuidanceScale,
              imageGuidanceScale: imageGuidanceScale, guidanceEmbed: guidanceEmbed,
              startStep: (integral: 0, fractional: 0),
              endStep: (
                integral: sampling.steps,
                fractional: Float(sampling.steps)
              ), originalSize: originalSize,
              cropTopLeft: cropTopLeft, targetSize: targetSize, aestheticScore: aestheticScore,
              negativeOriginalSize: negativeOriginalSize,
              negativeAestheticScore: negativeAestheticScore,
              zeroNegativePrompt: zeroNegativePrompt, refiner: refiner, fpsId: fpsId,
              motionBucketId: motionBucketId, condAug: condAug, startFrameCfg: startFrameCfg,
              sharpness: sharpness, sampling: sampling, cancellation: cancellation
            ) { step, tensor in
              feedback(.sampling(step), signposts, tensor)
            }, sampler: sampler, scale: firstPassScale, tokenLengthUncond: tokenLengthUncond,
            tokenLengthCond: tokenLengthCond)
      else {
        return (nil, 1)
      }
      guard feedback(.sampling(sampling.steps), signposts, nil) else {
        return (nil, 1)
      }
      let isHighPrecisionVAEFallbackEnabled = DeviceCapability.isHighPrecisionVAEFallbackEnabled(
        scale: imageScale)

      firstStage = FirstStage<FloatType>(
        filePath: ModelZoo.filePathForModelDownloaded(autoencoderFile), version: modelVersion,
        latentsScaling: latentsScaling, highPrecisionKeysAndValues: highPrecisionForAutoencoder,
        highPrecisionFallback: isHighPrecisionVAEFallbackEnabled,
        tiledDecoding: tiledDecoding, tiledDiffusion: tiledDiffusion,
        externalOnDemand: vaeExternalOnDemand, alternativeUsesFlashAttention: isMFAEnabled,
        alternativeFilePath: alternativeDecoderFilePath,
        alternativeDecoderVersion: alternativeDecoderVersion,
        deviceProperties: DeviceCapability.deviceProperties)

      if DeviceCapability.isLowPerformance {
        graph.garbageCollect()
      }
      if modelVersion == .v2 || modelVersion == .sdxlBase || modelVersion == .sdxlRefiner
        || modelVersion == .ssd1b || modelVersion == .wurstchenStageC
      {
        DynamicGraph.flags = []
      }
      if !DeviceCapability.isMemoryMapBufferSupported {
        DynamicGraph.flags.insert(.disableMmapMTLBuffer)
      }
      if !isMFAEnabled {
        DynamicGraph.flags.insert(.disableMetalFlashAttention)
      } else {
        DynamicGraph.flags.remove(.disableMetalFlashAttention)
        if !DeviceCapability.isMFAGEMMFaster {
          DynamicGraph.flags.insert(.disableMFAGEMM)
        }
      }
      var firstStageResult: DynamicGraph.Tensor<FloatType>
      if modelVersion == .wurstchenStageC {
        firstStageResult = x
      } else {
        // For Wurstchen model, we don't need to run decode.
        firstStageResult = modelPreloader.consumeFirstStageDecode(
          firstStage.decode(
            x, batchSize: (hiresFixEnabled ? batchSize : (batchSize.0 - batchSize.1, 0)),
            decoder: modelPreloader.retrieveFirstStageDecoder(
              firstStage: firstStage, scale: firstPassScale), cancellation: cancellation),
          firstStage: firstStage, scale: firstPassScale)
        guard !isNaN(firstStageResult.rawValue.toCPU()) else { return (nil, 1) }
      }
      guard feedback(.imageDecoded, signposts, nil) else { return (nil, 1) }
      // We go through second image sampling with Wurstchen.
      guard hiresFixEnabled || modelVersion == .wurstchenStageC else {
        firstStageResult = faceRestoreImage(firstStageResult, configuration: configuration)
        if signposts.contains(.faceRestored) {
          guard feedback(.faceRestored, signposts, nil) else { return (nil, 1) }
        }
        let (result, scaleFactor) = upscaleImageAndToCPU(
          firstStageResult, configuration: configuration)
        if signposts.contains(.imageUpscaled) {
          let _ = feedback(.imageUpscaled, signposts, nil)
        }
        var batchSize = batchSize.0
        if ImageGeneratorUtils.isVideoModel(modelVersion) {
          batchSize = Int(configuration.numFrames)
        }
        guard batchSize > 1 else {
          return ([result], scaleFactor)
        }
        var batch = [Tensor<FloatType>]()
        let shape = result.shape
        for i in 0..<min(batchSize, shape[0]) {
          batch.append(result[i..<(i + 1), 0..<shape[1], 0..<shape[2], 0..<shape[3]].copied())
        }
        return (batch, scaleFactor)
      }
      let startWidth: Int
      let startHeight: Int
      if modelVersion == .wurstchenStageC {
        startWidth = Int(configuration.startWidth) * 16
        startHeight = Int(configuration.startHeight) * 16
      } else {
        startWidth = Int(configuration.startWidth) * 8
        startHeight = Int(configuration.startHeight) * 8
      }
      let firstStageImage: DynamicGraph.Tensor<FloatType>
      let sample: DynamicGraph.Tensor<FloatType>
      // Bypass decode / scale / encode for Wurstchen model.
      if modelVersion == .wurstchenStageC {
        firstStageImage = firstStageResult
        sample = firstStageResult
      } else {
        let shape = firstStageResult.shape
        if shape[3] > 3 {
          // Keep the last 3-components (RGB). Remove alpha channel.
          firstStageImage = Upsample(
            .bilinear, widthScale: Float(startWidth) / Float(firstPassStartWidth),
            heightScale: Float(startHeight) / Float(firstPassStartHeight))(
              firstStageResult[0..<shape[0], 0..<shape[1], 0..<shape[2], (shape[3] - 3)..<shape[3]]
                .copied())
        } else {
          firstStageImage = Upsample(
            .bilinear, widthScale: Float(startWidth) / Float(firstPassStartWidth),
            heightScale: Float(startHeight) / Float(firstPassStartHeight))(firstStageResult)
        }
        // encode image again.
        (sample, _, _) = firstStage.sample(
          DynamicGraph.Tensor<FloatType>(from: firstStageImage), individualFrames: batchSize.1,
          encoder: nil, cancellation: cancellation)
      }
      if modifier == .inpainting || modifier == .editing || modifier == .double {
        // TODO: Support this properly for Wurstchen models.
        var image =
          (image.flatMap {
            if $0.shape[1] == startHeight * 8 && $0.shape[2] == startWidth * 8 {
              return $0
            }
            return nil
          })
          ?? {
            let image = graph.variable(
              .GPU(0), .NHWC(1, startHeight * 8, startWidth * 8, 3),
              of: FloatType.self)
            image.full(0)
            return image
          }()
        let imageSize: Int
        (imageSize, image, _) = expandImageForEncoding(
          batchSize: (batchSize.0, 0), version: modelVersion, modifier: modifier, image: image)
        let encodedImage = modelPreloader.consumeFirstStageEncode(
          firstStage.encode(
            image,
            encoder: modelPreloader.retrieveFirstStageEncoder(
              firstStage: firstStage, scale: imageScale), cancellation: cancellation),
          firstStage: firstStage, scale: imageScale
        )
        if modifier == .inpainting {
          maskedImage = firstStage.scale(
            encodedImage[0..<imageSize, 0..<startHeight, 0..<startWidth, 0..<firstPassChannels]
              .copied())
          mask = graph.variable(.GPU(0), .NHWC(1, startHeight, startWidth, 1), of: FloatType.self)
          mask?.full(1)
          maskedImage = concatMaskWithMaskedImage(
            hasImage: hasImage, batchSize: batchSize, version: modelVersion,
            encodedImage: maskedImage!, encodedMask: mask!, imageNegMask: nil
          )
        } else if modifier == .editing {
          if modelVersion == .v1 {
            maskedImage = encodedImage[
              0..<imageSize, 0..<startHeight, 0..<startWidth,
              0..<firstPassChannels
            ]
            .copied()
          } else {
            maskedImage = firstStage.scale(
              encodedImage[
                0..<imageSize, 0..<startHeight, 0..<startWidth,
                0..<firstPassChannels
              ]
              .copied())
          }
        } else {
          maskedImage = encodedImage[
            0..<imageSize, 0..<startHeight, 0..<startWidth, 0..<firstPassChannels
          ]
          .copied()
        }
      }
      guard feedback(.secondPassImageEncoded, signposts, nil) else { return (nil, 1) }
      let secondPassDepthImage = depthImage.map {
        let depthHeight = $0.shape[1]
        let depthWidth = $0.shape[2]
        guard depthHeight != startHeight * 8 || depthWidth != startWidth * 8 else {
          return $0
        }
        return Upsample(
          .bilinear, widthScale: Float(startHeight * 8) / Float(depthHeight),
          heightScale: Float(startWidth * 8) / Float(depthWidth))($0)
      }
      let secondPassCustomImage = customImage.map {
        let customHeight = $0.shape[1]
        let customWidth = $0.shape[2]
        guard customHeight != startHeight * 8 || customWidth != startWidth * 8 else {
          return $0
        }
        return Upsample(
          .bilinear, widthScale: Float(startHeight * 8) / Float(customHeight),
          heightScale: Float(startWidth * 8) / Float(customWidth))($0)
      }
      let secondPassImageCond = encodeImageCond(
        image: image, depth: secondPassDepthImage, custom: secondPassCustomImage,
        modifier: modifier, version: modelVersion, firstStage: firstStage,
        usesFlashAttention: isMFAEnabled)
      let (
        canInjectControls, canInjectT2IAdapters, canInjectAttentionKVs, _, injectIPAdapterLengths,
        canInjectedControls
      ) =
        ImageGeneratorUtils.canInjectControls(
          hasImage: true, hasDepth: secondPassDepthImage != nil, hasHints: hasHints,
          hasCustom: custom != nil, shuffleCount: shuffles.count,
          controls: configuration.controls,
          version: modelVersion)
      let secondPassControlExternalOnDemand = modelPreloader.externalOnDemand(
        version: modelVersion,
        scale: DeviceCapability.Scale(
          widthScale: configuration.startWidth, heightScale: configuration.startHeight),
        variant: .control, injectedControls: canInjectedControls, suffix: "ctrl")
      if secondPassControlExternalOnDemand {
        for file in Set(configuration.controls.compactMap { $0.file }) {
          TensorData.makeExternalData(for: ModelZoo.filePathForModelDownloaded(file), graph: graph)
        }
      }

      let secondPassInjectedControls = generateInjectedControls(
        graph: graph, batchSize: batchSize.0, startHeight: startHeight, startWidth: startWidth,
        image: image ?? firstStageImage, depth: secondPassDepthImage, hints: hints, custom: custom,
        shuffles: shuffles, pose: poses.last?.0, mask: nil, controls: configuration.controls,
        version: modelVersion,
        tiledDiffusion: tiledDiffusion, usesFlashAttention: isMFAEnabled,
        externalOnDemand: secondPassControlExternalOnDemand, steps: sampling.steps,
        firstStage: firstStage, cancellation: cancellation)
      guard feedback(.controlsGenerated, signposts, nil) else { return (nil, 1) }

      let secondPassModelVersion: ModelVersion
      let secondPassModelFilePath: String
      if modelVersion == .wurstchenStageC {
        secondPassModelVersion = .wurstchenStageB
        secondPassModelFilePath = ModelZoo.filePathForModelDownloaded(
          ModelZoo.stageModelsForModel(file)[0])
      } else {
        secondPassModelVersion = modelVersion
        secondPassModelFilePath = ModelZoo.filePathForModelDownloaded(file)
      }
      let secondPassSampler = LocalImageGenerator.sampler(
        from: configuration.sampler, isCfgEnabled: isCfgEnabled,
        filePath: secondPassModelFilePath, modifier: modifier,
        version: secondPassModelVersion, qkNorm: qkNorm, dualAttentionLayers: dualAttentionLayers,
        distilledGuidanceLayers: distilledGuidanceLayers, usesFlashAttention: isMFAEnabled,
        objective: modelObjective,
        upcastAttention: modelUpcastAttention,
        externalOnDemand: externalOnDemand,
        injectControls: canInjectControls, injectT2IAdapters: canInjectT2IAdapters,
        injectAttentionKV: canInjectAttentionKVs,
        injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
        isGuidanceEmbedEnabled: isGuidanceEmbedEnabled, isQuantizedModel: isQuantizedModel,
        canRunLoRASeparately: canRunLoRASeparately,
        stochasticSamplingGamma: configuration.stochasticSamplingGamma,
        conditioning: conditioning, parameterization: denoiserParameterization,
        tiledDiffusion: tiledDiffusion, teaCache: teaCache, causalInference: causalInference,
        weightsCache: weightsCache,
        of: FloatType.self)
      let startStep: (integral: Int, fractional: Float)
      let xEnc: DynamicGraph.Tensor<FloatType>
      let secondPassTextGuidance: Float
      let secondPassSampling: Sampling
      if modelVersion == .wurstchenStageC {
        startStep = (integral: 0, fractional: 0)
        xEnc = randomLatentNoise(
          graph: graph, batchSize: batchSize.0, startHeight: startHeight,
          startWidth: startWidth, channels: 4, seed: configuration.seed,
          seedMode: configuration.seedMode)
        c.append(sample)
        secondPassTextGuidance = configuration.stage2Cfg
        secondPassSampling = Sampling(
          steps: Int(configuration.stage2Steps), shift: Double(configuration.stage2Shift))
      } else {
        startStep = (
          integral: initTimestep.roundedDownStartStep, fractional: initTimestep.startStep
        )
        let channels: Int
        switch modelVersion {
        case .wurstchenStageC, .sd3, .sd3Large, .flux1, .hunyuanVideo, .wan21_1_3b, .wan21_14b,
          .hiDreamI1:
          channels = 16
        case .auraflow, .kandinsky21, .pixart, .sdxlBase, .sdxlRefiner, .ssd1b, .svdI2v, .v1, .v2,
          .wurstchenStageB:
          channels = 4
        }
        let noise = graph.variable(
          .GPU(0), .NHWC(batchSize.0, startHeight, startWidth, channels), of: FloatType.self)
        noise.randn(std: 1, mean: 0)
        let sampleScaleFactor = secondPassSampler.sampleScaleFactor(
          at: initTimestep.startStep, sampling: sampling)
        let noiseScaleFactor = secondPassSampler.noiseScaleFactor(
          at: initTimestep.startStep, sampling: sampling)
        xEnc = sampleScaleFactor * sample + noiseScaleFactor * noise
        secondPassTextGuidance = textGuidanceScale
        secondPassSampling = sampling
      }
      if modelVersion == .v2 || modelVersion == .sdxlBase || modelVersion == .sdxlRefiner
        || modelVersion == .ssd1b || modelVersion == .svdI2v || modelVersion == .wurstchenStageC
        || modelVersion == .sd3 || modelVersion == .pixart
      {
        DynamicGraph.flags = .disableMixedMPSGEMM
      }
      if !DeviceCapability.isMemoryMapBufferSupported {
        DynamicGraph.flags.insert(.disableMmapMTLBuffer)
      }
      if !isMFAEnabled {
        DynamicGraph.flags.insert(.disableMetalFlashAttention)
      } else {
        DynamicGraph.flags.remove(.disableMetalFlashAttention)
        if !DeviceCapability.isMFAGEMMFaster {
          DynamicGraph.flags.insert(.disableMFAGEMM)
        }
      }
      guard
        let x =
          try? modelPreloader.consumeUNet(
            secondPassSampler.sample(
              xEnc,
              unets: modelPreloader.retrieveUNet(
                sampler: secondPassSampler, scale: imageScale, tokenLengthUncond: tokenLengthUncond,
                tokenLengthCond: tokenLengthCond),
              sample: nil, conditionImage: maskedImage ?? secondPassImageCond, mask: mask,
              negMask: nil, conditioning: c, tokenLengthUncond: tokenLengthUncond,
              tokenLengthCond: tokenLengthCond, extraProjection: extraProjection,
              injectedControls: secondPassInjectedControls,
              textGuidanceScale: secondPassTextGuidance,
              imageGuidanceScale: imageGuidanceScale, guidanceEmbed: guidanceEmbed,
              startStep: startStep,
              endStep: (
                integral: secondPassSampling.steps, fractional: Float(secondPassSampling.steps)
              ),
              originalSize: originalSize, cropTopLeft: cropTopLeft,
              targetSize: targetSize, aestheticScore: aestheticScore,
              negativeOriginalSize: negativeOriginalSize,
              negativeAestheticScore: negativeAestheticScore,
              zeroNegativePrompt: zeroNegativePrompt, refiner: refiner, fpsId: fpsId,
              motionBucketId: motionBucketId, condAug: condAug, startFrameCfg: startFrameCfg,
              sharpness: sharpness, sampling: secondPassSampling, cancellation: cancellation
            ) { step, tensor in
              feedback(.secondPassSampling(step), signposts, tensor)
            }, sampler: secondPassSampler, scale: imageScale,
            tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond)
      else {
        return (nil, 1)
      }
      guard
        feedback(
          .secondPassSampling(secondPassSampling.steps - initTimestep.roundedDownStartStep),
          signposts, nil)
      else {
        return (nil, 1)
      }
      if modelVersion == .v2 || modelVersion == .sdxlBase || modelVersion == .sdxlRefiner
        || modelVersion == .ssd1b || modelVersion == .wurstchenStageC || modelVersion == .sd3
        || modelVersion == .pixart
      {
        DynamicGraph.flags = []
      }
      if !DeviceCapability.isMemoryMapBufferSupported {
        DynamicGraph.flags.insert(.disableMmapMTLBuffer)
      }
      if !isMFAEnabled {
        DynamicGraph.flags.insert(.disableMetalFlashAttention)
      } else {
        DynamicGraph.flags.remove(.disableMetalFlashAttention)
        if !DeviceCapability.isMFAGEMMFaster {
          DynamicGraph.flags.insert(.disableMFAGEMM)
        }
      }
      var secondPassResult = modelPreloader.consumeFirstStageDecode(
        firstStage.decode(
          x, batchSize: (batchSize.0 - batchSize.1, 0),
          decoder: modelPreloader.retrieveFirstStageDecoder(
            firstStage: firstStage, scale: imageScale), cancellation: cancellation),
        firstStage: firstStage, scale: imageScale)
      guard !isNaN(secondPassResult.rawValue.toCPU()) else { return (nil, 1) }
      guard feedback(.secondPassImageDecoded, signposts, nil) else { return (nil, 1) }
      secondPassResult = faceRestoreImage(secondPassResult, configuration: configuration)
      if signposts.contains(.faceRestored) {
        guard feedback(.faceRestored, signposts, nil) else { return (nil, 1) }
      }
      let (result, scaleFactor) = upscaleImageAndToCPU(
        secondPassResult, configuration: configuration)
      if signposts.contains(.imageUpscaled) {
        let _ = feedback(.imageUpscaled, signposts, nil)
      }
      if ImageGeneratorUtils.isVideoModel(modelVersion) {
        batchSize.0 = Int(configuration.numFrames)
      }
      guard batchSize.0 > 1 else {
        return ([result], scaleFactor)
      }
      var batch = [Tensor<FloatType>]()
      let shape = result.shape
      for i in 0..<min(batchSize.0, shape[0]) {
        batch.append(result[i..<(i + 1), 0..<shape[1], 0..<shape[2], 0..<shape[3]].copied())
      }
      return (batch, scaleFactor)
    }
  }

  // This generate image variations with text as modifier and strength.
  private func generateImageOnly(
    _ image: Tensor<FloatType>, scaleFactor imageScaleFactor: Int, depth: Tensor<FloatType>?,
    hints: [ControlHintType: AnyTensor], custom: Tensor<FloatType>?,
    shuffles: [(Tensor<FloatType>, Float)], poses: [(Tensor<FloatType>, Float)],
    text: String,
    negativeText: String, configuration: GenerationConfiguration,
    denoiserParameterization: Denoiser.Parameterization, sampling: Sampling,
    cancellation: (@escaping () -> Void) -> Void,
    feedback: @escaping (ImageGeneratorSignpost, Set<ImageGeneratorSignpost>, Tensor<FloatType>?) ->
      Bool
  ) -> ([Tensor<FloatType>]?, Int) {
    let coreMLGuard = modelPreloader.beginCoreMLGuard()
    defer {
      if coreMLGuard {
        modelPreloader.endCoreMLGuard()
      }
    }
    let mfaGuard = modelPreloader.beginMFAGuard()
    defer {
      if mfaGuard {
        modelPreloader.endMFAGuard()
      }
    }
    let file =
      (configuration.model.flatMap {
        ModelZoo.isModelDownloaded($0) ? $0 : nil
      }) ?? ModelZoo.defaultSpecification.file
    let modifier = ImageGeneratorUtils.modifierForModel(
      file, LoRAs: configuration.loras.compactMap(\.file))
    let modelVersion = ModelZoo.versionForModel(file)
    let (qkNorm, dualAttentionLayers, distilledGuidanceLayers) =
      ModelZoo.MMDiTForModel(file).map {
        return ($0.qkNorm, $0.dualAttentionLayers, $0.distilledGuidanceLayers ?? 0)
      } ?? (false, [], 0)
    let textEncoderVersion = ModelZoo.textEncoderVersionForModel(file)
    let modelObjective = ModelZoo.objectiveForModel(file)
    let modelUpcastAttention = ModelZoo.isUpcastAttentionForModel(file)
    var textEncoderFiles: [String] =
      [
        ModelZoo.textEncoderForModel(file).flatMap {
          ModelZoo.isModelDownloaded($0) ? $0 : nil
        } ?? "clip_vit_l14_f16.ckpt"
      ]
      + ModelZoo.CLIPEncodersForModel(file).compactMap { ModelZoo.isModelDownloaded($0) ? $0 : nil }
    textEncoderFiles +=
      ((ModelZoo.T5EncoderForModel(file).flatMap { ModelZoo.isModelDownloaded($0) ? $0 : nil }).map
      { [$0] } ?? [])
    let diffusionMappingFile = ModelZoo.diffusionMappingForModel(file).flatMap {
      ModelZoo.isModelDownloaded($0) ? $0 : nil
    }
    let fpsId = Int(configuration.fpsId)
    let motionBucketId = Int(configuration.motionBucketId)
    let condAug = configuration.condAug
    let startFrameCfg = configuration.startFrameCfg
    let clipSkip = Int(configuration.clipSkip)
    let autoencoderFile =
      ModelZoo.autoencoderForModel(file).flatMap {
        ModelZoo.isModelDownloaded($0) ? $0 : nil
      } ?? ImageGeneratorUtils.defaultAutoencoder
    let isGuidanceEmbedEnabled =
      ModelZoo.guidanceEmbedForModel(file) && configuration.speedUpWithGuidanceEmbed
    var isCfgEnabled = !ModelZoo.isConsistencyModelForModel(file) && !isGuidanceEmbedEnabled
    let latentsScaling = ModelZoo.latentsScalingForModel(file)
    let paddedTextEncodingLength = ModelZoo.paddedTextEncodingLengthForModel(file)
    let conditioning = ModelZoo.conditioningForModel(file)
    let refinerVersion: ModelVersion? = configuration.refinerModel.flatMap {
      guard $0 != file, ModelZoo.isModelDownloaded($0) else { return nil }
      let version = ModelZoo.versionForModel($0)
      guard
        version == modelVersion
          || ([.sdxlBase, .sdxlRefiner, .ssd1b].contains(version)
            && [.sdxlBase, .sdxlRefiner, .ssd1b].contains(modelVersion))
      else { return nil }
      return ModelZoo.versionForModel($0)
    }
    let tiledDecoding = TiledConfiguration(
      isEnabled: configuration.tiledDecoding,
      tileSize: .init(
        width: Int(configuration.decodingTileWidth), height: Int(configuration.decodingTileHeight)),
      tileOverlap: Int(configuration.decodingTileOverlap))
    let tiledDiffusion = TiledConfiguration(
      isEnabled: configuration.tiledDiffusion,
      tileSize: .init(
        width: Int(configuration.diffusionTileWidth), height: Int(configuration.diffusionTileHeight)
      ), tileOverlap: Int(configuration.diffusionTileOverlap))
    var alternativeDecoderFilePath: String? = nil
    var alternativeDecoderVersion: AlternativeDecoderVersion? = nil
    let lora: [LoRAConfiguration] =
      (ModelZoo.builtinLoRAForModel(file)
        ? [
          LoRAConfiguration(
            file: ModelZoo.filePathForModelDownloaded(file), weight: 1, version: modelVersion,
            isLoHa: false, modifier: .none)
        ] : [])
      + configuration.loras.compactMap {
        guard let file = $0.file else { return nil }
        let loraVersion = LoRAZoo.versionForModel(file)
        guard LoRAZoo.isModelDownloaded(file),
          modelVersion == loraVersion || refinerVersion == loraVersion
            || (modelVersion == .kandinsky21 && loraVersion == .v1)
        else { return nil }
        if LoRAZoo.isConsistencyModelForModel(file) {
          isCfgEnabled = false
        }
        if let alternativeDecoder = LoRAZoo.alternativeDecoderForModel(file) {
          alternativeDecoderFilePath = LoRAZoo.filePathForModelDownloaded(alternativeDecoder.0)
          alternativeDecoderVersion = alternativeDecoder.1
        }
        return LoRAConfiguration(
          file: LoRAZoo.filePathForModelDownloaded(file), weight: $0.weight, version: loraVersion,
          isLoHa: LoRAZoo.isLoHaForModel(file), modifier: LoRAZoo.modifierForModel(file))
      }
    if modelVersion == .v2 || modelVersion == .sdxlBase || modelVersion == .sdxlRefiner
      || modelVersion == .ssd1b || modelVersion == .svdI2v || modelVersion == .wurstchenStageC
      || modelVersion == .sd3 || modelVersion == .pixart
    {
      DynamicGraph.flags = .disableMixedMPSGEMM
    }
    if !DeviceCapability.isMemoryMapBufferSupported {
      DynamicGraph.flags.insert(.disableMmapMTLBuffer)
    }
    let isMFAEnabled = DeviceCapability.isMFAEnabled.load(ordering: .acquiring)
    if !isMFAEnabled {
      DynamicGraph.flags.insert(.disableMetalFlashAttention)
    } else {
      DynamicGraph.flags.remove(.disableMetalFlashAttention)
      if !DeviceCapability.isMFAGEMMFaster {
        DynamicGraph.flags.insert(.disableMFAGEMM)
      }
    }
    var hasHints = Set(hints.keys)
    if !poses.isEmpty {
      hasHints.insert(.pose)
    }
    let (
      canInjectControls, canInjectT2IAdapters, canInjectAttentionKVs, _, injectIPAdapterLengths,
      canInjectedControls
    ) =
      ImageGeneratorUtils.canInjectControls(
        hasImage: true, hasDepth: depth != nil, hasHints: hasHints, hasCustom: custom != nil,
        shuffleCount: shuffles.count, controls: configuration.controls,
        version: modelVersion)
    let queueWatermark = DynamicGraph.queueWatermark
    if (canInjectControls && modelVersion == .v2) && !DeviceCapability.isMaxPerformance {
      DynamicGraph.queueWatermark = 8
    }
    defer {
      if (canInjectControls && modelVersion == .v2) && !DeviceCapability.isMaxPerformance {
        DynamicGraph.queueWatermark = queueWatermark
      }
    }
    let textGuidanceScale = configuration.guidanceScale
    let imageGuidanceScale = configuration.imageGuidanceScale
    let guidanceEmbed = configuration.guidanceEmbed
    let originalSize =
      configuration.originalImageWidth == 0 || configuration.originalImageHeight == 0
      ? (width: Int(configuration.startWidth) * 64, height: Int(configuration.startHeight) * 64)
      : (
        width: Int(configuration.originalImageWidth), height: Int(configuration.originalImageHeight)
      )
    let cropTopLeft = (top: Int(configuration.cropTop), left: Int(configuration.cropLeft))
    let targetSize =
      configuration.targetImageWidth == 0 || configuration.targetImageHeight == 0
      ? (width: Int(configuration.startWidth) * 64, height: Int(configuration.startHeight) * 64)
      : (width: Int(configuration.targetImageWidth), height: Int(configuration.targetImageHeight))
    let negativeOriginalSize =
      configuration.negativeOriginalImageWidth == 0
        || configuration.negativeOriginalImageHeight == 0
      ? originalSize
      : (
        width: Int(configuration.negativeOriginalImageWidth),
        height: Int(configuration.negativeOriginalImageHeight)
      )
    let aestheticScore = configuration.aestheticScore
    let negativeAestheticScore = configuration.negativeAestheticScore
    let zeroNegativePrompt = configuration.zeroNegativePrompt
    let sharpness = configuration.sharpness
    let strength = configuration.strength
    let isQuantizedModel = ModelZoo.isQuantizedModel(file)
    let is8BitModel = ModelZoo.is8BitModel(file)
    let canRunLoRASeparately = modelPreloader.canRunLoRASeparately
    let externalOnDemand = modelPreloader.externalOnDemand(
      version: modelVersion,
      scale: DeviceCapability.Scale(
        widthScale: configuration.startWidth, heightScale: configuration.startHeight),
      variant: .unet, injectedControls: canInjectedControls,
      is8BitModel: is8BitModel && (canRunLoRASeparately || lora.isEmpty))
    let refiner: Refiner? = configuration.refinerModel.flatMap {
      guard $0 != file, ModelZoo.isModelDownloaded($0) else { return nil }
      let version = ModelZoo.versionForModel($0)
      guard
        version == modelVersion
          || ([.sdxlBase, .sdxlRefiner, .ssd1b].contains(version)
            && [.sdxlBase, .sdxlRefiner, .ssd1b].contains(modelVersion))
      else { return nil }
      return Refiner(
        start: configuration.refinerStart, filePath: ModelZoo.filePathForModelDownloaded($0),
        externalOnDemand: externalOnDemand, version: ModelZoo.versionForModel($0),
        isQuantizedModel: ModelZoo.isQuantizedModel($0),
        isConsistencyModel: ModelZoo.isConsistencyModelForModel($0))
    }
    let controlExternalOnDemand = modelPreloader.externalOnDemand(
      version: modelVersion,
      scale: DeviceCapability.Scale(
        widthScale: configuration.startWidth, heightScale: configuration.startHeight),
      variant: .control, injectedControls: canInjectedControls, suffix: "ctrl")
    let textEncoderExternalOnDemand = modelPreloader.externalOnDemand(
      version: modelVersion,
      scale: DeviceCapability.Scale(
        widthScale: configuration.startWidth, heightScale: configuration.startHeight),
      variant: .textEncoder, injectedControls: 0)
    let vaeExternalOnDemand = modelPreloader.externalOnDemand(
      version: modelVersion,
      scale: DeviceCapability.Scale(
        widthScale: configuration.startWidth, heightScale: configuration.startHeight),
      variant: .autoencoder, injectedControls: canInjectedControls)
    let dmExternalOnDemand = modelPreloader.externalOnDemand(
      version: modelVersion,
      scale: DeviceCapability.Scale(
        widthScale: configuration.startWidth, heightScale: configuration.startHeight),
      variant: .diffusionMapping, injectedControls: 0)
    let teaCache =
      ModelZoo.teaCacheCoefficientsForModel(file).map {
        var teaCacheEnd =
          configuration.teaCacheEnd < 0
          ? Int(configuration.steps) + 1 + Int(configuration.teaCacheEnd)
          : Int(configuration.teaCacheEnd)
        let teaCacheStart = min(max(Int(configuration.teaCacheStart), 0), Int(configuration.steps))
        teaCacheEnd = min(max(max(teaCacheStart, teaCacheEnd), 0), Int(configuration.steps))
        return TeaCacheConfiguration(
          coefficients: $0,
          steps: min(teaCacheStart, teaCacheEnd)...max(teaCacheStart, teaCacheEnd),
          threshold: configuration.teaCache ? configuration.teaCacheThreshold : 0,
          maxSkipSteps: Int(configuration.teaCacheMaxSkipSteps))
      }
      ?? TeaCacheConfiguration(
        coefficients: (0, 0, 0, 0, 0), steps: 0...0, threshold: 0, maxSkipSteps: 0)
    let causalInference =
      configuration.causalInferenceEnabled ? Int(configuration.causalInference) : 0
    let sampler = LocalImageGenerator.sampler(
      from: configuration.sampler, isCfgEnabled: isCfgEnabled,
      filePath: ModelZoo.filePathForModelDownloaded(file), modifier: modifier,
      version: modelVersion, qkNorm: qkNorm, dualAttentionLayers: dualAttentionLayers,
      distilledGuidanceLayers: distilledGuidanceLayers, usesFlashAttention: isMFAEnabled,
      objective: modelObjective,
      upcastAttention: modelUpcastAttention,
      externalOnDemand: externalOnDemand, injectControls: canInjectControls,
      injectT2IAdapters: canInjectT2IAdapters, injectAttentionKV: canInjectAttentionKVs,
      injectIPAdapterLengths: injectIPAdapterLengths,
      lora: lora, isGuidanceEmbedEnabled: isGuidanceEmbedEnabled,
      isQuantizedModel: isQuantizedModel,
      canRunLoRASeparately: canRunLoRASeparately,
      stochasticSamplingGamma: configuration.stochasticSamplingGamma,
      conditioning: conditioning, parameterization: denoiserParameterization,
      tiledDiffusion: tiledDiffusion, teaCache: teaCache, causalInference: causalInference,
      weightsCache: weightsCache,
      of: FloatType.self)
    let initTimestep = sampler.timestep(for: strength, sampling: sampling)
    guard initTimestep.startStep > 0 || modelVersion == .svdI2v else {  // TODO: This check should be removed as text only should be capable of handling svdI2v too.
      return generateTextOnly(
        image, scaleFactor: imageScaleFactor,
        depth: depth, hints: hints, custom: custom, shuffles: shuffles, poses: poses,
        text: text, negativeText: negativeText, configuration: configuration,
        denoiserParameterization: denoiserParameterization, sampling: sampling,
        cancellation: cancellation, feedback: feedback)
    }
    let batchSize =
      ImageGeneratorUtils.isVideoModel(modelVersion) ? 1 : Int(configuration.batchSize)
    precondition(batchSize > 0)
    precondition(strength >= 0 && strength <= 1)
    let highPrecisionForAutoencoder = ModelZoo.isHighPrecisionAutoencoderForModel(file)
    precondition(image.shape[2] % (64 * imageScaleFactor) == 0)
    precondition(image.shape[1] % (64 * imageScaleFactor) == 0)
    let startWidth: Int
    let startHeight: Int
    let channels: Int
    let firstStageFilePath: String
    if modelVersion == .wurstchenStageC {
      (startWidth, startHeight) = stageCLatentsSize(configuration)
      channels = 16
      firstStageFilePath = ModelZoo.filePathForModelDownloaded(file)
    } else {
      startWidth = image.shape[2] / 8 / imageScaleFactor
      startHeight = image.shape[1] / 8 / imageScaleFactor
      switch modelVersion {
      case .wurstchenStageC, .sd3, .sd3Large, .flux1, .hunyuanVideo, .wan21_1_3b, .wan21_14b,
        .hiDreamI1:
        channels = 16
      case .auraflow, .kandinsky21, .pixart, .sdxlBase, .sdxlRefiner, .ssd1b, .svdI2v, .v1, .v2,
        .wurstchenStageB:
        channels = 4
      }
      firstStageFilePath = ModelZoo.filePathForModelDownloaded(autoencoderFile)
    }
    let imageScale = DeviceCapability.Scale(
      widthScale: UInt16(startWidth / 8), heightScale: UInt16(startHeight / 8))
    let isHighPrecisionVAEFallbackEnabled = DeviceCapability.isHighPrecisionVAEFallbackEnabled(
      scale: imageScale)
    let graph = DynamicGraph()
    if externalOnDemand
      || externalOnDemandPartially(
        version: modelVersion, memoryCapacity: DeviceCapability.memoryCapacity,
        externalOnDemand: externalOnDemand)
    {
      TensorData.makeExternalData(for: ModelZoo.filePathForModelDownloaded(file), graph: graph)
      for stageModel in ModelZoo.stageModelsForModel(file) {
        TensorData.makeExternalData(
          for: ModelZoo.filePathForModelDownloaded(stageModel), graph: graph)
      }
    }
    if textEncoderExternalOnDemand {
      TensorData.makeExternalData(
        for: ModelZoo.filePathForModelDownloaded(textEncoderFiles[0]), graph: graph)
    }
    if controlExternalOnDemand {
      for file in Set(configuration.controls.compactMap { $0.file }) {
        TensorData.makeExternalData(for: ModelZoo.filePathForModelDownloaded(file), graph: graph)
      }
    }
    if vaeExternalOnDemand {
      TensorData.makeExternalData(
        for: ModelZoo.filePathForModelDownloaded(autoencoderFile), graph: graph)
    }
    if dmExternalOnDemand, let diffusionMappingFile = diffusionMappingFile {
      TensorData.makeExternalData(
        for: ModelZoo.filePathForModelDownloaded(diffusionMappingFile), graph: graph)
    }
    let potentials = lora.map { ($0.file as NSString).lastPathComponent }
    let (
      tokensTensors, positionTensors, embedMask, injectedEmbeddings, unconditionalAttentionWeights,
      attentionWeights, hasNonOneWeights, tokenLengthUncond, tokenLengthCond, lengthsOfUncond,
      lengthsOfCond
    ) = tokenize(
      graph: graph, modelVersion: modelVersion, textEncoderVersion: textEncoderVersion,
      paddedTextEncodingLength: paddedTextEncodingLength, text: text, negativeText: negativeText,
      negativePromptForImagePrior: configuration.negativePromptForImagePrior,
      potentials: potentials, T5TextEncoder: configuration.t5TextEncoder,
      clipL: configuration.separateClipL ? (configuration.clipLText ?? "") : nil,
      openClipG: configuration.separateOpenClipG ? (configuration.openClipGText ?? "") : nil,
      t5: configuration.separateT5 ? (configuration.t5Text ?? "") : nil
    )
    var signposts = Set<ImageGeneratorSignpost>([
      .textEncoded, .imageEncoded, .sampling(sampling.steps - initTimestep.roundedDownStartStep),
      .imageDecoded,
    ])
    if let faceRestoration = configuration.faceRestoration,
      EverythingZoo.isModelDownloaded(faceRestoration)
        && EverythingZoo.isModelDownloaded(EverythingZoo.parsenetForModel(faceRestoration))
    {
      signposts.insert(.faceRestored)
    }
    if let upscaler = configuration.upscaler, UpscalerZoo.isModelDownloaded(upscaler) {
      signposts.insert(.imageUpscaled)
    }
    if modelVersion == .wurstchenStageC {
      let initTimestep = sampler.timestep(
        for: strength,
        sampling: Sampling(
          steps: Int(configuration.stage2Steps), shift: Double(configuration.stage2Shift)))
      signposts.insert(
        .secondPassSampling(Int(configuration.stage2Steps) - initTimestep.roundedDownStartStep))
      signposts.insert(.secondPassImageEncoded)
    }
    return graph.withNoGrad {
      let injectedTextEmbeddings = generateInjectedTextEmbeddings(
        batchSize: batchSize, startHeight: startHeight, startWidth: startWidth, image: image,
        graph: graph, hints: hints, custom: custom, shuffles: shuffles, pose: poses.first?.0,
        controls: configuration.controls,
        version: modelVersion, tiledDiffusion: tiledDiffusion, usesFlashAttention: isMFAEnabled,
        externalOnDemand: controlExternalOnDemand, cancellation: cancellation)
      let (tokenLengthUncond, tokenLengthCond) = ControlModel<FloatType>.modifyTextEmbeddings(
        tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
        injecteds: injectedTextEmbeddings)
      let tokenLength = max(tokenLengthUncond, tokenLengthCond)
      let textEncoder = TextEncoder<FloatType>(
        filePaths: textEncoderFiles.map { ModelZoo.filePathForModelDownloaded($0) },
        version: modelVersion, textEncoderVersion: textEncoderVersion,
        isCfgEnabled: isCfgEnabled,
        usesFlashAttention: isMFAEnabled && DeviceCapability.isMFACausalAttentionMaskSupported,
        injectEmbeddings: !injectedEmbeddings.isEmpty,
        externalOnDemand: textEncoderExternalOnDemand,
        deviceProperties: DeviceCapability.deviceProperties, weightsCache: weightsCache,
        maxLength: tokenLength, clipSkip: clipSkip, lora: lora)
      let image = downscaleImageAndToGPU(
        graph.variable(image), scaleFactor: imageScaleFactor)
      let textEncodings = modelPreloader.consumeTextModels(
        textEncoder.encode(
          tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
          tokens: tokensTensors, positions: positionTensors, mask: embedMask,
          injectedEmbeddings: injectedEmbeddings, image: [image], lengthsOfUncond: lengthsOfUncond,
          lengthsOfCond: lengthsOfCond, injectedTextEmbeddings: injectedTextEmbeddings,
          textModels: modelPreloader.retrieveTextModels(textEncoder: textEncoder)),
        textEncoder: textEncoder)
      var c: [DynamicGraph.Tensor<FloatType>]
      var extraProjection: DynamicGraph.Tensor<FloatType>?
      DynamicGraph.setSeed(configuration.seed)
      if modelVersion == .kandinsky21, let diffusionMappingFile = diffusionMappingFile {
        let diffusionMapping = DiffusionMapping<FloatType>(
          filePath: ModelZoo.filePathForModelDownloaded(diffusionMappingFile),
          usesFlashAttention: isMFAEnabled && DeviceCapability.isMFACausalAttentionMaskSupported,
          steps: Int(configuration.imagePriorSteps),
          negativePromptForImagePrior: configuration.negativePromptForImagePrior,
          CLIPWeight: configuration.clipWeight, externalOnDemand: dmExternalOnDemand)
        let imageEmb = diffusionMapping.sample(
          textEncoding: textEncodings[2], textEmbedding: textEncodings[3], tokens: tokensTensors[1])
        let kandinskyEmbedding = KandinskyEmbedding<FloatType>(
          filePath: ModelZoo.filePathForModelDownloaded(file))
        let (xfProj, xfOut) = kandinskyEmbedding.encode(
          textEncoding: textEncodings[0], textEmbedding: textEncodings[1], imageEmbedding: imageEmb)
        extraProjection = xfProj
        c = [xfOut]
      } else {
        extraProjection = nil
        c = textEncodings
      }
      (c, extraProjection) = repeatConditionsToMatchBatchSize(
        c: c, extraProjection: extraProjection,
        unconditionalAttentionWeights: unconditionalAttentionWeights,
        attentionWeights: attentionWeights, version: modelVersion, tokenLength: tokenLength,
        batchSize: batchSize, hasNonOneWeights: hasNonOneWeights)
      guard feedback(.textEncoded, signposts, nil) else { return (nil, 1) }
      var firstStage = FirstStage<FloatType>(
        filePath: firstStageFilePath, version: modelVersion,
        latentsScaling: latentsScaling, highPrecisionKeysAndValues: highPrecisionForAutoencoder,
        highPrecisionFallback: isHighPrecisionVAEFallbackEnabled,
        tiledDecoding: tiledDecoding, tiledDiffusion: tiledDiffusion,
        externalOnDemand: vaeExternalOnDemand, alternativeUsesFlashAttention: isMFAEnabled,
        alternativeFilePath: alternativeDecoderFilePath,
        alternativeDecoderVersion: alternativeDecoderVersion,
        deviceProperties: DeviceCapability.deviceProperties)
      // Check if strength is 0.
      guard initTimestep.roundedDownStartStep < sampling.steps && configuration.strength > 0 else {
        let image = faceRestoreImage(image, configuration: configuration)
        // Otherwise, just run upscaler if needed.
        let (result, scaleFactor) = upscaleImageAndToCPU(image, configuration: configuration)
        // Because we just run the upscaler, there is no more than 1 image generation, return directly.
        return ([result], scaleFactor)
      }
      var firstPassImage: DynamicGraph.Tensor<FloatType>
      if modelVersion == .wurstchenStageC {
        // Try to resize the input image so we can encode with EfficientNetv2s properly.
        if image.shape[1] != startHeight * 32 || image.shape[2] != startWidth * 32 {
          firstPassImage = Upsample(
            .bilinear, widthScale: Float(startWidth * 32) / Float(image.shape[2]),
            heightScale: Float(startHeight * 32) / Float(image.shape[1]))(image)
        } else {
          firstPassImage = image
        }
      } else {
        firstPassImage = image
      }
      var batchSize = (batchSize, 0)
      switch modelVersion {
      case .svdI2v:
        batchSize = (Int(configuration.numFrames), 0)
      case .hunyuanVideo, .wan21_1_3b, .wan21_14b:
        batchSize = injectReferenceFrames(
          batchSize: ((Int(configuration.numFrames) - 1) / 4) + 1, version: modelVersion,
          canInjectControls: canInjectControls, shuffleCount: shuffles.count,
          hasCustom: custom != nil)
      case .auraflow, .flux1, .kandinsky21, .pixart, .sd3, .sd3Large, .sdxlBase, .sdxlRefiner,
        .ssd1b, .v1, .v2, .wurstchenStageB, .wurstchenStageC, .hiDreamI1:
        break
      }
      let imageSize: Int
      let firstPassImageForSample: DynamicGraph.Tensor<FloatType>?
      (imageSize, firstPassImage, firstPassImageForSample) = expandImageForEncoding(
        batchSize: batchSize, version: modelVersion, modifier: modifier, image: firstPassImage)
      var sample: DynamicGraph.Tensor<FloatType>
      let encodedImage: DynamicGraph.Tensor<FloatType>
      (sample, encodedImage) = modelPreloader.consumeFirstStageSample(
        firstStage.sample(
          firstPassImage,
          encoder: modelPreloader.retrieveFirstStageEncoder(
            firstStage: firstStage, scale: imageScale), cancellation: cancellation),
        firstStage: firstStage, scale: imageScale)
      if let firstPassImageForSample = firstPassImageForSample {
        (sample, _) = modelPreloader.consumeFirstStageSample(
          firstStage.sample(
            firstPassImageForSample,
            encoder: modelPreloader.retrieveFirstStageEncoder(
              firstStage: firstStage, scale: imageScale), cancellation: cancellation),
          firstStage: firstStage, scale: imageScale)
      }
      var maskedImage: DynamicGraph.Tensor<FloatType>? = nil
      var mask: DynamicGraph.Tensor<FloatType>? = nil
      if modifier == .inpainting {
        maskedImage = firstStage.scale(
          encodedImage[0..<imageSize, 0..<startHeight, 0..<startWidth, 0..<channels].copied())
        mask = graph.variable(.GPU(0), .NHWC(1, startHeight, startWidth, 1), of: FloatType.self)
        mask?.full(1)
        maskedImage = concatMaskWithMaskedImage(
          hasImage: true, batchSize: batchSize,
          version: modelVersion, encodedImage: maskedImage!, encodedMask: mask!, imageNegMask: nil)
      } else if modifier == .editing {
        if modelVersion == .v1 {
          maskedImage = encodedImage[0..<imageSize, 0..<startHeight, 0..<startWidth, 0..<channels]
            .copied()
        } else {
          maskedImage = firstStage.scale(
            encodedImage[0..<imageSize, 0..<startHeight, 0..<startWidth, 0..<channels].copied())
        }
      } else if modelVersion == .svdI2v {
        maskedImage = encodedImage[0..<imageSize, 0..<startHeight, 0..<startWidth, 0..<channels]
          .copied()
      }
      guard feedback(.imageEncoded, signposts, nil) else { return (nil, 1) }
      let noise = randomLatentNoise(
        graph: graph, batchSize: batchSize.0, startHeight: startHeight,
        startWidth: startWidth, channels: channels, seed: configuration.seed,
        seedMode: configuration.seedMode)
      let depthImage = depth.map {
        let depthImage = graph.variable($0.toGPU(0))
        let depthHeight = depthImage.shape[1]
        let depthWidth = depthImage.shape[2]
        guard depthHeight != startHeight * 8 || depthWidth != startWidth * 8 else {
          return depthImage
        }
        return Upsample(
          .bilinear, widthScale: Float(startHeight * 8) / Float(depthHeight),
          heightScale: Float(startWidth * 8) / Float(depthWidth))(depthImage)
      }
      let customImage = custom.map {
        let customImage = graph.variable($0.toGPU(0))
        let customHeight = customImage.shape[1]
        let customWidth = customImage.shape[2]
        guard customHeight != startHeight * 8 || customWidth != startWidth * 8 else {
          return customImage
        }
        return Upsample(
          .bilinear, widthScale: Float(startHeight * 8) / Float(customHeight),
          heightScale: Float(startWidth * 8) / Float(customWidth))(customImage)
      }
      let injectedControls = generateInjectedControls(
        graph: graph, batchSize: batchSize.0, startHeight: startHeight, startWidth: startWidth,
        image: image,
        depth: depthImage, hints: hints, custom: custom, shuffles: shuffles, pose: poses.last?.0,
        mask: nil, controls: configuration.controls, version: modelVersion,
        tiledDiffusion: tiledDiffusion, usesFlashAttention: isMFAEnabled,
        externalOnDemand: controlExternalOnDemand, steps: sampling.steps, firstStage: firstStage,
        cancellation: cancellation)
      guard feedback(.controlsGenerated, signposts, nil) else { return (nil, 1) }
      if let image = maskedImage {
        maskedImage = injectVACEFrames(
          batchSize: batchSize, version: modelVersion, image: image,
          injectedControls: injectedControls)
      }
      firstPassImage = injectVACEFrames(
        batchSize: batchSize, version: modelVersion, image: firstPassImage,
        injectedControls: injectedControls)
      sample = injectVACEFrames(
        batchSize: batchSize, version: modelVersion, image: sample,
        injectedControls: injectedControls)
      let x_T: DynamicGraph.Tensor<FloatType>
      if initTimestep.startStep > 0 {
        let sampleScaleFactor = sampler.sampleScaleFactor(
          at: initTimestep.startStep, sampling: sampling)
        let noiseScaleFactor = sampler.noiseScaleFactor(
          at: initTimestep.startStep, sampling: sampling)
        let zEnc = sampleScaleFactor * sample + noiseScaleFactor * noise
        x_T = zEnc
      } else {
        x_T = noise
      }
      let imageCond = encodeImageCond(
        image: firstPassImage, depth: depthImage, custom: customImage, modifier: modifier,
        version: modelVersion, firstStage: firstStage, usesFlashAttention: isMFAEnabled)
      guard
        var x =
          try? modelPreloader.consumeUNet(
            sampler.sample(
              x_T,
              unets: modelPreloader.retrieveUNet(
                sampler: sampler, scale: imageScale, tokenLengthUncond: tokenLengthUncond,
                tokenLengthCond: tokenLengthCond), sample: nil,
              conditionImage: maskedImage ?? imageCond,
              mask: mask, negMask: nil, conditioning: c, tokenLengthUncond: tokenLengthUncond,
              tokenLengthCond: tokenLengthCond, extraProjection: extraProjection,
              injectedControls: injectedControls, textGuidanceScale: textGuidanceScale,
              imageGuidanceScale: imageGuidanceScale, guidanceEmbed: guidanceEmbed,
              startStep: (
                integral: initTimestep.roundedDownStartStep, fractional: initTimestep.startStep
              ),
              endStep: (integral: sampling.steps, fractional: Float(sampling.steps)),
              originalSize: originalSize, cropTopLeft: cropTopLeft,
              targetSize: targetSize, aestheticScore: aestheticScore,
              negativeOriginalSize: negativeOriginalSize,
              negativeAestheticScore: negativeAestheticScore,
              zeroNegativePrompt: zeroNegativePrompt, refiner: refiner, fpsId: fpsId,
              motionBucketId: motionBucketId, condAug: condAug, startFrameCfg: startFrameCfg,
              sharpness: sharpness, sampling: sampling, cancellation: cancellation
            ) { step, tensor in
              feedback(.sampling(step), signposts, tensor)
            }, sampler: sampler, scale: imageScale, tokenLengthUncond: tokenLengthUncond,
            tokenLengthCond: tokenLengthCond)
      else {
        return (nil, 1)
      }
      guard
        feedback(.sampling(sampling.steps - initTimestep.roundedDownStartStep), signposts, nil)
      else {
        return (nil, 1)
      }
      // If it is Wurstchen, run the stage 2.
      if modelVersion == .wurstchenStageC {
        guard feedback(.imageDecoded, signposts, nil) else { return (nil, 1) }
        firstStage = FirstStage<FloatType>(
          filePath: ModelZoo.filePathForModelDownloaded(autoencoderFile), version: .wurstchenStageB,
          latentsScaling: latentsScaling, highPrecisionKeysAndValues: highPrecisionForAutoencoder,
          highPrecisionFallback: isHighPrecisionVAEFallbackEnabled,
          tiledDecoding: TiledConfiguration(
            isEnabled: false, tileSize: .init(width: 0, height: 0), tileOverlap: 0),
          tiledDiffusion: tiledDiffusion,
          externalOnDemand: vaeExternalOnDemand, alternativeUsesFlashAttention: isMFAEnabled,
          alternativeFilePath: alternativeDecoderFilePath,
          alternativeDecoderVersion: alternativeDecoderVersion,
          deviceProperties: DeviceCapability.deviceProperties)
        let (sample, encodedImage) = modelPreloader.consumeFirstStageSample(
          firstStage.sample(
            image,
            encoder: modelPreloader.retrieveFirstStageEncoder(
              firstStage: firstStage, scale: imageScale), cancellation: cancellation),
          firstStage: firstStage, scale: imageScale
        )
        let startHeight = Int(configuration.startHeight) * 16
        let startWidth = Int(configuration.startWidth) * 16
        let channels = 4
        var maskedImage: DynamicGraph.Tensor<FloatType>? = nil
        var mask: DynamicGraph.Tensor<FloatType>? = nil
        if modifier == .inpainting {
          maskedImage = firstStage.scale(
            encodedImage[0..<1, 0..<startHeight, 0..<startWidth, 0..<channels].copied())
          mask = graph.variable(.GPU(0), .NHWC(1, startHeight, startWidth, 1), of: FloatType.self)
          mask?.full(1)
          maskedImage = concatMaskWithMaskedImage(
            hasImage: true, batchSize: batchSize,
            version: modelVersion, encodedImage: maskedImage!, encodedMask: mask!, imageNegMask: nil
          )
        } else if modifier == .editing {
          if modelVersion == .v1 {
            maskedImage = encodedImage[0..<1, 0..<startHeight, 0..<startWidth, 0..<channels]
              .copied()
          } else {
            maskedImage = firstStage.scale(
              encodedImage[0..<1, 0..<startHeight, 0..<startWidth, 0..<channels]
                .copied())
          }
        } else if modelVersion == .svdI2v {
          maskedImage = encodedImage[0..<1, 0..<startHeight, 0..<startWidth, 0..<channels].copied()
        }
        guard feedback(.secondPassImageEncoded, signposts, nil) else { return (nil, 1) }
        let secondPassModelVersion = ModelVersion.wurstchenStageB
        let secondPassModelFilePath = ModelZoo.filePathForModelDownloaded(
          ModelZoo.stageModelsForModel(file)[0])
        let secondPassSampler = LocalImageGenerator.sampler(
          from: configuration.sampler, isCfgEnabled: isCfgEnabled,
          filePath: secondPassModelFilePath, modifier: modifier,
          version: secondPassModelVersion, qkNorm: qkNorm, dualAttentionLayers: dualAttentionLayers,
          distilledGuidanceLayers: distilledGuidanceLayers, usesFlashAttention: isMFAEnabled,
          objective: modelObjective,
          upcastAttention: modelUpcastAttention,
          externalOnDemand: externalOnDemand,
          injectControls: canInjectControls, injectT2IAdapters: canInjectT2IAdapters,
          injectAttentionKV: canInjectAttentionKVs,
          injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
          isGuidanceEmbedEnabled: isGuidanceEmbedEnabled, isQuantizedModel: isQuantizedModel,
          canRunLoRASeparately: canRunLoRASeparately,
          stochasticSamplingGamma: configuration.stochasticSamplingGamma,
          conditioning: conditioning, parameterization: denoiserParameterization,
          tiledDiffusion: tiledDiffusion, teaCache: teaCache, causalInference: causalInference,
          weightsCache: weightsCache,
          of: FloatType.self
        )
        let noise = randomLatentNoise(
          graph: graph, batchSize: batchSize.0, startHeight: startHeight,
          startWidth: startWidth, channels: channels, seed: configuration.seed,
          seedMode: configuration.seedMode)
        c.append(x)
        let secondPassTextGuidance = configuration.stage2Cfg
        let secondPassSampling = Sampling(
          steps: Int(configuration.stage2Steps), shift: Double(configuration.stage2Shift))
        let initTimestep = sampler.timestep(for: strength, sampling: secondPassSampling)
        let x_T: DynamicGraph.Tensor<FloatType>
        if initTimestep.startStep > 0 {
          let sampleScaleFactor = sampler.sampleScaleFactor(
            at: initTimestep.startStep, sampling: secondPassSampling)
          let noiseScaleFactor = sampler.noiseScaleFactor(
            at: initTimestep.startStep, sampling: secondPassSampling)
          let zEnc = sampleScaleFactor * sample + noiseScaleFactor * noise
          x_T = zEnc
        } else {
          x_T = noise
        }
        guard
          let b =
            try? modelPreloader.consumeUNet(
              secondPassSampler.sample(
                x_T,
                unets: modelPreloader.retrieveUNet(
                  sampler: secondPassSampler, scale: imageScale,
                  tokenLengthUncond: tokenLengthUncond,
                  tokenLengthCond: tokenLengthCond),
                sample: nil, conditionImage: maskedImage, mask: mask,
                negMask: nil, conditioning: c, tokenLengthUncond: tokenLengthUncond,
                tokenLengthCond: tokenLengthCond, extraProjection: extraProjection,
                injectedControls: [],  // TODO: Support injectedControls for this.
                textGuidanceScale: secondPassTextGuidance,
                imageGuidanceScale: imageGuidanceScale, guidanceEmbed: guidanceEmbed,
                startStep: (
                  integral: initTimestep.roundedDownStartStep, fractional: initTimestep.startStep
                ),
                endStep: (
                  integral: secondPassSampling.steps, fractional: Float(secondPassSampling.steps)
                ),
                originalSize: originalSize, cropTopLeft: cropTopLeft,
                targetSize: targetSize, aestheticScore: aestheticScore,
                negativeOriginalSize: negativeOriginalSize,
                negativeAestheticScore: negativeAestheticScore,
                zeroNegativePrompt: zeroNegativePrompt, refiner: refiner, fpsId: fpsId,
                motionBucketId: motionBucketId, condAug: condAug, startFrameCfg: startFrameCfg,
                sharpness: sharpness, sampling: secondPassSampling, cancellation: cancellation
              ) { step, tensor in
                feedback(.secondPassSampling(step), signposts, tensor)
              }, sampler: secondPassSampler, scale: imageScale,
              tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond)
        else {
          return (nil, 1)
        }
        guard
          feedback(
            .secondPassSampling(secondPassSampling.steps - initTimestep.roundedDownStartStep),
            signposts, nil)
        else {
          return (nil, 1)
        }
        x = b
      }
      if DeviceCapability.isLowPerformance {
        graph.garbageCollect()
      }
      if modelVersion == .v2 || modelVersion == .sdxlBase || modelVersion == .sdxlRefiner
        || modelVersion == .ssd1b
      {
        DynamicGraph.flags = []
      }
      if !DeviceCapability.isMemoryMapBufferSupported {
        DynamicGraph.flags.insert(.disableMmapMTLBuffer)
      }
      if !isMFAEnabled {
        DynamicGraph.flags.insert(.disableMetalFlashAttention)
      } else {
        DynamicGraph.flags.remove(.disableMetalFlashAttention)
        if !DeviceCapability.isMFAGEMMFaster {
          DynamicGraph.flags.insert(.disableMFAGEMM)
        }
      }
      var firstStageResult = modelPreloader.consumeFirstStageDecode(
        firstStage.decode(
          x, batchSize: (batchSize.0 - batchSize.1, 0),
          decoder: modelPreloader.retrieveFirstStageDecoder(
            firstStage: firstStage, scale: imageScale), cancellation: cancellation),
        firstStage: firstStage, scale: imageScale)
      guard !isNaN(firstStageResult.rawValue.toCPU()) else { return (nil, 1) }
      if modelVersion == .wurstchenStageC {
        guard feedback(.secondPassImageDecoded, signposts, nil) else { return (nil, 1) }
      } else {
        guard feedback(.imageDecoded, signposts, nil) else { return (nil, 1) }
      }
      firstStageResult = faceRestoreImage(firstStageResult, configuration: configuration)
      if signposts.contains(.faceRestored) {
        guard feedback(.faceRestored, signposts, nil) else { return (nil, 1) }
      }
      let (result, scaleFactor) = upscaleImageAndToCPU(
        firstStageResult, configuration: configuration)
      if signposts.contains(.imageUpscaled) {
        let _ = feedback(.imageUpscaled, signposts, nil)
      }
      if ImageGeneratorUtils.isVideoModel(modelVersion) {
        batchSize.0 = Int(configuration.numFrames)
      }
      guard batchSize.0 > 1 else {
        return ([result], scaleFactor)
      }
      var batch = [Tensor<FloatType>]()
      let shape = result.shape
      for i in 0..<min(batchSize.0, shape[0]) {
        batch.append(result[i..<(i + 1), 0..<shape[1], 0..<shape[2], 0..<shape[3]].copied())
      }
      return (batch, scaleFactor)
    }
  }

  private func imageWithMask(
    _ image: inout Tensor<FloatType>, original: Tensor<FloatType>, mask: Tensor<UInt8>,
    maskBlur: Float, maskBlurOutset: Int, overwrite0: Bool, transparent: Bool
  ) {
    precondition(image.shape[2] % 64 == 0)
    let imageWidth = image.shape[2]
    precondition(image.shape[1] % 64 == 0)
    let imageHeight = image.shape[1]
    precondition(image.shape[2] == mask.shape[1])
    precondition(image.shape[1] == mask.shape[0])
    precondition(original.shape[2] == mask.shape[1])
    precondition(original.shape[1] == mask.shape[0])
    let channelStart = image.shape[3] - 3
    let transparent = transparent && (channelStart == 1)
    if maskBlur > 0 {
      var maskBeforeBlurred = ccv_dense_matrix_new(
        Int32(imageHeight), Int32(imageWidth), Int32(CCV_8U | CCV_C1), nil, 0)!
      precondition(imageWidth % 4 == 0)
      let u8 = maskBeforeBlurred.pointee.data.u8!
      for y in 0..<imageHeight {
        for x in 0..<imageWidth {
          let byteMask = mask[y, x]
          guard
            (byteMask & 7) == 3 || ((byteMask & 7) == 1 && transparent)
              || (!overwrite0 && (byteMask & 7) == 0)
          else {
            u8[y * imageWidth + x] = 0
            continue
          }
          u8[y * imageWidth + x] = 255
        }
      }
      if maskBlurOutset != 0 {
        var maskAfterMorph = ccv_dense_matrix_new(
          Int32(imageHeight), Int32(imageWidth), Int32(CCV_8U | CCV_C1), nil, 0)
        if maskBlurOutset < 0 {
          for _ in 0..<(-maskBlurOutset) {
            ccv_dilate(maskBeforeBlurred, &maskAfterMorph, 0, 3)
            let mask = maskBeforeBlurred
            maskBeforeBlurred = maskAfterMorph!
            maskAfterMorph = mask
          }
        } else {
          for _ in 0..<maskBlurOutset {
            ccv_erode(maskBeforeBlurred, &maskAfterMorph, 0, 3)
            let mask = maskBeforeBlurred
            maskBeforeBlurred = maskAfterMorph!
            maskAfterMorph = mask
          }
        }
        ccv_matrix_free(maskAfterMorph)
      }
      var blurred: UnsafeMutablePointer<ccv_dense_matrix_t>? = nil
      ccv_blur(maskBeforeBlurred, &blurred, Int32(CCV_8U | CCV_C1), Double(maskBlur))
      ccv_matrix_free(maskBeforeBlurred)
      let blurU8 = blurred!.pointee.data.u8!
      for y in 0..<imageHeight {
        for x in 0..<imageWidth {
          let byteMask = mask[y, x]
          // 1 is nothing, thus, we have to take whatever in the generated image (because nothing in the original).
          // Note that if it is 1 and have alpha, there might still be something under.
          guard transparent || (byteMask & 7) != 1 else {
            let alpha = (byteMask & 0xf8)
            guard alpha > 0 else { continue }
            let fp16alpha = FloatType(alpha) / 255
            // Because it is already blended in, we are weaken it to alpha^2, and the other side should be similar.
            let fp16alphaSqr = fp16alpha * fp16alpha
            let negAlphaSqr = 1 - fp16alphaSqr
            image[0, y, x, channelStart] =
              original[0, y, x, 0] * fp16alphaSqr + image[0, y, x, channelStart] * negAlphaSqr
            image[0, y, x, channelStart + 1] =
              original[0, y, x, 1] * fp16alphaSqr + image[0, y, x, channelStart + 1] * negAlphaSqr
            image[0, y, x, channelStart + 2] =
              original[0, y, x, 2] * fp16alphaSqr + image[0, y, x, channelStart + 2] * negAlphaSqr
            continue
          }
          if transparent && (byteMask & 7) == 1 {
            // Copy over the alpha channel.
            let alpha = FloatType(byteMask & 0xf8)
            image[0, y, x, 0] = alpha / 255
          }
          let alpha = blurU8[y * imageWidth + x]
          guard alpha > 0 else { continue }
          if alpha == 255 {
            image[0, y, x, channelStart] = original[0, y, x, 0]
            image[0, y, x, channelStart + 1] = original[0, y, x, 1]
            image[0, y, x, channelStart + 2] = original[0, y, x, 2]
          } else {
            let fp16alpha = FloatType(alpha) / 255
            let negAlpha = 1 - fp16alpha
            image[0, y, x, channelStart] =
              original[0, y, x, 0] * fp16alpha + image[0, y, x, channelStart] * negAlpha
            image[0, y, x, channelStart + 1] =
              original[0, y, x, 1] * fp16alpha + image[0, y, x, channelStart + 1] * negAlpha
            image[0, y, x, channelStart + 2] =
              original[0, y, x, 2] * fp16alpha + image[0, y, x, channelStart + 2] * negAlpha
          }
        }
      }
      ccv_matrix_free(blurred)
    } else {
      for y in 0..<imageHeight {
        for x in 0..<imageWidth {
          let byteMask = mask[y, x]
          guard
            (byteMask & 7) == 3 || ((byteMask & 7) == 1 && transparent)
              || (!overwrite0 && (byteMask & 7) == 0)
          else {
            let alpha = (byteMask & 0xf8)
            guard alpha > 0 else { continue }
            let fp16alpha = FloatType(alpha) / 255
            // Because it is already blended in, we are weaken it to alpha^2, and the other side should be similar.
            let fp16alphaSqr = fp16alpha * fp16alpha
            let negAlphaSqr = 1 - fp16alphaSqr
            image[0, y, x, channelStart] =
              original[0, y, x, 0] * fp16alphaSqr + image[0, y, x, channelStart] * negAlphaSqr
            image[0, y, x, 1] =
              original[0, y, x, channelStart + 1] * fp16alphaSqr + image[0, y, x, channelStart + 1]
              * negAlphaSqr
            image[0, y, x, 2] =
              original[0, y, x, channelStart + 2] * fp16alphaSqr + image[0, y, x, channelStart + 2]
              * negAlphaSqr
            continue
          }
          if transparent && (byteMask & 7) == 1 {
            // Copy over the alpha channel.
            let alpha = FloatType(byteMask & 0xf8)
            image[0, y, x, 0] = alpha / 255
          }
          image[0, y, x, channelStart] = original[0, y, x, 0]
          image[0, y, x, channelStart + 1] = original[0, y, x, 1]
          image[0, y, x, channelStart + 2] = original[0, y, x, 2]
        }
      }
    }
  }

  public static func isInpainting(
    for binaryMask: Tensor<UInt8>?, configuration: GenerationConfiguration
  ) -> Bool {
    guard let binaryMask = binaryMask else { return false }
    let modelVersion = ModelZoo.versionForModel(configuration.model ?? "")
    let refinerVersion: ModelVersion? = configuration.refinerModel.flatMap {
      guard $0 != configuration.model, ModelZoo.isModelDownloaded($0) else { return nil }
      let version = ModelZoo.versionForModel($0)
      guard
        version == modelVersion
          || ([.sdxlBase, .sdxlRefiner, .ssd1b].contains(version)
            && [.sdxlBase, .sdxlRefiner, .ssd1b].contains(modelVersion))
      else { return nil }
      return ModelZoo.versionForModel($0)
    }
    var alternativeDecoderVersion: AlternativeDecoderVersion? = nil
    for lora in configuration.loras {
      guard let file = lora.file else { continue }
      let loraVersion = LoRAZoo.versionForModel(file)
      guard LoRAZoo.isModelDownloaded(file),
        modelVersion == loraVersion || refinerVersion == loraVersion
          || (modelVersion == .kandinsky21 && loraVersion == .v1)
      else { continue }
      if let alternativeDecoder = LoRAZoo.alternativeDecoderForModel(file) {
        alternativeDecoderVersion = alternativeDecoder.1
      }
    }
    let imageWidth = binaryMask.shape[1]
    let imageHeight = binaryMask.shape[0]
    // See detail explanation below.
    // This effectively tells whether we have any skip all (3, or in some conditions 0 can skip all).
    // It should be exists3 || (!(exists0 && exists1 && exists2 && exists3) && exists2) but can be simplified to below.
    for y in 0..<imageHeight {
      for x in 0..<imageWidth {
        let byteMask = (binaryMask[y, x] & 7)
        if byteMask == 3 && alternativeDecoderVersion != .transparent {
          return true
        } else if byteMask == 2 || byteMask == 4 {  // 4 is the same as 2.
          return true
        }
      }
    }
    return false
  }

  // This does inpainting, image doesn't change, but the parts masked out filled in.
  private func generateImageWithMask(
    _ image: Tensor<FloatType>, scaleFactor: Int, mask: Tensor<UInt8>, depth: Tensor<FloatType>?,
    hints: [ControlHintType: AnyTensor], custom: Tensor<FloatType>?,
    shuffles: [(Tensor<FloatType>, Float)], poses: [(Tensor<FloatType>, Float)],
    text: String,
    negativeText: String, configuration: GenerationConfiguration,
    denoiserParameterization: Denoiser.Parameterization, sampling: Sampling,
    cancellation: (@escaping () -> Void) -> Void,
    feedback: @escaping (ImageGeneratorSignpost, Set<ImageGeneratorSignpost>, Tensor<FloatType>?) ->
      Bool
  ) -> ([Tensor<FloatType>]?, Int) {
    // The binary mask is a shape of (height, width), with content of 0, 1, 2, 3
    // 2 means it is explicit masked, if 2 is presented, we will treat 0 as areas to retain, and
    // 1 as areas to fill in from pure noise. If 2 is not presented, we will fill in 1 as pure noise
    // still, but treat 0 as areas masked. If no 1 or 2 presented, this degrades back to generate
    // from image.
    // In more academic point of view, when 1 is presented, we will go from 0 to step - tEnc to
    // generate things from noise with text guidance in these areas. When 2 is explicitly masked, we will
    // retain these areas during 0 to step - tEnc, and make these areas mixing during step - tEnc to end.
    // When 2 is explicitly masked, we will retain areas marked as 0 during 0 to steps, otherwise
    // we will only retain them during 0 to step - tEnc (depending on whether we have 1, if we don't,
    // we don't need to step through 0 to step - tEnc, and if we don't, this degrades to generateImageOnly).
    // Regardless of these, when marked as 3, it will be retained.
    precondition(image.shape[2] % (64 * scaleFactor) == 0)
    precondition(image.shape[1] % (64 * scaleFactor) == 0)
    let modelVersion = ModelZoo.versionForModel(configuration.model ?? "")
    let refinerVersion: ModelVersion? = configuration.refinerModel.flatMap {
      guard $0 != configuration.model, ModelZoo.isModelDownloaded($0) else { return nil }
      let version = ModelZoo.versionForModel($0)
      guard
        version == modelVersion
          || ([.sdxlBase, .sdxlRefiner, .ssd1b].contains(version)
            && [.sdxlBase, .sdxlRefiner, .ssd1b].contains(modelVersion))
      else { return nil }
      return ModelZoo.versionForModel($0)
    }
    var alternativeDecoderVersion: AlternativeDecoderVersion? = nil
    for lora in configuration.loras {
      guard let file = lora.file else { continue }
      let loraVersion = LoRAZoo.versionForModel(file)
      guard LoRAZoo.isModelDownloaded(file),
        modelVersion == loraVersion || refinerVersion == loraVersion
          || (modelVersion == .kandinsky21 && loraVersion == .v1)
      else { continue }
      if let alternativeDecoder = LoRAZoo.alternativeDecoderForModel(file) {
        alternativeDecoderVersion = alternativeDecoder.1
      }
    }
    let startWidth: Int
    let startHeight: Int
    let latentsZoomFactor: Int
    if modelVersion == .wurstchenStageC {
      startWidth = image.shape[2] / 4 / scaleFactor
      startHeight = image.shape[1] / 4 / scaleFactor
      latentsZoomFactor = 4
    } else {
      startWidth = image.shape[2] / 8 / scaleFactor
      startHeight = image.shape[1] / 8 / scaleFactor
      latentsZoomFactor = 8
    }
    precondition(image.shape[2] / scaleFactor == mask.shape[1])
    precondition(image.shape[1] / scaleFactor == mask.shape[0])
    var imageNegMask2 = Tensor<FloatType>(
      .CPU, .NHWC(1, startHeight * latentsZoomFactor, startWidth * latentsZoomFactor, 1))
    // Anything marked 0 in mask2 will be retained until step - tEnc
    var mask2 = Tensor<FloatType>(.CPU, .NHWC(1, startHeight, startWidth, 1))
    var imageNegMask3 = Tensor<FloatType>(
      .CPU, .NHWC(1, startHeight * latentsZoomFactor, startWidth * latentsZoomFactor, 1))
    // Anything marked 0 in mask3 will be retained until the end
    var mask3 = Tensor<FloatType>(.CPU, .NHWC(1, startHeight, startWidth, 1))
    for y in 0..<startHeight {
      for x in 0..<startWidth {
        mask2[0, y, x, 0] = 0
        mask3[0, y, x, 0] = 0
      }
    }
    var exists0 = false
    var exists1 = false
    var exists2 = false
    var exists3 = false
    // have all 0, 1, 2, 3: 3 - skip all, 2 - mask3, 1 - mask2, mask3, 0 - mask3
    // have only 0, 1: 1 - mask2, mask3, 0 - mask3
    // have only 0, 2: 2 - mask3, 0 - skip all
    // have only 0, 3: 3 - skip all, 0 - mask3
    // have only 0, 1, 2: 2 - mask3, 1 - mask2, mask3, 0 - skip all
    // have only 0, 1, 3: 3 - skip all, 1 - mask2, mask3, 0 - mask3
    // have only 0, 2, 3: 3 - skip all, 2 - mask3, 0 - skip all
    for y in 0..<startHeight * latentsZoomFactor {
      let iy = y / latentsZoomFactor
      for x in 0..<startWidth * latentsZoomFactor {
        let ix = x / latentsZoomFactor
        let byteMask = mask[y, x]
        if (byteMask & 7) == 3 {
          imageNegMask2[0, y, x, 0] = 1
          imageNegMask3[0, y, x, 0] = 1
          exists3 = true
        } else if (byteMask & 7) == 2 || (byteMask & 7) == 4 {  // 4 is the same as 2.
          mask3[0, iy, ix, 0] = 1
          imageNegMask2[0, y, x, 0] = 1
          let alpha = FloatType(byteMask & 0xf8) / 255
          imageNegMask3[0, y, x, 0] = alpha
          exists2 = true
        } else if (byteMask & 7) == 1 {
          mask3[0, iy, ix, 0] = 1
          mask2[0, iy, ix, 0] = 1
          let alpha = FloatType(byteMask & 0xf8) / 255
          imageNegMask2[0, y, x, 0] = alpha
          imageNegMask3[0, y, x, 0] = alpha
          exists1 = true
        } else {
          imageNegMask2[0, y, x, 0] = 1
          imageNegMask3[0, y, x, 0] = 1
          exists0 = true
        }
      }
    }
    // Depends on the above combination, if exists0, we need to decide whether we skip all or put them in mask3.
    let overwrite0 = exists0 && (!exists2 || (exists3 && exists2 && exists1))
    if overwrite0 {
      for y in 0..<startHeight * latentsZoomFactor {
        let iy = y / latentsZoomFactor
        for x in 0..<startWidth * latentsZoomFactor {
          let ix = x / latentsZoomFactor
          let byteMask = mask[y, x]
          if (byteMask & 7) == 0 {
            mask3[0, iy, ix, 0] = 1
            let alpha = FloatType(byteMask & 0xf8) / 255
            imageNegMask3[0, y, x, 0] = alpha
          }
        }
      }
    }
    // If we use transparent generation, we don't care about area masked with 1 (i.e. nothing there). For these, we will turn off mask3.
    if alternativeDecoderVersion == .transparent {
      for y in 0..<startHeight * latentsZoomFactor {
        let iy = y / latentsZoomFactor
        for x in 0..<startWidth * latentsZoomFactor {
          let ix = x / latentsZoomFactor
          let byteMask = mask[y, x]
          if (byteMask & 7) == 1 {
            mask3[0, iy, ix, 0] = 0
            imageNegMask3[0, y, x, 0] = 1
          }
        }
      }
    }
    if configuration.maskBlurOutset > 0 {
      // Apply very soft outset on the latent space.
      // This is dilate because we want to increase the area under mask.
      let mask2f32 = Tensor<Float>(from: mask2.reshaped(.HWC(startHeight, startWidth, 1)))
      let maskBlurOutset =
        (Int(configuration.maskBlurOutset) + latentsZoomFactor - 1) / latentsZoomFactor  // Apply this much of mask blur outset.
      var b: UnsafeMutablePointer<ccv_dense_matrix_t>? = ccv_dense_matrix_new(
        Int32(startHeight), Int32(startWidth), Int32(CCV_C1 | CCV_32F), nil, 0)
      ccv_dilate(
        UnsafeMutableRawPointer(mask2f32.cTensor).assumingMemoryBound(to: ccv_dense_matrix_t.self),
        &b, 0, 3)
      for _ in 0..<(maskBlurOutset - 1) {
        ccv_dilate(b, &b, 0, 3)
      }
      // Now this is properly resized, we can claim a few things:
      // We can shift the viewModel to the new one, and update the image to use the new one as well.
      mask2 = Tensor<FloatType>(
        from: Tensor<Float>(
          .CPU, format: .NHWC, shape: [1, startHeight, startWidth, 1],
          unsafeMutablePointer: b!.pointee.data.f32, bindLifetimeOf: b!
        ).copied())
      let mask3f32 = Tensor<Float>(from: mask3.reshaped(.HWC(startHeight, startWidth, 1)))
      ccv_dilate(
        UnsafeMutableRawPointer(mask3f32.cTensor).assumingMemoryBound(to: ccv_dense_matrix_t.self),
        &b, 0, 3)
      for _ in 0..<(maskBlurOutset - 1) {
        ccv_dilate(b, &b, 0, 3)
      }
      // Now this is properly resized, we can claim a few things:
      // We can shift the viewModel to the new one, and update the image to use the new one as well.
      mask3 = Tensor<FloatType>(
        from: Tensor<Float>(
          .CPU, format: .NHWC, shape: [1, startHeight, startWidth, 1],
          unsafeMutablePointer: b!.pointee.data.f32, bindLifetimeOf: b!
        ).copied())
      ccv_matrix_free(b)
    }
    // Impossible to have none.
    assert(!(!exists0 && !exists1 && !exists2 && !exists3))
    // If no sophisticated mask, nothing to be done. For transparent, that means no exists2.
    if !exists2 && alternativeDecoderVersion == .transparent {
      return generateImageOnly(
        image, scaleFactor: scaleFactor, depth: depth, hints: hints, custom: custom,
        shuffles: shuffles, poses: poses, text: text,
        negativeText: negativeText, configuration: configuration,
        denoiserParameterization: denoiserParameterization, sampling: sampling,
        cancellation: cancellation, feedback: feedback)
    }
    // If no sophisticated mask, nothing to be done.
    // If only 3, meaning we are going to retain everything (???) then we just go as if image generation.
    // If only 0, meaning whatever, then we just go as if image generation.
    if (exists0 && !exists1 && !exists2 && !exists3)
      || (!exists0 && !exists1 && !exists2 && exists3)
    {
      return generateImageOnly(
        image, scaleFactor: scaleFactor, depth: depth, hints: hints, custom: custom,
        shuffles: shuffles, poses: poses, text: text,
        negativeText: negativeText, configuration: configuration,
        denoiserParameterization: denoiserParameterization, sampling: sampling,
        cancellation: cancellation, feedback: feedback)
    } else if !exists0 && exists1 && !exists2 && !exists3 {
      // If masked due to nothing only the whole page, run text generation only.
      return generateTextOnly(
        image, scaleFactor: scaleFactor, depth: depth, hints: hints, custom: custom,
        shuffles: shuffles, poses: poses, text: text,
        negativeText: negativeText, configuration: configuration,
        denoiserParameterization: denoiserParameterization, sampling: sampling,
        cancellation: cancellation, feedback: feedback)
    }
    var signposts = Set<ImageGeneratorSignpost>()
    if let faceRestoration = configuration.faceRestoration,
      EverythingZoo.isModelDownloaded(faceRestoration)
        && EverythingZoo.isModelDownloaded(EverythingZoo.parsenetForModel(faceRestoration))
    {
      signposts.insert(.faceRestored)
    }
    if let upscaler = configuration.upscaler, UpscalerZoo.isModelDownloaded(upscaler) {
      signposts.insert(.imageUpscaled)
    }
    // Either we're missing 1s, or we are in transparent mode (where 1 is ignored), go into here.
    if !exists1 || alternativeDecoderVersion == .transparent {
      // mask3 is not a typo. if no 1 exists, we only have mask3 relevant here if exists1 is missing.
      guard
        var result = generateImageWithMask2(
          image, scaleFactor: scaleFactor, imageNegMask2: imageNegMask3, mask2: mask3, depth: depth,
          hints: hints, custom: custom, shuffles: shuffles, poses: poses, text: text,
          negativeText: negativeText,
          configuration: configuration, denoiserParameterization: denoiserParameterization,
          sampling: sampling, signposts: &signposts, cancellation: cancellation, feedback: feedback)
      else {
        return (nil, 1)
      }
      if configuration.preserveOriginalAfterInpaint {
        let original = downscaleImage(image, scaleFactor: scaleFactor)
        for i in 0..<result.count {
          imageWithMask(
            &result[i], original: original, mask: mask, maskBlur: configuration.maskBlur,
            maskBlurOutset: Int(configuration.maskBlurOutset), overwrite0: overwrite0,
            transparent: alternativeDecoderVersion == .transparent)
        }
      }
      result = faceRestoreImages(result, configuration: configuration)
      if signposts.contains(.faceRestored) {
        guard feedback(.faceRestored, signposts, nil) else { return (nil, 1) }
      }
      let batch = upscaleImages(result, configuration: configuration)
      if signposts.contains(.imageUpscaled) {
        let _ = feedback(.imageUpscaled, signposts, nil)
      }
      return batch
    }
    // If there is no 2 or 3, only 0 or 1, we don't need to use mask2 (as it covered everything).
    if !exists2 && !exists3 {
      guard
        var result =
          generateImageWithMask1AndMask2(
            image, scaleFactor: scaleFactor, imageNegMask1: imageNegMask2, imageNegMask2: nil,
            mask1: mask2, mask2: nil, depth: depth, hints: hints, custom: custom,
            shuffles: shuffles, poses: poses, text: text,
            negativeText: negativeText, configuration: configuration,
            denoiserParameterization: denoiserParameterization, sampling: sampling,
            signposts: &signposts, cancellation: cancellation, feedback: feedback)
      else { return (nil, 1) }
      if configuration.strength == 0 && configuration.preserveOriginalAfterInpaint {
        let original = downscaleImage(image, scaleFactor: scaleFactor)
        for i in 0..<result.count {
          // If strength is zero, we need to put original back for mask = 0 case.
          imageWithMask(
            &result[i], original: original, mask: mask, maskBlur: configuration.maskBlur,
            maskBlurOutset: Int(configuration.maskBlurOutset), overwrite0: false, transparent: false
          )
        }
      }
      result = faceRestoreImages(result, configuration: configuration)
      if signposts.contains(.faceRestored) {
        guard feedback(.faceRestored, signposts, nil) else { return (nil, 1) }
      }
      let batch = upscaleImages(result, configuration: configuration)
      if signposts.contains(.imageUpscaled) {
        let _ = feedback(.imageUpscaled, signposts, nil)
      }
      return batch
    }
    guard
      var result = generateImageWithMask1AndMask2(
        image, scaleFactor: scaleFactor, imageNegMask1: imageNegMask2, imageNegMask2: imageNegMask3,
        mask1: mask2, mask2: mask3, depth: depth, hints: hints, custom: custom, shuffles: shuffles,
        poses: poses, text: text,
        negativeText: negativeText, configuration: configuration,
        denoiserParameterization: denoiserParameterization, sampling: sampling,
        signposts: &signposts, cancellation: cancellation, feedback: feedback)
    else {
      return (nil, 1)
    }
    if configuration.preserveOriginalAfterInpaint {
      let original = downscaleImage(image, scaleFactor: scaleFactor)
      for i in 0..<result.count {
        // If strength is zero, we need to put original back for mask = 0 case.
        imageWithMask(
          &result[i], original: original, mask: mask, maskBlur: configuration.maskBlur,
          maskBlurOutset: Int(configuration.maskBlurOutset),
          overwrite0: overwrite0 && configuration.strength > 0, transparent: false)
      }
    }
    result = faceRestoreImages(result, configuration: configuration)
    if signposts.contains(.faceRestored) {
      guard feedback(.faceRestored, signposts, nil) else { return (nil, 1) }
    }
    let batch = upscaleImages(result, configuration: configuration)
    if signposts.contains(.imageUpscaled) {
      let _ = feedback(.imageUpscaled, signposts, nil)
    }
    return batch
  }

  // This is vanilla inpainting, we directly go to steps - tEnc, and retain what we need till the end.
  private func generateImageWithMask2(
    _ image: Tensor<FloatType>, scaleFactor imageScaleFactor: Int, imageNegMask2: Tensor<FloatType>,
    mask2: Tensor<FloatType>, depth: Tensor<FloatType>?, hints: [ControlHintType: AnyTensor],
    custom: Tensor<FloatType>?, shuffles: [(Tensor<FloatType>, Float)],
    poses: [(Tensor<FloatType>, Float)], text: String,
    negativeText: String,
    configuration: GenerationConfiguration, denoiserParameterization: Denoiser.Parameterization,
    sampling: Sampling, signposts: inout Set<ImageGeneratorSignpost>,
    cancellation: (@escaping () -> Void) -> Void,
    feedback: @escaping (ImageGeneratorSignpost, Set<ImageGeneratorSignpost>, Tensor<FloatType>?) ->
      Bool
  ) -> [Tensor<FloatType>]? {
    let coreMLGuard = modelPreloader.beginCoreMLGuard()
    defer {
      if coreMLGuard {
        modelPreloader.endCoreMLGuard()
      }
    }
    let mfaGuard = modelPreloader.beginMFAGuard()
    defer {
      if mfaGuard {
        modelPreloader.endMFAGuard()
      }
    }
    let file =
      (configuration.model.flatMap {
        ModelZoo.isModelDownloaded($0) ? $0 : nil
      }) ?? ModelZoo.defaultSpecification.file
    let modifier = ImageGeneratorUtils.modifierForModel(
      file, LoRAs: configuration.loras.compactMap(\.file))
    let modelVersion = ModelZoo.versionForModel(file)
    let (qkNorm, dualAttentionLayers, distilledGuidanceLayers) =
      ModelZoo.MMDiTForModel(file).map {
        return ($0.qkNorm, $0.dualAttentionLayers, $0.distilledGuidanceLayers ?? 0)
      } ?? (false, [], 0)
    let textEncoderVersion = ModelZoo.textEncoderVersionForModel(file)
    let modelObjective = ModelZoo.objectiveForModel(file)
    let modelUpcastAttention = ModelZoo.isUpcastAttentionForModel(file)
    var textEncoderFiles: [String] =
      [
        ModelZoo.textEncoderForModel(file).flatMap {
          ModelZoo.isModelDownloaded($0) ? $0 : nil
        } ?? "clip_vit_l14_f16.ckpt"
      ]
      + ModelZoo.CLIPEncodersForModel(file).compactMap { ModelZoo.isModelDownloaded($0) ? $0 : nil }
    textEncoderFiles +=
      ((ModelZoo.T5EncoderForModel(file).flatMap { ModelZoo.isModelDownloaded($0) ? $0 : nil }).map
      { [$0] } ?? [])
    let diffusionMappingFile = ModelZoo.diffusionMappingForModel(file).flatMap {
      ModelZoo.isModelDownloaded($0) ? $0 : nil
    }
    let fpsId = Int(configuration.fpsId)
    let motionBucketId = Int(configuration.motionBucketId)
    let condAug = configuration.condAug
    let startFrameCfg = configuration.startFrameCfg
    let clipSkip = Int(configuration.clipSkip)
    let autoencoderFile =
      ModelZoo.autoencoderForModel(file).flatMap {
        ModelZoo.isModelDownloaded($0) ? $0 : nil
      } ?? ImageGeneratorUtils.defaultAutoencoder
    let isGuidanceEmbedEnabled =
      ModelZoo.guidanceEmbedForModel(file) && configuration.speedUpWithGuidanceEmbed
    var isCfgEnabled = !ModelZoo.isConsistencyModelForModel(file) && !isGuidanceEmbedEnabled
    let latentsScaling = ModelZoo.latentsScalingForModel(file)
    let paddedTextEncodingLength = ModelZoo.paddedTextEncodingLengthForModel(file)
    let conditioning = ModelZoo.conditioningForModel(file)
    let refinerVersion: ModelVersion? = configuration.refinerModel.flatMap {
      guard $0 != file, ModelZoo.isModelDownloaded($0) else { return nil }
      let version = ModelZoo.versionForModel($0)
      guard
        version == modelVersion
          || ([.sdxlBase, .sdxlRefiner, .ssd1b].contains(version)
            && [.sdxlBase, .sdxlRefiner, .ssd1b].contains(modelVersion))
      else { return nil }
      return ModelZoo.versionForModel($0)
    }
    let tiledDecoding = TiledConfiguration(
      isEnabled: configuration.tiledDecoding,
      tileSize: .init(
        width: Int(configuration.decodingTileWidth), height: Int(configuration.decodingTileHeight)),
      tileOverlap: Int(configuration.decodingTileOverlap))
    let tiledDiffusion = TiledConfiguration(
      isEnabled: configuration.tiledDiffusion,
      tileSize: .init(
        width: Int(configuration.diffusionTileWidth), height: Int(configuration.diffusionTileHeight)
      ), tileOverlap: Int(configuration.diffusionTileOverlap))
    var alternativeDecoderFilePath: String? = nil
    var alternativeDecoderVersion: AlternativeDecoderVersion? = nil
    let lora: [LoRAConfiguration] =
      (ModelZoo.builtinLoRAForModel(file)
        ? [
          LoRAConfiguration(
            file: ModelZoo.filePathForModelDownloaded(file), weight: 1, version: modelVersion,
            isLoHa: false, modifier: .none)
        ] : [])
      + configuration.loras.compactMap {
        guard let file = $0.file else { return nil }
        let loraVersion = LoRAZoo.versionForModel(file)
        guard LoRAZoo.isModelDownloaded(file),
          modelVersion == loraVersion || refinerVersion == loraVersion
            || (modelVersion == .kandinsky21 && loraVersion == .v1)
        else { return nil }
        if LoRAZoo.isConsistencyModelForModel(file) {
          isCfgEnabled = false
        }
        if let alternativeDecoder = LoRAZoo.alternativeDecoderForModel(file) {
          alternativeDecoderFilePath = LoRAZoo.filePathForModelDownloaded(alternativeDecoder.0)
          alternativeDecoderVersion = alternativeDecoder.1
        }
        return LoRAConfiguration(
          file: LoRAZoo.filePathForModelDownloaded(file), weight: $0.weight, version: loraVersion,
          isLoHa: LoRAZoo.isLoHaForModel(file), modifier: LoRAZoo.modifierForModel(file))
      }
    if modelVersion == .v2 || modelVersion == .sdxlBase || modelVersion == .sdxlRefiner
      || modelVersion == .ssd1b || modelVersion == .svdI2v || modelVersion == .wurstchenStageC
      || modelVersion == .sd3 || modelVersion == .pixart
    {
      DynamicGraph.flags = .disableMixedMPSGEMM
    }
    if !DeviceCapability.isMemoryMapBufferSupported {
      DynamicGraph.flags.insert(.disableMmapMTLBuffer)
    }
    let isMFAEnabled = DeviceCapability.isMFAEnabled.load(ordering: .acquiring)
    if !isMFAEnabled {
      DynamicGraph.flags.insert(.disableMetalFlashAttention)
    } else {
      DynamicGraph.flags.remove(.disableMetalFlashAttention)
      if !DeviceCapability.isMFAGEMMFaster {
        DynamicGraph.flags.insert(.disableMFAGEMM)
      }
    }
    var hasHints = Set(hints.keys)
    if !poses.isEmpty {
      hasHints.insert(.pose)
    }
    let (
      canInjectControls, canInjectT2IAdapters, canInjectAttentionKVs, _, injectIPAdapterLengths,
      canInjectedControls
    ) =
      ImageGeneratorUtils.canInjectControls(
        hasImage: true, hasDepth: depth != nil, hasHints: hasHints, hasCustom: custom != nil,
        shuffleCount: shuffles.count, controls: configuration.controls,
        version: modelVersion)
    let queueWatermark = DynamicGraph.queueWatermark
    if (canInjectControls && modelVersion == .v2) && !DeviceCapability.isMaxPerformance {
      DynamicGraph.queueWatermark = 8
    }
    defer {
      if (canInjectControls && modelVersion == .v2) && !DeviceCapability.isMaxPerformance {
        DynamicGraph.queueWatermark = queueWatermark
      }
    }
    let batchSize =
      ImageGeneratorUtils.isVideoModel(modelVersion) ? 1 : Int(configuration.batchSize)
    precondition(batchSize > 0)
    let textGuidanceScale = configuration.guidanceScale
    let imageGuidanceScale = configuration.imageGuidanceScale
    let guidanceEmbed = configuration.guidanceEmbed
    let originalSize =
      configuration.originalImageWidth == 0 || configuration.originalImageHeight == 0
      ? (width: Int(configuration.startWidth) * 64, height: Int(configuration.startHeight) * 64)
      : (
        width: Int(configuration.originalImageWidth), height: Int(configuration.originalImageHeight)
      )
    let cropTopLeft = (top: Int(configuration.cropTop), left: Int(configuration.cropLeft))
    let targetSize =
      configuration.targetImageWidth == 0 || configuration.targetImageHeight == 0
      ? (width: Int(configuration.startWidth) * 64, height: Int(configuration.startHeight) * 64)
      : (width: Int(configuration.targetImageWidth), height: Int(configuration.targetImageHeight))
    let negativeOriginalSize =
      configuration.negativeOriginalImageWidth == 0
        || configuration.negativeOriginalImageHeight == 0
      ? originalSize
      : (
        width: Int(configuration.negativeOriginalImageWidth),
        height: Int(configuration.negativeOriginalImageHeight)
      )
    let aestheticScore = configuration.aestheticScore
    let negativeAestheticScore = configuration.negativeAestheticScore
    let zeroNegativePrompt = configuration.zeroNegativePrompt
    let sharpness = configuration.sharpness
    let strength = configuration.strength
    let highPrecisionForAutoencoder = ModelZoo.isHighPrecisionAutoencoderForModel(file)
    precondition(image.shape[2] % (64 * imageScaleFactor) == 0)
    precondition(image.shape[1] % (64 * imageScaleFactor) == 0)
    let startWidth: Int
    let startHeight: Int
    let channels: Int
    let firstStageFilePath: String
    if modelVersion == .wurstchenStageC {
      (startWidth, startHeight) = stageCLatentsSize(configuration)
      channels = 16
      firstStageFilePath = ModelZoo.filePathForModelDownloaded(file)
    } else {
      startWidth = image.shape[2] / 8 / imageScaleFactor
      startHeight = image.shape[1] / 8 / imageScaleFactor
      switch modelVersion {
      case .wurstchenStageC, .sd3, .sd3Large, .flux1, .hunyuanVideo, .wan21_1_3b, .wan21_14b,
        .hiDreamI1:
        channels = 16
      case .auraflow, .kandinsky21, .pixart, .sdxlBase, .sdxlRefiner, .ssd1b, .svdI2v, .v1, .v2,
        .wurstchenStageB:
        channels = 4
      }
      firstStageFilePath = ModelZoo.filePathForModelDownloaded(autoencoderFile)
    }
    let imageScale = DeviceCapability.Scale(
      widthScale: UInt16(startWidth / 8), heightScale: UInt16(startHeight / 8))
    let isHighPrecisionVAEFallbackEnabled = DeviceCapability.isHighPrecisionVAEFallbackEnabled(
      scale: imageScale)
    let isQuantizedModel = ModelZoo.isQuantizedModel(file)
    let is8BitModel = ModelZoo.is8BitModel(file)
    let canRunLoRASeparately = modelPreloader.canRunLoRASeparately
    let externalOnDemand = modelPreloader.externalOnDemand(
      version: modelVersion,
      scale: DeviceCapability.Scale(
        widthScale: configuration.startWidth, heightScale: configuration.startHeight),
      variant: .unet, injectedControls: canInjectedControls,
      is8BitModel: is8BitModel && (canRunLoRASeparately || lora.isEmpty))
    let refiner: Refiner? = configuration.refinerModel.flatMap {
      guard $0 != file, ModelZoo.isModelDownloaded($0) else { return nil }
      let version = ModelZoo.versionForModel($0)
      guard
        version == modelVersion
          || ([.sdxlBase, .sdxlRefiner, .ssd1b].contains(version)
            && [.sdxlBase, .sdxlRefiner, .ssd1b].contains(modelVersion))
      else { return nil }
      return Refiner(
        start: configuration.refinerStart, filePath: ModelZoo.filePathForModelDownloaded($0),
        externalOnDemand: externalOnDemand, version: ModelZoo.versionForModel($0),
        isQuantizedModel: ModelZoo.isQuantizedModel($0),
        isConsistencyModel: ModelZoo.isConsistencyModelForModel($0))
    }
    let controlExternalOnDemand = modelPreloader.externalOnDemand(
      version: modelVersion,
      scale: DeviceCapability.Scale(
        widthScale: configuration.startWidth, heightScale: configuration.startHeight),
      variant: .control, injectedControls: canInjectedControls, suffix: "ctrl")
    let textEncoderExternalOnDemand = modelPreloader.externalOnDemand(
      version: modelVersion,
      scale: DeviceCapability.Scale(
        widthScale: configuration.startWidth, heightScale: configuration.startHeight),
      variant: .textEncoder, injectedControls: 0)
    let vaeExternalOnDemand = modelPreloader.externalOnDemand(
      version: modelVersion,
      scale: DeviceCapability.Scale(
        widthScale: configuration.startWidth, heightScale: configuration.startHeight),
      variant: .autoencoder, injectedControls: canInjectedControls)
    let dmExternalOnDemand = modelPreloader.externalOnDemand(
      version: modelVersion,
      scale: DeviceCapability.Scale(
        widthScale: configuration.startWidth, heightScale: configuration.startHeight),
      variant: .diffusionMapping, injectedControls: 0)
    let teaCache =
      ModelZoo.teaCacheCoefficientsForModel(file).map {
        var teaCacheEnd =
          configuration.teaCacheEnd < 0
          ? Int(configuration.steps) + 1 + Int(configuration.teaCacheEnd)
          : Int(configuration.teaCacheEnd)
        let teaCacheStart = min(max(Int(configuration.teaCacheStart), 0), Int(configuration.steps))
        teaCacheEnd = min(max(max(teaCacheStart, teaCacheEnd), 0), Int(configuration.steps))
        return TeaCacheConfiguration(
          coefficients: $0,
          steps: min(teaCacheStart, teaCacheEnd)...max(teaCacheStart, teaCacheEnd),
          threshold: configuration.teaCache ? configuration.teaCacheThreshold : 0,
          maxSkipSteps: Int(configuration.teaCacheMaxSkipSteps))
      }
      ?? TeaCacheConfiguration(
        coefficients: (0, 0, 0, 0, 0), steps: 0...0, threshold: 0, maxSkipSteps: 0)
    let causalInference =
      configuration.causalInferenceEnabled ? Int(configuration.causalInference) : 0
    let sampler = LocalImageGenerator.sampler(
      from: configuration.sampler, isCfgEnabled: isCfgEnabled,
      filePath: ModelZoo.filePathForModelDownloaded(file), modifier: modifier,
      version: modelVersion, qkNorm: qkNorm, dualAttentionLayers: dualAttentionLayers,
      distilledGuidanceLayers: distilledGuidanceLayers, usesFlashAttention: isMFAEnabled,
      objective: modelObjective,
      upcastAttention: modelUpcastAttention,
      externalOnDemand: externalOnDemand, injectControls: canInjectControls,
      injectT2IAdapters: canInjectT2IAdapters, injectAttentionKV: canInjectAttentionKVs,
      injectIPAdapterLengths: injectIPAdapterLengths,
      lora: lora, isGuidanceEmbedEnabled: isGuidanceEmbedEnabled,
      isQuantizedModel: isQuantizedModel,
      canRunLoRASeparately: canRunLoRASeparately,
      stochasticSamplingGamma: configuration.stochasticSamplingGamma,
      conditioning: conditioning, parameterization: denoiserParameterization,
      tiledDiffusion: tiledDiffusion, teaCache: teaCache, causalInference: causalInference,
      weightsCache: weightsCache,
      of: FloatType.self)
    let initTimestep = sampler.timestep(for: strength, sampling: sampling)
    let graph = DynamicGraph()
    if externalOnDemand
      || externalOnDemandPartially(
        version: modelVersion, memoryCapacity: DeviceCapability.memoryCapacity,
        externalOnDemand: externalOnDemand)
    {
      TensorData.makeExternalData(for: ModelZoo.filePathForModelDownloaded(file), graph: graph)
      for stageModel in ModelZoo.stageModelsForModel(file) {
        TensorData.makeExternalData(
          for: ModelZoo.filePathForModelDownloaded(stageModel), graph: graph)
      }
    }
    if textEncoderExternalOnDemand {
      TensorData.makeExternalData(
        for: ModelZoo.filePathForModelDownloaded(textEncoderFiles[0]), graph: graph)
    }
    if controlExternalOnDemand {
      for file in Set(configuration.controls.compactMap { $0.file }) {
        TensorData.makeExternalData(for: ModelZoo.filePathForModelDownloaded(file), graph: graph)
      }
    }
    if vaeExternalOnDemand {
      TensorData.makeExternalData(
        for: ModelZoo.filePathForModelDownloaded(autoencoderFile), graph: graph)
    }
    if dmExternalOnDemand, let diffusionMappingFile = diffusionMappingFile {
      TensorData.makeExternalData(
        for: ModelZoo.filePathForModelDownloaded(diffusionMappingFile), graph: graph)
    }
    let potentials = lora.map { ($0.file as NSString).lastPathComponent }
    let (
      tokensTensors, positionTensors, embedMask, injectedEmbeddings, unconditionalAttentionWeights,
      attentionWeights, hasNonOneWeights, tokenLengthUncond, tokenLengthCond, lengthsOfUncond,
      lengthsOfCond
    ) = tokenize(
      graph: graph, modelVersion: modelVersion, textEncoderVersion: textEncoderVersion,
      paddedTextEncodingLength: paddedTextEncodingLength, text: text, negativeText: negativeText,
      negativePromptForImagePrior: configuration.negativePromptForImagePrior,
      potentials: potentials, T5TextEncoder: configuration.t5TextEncoder,
      clipL: configuration.separateClipL ? (configuration.clipLText ?? "") : nil,
      openClipG: configuration.separateOpenClipG ? (configuration.openClipGText ?? "") : nil,
      t5: configuration.separateT5 ? (configuration.t5Text ?? "") : nil
    )
    signposts.formUnion([
      .textEncoded, .imageEncoded, .sampling(sampling.steps - initTimestep.roundedDownStartStep),
      .imageDecoded,
    ])
    if modelVersion == .wurstchenStageC {
      let initTimestep = sampler.timestep(
        for: strength,
        sampling: Sampling(
          steps: Int(configuration.stage2Steps), shift: Double(configuration.stage2Shift)))
      signposts.insert(
        .secondPassSampling(Int(configuration.stage2Steps) - initTimestep.roundedDownStartStep))
      signposts.insert(.secondPassImageEncoded)
    }
    return graph.withNoGrad {
      let injectedTextEmbeddings = generateInjectedTextEmbeddings(
        batchSize: batchSize, startHeight: startHeight, startWidth: startWidth, image: image,
        graph: graph, hints: hints, custom: custom, shuffles: shuffles, pose: poses.first?.0,
        controls: configuration.controls,
        version: modelVersion, tiledDiffusion: tiledDiffusion, usesFlashAttention: isMFAEnabled,
        externalOnDemand: controlExternalOnDemand, cancellation: cancellation)
      let (tokenLengthUncond, tokenLengthCond) = ControlModel<FloatType>.modifyTextEmbeddings(
        tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
        injecteds: injectedTextEmbeddings)
      let tokenLength = max(tokenLengthUncond, tokenLengthCond)
      let textEncoder = TextEncoder<FloatType>(
        filePaths: textEncoderFiles.map { ModelZoo.filePathForModelDownloaded($0) },
        version: modelVersion, textEncoderVersion: textEncoderVersion,
        isCfgEnabled: isCfgEnabled,
        usesFlashAttention: isMFAEnabled && DeviceCapability.isMFACausalAttentionMaskSupported,
        injectEmbeddings: !injectedEmbeddings.isEmpty,
        externalOnDemand: textEncoderExternalOnDemand,
        deviceProperties: DeviceCapability.deviceProperties, weightsCache: weightsCache,
        maxLength: tokenLength, clipSkip: clipSkip, lora: lora)
      let image = downscaleImageAndToGPU(
        graph.variable(image), scaleFactor: imageScaleFactor)
      let textEncodings = modelPreloader.consumeTextModels(
        textEncoder.encode(
          tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
          tokens: tokensTensors, positions: positionTensors, mask: embedMask,
          injectedEmbeddings: injectedEmbeddings, image: [image], lengthsOfUncond: lengthsOfUncond,
          lengthsOfCond: lengthsOfCond, injectedTextEmbeddings: injectedTextEmbeddings,
          textModels: modelPreloader.retrieveTextModels(textEncoder: textEncoder)),
        textEncoder: textEncoder)
      var c: [DynamicGraph.Tensor<FloatType>]
      var extraProjection: DynamicGraph.Tensor<FloatType>?
      DynamicGraph.setSeed(configuration.seed)
      if modelVersion == .kandinsky21, let diffusionMappingFile = diffusionMappingFile {
        let diffusionMapping = DiffusionMapping<FloatType>(
          filePath: ModelZoo.filePathForModelDownloaded(diffusionMappingFile),
          usesFlashAttention: isMFAEnabled && DeviceCapability.isMFACausalAttentionMaskSupported,
          steps: Int(configuration.imagePriorSteps),
          negativePromptForImagePrior: configuration.negativePromptForImagePrior,
          CLIPWeight: configuration.clipWeight, externalOnDemand: dmExternalOnDemand)
        let imageEmb = diffusionMapping.sample(
          textEncoding: textEncodings[2], textEmbedding: textEncodings[3], tokens: tokensTensors[1])
        let kandinskyEmbedding = KandinskyEmbedding<FloatType>(
          filePath: ModelZoo.filePathForModelDownloaded(file))
        let (xfProj, xfOut) = kandinskyEmbedding.encode(
          textEncoding: textEncodings[0], textEmbedding: textEncodings[1], imageEmbedding: imageEmb)
        extraProjection = xfProj
        c = [xfOut]
      } else {
        extraProjection = nil
        c = textEncodings
      }
      (c, extraProjection) = repeatConditionsToMatchBatchSize(
        c: c, extraProjection: extraProjection,
        unconditionalAttentionWeights: unconditionalAttentionWeights,
        attentionWeights: attentionWeights, version: modelVersion, tokenLength: tokenLength,
        batchSize: batchSize, hasNonOneWeights: hasNonOneWeights)
      guard feedback(.textEncoded, signposts, nil) else { return nil }
      var firstStage = FirstStage<FloatType>(
        filePath: firstStageFilePath, version: modelVersion,
        latentsScaling: latentsScaling, highPrecisionKeysAndValues: highPrecisionForAutoencoder,
        highPrecisionFallback: isHighPrecisionVAEFallbackEnabled,
        tiledDecoding: tiledDecoding, tiledDiffusion: tiledDiffusion,
        externalOnDemand: vaeExternalOnDemand, alternativeUsesFlashAttention: isMFAEnabled,
        alternativeFilePath: alternativeDecoderFilePath,
        alternativeDecoderVersion: alternativeDecoderVersion,
        deviceProperties: DeviceCapability.deviceProperties)
      var firstPassImage: DynamicGraph.Tensor<FloatType>
      if modelVersion == .wurstchenStageC {
        // Try to resize the input image so we can encode with EfficientNetv2s properly.
        if image.shape[1] != startHeight * 32 || image.shape[2] != startWidth * 32 {
          firstPassImage = Upsample(
            .bilinear, widthScale: Float(startWidth * 32) / Float(image.shape[2]),
            heightScale: Float(startHeight * 32) / Float(image.shape[1]))(image)
        } else {
          firstPassImage = image
        }
      } else {
        firstPassImage = image
      }
      var batchSize = (batchSize, 0)
      switch modelVersion {
      case .svdI2v:
        batchSize = (Int(configuration.numFrames), 0)
      case .hunyuanVideo, .wan21_1_3b, .wan21_14b:
        batchSize = injectReferenceFrames(
          batchSize: ((Int(configuration.numFrames) - 1) / 4) + 1, version: modelVersion,
          canInjectControls: canInjectControls, shuffleCount: shuffles.count,
          hasCustom: custom != nil)
      case .auraflow, .flux1, .kandinsky21, .pixart, .sd3, .sd3Large, .sdxlBase, .sdxlRefiner,
        .ssd1b, .v1, .v2, .wurstchenStageB, .wurstchenStageC, .hiDreamI1:
        break
      }
      let imageSize: Int
      let firstPassImageForSample: DynamicGraph.Tensor<FloatType>?
      (imageSize, firstPassImage, firstPassImageForSample) = expandImageForEncoding(
        batchSize: batchSize, version: modelVersion, modifier: modifier, image: firstPassImage)
      var (sample, _) = modelPreloader.consumeFirstStageSample(
        firstStage.sample(
          firstPassImageForSample ?? firstPassImage,
          encoder: modelPreloader.retrieveFirstStageEncoder(
            firstStage: firstStage, scale: imageScale), cancellation: cancellation),
        firstStage: firstStage, scale: imageScale)
      let depthImage = depth.map {
        let depthImage = graph.variable($0.toGPU(0))
        let depthHeight = depthImage.shape[1]
        let depthWidth = depthImage.shape[2]
        guard depthHeight != startHeight * 8 || depthWidth != startWidth * 8 else {
          return depthImage
        }
        return Upsample(
          .bilinear, widthScale: Float(startHeight * 8) / Float(depthHeight),
          heightScale: Float(startWidth * 8) / Float(depthWidth))(depthImage)
      }
      let injectedControls = generateInjectedControls(
        graph: graph, batchSize: batchSize.0, startHeight: startHeight, startWidth: startWidth,
        image: image,
        depth: depthImage, hints: hints, custom: custom, shuffles: shuffles, pose: poses.last?.0,
        mask: imageNegMask2, controls: configuration.controls, version: modelVersion,
        tiledDiffusion: tiledDiffusion, usesFlashAttention: isMFAEnabled,
        externalOnDemand: controlExternalOnDemand, steps: sampling.steps, firstStage: firstStage,
        cancellation: cancellation)
      guard feedback(.controlsGenerated, signposts, nil) else { return nil }
      var maskedImage: DynamicGraph.Tensor<FloatType>? = nil
      if modifier == .inpainting || modifier == .editing || modifier == .double
        || modelVersion == .svdI2v
      {
        if !isI2v(version: modelVersion, modifier: modifier) && modifier != .editing {
          firstPassImage = firstPassImage .* graph.variable(imageNegMask2.toGPU(0))
        }
        let encodedImage = modelPreloader.consumeFirstStageEncode(
          firstStage.encode(
            firstPassImage,
            encoder: modelPreloader.retrieveFirstStageEncoder(
              firstStage: firstStage, scale: imageScale), cancellation: cancellation),
          firstStage: firstStage, scale: imageScale
        )
        if modifier == .inpainting {
          maskedImage = firstStage.scale(
            encodedImage[0..<imageSize, 0..<startHeight, 0..<startWidth, 0..<channels].copied())
        } else if modifier == .editing {
          if modelVersion == .v1 {
            maskedImage = encodedImage[0..<imageSize, 0..<startHeight, 0..<startWidth, 0..<channels]
              .copied()
          } else {
            maskedImage = firstStage.scale(
              encodedImage[0..<imageSize, 0..<startHeight, 0..<startWidth, 0..<channels].copied())
          }
        } else {
          maskedImage = encodedImage[0..<imageSize, 0..<startHeight, 0..<startWidth, 0..<channels]
            .copied()
        }
      }
      if let image = maskedImage {
        maskedImage = injectVACEFrames(
          batchSize: batchSize, version: modelVersion, image: image,
          injectedControls: injectedControls)
      }
      firstPassImage = injectVACEFrames(
        batchSize: batchSize, version: modelVersion, image: firstPassImage,
        injectedControls: injectedControls)
      sample = injectVACEFrames(
        batchSize: batchSize, version: modelVersion, image: sample,
        injectedControls: injectedControls)
      guard feedback(.imageEncoded, signposts, nil) else { return nil }
      let noise = randomLatentNoise(
        graph: graph, batchSize: batchSize.0, startHeight: startHeight,
        startWidth: startWidth, channels: channels, seed: configuration.seed,
        seedMode: configuration.seedMode)
      var initMask = graph.variable(mask2.toGPU(0))
      if initMask.shape[1] != startHeight || initMask.shape[2] != startWidth {
        initMask = Upsample(
          .nearest, widthScale: Float(startWidth) / Float(initMask.shape[2]),
          heightScale: Float(startHeight) / Float(initMask.shape[1]))(initMask)
      }
      var initNegMask = graph.variable(
        .GPU(0), .NHWC(1, startHeight, startWidth, 1), of: FloatType.self)
      initNegMask.full(1)
      initNegMask = initNegMask - initMask
      let x_T: DynamicGraph.Tensor<FloatType>
      if initTimestep.startStep > 0 {
        let sampleScaleFactor = sampler.sampleScaleFactor(
          at: initTimestep.startStep, sampling: sampling)
        let noiseScaleFactor = sampler.noiseScaleFactor(
          at: initTimestep.startStep, sampling: sampling)
        let zEnc = sampleScaleFactor * sample + noiseScaleFactor * noise
        x_T = zEnc
      } else {
        x_T = noise
      }
      let customImage = custom.map {
        let customImage = graph.variable($0.toGPU(0))
        let customHeight = customImage.shape[1]
        let customWidth = customImage.shape[2]
        guard customHeight != startHeight * 8 || customWidth != startWidth * 8 else {
          return customImage
        }
        return Upsample(
          .bilinear, widthScale: Float(startHeight * 8) / Float(customHeight),
          heightScale: Float(startWidth * 8) / Float(customWidth))(customImage)
      }
      let imageCond = encodeImageCond(
        image: firstPassImage, depth: depthImage, custom: customImage, modifier: modifier,
        version: modelVersion, firstStage: firstStage, usesFlashAttention: isMFAEnabled)
      var initMaskMaybe: DynamicGraph.Tensor<FloatType>? = initMask
      var initNegMaskMaybe: DynamicGraph.Tensor<FloatType>? = initNegMask
      if modifier == .inpainting {
        maskedImage = concatMaskWithMaskedImage(
          hasImage: true, batchSize: batchSize,
          version: modelVersion, encodedImage: maskedImage!, encodedMask: initMask,
          imageNegMask: graph.variable(imageNegMask2.toGPU(0)))
        if !configuration.preserveOriginalAfterInpaint {
          initMaskMaybe = nil
          initNegMaskMaybe = nil
        }
      }
      guard
        var x =
          try? modelPreloader.consumeUNet(
            sampler.sample(
              x_T,
              unets: modelPreloader.retrieveUNet(
                sampler: sampler, scale: imageScale, tokenLengthUncond: tokenLengthUncond,
                tokenLengthCond: tokenLengthCond), sample: sample,
              conditionImage: maskedImage ?? imageCond, mask: initMaskMaybe,
              negMask: initNegMaskMaybe,
              conditioning: c, tokenLengthUncond: tokenLengthUncond,
              tokenLengthCond: tokenLengthCond, extraProjection: extraProjection,
              injectedControls: injectedControls, textGuidanceScale: textGuidanceScale,
              imageGuidanceScale: imageGuidanceScale, guidanceEmbed: guidanceEmbed,
              startStep: (
                integral: initTimestep.roundedDownStartStep, fractional: initTimestep.startStep
              ),
              endStep: (integral: sampling.steps, fractional: Float(sampling.steps)),
              originalSize: originalSize, cropTopLeft: cropTopLeft,
              targetSize: targetSize, aestheticScore: aestheticScore,
              negativeOriginalSize: negativeOriginalSize,
              negativeAestheticScore: negativeAestheticScore,
              zeroNegativePrompt: zeroNegativePrompt, refiner: refiner, fpsId: fpsId,
              motionBucketId: motionBucketId, condAug: condAug, startFrameCfg: startFrameCfg,
              sharpness: sharpness, sampling: sampling, cancellation: cancellation
            ) { step, tensor in
              feedback(.sampling(step), signposts, tensor)
            }, sampler: sampler, scale: imageScale, tokenLengthUncond: tokenLengthUncond,
            tokenLengthCond: tokenLengthCond)
      else {
        return nil
      }
      guard
        feedback(.sampling(sampling.steps - initTimestep.roundedDownStartStep), signposts, nil)
      else {
        return nil
      }
      // If it is Wurstchen, run the stage 2.
      if modelVersion == .wurstchenStageC {
        guard feedback(.imageDecoded, signposts, nil) else { return nil }
        firstStage = FirstStage<FloatType>(
          filePath: ModelZoo.filePathForModelDownloaded(autoencoderFile), version: .wurstchenStageB,
          latentsScaling: latentsScaling, highPrecisionKeysAndValues: highPrecisionForAutoencoder,
          highPrecisionFallback: isHighPrecisionVAEFallbackEnabled,
          tiledDecoding: TiledConfiguration(
            isEnabled: false,
            tileSize: .init(width: 0, height: 0), tileOverlap: 0),
          tiledDiffusion: tiledDiffusion,
          externalOnDemand: vaeExternalOnDemand, alternativeUsesFlashAttention: isMFAEnabled,
          alternativeFilePath: alternativeDecoderFilePath,
          alternativeDecoderVersion: alternativeDecoderVersion,
          deviceProperties: DeviceCapability.deviceProperties)
        let (sample, _) = modelPreloader.consumeFirstStageSample(
          firstStage.sample(
            image,
            encoder: modelPreloader.retrieveFirstStageEncoder(
              firstStage: firstStage, scale: imageScale), cancellation: cancellation),
          firstStage: firstStage, scale: imageScale
        )
        let startHeight = Int(configuration.startHeight) * 16
        let startWidth = Int(configuration.startWidth) * 16
        let channels = 4
        guard feedback(.secondPassImageEncoded, signposts, nil) else { return nil }
        let secondPassModelVersion = ModelVersion.wurstchenStageB
        let secondPassModelFilePath = ModelZoo.filePathForModelDownloaded(
          ModelZoo.stageModelsForModel(file)[0])
        let secondPassSampler = LocalImageGenerator.sampler(
          from: configuration.sampler, isCfgEnabled: isCfgEnabled,
          filePath: secondPassModelFilePath, modifier: modifier,
          version: secondPassModelVersion, qkNorm: qkNorm, dualAttentionLayers: dualAttentionLayers,
          distilledGuidanceLayers: distilledGuidanceLayers, usesFlashAttention: isMFAEnabled,
          objective: modelObjective,
          upcastAttention: modelUpcastAttention,
          externalOnDemand: externalOnDemand,
          injectControls: canInjectControls, injectT2IAdapters: canInjectT2IAdapters,
          injectAttentionKV: canInjectAttentionKVs,
          injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
          isGuidanceEmbedEnabled: isGuidanceEmbedEnabled, isQuantizedModel: isQuantizedModel,
          canRunLoRASeparately: canRunLoRASeparately,
          stochasticSamplingGamma: configuration.stochasticSamplingGamma,
          conditioning: conditioning, parameterization: denoiserParameterization,
          tiledDiffusion: tiledDiffusion, teaCache: teaCache, causalInference: causalInference,
          weightsCache: weightsCache,
          of: FloatType.self
        )
        let noise = randomLatentNoise(
          graph: graph, batchSize: batchSize.0, startHeight: startHeight,
          startWidth: startWidth, channels: channels, seed: configuration.seed,
          seedMode: configuration.seedMode)
        c.append(x)
        var initMask = graph.variable(mask2.toGPU(0))
        if initMask.shape[1] != startHeight || initMask.shape[2] != startWidth {
          initMask = Upsample(
            .nearest, widthScale: Float(startWidth) / Float(initMask.shape[2]),
            heightScale: Float(startHeight) / Float(initMask.shape[1]))(initMask)
        }
        var initNegMask = graph.variable(
          .GPU(0), .NHWC(1, startHeight, startWidth, 1), of: FloatType.self)
        initNegMask.full(1)
        initNegMask = initNegMask - initMask
        let secondPassTextGuidance = configuration.stage2Cfg
        let secondPassSampling = Sampling(
          steps: Int(configuration.stage2Steps), shift: Double(configuration.stage2Shift))
        let initTimestep = sampler.timestep(for: strength, sampling: secondPassSampling)
        let x_T: DynamicGraph.Tensor<FloatType>
        if initTimestep.startStep > 0 {
          let sampleScaleFactor = sampler.sampleScaleFactor(
            at: initTimestep.startStep, sampling: secondPassSampling)
          let noiseScaleFactor = sampler.noiseScaleFactor(
            at: initTimestep.startStep, sampling: secondPassSampling)
          let zEnc = sampleScaleFactor * sample + noiseScaleFactor * noise
          x_T = zEnc
        } else {
          x_T = noise
        }
        guard
          let b =
            try? modelPreloader.consumeUNet(
              secondPassSampler.sample(
                x_T,
                unets: modelPreloader.retrieveUNet(
                  sampler: secondPassSampler, scale: imageScale,
                  tokenLengthUncond: tokenLengthUncond,
                  tokenLengthCond: tokenLengthCond),
                sample: sample, conditionImage: nil, mask: initMask,
                negMask: initNegMask, conditioning: c, tokenLengthUncond: tokenLengthUncond,
                tokenLengthCond: tokenLengthCond, extraProjection: extraProjection,
                injectedControls: [],  // TODO: Support injectedControls for this.
                textGuidanceScale: secondPassTextGuidance,
                imageGuidanceScale: imageGuidanceScale, guidanceEmbed: guidanceEmbed,
                startStep: (
                  integral: initTimestep.roundedDownStartStep, fractional: initTimestep.startStep
                ),
                endStep: (
                  integral: secondPassSampling.steps, fractional: Float(secondPassSampling.steps)
                ),
                originalSize: originalSize, cropTopLeft: cropTopLeft,
                targetSize: targetSize, aestheticScore: aestheticScore,
                negativeOriginalSize: negativeOriginalSize,
                negativeAestheticScore: negativeAestheticScore,
                zeroNegativePrompt: zeroNegativePrompt, refiner: refiner, fpsId: fpsId,
                motionBucketId: motionBucketId, condAug: condAug, startFrameCfg: startFrameCfg,
                sharpness: sharpness, sampling: secondPassSampling, cancellation: cancellation
              ) { step, tensor in
                feedback(.secondPassSampling(step), signposts, tensor)
              }, sampler: secondPassSampler, scale: imageScale,
              tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond)
        else {
          return nil
        }
        guard
          feedback(
            .secondPassSampling(secondPassSampling.steps - initTimestep.roundedDownStartStep),
            signposts, nil)
        else {
          return nil
        }
        x = b
      }
      if DeviceCapability.isLowPerformance {
        graph.garbageCollect()
      }
      if modelVersion == .v2 || modelVersion == .sdxlBase || modelVersion == .sdxlRefiner
        || modelVersion == .ssd1b
      {
        DynamicGraph.flags = []
      }
      if !DeviceCapability.isMemoryMapBufferSupported {
        DynamicGraph.flags.insert(.disableMmapMTLBuffer)
      }
      if !isMFAEnabled {
        DynamicGraph.flags.insert(.disableMetalFlashAttention)
      } else {
        DynamicGraph.flags.remove(.disableMetalFlashAttention)
        if !DeviceCapability.isMFAGEMMFaster {
          DynamicGraph.flags.insert(.disableMFAGEMM)
        }
      }
      let result = DynamicGraph.Tensor<FloatType>(
        from: modelPreloader.consumeFirstStageDecode(
          firstStage.decode(
            x, batchSize: (batchSize.0 - batchSize.1, 0),
            decoder: modelPreloader.retrieveFirstStageDecoder(
              firstStage: firstStage, scale: imageScale), cancellation: cancellation),
          firstStage: firstStage, scale: imageScale
        )
      ).rawValue.toCPU()
      guard !isNaN(result) else { return nil }
      if modelVersion == .wurstchenStageC {
        guard feedback(.secondPassImageDecoded, signposts, nil) else { return nil }
      } else {
        guard feedback(.imageDecoded, signposts, nil) else { return nil }
      }
      if ImageGeneratorUtils.isVideoModel(modelVersion) {
        batchSize.0 = Int(configuration.numFrames)
      }
      guard batchSize.0 > 1 else {
        return [result]
      }
      var batch = [Tensor<FloatType>]()
      let shape = result.shape
      for i in 0..<min(batchSize.0, shape[0]) {
        batch.append(result[i..<(i + 1), 0..<shape[1], 0..<shape[2], 0..<shape[3]].copied())
      }
      return batch
    }
  }

  // This is a custom inpainting, we directly go to steps - tEnc, and retain what we need till the end.
  private func generateImageWithMask1AndMask2(
    _ image: Tensor<FloatType>, scaleFactor imageScaleFactor: Int, imageNegMask1: Tensor<FloatType>,
    imageNegMask2: Tensor<FloatType>?, mask1: Tensor<FloatType>, mask2: Tensor<FloatType>?,
    depth: Tensor<FloatType>?, hints: [ControlHintType: AnyTensor], custom: Tensor<FloatType>?,
    shuffles: [(Tensor<FloatType>, Float)], poses: [(Tensor<FloatType>, Float)],
    text: String, negativeText: String, configuration: GenerationConfiguration,
    denoiserParameterization: Denoiser.Parameterization, sampling: Sampling,
    signposts: inout Set<ImageGeneratorSignpost>, cancellation: (@escaping () -> Void) -> Void,
    feedback: @escaping (ImageGeneratorSignpost, Set<ImageGeneratorSignpost>, Tensor<FloatType>?) ->
      Bool
  ) -> [Tensor<FloatType>]? {
    let coreMLGuard = modelPreloader.beginCoreMLGuard()
    defer {
      if coreMLGuard {
        modelPreloader.endCoreMLGuard()
      }
    }
    let mfaGuard = modelPreloader.beginMFAGuard()
    defer {
      if mfaGuard {
        modelPreloader.endMFAGuard()
      }
    }
    let file =
      (configuration.model.flatMap {
        ModelZoo.isModelDownloaded($0) ? $0 : nil
      }) ?? ModelZoo.defaultSpecification.file
    let modifier = ImageGeneratorUtils.modifierForModel(
      file, LoRAs: configuration.loras.compactMap(\.file))
    let modelVersion = ModelZoo.versionForModel(file)
    let (qkNorm, dualAttentionLayers, distilledGuidanceLayers) =
      ModelZoo.MMDiTForModel(file).map {
        return ($0.qkNorm, $0.dualAttentionLayers, $0.distilledGuidanceLayers ?? 0)
      } ?? (false, [], 0)
    let textEncoderVersion = ModelZoo.textEncoderVersionForModel(file)
    let modelObjective = ModelZoo.objectiveForModel(file)
    let modelUpcastAttention = ModelZoo.isUpcastAttentionForModel(file)
    var textEncoderFiles: [String] =
      [
        ModelZoo.textEncoderForModel(file).flatMap {
          ModelZoo.isModelDownloaded($0) ? $0 : nil
        } ?? "clip_vit_l14_f16.ckpt"
      ]
      + ModelZoo.CLIPEncodersForModel(file).compactMap { ModelZoo.isModelDownloaded($0) ? $0 : nil }
    textEncoderFiles +=
      ((ModelZoo.T5EncoderForModel(file).flatMap { ModelZoo.isModelDownloaded($0) ? $0 : nil }).map
      { [$0] } ?? [])
    let diffusionMappingFile = ModelZoo.diffusionMappingForModel(file).flatMap {
      ModelZoo.isModelDownloaded($0) ? $0 : nil
    }
    let fpsId = Int(configuration.fpsId)
    let motionBucketId = Int(configuration.motionBucketId)
    let condAug = configuration.condAug
    let startFrameCfg = configuration.startFrameCfg
    let clipSkip = Int(configuration.clipSkip)
    let autoencoderFile =
      ModelZoo.autoencoderForModel(file).flatMap {
        ModelZoo.isModelDownloaded($0) ? $0 : nil
      } ?? ImageGeneratorUtils.defaultAutoencoder
    let isGuidanceEmbedEnabled =
      ModelZoo.guidanceEmbedForModel(file) && configuration.speedUpWithGuidanceEmbed
    var isCfgEnabled = !ModelZoo.isConsistencyModelForModel(file) && !isGuidanceEmbedEnabled
    let latentsScaling = ModelZoo.latentsScalingForModel(file)
    let paddedTextEncodingLength = ModelZoo.paddedTextEncodingLengthForModel(file)
    let conditioning = ModelZoo.conditioningForModel(file)
    let refinerVersion: ModelVersion? = configuration.refinerModel.flatMap {
      guard $0 != file, ModelZoo.isModelDownloaded($0) else { return nil }
      let version = ModelZoo.versionForModel($0)
      guard
        version == modelVersion
          || ([.sdxlBase, .sdxlRefiner, .ssd1b].contains(version)
            && [.sdxlBase, .sdxlRefiner, .ssd1b].contains(modelVersion))
      else { return nil }
      return ModelZoo.versionForModel($0)
    }
    let tiledDecoding = TiledConfiguration(
      isEnabled: configuration.tiledDecoding,
      tileSize: .init(
        width: Int(configuration.decodingTileWidth), height: Int(configuration.decodingTileHeight)),
      tileOverlap: Int(configuration.decodingTileOverlap))
    let tiledDiffusion = TiledConfiguration(
      isEnabled: configuration.tiledDiffusion,
      tileSize: .init(
        width: Int(configuration.diffusionTileWidth), height: Int(configuration.diffusionTileHeight)
      ), tileOverlap: Int(configuration.diffusionTileOverlap))
    var alternativeDecoderFilePath: String? = nil
    var alternativeDecoderVersion: AlternativeDecoderVersion? = nil
    let lora: [LoRAConfiguration] =
      (ModelZoo.builtinLoRAForModel(file)
        ? [
          LoRAConfiguration(
            file: ModelZoo.filePathForModelDownloaded(file), weight: 1, version: modelVersion,
            isLoHa: false, modifier: .none)
        ] : [])
      + configuration.loras.compactMap {
        guard let file = $0.file else { return nil }
        let loraVersion = LoRAZoo.versionForModel(file)
        guard LoRAZoo.isModelDownloaded(file),
          modelVersion == loraVersion || refinerVersion == loraVersion
            || (modelVersion == .kandinsky21 && loraVersion == .v1)
        else { return nil }
        if LoRAZoo.isConsistencyModelForModel(file) {
          isCfgEnabled = false
        }
        if let alternativeDecoder = LoRAZoo.alternativeDecoderForModel(file) {
          alternativeDecoderFilePath = LoRAZoo.filePathForModelDownloaded(alternativeDecoder.0)
          alternativeDecoderVersion = alternativeDecoder.1
        }
        return LoRAConfiguration(
          file: LoRAZoo.filePathForModelDownloaded(file), weight: $0.weight, version: loraVersion,
          isLoHa: LoRAZoo.isLoHaForModel(file), modifier: LoRAZoo.modifierForModel(file))
      }
    if modelVersion == .v2 || modelVersion == .sdxlBase || modelVersion == .sdxlRefiner
      || modelVersion == .ssd1b || modelVersion == .svdI2v || modelVersion == .wurstchenStageC
      || modelVersion == .sd3 || modelVersion == .pixart
    {
      DynamicGraph.flags = .disableMixedMPSGEMM
    }
    if !DeviceCapability.isMemoryMapBufferSupported {
      DynamicGraph.flags.insert(.disableMmapMTLBuffer)
    }
    let isMFAEnabled = DeviceCapability.isMFAEnabled.load(ordering: .acquiring)
    if !isMFAEnabled {
      DynamicGraph.flags.insert(.disableMetalFlashAttention)
    } else {
      DynamicGraph.flags.remove(.disableMetalFlashAttention)
      if !DeviceCapability.isMFAGEMMFaster {
        DynamicGraph.flags.insert(.disableMFAGEMM)
      }
    }
    var hasHints = Set(hints.keys)
    if !poses.isEmpty {
      hasHints.insert(.pose)
    }
    let (
      canInjectControls, canInjectT2IAdapters, canInjectAttentionKVs, _, injectIPAdapterLengths,
      canInjectedControls
    ) =
      ImageGeneratorUtils.canInjectControls(
        hasImage: true, hasDepth: depth != nil, hasHints: hasHints, hasCustom: custom != nil,
        shuffleCount: shuffles.count, controls: configuration.controls,
        version: modelVersion)
    let queueWatermark = DynamicGraph.queueWatermark
    if (canInjectControls && modelVersion == .v2) && !DeviceCapability.isMaxPerformance {
      DynamicGraph.queueWatermark = 8
    }
    defer {
      if (canInjectControls && modelVersion == .v2) && !DeviceCapability.isMaxPerformance {
        DynamicGraph.queueWatermark = queueWatermark
      }
    }
    let batchSize =
      ImageGeneratorUtils.isVideoModel(modelVersion) ? 1 : Int(configuration.batchSize)
    precondition(batchSize > 0)
    let textGuidanceScale = configuration.guidanceScale
    let imageGuidanceScale = configuration.imageGuidanceScale
    let guidanceEmbed = configuration.guidanceEmbed
    let originalSize =
      configuration.originalImageWidth == 0 || configuration.originalImageHeight == 0
      ? (width: Int(configuration.startWidth) * 64, height: Int(configuration.startHeight) * 64)
      : (
        width: Int(configuration.originalImageWidth), height: Int(configuration.originalImageHeight)
      )
    let cropTopLeft = (top: Int(configuration.cropTop), left: Int(configuration.cropLeft))
    let targetSize =
      configuration.targetImageWidth == 0 || configuration.targetImageHeight == 0
      ? (width: Int(configuration.startWidth) * 64, height: Int(configuration.startHeight) * 64)
      : (width: Int(configuration.targetImageWidth), height: Int(configuration.targetImageHeight))
    let negativeOriginalSize =
      configuration.negativeOriginalImageWidth == 0
        || configuration.negativeOriginalImageHeight == 0
      ? originalSize
      : (
        width: Int(configuration.negativeOriginalImageWidth),
        height: Int(configuration.negativeOriginalImageHeight)
      )
    let aestheticScore = configuration.aestheticScore
    let negativeAestheticScore = configuration.negativeAestheticScore
    let zeroNegativePrompt = configuration.zeroNegativePrompt
    let sharpness = configuration.sharpness
    let strength = configuration.strength
    let highPrecisionForAutoencoder = ModelZoo.isHighPrecisionAutoencoderForModel(file)
    precondition(image.shape[2] % (64 * imageScaleFactor) == 0)
    precondition(image.shape[1] % (64 * imageScaleFactor) == 0)
    let startWidth: Int
    let startHeight: Int
    let channels: Int
    let firstStageFilePath: String
    if modelVersion == .wurstchenStageC {
      (startWidth, startHeight) = stageCLatentsSize(configuration)
      channels = 16
      firstStageFilePath = ModelZoo.filePathForModelDownloaded(file)
    } else {
      startWidth = image.shape[2] / 8 / imageScaleFactor
      startHeight = image.shape[1] / 8 / imageScaleFactor
      switch modelVersion {
      case .wurstchenStageC, .sd3, .sd3Large, .flux1, .hunyuanVideo, .wan21_1_3b, .wan21_14b,
        .hiDreamI1:
        channels = 16
      case .auraflow, .kandinsky21, .pixart, .sdxlBase, .sdxlRefiner, .ssd1b, .svdI2v, .v1, .v2,
        .wurstchenStageB:
        channels = 4
      }
      firstStageFilePath = ModelZoo.filePathForModelDownloaded(autoencoderFile)
    }
    let imageScale = DeviceCapability.Scale(
      widthScale: UInt16(startWidth / 8), heightScale: UInt16(startHeight / 8))
    let isHighPrecisionVAEFallbackEnabled = DeviceCapability.isHighPrecisionVAEFallbackEnabled(
      scale: imageScale)
    let isQuantizedModel = ModelZoo.isQuantizedModel(file)
    let is8BitModel = ModelZoo.is8BitModel(file)
    let canRunLoRASeparately = modelPreloader.canRunLoRASeparately
    let externalOnDemand = modelPreloader.externalOnDemand(
      version: modelVersion,
      scale: DeviceCapability.Scale(
        widthScale: configuration.startWidth, heightScale: configuration.startHeight),
      variant: .unet, injectedControls: canInjectedControls,
      is8BitModel: is8BitModel && (canRunLoRASeparately || lora.isEmpty))
    let refiner: Refiner? = configuration.refinerModel.flatMap {
      guard $0 != file, ModelZoo.isModelDownloaded($0) else { return nil }
      let version = ModelZoo.versionForModel($0)
      guard
        version == modelVersion
          || ([.sdxlBase, .sdxlRefiner, .ssd1b].contains(version)
            && [.sdxlBase, .sdxlRefiner, .ssd1b].contains(modelVersion))
      else { return nil }
      return Refiner(
        start: configuration.refinerStart, filePath: ModelZoo.filePathForModelDownloaded($0),
        externalOnDemand: externalOnDemand, version: ModelZoo.versionForModel($0),
        isQuantizedModel: ModelZoo.isQuantizedModel($0),
        isConsistencyModel: ModelZoo.isConsistencyModelForModel($0))
    }
    let controlExternalOnDemand = modelPreloader.externalOnDemand(
      version: modelVersion,
      scale: DeviceCapability.Scale(
        widthScale: configuration.startWidth, heightScale: configuration.startHeight),
      variant: .control, injectedControls: canInjectedControls, suffix: "ctrl")
    let textEncoderExternalOnDemand = modelPreloader.externalOnDemand(
      version: modelVersion,
      scale: DeviceCapability.Scale(
        widthScale: configuration.startWidth, heightScale: configuration.startHeight),
      variant: .textEncoder, injectedControls: 0)
    let vaeExternalOnDemand = modelPreloader.externalOnDemand(
      version: modelVersion,
      scale: DeviceCapability.Scale(
        widthScale: configuration.startWidth, heightScale: configuration.startHeight),
      variant: .autoencoder, injectedControls: canInjectedControls)
    let dmExternalOnDemand = modelPreloader.externalOnDemand(
      version: modelVersion,
      scale: DeviceCapability.Scale(
        widthScale: configuration.startWidth, heightScale: configuration.startHeight),
      variant: .diffusionMapping, injectedControls: 0)
    let teaCache =
      ModelZoo.teaCacheCoefficientsForModel(file).map {
        var teaCacheEnd =
          configuration.teaCacheEnd < 0
          ? Int(configuration.steps) + 1 + Int(configuration.teaCacheEnd)
          : Int(configuration.teaCacheEnd)
        let teaCacheStart = min(max(Int(configuration.teaCacheStart), 0), Int(configuration.steps))
        teaCacheEnd = min(max(max(teaCacheStart, teaCacheEnd), 0), Int(configuration.steps))
        return TeaCacheConfiguration(
          coefficients: $0,
          steps: min(teaCacheStart, teaCacheEnd)...max(teaCacheStart, teaCacheEnd),
          threshold: configuration.teaCache ? configuration.teaCacheThreshold : 0,
          maxSkipSteps: Int(configuration.teaCacheMaxSkipSteps))
      }
      ?? TeaCacheConfiguration(
        coefficients: (0, 0, 0, 0, 0), steps: 0...0, threshold: 0, maxSkipSteps: 0)
    let causalInference =
      configuration.causalInferenceEnabled ? Int(configuration.causalInference) : 0
    let sampler = LocalImageGenerator.sampler(
      from: configuration.sampler, isCfgEnabled: isCfgEnabled,
      filePath: ModelZoo.filePathForModelDownloaded(file), modifier: modifier,
      version: modelVersion, qkNorm: qkNorm, dualAttentionLayers: dualAttentionLayers,
      distilledGuidanceLayers: distilledGuidanceLayers, usesFlashAttention: isMFAEnabled,
      objective: modelObjective,
      upcastAttention: modelUpcastAttention,
      externalOnDemand: externalOnDemand, injectControls: canInjectControls,
      injectT2IAdapters: canInjectT2IAdapters, injectAttentionKV: canInjectAttentionKVs,
      injectIPAdapterLengths: injectIPAdapterLengths,
      lora: lora, isGuidanceEmbedEnabled: isGuidanceEmbedEnabled,
      isQuantizedModel: isQuantizedModel,
      canRunLoRASeparately: canRunLoRASeparately,
      stochasticSamplingGamma: configuration.stochasticSamplingGamma,
      conditioning: conditioning, parameterization: denoiserParameterization,
      tiledDiffusion: tiledDiffusion, teaCache: teaCache, causalInference: causalInference,
      weightsCache: weightsCache,
      of: FloatType.self)
    let initTimestep = sampler.timestep(for: strength, sampling: sampling)
    let graph = DynamicGraph()
    if externalOnDemand
      || externalOnDemandPartially(
        version: modelVersion, memoryCapacity: DeviceCapability.memoryCapacity,
        externalOnDemand: externalOnDemand)
    {
      TensorData.makeExternalData(for: ModelZoo.filePathForModelDownloaded(file), graph: graph)
      for stageModel in ModelZoo.stageModelsForModel(file) {
        TensorData.makeExternalData(
          for: ModelZoo.filePathForModelDownloaded(stageModel), graph: graph)
      }
    }
    if textEncoderExternalOnDemand {
      TensorData.makeExternalData(
        for: ModelZoo.filePathForModelDownloaded(textEncoderFiles[0]), graph: graph)
    }
    if controlExternalOnDemand {
      for file in Set(configuration.controls.compactMap { $0.file }) {
        TensorData.makeExternalData(for: ModelZoo.filePathForModelDownloaded(file), graph: graph)
      }
    }
    if vaeExternalOnDemand {
      TensorData.makeExternalData(
        for: ModelZoo.filePathForModelDownloaded(autoencoderFile), graph: graph)
    }
    if dmExternalOnDemand, let diffusionMappingFile = diffusionMappingFile {
      TensorData.makeExternalData(
        for: ModelZoo.filePathForModelDownloaded(diffusionMappingFile), graph: graph)
    }
    let potentials = lora.map { ($0.file as NSString).lastPathComponent }
    let (
      tokensTensors, positionTensors, embedMask, injectedEmbeddings, unconditionalAttentionWeights,
      attentionWeights, hasNonOneWeights, tokenLengthUncond, tokenLengthCond, lengthsOfUncond,
      lengthsOfCond
    ) = tokenize(
      graph: graph, modelVersion: modelVersion, textEncoderVersion: textEncoderVersion,
      paddedTextEncodingLength: paddedTextEncodingLength, text: text, negativeText: negativeText,
      negativePromptForImagePrior: configuration.negativePromptForImagePrior,
      potentials: potentials, T5TextEncoder: configuration.t5TextEncoder,
      clipL: configuration.separateClipL ? (configuration.clipLText ?? "") : nil,
      openClipG: configuration.separateOpenClipG ? (configuration.openClipGText ?? "") : nil,
      t5: configuration.separateT5 ? (configuration.t5Text ?? "") : nil
    )
    signposts.formUnion([
      .textEncoded, .imageEncoded, .sampling(sampling.steps), .imageDecoded,
    ])
    if modelVersion == .wurstchenStageC {
      let initTimestep = sampler.timestep(
        for: strength,
        sampling: Sampling(
          steps: Int(configuration.stage2Steps), shift: Double(configuration.stage2Shift)))
      signposts.insert(
        .secondPassSampling(Int(configuration.stage2Steps) - initTimestep.roundedDownStartStep))
      signposts.insert(.secondPassImageEncoded)
    }
    return graph.withNoGrad {
      let injectedTextEmbeddings = generateInjectedTextEmbeddings(
        batchSize: batchSize, startHeight: startHeight, startWidth: startWidth, image: image,
        graph: graph, hints: hints, custom: custom, shuffles: shuffles, pose: poses.first?.0,
        controls: configuration.controls,
        version: modelVersion, tiledDiffusion: tiledDiffusion, usesFlashAttention: isMFAEnabled,
        externalOnDemand: controlExternalOnDemand, cancellation: cancellation)
      let (tokenLengthUncond, tokenLengthCond) = ControlModel<FloatType>.modifyTextEmbeddings(
        tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
        injecteds: injectedTextEmbeddings)
      let tokenLength = max(tokenLengthUncond, tokenLengthCond)
      let textEncoder = TextEncoder<FloatType>(
        filePaths: textEncoderFiles.map { ModelZoo.filePathForModelDownloaded($0) },
        version: modelVersion, textEncoderVersion: textEncoderVersion,
        isCfgEnabled: isCfgEnabled,
        usesFlashAttention: isMFAEnabled && DeviceCapability.isMFACausalAttentionMaskSupported,
        injectEmbeddings: !injectedEmbeddings.isEmpty,
        externalOnDemand: textEncoderExternalOnDemand,
        deviceProperties: DeviceCapability.deviceProperties, weightsCache: weightsCache,
        maxLength: tokenLength, clipSkip: clipSkip, lora: lora)
      let image = downscaleImageAndToGPU(
        graph.variable(image), scaleFactor: imageScaleFactor)
      let textEncodings = modelPreloader.consumeTextModels(
        textEncoder.encode(
          tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
          tokens: tokensTensors, positions: positionTensors, mask: embedMask,
          injectedEmbeddings: injectedEmbeddings, image: [image], lengthsOfUncond: lengthsOfUncond,
          lengthsOfCond: lengthsOfCond, injectedTextEmbeddings: injectedTextEmbeddings,
          textModels: modelPreloader.retrieveTextModels(textEncoder: textEncoder)),
        textEncoder: textEncoder)
      var c: [DynamicGraph.Tensor<FloatType>]
      var extraProjection: DynamicGraph.Tensor<FloatType>?
      DynamicGraph.setSeed(configuration.seed)
      if modelVersion == .kandinsky21, let diffusionMappingFile = diffusionMappingFile {
        let diffusionMapping = DiffusionMapping<FloatType>(
          filePath: ModelZoo.filePathForModelDownloaded(diffusionMappingFile),
          usesFlashAttention: isMFAEnabled && DeviceCapability.isMFACausalAttentionMaskSupported,
          steps: Int(configuration.imagePriorSteps),
          negativePromptForImagePrior: configuration.negativePromptForImagePrior,
          CLIPWeight: configuration.clipWeight, externalOnDemand: dmExternalOnDemand)
        let imageEmb = diffusionMapping.sample(
          textEncoding: textEncodings[2], textEmbedding: textEncodings[3], tokens: tokensTensors[1])
        let kandinskyEmbedding = KandinskyEmbedding<FloatType>(
          filePath: ModelZoo.filePathForModelDownloaded(file))
        let (xfProj, xfOut) = kandinskyEmbedding.encode(
          textEncoding: textEncodings[0], textEmbedding: textEncodings[1], imageEmbedding: imageEmb)
        extraProjection = xfProj
        c = [xfOut]
      } else {
        extraProjection = nil
        c = textEncodings
      }
      (c, extraProjection) = repeatConditionsToMatchBatchSize(
        c: c, extraProjection: extraProjection,
        unconditionalAttentionWeights: unconditionalAttentionWeights,
        attentionWeights: attentionWeights, version: modelVersion, tokenLength: tokenLength,
        batchSize: batchSize, hasNonOneWeights: hasNonOneWeights)
      guard feedback(.textEncoded, signposts, nil) else { return nil }
      var firstStage = FirstStage<FloatType>(
        filePath: firstStageFilePath, version: modelVersion,
        latentsScaling: latentsScaling, highPrecisionKeysAndValues: highPrecisionForAutoencoder,
        highPrecisionFallback: isHighPrecisionVAEFallbackEnabled,
        tiledDecoding: tiledDecoding, tiledDiffusion: tiledDiffusion,
        externalOnDemand: vaeExternalOnDemand, alternativeUsesFlashAttention: isMFAEnabled,
        alternativeFilePath: alternativeDecoderFilePath,
        alternativeDecoderVersion: alternativeDecoderVersion,
        deviceProperties: DeviceCapability.deviceProperties)
      var firstPassImage: DynamicGraph.Tensor<FloatType>
      if modelVersion == .wurstchenStageC {
        // Try to resize the input image so we can encode with EfficientNetv2s properly.
        if image.shape[1] != startHeight * 32 || image.shape[2] != startWidth * 32 {
          firstPassImage = Upsample(
            .bilinear, widthScale: Float(startWidth * 32) / Float(image.shape[2]),
            heightScale: Float(startHeight * 32) / Float(image.shape[1]))(image)
        } else {
          firstPassImage = image
        }
      } else {
        firstPassImage = image
      }
      var batchSize = (batchSize, 0)
      switch modelVersion {
      case .svdI2v:
        batchSize = (Int(configuration.numFrames), 0)
      case .hunyuanVideo, .wan21_1_3b, .wan21_14b:
        batchSize = injectReferenceFrames(
          batchSize: ((Int(configuration.numFrames) - 1) / 4) + 1, version: modelVersion,
          canInjectControls: canInjectControls, shuffleCount: shuffles.count,
          hasCustom: custom != nil)
      case .auraflow, .flux1, .kandinsky21, .pixart, .sd3, .sd3Large, .sdxlBase, .sdxlRefiner,
        .ssd1b, .v1, .v2, .wurstchenStageB, .wurstchenStageC, .hiDreamI1:
        break
      }
      let imageSize: Int
      let firstPassImageForSample: DynamicGraph.Tensor<FloatType>?
      (imageSize, firstPassImage, firstPassImageForSample) = expandImageForEncoding(
        batchSize: batchSize, version: modelVersion, modifier: modifier, image: firstPassImage)
      var (sample, _) = modelPreloader.consumeFirstStageSample(
        firstStage.sample(
          firstPassImageForSample ?? firstPassImage,
          encoder: modelPreloader.retrieveFirstStageEncoder(
            firstStage: firstStage, scale: imageScale), cancellation: cancellation),
        firstStage: firstStage, scale: imageScale)
      var maskedImage1: DynamicGraph.Tensor<FloatType>? = nil
      var maskedImage2: DynamicGraph.Tensor<FloatType>? = nil
      if modifier == .inpainting || modifier == .editing || modifier == .double
        || modelVersion == .svdI2v
      {
        var batch = [DynamicGraph.Tensor<FloatType>]()
        if isI2v(version: modelVersion, modifier: modifier) && modifier != .editing {
          batch = [firstPassImage]
        } else {
          batch.append(firstPassImage .* graph.variable(imageNegMask1.toGPU(0)))
          if let imageNegMask2 = imageNegMask2 {
            batch.append(firstPassImage .* graph.variable(imageNegMask2.toGPU(0)))
          } else {
            let shape = firstPassImage.shape
            let batch1 = graph.variable(.GPU(0), format: .NHWC, shape: shape, of: FloatType.self)
            batch1.full(0)
            batch.append(batch1)
          }
        }
        let encodedBatch = modelPreloader.consumeFirstStageEncode(
          firstStage.encode(
            batch,
            encoder: modelPreloader.retrieveFirstStageEncoder(
              firstStage: firstStage, scale: imageScale), cancellation: cancellation),
          firstStage: firstStage, scale: imageScale
        )
        if modifier == .inpainting {
          maskedImage1 = firstStage.scale(
            encodedBatch[0][0..<imageSize, 0..<startHeight, 0..<startWidth, 0..<channels].copied())
          maskedImage2 = firstStage.scale(
            encodedBatch[encodedBatch.count - 1][
              0..<imageSize, 0..<startHeight, 0..<startWidth, 0..<channels
            ].copied())
        } else if modifier == .editing {
          if modelVersion == .v1 {
            maskedImage1 = encodedBatch[0][
              0..<imageSize, 0..<startHeight, 0..<startWidth, 0..<channels
            ].copied()
          } else {
            maskedImage1 = firstStage.scale(
              encodedBatch[0][0..<imageSize, 0..<startHeight, 0..<startWidth, 0..<channels].copied()
            )
          }
          maskedImage2 = maskedImage1
        } else if modelVersion == .svdI2v {
          maskedImage1 = encodedBatch[0]
          maskedImage2 = encodedBatch[0]
        } else {
          maskedImage1 = encodedBatch[0][
            0..<imageSize, 0..<startHeight, 0..<startWidth, 0..<channels
          ].copied()
          maskedImage2 = maskedImage1
        }
      }
      guard feedback(.imageEncoded, signposts, nil) else { return nil }
      var initMask1 = graph.variable(mask1.toGPU(0))
      if initMask1.shape[1] != startHeight || initMask1.shape[2] != startWidth {
        initMask1 = Upsample(
          .nearest, widthScale: Float(startWidth) / Float(initMask1.shape[2]),
          heightScale: Float(startHeight) / Float(initMask1.shape[1]))(initMask1)
      }
      var initNegMask = graph.variable(
        .GPU(0), .NHWC(1, startHeight, startWidth, 1), of: FloatType.self)
      initNegMask.full(1)
      initNegMask = initNegMask - initMask1
      let depthImage = depth.map {
        let depthImage = graph.variable($0.toGPU(0))
        let depthHeight = depthImage.shape[1]
        let depthWidth = depthImage.shape[2]
        guard depthHeight != startHeight * 8 || depthWidth != startWidth * 8 else {
          return depthImage
        }
        return Upsample(
          .bilinear, widthScale: Float(startHeight * 8) / Float(depthHeight),
          heightScale: Float(startWidth * 8) / Float(depthWidth))(depthImage)
      }
      var injectedControls = generateInjectedControls(
        graph: graph, batchSize: batchSize.0, startHeight: startHeight, startWidth: startWidth,
        image: image,
        depth: depthImage, hints: hints, custom: custom, shuffles: shuffles, pose: poses.last?.0,
        mask: imageNegMask1, controls: configuration.controls, version: modelVersion,
        tiledDiffusion: tiledDiffusion,
        usesFlashAttention: isMFAEnabled, externalOnDemand: controlExternalOnDemand,
        steps: sampling.steps, firstStage: firstStage, cancellation: cancellation)
      guard feedback(.controlsGenerated, signposts, nil) else { return nil }
      let redoInjectedControls = configuration.controls.contains { control in
        control.file.map {
          (ControlNetZoo.modifierForModel($0) ?? ControlHintType(from: control.inputOverride))
            == .inpaint
        } ?? false
      }
      if let image = maskedImage1 {
        maskedImage1 = injectVACEFrames(
          batchSize: batchSize, version: modelVersion, image: image,
          injectedControls: injectedControls)
      }
      if let image = maskedImage2 {
        maskedImage2 = injectVACEFrames(
          batchSize: batchSize, version: modelVersion, image: image,
          injectedControls: injectedControls)
      }
      firstPassImage = injectVACEFrames(
        batchSize: batchSize, version: modelVersion, image: firstPassImage,
        injectedControls: injectedControls)
      sample = injectVACEFrames(
        batchSize: batchSize, version: modelVersion, image: sample,
        injectedControls: injectedControls)
      let noise = randomLatentNoise(
        graph: graph, batchSize: batchSize.0, startHeight: startHeight,
        startWidth: startWidth, channels: channels, seed: configuration.seed,
        seedMode: configuration.seedMode)
      let customImage = custom.map {
        let customImage = graph.variable($0.toGPU(0))
        let customHeight = customImage.shape[1]
        let customWidth = customImage.shape[2]
        guard customHeight != startHeight * 8 || customWidth != startWidth * 8 else {
          return customImage
        }
        return Upsample(
          .bilinear, widthScale: Float(startHeight * 8) / Float(customHeight),
          heightScale: Float(startWidth * 8) / Float(customWidth))(customImage)
      }
      let imageCond = encodeImageCond(
        image: firstPassImage, depth: depthImage, custom: customImage, modifier: modifier,
        version: modelVersion, firstStage: firstStage, usesFlashAttention: isMFAEnabled)
      var initMask1Maybe: DynamicGraph.Tensor<FloatType>? = initMask1
      var initNegMaskMaybe: DynamicGraph.Tensor<FloatType>? = initNegMask
      if modifier == .inpainting {
        maskedImage1 = concatMaskWithMaskedImage(
          hasImage: true, batchSize: batchSize,
          version: modelVersion, encodedImage: maskedImage1!, encodedMask: initMask1,
          imageNegMask: graph.variable(imageNegMask1.toGPU(0)))
        if configuration.preserveOriginalAfterInpaint {
          initMask1Maybe = nil
          initNegMaskMaybe = nil
        }
      }
      let intermediateResultWithError = sampler.sample(
        noise,
        unets: modelPreloader.retrieveUNet(
          sampler: sampler, scale: imageScale, tokenLengthUncond: tokenLengthUncond,
          tokenLengthCond: tokenLengthCond), sample: sample,
        conditionImage: maskedImage1 ?? imageCond, mask: initMask1Maybe,
        negMask: initNegMaskMaybe,
        conditioning: c, tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
        extraProjection: extraProjection, injectedControls: injectedControls,
        textGuidanceScale: textGuidanceScale, imageGuidanceScale: imageGuidanceScale,
        guidanceEmbed: guidanceEmbed,
        startStep: (integral: 0, fractional: 0),
        endStep: (integral: initTimestep.roundedUpStartStep, fractional: initTimestep.startStep),
        originalSize: originalSize,
        cropTopLeft: cropTopLeft, targetSize: targetSize, aestheticScore: aestheticScore,
        negativeOriginalSize: negativeOriginalSize,
        negativeAestheticScore: negativeAestheticScore, zeroNegativePrompt: zeroNegativePrompt,
        refiner: refiner, fpsId: fpsId, motionBucketId: motionBucketId, condAug: condAug,
        startFrameCfg: startFrameCfg, sharpness: sharpness, sampling: sampling,
        cancellation: cancellation
      ) { step, tensor in
        feedback(.sampling(step), signposts, tensor)
      }
      var intermediateResult = try? intermediateResultWithError.get()
      guard let x = intermediateResult?.x else {
        // Make sure on failure case, the model can be cached.
        let _ = try? modelPreloader.consumeUNet(
          intermediateResultWithError, sampler: sampler, scale: imageScale,
          tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond)
        return nil
      }
      let initNegMask2: DynamicGraph.Tensor<FloatType>?
      let initMask2: DynamicGraph.Tensor<FloatType>?
      var x_T: DynamicGraph.Tensor<FloatType>
      let zEnc: DynamicGraph.Tensor<FloatType>
      let pureNoise: Bool
      if initTimestep.startStep > 0 {
        // Redo noise only if it is used, otherwise it is not used, no need to redo.
        noise.randn(std: 1, mean: 0)
        let sampleScaleFactor = sampler.sampleScaleFactor(
          at: initTimestep.startStep, sampling: sampling)
        let noiseScaleFactor = sampler.noiseScaleFactor(
          at: initTimestep.startStep, sampling: sampling)
        zEnc = sampleScaleFactor * sample + noiseScaleFactor * noise
        pureNoise = false
      } else {
        zEnc = noise
        pureNoise = true
      }
      if let mask2 = mask2 {
        var initMask = graph.variable(mask2.toGPU(0))
        if initMask.shape[1] != startHeight || initMask.shape[2] != startWidth {
          initMask = Upsample(
            .nearest, widthScale: Float(startWidth) / Float(initMask.shape[2]),
            heightScale: Float(startHeight) / Float(initMask.shape[1]))(initMask)
        }
        initNegMask.full(1)
        initNegMask = initNegMask - initMask
        // If it is pure noise, no need to mix two noises (x is from noise too).
        x_T = pureNoise ? zEnc : zEnc .* initNegMask + x .* initMask
        initMask2 = initMask
        initNegMask2 = initNegMask
      } else {
        // If it is pure noise, no need to mix two noises (x is from noise too).
        x_T = pureNoise ? zEnc : zEnc .* initNegMask + x .* initMask1
        initNegMask2 = nil  // This will disable the mask run during sampling.
        if modifier == .inpainting {
          initMask2 = graph.variable(
            .GPU(0), .NHWC(1, startHeight, startWidth, 1), of: FloatType.self)
          initMask2?.full(1)
        } else {
          initMask2 = nil
        }
      }
      if redoInjectedControls {
        let imageNegMask2 =
          imageNegMask2
          ?? {
            let imageNegMask2 = graph.variable(
              .GPU(0), format: imageNegMask1.format, shape: imageNegMask1.shape, of: FloatType.self)
            imageNegMask2.full(0)
            return imageNegMask2.rawValue.toCPU()
          }()
        injectedControls = generateInjectedControls(
          graph: graph, batchSize: batchSize.0, startHeight: startHeight, startWidth: startWidth,
          image: firstPassImage,
          depth: depthImage, hints: hints, custom: custom, shuffles: shuffles, pose: poses.last?.0,
          mask: imageNegMask2, controls: configuration.controls, version: modelVersion,
          tiledDiffusion: tiledDiffusion,
          usesFlashAttention: isMFAEnabled, externalOnDemand: controlExternalOnDemand,
          steps: sampling.steps, firstStage: firstStage, cancellation: cancellation
        )
      }
      var initMask2Maybe: DynamicGraph.Tensor<FloatType>? = initMask2
      var initNegMask2Maybe: DynamicGraph.Tensor<FloatType>? = initNegMask2
      if modifier == .inpainting {
        maskedImage2 = concatMaskWithMaskedImage(
          hasImage: true, batchSize: batchSize,
          version: modelVersion, encodedImage: maskedImage2!, encodedMask: initMask2!,
          imageNegMask: imageNegMask2.map { graph.variable($0.toGPU(0)) })
        if !configuration.preserveOriginalAfterInpaint {
          initMask2Maybe = nil
          initNegMask2Maybe = nil
        }
      }
      guard
        var x =
          try? modelPreloader.consumeUNet(
            sampler.sample(
              x_T, unets: intermediateResult?.unets ?? [nil], sample: sample,
              conditionImage: maskedImage2 ?? imageCond, mask: initMask2Maybe,
              negMask: initNegMask2Maybe,
              conditioning: c,
              tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
              extraProjection: extraProjection, injectedControls: injectedControls,
              textGuidanceScale: textGuidanceScale, imageGuidanceScale: imageGuidanceScale,
              guidanceEmbed: guidanceEmbed,
              startStep: (
                integral: initTimestep.roundedDownStartStep, fractional: initTimestep.startStep
              ),
              endStep: (
                integral: sampling.steps,
                fractional: Float(sampling.steps)
              ),
              originalSize: originalSize, cropTopLeft: cropTopLeft, targetSize: targetSize,
              aestheticScore: aestheticScore, negativeOriginalSize: negativeOriginalSize,
              negativeAestheticScore: negativeAestheticScore,
              zeroNegativePrompt: zeroNegativePrompt, refiner: refiner, fpsId: fpsId,
              motionBucketId: motionBucketId, condAug: condAug, startFrameCfg: startFrameCfg,
              sharpness: sharpness, sampling: sampling, cancellation: cancellation
            ) { step, tensor in
              feedback(.sampling(initTimestep.roundedDownStartStep + step), signposts, tensor)
            }, sampler: sampler, scale: imageScale, tokenLengthUncond: tokenLengthUncond,
            tokenLengthCond: tokenLengthCond)
      else {
        return nil
      }
      intermediateResult = nil
      guard feedback(.sampling(sampling.steps), signposts, nil) else { return nil }
      // If it is Wurstchen, run the stage 2.
      if modelVersion == .wurstchenStageC {
        guard feedback(.imageDecoded, signposts, nil) else { return nil }
        firstStage = FirstStage<FloatType>(
          filePath: ModelZoo.filePathForModelDownloaded(autoencoderFile), version: .wurstchenStageB,
          latentsScaling: latentsScaling, highPrecisionKeysAndValues: highPrecisionForAutoencoder,
          highPrecisionFallback: isHighPrecisionVAEFallbackEnabled,
          tiledDecoding: TiledConfiguration(
            isEnabled: false,
            tileSize: .init(width: 0, height: 0), tileOverlap: 0),
          tiledDiffusion: tiledDiffusion,
          externalOnDemand: vaeExternalOnDemand, alternativeUsesFlashAttention: isMFAEnabled,
          alternativeFilePath: alternativeDecoderFilePath,
          alternativeDecoderVersion: alternativeDecoderVersion,
          deviceProperties: DeviceCapability.deviceProperties)
        let (sample, _) = modelPreloader.consumeFirstStageSample(
          firstStage.sample(
            image,
            encoder: modelPreloader.retrieveFirstStageEncoder(
              firstStage: firstStage, scale: imageScale), cancellation: cancellation),
          firstStage: firstStage, scale: imageScale
        )
        let startHeight = Int(configuration.startHeight) * 16
        let startWidth = Int(configuration.startWidth) * 16
        let channels = 4
        guard feedback(.secondPassImageEncoded, signposts, nil) else { return nil }
        let secondPassModelVersion = ModelVersion.wurstchenStageB
        let secondPassModelFilePath = ModelZoo.filePathForModelDownloaded(
          ModelZoo.stageModelsForModel(file)[0])
        let secondPassSampler = LocalImageGenerator.sampler(
          from: configuration.sampler, isCfgEnabled: isCfgEnabled,
          filePath: secondPassModelFilePath, modifier: modifier,
          version: secondPassModelVersion, qkNorm: qkNorm, dualAttentionLayers: dualAttentionLayers,
          distilledGuidanceLayers: distilledGuidanceLayers, usesFlashAttention: isMFAEnabled,
          objective: modelObjective,
          upcastAttention: modelUpcastAttention,
          externalOnDemand: externalOnDemand,
          injectControls: canInjectControls, injectT2IAdapters: canInjectT2IAdapters,
          injectAttentionKV: canInjectAttentionKVs,
          injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
          isGuidanceEmbedEnabled: isGuidanceEmbedEnabled, isQuantizedModel: isQuantizedModel,
          canRunLoRASeparately: canRunLoRASeparately,
          stochasticSamplingGamma: configuration.stochasticSamplingGamma,
          conditioning: conditioning, parameterization: denoiserParameterization,
          tiledDiffusion: tiledDiffusion, teaCache: teaCache, causalInference: causalInference,
          weightsCache: weightsCache,
          of: FloatType.self
        )
        let noise = randomLatentNoise(
          graph: graph, batchSize: batchSize.0, startHeight: startHeight,
          startWidth: startWidth, channels: channels, seed: configuration.seed,
          seedMode: configuration.seedMode)
        c.append(x)
        var initMask1 = graph.variable(mask1.toGPU(0))
        if initMask1.shape[1] != startHeight || initMask1.shape[2] != startWidth {
          initMask1 = Upsample(
            .nearest, widthScale: Float(startWidth) / Float(initMask1.shape[2]),
            heightScale: Float(startHeight) / Float(initMask1.shape[1]))(initMask1)
        }
        var initNegMask = graph.variable(
          .GPU(0), .NHWC(1, startHeight, startWidth, 1), of: FloatType.self)
        initNegMask.full(1)
        initNegMask = initNegMask - initMask1
        let secondPassTextGuidance = configuration.stage2Cfg
        let secondPassSampling = Sampling(
          steps: Int(configuration.stage2Steps), shift: Double(configuration.stage2Shift))
        let initTimestep = sampler.timestep(for: strength, sampling: secondPassSampling)
        var intermediateResult =
          try?
          (secondPassSampler.sample(
            noise,
            unets: modelPreloader.retrieveUNet(
              sampler: secondPassSampler, scale: imageScale, tokenLengthUncond: tokenLengthUncond,
              tokenLengthCond: tokenLengthCond),
            sample: sample, conditionImage: nil, mask: initMask1,
            negMask: initNegMask, conditioning: c, tokenLengthUncond: tokenLengthUncond,
            tokenLengthCond: tokenLengthCond, extraProjection: extraProjection,
            injectedControls: [],  // TODO: Support injectedControls for this.
            textGuidanceScale: secondPassTextGuidance,
            imageGuidanceScale: imageGuidanceScale, guidanceEmbed: guidanceEmbed,
            startStep: (integral: 0, fractional: 0),
            endStep: (
              integral: initTimestep.roundedUpStartStep, fractional: initTimestep.startStep
            ),
            originalSize: originalSize, cropTopLeft: cropTopLeft,
            targetSize: targetSize, aestheticScore: aestheticScore,
            negativeOriginalSize: negativeOriginalSize,
            negativeAestheticScore: negativeAestheticScore,
            zeroNegativePrompt: zeroNegativePrompt, refiner: refiner, fpsId: fpsId,
            motionBucketId: motionBucketId, condAug: condAug, startFrameCfg: startFrameCfg,
            sharpness: sharpness, sampling: secondPassSampling, cancellation: cancellation
          ) { step, tensor in
            feedback(.secondPassSampling(step), signposts, tensor)
          }).get()
        guard let b = intermediateResult?.x else { return nil }
        let initNegMask2: DynamicGraph.Tensor<FloatType>?
        let initMask2: DynamicGraph.Tensor<FloatType>?
        var x_T: DynamicGraph.Tensor<FloatType>
        let zEnc: DynamicGraph.Tensor<FloatType>
        let pureNoise: Bool
        if initTimestep.startStep > 0 {
          // Redo noise only if it is used, otherwise it is not used, no need to redo.
          noise.randn(std: 1, mean: 0)
          let sampleScaleFactor = sampler.sampleScaleFactor(
            at: initTimestep.startStep, sampling: sampling)
          let noiseScaleFactor = sampler.noiseScaleFactor(
            at: initTimestep.startStep, sampling: sampling)
          zEnc = sampleScaleFactor * sample + noiseScaleFactor * noise
          pureNoise = false
        } else {
          zEnc = noise
          pureNoise = true
        }
        if let mask2 = mask2 {
          var initMask = graph.variable(mask2.toGPU(0))
          if initMask.shape[1] != startHeight || initMask.shape[2] != startWidth {
            initMask = Upsample(
              .nearest, widthScale: Float(startWidth) / Float(initMask.shape[2]),
              heightScale: Float(startHeight) / Float(initMask.shape[1]))(initMask)
          }
          initNegMask.full(1)
          initNegMask = initNegMask - initMask
          // If it is pure noise, no need to mix two noises (x is from noise too).
          x_T = pureNoise ? zEnc : zEnc .* initNegMask + b .* initMask
          initMask2 = initMask
          initNegMask2 = initNegMask
        } else {
          // If it is pure noise, no need to mix two noises (x is from noise too).
          x_T = pureNoise ? zEnc : zEnc .* initNegMask + b .* initMask1
          initNegMask2 = nil  // This will disable the mask run during sampling.
          if modifier == .inpainting {
            initMask2 = graph.variable(
              .GPU(0), .NHWC(1, startHeight, startWidth, 1), of: FloatType.self)
            initMask2?.full(1)
          } else {
            initMask2 = nil
          }
        }
        guard
          let b =
            try? modelPreloader.consumeUNet(
              secondPassSampler.sample(
                x_T, unets: intermediateResult?.unets ?? [nil], sample: sample,
                conditionImage: maskedImage2 ?? imageCond, mask: initMask2, negMask: initNegMask2,
                conditioning: c,
                tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
                extraProjection: extraProjection, injectedControls: injectedControls,
                textGuidanceScale: secondPassTextGuidance, imageGuidanceScale: imageGuidanceScale,
                guidanceEmbed: guidanceEmbed,
                startStep: (
                  integral: initTimestep.roundedDownStartStep, fractional: initTimestep.startStep
                ),
                endStep: (
                  integral: secondPassSampling.steps,
                  fractional: Float(secondPassSampling.steps)
                ),
                originalSize: originalSize, cropTopLeft: cropTopLeft, targetSize: targetSize,
                aestheticScore: aestheticScore, negativeOriginalSize: negativeOriginalSize,
                negativeAestheticScore: negativeAestheticScore,
                zeroNegativePrompt: zeroNegativePrompt, refiner: refiner, fpsId: fpsId,
                motionBucketId: motionBucketId, condAug: condAug, startFrameCfg: startFrameCfg,
                sharpness: sharpness, sampling: secondPassSampling, cancellation: cancellation
              ) { step, tensor in
                feedback(.sampling(initTimestep.roundedDownStartStep + step), signposts, tensor)
              }, sampler: secondPassSampler, scale: imageScale,
              tokenLengthUncond: tokenLengthUncond,
              tokenLengthCond: tokenLengthCond)
        else {
          return nil
        }
        intermediateResult = nil
        guard
          feedback(
            .secondPassSampling(secondPassSampling.steps),
            signposts, nil)
        else {
          return nil
        }
        x = b
      }
      if DeviceCapability.isLowPerformance {
        graph.garbageCollect()
      }
      if modelVersion == .v2 || modelVersion == .sdxlBase || modelVersion == .sdxlRefiner
        || modelVersion == .ssd1b
      {
        DynamicGraph.flags = []
      }
      if !DeviceCapability.isMemoryMapBufferSupported {
        DynamicGraph.flags.insert(.disableMmapMTLBuffer)
      }
      if !isMFAEnabled {
        DynamicGraph.flags.insert(.disableMetalFlashAttention)
      } else {
        DynamicGraph.flags.remove(.disableMetalFlashAttention)
        if !DeviceCapability.isMFAGEMMFaster {
          DynamicGraph.flags.insert(.disableMFAGEMM)
        }
      }
      let result = DynamicGraph.Tensor<FloatType>(
        from: modelPreloader.consumeFirstStageDecode(
          firstStage.decode(
            x, batchSize: (batchSize.0 - batchSize.1, 0),
            decoder: modelPreloader.retrieveFirstStageDecoder(
              firstStage: firstStage, scale: imageScale), cancellation: cancellation),
          firstStage: firstStage, scale: imageScale
        )
      ).rawValue.toCPU()
      guard !isNaN(result) else { return nil }
      if modelVersion == .wurstchenStageC {
        guard feedback(.secondPassImageDecoded, signposts, nil) else { return nil }
      } else {
        guard feedback(.imageDecoded, signposts, nil) else { return nil }
      }
      if ImageGeneratorUtils.isVideoModel(modelVersion) {
        batchSize.0 = Int(configuration.numFrames)
      }
      guard batchSize.0 > 1 else {
        return [result]
      }
      var batch = [Tensor<FloatType>]()
      let shape = result.shape
      for i in 0..<min(batchSize.0, shape[0]) {
        batch.append(result[i..<(i + 1), 0..<shape[1], 0..<shape[2], 0..<shape[3]].copied())
      }
      return batch
    }
  }
}
