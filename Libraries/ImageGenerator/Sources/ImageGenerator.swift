import C_ccv
import DataModels
import Dflat
import Diffusion
import DiffusionCoreML
import DiffusionPreprocessors
import FaceRestorer
import Foundation
import ModelZoo
import NNC
import Upscaler

public func xorshift(_ a: UInt32) -> UInt32 {
  var x = a == 0 ? 0xbad_5eed : a
  x ^= x &<< 13
  x ^= x &>> 17
  x ^= x &<< 5
  return x
}

public protocol PoseDrawer {
  func drawPose(_ pose: Tensor<Float>, startWidth: Int, startHeight: Int) -> Tensor<FloatType>?
}

public struct ImageGenerator {
  public static let defaultTextEncoder = "clip_vit_l14_f16.ckpt"
  public static let defaultAutoencoder = "vae_ft_mse_840000_f16.ckpt"
  public let modelPreloader: ModelPreloader
  public var tokenizerV1: TextualInversionAttentionCLIPTokenizer
  public var tokenizerV2: TextualInversionAttentionCLIPTokenizer
  public var tokenizerXL: TextualInversionAttentionCLIPTokenizer
  public var tokenizerKandinsky: SentencePieceTokenizer
  public var tokenizerT5: SentencePieceTokenizer
  public var tokenizerPileT5: SentencePieceTokenizer
  public var tokenizerChatGLM3: SentencePieceTokenizer
  let poseDrawer: PoseDrawer
  private let queue: DispatchQueue
  public init(
    queue: DispatchQueue, configurations: FetchedResult<GenerationConfiguration>,
    workspace: Workspace, tokenizerV1: TextualInversionAttentionCLIPTokenizer,
    tokenizerV2: TextualInversionAttentionCLIPTokenizer,
    tokenizerXL: TextualInversionAttentionCLIPTokenizer,
    tokenizerKandinsky: SentencePieceTokenizer,
    tokenizerT5: SentencePieceTokenizer,
    tokenizerPileT5: SentencePieceTokenizer,
    tokenizerChatGLM3: SentencePieceTokenizer,
    poseDrawer: PoseDrawer
  ) {
    self.queue = queue
    self.tokenizerV1 = tokenizerV1
    self.tokenizerV2 = tokenizerV2
    self.tokenizerXL = tokenizerXL
    self.tokenizerKandinsky = tokenizerKandinsky
    self.tokenizerT5 = tokenizerT5
    self.tokenizerPileT5 = tokenizerPileT5
    self.tokenizerChatGLM3 = tokenizerChatGLM3
    self.poseDrawer = poseDrawer
    modelPreloader = ModelPreloader(
      queue: queue, configurations: configurations, workspace: workspace)
  }
}

extension ImageGenerator {
  public enum Signpost: Equatable & Hashable {
    case textEncoded
    case imageEncoded
    case sampling(Int)
    case imageDecoded
    case secondPassImageEncoded
    case secondPassSampling(Int)
    case secondPassImageDecoded
    case faceRestored
    case imageUpscaled
  }
}

extension ImageGenerator {
  public static func sampler<FloatType: TensorNumeric & BinaryFloatingPoint>(
    from type: SamplerType, isConsistencyModel: Bool, filePath: String, modifier: SamplerModifier,
    version: ModelVersion,
    usesFlashAttention: Bool, objective: Denoiser.Objective, upcastAttention: Bool,
    externalOnDemand: Bool, injectControls: Bool, injectT2IAdapters: Bool,
    injectIPAdapterLengths: [Int],
    lora: [LoRAConfiguration], is8BitModel: Bool, canRunLoRASeparately: Bool,
    stochasticSamplingGamma: Float, conditioning: Denoiser.Conditioning,
    parameterization: Denoiser.Parameterization, tiledDiffusion: TiledConfiguration,
    of: FloatType.Type
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
      samplingTimesteps = [999, 845, 730, 587, 443, 310, 193, 116, 53, 13, 0]
      samplingSigmas = []
    case .svdI2v:
      samplingTimesteps = []
      samplingSigmas = [
        700.00, 54.5, 15.886, 7.977, 4.248, 1.789, 0.981, 0.403, 0.173, 0.034, 0.002,
      ]
    case .sd3, .pixart, .auraflow, .kandinsky21, .wurstchenStageB, .wurstchenStageC:
      samplingTimesteps = []
      samplingSigmas = []
    }
    let isCfgEnabled = !isConsistencyModel
    guard version != .wurstchenStageC && version != .wurstchenStageB else {
      switch type {
      case .dPMPP2MKarras, .DPMPP2MAYS, .dPMPP2MTrailing:
        return DPMPP2MSampler<FloatType, UNetWrapper<FloatType>, Denoiser.CosineDiscretization>(
          filePath: filePath, modifier: modifier, version: version,
          usesFlashAttention: usesFlashAttention,
          upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
          injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
          injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
          classifierFreeGuidance: isCfgEnabled, is8BitModel: is8BitModel,
          canRunLoRASeparately: canRunLoRASeparately, conditioning: conditioning,
          tiledDiffusion: tiledDiffusion,
          discretization: Denoiser.CosineDiscretization(parameterization, objective: objective))
      case .eulerA, .eulerASubstep, .eulerATrailing, .eulerAAYS:
        return EulerASampler<FloatType, UNetWrapper<FloatType>, Denoiser.CosineDiscretization>(
          filePath: filePath, modifier: modifier, version: version,
          usesFlashAttention: usesFlashAttention,
          upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
          injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
          injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
          classifierFreeGuidance: isCfgEnabled, is8BitModel: is8BitModel,
          canRunLoRASeparately: canRunLoRASeparately, conditioning: conditioning,
          tiledDiffusion: tiledDiffusion,
          discretization: Denoiser.CosineDiscretization(parameterization, objective: objective))
      case .DDIM, .dDIMTrailing:
        return DDIMSampler<FloatType, UNetWrapper<FloatType>, Denoiser.CosineDiscretization>(
          filePath: filePath, modifier: modifier, version: version,
          usesFlashAttention: usesFlashAttention,
          upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
          injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
          injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
          classifierFreeGuidance: isCfgEnabled, is8BitModel: is8BitModel,
          canRunLoRASeparately: canRunLoRASeparately, conditioning: conditioning,
          tiledDiffusion: tiledDiffusion,
          discretization: Denoiser.CosineDiscretization(parameterization, objective: objective))
      case .PLMS:
        return PLMSSampler<FloatType, UNetWrapper<FloatType>, Denoiser.CosineDiscretization>(
          filePath: filePath, modifier: modifier, version: version,
          usesFlashAttention: usesFlashAttention,
          upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
          injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
          injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
          classifierFreeGuidance: isCfgEnabled, is8BitModel: is8BitModel,
          canRunLoRASeparately: canRunLoRASeparately, conditioning: conditioning,
          tiledDiffusion: tiledDiffusion,
          discretization: Denoiser.CosineDiscretization(parameterization, objective: objective))
      case .dPMPPSDEKarras, .dPMPPSDESubstep, .dPMPPSDETrailing, .DPMPPSDEAYS:
        return DPMPPSDESampler<FloatType, UNetWrapper<FloatType>, Denoiser.CosineDiscretization>(
          filePath: filePath, modifier: modifier, version: version,
          usesFlashAttention: usesFlashAttention,
          upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
          injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
          injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
          classifierFreeGuidance: isCfgEnabled, is8BitModel: is8BitModel,
          canRunLoRASeparately: canRunLoRASeparately, conditioning: conditioning,
          tiledDiffusion: tiledDiffusion,
          discretization: Denoiser.CosineDiscretization(parameterization, objective: objective))
      case .uniPC:
        return UniPCSampler<FloatType, UNetWrapper<FloatType>, Denoiser.CosineDiscretization>(
          filePath: filePath, modifier: modifier, version: version,
          usesFlashAttention: usesFlashAttention,
          upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
          injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
          injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
          classifierFreeGuidance: isCfgEnabled, is8BitModel: is8BitModel,
          canRunLoRASeparately: canRunLoRASeparately, conditioning: conditioning,
          tiledDiffusion: tiledDiffusion,
          discretization: Denoiser.CosineDiscretization(parameterization, objective: objective))
      case .LCM:
        return LCMSampler<FloatType, UNetWrapper<FloatType>, Denoiser.CosineDiscretization>(
          filePath: filePath, modifier: modifier, version: version,
          usesFlashAttention: usesFlashAttention,
          upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
          injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
          injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
          is8BitModel: is8BitModel, canRunLoRASeparately: canRunLoRASeparately,
          conditioning: conditioning, tiledDiffusion: tiledDiffusion,
          discretization: Denoiser.CosineDiscretization(parameterization, objective: objective))
      case .TCD:
        return TCDSampler<FloatType, UNetWrapper<FloatType>, Denoiser.CosineDiscretization>(
          filePath: filePath, modifier: modifier, version: version,
          usesFlashAttention: usesFlashAttention,
          upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
          injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
          injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
          is8BitModel: is8BitModel, canRunLoRASeparately: canRunLoRASeparately,
          stochasticSamplingGamma: stochasticSamplingGamma,
          conditioning: conditioning, tiledDiffusion: tiledDiffusion,
          discretization: Denoiser.CosineDiscretization(parameterization, objective: objective))
      }
    }
    switch type {
    case .dPMPP2MKarras:
      return DPMPP2MSampler<FloatType, UNetWrapper<FloatType>, Denoiser.KarrasDiscretization>(
        filePath: filePath, modifier: modifier, version: version,
        usesFlashAttention: usesFlashAttention,
        upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
        injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
        injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
        classifierFreeGuidance: isCfgEnabled, is8BitModel: is8BitModel,
        canRunLoRASeparately: canRunLoRASeparately, conditioning: conditioning,
        tiledDiffusion: tiledDiffusion,
        discretization: Denoiser.KarrasDiscretization(parameterization, objective: objective))
    case .DPMPP2MAYS:
      if samplingTimesteps.isEmpty {
        return DPMPP2MSampler<
          FloatType, UNetWrapper<FloatType>, Denoiser.AYSLogLinearInterpolatedKarrasDiscretization
        >(
          filePath: filePath, modifier: modifier, version: version,
          usesFlashAttention: usesFlashAttention,
          upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
          injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
          injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
          classifierFreeGuidance: isCfgEnabled, is8BitModel: is8BitModel,
          canRunLoRASeparately: canRunLoRASeparately, conditioning: conditioning,
          tiledDiffusion: tiledDiffusion,
          discretization: Denoiser.AYSLogLinearInterpolatedKarrasDiscretization(
            parameterization, objective: objective, samplingSigmas: samplingSigmas))
      } else {
        return DPMPP2MSampler<
          FloatType, UNetWrapper<FloatType>, Denoiser.AYSLogLinearInterpolatedTimestepDiscretization
        >(
          filePath: filePath, modifier: modifier, version: version,
          usesFlashAttention: usesFlashAttention,
          upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
          injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
          injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
          classifierFreeGuidance: isCfgEnabled, is8BitModel: is8BitModel,
          canRunLoRASeparately: canRunLoRASeparately, conditioning: conditioning,
          tiledDiffusion: tiledDiffusion,
          discretization: Denoiser.AYSLogLinearInterpolatedTimestepDiscretization(
            parameterization, objective: objective, samplingTimesteps: samplingTimesteps))
      }
    case .dPMPP2MTrailing:
      return DPMPP2MSampler<FloatType, UNetWrapper<FloatType>, Denoiser.LinearDiscretization>(
        filePath: filePath, modifier: modifier, version: version,
        usesFlashAttention: usesFlashAttention,
        upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
        injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
        injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
        classifierFreeGuidance: isCfgEnabled, is8BitModel: is8BitModel,
        canRunLoRASeparately: canRunLoRASeparately, conditioning: conditioning,
        tiledDiffusion: tiledDiffusion,
        discretization: Denoiser.LinearDiscretization(
          parameterization, objective: objective, timestepSpacing: .trailing))
    case .eulerA:
      return EulerASampler<FloatType, UNetWrapper<FloatType>, Denoiser.LinearDiscretization>(
        filePath: filePath, modifier: modifier, version: version,
        usesFlashAttention: usesFlashAttention,
        upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
        injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
        injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
        classifierFreeGuidance: isCfgEnabled, is8BitModel: is8BitModel,
        canRunLoRASeparately: canRunLoRASeparately, conditioning: conditioning,
        tiledDiffusion: tiledDiffusion,
        discretization: Denoiser.LinearDiscretization(parameterization, objective: objective))
    case .eulerATrailing:
      return EulerASampler<FloatType, UNetWrapper<FloatType>, Denoiser.LinearDiscretization>(
        filePath: filePath, modifier: modifier, version: version,
        usesFlashAttention: usesFlashAttention,
        upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
        injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
        injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
        classifierFreeGuidance: isCfgEnabled, is8BitModel: is8BitModel,
        canRunLoRASeparately: canRunLoRASeparately, conditioning: conditioning,
        tiledDiffusion: tiledDiffusion,
        discretization: Denoiser.LinearDiscretization(
          parameterization, objective: objective, timestepSpacing: .trailing))
    case .eulerAAYS:
      if samplingTimesteps.isEmpty {
        return EulerASampler<
          FloatType, UNetWrapper<FloatType>, Denoiser.AYSLogLinearInterpolatedKarrasDiscretization
        >(
          filePath: filePath, modifier: modifier, version: version,
          usesFlashAttention: usesFlashAttention,
          upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
          injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
          injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
          classifierFreeGuidance: isCfgEnabled, is8BitModel: is8BitModel,
          canRunLoRASeparately: canRunLoRASeparately, conditioning: conditioning,
          tiledDiffusion: tiledDiffusion,
          discretization: Denoiser.AYSLogLinearInterpolatedKarrasDiscretization(
            parameterization, objective: objective, samplingSigmas: samplingSigmas))
      } else {
        return EulerASampler<
          FloatType, UNetWrapper<FloatType>, Denoiser.AYSLogLinearInterpolatedTimestepDiscretization
        >(
          filePath: filePath, modifier: modifier, version: version,
          usesFlashAttention: usesFlashAttention,
          upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
          injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
          injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
          classifierFreeGuidance: isCfgEnabled, is8BitModel: is8BitModel,
          canRunLoRASeparately: canRunLoRASeparately, conditioning: conditioning,
          tiledDiffusion: tiledDiffusion,
          discretization: Denoiser.AYSLogLinearInterpolatedTimestepDiscretization(
            parameterization, objective: objective, samplingTimesteps: samplingTimesteps))
      }
    case .DDIM:
      return DDIMSampler<FloatType, UNetWrapper<FloatType>, Denoiser.LinearDiscretization>(
        filePath: filePath, modifier: modifier, version: version,
        usesFlashAttention: usesFlashAttention,
        upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
        injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
        injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
        classifierFreeGuidance: isCfgEnabled, is8BitModel: is8BitModel,
        canRunLoRASeparately: canRunLoRASeparately, conditioning: conditioning,
        tiledDiffusion: tiledDiffusion,
        discretization: Denoiser.LinearDiscretization(
          parameterization, objective: objective, timestepSpacing: .leading))
    case .dDIMTrailing:
      return DDIMSampler<FloatType, UNetWrapper<FloatType>, Denoiser.LinearDiscretization>(
        filePath: filePath, modifier: modifier, version: version,
        usesFlashAttention: usesFlashAttention,
        upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
        injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
        injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
        classifierFreeGuidance: isCfgEnabled, is8BitModel: is8BitModel,
        canRunLoRASeparately: canRunLoRASeparately, conditioning: conditioning,
        tiledDiffusion: tiledDiffusion,
        discretization: Denoiser.LinearDiscretization(
          parameterization, objective: objective, timestepSpacing: .trailing))
    case .PLMS:
      return PLMSSampler<FloatType, UNetWrapper<FloatType>, Denoiser.LinearDiscretization>(
        filePath: filePath, modifier: modifier, version: version,
        usesFlashAttention: usesFlashAttention,
        upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
        injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
        injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
        classifierFreeGuidance: isCfgEnabled, is8BitModel: is8BitModel,
        canRunLoRASeparately: canRunLoRASeparately, conditioning: conditioning,
        tiledDiffusion: tiledDiffusion,
        discretization: Denoiser.LinearDiscretization(
          parameterization, objective: objective, timestepSpacing: .leading))
    case .dPMPPSDEKarras:
      return DPMPPSDESampler<FloatType, UNetWrapper<FloatType>, Denoiser.KarrasDiscretization>(
        filePath: filePath, modifier: modifier, version: version,
        usesFlashAttention: usesFlashAttention,
        upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
        injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
        injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
        classifierFreeGuidance: isCfgEnabled, is8BitModel: is8BitModel,
        canRunLoRASeparately: canRunLoRASeparately, conditioning: conditioning,
        tiledDiffusion: tiledDiffusion,
        discretization: Denoiser.KarrasDiscretization(parameterization, objective: objective))
    case .dPMPPSDETrailing:
      return DPMPPSDESampler<FloatType, UNetWrapper<FloatType>, Denoiser.LinearDiscretization>(
        filePath: filePath, modifier: modifier, version: version,
        usesFlashAttention: usesFlashAttention,
        upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
        injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
        injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
        classifierFreeGuidance: isCfgEnabled, is8BitModel: is8BitModel,
        canRunLoRASeparately: canRunLoRASeparately, conditioning: conditioning,
        tiledDiffusion: tiledDiffusion,
        discretization: Denoiser.LinearDiscretization(
          parameterization, objective: objective, timestepSpacing: .trailing))
    case .DPMPPSDEAYS:
      if samplingTimesteps.isEmpty {
        return DPMPPSDESampler<
          FloatType, UNetWrapper<FloatType>, Denoiser.AYSLogLinearInterpolatedKarrasDiscretization
        >(
          filePath: filePath, modifier: modifier, version: version,
          usesFlashAttention: usesFlashAttention,
          upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
          injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
          injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
          classifierFreeGuidance: isCfgEnabled, is8BitModel: is8BitModel,
          canRunLoRASeparately: canRunLoRASeparately, conditioning: conditioning,
          tiledDiffusion: tiledDiffusion,
          discretization: Denoiser.AYSLogLinearInterpolatedKarrasDiscretization(
            parameterization, objective: objective, samplingSigmas: samplingSigmas))
      } else {
        return DPMPPSDESampler<
          FloatType, UNetWrapper<FloatType>, Denoiser.AYSLogLinearInterpolatedTimestepDiscretization
        >(
          filePath: filePath, modifier: modifier, version: version,
          usesFlashAttention: usesFlashAttention,
          upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
          injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
          injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
          classifierFreeGuidance: isCfgEnabled, is8BitModel: is8BitModel,
          canRunLoRASeparately: canRunLoRASeparately, conditioning: conditioning,
          tiledDiffusion: tiledDiffusion,
          discretization: Denoiser.AYSLogLinearInterpolatedTimestepDiscretization(
            parameterization, objective: objective, samplingTimesteps: samplingTimesteps))
      }
    case .uniPC:
      return UniPCSampler<FloatType, UNetWrapper<FloatType>, Denoiser.LinearDiscretization>(
        filePath: filePath, modifier: modifier, version: version,
        usesFlashAttention: usesFlashAttention,
        upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
        injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
        injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
        classifierFreeGuidance: isCfgEnabled, is8BitModel: is8BitModel,
        canRunLoRASeparately: canRunLoRASeparately, conditioning: conditioning,
        tiledDiffusion: tiledDiffusion,
        discretization: Denoiser.LinearDiscretization(parameterization, objective: objective))
    case .LCM:
      return LCMSampler<FloatType, UNetWrapper<FloatType>, Denoiser.LinearDiscretization>(
        filePath: filePath, modifier: modifier, version: version,
        usesFlashAttention: usesFlashAttention,
        upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
        injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
        injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
        is8BitModel: is8BitModel, canRunLoRASeparately: canRunLoRASeparately,
        conditioning: conditioning, tiledDiffusion: tiledDiffusion,
        discretization: Denoiser.LinearDiscretization(parameterization, objective: objective))
    case .TCD:
      return TCDSampler<FloatType, UNetWrapper<FloatType>, Denoiser.LinearDiscretization>(
        filePath: filePath, modifier: modifier, version: version,
        usesFlashAttention: usesFlashAttention,
        upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
        injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
        injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
        is8BitModel: is8BitModel, canRunLoRASeparately: canRunLoRASeparately,
        stochasticSamplingGamma: stochasticSamplingGamma,
        conditioning: conditioning, tiledDiffusion: tiledDiffusion,
        discretization: Denoiser.LinearDiscretization(parameterization, objective: objective))
    case .eulerASubstep:
      return EulerASampler<FloatType, UNetWrapper<FloatType>, Denoiser.LinearManualDiscretization>(
        filePath: filePath, modifier: modifier, version: version,
        usesFlashAttention: usesFlashAttention,
        upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
        injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
        injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
        classifierFreeGuidance: false, is8BitModel: is8BitModel,
        canRunLoRASeparately: canRunLoRASeparately, conditioning: conditioning,
        tiledDiffusion: tiledDiffusion,
        discretization: Denoiser.LinearManualDiscretization(
          parameterization, objective: objective, manual: manualSubsteps))
    case .dPMPPSDESubstep:
      return DPMPPSDESampler<
        FloatType, UNetWrapper<FloatType>, Denoiser.LinearManualDiscretization
      >(
        filePath: filePath, modifier: modifier, version: version,
        usesFlashAttention: usesFlashAttention,
        upcastAttention: upcastAttention, externalOnDemand: externalOnDemand,
        injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
        injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
        classifierFreeGuidance: false, is8BitModel: is8BitModel,
        canRunLoRASeparately: canRunLoRASeparately, conditioning: conditioning,
        tiledDiffusion: tiledDiffusion,
        discretization: Denoiser.LinearManualDiscretization(
          parameterization, objective: objective, manual: manualSubsteps))
    }
  }
}

extension ImageGenerator {
  public static func expectedSignposts(
    _ image: Bool, mask: Bool, text: String, negativeText: String,
    configuration: GenerationConfiguration, version: ModelVersion
  ) -> Set<Signpost> {
    var signposts = Set<Signpost>([.textEncoded, .imageDecoded])
    if let faceRestoration = configuration.faceRestoration,
      EverythingZoo.isModelDownloaded(faceRestoration)
        && EverythingZoo.isModelDownloaded(EverythingZoo.parsenetForModel(faceRestoration))
    {
      signposts.insert(.faceRestored)
    }
    if let upscaler = configuration.upscaler, UpscalerZoo.isModelDownloaded(upscaler) {
      signposts.insert(.imageUpscaled)
    }
    guard image else {
      signposts.insert(.sampling(Int(configuration.steps)))
      if configuration.hiresFix || version == .wurstchenStageC {
        signposts.insert(.secondPassImageEncoded)
        if version == .wurstchenStageC {
          let tEnc = Int(configuration.stage2Steps)
          signposts.insert(.secondPassSampling(tEnc))
        } else {
          let tEnc = Int(configuration.hiresFixStrength * Float(Int(configuration.steps)))
          signposts.insert(.secondPassSampling(tEnc))
        }
        signposts.insert(.secondPassImageDecoded)
      }
      return signposts
    }
    signposts.insert(.imageEncoded)
    let tEnc = Int(configuration.strength * Float(configuration.steps))
    signposts.insert(.sampling(tEnc))
    if version == .wurstchenStageC {
      signposts.insert(.secondPassImageEncoded)
      let tEnc = Int(configuration.strength * Float(configuration.stage2Steps))
      signposts.insert(.secondPassSampling(tEnc))
      signposts.insert(.secondPassImageDecoded)
    }
    return signposts
  }
  public func startGenerating() {
    modelPreloader.startGenerating()
  }
  public func stopGenerating() {
    modelPreloader.stopGenerating()
  }
  public func generate(
    _ image: Tensor<FloatType>?, scaleFactor: Int, mask: Tensor<UInt8>?, depth: Tensor<FloatType>?,
    hints: [ControlHintType: AnyTensor], custom: Tensor<FloatType>?,
    shuffles: [(Tensor<FloatType>, Float)],
    text: String,
    negativeText: String, configuration: GenerationConfiguration,
    feedback: @escaping (Signpost, Set<Signpost>, Tensor<FloatType>?) -> Bool
  ) -> ([Tensor<FloatType>]?, Int) {
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
    let sampling = Sampling(steps: Int(configuration.steps), shift: Double(configuration.shift))
    guard let image = image else {
      return generateTextOnly(
        nil, scaleFactor: scaleFactor, depth: depth, hints: hints, custom: custom,
        shuffles: shuffles,
        text: text, negativeText: negativeText, configuration: configuration,
        denoiserParameterization: denoiserParameterization, sampling: sampling, feedback: feedback)
    }
    guard let mask = mask else {
      return generateImageOnly(
        image, scaleFactor: scaleFactor, depth: depth, hints: hints, custom: custom,
        shuffles: shuffles, text: text,
        negativeText: negativeText, configuration: configuration,
        denoiserParameterization: denoiserParameterization, sampling: sampling, feedback: feedback)
    }
    return generateImageWithMask(
      image, scaleFactor: scaleFactor, mask: mask, depth: depth, hints: hints, custom: custom,
      shuffles: shuffles,
      text: text, negativeText: negativeText, configuration: configuration,
      denoiserParameterization: denoiserParameterization, sampling: sampling, feedback: feedback)
  }
}

extension ImageGenerator {
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
    endLength: Int = 1, maxLength: Int = 77, paddingLength: Int = 77
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
    let tokenLength = max(unconditionalTokensCount, tokensCount)
    if tokenLength > paddingLength {
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
    positionTensor[0] = 0
    positionTensor[tokenLength - 1] = 76
    positionTensor[tokenLength] = 0
    positionTensor[tokenLength * 2 - 1] = 76
    // For everything else, we will go through lengths of each, and assigning accordingly.
    j = startLength
    var maxPosition = 0
    prefixLength = startLength
    for length in lengthsOfUncond {
      for i in 0..<length {
        var position =
          length <= 75 ? i + 1 : Int(((Float(i) + 0.5) * 75 / Float(length) + 0.5).rounded())
        position = min(max(position, 1), 75)
        positionTensor[j] = Int32(position)
        j += 1
      }
      maxPosition = max(length, maxPosition)
      prefixLength += length
    }
    var tokenLengthUncond = tokenLength
    // We shouldn't have anything to fill between maxPosition and tokenLength - 1 if we are longer than paddingLength.
    if prefixLength < tokenLength - 1 {
      if maxPosition + endLength + startLength > paddingLength {  // If it is paddingLength, we can go to later to find i
        tokenLengthUncond = prefixLength + 1
      }
      var position = maxPosition + 1
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
          length <= 75 ? i + 1 : Int(((Float(i) + 0.5) * 75 / Float(length) + 0.5).rounded())
        position = min(max(position, 1), 75)
        positionTensor[j] = Int32(position)
        j += 1
      }
      maxPosition = max(length, maxPosition)
      prefixLength += length
    }
    var tokenLengthCond = tokenLength
    // We shouldn't have anything to fill between maxPosition and tokenLength - 1 if we are longer than paddingLength.
    if prefixLength < tokenLength - 1 {
      if maxPosition + endLength + startLength > paddingLength {  // If it is paddingLength, we can go to later to find i
        tokenLengthCond = prefixLength + 1
      }
      var position = maxPosition + 1
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
    text: String, negativeText: String, negativePromptForImagePrior: Bool, potentials: [String],
    T5TextEncoder: Bool, clipL: String?, openClipG: String?
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
          potentials: potentials,
          startLength: 0, endLength: 0, maxLength: 0, paddingLength: 0)
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
      forcedScaleFactor: forcedScaleFactor, numberOfBlocks: numberOfBlocks)
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

  private func downscaleDepthForDepth2Img(_ depth: DynamicGraph.Tensor<FloatType>)
    -> DynamicGraph.Tensor<FloatType>
  {
    return Functional.averagePool(depth, filterSize: [8, 8], hint: Hint(stride: [8, 8]))
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
        forcedScaleFactor: forcedScaleFactor, numberOfBlocks: numberOfBlocks)
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
  }

  private func faceRestoreImages(
    _ images: [Tensor<FloatType>], configuration: GenerationConfiguration
  ) -> [Tensor<FloatType>] {
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
            original[0..<shape[0], 0..<shape[1], 0..<shape[2], (shape[3] - 3)..<shape[3]].copied(),
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

  public static func canInjectControls(
    hasImage: Bool, hasDepth: Bool, hasHints: Set<ControlHintType>, hasCustom: Bool,
    shuffleCount: Int, controls: [Control], version: ModelVersion
  ) -> (
    injectControls: Bool, injectT2IAdapters: Bool, injectIPAdapterLengths: [Int],
    injectedControls: Int
  ) {
    var injectControls = false
    var injectT2IAdapters = false
    var injectIPAdapterLengths = [Int]()
    var injectedControls = 0
    for control in controls {
      guard let file = control.file, let specification = ControlNetZoo.specificationForModel(file),
        ControlNetZoo.isModelDownloaded(specification)
      else { continue }
      guard ControlNetZoo.versionForModel(file) == version else { continue }
      guard control.weight > 0 else { continue }
      let modifier = ControlNetZoo.modifierForModel(file)
      let type = ControlNetZoo.typeForModel(file)
      let isPreprocessorDownloaded =
        ControlNetZoo.preprocessorForModel(file).map { ControlNetZoo.isModelDownloaded($0) } ?? true
      switch type {
      case .controlnet, .controlnetlora:
        switch modifier {
        case .canny, .mlsd, .tile:
          injectControls = injectControls || hasImage || hasCustom
          injectedControls += hasImage || hasCustom ? 1 : 0
        case .depth:
          let hasDepth = (hasImage && Self.isDepthModelAvailable) || hasDepth
          injectControls = injectControls || hasDepth
          injectedControls += hasDepth ? 1 : 0
        case .pose:
          injectControls = injectControls || hasHints.contains(modifier) || hasCustom
          injectedControls += hasHints.contains(modifier) || hasCustom ? 1 : 0
        case .scribble:
          injectControls =
            injectControls || hasHints.contains(modifier) || (isPreprocessorDownloaded && hasImage)
          injectedControls +=
            hasHints.contains(modifier) || (isPreprocessorDownloaded && hasImage) ? 1 : 0
        case .color:
          injectControls = injectControls || hasHints.contains(modifier) || hasImage
          injectedControls += hasHints.contains(modifier) || hasImage ? 1 : 0
        case .softedge:
          injectControls = injectControls || (isPreprocessorDownloaded && hasImage) || hasCustom
          injectedControls += (isPreprocessorDownloaded && hasImage) || hasCustom ? 1 : 0
        case .normalbae, .lineart, .seg, .custom:
          injectControls = injectControls || hasCustom
          injectedControls += hasCustom ? 1 : 0
        case .shuffle:
          injectControls = injectControls || shuffleCount > 0 || hasCustom
          injectedControls += shuffleCount > 0 || hasCustom ? 1 : 0
        case .inpaint, .ip2p:
          injectControls = injectControls || hasImage
          injectedControls += hasImage ? 1 : 0
        }
      case .ipadapterplus:
        if hasCustom || shuffleCount > 0 {
          injectIPAdapterLengths.append((shuffleCount > 0 ? shuffleCount : 1) * 16)
        }
      case .ipadapterfull:
        if hasCustom || shuffleCount > 0 {
          injectIPAdapterLengths.append((shuffleCount > 0 ? shuffleCount : 1) * 257)
        }
      case .t2iadapter:
        switch modifier {
        case .canny, .mlsd, .tile:
          injectT2IAdapters = injectT2IAdapters || hasImage || hasCustom
        case .depth:
          injectT2IAdapters =
            injectT2IAdapters || (hasImage && Self.isDepthModelAvailable) || hasDepth
        case .pose:
          injectT2IAdapters = injectT2IAdapters || hasHints.contains(modifier) || hasCustom
        case .scribble:
          injectT2IAdapters = injectT2IAdapters || hasHints.contains(modifier)
        case .color:
          injectT2IAdapters = injectT2IAdapters || hasHints.contains(modifier) || hasImage
        case .normalbae, .lineart, .softedge, .seg, .inpaint, .ip2p, .shuffle, .custom:
          break
        }
      }
    }
    return (injectControls, injectT2IAdapters, injectIPAdapterLengths, injectedControls)
  }

  public static var isDepthModelAvailable: Bool {
    EverythingZoo.isModelDownloaded("dino_v2_f16.ckpt")
      && EverythingZoo.isModelDownloaded("depth_anything_v1.0_f16.ckpt")
  }

  public static func extractDepthMap(
    _ image: DynamicGraph.Tensor<FloatType>, imageWidth: Int, imageHeight: Int,
    usesFlashAttention: Bool
  ) -> Tensor<FloatType> {
    let depthEstimator = DepthEstimator<FloatType>(
      filePaths: (
        EverythingZoo.filePathForModelDownloaded("dino_v2_f16.ckpt"),
        EverythingZoo.filePathForModelDownloaded("depth_anything_v1.0_f16.ckpt")
      ), usesFlashAttention: usesFlashAttention)
    var depthMap = depthEstimator.estimate(image)
    let shape = depthMap.shape
    if shape[1] != imageHeight || shape[2] != imageWidth {
      depthMap = Upsample(
        .bilinear, widthScale: Float(imageWidth) / Float(shape[2]),
        heightScale: Float(imageHeight) / Float(shape[1]))(depthMap)
    }
    var depthMapRawValue = depthMap.rawValue.toCPU()
    var maxVal = depthMapRawValue[0, 0, 0, 0]
    var minVal = maxVal
    depthMapRawValue.withUnsafeBytes {
      guard let f16 = $0.baseAddress?.assumingMemoryBound(to: FloatType.self) else { return }
      for i in 0..<(imageHeight * imageWidth) {
        let val = f16[i]
        maxVal = max(maxVal, val)
        minVal = min(minVal, val)
      }
    }
    if maxVal - minVal > 0 {
      let scale = (maxVal - minVal) * 0.5
      let midVal = scale + minVal
      depthMapRawValue.withUnsafeMutableBytes {
        guard let f16 = $0.baseAddress?.assumingMemoryBound(to: FloatType.self) else { return }
        for i in 0..<(imageHeight * imageWidth) {
          f16[i] = (f16[i] - midVal) / scale
        }
      }
    }
    return depthMapRawValue
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
      case .clipL14_336:
        imageSize = 336
      case .openClipH14:
        imageSize = 224
      }
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
    return rgbResults
  }

  private func shuffleRGB(
    shuffles: [(Tensor<FloatType>, Float)], graph: DynamicGraph, startHeight: Int, startWidth: Int
  ) -> [(DynamicGraph.Tensor<FloatType>, Float)] {
    var rgbResults = [(DynamicGraph.Tensor<FloatType>, Float)]()
    for (shuffle, strength) in shuffles {
      let input = graph.variable(Tensor<FloatType>(shuffle).toGPU(0))
      let inputHeight = input.shape[1]
      let inputWidth = input.shape[2]
      precondition(input.shape[3] == 3)
      if inputHeight != startHeight * 8 || inputWidth != startWidth * 8 {
        rgbResults.append(
          (
            0.5
              * (Upsample(
                .bilinear, widthScale: Float(startWidth * 8) / Float(inputWidth),
                heightScale: Float(startHeight * 8) / Float(inputHeight))(input) + 1),
            strength
          ))
      } else {
        rgbResults.append((0.5 * (input + 1), strength))
      }
    }
    return rgbResults
  }

  private func generateInjectedControls(
    graph: DynamicGraph, startHeight: Int, startWidth: Int, image: DynamicGraph.Tensor<FloatType>?,
    depth: DynamicGraph.Tensor<FloatType>?, hints: [ControlHintType: AnyTensor],
    custom: Tensor<FloatType>?, shuffles: [(Tensor<FloatType>, Float)], mask: Tensor<FloatType>?,
    controls: [Control], version: ModelVersion, tiledDiffusion: TiledConfiguration,
    usesFlashAttention: Bool, externalOnDemand: Bool, steps: Int
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
      let startStep = Int(floor(Float(steps - 1) * control.guidanceStart + 0.5))
      let endStep = Int(ceil(Float(steps - 1) * control.guidanceEnd + 0.5))
      let modifier = ControlNetZoo.modifierForModel(file)
      let type = ControlNetZoo.typeForModel(file)
      let isPreprocessorDownloaded =
        ControlNetZoo.preprocessorForModel(file).map { ControlNetZoo.isModelDownloaded($0) }
        ?? false
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
      let controlModel = ControlModel<FloatType>(
        filePaths: filePaths, type: type, modifier: modifier,
        externalOnDemand: externalOnDemand, version: version,
        tiledDiffusion: tiledDiffusion, usesFlashAttention: usesFlashAttention,
        startStep: startStep, endStep: endStep, controlMode: controlMode,
        globalAveragePooling: globalAveragePooling, transformerBlocks: transformerBlocks,
        targetBlocks: control.targetBlocks, imageEncoderVersion: imageEncoderVersion,
        ipAdapterConfig: ipAdapterConfig)
      let customRGB: (Bool) -> DynamicGraph.Tensor<FloatType>? = { convert in
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
            return 0.5 * (input + 1)
          }
          return 0.5
            * (Upsample(
              .bilinear, widthScale: Float(startWidth * 8) / Float(inputWidth),
              heightScale: Float(startHeight * 8) / Float(inputHeight))(input) + 1)
        })
      }
      switch type {
      case .controlnet, .controlnetlora:
        switch modifier {
        case .canny:
          guard let image = image else {
            guard let rgb = customRGB(true) else { return nil }
            let hints: [([DynamicGraph.Tensor<FloatType>], Float)] = controlModel.hint(inputs: [
              (hint: rgb, weight: 1)
            ]).map { ($0, control.weight) }
            return (model: controlModel, hints: hints)
          }
          let canny = graph.variable(ControlModel<FloatType>.canny(image.rawValue.toCPU()).toGPU(0))
          let hints: [([DynamicGraph.Tensor<FloatType>], Float)] = controlModel.hint(inputs: [
            (hint: canny, weight: 1)
          ]).map { ($0, control.weight) }
          return (model: controlModel, hints: hints)
        case .softedge:
          guard isPreprocessorDownloaded, let image = image else {
            guard let rgb = customRGB(true) else { return nil }
            let hints: [([DynamicGraph.Tensor<FloatType>], Float)] = controlModel.hint(inputs: [
              (hint: rgb, weight: 1)
            ]).map { ($0, control.weight) }
            return (model: controlModel, hints: hints)
          }
          var softedge = graph.variable(image.rawValue.toGPU(0))
          let shape = softedge.shape
          if let preprocessor = ControlNetZoo.preprocessorForModel(file) {
            let preprocessed = ControlModel<FloatType>.hed(
              softedge,
              modelFilePath: ControlNetZoo.filePathForModelDownloaded(preprocessor))
            var softedgeRGB = graph.variable(
              .GPU(0), .NHWC(shape[0], shape[1], shape[2], 3), of: FloatType.self)
            softedgeRGB[0..<shape[0], 0..<shape[1], 0..<shape[2], 0..<1] = preprocessed
            softedgeRGB[0..<shape[0], 0..<shape[1], 0..<shape[2], 1..<2] = preprocessed
            softedgeRGB[0..<shape[0], 0..<shape[1], 0..<shape[2], 2..<3] = preprocessed
            softedge = softedgeRGB
          } else {
            return nil
          }
          let hints: [([DynamicGraph.Tensor<FloatType>], Float)] = controlModel.hint(inputs: [
            (hint: softedge, weight: 1)
          ]).map { ($0, control.weight) }
          return (model: controlModel, hints: hints)
        case .mlsd:
          guard let image = image else {
            guard let rgb = customRGB(true) else { return nil }
            let hints: [([DynamicGraph.Tensor<FloatType>], Float)] = controlModel.hint(inputs: [
              (hint: rgb, weight: 1)
            ]).map { ($0, control.weight) }
            return (model: controlModel, hints: hints)
          }
          let imageTensor = image.rawValue.toCPU()
          guard
            let mlsdTensor = ControlModel<FloatType>.mlsd(graph.variable(imageTensor.toGPU(0)))
          else {
            return nil
          }
          let hints: [([DynamicGraph.Tensor<FloatType>], Float)] = controlModel.hint(inputs: [
            (hint: mlsdTensor.toGPU(0), weight: 1)
          ]).map { ($0, control.weight) }
          return (model: controlModel, hints: hints)
        case .depth:
          guard
            var depth = depth
              ?? (image.flatMap {
                guard Self.isDepthModelAvailable else { return nil }
                return graph.variable(
                  Self.extractDepthMap(
                    $0, imageWidth: startWidth * 8, imageHeight: startHeight * 8,
                    usesFlashAttention: usesFlashAttention
                  )
                  .toGPU(0))
              })
          else { return nil }
          // ControlNet input is always RGB at 0~1 range.
          let shape = depth.shape
          precondition(shape[3] == 1)
          depth = 0.5 * (depth + 1)
          var depthRGB = graph.variable(
            .GPU(0), .NHWC(shape[0], shape[1], shape[2], 3), of: FloatType.self)
          depthRGB[0..<shape[0], 0..<shape[1], 0..<shape[2], 0..<1] = depth
          depthRGB[0..<shape[0], 0..<shape[1], 0..<shape[2], 1..<2] = depth
          depthRGB[0..<shape[0], 0..<shape[1], 0..<shape[2], 2..<3] = depth
          let hints: [([DynamicGraph.Tensor<FloatType>], Float)] = controlModel.hint(inputs: [
            (hint: depthRGB, weight: 1)
          ]).map { ($0, control.weight) }
          return (model: controlModel, hints: hints)
        case .scribble:
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
            scribble = rgb

          } else if isPreprocessorDownloaded, let image = image {
            let rawImage = graph.variable(image.rawValue.toGPU(0))
            let shape = rawImage.shape
            if let preprocessor = ControlNetZoo.preprocessorForModel(file) {
              let preprocessed = ControlModel<FloatType>.hed(
                rawImage,
                modelFilePath: ControlNetZoo.filePathForModelDownloaded(preprocessor))
              var scribbleRGB = graph.variable(
                .GPU(0), .NHWC(shape[0], shape[1], shape[2], 3), of: FloatType.self)
              scribbleRGB[0..<shape[0], 0..<shape[1], 0..<shape[2], 0..<1] = preprocessed
              scribbleRGB[0..<shape[0], 0..<shape[1], 0..<shape[2], 1..<2] = preprocessed
              scribbleRGB[0..<shape[0], 0..<shape[1], 0..<shape[2], 2..<3] = preprocessed
              scribble = scribbleRGB
            }
          }
          guard let scribble = scribble else {
            return nil
          }
          let hints: [([DynamicGraph.Tensor<FloatType>], Float)] = controlModel.hint(inputs: [
            (hint: scribble, weight: 1)
          ]).map { ($0, control.weight) }
          return (model: controlModel, hints: hints)
        case .pose:
          guard let pose = hints[.pose].map({ Tensor<Float>($0) }) else {
            guard let rgb = customRGB(true) else { return nil }
            let hints: [([DynamicGraph.Tensor<FloatType>], Float)] = controlModel.hint(inputs: [
              (hint: rgb, weight: 1)
            ]).map { ($0, control.weight) }
            return (model: controlModel, hints: hints)
          }
          guard
            let rgb = poseDrawer.drawPose(pose, startWidth: startWidth, startHeight: startHeight)
          else { return nil }
          let hints: [([DynamicGraph.Tensor<FloatType>], Float)] = controlModel.hint(inputs: [
            (hint: graph.variable(rgb.toGPU(0)), weight: 1)
          ]).map { ($0, control.weight) }
          return (model: controlModel, hints: hints)
        case .lineart:
          guard var rgb = customRGB(false) else { return nil }
          rgb = 0.5 * (1 - rgb)
          let hints: [([DynamicGraph.Tensor<FloatType>], Float)] = controlModel.hint(inputs: [
            (hint: rgb, weight: 1)
          ]).map { ($0, control.weight) }
          return (model: controlModel, hints: hints)
        case .normalbae, .seg, .custom:
          guard let rgb = customRGB(true) else { return nil }
          let hints: [([DynamicGraph.Tensor<FloatType>], Float)] = controlModel.hint(inputs: [
            (hint: rgb, weight: 1)
          ]).map { ($0, control.weight) }
          return (model: controlModel, hints: hints)
        case .shuffle:
          guard !shuffles.isEmpty else {
            guard let rgb = customRGB(true) else { return nil }
            let hints: [([DynamicGraph.Tensor<FloatType>], Float)] = controlModel.hint(inputs: [
              (hint: rgb, weight: 1)
            ]).map { ($0, control.weight) }
            return (model: controlModel, hints: hints)
          }
          let rgbs = shuffleRGB(
            shuffles: shuffles, graph: graph, startHeight: startHeight, startWidth: startWidth)
          let hints: [([DynamicGraph.Tensor<FloatType>], Float)] = zip(
            rgbs, controlModel.hint(inputs: rgbs.map { (hint: $0.0, weight: 1) })
          ).map { ($0.1, $0.0.1 * control.weight) }
          return (model: controlModel, hints: hints)
        case .inpaint:
          guard var input = image else { return nil }
          let inputHeight = input.shape[1]
          let inputWidth = input.shape[2]
          precondition(input.shape[3] == 3)
          if inputHeight != startHeight * 8 || inputWidth != startWidth * 8 {
            input =
              0.5
              * (Upsample(
                .bilinear, widthScale: Float(startWidth * 8) / Float(inputWidth),
                heightScale: Float(startHeight * 8) / Float(inputHeight))(input) + 1)
          } else {
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
            input = input .* inputMask + (inputMask - 1)
          }
          let hints: [([DynamicGraph.Tensor<FloatType>], Float)] = controlModel.hint(inputs: [
            (hint: input, weight: 1)
          ]).map { ($0, control.weight) }
          return (model: controlModel, hints: hints)
        case .ip2p:
          guard var input = image else { return nil }
          let inputHeight = input.shape[1]
          let inputWidth = input.shape[2]
          precondition(input.shape[3] == 3)
          if inputHeight != startHeight * 8 || inputWidth != startWidth * 8 {
            input =
              0.5
              * (Upsample(
                .bilinear, widthScale: Float(startWidth * 8) / Float(inputWidth),
                heightScale: Float(startHeight * 8) / Float(inputHeight))(input) + 1)
          } else {
            input = 0.5 * (input + 1)
          }
          let hints: [([DynamicGraph.Tensor<FloatType>], Float)] = controlModel.hint(inputs: [
            (hint: input, weight: 1)
          ]).map { ($0, control.weight) }
          return (model: controlModel, hints: hints)
        case .tile:
          // Prefer custom for tile.
          if let rgb = customRGB(true) {
            let hint = controlModel.hint(inputs: [(hint: rgb, weight: 1)])[0]
            return (model: controlModel, hints: [(hint, control.weight)])
          }
          guard var input = image else { return nil }
          let inputHeight = input.shape[1]
          let inputWidth = input.shape[2]
          precondition(input.shape[3] == 3)
          if inputHeight != startHeight * 8 || inputWidth != startWidth * 8 {
            input =
              0.5
              * (Upsample(
                .bilinear, widthScale: Float(startWidth * 8) / Float(inputWidth),
                heightScale: Float(startHeight * 8) / Float(inputHeight))(input) + 1)
          } else {
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
          let hints: [([DynamicGraph.Tensor<FloatType>], Float)] = controlModel.hint(inputs: [
            (hint: input, weight: 1)
          ]).map { ($0, control.weight) }
          return (model: controlModel, hints: hints)
        case .color:
          return nil  // Not supported at the moment.
        }
      case .ipadapterplus, .ipadapterfull:
        var shuffles = shuffles
        if shuffles.isEmpty {
          guard let custom = custom else { return nil }
          shuffles = [(custom, 1)]
        }
        let rgbs = ipAdapterRGB(
          shuffles: shuffles, imageEncoderVersion: imageEncoderVersion, graph: graph)
        let hints: [([DynamicGraph.Tensor<FloatType>], Float)] = controlModel.hint(
          inputs: rgbs.map { (hint: $0.0, weight: $0.1 * control.weight) }
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
            let hint = controlModel.hint(inputs: [(hint: input, weight: control.weight)])[0]
            return (model: controlModel, hints: [(hint, 1)])
          }
          let canny = graph.variable(ControlModel<FloatType>.canny(image.rawValue.toCPU()).toGPU(0))
          let shape = canny.shape
          let input = canny[0..<shape[0], 0..<shape[1], 0..<shape[2], 0..<1].copied().reshaped(
            format: .NHWC, shape: [shape[0], startHeight, 8, startWidth, 8]
          ).permuted(0, 1, 3, 2, 4).copied().reshaped(
            .NHWC(shape[0], startHeight, startWidth, 64))
          let hints: [([DynamicGraph.Tensor<FloatType>], Float)] = controlModel.hint(inputs: [
            (hint: input, weight: control.weight)
          ]).map { ($0, 1) }
          return (model: controlModel, hints: hints)
        case .depth:
          guard var depth = depth else { return nil }
          // ControlNet input is always RGB at 0~1 range.
          let shape = depth.shape
          precondition(shape[3] == 1)
          depth = 0.5 * (depth + 1)
          var depthRGB = graph.variable(
            .GPU(0), .NHWC(shape[0], shape[1], shape[2], 3), of: FloatType.self)
          depthRGB[0..<shape[0], 0..<shape[1], 0..<shape[2], 0..<1] = depth
          depthRGB[0..<shape[0], 0..<shape[1], 0..<shape[2], 1..<2] = depth
          depthRGB[0..<shape[0], 0..<shape[1], 0..<shape[2], 2..<3] = depth
          let input = depthRGB.reshaped(
            format: .NHWC, shape: [shape[0], startHeight, 8, startWidth, 8, 3]
          ).permuted(0, 1, 3, 5, 2, 4).copied().reshaped(
            .NHWC(shape[0], startHeight, startWidth, 64 * 3))
          let hints: [([DynamicGraph.Tensor<FloatType>], Float)] = controlModel.hint(inputs: [
            (hint: input, weight: control.weight)
          ]).map { ($0, 1) }
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
          let hints: [([DynamicGraph.Tensor<FloatType>], Float)] = controlModel.hint(inputs: [
            (hint: input, weight: control.weight)
          ]).map { ($0, 1) }
          return (model: controlModel, hints: hints)
        case .pose:
          guard let pose = hints[.pose].map({ Tensor<Float>($0) }) else {
            guard let rgb = customRGB(true) else { return nil }
            let shape = rgb.shape
            let input = rgb.reshaped(
              format: .NHWC, shape: [shape[0], startHeight, 8, startWidth, 8, 3]
            ).permuted(0, 1, 3, 5, 2, 4).copied().reshaped(
              .NHWC(shape[0], startHeight, startWidth, 64 * 3))
            let hints: [([DynamicGraph.Tensor<FloatType>], Float)] = controlModel.hint(inputs: [
              (hint: input, weight: control.weight)
            ]).map { ($0, 1) }
            return (model: controlModel, hints: hints)
          }
          guard
            let rgb = poseDrawer.drawPose(pose, startWidth: startWidth, startHeight: startHeight)
          else { return nil }
          let shape = rgb.shape
          let input = graph.variable(rgb.toGPU(0)).reshaped(
            format: .NHWC, shape: [shape[0], startHeight, 8, startWidth, 8, 3]
          ).permuted(0, 1, 3, 5, 2, 4).copied().reshaped(
            .NHWC(shape[0], startHeight, startWidth, 64 * 3))
          let hints: [([DynamicGraph.Tensor<FloatType>], Float)] = controlModel.hint(inputs: [
            (hint: input, weight: control.weight)
          ]).map { ($0, 1) }
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
          rgb = 0.5 * (rgb + 1)
          let input = rgb.reshaped(
            format: .NHWC, shape: [shape[0], startHeight, 8, startWidth, 8, 3]
          ).permuted(0, 1, 3, 5, 2, 4).copied().reshaped(
            .NHWC(shape[0], startHeight, startWidth, 64 * 3))
          let hints: [([DynamicGraph.Tensor<FloatType>], Float)] = controlModel.hint(inputs: [
            (hint: input, weight: control.weight)
          ]).map { ($0, 1) }
          return (model: controlModel, hints: hints)
        case .normalbae, .lineart, .softedge, .seg, .inpaint, .ip2p, .shuffle, .mlsd, .tile,
          .custom:
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

  private func generateTextOnly(
    _ image: Tensor<FloatType>?, scaleFactor imageScaleFactor: Int,
    depth: Tensor<FloatType>?, hints: [ControlHintType: AnyTensor], custom: Tensor<FloatType>?,
    shuffles: [(Tensor<FloatType>, Float)],
    text: String, negativeText: String, configuration: GenerationConfiguration,
    denoiserParameterization: Denoiser.Parameterization, sampling: Sampling,
    feedback: @escaping (Signpost, Set<Signpost>, Tensor<FloatType>?) -> Bool
  ) -> ([Tensor<FloatType>]?, Int) {
    dispatchPrecondition(condition: .onQueue(queue))
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
    var file =
      (configuration.model.flatMap {
        ModelZoo.isModelDownloaded($0) ? $0 : nil
      }) ?? ModelZoo.defaultSpecification.file
    var modifier = ModelPreloader.modifierForModel(
      file, LoRAs: configuration.loras.compactMap(\.file))
    if modifier == .depth && depth == nil {
      // Revert to default file.
      modifier = .none
      file = ModelZoo.defaultSpecification.file
    }
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
        } ?? Self.defaultTextEncoder
      ]
      + ((ModelZoo.CLIPEncoderForModel(file).flatMap { ModelZoo.isModelDownloaded($0) ? $0 : nil })
        .map { [$0] } ?? [])
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
      } ?? Self.defaultAutoencoder
    var isConsistencyModel = ModelZoo.isConsistencyModelForModel(file)
    let latentsScaling = ModelZoo.latentsScalingForModel(file)
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
    let lora: [LoRAConfiguration] = configuration.loras.compactMap {
      guard let file = $0.file else { return nil }
      let loraVersion = LoRAZoo.versionForModel(file)
      guard LoRAZoo.isModelDownloaded(file),
        modelVersion == loraVersion || refinerVersion == loraVersion
          || (modelVersion == .kandinsky21 && loraVersion == .v1)
      else { return nil }
      if LoRAZoo.isConsistencyModelForModel(file) {
        isConsistencyModel = true
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
    var signposts = Set<Signpost>([
      .textEncoded, .sampling(sampling.steps), .imageDecoded,
    ])
    if modifier == .inpainting || modifier == .editing {
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
    let hasHints = Set(hints.keys)
    let (canInjectControls, canInjectT2IAdapters, injectIPAdapterLengths, canInjectedControls) =
      Self.canInjectControls(
        hasImage: image != nil, hasDepth: depth != nil, hasHints: hasHints,
        hasCustom: custom != nil,
        shuffleCount: shuffles.count, controls: configuration.controls, version: modelVersion)
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
    let batchSize = Int(configuration.batchSize)
    precondition(batchSize > 0)
    let textGuidanceScale = configuration.guidanceScale
    let imageGuidanceScale = configuration.imageGuidanceScale
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
        is8BitModel: ModelZoo.is8BitModel($0),
        isConsistencyModel: ModelZoo.isConsistencyModelForModel($0))
    }
    let hiresFixStrength = configuration.hiresFixStrength
    let isMemoryEfficient = DynamicGraph.memoryEfficient
    if (canInjectControls && modelVersion == .v2) && !DeviceCapability.isMaxPerformance {
      DynamicGraph.memoryEfficient = true
    }
    defer {
      if (canInjectControls && modelVersion == .v2) && !DeviceCapability.isMaxPerformance {
        DynamicGraph.memoryEfficient = isMemoryEfficient
      }
    }
    let sampler = ImageGenerator.sampler(
      from: configuration.sampler, isConsistencyModel: isConsistencyModel,
      filePath: ModelZoo.filePathForModelDownloaded(file), modifier: modifier,
      version: modelVersion, usesFlashAttention: isMFAEnabled, objective: modelObjective,
      upcastAttention: modelUpcastAttention,
      externalOnDemand: externalOnDemand, injectControls: canInjectControls,
      injectT2IAdapters: canInjectT2IAdapters, injectIPAdapterLengths: injectIPAdapterLengths,
      lora: lora, is8BitModel: is8BitModel, canRunLoRASeparately: canRunLoRASeparately,
      stochasticSamplingGamma: configuration.stochasticSamplingGamma,
      conditioning: conditioning, parameterization: denoiserParameterization,
      tiledDiffusion: tiledDiffusion,
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
      if modelVersion == .sd3 {
        firstPassChannels = 16
      } else {
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
    if externalOnDemand {
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
      graph: graph, modelVersion: modelVersion, textEncoderVersion: textEncoderVersion, text: text,
      negativeText: negativeText,
      negativePromptForImagePrior: configuration.negativePromptForImagePrior,
      potentials: potentials, T5TextEncoder: configuration.t5TextEncoder,
      clipL: configuration.separateClipL ? (configuration.clipLText ?? "") : nil,
      openClipG: configuration.separateOpenClipG ? (configuration.openClipGText ?? "") : nil
    )
    let tokenLength = max(tokenLengthUncond, tokenLengthCond)
    return graph.withNoGrad {
      let textEncoder = TextEncoder<FloatType>(
        filePaths: textEncoderFiles.map { ModelZoo.filePathForModelDownloaded($0) },
        version: modelVersion, textEncoderVersion: textEncoderVersion,
        usesFlashAttention: isMFAEnabled && DeviceCapability.isMFACausalAttentionMaskSupported,
        injectEmbeddings: !injectedEmbeddings.isEmpty,
        externalOnDemand: textEncoderExternalOnDemand, maxLength: tokenLength, clipSkip: clipSkip,
        lora: lora)
      let textEncodings = modelPreloader.consumeTextModels(
        textEncoder.encode(
          tokens: tokensTensors, positions: positionTensors, mask: embedMask,
          injectedEmbeddings: injectedEmbeddings, image: [], lengthsOfUncond: lengthsOfUncond,
          lengthsOfCond: lengthsOfCond,
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
      if hasNonOneWeights {
        // Need to scale c according to these weights. C has two parts, the unconditional and conditional.
        // We also want to do z-score type of scaling, so we need to compute mean of both parts separately.
        c = c.map { c in
          let conditionalLength = c.shape[2]
          guard tokenLength == c.shape[1] else { return c }
          var c = c
          let uc = c[0..<1, 0..<tokenLength, 0..<conditionalLength]
          let cc = c[1..<2, 0..<tokenLength, 0..<conditionalLength]
          let umean = uc.reduced(.mean, axis: [1, 2])
          let cmean = cc.reduced(.mean, axis: [1, 2])
          let uw = graph.variable(
            Tensor<FloatType>(
              from: Tensor(unconditionalAttentionWeights, .CPU, .HWC(1, tokenLength, 1))
            )
            .toGPU(0))
          let cw = graph.variable(
            Tensor<FloatType>(from: Tensor(attentionWeights, .CPU, .HWC(1, tokenLength, 1))).toGPU(
              0))
          // Keep the mean unchanged while scale it.
          c[0..<1, 0..<tokenLength, 0..<conditionalLength] = (uc - umean) .* uw + umean
          c[1..<2, 0..<tokenLength, 0..<conditionalLength] = (cc - cmean) .* cw + cmean
          return c
        }
      }
      if batchSize > 1 {
        c = c.map { c in
          var c = c
          let oldC = c
          let shape = c.shape
          if shape.count == 3 {
            c = graph.variable(
              .GPU(0), .HWC(batchSize * 2, shape[1], shape[2]), of: FloatType.self)
            for i in 0..<batchSize {
              c[i..<(i + 1), 0..<shape[1], 0..<shape[2]] =
                oldC[0..<1, 0..<shape[1], 0..<shape[2]]
              c[(batchSize + i)..<(batchSize + i + 1), 0..<shape[1], 0..<shape[2]] =
                oldC[1..<2, 0..<shape[1], 0..<shape[2]]
            }
          } else if shape.count == 2 {
            c = graph.variable(
              .GPU(0), .WC(batchSize * 2, shape[1]), of: FloatType.self)
            for i in 0..<batchSize {
              c[i..<(i + 1), 0..<shape[1]] = oldC[0..<1, 0..<shape[1]]
              c[(batchSize + i)..<(batchSize + i + 1), 0..<shape[1]] = oldC[1..<2, 0..<shape[1]]
            }
          }
          return c
        }
        if let oldProj = extraProjection {
          let shape = oldProj.shape
          var xfProj = graph.variable(
            .GPU(0), .HWC(batchSize * 2, shape[1], shape[2]), of: FloatType.self)
          for i in 0..<batchSize {
            xfProj[i..<(i + 1), 0..<shape[1], 0..<shape[2]] =
              oldProj[0..<1, 0..<shape[1], 0..<shape[2]]
            xfProj[(batchSize + i)..<(batchSize + i + 1), 0..<shape[1], 0..<shape[2]] =
              oldProj[1..<2, 0..<shape[1], 0..<shape[2]]
          }
          extraProjection = xfProj
        }
      }
      guard feedback(.textEncoded, signposts, nil) else { return (nil, 1) }
      var maskedImage: DynamicGraph.Tensor<FloatType>? = nil
      var mask: DynamicGraph.Tensor<FloatType>? = nil
      let image = image.map {
        downscaleImageAndToGPU(graph.variable($0), scaleFactor: imageScaleFactor)
      }
      var firstPassImage: DynamicGraph.Tensor<FloatType>? = nil
      if modifier == .inpainting || modifier == .editing || canInjectControls
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
      if modifier == .inpainting || modifier == .editing {
        // TODO: This needs to be properly handled for Wurstchen (i.e. using EfficientNet to encode image).
        let firstStage = FirstStage<FloatType>(
          filePath: ModelZoo.filePathForModelDownloaded(autoencoderFile), version: modelVersion,
          latentsScaling: latentsScaling, highPrecision: highPrecisionForAutoencoder,
          highPrecisionFallback: isHighPrecisionVAEFallbackEnabled,
          tiledDecoding: tiledDecoding, tiledDiffusion: tiledDiffusion,
          externalOnDemand: vaeExternalOnDemand, alternativeUsesFlashAttention: isMFAEnabled,
          alternativeFilePath: alternativeDecoderFilePath,
          alternativeDecoderVersion: alternativeDecoderVersion)
        // Only apply the image fill logic (for image encoding purpose) when it is inpainting or editing.
        let firstPassImage =
          firstPassImage
          ?? {
            let image = graph.variable(
              .GPU(0), .NHWC(1, firstPassStartHeight * 8, firstPassStartWidth * 8, 3),
              of: FloatType.self)
            image.full(0)
            return image
          }()
        let encodedImage = modelPreloader.consumeFirstStageEncode(
          firstStage.encode(
            firstPassImage,
            encoder: modelPreloader.retrieveFirstStageEncoder(
              firstStage: firstStage, scale: firstPassScale)), firstStage: firstStage,
          scale: firstPassScale)
        if modifier == .inpainting {
          maskedImage = firstStage.scale(
            encodedImage[0..<1, 0..<firstPassStartHeight, 0..<firstPassStartWidth, 0..<4].copied())
          mask = graph.variable(
            .GPU(0), .NHWC(1, firstPassStartHeight, firstPassStartWidth, 1), of: FloatType.self)
          mask?.full(1)
        } else {
          maskedImage = encodedImage[
            0..<1, 0..<firstPassStartHeight, 0..<firstPassStartWidth, 0..<4
          ].copied()
        }
        guard feedback(.imageEncoded, signposts, nil) else { return (nil, 1) }
      }
      let x_T = randomLatentNoise(
        graph: graph, batchSize: batchSize, startHeight: firstPassStartHeight,
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
      let firstPassDepth2Img = firstPassDepthImage.map { downscaleDepthForDepth2Img($0) }
      let injectedControls = generateInjectedControls(
        graph: graph, startHeight: firstPassStartHeight, startWidth: firstPassStartWidth,
        image: firstPassImage, depth: firstPassDepthImage, hints: hints, custom: custom,
        shuffles: shuffles, mask: nil, controls: configuration.controls, version: modelVersion,
        tiledDiffusion: tiledDiffusion, usesFlashAttention: isMFAEnabled,
        externalOnDemand: controlExternalOnDemand, steps: sampling.steps)
      guard
        let x =
          try? modelPreloader.consumeUNet(
            (sampler.sample(
              x_T,
              unets: modelPreloader.retrieveUNet(
                sampler: sampler, scale: firstPassScale, tokenLengthUncond: tokenLengthUncond,
                tokenLengthCond: tokenLengthCond), sample: nil,
              maskedImage: maskedImage, depthImage: firstPassDepth2Img,
              mask: mask, negMask: nil, conditioning: c, tokenLengthUncond: tokenLengthUncond,
              tokenLengthCond: tokenLengthCond, extraProjection: extraProjection,
              injectedControls: injectedControls, textGuidanceScale: textGuidanceScale,
              imageGuidanceScale: imageGuidanceScale, startStep: (integral: 0, fractional: 0),
              endStep: (
                integral: sampling.steps,
                fractional: Float(sampling.steps)
              ), originalSize: originalSize,
              cropTopLeft: cropTopLeft, targetSize: targetSize, aestheticScore: aestheticScore,
              negativeOriginalSize: negativeOriginalSize,
              negativeAestheticScore: negativeAestheticScore,
              zeroNegativePrompt: zeroNegativePrompt, refiner: refiner, fpsId: fpsId,
              motionBucketId: motionBucketId, condAug: condAug, startFrameCfg: startFrameCfg,
              sharpness: sharpness, sampling: sampling
            ) { step, tensor in
              feedback(.sampling(step), signposts, tensor)
            }).get(), sampler: sampler, scale: firstPassScale, tokenLengthUncond: tokenLengthUncond,
            tokenLengthCond: tokenLengthCond)
      else {
        return (nil, 1)
      }
      guard feedback(.sampling(sampling.steps), signposts, nil) else {
        return (nil, 1)
      }
      let isHighPrecisionVAEFallbackEnabled = DeviceCapability.isHighPrecisionVAEFallbackEnabled(
        scale: imageScale)
      let firstStage = FirstStage<FloatType>(
        filePath: ModelZoo.filePathForModelDownloaded(autoencoderFile), version: modelVersion,
        latentsScaling: latentsScaling, highPrecision: highPrecisionForAutoencoder,
        highPrecisionFallback: isHighPrecisionVAEFallbackEnabled,
        tiledDecoding: tiledDecoding, tiledDiffusion: tiledDiffusion,
        externalOnDemand: vaeExternalOnDemand, alternativeUsesFlashAttention: isMFAEnabled,
        alternativeFilePath: alternativeDecoderFilePath,
        alternativeDecoderVersion: alternativeDecoderVersion)
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
            x,
            decoder: modelPreloader.retrieveFirstStageDecoder(
              firstStage: firstStage, scale: firstPassScale)), firstStage: firstStage,
          scale: firstPassScale)
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
        guard batchSize > 1 else {
          return ([result], scaleFactor)
        }
        var batch = [Tensor<FloatType>]()
        let shape = result.shape
        for i in 0..<batchSize {
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
          DynamicGraph.Tensor<FloatType>(from: firstStageImage), encoder: nil)
      }
      if modifier == .inpainting || modifier == .editing {
        // TODO: Support this properly for Wurstchen models.
        let image =
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
        let encodedImage = modelPreloader.consumeFirstStageEncode(
          firstStage.encode(
            image,
            encoder: modelPreloader.retrieveFirstStageEncoder(
              firstStage: firstStage, scale: imageScale)), firstStage: firstStage, scale: imageScale
        )
        if modifier == .inpainting {
          maskedImage = firstStage.scale(
            encodedImage[0..<1, 0..<startHeight, 0..<startWidth, 0..<4].copied())
          mask = graph.variable(.GPU(0), .NHWC(1, startHeight, startWidth, 1), of: FloatType.self)
          mask?.full(1)
        } else {
          maskedImage = encodedImage[0..<1, 0..<startHeight, 0..<startWidth, 0..<4].copied()
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
      let secondPassDepth2Img = secondPassDepthImage.map { downscaleDepthForDepth2Img($0) }
      let (canInjectControls, canInjectT2IAdapters, injectIPAdapterLengths, canInjectedControls) =
        Self.canInjectControls(
          hasImage: true, hasDepth: secondPassDepthImage != nil, hasHints: hasHints,
          hasCustom: custom != nil, shuffleCount: shuffles.count, controls: configuration.controls,
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
        graph: graph, startHeight: startHeight, startWidth: startWidth,
        image: image ?? firstStageImage, depth: secondPassDepthImage, hints: hints, custom: custom,
        shuffles: shuffles, mask: nil, controls: configuration.controls, version: modelVersion,
        tiledDiffusion: tiledDiffusion, usesFlashAttention: isMFAEnabled,
        externalOnDemand: secondPassControlExternalOnDemand, steps: sampling.steps)
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
      let secondPassSampler = ImageGenerator.sampler(
        from: configuration.sampler, isConsistencyModel: isConsistencyModel,
        filePath: secondPassModelFilePath, modifier: modifier,
        version: secondPassModelVersion, usesFlashAttention: isMFAEnabled,
        objective: modelObjective,
        upcastAttention: modelUpcastAttention,
        externalOnDemand: externalOnDemand,
        injectControls: canInjectControls, injectT2IAdapters: canInjectT2IAdapters,
        injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
        is8BitModel: is8BitModel, canRunLoRASeparately: canRunLoRASeparately,
        stochasticSamplingGamma: configuration.stochasticSamplingGamma,
        conditioning: conditioning, parameterization: denoiserParameterization,
        tiledDiffusion: tiledDiffusion, of: FloatType.self)
      let startStep: (integral: Int, fractional: Float)
      let xEnc: DynamicGraph.Tensor<FloatType>
      let secondPassTextGuidance: Float
      let secondPassSampling: Sampling
      if modelVersion == .wurstchenStageC {
        startStep = (integral: 0, fractional: 0)
        xEnc = randomLatentNoise(
          graph: graph, batchSize: batchSize, startHeight: startHeight,
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
        if modelVersion == .sd3 {
          channels = 16
        } else {
          channels = 4
        }
        let noise = graph.variable(
          .GPU(0), .NHWC(batchSize, startHeight, startWidth, channels), of: FloatType.self)
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
            (secondPassSampler.sample(
              xEnc,
              unets: modelPreloader.retrieveUNet(
                sampler: secondPassSampler, scale: imageScale, tokenLengthUncond: tokenLengthUncond,
                tokenLengthCond: tokenLengthCond),
              sample: nil, maskedImage: maskedImage, depthImage: secondPassDepth2Img, mask: mask,
              negMask: nil, conditioning: c, tokenLengthUncond: tokenLengthUncond,
              tokenLengthCond: tokenLengthCond, extraProjection: extraProjection,
              injectedControls: secondPassInjectedControls,
              textGuidanceScale: secondPassTextGuidance,
              imageGuidanceScale: imageGuidanceScale,
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
              sharpness: sharpness, sampling: secondPassSampling
            ) { step, tensor in
              feedback(.secondPassSampling(step), signposts, tensor)
            }).get(), sampler: secondPassSampler, scale: imageScale,
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
          x,
          decoder: modelPreloader.retrieveFirstStageDecoder(
            firstStage: firstStage, scale: imageScale)), firstStage: firstStage, scale: imageScale)
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
      guard batchSize > 1 else {
        return ([result], scaleFactor)
      }
      var batch = [Tensor<FloatType>]()
      let shape = result.shape
      for i in 0..<batchSize {
        batch.append(result[i..<(i + 1), 0..<shape[1], 0..<shape[2], 0..<shape[3]].copied())
      }
      return (batch, scaleFactor)
    }
  }

  // This generate image variations with text as modifier and strength.
  private func generateImageOnly(
    _ image: Tensor<FloatType>, scaleFactor imageScaleFactor: Int, depth: Tensor<FloatType>?,
    hints: [ControlHintType: AnyTensor], custom: Tensor<FloatType>?,
    shuffles: [(Tensor<FloatType>, Float)],
    text: String,
    negativeText: String, configuration: GenerationConfiguration,
    denoiserParameterization: Denoiser.Parameterization, sampling: Sampling,
    feedback: @escaping (Signpost, Set<Signpost>, Tensor<FloatType>?) -> Bool
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
    var file =
      (configuration.model.flatMap {
        ModelZoo.isModelDownloaded($0) ? $0 : nil
      }) ?? ModelZoo.defaultSpecification.file
    var modifier = ModelPreloader.modifierForModel(
      file, LoRAs: configuration.loras.compactMap(\.file))
    if modifier == .depth && depth == nil {
      // Revert to default file.
      modifier = .none
      file = ModelZoo.defaultSpecification.file
    }
    let modelVersion = ModelZoo.versionForModel(file)
    let textEncoderVersion = ModelZoo.textEncoderVersionForModel(file)
    let modelObjective = ModelZoo.objectiveForModel(file)
    let modelUpcastAttention = ModelZoo.isUpcastAttentionForModel(file)
    var textEncoderFiles: [String] =
      [
        ModelZoo.textEncoderForModel(file).flatMap {
          ModelZoo.isModelDownloaded($0) ? $0 : nil
        } ?? "clip_vit_l14_f16.ckpt"
      ]
      + ((ModelZoo.CLIPEncoderForModel(file).flatMap { ModelZoo.isModelDownloaded($0) ? $0 : nil })
        .map { [$0] } ?? [])
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
      } ?? "vae_ft_mse_840000_f16.ckpt"
    var isConsistencyModel = ModelZoo.isConsistencyModelForModel(file)
    let latentsScaling = ModelZoo.latentsScalingForModel(file)
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
    let lora: [LoRAConfiguration] = configuration.loras.compactMap {
      guard let file = $0.file else { return nil }
      let loraVersion = LoRAZoo.versionForModel(file)
      guard LoRAZoo.isModelDownloaded(file),
        modelVersion == loraVersion || refinerVersion == loraVersion
          || (modelVersion == .kandinsky21 && loraVersion == .v1)
      else { return nil }
      if LoRAZoo.isConsistencyModelForModel(file) {
        isConsistencyModel = true
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
    let (canInjectControls, canInjectT2IAdapters, injectIPAdapterLengths, canInjectedControls) =
      Self.canInjectControls(
        hasImage: true, hasDepth: depth != nil, hasHints: Set(hints.keys), hasCustom: custom != nil,
        shuffleCount: shuffles.count, controls: configuration.controls, version: modelVersion)
    let isMemoryEfficient = DynamicGraph.memoryEfficient
    if (canInjectControls && modelVersion == .v2) && !DeviceCapability.isMaxPerformance {
      DynamicGraph.memoryEfficient = true
    }
    defer {
      if (canInjectControls && modelVersion == .v2) && !DeviceCapability.isMaxPerformance {
        DynamicGraph.memoryEfficient = isMemoryEfficient
      }
    }
    let textGuidanceScale = configuration.guidanceScale
    let imageGuidanceScale = configuration.imageGuidanceScale
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
        is8BitModel: ModelZoo.is8BitModel($0),
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
    let sampler = ImageGenerator.sampler(
      from: configuration.sampler, isConsistencyModel: isConsistencyModel,
      filePath: ModelZoo.filePathForModelDownloaded(file), modifier: modifier,
      version: modelVersion, usesFlashAttention: isMFAEnabled, objective: modelObjective,
      upcastAttention: modelUpcastAttention,
      externalOnDemand: externalOnDemand, injectControls: canInjectControls,
      injectT2IAdapters: canInjectT2IAdapters, injectIPAdapterLengths: injectIPAdapterLengths,
      lora: lora, is8BitModel: is8BitModel, canRunLoRASeparately: canRunLoRASeparately,
      stochasticSamplingGamma: configuration.stochasticSamplingGamma,
      conditioning: conditioning, parameterization: denoiserParameterization,
      tiledDiffusion: tiledDiffusion, of: FloatType.self)
    let initTimestep = sampler.timestep(for: strength, sampling: sampling)
    guard initTimestep.startStep > 0 || modelVersion == .svdI2v else {
      return generateTextOnly(
        image, scaleFactor: imageScaleFactor,
        depth: depth, hints: hints, custom: custom, shuffles: shuffles,
        text: text, negativeText: negativeText, configuration: configuration,
        denoiserParameterization: denoiserParameterization, sampling: sampling, feedback: feedback)
    }
    let batchSize = Int(configuration.batchSize)
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
      if modelVersion == .sd3 {
        channels = 16
      } else {
        channels = 4
      }
      firstStageFilePath = ModelZoo.filePathForModelDownloaded(autoencoderFile)
    }
    let imageScale = DeviceCapability.Scale(
      widthScale: UInt16(startWidth / 8), heightScale: UInt16(startHeight / 8))
    let isHighPrecisionVAEFallbackEnabled = DeviceCapability.isHighPrecisionVAEFallbackEnabled(
      scale: imageScale)
    let graph = DynamicGraph()
    if externalOnDemand {
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
      graph: graph, modelVersion: modelVersion, textEncoderVersion: textEncoderVersion, text: text,
      negativeText: negativeText,
      negativePromptForImagePrior: configuration.negativePromptForImagePrior,
      potentials: potentials, T5TextEncoder: configuration.t5TextEncoder,
      clipL: configuration.separateClipL ? (configuration.clipLText ?? "") : nil,
      openClipG: configuration.separateOpenClipG ? (configuration.openClipGText ?? "") : nil
    )
    let tokenLength = max(tokenLengthUncond, tokenLengthCond)
    var signposts = Set<Signpost>([
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
      let textEncoder = TextEncoder<FloatType>(
        filePaths: textEncoderFiles.map { ModelZoo.filePathForModelDownloaded($0) },
        version: modelVersion, textEncoderVersion: textEncoderVersion,
        usesFlashAttention: isMFAEnabled && DeviceCapability.isMFACausalAttentionMaskSupported,
        injectEmbeddings: !injectedEmbeddings.isEmpty,
        externalOnDemand: textEncoderExternalOnDemand, maxLength: tokenLength, clipSkip: clipSkip,
        lora: lora)
      let image = downscaleImageAndToGPU(
        graph.variable(image), scaleFactor: imageScaleFactor)
      let textEncodings = modelPreloader.consumeTextModels(
        textEncoder.encode(
          tokens: tokensTensors, positions: positionTensors, mask: embedMask,
          injectedEmbeddings: injectedEmbeddings, image: [image], lengthsOfUncond: lengthsOfUncond,
          lengthsOfCond: lengthsOfCond,
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
      if hasNonOneWeights {
        // Need to scale c according to these weights. C has two parts, the unconditional and conditional.
        // We also want to do z-score type of scaling, so we need to compute mean of both parts separately.
        c = c.map { c in
          let conditionalLength = c.shape[2]
          guard tokenLength == c.shape[1] else { return c }
          var c = c
          let uc = c[0..<1, 0..<tokenLength, 0..<conditionalLength]
          let cc = c[1..<2, 0..<tokenLength, 0..<conditionalLength]
          let umean = uc.reduced(.mean, axis: [1, 2])
          let cmean = cc.reduced(.mean, axis: [1, 2])
          let uw = graph.variable(
            Tensor<FloatType>(
              from: Tensor(unconditionalAttentionWeights, .CPU, .HWC(1, tokenLength, 1))
            )
            .toGPU(0))
          let cw = graph.variable(
            Tensor<FloatType>(from: Tensor(attentionWeights, .CPU, .HWC(1, tokenLength, 1))).toGPU(
              0))
          // Keep the mean unchanged while scale it.
          c[0..<1, 0..<tokenLength, 0..<conditionalLength] = (uc - umean) .* uw + umean
          c[1..<2, 0..<tokenLength, 0..<conditionalLength] = (cc - cmean) .* cw + cmean
          return c
        }
      }
      if batchSize > 1 {
        c = c.map { c in
          var c = c
          let oldC = c
          let shape = c.shape
          if shape.count == 3 {
            c = graph.variable(
              .GPU(0), .HWC(batchSize * 2, shape[1], shape[2]), of: FloatType.self)
            for i in 0..<batchSize {
              c[i..<(i + 1), 0..<shape[1], 0..<shape[2]] =
                oldC[0..<1, 0..<shape[1], 0..<shape[2]]
              c[(batchSize + i)..<(batchSize + i + 1), 0..<shape[1], 0..<shape[2]] =
                oldC[1..<2, 0..<shape[1], 0..<shape[2]]
            }
          } else if shape.count == 2 {
            c = graph.variable(
              .GPU(0), .WC(batchSize * 2, shape[1]), of: FloatType.self)
            for i in 0..<batchSize {
              c[i..<(i + 1), 0..<shape[1]] = oldC[0..<1, 0..<shape[1]]
              c[(batchSize + i)..<(batchSize + i + 1), 0..<shape[1]] = oldC[1..<2, 0..<shape[1]]
            }
          }
          return c
        }
        if let oldProj = extraProjection {
          let shape = oldProj.shape
          var xfProj = graph.variable(
            .GPU(0), .HWC(batchSize * 2, shape[1], shape[2]), of: FloatType.self)
          for i in 0..<batchSize {
            xfProj[i..<(i + 1), 0..<shape[1], 0..<shape[2]] =
              oldProj[0..<1, 0..<shape[1], 0..<shape[2]]
            xfProj[(batchSize + i)..<(batchSize + i + 1), 0..<shape[1], 0..<shape[2]] =
              oldProj[1..<2, 0..<shape[1], 0..<shape[2]]
          }
          extraProjection = xfProj
        }
      }
      guard feedback(.textEncoded, signposts, nil) else { return (nil, 1) }
      var firstStage = FirstStage<FloatType>(
        filePath: firstStageFilePath, version: modelVersion,
        latentsScaling: latentsScaling, highPrecision: highPrecisionForAutoencoder,
        highPrecisionFallback: isHighPrecisionVAEFallbackEnabled,
        tiledDecoding: tiledDecoding, tiledDiffusion: tiledDiffusion,
        externalOnDemand: vaeExternalOnDemand, alternativeUsesFlashAttention: isMFAEnabled,
        alternativeFilePath: alternativeDecoderFilePath,
        alternativeDecoderVersion: alternativeDecoderVersion)
      // Check if strength is 0.
      guard initTimestep.roundedDownStartStep < sampling.steps && configuration.strength > 0 else {
        let image = faceRestoreImage(image, configuration: configuration)
        // Otherwise, just run upscaler if needed.
        let (result, scaleFactor) = upscaleImageAndToCPU(image, configuration: configuration)
        // Because we just run the upscaler, there is no more than 1 image generation, return directly.
        return ([result], scaleFactor)
      }
      let firstPassImage: DynamicGraph.Tensor<FloatType>
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
      let (sample, encodedImage) = modelPreloader.consumeFirstStageSample(
        firstStage.sample(
          firstPassImage,
          encoder: modelPreloader.retrieveFirstStageEncoder(
            firstStage: firstStage, scale: imageScale)), firstStage: firstStage, scale: imageScale)
      var maskedImage: DynamicGraph.Tensor<FloatType>? = nil
      var mask: DynamicGraph.Tensor<FloatType>? = nil
      if modifier == .inpainting {
        maskedImage = firstStage.scale(
          encodedImage[0..<1, 0..<startHeight, 0..<startWidth, 0..<channels].copied())
        mask = graph.variable(.GPU(0), .NHWC(1, startHeight, startWidth, 1), of: FloatType.self)
        mask?.full(1)
      } else if modifier == .editing || modelVersion == .svdI2v {
        maskedImage = encodedImage[0..<1, 0..<startHeight, 0..<startWidth, 0..<channels].copied()
      }
      guard feedback(.imageEncoded, signposts, nil) else { return (nil, 1) }
      var batchSize = batchSize
      if modelVersion == .svdI2v {
        batchSize = Int(configuration.numFrames)
      }
      let noise = randomLatentNoise(
        graph: graph, batchSize: batchSize, startHeight: startHeight,
        startWidth: startWidth, channels: channels, seed: configuration.seed,
        seedMode: configuration.seedMode)
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
      let depth2Img = depthImage.map { downscaleDepthForDepth2Img($0) }
      let injectedControls = generateInjectedControls(
        graph: graph, startHeight: startHeight, startWidth: startWidth, image: image,
        depth: depthImage, hints: hints, custom: custom, shuffles: shuffles, mask: nil,
        controls: configuration.controls, version: modelVersion, tiledDiffusion: tiledDiffusion,
        usesFlashAttention: isMFAEnabled, externalOnDemand: controlExternalOnDemand,
        steps: sampling.steps)
      guard
        var x =
          try? modelPreloader.consumeUNet(
            (sampler.sample(
              x_T,
              unets: modelPreloader.retrieveUNet(
                sampler: sampler, scale: imageScale, tokenLengthUncond: tokenLengthUncond,
                tokenLengthCond: tokenLengthCond), sample: nil,
              maskedImage: maskedImage, depthImage: depth2Img,
              mask: mask, negMask: nil, conditioning: c, tokenLengthUncond: tokenLengthUncond,
              tokenLengthCond: tokenLengthCond, extraProjection: extraProjection,
              injectedControls: injectedControls, textGuidanceScale: textGuidanceScale,
              imageGuidanceScale: imageGuidanceScale,
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
              sharpness: sharpness, sampling: sampling
            ) { step, tensor in
              feedback(.sampling(step), signposts, tensor)
            }).get(), sampler: sampler, scale: imageScale, tokenLengthUncond: tokenLengthUncond,
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
          latentsScaling: latentsScaling, highPrecision: highPrecisionForAutoencoder,
          highPrecisionFallback: isHighPrecisionVAEFallbackEnabled,
          tiledDecoding: TiledConfiguration(
            isEnabled: false, tileSize: .init(width: 0, height: 0), tileOverlap: 0),
          tiledDiffusion: tiledDiffusion,
          externalOnDemand: vaeExternalOnDemand, alternativeUsesFlashAttention: isMFAEnabled,
          alternativeFilePath: alternativeDecoderFilePath,
          alternativeDecoderVersion: alternativeDecoderVersion)
        let (sample, encodedImage) = modelPreloader.consumeFirstStageSample(
          firstStage.sample(
            image,
            encoder: modelPreloader.retrieveFirstStageEncoder(
              firstStage: firstStage, scale: imageScale)), firstStage: firstStage, scale: imageScale
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
        } else if modifier == .editing || modelVersion == .svdI2v {
          maskedImage = encodedImage[0..<1, 0..<startHeight, 0..<startWidth, 0..<channels].copied()
        }
        guard feedback(.secondPassImageEncoded, signposts, nil) else { return (nil, 1) }
        let secondPassModelVersion = ModelVersion.wurstchenStageB
        let secondPassModelFilePath = ModelZoo.filePathForModelDownloaded(
          ModelZoo.stageModelsForModel(file)[0])
        let secondPassSampler = ImageGenerator.sampler(
          from: configuration.sampler, isConsistencyModel: isConsistencyModel,
          filePath: secondPassModelFilePath, modifier: modifier,
          version: secondPassModelVersion, usesFlashAttention: isMFAEnabled,
          objective: modelObjective,
          upcastAttention: modelUpcastAttention,
          externalOnDemand: externalOnDemand,
          injectControls: canInjectControls, injectT2IAdapters: canInjectT2IAdapters,
          injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
          is8BitModel: is8BitModel, canRunLoRASeparately: canRunLoRASeparately,
          stochasticSamplingGamma: configuration.stochasticSamplingGamma,
          conditioning: conditioning, parameterization: denoiserParameterization,
          tiledDiffusion: tiledDiffusion, of: FloatType.self
        )
        let noise = randomLatentNoise(
          graph: graph, batchSize: batchSize, startHeight: startHeight,
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
              (secondPassSampler.sample(
                x_T,
                unets: modelPreloader.retrieveUNet(
                  sampler: secondPassSampler, scale: imageScale,
                  tokenLengthUncond: tokenLengthUncond,
                  tokenLengthCond: tokenLengthCond),
                sample: nil, maskedImage: maskedImage, depthImage: nil, mask: mask,
                negMask: nil, conditioning: c, tokenLengthUncond: tokenLengthUncond,
                tokenLengthCond: tokenLengthCond, extraProjection: extraProjection,
                injectedControls: [],  // TODO: Support injectedControls for this.
                textGuidanceScale: secondPassTextGuidance,
                imageGuidanceScale: imageGuidanceScale,
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
                sharpness: sharpness, sampling: secondPassSampling
              ) { step, tensor in
                feedback(.secondPassSampling(step), signposts, tensor)
              }).get(), sampler: secondPassSampler, scale: imageScale,
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
          x,
          decoder: modelPreloader.retrieveFirstStageDecoder(
            firstStage: firstStage, scale: imageScale)), firstStage: firstStage, scale: imageScale)
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
      guard batchSize > 1 else {
        return ([result], scaleFactor)
      }
      var batch = [Tensor<FloatType>]()
      let shape = result.shape
      for i in 0..<batchSize {
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
    shuffles: [(Tensor<FloatType>, Float)],
    text: String,
    negativeText: String, configuration: GenerationConfiguration,
    denoiserParameterization: Denoiser.Parameterization, sampling: Sampling,
    feedback: @escaping (Signpost, Set<Signpost>, Tensor<FloatType>?) -> Bool
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
        shuffles: shuffles, text: text,
        negativeText: negativeText, configuration: configuration,
        denoiserParameterization: denoiserParameterization, sampling: sampling, feedback: feedback)
    }
    // If no sophisticated mask, nothing to be done.
    if exists0 && !exists1 && !exists2 && !exists3 {
      return generateImageOnly(
        image, scaleFactor: scaleFactor, depth: depth, hints: hints, custom: custom,
        shuffles: shuffles, text: text,
        negativeText: negativeText, configuration: configuration,
        denoiserParameterization: denoiserParameterization, sampling: sampling, feedback: feedback)
    } else if !exists0 && exists1 && !exists2 && !exists3 {
      // If masked due to nothing only the whole page, run text generation only.
      return generateTextOnly(
        image, scaleFactor: scaleFactor, depth: depth, hints: hints, custom: custom,
        shuffles: shuffles, text: text,
        negativeText: negativeText, configuration: configuration,
        denoiserParameterization: denoiserParameterization, sampling: sampling, feedback: feedback)
    }
    var signposts = Set<Signpost>()
    if let faceRestoration = configuration.faceRestoration,
      EverythingZoo.isModelDownloaded(faceRestoration)
        && EverythingZoo.isModelDownloaded(EverythingZoo.parsenetForModel(faceRestoration))
    {
      signposts.insert(.faceRestored)
    }
    if let upscaler = configuration.upscaler, UpscalerZoo.isModelDownloaded(upscaler) {
      signposts.insert(.imageUpscaled)
    }
    // If only 3, meaning we are going to retain everything, just return.
    if !exists0 && !exists1 && !exists2 && exists3 {
      let images = faceRestoreImages([image], configuration: configuration)
      let (result, scaleFactor) = upscaleImages(images, configuration: configuration)
      return (result, scaleFactor)
    }
    // Either we missing 1s, or we are in transparent mode (where 1 is ignored), go into here.
    if !exists1 || alternativeDecoderVersion == .transparent {
      // mask3 is not a typo. if no 1 exists, we only have mask3 relevant here if exists1 is missing.
      guard
        var result = generateImageWithMask2(
          image, scaleFactor: scaleFactor, imageNegMask2: imageNegMask3, mask2: mask3, depth: depth,
          hints: hints, custom: custom, shuffles: shuffles, text: text, negativeText: negativeText,
          configuration: configuration, denoiserParameterization: denoiserParameterization,
          sampling: sampling, signposts: &signposts, feedback: feedback)
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
            shuffles: shuffles, text: text,
            negativeText: negativeText, configuration: configuration,
            denoiserParameterization: denoiserParameterization, sampling: sampling,
            signposts: &signposts, feedback: feedback)
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
        text: text,
        negativeText: negativeText, configuration: configuration,
        denoiserParameterization: denoiserParameterization, sampling: sampling,
        signposts: &signposts, feedback: feedback)
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
    custom: Tensor<FloatType>?, shuffles: [(Tensor<FloatType>, Float)], text: String,
    negativeText: String,
    configuration: GenerationConfiguration, denoiserParameterization: Denoiser.Parameterization,
    sampling: Sampling, signposts: inout Set<Signpost>,
    feedback: @escaping (Signpost, Set<Signpost>, Tensor<FloatType>?) -> Bool
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
    var file =
      (configuration.model.flatMap {
        ModelZoo.isModelDownloaded($0) ? $0 : nil
      }) ?? ModelZoo.defaultSpecification.file
    var modifier = ModelPreloader.modifierForModel(
      file, LoRAs: configuration.loras.compactMap(\.file))
    if modifier == .depth && depth == nil {
      // Revert to default file.
      modifier = .none
      file = ModelZoo.defaultSpecification.file
    }
    let modelVersion = ModelZoo.versionForModel(file)
    let textEncoderVersion = ModelZoo.textEncoderVersionForModel(file)
    let modelObjective = ModelZoo.objectiveForModel(file)
    let modelUpcastAttention = ModelZoo.isUpcastAttentionForModel(file)
    var textEncoderFiles: [String] =
      [
        ModelZoo.textEncoderForModel(file).flatMap {
          ModelZoo.isModelDownloaded($0) ? $0 : nil
        } ?? "clip_vit_l14_f16.ckpt"
      ]
      + ((ModelZoo.CLIPEncoderForModel(file).flatMap { ModelZoo.isModelDownloaded($0) ? $0 : nil })
        .map { [$0] } ?? [])
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
      } ?? "vae_ft_mse_840000_f16.ckpt"
    var isConsistencyModel = ModelZoo.isConsistencyModelForModel(file)
    let latentsScaling = ModelZoo.latentsScalingForModel(file)
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
    let lora: [LoRAConfiguration] = configuration.loras.compactMap {
      guard let file = $0.file else { return nil }
      let loraVersion = LoRAZoo.versionForModel(file)
      guard LoRAZoo.isModelDownloaded(file),
        modelVersion == loraVersion || refinerVersion == loraVersion
          || (modelVersion == .kandinsky21 && loraVersion == .v1)
      else { return nil }
      if LoRAZoo.isConsistencyModelForModel(file) {
        isConsistencyModel = true
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
    let (canInjectControls, canInjectT2IAdapters, injectIPAdapterLengths, canInjectedControls) =
      Self.canInjectControls(
        hasImage: true, hasDepth: depth != nil, hasHints: Set(hints.keys), hasCustom: custom != nil,
        shuffleCount: shuffles.count, controls: configuration.controls, version: modelVersion)
    let isMemoryEfficient = DynamicGraph.memoryEfficient
    if (canInjectControls && modelVersion == .v2) && !DeviceCapability.isMaxPerformance {
      DynamicGraph.memoryEfficient = true
    }
    defer {
      if (canInjectControls && modelVersion == .v2) && !DeviceCapability.isMaxPerformance {
        DynamicGraph.memoryEfficient = isMemoryEfficient
      }
    }
    let batchSize = Int(configuration.batchSize)
    precondition(batchSize > 0)
    let textGuidanceScale = configuration.guidanceScale
    let imageGuidanceScale = configuration.imageGuidanceScale
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
      if modelVersion == .sd3 {
        channels = 16
      } else {
        channels = 4
      }
      firstStageFilePath = ModelZoo.filePathForModelDownloaded(autoencoderFile)
    }
    let imageScale = DeviceCapability.Scale(
      widthScale: UInt16(startWidth / 8), heightScale: UInt16(startHeight / 8))
    let isHighPrecisionVAEFallbackEnabled = DeviceCapability.isHighPrecisionVAEFallbackEnabled(
      scale: imageScale)
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
        is8BitModel: ModelZoo.is8BitModel($0),
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
    let sampler = ImageGenerator.sampler(
      from: configuration.sampler, isConsistencyModel: isConsistencyModel,
      filePath: ModelZoo.filePathForModelDownloaded(file), modifier: modifier,
      version: modelVersion, usesFlashAttention: isMFAEnabled, objective: modelObjective,
      upcastAttention: modelUpcastAttention,
      externalOnDemand: externalOnDemand, injectControls: canInjectControls,
      injectT2IAdapters: canInjectT2IAdapters, injectIPAdapterLengths: injectIPAdapterLengths,
      lora: lora, is8BitModel: is8BitModel, canRunLoRASeparately: canRunLoRASeparately,
      stochasticSamplingGamma: configuration.stochasticSamplingGamma,
      conditioning: conditioning, parameterization: denoiserParameterization,
      tiledDiffusion: tiledDiffusion, of: FloatType.self)
    let initTimestep = sampler.timestep(for: strength, sampling: sampling)
    let graph = DynamicGraph()
    if externalOnDemand {
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
      graph: graph, modelVersion: modelVersion, textEncoderVersion: textEncoderVersion, text: text,
      negativeText: negativeText,
      negativePromptForImagePrior: configuration.negativePromptForImagePrior,
      potentials: potentials, T5TextEncoder: configuration.t5TextEncoder,
      clipL: configuration.separateClipL ? (configuration.clipLText ?? "") : nil,
      openClipG: configuration.separateOpenClipG ? (configuration.openClipGText ?? "") : nil
    )
    let tokenLength = max(tokenLengthUncond, tokenLengthCond)
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
      let textEncoder = TextEncoder<FloatType>(
        filePaths: textEncoderFiles.map { ModelZoo.filePathForModelDownloaded($0) },
        version: modelVersion, textEncoderVersion: textEncoderVersion,
        usesFlashAttention: isMFAEnabled && DeviceCapability.isMFACausalAttentionMaskSupported,
        injectEmbeddings: !injectedEmbeddings.isEmpty,
        externalOnDemand: textEncoderExternalOnDemand, maxLength: tokenLength, clipSkip: clipSkip,
        lora: lora)
      var image = downscaleImageAndToGPU(
        graph.variable(image), scaleFactor: imageScaleFactor)
      let textEncodings = modelPreloader.consumeTextModels(
        textEncoder.encode(
          tokens: tokensTensors, positions: positionTensors, mask: embedMask,
          injectedEmbeddings: injectedEmbeddings, image: [image], lengthsOfUncond: lengthsOfUncond,
          lengthsOfCond: lengthsOfCond,
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
      if hasNonOneWeights {
        // Need to scale c according to these weights. C has two parts, the unconditional and conditional.
        // We also want to do z-score type of scaling, so we need to compute mean of both parts separately.
        c = c.map { c in
          let conditionalLength = c.shape[2]
          guard tokenLength == c.shape[1] else { return c }
          var c = c
          let uc = c[0..<1, 0..<tokenLength, 0..<conditionalLength]
          let cc = c[1..<2, 0..<tokenLength, 0..<conditionalLength]
          let umean = uc.reduced(.mean, axis: [1, 2])
          let cmean = cc.reduced(.mean, axis: [1, 2])
          let uw = graph.variable(
            Tensor<FloatType>(
              from: Tensor(unconditionalAttentionWeights, .CPU, .HWC(1, tokenLength, 1))
            )
            .toGPU(0))
          let cw = graph.variable(
            Tensor<FloatType>(from: Tensor(attentionWeights, .CPU, .HWC(1, tokenLength, 1))).toGPU(
              0))
          // Keep the mean unchanged while scale it.
          c[0..<1, 0..<tokenLength, 0..<conditionalLength] = (uc - umean) .* uw + umean
          c[1..<2, 0..<tokenLength, 0..<conditionalLength] = (cc - cmean) .* cw + cmean
          return c
        }
      }
      if batchSize > 1 {
        c = c.map { c in
          var c = c
          let oldC = c
          let shape = c.shape
          if shape.count == 3 {
            c = graph.variable(
              .GPU(0), .HWC(batchSize * 2, shape[1], shape[2]), of: FloatType.self)
            for i in 0..<batchSize {
              c[i..<(i + 1), 0..<shape[1], 0..<shape[2]] =
                oldC[0..<1, 0..<shape[1], 0..<shape[2]]
              c[(batchSize + i)..<(batchSize + i + 1), 0..<shape[1], 0..<shape[2]] =
                oldC[1..<2, 0..<shape[1], 0..<shape[2]]
            }
          } else if shape.count == 2 {
            c = graph.variable(
              .GPU(0), .WC(batchSize * 2, shape[1]), of: FloatType.self)
            for i in 0..<batchSize {
              c[i..<(i + 1), 0..<shape[1]] = oldC[0..<1, 0..<shape[1]]
              c[(batchSize + i)..<(batchSize + i + 1), 0..<shape[1]] = oldC[1..<2, 0..<shape[1]]
            }
          }
          return c
        }
        if let oldProj = extraProjection {
          let shape = oldProj.shape
          var xfProj = graph.variable(
            .GPU(0), .HWC(batchSize * 2, shape[1], shape[2]), of: FloatType.self)
          for i in 0..<batchSize {
            xfProj[i..<(i + 1), 0..<shape[1], 0..<shape[2]] =
              oldProj[0..<1, 0..<shape[1], 0..<shape[2]]
            xfProj[(batchSize + i)..<(batchSize + i + 1), 0..<shape[1], 0..<shape[2]] =
              oldProj[1..<2, 0..<shape[1], 0..<shape[2]]
          }
          extraProjection = xfProj
        }
      }
      guard feedback(.textEncoded, signposts, nil) else { return nil }
      var firstStage = FirstStage<FloatType>(
        filePath: firstStageFilePath, version: modelVersion,
        latentsScaling: latentsScaling, highPrecision: highPrecisionForAutoencoder,
        highPrecisionFallback: isHighPrecisionVAEFallbackEnabled,
        tiledDecoding: tiledDecoding, tiledDiffusion: tiledDiffusion,
        externalOnDemand: vaeExternalOnDemand, alternativeUsesFlashAttention: isMFAEnabled,
        alternativeFilePath: alternativeDecoderFilePath,
        alternativeDecoderVersion: alternativeDecoderVersion)
      let firstPassImage: DynamicGraph.Tensor<FloatType>
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
      let (sample, _) = modelPreloader.consumeFirstStageSample(
        firstStage.sample(
          firstPassImage,
          encoder: modelPreloader.retrieveFirstStageEncoder(
            firstStage: firstStage, scale: imageScale)), firstStage: firstStage, scale: imageScale)
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
        graph: graph, startHeight: startHeight, startWidth: startWidth, image: image,
        depth: depthImage, hints: hints, custom: custom, shuffles: shuffles, mask: imageNegMask2,
        controls: configuration.controls, version: modelVersion, tiledDiffusion: tiledDiffusion,
        usesFlashAttention: isMFAEnabled, externalOnDemand: controlExternalOnDemand,
        steps: sampling.steps)
      var maskedImage: DynamicGraph.Tensor<FloatType>? = nil
      if modifier == .inpainting || modifier == .editing || modelVersion == .svdI2v {
        if modelVersion != .svdI2v {
          image = image .* graph.variable(imageNegMask2.toGPU(0))
        }
        let encodedImage = modelPreloader.consumeFirstStageEncode(
          firstStage.encode(
            image,
            encoder: modelPreloader.retrieveFirstStageEncoder(
              firstStage: firstStage, scale: imageScale)), firstStage: firstStage, scale: imageScale
        )
        if modifier == .inpainting {
          maskedImage = firstStage.scale(
            encodedImage[0..<1, 0..<startHeight, 0..<startWidth, 0..<channels].copied())
        } else {
          maskedImage = encodedImage[0..<1, 0..<startHeight, 0..<startWidth, 0..<channels].copied()
        }
      }
      guard feedback(.imageEncoded, signposts, nil) else { return nil }
      var batchSize = batchSize
      if modelVersion == .svdI2v {
        batchSize = Int(configuration.numFrames)
      }
      let noise = randomLatentNoise(
        graph: graph, batchSize: batchSize, startHeight: startHeight,
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
      let depth2Img = depthImage.map { downscaleDepthForDepth2Img($0) }
      guard
        var x =
          try? modelPreloader.consumeUNet(
            (sampler.sample(
              x_T,
              unets: modelPreloader.retrieveUNet(
                sampler: sampler, scale: imageScale, tokenLengthUncond: tokenLengthUncond,
                tokenLengthCond: tokenLengthCond), sample: sample,
              maskedImage: maskedImage, depthImage: depth2Img, mask: initMask, negMask: initNegMask,
              conditioning: c, tokenLengthUncond: tokenLengthUncond,
              tokenLengthCond: tokenLengthCond, extraProjection: extraProjection,
              injectedControls: injectedControls, textGuidanceScale: textGuidanceScale,
              imageGuidanceScale: imageGuidanceScale,
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
              sharpness: sharpness, sampling: sampling
            ) { step, tensor in
              feedback(.sampling(step), signposts, tensor)
            }).get(), sampler: sampler, scale: imageScale, tokenLengthUncond: tokenLengthUncond,
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
          latentsScaling: latentsScaling, highPrecision: highPrecisionForAutoencoder,
          highPrecisionFallback: isHighPrecisionVAEFallbackEnabled,
          tiledDecoding: TiledConfiguration(
            isEnabled: false,
            tileSize: .init(width: 0, height: 0), tileOverlap: 0),
          tiledDiffusion: tiledDiffusion,
          externalOnDemand: vaeExternalOnDemand, alternativeUsesFlashAttention: isMFAEnabled,
          alternativeFilePath: alternativeDecoderFilePath,
          alternativeDecoderVersion: alternativeDecoderVersion)
        let (sample, _) = modelPreloader.consumeFirstStageSample(
          firstStage.sample(
            image,
            encoder: modelPreloader.retrieveFirstStageEncoder(
              firstStage: firstStage, scale: imageScale)), firstStage: firstStage, scale: imageScale
        )
        let startHeight = Int(configuration.startHeight) * 16
        let startWidth = Int(configuration.startWidth) * 16
        let channels = 4
        guard feedback(.secondPassImageEncoded, signposts, nil) else { return nil }
        let secondPassModelVersion = ModelVersion.wurstchenStageB
        let secondPassModelFilePath = ModelZoo.filePathForModelDownloaded(
          ModelZoo.stageModelsForModel(file)[0])
        let secondPassSampler = ImageGenerator.sampler(
          from: configuration.sampler, isConsistencyModel: isConsistencyModel,
          filePath: secondPassModelFilePath, modifier: modifier,
          version: secondPassModelVersion, usesFlashAttention: isMFAEnabled,
          objective: modelObjective,
          upcastAttention: modelUpcastAttention,
          externalOnDemand: externalOnDemand,
          injectControls: canInjectControls, injectT2IAdapters: canInjectT2IAdapters,
          injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
          is8BitModel: is8BitModel, canRunLoRASeparately: canRunLoRASeparately,
          stochasticSamplingGamma: configuration.stochasticSamplingGamma,
          conditioning: conditioning, parameterization: denoiserParameterization,
          tiledDiffusion: tiledDiffusion, of: FloatType.self
        )
        let noise = randomLatentNoise(
          graph: graph, batchSize: batchSize, startHeight: startHeight,
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
              (secondPassSampler.sample(
                x_T,
                unets: modelPreloader.retrieveUNet(
                  sampler: secondPassSampler, scale: imageScale,
                  tokenLengthUncond: tokenLengthUncond,
                  tokenLengthCond: tokenLengthCond),
                sample: sample, maskedImage: nil, depthImage: nil, mask: initMask,
                negMask: initNegMask, conditioning: c, tokenLengthUncond: tokenLengthUncond,
                tokenLengthCond: tokenLengthCond, extraProjection: extraProjection,
                injectedControls: [],  // TODO: Support injectedControls for this.
                textGuidanceScale: secondPassTextGuidance,
                imageGuidanceScale: imageGuidanceScale,
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
                sharpness: sharpness, sampling: secondPassSampling
              ) { step, tensor in
                feedback(.secondPassSampling(step), signposts, tensor)
              }).get(), sampler: secondPassSampler, scale: imageScale,
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
            x,
            decoder: modelPreloader.retrieveFirstStageDecoder(
              firstStage: firstStage, scale: imageScale)), firstStage: firstStage, scale: imageScale
        )
      )
      .rawValue
      .toCPU()
      guard !isNaN(result) else { return nil }
      if modelVersion == .wurstchenStageC {
        guard feedback(.secondPassImageDecoded, signposts, nil) else { return nil }
      } else {
        guard feedback(.imageDecoded, signposts, nil) else { return nil }
      }
      guard batchSize > 1 else {
        return [result]
      }
      var batch = [Tensor<FloatType>]()
      let shape = result.shape
      for i in 0..<batchSize {
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
    shuffles: [(Tensor<FloatType>, Float)],
    text: String, negativeText: String, configuration: GenerationConfiguration,
    denoiserParameterization: Denoiser.Parameterization, sampling: Sampling,
    signposts: inout Set<Signpost>,
    feedback: @escaping (Signpost, Set<Signpost>, Tensor<FloatType>?) -> Bool
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
    var file =
      (configuration.model.flatMap {
        ModelZoo.isModelDownloaded($0) ? $0 : nil
      }) ?? ModelZoo.defaultSpecification.file
    var modifier = ModelPreloader.modifierForModel(
      file, LoRAs: configuration.loras.compactMap(\.file))
    if modifier == .depth && depth == nil {
      // Revert to default file.
      modifier = .none
      file = ModelZoo.defaultSpecification.file
    }
    let modelVersion = ModelZoo.versionForModel(file)
    let textEncoderVersion = ModelZoo.textEncoderVersionForModel(file)
    let modelObjective = ModelZoo.objectiveForModel(file)
    let modelUpcastAttention = ModelZoo.isUpcastAttentionForModel(file)
    var textEncoderFiles: [String] =
      [
        ModelZoo.textEncoderForModel(file).flatMap {
          ModelZoo.isModelDownloaded($0) ? $0 : nil
        } ?? "clip_vit_l14_f16.ckpt"
      ]
      + ((ModelZoo.CLIPEncoderForModel(file).flatMap { ModelZoo.isModelDownloaded($0) ? $0 : nil })
        .map { [$0] } ?? [])
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
      } ?? "vae_ft_mse_840000_f16.ckpt"
    var isConsistencyModel = ModelZoo.isConsistencyModelForModel(file)
    let latentsScaling = ModelZoo.latentsScalingForModel(file)
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
    let lora: [LoRAConfiguration] = configuration.loras.compactMap {
      guard let file = $0.file else { return nil }
      let loraVersion = LoRAZoo.versionForModel(file)
      guard LoRAZoo.isModelDownloaded(file),
        modelVersion == loraVersion || refinerVersion == loraVersion
          || (modelVersion == .kandinsky21 && loraVersion == .v1)
      else { return nil }
      if LoRAZoo.isConsistencyModelForModel(file) {
        isConsistencyModel = true
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
    let (canInjectControls, canInjectT2IAdapters, injectIPAdapterLengths, canInjectedControls) =
      Self.canInjectControls(
        hasImage: true, hasDepth: depth != nil, hasHints: Set(hints.keys), hasCustom: custom != nil,
        shuffleCount: shuffles.count, controls: configuration.controls, version: modelVersion)
    let isMemoryEfficient = DynamicGraph.memoryEfficient
    if (canInjectControls && modelVersion == .v2) && !DeviceCapability.isMaxPerformance {
      DynamicGraph.memoryEfficient = true
    }
    defer {
      if (canInjectControls && modelVersion == .v2) && !DeviceCapability.isMaxPerformance {
        DynamicGraph.memoryEfficient = isMemoryEfficient
      }
    }
    let batchSize = Int(configuration.batchSize)
    precondition(batchSize > 0)
    let textGuidanceScale = configuration.guidanceScale
    let imageGuidanceScale = configuration.imageGuidanceScale
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
      if modelVersion == .sd3 {
        channels = 16
      } else {
        channels = 4
      }
      firstStageFilePath = ModelZoo.filePathForModelDownloaded(autoencoderFile)
    }
    let imageScale = DeviceCapability.Scale(
      widthScale: UInt16(startWidth / 8), heightScale: UInt16(startHeight / 8))
    let isHighPrecisionVAEFallbackEnabled = DeviceCapability.isHighPrecisionVAEFallbackEnabled(
      scale: imageScale)
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
        is8BitModel: ModelZoo.is8BitModel($0),
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
    let sampler = ImageGenerator.sampler(
      from: configuration.sampler, isConsistencyModel: isConsistencyModel,
      filePath: ModelZoo.filePathForModelDownloaded(file), modifier: modifier,
      version: modelVersion, usesFlashAttention: isMFAEnabled, objective: modelObjective,
      upcastAttention: modelUpcastAttention,
      externalOnDemand: externalOnDemand, injectControls: canInjectControls,
      injectT2IAdapters: canInjectT2IAdapters, injectIPAdapterLengths: injectIPAdapterLengths,
      lora: lora, is8BitModel: is8BitModel, canRunLoRASeparately: canRunLoRASeparately,
      stochasticSamplingGamma: configuration.stochasticSamplingGamma,
      conditioning: conditioning, parameterization: denoiserParameterization,
      tiledDiffusion: tiledDiffusion, of: FloatType.self)
    let initTimestep = sampler.timestep(for: strength, sampling: sampling)
    let graph = DynamicGraph()
    if externalOnDemand {
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
      graph: graph, modelVersion: modelVersion, textEncoderVersion: textEncoderVersion, text: text,
      negativeText: negativeText,
      negativePromptForImagePrior: configuration.negativePromptForImagePrior,
      potentials: potentials, T5TextEncoder: configuration.t5TextEncoder,
      clipL: configuration.separateClipL ? (configuration.clipLText ?? "") : nil,
      openClipG: configuration.separateOpenClipG ? (configuration.openClipGText ?? "") : nil
    )
    let tokenLength = max(tokenLengthUncond, tokenLengthCond)
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
      let textEncoder = TextEncoder<FloatType>(
        filePaths: textEncoderFiles.map { ModelZoo.filePathForModelDownloaded($0) },
        version: modelVersion, textEncoderVersion: textEncoderVersion,
        usesFlashAttention: isMFAEnabled && DeviceCapability.isMFACausalAttentionMaskSupported,
        injectEmbeddings: !injectedEmbeddings.isEmpty,
        externalOnDemand: textEncoderExternalOnDemand, maxLength: tokenLength, clipSkip: clipSkip,
        lora: lora)
      let image = downscaleImageAndToGPU(
        graph.variable(image), scaleFactor: imageScaleFactor)
      let textEncodings = modelPreloader.consumeTextModels(
        textEncoder.encode(
          tokens: tokensTensors, positions: positionTensors, mask: embedMask,
          injectedEmbeddings: injectedEmbeddings, image: [image], lengthsOfUncond: lengthsOfUncond,
          lengthsOfCond: lengthsOfCond,
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
      if hasNonOneWeights {
        // Need to scale c according to these weights. C has two parts, the unconditional and conditional.
        // We also want to do z-score type of scaling, so we need to compute mean of both parts separately.
        c = c.map { c in
          let conditionalLength = c.shape[2]
          guard tokenLength == c.shape[1] else { return c }
          var c = c
          let uc = c[0..<1, 0..<tokenLength, 0..<conditionalLength]
          let cc = c[1..<2, 0..<tokenLength, 0..<conditionalLength]
          let umean = uc.reduced(.mean, axis: [1, 2])
          let cmean = cc.reduced(.mean, axis: [1, 2])
          let uw = graph.variable(
            Tensor<FloatType>(
              from: Tensor(unconditionalAttentionWeights, .CPU, .HWC(1, tokenLength, 1))
            )
            .toGPU(0))
          let cw = graph.variable(
            Tensor<FloatType>(from: Tensor(attentionWeights, .CPU, .HWC(1, tokenLength, 1))).toGPU(
              0))
          // Keep the mean unchanged while scale it.
          c[0..<1, 0..<tokenLength, 0..<conditionalLength] = (uc - umean) .* uw + umean
          c[1..<2, 0..<tokenLength, 0..<conditionalLength] = (cc - cmean) .* cw + cmean
          return c
        }
      }
      if batchSize > 1 {
        c = c.map { c in
          var c = c
          let oldC = c
          let shape = c.shape
          if shape.count == 3 {
            c = graph.variable(
              .GPU(0), .HWC(batchSize * 2, shape[1], shape[2]), of: FloatType.self)
            for i in 0..<batchSize {
              c[i..<(i + 1), 0..<shape[1], 0..<shape[2]] =
                oldC[0..<1, 0..<shape[1], 0..<shape[2]]
              c[(batchSize + i)..<(batchSize + i + 1), 0..<shape[1], 0..<shape[2]] =
                oldC[1..<2, 0..<shape[1], 0..<shape[2]]
            }
          } else if shape.count == 2 {
            c = graph.variable(
              .GPU(0), .WC(batchSize * 2, shape[1]), of: FloatType.self)
            for i in 0..<batchSize {
              c[i..<(i + 1), 0..<shape[1]] = oldC[0..<1, 0..<shape[1]]
              c[(batchSize + i)..<(batchSize + i + 1), 0..<shape[1]] = oldC[1..<2, 0..<shape[1]]
            }
          }
          return c
        }
        if let oldProj = extraProjection {
          let shape = oldProj.shape
          var xfProj = graph.variable(
            .GPU(0), .HWC(batchSize * 2, shape[1], shape[2]), of: FloatType.self)
          for i in 0..<batchSize {
            xfProj[i..<(i + 1), 0..<shape[1], 0..<shape[2]] =
              oldProj[0..<1, 0..<shape[1], 0..<shape[2]]
            xfProj[(batchSize + i)..<(batchSize + i + 1), 0..<shape[1], 0..<shape[2]] =
              oldProj[1..<2, 0..<shape[1], 0..<shape[2]]
          }
          extraProjection = xfProj
        }
      }
      guard feedback(.textEncoded, signposts, nil) else { return nil }
      var firstStage = FirstStage<FloatType>(
        filePath: firstStageFilePath, version: modelVersion,
        latentsScaling: latentsScaling, highPrecision: highPrecisionForAutoencoder,
        highPrecisionFallback: isHighPrecisionVAEFallbackEnabled,
        tiledDecoding: tiledDecoding, tiledDiffusion: tiledDiffusion,
        externalOnDemand: vaeExternalOnDemand, alternativeUsesFlashAttention: isMFAEnabled,
        alternativeFilePath: alternativeDecoderFilePath,
        alternativeDecoderVersion: alternativeDecoderVersion)
      let firstPassImage: DynamicGraph.Tensor<FloatType>
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
      let (sample, _) = modelPreloader.consumeFirstStageSample(
        firstStage.sample(
          firstPassImage,
          encoder: modelPreloader.retrieveFirstStageEncoder(
            firstStage: firstStage, scale: imageScale)), firstStage: firstStage, scale: imageScale)
      var maskedImage1: DynamicGraph.Tensor<FloatType>? = nil
      var maskedImage2: DynamicGraph.Tensor<FloatType>? = nil
      if modifier == .inpainting || modifier == .editing || modelVersion == .svdI2v {
        var batch: DynamicGraph.Tensor<FloatType>
        if modelVersion == .svdI2v {
          batch = image
        } else {
          batch = graph.variable(
            .GPU(0), .NHWC(2, startHeight * 8, startWidth * 8, 3), of: FloatType.self)
          batch[0..<1, 0..<(startHeight * 8), 0..<(startWidth * 8), 0..<3] = image
            .* graph.variable(imageNegMask1.toGPU(0))
          if let imageNegMask2 = imageNegMask2 {
            batch[1..<2, 0..<(startHeight * 8), 0..<(startWidth * 8), 0..<3] = image
              .* graph.variable(imageNegMask2.toGPU(0))
          } else {
            batch[1..<2, 0..<(startHeight * 8), 0..<(startWidth * 8), 0..<3].full(0)
          }
        }
        let encodedBatch = modelPreloader.consumeFirstStageEncode(
          firstStage.encode(
            batch,
            encoder: modelPreloader.retrieveFirstStageEncoder(
              firstStage: firstStage, scale: imageScale)), firstStage: firstStage, scale: imageScale
        )
        if modifier == .inpainting {
          maskedImage1 = firstStage.scale(
            encodedBatch[0..<1, 0..<startHeight, 0..<startWidth, 0..<channels].copied())
          maskedImage2 = firstStage.scale(
            encodedBatch[1..<2, 0..<startHeight, 0..<startWidth, 0..<channels].copied())
        } else if modelVersion == .svdI2v {
          maskedImage1 = encodedBatch
          maskedImage2 = encodedBatch
        } else {
          maskedImage1 = encodedBatch[0..<1, 0..<startHeight, 0..<startWidth, 0..<channels].copied()
          maskedImage2 = encodedBatch[1..<2, 0..<startHeight, 0..<startWidth, 0..<channels].copied()
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
        graph: graph, startHeight: startHeight, startWidth: startWidth, image: image,
        depth: depthImage, hints: hints, custom: custom, shuffles: shuffles, mask: imageNegMask1,
        controls: configuration.controls, version: modelVersion, tiledDiffusion: tiledDiffusion,
        usesFlashAttention: isMFAEnabled, externalOnDemand: controlExternalOnDemand,
        steps: sampling.steps)
      let redoInjectedControls = configuration.controls.contains {
        $0.file.map { ControlNetZoo.modifierForModel($0) == .inpaint } ?? false
      }
      var batchSize = batchSize
      if modelVersion == .svdI2v {
        batchSize = Int(configuration.numFrames)
      }
      let noise = randomLatentNoise(
        graph: graph, batchSize: batchSize, startHeight: startHeight,
        startWidth: startWidth, channels: channels, seed: configuration.seed,
        seedMode: configuration.seedMode)
      let depth2Img = depthImage.map { downscaleDepthForDepth2Img($0) }
      var intermediateResult =
        try?
        (sampler.sample(
          noise,
          unets: modelPreloader.retrieveUNet(
            sampler: sampler, scale: imageScale, tokenLengthUncond: tokenLengthUncond,
            tokenLengthCond: tokenLengthCond), sample: sample,
          maskedImage: maskedImage1, depthImage: depth2Img, mask: initMask1, negMask: initNegMask,
          conditioning: c, tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
          extraProjection: extraProjection, injectedControls: injectedControls,
          textGuidanceScale: textGuidanceScale, imageGuidanceScale: imageGuidanceScale,
          startStep: (integral: 0, fractional: 0),
          endStep: (integral: initTimestep.roundedUpStartStep, fractional: initTimestep.startStep),
          originalSize: originalSize,
          cropTopLeft: cropTopLeft, targetSize: targetSize, aestheticScore: aestheticScore,
          negativeOriginalSize: negativeOriginalSize,
          negativeAestheticScore: negativeAestheticScore, zeroNegativePrompt: zeroNegativePrompt,
          refiner: refiner, fpsId: fpsId, motionBucketId: motionBucketId, condAug: condAug,
          startFrameCfg: startFrameCfg, sharpness: sharpness, sampling: sampling
        ) { step, tensor in
          feedback(.sampling(step), signposts, tensor)
        }).get()
      guard let x = intermediateResult?.x else {
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
          graph: graph, startHeight: startHeight, startWidth: startWidth, image: image,
          depth: depthImage, hints: hints, custom: custom, shuffles: shuffles, mask: imageNegMask2,
          controls: configuration.controls, version: modelVersion, tiledDiffusion: tiledDiffusion,
          usesFlashAttention: isMFAEnabled, externalOnDemand: controlExternalOnDemand,
          steps: sampling.steps
        )
      }
      guard
        var x =
          try? modelPreloader.consumeUNet(
            (sampler.sample(
              x_T, unets: intermediateResult?.unets ?? [nil], sample: sample,
              maskedImage: maskedImage2,
              depthImage: depth2Img, mask: initMask2, negMask: initNegMask2, conditioning: c,
              tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
              extraProjection: extraProjection, injectedControls: injectedControls,
              textGuidanceScale: textGuidanceScale, imageGuidanceScale: imageGuidanceScale,
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
              sharpness: sharpness, sampling: sampling
            ) { step, tensor in
              feedback(.sampling(initTimestep.roundedDownStartStep + step), signposts, tensor)
            }).get(), sampler: sampler, scale: imageScale, tokenLengthUncond: tokenLengthUncond,
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
          latentsScaling: latentsScaling, highPrecision: highPrecisionForAutoencoder,
          highPrecisionFallback: isHighPrecisionVAEFallbackEnabled,
          tiledDecoding: TiledConfiguration(
            isEnabled: false,
            tileSize: .init(width: 0, height: 0), tileOverlap: 0),
          tiledDiffusion: tiledDiffusion,
          externalOnDemand: vaeExternalOnDemand, alternativeUsesFlashAttention: isMFAEnabled,
          alternativeFilePath: alternativeDecoderFilePath,
          alternativeDecoderVersion: alternativeDecoderVersion)
        let (sample, _) = modelPreloader.consumeFirstStageSample(
          firstStage.sample(
            image,
            encoder: modelPreloader.retrieveFirstStageEncoder(
              firstStage: firstStage, scale: imageScale)), firstStage: firstStage, scale: imageScale
        )
        let startHeight = Int(configuration.startHeight) * 16
        let startWidth = Int(configuration.startWidth) * 16
        let channels = 4
        guard feedback(.secondPassImageEncoded, signposts, nil) else { return nil }
        let secondPassModelVersion = ModelVersion.wurstchenStageB
        let secondPassModelFilePath = ModelZoo.filePathForModelDownloaded(
          ModelZoo.stageModelsForModel(file)[0])
        let secondPassSampler = ImageGenerator.sampler(
          from: configuration.sampler, isConsistencyModel: isConsistencyModel,
          filePath: secondPassModelFilePath, modifier: modifier,
          version: secondPassModelVersion, usesFlashAttention: isMFAEnabled,
          objective: modelObjective,
          upcastAttention: modelUpcastAttention,
          externalOnDemand: externalOnDemand,
          injectControls: canInjectControls, injectT2IAdapters: canInjectT2IAdapters,
          injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
          is8BitModel: is8BitModel, canRunLoRASeparately: canRunLoRASeparately,
          stochasticSamplingGamma: configuration.stochasticSamplingGamma,
          conditioning: conditioning, parameterization: denoiserParameterization,
          tiledDiffusion: tiledDiffusion, of: FloatType.self
        )
        let noise = randomLatentNoise(
          graph: graph, batchSize: batchSize, startHeight: startHeight,
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
            sample: sample, maskedImage: nil, depthImage: nil, mask: initMask1,
            negMask: initNegMask, conditioning: c, tokenLengthUncond: tokenLengthUncond,
            tokenLengthCond: tokenLengthCond, extraProjection: extraProjection,
            injectedControls: [],  // TODO: Support injectedControls for this.
            textGuidanceScale: secondPassTextGuidance,
            imageGuidanceScale: imageGuidanceScale,
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
            sharpness: sharpness, sampling: secondPassSampling
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
              (secondPassSampler.sample(
                x_T, unets: intermediateResult?.unets ?? [nil], sample: sample,
                maskedImage: maskedImage2,
                depthImage: depth2Img, mask: initMask2, negMask: initNegMask2, conditioning: c,
                tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
                extraProjection: extraProjection, injectedControls: injectedControls,
                textGuidanceScale: secondPassTextGuidance, imageGuidanceScale: imageGuidanceScale,
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
                sharpness: sharpness, sampling: secondPassSampling
              ) { step, tensor in
                feedback(.sampling(initTimestep.roundedDownStartStep + step), signposts, tensor)
              }).get(), sampler: secondPassSampler, scale: imageScale,
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
            x,
            decoder: modelPreloader.retrieveFirstStageDecoder(
              firstStage: firstStage, scale: imageScale)), firstStage: firstStage, scale: imageScale
        )
      )
      .rawValue
      .toCPU()
      guard !isNaN(result) else { return nil }
      if modelVersion == .wurstchenStageC {
        guard feedback(.secondPassImageDecoded, signposts, nil) else { return nil }
      } else {
        guard feedback(.imageDecoded, signposts, nil) else { return nil }
      }
      guard batchSize > 1 else {
        return [result]
      }
      var batch = [Tensor<FloatType>]()
      let shape = result.shape
      for i in 0..<batchSize {
        batch.append(result[i..<(i + 1), 0..<shape[1], 0..<shape[2], 0..<shape[3]].copied())
      }
      return batch
    }
  }
}
