import Dflat
import Diffusion
import NNC

#if !os(Linux)
  import DiffusionCoreML
#endif

public struct UNetWrapper<FloatType: TensorNumeric & BinaryFloatingPoint>: UNetProtocol {

  private var unetFromNNC = UNetFromNNC<FloatType>()
  #if !os(Linux)
    private var unetFromCoreML = UNetFromCoreML<FloatType>()
  #endif
  private var preferCoreML = false
  public init() {}
  public var version: ModelVersion {
    #if !os(Linux)
      if preferCoreML {
        return unetFromCoreML.version
      } else {
        return unetFromNNC.version
      }
    #else
      return unetFromNNC.version
    #endif

  }
  public var isLoaded: Bool {
    #if !os(Linux)
      return unetFromCoreML.isLoaded || unetFromNNC.isLoaded
    #else
      return unetFromNNC.isLoaded
    #endif
  }
  public func unloadResources() {
    unetFromNNC.unloadResources()
    #if !os(Linux)
      unetFromCoreML.unloadResources()
    #endif

  }
}

extension UNetWrapper {
  public var modelAndWeightMapper: (Model, ModelWeightMapper)? {
    guard unetFromNNC.isLoaded else { return nil }
    return unetFromNNC.modelAndWeightMapper
  }
  public mutating func compileModel(
    filePath: String, externalOnDemand: Bool, version: ModelVersion, upcastAttention: Bool,
    usesFlashAttention: Bool, injectControls: Bool, injectT2IAdapters: Bool,
    injectAttentionKV: Bool,
    injectIPAdapterLengths: [Int], lora: [LoRAConfiguration],
    is8BitModel: Bool, canRunLoRASeparately: Bool, inputs xT: DynamicGraph.Tensor<FloatType>,
    _ timestep: DynamicGraph.Tensor<FloatType>?,
    _ c: [DynamicGraph.Tensor<FloatType>], tokenLengthUncond: Int, tokenLengthCond: Int,
    extraProjection: DynamicGraph.Tensor<FloatType>?,
    injectedControlsAndAdapters: InjectedControlsAndAdapters<FloatType>,
    tiledDiffusion: TiledConfiguration
  ) -> Bool {
    #if !os(Linux)

      if unetFromCoreML.compileModel(
        filePath: filePath, externalOnDemand: externalOnDemand, version: version,
        upcastAttention: upcastAttention, usesFlashAttention: usesFlashAttention,
        injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
        injectAttentionKV: injectAttentionKV,
        injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
        is8BitModel: is8BitModel, canRunLoRASeparately: canRunLoRASeparately, inputs: xT,
        timestep, c,
        tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
        extraProjection: extraProjection, injectedControlsAndAdapters: injectedControlsAndAdapters,
        tiledDiffusion: tiledDiffusion)
      {
        preferCoreML = true
        return true
      }
    #endif
    let _ = unetFromNNC.compileModel(
      filePath: filePath, externalOnDemand: externalOnDemand, version: version,
      upcastAttention: upcastAttention, usesFlashAttention: usesFlashAttention,
      injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
      injectAttentionKV: injectAttentionKV,
      injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
      is8BitModel: is8BitModel, canRunLoRASeparately: canRunLoRASeparately, inputs: xT, timestep, c,
      tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
      extraProjection: extraProjection, injectedControlsAndAdapters: injectedControlsAndAdapters,
      tiledDiffusion: tiledDiffusion)
    return true
  }

  public func callAsFunction(
    timestep t: Float,
    inputs xT: DynamicGraph.Tensor<FloatType>, _ timestep: DynamicGraph.Tensor<FloatType>?,
    _ c: [DynamicGraph.Tensor<FloatType>], extraProjection: DynamicGraph.Tensor<FloatType>?,
    injectedControlsAndAdapters: (
      _ xT: DynamicGraph.Tensor<FloatType>, _ inputStartYPad: Int, _ inputEndYPad: Int,
      _ inputStartXPad: Int, _ inputEndXPad: Int, _ existingControlNets: inout [Model?]
    ) -> (
      injectedControls: [DynamicGraph.Tensor<FloatType>],
      injectedT2IAdapters: [DynamicGraph.Tensor<FloatType>],
      injectedAttentionKVs: [NNC.DynamicGraph.Tensor<FloatType>]
    ),
    injectedIPAdapters: [DynamicGraph.Tensor<FloatType>],
    tiledDiffusion: TiledConfiguration,
    controlNets: inout [Model?]
  ) -> DynamicGraph.Tensor<FloatType> {
    #if !os(Linux)

      if preferCoreML {
        return unetFromCoreML(
          timestep: t, inputs: xT, timestep, c, extraProjection: extraProjection,
          injectedControlsAndAdapters: injectedControlsAndAdapters,
          injectedIPAdapters: injectedIPAdapters, tiledDiffusion: tiledDiffusion,
          controlNets: &controlNets)
      }
    #endif

    return unetFromNNC(
      timestep: t, inputs: xT, timestep, c, extraProjection: extraProjection,
      injectedControlsAndAdapters: injectedControlsAndAdapters,
      injectedIPAdapters: injectedIPAdapters, tiledDiffusion: tiledDiffusion,
      controlNets: &controlNets)
  }

  public func decode(_ x: DynamicGraph.Tensor<FloatType>) -> DynamicGraph.Tensor<FloatType> {
    #if !os(Linux)
      if preferCoreML {
        return unetFromCoreML.decode(x)
      }
    #endif

    return unetFromNNC.decode(x)

  }
}
