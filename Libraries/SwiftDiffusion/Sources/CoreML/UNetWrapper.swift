import Diffusion
import NNC

public struct UNetWrapper<FloatType: TensorNumeric & BinaryFloatingPoint>: UNetProtocol {
  private var unetFromNNC = UNetFromNNC<FloatType>()
  private var unetFromCoreML = UNetFromCoreML<FloatType>()
  private var preferCoreML = false
  public init() {}
  public var version: ModelVersion {
    if preferCoreML {
      return unetFromCoreML.version
    } else {
      return unetFromNNC.version
    }
  }
  public var isLoaded: Bool { unetFromCoreML.isLoaded || unetFromNNC.isLoaded }
  public func unloadResources() {
    unetFromNNC.unloadResources()
    unetFromCoreML.unloadResources()
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
    injectIPAdapterLengths: [Int], lora: [LoRAConfiguration],
    is8BitModel: Bool, canRunLoRASeparately: Bool, inputs xT: DynamicGraph.Tensor<FloatType>,
    _ timestep: DynamicGraph.Tensor<FloatType>?,
    _ c: [DynamicGraph.Tensor<FloatType>], tokenLengthUncond: Int, tokenLengthCond: Int,
    extraProjection: DynamicGraph.Tensor<FloatType>?,
    injectedControls: [DynamicGraph.Tensor<FloatType>],
    injectedT2IAdapters: [DynamicGraph.Tensor<FloatType>],
    injectedIPAdapters: [DynamicGraph.Tensor<FloatType>],
    tiledDiffusion: TiledConfiguration
  ) -> Bool {
    if unetFromCoreML.compileModel(
      filePath: filePath, externalOnDemand: externalOnDemand, version: version,
      upcastAttention: upcastAttention, usesFlashAttention: usesFlashAttention,
      injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
      injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
      is8BitModel: is8BitModel, canRunLoRASeparately: canRunLoRASeparately, inputs: xT, timestep, c,
      tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
      extraProjection: extraProjection, injectedControls: injectedControls,
      injectedT2IAdapters: injectedT2IAdapters, injectedIPAdapters: injectedIPAdapters,
      tiledDiffusion: tiledDiffusion)
    {
      preferCoreML = true
      return true
    }
    let _ = unetFromNNC.compileModel(
      filePath: filePath, externalOnDemand: externalOnDemand, version: version,
      upcastAttention: upcastAttention, usesFlashAttention: usesFlashAttention,
      injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
      injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
      is8BitModel: is8BitModel, canRunLoRASeparately: canRunLoRASeparately, inputs: xT, timestep, c,
      tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
      extraProjection: extraProjection, injectedControls: injectedControls,
      injectedT2IAdapters: injectedT2IAdapters, injectedIPAdapters: injectedIPAdapters,
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
      injectedT2IAdapters: [DynamicGraph.Tensor<FloatType>]
    ),
    injectedIPAdapters: [DynamicGraph.Tensor<FloatType>],
    tiledDiffusion: TiledConfiguration,
    controlNets: inout [Model?]
  ) -> DynamicGraph.Tensor<FloatType> {
    if preferCoreML {
      return unetFromCoreML(
        timestep: t, inputs: xT, timestep, c, extraProjection: extraProjection,
        injectedControlsAndAdapters: injectedControlsAndAdapters,
        injectedIPAdapters: injectedIPAdapters, tiledDiffusion: tiledDiffusion,
        controlNets: &controlNets)
    }
    return unetFromNNC(
      timestep: t, inputs: xT, timestep, c, extraProjection: extraProjection,
      injectedControlsAndAdapters: injectedControlsAndAdapters,
      injectedIPAdapters: injectedIPAdapters, tiledDiffusion: tiledDiffusion,
      controlNets: &controlNets)
  }

  public func decode(_ x: DynamicGraph.Tensor<FloatType>) -> DynamicGraph.Tensor<FloatType> {
    if preferCoreML {
      return unetFromCoreML.decode(x)
    } else {
      return unetFromNNC.decode(x)
    }
  }
}
