import Foundation
import NNC

private func xPrevAndPredX0<T: TensorNumeric>(
  x: DynamicGraph.Tensor<T>, et: DynamicGraph.Tensor<T>, alpha: Double,
  alphaPrev: Double
) -> (DynamicGraph.Tensor<T>, DynamicGraph.Tensor<T>) {
  let predX0 = Float(1 / alpha.squareRoot()) * (x - Float((1 - alpha).squareRoot()) * et)
  let dirXt = Float((1 - alphaPrev).squareRoot()) * et
  let xPrev = Float(alphaPrev.squareRoot()) * predX0 + dirXt
  return (xPrev, predX0)
}

public struct PLMSSampler<
  FloatType: TensorNumeric & BinaryFloatingPoint, UNet: UNetProtocol,
  Discretization: Denoiser.Discretization
>
where UNet.FloatType == FloatType {
  public let filePath: String
  public let modifier: SamplerModifier
  public let version: ModelVersion
  public let usesFlashAttention: Bool
  public let upcastAttention: Bool
  public let externalOnDemand: Bool
  public let injectControls: Bool
  public let injectT2IAdapters: Bool
  public let injectIPAdapterLengths: [Int]
  public let lora: [LoRAConfiguration]
  public let is8BitModel: Bool
  public let canRunLoRASeparately: Bool
  public let conditioning: Denoiser.Conditioning
  public let tiledDiffusion: TiledDiffusionConfiguration
  private let discretization: Discretization
  public init(
    filePath: String, modifier: SamplerModifier, version: ModelVersion, usesFlashAttention: Bool,
    upcastAttention: Bool, externalOnDemand: Bool, injectControls: Bool,
    injectT2IAdapters: Bool, injectIPAdapterLengths: [Int], lora: [LoRAConfiguration],
    is8BitModel: Bool, canRunLoRASeparately: Bool, conditioning: Denoiser.Conditioning,
    tiledDiffusion: TiledDiffusionConfiguration, discretization: Discretization
  ) {
    self.filePath = filePath
    self.modifier = modifier
    self.version = version
    self.usesFlashAttention = usesFlashAttention
    self.upcastAttention = upcastAttention
    self.externalOnDemand = externalOnDemand
    self.injectControls = injectControls
    self.injectT2IAdapters = injectT2IAdapters
    self.injectIPAdapterLengths = injectIPAdapterLengths
    self.lora = lora
    self.is8BitModel = is8BitModel
    self.canRunLoRASeparately = canRunLoRASeparately
    self.conditioning = conditioning
    self.tiledDiffusion = tiledDiffusion
    self.discretization = discretization
  }
}

extension PLMSSampler: Sampler {
  public func sample(
    _ x_T: DynamicGraph.Tensor<FloatType>, unets existingUNets: [UNet?],
    sample: DynamicGraph.Tensor<FloatType>?, maskedImage: DynamicGraph.Tensor<FloatType>?,
    depthImage: DynamicGraph.Tensor<FloatType>?,
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
  ) -> Result<SamplerOutput<FloatType, UNet>, Error> {
    let startStep = startStep.integral
    let endStep = endStep.integral
    var x = x_T
    var c0 = c[0]
    let batchSize = x.shape[0]
    let startHeight = x.shape[1]
    let startWidth = x.shape[2]
    let channels = x.shape[3]
    let graph = x.graph
    var isCfgEnabled = isCfgEnabled(
      textGuidanceScale: textGuidanceScale, startFrameCfg: startFrameCfg, version: version)
    let cfgChannels: Int
    let inChannels: Int
    if version == .svdI2v {
      cfgChannels = 1
      inChannels = channels * 2
    } else {
      switch modifier {
      case .inpainting:
        cfgChannels = isCfgEnabled ? 2 : 1
        inChannels = channels * 2 + 1
      case .depth:
        cfgChannels = isCfgEnabled ? 2 : 1
        inChannels = channels + 1
      case .editing:
        cfgChannels = 3
        inChannels = channels * 2
        isCfgEnabled = true
      case .none:
        cfgChannels = isCfgEnabled ? 2 : 1
        inChannels = channels
      }
    }
    var xIn = graph.variable(
      .GPU(0), .NHWC(cfgChannels * batchSize, startHeight, startWidth, inChannels),
      of: FloatType.self
    )
    switch modifier {
    case .inpainting:
      let maskedImage = maskedImage!
      let mask = mask!
      for i in 0..<batchSize {
        xIn[i..<(i + 1), 0..<startHeight, 0..<startWidth, channels..<(channels + 1)] = mask
        xIn[i..<(i + 1), 0..<startHeight, 0..<startWidth, (channels + 1)..<(channels * 2 + 1)] =
          maskedImage
        if isCfgEnabled {
          xIn[
            (batchSize + i)..<(batchSize + i + 1), 0..<startHeight, 0..<startWidth,
            channels..<(channels + 1)] = mask
          xIn[
            (batchSize + i)..<(batchSize + i + 1), 0..<startHeight, 0..<startWidth,
            (channels + 1)..<(channels * 2 + 1)] =
            maskedImage
        }
      }
    case .editing:
      let maskedImage = maskedImage!
      for i in 0..<batchSize {
        xIn[i..<(i + 1), 0..<startHeight, 0..<startWidth, channels..<(channels * 2)] = maskedImage
        xIn[
          (batchSize + i)..<(batchSize + i + 1), 0..<startHeight, 0..<startWidth,
          channels..<(channels * 2)] =
          maskedImage
        xIn[
          (batchSize * 2 + i)..<(batchSize * 2 + i + 1), 0..<startHeight, 0..<startWidth,
          channels..<(channels * 2)
        ]
        .full(0)
      }
      let oldC = c0
      c0 = graph.variable(
        .GPU(0), .HWC(3 * batchSize, oldC.shape[1], oldC.shape[2]), of: FloatType.self)
      // Expanding c.
      c0[0..<(batchSize * 2), 0..<oldC.shape[1], 0..<oldC.shape[2]] = oldC
      c0[(batchSize * 2)..<(batchSize * 3), 0..<oldC.shape[1], 0..<oldC.shape[2]] =
        oldC[0..<batchSize, 0..<oldC.shape[1], 0..<oldC.shape[2]]
    case .depth:
      let depthImage = depthImage!
      for i in 0..<batchSize {
        xIn[i..<(i + 1), 0..<startHeight, 0..<startWidth, channels..<(channels + 1)] = depthImage
        if isCfgEnabled {
          xIn[
            (batchSize + i)..<(batchSize + i + 1), 0..<startHeight, 0..<startWidth,
            channels..<(channels + 1)] =
            depthImage
        }
      }
    case .none:
      break
    }
    var c = c
    var extraProjection = extraProjection
    var tokenLengthUncond = tokenLengthUncond
    if !isCfgEnabled && version != .svdI2v {
      for i in 0..<c.count {
        let shape = c[i].shape
        if shape.count == 3 {
          let conditionalLength = version == .kandinsky21 ? shape[1] : tokenLengthCond
          // Only tokenLengthCond is used.
          c[i] = c[i][batchSize..<(batchSize * 2), 0..<conditionalLength, 0..<shape[2]].copied()
        } else if shape.count == 2 {
          c[i] = c[i][batchSize..<(batchSize * 2), 0..<shape[1]].copied()
        }
      }
      if var projection = extraProjection {
        let shape = projection.shape
        if shape.count == 3 {
          // Only tokenLengthCond is used.
          projection = projection[batchSize..<(batchSize * 2), 0..<shape[1], 0..<shape[2]].copied()
        } else if shape.count == 2 {
          projection = projection[batchSize..<(batchSize * 2), 0..<shape[1]].copied()
        }
        extraProjection = projection
      }
      // There is no tokenLengthUncond any more.
      tokenLengthUncond = tokenLengthCond
    }
    let oldC = c
    let fixedEncoder = UNetFixedEncoder<FloatType>(
      filePath: filePath, version: version, usesFlashAttention: usesFlashAttention,
      zeroNegativePrompt: zeroNegativePrompt)
    let injectedControlsC: [[DynamicGraph.Tensor<FloatType>]]
    if c.count >= 2 || version == .svdI2v {
      let vector = fixedEncoder.vector(
        textEmbedding: c[c.count - 1], originalSize: originalSize,
        cropTopLeft: cropTopLeft,
        targetSize: targetSize, aestheticScore: aestheticScore,
        negativeOriginalSize: negativeOriginalSize, negativeAestheticScore: negativeAestheticScore,
        fpsId: fpsId, motionBucketId: motionBucketId, condAug: condAug)
      let (encodings, weightMapper) = fixedEncoder.encode(
        textEncoding: c, batchSize: batchSize, startHeight: startHeight, startWidth: startWidth,
        tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond, lora: lora)
      c = vector + encodings
      injectedControlsC = injectedControls.map {
        $0.model.encode(
          textEncoding: oldC, vector: vector.first, batchSize: batchSize, startHeight: startHeight,
          startWidth: startWidth, tokenLengthUncond: tokenLengthUncond,
          tokenLengthCond: tokenLengthCond, zeroNegativePrompt: zeroNegativePrompt,
          mainUNetFixed: (fixedEncoder.filePath, weightMapper))
      }
    } else {
      injectedControlsC = injectedControls.map {
        $0.model.encode(
          textEncoding: oldC, vector: nil, batchSize: batchSize, startHeight: startHeight,
          startWidth: startWidth, tokenLengthUncond: tokenLengthUncond,
          tokenLengthCond: tokenLengthCond, zeroNegativePrompt: zeroNegativePrompt,
          mainUNetFixed: (fixedEncoder.filePath, nil))
      }
    }
    var unet = existingUNets[0] ?? UNet()
    var controlNets = [Model?](repeating: nil, count: injectedControls.count)
    if existingUNets[0] == nil {
      let firstTimestep =
        discretization.timesteps - discretization.timesteps / Float(sampling.steps) + 1
      let t = unet.timeEmbed(
        graph: graph, batchSize: cfgChannels * batchSize, timestep: firstTimestep, version: version)
      let (injectedControls, injectedT2IAdapters, injectedIPAdapters) =
        ControlModel<FloatType>
        .emptyInjectedControlsAndAdapters(
          injecteds: injectedControls, step: 0, version: version, inputs: xIn,
          tiledDiffusion: tiledDiffusion)
      let newC: [DynamicGraph.Tensor<FloatType>]
      if version == .svdI2v {
        newC = Array(c[0..<(1 + (c.count - 1) / 2)])
      } else {
        newC = c
      }
      let _ = unet.compileModel(
        filePath: filePath, externalOnDemand: externalOnDemand, version: version,
        upcastAttention: upcastAttention, usesFlashAttention: usesFlashAttention,
        injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
        injectIPAdapterLengths: injectIPAdapterLengths, lora: lora,
        is8BitModel: is8BitModel, canRunLoRASeparately: canRunLoRASeparately,
        inputs: xIn, t, newC,
        tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
        extraProjection: extraProjection, injectedControls: injectedControls,
        injectedT2IAdapters: injectedT2IAdapters, injectedIPAdapters: injectedIPAdapters,
        tiledDiffusion: tiledDiffusion)
    }
    let alphasCumprod = discretization.alphasCumprod(steps: sampling.steps, shift: sampling.shift)
    var noise: DynamicGraph.Tensor<FloatType>? = nil
    if mask != nil {
      noise = graph.variable(.GPU(0), .NHWC(batchSize, startHeight, startWidth, channels))
    }
    let condAugFrames: DynamicGraph.Tensor<FloatType>?
    let textGuidanceVector: DynamicGraph.Tensor<FloatType>?
    if version == .svdI2v {
      let scaleCPU = graph.variable(.CPU, .NHWC(batchSize, 1, 1, 1), of: FloatType.self)
      for i in 0..<batchSize {
        scaleCPU[i, 0, 0, 0] = FloatType(
          Float(i) * (textGuidanceScale - startFrameCfg) / Float(batchSize - 1) + startFrameCfg)
      }
      textGuidanceVector = scaleCPU.toGPU(0)
      let maskedImage = maskedImage!
      var frames = graph.variable(
        .GPU(0), .NHWC(batchSize, startHeight, startWidth, channels), of: FloatType.self)
      for i in 0..<batchSize {
        frames[i..<(i + 1), 0..<startHeight, 0..<startWidth, 0..<channels] = maskedImage
      }
      if condAug > 0 {
        let noise = graph.variable(like: frames)
        noise.randn(std: condAug)
        frames = frames .+ noise
      }
      condAugFrames = frames
    } else {
      textGuidanceVector = nil
      condAugFrames = nil
    }
    let blur: Model?
    if sharpness > 0 {
      blur = Blur(filters: channels, sigma: 3.0, size: 13, input: x)
    } else {
      blur = nil
    }
    let streamContext = StreamContext(.GPU(0))
    let injecteds = injectedControls
    var refinerKickIn = refiner.map { (1 - $0.start) * discretization.timesteps } ?? -1
    var unets: [UNet?] = [unet]
    var currentModelVersion = version
    let result: Result<SamplerOutput<FloatType, UNet>, Error> = graph.withStream(streamContext) {
      var oldEps = [DynamicGraph.Tensor<FloatType>]()
      // Now do PLMS sampling.
      for i in startStep..<endStep {
        let rawValue: Tensor<FloatType>? =
          (i > max(startStep, sampling.steps / 2) || i % 5 == 4) ? x.rawValue.toCPU() : nil
        if i % 5 == 4, let rawValue = rawValue {
          if isNaN(rawValue) {
            return .failure(SamplerError.isNaN)
          }
        }
        guard feedback(i - startStep, rawValue) else { return .failure(SamplerError.cancelled) }
        let timestep = discretization.timestep(for: alphasCumprod[i])
        if timestep < refinerKickIn, let refiner = refiner {
          unets = [nil]
          let fixedEncoder = UNetFixedEncoder<FloatType>(
            filePath: refiner.filePath, version: refiner.version,
            usesFlashAttention: usesFlashAttention, zeroNegativePrompt: zeroNegativePrompt)
          if oldC.count >= 2 || refiner.version == .svdI2v {
            let vector = fixedEncoder.vector(
              textEmbedding: oldC[oldC.count - 1], originalSize: originalSize,
              cropTopLeft: cropTopLeft,
              targetSize: targetSize, aestheticScore: aestheticScore,
              negativeOriginalSize: negativeOriginalSize,
              negativeAestheticScore: negativeAestheticScore, fpsId: fpsId,
              motionBucketId: motionBucketId, condAug: condAug)
            c =
              vector
              + fixedEncoder.encode(
                textEncoding: oldC, batchSize: batchSize, startHeight: startHeight,
                startWidth: startWidth, tokenLengthUncond: tokenLengthUncond,
                tokenLengthCond: tokenLengthCond, lora: lora
              ).0
          }
          unet = UNet()
          currentModelVersion = refiner.version
          let firstTimestep =
            discretization.timesteps - discretization.timesteps / Float(sampling.steps) + 1
          let t = unet.timeEmbed(
            graph: graph, batchSize: cfgChannels * batchSize, timestep: firstTimestep,
            version: currentModelVersion)
          let (injectedControls, injectedT2IAdapters, injectedIPAdapters) =
            ControlModel<FloatType>
            .emptyInjectedControlsAndAdapters(
              injecteds: injectedControls, step: 0, version: refiner.version, inputs: xIn,
              tiledDiffusion: tiledDiffusion)
          let newC: [DynamicGraph.Tensor<FloatType>]
          if version == .svdI2v {
            newC = Array(c[0..<(1 + (c.count - 1) / 2)])
          } else {
            newC = c
          }
          let _ = unet.compileModel(
            filePath: refiner.filePath, externalOnDemand: refiner.externalOnDemand,
            version: refiner.version, upcastAttention: upcastAttention,
            usesFlashAttention: usesFlashAttention, injectControls: injectControls,
            injectT2IAdapters: injectT2IAdapters, injectIPAdapterLengths: injectIPAdapterLengths,
            lora: lora, is8BitModel: refiner.is8BitModel,
            canRunLoRASeparately: canRunLoRASeparately,
            inputs: xIn, t, newC,
            tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
            extraProjection: extraProjection, injectedControls: injectedControls,
            injectedT2IAdapters: injectedT2IAdapters, injectedIPAdapters: injectedIPAdapters,
            tiledDiffusion: tiledDiffusion)
          refinerKickIn = -1
          unets.append(unet)
        }
        let cNoise: Float
        switch conditioning {
        case .noise:
          cNoise = discretization.noise(for: alphasCumprod[i])
        case .timestep:
          cNoise = timestep
        }
        let t = unet.timeEmbed(
          graph: graph, batchSize: cfgChannels * batchSize, timestep: cNoise,
          version: currentModelVersion)
        var et: DynamicGraph.Tensor<FloatType>
        if version == .svdI2v, let textGuidanceVector = textGuidanceVector,
          let condAugFrames = condAugFrames
        {
          xIn[0..<batchSize, 0..<startHeight, 0..<startWidth, 0..<channels] = x
          xIn[0..<batchSize, 0..<startHeight, 0..<startWidth, channels..<(channels * 2)] =
            condAugFrames
          let injectedIPAdapters = ControlModel<FloatType>
            .injectedIPAdapters(
              injecteds: injectedControls, step: i, version: unet.version,
              usesFlashAttention: usesFlashAttention, inputs: xIn, t, injectedControlsC,
              tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
              isCfgEnabled: isCfgEnabled, mainUNetAndWeightMapper: unet.modelAndWeightMapper,
              controlNets: &controlNets)
          let injectedControlsAndAdapters = ControlModel<FloatType>
            .injectedControlsAndAdapters(
              injecteds: injectedControls, step: i, version: unet.version,
              usesFlashAttention: usesFlashAttention, inputs: xIn, t, injectedControlsC,
              tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
              isCfgEnabled: isCfgEnabled, mainUNetAndWeightMapper: unet.modelAndWeightMapper,
              controlNets: &controlNets)
          let cCond = Array(c[0..<(1 + (c.count - 1) / 2)])
          var etCond = unet(
            timestep: cNoise, inputs: xIn, t, cCond, extraProjection: extraProjection,
            injectedControlsAndAdapters: injectedControlsAndAdapters,
            injectedIPAdapters: injectedIPAdapters, tiledDiffusion: tiledDiffusion,
            controlNets: &controlNets)
          let alpha =
            0.001 * sharpness * (discretization.timesteps - timestep)
            / discretization.timesteps
          if isCfgEnabled {
            xIn[0..<batchSize, 0..<startHeight, 0..<startWidth, channels..<(channels * 2)].full(0)
            let cUncond = Array([c[0]] + c[(1 + (c.count - 1) / 2)...])
            let etUncond = unet(
              timestep: cNoise, inputs: xIn, t, cUncond, extraProjection: extraProjection,
              injectedControlsAndAdapters: injectedControlsAndAdapters,
              injectedIPAdapters: injectedIPAdapters, tiledDiffusion: tiledDiffusion,
              controlNets: &controlNets)
            if let blur = blur {
              let etCondDegraded = blur(inputs: etCond)[0].as(of: FloatType.self)
              etCond = Functional.add(
                left: etCondDegraded, right: etCond, leftScalar: alpha, rightScalar: 1 - alpha)
            }
            et = etUncond + textGuidanceVector .* (etCond - etUncond)
          } else {
            if let blur = blur {
              let etCondDegraded = blur(inputs: etCond)[0].as(of: FloatType.self)
              etCond = Functional.add(
                left: etCondDegraded, right: etCond, leftScalar: alpha, rightScalar: 1 - alpha)
            }
            et = etCond
          }
        } else {
          xIn[0..<batchSize, 0..<startHeight, 0..<startWidth, 0..<channels] = x
          if isCfgEnabled {
            xIn[batchSize..<(batchSize * 2), 0..<startHeight, 0..<startWidth, 0..<channels] = x
            if modifier == .editing {
              xIn[
                (batchSize * 2)..<(batchSize * 3), 0..<startHeight, 0..<startWidth, 0..<channels] =
                x
            }
          }
          let injectedIPAdapters = ControlModel<FloatType>
            .injectedIPAdapters(
              injecteds: injecteds, step: i, version: unet.version,
              usesFlashAttention: usesFlashAttention, inputs: xIn, t, injectedControlsC,
              tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
              isCfgEnabled: isCfgEnabled, mainUNetAndWeightMapper: unet.modelAndWeightMapper,
              controlNets: &controlNets)
          let injectedControlsAndAdapters = ControlModel<FloatType>
            .injectedControlsAndAdapters(
              injecteds: injecteds, step: i, version: unet.version,
              usesFlashAttention: usesFlashAttention, inputs: xIn, t, injectedControlsC,
              tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
              isCfgEnabled: isCfgEnabled, mainUNetAndWeightMapper: unet.modelAndWeightMapper,
              controlNets: &controlNets)
          var etOut = unet(
            timestep: cNoise, inputs: xIn, t, c, extraProjection: extraProjection,
            injectedControlsAndAdapters: injectedControlsAndAdapters,
            injectedIPAdapters: injectedIPAdapters, tiledDiffusion: tiledDiffusion,
            controlNets: &controlNets)
          let alpha =
            0.001 * sharpness * (discretization.timesteps - timestep)
            / discretization.timesteps
          if isCfgEnabled {
            var etUncond = graph.variable(
              .GPU(0), .NHWC(batchSize, startHeight, startWidth, channels), of: FloatType.self)
            var etCond = graph.variable(
              .GPU(0), .NHWC(batchSize, startHeight, startWidth, channels), of: FloatType.self)
            etUncond[0..<batchSize, 0..<startHeight, 0..<startWidth, 0..<channels] =
              etOut[0..<batchSize, 0..<startHeight, 0..<startWidth, 0..<channels]
            etCond[0..<batchSize, 0..<startHeight, 0..<startWidth, 0..<channels] =
              etOut[batchSize..<(batchSize * 2), 0..<startHeight, 0..<startWidth, 0..<channels]
            if let blur = blur {
              let etCondDegraded = blur(inputs: etCond)[0].as(of: FloatType.self)
              etCond = Functional.add(
                left: etCondDegraded, right: etCond, leftScalar: alpha, rightScalar: 1 - alpha)
            }
            if modifier == .editing {
              var etAllUncond = graph.variable(
                .GPU(0), .NHWC(batchSize, startHeight, startWidth, channels), of: FloatType.self)
              etAllUncond[0..<batchSize, 0..<startHeight, 0..<startWidth, 0..<channels] =
                etOut[
                  (batchSize * 2)..<(batchSize * 3), 0..<startHeight, 0..<startWidth, 0..<channels]
              et =
                etAllUncond + textGuidanceScale * (etCond - etUncond) + imageGuidanceScale
                * (etUncond - etAllUncond)
            } else {
              et = etUncond + textGuidanceScale * (etCond - etUncond)
            }
          } else {
            if let blur = blur {
              let etOutDegraded = blur(inputs: etOut)[0].as(of: FloatType.self)
              etOut = Functional.add(
                left: etOutDegraded, right: etOut, leftScalar: alpha, rightScalar: 1 - alpha)
            }
            et = etOut
          }
        }
        let alpha = alphasCumprod[i]
        let alphaPrev = alphasCumprod[i + 1]
        switch discretization.objective {
        case .v:
          et = Float(alpha.squareRoot()) * et + Float((1 - alpha).squareRoot()) * x
        case .epsilon:
          break
        case .edm(_):
          break
        }
        let etPrime: DynamicGraph.Tensor<FloatType>
        switch oldEps.count {
        case 0:
          let (xPrev, _) = xPrevAndPredX0(x: x, et: et, alpha: alpha, alphaPrev: alphaPrev)
          let timestepNext = discretization.timestep(for: alphasCumprod[i + 1])
          let cNoiseNext: Float
          switch conditioning {
          case .noise:
            cNoiseNext = discretization.noise(for: alphasCumprod[i + 1])
          case .timestep:
            cNoiseNext = timestepNext
          }
          let tNext = unet.timeEmbed(
            graph: graph, batchSize: cfgChannels * batchSize, timestep: cNoiseNext,
            version: currentModelVersion)
          var etNext: DynamicGraph.Tensor<FloatType>
          if version == .svdI2v, let textGuidanceVector = textGuidanceVector,
            let condAugFrames = condAugFrames
          {
            xIn[0..<batchSize, 0..<startHeight, 0..<startWidth, 0..<channels] = xPrev
            xIn[0..<batchSize, 0..<startHeight, 0..<startWidth, channels..<(channels * 2)] =
              condAugFrames
            let injectedIPAdapters = ControlModel<FloatType>.injectedIPAdapters(
              injecteds: injectedControls, step: i, version: unet.version,
              usesFlashAttention: usesFlashAttention, inputs: xIn, t, injectedControlsC,
              tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
              isCfgEnabled: isCfgEnabled, mainUNetAndWeightMapper: unet.modelAndWeightMapper,
              controlNets: &controlNets)
            let injectedControlsAndAdapters = ControlModel<FloatType>.injectedControlsAndAdapters(
              injecteds: injectedControls, step: i, version: unet.version,
              usesFlashAttention: usesFlashAttention, inputs: xIn, t, injectedControlsC,
              tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
              isCfgEnabled: isCfgEnabled, mainUNetAndWeightMapper: unet.modelAndWeightMapper,
              controlNets: &controlNets)
            let cCond = Array(c[0..<(1 + (c.count - 1) / 2)])
            var etNextCond = unet(
              timestep: cNoiseNext, inputs: xIn, tNext, cCond, extraProjection: extraProjection,
              injectedControlsAndAdapters: injectedControlsAndAdapters,
              injectedIPAdapters: injectedIPAdapters, tiledDiffusion: tiledDiffusion,
              controlNets: &controlNets)
            let alpha =
              0.001 * sharpness * (discretization.timesteps - timestepNext)
              / discretization.timesteps
            if isCfgEnabled {
              xIn[0..<batchSize, 0..<startHeight, 0..<startWidth, channels..<(channels * 2)].full(0)
              let cUncond = Array([c[0]] + c[(1 + (c.count - 1) / 2)...])
              let etNextUncond = unet(
                timestep: cNoiseNext, inputs: xIn, tNext, cUncond, extraProjection: extraProjection,
                injectedControlsAndAdapters: injectedControlsAndAdapters,
                injectedIPAdapters: injectedIPAdapters, tiledDiffusion: tiledDiffusion,
                controlNets: &controlNets)
              if let blur = blur {
                let etNextCondDegraded = blur(inputs: etNextCond)[0].as(of: FloatType.self)
                etNextCond = Functional.add(
                  left: etNextCondDegraded, right: etNextCond, leftScalar: alpha,
                  rightScalar: 1 - alpha)
              }
              etNext = etNextUncond + textGuidanceVector .* (etNextCond - etNextUncond)
            } else {
              if let blur = blur {
                let etNextCondDegraded = blur(inputs: etNextCond)[0].as(of: FloatType.self)
                etNextCond = Functional.add(
                  left: etNextCondDegraded, right: etNextCond, leftScalar: alpha,
                  rightScalar: 1 - alpha)
              }
              etNext = etNextCond
            }
          } else {
            // Compute etNext.
            xIn[0..<batchSize, 0..<startHeight, 0..<startWidth, 0..<channels] = xPrev
            if isCfgEnabled {
              xIn[batchSize..<(batchSize * 2), 0..<startHeight, 0..<startWidth, 0..<channels] =
                xPrev
            }
            let injectedIPAdapters = ControlModel<FloatType>.injectedIPAdapters(
              injecteds: injecteds, step: i, version: unet.version,
              usesFlashAttention: usesFlashAttention, inputs: xIn, t, injectedControlsC,
              tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
              isCfgEnabled: isCfgEnabled, mainUNetAndWeightMapper: unet.modelAndWeightMapper,
              controlNets: &controlNets)
            let injectedControlsAndAdapters = ControlModel<FloatType>.injectedControlsAndAdapters(
              injecteds: injecteds, step: i, version: unet.version,
              usesFlashAttention: usesFlashAttention, inputs: xIn, t, injectedControlsC,
              tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
              isCfgEnabled: isCfgEnabled, mainUNetAndWeightMapper: unet.modelAndWeightMapper,
              controlNets: &controlNets)
            var etNextOut = unet(
              timestep: cNoiseNext, inputs: xIn, tNext, c, extraProjection: extraProjection,
              injectedControlsAndAdapters: injectedControlsAndAdapters,
              injectedIPAdapters: injectedIPAdapters, tiledDiffusion: tiledDiffusion,
              controlNets: &controlNets
            )
            let alpha =
              0.001 * sharpness * (discretization.timesteps - timestepNext)
              / discretization.timesteps
            if isCfgEnabled {
              var etNextUncond = graph.variable(
                .GPU(0), .NHWC(batchSize, startHeight, startWidth, channels), of: FloatType.self)
              var etNextCond = graph.variable(
                .GPU(0), .NHWC(batchSize, startHeight, startWidth, channels), of: FloatType.self)
              etNextUncond[0..<batchSize, 0..<startHeight, 0..<startWidth, 0..<channels] =
                etNextOut[0..<batchSize, 0..<startHeight, 0..<startWidth, 0..<channels]
              etNextCond[0..<batchSize, 0..<startHeight, 0..<startWidth, 0..<channels] =
                etNextOut[
                  batchSize..<(batchSize * 2), 0..<startHeight, 0..<startWidth, 0..<channels]
              if let blur = blur {
                let etNextCondDegraded = blur(inputs: etNextOut)[0].as(of: FloatType.self)
                etNextCond = Functional.add(
                  left: etNextCondDegraded, right: etNextCond, leftScalar: alpha,
                  rightScalar: 1 - alpha)
              }
              if modifier == .editing {
                var etNextAllUncond = graph.variable(
                  .GPU(0), .NHWC(batchSize, startHeight, startWidth, channels), of: FloatType.self)
                etNextAllUncond[0..<batchSize, 0..<startHeight, 0..<startWidth, 0..<channels] =
                  etNextOut[
                    (batchSize * 2)..<(batchSize * 3), 0..<startHeight, 0..<startWidth, 0..<channels
                  ]
                etNext =
                  etNextAllUncond + textGuidanceScale * (etNextCond - etNextUncond)
                  + imageGuidanceScale
                  * (etNextUncond - etNextAllUncond)
              } else {
                etNext = etNextUncond + textGuidanceScale * (etNextCond - etNextUncond)
              }
            } else {
              if let blur = blur {
                let etNextOutDegraded = blur(inputs: etNextOut)[0].as(of: FloatType.self)
                etNextOut = Functional.add(
                  left: etNextOutDegraded, right: etNextOut, leftScalar: alpha,
                  rightScalar: 1 - alpha)
              }
              etNext = etNextOut
            }
          }
          switch discretization.objective {
          case .v:
            etNext =
              Float(alphaPrev.squareRoot()) * etNext + Float((1 - alphaPrev).squareRoot()) * xPrev
          case .epsilon:
            break
          case .edm(_):
            break
          }
          etPrime = 0.5 * (et + etNext)
        case 1:
          etPrime = 0.5 * (3 * et - oldEps[0])
        case 2:
          etPrime =
            Float(1) / Float(12) * (Float(23) * et - Float(16) * oldEps[1] + Float(5) * oldEps[0])
        case 3:
          etPrime =
            Float(1) / Float(24)
            * (Float(55) * et - Float(59) * oldEps[2] + Float(37) * oldEps[1] - Float(9) * oldEps[0])
        default:
          fatalError()
        }
        let (xPrev, _) = xPrevAndPredX0(x: x, et: etPrime, alpha: alpha, alphaPrev: alphaPrev)
        x = xPrev
        oldEps.append(et)
        if oldEps.count > 3 {
          oldEps.removeFirst()
        }
        if i < endStep - 1, let noise = noise, let sample = sample, let mask = mask,
          let negMask = negMask
        {
          noise.randn(std: 1, mean: 0)
          let qSample =
            Float(alphaPrev.squareRoot()) * sample + Float((1 - alphaPrev).squareRoot()) * noise
          x = qSample .* negMask + x .* mask
        }
        if i == endStep - 1 {
          if isNaN(x.rawValue.toCPU()) {
            return .failure(SamplerError.isNaN)
          }
        }
      }
      return .success(SamplerOutput(x: x, unets: unets))
    }
    streamContext.joined()
    return result
  }

  public func timestep(for strength: Float, sampling: Sampling) -> (
    timestep: Float, startStep: Float, roundedDownStartStep: Int, roundedUpStartStep: Int
  ) {
    let tEnc = Int(strength * Float(sampling.steps))
    let initTimestep =
      discretization.timesteps - discretization.timesteps
      / Float(sampling.steps * (sampling.steps - tEnc)) + 1
    let startStep = sampling.steps - tEnc
    return (
      timestep: initTimestep, startStep: Float(startStep), roundedDownStartStep: startStep,
      roundedUpStartStep: startStep
    )
  }

  public func sampleScaleFactor(at step: Float, sampling: Sampling) -> Float {
    let step = Int(step.rounded())
    let alphasCumprod = discretization.alphasCumprod(steps: sampling.steps, shift: sampling.shift)
    let alphaCumprod = alphasCumprod[max(0, min(alphasCumprod.count - 1, step))]
    return Float(alphaCumprod.squareRoot())
  }

  public func noiseScaleFactor(at step: Float, sampling: Sampling) -> Float {
    let step = Int(step.rounded())
    let alphasCumprod = discretization.alphasCumprod(steps: sampling.steps, shift: sampling.shift)
    let alphaCumprod = alphasCumprod[max(0, min(alphasCumprod.count - 1, step))]
    return Float((1 - alphaCumprod).squareRoot())
  }
}
