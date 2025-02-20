import Foundation
import NNC

public struct TCDSampler<
  FloatType: TensorNumeric & BinaryFloatingPoint, UNet: UNetProtocol,
  Discretization: Denoiser.Discretization
>
where UNet.FloatType == FloatType {
  public let filePath: String
  public let modifier: SamplerModifier
  public let version: ModelVersion
  public let qkNorm: Bool
  public let dualAttentionLayers: [Int]
  public let usesFlashAttention: Bool
  public let upcastAttention: Bool
  public let externalOnDemand: Bool
  public let injectControls: Bool
  public let injectT2IAdapters: Bool
  public let injectAttentionKV: Bool

  public let injectIPAdapterLengths: [Int]
  public let lora: [LoRAConfiguration]
  public let isGuidanceEmbedEnabled: Bool
  public let isQuantizedModel: Bool
  public let canRunLoRASeparately: Bool
  public let stochasticSamplingGamma: Float
  public let conditioning: Denoiser.Conditioning
  public let tiledDiffusion: TiledConfiguration
  private let discretization: Discretization
  public init(
    filePath: String, modifier: SamplerModifier, version: ModelVersion, qkNorm: Bool,
    dualAttentionLayers: [Int], usesFlashAttention: Bool,
    upcastAttention: Bool, externalOnDemand: Bool, injectControls: Bool,
    injectT2IAdapters: Bool, injectAttentionKV: Bool, injectIPAdapterLengths: [Int],
    lora: [LoRAConfiguration],
    isGuidanceEmbedEnabled: Bool, isQuantizedModel: Bool, canRunLoRASeparately: Bool,
    stochasticSamplingGamma: Float, conditioning: Denoiser.Conditioning,
    tiledDiffusion: TiledConfiguration, discretization: Discretization
  ) {
    self.filePath = filePath
    self.modifier = modifier
    self.version = version
    self.qkNorm = qkNorm
    self.dualAttentionLayers = dualAttentionLayers
    self.usesFlashAttention = usesFlashAttention
    self.upcastAttention = upcastAttention
    self.externalOnDemand = externalOnDemand
    self.injectControls = injectControls
    self.injectT2IAdapters = injectT2IAdapters
    self.injectAttentionKV = injectAttentionKV

    self.injectIPAdapterLengths = injectIPAdapterLengths
    self.lora = lora
    self.isGuidanceEmbedEnabled = isGuidanceEmbedEnabled
    self.isQuantizedModel = isQuantizedModel
    self.canRunLoRASeparately = canRunLoRASeparately
    self.stochasticSamplingGamma = stochasticSamplingGamma
    self.conditioning = conditioning
    self.tiledDiffusion = tiledDiffusion
    self.discretization = discretization
  }
}

extension TCDSampler: Sampler {
  public func sample(
    _ x_T: DynamicGraph.Tensor<FloatType>, unets existingUNets: [UNet?],
    sample: DynamicGraph.Tensor<FloatType>?, conditionImage: DynamicGraph.Tensor<FloatType>?,
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
  ) -> Result<SamplerOutput<FloatType, UNet>, Error> {
    guard endStep.integral > startStep.integral else {
      return .success(SamplerOutput(x: x_T, unets: [nil]))
    }
    var x = x_T
    let batchSize = x.shape[0]
    let startHeight = x.shape[1]
    let startWidth = x.shape[2]
    let channels = x.shape[3]
    let graph = x.graph
    let inChannels: Int
    if version == .svdI2v {
      inChannels = channels * 2
    } else {
      switch modifier {
      case .inpainting, .depth, .canny:
        inChannels = channels + (conditionImage?.shape[3] ?? 0)
      case .editing:
        inChannels = channels * 2
      case .double:
        inChannels = channels * 2
      case .none:
        inChannels = channels
      }
    }
    let zeroNegativePrompt = false
    var xIn = graph.variable(
      .GPU(0), .NHWC(batchSize, startHeight, startWidth, inChannels),
      of: FloatType.self
    )
    switch modifier {
    case .inpainting, .depth, .canny:
      let maskedImage = conditionImage!
      let shape = maskedImage.shape
      for i in 0..<batchSize {
        xIn[i..<(i + 1), 0..<startHeight, 0..<startWidth, channels..<(channels + shape[3])] =
          maskedImage
      }
    case .editing:
      let maskedImage = conditionImage!
      for i in 0..<batchSize {
        xIn[i..<(i + 1), 0..<startHeight, 0..<startWidth, channels..<(channels * 2)] = maskedImage
      }
    case .double:
      let maskedImage = conditionImage!
      let maskedImageChannels = maskedImage.shape[3]
      for i in 0..<batchSize {
        xIn[
          i..<(i + 1), 0..<startHeight, 0..<startWidth, channels..<(channels + maskedImageChannels)] =
          maskedImage
      }
    case .none:
      break
    }
    var c = c
    var extraProjection = extraProjection
    // There is no tokenLengthUncond any more.
    let tokenLengthUncond = tokenLengthCond
    if version != .svdI2v {
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
    }
    let oldC = c
    let fixedEncoder = UNetFixedEncoder<FloatType>(
      filePath: filePath, version: version, dualAttentionLayers: dualAttentionLayers,
      usesFlashAttention: usesFlashAttention,
      zeroNegativePrompt: zeroNegativePrompt, isQuantizedModel: isQuantizedModel,
      canRunLoRASeparately: canRunLoRASeparately, externalOnDemand: externalOnDemand)
    let injectedControlsC: [[DynamicGraph.Tensor<FloatType>]]
    let alphasCumprod = discretization.alphasCumprod(
      steps: sampling.steps + 1, shift: sampling.shift)
    let sigmas = alphasCumprod.map { discretization.sigma(from: $0) }
    let timesteps = (startStep.integral..<endStep.integral).map {
      let alphaCumprod: Double
      if $0 == startStep.integral && Float(startStep.integral) != startStep.fractional {
        let lowTimestep = discretization.timestep(
          for: alphasCumprod[max(0, min(Int(startStep.integral), alphasCumprod.count - 1))])
        let highTimestep = discretization.timestep(
          for: alphasCumprod[
            max(0, min(Int(startStep.fractional.rounded(.up)), alphasCumprod.count - 1))])
        let timestep =
          lowTimestep
          + Float(highTimestep - lowTimestep) * (startStep.fractional - Float(startStep.integral))
        alphaCumprod = discretization.alphaCumprod(timestep: timestep, shift: 1)
      } else {
        alphaCumprod = discretization.alphaCumprod(from: sigmas[$0])
      }
      switch conditioning {
      case .noise:
        return discretization.noise(for: alphaCumprod)
      case .timestep:
        return discretization.timestep(for: alphaCumprod)
      }
    }
    if UNetFixedEncoder<FloatType>.isFixedEncoderRequired(version: version) {
      let vector = fixedEncoder.vector(
        textEmbedding: c[c.count - 1], originalSize: originalSize,
        cropTopLeft: cropTopLeft,
        targetSize: targetSize, aestheticScore: aestheticScore,
        negativeOriginalSize: negativeOriginalSize, negativeAestheticScore: negativeAestheticScore,
        fpsId: fpsId, motionBucketId: motionBucketId, condAug: condAug)
      let (encodings, weightMapper) = fixedEncoder.encode(
        isCfgEnabled: false, textGuidanceScale: textGuidanceScale, guidanceEmbed: guidanceEmbed,
        isGuidanceEmbedEnabled: isGuidanceEmbedEnabled,
        textEncoding: c, timesteps: timesteps, batchSize: batchSize, startHeight: startHeight,
        startWidth: startWidth,
        tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond, lora: lora,
        tiledDiffusion: tiledDiffusion, injectedControls: injectedControls)
      c = vector + encodings
      injectedControlsC = injectedControls.map {
        $0.model.encode(
          isCfgEnabled: false, textGuidanceScale: textGuidanceScale, guidanceEmbed: guidanceEmbed,
          isGuidanceEmbedEnabled: isGuidanceEmbedEnabled, textEncoding: oldC, timesteps: timesteps,
          vector: vector.first, batchSize: batchSize, startHeight: startHeight,
          startWidth: startWidth, tokenLengthUncond: tokenLengthUncond,
          tokenLengthCond: tokenLengthCond, zeroNegativePrompt: zeroNegativePrompt,
          mainUNetFixed: (fixedEncoder.filePath, weightMapper))
      }
    } else {
      injectedControlsC = injectedControls.map {
        $0.model.encode(
          isCfgEnabled: false, textGuidanceScale: textGuidanceScale, guidanceEmbed: guidanceEmbed,
          isGuidanceEmbedEnabled: isGuidanceEmbedEnabled, textEncoding: oldC, timesteps: timesteps,
          vector: nil, batchSize: batchSize, startHeight: startHeight,
          startWidth: startWidth, tokenLengthUncond: tokenLengthUncond,
          tokenLengthCond: tokenLengthCond, zeroNegativePrompt: zeroNegativePrompt,
          mainUNetFixed: (fixedEncoder.filePath, nil))
      }
    }
    var unet = existingUNets[0] ?? UNet()
    defer {
      cancellation {}  // In this way, we are not holding the graph any more.
    }
    cancellation {
      unet.cancel()
    }
    var controlNets = [Model?](repeating: nil, count: injectedControls.count)
    let injectControlsAndAdapters = InjectControlsAndAdapters(
      injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
      injectAttentionKV: injectAttentionKV, injectIPAdapterLengths: injectIPAdapterLengths,
      injectControlModels: injectedControls.map { $0.model })
    if existingUNets[0] == nil {
      let firstTimestep =
        discretization.timesteps - discretization.timesteps / Float(sampling.steps) + 1
      let t = unet.timeEmbed(
        graph: graph, batchSize: batchSize, timestep: firstTimestep, version: version)
      let emptyInjectedControlsAndAdapters =
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
        filePath: filePath, externalOnDemand: externalOnDemand, version: version, qkNorm: qkNorm,
        dualAttentionLayers: dualAttentionLayers,
        upcastAttention: upcastAttention, usesFlashAttention: usesFlashAttention,
        injectControlsAndAdapters: injectControlsAndAdapters, lora: lora,
        isQuantizedModel: isQuantizedModel, canRunLoRASeparately: canRunLoRASeparately,
        inputs: xIn, t,
        UNetExtractConditions(
          of: FloatType.self,
          graph: graph, index: 0, batchSize: batchSize, tokenLengthUncond: tokenLengthUncond,
          tokenLengthCond: tokenLengthCond, conditions: newC, version: version, isCfgEnabled: false),
        tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
        isCfgEnabled: false, extraProjection: extraProjection,
        injectedControlsAndAdapters: emptyInjectedControlsAndAdapters,
        tiledDiffusion: tiledDiffusion)
    }
    let noise: DynamicGraph.Tensor<FloatType> = graph.variable(
      .GPU(0), .NHWC(batchSize, startHeight, startWidth, channels))
    let streamContext = StreamContext(.GPU(0))
    var refinerKickIn = refiner.map { (1 - $0.start) * discretization.timesteps } ?? -1
    var unets: [UNet?] = [unet]
    let blur: Model?
    if sharpness > 0 {
      blur = Blur(filters: channels, sigma: 3.0, size: 13, input: x)
    } else {
      blur = nil
    }
    var currentModelVersion = version
    var indexOffset = startStep.integral
    let result: Result<SamplerOutput<FloatType, UNet>, Error> = graph.withStream(streamContext) {
      if version == .svdI2v {
        let maskedImage = conditionImage!
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
        xIn[0..<batchSize, 0..<startHeight, 0..<startWidth, channels..<(channels * 2)] = frames
      }
      var oldDenoised: DynamicGraph.Tensor<FloatType>? = nil
      for i in startStep.integral..<endStep.integral {
        let sigma: Double
        if i == startStep.integral && Float(startStep.integral) != startStep.fractional {
          let lowTimestep = discretization.timestep(
            for: alphasCumprod[max(0, min(Int(startStep.integral), alphasCumprod.count - 1))])
          let highTimestep = discretization.timestep(
            for: alphasCumprod[
              max(0, min(Int(startStep.fractional.rounded(.up)), alphasCumprod.count - 1))])
          let timestep =
            lowTimestep
            + Float(highTimestep - lowTimestep) * (startStep.fractional - Float(startStep.integral))
          let alphaCumprod = discretization.alphaCumprod(timestep: timestep, shift: 1)
          sigma = discretization.sigma(from: alphaCumprod)
        } else {
          sigma = sigmas[i]
        }
        let alphaCumprod = discretization.alphaCumprod(from: sigma)
        let rawValue: Tensor<FloatType>? =
          (i > max(startStep.integral, sampling.steps / 2) || i % 2 == 1)
          ? (oldDenoised.map { unet.decode($0) })?.rawValue.toCPU() : nil
        if i % 5 == 4, let rawValue = rawValue {
          if isNaN(rawValue) {
            return .failure(SamplerError.isNaN)
          }
        }
        guard feedback(i - startStep.integral, rawValue) else {
          return .failure(SamplerError.cancelled)
        }
        let timestep = discretization.timestep(for: alphaCumprod)
        if Float(timestep) < refinerKickIn, let refiner = refiner {
          let timesteps = (i..<endStep.integral).map {
            let alphaCumprod =
              $0 == i ? alphaCumprod : discretization.alphaCumprod(from: sigmas[$0])
            switch conditioning {
            case .noise:
              return discretization.noise(for: alphaCumprod)
            case .timestep:
              return discretization.timestep(for: alphaCumprod)
            }
          }
          unets = [nil]
          let fixedEncoder = UNetFixedEncoder<FloatType>(
            filePath: refiner.filePath, version: refiner.version,
            dualAttentionLayers: dualAttentionLayers,
            usesFlashAttention: usesFlashAttention, zeroNegativePrompt: zeroNegativePrompt,
            isQuantizedModel: isQuantizedModel, canRunLoRASeparately: canRunLoRASeparately,
            externalOnDemand: externalOnDemand)
          if UNetFixedEncoder<FloatType>.isFixedEncoderRequired(version: refiner.version) {
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
                isCfgEnabled: false, textGuidanceScale: textGuidanceScale,
                guidanceEmbed: guidanceEmbed, isGuidanceEmbedEnabled: isGuidanceEmbedEnabled,
                textEncoding: oldC, timesteps: timesteps, batchSize: batchSize,
                startHeight: startHeight,
                startWidth: startWidth, tokenLengthUncond: tokenLengthUncond,
                tokenLengthCond: tokenLengthCond, lora: lora, tiledDiffusion: tiledDiffusion,
                injectedControls: injectedControls
              ).0
            indexOffset = i
          }
          unet = UNet()
          cancellation {
            unet.cancel()
          }
          currentModelVersion = refiner.version
          let firstTimestep =
            discretization.timesteps - discretization.timesteps / Float(sampling.steps) + 1
          let t = unet.timeEmbed(
            graph: graph, batchSize: batchSize, timestep: Float(firstTimestep),
            version: currentModelVersion)
          let emptyInjectedControlsAndAdapters =
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
            version: refiner.version, qkNorm: qkNorm, dualAttentionLayers: dualAttentionLayers,
            upcastAttention: upcastAttention,
            usesFlashAttention: usesFlashAttention,
            injectControlsAndAdapters: injectControlsAndAdapters,
            lora: lora, isQuantizedModel: refiner.isQuantizedModel,
            canRunLoRASeparately: canRunLoRASeparately,
            inputs: xIn, t,
            UNetExtractConditions(
              of: FloatType.self,
              graph: graph, index: 0, batchSize: batchSize, tokenLengthUncond: tokenLengthUncond,
              tokenLengthCond: tokenLengthCond, conditions: newC,
              version: currentModelVersion, isCfgEnabled: false),
            tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
            isCfgEnabled: false, extraProjection: extraProjection,
            injectedControlsAndAdapters: emptyInjectedControlsAndAdapters,
            tiledDiffusion: tiledDiffusion)
          refinerKickIn = -1
          unets.append(unet)
        }
        let cNoise: Float
        switch conditioning {
        case .noise:
          cNoise = discretization.noise(for: alphaCumprod)
        case .timestep:
          cNoise = timestep
        }
        let t = unet.timeEmbed(
          graph: graph, batchSize: batchSize, timestep: cNoise, version: currentModelVersion)
        let c = UNetExtractConditions(
          of: FloatType.self,
          graph: graph, index: i - indexOffset, batchSize: batchSize,
          tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond, conditions: c,
          version: currentModelVersion, isCfgEnabled: false)
        xIn[0..<batchSize, 0..<startHeight, 0..<startWidth, 0..<channels] = x
        let injectedIPAdapters = ControlModel<FloatType>
          .injectedIPAdapters(
            injecteds: injectedControls, step: i, version: unet.version,
            usesFlashAttention: usesFlashAttention, inputs: xIn, t, injectedControlsC,
            tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
            isCfgEnabled: false, index: i - startStep.integral,
            mainUNetAndWeightMapper: unet.modelAndWeightMapper,
            controlNets: &controlNets)
        let injectedControlsAndAdapters = ControlModel<FloatType>
          .injectedControlsAndAdapters(
            injecteds: injectedControls, step: i, version: unet.version,
            usesFlashAttention: usesFlashAttention, inputs: xIn, t, injectedControlsC,
            tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
            isCfgEnabled: false, index: i - startStep.integral,
            mainUNetAndWeightMapper: unet.modelAndWeightMapper,
            controlNets: &controlNets)
        let newC: [DynamicGraph.Tensor<FloatType>]
        if version == .svdI2v {
          newC = Array(c[0..<(1 + (c.count - 1) / 2)])
        } else {
          newC = c
        }
        var etOut = unet(
          timestep: cNoise, inputs: xIn, t, newC, extraProjection: extraProjection,
          injectedControlsAndAdapters: injectedControlsAndAdapters,
          injectedIPAdapters: injectedIPAdapters, tokenLengthUncond: tokenLengthUncond,
          tokenLengthCond: tokenLengthCond, isCfgEnabled: false, tiledDiffusion: tiledDiffusion,
          controlNets: &controlNets)
        if channels < etOut.shape[3] {
          etOut = etOut[0..<batchSize, 0..<startHeight, 0..<startWidth, 0..<channels].copied()
        }
        if let blur = blur {
          let alpha =
            0.001 * sharpness * (discretization.timesteps - Float(timestep))
            / discretization.timesteps
          let etOutDegraded = blur(inputs: etOut)[0].as(of: FloatType.self)
          etOut = Functional.add(
            left: etOutDegraded, right: etOut, leftScalar: alpha, rightScalar: 1 - alpha)
        }
        let predictOriginalSample: DynamicGraph.Tensor<FloatType>
        let predictNoisedSample: DynamicGraph.Tensor<FloatType>
        let alphaPrev = alphasCumprod[i + 1]
        let timestepPrev = discretization.timestep(for: alphaPrev)
        let timestepS = (1 - stochasticSamplingGamma) * timestepPrev
        let alphaCumprodS = discretization.alphaCumprod(timestep: timestepS, shift: sampling.shift)
        switch discretization.objective {
        case .u(_):
          let sigmaS = discretization.sigma(from: alphaCumprodS)
          predictOriginalSample = Functional.add(
            left: x, right: etOut, leftScalar: 1, rightScalar: Float(-sigma))
          predictNoisedSample = Functional.add(
            left: x, right: predictOriginalSample,
            leftScalar: Float(sigmaS / sigma),
            rightScalar: Float(1 - sigmaS / sigma))
        case .v:
          let sqrtAlphaCumprod = 1.0 / (sigma * sigma + 1).squareRoot()
          predictOriginalSample = Functional.add(
            left: x, right: etOut, leftScalar: Float(sqrtAlphaCumprod),
            rightScalar: Float(-sigma * sqrtAlphaCumprod))
          let predictEpsilon = Functional.add(
            left: x, right: etOut, leftScalar: Float((1 - alphaCumprod).squareRoot()),
            rightScalar: Float(sqrtAlphaCumprod))
          predictNoisedSample = Functional.add(
            left: predictOriginalSample, right: predictEpsilon,
            leftScalar: Float(alphaCumprodS.squareRoot()),
            rightScalar: Float((1 - alphaCumprodS).squareRoot()))
        case .epsilon:
          predictOriginalSample = Functional.add(
            left: x, right: etOut, leftScalar: Float(1.0 / alphaCumprod.squareRoot()),
            rightScalar: Float(-sigma))
          predictNoisedSample = Functional.add(
            left: predictOriginalSample, right: etOut,
            leftScalar: Float(alphaCumprodS.squareRoot()),
            rightScalar: Float((1 - alphaCumprodS).squareRoot()))
        case .edm(let sigmaData):
          let sigmaData2 = sigmaData * sigmaData
          predictOriginalSample = Functional.add(
            left: x, right: etOut,
            leftScalar: Float(sigmaData2 / (sigma * sigma + sigmaData2).squareRoot()),
            rightScalar: Float(sigma * sigmaData / (sigma * sigma + sigmaData2).squareRoot()))
          predictNoisedSample = Functional.add(
            left: predictOriginalSample, right: etOut,
            leftScalar: Float(alphaCumprodS.squareRoot()),
            rightScalar: Float((1 - alphaCumprodS).squareRoot()))
        }
        oldDenoised = predictOriginalSample
        if i < sampling.steps - 1 {
          if stochasticSamplingGamma > 0 {
            noise.randn(std: 1, mean: 0)
            if case .u(_) = discretization.objective {
              x = Functional.add(
                left: predictNoisedSample, right: predictOriginalSample, leftScalar: 1,
                rightScalar: Float(alphaCumprodS - alphaPrev))
              let sigmaPrev = 1 - alphaPrev
              let sigmaS = 1 - alphaCumprodS
              x = Functional.add(
                left: x, right: noise, leftScalar: 1,
                rightScalar: Float((sigmaPrev * sigmaPrev - sigmaS * sigmaS).squareRoot()))
            } else {
              x = Functional.add(
                left: predictNoisedSample, right: noise,
                leftScalar: Float((alphaPrev / alphaCumprodS).squareRoot()),
                rightScalar: Float((1 - alphaPrev / alphaCumprodS).squareRoot()))
            }
          } else {
            x = predictNoisedSample
          }
        } else {
          x = predictOriginalSample
        }
        if i < endStep.integral - 1, let sample = sample, let mask = mask,
          let negMask = negMask
        {
          noise.randn(std: 1, mean: 0)
          let qSample: DynamicGraph.Tensor<FloatType>
          if case .u(_) = discretization.objective {
            qSample = Functional.add(
              left: sample, right: noise, leftScalar: Float(alphaPrev),
              rightScalar: Float(1 - alphaPrev))
          } else {
            qSample =
              Float(alphaPrev.squareRoot()) * sample + Float((1 - alphaPrev).squareRoot()) * noise
          }
          x = qSample .* negMask + x .* mask
        }
        if i == endStep.integral - 1 {
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
    let tEnc = strength * discretization.timesteps
    let initTimestep = tEnc
    let alphasCumprod = discretization.alphasCumprod(steps: sampling.steps, shift: sampling.shift)
    var previousTimestep = discretization.timesteps
    for (i, alphaCumprod) in alphasCumprod.enumerated() {
      let timestep = discretization.timestep(for: alphaCumprod)
      if initTimestep >= timestep {
        guard i > 0 else {
          return (
            timestep: timestep, startStep: 0, roundedDownStartStep: 0, roundedUpStartStep: 0
          )
        }
        guard initTimestep > timestep + 1e-3 else {
          return (
            timestep: initTimestep, startStep: Float(i), roundedDownStartStep: i,
            roundedUpStartStep: i
          )
        }
        return (
          timestep: Float(initTimestep),
          startStep: Float(i - 1) + Float(initTimestep - previousTimestep)
            / Float(timestep - previousTimestep), roundedDownStartStep: i - 1, roundedUpStartStep: i
        )
      }
      previousTimestep = timestep
    }
    return (
      timestep: discretization.timestep(for: alphasCumprod[0]),
      startStep: Float(alphasCumprod.count - 1),
      roundedDownStartStep: alphasCumprod.count - 1, roundedUpStartStep: alphasCumprod.count - 1
    )
  }

  public func sampleScaleFactor(at step: Float, sampling: Sampling) -> Float {
    let step = Int(step.rounded())
    let alphasCumprod = discretization.alphasCumprod(steps: sampling.steps, shift: sampling.shift)
    let alphaCumprod = alphasCumprod[max(0, min(alphasCumprod.count - 1, step))]
    if case .u(_) = discretization.objective {
      return Float(alphaCumprod)
    } else {
      return Float(alphaCumprod.squareRoot())
    }
  }

  public func noiseScaleFactor(at step: Float, sampling: Sampling) -> Float {
    let step = Int(step.rounded())
    let alphasCumprod = discretization.alphasCumprod(steps: sampling.steps, shift: sampling.shift)
    let alphaCumprod = alphasCumprod[max(0, min(alphasCumprod.count - 1, step))]
    if case .u(_) = discretization.objective {
      return Float(1 - alphaCumprod)
    } else {
      return Float((1 - alphaCumprod).squareRoot())
    }
  }
}
