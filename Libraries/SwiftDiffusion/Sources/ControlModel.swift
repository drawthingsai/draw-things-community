import C_ccv
import NNC

public enum ControlHintType: String, Codable & CaseIterable {
  case custom
  case depth
  case canny
  case scribble
  case pose
  case normalbae
  case color
  case lineart
  case softedge
  case seg
  case inpaint
  case ip2p
  case shuffle
  case mlsd
  case tile
}

public enum ControlType: String, Codable {
  case controlnet
  case t2iadapter
  case ipadapterplus
  case ipadapterfull
  case controlnetlora
  case injectKV = "inject_kv"
}

public enum ControlMode {
  case balanced
  case prompt
  case control
}

public struct IPAdapterConfig: Codable {
  public var inputDim: Int
  public var queryDim: Int
  public var outputDim: Int
  public var headDim: Int
  public var numHeads: Int
  public var grid: Int
  public init(inputDim: Int, queryDim: Int, outputDim: Int, headDim: Int, numHeads: Int, grid: Int)
  {
    self.inputDim = inputDim
    self.queryDim = queryDim
    self.outputDim = outputDim
    self.headDim = headDim
    self.numHeads = numHeads
    self.grid = grid
  }
}

public struct ControlModel<FloatType: TensorNumeric & BinaryFloatingPoint> {
  public let filePaths: [String]
  public let type: ControlType
  public let modifier: ControlHintType
  public let externalOnDemand: Bool
  public let version: ModelVersion
  public let tiledDiffusion: TiledConfiguration
  public let usesFlashAttention: Bool
  public let startStep: Int
  public let endStep: Int
  public let controlMode: ControlMode
  public let globalAveragePooling: Bool
  public let transformerBlocks: [Int]
  public let targetBlocks: [String]
  public let imageEncoderVersion: ImageEncoderVersion
  public let ipAdapterConfig: IPAdapterConfig?
  public init(
    filePaths: [String], type: ControlType, modifier: ControlHintType,
    externalOnDemand: Bool, version: ModelVersion, tiledDiffusion: TiledConfiguration,
    usesFlashAttention: Bool, startStep: Int, endStep: Int, controlMode: ControlMode,
    globalAveragePooling: Bool, transformerBlocks: [Int], targetBlocks: [String],
    imageEncoderVersion: ImageEncoderVersion, ipAdapterConfig: IPAdapterConfig?
  ) {
    self.filePaths = filePaths
    self.type = type
    self.modifier = modifier
    self.externalOnDemand = externalOnDemand
    self.version = version
    self.tiledDiffusion = tiledDiffusion
    self.usesFlashAttention = usesFlashAttention
    self.startStep = startStep
    self.endStep = endStep
    self.controlMode = controlMode
    self.globalAveragePooling = globalAveragePooling
    self.transformerBlocks = transformerBlocks
    self.targetBlocks = targetBlocks
    self.imageEncoderVersion = imageEncoderVersion
    self.ipAdapterConfig = ipAdapterConfig
  }
}

extension ControlModel {
  static func injectedIPAdapters(
    injecteds: [(
      model: ControlModel<FloatType>, hints: [([DynamicGraph.Tensor<FloatType>], Float)]
    )],
    step: Int, version: ModelVersion, usesFlashAttention: Bool,
    inputs xT: DynamicGraph.Tensor<FloatType>,
    _ timestep: DynamicGraph.Tensor<FloatType>?, _ c: [[DynamicGraph.Tensor<FloatType>]],
    tokenLengthUncond: Int, tokenLengthCond: Int, isCfgEnabled: Bool,
    mainUNetAndWeightMapper: (Model, ModelWeightMapper)?,
    controlNets existingControlNets: inout [Model?]
  ) -> (
    [DynamicGraph.Tensor<FloatType>]
  ) {
    var injectedIPAdapters = [DynamicGraph.Tensor<FloatType>]()
    for (i, injected) in injecteds.enumerated() {
      guard injected.model.version == version else { continue }
      switch injected.model.type {
      case .controlnet, .controlnetlora, .t2iadapter, .injectKV:
        continue
      case .ipadapterplus, .ipadapterfull:
        var instanceInjectedIPAdapters = [DynamicGraph.Tensor<FloatType>]()
        for (hint, strength) in injected.hints {
          let newInjectedIPAdapters = injected.model(
            inputStartYPad: 0, inputEndYPad: 0, inputStartXPad: 0, inputEndXPad: 0,
            step: step, inputs: xT, hint, strength: strength, timestep, c[i],
            tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
            isCfgEnabled: isCfgEnabled, mainUNetAndWeightMapper: mainUNetAndWeightMapper,
            controlNet: &existingControlNets[i])
          if instanceInjectedIPAdapters.isEmpty {
            instanceInjectedIPAdapters = newInjectedIPAdapters
          } else {
            instanceInjectedIPAdapters = zip(instanceInjectedIPAdapters, newInjectedIPAdapters).map
            {
              if usesFlashAttention {
                return Concat(axis: 1)($0, $1)
              } else {
                return Concat(axis: 2)($0, $1)
              }
            }
          }
        }
        injectedIPAdapters.append(contentsOf: instanceInjectedIPAdapters)
      }
    }
    return injectedIPAdapters
  }

  public static func injectedControlsAndAdapters(
    injecteds: [(
      model: ControlModel<FloatType>, hints: [([DynamicGraph.Tensor<FloatType>], Float)]
    )],
    step: Int, version: ModelVersion, usesFlashAttention: Bool,
    inputs xT: DynamicGraph.Tensor<FloatType>,
    _ timestep: DynamicGraph.Tensor<FloatType>?, _ c: [[DynamicGraph.Tensor<FloatType>]],
    tokenLengthUncond: Int, tokenLengthCond: Int, isCfgEnabled: Bool,
    mainUNetAndWeightMapper: (Model, ModelWeightMapper)?,
    controlNets existingControlNets: inout [Model?]
  ) -> (
    (
      _ xT: DynamicGraph.Tensor<FloatType>, _ inputStartYPad: Int, _ inputEndYPad: Int,
      _ inputStartXPad: Int, _ inputEndXPad: Int, _ existingControlNets: inout [Model?]
    ) -> (
      injectedControls: [DynamicGraph.Tensor<FloatType>],
      injectedT2IAdapters: [DynamicGraph.Tensor<FloatType>],
      injectedAttentionKVs: [DynamicGraph.Tensor<FloatType>]
    )
  ) {
    var injectedT2IAdapters = [DynamicGraph.Tensor<FloatType>]()
    var injectedKVs = [DynamicGraph.Tensor<FloatType>]()

    for (i, injected) in injecteds.enumerated() {
      guard injected.model.version == version else { continue }
      switch injected.model.type {
      case .controlnet, .controlnetlora, .ipadapterplus, .ipadapterfull:
        continue
      case .injectKV:
        for (hint, strength) in injected.hints {
          let newHint = injected.model(
            inputStartYPad: 0, inputEndYPad: 0, inputStartXPad: 0, inputEndXPad: 0,
            step: step, inputs: xT, hint, strength: strength, timestep, c[i],
            tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
            isCfgEnabled: isCfgEnabled, mainUNetAndWeightMapper: mainUNetAndWeightMapper,
            controlNet: &existingControlNets[i])
          if injectedKVs.isEmpty {
            injectedKVs = newHint
          } else {
            injectedKVs = zip(injectedKVs, newHint).map { $0 + $1 }
          }
        }
      case .t2iadapter:
        for (hint, strength) in injected.hints {
          let newInjectedT2IAdapters = injected.model(
            inputStartYPad: 0, inputEndYPad: 0, inputStartXPad: 0, inputEndXPad: 0,
            step: step, inputs: xT, hint, strength: strength, timestep, c[i],
            tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
            isCfgEnabled: isCfgEnabled, mainUNetAndWeightMapper: mainUNetAndWeightMapper,
            controlNet: &existingControlNets[i])
          if injectedT2IAdapters.isEmpty {
            injectedT2IAdapters = newInjectedT2IAdapters
          } else {
            injectedT2IAdapters = zip(injectedT2IAdapters, newInjectedT2IAdapters).map { $0 + $1 }
          }
        }
      }
    }
    return { xT, inputStartYPad, inputEndYPad, inputStartXPad, inputEndXPad, existingControlNets in
      var injectedControls = [DynamicGraph.Tensor<FloatType>]()
      for (i, injected) in injecteds.enumerated() {
        guard injected.model.version == version else { continue }
        switch injected.model.type {
        case .controlnet, .controlnetlora:
          for (hint, strength) in injected.hints {
            let newInjectedControls = injected.model(
              inputStartYPad: inputStartYPad, inputEndYPad: inputEndYPad,
              inputStartXPad: inputStartXPad, inputEndXPad: inputEndXPad,
              step: step, inputs: xT, hint, strength: strength, timestep, c[i],
              tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
              isCfgEnabled: isCfgEnabled, mainUNetAndWeightMapper: mainUNetAndWeightMapper,
              controlNet: &existingControlNets[i])
            if injectedControls.isEmpty {
              injectedControls = newInjectedControls
            } else {
              injectedControls = zip(injectedControls, newInjectedControls).map { $0 + $1 }
            }
          }
        case .ipadapterplus, .ipadapterfull, .t2iadapter, .injectKV:
          continue
        }
      }
      return (injectedControls, injectedT2IAdapters, injectedKVs)
    }
  }

  static func emptyInjectedControlsAndAdapters(
    injecteds: [(
      model: ControlModel<FloatType>, hints: [([DynamicGraph.Tensor<FloatType>], Float)]
    )],
    step: Int, version: ModelVersion, inputs xT: DynamicGraph.Tensor<FloatType>,
    tiledDiffusion: TiledConfiguration
  ) -> (
    [DynamicGraph.Tensor<FloatType>], [DynamicGraph.Tensor<FloatType>],
    [DynamicGraph.Tensor<FloatType>], [DynamicGraph.Tensor<FloatType>]
  ) {
    var injectedControls = [DynamicGraph.Tensor<FloatType>]()
    var injectedT2IAdapters = [DynamicGraph.Tensor<FloatType>]()
    var injectedIPAdapters = [DynamicGraph.Tensor<FloatType>]()
    var injectedKVs = [DynamicGraph.Tensor<FloatType>]()
    let graph = xT.graph
    let batchSize = xT.shape[0]
    let startHeight = xT.shape[1]
    let startWidth = xT.shape[2]
    let tiledHeight: Int
    let tiledWidth: Int
    switch version {
    case .v1, .v2, .sdxlBase, .ssd1b, .sdxlRefiner, .svdI2v, .kandinsky21, .sd3, .pixart, .auraflow,
      .flux1:
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
    case .wurstchenStageC:
      tiledHeight = startHeight
      tiledWidth = startWidth
    case .wurstchenStageB:
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 16, startHeight) : startHeight
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 16, startWidth) : startWidth
    }
    for injected in injecteds {
      guard injected.model.version == version else { continue }
      switch injected.model.type {
      case .controlnet, .controlnetlora:
        if injectedControls.isEmpty {
          injectedControls = emptyControls(
            graph: graph, batchSize: batchSize, startWidth: tiledWidth, startHeight: tiledHeight,
            version: injected.model.version)
        }
      case .ipadapterplus:
        guard injected.hints.count > 0 else { continue }
        injectedIPAdapters += emptyIPAdapters(
          graph: graph, batchSize: batchSize, length: injected.hints.count, numTokens: 16,
          version: injected.model.version, usesFlashAttention: injected.model.usesFlashAttention)
      case .ipadapterfull:
        guard injected.hints.count > 0 else { continue }
        injectedIPAdapters += emptyIPAdapters(
          graph: graph, batchSize: batchSize, length: injected.hints.count, numTokens: 256,
          version: injected.model.version, usesFlashAttention: injected.model.usesFlashAttention)
      case .t2iadapter:
        if injectedT2IAdapters.isEmpty {
          injectedT2IAdapters = emptyAdapters(
            graph: graph, startWidth: tiledWidth, startHeight: tiledHeight)
        }
      case .injectKV:
        if injectedKVs.isEmpty {
          injectedKVs = emptyInjectKV(
            graph: graph, batchSize: batchSize, startWidth: tiledWidth, startHeight: tiledHeight)
        }
      }
    }
    return (injectedControls, injectedT2IAdapters, injectedIPAdapters, injectedKVs)
  }
}

extension ControlModel {
  private func zeroTensor(dataType: DataType, format: TensorFormat, shape: TensorShape) -> AnyTensor
  {
    switch dataType {
    case .Float16:
      #if !((os(macOS) || (os(iOS) && targetEnvironment(macCatalyst))) && (arch(i386) || arch(x86_64)))
        var tensor = Tensor<Float16>(.CPU, format: format, shape: shape)
        tensor.withUnsafeMutableBytes {
          let size = shape.reduce(MemoryLayout<Float16>.size, *)
          memset($0.baseAddress!, 0, size)
        }
        return tensor
      #else
        var tensor = Tensor<UInt16>(.CPU, format: format, shape: shape)
        tensor.withUnsafeMutableBytes {
          let size = shape.reduce(MemoryLayout<UInt16>.size, *)
          memset($0.baseAddress!, 0, size)
        }
        return tensor
      #endif
    case .Float32:
      var tensor = Tensor<Float32>(.CPU, format: format, shape: shape)
      tensor.withUnsafeMutableBytes {
        let size = shape.reduce(MemoryLayout<Float32>.size, *)
        memset($0.baseAddress!, 0, size)
      }
      return tensor
    case .Float64, .Int32, .Int64, .UInt8:
      fatalError()
    }
  }
  public func hint(inputs: [(hint: DynamicGraph.Tensor<FloatType>, weight: Float)])
    -> [[DynamicGraph.Tensor<FloatType>]]
  {
    guard inputs.count > 0 else {
      return []
    }
    let graph = inputs[0].hint.graph
    switch type {
    case .controlnet, .controlnetlora:
      let shape = inputs[0].hint.shape
      let startHeight = shape[1]
      let startWidth = shape[2]
      let tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 64, startHeight) : startHeight
      let tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 64, startWidth) : startWidth
      let tileOverlap = min(
        min(
          tiledDiffusion.tileOverlap * 64,
          Int((Double(tiledHeight / 3) / 64).rounded(.down)) * 64),
        Int((Double(tiledWidth / 3) / 64).rounded(.down)) * 64)
      let tiledDiffusionIsEnabled = (startWidth > tiledWidth) || (startHeight > tiledHeight)
      // For ControlNet, we only compute hint.
      let outputChannels = 320
      let hintNet = HintNet(channels: outputChannels).0
      if tiledDiffusionIsEnabled {
        hintNet.compile(
          inputs: inputs[0].hint[0..<shape[0], 0..<tiledHeight, 0..<tiledWidth, 0..<shape[3]])
      } else {
        hintNet.compile(inputs: inputs[0].hint)
      }
      graph.openStore(
        filePaths[0], flags: .readOnly,
        externalStore: TensorData.externalStore(filePath: filePaths[0])
      ) {
        $0.read("hintnet", model: hintNet, codec: [.ezm7, .q6p, .q8p, .jit, .externalData])
      }
      // Note that the Hint network for ControlNet is pretty lightweight, it has 7 convnet with 3x3 reception field. That means as long as we have 3 (scale 1) + 2 * 2 (scale 2) + 2 * 4 (scale 4) + 1 * 8 (scale 8) overlap (23 pixels), the result will exact match (also no group norm). Thus, when we do tile, we don't do the cross-fade but do direct copy.
      if tiledDiffusionIsEnabled {
        let yTiles =
          (startHeight - tileOverlap * 2 + (tiledHeight - tileOverlap * 2) - 1)
          / (tiledHeight - tileOverlap * 2)
        let xTiles =
          (startWidth - tileOverlap * 2 + (tiledWidth - tileOverlap * 2) - 1)
          / (tiledWidth - tileOverlap * 2)
        return inputs.map {
          let hint = $0.hint
          var result = graph.variable(
            hint.kind, .NHWC(shape[0], startHeight / 8, startWidth / 8, outputChannels),
            of: FloatType.self)
          for y in 0..<yTiles {
            let yOfs = y * (tiledHeight - tileOverlap * 2) + (y > 0 ? tileOverlap : 0)
            let (inputStartYPad, inputEndYPad) = paddedTileStartAndEnd(
              iOfs: yOfs, length: startHeight, tileSize: tiledHeight, tileOverlap: tileOverlap)
            let srcYStart: Int
            let srcYEnd: Int
            let dstYStart: Int
            let dstYEnd: Int
            if y == 0 {
              dstYStart = 0
              dstYEnd = yTiles == 1 ? inputEndYPad / 8 : (inputEndYPad - tileOverlap) / 8
              srcYStart = 0
              srcYEnd = dstYEnd
            } else if y == yTiles - 1 {
              dstYStart = (inputStartYPad + tileOverlap) / 8
              dstYEnd = inputEndYPad / 8
              srcYStart = tileOverlap / 8
              srcYEnd = tiledHeight / 8
            } else {
              dstYStart = (inputStartYPad + tileOverlap) / 8
              dstYEnd = (inputEndYPad - tileOverlap) / 8
              srcYStart = tileOverlap / 8
              srcYEnd = (tiledHeight - tileOverlap) / 8
            }
            for x in 0..<xTiles {
              let xOfs = x * (tiledWidth - tileOverlap * 2) + (x > 0 ? tileOverlap : 0)
              let (inputStartXPad, inputEndXPad) = paddedTileStartAndEnd(
                iOfs: xOfs, length: startWidth, tileSize: tiledWidth, tileOverlap: tileOverlap)
              let srcXStart: Int
              let srcXEnd: Int
              let dstXStart: Int
              let dstXEnd: Int
              if x == 0 {
                dstXStart = 0
                dstXEnd = xTiles == 1 ? inputEndXPad / 8 : (inputEndXPad - tileOverlap) / 8
                srcXStart = 0
                srcXEnd = dstXEnd
              } else if x == xTiles - 1 {
                dstXStart = (inputStartXPad + tileOverlap) / 8
                dstXEnd = inputEndXPad / 8
                srcXStart = tileOverlap / 8
                srcXEnd = tiledWidth / 8
              } else {
                dstXStart = (inputStartXPad + tileOverlap) / 8
                dstXEnd = (inputEndXPad - tileOverlap) / 8
                srcXStart = tileOverlap / 8
                srcXEnd = (tiledWidth - tileOverlap) / 8
              }
              let tiled = hintNet(
                inputs: hint[
                  0..<shape[0], inputStartYPad..<inputEndYPad, inputStartXPad..<inputEndXPad,
                  0..<shape[3]
                ].copied())[0].as(of: FloatType.self)
              result[0..<shape[0], dstYStart..<dstYEnd, dstXStart..<dstXEnd, 0..<outputChannels] =
                tiled[0..<shape[0], srcYStart..<srcYEnd, srcXStart..<srcXEnd, 0..<outputChannels]
            }
          }
          return [result]
        }
      } else {
        return inputs.map { hintNet(inputs: $0.hint).map { $0.as(of: FloatType.self) } }
      }
    case .injectKV:
      // TODO: this is not efficient as for many more samples, we will load the model many more times.
      var injectedKVs = inputs.map {
        let image = $0.hint

        // vae encoding
        let x = image
        let shape = x.shape
        let startHeight = shape[1] / 8
        let startWidth = shape[2] / 8
        let tiledWidth =
          tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
        let tiledHeight =
          tiledDiffusion.isEnabled
          ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
        let externalData: DynamicGraph.Store.Codec =
          externalOnDemand ? .externalOnDemand : .externalData
        let vaeEncoding = graph.withNoGrad {
          let encoder = Encoder(
            channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: tiledWidth,
            startHeight: tiledHeight, usesFlashAttention: false
          ).0
          let input: DynamicGraph.Tensor<FloatType>
          if shape[1] != tiledHeight * 8 || shape[2] != tiledWidth * 8 {
            input =
              Upsample(
                .bilinear, widthScale: Float(tiledWidth * 8) / Float(shape[2]),
                heightScale: Float(tiledHeight * 8) / Float(shape[1]))(x)
          } else {
            input = x
          }
          encoder.compile(inputs: input)
          let vaeFilePath = filePaths[2]
          graph.openStore(
            vaeFilePath, flags: .readOnly,
            externalStore: TensorData.externalStore(filePath: vaeFilePath)
          ) {
            $0.read("encoder", model: encoder, codec: [.jit, .q8p, .ezm7, externalData])
          }
          var vaeEncoding = encoder(inputs: input)[0].as(of: FloatType.self)
          vaeEncoding = vaeEncoding
          return vaeEncoding
        }

        // clip image encoding
        let inputHeight = image.shape[1]
        let inputWidth = image.shape[2]
        precondition(image.shape[3] == 3)
        let imageOut = graph.withNoGrad {
          let mean = graph.variable(
            Tensor<FloatType>(
              [
                FloatType(2 * 0.48145466 - 1), FloatType(2 * 0.4578275 - 1),
                FloatType(2 * 0.40821073 - 1),
              ], .GPU(0), .NHWC(1, 1, 1, 3)))
          let invStd = graph.variable(
            Tensor<FloatType>(
              [
                FloatType(0.5 / 0.26862954), FloatType(0.5 / 0.26130258),
                FloatType(0.5 / 0.27577711),
              ],
              .GPU(0), .NHWC(1, 1, 1, 3)))
          var imageTensorsGPU: DynamicGraph.Tensor<FloatType> = image.toGPU(0)
          if inputHeight != 224 || inputWidth != 224 {
            imageTensorsGPU =
              (Upsample(
                .bilinear, widthScale: Float(224) / Float(inputWidth),
                heightScale: Float(224) / Float(inputHeight))(imageTensorsGPU) - mean) .* invStd
          } else {
            imageTensorsGPU = (imageTensorsGPU - mean) .* invStd
          }
          let vit = CLIPVisionTransformer(
            FloatType.self, grid: 16, width: 1024, layers: 24, heads: 16, batchSize: 1
          )
          vit.compile(inputs: imageTensorsGPU)
          let visualProj = graph.variable(.GPU(0), .NC(1024, 768), of: FloatType.self)
          graph.openStore(
            filePaths[1], flags: .readOnly,
            externalStore: TensorData.externalStore(filePath: filePaths[1])
          ) {
            $0.read("vision_model", model: vit, codec: [externalData, .q8p, .ezm7])
            $0.read("visual_proj", variable: visualProj, codec: [.externalData, .q8p, .ezm7])
          }
          var imageOut = vit(inputs: imageTensorsGPU)[0].as(of: FloatType.self)
          imageOut = imageOut * visualProj
          return imageOut
        }
        let vaeShape = vaeEncoding.shape
        let vaeEmbedsTensor = vaeEncoding[0..<1, 0..<vaeShape[1], 0..<vaeShape[2], 0..<4]
          .contiguous()

        // preset clip text encoding
        var textEmbedings: DynamicGraph.Tensor<FloatType>? = nil
        graph.openStore(
          filePaths[0], externalStore: TensorData.externalStore(filePath: filePaths[0])
        ) {
          if let tensor = $0.read("text_embeds", codec: [.externalData, .q8p, .ezm7]) {
            textEmbedings = graph.variable(tensor as! Tensor<FloatType>).toGPU(0)
          }
        }
        let textEmbeds =
          textEmbedings ?? graph.variable(.GPU(0), .HWC(1, 1, 768), of: FloatType.self)

        var embeddingLength = textEmbeds.shape[1] + 1
        var promptEmbeds = graph.variable(
          .GPU(0), .HWC(2, embeddingLength, 768), of: FloatType.self)
        promptEmbeds[0..<1, 0..<textEmbeds.shape[1], 0..<768] = textEmbeds
        promptEmbeds[0..<1, textEmbeds.shape[1]..<embeddingLength, 0..<768] = imageOut
        promptEmbeds[1..<2, 0..<textEmbeds.shape[1], 0..<768] = textEmbeds
        promptEmbeds[1..<2, textEmbeds.shape[1]..<embeddingLength, 0..<768] = imageOut

        let channels = vaeEmbedsTensor.shape[3]
        let t = graph.variable(
          Tensor<FloatType>(
            from: timeEmbedding(
              timestep: 0, batchSize: 2, embeddingSize: 320, maxPeriod: 10_000
            )
          ).toGPU(0))

        var xIn = graph.variable(
          .GPU(0), .NHWC(2, tiledHeight, tiledWidth, channels),
          of: FloatType.self
        )
        xIn.full(0)
        xIn[1..<2, 0..<tiledHeight, 0..<tiledWidth, 0..<channels] = vaeEmbedsTensor
        let newC = [promptEmbeds]

        let garmUnet = GarmentUNet(
          batchSize: 2, embeddingLength: (embeddingLength, embeddingLength),
          startWidth: tiledWidth, startHeight: tiledHeight,
          usesFlashAttention: usesFlashAttention ? .scaleMerged : .none,
          injectControls: false, injectT2IAdapters: false,
          injectIPAdapterLengths: [], injectAttentionKV: false
        ).0

        var inputs = [DynamicGraph.Tensor<FloatType>]()
        inputs.append(t)
        inputs.append(contentsOf: newC)
        garmUnet.compile(inputs: [xIn] + inputs)
        graph.openStore(
          filePaths[0], flags: .readOnly,
          externalStore: TensorData.externalStore(filePath: filePaths[0])
        ) {
          $0.read("unet", model: garmUnet, codec: [.jit, .q8p, .ezm7, externalData])
        }

        return garmUnet(inputs: xIn, [t] + newC).map { $0.as(of: FloatType.self) }
      }
      injectedKVs = zip(inputs, injectedKVs).map {
        guard $0.weight != 1 else {
          return $1
        }
        let weight = $0.weight
        // This can be improved, it makes more sense to manipulate key / value. But this looks OK for now.
        // The reason to do this is to push the magnitude so it is more likely to attend these keys, rather
        // than averaging out the variance, which will be more likely to loss information.
        return $1.map {
          let mean = $0.reduced(.mean, axis: [1])
          return Functional.add(left: $0, right: mean, rightScalar: weight - 1)
        }
      }
      return injectedKVs
    case .ipadapterplus:
      let imageEncoder = ImageEncoder<FloatType>(
        filePath: filePaths[1], version: imageEncoderVersion)
      let imageSize: Int
      switch imageEncoderVersion {
      case .clipL14_336:
        imageSize = 336
      case .openClipH14:
        imageSize = 224
      }
      let zeroEmbeds = graph.variable(
        .GPU(0), .NHWC(1, imageSize, imageSize, 3), of: FloatType.self)
      zeroEmbeds.full(0)
      let imageEmbeds = imageEncoder.encode(inputs.map(\.hint) + [zeroEmbeds])
      let resampler: Model
      if let ipAdapterConfig = ipAdapterConfig {
        resampler = Resampler(
          FloatType.self, inputDim: ipAdapterConfig.inputDim, queryDim: ipAdapterConfig.queryDim,
          outputDim: ipAdapterConfig.outputDim, headDim: ipAdapterConfig.headDim,
          heads: ipAdapterConfig.numHeads, grid: ipAdapterConfig.grid, queries: 16, layers: 4,
          batchSize: 1)
      } else {
        switch version {
        case .v1:
          resampler = Resampler(
            FloatType.self, inputDim: 768, queryDim: 768, outputDim: 768, headDim: 64, heads: 12,
            grid: 16, queries: 16, layers: 4, batchSize: 1)
        case .sdxlBase:
          resampler = Resampler(
            FloatType.self, inputDim: 1280, queryDim: 1280, outputDim: 2048, headDim: 64, heads: 20,
            grid: 16, queries: 16, layers: 4, batchSize: 1)
        case .v2, .sd3, .pixart, .auraflow, .flux1, .sdxlRefiner, .kandinsky21, .ssd1b, .svdI2v,
          .wurstchenStageC, .wurstchenStageB:
          fatalError()
        }
      }
      resampler.compile(inputs: imageEmbeds[0])
      graph.openStore(
        filePaths[0], flags: .readOnly,
        externalStore: TensorData.externalStore(filePath: filePaths[0])
      ) {
        $0.read("resampler", model: resampler, codec: [.ezm7, .q6p, .q8p, .jit, .externalData])
      }
      let imagePromptEmbeds = imageEmbeds.map {
        resampler(inputs: $0)[0].as(of: FloatType.self)
      }
      let batchedImagePromptEmbeds = (0..<inputs.count).map { i in
        switch version {
        case .v1:
          var imagePromptEmbed = graph.variable(.GPU(0), .HWC(2, 16, 768), of: FloatType.self)
          imagePromptEmbed[0..<1, 0..<16, 0..<768] = imagePromptEmbeds[inputs.count]  // The zero prompt embed.
          imagePromptEmbed[1..<2, 0..<16, 0..<768] = imagePromptEmbeds[i]
          return imagePromptEmbed
        case .sdxlBase:
          var imagePromptEmbed = graph.variable(.GPU(0), .HWC(2, 16, 2048), of: FloatType.self)
          imagePromptEmbed[0..<1, 0..<16, 0..<2048] = imagePromptEmbeds[inputs.count]  // The zero prompt embed.
          imagePromptEmbed[1..<2, 0..<16, 0..<2048] = imagePromptEmbeds[i]
          return imagePromptEmbed
        case .v2, .sd3, .pixart, .auraflow, .flux1, .sdxlRefiner, .kandinsky21, .ssd1b, .svdI2v,
          .wurstchenStageC, .wurstchenStageB:
          fatalError()
        }
      }
      // Redo imagePromptEmbeds to be batch of 2.
      let unetIPFixedMapper: ModelWeightMapper
      let unetIPFixed: Model
      switch version {
      case .v1:
        (unetIPFixedMapper, unetIPFixed) = UNetIPFixed(
          batchSize: 2, embeddingLength: (16, 16), startWidth: 64, startHeight: 64,
          usesFlashAttention: usesFlashAttention ? .scaleMerged : .none)
      case .sdxlBase:
        (unetIPFixedMapper, unetIPFixed) = UNetXLIPFixed(
          batchSize: 2, startHeight: 128, startWidth: 128, channels: [320, 640, 1280],
          embeddingLength: 16, attentionRes: [2: 2, 4: 10],
          usesFlashAttention: usesFlashAttention ? .scaleMerged : .none)
      case .v2, .sd3, .pixart, .auraflow, .flux1, .sdxlRefiner, .kandinsky21, .ssd1b, .svdI2v,
        .wurstchenStageC, .wurstchenStageB:
        fatalError()
      }
      unetIPFixed.compile(inputs: batchedImagePromptEmbeds[0])
      graph.openStore(
        filePaths[0], flags: .readOnly,
        externalStore: TensorData.externalStore(filePath: filePaths[0])
      ) {
        if targetBlocks.isEmpty {
          $0.read(
            "unet_ip_fixed", model: unetIPFixed, codec: [.ezm7, .q6p, .q8p, .jit, .externalData])
        } else {
          let mapping = unetIPFixedMapper(.diffusers)
          var reverseMapping = [String: String]()
          for (key, values) in mapping {
            guard values.count == 1 else { continue }
            reverseMapping["__unet_ip_fixed__[\(values[0])]"] = key
          }
          $0.read(
            "unet_ip_fixed", model: unetIPFixed, codec: [.ezm7, .q6p, .q8p, .jit, .externalData]
          ) { name, dataType, format, shape in
            guard let diffusersName = reverseMapping[name] else { return .continue(name) }
            guard diffusersName.hasSuffix("to_v.weight") else { return .continue(name) }
            // Only retain the ones with this name.
            for targetBlock in targetBlocks {
              if diffusersName.contains(targetBlock) {
                return .continue(name)
              }
            }
            return .final(zeroTensor(dataType: dataType, format: format, shape: shape))
          }
        }
      }
      let kvs = batchedImagePromptEmbeds.map {
        unetIPFixed(inputs: $0).map { $0.as(of: FloatType.self) }
      }
      return zip(kvs, inputs).map { (kvs, input) in
        guard input.weight != 1 || controlMode != .prompt else { return kvs }
        return kvs.enumerated().map {
          let weight: Float
          if controlMode == .prompt {
            weight = input.weight
          } else {
            switch version {
            case .v1:
              weight = input.weight * Float($0.element.shape[3]) / 160
            case .sdxlBase:
              weight = input.weight * Float($0.element.shape[usesFlashAttention ? 2 : 1]) / 20
            case .v2, .sd3, .pixart, .auraflow, .flux1, .sdxlRefiner, .kandinsky21, .ssd1b, .svdI2v,
              .wurstchenStageC, .wurstchenStageB:
              fatalError()
            }
          }
          if $0.offset % 2 == 0 {
            // This is for K. Note that after weighting, the formulation is slightly different than IP-Adapter training recipe. Basically, it reducies model's "certainty" around certain values.
            return weight * $0.element
          } else {
            // This is a formulation to shift the mean but retain the variance.
            if usesFlashAttention {
              let v = $0.element
              let mean = v.reduced(.mean, axis: [1])
              return Functional.add(left: v, right: mean, rightScalar: weight - 1)
            } else {
              let v = $0.element
              let mean = v.reduced(.mean, axis: [2])
              return Functional.add(left: v, right: mean, rightScalar: weight - 1)
            }
          }
        }
      }
    case .ipadapterfull:
      let imageEncoder = ImageEncoder<FloatType>(
        filePath: filePaths[1], version: imageEncoderVersion)
      let zeroEmbeds = graph.variable(.GPU(0), .NHWC(1, 224, 224, 3), of: FloatType.self)
      zeroEmbeds.full(0)
      let imageEmbeds = imageEncoder.encode(inputs.map(\.hint) + [zeroEmbeds])
      let projModel: Model
      switch version {
      case .v1:
        projModel = MLPProjModel(width: 1280, outputDim: 768)
      case .sdxlBase:
        projModel = MLPProjModel(width: 1280, outputDim: 2048)
      case .v2, .sd3, .pixart, .auraflow, .flux1, .sdxlRefiner, .kandinsky21, .ssd1b, .svdI2v,
        .wurstchenStageC, .wurstchenStageB:
        fatalError()
      }
      projModel.compile(inputs: imageEmbeds[0])
      graph.openStore(
        filePaths[0], flags: .readOnly,
        externalStore: TensorData.externalStore(filePath: filePaths[0])
      ) {
        $0.read("proj_model", model: projModel, codec: [.ezm7, .q6p, .q8p, .jit, .externalData])
      }
      let imagePromptEmbeds = imageEmbeds.map {
        projModel(inputs: $0)[0].as(of: FloatType.self)
      }
      let batchedImagePromptEmbeds = (0..<inputs.count).map { i in
        switch version {
        case .v1:
          var imagePromptEmbed = graph.variable(.GPU(0), .HWC(2, 257, 768), of: FloatType.self)
          imagePromptEmbed[0..<1, 0..<257, 0..<768] = imagePromptEmbeds[inputs.count]  // The zero prompt embed.
          imagePromptEmbed[1..<2, 0..<257, 0..<768] = imagePromptEmbeds[i]
          return imagePromptEmbed
        case .sdxlBase:
          var imagePromptEmbed = graph.variable(.GPU(0), .HWC(2, 257, 2048), of: FloatType.self)
          imagePromptEmbed[0..<1, 0..<257, 0..<2048] = imagePromptEmbeds[inputs.count]  // The zero prompt embed.
          imagePromptEmbed[1..<2, 0..<257, 0..<2048] = imagePromptEmbeds[i]
          return imagePromptEmbed
        case .v2, .sd3, .pixart, .auraflow, .flux1, .sdxlRefiner, .kandinsky21, .ssd1b, .svdI2v,
          .wurstchenStageC, .wurstchenStageB:
          fatalError()
        }
      }
      // Redo imagePromptEmbeds to be batch of 2.
      let unetIPFixedMapper: ModelWeightMapper
      let unetIPFixed: Model
      switch version {
      case .v1:
        (unetIPFixedMapper, unetIPFixed) = UNetIPFixed(
          batchSize: 2, embeddingLength: (257, 257), startWidth: 64, startHeight: 64,
          usesFlashAttention: usesFlashAttention ? .scaleMerged : .none)
      case .sdxlBase:
        (unetIPFixedMapper, unetIPFixed) = UNetXLIPFixed(
          batchSize: 2, startHeight: 128, startWidth: 128, channels: [320, 640, 1280],
          embeddingLength: 257, attentionRes: [2: 2, 4: 10],
          usesFlashAttention: usesFlashAttention ? .scaleMerged : .none)
      case .v2, .sd3, .pixart, .auraflow, .flux1, .sdxlRefiner, .kandinsky21, .ssd1b, .svdI2v,
        .wurstchenStageC, .wurstchenStageB:
        fatalError()
      }
      unetIPFixed.compile(inputs: batchedImagePromptEmbeds[0])
      graph.openStore(
        filePaths[0], flags: .readOnly,
        externalStore: TensorData.externalStore(filePath: filePaths[0])
      ) {
        if targetBlocks.isEmpty {
          $0.read(
            "unet_ip_fixed", model: unetIPFixed, codec: [.ezm7, .q6p, .q8p, .jit, .externalData])
        } else {

          let mapping = unetIPFixedMapper(.diffusers)
          var reverseMapping = [String: String]()
          for (key, values) in mapping {
            guard values.count == 1 else { continue }
            reverseMapping["__unet_ip_fixed__[\(values[0])]"] = key
          }
          $0.read(
            "unet_ip_fixed", model: unetIPFixed, codec: [.ezm7, .q6p, .q8p, .jit, .externalData]
          ) { name, dataType, format, shape in
            guard let diffusersName = reverseMapping[name] else { return .continue(name) }
            guard diffusersName.hasSuffix("to_v.weight") else { return .continue(name) }
            // Only retain the ones with this name.
            for targetBlock in targetBlocks {
              if diffusersName.contains(targetBlock) {
                return .continue(name)
              }
            }
            return .final(zeroTensor(dataType: dataType, format: format, shape: shape))
          }
        }
      }
      let kvs = batchedImagePromptEmbeds.map {
        unetIPFixed(inputs: $0).map { $0.as(of: FloatType.self) }
      }
      return zip(kvs, inputs).map { (kvs, input) in
        guard input.weight != 1 || controlMode != .prompt else { return kvs }
        return kvs.enumerated().map {
          let weight: Float
          if controlMode == .prompt {
            weight = input.weight
          } else {
            switch version {
            case .v1:
              weight = input.weight * Float($0.element.shape[3]) / 160
            case .sdxlBase:
              weight = input.weight * Float($0.element.shape[usesFlashAttention ? 2 : 1]) / 20
            case .v2, .sd3, .pixart, .auraflow, .flux1, .sdxlRefiner, .kandinsky21, .ssd1b, .svdI2v,
              .wurstchenStageC, .wurstchenStageB:
              fatalError()
            }
          }
          if $0.offset % 2 == 0 {
            // This is for K. Note that after weighting, the formulation is slightly different than IP-Adapter training recipe. Basically, it reducies model's "certainty" around certain values.
            return weight * $0.element
          } else {
            // This is a formulation to shift the mean but retain the variance.
            if usesFlashAttention {
              let v = $0.element
              let mean = v.reduced(.mean, axis: [1])
              return Functional.add(left: v, right: mean, rightScalar: weight - 1)
            } else {
              let v = $0.element
              let mean = v.reduced(.mean, axis: [2])
              return Functional.add(left: v, right: mean, rightScalar: weight - 1)
            }
          }
        }
      }
    case .t2iadapter:
      // For T2I-Adapter, we go all the way.
      let adapter: Model
      if modifier == .color {
        adapter = AdapterLight(channels: [320, 640, 1280, 1280], numRepeat: 4)
      } else {
        adapter = Adapter(channels: [320, 640, 1280, 1280], numRepeat: 2)
      }
      // Hint is already in input size, not in image size.
      let shape = inputs[0].hint.shape
      let startHeight = shape[1]
      let startWidth = shape[2]
      let tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
      let tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
      let tiledDiffusionIsEnabled = (startWidth > tiledWidth) || (startHeight > tiledHeight)
      if tiledDiffusionIsEnabled {
        adapter.compile(
          inputs: inputs[0].hint[0..<shape[0], 0..<tiledHeight, 0..<tiledWidth, 0..<shape[3]])
      } else {
        adapter.compile(inputs: inputs[0].hint)
      }
      graph.openStore(
        filePaths[0], flags: .readOnly,
        externalStore: TensorData.externalStore(filePath: filePaths[0])
      ) {
        $0.read("adapter", model: adapter, codec: [.ezm7, .q6p, .q8p, .jit, .externalData])
      }
      if tiledDiffusionIsEnabled {
        let tileOverlap = min(
          min(
            tiledDiffusion.tileOverlap * 8,
            Int((Double(tiledHeight / 3) / 8).rounded(.down)) * 8),
          Int((Double(tiledWidth / 3) / 8).rounded(.down)) * 8)
        let yTiles =
          (startHeight - tileOverlap * 2 + (tiledHeight - tileOverlap * 2) - 1)
          / (tiledHeight - tileOverlap * 2)
        let xTiles =
          (startWidth - tileOverlap * 2 + (tiledWidth - tileOverlap * 2) - 1)
          / (tiledWidth - tileOverlap * 2)
        return inputs.map { input in
          var result = [DynamicGraph.Tensor<FloatType>]()
          for y in 0..<yTiles {
            let yOfs = y * (tiledHeight - tileOverlap * 2) + (y > 0 ? tileOverlap : 0)
            let (inputStartYPad, inputEndYPad) = paddedTileStartAndEnd(
              iOfs: yOfs, length: startHeight, tileSize: tiledHeight, tileOverlap: tileOverlap)
            for x in 0..<xTiles {
              let xOfs = x * (tiledWidth - tileOverlap * 2) + (x > 0 ? tileOverlap : 0)
              let (inputStartXPad, inputEndXPad) = paddedTileStartAndEnd(
                iOfs: xOfs, length: startWidth, tileSize: tiledWidth, tileOverlap: tileOverlap)
              let tiles = adapter(
                inputs: input.hint[
                  0..<shape[0], inputStartYPad..<inputEndYPad, inputStartXPad..<inputEndXPad,
                  0..<shape[3]
                ].copied()
              ).map { input.weight * $0.as(of: FloatType.self) }
              if result.isEmpty {
                for tile in tiles {
                  var shape = tile.shape
                  let batchSize = shape[0]
                  shape[0] = shape[0] * (xTiles * yTiles)
                  var z = graph.variable(
                    tile.kind, format: tile.format, shape: shape, of: FloatType.self)
                  let index = y * xTiles + x
                  z[
                    (index * batchSize)..<((index + 1) * batchSize), 0..<shape[1], 0..<shape[2],
                    0..<shape[3]] = tile
                  result.append(z)
                }
              } else {
                for (i, tile) in tiles.enumerated() {
                  let shape = tile.shape
                  var z = result[i]
                  let index = y * xTiles + x
                  z[
                    (index * shape[0])..<((index + 1) * shape[0]), 0..<shape[1], 0..<shape[2],
                    0..<shape[3]] = tile
                  result[i] = z
                }
              }
            }
          }
          return result
        }
      } else {
        // compute with weight on adapter right now, to avoid doing it every time during denoising iteration(like control net).
        return inputs.map { input in
          adapter(inputs: input.hint).map { input.weight * $0.as(of: FloatType.self) }
        }
      }
    }
  }
}

extension ControlModel {
  private static func emptyControls(
    graph: DynamicGraph, batchSize: Int, startWidth: Int, startHeight: Int, version: ModelVersion
  )
    -> [DynamicGraph.Tensor<FloatType>]
  {
    precondition(startWidth % 8 == 0)
    precondition(startHeight % 8 == 0)
    let emptyControls: [DynamicGraph.Tensor<FloatType>]
    switch version {
    case .v1, .v2:
      emptyControls = [
        graph.variable(.GPU(0), .NHWC(batchSize, startHeight, startWidth, 320)),
        graph.variable(.GPU(0), .NHWC(batchSize, startHeight, startWidth, 320)),
        graph.variable(.GPU(0), .NHWC(batchSize, startHeight, startWidth, 320)),
        graph.variable(.GPU(0), .NHWC(batchSize, startHeight / 2, startWidth / 2, 320)),
        graph.variable(.GPU(0), .NHWC(batchSize, startHeight / 2, startWidth / 2, 640)),
        graph.variable(.GPU(0), .NHWC(batchSize, startHeight / 2, startWidth / 2, 640)),
        graph.variable(.GPU(0), .NHWC(batchSize, startHeight / 4, startWidth / 4, 640)),
        graph.variable(.GPU(0), .NHWC(batchSize, startHeight / 4, startWidth / 4, 1280)),
        graph.variable(.GPU(0), .NHWC(batchSize, startHeight / 4, startWidth / 4, 1280)),
        graph.variable(.GPU(0), .NHWC(batchSize, startHeight / 8, startWidth / 8, 1280)),
        graph.variable(.GPU(0), .NHWC(batchSize, startHeight / 8, startWidth / 8, 1280)),
        graph.variable(.GPU(0), .NHWC(batchSize, startHeight / 8, startWidth / 8, 1280)),
        graph.variable(.GPU(0), .NHWC(batchSize, startHeight / 8, startWidth / 8, 1280)),
      ]
    case .sdxlBase:
      emptyControls = [
        graph.variable(.GPU(0), .NHWC(batchSize, startHeight, startWidth, 320)),
        graph.variable(.GPU(0), .NHWC(batchSize, startHeight, startWidth, 320)),
        graph.variable(.GPU(0), .NHWC(batchSize, startHeight, startWidth, 320)),
        graph.variable(.GPU(0), .NHWC(batchSize, startHeight / 2, startWidth / 2, 320)),
        graph.variable(.GPU(0), .NHWC(batchSize, startHeight / 2, startWidth / 2, 640)),
        graph.variable(.GPU(0), .NHWC(batchSize, startHeight / 2, startWidth / 2, 640)),
        graph.variable(.GPU(0), .NHWC(batchSize, startHeight / 4, startWidth / 4, 640)),
        graph.variable(.GPU(0), .NHWC(batchSize, startHeight / 4, startWidth / 4, 1280)),
        graph.variable(.GPU(0), .NHWC(batchSize, startHeight / 4, startWidth / 4, 1280)),
        graph.variable(.GPU(0), .NHWC(batchSize, startHeight / 4, startWidth / 4, 1280)),
      ]
    case .sd3, .pixart, .auraflow, .flux1, .kandinsky21, .svdI2v, .sdxlRefiner, .ssd1b,
      .wurstchenStageC,
      .wurstchenStageB:
      fatalError()
    }
    for emptyControl in emptyControls {
      emptyControl.full(0)
    }
    return emptyControls
  }

  private static func emptyInjectKV(
    graph: DynamicGraph, batchSize: Int, startWidth: Int, startHeight: Int
  )
    -> [DynamicGraph.Tensor<FloatType>]
  {
    precondition(startWidth % 8 == 0)
    precondition(startHeight % 8 == 0)
    let emptyInjectKVs: [DynamicGraph.Tensor<FloatType>] = [
      graph.variable(.GPU(0), .HWC(batchSize, startHeight * startWidth, 320)),
      graph.variable(.GPU(0), .HWC(batchSize, startHeight * startWidth, 320)),

      graph.variable(.GPU(0), .HWC(batchSize, (startHeight / 2) * (startWidth / 2), 640)),
      graph.variable(.GPU(0), .HWC(batchSize, (startHeight / 2) * (startWidth / 2), 640)),

      graph.variable(.GPU(0), .HWC(batchSize, (startHeight / 4) * (startWidth / 4), 1280)),
      graph.variable(.GPU(0), .HWC(batchSize, (startHeight / 4) * (startWidth / 4), 1280)),

      graph.variable(.GPU(0), .HWC(batchSize, (startHeight / 8) * (startWidth / 8), 1280)),

      graph.variable(.GPU(0), .HWC(batchSize, (startHeight / 4) * (startWidth / 4), 1280)),
      graph.variable(.GPU(0), .HWC(batchSize, (startHeight / 4) * (startWidth / 4), 1280)),
      graph.variable(.GPU(0), .HWC(batchSize, (startHeight / 4) * (startWidth / 4), 1280)),

      graph.variable(.GPU(0), .HWC(batchSize, (startHeight / 2) * (startWidth / 2), 640)),
      graph.variable(.GPU(0), .HWC(batchSize, (startHeight / 2) * (startWidth / 2), 640)),
      graph.variable(.GPU(0), .HWC(batchSize, (startHeight / 2) * (startWidth / 2), 640)),

      graph.variable(.GPU(0), .HWC(batchSize, startHeight * startWidth, 320)),
      graph.variable(.GPU(0), .HWC(batchSize, startHeight * startWidth, 320)),
      graph.variable(.GPU(0), .HWC(batchSize, startHeight * startWidth, 320)),
    ]
    for emptyInjectKV in emptyInjectKVs {
      emptyInjectKV.full(0)
    }
    return emptyInjectKVs
  }

  private static func emptyAdapters(graph: DynamicGraph, startWidth: Int, startHeight: Int)
    -> [DynamicGraph.Tensor<FloatType>]
  {
    precondition(startWidth % 8 == 0)
    precondition(startHeight % 8 == 0)
    let emptyAdapters: [DynamicGraph.Tensor<FloatType>] = [
      graph.variable(.GPU(0), .NHWC(1, startHeight, startWidth, 320)),
      graph.variable(.GPU(0), .NHWC(1, startHeight / 2, startWidth / 2, 640)),
      graph.variable(.GPU(0), .NHWC(1, startHeight / 4, startWidth / 4, 1280)),
      graph.variable(.GPU(0), .NHWC(1, startHeight / 8, startWidth / 8, 1280)),
    ]
    for emptyAdapter in emptyAdapters {
      emptyAdapter.full(0)
    }
    return emptyAdapters
  }

  private static func emptyIPAdapters(
    graph: DynamicGraph, batchSize: Int, length: Int, numTokens: Int, version: ModelVersion,
    usesFlashAttention: Bool
  )
    -> [DynamicGraph.Tensor<FloatType>]
  {
    var emptyAdapters: [DynamicGraph.Tensor<FloatType>] = []
    switch version {
    case .v1:
      if usesFlashAttention {
        for _ in 0..<4 {
          emptyAdapters.append(graph.variable(.GPU(0), .NHWC(batchSize, numTokens * length, 8, 40)))
        }
        for _ in 4..<8 {
          emptyAdapters.append(graph.variable(.GPU(0), .NHWC(batchSize, numTokens * length, 8, 80)))
        }
        for _ in 8..<20 {
          emptyAdapters.append(
            graph.variable(.GPU(0), .NHWC(batchSize, numTokens * length, 8, 160)))
        }
        for _ in 20..<26 {
          emptyAdapters.append(graph.variable(.GPU(0), .NHWC(batchSize, numTokens * length, 8, 80)))
        }
        for _ in 26..<32 {
          emptyAdapters.append(graph.variable(.GPU(0), .NHWC(batchSize, numTokens * length, 8, 40)))
        }
      } else {
        for _ in 0..<4 {
          emptyAdapters.append(graph.variable(.GPU(0), .NHWC(batchSize, 8, numTokens * length, 40)))
        }
        for _ in 4..<8 {
          emptyAdapters.append(graph.variable(.GPU(0), .NHWC(batchSize, 8, numTokens * length, 80)))
        }
        for _ in 8..<20 {
          emptyAdapters.append(
            graph.variable(.GPU(0), .NHWC(batchSize, 8, numTokens * length, 160)))
        }
        for _ in 20..<26 {
          emptyAdapters.append(graph.variable(.GPU(0), .NHWC(batchSize, 8, numTokens * length, 80)))
        }
        for _ in 26..<32 {
          emptyAdapters.append(graph.variable(.GPU(0), .NHWC(batchSize, 8, numTokens * length, 40)))
        }
      }
    case .sdxlBase:
      if usesFlashAttention {
        for _ in 0..<8 {
          emptyAdapters.append(
            graph.variable(.GPU(0), .NHWC(batchSize, numTokens * length, 10, 64)))
        }
        for _ in 8..<128 {
          emptyAdapters.append(
            graph.variable(.GPU(0), .NHWC(batchSize, numTokens * length, 20, 64)))
        }
        for _ in 128..<140 {
          emptyAdapters.append(
            graph.variable(.GPU(0), .NHWC(batchSize, numTokens * length, 10, 64)))
        }
      } else {
        for _ in 0..<8 {
          emptyAdapters.append(
            graph.variable(.GPU(0), .NHWC(batchSize, 10, numTokens * length, 64)))
        }
        for _ in 8..<128 {
          emptyAdapters.append(
            graph.variable(.GPU(0), .NHWC(batchSize, 20, numTokens * length, 64)))
        }
        for _ in 128..<140 {
          emptyAdapters.append(
            graph.variable(.GPU(0), .NHWC(batchSize, 10, numTokens * length, 64)))
        }
      }
    case .v2, .sd3, .pixart, .auraflow, .flux1, .sdxlRefiner, .kandinsky21, .ssd1b, .svdI2v,
      .wurstchenStageC, .wurstchenStageB:
      fatalError()
    }
    for emptyAdapter in emptyAdapters {
      emptyAdapter.full(0)
    }
    return emptyAdapters
  }

  private static func flattenWeightMapping(_ weightMapper: ModelWeightMapper) -> [String: String] {
    let mapping = weightMapper(.diffusers)
    var flattenMapping = [String: String]()
    for (key, values) in mapping {
      guard values.count > 1 else {
        flattenMapping[key] = values[0]
        continue
      }
      for (i, value) in values.enumerated() {
        flattenMapping[key + ".\(i)"] = value
      }
    }
    return flattenMapping
  }

  private static func reversed(_ dictionary: [String: String]) -> [String: String] {
    var reversedDictionary = [String: String]()
    for (key, value) in dictionary {
      reversedDictionary[value] = key
    }
    return reversedDictionary
  }

  public func encode(
    textEncoding: [DynamicGraph.Tensor<FloatType>], vector: DynamicGraph.Tensor<FloatType>?,
    batchSize: Int, startHeight: Int, startWidth: Int, tokenLengthUncond: Int, tokenLengthCond: Int,
    zeroNegativePrompt: Bool, mainUNetFixed: (filePath: String, weightMapper: ModelWeightMapper?)
  ) -> [DynamicGraph.Tensor<FloatType>] {
    guard version == .sdxlBase && (type == .controlnet || type == .controlnetlora) else {
      return textEncoding
    }
    let graph = textEncoding[0].graph
    var textEncoding = textEncoding
    graph.openStore(
      filePaths[0], flags: .readOnly,
      externalStore: TensorData.externalStore(filePath: filePaths[0])
    ) {
      if $0.read(like: "__encoder_hid_proj__[t-0-0]") != nil {
        let encoderHidProj = Dense(count: 2_048)
        encoderHidProj.compile(inputs: textEncoding[0])
        $0.read(
          "encoder_hid_proj", model: encoderHidProj,
          codec: [.jit, .q6p, .q8p, .ezm7, .externalData])
        textEncoding = encoderHidProj(inputs: textEncoding[0]).map { $0.as(of: FloatType.self) }
      }
    }
    let batchSize = textEncoding[0].shape[0]
    let maxTokenLength = textEncoding[0].shape[1]
    var crossattn = graph.variable(
      textEncoding[0].kind, .HWC(batchSize, maxTokenLength, 2048), of: FloatType.self)
    if textEncoding.count >= 2 {
      crossattn[0..<batchSize, 0..<maxTokenLength, 0..<768] = textEncoding[0]
      crossattn[0..<batchSize, 0..<maxTokenLength, 768..<2048] = textEncoding[1]
    } else {
      crossattn[0..<batchSize, 0..<maxTokenLength, 0..<2048] = textEncoding[0]
    }
    if zeroNegativePrompt && (version == .sdxlBase || version == .ssd1b) {
      crossattn[0..<(batchSize / 2), 0..<maxTokenLength, 0..<2048].full(0)
    }
    let inputAttentionRes: KeyValuePairs<Int, [Int]>
    let middleAttentionBlocks: Int
    if transformerBlocks.count == 4 {
      inputAttentionRes = [
        2: [transformerBlocks[1], transformerBlocks[1]],
        4: [transformerBlocks[2], transformerBlocks[2]],
      ]
      middleAttentionBlocks = transformerBlocks[3]
    } else {
      inputAttentionRes = [2: [2, 2], 4: [10, 10]]
      middleAttentionBlocks = 10
    }
    guard transformerBlocks.isEmpty || transformerBlocks.contains(where: { $0 > 0 }) else {
      return vector.map({ [$0] }) ?? []
    }
    let controlNetFixed: Model
    switch type {
    case .controlnet:
      controlNetFixed =
        ControlNetXLFixed(
          batchSize: batchSize, startHeight: startHeight, startWidth: startWidth,
          channels: [320, 640, 1280], embeddingLength: (tokenLengthUncond, tokenLengthCond),
          inputAttentionRes: inputAttentionRes, middleAttentionBlocks: middleAttentionBlocks,
          usesFlashAttention: usesFlashAttention ? .scaleMerged : .none
        ).0
      controlNetFixed.compile(inputs: crossattn)
      graph.openStore(
        filePaths[0], flags: .readOnly,
        externalStore: TensorData.externalStore(filePath: filePaths[0])
      ) { store in
        store.read(
          "controlnet_fixed", model: controlNetFixed,
          codec: [.q6p, .q8p, .ezm7, .jit, .externalData])
      }
    case .controlnetlora:
      guard let weightMapper = mainUNetFixed.weightMapper else { return textEncoding }
      let rank = LoRALoader<FloatType>.rank(
        graph, of: [filePaths[0]], inspectFilesRequireMerge: false
      ).rank
      let configuration = LoRANetworkConfiguration(rank: rank, scale: 1, highPrecision: false)
      let controlNetFixedWeightMapper: ModelWeightMapper
      (controlNetFixed, controlNetFixedWeightMapper) =
        LoRAControlNetXLFixed(
          batchSize: batchSize, startHeight: startHeight, startWidth: startWidth,
          channels: [320, 640, 1280], embeddingLength: (tokenLengthUncond, tokenLengthCond),
          inputAttentionRes: inputAttentionRes, middleAttentionBlocks: middleAttentionBlocks,
          usesFlashAttention: usesFlashAttention ? .scaleMerged : .none,
          LoRAConfiguration: configuration
        )
      controlNetFixed.compile(inputs: crossattn)
      let unetFixedMapping = Self.flattenWeightMapping(weightMapper)
      let reversedControlNetFixedMapping = Self.reversed(
        Self.flattenWeightMapping(controlNetFixedWeightMapper))
      graph.openStore(
        filePaths[0], flags: .readOnly,
        externalStore: TensorData.externalStore(filePath: filePaths[0])
      ) {
        let keys = Set($0.keys)
        // First, read as much parameters from this file first.
        $0.read(
          "controlnet_fixed", model: controlNetFixed,
          codec: [.q6p, .q8p, .ezm7, .jit, .externalData])
        // Then, read from the main UNet with a rewrite rule.
        graph.openStore(
          mainUNetFixed.filePath, flags: .readOnly,
          externalStore: TensorData.externalStore(filePath: mainUNetFixed.filePath)
        ) {
          $0.read(
            "controlnet_fixed", model: controlNetFixed,
            codec: [.q6p, .q8p, .ezm7, .jit, .externalData]
          ) {
            name, _, _, _ in
            guard !keys.contains(name) else {
              return .fail
            }
            // If cannot find it in the file, we need to load these from mainUNetFixedStore.
            guard name.hasPrefix("__controlnet_fixed__[") else {
              return .fail
            }
            let tensorName = String(name.dropFirst(21).dropLast())
            // Look up its mapping name, and use that to look up mainUNetFixed name.
            guard let mappingName = reversedControlNetFixedMapping[tensorName],
              let mainUNetFixedTensorName = unetFixedMapping[mappingName]
            else {
              return .fail
            }
            return .continue("__unet_fixed__[\(mainUNetFixedTensorName)]")
          }
        }
      }
    case .ipadapterfull, .ipadapterplus, .t2iadapter, .injectKV:
      fatalError()
    }
    return (vector.map({ [$0] }) ?? [])
      + controlNetFixed(inputs: crossattn).map { $0.as(of: FloatType.self) }
  }

  public func callAsFunction(
    inputStartYPad: Int, inputEndYPad: Int, inputStartXPad: Int, inputEndXPad: Int,
    step: Int, inputs xT: DynamicGraph.Tensor<FloatType>, _ hint: [DynamicGraph.Tensor<FloatType>],
    strength: Float, _ timestep: DynamicGraph.Tensor<FloatType>?,
    _ c: [DynamicGraph.Tensor<FloatType>], tokenLengthUncond: Int, tokenLengthCond: Int,
    isCfgEnabled: Bool, mainUNetAndWeightMapper: (Model, ModelWeightMapper)?,
    controlNet existingControlNet: inout Model?
  ) -> [DynamicGraph.Tensor<FloatType>] {
    let graph = xT.graph
    let batchSize = xT.shape[0]
    let startHeight = xT.shape[1]
    let startWidth = xT.shape[2]
    let channels = xT.shape[3]
    guard step >= startStep && step < endStep else {
      if step >= endStep {
        // If we already ended, we can nil out the model.
        existingControlNet = nil
      }
      switch type {
      case .controlnet, .controlnetlora:
        return Self.emptyControls(
          graph: graph, batchSize: batchSize, startWidth: startWidth, startHeight: startHeight,
          version: version)
      case .ipadapterplus:
        return Self.emptyIPAdapters(
          graph: graph, batchSize: batchSize, length: 1, numTokens: 16, version: version,
          usesFlashAttention: usesFlashAttention)
      case .ipadapterfull:
        return Self.emptyIPAdapters(
          graph: graph, batchSize: batchSize, length: 1, numTokens: 257, version: version,
          usesFlashAttention: usesFlashAttention)
      case .t2iadapter:
        return Self.emptyAdapters(graph: graph, startWidth: startWidth, startHeight: startHeight)
      case .injectKV:
        return Self.emptyInjectKV(
          graph: graph, batchSize: batchSize, startWidth: startWidth, startHeight: startHeight)
      }
    }
    guard type == .controlnet || type == .controlnetlora else {
      switch type {
      case .injectKV:
        return hint.map {
          let shape = $0.shape
          guard shape[0] != batchSize else { return $0 }
          precondition(shape[0] == 2)
          var x = graph.variable(
            .GPU(0), .HWC(batchSize, shape[1], shape[2]), of: FloatType.self)
          if isCfgEnabled {
            for i in 0..<(batchSize / 2) {
              x[i..<(i + 1), 0..<shape[1], 0..<shape[2]] =
                $0[0..<1, 0..<shape[1], 0..<shape[2]]
            }
            for i in (batchSize / 2)..<batchSize {
              x[i..<(i + 1), 0..<shape[1], 0..<shape[2]] =
                $0[1..<2, 0..<shape[1], 0..<shape[2]]
            }
          } else {
            for i in 0..<batchSize {
              x[i..<(i + 1), 0..<shape[1], 0..<shape[2]] =
                $0[(shape[0] - 1)..<shape[0], 0..<shape[1], 0..<shape[2]]
            }
          }
          return x

        }
      case .ipadapterplus, .ipadapterfull:
        return hint.map {
          let shape = $0.shape
          guard shape[0] != batchSize else { return $0 }
          if shape[0] == 1 {
            var x = graph.variable(
              .GPU(0), .NHWC(batchSize, shape[1], shape[2], shape[3]), of: FloatType.self)
            for i in 0..<batchSize {
              x[i..<(i + 1), 0..<shape[1], 0..<shape[2], 0..<shape[3]] =
                $0[0..<1, 0..<shape[1], 0..<shape[2], 0..<shape[3]]
            }
            return x
          } else {
            precondition(shape[0] == 2)
            var x = graph.variable(
              .GPU(0), .NHWC(batchSize, shape[1], shape[2], shape[3]), of: FloatType.self)
            if isCfgEnabled {
              for i in 0..<(batchSize / 2) {
                x[i..<(i + 1), 0..<shape[1], 0..<shape[2], 0..<shape[3]] =
                  $0[0..<1, 0..<shape[1], 0..<shape[2], 0..<shape[3]]
              }
              for i in (batchSize / 2)..<batchSize {
                x[i..<(i + 1), 0..<shape[1], 0..<shape[2], 0..<shape[3]] =
                  $0[1..<2, 0..<shape[1], 0..<shape[2], 0..<shape[3]]
              }
            } else {
              for i in 0..<batchSize {
                x[i..<(i + 1), 0..<shape[1], 0..<shape[2], 0..<shape[3]] =
                  $0[(shape[0] - 1)..<shape[0], 0..<shape[1], 0..<shape[2], 0..<shape[3]]
              }
            }
            return x
          }
        }
      case .t2iadapter:
        guard controlMode == .control && isCfgEnabled else { return hint }
        return hint.map {
          let shape = $0.shape
          precondition(shape[0] == 1)
          var x = graph.variable(
            .GPU(0), .NHWC(batchSize, shape[1], shape[2], shape[3]), of: FloatType.self)
          x[0..<(batchSize / 2), 0..<shape[1], 0..<shape[2], 0..<shape[3]].full(0)
          for i in (batchSize / 2)..<batchSize {
            x[i..<(i + 1), 0..<shape[1], 0..<shape[2], 0..<shape[3]] = $0
          }
          return x
        }
      case .controlnet, .controlnetlora:
        fatalError()
      }
    }
    let xIn =
      channels == 4 ? xT : xT[0..<batchSize, 0..<startHeight, 0..<startWidth, 0..<4].copied()
    let controlNet: Model
    let controlNetWeightMapper: ModelWeightMapper?
    if let existingControlNet = existingControlNet {
      controlNet = existingControlNet
      controlNetWeightMapper = nil
    } else {
      switch version {
      case .v1:
        controlNet =
          ControlNet(
            batchSize: batchSize, embeddingLength: (tokenLengthUncond, tokenLengthCond),
            startWidth: startWidth, startHeight: startHeight,
            usesFlashAttention: usesFlashAttention ? .scaleMerged : .none
          ).0
        controlNetWeightMapper = nil
      case .v2:
        controlNet =
          ControlNetv2(
            batchSize: batchSize, embeddingLength: (tokenLengthUncond, tokenLengthCond),
            startWidth: startWidth, startHeight: startHeight, upcastAttention: false,
            usesFlashAttention: usesFlashAttention ? .scaleMerged : .none
          ).0
        controlNetWeightMapper = nil
      case .sdxlBase:
        let inputAttentionRes: KeyValuePairs<Int, [Int]>
        let middleAttentionBlocks: Int
        if transformerBlocks.count == 4 {
          inputAttentionRes = [
            2: [transformerBlocks[1], transformerBlocks[1]],
            4: [transformerBlocks[2], transformerBlocks[2]],
          ]
          middleAttentionBlocks = transformerBlocks[3]
        } else {
          inputAttentionRes = [2: [2, 2], 4: [10, 10]]
          middleAttentionBlocks = 10
        }
        if type == .controlnetlora {
          let rank = LoRALoader<FloatType>.rank(
            graph, of: [filePaths[0]], inspectFilesRequireMerge: false
          ).rank
          let configuration = LoRANetworkConfiguration(rank: rank, scale: 1, highPrecision: false)
          (controlNet, controlNetWeightMapper) =
            LoRAControlNetXL(
              batchSize: batchSize, startWidth: startWidth, startHeight: startHeight,
              channels: [320, 640, 1280], embeddingLength: (tokenLengthUncond, tokenLengthCond),
              inputAttentionRes: inputAttentionRes, middleAttentionBlocks: middleAttentionBlocks,
              usesFlashAttention: usesFlashAttention ? .scaleMerged : .none,
              LoRAConfiguration: configuration
            )
        } else {
          controlNet =
            ControlNetXL(
              batchSize: batchSize, startWidth: startWidth, startHeight: startHeight,
              channels: [320, 640, 1280], embeddingLength: (tokenLengthUncond, tokenLengthCond),
              inputAttentionRes: inputAttentionRes, middleAttentionBlocks: middleAttentionBlocks,
              usesFlashAttention: usesFlashAttention ? .scaleMerged : .none
            ).0
          controlNetWeightMapper = nil
        }
      case .sd3, .pixart, .auraflow, .flux1, .kandinsky21, .sdxlRefiner, .ssd1b, .svdI2v,
        .wurstchenStageC,
        .wurstchenStageB:
        fatalError()
      }
    }
    let tiledDiffusionIsEnabled = inputEndYPad > 0 && inputEndXPad > 0
    if existingControlNet == nil {
      if tiledDiffusionIsEnabled {
        let shape = hint[0].shape
        controlNet.compile(
          inputs: [
            xIn,
            hint[0][
              0..<shape[0], inputStartYPad..<inputEndYPad, inputStartXPad..<inputEndXPad,
              0..<shape[3]],
          ] + (timestep.map { [$0] } ?? []) + c)
      } else {
        controlNet.compile(inputs: [xIn, hint[0]] + (timestep.map { [$0] } ?? []) + c)
      }
      let externalData: DynamicGraph.Store.Codec =
        externalOnDemand ? .externalOnDemand : .externalData
      if let controlNetWeightMapper = controlNetWeightMapper,
        let mainUNetAndWeightMapper = mainUNetAndWeightMapper
      {
        precondition(type == .controlnetlora)
        let reversedControlNetWeightMapping = Self.reversed(
          Self.flattenWeightMapping(controlNetWeightMapper))
        let mainUNetWeightMapping = Self.flattenWeightMapping(mainUNetAndWeightMapper.1)
        graph.openStore(
          filePaths[0], flags: .readOnly,
          externalStore: TensorData.externalStore(filePath: filePaths[0])
        ) {
          let keys = Set(
            $0.keys.filter { $0.hasPrefix("__controlnet__[") }.compactMap {
              reversedControlNetWeightMapping[String($0.dropFirst(15).dropLast())]
            })
          // First read weights from disk.
          $0.read(
            "controlnet", model: controlNet, codec: [.ezm7, .q6p, .q8p, .fpzip, .jit, externalData])
          // Then check if we can share weights from main model.
          controlNet.parameters.share(from: mainUNetAndWeightMapper.0.parameters) {
            controlNetName, _ in
            guard let controlNetMapping = reversedControlNetWeightMapping[controlNetName] else {
              return .fail
            }
            // If we have the key in the file, we don't need to load from the main model.
            guard !keys.contains(controlNetMapping) else { return .fail }
            guard let mainUNetMapping = mainUNetWeightMapping[controlNetMapping] else {
              return .fail
            }
            return .continue(mainUNetMapping)
          }
        }
      } else {
        graph.openStore(
          filePaths[0], flags: .readOnly,
          externalStore: TensorData.externalStore(filePath: filePaths[0])
        ) {
          $0.read(
            "controlnet", model: controlNet, codec: [.ezm7, .q6p, .q8p, .fpzip, .jit, externalData])
        }
      }
      existingControlNet = controlNet
    }
    var result: [DynamicGraph.Tensor<FloatType>]
    if tiledDiffusionIsEnabled {
      let shape = hint[0].shape
      result = controlNet(
        inputs: xIn,
        [
          hint[0][
            0..<shape[0], inputStartYPad..<inputEndYPad, inputStartXPad..<inputEndXPad, 0..<shape[3]
          ].copied()
        ] + (timestep.map { [$0] } ?? []) + c
      ).map { $0.as(of: FloatType.self) }
    } else {
      result = controlNet(inputs: xIn, [hint[0]] + (timestep.map { [$0] } ?? []) + c).map {
        $0.as(of: FloatType.self)
      }
    }
    if controlMode == .control && isCfgEnabled {
      for x in result {
        let shape = x.shape
        x[0..<(batchSize / 2), 0..<shape[1], 0..<shape[2], 0..<shape[3]].full(0)
      }
    }
    if globalAveragePooling {
      result = result.map {
        $0.reduced(.mean, axis: [1, 2])
      }
    }
    // In ControlNet implementation, we degrade influence as it approaches lower level.
    if controlMode != .balanced {
      return result.enumerated().map {
        return powf(0.825, Float(12 - $0)) * $1 * strength
      }
    } else {
      if strength != 1 {
        return result.map { $0 * strength }
      }
      return result
    }
  }
}

extension ControlModel {
  // Assuming input is in the range of -1 to 1, and in form of NHWC with 3 channels.
  public static func canny(_ x: Tensor<FloatType>) -> Tensor<FloatType> {
    let shape = x.shape
    precondition(x.kind == .CPU)
    precondition(shape.count == 4)
    precondition(shape[3] == 3)
    precondition(shape[0] == 1)
    let startWidth = shape[2]
    let startHeight = shape[1]
    precondition(startWidth % 4 == 0)
    let u8Img = ccv_dense_matrix_new(
      Int32(startHeight), Int32(startWidth), Int32(CCV_8U | CCV_C1), nil, 0)!
    x.withUnsafeBytes {
      guard let fp16 = $0.baseAddress?.assumingMemoryBound(to: FloatType.self) else { return }
      for i in 0..<startHeight {
        for j in 0..<startWidth {
          let r = Int32(max(min(Int32((fp16[i * startWidth * 3 + j * 3] + 1) * 127.5), 255), 0))
          let g = Int32(max(min(Int32((fp16[i * startWidth * 3 + j * 3 + 1] + 1) * 127.5), 255), 0))
          let b = Int32(max(min(Int32((fp16[i * startWidth * 3 + j * 3 + 2] + 1) * 127.5), 255), 0))
          u8Img.pointee.data.u8[i * startWidth + j] = UInt8(
            (r * 6969 + g * 23434 + b * 2365) >> 15)
        }
      }
    }
    var cannyImg: UnsafeMutablePointer<ccv_dense_matrix_t>? = nil
    ccv_canny(u8Img, &cannyImg, 0, 3, 100, 200)
    var y = Tensor<FloatType>(.CPU, .NHWC(1, startHeight, startWidth, 3))
    if let cannyImg = cannyImg {
      for i in 0..<startHeight {
        for j in 0..<startWidth {
          if cannyImg.pointee.data.u8[i * startWidth + j] == 0 {
            y[0, i, j, 0] = 0
            y[0, i, j, 1] = 0
            y[0, i, j, 2] = 0
          } else {
            y[0, i, j, 0] = 1
            y[0, i, j, 1] = 1
            y[0, i, j, 2] = 1
          }
        }
      }
    }
    return y
  }
}
