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
}

public enum ControlMode {
  case balanced
  case prompt
  case control
}

public struct ControlModel<FloatType: TensorNumeric & BinaryFloatingPoint> {
  public let filePaths: [String]
  public let type: ControlType
  public let modifier: ControlHintType
  public let externalOnDemand: Bool
  public let version: ModelVersion
  public let tiledDiffusion: TiledDiffusionConfiguration
  public let usesFlashAttention: Bool
  public let startStep: Int
  public let endStep: Int
  public let controlMode: ControlMode
  public let globalAveragePooling: Bool
  public let transformerBlocks: [Int]
  public init(
    filePaths: [String], type: ControlType, modifier: ControlHintType,
    externalOnDemand: Bool, version: ModelVersion, tiledDiffusion: TiledDiffusionConfiguration,
    usesFlashAttention: Bool, startStep: Int, endStep: Int, controlMode: ControlMode,
    globalAveragePooling: Bool, transformerBlocks: [Int]
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
  }
}

extension ControlModel {
  static func injectedControlsAndAdapters(
    injecteds: [(
      model: ControlModel<FloatType>, hints: [([DynamicGraph.Tensor<FloatType>], Float)]
    )],
    step: Int, version: ModelVersion, usesFlashAttention: Bool,
    inputs xT: DynamicGraph.Tensor<FloatType>,
    _ timestep: DynamicGraph.Tensor<FloatType>, _ c: [[DynamicGraph.Tensor<FloatType>]],
    tokenLengthUncond: Int, tokenLengthCond: Int,
    mainUNetAndWeightMapper: (Model, ModelWeightMapper)?,
    controlNets existingControlNets: inout [Model?]
  ) -> (
    [DynamicGraph.Tensor<FloatType>], [DynamicGraph.Tensor<FloatType>],
    [DynamicGraph.Tensor<FloatType>]
  ) {
    var injectedControls = [DynamicGraph.Tensor<FloatType>]()
    var injectedT2IAdapters = [DynamicGraph.Tensor<FloatType>]()
    var injectedIPAdapters = [DynamicGraph.Tensor<FloatType>]()
    for (i, injected) in injecteds.enumerated() {
      guard injected.model.version == version else { continue }
      switch injected.model.type {
      case .controlnet, .controlnetlora:
        for (hint, strength) in injected.hints {
          let newInjectedControls = injected.model(
            step: step, inputs: xT, hint, strength: strength, timestep, c[i],
            tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
            mainUNetAndWeightMapper: mainUNetAndWeightMapper, controlNet: &existingControlNets[i])
          if injectedControls.isEmpty {
            injectedControls = newInjectedControls
          } else {
            injectedControls = zip(injectedControls, newInjectedControls).map { $0 + $1 }
          }
        }
      case .ipadapterplus, .ipadapterfull:
        var instanceInjectedIPAdapters = [DynamicGraph.Tensor<FloatType>]()
        for (hint, strength) in injected.hints {
          let newInjectedIPAdapters = injected.model(
            step: step, inputs: xT, hint, strength: strength, timestep, c[i],
            tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
            mainUNetAndWeightMapper: mainUNetAndWeightMapper, controlNet: &existingControlNets[i])
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
      case .t2iadapter:
        for (hint, strength) in injected.hints {
          let newInjectedT2IAdapters = injected.model(
            step: step, inputs: xT, hint, strength: strength, timestep, c[i],
            tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
            mainUNetAndWeightMapper: mainUNetAndWeightMapper, controlNet: &existingControlNets[i])
          if injectedT2IAdapters.isEmpty {
            injectedT2IAdapters = newInjectedT2IAdapters
          } else {
            injectedT2IAdapters = zip(injectedT2IAdapters, newInjectedT2IAdapters).map { $0 + $1 }
          }
        }
      }
    }
    return (injectedControls, injectedT2IAdapters, injectedIPAdapters)
  }

  static func emptyInjectedControlsAndAdapters(
    injecteds: [(
      model: ControlModel<FloatType>, hints: [([DynamicGraph.Tensor<FloatType>], Float)]
    )],
    step: Int, version: ModelVersion, inputs xT: DynamicGraph.Tensor<FloatType>
  ) -> (
    [DynamicGraph.Tensor<FloatType>], [DynamicGraph.Tensor<FloatType>],
    [DynamicGraph.Tensor<FloatType>]
  ) {
    var injectedControls = [DynamicGraph.Tensor<FloatType>]()
    var injectedT2IAdapters = [DynamicGraph.Tensor<FloatType>]()
    var injectedIPAdapters = [DynamicGraph.Tensor<FloatType>]()
    let graph = xT.graph
    let batchSize = xT.shape[0]
    let startHeight = xT.shape[1]
    let startWidth = xT.shape[2]
    for injected in injecteds {
      guard injected.model.version == version else { continue }
      switch injected.model.type {
      case .controlnet, .controlnetlora:
        if injectedControls.isEmpty {
          injectedControls = emptyControls(
            graph: graph, batchSize: batchSize, startWidth: startWidth, startHeight: startHeight,
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
            graph: graph, startWidth: startWidth, startHeight: startHeight)
        }
      }
    }
    return (injectedControls, injectedT2IAdapters, injectedIPAdapters)
  }
}

extension ControlModel {
  public func hint(inputs: [(hint: DynamicGraph.Tensor<FloatType>, weight: Float)])
    -> [[DynamicGraph.Tensor<FloatType>]]
  {
    guard inputs.count > 0 else {
      return []
    }
    let graph = inputs[0].hint.graph
    switch type {
    case .controlnet, .controlnetlora:
      // For ControlNet, we only compute hint.
      let hintNet = HintNet(channels: 320).0
      hintNet.compile(inputs: inputs[0].hint)
      graph.openStore(
        filePaths[0], flags: .readOnly,
        externalStore: TensorData.externalStore(filePath: filePaths[0])
      ) {
        $0.read("hintnet", model: hintNet, codec: [.ezm7, .q6p, .q8p, .jit, .externalData])
      }
      return inputs.map { hintNet(inputs: $0.hint).map { $0.as(of: FloatType.self) } }
    case .ipadapterplus:
      let imageEncoder = ImageEncoder<FloatType>(filePath: filePaths[1])
      let zeroEmbeds = graph.variable(.GPU(0), .NHWC(1, 224, 224, 3), of: FloatType.self)
      zeroEmbeds.full(0)
      let imageEmbeds = imageEncoder.encode(inputs.map(\.hint) + [zeroEmbeds])
      let resampler: Model
      switch version {
      case .v1:
        resampler = Resampler(
          FloatType.self, width: 768, outputDim: 768, heads: 12, grid: 16, queries: 16, layers: 4,
          batchSize: 1)
      case .sdxlBase:
        resampler = Resampler(
          FloatType.self, width: 1280, outputDim: 2048, heads: 20, grid: 16, queries: 16, layers: 4,
          batchSize: 1)
      case .v2, .sdxlRefiner, .kandinsky21, .ssd1b, .svdI2v, .wurstchenStageC, .wurstchenStageB:
        fatalError()
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
        case .v2, .sdxlRefiner, .kandinsky21, .ssd1b, .svdI2v, .wurstchenStageC, .wurstchenStageB:
          fatalError()
        }
      }
      // Redo imagePromptEmbeds to be batch of 2.
      let unetIPFixed: Model
      switch version {
      case .v1:
        unetIPFixed = UNetIPFixed(
          batchSize: 2, embeddingLength: (16, 16), startWidth: 64, startHeight: 64,
          usesFlashAttention: usesFlashAttention ? .scaleMerged : .none)
      case .sdxlBase:
        unetIPFixed = UNetXLIPFixed(
          batchSize: 2, startHeight: 128, startWidth: 128, channels: [320, 640, 1280],
          embeddingLength: 16, attentionRes: [2: 2, 4: 10],
          usesFlashAttention: usesFlashAttention ? .scaleMerged : .none)
      case .v2, .sdxlRefiner, .kandinsky21, .ssd1b, .svdI2v, .wurstchenStageC, .wurstchenStageB:
        fatalError()
      }
      unetIPFixed.compile(inputs: batchedImagePromptEmbeds[0])
      graph.openStore(
        filePaths[0], flags: .readOnly,
        externalStore: TensorData.externalStore(filePath: filePaths[0])
      ) {
        $0.read(
          "unet_ip_fixed", model: unetIPFixed, codec: [.ezm7, .q6p, .q8p, .jit, .externalData])
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
            case .v2, .sdxlRefiner, .kandinsky21, .ssd1b, .svdI2v, .wurstchenStageC,
              .wurstchenStageB:
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
      let imageEncoder = ImageEncoder<FloatType>(filePath: filePaths[1])
      let zeroEmbeds = graph.variable(.GPU(0), .NHWC(1, 224, 224, 3), of: FloatType.self)
      zeroEmbeds.full(0)
      let imageEmbeds = imageEncoder.encode(inputs.map(\.hint) + [zeroEmbeds])
      let projModel: Model
      switch version {
      case .v1:
        projModel = MLPProjModel(width: 1280, outputDim: 768)
      case .sdxlBase:
        projModel = MLPProjModel(width: 1280, outputDim: 2048)
      case .v2, .sdxlRefiner, .kandinsky21, .ssd1b, .svdI2v, .wurstchenStageC, .wurstchenStageB:
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
        case .v2, .sdxlRefiner, .kandinsky21, .ssd1b, .svdI2v, .wurstchenStageC, .wurstchenStageB:
          fatalError()
        }
      }
      // Redo imagePromptEmbeds to be batch of 2.
      let unetIPFixed: Model
      switch version {
      case .v1:
        unetIPFixed = UNetIPFixed(
          batchSize: 2, embeddingLength: (257, 257), startWidth: 64, startHeight: 64,
          usesFlashAttention: usesFlashAttention ? .scaleMerged : .none)
      case .sdxlBase:
        unetIPFixed = UNetXLIPFixed(
          batchSize: 2, startHeight: 128, startWidth: 128, channels: [320, 640, 1280],
          embeddingLength: 257, attentionRes: [2: 2, 4: 10],
          usesFlashAttention: usesFlashAttention ? .scaleMerged : .none)
      case .v2, .sdxlRefiner, .kandinsky21, .ssd1b, .svdI2v, .wurstchenStageC, .wurstchenStageB:
        fatalError()
      }
      unetIPFixed.compile(inputs: batchedImagePromptEmbeds[0])
      graph.openStore(
        filePaths[0], flags: .readOnly,
        externalStore: TensorData.externalStore(filePath: filePaths[0])
      ) {
        $0.read(
          "unet_ip_fixed", model: unetIPFixed, codec: [.ezm7, .q6p, .q8p, .jit, .externalData])
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
            case .v2, .sdxlRefiner, .kandinsky21, .ssd1b, .svdI2v, .wurstchenStageC,
              .wurstchenStageB:
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
      adapter.compile(inputs: inputs[0].hint)
      graph.openStore(
        filePaths[0], flags: .readOnly,
        externalStore: TensorData.externalStore(filePath: filePaths[0])
      ) {
        $0.read("adapter", model: adapter, codec: [.ezm7, .q6p, .q8p, .jit, .externalData])
      }
      // compute with weight on adapter right now, to avoid doing it every time during denoising iteration(like control net).
      return inputs.map { input in
        adapter(inputs: input.hint).map { input.weight * $0.as(of: FloatType.self) }
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
    case .kandinsky21, .svdI2v, .sdxlRefiner, .ssd1b, .wurstchenStageC, .wurstchenStageB:
      fatalError()
    }
    for emptyControl in emptyControls {
      emptyControl.full(0)
    }
    return emptyControls
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
    case .v2, .sdxlRefiner, .kandinsky21, .ssd1b, .svdI2v, .wurstchenStageC, .wurstchenStageB:
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
    let batchSize = textEncoding[0].shape[0]
    let maxTokenLength = textEncoding[0].shape[1]
    var crossattn = graph.variable(
      textEncoding[0].kind, .HWC(batchSize, maxTokenLength, 2048), of: FloatType.self)
    crossattn[0..<batchSize, 0..<maxTokenLength, 0..<768] = textEncoding[0]
    crossattn[0..<batchSize, 0..<maxTokenLength, 768..<2048] = textEncoding[1]
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
    case .ipadapterfull, .ipadapterplus, .t2iadapter:
      fatalError()
    }
    return (vector.map({ [$0] }) ?? [])
      + controlNetFixed(inputs: crossattn).map { $0.as(of: FloatType.self) }
  }

  public func callAsFunction(
    step: Int,
    inputs xT: DynamicGraph.Tensor<FloatType>, _ hint: [DynamicGraph.Tensor<FloatType>],
    strength: Float,
    _ timestep: DynamicGraph.Tensor<FloatType>, _ c: [DynamicGraph.Tensor<FloatType>],
    tokenLengthUncond: Int, tokenLengthCond: Int,
    mainUNetAndWeightMapper: (Model, ModelWeightMapper)?,
    controlNet existingControlNet: inout Model?
  ) -> [DynamicGraph.Tensor<FloatType>] {
    let graph = xT.graph
    let batchSize = xT.shape[0]
    let startHeight = xT.shape[1]
    let startWidth = xT.shape[2]
    let channels = xT.shape[3]
    guard step >= startStep && step < endStep else {
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
      }
    }
    guard type == .controlnet || type == .controlnetlora else {
      switch type {
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
            for i in 0..<(batchSize / 2) {
              x[i..<(i + 1), 0..<shape[1], 0..<shape[2], 0..<shape[3]] =
                $0[0..<1, 0..<shape[1], 0..<shape[2], 0..<shape[3]]
            }
            for i in (batchSize / 2)..<batchSize {
              x[i..<(i + 1), 0..<shape[1], 0..<shape[2], 0..<shape[3]] =
                $0[1..<2, 0..<shape[1], 0..<shape[2], 0..<shape[3]]
            }
            return x
          }
        }
      case .t2iadapter:
        guard controlMode == .control else { return hint }
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
      case .kandinsky21, .sdxlRefiner, .ssd1b, .svdI2v, .wurstchenStageC, .wurstchenStageB:
        fatalError()
      }
    }
    if existingControlNet == nil {
      controlNet.compile(inputs: [xIn, hint[0], timestep] + c)
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
    var result = controlNet(inputs: xIn, [hint[0], timestep] + c).map { $0.as(of: FloatType.self) }
    if controlMode == .control {
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
