import Collections
import NNC

public protocol UNetProtocol {
  associatedtype FloatType: TensorNumeric & BinaryFloatingPoint
  init()
  var isLoaded: Bool { get }
  func unloadResources()
  var version: ModelVersion { get }
  var modelAndWeightMapper: (Model, ModelWeightMapper)? { get }
  mutating func compileModel(
    filePath: String, externalOnDemand: Bool, version: ModelVersion, upcastAttention: Bool,
    usesFlashAttention: Bool, injectControls: Bool, injectT2IAdapters: Bool,
    injectIPAdapterLengths: [Int], lora: [LoRAConfiguration],
    is8BitModel: Bool, canRunLoRASeparately: Bool, inputs xT: DynamicGraph.Tensor<FloatType>,
    _ timestep: DynamicGraph.Tensor<FloatType>,
    _ c: [DynamicGraph.Tensor<FloatType>], tokenLengthUncond: Int, tokenLengthCond: Int,
    extraProjection: DynamicGraph.Tensor<FloatType>?,
    injectedControls: [DynamicGraph.Tensor<FloatType>],
    injectedT2IAdapters: [DynamicGraph.Tensor<FloatType>],
    injectedIPAdapters: [DynamicGraph.Tensor<FloatType>]
  ) -> Bool
  func callAsFunction(
    timestep: Float,
    inputs: DynamicGraph.Tensor<FloatType>, _: DynamicGraph.Tensor<FloatType>,
    _: [DynamicGraph.Tensor<FloatType>], extraProjection: DynamicGraph.Tensor<FloatType>?,
    injectedControls: [DynamicGraph.Tensor<FloatType>],
    injectedT2IAdapters: [DynamicGraph.Tensor<FloatType>],
    injectedIPAdapters: [DynamicGraph.Tensor<FloatType>]
  ) -> DynamicGraph.Tensor<FloatType>
  func decode(_ x: DynamicGraph.Tensor<FloatType>) -> DynamicGraph.Tensor<FloatType>
}

extension UNetProtocol {
  public func timeEmbed(graph: DynamicGraph, batchSize: Int, timestep: Float, version: ModelVersion)
    -> DynamicGraph.Tensor<FloatType>
  {
    switch version {
    case .v1, .v2, .sdxlBase, .sdxlRefiner, .kandinsky21, .ssd1b, .svdI2v:
      let timeEmbeddingSize = version == .kandinsky21 || version == .sdxlRefiner ? 384 : 320
      return graph.variable(
        Tensor<FloatType>(
          from: timeEmbedding(
            timestep: timestep, batchSize: batchSize,
            embeddingSize: timeEmbeddingSize,
            maxPeriod: 10_000)
        ).toGPU(0))
    case .wurstchenStageC:
      let rTimeEmbed = rEmbedding(
        timesteps: timestep, batchSize: batchSize, embeddingSize: 64, maxPeriod: 10_000)
      let rZeros = rEmbedding(
        timesteps: 0, batchSize: batchSize, embeddingSize: 64, maxPeriod: 10_000)
      var rEmbed = Tensor<Float>(.CPU, .WC(batchSize, 192))
      rEmbed[0..<batchSize, 0..<64] = rTimeEmbed
      rEmbed[0..<batchSize, 64..<128] = rZeros
      rEmbed[0..<batchSize, 128..<192] = rZeros
      return graph.variable(Tensor<FloatType>(from: rEmbed).toGPU(0))
    case .wurstchenStageB:
      let rTimeEmbed = rEmbedding(
        timesteps: timestep, batchSize: batchSize, embeddingSize: 64, maxPeriod: 10_000)
      let rZeros = rEmbedding(
        timesteps: 0, batchSize: batchSize, embeddingSize: 64, maxPeriod: 10_000)
      var rEmbed = Tensor<Float>(.CPU, .WC(batchSize, 128))
      rEmbed[0..<batchSize, 0..<64] = rTimeEmbed
      rEmbed[0..<batchSize, 64..<128] = rZeros
      return graph.variable(Tensor<FloatType>(from: rEmbed).toGPU(0))
    }
  }
}

public struct UNetFromNNC<FloatType: TensorNumeric & BinaryFloatingPoint>: UNetProtocol {
  var unet: Model? = nil
  var previewer: Model? = nil
  var unetWeightMapper: ModelWeightMapper? = nil
  var timeEmbed: Model? = nil
  public private(set) var version: ModelVersion = .v1
  public init() {}
  public var isLoaded: Bool { unet != nil }
  public func unloadResources() {}
}

extension UNetFromNNC {
  public var modelAndWeightMapper: (Model, ModelWeightMapper)? {
    guard let unet = unet, let unetWeightMapper = unetWeightMapper else { return nil }
    return (unet, unetWeightMapper)
  }
  public mutating func compileModel(
    filePath: String, externalOnDemand: Bool, version: ModelVersion, upcastAttention: Bool,
    usesFlashAttention: Bool, injectControls: Bool, injectT2IAdapters: Bool,
    injectIPAdapterLengths: [Int], lora: [LoRAConfiguration],
    is8BitModel: Bool, canRunLoRASeparately: Bool, inputs xT: DynamicGraph.Tensor<FloatType>,
    _ timestep: DynamicGraph.Tensor<FloatType>,
    _ c: [DynamicGraph.Tensor<FloatType>], tokenLengthUncond: Int, tokenLengthCond: Int,
    extraProjection: DynamicGraph.Tensor<FloatType>?,
    injectedControls: [DynamicGraph.Tensor<FloatType>],
    injectedT2IAdapters: [DynamicGraph.Tensor<FloatType>],
    injectedIPAdapters: [DynamicGraph.Tensor<FloatType>]
  ) -> Bool {
    guard unet == nil else { return true }
    let batchSize = xT.shape[0]
    let startHeight = xT.shape[1]
    let startWidth = xT.shape[2]
    let graph = xT.graph
    let unet: Model
    let lora = Array(
      (OrderedDictionary<String, LoRAConfiguration>(
        lora.filter({ $0.version == version }).map {
          ($0.file, $0)
        }
      ) {
        LoRAConfiguration(
          file: $0.file, weight: $0.weight + $1.weight, version: $0.version, isLoHa: $0.isLoHa,
          modifier: $0.modifier)
      })
      .values
    ).filter { $0.weight != 0 }
    let (rankOfLoRA, filesRequireMerge) = LoRALoader<FloatType>.rank(
      graph, of: lora.map { $0.file })
    let isLoHa = lora.contains { $0.isLoHa }
    let configuration = LoRANetworkConfiguration(rank: rankOfLoRA, scale: 1, highPrecision: false)
    let runLoRASeparatelyIsPreferred = is8BitModel || externalOnDemand
    switch version {
    case .v1:
      if !lora.isEmpty && rankOfLoRA > 0 && !isLoHa && runLoRASeparatelyIsPreferred
        && canRunLoRASeparately
      {
        unet =
          LoRAUNet(
            batchSize: batchSize, embeddingLength: (tokenLengthUncond, tokenLengthCond),
            startWidth: startWidth, startHeight: startHeight,
            usesFlashAttention: usesFlashAttention ? .scaleMerged : .none,
            injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
            injectIPAdapterLengths: injectIPAdapterLengths, LoRAConfiguration: configuration
          )
      } else {
        unet =
          UNet(
            batchSize: batchSize, embeddingLength: (tokenLengthUncond, tokenLengthCond),
            startWidth: startWidth, startHeight: startHeight,
            usesFlashAttention: usesFlashAttention ? .scaleMerged : .none,
            injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
            injectIPAdapterLengths: injectIPAdapterLengths
          ).0
      }
    case .v2:
      if !lora.isEmpty && rankOfLoRA > 0 && !isLoHa && runLoRASeparatelyIsPreferred
        && canRunLoRASeparately
      {
        unet =
          LoRAUNetv2(
            batchSize: batchSize, embeddingLength: (tokenLengthUncond, tokenLengthCond),
            startWidth: startWidth, startHeight: startHeight, upcastAttention: upcastAttention,
            usesFlashAttention: usesFlashAttention ? .scaleMerged : .none,
            injectControls: injectControls,
            LoRAConfiguration: configuration
          )
      } else {
        unet =
          UNetv2(
            batchSize: batchSize, embeddingLength: (tokenLengthUncond, tokenLengthCond),
            startWidth: startWidth, startHeight: startHeight, upcastAttention: upcastAttention,
            usesFlashAttention: usesFlashAttention ? .scaleMerged : .none,
            injectControls: injectControls
          ).0
      }
    case .svdI2v:
      (unet, _, unetWeightMapper) =
        UNetXL(
          batchSize: batchSize, startHeight: startHeight, startWidth: startWidth,
          channels: [320, 640, 1280, 1280],
          inputAttentionRes: [1: [1, 1], 2: [1, 1], 4: [1, 1]], middleAttentionBlocks: 1,
          outputAttentionRes: [1: [1, 1, 1], 2: [1, 1, 1], 4: [1, 1, 1]], embeddingLength: (1, 1),
          injectIPAdapterLengths: [], upcastAttention: ([:], false, [1: [0, 1, 2]]),
          usesFlashAttention: usesFlashAttention ? .scale1 : .none, injectControls: false,
          isTemporalMixEnabled: true, of: FloatType.self
        )
    case .kandinsky21:
      unet = UNetKandinsky(
        batchSize: batchSize, channels: 384, outChannels: 8, channelMult: [1, 2, 3, 4],
        numResBlocks: 3, numHeadChannels: 64, t: 87, startHeight: startHeight,
        startWidth: startWidth, attentionResolutions: Set([2, 4, 8]),
        usesFlashAttention: usesFlashAttention)
      timeEmbed = timestepEmbedding(prefix: "time_embed", channels: 384 * 4)
    case .sdxlBase:
      if !lora.isEmpty && rankOfLoRA > 0 && !isLoHa && runLoRASeparatelyIsPreferred
        && canRunLoRASeparately
      {
        (unet, unetWeightMapper) =
          LoRAUNetXL(
            batchSize: batchSize, startHeight: startHeight, startWidth: startWidth,
            channels: [320, 640, 1280], inputAttentionRes: [2: [2, 2], 4: [10, 10]],
            middleAttentionBlocks: 10, outputAttentionRes: [2: [2, 2, 2], 4: [10, 10, 10]],
            embeddingLength: (tokenLengthUncond, tokenLengthCond),
            injectIPAdapterLengths: injectIPAdapterLengths, upcastAttention: ([:], false, [:]),
            usesFlashAttention: usesFlashAttention ? .scaleMerged : .none,
            injectControls: injectControls, LoRAConfiguration: configuration
          )
      } else {
        (unet, _, unetWeightMapper) =
          UNetXL(
            batchSize: batchSize, startHeight: startHeight, startWidth: startWidth,
            channels: [320, 640, 1280], inputAttentionRes: [2: [2, 2], 4: [10, 10]],
            middleAttentionBlocks: 10, outputAttentionRes: [2: [2, 2, 2], 4: [10, 10, 10]],
            embeddingLength: (tokenLengthUncond, tokenLengthCond),
            injectIPAdapterLengths: injectIPAdapterLengths, upcastAttention: ([:], false, [:]),
            usesFlashAttention: usesFlashAttention ? .scaleMerged : .none,
            injectControls: injectControls, isTemporalMixEnabled: false, of: FloatType.self
          )
      }
    case .sdxlRefiner:
      if !lora.isEmpty && rankOfLoRA > 0 && !isLoHa && runLoRASeparatelyIsPreferred
        && canRunLoRASeparately
      {
        (unet, unetWeightMapper) =
          LoRAUNetXL(
            batchSize: batchSize, startHeight: startHeight, startWidth: startWidth,
            channels: [384, 768, 1536, 1536], inputAttentionRes: [2: [4, 4], 4: [4, 4]],
            middleAttentionBlocks: 4, outputAttentionRes: [2: [4, 4, 4], 4: [4, 4, 4]],
            embeddingLength: (tokenLengthUncond, tokenLengthCond), injectIPAdapterLengths: [],
            upcastAttention: ([:], false, [:]),
            usesFlashAttention: usesFlashAttention ? .scaleMerged : .none,
            injectControls: false, LoRAConfiguration: configuration
          )
      } else {
        (unet, _, unetWeightMapper) =
          UNetXL(
            batchSize: batchSize, startHeight: startHeight, startWidth: startWidth,
            channels: [384, 768, 1536, 1536], inputAttentionRes: [2: [4, 4], 4: [4, 4]],
            middleAttentionBlocks: 4, outputAttentionRes: [2: [4, 4, 4], 4: [4, 4, 4]],
            embeddingLength: (tokenLengthUncond, tokenLengthCond), injectIPAdapterLengths: [],
            upcastAttention: ([:], false, [:]),
            usesFlashAttention: usesFlashAttention ? .scaleMerged : .none, injectControls: false,
            isTemporalMixEnabled: false, of: FloatType.self
          )
      }
    case .ssd1b:
      if !lora.isEmpty && rankOfLoRA > 0 && !isLoHa && runLoRASeparatelyIsPreferred
        && canRunLoRASeparately
      {
        (unet, unetWeightMapper) =
          LoRAUNetXL(
            batchSize: batchSize, startHeight: startHeight, startWidth: startWidth,
            channels: [320, 640, 1280], inputAttentionRes: [2: [2, 2], 4: [4, 4]],
            middleAttentionBlocks: 0, outputAttentionRes: [2: [2, 1, 1], 4: [4, 4, 10]],
            embeddingLength: (tokenLengthUncond, tokenLengthCond),
            injectIPAdapterLengths: injectIPAdapterLengths,
            upcastAttention: ([:], false, [:]),
            usesFlashAttention: usesFlashAttention ? .scale1 : .none,
            injectControls: false, LoRAConfiguration: configuration
          )
      } else {
        (unet, _, unetWeightMapper) =
          UNetXL(
            batchSize: batchSize, startHeight: startHeight, startWidth: startWidth,
            channels: [320, 640, 1280], inputAttentionRes: [2: [2, 2], 4: [4, 4]],
            middleAttentionBlocks: 0, outputAttentionRes: [2: [2, 1, 1], 4: [4, 4, 10]],
            embeddingLength: (tokenLengthUncond, tokenLengthCond),
            injectIPAdapterLengths: injectIPAdapterLengths,
            upcastAttention: ([:], false, [:]),
            usesFlashAttention: usesFlashAttention ? .scale1 : .none, injectControls: false,
            isTemporalMixEnabled: false, of: FloatType.self
          )
      }
    case .wurstchenStageC:
      unet = WurstchenStageC(
        batchSize: batchSize, height: startHeight, width: startWidth,
        t: (tokenLengthUncond + 8, tokenLengthCond + 8),
        usesFlashAttention: usesFlashAttention ? .scaleMerged : .none)
      previewer = WurstchenStageCPreviewer()
    case .wurstchenStageB:
      unet = WurstchenStageB(
        batchSize: batchSize, cIn: 4, height: startHeight, width: startWidth,
        usesFlashAttention: usesFlashAttention ? .scaleMerged : .none)
    }
    var c = c
    if injectedIPAdapters.count > 0 {
      switch version {
      case .v1:
        let injectIPAdapters = injectedIPAdapters.count / 32
        var newC = [c[0]]
        for i in stride(from: 0, to: 32, by: 2) {
          for j in 0..<injectIPAdapters {
            newC.append(injectedIPAdapters[i + j * 32])  // ip_k
            newC.append(injectedIPAdapters[i + 1 + j * 32])  // ip_v
          }
        }
        c = newC
      case .sdxlBase, .sdxlRefiner, .ssd1b:
        precondition(injectedIPAdapters.count % (c.count - 1) == 0)
        precondition((c.count - 1) % 2 == 0)
        let injectIPAdapters = injectedIPAdapters.count / (c.count - 1)
        var newC = [c[0]]
        for i in stride(from: 0, to: c.count - 1, by: 2) {
          newC.append(c[i + 1])  // k
          newC.append(c[i + 2])  // v
          for j in 0..<injectIPAdapters {
            newC.append(injectedIPAdapters[i + j * (c.count - 1)])  // ip_k
            newC.append(injectedIPAdapters[i + 1 + j * (c.count - 1)])  // ip_v
          }
        }
        c = newC
      case .v2, .kandinsky21, .svdI2v, .wurstchenStageC, .wurstchenStageB:
        fatalError()
      }
    }
    var inputs = [xT, extraProjection?.reshaped(.WC(batchSize, 384 * 4)) ?? timestep] + c
    if injectControls {
      inputs.append(contentsOf: injectedControls)
    }
    if injectT2IAdapters {
      inputs.append(contentsOf: injectedT2IAdapters)
    }
    unet.compile(inputs: inputs)
    if let timeEmbed = timeEmbed {
      timeEmbed.compile(inputs: timestep)
    }
    let modelKey: String
    switch version {
    case .v1, .v2, .sdxlBase, .sdxlRefiner, .kandinsky21, .ssd1b, .svdI2v:
      modelKey = "unet"
    case .wurstchenStageB:
      modelKey = "stage_b"
    case .wurstchenStageC:
      modelKey = "stage_c"
    }
    let externalData: DynamicGraph.Store.Codec =
      externalOnDemand ? .externalOnDemand : .externalData
    graph.openStore(
      filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
    ) { store in
      if !lora.isEmpty && version != .kandinsky21 {
        if !isLoHa && runLoRASeparatelyIsPreferred && rankOfLoRA > 0 && canRunLoRASeparately {
          let mapping: [Int: Int] = {
            switch version {
            case .sdxlBase:
              return LoRAMapping.SDUNetXLBase
            case .sdxlRefiner:
              return LoRAMapping.SDUNetXLRefiner
            case .ssd1b:
              return LoRAMapping.SDUNetXLSSD1B
            case .v1, .v2:
              return LoRAMapping.SDUNet
            case .kandinsky21, .svdI2v, .wurstchenStageC, .wurstchenStageB:
              fatalError()
            }
          }()
          LoRALoader<FloatType>.openStore(graph, lora: lora) { loader in
            store.read(modelKey, model: unet, codec: [.jit, .q6p, .q8p, .ezm7, externalData]) {
              name, dataType, format, shape in
              return loader.concatenateLoRA(
                graph, LoRAMapping: mapping, filesRequireMerge: filesRequireMerge, name: name,
                store: store, dataType: dataType, format: format, shape: shape)
            }
          }
        } else {
          LoRALoader<FloatType>.openStore(graph, lora: lora) { loader in
            store.read(modelKey, model: unet, codec: [.jit, .q6p, .q8p, .ezm7, externalData]) {
              name, _, _, shape in
              return loader.mergeLoRA(graph, name: name, store: store, shape: shape)
            }
          }
        }
      } else {
        store.read(modelKey, model: unet, codec: [.jit, .q6p, .q8p, .ezm7, externalData])
      }
      if let timeEmbed = timeEmbed {
        store.read("time_embed", model: timeEmbed, codec: [.q6p, .q8p, .ezm7, .externalData])
      }
      if let previewer = previewer {
        previewer.compile(inputs: xT)
        store.read("previewer", model: previewer, codec: [.q6p, .q8p, .ezm7, .externalData])
      }
    }
    self.version = version
    self.unet = unet
    return true
  }

  public func callAsFunction(
    timestep _: Float,
    inputs xT: DynamicGraph.Tensor<FloatType>, _ timestep: DynamicGraph.Tensor<FloatType>,
    _ c: [DynamicGraph.Tensor<FloatType>], extraProjection: DynamicGraph.Tensor<FloatType>?,
    injectedControls: [DynamicGraph.Tensor<FloatType>],
    injectedT2IAdapters: [DynamicGraph.Tensor<FloatType>],
    injectedIPAdapters: [DynamicGraph.Tensor<FloatType>]
  ) -> DynamicGraph.Tensor<FloatType> {
    if let extraProjection = extraProjection, let timeEmbed = timeEmbed {
      let batchSize = xT.shape[0]
      var embGPU = timeEmbed(inputs: timestep)[0].as(of: FloatType.self)
      embGPU = embGPU + extraProjection.reshaped(.NC(batchSize, 384 * 4))
      return unet!(inputs: xT, embGPU, c[0])[0].as(of: FloatType.self)
    }
    if injectedControls.count > 0 || injectedT2IAdapters.count > 0 || injectedIPAdapters.count > 0 {
      // Interleaving injectedAdapters with c.
      var c = c
      if injectedIPAdapters.count > 0 {
        switch version {
        case .v1:
          let injectIPAdapters = injectedIPAdapters.count / 32
          var newC = [c[0]]
          for i in stride(from: 0, to: 32, by: 2) {
            for j in 0..<injectIPAdapters {
              newC.append(injectedIPAdapters[i + j * 32])  // ip_k
              newC.append(injectedIPAdapters[i + 1 + j * 32])  // ip_v
            }
          }
          c = newC
        case .sdxlBase, .sdxlRefiner, .ssd1b:
          precondition(injectedIPAdapters.count % (c.count - 1) == 0)
          precondition((c.count - 1) % 2 == 0)
          let injectIPAdapters = injectedIPAdapters.count / (c.count - 1)
          var newC = [c[0]]
          for i in stride(from: 0, to: c.count - 1, by: 2) {
            newC.append(c[i + 1])  // k
            newC.append(c[i + 2])  // v
            for j in 0..<injectIPAdapters {
              newC.append(injectedIPAdapters[i + j * (c.count - 1)])  // ip_k
              newC.append(injectedIPAdapters[i + 1 + j * (c.count - 1)])  // ip_v
            }
          }
          c = newC
        case .v2, .kandinsky21, .svdI2v, .wurstchenStageC, .wurstchenStageB:
          fatalError()
        }
      }
      var inputs = [timestep] + c
      inputs.append(contentsOf: injectedControls)
      inputs.append(contentsOf: injectedT2IAdapters)
      return unet!(inputs: xT, inputs)[0].as(of: FloatType.self)
    } else {
      return unet!(inputs: xT, [timestep] + c)[0].as(of: FloatType.self)
    }
  }

  public func decode(_ x: DynamicGraph.Tensor<FloatType>) -> DynamicGraph.Tensor<FloatType> {
    switch version {
    case .wurstchenStageC:
      if let previewer = previewer {
        return previewer(inputs: x)[0].as(of: FloatType.self)
      }
      return x
    case .v1, .v2, .sdxlBase, .sdxlRefiner, .ssd1b, .svdI2v, .kandinsky21, .wurstchenStageB:
      return x
    }
  }
}
