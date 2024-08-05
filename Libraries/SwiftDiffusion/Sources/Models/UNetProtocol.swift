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
    _ timestep: DynamicGraph.Tensor<FloatType>?,
    _ c: [DynamicGraph.Tensor<FloatType>], tokenLengthUncond: Int, tokenLengthCond: Int,
    extraProjection: DynamicGraph.Tensor<FloatType>?,
    injectedControls: [DynamicGraph.Tensor<FloatType>],
    injectedT2IAdapters: [DynamicGraph.Tensor<FloatType>],
    injectedIPAdapters: [DynamicGraph.Tensor<FloatType>],
    tiledDiffusion: TiledConfiguration
  ) -> Bool
  func callAsFunction(
    timestep: Float,
    inputs: DynamicGraph.Tensor<FloatType>, _: DynamicGraph.Tensor<FloatType>?,
    _: [DynamicGraph.Tensor<FloatType>], extraProjection: DynamicGraph.Tensor<FloatType>?,
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
  ) -> DynamicGraph.Tensor<FloatType>
  func decode(_ x: DynamicGraph.Tensor<FloatType>) -> DynamicGraph.Tensor<FloatType>
}

extension UNetProtocol {
  public func timeEmbed(graph: DynamicGraph, batchSize: Int, timestep: Float, version: ModelVersion)
    -> DynamicGraph.Tensor<FloatType>?
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
    case .sd3, .pixart, .auraflow, .flux1:
      return nil
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

extension UNetProtocol {
  public func extractConditions(
    graph: DynamicGraph, index: Int, batchSize: Int, conditions: [DynamicGraph.Tensor<FloatType>],
    version: ModelVersion
  )
    -> [DynamicGraph.Tensor<FloatType>]
  {
    switch version {
    case .kandinsky21, .sdxlBase, .sdxlRefiner, .ssd1b, .svdI2v, .v1, .v2, .wurstchenStageB,
      .wurstchenStageC:
      return conditions
    case .sd3, .auraflow:
      return [conditions[0]]
        + conditions[1..<conditions.count].map {
          let shape = $0.shape
          return $0[(index * batchSize)..<((index + 1) * batchSize), 0..<shape[1], 0..<shape[2]]
            .copied()
        }
    case .flux1:
      return conditions[0..<2]
        + conditions[2..<conditions.count].map {
          let shape = $0.shape
          return $0[(index * batchSize)..<((index + 1) * batchSize), 0..<shape[1], 0..<shape[2]]
            .copied()
        }
    case .pixart:
      var extractedConditions = [conditions[0]]
      let layers = (conditions.count - 3) / 8
      for i in 0..<layers {
        let shape = conditions[1 + i * 8].shape
        extractedConditions.append(contentsOf: [
          conditions[1 + i * 8][index..<(index + 1), 0..<1, 0..<shape[2]].copied(),
          conditions[1 + i * 8 + 1][index..<(index + 1), 0..<1, 0..<shape[2]].copied(),
          conditions[1 + i * 8 + 2][index..<(index + 1), 0..<1, 0..<shape[2]].copied(),
          conditions[1 + i * 8 + 3],
          conditions[1 + i * 8 + 4],
          conditions[1 + i * 8 + 5][index..<(index + 1), 0..<1, 0..<shape[2]].copied(),
          conditions[1 + i * 8 + 6][index..<(index + 1), 0..<1, 0..<shape[2]].copied(),
          conditions[1 + i * 8 + 7][index..<(index + 1), 0..<1, 0..<shape[2]].copied(),
        ])
      }
      let shape = conditions[conditions.count - 2].shape
      extractedConditions.append(contentsOf: [
        conditions[conditions.count - 2][index..<(index + 1), 0..<1, 0..<shape[2]].copied(),
        conditions[conditions.count - 1][index..<(index + 1), 0..<1, 0..<shape[2]].copied(),
      ])
      return extractedConditions
    }
  }
}

public struct UNetFromNNC<FloatType: TensorNumeric & BinaryFloatingPoint>: UNetProtocol {
  var unet: Model? = nil
  var previewer: Model? = nil
  var unetWeightMapper: ModelWeightMapper? = nil
  var timeEmbed: Model? = nil
  var yTileWeightsAndIndexes: [[(weight: Float, index: Int, offset: Int)]]? = nil
  var xTileWeightsAndIndexes: [[(weight: Float, index: Int, offset: Int)]]? = nil
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
    _ timestep: DynamicGraph.Tensor<FloatType>?,
    _ c: [DynamicGraph.Tensor<FloatType>], tokenLengthUncond: Int, tokenLengthCond: Int,
    extraProjection: DynamicGraph.Tensor<FloatType>?,
    injectedControls: [DynamicGraph.Tensor<FloatType>],
    injectedT2IAdapters: [DynamicGraph.Tensor<FloatType>],
    injectedIPAdapters: [DynamicGraph.Tensor<FloatType>],
    tiledDiffusion: TiledConfiguration
  ) -> Bool {
    guard unet == nil else { return true }
    let shape = xT.shape
    let batchSize = shape[0]
    let startHeight = shape[1]
    let startWidth = shape[2]
    let tiledWidth: Int
    let tiledHeight: Int
    let tileScaleFactor: Int
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
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
      tileScaleFactor = 8
      if !lora.isEmpty && rankOfLoRA > 0 && !isLoHa && runLoRASeparatelyIsPreferred
        && canRunLoRASeparately
      {
        unet =
          LoRAUNet(
            batchSize: batchSize, embeddingLength: (tokenLengthUncond, tokenLengthCond),
            startWidth: tiledWidth, startHeight: tiledHeight,
            usesFlashAttention: usesFlashAttention ? .scaleMerged : .none,
            injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
            injectIPAdapterLengths: injectIPAdapterLengths, LoRAConfiguration: configuration
          )
      } else {
        unet =
          UNet(
            batchSize: batchSize, embeddingLength: (tokenLengthUncond, tokenLengthCond),
            startWidth: tiledWidth, startHeight: tiledHeight,
            usesFlashAttention: usesFlashAttention ? .scaleMerged : .none,
            injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
            injectIPAdapterLengths: injectIPAdapterLengths, injectAttentionKV: false,
            outputSpatialAttnInput: false
          ).0
      }
    case .v2:
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
      tileScaleFactor = 8
      if !lora.isEmpty && rankOfLoRA > 0 && !isLoHa && runLoRASeparatelyIsPreferred
        && canRunLoRASeparately
      {
        unet =
          LoRAUNetv2(
            batchSize: batchSize, embeddingLength: (tokenLengthUncond, tokenLengthCond),
            startWidth: tiledWidth, startHeight: tiledHeight, upcastAttention: upcastAttention,
            usesFlashAttention: usesFlashAttention ? .scaleMerged : .none,
            injectControls: injectControls,
            LoRAConfiguration: configuration
          )
      } else {
        unet =
          UNetv2(
            batchSize: batchSize, embeddingLength: (tokenLengthUncond, tokenLengthCond),
            startWidth: tiledWidth, startHeight: tiledHeight, upcastAttention: upcastAttention,
            usesFlashAttention: usesFlashAttention ? .scaleMerged : .none,
            injectControls: injectControls
          ).0
      }
    case .svdI2v:
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
      tileScaleFactor = 8
      (unet, _, unetWeightMapper) =
        UNetXL(
          batchSize: batchSize, startHeight: tiledHeight, startWidth: tiledWidth,
          channels: [320, 640, 1280, 1280],
          inputAttentionRes: [1: [1, 1], 2: [1, 1], 4: [1, 1]], middleAttentionBlocks: 1,
          outputAttentionRes: [1: [1, 1, 1], 2: [1, 1, 1], 4: [1, 1, 1]], embeddingLength: (1, 1),
          injectIPAdapterLengths: [], upcastAttention: ([:], false, [1: [0, 1, 2]]),
          usesFlashAttention: usesFlashAttention ? .scale1 : .none, injectControls: false,
          isTemporalMixEnabled: true, of: FloatType.self
        )
    case .kandinsky21:
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
      tileScaleFactor = 8
      unet = UNetKandinsky(
        batchSize: batchSize, channels: 384, outChannels: 8, channelMult: [1, 2, 3, 4],
        numResBlocks: 3, numHeadChannels: 64, t: 87, startHeight: tiledHeight,
        startWidth: tiledWidth, attentionResolutions: Set([2, 4, 8]),
        usesFlashAttention: usesFlashAttention)
      timeEmbed = timestepEmbedding(prefix: "time_embed", channels: 384 * 4)
    case .sdxlBase:
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
      tileScaleFactor = 8
      if !lora.isEmpty && rankOfLoRA > 0 && !isLoHa && runLoRASeparatelyIsPreferred
        && canRunLoRASeparately
      {
        (unet, unetWeightMapper) =
          LoRAUNetXL(
            batchSize: batchSize, startHeight: tiledHeight, startWidth: tiledWidth,
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
            batchSize: batchSize, startHeight: tiledHeight, startWidth: tiledWidth,
            channels: [320, 640, 1280], inputAttentionRes: [2: [2, 2], 4: [10, 10]],
            middleAttentionBlocks: 10, outputAttentionRes: [2: [2, 2, 2], 4: [10, 10, 10]],
            embeddingLength: (tokenLengthUncond, tokenLengthCond),
            injectIPAdapterLengths: injectIPAdapterLengths, upcastAttention: ([:], false, [:]),
            usesFlashAttention: usesFlashAttention ? .scaleMerged : .none,
            injectControls: injectControls, isTemporalMixEnabled: false, of: FloatType.self
          )
      }
    case .sdxlRefiner:
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
      tileScaleFactor = 8
      if !lora.isEmpty && rankOfLoRA > 0 && !isLoHa && runLoRASeparatelyIsPreferred
        && canRunLoRASeparately
      {
        (unet, unetWeightMapper) =
          LoRAUNetXL(
            batchSize: batchSize, startHeight: tiledHeight, startWidth: tiledWidth,
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
            batchSize: batchSize, startHeight: tiledHeight, startWidth: tiledWidth,
            channels: [384, 768, 1536, 1536], inputAttentionRes: [2: [4, 4], 4: [4, 4]],
            middleAttentionBlocks: 4, outputAttentionRes: [2: [4, 4, 4], 4: [4, 4, 4]],
            embeddingLength: (tokenLengthUncond, tokenLengthCond), injectIPAdapterLengths: [],
            upcastAttention: ([:], false, [:]),
            usesFlashAttention: usesFlashAttention ? .scaleMerged : .none, injectControls: false,
            isTemporalMixEnabled: false, of: FloatType.self
          )
      }
    case .ssd1b:
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
      tileScaleFactor = 8
      if !lora.isEmpty && rankOfLoRA > 0 && !isLoHa && runLoRASeparatelyIsPreferred
        && canRunLoRASeparately
      {
        (unet, unetWeightMapper) =
          LoRAUNetXL(
            batchSize: batchSize, startHeight: tiledHeight, startWidth: tiledWidth,
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
            batchSize: batchSize, startHeight: tiledHeight, startWidth: tiledWidth,
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
      tiledWidth = startWidth
      tiledHeight = startHeight
      tileScaleFactor = 1
      (unet, _) = WurstchenStageC(
        batchSize: batchSize, height: startHeight, width: startWidth,
        t: (tokenLengthUncond + 8, tokenLengthCond + 8),
        usesFlashAttention: usesFlashAttention ? .scaleMerged : .none)
      previewer = WurstchenStageCPreviewer()
    case .wurstchenStageB:
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 16, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 16, startHeight) : startHeight
      tileScaleFactor = 16
      (unet, _) = WurstchenStageB(
        batchSize: batchSize, cIn: 4, height: tiledHeight, width: tiledWidth,
        usesFlashAttention: usesFlashAttention ? .scaleMerged : .none)
    case .sd3:
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
      tileScaleFactor = 8
      if !lora.isEmpty && rankOfLoRA > 0 && !isLoHa && runLoRASeparatelyIsPreferred
        && canRunLoRASeparately
      {
        (_, unet) =
          LoRAMMDiT(
            batchSize: batchSize, t: c[0].shape[1], height: tiledHeight,
            width: tiledWidth, channels: 1536, layers: 24,
            usesFlashAttention: usesFlashAttention ? .scaleMerged : .none,
            LoRAConfiguration: configuration, of: FloatType.self)
      } else {
        (_, unet) =
          MMDiT(
            batchSize: batchSize, t: c[0].shape[1], height: tiledHeight,
            width: tiledWidth, channels: 1536, layers: 24,
            usesFlashAttention: usesFlashAttention ? .scaleMerged : .none, of: FloatType.self)
      }
    case .pixart:
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
      tileScaleFactor = 8
      if !lora.isEmpty && rankOfLoRA > 0 && !isLoHa && runLoRASeparatelyIsPreferred
        && canRunLoRASeparately
      {
        (_, unet) = LoRAPixArt(
          batchSize: batchSize, height: tiledHeight, width: tiledWidth, channels: 1152, layers: 28,
          tokenLength: (tokenLengthUncond, tokenLengthCond), usesFlashAttention: usesFlashAttention,
          LoRAConfiguration: configuration, of: FloatType.self)
      } else {
        (_, unet) = PixArt(
          batchSize: batchSize, height: tiledHeight, width: tiledWidth, channels: 1152, layers: 28,
          tokenLength: (tokenLengthUncond, tokenLengthCond), usesFlashAttention: usesFlashAttention,
          of: FloatType.self)
      }
    case .auraflow:
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
      tileScaleFactor = 8
      (_, unet) = AuraFlow(
        batchSize: batchSize, tokenLength: max(256, max(tokenLengthCond, tokenLengthUncond)),
        height: tiledHeight, width: tiledWidth, channels: 3072, layers: (4, 32),
        usesFlashAttention: usesFlashAttention ? .scaleMerged : .none, of: FloatType.self)
    case .flux1:
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
      tileScaleFactor = 8
      (_, unet) = Flux1(
        batchSize: batchSize, tokenLength: max(256, max(tokenLengthCond, tokenLengthUncond)),
        height: tiledHeight, width: tiledWidth, channels: 3072, layers: (19, 38),
        usesFlashAttention: usesFlashAttention ? .scaleMerged : .none)
    }
    // Need to assign version now such that sliceInputs will have the correct version.
    self.version = version
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
      case .v2, .sd3, .pixart, .auraflow, .flux1, .kandinsky21, .svdI2v, .wurstchenStageC,
        .wurstchenStageB:
        fatalError()
      }
    }
    var inputs = [DynamicGraph.Tensor<FloatType>]()
    if let extraProjection = extraProjection {
      inputs.append(extraProjection.reshaped(.WC(batchSize, 384 * 4)))
    } else if let timestep = timestep {
      inputs.append(timestep)
    }
    inputs.append(contentsOf: c)
    if injectControls {
      inputs.append(contentsOf: injectedControls)
    }
    if injectT2IAdapters {
      inputs.append(contentsOf: injectedT2IAdapters)
    }
    let tileOverlap = min(
      min(
        tiledDiffusion.tileOverlap * tileScaleFactor,
        Int((Double(tiledHeight / 3) / Double(tileScaleFactor)).rounded(.down)) * tileScaleFactor),
      Int((Double(tiledWidth / 3) / Double(tileScaleFactor)).rounded(.down)) * tileScaleFactor)
    let yTiles =
      (startHeight - tileOverlap * 2 + (tiledHeight - tileOverlap * 2) - 1)
      / (tiledHeight - tileOverlap * 2)
    let xTiles =
      (startWidth - tileOverlap * 2 + (tiledWidth - tileOverlap * 2) - 1)
      / (tiledWidth - tileOverlap * 2)
    if startWidth > tiledWidth || startHeight > tiledHeight {
      let inputs = sliceInputs(
        inputs, originalShape: shape, xyTiles: xTiles * yTiles, index: 0, inputStartYPad: 0,
        inputEndYPad: tiledHeight, inputStartXPad: 0, inputEndXPad: tiledWidth)
      unet.compile(
        inputs: [xT[0..<shape[0], 0..<tiledHeight, 0..<tiledWidth, 0..<shape[3]]] + inputs)
    } else {
      unet.compile(inputs: [xT] + inputs)
    }
    if let timeEmbed = timeEmbed, let timestep = timestep {
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
    case .sd3, .pixart, .auraflow, .flux1:
      modelKey = "dit"
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
            case .sd3:
              return [Int: Int](
                uniqueKeysWithValues: (0..<24).map {
                  return ($0, $0)
                })
            case .pixart:
              return [Int: Int](
                uniqueKeysWithValues: (0..<28).map {
                  return ($0, $0)
                })
            case .auraflow, .flux1:
              fatalError()
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
    self.unet = unet
    if startWidth > tiledWidth || startHeight > tiledHeight {
      (xTileWeightsAndIndexes, yTileWeightsAndIndexes) = xyTileWeightsAndIndexes(
        width: startWidth, height: startHeight, xTiles: xTiles, yTiles: yTiles,
        tileSize: (width: tiledWidth, height: tiledHeight), tileOverlap: tileOverlap)
    }
    return true
  }

  private func sliceInputs(
    _ inputs: [DynamicGraph.Tensor<FloatType>], originalShape: TensorShape, xyTiles: Int,
    index: Int, inputStartYPad: Int, inputEndYPad: Int, inputStartXPad: Int, inputEndXPad: Int
  ) -> [DynamicGraph.Tensor<FloatType>] {
    return inputs.enumerated().map {
      // For FLUX.1, if it is the first one, we need to handle its slicing (rotary encoding).
      if $0.0 == 0 && version == .flux1 {
        let shape = $0.1.shape
        let tokenLength = shape[1] - (originalShape[1] / 2) * (originalShape[2] / 2)
        let graph = $0.1.graph
        let tokenEncoding = $0.1[0..<shape[0], 0..<tokenLength, 0..<shape[2], 0..<shape[3]].copied()
        let imageEncoding = $0.1[0..<shape[0], tokenLength..<shape[1], 0..<shape[2], 0..<shape[3]]
          .copied().reshaped(.NHWC(shape[0], originalShape[1] / 2, originalShape[2] / 2, shape[3]))
        let h = inputEndYPad / 2 - inputStartYPad / 2
        let w = inputEndXPad / 2 - inputStartXPad / 2
        let sliceEncoding = imageEncoding[
          0..<shape[0], (inputStartYPad / 2)..<(inputEndYPad / 2),
          (inputStartXPad / 2)..<(inputEndXPad / 2), 0..<shape[3]
        ].copied().reshaped(.NHWC(shape[0], h * w, 1, shape[3]))
        var finalEncoding = graph.variable(
          $0.1.kind, .NHWC(shape[0], h * w + tokenLength, 1, shape[3]), of: FloatType.self)
        finalEncoding[0..<shape[0], 0..<tokenLength, 0..<1, 0..<shape[3]] = tokenEncoding
        finalEncoding[0..<shape[0], tokenLength..<(h * w), 0..<1, 0..<shape[3]] = sliceEncoding
        return finalEncoding
      }
      let shape = $0.1.shape
      guard shape.count == 4 else { return $0.1 }
      if shape[0] == originalShape[0] {
        // This is likely a one with xT same shape, from Wurstchen B model.
        if version == .wurstchenStageB || version == .wurstchenStageC {
          if (originalShape[1] % shape[1]) == 0 && (originalShape[2] % shape[2]) == 0
            && ((originalShape[1] / shape[1]) == (originalShape[2] / shape[2]))
          {
            // This may have issues with 3x3 convolution downsampling with strides, but luckily in UNet we deal with, these don't exist.
            let scaleFactor = originalShape[1] / shape[1]
            return $0.1[
              0..<shape[0], (inputStartYPad / scaleFactor)..<(inputEndYPad / scaleFactor),
              (inputStartXPad / scaleFactor)..<(inputEndXPad / scaleFactor), 0..<shape[3]
            ].copied()
          }
        }
      } else if originalShape[0] * xyTiles == shape[0] {
        return $0.1[
          (index * originalShape[0])..<((index + 1) * originalShape[0]), 0..<shape[1], 0..<shape[2],
          0..<shape[3]
        ].copied()
      }
      return $0.1
    }
  }

  private func internalDiffuse(
    xyTiles: Int, index: Int, inputStartYPad: Int, inputEndYPad: Int, inputStartXPad: Int,
    inputEndXPad: Int, xT: DynamicGraph.Tensor<FloatType>, inputs: [DynamicGraph.Tensor<FloatType>],
    injectedControlsAndAdapters: (
      _ xT: DynamicGraph.Tensor<FloatType>, _ inputStartYPad: Int, _ inputEndYPad: Int,
      _ inputStartXPad: Int, _ inputEndXPad: Int, _ existingControlNets: inout [Model?]
    ) -> (
      injectedControls: [DynamicGraph.Tensor<FloatType>],
      injectedT2IAdapters: [DynamicGraph.Tensor<FloatType>]
    ), controlNets: inout [Model?]
  ) -> DynamicGraph.Tensor<FloatType> {
    let shape = xT.shape
    let xT = xT[
      0..<shape[0], inputStartYPad..<inputEndYPad, inputStartXPad..<inputEndXPad, 0..<shape[3]
    ].copied()
    // Need to rework the shape. For Wurstchen B, we need to slice them up.
    // For ControlNet, we already sliced them up into batch dimension, now need to extract them out.
    let (injectedControls, injectedT2IAdapters) = injectedControlsAndAdapters(
      xT, inputStartYPad, inputEndYPad, inputStartXPad, inputEndXPad, &controlNets)
    let inputs = sliceInputs(
      inputs + injectedControls + injectedT2IAdapters, originalShape: shape, xyTiles: xyTiles,
      index: index, inputStartYPad: inputStartYPad,
      inputEndYPad: inputEndYPad, inputStartXPad: inputStartXPad, inputEndXPad: inputEndXPad)
    return unet!(inputs: xT, inputs)[0].as(of: FloatType.self)
  }

  private func tiledDiffuse(
    tiledDiffusion: TiledConfiguration, xT: DynamicGraph.Tensor<FloatType>,
    inputs: [DynamicGraph.Tensor<FloatType>],
    injectedControlsAndAdapters: (
      _ xT: DynamicGraph.Tensor<FloatType>, _ inputStartYPad: Int, _ inputEndYPad: Int,
      _ inputStartXPad: Int, _ inputEndXPad: Int, _ existingControlNets: inout [Model?]
    ) -> (
      injectedControls: [DynamicGraph.Tensor<FloatType>],
      injectedT2IAdapters: [DynamicGraph.Tensor<FloatType>]
    ), controlNets: inout [Model?]
  ) -> DynamicGraph.Tensor<FloatType> {
    guard let xTileWeightsAndIndexes = xTileWeightsAndIndexes,
      let yTileWeightsAndIndexes = yTileWeightsAndIndexes
    else {
      let (injectedControls, injectedT2IAdapters) = injectedControlsAndAdapters(
        xT, 0, 0, 0, 0, &controlNets)
      return unet!(inputs: xT, inputs + injectedControls + injectedT2IAdapters)[0].as(
        of: FloatType.self)
    }
    let shape = xT.shape
    let startHeight = shape[1]
    let startWidth = shape[2]
    let tileScaleFactor = version == .wurstchenStageB ? 16 : 8
    let tiledWidth =
      tiledDiffusion.isEnabled
      ? min(tiledDiffusion.tileSize.width * tileScaleFactor, startWidth) : startWidth
    let tiledHeight =
      tiledDiffusion.isEnabled
      ? min(tiledDiffusion.tileSize.height * tileScaleFactor, startHeight) : startHeight
    let tileOverlap = min(
      min(
        tiledDiffusion.tileOverlap * tileScaleFactor,
        Int((Double(tiledHeight / 3) / Double(tileScaleFactor)).rounded(.down)) * tileScaleFactor),
      Int((Double(tiledWidth / 3) / Double(tileScaleFactor)).rounded(.down)) * tileScaleFactor)
    let yTiles =
      (startHeight - tileOverlap * 2 + (tiledHeight - tileOverlap * 2) - 1)
      / (tiledHeight - tileOverlap * 2)
    let xTiles =
      (startWidth - tileOverlap * 2 + (tiledWidth - tileOverlap * 2) - 1)
      / (tiledWidth - tileOverlap * 2)
    var et = [DynamicGraph.Tensor<FloatType>]()
    for y in 0..<yTiles {
      let yOfs = y * (tiledHeight - tileOverlap * 2) + (y > 0 ? tileOverlap : 0)
      let (inputStartYPad, inputEndYPad) = paddedTileStartAndEnd(
        iOfs: yOfs, length: startHeight, tileSize: tiledHeight, tileOverlap: tileOverlap)
      for x in 0..<xTiles {
        let xOfs = x * (tiledWidth - tileOverlap * 2) + (x > 0 ? tileOverlap : 0)
        let (inputStartXPad, inputEndXPad) = paddedTileStartAndEnd(
          iOfs: xOfs, length: startWidth, tileSize: tiledWidth, tileOverlap: tileOverlap)
        et.append(
          internalDiffuse(
            xyTiles: xTiles * yTiles, index: y * xTiles + x, inputStartYPad: inputStartYPad,
            inputEndYPad: inputEndYPad, inputStartXPad: inputStartXPad, inputEndXPad: inputEndXPad,
            xT: xT, inputs: inputs, injectedControlsAndAdapters: injectedControlsAndAdapters,
            controlNets: &controlNets))
      }
    }
    let etRawValues = et.map { $0.rawValue.toCPU() }
    let channels = etRawValues[0].shape[3]
    var etRaw = Tensor<FloatType>(.CPU, .NHWC(shape[0], startHeight, startWidth, channels))
    etRaw.withUnsafeMutableBytes {
      guard var fp = $0.baseAddress?.assumingMemoryBound(to: FloatType.self) else { return }
      for b in 0..<shape[0] {
        for j in 0..<startHeight {
          let yWeightAndIndex = yTileWeightsAndIndexes[j]
          for i in 0..<startWidth {
            let xWeightAndIndex = xTileWeightsAndIndexes[i]
            for k in 0..<channels {
              fp[k] = 0
            }
            for y in yWeightAndIndex {
              for x in xWeightAndIndex {
                let weight = FloatType(x.weight * y.weight)
                let index = y.index * xTiles + x.index
                let tensor = etRawValues[index]
                tensor.withUnsafeBytes {
                  guard var v = $0.baseAddress?.assumingMemoryBound(to: FloatType.self) else {
                    return
                  }
                  // Note that while result is outputChannels, this is padded to 4 i.e. channels.
                  v =
                    v + b * tiledHeight * tiledWidth * channels + x.offset * channels + y.offset
                    * tiledWidth * channels
                  for k in 0..<channels {
                    fp[k] += v[k] * weight
                  }
                }
              }
            }
            fp += channels
          }
        }
      }
    }
    let graph = xT.graph
    return graph.variable(etRaw.toGPU(0))
  }

  public func callAsFunction(
    timestep _: Float,
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
    tiledDiffusion: TiledConfiguration, controlNets: inout [Model?]
  ) -> DynamicGraph.Tensor<FloatType> {
    if let extraProjection = extraProjection, let timeEmbed = timeEmbed, let timestep = timestep {
      let batchSize = xT.shape[0]
      var embGPU = timeEmbed(inputs: timestep)[0].as(of: FloatType.self)
      embGPU = embGPU + extraProjection.reshaped(.NC(batchSize, 384 * 4))
      if tiledDiffusion.isEnabled {
        return tiledDiffuse(
          tiledDiffusion: tiledDiffusion, xT: xT, inputs: [embGPU, c[0]],
          injectedControlsAndAdapters: injectedControlsAndAdapters, controlNets: &controlNets)
      } else {
        return unet!(inputs: xT, embGPU, c[0])[0].as(of: FloatType.self)
      }
    }
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
      case .v2, .sd3, .pixart, .auraflow, .flux1, .kandinsky21, .svdI2v, .wurstchenStageC,
        .wurstchenStageB:
        fatalError()
      }
    }
    if tiledDiffusion.isEnabled {
      return tiledDiffuse(
        tiledDiffusion: tiledDiffusion, xT: xT, inputs: (timestep.map { [$0] } ?? []) + c,
        injectedControlsAndAdapters: injectedControlsAndAdapters, controlNets: &controlNets)
    } else {
      let (injectedControls, injectedT2IAdapters) = injectedControlsAndAdapters(
        xT, 0, 0, 0, 0, &controlNets)
      return unet!(
        inputs: xT, (timestep.map { [$0] } ?? []) + c + injectedControls + injectedT2IAdapters)[0]
        .as(of: FloatType.self)
    }
  }

  public func decode(_ x: DynamicGraph.Tensor<FloatType>) -> DynamicGraph.Tensor<FloatType> {
    switch version {
    case .wurstchenStageC:
      if let previewer = previewer {
        return previewer(inputs: x)[0].as(of: FloatType.self)
      }
      return x
    case .v1, .v2, .sd3, .pixart, .auraflow, .flux1, .sdxlBase, .sdxlRefiner, .ssd1b, .svdI2v,
      .kandinsky21,
      .wurstchenStageB:
      return x
    }
  }
}
