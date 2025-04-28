import Atomics
import Collections
import NNC

public struct InjectedControlsAndAdapters<FloatType: TensorNumeric & BinaryFloatingPoint> {
  var injectedControls: [DynamicGraph.Tensor<FloatType>]
  var injectedT2IAdapters: [DynamicGraph.Tensor<FloatType>]
  var injectedIPAdapters: [DynamicGraph.Tensor<FloatType>]
  var injectedAttentionKVs: [DynamicGraph.Tensor<FloatType>]
  public init(
    injectedControls: [DynamicGraph.Tensor<FloatType>],
    injectedT2IAdapters: [DynamicGraph.Tensor<FloatType>],
    injectedIPAdapters: [DynamicGraph.Tensor<FloatType>],
    injectedAttentionKVs: [DynamicGraph.Tensor<FloatType>]
  ) {
    self.injectedControls = injectedControls
    self.injectedT2IAdapters = injectedT2IAdapters
    self.injectedIPAdapters = injectedIPAdapters
    self.injectedAttentionKVs = injectedAttentionKVs
  }
}

public struct InjectControlsAndAdapters<T: TensorNumeric & BinaryFloatingPoint> {
  public var injectControls: Bool
  public var injectT2IAdapters: Bool
  public var injectAttentionKV: Bool
  public var injectIPAdapterLengths: [Int]
  public var injectControlModels: [ControlModel<T>]
  public init(
    injectControls: Bool, injectT2IAdapters: Bool, injectAttentionKV: Bool,
    injectIPAdapterLengths: [Int], injectControlModels: [ControlModel<T>]
  ) {
    self.injectControls = injectControls
    self.injectT2IAdapters = injectT2IAdapters
    self.injectAttentionKV = injectAttentionKV
    self.injectIPAdapterLengths = injectIPAdapterLengths
    self.injectControlModels = injectControlModels
  }
}

public protocol UNetProtocol {
  associatedtype FloatType: TensorNumeric & BinaryFloatingPoint
  init()
  var isLoaded: Bool { get }
  func unloadResources()
  var version: ModelVersion { get }
  var modelAndWeightMapper: (AnyModel, ModelWeightMapper)? { get }
  mutating func compileModel(
    filePath: String, externalOnDemand: Bool, version: ModelVersion, modifier: SamplerModifier,
    qkNorm: Bool, dualAttentionLayers: [Int], upcastAttention: Bool, usesFlashAttention: Bool,
    injectControlsAndAdapters: InjectControlsAndAdapters<FloatType>, lora: [LoRAConfiguration],
    isQuantizedModel: Bool, canRunLoRASeparately: Bool, inputs xT: DynamicGraph.Tensor<FloatType>,
    _ timestep: DynamicGraph.Tensor<FloatType>?, _ c: [DynamicGraph.AnyTensor],
    tokenLengthUncond: Int, tokenLengthCond: Int, isCfgEnabled: Bool,
    extraProjection: DynamicGraph.Tensor<FloatType>?,
    injectedControlsAndAdapters: InjectedControlsAndAdapters<FloatType>,
    tiledDiffusion: TiledConfiguration, teaCache: TeaCacheConfiguration
  ) -> Bool

  func callAsFunction(
    timestep: Float,
    inputs: DynamicGraph.Tensor<FloatType>, _: DynamicGraph.Tensor<FloatType>?,
    _: [DynamicGraph.AnyTensor], extraProjection: DynamicGraph.Tensor<FloatType>?,
    injectedControlsAndAdapters: (
      _ xT: DynamicGraph.Tensor<FloatType>, _ inputStartYPad: Int, _ inputEndYPad: Int,
      _ inputStartXPad: Int, _ inputEndXPad: Int, _ existingControlNets: inout [Model?]
    ) -> (
      injectedControls: [DynamicGraph.Tensor<FloatType>],
      injectedT2IAdapters: [DynamicGraph.Tensor<FloatType>],
      injectedAttentionKVs: [DynamicGraph.Tensor<FloatType>]
    ),
    injectedIPAdapters: [DynamicGraph.Tensor<FloatType>], step: Int,
    tokenLengthUncond: Int, tokenLengthCond: Int, isCfgEnabled: Bool,
    tiledDiffusion: TiledConfiguration, controlNets: inout [Model?]
  ) -> DynamicGraph.Tensor<FloatType>

  func decode(_ x: DynamicGraph.Tensor<FloatType>) -> DynamicGraph.Tensor<FloatType>

  // This is for best-effort.
  func cancel()
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
    case .sd3, .pixart, .auraflow, .flux1, .sd3Large, .hunyuanVideo, .wan21_1_3b, .wan21_14b,
      .hiDreamI1:
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

public func UNetExtractConditions<FloatType: TensorNumeric & BinaryFloatingPoint>(
  of: FloatType.Type = FloatType.self, graph: DynamicGraph, index: Int, batchSize: Int,
  tokenLengthUncond: Int, tokenLengthCond: Int, conditions: [DynamicGraph.AnyTensor],
  version: ModelVersion, isCfgEnabled: Bool
)
  -> [DynamicGraph.AnyTensor]
{
  switch version {
  case .kandinsky21, .sdxlBase, .sdxlRefiner, .ssd1b, .svdI2v, .v1, .v2, .wurstchenStageB,
    .wurstchenStageC:
    return conditions
  case .sd3, .auraflow, .sd3Large:
    return [conditions[0]]
      + conditions[1..<conditions.count].map {
        let shape = $0.shape
        return DynamicGraph.Tensor<FloatType>($0)[
          (index * batchSize)..<((index + 1) * batchSize), 0..<shape[1], 0..<shape[2]
        ]
        .copied()
      }
  case .flux1:
    return conditions[0..<2]
      + conditions[2..<conditions.count].map {
        let shape = $0.shape
        return DynamicGraph.Tensor<FloatType>($0)[
          (index * batchSize)..<((index + 1) * batchSize), 0..<shape[1], 0..<shape[2]
        ]
        .copied()
      }
  case .hiDreamI1:
    return conditions[0..<50]
      + conditions[50..<conditions.count].map {
        let shape = $0.shape
        return DynamicGraph.Tensor<FloatType>($0)[
          (index * batchSize)..<((index + 1) * batchSize), 0..<shape[1], 0..<shape[2]
        ]
        .copied()
      }
  case .hunyuanVideo:
    return conditions[0..<2]
      + conditions[2..<conditions.count].enumerated().map {
        let shape = $0.1.shape
        if !isCfgEnabled {
          if $0.0 == 0 {
            return DynamicGraph.Tensor<FloatType>($0.1)[
              (index * tokenLengthCond)..<((index + 1) * tokenLengthCond), 0..<shape[1]
            ]
            .reshaped(.HWC(1, tokenLengthCond, shape[1])).copied()
          }
          return DynamicGraph.Tensor<FloatType>($0.1)[
            index..<(index + 1), 0..<shape[1], 0..<shape[2]
          ]
          .copied()
        } else {
          // Note that for Hunyuan, batchSize is num of frames.
          precondition(batchSize % 2 == 0)
          if $0.0 == 0 {
            let timesteps = shape[0] / (tokenLengthUncond + tokenLengthCond)
            return Functional.concat(
              axis: 1,
              DynamicGraph.Tensor<FloatType>($0.1)[
                (index * tokenLengthUncond)..<((index + 1) * tokenLengthUncond), 0..<shape[1]
              ]
              .reshaped(.HWC(1, tokenLengthUncond, shape[1])),
              DynamicGraph.Tensor<FloatType>($0.1)[
                (index * tokenLengthCond + tokenLengthUncond * timesteps)..<((index + 1)
                  * tokenLengthCond + tokenLengthUncond * timesteps), 0..<shape[1]
              ].reshaped(.HWC(1, tokenLengthCond, shape[1])))
          }
          let timesteps = shape[0] / 2
          return Functional.concat(
            axis: 0,
            DynamicGraph.Tensor<FloatType>($0.1)[index..<(index + 1), 0..<shape[1], 0..<shape[2]],
            DynamicGraph.Tensor<FloatType>($0.1)[
              (index + timesteps)..<(index + timesteps + 1), 0..<shape[1], 0..<shape[2]])
        }
      }
  case .wan21_1_3b, .wan21_14b:
    return conditions[0..<1]
      + conditions[1..<7].map({
        let shape = $0.shape
        return DynamicGraph.Tensor<Float>($0)[
          index..<(index + 1), 0..<shape[1], 0..<shape[2]
        ].copied()
      }) + conditions[7..<(conditions.count - 2)]
      + conditions[(conditions.count - 2)...].map({
        let shape = $0.shape
        return DynamicGraph.Tensor<Float>($0)[
          index..<(index + 1), 0..<shape[1], 0..<shape[2]
        ].copied()
      })
  case .pixart:
    var extractedConditions = [conditions[0]]
    let layers = (conditions.count - 3) / 8
    for i in 0..<layers {
      let shape = conditions[1 + i * 8].shape
      extractedConditions.append(contentsOf: [
        DynamicGraph.Tensor<FloatType>(conditions[1 + i * 8])[
          index..<(index + 1), 0..<1, 0..<shape[2]
        ].copied(),
        DynamicGraph.Tensor<FloatType>(conditions[1 + i * 8 + 1])[
          index..<(index + 1), 0..<1, 0..<shape[2]
        ].copied(),
        DynamicGraph.Tensor<FloatType>(conditions[1 + i * 8 + 2])[
          index..<(index + 1), 0..<1, 0..<shape[2]
        ].copied(),
        conditions[1 + i * 8 + 3],
        conditions[1 + i * 8 + 4],
        DynamicGraph.Tensor<FloatType>(conditions[1 + i * 8 + 5])[
          index..<(index + 1), 0..<1, 0..<shape[2]
        ].copied(),
        DynamicGraph.Tensor<FloatType>(conditions[1 + i * 8 + 6])[
          index..<(index + 1), 0..<1, 0..<shape[2]
        ].copied(),
        DynamicGraph.Tensor<FloatType>(conditions[1 + i * 8 + 7])[
          index..<(index + 1), 0..<1, 0..<shape[2]
        ].copied(),
      ])
    }
    let shape = conditions[conditions.count - 2].shape
    extractedConditions.append(contentsOf: [
      DynamicGraph.Tensor<FloatType>(conditions[conditions.count - 2])[
        index..<(index + 1), 0..<1, 0..<shape[2]
      ].copied(),
      DynamicGraph.Tensor<FloatType>(conditions[conditions.count - 1])[
        index..<(index + 1), 0..<1, 0..<shape[2]
      ].copied(),
    ])
    return extractedConditions
  }
}

enum ModelBuilderOrModel {
  case modelBuilder(ModelBuilder<Void>)
  case model(Model)
  public var unwrapped: AnyModel {
    switch self {
    case .model(let model):
      return model
    case .modelBuilder(let modelBuilder):
      return modelBuilder
    }
  }
  public var maxConcurrency: StreamContext.Concurrency {
    get {
      switch self {
      case .model(let model):
        return model.maxConcurrency
      case .modelBuilder(let modelBuilder):
        return modelBuilder.maxConcurrency
      }
    }
    set {
      switch self {
      case .model(let model):
        model.maxConcurrency = newValue
      case .modelBuilder(let modelBuilder):
        modelBuilder.maxConcurrency = newValue
      }
    }
  }
  public func cancel() {
    switch self {
    case .model(let model):
      model.cancel()
    case .modelBuilder(let modelBuilder):
      modelBuilder.cancel()
    }
  }
  public func compile(inputs: [DynamicGraph_Any], isEager: Bool = false) {
    switch self {
    case .model(let model):
      model.compile(inputs: inputs, isEager: isEager)
    case .modelBuilder(let modelBuilder):
      modelBuilder.compile(inputs: inputs, isEager: isEager)
    }
  }
  public func compile(inputs: DynamicGraph_Any..., isEager: Bool = false) {
    compile(inputs: inputs, isEager: isEager)
  }
  public func callAsFunction<T: DynamicGraph.AnyTensorGroup>(
    inputs firstInput: T, _ restInputs: [DynamicGraph_Any], streamContext: StreamContext? = nil
  ) -> [T.AnyTensor] {
    switch self {
    case .model(let model):
      return model(inputs: firstInput, restInputs, streamContext: streamContext)
    case .modelBuilder(let modelBuilder):
      return modelBuilder(inputs: firstInput, restInputs, streamContext: streamContext)
    }
  }
  public func callAsFunction<T: DynamicGraph.AnyTensorGroup>(
    inputs firstInput: T, _ restInputs: DynamicGraph_Any..., streamContext: StreamContext? = nil
  ) -> [T.AnyTensor] {
    return self(inputs: firstInput, restInputs, streamContext: streamContext)
  }
}

public struct UNetFromNNC<FloatType: TensorNumeric & BinaryFloatingPoint>: UNetProtocol {
  var teaCache: TeaCache<FloatType>? = nil
  var unet: ModelBuilderOrModel? = nil
  var previewer: Model? = nil
  var unetWeightMapper: ModelWeightMapper? = nil
  var timeEmbed: Model? = nil
  var modifier: SamplerModifier = .none
  var yTileWeightsAndIndexes: [[(weight: Float, index: Int, offset: Int)]]? = nil
  var xTileWeightsAndIndexes: [[(weight: Float, index: Int, offset: Int)]]? = nil
  let isCancelled = ManagedAtomic<Bool>(false)
  public private(set) var version: ModelVersion = .v1
  public init() {}
  public var isLoaded: Bool { unet != nil }
  public func unloadResources() {}
}

extension UNetFromNNC {
  public var modelAndWeightMapper: (AnyModel, ModelWeightMapper)? {
    guard let unet = unet, let unetWeightMapper = unetWeightMapper else { return nil }
    return (unet.unwrapped, unetWeightMapper)
  }
  public mutating func compileModel(
    filePath: String, externalOnDemand: Bool, version: ModelVersion, modifier: SamplerModifier,
    qkNorm: Bool, dualAttentionLayers: [Int], upcastAttention: Bool, usesFlashAttention: Bool,
    injectControlsAndAdapters: InjectControlsAndAdapters<FloatType>, lora: [LoRAConfiguration],
    isQuantizedModel: Bool, canRunLoRASeparately: Bool, inputs xT: DynamicGraph.Tensor<FloatType>,
    _ timestep: DynamicGraph.Tensor<FloatType>?, _ c: [DynamicGraph.AnyTensor],
    tokenLengthUncond: Int, tokenLengthCond: Int, isCfgEnabled: Bool,
    extraProjection: DynamicGraph.Tensor<FloatType>?,
    injectedControlsAndAdapters: InjectedControlsAndAdapters<FloatType>,
    tiledDiffusion: TiledConfiguration, teaCache teaCacheConfiguration: TeaCacheConfiguration
  ) -> Bool {
    guard unet == nil else { return true }
    isCancelled.store(false, ordering: .releasing)
    let injectedControls = injectedControlsAndAdapters.injectedControls
    let injectedIPAdapters = injectedControlsAndAdapters.injectedIPAdapters
    let injectedT2IAdapters = injectedControlsAndAdapters.injectedT2IAdapters
    let injectedAttentionKVs = injectedControlsAndAdapters.injectedAttentionKVs
    let shape = xT.shape
    let batchSize = shape[0]
    let startHeight = shape[1]
    let startWidth = shape[2]
    let tiledWidth: Int
    let tiledHeight: Int
    let tileScaleFactor: Int
    let graph = xT.graph
    var unet: ModelBuilderOrModel
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
      graph, of: lora.map { $0.file }, modelFile: filePath)
    let isLoHa = lora.contains { $0.isLoHa }
    var configuration = LoRANetworkConfiguration(rank: rankOfLoRA, scale: 1, highPrecision: false)
    let runLoRASeparatelyIsPreferred = isQuantizedModel || externalOnDemand
    let isTeaCacheEnabled = teaCacheConfiguration.threshold > 0
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
          ModelBuilderOrModel.model(
            LoRAUNet(
              batchSize: batchSize, embeddingLength: (tokenLengthUncond, tokenLengthCond),
              startWidth: tiledWidth, startHeight: tiledHeight,
              usesFlashAttention: usesFlashAttention ? .scaleMerged : .none,
              injectControls: injectControlsAndAdapters.injectControls,
              injectT2IAdapters: injectControlsAndAdapters.injectT2IAdapters,
              injectIPAdapterLengths: injectControlsAndAdapters.injectIPAdapterLengths,
              LoRAConfiguration: configuration
            ))
      } else {
        unet =
          ModelBuilderOrModel.model(
            UNet(
              batchSize: batchSize, embeddingLength: (tokenLengthUncond, tokenLengthCond),
              startWidth: tiledWidth, startHeight: tiledHeight,
              usesFlashAttention: usesFlashAttention ? .scaleMerged : .none,
              injectControls: injectControlsAndAdapters.injectControls,
              injectT2IAdapters: injectControlsAndAdapters.injectT2IAdapters,
              injectIPAdapterLengths: injectControlsAndAdapters.injectIPAdapterLengths,
              injectAttentionKV: injectControlsAndAdapters.injectAttentionKV
            ).0)
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
          ModelBuilderOrModel.model(
            LoRAUNetv2(
              batchSize: batchSize, embeddingLength: (tokenLengthUncond, tokenLengthCond),
              startWidth: tiledWidth, startHeight: tiledHeight, upcastAttention: upcastAttention,
              usesFlashAttention: usesFlashAttention ? .scaleMerged : .none,
              injectControls: injectControlsAndAdapters.injectControls,
              LoRAConfiguration: configuration
            ))
      } else {
        unet =
          ModelBuilderOrModel.model(
            UNetv2(
              batchSize: batchSize, embeddingLength: (tokenLengthUncond, tokenLengthCond),
              startWidth: tiledWidth, startHeight: tiledHeight, upcastAttention: upcastAttention,
              usesFlashAttention: usesFlashAttention ? .scaleMerged : .none,
              injectControls: injectControlsAndAdapters.injectControls
            ).0)
      }
    case .svdI2v:
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
      tileScaleFactor = 8
      let model: Model
      (model, _, unetWeightMapper) =
        UNetXL(
          batchSize: batchSize, startHeight: tiledHeight, startWidth: tiledWidth,
          channels: [320, 640, 1280, 1280],
          inputAttentionRes: [1: [1, 1], 2: [1, 1], 4: [1, 1]], middleAttentionBlocks: 1,
          outputAttentionRes: [1: [1, 1, 1], 2: [1, 1, 1], 4: [1, 1, 1]], embeddingLength: (1, 1),
          injectIPAdapterLengths: [], upcastAttention: ([:], false, [1: [0, 1, 2]]),
          usesFlashAttention: usesFlashAttention ? .scale1 : .none, injectControls: false,
          isTemporalMixEnabled: true, of: FloatType.self
        )
      unet = ModelBuilderOrModel.model(model)
    case .kandinsky21:
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
      tileScaleFactor = 8
      unet = ModelBuilderOrModel.model(
        UNetKandinsky(
          batchSize: batchSize, channels: 384, outChannels: 8, channelMult: [1, 2, 3, 4],
          numResBlocks: 3, numHeadChannels: 64, t: 87, startHeight: tiledHeight,
          startWidth: tiledWidth, attentionResolutions: Set([2, 4, 8]),
          usesFlashAttention: usesFlashAttention))
      timeEmbed = timestepEmbedding(prefix: "time_embed", channels: 384 * 4)
    case .sdxlBase:
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
      tileScaleFactor = 8
      let model: Model
      if !lora.isEmpty && rankOfLoRA > 0 && !isLoHa && runLoRASeparatelyIsPreferred
        && canRunLoRASeparately
      {
        (model, unetWeightMapper) =
          LoRAUNetXL(
            batchSize: batchSize, startHeight: tiledHeight, startWidth: tiledWidth,
            channels: [320, 640, 1280], inputAttentionRes: [2: [2, 2], 4: [10, 10]],
            middleAttentionBlocks: 10, outputAttentionRes: [2: [2, 2, 2], 4: [10, 10, 10]],
            embeddingLength: (tokenLengthUncond, tokenLengthCond),
            injectIPAdapterLengths: injectControlsAndAdapters.injectIPAdapterLengths,
            upcastAttention: upcastAttention ? ([:], false, [2: [0, 1, 2]]) : ([:], false, [:]),
            usesFlashAttention: usesFlashAttention ? .scaleMerged : .none,
            injectControls: injectControlsAndAdapters.injectControls,
            LoRAConfiguration: configuration
          )
      } else {
        (model, _, unetWeightMapper) =
          UNetXL(
            batchSize: batchSize, startHeight: tiledHeight, startWidth: tiledWidth,
            channels: [320, 640, 1280], inputAttentionRes: [2: [2, 2], 4: [10, 10]],
            middleAttentionBlocks: 10, outputAttentionRes: [2: [2, 2, 2], 4: [10, 10, 10]],
            embeddingLength: (tokenLengthUncond, tokenLengthCond),
            injectIPAdapterLengths: injectControlsAndAdapters.injectIPAdapterLengths,
            upcastAttention: upcastAttention ? ([:], false, [2: [0, 1, 2]]) : ([:], false, [:]),
            usesFlashAttention: usesFlashAttention ? .scaleMerged : .none,
            injectControls: injectControlsAndAdapters.injectControls, isTemporalMixEnabled: false,
            of: FloatType.self
          )
      }
      unet = ModelBuilderOrModel.model(model)
    case .sdxlRefiner:
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
      tileScaleFactor = 8
      let model: Model
      if !lora.isEmpty && rankOfLoRA > 0 && !isLoHa && runLoRASeparatelyIsPreferred
        && canRunLoRASeparately
      {
        (model, unetWeightMapper) =
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
        (model, _, unetWeightMapper) =
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
      unet = ModelBuilderOrModel.model(model)
    case .ssd1b:
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
      tileScaleFactor = 8
      let model: Model
      if !lora.isEmpty && rankOfLoRA > 0 && !isLoHa && runLoRASeparatelyIsPreferred
        && canRunLoRASeparately
      {
        (model, unetWeightMapper) =
          LoRAUNetXL(
            batchSize: batchSize, startHeight: tiledHeight, startWidth: tiledWidth,
            channels: [320, 640, 1280], inputAttentionRes: [2: [2, 2], 4: [4, 4]],
            middleAttentionBlocks: 0, outputAttentionRes: [2: [2, 1, 1], 4: [4, 4, 10]],
            embeddingLength: (tokenLengthUncond, tokenLengthCond),
            injectIPAdapterLengths: injectControlsAndAdapters.injectIPAdapterLengths,
            upcastAttention: ([:], false, [:]),
            usesFlashAttention: usesFlashAttention ? .scale1 : .none,
            injectControls: false, LoRAConfiguration: configuration
          )
      } else {
        (model, _, unetWeightMapper) =
          UNetXL(
            batchSize: batchSize, startHeight: tiledHeight, startWidth: tiledWidth,
            channels: [320, 640, 1280], inputAttentionRes: [2: [2, 2], 4: [4, 4]],
            middleAttentionBlocks: 0, outputAttentionRes: [2: [2, 1, 1], 4: [4, 4, 10]],
            embeddingLength: (tokenLengthUncond, tokenLengthCond),
            injectIPAdapterLengths: injectControlsAndAdapters.injectIPAdapterLengths,
            upcastAttention: ([:], false, [:]),
            usesFlashAttention: usesFlashAttention ? .scale1 : .none, injectControls: false,
            isTemporalMixEnabled: false, of: FloatType.self
          )
      }
      unet = ModelBuilderOrModel.model(model)
    case .wurstchenStageC:
      tiledWidth = startWidth
      tiledHeight = startHeight
      tileScaleFactor = 1
      unet = ModelBuilderOrModel.model(
        WurstchenStageC(
          batchSize: batchSize, height: startHeight, width: startWidth,
          t: (tokenLengthUncond + 8, tokenLengthCond + 8),
          usesFlashAttention: usesFlashAttention ? .scaleMerged : .none
        ).0)
      previewer = WurstchenStageCPreviewer()
    case .wurstchenStageB:
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 16, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 16, startHeight) : startHeight
      tileScaleFactor = 16
      unet = ModelBuilderOrModel.model(
        WurstchenStageB(
          batchSize: batchSize, cIn: 4, height: tiledHeight, width: tiledWidth,
          usesFlashAttention: usesFlashAttention ? .scaleMerged : .none
        ).0)
    case .sd3:
      var posEmbedMaxSize = 192
      graph.openStore(
        filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
      ) {
        guard let shape = $0.read(like: "__dit__[t-pos_embed-0-0]")?.shape else { return }
        posEmbedMaxSize = Int(Double(shape.reduce(1, *) / 1536).squareRoot().rounded())
      }
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
          ModelBuilderOrModel.model(
            LoRAMMDiT(
              batchSize: batchSize, t: c[0].shape[1], height: tiledHeight,
              width: tiledWidth, channels: 1536, layers: 24, upcast: false, qkNorm: qkNorm,
              dualAttentionLayers: dualAttentionLayers, posEmbedMaxSize: posEmbedMaxSize,
              usesFlashAttention: usesFlashAttention ? .scaleMerged : .none,
              LoRAConfiguration: configuration, of: FloatType.self
            ).1)
      } else {
        unet =
          ModelBuilderOrModel.model(
            MMDiT(
              batchSize: batchSize, t: c[0].shape[1], height: tiledHeight,
              width: tiledWidth, channels: 1536, layers: 24, upcast: false, qkNorm: qkNorm,
              dualAttentionLayers: dualAttentionLayers, posEmbedMaxSize: posEmbedMaxSize,
              usesFlashAttention: usesFlashAttention ? .scaleMerged : .none, of: FloatType.self
            ).1)
      }
    case .sd3Large:
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
      tileScaleFactor = 8
      if !lora.isEmpty && rankOfLoRA > 0 && !isLoHa && runLoRASeparatelyIsPreferred
        && canRunLoRASeparately
      {
        let keys = LoRALoader<FloatType>.keys(graph, of: lora.map { $0.file }, modelFile: filePath)
        configuration.keys = keys
        unet =
          ModelBuilderOrModel.model(
            LoRAMMDiT(
              batchSize: batchSize, t: c[0].shape[1], height: tiledHeight,
              width: tiledWidth, channels: 2432, layers: 38, upcast: true, qkNorm: true,
              dualAttentionLayers: [], posEmbedMaxSize: 192,
              usesFlashAttention: usesFlashAttention ? .scaleMerged : .none,
              LoRAConfiguration: configuration, of: FloatType.self
            ).1)
      } else {
        unet =
          ModelBuilderOrModel.model(
            MMDiT(
              batchSize: batchSize, t: c[0].shape[1], height: tiledHeight,
              width: tiledWidth, channels: 2432, layers: 38, upcast: true, qkNorm: true,
              dualAttentionLayers: [], posEmbedMaxSize: 192,
              usesFlashAttention: usesFlashAttention ? .scaleMerged : .none, of: FloatType.self
            ).1)
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
        unet = ModelBuilderOrModel.model(
          LoRAPixArt(
            batchSize: batchSize, height: tiledHeight, width: tiledWidth, channels: 1152,
            layers: 28,
            tokenLength: (tokenLengthUncond, tokenLengthCond),
            usesFlashAttention: usesFlashAttention,
            LoRAConfiguration: configuration, of: FloatType.self
          ).1)
      } else {
        unet = ModelBuilderOrModel.model(
          PixArt(
            batchSize: batchSize, height: tiledHeight, width: tiledWidth, channels: 1152,
            layers: 28,
            tokenLength: (tokenLengthUncond, tokenLengthCond),
            usesFlashAttention: usesFlashAttention,
            of: FloatType.self
          ).1)
      }
    case .auraflow:
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
      tileScaleFactor = 8
      unet = ModelBuilderOrModel.model(
        AuraFlow(
          batchSize: batchSize, tokenLength: max(256, max(tokenLengthCond, tokenLengthUncond)),
          height: tiledHeight, width: tiledWidth, channels: 3072, layers: (4, 32),
          usesFlashAttention: usesFlashAttention ? .scaleMerged : .none, of: FloatType.self
        ).1)
    case .flux1:
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
      tileScaleFactor = 8
      var injectIPAdapterLengths = [Int: [Int]]()
      for i in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18] {
        injectIPAdapterLengths[i] = injectControlsAndAdapters.injectIPAdapterLengths
      }
      for i in [0, 4, 8, 12, 16, 20, 24, 28, 32, 36] {
        injectIPAdapterLengths[19 + i] = injectControlsAndAdapters.injectIPAdapterLengths
      }
      let tokenLength = c[1].shape[1]
      if !lora.isEmpty && rankOfLoRA > 0 && !isLoHa && runLoRASeparatelyIsPreferred
        && canRunLoRASeparately
      {
        let keys = LoRALoader<FloatType>.keys(graph, of: lora.map { $0.file }, modelFile: filePath)
        configuration.keys = keys
        unet = ModelBuilderOrModel.model(
          LoRAFlux1(
            batchSize: isTeaCacheEnabled ? 1 : batchSize, tokenLength: tokenLength,
            height: tiledHeight, width: tiledWidth, channels: 3072, layers: (19, 38),
            usesFlashAttention: usesFlashAttention ? .scaleMerged : .none,
            contextPreloaded: true,
            injectControls: injectControlsAndAdapters.injectControls,
            injectIPAdapterLengths: injectIPAdapterLengths, outputResidual: isTeaCacheEnabled,
            inputResidual: false, LoRAConfiguration: configuration
          ).1)
        if isTeaCacheEnabled {
          teaCache = TeaCache(
            modelVersion: version, coefficients: teaCacheConfiguration.coefficients,
            threshold: teaCacheConfiguration.threshold, steps: teaCacheConfiguration.steps,
            reducedModel: LoRAFlux1(
              batchSize: 1, tokenLength: tokenLength,
              height: tiledHeight, width: tiledWidth, channels: 3072, layers: (0, 0),
              usesFlashAttention: usesFlashAttention ? .scaleMerged : .none,
              contextPreloaded: true,
              injectControls: injectControlsAndAdapters.injectControls,
              injectIPAdapterLengths: injectIPAdapterLengths, outputResidual: false,
              inputResidual: true, LoRAConfiguration: configuration
            ).1,
            inferModel: LoRAFlux1Norm1(
              batchSize: 1, height: tiledHeight, width: tiledWidth, channels: 3072,
              LoRAConfiguration: configuration))
        }
      } else {
        unet = ModelBuilderOrModel.model(
          Flux1(
            batchSize: isTeaCacheEnabled ? 1 : batchSize, tokenLength: tokenLength,
            height: tiledHeight, width: tiledWidth, channels: 3072, layers: (19, 38),
            usesFlashAttention: usesFlashAttention ? .scaleMerged : .none,
            contextPreloaded: true,
            injectControls: injectControlsAndAdapters.injectControls,
            injectIPAdapterLengths: injectIPAdapterLengths, outputResidual: isTeaCacheEnabled,
            inputResidual: false
          ).1)
        if isTeaCacheEnabled {
          teaCache = TeaCache(
            modelVersion: version, coefficients: teaCacheConfiguration.coefficients,
            threshold: teaCacheConfiguration.threshold, steps: teaCacheConfiguration.steps,
            reducedModel: Flux1(
              batchSize: 1, tokenLength: tokenLength,
              height: tiledHeight, width: tiledWidth, channels: 3072, layers: (0, 0),
              usesFlashAttention: usesFlashAttention ? .scaleMerged : .none,
              contextPreloaded: true,
              injectControls: injectControlsAndAdapters.injectControls,
              injectIPAdapterLengths: injectIPAdapterLengths, outputResidual: false,
              inputResidual: true
            ).1,
            inferModel: Flux1Norm1(
              batchSize: 1, height: tiledHeight, width: tiledWidth, channels: 3072))
        }
      }
    case .hunyuanVideo:
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
      tileScaleFactor = 8
      if !lora.isEmpty && rankOfLoRA > 0 && !isLoHa && runLoRASeparatelyIsPreferred
        && canRunLoRASeparately
      {
        let keys = LoRALoader<FloatType>.keys(graph, of: lora.map { $0.file }, modelFile: filePath)
        configuration.keys = keys
        unet = ModelBuilderOrModel.modelBuilder(
          ModelBuilder {
            return LoRAHunyuan(
              time: $0[0].shape[0], height: tiledHeight, width: tiledWidth,
              textLength: $0[3].shape[1],
              channels: 3072, layers: (20, 40),
              usesFlashAttention: usesFlashAttention ? .scaleMerged : .none,
              outputResidual: isTeaCacheEnabled, inputResidual: false,
              LoRAConfiguration: configuration
            ).1
          })
        if isTeaCacheEnabled {
          teaCache = TeaCache(
            modelVersion: version, coefficients: teaCacheConfiguration.coefficients,
            threshold: teaCacheConfiguration.threshold, steps: teaCacheConfiguration.steps,
            reducedModel: LoRAHunyuan(
              time: isCfgEnabled ? batchSize / 2 : batchSize, height: tiledHeight,
              width: tiledWidth,
              textLength: 0,
              channels: 3072, layers: (0, 0),
              usesFlashAttention: usesFlashAttention ? .scaleMerged : .none,
              outputResidual: false, inputResidual: true, LoRAConfiguration: configuration
            ).1,
            inferModel: HunyuanNorm1(
              time: isCfgEnabled ? batchSize / 2 : batchSize, height: tiledHeight,
              width: tiledWidth, channels: 3072))
        }
      } else {
        unet = ModelBuilderOrModel.modelBuilder(
          ModelBuilder {
            return Hunyuan(
              time: $0[0].shape[0], height: tiledHeight, width: tiledWidth,
              textLength: $0[3].shape[1],
              channels: 3072, layers: (20, 40),
              usesFlashAttention: usesFlashAttention ? .scaleMerged : .none,
              outputResidual: isTeaCacheEnabled, inputResidual: false
            ).1
          })
        if isTeaCacheEnabled {
          teaCache = TeaCache(
            modelVersion: version, coefficients: teaCacheConfiguration.coefficients,
            threshold: teaCacheConfiguration.threshold, steps: teaCacheConfiguration.steps,
            reducedModel: Hunyuan(
              time: isCfgEnabled ? batchSize / 2 : batchSize, height: tiledHeight,
              width: tiledWidth,
              textLength: 0,
              channels: 3072, layers: (0, 0),
              usesFlashAttention: usesFlashAttention ? .scaleMerged : .none,
              outputResidual: false, inputResidual: true
            ).1,
            inferModel: HunyuanNorm1(
              time: isCfgEnabled ? batchSize / 2 : batchSize, height: tiledHeight,
              width: tiledWidth, channels: 3072))
        }
      }
    case .wan21_1_3b:
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
      tileScaleFactor = 8
      let injectImage = c.count > 9 + 4 * 30
      let textLength = c[7].shape[1]
      if !lora.isEmpty && rankOfLoRA > 0 && !isLoHa && runLoRASeparatelyIsPreferred
        && canRunLoRASeparately
      {
        let keys = LoRALoader<FloatType>.keys(graph, of: lora.map { $0.file }, modelFile: filePath)
        configuration.keys = keys
        unet = ModelBuilderOrModel.model(
          LoRAWan(
            channels: 1_536, layers: 30, intermediateSize: 8_960,
            time: isCfgEnabled ? batchSize / 2 : batchSize, height: tiledHeight, width: tiledWidth,
            textLength: textLength, injectImage: injectImage, outputResidual: isTeaCacheEnabled,
            inputResidual: false, LoRAConfiguration: configuration
          ).1)
        if isTeaCacheEnabled {
          teaCache = TeaCache(
            modelVersion: version, coefficients: teaCacheConfiguration.coefficients,
            threshold: teaCacheConfiguration.threshold, steps: teaCacheConfiguration.steps,
            reducedModel: LoRAWan(
              channels: 1_536, layers: 0, intermediateSize: 8_960,
              time: isCfgEnabled ? batchSize / 2 : batchSize, height: tiledHeight,
              width: tiledWidth, textLength: textLength, injectImage: injectImage,
              outputResidual: false, inputResidual: true, LoRAConfiguration: configuration
            ).1)
        }
      } else {
        unet = ModelBuilderOrModel.model(
          Wan(
            channels: 1_536, layers: 30, intermediateSize: 8_960,
            time: isCfgEnabled ? batchSize / 2 : batchSize, height: tiledHeight, width: tiledWidth,
            textLength: textLength, injectImage: injectImage, outputResidual: isTeaCacheEnabled,
            inputResidual: false
          ).1)
        if isTeaCacheEnabled {
          teaCache = TeaCache(
            modelVersion: version, coefficients: teaCacheConfiguration.coefficients,
            threshold: teaCacheConfiguration.threshold, steps: teaCacheConfiguration.steps,
            reducedModel: Wan(
              channels: 1_536, layers: 0, intermediateSize: 8_960,
              time: isCfgEnabled ? batchSize / 2 : batchSize, height: tiledHeight,
              width: tiledWidth, textLength: textLength, injectImage: injectImage,
              outputResidual: false, inputResidual: true
            ).1)
        }
      }
    case .wan21_14b:
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
      tileScaleFactor = 8
      let injectImage = c.count > 9 + 4 * 40
      let textLength = c[7].shape[1]
      if !lora.isEmpty && rankOfLoRA > 0 && !isLoHa && runLoRASeparatelyIsPreferred
        && canRunLoRASeparately
      {
        let keys = LoRALoader<FloatType>.keys(graph, of: lora.map { $0.file }, modelFile: filePath)
        configuration.keys = keys
        unet = ModelBuilderOrModel.model(
          LoRAWan(
            channels: 5_120, layers: 40, intermediateSize: 13_824,
            time: isCfgEnabled ? batchSize / 2 : batchSize, height: tiledHeight, width: tiledWidth,
            textLength: textLength, injectImage: injectImage, outputResidual: false,
            inputResidual: false, LoRAConfiguration: configuration
          ).1)
      } else {
        unet = ModelBuilderOrModel.model(
          Wan(
            channels: 5_120, layers: 40, intermediateSize: 13_824,
            time: isCfgEnabled ? batchSize / 2 : batchSize, height: tiledHeight, width: tiledWidth,
            textLength: textLength, injectImage: injectImage, outputResidual: false,
            inputResidual: false
          ).1)
      }
    case .hiDreamI1:
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
      tileScaleFactor = 8
      let llama3Length = c[48].shape[1]
      let t5Length = c[49].shape[1] - llama3Length
      unet = ModelBuilderOrModel.model(
        HiDream(
          batchSize: 1, height: tiledHeight,
          width: modifier == .editing ? tiledWidth * 2 : tiledWidth,
          textLength: (t5Length, llama3Length), layers: (16, 32)
        ).0)
    }
    // Need to assign version now such that sliceInputs will have the correct version.
    self.version = version
    self.modifier = modifier
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
      case .flux1:
        c.append(contentsOf: injectedIPAdapters)
      case .v2, .sd3, .sd3Large, .pixart, .auraflow, .kandinsky21, .svdI2v, .wurstchenStageC,
        .wurstchenStageB, .hunyuanVideo, .wan21_1_3b, .wan21_14b, .hiDreamI1:
        fatalError()
      }
    }
    var inputs = [DynamicGraph.AnyTensor]()
    if let extraProjection = extraProjection {
      inputs.append(extraProjection.reshaped(.WC(batchSize, 384 * 4)))
    } else if let timestep = timestep {
      inputs.append(timestep)
    }
    inputs.append(contentsOf: c)
    if injectControlsAndAdapters.injectControls {
      inputs.append(contentsOf: injectedControls)
    }
    if injectControlsAndAdapters.injectT2IAdapters {
      inputs.append(contentsOf: injectedT2IAdapters)
    }
    if !injectedAttentionKVs.isEmpty {
      inputs.append(contentsOf: injectedAttentionKVs)
    }
    unet.maxConcurrency = .limit(4)
    let tileOverlap = min(
      min(
        tiledDiffusion.tileOverlap * tileScaleFactor / 2,
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
      compile(
        unet, tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
        isCfgEnabled: isCfgEnabled,
        inputs: [xT[0..<shape[0], 0..<tiledHeight, 0..<tiledWidth, 0..<shape[3]]] + inputs)
    } else {
      compile(
        unet, tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
        isCfgEnabled: isCfgEnabled, inputs: [xT] + inputs)
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
    case .sd3, .pixart, .auraflow, .flux1, .sd3Large, .hunyuanVideo, .wan21_1_3b, .wan21_14b,
      .hiDreamI1:
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
            case .flux1:
              return [Int: Int](
                uniqueKeysWithValues: (0..<(19 + 38)).map {
                  return ($0, $0)
                })
            case .sd3Large:
              return [Int: Int](
                uniqueKeysWithValues: (0..<38).map {
                  return ($0, $0)
                })
            case .hunyuanVideo:
              return [Int: Int](
                uniqueKeysWithValues: (0..<(20 + 40)).map {
                  return ($0, $0)
                })
            case .wan21_1_3b, .wan21_14b:
              return [Int: Int](
                uniqueKeysWithValues: (0..<40).map {
                  return ($0, $0)
                })
            case .hiDreamI1:
              return [Int: Int](
                uniqueKeysWithValues: (0..<(16 + 32)).map {
                  return ($0, $0)
                })
            case .auraflow:
              fatalError()
            case .kandinsky21, .svdI2v, .wurstchenStageC, .wurstchenStageB:
              fatalError()
            }
          }()
          ControlModelLoader<FloatType>.openStore(
            graph, injectControlModels: injectControlsAndAdapters.injectControlModels,
            version: version
          ) { controlModelLoader in
            LoRALoader<FloatType>.openStore(graph, lora: lora) { loader in
              try! store.read(
                modelKey, model: unet.unwrapped, strict: true,
                codec: [.jit, .q6p, .q8p, .ezm7, externalData]
              ) {
                name, dataType, format, shape in
                if let result = controlModelLoader.loadMergedWeight(name: name) {
                  return result
                }
                // Patch for bias weights which missing a 1/8 scale. Note that this is not needed if we merge this into the model import step like we do for Hunyuan.
                if version == .flux1
                  && (name.hasSuffix("_out_proj-17-1]") || name.hasSuffix("_out_proj-18-1]")),
                  let tensor = store.read(name, codec: [.ezm7, .externalData, .q6p, .q8p])
                {
                  return .final(
                    graph.withNoGrad {
                      let scaleFactor: Float = 8
                      return
                        ((1 / scaleFactor)
                        * graph.variable(Tensor<FloatType>(from: tensor)).toGPU(0)).rawValue.toCPU()
                    })
                }
                return loader.concatenateLoRA(
                  graph, LoRAMapping: mapping, filesRequireMerge: filesRequireMerge, name: name,
                  store: store, dataType: dataType, format: format, shape: shape)
              }
            }
          }
        } else {
          ControlModelLoader<FloatType>.openStore(
            graph, injectControlModels: injectControlsAndAdapters.injectControlModels,
            version: version
          ) { controlModelLoader in
            LoRALoader<FloatType>.openStore(graph, lora: lora) { loader in
              store.read(
                modelKey, model: unet.unwrapped, codec: [.jit, .q6p, .q8p, .ezm7, externalData]
              ) {
                name, _, _, shape in
                if let result = controlModelLoader.loadMergedWeight(name: name) {
                  return result
                }
                // Patch for bias weights which missing a 1/8 scale. Note that this is not needed if we merge this into the model import step like we do for Hunyuan.
                if version == .flux1
                  && (name.hasSuffix("_out_proj-17-1]") || name.hasSuffix("_out_proj-18-1]")),
                  let tensor = store.read(name, codec: [.ezm7, .externalData, .q6p, .q8p])
                {
                  return .final(
                    graph.withNoGrad {
                      let scaleFactor: Float = 8
                      return
                        ((1 / scaleFactor)
                        * graph.variable(Tensor<FloatType>(from: tensor)).toGPU(0)).rawValue.toCPU()
                    })
                }
                return loader.mergeLoRA(graph, name: name, store: store, shape: shape)
              }
            }
          }
        }
      } else {
        ControlModelLoader<FloatType>.openStore(
          graph, injectControlModels: injectControlsAndAdapters.injectControlModels,
          version: version
        ) { controlModelLoader in
          store.read(
            modelKey, model: unet.unwrapped, codec: [.jit, .q6p, .q8p, .ezm7, externalData]
          ) {
            name, _, _, _ in
            if let result = controlModelLoader.loadMergedWeight(name: name) {
              return result
            }
            // Patch for bias weights which missing a 1/8 scale. Note that this is not needed if we merge this into the model import step like we do for Hunyuan.
            if version == .flux1
              && (name.hasSuffix("_out_proj-17-1]") || name.hasSuffix("_out_proj-18-1]")),
              let tensor = store.read(name, codec: [.ezm7, .externalData, .q6p, .q8p])
            {
              return .final(
                graph.withNoGrad {
                  let scaleFactor: Float = 8
                  return
                    ((1 / scaleFactor) * graph.variable(Tensor<FloatType>(from: tensor)).toGPU(0))
                    .rawValue.toCPU()
                })
            }
            return .continue(name)
          }
        }
      }
      if let timeEmbed = timeEmbed {
        store.read("time_embed", model: timeEmbed, codec: [.q6p, .q8p, .ezm7, .externalData])
      }
      if let previewer = previewer {
        previewer.maxConcurrency = .limit(4)
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
    _ inputs: [DynamicGraph.AnyTensor], originalShape: TensorShape, xyTiles: Int,
    index: Int, inputStartYPad: Int, inputEndYPad: Int, inputStartXPad: Int, inputEndXPad: Int
  ) -> [DynamicGraph.AnyTensor] {
    return inputs.enumerated().map {
      // For FLUX.1, if it is the first one, we need to handle its slicing (rotary encoding).
      switch version {
      case .flux1:
        if $0.0 == 0 {
          let shape = $0.1.shape
          let tokenLength = shape[1] - (originalShape[1] / 2) * (originalShape[2] / 2)
          let graph = $0.1.graph
          let tokenEncoding = DynamicGraph.Tensor<FloatType>($0.1)[
            0..<shape[0], 0..<tokenLength, 0..<shape[2], 0..<shape[3]
          ]
          .copied()
          let imageEncoding = DynamicGraph.Tensor<FloatType>($0.1)[
            0..<shape[0], tokenLength..<shape[1], 0..<shape[2], 0..<shape[3]
          ]
          .copied().reshaped(
            .NHWC(shape[0], originalShape[1] / 2, originalShape[2] / 2, shape[3]))
          let h = inputEndYPad / 2 - inputStartYPad / 2
          let w = inputEndXPad / 2 - inputStartXPad / 2
          let sliceEncoding = imageEncoding[
            0..<shape[0], (inputStartYPad / 2)..<(inputEndYPad / 2),
            (inputStartXPad / 2)..<(inputEndXPad / 2), 0..<shape[3]
          ].copied().reshaped(.NHWC(shape[0], h * w, 1, shape[3]))
          var finalEncoding = graph.variable(
            $0.1.kind, .NHWC(shape[0], h * w + tokenLength, 1, shape[3]), of: FloatType.self)
          finalEncoding[0..<shape[0], 0..<tokenLength, 0..<1, 0..<shape[3]] = tokenEncoding
          finalEncoding[0..<shape[0], tokenLength..<(tokenLength + h * w), 0..<1, 0..<shape[3]] =
            sliceEncoding
          return finalEncoding
        }
      case .hiDreamI1:
        if $0.0 == 0 {
          let shape = $0.1.shape
          let tokenLength = shape[1] - (originalShape[1] / 2) * (originalShape[2] / 2)
          let graph = $0.1.graph
          let imageEncoding = DynamicGraph.Tensor<FloatType>($0.1)[
            0..<shape[0], 0..<(shape[1] - tokenLength), 0..<shape[2], 0..<shape[3]
          ].copied().reshaped(
            .NHWC(shape[0], originalShape[1] / 2, originalShape[2] / 2, shape[3]))
          let tokenEncoding = DynamicGraph.Tensor<FloatType>($0.1)[
            0..<shape[0], (shape[1] - tokenLength)..<shape[1], 0..<shape[2], 0..<shape[3]
          ].copied()
          let h = inputEndYPad / 2 - inputStartYPad / 2
          let w = inputEndXPad / 2 - inputStartXPad / 2
          let sliceEncoding = imageEncoding[
            0..<shape[0], (inputStartYPad / 2)..<(inputEndYPad / 2),
            (inputStartXPad / 2)..<(inputEndXPad / 2), 0..<shape[3]
          ].copied().reshaped(.NHWC(shape[0], h * w, 1, shape[3]))
          var finalEncoding = graph.variable(
            $0.1.kind, .NHWC(shape[0], h * w + tokenLength, 1, shape[3]), of: FloatType.self)
          finalEncoding[0..<shape[0], 0..<(h * w), 0..<1, 0..<shape[3]] =
            sliceEncoding
          finalEncoding[0..<shape[0], (h * w)..<(h * w + tokenLength), 0..<1, 0..<shape[3]] =
            tokenEncoding
          return finalEncoding
        }
      case .hunyuanVideo:
        if $0.0 == 0 {
          let shape = $0.1.shape
          let t = shape[1] / ((originalShape[1] / 2) * (originalShape[2] / 2))
          let imageEncoding = DynamicGraph.Tensor<FloatType>($0.1).reshaped(
            .NHWC(t, originalShape[1] / 2, originalShape[2] / 2, shape[3]))
          let h = inputEndYPad / 2 - inputStartYPad / 2
          let w = inputEndXPad / 2 - inputStartXPad / 2
          return imageEncoding[
            0..<t, (inputStartYPad / 2)..<(inputEndYPad / 2),
            (inputStartXPad / 2)..<(inputEndXPad / 2), 0..<shape[3]
          ].copied().reshaped(.NHWC(shape[0], t * h * w, 1, shape[3]))
        } else if $0.0 == 1 {
          let shape = $0.1.shape
          let t = inputs[0].shape[1] / ((originalShape[1] / 2) * (originalShape[2] / 2))
          let tokenLength =
            shape[1] - t * (originalShape[1] / 2) * (originalShape[2] / 2)
          let graph = $0.1.graph
          let imageEncoding = DynamicGraph.Tensor<FloatType>($0.1)[
            0..<shape[0], 0..<(shape[1] - tokenLength), 0..<shape[2], 0..<shape[3]
          ].copied().reshaped(
            .NHWC(t, originalShape[1] / 2, originalShape[2] / 2, shape[3]))
          let tokenEncoding = DynamicGraph.Tensor<FloatType>($0.1)[
            0..<shape[0], (shape[1] - tokenLength)..<shape[1], 0..<shape[2], 0..<shape[3]
          ]
          .copied()
          let h = inputEndYPad / 2 - inputStartYPad / 2
          let w = inputEndXPad / 2 - inputStartXPad / 2
          let sliceEncoding = imageEncoding[
            0..<t, (inputStartYPad / 2)..<(inputEndYPad / 2),
            (inputStartXPad / 2)..<(inputEndXPad / 2), 0..<shape[3]
          ].copied().reshaped(.NHWC(shape[0], t * h * w, 1, shape[3]))
          var finalEncoding = graph.variable(
            $0.1.kind, .NHWC(shape[0], t * h * w + tokenLength, 1, shape[3]), of: FloatType.self)
          finalEncoding[0..<shape[0], 0..<(t * h * w), 0..<1, 0..<shape[3]] = sliceEncoding
          finalEncoding[
            0..<shape[0], (t * h * w)..<(tokenLength + t * h * w), 0..<1, 0..<shape[3]] =
            tokenEncoding
          return finalEncoding
        }
      case .auraflow, .kandinsky21, .pixart, .sd3, .sd3Large, .sdxlBase, .sdxlRefiner, .ssd1b,
        .svdI2v, .v1, .v2, .wurstchenStageB, .wurstchenStageC:
        break
      case .wan21_1_3b, .wan21_14b:
        if $0.0 == 0 {
          let shape = $0.1.shape
          let t = shape[1] / ((originalShape[1] / 2) * (originalShape[2] / 2))
          let imageEncoding = DynamicGraph.Tensor<FloatType>($0.1).reshaped(
            .NHWC(t, originalShape[1] / 2, originalShape[2] / 2, shape[3]))
          let h = inputEndYPad / 2 - inputStartYPad / 2
          let w = inputEndXPad / 2 - inputStartXPad / 2
          return imageEncoding[
            0..<t, (inputStartYPad / 2)..<(inputEndYPad / 2),
            (inputStartXPad / 2)..<(inputEndXPad / 2), 0..<shape[3]
          ].copied().reshaped(.NHWC(shape[0], t * h * w, 1, shape[3]))
        }
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
            return DynamicGraph.Tensor<FloatType>($0.1)[
              0..<shape[0], (inputStartYPad / scaleFactor)..<(inputEndYPad / scaleFactor),
              (inputStartXPad / scaleFactor)..<(inputEndXPad / scaleFactor), 0..<shape[3]
            ].copied()
          }
        }
      } else if originalShape[0] * xyTiles == shape[0] {
        return DynamicGraph.Tensor<FloatType>($0.1)[
          (index * originalShape[0])..<((index + 1) * originalShape[0]), 0..<shape[1], 0..<shape[2],
          0..<shape[3]
        ].copied()
      }
      return $0.1
    }
  }

  private func compile(
    _ unet: ModelBuilderOrModel, tokenLengthUncond: Int, tokenLengthCond: Int, isCfgEnabled: Bool,
    inputs: [DynamicGraph.AnyTensor]
  ) {
    switch version {
    case .hunyuanVideo:
      guard isCfgEnabled else {
        unet.compile(inputs: inputs)
        teaCache?.compile(model: unet, inputs: inputs)
        return
      }
      let inputs: [DynamicGraph.AnyTensor] = inputs.enumerated().map {
        let shape = $0.1.shape
        switch $0.0 {
        case 0:
          return DynamicGraph.Tensor<FloatType>($0.1)[
            0..<(shape[0] / 2), 0..<shape[1], 0..<shape[2], 0..<shape[3]]
        case 1...2:
          return $0.1
        case 3:
          return DynamicGraph.Tensor<FloatType>($0.1)[
            0..<shape[0], 0..<max(tokenLengthUncond, tokenLengthCond), 0..<shape[2]
          ]
          .copied()
        default:
          return DynamicGraph.Tensor<FloatType>($0.1)[0..<1, 0..<shape[1], 0..<shape[2]]
        }
      }
      unet.compile(inputs: inputs)
      teaCache?.compile(model: unet, inputs: inputs)
      return
    case .wan21_1_3b, .wan21_14b:
      guard isCfgEnabled else {
        unet.compile(inputs: inputs)
        teaCache?.compile(model: unet, inputs: inputs)
        return
      }
      let injectImage: Bool
      if version == .wan21_1_3b {
        injectImage = inputs.count > 10 + 4 * 30
      } else {
        injectImage = inputs.count > 10 + 4 * 40
      }
      let inputs: [DynamicGraph.AnyTensor] = inputs.enumerated().compactMap {
        let shape = $0.1.shape
        switch $0.0 {
        case 0:
          return DynamicGraph.Tensor<FloatType>($0.1)[
            0..<(shape[0] / 2), 0..<shape[1], 0..<shape[2], 0..<shape[3]]
        case 1...7, (inputs.count - 2)..<inputs.count:
          return $0.1
        default:
          if injectImage {
            if ($0.0 - 8) % 6 == 1 || ($0.0 - 8) % 6 == 3 {
              return nil  // Remove positive ones.
            }
            return $0.1
          } else {
            if $0.0 % 2 == 0 {
              return $0.1
            }
            return nil
          }
        }
      }
      unet.compile(inputs: inputs)
      teaCache?.compile(model: unet, inputs: inputs)
      return
    case .flux1:
      if let teaCache = teaCache {
        let inputs = inputs.map {
          var shape = $0.shape
          guard shape[0] > 1 else {
            return $0
          }
          shape[0] = 1
          return DynamicGraph.Tensor<FloatType>($0).reshaped(format: $0.format, shape: shape)
        }
        unet.compile(inputs: inputs)
        teaCache.compile(model: unet, inputs: inputs)
        return
      }
    case .auraflow, .kandinsky21, .pixart, .sd3, .sd3Large, .sdxlBase, .sdxlRefiner,
      .ssd1b, .svdI2v, .v1, .v2, .wurstchenStageB, .wurstchenStageC:
      break
    case .hiDreamI1:
      let inputs = inputs.map {
        var shape = $0.shape
        guard shape[0] > 1 else {
          return $0
        }
        shape[0] = 1
        return DynamicGraph.Tensor<FloatType>($0).reshaped(format: $0.format, shape: shape)
      }
      unet.compile(inputs: inputs)
      return
    }
    unet.compile(inputs: inputs)
  }

  private func callAsFunction(
    step: Int,
    tokenLengthUncond: Int, tokenLengthCond: Int, isCfgEnabled: Bool,
    inputs firstInput: DynamicGraph.Tensor<FloatType>,
    _ restInputs: [DynamicGraph.AnyTensor]
  ) -> DynamicGraph.Tensor<FloatType> {
    switch version {
    case .hunyuanVideo:
      guard isCfgEnabled else {
        let shouldUseCache =
          teaCache?.shouldUseCacheForTimeEmbedding(
            [firstInput] + restInputs, model: unet!, step: step, marker: 0, of: FloatType.self)
          ?? false
        let et: DynamicGraph.Tensor<FloatType>
        if shouldUseCache,
          let result = teaCache!(model: unet!, inputs: firstInput, restInputs, marker: 0)
        {
          et = result
        } else {
          let result = unet!(
            inputs: firstInput, restInputs
          )
          et = result[0].as(of: FloatType.self)
          teaCache?.cache(outputs: result, marker: 0)
        }
        return et
      }
      let shape = firstInput.shape
      let tokenLength = max(tokenLengthUncond, tokenLengthCond)
      let etUncond: DynamicGraph.Tensor<FloatType>
      let etCond: DynamicGraph.Tensor<FloatType>
      if tokenLengthCond > tokenLengthUncond {
        // This if-clause is useful because we compiled the graph with longest token, so later we don't need to trigger the automatic re-compilation.
        let xCond = firstInput[
          (shape[0] / 2)..<shape[0], 0..<shape[1], 0..<shape[2], 0..<shape[3]
        ]
        .copied()
        let otherConds = restInputs.enumerated().map {
          let shape = $0.1.shape
          switch $0.0 {
          case 0:
            return $0.1
          case 1:
            let imageLength = shape[1] - tokenLength
            return DynamicGraph.Tensor<FloatType>($0.1)[
              0..<shape[0], 0..<(imageLength + tokenLengthCond), 0..<shape[2], 0..<shape[3]
            ].copied()
          case 2:
            return DynamicGraph.Tensor<FloatType>($0.1)[
              0..<shape[0], tokenLengthUncond..<(tokenLengthUncond + tokenLengthCond),
              0..<shape[2]
            ].copied()
          default:
            return DynamicGraph.Tensor<FloatType>($0.1)[1..<2, 0..<shape[1], 0..<shape[2]]
              .copied()
          }
        }
        // While xCond == xUncond, the pooled condition (used for adaptive layernorm) is different between cond / uncond branches, therefore, we need to check this for both cond / uncond branches.
        let shouldUseCacheCond =
          teaCache?.shouldUseCacheForTimeEmbedding(
            [xCond] + otherConds, model: unet!, step: step, marker: 0, of: FloatType.self) ?? false
        if shouldUseCacheCond,
          let result = teaCache!(model: unet!, inputs: xCond, otherConds, marker: 0)
        {
          etCond = result
        } else {
          let result = unet!(inputs: xCond, otherConds)
          etCond = result[0].as(of: FloatType.self)
          teaCache?.cache(outputs: result, marker: 0)
        }
        etCond.graph.joined()  // Wait for the result to be fully populated. Seems otherwise I can have Metal error for very large executions.
        guard !isCancelled.load(ordering: .acquiring) else {
          return Functional.concat(axis: 0, etCond, etCond)
        }
        let xUncond = firstInput[0..<(shape[0] / 2), 0..<shape[1], 0..<shape[2], 0..<shape[3]]
          .copied()
        let otherUnconds = restInputs.enumerated().map {
          let shape = $0.1.shape
          switch $0.0 {
          case 0:
            return $0.1
          case 1:
            let imageLength = shape[1] - tokenLength
            return DynamicGraph.Tensor<FloatType>($0.1)[
              0..<shape[0], 0..<(imageLength + tokenLengthUncond), 0..<shape[2], 0..<shape[3]
            ].copied()
          case 2:
            return DynamicGraph.Tensor<FloatType>($0.1)[
              0..<shape[0], 0..<tokenLengthUncond, 0..<shape[2]
            ].copied()
          default:
            return DynamicGraph.Tensor<FloatType>($0.1)[0..<1, 0..<shape[1], 0..<shape[2]]
              .copied()
          }
        }
        let shouldUseCacheUncond =
          teaCache?.shouldUseCacheForTimeEmbedding(
            [xUncond] + otherUnconds, model: unet!, step: step, marker: 1, of: FloatType.self)
          ?? false
        if shouldUseCacheUncond,
          let result = teaCache!(model: unet!, inputs: xUncond, otherUnconds, marker: 1)
        {
          etUncond = result
        } else {
          let result = unet!(inputs: xUncond, otherUnconds)
          etUncond = result[0].as(of: FloatType.self)
          teaCache?.cache(outputs: result, marker: 1)
        }
      } else {
        let xUncond = firstInput[0..<(shape[0] / 2), 0..<shape[1], 0..<shape[2], 0..<shape[3]]
          .copied()
        let otherUnconds = restInputs.enumerated().map {
          let shape = $0.1.shape
          switch $0.0 {
          case 0:
            return $0.1
          case 1:
            let imageLength = shape[1] - tokenLength
            return DynamicGraph.Tensor<FloatType>($0.1)[
              0..<shape[0], 0..<(imageLength + tokenLengthUncond), 0..<shape[2], 0..<shape[3]
            ].copied()
          case 2:
            return DynamicGraph.Tensor<FloatType>($0.1)[
              0..<shape[0], 0..<tokenLengthUncond, 0..<shape[2]
            ].copied()
          default:
            return DynamicGraph.Tensor<FloatType>($0.1)[0..<1, 0..<shape[1], 0..<shape[2]]
              .copied()
          }
        }
        let shouldUseCacheUncond =
          teaCache?.shouldUseCacheForTimeEmbedding(
            [xUncond] + otherUnconds, model: unet!, step: step, marker: 1, of: FloatType.self)
          ?? false
        if shouldUseCacheUncond,
          let result = teaCache!(model: unet!, inputs: xUncond, otherUnconds, marker: 1)
        {
          etUncond = result
        } else {
          let result = unet!(inputs: xUncond, otherUnconds)
          etUncond = result[0].as(of: FloatType.self)
          teaCache?.cache(outputs: result, marker: 1)
        }
        etUncond.graph.joined()  // Wait for the result to be fully populated. Seems otherwise I can have Metal error for very large executions.
        guard !isCancelled.load(ordering: .acquiring) else {
          return Functional.concat(axis: 0, etUncond, etUncond)
        }
        let xCond = firstInput[
          (shape[0] / 2)..<shape[0], 0..<shape[1], 0..<shape[2], 0..<shape[3]
        ]
        .copied()
        let otherConds = restInputs.enumerated().map {
          let shape = $0.1.shape
          switch $0.0 {
          case 0:
            return $0.1
          case 1:
            let imageLength = shape[1] - tokenLength
            return DynamicGraph.Tensor<FloatType>($0.1)[
              0..<shape[0], 0..<(imageLength + tokenLengthCond), 0..<shape[2], 0..<shape[3]
            ].copied()
          case 2:
            return DynamicGraph.Tensor<FloatType>($0.1)[
              0..<shape[0], tokenLengthUncond..<(tokenLengthUncond + tokenLengthCond),
              0..<shape[2]
            ].copied()
          default:
            return DynamicGraph.Tensor<FloatType>($0.1)[1..<2, 0..<shape[1], 0..<shape[2]]
              .copied()
          }
        }
        let shouldUseCacheCond =
          teaCache?.shouldUseCacheForTimeEmbedding(
            [xCond] + otherConds, model: unet!, step: step, marker: 0, of: FloatType.self) ?? false
        if shouldUseCacheCond,
          let result = teaCache!(model: unet!, inputs: xCond, otherConds, marker: 0)
        {
          etCond = result
        } else {
          let result = unet!(inputs: xCond, otherConds)
          etCond = result[0].as(of: FloatType.self)
          teaCache?.cache(outputs: result, marker: 0)
        }
      }
      return Functional.concat(axis: 0, etUncond, etCond)
    case .wan21_1_3b, .wan21_14b:
      let shouldUseCache =
        teaCache?.shouldUseCacheForTimeEmbedding(
          Array(restInputs[1..<7]), model: unet!, step: step, marker: 0, of: Float.self)
        ?? false
      guard isCfgEnabled else {
        let et: DynamicGraph.Tensor<FloatType>
        if shouldUseCache,
          let result = teaCache!(model: unet!, inputs: firstInput, restInputs, marker: 0)
        {
          et = result
        } else {
          let result = unet!(
            inputs: firstInput, restInputs
          )
          et = result[0].as(of: FloatType.self)
          teaCache?.cache(outputs: result, marker: 0)
        }
        return et
      }
      let injectImage: Bool
      if version == .wan21_1_3b {
        injectImage = restInputs.count > 9 + 4 * 30
      } else {
        injectImage = restInputs.count > 9 + 4 * 40
      }
      let shape = firstInput.shape
      let etUncond: DynamicGraph.Tensor<FloatType>
      let etCond: DynamicGraph.Tensor<FloatType>
      let xUncond = firstInput[0..<(shape[0] / 2), 0..<shape[1], 0..<shape[2], 0..<shape[3]]
        .copied()
      let restInputsUncond: [DynamicGraph.AnyTensor] = restInputs.enumerated().compactMap {
        switch $0.0 {
        case 0..<7, (restInputs.count - 2)..<restInputs.count:
          return $0.1
        default:
          if injectImage {
            if ($0.0 - 7) % 6 == 1 || ($0.0 - 7) % 6 == 3 {
              return nil  // Remove positive ones.
            }
            return $0.1
          } else {
            if $0.0 % 2 == 1 {
              return $0.1
            }
            return nil
          }
        }
      }
      if shouldUseCache,
        let uncond = teaCache!(model: unet!, inputs: xUncond, restInputsUncond, marker: 0)
      {
        etUncond = uncond
      } else {
        let result = unet!(
          inputs: xUncond, restInputsUncond
        )
        etUncond = result[0].as(of: FloatType.self)
        teaCache?.cache(outputs: result, marker: 0)
      }
      etUncond.graph.joined()  // Wait for the result to be fully populated. Seems otherwise I can have Metal error for very large executions.
      guard !isCancelled.load(ordering: .acquiring) else {
        return Functional.concat(axis: 0, etUncond, etUncond)
      }
      let xCond = firstInput[(shape[0] / 2)..<shape[0], 0..<shape[1], 0..<shape[2], 0..<shape[3]]
        .copied()
      let restInputsCond: [DynamicGraph.AnyTensor] = restInputs.enumerated().compactMap {
        switch $0.0 {
        case 0..<7, (restInputs.count - 2)..<restInputs.count:
          return $0.1
        default:
          if injectImage {
            if ($0.0 - 7) % 6 == 0 || ($0.0 - 7) % 6 == 2 {
              return nil  // Remove negative ones.
            }
            return $0.1
          } else {
            if $0.0 % 2 == 0 {
              return $0.1
            }
            return nil
          }
        }
      }
      if shouldUseCache,
        let cond = teaCache!(model: unet!, inputs: xCond, restInputsCond, marker: 1)
      {
        etCond = cond
      } else {
        let result = unet!(
          inputs: xCond, restInputsCond
        )
        etCond = result[0].as(of: FloatType.self)
        teaCache?.cache(outputs: result, marker: 1)
      }
      return Functional.concat(axis: 0, etUncond, etCond)
    case .flux1:
      if let teaCache = teaCache {
        let shape = firstInput.shape
        let batchSize = shape[0]
        guard batchSize > 1 else {
          let shouldUseCache = teaCache.shouldUseCacheForTimeEmbedding(
            [firstInput] + restInputs, model: unet!, step: step, marker: 0, of: FloatType.self)
          let et: DynamicGraph.Tensor<FloatType>
          if shouldUseCache,
            let result = teaCache(model: unet!, inputs: firstInput, restInputs, marker: 0)
          {
            et = result
          } else {
            let result = unet!(
              inputs: firstInput, restInputs
            )
            et = result[0].as(of: FloatType.self)
            teaCache.cache(outputs: result, marker: 0)
          }
          return et
        }
        let graph = firstInput.graph
        var et = graph.variable(like: firstInput)
        for i in 0..<batchSize {
          let x0 = firstInput[i..<(i + 1), 0..<shape[1], 0..<shape[2], 0..<shape[3]].copied()
          let others = restInputs.map {
            var shape = $0.shape
            guard shape[0] > 1 else { return $0 }
            shape[0] = 1
            return DynamicGraph.Tensor<FloatType>($0).reshaped(
              format: $0.format, shape: shape, offset: [i]
            ).copied()
          }
          let shouldUseCache =
            teaCache.shouldUseCacheForTimeEmbedding(
              [x0] + others, model: unet!, step: step, marker: i, of: FloatType.self)
          let et0: DynamicGraph.Tensor<FloatType>
          if shouldUseCache,
            let result = teaCache(model: unet!, inputs: x0, others, marker: i)
          {
            et0 = result
          } else {
            let result = unet!(
              inputs: x0, others
            )
            et0 = result[0].as(of: FloatType.self)
            teaCache.cache(outputs: result, marker: i)
          }
          et[i..<(i + 1), 0..<shape[1], 0..<shape[2], 0..<shape[3]] = et0
        }
        return et
      }
    case .hiDreamI1:
      let shape = firstInput.shape
      let batchSize = shape[0]
      guard batchSize > 1 else {
        let et = unet!(inputs: firstInput, restInputs)[0].as(of: FloatType.self)
        return et
      }
      let graph = firstInput.graph
      var et = graph.variable(like: firstInput)
      for i in 0..<batchSize {
        let x0 = firstInput[i..<(i + 1), 0..<shape[1], 0..<shape[2], 0..<shape[3]].copied()
        let others = restInputs.map {
          var shape = $0.shape
          guard shape[0] > 1 else { return $0 }
          shape[0] = 1
          return DynamicGraph.Tensor<FloatType>($0).reshaped(
            format: $0.format, shape: shape, offset: [i]
          ).copied()
        }
        et[i..<(i + 1), 0..<shape[1], 0..<shape[2], 0..<shape[3]] = unet!(inputs: x0, others)[0].as(
          of: FloatType.self)
      }
      return et
    case .auraflow, .kandinsky21, .pixart, .sd3, .sd3Large, .sdxlBase, .sdxlRefiner,
      .ssd1b, .svdI2v, .v1, .v2, .wurstchenStageB, .wurstchenStageC:
      break
    }
    return unet!(inputs: firstInput, restInputs)[0].as(of: FloatType.self)
  }

  private func internalDiffuse(
    xyTiles: Int, index: Int, inputStartYPad: Int, inputEndYPad: Int, inputStartXPad: Int,
    inputEndXPad: Int, xT: DynamicGraph.Tensor<FloatType>, inputs: [DynamicGraph.AnyTensor],
    injectedControlsAndAdapters: (
      _ xT: DynamicGraph.Tensor<FloatType>, _ inputStartYPad: Int, _ inputEndYPad: Int,
      _ inputStartXPad: Int, _ inputEndXPad: Int, _ existingControlNets: inout [Model?]
    ) -> (
      injectedControls: [DynamicGraph.Tensor<FloatType>],
      injectedT2IAdapters: [DynamicGraph.Tensor<FloatType>],
      injectedAttentionKVs: [DynamicGraph.Tensor<FloatType>]
    ), step: Int, tokenLengthUncond: Int, tokenLengthCond: Int, isCfgEnabled: Bool,
    controlNets: inout [Model?]
  ) -> DynamicGraph.Tensor<FloatType> {
    let shape = xT.shape
    let xT = xT[
      0..<shape[0], inputStartYPad..<inputEndYPad, inputStartXPad..<inputEndXPad, 0..<shape[3]
    ].copied()
    // Need to rework the shape. For Wurstchen B, we need to slice them up.
    // For ControlNet, we already sliced them up into batch dimension, now need to extract them out.
    let (injectedControls, injectedT2IAdapters, injectedAttentionKVs) = injectedControlsAndAdapters(
      xT, inputStartYPad, inputEndYPad, inputStartXPad, inputEndXPad, &controlNets)
    let inputs = sliceInputs(
      inputs + injectedControls + injectedT2IAdapters, originalShape: shape, xyTiles: xyTiles,
      index: index, inputStartYPad: inputStartYPad,
      inputEndYPad: inputEndYPad, inputStartXPad: inputStartXPad, inputEndXPad: inputEndXPad)
    return self(
      step: step,
      tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
      isCfgEnabled: isCfgEnabled, inputs: xT, inputs + injectedAttentionKVs)
  }

  private func tiledDiffuse(
    tiledDiffusion: TiledConfiguration, xT: DynamicGraph.Tensor<FloatType>,
    inputs: [DynamicGraph.AnyTensor],
    injectedControlsAndAdapters: (
      _ xT: DynamicGraph.Tensor<FloatType>, _ inputStartYPad: Int, _ inputEndYPad: Int,
      _ inputStartXPad: Int, _ inputEndXPad: Int, _ existingControlNets: inout [Model?]
    ) -> (
      injectedControls: [DynamicGraph.Tensor<FloatType>],
      injectedT2IAdapters: [DynamicGraph.Tensor<FloatType>],
      injectedAttentionKVs: [DynamicGraph.Tensor<FloatType>]
    ), step: Int, tokenLengthUncond: Int, tokenLengthCond: Int, isCfgEnabled: Bool,
    controlNets: inout [Model?]
  ) -> DynamicGraph.Tensor<FloatType> {
    guard let xTileWeightsAndIndexes = xTileWeightsAndIndexes,
      let yTileWeightsAndIndexes = yTileWeightsAndIndexes
    else {
      let (injectedControls, injectedT2IAdapters, injectedAttentionKVs) =
        injectedControlsAndAdapters(
          xT, 0, 0, 0, 0, &controlNets)
      return self(
        step: step,
        tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
        isCfgEnabled: isCfgEnabled,
        inputs: xT, inputs + injectedControls + injectedT2IAdapters + injectedAttentionKVs)
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
        tiledDiffusion.tileOverlap * tileScaleFactor / 2,
        Int((Double(tiledHeight / 3) / Double(tileScaleFactor)).rounded(.down)) * tileScaleFactor),
      Int((Double(tiledWidth / 3) / Double(tileScaleFactor)).rounded(.down)) * tileScaleFactor)
    let yTiles =
      (startHeight - tileOverlap * 2 + (tiledHeight - tileOverlap * 2) - 1)
      / (tiledHeight - tileOverlap * 2)
    let xTiles =
      (startWidth - tileOverlap * 2 + (tiledWidth - tileOverlap * 2) - 1)
      / (tiledWidth - tileOverlap * 2)
    var et = [DynamicGraph.Tensor<FloatType>]()
    let graph = xT.graph
    guard !isCancelled.load(ordering: .acquiring) else {
      return graph.variable(
        Tensor<FloatType>(.GPU(0), .NHWC(shape[0], startHeight, startWidth, shape[3])))
    }
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
            step: step, tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
            isCfgEnabled: isCfgEnabled, controlNets: &controlNets))
        guard !isCancelled.load(ordering: .acquiring) else {
          return graph.variable(
            Tensor<FloatType>(.GPU(0), .NHWC(shape[0], startHeight, startWidth, shape[3])))
        }
      }
    }
    graph.joined()
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
    return graph.variable(etRaw.toGPU(0))
  }

  public func callAsFunction(
    timestep _: Float,
    inputs xT: DynamicGraph.Tensor<FloatType>, _ timestep: DynamicGraph.Tensor<FloatType>?,
    _ c: [DynamicGraph.AnyTensor], extraProjection: DynamicGraph.Tensor<FloatType>?,
    injectedControlsAndAdapters: (
      _ xT: DynamicGraph.Tensor<FloatType>, _ inputStartYPad: Int, _ inputEndYPad: Int,
      _ inputStartXPad: Int, _ inputEndXPad: Int, _ existingControlNets: inout [Model?]
    ) -> (
      injectedControls: [DynamicGraph.Tensor<FloatType>],
      injectedT2IAdapters: [DynamicGraph.Tensor<FloatType>],
      injectedAttentionKVs: [NNC.DynamicGraph.Tensor<FloatType>]
    ),
    injectedIPAdapters: [DynamicGraph.Tensor<FloatType>], step: Int,
    tokenLengthUncond: Int, tokenLengthCond: Int, isCfgEnabled: Bool,
    tiledDiffusion: TiledConfiguration, controlNets: inout [Model?]
  ) -> DynamicGraph.Tensor<FloatType> {
    if let extraProjection = extraProjection, let timeEmbed = timeEmbed, let timestep = timestep {
      let batchSize = xT.shape[0]
      var embGPU = timeEmbed(inputs: timestep)[0].as(of: FloatType.self)
      embGPU = embGPU + extraProjection.reshaped(.NC(batchSize, 384 * 4))
      if tiledDiffusion.isEnabled {
        return tiledDiffuse(
          tiledDiffusion: tiledDiffusion, xT: xT, inputs: [embGPU, c[0]],
          injectedControlsAndAdapters: injectedControlsAndAdapters, step: step,
          tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
          isCfgEnabled: isCfgEnabled, controlNets: &controlNets)
      } else {
        return self(
          step: step,
          tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
          isCfgEnabled: isCfgEnabled, inputs: xT, [embGPU, c[0]])
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
      case .flux1:
        let injectIPAdapters = injectedIPAdapters.count / 40
        var newC = c
        for i in stride(from: 0, to: 40, by: 2) {
          for j in 0..<injectIPAdapters {
            newC.append(injectedIPAdapters[i + j * 40])  // ip_k
            newC.append(injectedIPAdapters[i + 1 + j * 40])  // ip_v
          }
        }
        c = newC
      case .v2, .sd3, .sd3Large, .pixart, .auraflow, .kandinsky21, .svdI2v, .wurstchenStageC,
        .wurstchenStageB, .hunyuanVideo, .wan21_1_3b, .wan21_14b, .hiDreamI1:
        fatalError()
      }
    }
    if tiledDiffusion.isEnabled {
      return tiledDiffuse(
        tiledDiffusion: tiledDiffusion, xT: xT, inputs: (timestep.map { [$0] } ?? []) + c,
        injectedControlsAndAdapters: injectedControlsAndAdapters, step: step,
        tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
        isCfgEnabled: isCfgEnabled, controlNets: &controlNets)
    } else {
      let (injectedControls, injectedT2IAdapters, injectedAttentionKVs) =
        injectedControlsAndAdapters(
          xT, 0, 0, 0, 0, &controlNets)
      return self(
        step: step,
        tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
        isCfgEnabled: isCfgEnabled, inputs: xT,
        (timestep.map { [$0] } ?? []) + c + injectedControls + injectedT2IAdapters
          + injectedAttentionKVs)
    }
  }

  public func decode(_ x: DynamicGraph.Tensor<FloatType>) -> DynamicGraph.Tensor<FloatType> {
    switch version {
    case .wurstchenStageC:
      if let previewer = previewer {
        return previewer(inputs: x)[0].as(of: FloatType.self)
      }
      return x
    case .v1, .v2, .sd3, .sd3Large, .pixart, .auraflow, .flux1, .sdxlBase, .sdxlRefiner, .ssd1b,
      .svdI2v, .kandinsky21, .wurstchenStageB, .hunyuanVideo, .wan21_1_3b, .wan21_14b, .hiDreamI1:
      return x
    }
  }

  public func cancel() {
    isCancelled.store(true, ordering: .releasing)
    unet?.cancel()
  }
}
