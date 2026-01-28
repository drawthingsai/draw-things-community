import Collections
import NNC
import WeightsCache

#if canImport(C_ccv)
  import C_ccv
#elseif canImport(C_swiftpm_ccv)
  import C_swiftpm_ccv
#endif

public struct TextEncoder<FloatType: TensorNumeric & BinaryFloatingPoint> {
  public let filePaths: [String]
  public let version: ModelVersion
  public let textEncoderVersion: TextEncoderVersion?
  public let isCfgEnabled: Bool
  public let usesFlashAttention: Bool
  public let injectEmbeddings: Bool
  public let maxLength: Int
  public let clipSkip: Int
  public let lora: [LoRAConfiguration]
  public let externalOnDemand: Bool
  public let deviceProperties: DeviceProperties
  private let weightsCache: WeightsCache
  public init(
    filePaths: [String], version: ModelVersion, textEncoderVersion: TextEncoderVersion?,
    isCfgEnabled: Bool, usesFlashAttention: Bool, injectEmbeddings: Bool, externalOnDemand: Bool,
    deviceProperties: DeviceProperties, weightsCache: WeightsCache, maxLength: Int = 77,
    clipSkip: Int = 1, lora: [LoRAConfiguration] = []
  ) {
    self.filePaths = filePaths
    self.version = version
    self.textEncoderVersion = textEncoderVersion
    self.isCfgEnabled = isCfgEnabled
    self.usesFlashAttention = usesFlashAttention
    self.injectEmbeddings = injectEmbeddings
    self.externalOnDemand = externalOnDemand
    self.deviceProperties = deviceProperties
    self.weightsCache = weightsCache
    self.maxLength = maxLength
    self.clipSkip = clipSkip
    self.lora = lora.filter { $0.version == version }
  }
}

extension TextEncoder {
  private func encodeKandinsky(
    tokens: [DynamicGraph.Tensor<Int32>], positions: [DynamicGraph.Tensor<Int32>]
  ) -> ([DynamicGraph.Tensor<FloatType>], [Model]) {
    let graph = tokens[0].graph
    let tokensTensor = tokens[0]
    var unconditionalTokenLength: Int? = nil
    var tokenLength: Int? = nil
    for i in 0..<77 {
      if tokensTensor[i] == 2 && unconditionalTokenLength == nil {
        unconditionalTokenLength = i + 1
      }
      if tokensTensor[i + 77] == 2 && tokenLength == nil {
        tokenLength = i + 1
      }
    }
    let attentionMask = graph.variable(.CPU, .NHWC(2, 1, 1, 77), of: FloatType.self)
    for i in 0..<77 {
      attentionMask[0, 0, 0, i] = 0
      attentionMask[1, 0, 0, i] = 0
    }
    if let unconditionalTokenLength = unconditionalTokenLength {
      for i in unconditionalTokenLength..<77 {
        attentionMask[0, 0, 0, i] = -FloatType.greatestFiniteMagnitude
      }
    }
    if let tokenLength = tokenLength {
      for i in tokenLength..<77 {
        attentionMask[1, 0, 0, i] = -FloatType.greatestFiniteMagnitude
      }
    }
    var causalAttentionMask = Tensor<FloatType>(
      Array(repeating: 0, count: maxLength * maxLength), .CPU, .NHWC(1, 1, maxLength, maxLength)
    )
    for i in 0..<(maxLength - 1) {
      for j in (i + 1)..<maxLength {
        causalAttentionMask[0, 0, i, j] = -FloatType.greatestFiniteMagnitude
      }
    }
    var fullEmb: DynamicGraph.Tensor<FloatType>? = nil
    var poolEmb: DynamicGraph.Tensor<FloatType>? = nil
    let externalData: DynamicGraph.Store.Codec =
      externalOnDemand
      ? .externalOnDemand : .externalData(deviceProperties.isFreadPreferred ? .fread : .mmap)
    graph.openStore(
      filePaths[0], flags: .readOnly,
      externalStore: TensorData.externalStore(filePath: filePaths[0])
    ) {
      let tokensTensorGPU = tokensTensor.toGPU(0)
      let positionTensorGPU = positions[0].toGPU(0)
      let tokenTypesTensor = graph.variable(.CPU, .C(2 * 77), of: Int32.self)
      for i in 0..<(2 * 77) {
        tokenTypesTensor[i] = 0
      }
      let tokenTypesTensorGPU = tokenTypesTensor.toGPU(0)
      let attentionMaskGPU = attentionMask.toGPU(0)
      let textEncoder = XLMRobertaTextEmbedding(
        FloatType.self, prefix: "model.transformer.embeddings", vocabularySize: 250_002,
        maxLength: 514, tokenTypes: 1, embeddingSize: 1_024)
      textEncoder.compile(inputs: tokensTensorGPU, positionTensorGPU, tokenTypesTensorGPU)
      $0.read("embedding", model: textEncoder, codec: [.jit, .q6p, .q8p, .ezm7, externalData])
      let embeddings = textEncoder(inputs: tokensTensorGPU, positionTensorGPU, tokenTypesTensorGPU)[
        0
      ].as(of: FloatType.self)
      let layer = XLMRobertaModel(numberOfLayers: 24, k: 64, h: 16, b: 2, t: 77)
      layer.compile(inputs: embeddings, attentionMaskGPU)
      $0.read("roberta", model: layer, codec: [.jit, .q6p, .q8p, .ezm7, externalData])
      let textEncoderEmb = layer(inputs: embeddings, attentionMaskGPU)[0].as(of: FloatType.self)
        .reshaped(.HWC(2, 77, 1024))
      fullEmb = textEncoderEmb
      let poolingMask = graph.variable(.CPU, .HWC(2, 1, 77), of: FloatType.self)
      let weightedMask = graph.variable(.CPU, .HWC(2, 1, 1), of: FloatType.self)
      for i in 0..<77 {
        poolingMask[0, 0, i] = i < (unconditionalTokenLength ?? 77) ? 1 : 0
        poolingMask[1, 0, i] = i < (tokenLength ?? 77) ? 1 : 0
      }
      weightedMask[0, 0, 0] = FloatType(1 / Float(unconditionalTokenLength ?? 77))
      weightedMask[1, 0, 0] = FloatType(1 / Float(tokenLength ?? 77))
      let middlePoolEmb = weightedMask.toGPU(0) .* (poolingMask.toGPU(0) * textEncoderEmb)
      let linearTransformation = Dense(count: 768)
      linearTransformation.compile(inputs: middlePoolEmb)
      $0.read(
        "linear_transformation", model: linearTransformation,
        codec: [.jit, .q6p, .q8p, .ezm7, externalData])
      poolEmb = linearTransformation(inputs: middlePoolEmb)[0].as(of: FloatType.self)
    }
    var CLIPTextEmb: DynamicGraph.Tensor<FloatType>? = nil
    var CLIPTextEnc: DynamicGraph.Tensor<FloatType>? = nil
    graph.openStore(
      filePaths[1], flags: .readOnly,
      externalStore: TensorData.externalStore(filePath: filePaths[1])
    ) { store in
      let tokensTensor = tokens[1]
      var unconditionalTokenLength = 77
      var tokenLength = 77
      for i in 0..<77 {
        if tokensTensor[i] == 49407 && unconditionalTokenLength == 77 {
          unconditionalTokenLength = i + 1
        }
        if tokensTensor[i + 77] == 49407 && tokenLength == 77 {
          tokenLength = i + 1
        }
      }
      let CLIPTokensTensorGPU = tokensTensor.toGPU(0)
      let CLIPPositionTensorGPU = positions[1].toGPU(0)
      let causalAttentionMaskGPU = graph.variable(causalAttentionMask.toGPU())
      let textModel = CLIPTextModel(
        FloatType.self, injectEmbeddings: false,
        vocabularySize: 49408, maxLength: 77, maxTokenLength: maxLength, embeddingSize: 768,
        numLayers: 12, numHeads: 12, batchSize: 3, intermediateSize: 3072,
        usesFlashAttention: usesFlashAttention
      ).0
      textModel.compile(inputs: CLIPTokensTensorGPU, CLIPPositionTensorGPU, causalAttentionMaskGPU)
      if lora.count > 0 {
        LoRALoader.openStore(graph, lora: lora) { loader in
          if clipSkip > 1 {
            store.read(
              "text_model", model: textModel, codec: [.jit, .q6p, .q8p, .ezm7, externalData]
            ) { name, dataType, _, shape in
              // Retrieve the right final layer norm parameters.
              var name = name
              if name == "__text_model__[t-\(98 - (min(clipSkip, 12) - 1) * 8)-0]" {
                name = "__text_model__[t-98-0]"
              } else if name == "__text_model__[t-\(98 - (min(clipSkip, 12) - 1) * 8)-1]" {
                name = "__text_model__[t-98-1]"
              }
              return loader.mergeLoRA(
                graph, name: name, store: store, dataType: dataType, shape: shape,
                of: FloatType.self)
            }
          } else {
            store.read(
              "text_model", model: textModel, codec: [.jit, .q6p, .q8p, .ezm7, externalData]
            ) { name, dataType, _, shape in
              return loader.mergeLoRA(
                graph, name: name, store: store, dataType: dataType, shape: shape,
                of: FloatType.self)
            }
          }
        }
      } else {
        if clipSkip > 1 {
          store.read(
            "text_model", model: textModel, codec: [.jit, .q6p, .q8p, .ezm7, externalData]
          ) { name, _, _, _ in
            // Retrieve the right final layer norm parameters.
            var name = name
            if name == "__text_model__[t-\(98 - (min(clipSkip, 12) - 1) * 8)-0]" {
              name = "__text_model__[t-98-0]"
            } else if name == "__text_model__[t-\(98 - (min(clipSkip, 12) - 1) * 8)-1]" {
              name = "__text_model__[t-98-1]"
            }
            return .continue(name)
          }
        } else {
          store.read(
            "text_model", model: textModel, codec: [.jit, .q6p, .q8p, .ezm7, externalData])
        }
      }
      let c = textModel(inputs: CLIPTokensTensorGPU, CLIPPositionTensorGPU, causalAttentionMaskGPU)[
        0
      ].as(of: FloatType.self)
      let tensorIndex = graph.variable(.CPU, .C(3), of: Int32.self)
      tensorIndex[0] = Int32(unconditionalTokenLength) - 1
      tensorIndex[1] = Int32(tokenLength) + 77 - 1
      tensorIndex[2] = 77 * 2 + 1
      CLIPTextEmb = Functional.indexSelect(
        input: c.reshaped(.WC(3 * 77, 768)), index: tensorIndex.toGPU(0))
      CLIPTextEnc = c.reshaped(.HWC(3, 77, 768))
    }
    return ([fullEmb!, poolEmb!, CLIPTextEnc!, CLIPTextEmb!], [])
  }

  private func encodeSDXL(
    tokens: [DynamicGraph.Tensor<Int32>], positions: [DynamicGraph.Tensor<Int32>],
    mask: [DynamicGraph.Tensor<FloatType>], injectedEmbeddings: [DynamicGraph.Tensor<FloatType>],
    lengthsOfUncond: [Int], lengthsOfCond: [Int], textModels existingTextModels: [Model?]
  )
    -> ([DynamicGraph.Tensor<FloatType>], [Model])
  {
    var causalAttentionMask = Tensor<FloatType>(
      Array(repeating: 0, count: 2 * maxLength * maxLength), .CPU, .NHWC(2, 1, maxLength, maxLength)
    )
    for i in 0..<(maxLength - 1) {
      for j in (i + 1)..<maxLength {
        causalAttentionMask[0, 0, i, j] = -FloatType.greatestFiniteMagnitude
        causalAttentionMask[1, 0, i, j] = -FloatType.greatestFiniteMagnitude
      }
    }
    var j = 0
    var prefixLength = 0
    for i in 0..<maxLength {
      // Mask out anything before this, except padding / ending.
      guard j < lengthsOfUncond.count else { break }
      if i - 1 >= lengthsOfUncond[j] + prefixLength {
        prefixLength += lengthsOfUncond[j]
        j += 1
      }
      if prefixLength > 0 && j < lengthsOfUncond.count {
        for k in 1..<(prefixLength + 1) {
          causalAttentionMask[0, 0, i, k] = -FloatType.greatestFiniteMagnitude
        }
      }
    }
    j = 0
    prefixLength = 0
    for i in 0..<maxLength {
      // Mask out anything before this, except padding / ending.
      guard j < lengthsOfCond.count else { break }
      if i - 1 >= lengthsOfCond[j] + prefixLength {
        prefixLength += lengthsOfCond[j]
        j += 1
      }
      if prefixLength > 0 && j < lengthsOfCond.count {
        for k in 1..<(prefixLength + 1) {
          causalAttentionMask[1, 0, i, k] = -FloatType.greatestFiniteMagnitude
        }
      }
    }
    let graph = tokens[0].graph
    let tokens0TensorGPU = tokens[0].toGPU(0)
    let positionTensorGPU = positions[0].toGPU(0)
    let causalAttentionMaskGPU = graph.variable(causalAttentionMask.toGPU())
    let maskGPU = mask.map { $0.toGPU(0) }
    let injectedEmbeddingsGPU = injectedEmbeddings.map { $0.toGPU(0) }
    let externalData: DynamicGraph.Store.Codec =
      externalOnDemand
      ? .externalOnDemand : .externalData(deviceProperties.isFreadPreferred ? .fread : .mmap)
    var textModel: Model
    textModel =
      CLIPTextModel(
        FloatType.self, injectEmbeddings: injectEmbeddings,
        vocabularySize: 49408, maxLength: 77, maxTokenLength: maxLength, embeddingSize: 768,
        numLayers: 13 - min(max(clipSkip, 1), 12), numHeads: 12, batchSize: 2,
        intermediateSize: 3072, usesFlashAttention: usesFlashAttention, noFinalLayerNorm: true
      ).0
    if let maskGPU = maskGPU.first, let injectedEmbeddingsGPU = injectedEmbeddingsGPU.first {
      textModel.compile(
        inputs: tokens0TensorGPU, positionTensorGPU, causalAttentionMaskGPU, maskGPU,
        injectedEmbeddingsGPU)
    } else {
      textModel.compile(inputs: tokens0TensorGPU, positionTensorGPU, causalAttentionMaskGPU)
    }
    let c0: DynamicGraph.Tensor<FloatType>
    if filePaths.count > 1 {
      graph.openStore(
        filePaths[1], flags: .readOnly,
        externalStore: TensorData.externalStore(filePath: filePaths[1])
      ) { store in
        if lora.count > 0 {
          LoRALoader.openStore(graph, lora: lora) { loader in
            store.read(
              "text_model", model: textModel, codec: [.jit, .q6p, .q8p, .ezm7, externalData]
            ) { name, dataType, _, shape in
              var name = name
              if name == "__text_model__[t-\(98 - (min(clipSkip, 12) - 1) * 8)-0]" {
                name = "__text_model__[t-98-0]"
              } else if name == "__text_model__[t-\(98 - (min(clipSkip, 12) - 1) * 8)-1]" {
                name = "__text_model__[t-98-1]"
              }
              return loader.mergeLoRA(
                graph, name: name, store: store, dataType: dataType, shape: shape,
                of: FloatType.self)
            }
          }
        } else {
          if clipSkip > 1 {
            store.read(
              "text_model", model: textModel, codec: [.jit, .q6p, .q8p, .ezm7, externalData]
            ) { name, _, _, _ in
              // Retrieve the right final layer norm parameters.
              var name = name
              if name == "__text_model__[t-\(98 - (min(clipSkip, 12) - 1) * 8)-0]" {
                name = "__text_model__[t-98-0]"
              } else if name == "__text_model__[t-\(98 - (min(clipSkip, 12) - 1) * 8)-1]" {
                name = "__text_model__[t-98-1]"
              }
              return .continue(name)
            }
          } else {
            store.read(
              "text_model", model: textModel, codec: [.jit, .q6p, .q8p, .ezm7, externalData])
          }
        }
      }
      if let maskGPU = maskGPU.first, let injectedEmbeddingsGPU = injectedEmbeddingsGPU.first {
        c0 = textModel(
          inputs: tokens0TensorGPU, positionTensorGPU, causalAttentionMaskGPU, maskGPU,
          injectedEmbeddingsGPU)[0].as(
            of: FloatType.self
          ).reshaped(.HWC(2, maxLength, 768))
      } else {
        c0 = textModel(
          inputs: tokens0TensorGPU, positionTensorGPU, causalAttentionMaskGPU)[0].as(
            of: FloatType.self
          ).reshaped(.HWC(2, maxLength, 768))
      }
    } else {
      c0 = graph.variable(.GPU(0), .HWC(2, maxLength, 768))
      c0.full(0)
    }
    let tokens1TensorGPU = tokens[1].toGPU(0)
    if let existingTextModel = existingTextModels[0] {
      textModel = existingTextModel
    } else {
      textModel =
        OpenCLIPTextModel(
          FloatType.self, injectEmbeddings: injectEmbeddings,
          vocabularySize: 49408, maxLength: 77, maxTokenLength: maxLength, embeddingSize: 1280,
          numLayers: 32 - min(max(clipSkip - 2, 0), 30), numHeads: 20, batchSize: 2,
          intermediateSize: 5120, usesFlashAttention: usesFlashAttention, outputPenultimate: true
        ).0
    }
    if let maskGPU = maskGPU.last, let injectedEmbeddingsGPU = injectedEmbeddingsGPU.last {
      textModel.compile(
        inputs: tokens1TensorGPU, positionTensorGPU, causalAttentionMaskGPU, maskGPU,
        injectedEmbeddingsGPU)
    } else {
      textModel.compile(
        inputs: tokens1TensorGPU, positionTensorGPU, causalAttentionMaskGPU)
    }
    let textProjection = graph.variable(.GPU(0), .WC(1280, 1280), of: FloatType.self)
    graph.openStore(
      filePaths[0], flags: .readOnly,
      externalStore: TensorData.externalStore(filePath: filePaths[0])
    ) { store in
      if lora.count > 0 {
        LoRALoader.openStore(graph, lora: lora) { loader in
          store.read(
            "text_model", model: textModel, codec: [.jit, .q6p, .q8p, .ezm7, externalData]
          ) { name, dataType, _, shape in
            // Retrieve the right final layer norm parameters.
            var name = name
            if name == "__text_model__[t-\(258 - (min(max(clipSkip - 1, 1), 31) - 1) * 8)-0]" {
              name = "__text_model__[t-258-0]"
            } else if name == "__text_model__[t-\(258 - (min(max(clipSkip - 1, 1), 31) - 1) * 8)-1]"
            {
              name = "__text_model__[t-258-1]"
            }
            return loader.mergeLoRA(
              graph, name: name, store: store, dataType: dataType, shape: shape, of: FloatType.self,
              prefix: "__te2")
          }
        }
      } else if clipSkip > 1 {
        store.read("text_model", model: textModel, codec: [.jit, .q6p, .q8p, .ezm7, externalData]) {
          name, _, _, _ in
          // Retrieve the right final layer norm parameters.
          var name = name
          if name == "__text_model__[t-\(258 - (min(max(clipSkip - 1, 1), 31) - 1) * 8)-0]" {
            name = "__text_model__[t-258-0]"
          } else if name == "__text_model__[t-\(258 - (min(max(clipSkip - 1, 1), 31) - 1) * 8)-1]" {
            name = "__text_model__[t-258-1]"
          }
          return .continue(name)
        }
      } else {
        store.read("text_model", model: textModel, codec: [.jit, .q6p, .q8p, .ezm7, externalData])
      }
      store.read(
        "text_projection", variable: textProjection,
        codec: [
          .q6p, .q8p, .ezm7, .externalData(deviceProperties.isFreadPreferred ? .fread : .mmap),
        ])
    }
    let c1Out: [DynamicGraph.Tensor<FloatType>]
    if let maskGPU = maskGPU.last, let injectedEmbeddingsGPU = injectedEmbeddingsGPU.last {
      c1Out = textModel(
        inputs: tokens1TensorGPU, positionTensorGPU, causalAttentionMaskGPU, maskGPU,
        injectedEmbeddingsGPU
      ).map { $0.as(of: FloatType.self) }
    } else {
      c1Out = textModel(
        inputs: tokens1TensorGPU, positionTensorGPU, causalAttentionMaskGPU
      ).map { $0.as(of: FloatType.self) }
    }
    let c1 = c1Out[0].reshaped(.HWC(2, maxLength, 1280))
    var pooled = graph.variable(.GPU(0), .WC(2, 1280), of: FloatType.self)
    var unconditionalTokenEnd: Int? = nil
    var tokenEnd: Int? = nil
    if mask.count > 1 {
      for i in 0..<maxLength {
        if tokens[1][i] == 49407 && mask[1][i, 0] > 0 && unconditionalTokenEnd == nil {
          unconditionalTokenEnd = i
        }
        if tokens[1][i + maxLength] == 49407 && mask[1][i + maxLength, 0] > 0 && tokenEnd == nil {
          tokenEnd = i
        }
      }
    } else {
      for i in 0..<maxLength {
        if tokens[1][i] == 49407 && unconditionalTokenEnd == nil {
          unconditionalTokenEnd = i
        }
        if tokens[1][i + maxLength] == 49407 && tokenEnd == nil {
          tokenEnd = i
        }
      }
    }
    if let unconditionalTokenEnd = unconditionalTokenEnd, let tokenEnd = tokenEnd {
      pooled[0..<1, 0..<1280] =
        c1Out[1][unconditionalTokenEnd..<(unconditionalTokenEnd + 1), 0..<1280] * textProjection
      pooled[1..<2, 0..<1280] =
        c1Out[1][(maxLength + tokenEnd)..<(maxLength + tokenEnd + 1), 0..<1280] * textProjection
    }
    return ([c0, c1, pooled], [textModel])
  }

  private func encodeI2v(
    images: [DynamicGraph.Tensor<FloatType>], textModels existingTextModels: [Model?]
  ) -> ([DynamicGraph.Tensor<FloatType>], [Model]) {
    let graph = images[0].graph
    let vit: Model
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
    var input = images[0]
    let inputHeight = input.shape[1]
    let inputWidth = input.shape[2]
    precondition(input.shape[3] == 3)
    if inputHeight != 224 || inputWidth != 224 {
      input =
        (Upsample(
          .bilinear, widthScale: Float(224) / Float(inputWidth),
          heightScale: Float(224) / Float(inputHeight))(input) - mean) .* invStd
    } else {
      input = (input - mean) .* invStd
    }
    let externalData: DynamicGraph.Store.Codec =
      externalOnDemand
      ? .externalOnDemand : .externalData(deviceProperties.isFreadPreferred ? .fread : .mmap)
    if existingTextModels.count >= 1, let existingTextModel = existingTextModels[0] {
      vit = existingTextModel
    } else {
      vit = VisionTransformer(
        FloatType.self, grid: 16, width: 1280, layers: 32, heads: 16, batchSize: 1)
      vit.compile(inputs: input)
      graph.openStore(
        filePaths[0], flags: .readOnly,
        externalStore: TensorData.externalStore(filePath: filePaths[0])
      ) {
        $0.read("vision_model", model: vit, codec: [.jit, .q6p, .q8p, .ezm7, externalData])
      }
    }
    let imageEmbeds = vit(inputs: input)[0].as(of: FloatType.self).reshaped(.CHW(1, 1, 1280))
    let visualProj: Model
    if existingTextModels.count >= 2, let existingTextModel = existingTextModels[1] {
      visualProj = existingTextModel
    } else {
      visualProj = Dense(count: 1024, noBias: true)
      visualProj.compile(inputs: imageEmbeds)
      graph.openStore(
        filePaths[1], flags: .readOnly,
        externalStore: TensorData.externalStore(filePath: filePaths[1])
      ) {
        $0.read(
          "visual_proj", model: visualProj,
          codec: [
            .jit, .q6p, .q8p, .ezm7,
            .externalData(deviceProperties.isFreadPreferred ? .fread : .mmap),
          ])
      }
    }
    let imageProj = visualProj(inputs: imageEmbeds)[0].as(of: FloatType.self)
    return ([imageProj], [vit, visualProj])
  }

  private func encodeWurstchen(
    tokens: [DynamicGraph.Tensor<Int32>], positions: [DynamicGraph.Tensor<Int32>],
    mask: [DynamicGraph.Tensor<FloatType>], injectedEmbeddings: [DynamicGraph.Tensor<FloatType>],
    lengthsOfUncond: [Int], lengthsOfCond: [Int], textModels existingTextModels: [Model?]
  )
    -> ([DynamicGraph.Tensor<FloatType>], [Model])
  {
    var causalAttentionMask = Tensor<FloatType>(
      Array(repeating: 0, count: 2 * maxLength * maxLength), .CPU, .NHWC(2, 1, maxLength, maxLength)
    )
    var unconditionalTokenEnd: Int? = nil
    var tokenEnd: Int? = nil
    if mask.count > 1 {
      for i in 0..<maxLength {
        if tokens[0][i] == 49407 && mask[0][i, 0] > 0 && unconditionalTokenEnd == nil {
          unconditionalTokenEnd = i
        }
        if tokens[0][i + maxLength] == 49407 && mask[0][i + maxLength, 0] > 0 && tokenEnd == nil {
          tokenEnd = i
        }
      }
    } else {
      for i in 0..<maxLength {
        if tokens[0][i] == 49407 && unconditionalTokenEnd == nil {
          unconditionalTokenEnd = i
        }
        if tokens[0][i + maxLength] == 49407 && tokenEnd == nil {
          tokenEnd = i
        }
      }
    }
    for i in 0..<maxLength {
      for j in (i + 1)..<maxLength {
        causalAttentionMask[0, 0, i, j] = -FloatType.greatestFiniteMagnitude
        causalAttentionMask[1, 0, i, j] = -FloatType.greatestFiniteMagnitude
      }
      // For Wurstchen, padding tokens are masked out.
      if tokens[0][i] == 49407, let unconditionalTokenEnd = unconditionalTokenEnd,
        i > unconditionalTokenEnd
      {
        for j in (unconditionalTokenEnd + 1)..<(i + 1) {
          causalAttentionMask[0, 0, i, j] = -FloatType.greatestFiniteMagnitude
        }
      }
      if tokens[0][i + maxLength] == 49407, let tokenEnd = tokenEnd, i > tokenEnd {
        for j in (tokenEnd + 1)..<(i + 1) {
          causalAttentionMask[1, 0, i, j] = -FloatType.greatestFiniteMagnitude
        }
      }
    }
    var j = 0
    var prefixLength = 0
    for i in 0..<maxLength {
      // Mask out anything before this, except padding / ending.
      guard j < lengthsOfUncond.count else { break }
      if i - 1 >= lengthsOfUncond[j] + prefixLength {
        prefixLength += lengthsOfUncond[j]
        j += 1
      }
      if prefixLength > 0 && j < lengthsOfUncond.count {
        for k in 1..<(prefixLength + 1) {
          causalAttentionMask[0, 0, i, k] = -FloatType.greatestFiniteMagnitude
        }
      }
    }
    j = 0
    prefixLength = 0
    for i in 0..<maxLength {
      // Mask out anything before this, except padding / ending.
      guard j < lengthsOfCond.count else { break }
      if i - 1 >= lengthsOfCond[j] + prefixLength {
        prefixLength += lengthsOfCond[j]
        j += 1
      }
      if prefixLength > 0 && j < lengthsOfCond.count {
        for k in 1..<(prefixLength + 1) {
          causalAttentionMask[1, 0, i, k] = -FloatType.greatestFiniteMagnitude
        }
      }
    }
    let graph = tokens[0].graph
    let tokensTensorGPU = tokens[0].toGPU(0)
    let positionTensorGPU = positions[0].toGPU(0)
    let causalAttentionMaskGPU = graph.variable(causalAttentionMask.toGPU())
    let maskGPU = mask.map { $0.toGPU(0) }
    let injectedEmbeddingsGPU = injectedEmbeddings.map { $0.toGPU(0) }
    let textModel: Model
    let externalData: DynamicGraph.Store.Codec =
      externalOnDemand
      ? .externalOnDemand : .externalData(deviceProperties.isFreadPreferred ? .fread : .mmap)
    if let existingTextModel = existingTextModels[0] {
      textModel = existingTextModel
    } else {
      textModel =
        OpenCLIPTextModel(
          FloatType.self, injectEmbeddings: injectEmbeddings,
          vocabularySize: 49408, maxLength: 77, maxTokenLength: maxLength, embeddingSize: 1280,
          numLayers: 32 - min(max(clipSkip - 1, 0), 30), numHeads: 20, batchSize: 2,
          intermediateSize: 5120, usesFlashAttention: usesFlashAttention, outputHiddenState: true
        ).0
    }
    if let maskGPU = maskGPU.last, let injectedEmbeddingsGPU = injectedEmbeddingsGPU.last {
      textModel.compile(
        inputs: tokensTensorGPU, positionTensorGPU, causalAttentionMaskGPU, maskGPU,
        injectedEmbeddingsGPU)
    } else {
      textModel.compile(
        inputs: tokensTensorGPU, positionTensorGPU, causalAttentionMaskGPU)
    }
    let textProjection = graph.variable(.GPU(0), .WC(1280, 1280), of: FloatType.self)
    graph.openStore(
      filePaths[0], flags: .readOnly,
      externalStore: TensorData.externalStore(filePath: filePaths[0])
    ) { store in
      if lora.count > 0 {
        LoRALoader.openStore(graph, lora: lora) { loader in
          store.read(
            "text_model", model: textModel, codec: [.jit, .q6p, .q8p, .ezm7, externalData]
          ) { name, dataType, _, shape in
            // Retrieve the right final layer norm parameters.
            var name = name
            if name == "__text_model__[t-\(258 - (min(max(clipSkip - 1, 1), 31) - 1) * 8)-0]" {
              name = "__text_model__[t-258-0]"
            } else if name == "__text_model__[t-\(258 - (min(max(clipSkip - 1, 1), 31) - 1) * 8)-1]"
            {
              name = "__text_model__[t-258-1]"
            }
            return loader.mergeLoRA(
              graph, name: name, store: store, dataType: dataType, shape: shape, of: FloatType.self,
              prefix: "__te2")
          }
        }
      } else if clipSkip > 1 {
        store.read("text_model", model: textModel, codec: [.jit, .q6p, .q8p, .ezm7, externalData]) {
          name, _, _, _ in
          // Retrieve the right final layer norm parameters.
          var name = name
          if name == "__text_model__[t-\(258 - (min(max(clipSkip - 1, 1), 31) - 1) * 8)-0]" {
            name = "__text_model__[t-258-0]"
          } else if name == "__text_model__[t-\(258 - (min(max(clipSkip - 1, 1), 31) - 1) * 8)-1]" {
            name = "__text_model__[t-258-1]"
          }
          return .continue(name)
        }
      } else {
        store.read("text_model", model: textModel, codec: [.jit, .q6p, .q8p, .ezm7, externalData])
      }
      store.read(
        "text_projection", variable: textProjection,
        codec: [
          .q6p, .q8p, .ezm7, .externalData(deviceProperties.isFreadPreferred ? .fread : .mmap),
        ])
    }
    let cOut: [DynamicGraph.Tensor<FloatType>]
    if let maskGPU = maskGPU.last, let injectedEmbeddingsGPU = injectedEmbeddingsGPU.last {
      cOut = textModel(
        inputs: tokensTensorGPU, positionTensorGPU, causalAttentionMaskGPU, maskGPU,
        injectedEmbeddingsGPU
      ).map { $0.as(of: FloatType.self) }
    } else {
      cOut = textModel(
        inputs: tokensTensorGPU, positionTensorGPU, causalAttentionMaskGPU
      ).map { $0.as(of: FloatType.self) }
    }
    let c = cOut[0].reshaped(.HWC(2, maxLength, 1280))
    var pooled = graph.variable(.GPU(0), .WC(2, 1280), of: FloatType.self)
    if let unconditionalTokenEnd = unconditionalTokenEnd, let tokenEnd = tokenEnd {
      pooled[0..<1, 0..<1280] =
        cOut[1][unconditionalTokenEnd..<(unconditionalTokenEnd + 1), 0..<1280] * textProjection
      pooled[1..<2, 0..<1280] =
        cOut[1][(maxLength + tokenEnd)..<(maxLength + tokenEnd + 1), 0..<1280] * textProjection
    }
    return ([c, pooled], [textModel])
  }

  private func encodeSD3(
    tokens: [DynamicGraph.Tensor<Int32>], positions: [DynamicGraph.Tensor<Int32>],
    mask: [DynamicGraph.Tensor<FloatType>], injectedEmbeddings: [DynamicGraph.Tensor<FloatType>],
    lengthsOfUncond: [Int], lengthsOfCond: [Int], textModels existingTextModels: [Model?]
  )
    -> ([DynamicGraph.Tensor<FloatType>], [Model])
  {
    var causalAttentionMask = Tensor<FloatType>(
      Array(repeating: 0, count: 2 * maxLength * maxLength), .CPU, .NHWC(2, 1, maxLength, maxLength)
    )
    for i in 0..<(maxLength - 1) {
      for j in (i + 1)..<maxLength {
        causalAttentionMask[0, 0, i, j] = -FloatType.greatestFiniteMagnitude
        causalAttentionMask[1, 0, i, j] = -FloatType.greatestFiniteMagnitude
      }
    }
    var j = 0
    var prefixLength = 0
    for i in 0..<maxLength {
      // Mask out anything before this, except padding / ending.
      guard j < lengthsOfUncond.count else { break }
      if i - 1 >= lengthsOfUncond[j] + prefixLength {
        prefixLength += lengthsOfUncond[j]
        j += 1
      }
      if prefixLength > 0 && j < lengthsOfUncond.count {
        for k in 1..<(prefixLength + 1) {
          causalAttentionMask[0, 0, i, k] = -FloatType.greatestFiniteMagnitude
        }
      }
    }
    j = 0
    prefixLength = 0
    for i in 0..<maxLength {
      // Mask out anything before this, except padding / ending.
      guard j < lengthsOfCond.count else { break }
      if i - 1 >= lengthsOfCond[j] + prefixLength {
        prefixLength += lengthsOfCond[j]
        j += 1
      }
      if prefixLength > 0 && j < lengthsOfCond.count {
        for k in 1..<(prefixLength + 1) {
          causalAttentionMask[1, 0, i, k] = -FloatType.greatestFiniteMagnitude
        }
      }
    }
    let graph = tokens[0].graph
    let tokens0TensorGPU = tokens[0].toGPU(0)
    let positionTensorGPU = positions[0].toGPU(0)
    let causalAttentionMaskGPU = graph.variable(causalAttentionMask.toGPU())
    let maskGPU = mask.map { $0.toGPU(0) }
    let injectedEmbeddingsGPU = injectedEmbeddings.map { $0.toGPU(0) }
    let externalData: DynamicGraph.Store.Codec =
      externalOnDemand
      ? .externalOnDemand : .externalData(deviceProperties.isFreadPreferred ? .fread : .mmap)
    var textModel: Model
    textModel =
      CLIPTextModel(
        FloatType.self, injectEmbeddings: injectEmbeddings,
        vocabularySize: 49408, maxLength: 77, maxTokenLength: maxLength, embeddingSize: 768,
        numLayers: 13 - min(max(clipSkip - 1, 1), 12), numHeads: 12, batchSize: 2,
        intermediateSize: 3072, usesFlashAttention: usesFlashAttention, outputPenultimate: true
      ).0
    if let maskGPU = maskGPU.first, let injectedEmbeddingsGPU = injectedEmbeddingsGPU.first {
      textModel.compile(
        inputs: tokens0TensorGPU, positionTensorGPU, causalAttentionMaskGPU, maskGPU,
        injectedEmbeddingsGPU)
    } else {
      textModel.compile(inputs: tokens0TensorGPU, positionTensorGPU, causalAttentionMaskGPU)
    }
    let c0Out: [DynamicGraph.Tensor<FloatType>]
    if filePaths.count > 1 {
      graph.openStore(
        filePaths[1], flags: .readOnly,
        externalStore: TensorData.externalStore(filePath: filePaths[1])
      ) { store in
        if lora.count > 0 {
          LoRALoader.openStore(graph, lora: lora) { loader in
            store.read(
              "text_model", model: textModel, codec: [.jit, .q6p, .q8p, .ezm7, externalData]
            ) { name, dataType, _, shape in
              var name = name
              if name == "__text_model__[t-\(98 - (min(max(clipSkip - 1, 1), 12) - 1) * 8)-0]" {
                name = "__text_model__[t-98-0]"
              } else if name
                == "__text_model__[t-\(98 - (min(max(clipSkip - 1, 1), 12) - 1) * 8)-1]"
              {
                name = "__text_model__[t-98-1]"
              }
              return loader.mergeLoRA(
                graph, name: name, store: store, dataType: dataType, shape: shape,
                of: FloatType.self)
            }
          }
        } else {
          if clipSkip > 1 {
            store.read(
              "text_model", model: textModel, codec: [.jit, .q6p, .q8p, .ezm7, externalData]
            ) { name, _, _, _ in
              // Retrieve the right final layer norm parameters.
              var name = name
              if name == "__text_model__[t-\(98 - (min(max(clipSkip - 1, 1), 12) - 1) * 8)-0]" {
                name = "__text_model__[t-98-0]"
              } else if name
                == "__text_model__[t-\(98 - (min(max(clipSkip - 1, 1), 12) - 1) * 8)-1]"
              {
                name = "__text_model__[t-98-1]"
              }
              return .continue(name)
            }
          } else {
            store.read(
              "text_model", model: textModel, codec: [.jit, .q6p, .q8p, .ezm7, externalData])
          }
        }
      }
      if let maskGPU = maskGPU.first, let injectedEmbeddingsGPU = injectedEmbeddingsGPU.first {
        c0Out = textModel(
          inputs: tokens0TensorGPU, positionTensorGPU, causalAttentionMaskGPU, maskGPU,
          injectedEmbeddingsGPU
        ).map {
          $0.as(
            of: FloatType.self
          )
        }
      } else {
        c0Out = textModel(
          inputs: tokens0TensorGPU, positionTensorGPU, causalAttentionMaskGPU
        ).map {
          $0.as(
            of: FloatType.self
          )
        }
      }
    } else {
      c0Out = [
        graph.variable(.GPU(0), .HWC(2, maxLength, 768)),
        graph.variable(.GPU(0), .WC(2 * maxLength, 768)),
      ]
      for c in c0Out {
        c.full(0)
      }
    }
    let c0 = c0Out[0].reshaped(.HWC(2, maxLength, 768))
    let tokens1TensorGPU = tokens[1].toGPU(0)
    if let existingTextModel = existingTextModels[0] {
      textModel = existingTextModel
    } else {
      textModel =
        OpenCLIPTextModel(
          FloatType.self, injectEmbeddings: injectEmbeddings,
          vocabularySize: 49408, maxLength: 77, maxTokenLength: maxLength, embeddingSize: 1280,
          numLayers: 32 - min(max(clipSkip - 2, 0), 30), numHeads: 20, batchSize: 2,
          intermediateSize: 5120, usesFlashAttention: usesFlashAttention, outputPenultimate: true
        ).0
    }
    if let maskGPU = maskGPU.last, let injectedEmbeddingsGPU = injectedEmbeddingsGPU.last {
      textModel.compile(
        inputs: tokens1TensorGPU, positionTensorGPU, causalAttentionMaskGPU, maskGPU,
        injectedEmbeddingsGPU)
    } else {
      textModel.compile(
        inputs: tokens1TensorGPU, positionTensorGPU, causalAttentionMaskGPU)
    }
    let textProjection = graph.variable(.GPU(0), .WC(1280, 1280), of: FloatType.self)
    graph.openStore(
      filePaths[0], flags: .readOnly,
      externalStore: TensorData.externalStore(filePath: filePaths[0])
    ) { store in
      if lora.count > 0 {
        LoRALoader.openStore(graph, lora: lora) { loader in
          store.read(
            "text_model", model: textModel, codec: [.jit, .q6p, .q8p, .ezm7, externalData]
          ) { name, dataType, _, shape in
            // Retrieve the right final layer norm parameters.
            var name = name
            if name == "__text_model__[t-\(258 - (min(max(clipSkip - 1, 1), 31) - 1) * 8)-0]" {
              name = "__text_model__[t-258-0]"
            } else if name == "__text_model__[t-\(258 - (min(max(clipSkip - 1, 1), 31) - 1) * 8)-1]"
            {
              name = "__text_model__[t-258-1]"
            }
            return loader.mergeLoRA(
              graph, name: name, store: store, dataType: dataType, shape: shape, of: FloatType.self,
              prefix: "__te2")
          }
        }
      } else if clipSkip > 1 {
        store.read("text_model", model: textModel, codec: [.jit, .q6p, .q8p, .ezm7, externalData]) {
          name, _, _, _ in
          // Retrieve the right final layer norm parameters.
          var name = name
          if name == "__text_model__[t-\(258 - (min(max(clipSkip - 1, 1), 31) - 1) * 8)-0]" {
            name = "__text_model__[t-258-0]"
          } else if name == "__text_model__[t-\(258 - (min(max(clipSkip - 1, 1), 31) - 1) * 8)-1]" {
            name = "__text_model__[t-258-1]"
          }
          return .continue(name)
        }
      } else {
        store.read("text_model", model: textModel, codec: [.jit, .q6p, .q8p, .ezm7, externalData])
      }
      store.read(
        "text_projection", variable: textProjection,
        codec: [
          .q6p, .q8p, .ezm7, .externalData(deviceProperties.isFreadPreferred ? .fread : .mmap),
        ])
    }
    let c1Out: [DynamicGraph.Tensor<FloatType>]
    if let maskGPU = maskGPU.last, let injectedEmbeddingsGPU = injectedEmbeddingsGPU.last {
      c1Out = textModel(
        inputs: tokens1TensorGPU, positionTensorGPU, causalAttentionMaskGPU, maskGPU,
        injectedEmbeddingsGPU
      ).map { $0.as(of: FloatType.self) }
    } else {
      c1Out = textModel(
        inputs: tokens1TensorGPU, positionTensorGPU, causalAttentionMaskGPU
      ).map { $0.as(of: FloatType.self) }
    }
    let c1 = c1Out[0].reshaped(.HWC(2, maxLength, 1280))
    var pooled = graph.variable(.GPU(0), .WC(2, 2048), of: FloatType.self)
    var unconditionalTokenEnd: Int? = nil
    var tokenEnd: Int? = nil
    if mask.count > 1 {
      for i in 0..<maxLength {
        if tokens[1][i] == 49407 && mask[1][i, 0] > 0 && unconditionalTokenEnd == nil {
          unconditionalTokenEnd = i
        }
        if tokens[1][i + maxLength] == 49407 && mask[1][i + maxLength, 0] > 0 && tokenEnd == nil {
          tokenEnd = i
        }
      }
    } else {
      for i in 0..<maxLength {
        if tokens[1][i] == 49407 && unconditionalTokenEnd == nil {
          unconditionalTokenEnd = i
        }
        if tokens[1][i + maxLength] == 49407 && tokenEnd == nil {
          tokenEnd = i
        }
      }
    }
    if let unconditionalTokenEnd = unconditionalTokenEnd, let tokenEnd = tokenEnd {
      pooled[0..<1, 0..<768] =
        c0Out[1][unconditionalTokenEnd..<(unconditionalTokenEnd + 1), 0..<768]
      pooled[1..<2, 0..<768] =
        c0Out[1][(maxLength + tokenEnd)..<(maxLength + tokenEnd + 1), 0..<768]
      pooled[0..<1, 768..<2048] =
        c1Out[1][unconditionalTokenEnd..<(unconditionalTokenEnd + 1), 0..<1280] * textProjection
      pooled[1..<2, 768..<2048] =
        c1Out[1][(maxLength + tokenEnd)..<(maxLength + tokenEnd + 1), 0..<1280] * textProjection
    }
    guard filePaths.count >= 3 && tokens.count >= 3 else {
      return ([c0, c1, pooled], [textModel])
    }
    // Now load T5 encoder.
    let tokenLength = tokens[2].shape[0] / 2
    let (_, t5) = T5ForConditionalGeneration(
      b: 2, t: tokenLength, attentionMask: false, of: FloatType.self)
    let relativePositionBuckets = relativePositionBuckets(
      sequenceLength: tokenLength, numBuckets: 32, maxDistance: 128)
    let tokens2TensorGPU = tokens[2].toGPU(0)
    let relativePositionBucketsGPU = graph.variable(relativePositionBuckets.toGPU(0))
    t5.compile(inputs: tokens2TensorGPU, relativePositionBucketsGPU)
    if !weightsCache.detach(filePaths[2], to: t5.parameters) {
      // If we have more than 24GiB RAM, and not forced to be on demand. We load the whole thing (better for weights cache).
      let externalData: DynamicGraph.Store.Codec =
        externalOnDemand || deviceProperties.memoryCapacity != .high
        ? .externalOnDemand : .externalData(deviceProperties.isFreadPreferred ? .fread : .mmap)
      // Move T5 to on-demand.
      TensorData.makeExternalData(for: filePaths[2], graph: graph)
      graph.openStore(
        filePaths[2], flags: .readOnly,
        externalStore: TensorData.externalStore(filePath: filePaths[2])
      ) {
        $0.read("text_model", model: t5, codec: [.q8p, .q6p, .q4p, .ezm7, .jit, externalData])
      }
    }
    let c2 = t5(inputs: tokens2TensorGPU, relativePositionBucketsGPU)[0].as(
      of: FloatType.self
    ).reshaped(.HWC(2, tokenLength, 4096))
    weightsCache.attach(filePaths[2], from: t5.parameters)
    return ([c0, c1, c2, pooled], [textModel])
  }

  private func encodePixArt(
    tokens: [DynamicGraph.Tensor<Int32>], positions: [DynamicGraph.Tensor<Int32>],
    mask: [DynamicGraph.Tensor<FloatType>], injectedEmbeddings: [DynamicGraph.Tensor<FloatType>],
    lengthsOfUncond: [Int], lengthsOfCond: [Int], textModels existingTextModels: [Model?]
  )
    -> ([DynamicGraph.Tensor<FloatType>], [Model])
  {
    let graph = tokens[0].graph
    let tokenLength = tokens[0].shape[0] / 2
    let lora = Array(
      (OrderedDictionary<String, LoRAConfiguration>(
        lora.filter({ $0.version == version }).map {
          ($0.file, $0)
        }
      ) {
        LoRAConfiguration(
          file: $0.file, weight: $0.weight + $1.weight, version: $0.version, isLoHa: $0.isLoHa,
          modifier: $0.modifier, mode: $0.mode)
      })
      .values
    ).filter { $0.weight != 0 }
    let (rankOfLoRA, filesRequireMerge) = LoRALoader.rank(
      graph, of: lora.map { $0.file }, prefix: "__text_model__")
    let configuration = LoRANetworkConfiguration(rank: rankOfLoRA, scale: 1, highPrecision: false)
    let textModel: Model
    if !lora.isEmpty && rankOfLoRA > 0 {
      (_, textModel) = LoRAT5ForConditionalGeneration(
        b: 2, t: tokenLength, LoRAConfiguration: configuration, of: FloatType.self)
    } else {
      (_, textModel) = T5ForConditionalGeneration(
        b: 2, t: tokenLength, attentionMask: false, of: FloatType.self)
    }
    let relativePositionBuckets = relativePositionBuckets(
      sequenceLength: tokenLength, numBuckets: 32, maxDistance: 128)
    let tokensTensorGPU = tokens[0].toGPU(0)
    let relativePositionBucketsGPU = graph.variable(relativePositionBuckets.toGPU(0))
    textModel.compile(inputs: tokensTensorGPU, relativePositionBucketsGPU)
    // If we have more than 24GiB RAM, and not forced to be on demand. We load the whole thing (better for weights cache).
    let externalData: DynamicGraph.Store.Codec =
      externalOnDemand || deviceProperties.memoryCapacity != .high
      ? .externalOnDemand : .externalData(deviceProperties.isFreadPreferred ? .fread : .mmap)
    // Move T5 to on-demand.
    TensorData.makeExternalData(for: filePaths[0], graph: graph)
    graph.openStore(
      filePaths[0], flags: .readOnly,
      externalStore: TensorData.externalStore(filePath: filePaths[0])
    ) { store in
      if !lora.isEmpty && rankOfLoRA > 0 {
        let mapping = [Int: Int](
          uniqueKeysWithValues: (0..<24).map {
            return ($0, $0)
          })
        LoRALoader.openStore(graph, lora: lora) { loader in
          store.read(
            "text_model", model: textModel, codec: [.jit, .q6p, .q8p, .ezm7, externalData]
          ) {
            name, dataType, format, shape in
            return loader.concatenateLoRA(
              graph, LoRAMapping: mapping, filesRequireMerge: filesRequireMerge, name: name,
              store: store, dataType: dataType, format: format, shape: shape, of: FloatType.self)
          }
        }
      } else {
        store.read(
          "text_model", model: textModel, codec: [.q8p, .q6p, .q4p, .ezm7, .jit, externalData])
      }
    }
    let c = textModel(inputs: tokensTensorGPU, relativePositionBucketsGPU)[0].as(
      of: FloatType.self
    ).reshaped(.HWC(2, tokenLength, 4096))
    return ([c], [textModel])
  }

  private func encodeAuraFlow(
    tokens: [DynamicGraph.Tensor<Int32>], positions: [DynamicGraph.Tensor<Int32>],
    mask: [DynamicGraph.Tensor<FloatType>], injectedEmbeddings: [DynamicGraph.Tensor<FloatType>],
    lengthsOfUncond: [Int], lengthsOfCond: [Int], textModels existingTextModels: [Model?]
  )
    -> ([DynamicGraph.Tensor<FloatType>], [Model])
  {
    let graph = tokens[0].graph
    let tokenLength = tokens[0].shape[0] / 2
    let (_, textModel) = UMT5ForConditionalGeneration(
      b: 2, t: tokenLength, vocabularySize: 32_128, channels: 2_048, intermediateSize: 5_120,
      upcast: false, of: FloatType.self)
    let relativePositionBuckets = relativePositionBuckets(
      sequenceLength: tokenLength, numBuckets: 32, maxDistance: 128)
    var attentionMask = Tensor<FloatType>(.CPU, .NHWC(2, 1, 1, tokenLength))
    let lengthOfCond = lengthsOfCond.reduce(0, +)
    let lengthOfUncond = lengthsOfUncond.reduce(0, +)
    for i in 0..<tokenLength {
      attentionMask[0, 0, 0, i] = i < lengthOfUncond + 1 ? 0 : -FloatType.greatestFiniteMagnitude
      attentionMask[1, 0, 0, i] = i < lengthOfCond + 1 ? 0 : -FloatType.greatestFiniteMagnitude
    }
    let tokensTensorGPU = tokens[0].toGPU(0)
    let relativePositionBucketsGPU = graph.variable(relativePositionBuckets.toGPU(0))
    let attentionMaskGPU = graph.variable(attentionMask.toGPU(0))
    textModel.compile(inputs: tokensTensorGPU, attentionMaskGPU, relativePositionBucketsGPU)
    if !weightsCache.detach(filePaths[0], to: textModel.parameters) {
      // If we have more than 24GiB RAM, and not forced to be on demand. We load the whole thing (better for weights cache).
      let externalData: DynamicGraph.Store.Codec =
        externalOnDemand || deviceProperties.memoryCapacity != .high
        ? .externalOnDemand : .externalData(deviceProperties.isFreadPreferred ? .fread : .mmap)
      // Move Pile T5 XL to on-demand.
      TensorData.makeExternalData(for: filePaths[0], graph: graph)
      graph.openStore(
        filePaths[0], flags: .readOnly,
        externalStore: TensorData.externalStore(filePath: filePaths[0])
      ) { store in
        store.read(
          "text_model", model: textModel, codec: [.q8p, .q6p, .q4p, .ezm7, .jit, externalData])
      }
    }
    var c = textModel(inputs: tokensTensorGPU, attentionMaskGPU, relativePositionBucketsGPU)[0].as(
      of: FloatType.self
    ).reshaped(.HWC(2, tokenLength, 2048))
    weightsCache.attach(filePaths[0], from: textModel.parameters)
    var encoderMask = Tensor<FloatType>(.CPU, .HWC(2, tokenLength, 1))
    for i in 0..<tokenLength {
      encoderMask[0, i, 0] = i < lengthOfUncond + 1 ? 1 : 0
      encoderMask[1, i, 0] = i < lengthOfCond + 1 ? 1 : 0
    }
    c = c .* graph.variable(encoderMask.toGPU(0))
    return ([c], [textModel])
  }

  private func encodeChatGLM3(
    tokens: [DynamicGraph.Tensor<Int32>], positions: [DynamicGraph.Tensor<Int32>],
    mask: [DynamicGraph.Tensor<FloatType>], injectedEmbeddings: [DynamicGraph.Tensor<FloatType>],
    lengthsOfUncond: [Int], lengthsOfCond: [Int], textModels existingTextModels: [Model?]
  )
    -> ([DynamicGraph.Tensor<FloatType>], [Model])
  {
    let graph = tokens[0].graph
    let lengthOfCond = lengthsOfCond.reduce(0, +)
    let lengthOfUncond = lengthsOfUncond.reduce(0, +)
    assert(tokens[0].shape[0] == 2 * max(lengthOfCond, lengthOfUncond))
    let tokenLength = max(max(lengthOfCond + 2, lengthOfUncond + 2), 256)
    var rightAlignedTokens = Tensor<Int32>(.CPU, format: .NHWC, shape: [tokenLength * 2])
    for i in 0..<(tokenLength - lengthOfUncond - 2) {
      rightAlignedTokens[i] = 0
    }
    rightAlignedTokens[tokenLength - lengthOfUncond - 2] = 64_790
    rightAlignedTokens[tokenLength - lengthOfUncond - 1] = 64_792
    for i in (tokenLength - lengthOfUncond)..<tokenLength {
      rightAlignedTokens[i] = tokens[0][i - (tokenLength - lengthOfUncond)]
    }
    for i in 0..<(tokenLength - lengthOfCond - 2) {
      rightAlignedTokens[tokenLength + i] = 0
    }
    rightAlignedTokens[tokenLength + tokenLength - lengthOfCond - 2] = 64_790
    rightAlignedTokens[tokenLength + tokenLength - lengthOfCond - 1] = 64_792
    for i in (tokenLength - lengthOfCond)..<tokenLength {
      rightAlignedTokens[tokenLength + i] =
        tokens[0][max(lengthOfCond, lengthOfUncond) + i - (tokenLength - lengthOfCond)]
    }
    let causalAttentionMask = graph.variable(
      .CPU, .NHWC(2, 1, tokenLength, tokenLength), of: FloatType.self)
    causalAttentionMask.full(0)
    for i in (tokenLength - lengthOfUncond - 2)..<(tokenLength - 1) {
      for j in (i + 1)..<tokenLength {
        causalAttentionMask[0, 0, i, j] = -FloatType.greatestFiniteMagnitude
      }
    }
    for i in (tokenLength - lengthOfUncond - 2)..<tokenLength {
      for j in 0..<(tokenLength - lengthOfUncond - 2) {
        causalAttentionMask[0, 0, i, j] = -FloatType.greatestFiniteMagnitude
      }
    }
    for i in (tokenLength - lengthOfCond - 2)..<(tokenLength - 1) {
      for j in (i + 1)..<tokenLength {
        causalAttentionMask[1, 0, i, j] = -FloatType.greatestFiniteMagnitude
      }
    }
    for i in (tokenLength - lengthOfCond - 2)..<tokenLength {
      for j in 0..<(tokenLength - lengthOfCond - 2) {
        causalAttentionMask[1, 0, i, j] = -FloatType.greatestFiniteMagnitude
      }
    }
    let rotaryEmbedding = GLMRotaryEmbedding(sequenceLength: tokenLength, of: FloatType.self)
    var rightAlignedRotaryEmbedding = Tensor<FloatType>(.CPU, .NHWC(2, tokenLength, 1, 128))
    for i in 0..<(tokenLength - lengthOfUncond - 2) {
      rightAlignedRotaryEmbedding[0..<1, i..<(i + 1), 0..<1, 0..<128] =
        rotaryEmbedding[0..<1, 0..<1, 0..<1, 0..<128]
    }
    rightAlignedRotaryEmbedding[
      0..<1, (tokenLength - lengthOfUncond - 2)..<tokenLength, 0..<1, 0..<128] =
      rotaryEmbedding[0..<1, 0..<(lengthOfUncond + 2), 0..<1, 0..<128]
    for i in 0..<(tokenLength - lengthOfCond - 2) {
      rightAlignedRotaryEmbedding[1..<2, i..<(i + 1), 0..<1, 0..<128] =
        rotaryEmbedding[0..<1, 0..<1, 0..<1, 0..<128]
    }
    rightAlignedRotaryEmbedding[
      1..<2, (tokenLength - lengthOfCond - 2)..<tokenLength, 0..<1, 0..<128] =
      rotaryEmbedding[0..<1, 0..<(lengthOfCond + 2), 0..<1, 0..<128]
    // ChatGLM3 alignment is a bit different, realign the token tensor.
    let (textModel, _) = GLMTransformer(
      FloatType.self, vocabularySize: 65_024, width: 4_096, tokenLength: tokenLength,
      layers: 29 - min(max(clipSkip - 1, 1), 27), MLP: 13_696, heads: 32, batchSize: 2,
      outputPenultimate: true, applyFinalNorm: false, usesFlashAttention: usesFlashAttention)
    let rightAlignedTokensTensorGPU = graph.variable(rightAlignedTokens.toGPU(0))
    let rightAlignedRotaryEmbeddingGPU = graph.variable(rightAlignedRotaryEmbedding.toGPU(0))
    let causalAttentionMaskGPU = causalAttentionMask.toGPU(0)
    textModel.compile(
      inputs: rightAlignedTokensTensorGPU, rightAlignedRotaryEmbeddingGPU, causalAttentionMaskGPU)
    if !weightsCache.detach(filePaths[0], to: textModel.parameters) {
      // If we have more than 24GiB RAM, and not forced to be on demand. We load the whole thing (better for weights cache).
      let externalData: DynamicGraph.Store.Codec =
        externalOnDemand || deviceProperties.memoryCapacity != .high
        ? .externalOnDemand : .externalData(deviceProperties.isFreadPreferred ? .fread : .mmap)
      // Move ChatGLM3 to on-demand.
      TensorData.makeExternalData(for: filePaths[0], graph: graph)
      graph.openStore(
        filePaths[0], flags: .readOnly,
        externalStore: TensorData.externalStore(filePath: filePaths[0])
      ) { store in
        store.read(
          "text_model", model: textModel, codec: [.q8p, .q6p, .q4p, .ezm7, .jit, externalData])
      }
    }
    let c = textModel(
      inputs: rightAlignedTokensTensorGPU, rightAlignedRotaryEmbeddingGPU, causalAttentionMaskGPU
    ).map {
      $0.as(
        of: FloatType.self
      )
    }
    weightsCache.attach(filePaths[0], from: textModel.parameters)
    var pooled = graph.variable(.GPU(0), .WC(2, 4096), of: FloatType.self)
    pooled[0..<1, 0..<4096] = c[1][(tokenLength - 1)..<tokenLength, 0..<4096]
    pooled[1..<2, 0..<4096] = c[1][(tokenLength * 2 - 1)..<(tokenLength * 2), 0..<4096]
    return ([c[0].reshaped(.HWC(2, tokenLength, 4096)), pooled], [textModel])
  }

  private func encodeFlux1(
    tokens: [DynamicGraph.Tensor<Int32>], positions: [DynamicGraph.Tensor<Int32>],
    mask: [DynamicGraph.Tensor<FloatType>], injectedEmbeddings: [DynamicGraph.Tensor<FloatType>],
    lengthsOfUncond: [Int], lengthsOfCond: [Int], textModels existingTextModels: [Model?]
  )
    -> ([DynamicGraph.Tensor<FloatType>], [Model])
  {
    let maxLength = tokens[1].shape[0] / 2
    var causalAttentionMask = Tensor<FloatType>(
      Array(repeating: 0, count: 2 * maxLength * maxLength), .CPU, .NHWC(2, 1, maxLength, maxLength)
    )
    for i in 0..<(maxLength - 1) {
      for j in (i + 1)..<maxLength {
        causalAttentionMask[0, 0, i, j] = -FloatType.greatestFiniteMagnitude
        causalAttentionMask[1, 0, i, j] = -FloatType.greatestFiniteMagnitude
      }
    }
    var j = 0
    var prefixLength = 0
    for i in 0..<maxLength {
      // Mask out anything before this, except padding / ending.
      guard j < lengthsOfUncond.count else { break }
      if i - 1 >= lengthsOfUncond[j] + prefixLength {
        prefixLength += lengthsOfUncond[j]
        j += 1
      }
      if prefixLength > 0 && j < lengthsOfUncond.count {
        for k in 1..<(prefixLength + 1) {
          causalAttentionMask[0, 0, i, k] = -FloatType.greatestFiniteMagnitude
        }
      }
    }
    j = 0
    prefixLength = 0
    for i in 0..<maxLength {
      // Mask out anything before this, except padding / ending.
      guard j < lengthsOfCond.count else { break }
      if i - 1 >= lengthsOfCond[j] + prefixLength {
        prefixLength += lengthsOfCond[j]
        j += 1
      }
      if prefixLength > 0 && j < lengthsOfCond.count {
        for k in 1..<(prefixLength + 1) {
          causalAttentionMask[1, 0, i, k] = -FloatType.greatestFiniteMagnitude
        }
      }
    }
    let graph = tokens[1].graph
    let tokens0TensorGPU = tokens[1].toGPU(0)
    let positionTensorGPU = positions[0].toGPU(0)
    let causalAttentionMaskGPU = graph.variable(causalAttentionMask.toGPU())
    let maskGPU = mask.map { $0.toGPU(0) }
    let injectedEmbeddingsGPU = injectedEmbeddings.map { $0.toGPU(0) }
    let externalData: DynamicGraph.Store.Codec =
      externalOnDemand
      ? .externalOnDemand : .externalData(deviceProperties.isFreadPreferred ? .fread : .mmap)
    var textModel: Model
    textModel =
      CLIPTextModel(
        FloatType.self, injectEmbeddings: injectEmbeddings,
        vocabularySize: 49408, maxLength: 77, maxTokenLength: maxLength, embeddingSize: 768,
        numLayers: 13 - min(max(clipSkip - 1, 1), 12), numHeads: 12, batchSize: 2,
        intermediateSize: 3072, usesFlashAttention: usesFlashAttention, outputPenultimate: true
      ).0
    if let maskGPU = maskGPU.first, let injectedEmbeddingsGPU = injectedEmbeddingsGPU.first {
      textModel.compile(
        inputs: tokens0TensorGPU, positionTensorGPU, causalAttentionMaskGPU, maskGPU,
        injectedEmbeddingsGPU)
    } else {
      textModel.compile(inputs: tokens0TensorGPU, positionTensorGPU, causalAttentionMaskGPU)
    }
    let c0Out: [DynamicGraph.Tensor<FloatType>]
    if filePaths.count > 1 {
      graph.openStore(
        filePaths[1], flags: .readOnly,
        externalStore: TensorData.externalStore(filePath: filePaths[1])
      ) { store in
        if lora.count > 0 {
          LoRALoader.openStore(graph, lora: lora) { loader in
            store.read(
              "text_model", model: textModel, codec: [.jit, .q6p, .q8p, .ezm7, externalData]
            ) { name, dataType, _, shape in
              var name = name
              if name == "__text_model__[t-\(98 - (min(max(clipSkip - 1, 1), 12) - 1) * 8)-0]" {
                name = "__text_model__[t-98-0]"
              } else if name
                == "__text_model__[t-\(98 - (min(max(clipSkip - 1, 1), 12) - 1) * 8)-1]"
              {
                name = "__text_model__[t-98-1]"
              }
              return loader.mergeLoRA(
                graph, name: name, store: store, dataType: dataType, shape: shape,
                of: FloatType.self)
            }
          }
        } else {
          if clipSkip > 1 {
            store.read(
              "text_model", model: textModel, codec: [.jit, .q6p, .q8p, .ezm7, externalData]
            ) { name, _, _, _ in
              // Retrieve the right final layer norm parameters.
              var name = name
              if name == "__text_model__[t-\(98 - (min(max(clipSkip - 1, 1), 12) - 1) * 8)-0]" {
                name = "__text_model__[t-98-0]"
              } else if name
                == "__text_model__[t-\(98 - (min(max(clipSkip - 1, 1), 12) - 1) * 8)-1]"
              {
                name = "__text_model__[t-98-1]"
              }
              return .continue(name)
            }
          } else {
            store.read(
              "text_model", model: textModel, codec: [.jit, .q6p, .q8p, .ezm7, externalData])
          }
        }
      }
      if let maskGPU = maskGPU.first, let injectedEmbeddingsGPU = injectedEmbeddingsGPU.first {
        c0Out = textModel(
          inputs: tokens0TensorGPU, positionTensorGPU, causalAttentionMaskGPU, maskGPU,
          injectedEmbeddingsGPU
        ).map {
          $0.as(
            of: FloatType.self
          )
        }
      } else {
        c0Out = textModel(
          inputs: tokens0TensorGPU, positionTensorGPU, causalAttentionMaskGPU
        ).map {
          $0.as(
            of: FloatType.self
          )
        }
      }
    } else {
      c0Out = [
        graph.variable(.GPU(0), .HWC(2, maxLength, 768)),
        graph.variable(.GPU(0), .WC(2 * maxLength, 768)),
      ]
      for c in c0Out {
        c.full(0)
      }
    }
    let batchSize = isCfgEnabled ? 2 : 1
    var pooled = graph.variable(.GPU(0), .WC(batchSize, 768), of: FloatType.self)
    var unconditionalTokenEnd: Int? = nil
    var tokenEnd: Int? = nil
    if mask.count > 1 {
      for i in 0..<maxLength {
        if tokens[1][i] == 49407 && mask[1][i, 0] > 0 && unconditionalTokenEnd == nil {
          unconditionalTokenEnd = i
        }
        if tokens[1][i + maxLength] == 49407 && mask[1][i + maxLength, 0] > 0 && tokenEnd == nil {
          tokenEnd = i
        }
      }
    } else {
      for i in 0..<maxLength {
        if tokens[1][i] == 49407 && unconditionalTokenEnd == nil {
          unconditionalTokenEnd = i
        }
        if tokens[1][i + maxLength] == 49407 && tokenEnd == nil {
          tokenEnd = i
        }
      }
    }
    if let unconditionalTokenEnd = unconditionalTokenEnd, let tokenEnd = tokenEnd {
      if isCfgEnabled {
        pooled[0..<1, 0..<768] =
          c0Out[1][unconditionalTokenEnd..<(unconditionalTokenEnd + 1), 0..<768]
        pooled[1..<2, 0..<768] =
          c0Out[1][(maxLength + tokenEnd)..<(maxLength + tokenEnd + 1), 0..<768]
      } else {
        pooled[0..<1, 0..<768] =
          c0Out[1][(maxLength + tokenEnd)..<(maxLength + tokenEnd + 1), 0..<768]
      }
    }
    // Now load T5 encoder.
    let tokenLength = tokens[0].shape[0] / 2
    let (_, t5) = T5ForConditionalGeneration(
      b: batchSize, t: tokenLength, attentionMask: false, of: FloatType.self)
    let relativePositionBuckets = relativePositionBuckets(
      sequenceLength: tokenLength, numBuckets: 32, maxDistance: 128)
    let tokens2TensorGPU: DynamicGraph.Tensor<Int32>
    if isCfgEnabled {
      tokens2TensorGPU = tokens[0].toGPU(0)
    } else {
      tokens2TensorGPU = tokens[0][tokenLength..<(tokenLength * 2)].toGPU(0)
    }
    let relativePositionBucketsGPU = graph.variable(relativePositionBuckets.toGPU(0))
    t5.compile(inputs: tokens2TensorGPU, relativePositionBucketsGPU)
    if !weightsCache.detach(filePaths[0], to: t5.parameters) {
      // If we have more than 24GiB RAM, and not forced to be on demand. We load the whole thing (better for weights cache).
      let externalData: DynamicGraph.Store.Codec =
        externalOnDemand || deviceProperties.memoryCapacity != .high
        ? .externalOnDemand : .externalData(deviceProperties.isFreadPreferred ? .fread : .mmap)
      // Move T5 to on-demand.
      TensorData.makeExternalData(for: filePaths[0], graph: graph)
      graph.openStore(
        filePaths[0], flags: .readOnly,
        externalStore: TensorData.externalStore(filePath: filePaths[0])
      ) {
        $0.read("text_model", model: t5, codec: [.q8p, .q6p, .q4p, .ezm7, .jit, externalData])
      }
    }
    let c2 = t5(inputs: tokens2TensorGPU, relativePositionBucketsGPU)[0].as(
      of: FloatType.self
    ).reshaped(.HWC(batchSize, tokenLength, 4096))
    weightsCache.attach(filePaths[0], from: t5.parameters)
    return ([c2, pooled], [textModel])
  }

  private func encodeHunyuan(
    tokenLengthUncond: Int, tokenLengthCond: Int,
    tokens: [DynamicGraph.Tensor<Int32>], positions: [DynamicGraph.Tensor<Int32>],
    mask: [DynamicGraph.Tensor<FloatType>], injectedEmbeddings: [DynamicGraph.Tensor<FloatType>],
    lengthsOfUncond: [Int], lengthsOfCond: [Int],
    injectedTextEmbeddings: [(
      model: ControlModel<FloatType>, hints: [([DynamicGraph.Tensor<FloatType>], Float)]
    )], textModels existingTextModels: [Model?]
  )
    -> ([DynamicGraph.Tensor<FloatType>], [Model])
  {
    let maxLength = tokens[1].shape[0] / 2
    var causalAttentionMask = Tensor<FloatType>(
      Array(repeating: 0, count: 2 * maxLength * maxLength), .CPU, .NHWC(2, 1, maxLength, maxLength)
    )
    for i in 0..<(maxLength - 1) {
      for j in (i + 1)..<maxLength {
        causalAttentionMask[0, 0, i, j] = -FloatType.greatestFiniteMagnitude
        causalAttentionMask[1, 0, i, j] = -FloatType.greatestFiniteMagnitude
      }
    }
    var j = 0
    var prefixLength = 0
    for i in 0..<maxLength {
      // Mask out anything before this, except padding / ending.
      guard j < lengthsOfUncond.count else { break }
      if i - 1 >= lengthsOfUncond[j] + prefixLength {
        prefixLength += lengthsOfUncond[j]
        j += 1
      }
      if prefixLength > 0 && j < lengthsOfUncond.count {
        for k in 1..<(prefixLength + 1) {
          causalAttentionMask[0, 0, i, k] = -FloatType.greatestFiniteMagnitude
        }
      }
    }
    j = 0
    prefixLength = 0
    for i in 0..<maxLength {
      // Mask out anything before this, except padding / ending.
      guard j < lengthsOfCond.count else { break }
      if i - 1 >= lengthsOfCond[j] + prefixLength {
        prefixLength += lengthsOfCond[j]
        j += 1
      }
      if prefixLength > 0 && j < lengthsOfCond.count {
        for k in 1..<(prefixLength + 1) {
          causalAttentionMask[1, 0, i, k] = -FloatType.greatestFiniteMagnitude
        }
      }
    }
    let graph = tokens[1].graph
    let tokens0TensorGPU = tokens[1].toGPU(0)
    let positionTensorGPU = positions[0].toGPU(0)
    let causalAttentionMaskGPU = graph.variable(causalAttentionMask.toGPU())
    let maskGPU = mask.map { $0.toGPU(0) }
    let injectedEmbeddingsGPU = injectedEmbeddings.map { $0.toGPU(0) }
    let externalData: DynamicGraph.Store.Codec =
      externalOnDemand
      ? .externalOnDemand : .externalData(deviceProperties.isFreadPreferred ? .fread : .mmap)
    var textModel: Model
    textModel =
      CLIPTextModel(
        FloatType.self, injectEmbeddings: injectEmbeddings,
        vocabularySize: 49408, maxLength: 77, maxTokenLength: maxLength, embeddingSize: 768,
        numLayers: 13 - min(max(clipSkip - 1, 1), 12), numHeads: 12, batchSize: 2,
        intermediateSize: 3072, usesFlashAttention: usesFlashAttention, outputPenultimate: true
      ).0
    if let maskGPU = maskGPU.first, let injectedEmbeddingsGPU = injectedEmbeddingsGPU.first {
      textModel.compile(
        inputs: tokens0TensorGPU, positionTensorGPU, causalAttentionMaskGPU, maskGPU,
        injectedEmbeddingsGPU)
    } else {
      textModel.compile(inputs: tokens0TensorGPU, positionTensorGPU, causalAttentionMaskGPU)
    }
    let c0Out: [DynamicGraph.Tensor<FloatType>]
    if filePaths.count > 1 {
      graph.openStore(
        filePaths[1], flags: .readOnly,
        externalStore: TensorData.externalStore(filePath: filePaths[1])
      ) { store in
        if lora.count > 0 {
          LoRALoader.openStore(graph, lora: lora) { loader in
            store.read(
              "text_model", model: textModel, codec: [.jit, .q6p, .q8p, .ezm7, externalData]
            ) { name, dataType, _, shape in
              var name = name
              if name == "__text_model__[t-\(98 - (min(max(clipSkip - 1, 1), 12) - 1) * 8)-0]" {
                name = "__text_model__[t-98-0]"
              } else if name
                == "__text_model__[t-\(98 - (min(max(clipSkip - 1, 1), 12) - 1) * 8)-1]"
              {
                name = "__text_model__[t-98-1]"
              }
              return loader.mergeLoRA(
                graph, name: name, store: store, dataType: dataType, shape: shape,
                of: FloatType.self)
            }
          }
        } else {
          if clipSkip > 1 {
            store.read(
              "text_model", model: textModel, codec: [.jit, .q6p, .q8p, .ezm7, externalData]
            ) { name, _, _, _ in
              // Retrieve the right final layer norm parameters.
              var name = name
              if name == "__text_model__[t-\(98 - (min(max(clipSkip - 1, 1), 12) - 1) * 8)-0]" {
                name = "__text_model__[t-98-0]"
              } else if name
                == "__text_model__[t-\(98 - (min(max(clipSkip - 1, 1), 12) - 1) * 8)-1]"
              {
                name = "__text_model__[t-98-1]"
              }
              return .continue(name)
            }
          } else {
            store.read(
              "text_model", model: textModel, codec: [.jit, .q6p, .q8p, .ezm7, externalData])
          }
        }
      }
      if let maskGPU = maskGPU.first, let injectedEmbeddingsGPU = injectedEmbeddingsGPU.first {
        c0Out = textModel(
          inputs: tokens0TensorGPU, positionTensorGPU, causalAttentionMaskGPU, maskGPU,
          injectedEmbeddingsGPU
        ).map {
          $0.as(
            of: FloatType.self
          )
        }
      } else {
        c0Out = textModel(
          inputs: tokens0TensorGPU, positionTensorGPU, causalAttentionMaskGPU
        ).map {
          $0.as(
            of: FloatType.self
          )
        }
      }
    } else {
      c0Out = [
        graph.variable(.GPU(0), .HWC(2, maxLength, 768)),
        graph.variable(.GPU(0), .WC(2 * maxLength, 768)),
      ]
      for c in c0Out {
        c.full(0)
      }
    }
    let batchSize = isCfgEnabled ? 2 : 1
    var pooled = graph.variable(.GPU(0), .WC(batchSize, 768), of: FloatType.self)
    var unconditionalTokenEnd: Int? = nil
    var tokenEnd: Int? = nil
    if mask.count > 1 {
      for i in 0..<maxLength {
        if tokens[1][i] == 49407 && mask[1][i, 0] > 0 && unconditionalTokenEnd == nil {
          unconditionalTokenEnd = i
        }
        if tokens[1][i + maxLength] == 49407 && mask[1][i + maxLength, 0] > 0 && tokenEnd == nil {
          tokenEnd = i
        }
      }
    } else {
      for i in 0..<maxLength {
        if tokens[1][i] == 49407 && unconditionalTokenEnd == nil {
          unconditionalTokenEnd = i
        }
        if tokens[1][i + maxLength] == 49407 && tokenEnd == nil {
          tokenEnd = i
        }
      }
    }
    if let unconditionalTokenEnd = unconditionalTokenEnd, let tokenEnd = tokenEnd {
      if isCfgEnabled {
        pooled[0..<1, 0..<768] =
          c0Out[1][unconditionalTokenEnd..<(unconditionalTokenEnd + 1), 0..<768]
        pooled[1..<2, 0..<768] =
          c0Out[1][(maxLength + tokenEnd)..<(maxLength + tokenEnd + 1), 0..<768]
      } else {
        pooled[0..<1, 0..<768] =
          c0Out[1][(maxLength + tokenEnd)..<(maxLength + tokenEnd + 1), 0..<768]
      }
    }
    // Now load Llama3 decoder.
    let tokenLength = tokens[0].shape[0] / 2
    let injectedTextEmbeddings = injectedTextEmbeddings.flatMap {
      $0.hints.flatMap {
        $0.0
      }
    }
    let injectedTextEmbedding: DynamicGraph.Tensor<FloatType>? =
      (injectedTextEmbeddings.count > 1
      ? Concat(axis: 1)(inputs: injectedEmbeddings[0], Array(injectedTextEmbeddings[1...]))[0].as(
        of: FloatType.self) : injectedTextEmbeddings.first).map {
        let shape = $0.shape
        return $0[1..<2, 0..<shape[1], 0..<shape[2]].copied().reshaped(.WC(shape[1], shape[2]))
      }
    let additionalTokenLength = max(isCfgEnabled ? tokenLength : 0, tokenLengthCond + 95)
    let tokenLengthCondWithoutAddition =
      tokenLengthCond + 95 - (injectedTextEmbedding?.shape[0] ?? 0)
    let llama3 = Llama3(
      FloatType.self, vocabularySize: 128_320, width: 4_096,
      tokenLength: (tokenLengthUncond + 95, tokenLengthCondWithoutAddition, additionalTokenLength),
      layers: 32, MLP: 14336, heads: 32, outputHiddenStates: [30], batchSize: batchSize,
      usesFlashAttention: true /* For now, to keep consistency on CUDA */)
    let tokens2TensorGPU: DynamicGraph.Tensor<Int32>
    if isCfgEnabled {
      tokens2TensorGPU = tokens[0].toGPU(0)
    } else {
      tokens2TensorGPU = tokens[0][
        tokenLength..<(tokenLength + additionalTokenLength - (injectedTextEmbedding?.shape[0] ?? 0))
      ].toGPU(0)
    }
    var causalAttentionMaskLlama3 = Tensor<FloatType>(
      Array(repeating: 0, count: additionalTokenLength * additionalTokenLength), .CPU,
      .NHWC(1, 1, additionalTokenLength, additionalTokenLength)
    )
    for i in 0..<(additionalTokenLength - 1) {
      for j in (i + 1)..<additionalTokenLength {
        causalAttentionMaskLlama3[0, 0, i, j] = -FloatType.greatestFiniteMagnitude
      }
    }
    let rotaryTensorGPU = graph.variable(
      LlamaRotaryEmbedding(sequenceLength: additionalTokenLength, of: FloatType.self).toGPU(0))
    let causalAttentionMaskLlama3GPU = graph.variable(causalAttentionMaskLlama3.toGPU(0))
    llama3.compile(
      inputs: [tokens2TensorGPU, rotaryTensorGPU, causalAttentionMaskLlama3GPU]
        + (injectedTextEmbedding.flatMap { [$0] } ?? []))
    if !weightsCache.detach(filePaths[0], to: llama3.parameters) {
      // If we have more than 24GiB RAM, and not forced to be on demand. We load the whole thing (better for weights cache).
      let externalData: DynamicGraph.Store.Codec =
        externalOnDemand || deviceProperties.memoryCapacity != .high
        ? .externalOnDemand : .externalData(deviceProperties.isFreadPreferred ? .fread : .mmap)
      // Move Llama3 8B to on-demand.
      TensorData.makeExternalData(for: filePaths[0], graph: graph)
      graph.openStore(
        filePaths[0], flags: .readOnly,
        externalStore: TensorData.externalStore(filePath: filePaths[0])
      ) {
        $0.read("llava", model: llama3, codec: [.q8p, .q6p, .q4p, .ezm7, .jit, externalData])
      }
    }
    let c2 = llama3(
      inputs: tokens2TensorGPU,
      [rotaryTensorGPU, causalAttentionMaskLlama3GPU]
        + (injectedTextEmbedding.flatMap { [$0] } ?? []))[0].as(
        of: FloatType.self
      ).reshaped(.HWC(batchSize, additionalTokenLength, 4096))[
        0..<batchSize, 95..<additionalTokenLength, 0..<4096
      ].copied()
    weightsCache.attach(filePaths[0], from: llama3.parameters)
    return ([c2, pooled], [textModel])
  }

  private func encodeWan(
    images: [DynamicGraph.Tensor<FloatType>],
    tokens: [DynamicGraph.Tensor<Int32>], positions: [DynamicGraph.Tensor<Int32>],
    mask: [DynamicGraph.Tensor<FloatType>], injectedEmbeddings: [DynamicGraph.Tensor<FloatType>],
    tokenLengthUncond: Int, tokenLengthCond: Int, textModels existingTextModels: [Model?]
  )
    -> ([DynamicGraph.Tensor<FloatType>], [Model])
  {
    let graph = tokens[0].graph
    let tokenLength = tokens[0].shape[0] / 2
    let (_, textModel) = UMT5ForConditionalGeneration(
      b: 2, t: tokenLength, vocabularySize: 256_384, channels: 4_096, intermediateSize: 10_240,
      upcast: true, of: FloatType.self)
    let relativePositionBuckets = relativePositionBuckets(
      sequenceLength: tokenLength, numBuckets: 32, maxDistance: 128)
    var attentionMask = Tensor<FloatType>(.CPU, .NHWC(2, 1, 1, tokenLength))
    for i in 0..<tokenLength {
      attentionMask[0, 0, 0, i] = i < tokenLengthUncond ? 0 : -FloatType.greatestFiniteMagnitude
      attentionMask[1, 0, 0, i] = i < tokenLengthCond ? 0 : -FloatType.greatestFiniteMagnitude
    }
    let tokensTensorGPU = tokens[0].toGPU(0)
    let relativePositionBucketsGPU = graph.variable(relativePositionBuckets.toGPU(0))
    let attentionMaskGPU = graph.variable(attentionMask.toGPU(0))
    textModel.compile(inputs: tokensTensorGPU, attentionMaskGPU, relativePositionBucketsGPU)
    if !weightsCache.detach(filePaths[0], to: textModel.parameters) {
      // If we have more than 24GiB RAM, and not forced to be on demand. We load the whole thing (better for weights cache).
      let externalData: DynamicGraph.Store.Codec =
        externalOnDemand || deviceProperties.memoryCapacity != .high
        ? .externalOnDemand : .externalData(deviceProperties.isFreadPreferred ? .fread : .mmap)
      // Move UMT5 XXL to on-demand.
      TensorData.makeExternalData(for: filePaths[0], graph: graph)
      graph.openStore(
        filePaths[0], flags: .readOnly,
        externalStore: TensorData.externalStore(filePath: filePaths[0])
      ) { store in
        store.read(
          "text_model", model: textModel, codec: [.q8p, .q6p, .q4p, .ezm7, .jit, externalData])
      }
    }
    var c = textModel(inputs: tokensTensorGPU, attentionMaskGPU, relativePositionBucketsGPU)[0].as(
      of: FloatType.self
    ).reshaped(.HWC(2, tokenLength, 4096))
    weightsCache.attach(filePaths[0], from: textModel.parameters)
    var encoderMask = Tensor<FloatType>(.CPU, .HWC(2, tokenLength, 1))
    for i in 0..<tokenLength {
      encoderMask[0, i, 0] = i < tokenLengthUncond ? 1 : 0
      encoderMask[1, i, 0] = i < tokenLengthCond ? 1 : 0
    }
    c = c .* graph.variable(encoderMask.toGPU(0))
    guard var input = images.first, filePaths.count >= 2 else {
      return ([c], [textModel])
    }
    // Use OpenCLIP model to get clip embedding of the input image.
    let vit: Model
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
    let inputHeight = input.shape[1]
    let inputWidth = input.shape[2]
    precondition(input.shape[3] == 3)
    if inputHeight != 224 || inputWidth != 224 {
      input =
        (Upsample(
          .bilinear, widthScale: Float(224) / Float(inputWidth),
          heightScale: Float(224) / Float(inputHeight))(input) - mean) .* invStd
    } else {
      input = (input - mean) .* invStd
    }
    let externalData: DynamicGraph.Store.Codec =
      externalOnDemand
      ? .externalOnDemand : .externalData(deviceProperties.isFreadPreferred ? .fread : .mmap)
    if existingTextModels.count >= 1, let existingTextModel = existingTextModels[0] {
      vit = existingTextModel
    } else {
      vit = VisionTransformer(
        FloatType.self, grid: 16, width: 1280, layers: 31, heads: 16, batchSize: 1,
        noFinalLayerNorm: true)
      vit.compile(inputs: input)
      graph.openStore(
        filePaths[1], flags: .readOnly,
        externalStore: TensorData.externalStore(filePath: filePaths[1])
      ) {
        $0.read("vision_model", model: vit, codec: [.jit, .q6p, .q8p, .ezm7, externalData])
      }
    }
    let imageEmbeds = vit(inputs: input)[0].as(of: FloatType.self).reshaped(.HWC(1, 257, 1280))
    return ([c, imageEmbeds], [textModel])
  }

  private func encodeQwen(
    images: [DynamicGraph.Tensor<FloatType>],
    tokens: [DynamicGraph.Tensor<Int32>], positions: [DynamicGraph.Tensor<Int32>],
    mask: [DynamicGraph.Tensor<FloatType>], injectedEmbeddings: [DynamicGraph.Tensor<FloatType>],
    tokenLengthUncond: inout Int, tokenLengthCond: inout Int,
    modifier: SamplerModifier, textModels existingTextModels: [Model?]
  )
    -> ([DynamicGraph.Tensor<FloatType>], [Model])
  {
    let graph = tokens[0].graph
    let externalData: DynamicGraph.Store.Codec =
      externalOnDemand || deviceProperties.memoryCapacity != .high
      ? .externalOnDemand : .externalData(deviceProperties.isFreadPreferred ? .fread : .mmap)
    var tokenLength = tokens[0].shape[0] / 2
    var tokens = tokens
    var injectedEmbeddings = [DynamicGraph.Tensor<FloatType>]()
    var referenceSizes = [(height: Int, width: Int)]()
    if !images.isEmpty, filePaths.count > 1 {
      // Try to find <|image_pad|> in negative prompt and positive prompt.
      var negativePromptPad = [Int]()
      var positivePromptPad = [Int]()
      for i in 0..<tokenLength {
        if tokens[0][i] == 151655, negativePromptPad.count < images.count {
          negativePromptPad.append(i)
        }
        if tokens[0][i + tokenLength] == 151655, positivePromptPad.count < images.count {
          positivePromptPad.append(i)
        }
      }
      if !negativePromptPad.isEmpty && negativePromptPad.count == positivePromptPad.count
        && negativePromptPad.count <= images.count
      {
        let c = images[0..<negativePromptPad.count].map { image in
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
          var input = (image.toGPU(0) - mean) .* invStd
          // Resize w / h to multiple of 28, it might have some stretches, it is OK. This is the smart_resize logic: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py#L55
          let shape = input.shape
          var height = shape[1]
          var width = shape[2]
          if modifier == .qwenimageEditPlus || modifier == .qwenimageEdit2511,
            height > 0 && width > 0
          {
            // Qwen Image Edit Plus pipeline modifies image to area of 384x384.
            let ratio = Double(width) / Double(height)
            let widthD = (384 * 384 * ratio).squareRoot()
            let heightD = widthD / ratio
            // https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit_plus.py#L158
            // choose to round to multiple of 32, but here we do 28 to avoid additional aspect ratio scaling artifacts.
            width = Int((widthD / 28).rounded()) * 28
            height = Int((heightD / 28).rounded()) * 28
          }
          var hBar = (Double(height) / 28).rounded() * 28
          var wBar = (Double(width) / 28).rounded() * 28
          if hBar * wBar > 12_845_056 {  // https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/preprocessor_config.json#L3
            let beta = (Double(height * width) / 12_845_056).squareRoot()
            hBar = max(28, (Double(height) / beta / 28).rounded(.down) * 28)
            wBar = max(28, (Double(width) / beta / 28).rounded(.down) * 28)
          } else if hBar * wBar < 3156 {
            let beta = (3156 / Double(height * width)).squareRoot()
            hBar = (Double(height) / beta / 28).rounded(.up) * 28
            wBar = (Double(width) / beta / 28).rounded(.up) * 28
          }
          let targetHeight = Int(hBar)
          let targetWidth = Int(wBar)
          if shape[1] != targetHeight || shape[2] != targetWidth {
            // Use high-quality resize operation.
            let f32 = Tensor<Float>(
              from: input.rawValue.toCPU().reshaped(.HWC(shape[1], shape[2], shape[3])))
            var b: UnsafeMutablePointer<ccv_dense_matrix_t>? = ccv_dense_matrix_new(
              Int32(targetHeight), Int32(targetWidth), Int32(CCV_C3 | CCV_32F), nil, 0)
            ccv_resample(
              UnsafeMutableRawPointer(f32.cTensor).assumingMemoryBound(to: ccv_dense_matrix_t.self),
              &b,
              0, hBar / Double(shape[1]), wBar / Double(shape[2]),
              Int32(CCV_INTER_AREA | CCV_INTER_CUBIC)
            )
            input = graph.variable(
              Tensor<FloatType>(
                from: Tensor<Float>(
                  .CPU, format: .NHWC, shape: [1, targetHeight, targetWidth, 3],
                  unsafeMutablePointer: b!.pointee.data.f32, bindLifetimeOf: b!
                ).copied()
              ).toGPU(0))
            ccv_matrix_free(b)
          }
          // Input is in h, w, 3 format, the expected format for QwenVL is complicated, let's break it down step-by-step:
          // 1. reformat to NCHW: 1, 3, h, w.
          // 2. move it into 14x14 patch: 1, 3, h / 14, 14, w / 14, 14, then move it to: h / 14, w / 14, 3, 14, 14
          // 3. making it two frames: h / 14, w / 14, 3, 2, 14, 14
          // 4. coordinating with the rotary encoding, but move it further into compact 2x2 blocks: h / 28, 2, w / 28, 2, 3, 2, 14, 14 -> h / 28, w / 28, 2, 2, 3, 2, 14, 14
          // 5. (optional) doing additional work to make sure it is a further 8x8 patch (112x112 in pixel space) for windowed attention.
          input = input.permuted(0, 3, 1, 2).copied().reshaped(
            format: .NHWC, shape: [3, targetHeight / 14, 14, targetWidth / 14, 14]
          ).permuted(1, 3, 0, 2, 4).copied().reshaped(
            .NHWC((targetHeight / 14) * (targetWidth / 14), 3, 14, 14))
          input = Functional.concat(axis: 2, input, input).reshaped(
            format: .NHWC, shape: [targetHeight / 28, 2, targetWidth / 28, 2, 3 * 2 * 14 * 14]
          ).permuted(0, 2, 1, 3, 4).copied().reshaped(
            .HWC(targetHeight / 28, targetWidth / 28, 4 * 3 * 2 * 14 * 14))
          // Step 4 is done, now doing step 5, copying it over to a 8x8 patch (in this case, 4x4 patch because already clustered to 2x2 patch).
          let gridX = targetWidth / 14
          let gridY = targetHeight / 14
          if gridX % 8 == 0, gridY % 8 == 0 {
            // Can directly permuted into the patch required.
            input = input.reshaped(
              format: .NHWC, shape: [gridY / 8, 4, gridX / 8, 4, 4 * 3 * 2 * 14 * 14]
            ).permuted(0, 2, 1, 3, 4).copied().reshaped(.WC(gridY * gridX, 3 * 2 * 14 * 14))
          } else if gridX % 8 == 0 {
            let shape = input.shape
            let top = input[0..<((gridY / 8) * 4), 0..<shape[1], 0..<shape[2]].copied().reshaped(
              format: .NHWC, shape: [gridY / 8, 4, gridX / 8, 4, 4 * 3 * 2 * 14 * 14]
            ).permuted(0, 2, 1, 3, 4).copied().reshaped(
              .WC((gridY / 8) * 8 * gridX, 3 * 2 * 14 * 14))
            let bottom = input[((gridY / 8) * 4)..<shape[0], 0..<shape[1], 0..<shape[2]].copied()
              .reshaped(
                format: .NHWC, shape: [1, (gridY / 2) % 4, gridX / 8, 4, 4 * 3 * 2 * 14 * 14]
              )
              .permuted(0, 2, 1, 3, 4).copied().reshaped(.WC((gridY % 8) * gridX, 3 * 2 * 14 * 14))
            input = Functional.concat(axis: 0, top, bottom)
          } else if gridY % 8 == 0 {
            let shape = input.shape
            let left = input[0..<shape[0], 0..<((gridX / 8) * 4), 0..<shape[2]].copied().reshaped(
              format: .NHWC, shape: [gridY / 8, 4, gridX / 8, 4, 4 * 3 * 2 * 14 * 14]
            ).permuted(0, 2, 1, 3, 4).copied().reshaped(
              .WC(gridY * (gridX / 8) * 8, 3 * 2 * 14 * 14))
            let right = input[0..<shape[0], ((gridX / 8) * 4)..<shape[1], 0..<shape[2]].copied()
              .reshaped(
                format: .NHWC, shape: [gridY / 8, 4, 1, (gridX / 2) % 4, 4 * 3 * 2 * 14 * 14]
              )
              .permuted(0, 2, 1, 3, 4).copied().reshaped(.WC(gridY * (gridX % 8), 3 * 2 * 14 * 14))
            input = Functional.concat(axis: 0, left, right)
          } else {
            let shape = input.shape
            let topLeft = input[0..<((gridY / 8) * 4), 0..<((gridX / 8) * 4), 0..<shape[2]].copied()
              .reshaped(format: .NHWC, shape: [gridY / 8, 4, gridX / 8, 4, 4 * 3 * 2 * 14 * 14])
              .permuted(0, 2, 1, 3, 4).copied().reshaped(
                .WC((gridY / 8) * 8 * (gridX / 8) * 8, 3 * 2 * 14 * 14))
            let right = input[0..<((gridY / 8) * 4), ((gridX / 8) * 4)..<shape[1], 0..<shape[2]]
              .copied().reshaped(
                format: .NHWC, shape: [gridY / 8, 4, 1, (gridX / 2) % 4, 4 * 3 * 2 * 14 * 14]
              ).permuted(0, 2, 1, 3, 4).copied().reshaped(
                .WC((gridY / 8) * 8 * (gridX % 8), 3 * 2 * 14 * 14))
            let bottom = input[((gridY / 8) * 4)..<shape[0], 0..<((gridX / 8) * 4), 0..<shape[2]]
              .copied().reshaped(
                format: .NHWC, shape: [1, (gridY / 2) % 4, gridX / 8, 4, 4 * 3 * 2 * 14 * 14]
              ).permuted(0, 2, 1, 3, 4).copied().reshaped(
                .WC((gridY % 8) * (gridX / 8) * 8, 3 * 2 * 14 * 14))
            let bottomRight = input[
              ((gridY / 8) * 4)..<shape[0], ((gridX / 8) * 4)..<shape[1], 0..<shape[2]
            ].copied().reshaped(
              format: .NHWC, shape: [1, (gridY / 2) % 4, 1, (gridX / 2) % 4, 4 * 3 * 2 * 14 * 14]
            ).permuted(0, 2, 1, 3, 4).copied().reshaped(
              .WC((gridY % 8) * (gridX % 8), 3 * 2 * 14 * 14))
            input = Functional.concat(axis: 0, topLeft, right, bottom, bottomRight)
          }
          let rotTensor = graph.variable(
            Qwen2VLViTRotaryEmbedding(gridX: gridX, gridY: gridY, of: FloatType.self).toGPU(0))
          let vit = Qwen2VLVisionTransformer(
            gridX: gridX, gridY: gridY, width: 1280, layers: 32,
            fullAttentionLayers: [7, 15, 23, 31],
            heads: 16, MLP: 3420, batchSize: 1, usesFlashAttention: usesFlashAttention)
          vit.compile(inputs: input, rotTensor)
          if !weightsCache.detach(filePaths[1], to: vit.parameters) {
            TensorData.makeExternalData(for: filePaths[1], graph: graph)
            graph.openStore(
              filePaths[1], flags: .readOnly,
              externalStore: TensorData.externalStore(filePath: filePaths[1])
            ) {
              $0.read(
                "vit", model: vit, codec: [.q8p, .q6p, .q4p, .ezm7, .jit, externalData])
            }
          }
          var c = vit(inputs: input, rotTensor)[0].as(of: FloatType.self)
          let resultShape = c.shape
          // Unshuffle.
          if gridX % 8 == 0, gridY % 8 == 0 {
            // Can directly permuted into the patch required.
            c = c.reshaped(
              format: .NHWC, shape: [gridY / 8, gridX / 8, 4, 4, resultShape[1]]
            ).permuted(0, 2, 1, 3, 4).copied().reshaped(
              .WC((gridY / 2) * (gridX / 2), resultShape[1]))
          } else if gridX % 8 == 0 {
            let oldC = c
            c = graph.variable(.GPU(0), .HWC(gridY / 2, gridX / 2, resultShape[1]))
            let top = oldC[0..<(gridX / 2) * ((gridY / 8) * 4), 0..<resultShape[1]].copied()
              .reshaped(
                format: .NHWC, shape: [gridY / 8, gridX / 8, 4, 4, resultShape[1]]
              ).permuted(0, 2, 1, 3, 4).copied().reshaped(
                .HWC((gridY / 8) * 4, gridX / 2, resultShape[1]))
            let bottom = oldC[
              ((gridX / 2) * ((gridY / 8) * 4))..<resultShape[0], 0..<resultShape[1]
            ]
            .copied()
            .reshaped(format: .NHWC, shape: [1, gridX / 8, (gridY / 2) % 4, 4, resultShape[1]])
            .permuted(0, 2, 1, 3, 4).copied().reshaped(
              .HWC((gridY / 2) % 4, gridX / 2, resultShape[1]))
            c[0..<((gridY / 8) * 4), 0..<(gridX / 2), 0..<resultShape[1]] = top
            c[((gridY / 8) * 4)..<(gridY / 2), 0..<(gridX / 2), 0..<resultShape[1]] = bottom
            c = c.reshaped(.WC((gridY / 2) * (gridX / 2), resultShape[1]))
          } else if gridY % 8 == 0 {
            let oldC = c
            c = graph.variable(.GPU(0), .HWC(gridY / 2, gridX / 2, resultShape[1]))
            let left = oldC[0..<(gridY / 2) * ((gridX / 8) * 4), 0..<resultShape[1]].copied()
              .reshaped(
                format: .NHWC, shape: [gridY / 8, gridX / 8, 4, 4, resultShape[1]]
              ).permuted(0, 2, 1, 3, 4).copied().reshaped(
                .HWC(gridY / 2, (gridX / 8) * 4, resultShape[1]))
            let right = oldC[((gridY / 2) * ((gridX / 8) * 4))..<resultShape[0], 0..<resultShape[1]]
              .copied()
              .reshaped(format: .NHWC, shape: [gridY / 8, 1, 4, (gridX / 2) % 4, resultShape[1]])
              .permuted(0, 2, 1, 3, 4).copied().reshaped(
                .HWC(gridY / 2, (gridX / 2) % 4, resultShape[1]))
            c[0..<(gridY / 2), 0..<((gridX / 8) * 4), 0..<resultShape[1]] = left
            c[0..<(gridY / 2), ((gridX / 8) * 4)..<(gridX / 2), 0..<resultShape[1]] = right
            c = c.reshaped(.WC((gridY / 2) * (gridX / 2), resultShape[1]))
          } else {
            let oldC = c
            c = graph.variable(.GPU(0), .HWC(gridY / 2, gridX / 2, resultShape[1]))
            let topLeft = oldC[0..<(((gridY / 8) * 4) * ((gridX / 8) * 4)), 0..<resultShape[1]]
              .copied()
              .reshaped(format: .NHWC, shape: [gridY / 8, gridX / 8, 4, 4, resultShape[1]])
              .permuted(0, 2, 1, 3, 4).copied().reshaped(
                .HWC((gridY / 8) * 4, (gridX / 8) * 4, resultShape[1]))
            let right = oldC[
              (((gridY / 8) * 4) * ((gridX / 8) * 4))..<(((gridY / 8) * 4) * (gridX / 2)),
              0..<resultShape[1]
            ]
            .copied().reshaped(
              format: .NHWC, shape: [gridY / 8, 1, 4, (gridX / 2) % 4, resultShape[1]]
            ).permuted(0, 2, 1, 3, 4).copied().reshaped(
              .HWC((gridY / 8) * 4, (gridX / 2) % 4, resultShape[1]))
            let lastCorner = resultShape[0] - ((gridY / 2) % 4) * ((gridX / 2) % 4)
            let bottom = oldC[
              (((gridY / 8) * 4) * (gridX / 2))..<lastCorner, 0..<resultShape[1]
            ]
            .copied().reshaped(
              format: .NHWC, shape: [1, gridX / 8, (gridY / 2) % 4, 4, resultShape[1]]
            ).permuted(0, 2, 1, 3, 4).copied().reshaped(
              .HWC((gridY / 2) % 4, (gridX / 8) * 4, resultShape[1]))
            let bottomRight = oldC[
              lastCorner..<resultShape[0], 0..<resultShape[1]
            ].copied().reshaped(
              format: .NHWC, shape: [1, 1, (gridY / 2) % 4, (gridX / 2) % 4, resultShape[1]]
            ).permuted(0, 2, 1, 3, 4).copied().reshaped(
              .HWC((gridY / 2) % 4, (gridX / 2) % 4, resultShape[1]))
            c[0..<((gridY / 8) * 4), 0..<((gridX / 8) * 4), 0..<resultShape[1]] = topLeft
            c[0..<((gridY / 8) * 4), ((gridX / 8) * 4)..<(gridX / 2), 0..<resultShape[1]] = right
            c[((gridY / 8) * 4)..<(gridY / 2), 0..<((gridX / 8) * 4), 0..<resultShape[1]] = bottom
            c[
              ((gridY / 8) * 4)..<(gridY / 2), ((gridX / 8) * 4)..<(gridX / 2), 0..<resultShape[1]] =
              bottomRight
            c = c.reshaped(.WC((gridY / 2) * (gridX / 2), resultShape[1]))
          }
          weightsCache.attach(filePaths[1], from: vit.parameters)
          referenceSizes.append((height: gridY / 2, gridX / 2))
          return c
        }
        let oldTokenLength = tokenLength
        tokenLength += c.reduce(0) {
          $0 + $1.shape[0] - 1
        }

        var token = Tensor<Int32>(.CPU, format: .NHWC, shape: [tokenLength * 2])
        var mask = Tensor<FloatType>(.CPU, .WC(tokenLength * 2, 1))
        var embeddings = graph.variable(
          .GPU(0), .WC(tokenLength * 2, c[0].shape[1]), of: FloatType.self)
        embeddings.full(0)
        // Copy negative ones over, inject as much pad as needed.
        for i in 0..<negativePromptPad[0] {
          token[i] = tokens[0][i]
          mask[i, 0] = 1
        }
        var offset = negativePromptPad[0]
        var oldOffset = negativePromptPad[0]
        for k in 0..<negativePromptPad.count {
          let resultShape = c[k].shape
          for i in 0..<resultShape[0] {
            token[i + offset] = 151655
            mask[i + offset, 0] = 0
          }
          embeddings[offset..<(offset + resultShape[0]), 0..<resultShape[1]] =
            c[k]
          offset += resultShape[0]
          oldOffset += 1
          for i
            in 0..<((k + 1 < negativePromptPad.count ? negativePromptPad[k + 1] : oldTokenLength)
            - negativePromptPad[k])
          {
            token[i + offset] = tokens[0][i + oldOffset]
            mask[i + offset, 0] = 1
          }
          tokenLengthUncond += resultShape[0] - 1
          if k < negativePromptPad.count - 1 {
            offset += negativePromptPad[k + 1] - negativePromptPad[k] - 1
            oldOffset += negativePromptPad[k + 1] - negativePromptPad[k] - 1
          }
        }
        for i in 0..<positivePromptPad[0] {
          token[i + tokenLength] = tokens[0][i + oldTokenLength]
          mask[i + tokenLength, 0] = 1
        }
        offset = positivePromptPad[0]
        oldOffset = positivePromptPad[0]
        for k in 0..<positivePromptPad.count {
          let resultShape = c[k].shape
          for i in 0..<resultShape[0] {
            token[i + tokenLength + offset] = 151655
            mask[i + tokenLength + offset, 0] = 0
          }
          embeddings[
            (offset + tokenLength)..<(offset + tokenLength + resultShape[0]), 0..<resultShape[1]] =
            c[k]
          offset += resultShape[0]
          oldOffset += 1
          for i
            in 0..<((k + 1 < positivePromptPad.count ? positivePromptPad[k + 1] : oldTokenLength)
            - positivePromptPad[k])
          {
            token[i + tokenLength + offset] = tokens[0][i + oldTokenLength + oldOffset]
            mask[i + tokenLength + offset, 0] = 1
          }
          tokenLengthCond += resultShape[0] - 1
          if k < positivePromptPad.count - 1 {
            offset += positivePromptPad[k + 1] - positivePromptPad[k] - 1
            oldOffset += positivePromptPad[k + 1] - positivePromptPad[k] - 1
          }
        }
        tokens[0] = graph.variable(token)
        injectedEmbeddings.append(contentsOf: [graph.variable(mask.toGPU(0)), embeddings])
      }
    }
    let textModel = Qwen2VL(
      FloatType.self, injectEmbeddings: !injectedEmbeddings.isEmpty, vocabularySize: 152_064,
      maxLength: tokenLength, width: 3_584,
      tokenLength: tokenLength, layers: 28, MLP: 18_944, heads: 28, outputHiddenStates: 28,
      batchSize: 2, usesFlashAttention: usesFlashAttention)
    var causalAttentionMask = Tensor<FloatType>(
      Array(repeating: 0, count: tokenLength * tokenLength), .CPU,
      .NHWC(1, 1, tokenLength, tokenLength)
    )
    for i in 0..<(tokenLength - 1) {
      for j in (i + 1)..<tokenLength {
        causalAttentionMask[0, 0, i, j] = -FloatType.greatestFiniteMagnitude
      }
    }
    let tokensTensorGPU = tokens[0].toGPU(0)
    let rotaryTensorGPU = graph.variable(
      Qwen2VLRotaryEmbedding(
        sequenceLength: tokenLength, token: tokens[0], referenceSizes: referenceSizes,
        of: FloatType.self
      ).toGPU(0))
    let causalAttentionMaskGPU = graph.variable(causalAttentionMask.toGPU(0))
    textModel.compile(
      inputs: [tokensTensorGPU, rotaryTensorGPU, causalAttentionMaskGPU] + injectedEmbeddings)
    if !weightsCache.detach(filePaths[0], to: textModel.parameters) {
      // If we have more than 24GiB RAM, and not forced to be on demand. We load the whole thing (better for weights cache).
      // Move Qwen 2.5 VL to on-demand.
      TensorData.makeExternalData(for: filePaths[0], graph: graph)
      graph.openStore(
        filePaths[0], flags: .readOnly,
        externalStore: TensorData.externalStore(filePath: filePaths[0])
      ) { store in
        store.read(
          "text_model", model: textModel, codec: [.q8p, .q6p, .q4p, .ezm7, .jit, externalData])
      }
    }
    if filePaths.count > 1, tokenLength > 64 {
      let c = textModel(
        inputs: tokensTensorGPU, [rotaryTensorGPU, causalAttentionMaskGPU] + injectedEmbeddings)[0]
        .as(
          of: FloatType.self
        ).reshaped(.HWC(2, tokenLength, 3584))[0..<2, 64..<tokenLength, 0..<3584].contiguous()
      weightsCache.attach(filePaths[0], from: textModel.parameters)
      return ([c], [textModel])
    } else {
      let c = textModel(
        inputs: tokensTensorGPU, [rotaryTensorGPU, causalAttentionMaskGPU] + injectedEmbeddings)[0]
        .as(
          of: FloatType.self
        ).reshaped(.HWC(2, tokenLength, 3584))[0..<2, 34..<tokenLength, 0..<3584].contiguous()
      weightsCache.attach(filePaths[0], from: textModel.parameters)
      return ([c], [textModel])
    }
  }

  private func encodeZImage(
    images: [DynamicGraph.Tensor<FloatType>],
    tokens: [DynamicGraph.Tensor<Int32>], positions: [DynamicGraph.Tensor<Int32>],
    mask: [DynamicGraph.Tensor<FloatType>], injectedEmbeddings: [DynamicGraph.Tensor<FloatType>],
    tokenLengthUncond: inout Int, tokenLengthCond: inout Int,
    modifier: SamplerModifier, textModels existingTextModels: [Model?]
  )
    -> ([DynamicGraph.Tensor<FloatType>], [Model])
  {
    let graph = tokens[0].graph
    let externalData: DynamicGraph.Store.Codec =
      externalOnDemand || deviceProperties.memoryCapacity != .high
      ? .externalOnDemand : .externalData(deviceProperties.isFreadPreferred ? .fread : .mmap)
    let tokenLength = tokens[0].shape[0] / 2
    let textModel = Qwen3(
      FloatType.self, vocabularySize: 151_936,
      maxLength: tokenLength, width: 2_560,
      tokenLength: tokenLength,
      layers: 35 /* Should be 36, but only 35 is enough for this purpose */, MLP: 9_728, heads: 32,
      outputHiddenStates: [34],
      noFinalNormalizedOutput: true, batchSize: 2, usesFlashAttention: usesFlashAttention)
    var causalAttentionMask = Tensor<FloatType>(
      Array(repeating: 0, count: tokenLength * tokenLength), .CPU,
      .NHWC(1, 1, tokenLength, tokenLength)
    )
    for i in 0..<(tokenLength - 1) {
      for j in (i + 1)..<tokenLength {
        causalAttentionMask[0, 0, i, j] = -FloatType.greatestFiniteMagnitude
      }
    }
    let tokensTensorGPU = tokens[0].toGPU(0)
    let rotaryTensorGPU = graph.variable(
      Qwen3RotaryEmbedding(
        sequenceLength: tokenLength, of: FloatType.self
      ).toGPU(0))
    let causalAttentionMaskGPU = graph.variable(causalAttentionMask.toGPU(0))
    textModel.compile(
      inputs: [tokensTensorGPU, rotaryTensorGPU, causalAttentionMaskGPU])
    if !weightsCache.detach(filePaths[0], to: textModel.parameters) {
      // If we have more than 24GiB RAM, and not forced to be on demand. We load the whole thing (better for weights cache).
      // Move Qwen 2.5 VL to on-demand.
      TensorData.makeExternalData(for: filePaths[0], graph: graph)
      graph.openStore(
        filePaths[0], flags: .readOnly,
        externalStore: TensorData.externalStore(filePath: filePaths[0])
      ) { store in
        store.read(
          "text_model", model: textModel, codec: [.q8p, .q6p, .q4p, .ezm7, .jit, externalData])
      }
    }
    let c = textModel(
      inputs: tokensTensorGPU, [rotaryTensorGPU, causalAttentionMaskGPU])[0]
      .as(
        of: FloatType.self
      ).reshaped(.HWC(2, tokenLength, 2560))
    weightsCache.attach(filePaths[0], from: textModel.parameters)
    return ([c], [textModel])
  }

  private func encodeHiDreamI1(
    tokens: [DynamicGraph.Tensor<Int32>], positions: [DynamicGraph.Tensor<Int32>],
    mask: [DynamicGraph.Tensor<FloatType>], injectedEmbeddings: [DynamicGraph.Tensor<FloatType>],
    lengthsOfUncond: [Int], lengthsOfCond: [Int], textModels existingTextModels: [Model?]
  )
    -> ([DynamicGraph.Tensor<FloatType>], [Model])
  {
    let maxLength0 = tokens[0].shape[0] / 2
    var causalAttentionMask0 = Tensor<FloatType>(
      Array(repeating: 0, count: maxLength0 * maxLength0), .CPU, .NHWC(1, 1, maxLength0, maxLength0)
    )
    for i in 0..<(maxLength0 - 1) {
      for j in (i + 1)..<maxLength0 {
        causalAttentionMask0[0, 0, i, j] = -FloatType.greatestFiniteMagnitude
      }
    }
    let maxLength1 = tokens[1].shape[0] / 2
    var causalAttentionMask1 = Tensor<FloatType>(
      Array(repeating: 0, count: maxLength1 * maxLength1), .CPU, .NHWC(1, 1, maxLength1, maxLength1)
    )
    for i in 0..<(maxLength1 - 1) {
      for j in (i + 1)..<maxLength1 {
        causalAttentionMask1[0, 0, i, j] = -FloatType.greatestFiniteMagnitude
      }
    }
    let graph = tokens[1].graph
    let tokens0TensorGPU = tokens[0].toGPU(0)
    let position0TensorGPU = positions[0].toGPU(0)
    let causalAttentionMask0GPU = graph.variable(causalAttentionMask0.toGPU())
    let externalData: DynamicGraph.Store.Codec =
      externalOnDemand
      ? .externalOnDemand : .externalData(deviceProperties.isFreadPreferred ? .fread : .mmap)
    let textModel0 =
      CLIPTextModel(
        FloatType.self, injectEmbeddings: false,
        vocabularySize: 49408, maxLength: 248, maxTokenLength: maxLength0, embeddingSize: 768,
        numLayers: 13 - min(max(clipSkip - 1, 1), 12), numHeads: 12, batchSize: 2,
        intermediateSize: 3072, usesFlashAttention: usesFlashAttention
      ).0
    textModel0.compile(inputs: tokens0TensorGPU, position0TensorGPU, causalAttentionMask0GPU)
    let c0Out: [DynamicGraph.Tensor<FloatType>]
    let textProjection0 = graph.variable(.GPU(0), .WC(768, 768), of: FloatType.self)
    if filePaths.count > 1 {
      graph.openStore(
        filePaths[1], flags: .readOnly,
        externalStore: TensorData.externalStore(filePath: filePaths[1])
      ) { store in
        if lora.count > 0 {
          LoRALoader.openStore(graph, lora: lora) { loader in
            store.read(
              "text_model", model: textModel0, codec: [.jit, .q6p, .q8p, .ezm7, externalData]
            ) { name, dataType, _, shape in
              var name = name
              if name == "__text_model__[t-\(98 - (min(max(clipSkip - 1, 1), 12) - 1) * 8)-0]" {
                name = "__text_model__[t-98-0]"
              } else if name
                == "__text_model__[t-\(98 - (min(max(clipSkip - 1, 1), 12) - 1) * 8)-1]"
              {
                name = "__text_model__[t-98-1]"
              }
              return loader.mergeLoRA(
                graph, name: name, store: store, dataType: dataType, shape: shape,
                of: FloatType.self)
            }
          }
        } else {
          if clipSkip > 1 {
            store.read(
              "text_model", model: textModel0, codec: [.jit, .q6p, .q8p, .ezm7, externalData]
            ) { name, _, _, _ in
              // Retrieve the right final layer norm parameters.
              var name = name
              if name == "__text_model__[t-\(98 - (min(max(clipSkip - 1, 1), 12) - 1) * 8)-0]" {
                name = "__text_model__[t-98-0]"
              } else if name
                == "__text_model__[t-\(98 - (min(max(clipSkip - 1, 1), 12) - 1) * 8)-1]"
              {
                name = "__text_model__[t-98-1]"
              }
              return .continue(name)
            }
          } else {
            store.read(
              "text_model", model: textModel0, codec: [.jit, .q6p, .q8p, .ezm7, externalData])
          }
        }
        store.read(
          "text_projection", variable: textProjection0,
          codec: [
            .q6p, .q8p, .ezm7, .externalData(deviceProperties.isFreadPreferred ? .fread : .mmap),
          ])
      }
      c0Out = textModel0(
        inputs: tokens0TensorGPU, position0TensorGPU, causalAttentionMask0GPU
      ).map {
        $0.as(
          of: FloatType.self
        )
      }
    } else {
      c0Out = [
        graph.variable(.GPU(0), .WC(2 * maxLength0, 768))
      ]
      for c in c0Out {
        c.full(0)
      }
    }
    let batchSize = isCfgEnabled ? 2 : 1
    var pooled = graph.variable(.GPU(0), .WC(batchSize, 2048), of: FloatType.self)
    pooled.full(0)
    var unconditionalTokenEnd0: Int? = nil
    var tokenEnd0: Int? = nil
    for i in 0..<maxLength0 {
      if tokens[0][i] == 49407 && unconditionalTokenEnd0 == nil {
        unconditionalTokenEnd0 = i
      }
      if tokens[0][i + maxLength0] == 49407 && tokenEnd0 == nil {
        tokenEnd0 = i
      }
    }
    if let unconditionalTokenEnd0 = unconditionalTokenEnd0, let tokenEnd0 = tokenEnd0 {
      if isCfgEnabled {
        pooled[0..<1, 0..<768] =
          c0Out[0][unconditionalTokenEnd0..<(unconditionalTokenEnd0 + 1), 0..<768] * textProjection0
        pooled[1..<2, 0..<768] =
          c0Out[0][(maxLength0 + tokenEnd0)..<(maxLength0 + tokenEnd0 + 1), 0..<768]
          * textProjection0
      } else {
        pooled[0..<1, 0..<768] =
          c0Out[0][(maxLength0 + tokenEnd0)..<(maxLength0 + tokenEnd0 + 1), 0..<768]
          * textProjection0
      }
    }
    // Now load OpenCLIP
    let tokens1TensorGPU = tokens[1].toGPU(0)
    let position1TensorGPU = positions[1].toGPU(0)
    let causalAttentionMask1GPU = graph.variable(causalAttentionMask1.toGPU())
    let textModel1 =
      OpenCLIPTextModel(
        FloatType.self, injectEmbeddings: injectEmbeddings,
        vocabularySize: 49408, maxLength: 218, maxTokenLength: maxLength1, embeddingSize: 1280,
        numLayers: 32 - min(max(clipSkip - 2, 0), 30), numHeads: 20, batchSize: 2,
        intermediateSize: 5120, usesFlashAttention: usesFlashAttention
      ).0
    textModel1.compile(
      inputs: tokens1TensorGPU, position1TensorGPU, causalAttentionMask1GPU)
    let textProjection1 = graph.variable(.GPU(0), .WC(1280, 1280), of: FloatType.self)
    let c1Out: [DynamicGraph.Tensor<FloatType>]
    if filePaths.count > 2 {
      graph.openStore(
        filePaths[2], flags: .readOnly,
        externalStore: TensorData.externalStore(filePath: filePaths[2])
      ) { store in
        if lora.count > 0 {
          LoRALoader.openStore(graph, lora: lora) { loader in
            store.read(
              "text_model", model: textModel1, codec: [.jit, .q6p, .q8p, .ezm7, externalData]
            ) { name, dataType, _, shape in
              // Retrieve the right final layer norm parameters.
              var name = name
              if name == "__text_model__[t-\(258 - (min(max(clipSkip - 1, 1), 31) - 1) * 8)-0]" {
                name = "__text_model__[t-258-0]"
              } else if name
                == "__text_model__[t-\(258 - (min(max(clipSkip - 1, 1), 31) - 1) * 8)-1]"
              {
                name = "__text_model__[t-258-1]"
              }
              return loader.mergeLoRA(
                graph, name: name, store: store, dataType: dataType, shape: shape,
                of: FloatType.self, prefix: "__te2")
            }
          }
        } else if clipSkip > 1 {
          store.read(
            "text_model", model: textModel1, codec: [.jit, .q6p, .q8p, .ezm7, externalData]
          ) {
            name, _, _, _ in
            // Retrieve the right final layer norm parameters.
            var name = name
            if name == "__text_model__[t-\(258 - (min(max(clipSkip - 1, 1), 31) - 1) * 8)-0]" {
              name = "__text_model__[t-258-0]"
            } else if name == "__text_model__[t-\(258 - (min(max(clipSkip - 1, 1), 31) - 1) * 8)-1]"
            {
              name = "__text_model__[t-258-1]"
            }
            return .continue(name)
          }
        } else {
          store.read(
            "text_model", model: textModel1, codec: [.jit, .q6p, .q8p, .ezm7, externalData])
        }
        store.read(
          "text_projection", variable: textProjection1,
          codec: [
            .q6p, .q8p, .ezm7, .externalData(deviceProperties.isFreadPreferred ? .fread : .mmap),
          ])
      }
      c1Out = textModel1(
        inputs: tokens1TensorGPU, position1TensorGPU, causalAttentionMask1GPU
      ).map {
        $0.as(
          of: FloatType.self
        )
      }
    } else {
      c1Out = [
        graph.variable(.GPU(0), .WC(2 * maxLength1, 1280))
      ]
      for c in c1Out {
        c.full(0)
      }
    }
    var unconditionalTokenEnd1: Int? = nil
    var tokenEnd1: Int? = nil
    for i in 0..<maxLength1 {
      if tokens[1][i] == 49407 && unconditionalTokenEnd1 == nil {
        unconditionalTokenEnd1 = i
      }
      if tokens[1][i + maxLength1] == 49407 && tokenEnd1 == nil {
        tokenEnd1 = i
      }
    }
    if let unconditionalTokenEnd1 = unconditionalTokenEnd1, let tokenEnd1 = tokenEnd1 {
      if isCfgEnabled {
        pooled[0..<1, 768..<2048] =
          c1Out[0][unconditionalTokenEnd1..<(unconditionalTokenEnd1 + 1), 0..<1280]
          * textProjection1
        pooled[1..<2, 768..<2048] =
          c1Out[0][(maxLength1 + tokenEnd1)..<(maxLength1 + tokenEnd1 + 1), 0..<1280]
          * textProjection1
      } else {
        pooled[0..<1, 768..<2048] =
          c1Out[0][(maxLength1 + tokenEnd1)..<(maxLength1 + tokenEnd1 + 1), 0..<1280]
          * textProjection1
      }
    }
    let c2: DynamicGraph.Tensor<FloatType>
    if filePaths.count > 3 {
      let tokenLength = tokens[2].shape[0] / 2
      let (_, t5) = T5ForConditionalGeneration(
        b: batchSize, t: tokenLength, attentionMask: tokenLength <= 128, of: FloatType.self)
      let relativePositionBuckets = relativePositionBuckets(
        sequenceLength: tokenLength, numBuckets: 32, maxDistance: 128)
      let tokens2TensorGPU: DynamicGraph.Tensor<Int32>
      let attentionMask: Tensor<FloatType>?
      if isCfgEnabled {
        tokens2TensorGPU = tokens[2].toGPU(0)
        if tokenLength <= 128 {
          var mask = Tensor<FloatType>(
            Array(repeating: 0, count: 2 * tokenLength), .CPU, .NHWC(2, 1, 1, tokenLength)
          )
          if lengthsOfUncond[0] < tokenLength {
            for i in lengthsOfUncond[0]..<tokenLength {
              mask[0, 0, 0, i] = -FloatType.greatestFiniteMagnitude
            }
          }
          if lengthsOfCond[0] < tokenLength {
            for i in lengthsOfCond[0]..<tokenLength {
              mask[1, 0, 0, i] = -FloatType.greatestFiniteMagnitude
            }
          }
          attentionMask = mask
        } else {
          attentionMask = nil
        }
      } else {
        tokens2TensorGPU = tokens[2][tokenLength..<(tokenLength * 2)].toGPU(0)
        if tokenLength <= 128 {
          var mask = Tensor<FloatType>(
            Array(repeating: 0, count: tokenLength), .CPU, .NHWC(1, 1, 1, tokenLength)
          )
          if lengthsOfCond[0] < tokenLength {
            for i in lengthsOfCond[0]..<tokenLength {
              mask[0, 0, 0, i] = -FloatType.greatestFiniteMagnitude
            }
          }
          attentionMask = mask
        } else {
          attentionMask = nil
        }
      }
      let relativePositionBucketsGPU = graph.variable(relativePositionBuckets.toGPU(0))
      let attentionMaskGPU = attentionMask.map { graph.variable($0.toGPU(0)) }
      t5.compile(
        inputs: [tokens2TensorGPU, relativePositionBucketsGPU]
          + (attentionMaskGPU.map { [$0] } ?? []))
      if !weightsCache.detach(filePaths[3], to: t5.parameters) {
        // If we have more than 24GiB RAM, and not forced to be on demand. We load the whole thing (better for weights cache).
        let externalData: DynamicGraph.Store.Codec =
          externalOnDemand || deviceProperties.memoryCapacity != .high
          ? .externalOnDemand : .externalData(deviceProperties.isFreadPreferred ? .fread : .mmap)
        // Move T5 to on-demand.
        TensorData.makeExternalData(for: filePaths[3], graph: graph)
        graph.openStore(
          filePaths[3], flags: .readOnly,
          externalStore: TensorData.externalStore(filePath: filePaths[3])
        ) {
          $0.read(
            "text_model", model: t5, codec: [.q8p, .q6p, .q4p, .ezm7, .jit, externalData])
        }
      }
      c2 = t5(
        inputs: tokens2TensorGPU,
        [relativePositionBucketsGPU] + (attentionMaskGPU.map { [$0] } ?? []))[0].as(
          of: FloatType.self
        ).reshaped(.HWC(batchSize, tokenLength, 4096))
      weightsCache.attach(filePaths[3], from: t5.parameters)
    } else {
      let tokenLength = tokens[2].shape[0] / 2
      c2 = graph.variable(.GPU(0), .HWC(batchSize, tokenLength, 4096))
      c2.full(0)
    }
    // Now run llama3.
    let tokenLength = tokens[3].shape[0] / 2
    let llama3 = Llama3(
      FloatType.self, vocabularySize: 128_256, width: 4_096,
      tokenLength: (tokenLength, tokenLength, tokenLength),
      layers: 32, MLP: 14336, heads: 32, outputHiddenStates: Set(1..<32), batchSize: batchSize,
      usesFlashAttention: usesFlashAttention)
    let tokens3TensorGPU: DynamicGraph.Tensor<Int32>
    var causalAttentionMaskLlama3 = Tensor<FloatType>(
      Array(repeating: 0, count: batchSize * tokenLength * tokenLength), .CPU,
      .NHWC(batchSize, 1, tokenLength, tokenLength)
    )
    if isCfgEnabled {
      tokens3TensorGPU = tokens[3].toGPU(0)
      for i in 0..<tokenLength {
        if i == tokenLength - 1 && lengthsOfUncond[1] >= tokenLength {
          continue
        }
        for j in min(i + 1, lengthsOfUncond[1])..<tokenLength {
          causalAttentionMaskLlama3[0, 0, i, j] = -FloatType.greatestFiniteMagnitude
        }
      }
      for i in 0..<tokenLength {
        if i == tokenLength - 1 && lengthsOfCond[1] >= tokenLength {
          continue
        }
        for j in min(i + 1, lengthsOfCond[1])..<tokenLength {
          causalAttentionMaskLlama3[1, 0, i, j] = -FloatType.greatestFiniteMagnitude
        }
      }
    } else {
      tokens3TensorGPU = tokens[3][tokenLength..<(tokenLength * 2)].toGPU(0)
      for i in 0..<tokenLength {
        if i == tokenLength - 1 && lengthsOfCond[1] >= tokenLength {
          continue
        }
        for j in min(i + 1, lengthsOfCond[1])..<tokenLength {
          causalAttentionMaskLlama3[0, 0, i, j] = -FloatType.greatestFiniteMagnitude
        }
      }
    }
    let rotaryTensorGPU = graph.variable(
      Llama3RotaryEmbedding(sequenceLength: tokenLength, of: FloatType.self).toGPU(0))
    let causalAttentionMaskLlama3GPU = graph.variable(causalAttentionMaskLlama3.toGPU(0))
    llama3.compile(
      inputs: [tokens3TensorGPU, rotaryTensorGPU, causalAttentionMaskLlama3GPU])
    if !weightsCache.detach(filePaths[0], to: llama3.parameters) {
      // If we have more than 24GiB RAM, and not forced to be on demand. We load the whole thing (better for weights cache).
      let externalData: DynamicGraph.Store.Codec =
        externalOnDemand || deviceProperties.memoryCapacity != .high
        ? .externalOnDemand : .externalData(deviceProperties.isFreadPreferred ? .fread : .mmap)
      // Move Llama3 8B to on-demand.
      TensorData.makeExternalData(for: filePaths[0], graph: graph)
      graph.openStore(
        filePaths[0], flags: .readOnly,
        externalStore: TensorData.externalStore(filePath: filePaths[0])
      ) {
        $0.read(
          "text_model", model: llama3, codec: [.q8p, .q6p, .q4p, .ezm7, .jit, externalData])
      }
    }
    let c3 = llama3(
      inputs: tokens3TensorGPU,
      [rotaryTensorGPU, causalAttentionMaskLlama3GPU]
    ).map {
      $0.as(
        of: FloatType.self
      ).reshaped(.HWC(batchSize, tokenLength, 4096))
    }
    weightsCache.attach(filePaths[0], from: llama3.parameters)
    return ([pooled, c2] + c3, [])
  }

  private func encodeFlux2Mistral(
    images: [DynamicGraph.Tensor<FloatType>],
    tokens: [DynamicGraph.Tensor<Int32>], positions: [DynamicGraph.Tensor<Int32>],
    mask: [DynamicGraph.Tensor<FloatType>], injectedEmbeddings: [DynamicGraph.Tensor<FloatType>],
    tokenLengthUncond: inout Int, tokenLengthCond: inout Int,
    modifier: SamplerModifier, textModels existingTextModels: [Model?]
  )
    -> ([DynamicGraph.Tensor<FloatType>], [Model])
  {
    let graph = tokens[0].graph
    let externalData: DynamicGraph.Store.Codec =
      externalOnDemand || deviceProperties.memoryCapacity != .high
      ? .externalOnDemand : .externalData(deviceProperties.isFreadPreferred ? .fread : .mmap)
    let tokenLength = tokens[0].shape[0] / 2
    let textModel = Mistral3(
      FloatType.self, vocabularySize: 131_072, maxLength: tokenLength, width: 5_120,
      tokenLength: tokenLength,
      layers: 30 /* Should be 40, but only 30 is enough for this purpose */, MLP: 32_768,
      heads: 32, outputHiddenStates: [9, 19, 29], noFinalNormalizedOutput: true, batchSize: 2,
      usesFlashAttention: usesFlashAttention)
    var causalAttentionMask = Tensor<FloatType>(
      Array(repeating: 0, count: tokenLength * tokenLength), .CPU,
      .NHWC(1, 1, tokenLength, tokenLength)
    )
    for i in 0..<(tokenLength - 1) {
      for j in (i + 1)..<tokenLength {
        causalAttentionMask[0, 0, i, j] = -FloatType.greatestFiniteMagnitude
      }
    }
    let tokensTensorGPU = tokens[0].toGPU(0)
    let maxTokenLength = 512
    let rotaryTensorGPUUncond = graph.variable(
      Mistral3RotaryEmbedding(
        sequenceLength: tokenLength, endAligned: max(0, maxTokenLength - tokenLengthUncond),
        of: FloatType.self
      ).toGPU(0))
    let rotaryTensorGPUCond = graph.variable(
      Mistral3RotaryEmbedding(
        sequenceLength: tokenLength, endAligned: max(0, maxTokenLength - tokenLengthCond),
        of: FloatType.self
      ).toGPU(0))
    let rotaryTensorGPU = Functional.concat(axis: 0, rotaryTensorGPUUncond, rotaryTensorGPUCond)
    let causalAttentionMaskGPU = graph.variable(causalAttentionMask.toGPU(0))
    // The pad token.
    let padToken = Tensor<Int32>([11], kind: .CPU, format: .NHWC, shape: [1])
    let padTokenGPU = graph.variable(padToken.toGPU(0))
    textModel.compile(
      inputs: [tokensTensorGPU, rotaryTensorGPU, causalAttentionMaskGPU, padTokenGPU])
    if !weightsCache.detach(filePaths[0], to: textModel.parameters) {
      // If we have more than 24GiB RAM, and not forced to be on demand. We load the whole thing (better for weights cache).
      TensorData.makeExternalData(for: filePaths[0], graph: graph)
      graph.openStore(
        filePaths[0], flags: .readOnly,
        externalStore: TensorData.externalStore(filePath: filePaths[0])
      ) { store in
        store.read(
          "text_model", model: textModel, codec: [.q8p, .q6p, .q4p, .ezm7, .jit, externalData])
      }
    }
    var c = textModel(
      inputs: tokensTensorGPU, [rotaryTensorGPU, causalAttentionMaskGPU, padTokenGPU]
    ).map { $0.as(of: FloatType.self) }
    c[0] = c[0].reshaped(.HWC(2, tokenLength, 5120))
    c[1] = c[1].reshaped(.HWC(2, tokenLength, 5120))
    c[2] = c[2].reshaped(.HWC(2, tokenLength, 5120))
    weightsCache.attach(filePaths[0], from: textModel.parameters)
    // Now handles C.
    let paddedTokenLength = max(tokenLength, maxTokenLength)
    var newC = graph.variable(.GPU(0), .HWC(2, paddedTokenLength, 15360), of: FloatType.self)
    newC[
      0..<1, max(0, (maxTokenLength - tokenLengthUncond))..<max(tokenLengthUncond, maxTokenLength),
      0..<5120] =
      c[0][0..<1, 0..<tokenLengthUncond, 0..<5120]
    newC[
      0..<1, max(0, (maxTokenLength - tokenLengthUncond))..<max(tokenLengthUncond, maxTokenLength),
      5120..<10240] =
      c[1][0..<1, 0..<tokenLengthUncond, 0..<5120]
    newC[
      0..<1, max(0, (maxTokenLength - tokenLengthUncond))..<max(tokenLengthUncond, maxTokenLength),
      10240..<15360] =
      c[2][0..<1, 0..<tokenLengthUncond, 0..<5120]
    let padEmbed = c[3].reshaped(.HWC(1, 1, 5120))
    if tokenLengthUncond < maxTokenLength {
      // Copy pad token to the beginning.
      for i in 0..<(maxTokenLength - tokenLengthUncond) {
        newC[0..<1, i..<(i + 1), 0..<5120] = padEmbed
        newC[0..<1, i..<(i + 1), 5120..<10240] = padEmbed
        newC[0..<1, i..<(i + 1), 10240..<15360] = padEmbed
      }
    }
    newC[
      1..<2, max(0, (maxTokenLength - tokenLengthCond))..<max(tokenLengthCond, maxTokenLength),
      0..<5120] =
      c[0][1..<2, 0..<tokenLengthCond, 0..<5120]
    newC[
      1..<2, max(0, (maxTokenLength - tokenLengthCond))..<max(tokenLengthCond, maxTokenLength),
      5120..<10240] =
      c[1][1..<2, 0..<tokenLengthCond, 0..<5120]
    newC[
      1..<2, max(0, (maxTokenLength - tokenLengthCond))..<max(tokenLengthCond, maxTokenLength),
      10240..<15360] =
      c[2][1..<2, 0..<tokenLengthCond, 0..<5120]
    if tokenLengthCond < maxTokenLength {
      // Copy pad token to the beginning.
      for i in 0..<(maxTokenLength - tokenLengthCond) {
        newC[0..<1, i..<(i + 1), 0..<5120] = padEmbed
        newC[0..<1, i..<(i + 1), 5120..<10240] = padEmbed
        newC[0..<1, i..<(i + 1), 10240..<15360] = padEmbed
      }
    }
    // Make sure it is padded to 512 at least.
    tokenLengthCond = max(maxTokenLength, tokenLengthCond)
    tokenLengthUncond = max(maxTokenLength, tokenLengthUncond)
    return ([newC], [textModel])
  }

  private func encodeFlux2Qwen3(
    images: [DynamicGraph.Tensor<FloatType>],
    tokens: [DynamicGraph.Tensor<Int32>], positions: [DynamicGraph.Tensor<Int32>],
    mask: [DynamicGraph.Tensor<FloatType>], injectedEmbeddings: [DynamicGraph.Tensor<FloatType>],
    tokenLengthUncond: inout Int, tokenLengthCond: inout Int,
    modifier: SamplerModifier, textModels existingTextModels: [Model?]
  )
    -> ([DynamicGraph.Tensor<FloatType>], [Model])
  {
    let graph = tokens[0].graph
    let externalData: DynamicGraph.Store.Codec =
      externalOnDemand || deviceProperties.memoryCapacity != .high
      ? .externalOnDemand : .externalData(deviceProperties.isFreadPreferred ? .fread : .mmap)
    let tokenLength = tokens[0].shape[0] / 2
    let channels: Int
    let MLP: Int
    if version == .flux2_9b {
      channels = 4_096
      MLP = 12_288
    } else {
      channels = 2_560
      MLP = 9_728
    }
    var tokenLength0: Int? = nil
    var tokenLength1: Int? = nil
    for i in (0..<tokenLength).reversed() {
      if tokenLength0 == nil && tokens[0][i] != 151643 {
        tokenLength0 = i + 1
      }
      if tokenLength1 == nil && tokens[0][i + tokenLength] != 151643 {
        tokenLength1 = i + 1
      }
      if tokenLength0 != nil && tokenLength1 != nil {
        break
      }
    }
    let textModel = Qwen3(
      FloatType.self, vocabularySize: 151_936,
      maxLength: tokenLength, width: channels,
      tokenLength: tokenLength,
      layers: 27 /* Should be 36, but only 27 is enough for this purpose */, MLP: MLP, heads: 32,
      outputHiddenStates: [8, 17, 26], noFinalNormalizedOutput: true,
      batchSize: 2, usesFlashAttention: usesFlashAttention)
    var causalAttentionMask = Tensor<FloatType>(
      Array(repeating: 0, count: tokenLength * tokenLength * 2), .CPU,
      .NHWC(2, 1, tokenLength, tokenLength)
    )
    for i in 0..<tokenLength {
      let tokenStart0 = min(tokenLength0 ?? tokenLength, i + 1)
      if tokenStart0 < tokenLength {
        for j in tokenStart0..<tokenLength {
          causalAttentionMask[0, 0, i, j] = -FloatType.greatestFiniteMagnitude
        }
      }
      let tokenStart1 = min(tokenLength1 ?? tokenLength, i + 1)
      if tokenStart1 < tokenLength {
        for j in tokenStart1..<tokenLength {
          causalAttentionMask[1, 0, i, j] = -FloatType.greatestFiniteMagnitude
        }
      }
    }
    let tokensTensorGPU = tokens[0].toGPU(0)
    let rotaryTensorGPU = graph.variable(
      Qwen3RotaryEmbedding(
        sequenceLength: tokenLength, of: FloatType.self
      ).toGPU(0))
    let causalAttentionMaskGPU = graph.variable(causalAttentionMask.toGPU(0))
    textModel.compile(
      inputs: [tokensTensorGPU, rotaryTensorGPU, causalAttentionMaskGPU])
    if !weightsCache.detach(filePaths[0], to: textModel.parameters) {
      TensorData.makeExternalData(for: filePaths[0], graph: graph)
      graph.openStore(
        filePaths[0], flags: .readOnly,
        externalStore: TensorData.externalStore(filePath: filePaths[0])
      ) { store in
        store.read(
          "text_model", model: textModel, codec: [.q8p, .q6p, .q4p, .ezm7, .jit, externalData])
      }
    }
    let c = textModel(
      inputs: tokensTensorGPU, [rotaryTensorGPU, causalAttentionMaskGPU]
    ).map {
      $0.as(
        of: FloatType.self
      ).reshaped(.HWC(2, tokenLength, channels))
    }
    weightsCache.attach(filePaths[0], from: textModel.parameters)
    var newC = graph.variable(.GPU(0), .HWC(2, tokenLength, channels * 3), of: FloatType.self)
    newC[0..<1, 0..<tokenLength, 0..<channels] = c[0][0..<1, 0..<tokenLength, 0..<channels]
    newC[0..<1, 0..<tokenLength, channels..<(channels * 2)] =
      c[1][0..<1, 0..<tokenLength, 0..<channels]
    newC[0..<1, 0..<tokenLength, (channels * 2)..<(channels * 3)] =
      c[2][0..<1, 0..<tokenLength, 0..<channels]
    newC[1..<2, 0..<tokenLength, 0..<channels] = c[0][1..<2, 0..<tokenLength, 0..<channels]
    newC[1..<2, 0..<tokenLength, channels..<(channels * 2)] =
      c[1][1..<2, 0..<tokenLength, 0..<channels]
    newC[1..<2, 0..<tokenLength, (channels * 2)..<(channels * 3)] =
      c[2][1..<2, 0..<tokenLength, 0..<channels]
    return ([newC], [textModel])
  }

  private func encodeLTX2(
    images: [DynamicGraph.Tensor<FloatType>],
    tokens: [DynamicGraph.Tensor<Int32>], positions: [DynamicGraph.Tensor<Int32>],
    mask: [DynamicGraph.Tensor<FloatType>], injectedEmbeddings: [DynamicGraph.Tensor<FloatType>],
    tokenLengthUncond: inout Int, tokenLengthCond: inout Int,
    modifier: SamplerModifier, textModels existingTextModels: [Model?]
  )
    -> ([DynamicGraph.Tensor<FloatType>], [Model])
  {
    let graph = tokens[0].graph
    let externalData: DynamicGraph.Store.Codec =
      externalOnDemand || deviceProperties.memoryCapacity != .high
      ? .externalOnDemand : .externalData(deviceProperties.isFreadPreferred ? .fread : .mmap)
    let tokenLength = tokens[0].shape[0] / 2
    let textModel = Gemma3(
      FloatType.self, vocabularySize: 262_208, maxLength: tokenLength, width: 3_840,
      tokenLength: tokenLength, layers: 48, MLP: 15_360, heads: 16, batchSize: 2,
      usesFlashAttention: usesFlashAttention)
    var causalAttentionMask = Tensor<FloatType>(
      Array(repeating: 0, count: tokenLength * tokenLength), .CPU,
      .NHWC(1, 1, tokenLength, tokenLength)
    )
    for i in 0..<(tokenLength - 1) {
      for j in (i + 1)..<tokenLength {
        causalAttentionMask[0, 0, i, j] = -FloatType.greatestFiniteMagnitude
      }
    }
    let tokensTensorGPU = tokens[0].toGPU(0)
    let maxTokenLength = 1024
    let (rotaryLocalUncond, rotaryUncond) = Gemma3RotaryEmbedding(
      sequenceLength: tokenLength, endAligned: max(0, maxTokenLength - tokenLengthUncond),
      of: FloatType.self
    )
    let (rotaryLocalCond, rotaryCond) = Gemma3RotaryEmbedding(
      sequenceLength: tokenLength, endAligned: max(0, maxTokenLength - tokenLengthCond),
      of: FloatType.self
    )
    let rotaryLocalGPUUncond = graph.variable(rotaryLocalUncond.toGPU(0))
    let rotaryGPUUncond = graph.variable(rotaryUncond.toGPU(0))
    let rotaryLocalGPUCond = graph.variable(rotaryLocalCond.toGPU(0))
    let rotaryGPUCond = graph.variable(rotaryCond.toGPU(0))
    let rotaryLocalTensorGPU = Functional.concat(axis: 0, rotaryLocalGPUUncond, rotaryLocalGPUCond)
    let rotaryTensorGPU = Functional.concat(axis: 0, rotaryGPUUncond, rotaryGPUCond)
    let causalAttentionMaskGPU = graph.variable(causalAttentionMask.toGPU(0))
    textModel.compile(
      inputs: [tokensTensorGPU, rotaryLocalTensorGPU, rotaryTensorGPU, causalAttentionMaskGPU])
    if !weightsCache.detach(filePaths[0], to: textModel.parameters) {
      // If we have more than 24GiB RAM, and not forced to be on demand. We load the whole thing (better for weights cache).
      // Move Qwen 2.5 VL to on-demand.
      TensorData.makeExternalData(for: filePaths[0], graph: graph)
      graph.openStore(
        filePaths[0], flags: .readOnly,
        externalStore: TensorData.externalStore(filePath: filePaths[0])
      ) { store in
        store.read(
          "text_model", model: textModel, codec: [.q8p, .q6p, .q4p, .ezm7, .jit, externalData])
      }
    }
    let outputHiddenStates = textModel(
      inputs: tokensTensorGPU, [rotaryLocalTensorGPU, rotaryTensorGPU, causalAttentionMaskGPU]
    ).map {
      $0.as(
        of: Float.self
      ).reshaped(.HWC(2, tokenLength, 3840))
    }
    weightsCache.attach(filePaths[0], from: textModel.parameters)
    var hiddenStates = graph.variable(.GPU(0), .NHWC(2, tokenLength, 3840, 49), of: Float.self)
    for i in 0..<49 {
      hiddenStates[0..<2, 0..<tokenLength, 0..<3840, i..<(i + 1)] = outputHiddenStates[i]
        .reshaped(.NHWC(2, tokenLength, 3840, 1))
    }
    // Separately normalize unconditional / conditional branch.
    var hiddenStatesUncond = hiddenStates[
      0..<1, 0..<min(tokenLength, tokenLengthUncond), 0..<3840, 0..<49
    ].contiguous()
    let meanUncond = hiddenStatesUncond.reduced(.mean, axis: [1, 2])
    let minUncond_ = hiddenStatesUncond.reduced(.min, axis: [1, 2])
    let maxUncond_ = hiddenStatesUncond.reduced(.max, axis: [1, 2])
    let rangeUncond_ = 8.0 * Functional.reciprocal(maxUncond_ - minUncond_)
    hiddenStatesUncond = (hiddenStates[0..<1, 0..<tokenLength, 0..<3840, 0..<49] - meanUncond)
      .* rangeUncond_
    var hiddenStatesCond = hiddenStates[
      1..<2, 0..<min(tokenLength, tokenLengthCond), 0..<3840, 0..<49
    ].contiguous()
    let meanCond = hiddenStatesCond.reduced(.mean, axis: [1, 2])
    let minCond_ = hiddenStatesCond.reduced(.min, axis: [1, 2])
    let maxCond_ = hiddenStatesCond.reduced(.max, axis: [1, 2])
    let rangeCond_ = 8.0 * Functional.reciprocal(maxCond_ - minCond_)
    hiddenStatesCond = (hiddenStates[1..<2, 0..<tokenLength, 0..<3840, 0..<49] - meanCond)
      .* rangeCond_
    let normedHiddenStates = DynamicGraph.Tensor<BFloat16>(
      from: Functional.concat(axis: 0, hiddenStatesUncond, hiddenStatesCond)
    ).reshaped(.HWC(2, tokenLength, 3840 * 49))
    let featureExtractorLinear = Dense(count: 3840, noBias: true)
    featureExtractorLinear.compile(inputs: normedHiddenStates)
    graph.openStore(
      filePaths[1], flags: .readOnly,
      externalStore: TensorData.externalStore(filePath: filePaths[1])
    ) {
      $0.read(
        "text_feature_extractor", model: featureExtractorLinear,
        codec: [.q8p, .q6p, .q4p, .ezm7, .jit, externalData])
    }
    let c = featureExtractorLinear(inputs: normedHiddenStates).map {
      DynamicGraph.Tensor<FloatType>(from: $0)
    }
    return (c, [textModel])
  }

  public func encode(
    tokenLengthUncond: inout Int, tokenLengthCond: inout Int,
    tokens: [DynamicGraph.Tensor<Int32>], positions: [DynamicGraph.Tensor<Int32>],
    mask: [DynamicGraph.Tensor<FloatType>], injectedEmbeddings: [DynamicGraph.Tensor<FloatType>],
    images: [DynamicGraph.Tensor<FloatType>], lengthsOfUncond: [Int], lengthsOfCond: [Int],
    injectedTextEmbeddings: [(
      model: ControlModel<FloatType>, hints: [([DynamicGraph.Tensor<FloatType>], Float)]
    )], modifier: SamplerModifier,
    textModels existingTextModels: [Model?]
  )
    -> ([DynamicGraph.Tensor<FloatType>], [Model])
  {
    let conditionalLength: Int
    switch version {
    case .v1:
      conditionalLength = 768
    case .v2:
      conditionalLength = 1024
    case .sd3, .sd3Large:
      return encodeSD3(
        tokens: tokens, positions: positions, mask: mask, injectedEmbeddings: injectedEmbeddings,
        lengthsOfUncond: lengthsOfUncond, lengthsOfCond: lengthsOfCond,
        textModels: existingTextModels)
    case .pixart:
      return encodePixArt(
        tokens: tokens, positions: positions, mask: mask, injectedEmbeddings: injectedEmbeddings,
        lengthsOfUncond: lengthsOfUncond, lengthsOfCond: lengthsOfCond,
        textModels: existingTextModels)
    case .auraflow:
      return encodeAuraFlow(
        tokens: tokens, positions: positions, mask: mask, injectedEmbeddings: injectedEmbeddings,
        lengthsOfUncond: lengthsOfUncond, lengthsOfCond: lengthsOfCond,
        textModels: existingTextModels)
    case .flux1:
      return encodeFlux1(
        tokens: tokens, positions: positions, mask: mask, injectedEmbeddings: injectedEmbeddings,
        lengthsOfUncond: lengthsOfUncond, lengthsOfCond: lengthsOfCond,
        textModels: existingTextModels)
    case .hiDreamI1:
      return encodeHiDreamI1(
        tokens: tokens, positions: positions, mask: mask, injectedEmbeddings: injectedEmbeddings,
        lengthsOfUncond: lengthsOfUncond, lengthsOfCond: lengthsOfCond,
        textModels: existingTextModels)
    case .zImage:
      return encodeZImage(
        images: images, tokens: tokens, positions: positions, mask: mask,
        injectedEmbeddings: injectedEmbeddings,
        tokenLengthUncond: &tokenLengthUncond, tokenLengthCond: &tokenLengthCond,
        modifier: modifier, textModels: existingTextModels)
    case .flux2:
      return encodeFlux2Mistral(
        images: images, tokens: tokens, positions: positions, mask: mask,
        injectedEmbeddings: injectedEmbeddings,
        tokenLengthUncond: &tokenLengthUncond, tokenLengthCond: &tokenLengthCond,
        modifier: modifier, textModels: existingTextModels)
    case .flux2_9b, .flux2_4b:
      return encodeFlux2Qwen3(
        images: images, tokens: tokens, positions: positions, mask: mask,
        injectedEmbeddings: injectedEmbeddings,
        tokenLengthUncond: &tokenLengthUncond, tokenLengthCond: &tokenLengthCond,
        modifier: modifier, textModels: existingTextModels)
    case .ltx2:
      return encodeLTX2(
        images: images, tokens: tokens, positions: positions, mask: mask,
        injectedEmbeddings: injectedEmbeddings,
        tokenLengthUncond: &tokenLengthUncond, tokenLengthCond: &tokenLengthCond,
        modifier: modifier, textModels: existingTextModels)
    case .kandinsky21:
      return encodeKandinsky(tokens: tokens, positions: positions)
    case .sdxlBase, .sdxlRefiner, .ssd1b:
      switch textEncoderVersion {
      case .chatglm3_6b:
        return encodeChatGLM3(
          tokens: tokens, positions: positions, mask: mask, injectedEmbeddings: injectedEmbeddings,
          lengthsOfUncond: lengthsOfUncond, lengthsOfCond: lengthsOfCond,
          textModels: existingTextModels)
      case nil:
        return encodeSDXL(
          tokens: tokens, positions: positions, mask: mask, injectedEmbeddings: injectedEmbeddings,
          lengthsOfUncond: lengthsOfUncond, lengthsOfCond: lengthsOfCond,
          textModels: existingTextModels)
      }
    case .svdI2v:
      return encodeI2v(images: images, textModels: existingTextModels)
    case .hunyuanVideo:
      return encodeHunyuan(
        tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
        tokens: tokens, positions: positions, mask: mask, injectedEmbeddings: injectedEmbeddings,
        lengthsOfUncond: lengthsOfUncond, lengthsOfCond: lengthsOfCond,
        injectedTextEmbeddings: injectedTextEmbeddings, textModels: existingTextModels)
    case .wan21_1_3b, .wan21_14b, .wan22_5b:
      return encodeWan(
        images: images,
        tokens: tokens, positions: positions, mask: mask, injectedEmbeddings: injectedEmbeddings,
        tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
        textModels: existingTextModels)
    case .qwenImage:
      return encodeQwen(
        images: images,
        tokens: tokens, positions: positions, mask: mask, injectedEmbeddings: injectedEmbeddings,
        tokenLengthUncond: &tokenLengthUncond, tokenLengthCond: &tokenLengthCond,
        modifier: modifier, textModels: existingTextModels)
    case .wurstchenStageC, .wurstchenStageB:
      return encodeWurstchen(
        tokens: tokens, positions: positions, mask: mask, injectedEmbeddings: injectedEmbeddings,
        lengthsOfUncond: lengthsOfUncond, lengthsOfCond: lengthsOfCond,
        textModels: existingTextModels)
    }
    var causalAttentionMask = Tensor<FloatType>(
      Array(repeating: 0, count: 2 * maxLength * maxLength), .CPU, .NHWC(2, 1, maxLength, maxLength)
    )
    for i in 0..<(maxLength - 1) {
      for j in (i + 1)..<maxLength {
        causalAttentionMask[0, 0, i, j] = -FloatType.greatestFiniteMagnitude
        causalAttentionMask[1, 0, i, j] = -FloatType.greatestFiniteMagnitude
      }
    }
    var j = 0
    var prefixLength = 0
    for i in 0..<maxLength {
      // Mask out anything before this, except padding / ending.
      guard j < lengthsOfUncond.count else { break }
      if i - 1 >= lengthsOfUncond[j] + prefixLength {
        prefixLength += lengthsOfUncond[j]
        j += 1
      }
      if prefixLength > 0 && j < lengthsOfUncond.count {
        for k in 1..<(prefixLength + 1) {
          causalAttentionMask[0, 0, i, k] = -FloatType.greatestFiniteMagnitude
        }
      }
    }
    j = 0
    prefixLength = 0
    for i in 0..<maxLength {
      // Mask out anything before this, except padding / ending.
      guard j < lengthsOfCond.count else { break }
      if i - 1 >= lengthsOfCond[j] + prefixLength {
        prefixLength += lengthsOfCond[j]
        j += 1
      }
      if prefixLength > 0 && j < lengthsOfCond.count {
        for k in 1..<(prefixLength + 1) {
          causalAttentionMask[1, 0, i, k] = -FloatType.greatestFiniteMagnitude
        }
      }
    }
    let graph = tokens[0].graph
    let tokensTensorGPU = tokens[0].toGPU(0)
    let positionTensorGPU = positions[0].toGPU(0)
    let causalAttentionMaskGPU = graph.variable(causalAttentionMask.toGPU())
    let maskGPU = mask.map { $0.toGPU(0) }
    let injectedEmbeddingsGPU = injectedEmbeddings.map { $0.toGPU(0) }
    let textModel: Model
    let externalData: DynamicGraph.Store.Codec =
      externalOnDemand
      ? .externalOnDemand : .externalData(deviceProperties.isFreadPreferred ? .fread : .mmap)
    if let existingTextModel = existingTextModels[0] {
      textModel = existingTextModel
    } else {
      switch version {
      case .v1:
        textModel =
          CLIPTextModel(
            FloatType.self, injectEmbeddings: injectEmbeddings,
            vocabularySize: 49408, maxLength: 77, maxTokenLength: maxLength, embeddingSize: 768,
            numLayers: 13 - min(max(clipSkip, 1), 12), numHeads: 12, batchSize: 2,
            intermediateSize: 3072, usesFlashAttention: usesFlashAttention
          ).0
      case .v2:
        textModel =
          OpenCLIPTextModel(
            FloatType.self, injectEmbeddings: injectEmbeddings,
            vocabularySize: 49408, maxLength: 77, maxTokenLength: maxLength, embeddingSize: 1024,
            numLayers: 24 - min(max(clipSkip, 1), 23), numHeads: 16, batchSize: 2,
            intermediateSize: 4096, usesFlashAttention: usesFlashAttention
          ).0
      case .sd3, .sd3Large, .pixart, .auraflow, .flux1, .kandinsky21, .sdxlBase, .sdxlRefiner,
        .ssd1b, .svdI2v, .wurstchenStageC, .wurstchenStageB, .hunyuanVideo, .wan21_1_3b, .wan21_14b,
        .hiDreamI1, .qwenImage, .wan22_5b, .zImage, .flux2, .flux2_9b, .flux2_4b, .ltx2:
        fatalError()
      }
      if let maskGPU = maskGPU.first, let injectedEmbeddingsGPU = injectedEmbeddingsGPU.first {
        textModel.compile(
          inputs: tokensTensorGPU, positionTensorGPU, causalAttentionMaskGPU, maskGPU,
          injectedEmbeddingsGPU)
      } else {
        textModel.compile(inputs: tokensTensorGPU, positionTensorGPU, causalAttentionMaskGPU)
      }
      graph.openStore(
        filePaths[0], flags: .readOnly,
        externalStore: TensorData.externalStore(filePath: filePaths[0])
      ) { store in
        if lora.count > 0 {
          LoRALoader.openStore(graph, lora: lora) { loader in
            if clipSkip > 1 {
              store.read(
                "text_model", model: textModel, codec: [.jit, .q6p, .q8p, .ezm7, externalData]
              ) { name, dataType, _, shape in
                // Retrieve the right final layer norm parameters.
                var name = name
                switch version {
                case .v1:
                  if name == "__text_model__[t-\(98 - (min(clipSkip, 12) - 1) * 8)-0]" {
                    name = "__text_model__[t-98-0]"
                  } else if name == "__text_model__[t-\(98 - (min(clipSkip, 12) - 1) * 8)-1]" {
                    name = "__text_model__[t-98-1]"
                  }
                case .v2:
                  if name == "__text_model__[t-\(186 - (min(clipSkip, 23) - 1) * 8)-0]" {
                    name = "__text_model__[t-186-0]"
                  } else if name == "__text_model__[t-\(186 - (min(clipSkip, 23) - 1) * 8)-1]" {
                    name = "__text_model__[t-186-1]"
                  }
                case .sd3, .sd3Large, .pixart, .auraflow, .flux1, .kandinsky21, .sdxlBase,
                  .sdxlRefiner, .ssd1b, .svdI2v, .wurstchenStageC, .wurstchenStageB, .hunyuanVideo,
                  .wan21_1_3b, .wan21_14b, .hiDreamI1, .qwenImage, .wan22_5b, .zImage, .flux2,
                  .flux2_9b, .flux2_4b, .ltx2:
                  fatalError()
                }
                return loader.mergeLoRA(
                  graph, name: name, store: store, dataType: dataType, shape: shape,
                  of: FloatType.self)
              }
            } else {
              store.read(
                "text_model", model: textModel, codec: [.jit, .q6p, .q8p, .ezm7, externalData]
              ) { name, dataType, _, shape in
                return loader.mergeLoRA(
                  graph, name: name, store: store, dataType: dataType, shape: shape,
                  of: FloatType.self)
              }
            }
          }
        } else {
          if clipSkip > 1 {
            store.read(
              "text_model", model: textModel, codec: [.jit, .q6p, .q8p, .ezm7, externalData]
            ) { name, _, _, _ in
              // Retrieve the right final layer norm parameters.
              var name = name
              switch version {
              case .v1:
                if name == "__text_model__[t-\(98 - (min(clipSkip, 12) - 1) * 8)-0]" {
                  name = "__text_model__[t-98-0]"
                } else if name == "__text_model__[t-\(98 - (min(clipSkip, 12) - 1) * 8)-1]" {
                  name = "__text_model__[t-98-1]"
                }
              case .v2:
                if name == "__text_model__[t-\(186 - (min(clipSkip, 23) - 1) * 8)-0]" {
                  name = "__text_model__[t-186-0]"
                } else if name == "__text_model__[t-\(186 - (min(clipSkip, 23) - 1) * 8)-1]" {
                  name = "__text_model__[t-186-1]"
                }
              case .sd3, .sd3Large, .pixart, .auraflow, .flux1, .kandinsky21, .sdxlBase,
                .sdxlRefiner, .ssd1b, .svdI2v, .wurstchenStageC, .wurstchenStageB, .hunyuanVideo,
                .wan21_1_3b, .wan21_14b, .hiDreamI1, .qwenImage, .wan22_5b, .zImage, .flux2,
                .flux2_9b, .flux2_4b, .ltx2:
                fatalError()
              }
              return .continue(name)
            }
          } else {
            store.read(
              "text_model", model: textModel, codec: [.jit, .q6p, .q8p, .ezm7, externalData])
          }
        }
      }
    }
    if let maskGPU = maskGPU.first, let injectedEmbeddingsGPU = injectedEmbeddingsGPU.first {
      return (
        [
          textModel(
            inputs: tokensTensorGPU, positionTensorGPU, causalAttentionMaskGPU, maskGPU,
            injectedEmbeddingsGPU)[0].as(
              of: FloatType.self
            ).reshaped(.HWC(2, maxLength, conditionalLength))
        ], [textModel]
      )
    } else {
      return (
        [
          textModel(
            inputs: tokensTensorGPU, positionTensorGPU, causalAttentionMaskGPU)[0].as(
              of: FloatType.self
            ).reshaped(.HWC(2, maxLength, conditionalLength))
        ], [textModel]
      )
    }
  }
}
