import Collections
import NNC
import WeightsCache

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
    image: [DynamicGraph.Tensor<FloatType>], textModels existingTextModels: [Model?]
  ) -> ([DynamicGraph.Tensor<FloatType>], [Model]) {
    let graph = image[0].graph
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
    var input = image[0]
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
          modifier: $0.modifier)
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
    image: [DynamicGraph.Tensor<FloatType>],
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
    guard var input = image.first, filePaths.count >= 2 else {
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

  public func encode(
    tokenLengthUncond: Int, tokenLengthCond: Int,
    tokens: [DynamicGraph.Tensor<Int32>], positions: [DynamicGraph.Tensor<Int32>],
    mask: [DynamicGraph.Tensor<FloatType>], injectedEmbeddings: [DynamicGraph.Tensor<FloatType>],
    image: [DynamicGraph.Tensor<FloatType>], lengthsOfUncond: [Int], lengthsOfCond: [Int],
    injectedTextEmbeddings: [(
      model: ControlModel<FloatType>, hints: [([DynamicGraph.Tensor<FloatType>], Float)]
    )],
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
      return encodeI2v(image: image, textModels: existingTextModels)
    case .hunyuanVideo:
      return encodeHunyuan(
        tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
        tokens: tokens, positions: positions, mask: mask, injectedEmbeddings: injectedEmbeddings,
        lengthsOfUncond: lengthsOfUncond, lengthsOfCond: lengthsOfCond,
        injectedTextEmbeddings: injectedTextEmbeddings, textModels: existingTextModels)
    case .wan21_1_3b, .wan21_14b:
      return encodeWan(
        image: image,
        tokens: tokens, positions: positions, mask: mask, injectedEmbeddings: injectedEmbeddings,
        tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
        textModels: existingTextModels)
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
        .hiDreamI1:
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
                  .wan21_1_3b, .wan21_14b, .hiDreamI1:
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
                .wan21_1_3b, .wan21_14b, .hiDreamI1:
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
