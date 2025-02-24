import Collections
import NNC

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
  public init(
    filePaths: [String], version: ModelVersion, textEncoderVersion: TextEncoderVersion?,
    isCfgEnabled: Bool, usesFlashAttention: Bool, injectEmbeddings: Bool, externalOnDemand: Bool,
    maxLength: Int = 77, clipSkip: Int = 1, lora: [LoRAConfiguration] = []
  ) {
    self.filePaths = filePaths
    self.version = version
    self.textEncoderVersion = textEncoderVersion
    self.isCfgEnabled = isCfgEnabled
    self.usesFlashAttention = usesFlashAttention
    self.injectEmbeddings = injectEmbeddings
    self.externalOnDemand = externalOnDemand
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
      externalOnDemand ? .externalOnDemand : .externalData
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
        LoRALoader<FloatType>.openStore(graph, lora: lora) { loader in
          if clipSkip > 1 {
            store.read(
              "text_model", model: textModel, codec: [.jit, .q6p, .q8p, .ezm7, externalData]
            ) { name, _, _, shape in
              // Retrieve the right final layer norm parameters.
              var name = name
              if name == "__text_model__[t-\(98 - (min(clipSkip, 12) - 1) * 8)-0]" {
                name = "__text_model__[t-98-0]"
              } else if name == "__text_model__[t-\(98 - (min(clipSkip, 12) - 1) * 8)-1]" {
                name = "__text_model__[t-98-1]"
              }
              return loader.mergeLoRA(graph, name: name, store: store, shape: shape)
            }
          } else {
            store.read(
              "text_model", model: textModel, codec: [.jit, .q6p, .q8p, .ezm7, externalData]
            ) { name, _, _, shape in
              return loader.mergeLoRA(graph, name: name, store: store, shape: shape)
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
      externalOnDemand ? .externalOnDemand : .externalData
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
          LoRALoader<FloatType>.openStore(graph, lora: lora) { loader in
            store.read(
              "text_model", model: textModel, codec: [.jit, .q6p, .q8p, .ezm7, externalData]
            ) { name, _, _, shape in
              var name = name
              if name == "__text_model__[t-\(98 - (min(clipSkip, 12) - 1) * 8)-0]" {
                name = "__text_model__[t-98-0]"
              } else if name == "__text_model__[t-\(98 - (min(clipSkip, 12) - 1) * 8)-1]" {
                name = "__text_model__[t-98-1]"
              }
              return loader.mergeLoRA(graph, name: name, store: store, shape: shape)
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
        LoRALoader<FloatType>.openStore(graph, lora: lora) { loader in
          store.read(
            "text_model", model: textModel, codec: [.jit, .q6p, .q8p, .ezm7, externalData]
          ) { name, _, _, shape in
            // Retrieve the right final layer norm parameters.
            var name = name
            if name == "__text_model__[t-\(258 - (min(max(clipSkip - 1, 1), 31) - 1) * 8)-0]" {
              name = "__text_model__[t-258-0]"
            } else if name == "__text_model__[t-\(258 - (min(max(clipSkip - 1, 1), 31) - 1) * 8)-1]"
            {
              name = "__text_model__[t-258-1]"
            }
            return loader.mergeLoRA(graph, name: name, store: store, shape: shape, prefix: "__te2")
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
        "text_projection", variable: textProjection, codec: [.q6p, .q8p, .ezm7, .externalData])
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
      externalOnDemand ? .externalOnDemand : .externalData
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
        $0.read("visual_proj", model: visualProj, codec: [.jit, .q6p, .q8p, .ezm7, .externalData])
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
      externalOnDemand ? .externalOnDemand : .externalData
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
        LoRALoader<FloatType>.openStore(graph, lora: lora) { loader in
          store.read(
            "text_model", model: textModel, codec: [.jit, .q6p, .q8p, .ezm7, externalData]
          ) { name, _, _, shape in
            // Retrieve the right final layer norm parameters.
            var name = name
            if name == "__text_model__[t-\(258 - (min(max(clipSkip - 1, 1), 31) - 1) * 8)-0]" {
              name = "__text_model__[t-258-0]"
            } else if name == "__text_model__[t-\(258 - (min(max(clipSkip - 1, 1), 31) - 1) * 8)-1]"
            {
              name = "__text_model__[t-258-1]"
            }
            return loader.mergeLoRA(graph, name: name, store: store, shape: shape, prefix: "__te2")
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
        "text_projection", variable: textProjection, codec: [.q6p, .q8p, .ezm7, .externalData])
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
      externalOnDemand ? .externalOnDemand : .externalData
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
          LoRALoader<FloatType>.openStore(graph, lora: lora) { loader in
            store.read(
              "text_model", model: textModel, codec: [.jit, .q6p, .q8p, .ezm7, externalData]
            ) { name, _, _, shape in
              var name = name
              if name == "__text_model__[t-\(98 - (min(max(clipSkip - 1, 1), 12) - 1) * 8)-0]" {
                name = "__text_model__[t-98-0]"
              } else if name
                == "__text_model__[t-\(98 - (min(max(clipSkip - 1, 1), 12) - 1) * 8)-1]"
              {
                name = "__text_model__[t-98-1]"
              }
              return loader.mergeLoRA(graph, name: name, store: store, shape: shape)
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
        LoRALoader<FloatType>.openStore(graph, lora: lora) { loader in
          store.read(
            "text_model", model: textModel, codec: [.jit, .q6p, .q8p, .ezm7, externalData]
          ) { name, _, _, shape in
            // Retrieve the right final layer norm parameters.
            var name = name
            if name == "__text_model__[t-\(258 - (min(max(clipSkip - 1, 1), 31) - 1) * 8)-0]" {
              name = "__text_model__[t-258-0]"
            } else if name == "__text_model__[t-\(258 - (min(max(clipSkip - 1, 1), 31) - 1) * 8)-1]"
            {
              name = "__text_model__[t-258-1]"
            }
            return loader.mergeLoRA(graph, name: name, store: store, shape: shape, prefix: "__te2")
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
        "text_projection", variable: textProjection, codec: [.q6p, .q8p, .ezm7, .externalData])
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
    let (_, t5) = T5ForConditionalGeneration(b: 2, t: tokenLength, of: FloatType.self)
    let relativePositionBuckets = relativePositionBuckets(
      sequenceLength: tokenLength, numBuckets: 32, maxDistance: 128)
    let tokens2TensorGPU = tokens[2].toGPU(0)
    let relativePositionBucketsGPU = graph.variable(relativePositionBuckets.toGPU(0))
    t5.compile(inputs: tokens2TensorGPU, relativePositionBucketsGPU)
    // Move T5 to on-demand.
    TensorData.makeExternalData(for: filePaths[2], graph: graph)
    graph.openStore(
      filePaths[2], flags: .readOnly,
      externalStore: TensorData.externalStore(filePath: filePaths[2])
    ) {
      $0.read("text_model", model: t5, codec: [.q8p, .q6p, .q4p, .ezm7, .jit, .externalOnDemand])
    }
    let c2 = t5(inputs: tokens2TensorGPU, relativePositionBucketsGPU)[0].as(
      of: FloatType.self
    ).reshaped(.HWC(2, tokenLength, 4096))
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
    let (rankOfLoRA, filesRequireMerge) = LoRALoader<FloatType>.rank(
      graph, of: lora.map { $0.file }, prefix: "__text_model__")
    let configuration = LoRANetworkConfiguration(rank: rankOfLoRA, scale: 1, highPrecision: false)
    let textModel: Model
    if !lora.isEmpty && rankOfLoRA > 0 {
      (_, textModel) = LoRAT5ForConditionalGeneration(
        b: 2, t: tokenLength, LoRAConfiguration: configuration, of: FloatType.self)
    } else {
      (_, textModel) = T5ForConditionalGeneration(b: 2, t: tokenLength, of: FloatType.self)
    }
    let relativePositionBuckets = relativePositionBuckets(
      sequenceLength: tokenLength, numBuckets: 32, maxDistance: 128)
    let tokensTensorGPU = tokens[0].toGPU(0)
    let relativePositionBucketsGPU = graph.variable(relativePositionBuckets.toGPU(0))
    textModel.compile(inputs: tokensTensorGPU, relativePositionBucketsGPU)
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
        LoRALoader<FloatType>.openStore(graph, lora: lora) { loader in
          store.read(
            "text_model", model: textModel, codec: [.jit, .q6p, .q8p, .ezm7, .externalOnDemand]
          ) {
            name, dataType, format, shape in
            return loader.concatenateLoRA(
              graph, LoRAMapping: mapping, filesRequireMerge: filesRequireMerge, name: name,
              store: store, dataType: dataType, format: format, shape: shape)
          }
        }
      } else {
        store.read(
          "text_model", model: textModel, codec: [.q8p, .q6p, .q4p, .ezm7, .jit, .externalOnDemand])
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
    let (_, textModel) = UMT5ForConditionalGeneration(b: 2, t: tokenLength, of: FloatType.self)
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
    // Move Pile T5 XL to on-demand.
    TensorData.makeExternalData(for: filePaths[0], graph: graph)
    graph.openStore(
      filePaths[0], flags: .readOnly,
      externalStore: TensorData.externalStore(filePath: filePaths[0])
    ) { store in
      store.read(
        "text_model", model: textModel, codec: [.q8p, .q6p, .q4p, .ezm7, .jit, .externalOnDemand])
    }
    var c = textModel(inputs: tokensTensorGPU, attentionMaskGPU, relativePositionBucketsGPU)[0].as(
      of: FloatType.self
    ).reshaped(.HWC(2, tokenLength, 2048))
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
    // Move ChatGLM3 to on-demand.
    TensorData.makeExternalData(for: filePaths[0], graph: graph)
    graph.openStore(
      filePaths[0], flags: .readOnly,
      externalStore: TensorData.externalStore(filePath: filePaths[0])
    ) { store in
      store.read(
        "text_model", model: textModel, codec: [.q8p, .q6p, .q4p, .ezm7, .jit, .externalOnDemand])
    }
    let c = textModel(
      inputs: rightAlignedTokensTensorGPU, rightAlignedRotaryEmbeddingGPU, causalAttentionMaskGPU
    ).map {
      $0.as(
        of: FloatType.self
      )
    }
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
      externalOnDemand ? .externalOnDemand : .externalData
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
          LoRALoader<FloatType>.openStore(graph, lora: lora) { loader in
            store.read(
              "text_model", model: textModel, codec: [.jit, .q6p, .q8p, .ezm7, externalData]
            ) { name, _, _, shape in
              var name = name
              if name == "__text_model__[t-\(98 - (min(max(clipSkip - 1, 1), 12) - 1) * 8)-0]" {
                name = "__text_model__[t-98-0]"
              } else if name
                == "__text_model__[t-\(98 - (min(max(clipSkip - 1, 1), 12) - 1) * 8)-1]"
              {
                name = "__text_model__[t-98-1]"
              }
              return loader.mergeLoRA(graph, name: name, store: store, shape: shape)
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
    let (_, t5) = T5ForConditionalGeneration(b: batchSize, t: tokenLength, of: FloatType.self)
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
    // Move T5 to on-demand.
    TensorData.makeExternalData(for: filePaths[0], graph: graph)
    graph.openStore(
      filePaths[0], flags: .readOnly,
      externalStore: TensorData.externalStore(filePath: filePaths[0])
    ) {
      $0.read("text_model", model: t5, codec: [.q8p, .q6p, .q4p, .ezm7, .jit, .externalOnDemand])
    }
    let c2 = t5(inputs: tokens2TensorGPU, relativePositionBucketsGPU)[0].as(
      of: FloatType.self
    ).reshaped(.HWC(batchSize, tokenLength, 4096))
    return ([c2, pooled], [textModel])
  }

  private func encodeLlama3(
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
      externalOnDemand ? .externalOnDemand : .externalData
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
          LoRALoader<FloatType>.openStore(graph, lora: lora) { loader in
            store.read(
              "text_model", model: textModel, codec: [.jit, .q6p, .q8p, .ezm7, externalData]
            ) { name, _, _, shape in
              var name = name
              if name == "__text_model__[t-\(98 - (min(max(clipSkip - 1, 1), 12) - 1) * 8)-0]" {
                name = "__text_model__[t-98-0]"
              } else if name
                == "__text_model__[t-\(98 - (min(max(clipSkip - 1, 1), 12) - 1) * 8)-1]"
              {
                name = "__text_model__[t-98-1]"
              }
              return loader.mergeLoRA(graph, name: name, store: store, shape: shape)
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
        guard !isCfgEnabled else {
          return $0
        }
        let shape = $0.shape
        return $0[1..<2, 0..<shape[1], 0..<shape[2]]
      }
    let additionalTokenLength = tokenLength + (injectedTextEmbedding?.shape[1] ?? 0)
    let llama3 = Llama3(
      FloatType.self, vocabularySize: 128_320, width: 4_096,
      tokenLength: (tokenLength, additionalTokenLength), layers: 32, MLP: 14336, heads: 32,
      outputHiddenStates: 29,
      batchSize: batchSize)
    let tokens2TensorGPU: DynamicGraph.Tensor<Int32>
    if isCfgEnabled {
      tokens2TensorGPU = tokens[0].toGPU(0)
    } else {
      tokens2TensorGPU = tokens[0][tokenLength..<(tokenLength * 2)].toGPU(0)
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
      Llama3RotaryEmbedding(sequenceLength: additionalTokenLength, of: FloatType.self).toGPU(0))
    let causalAttentionMaskLlama3GPU = graph.variable(causalAttentionMaskLlama3.toGPU(0))
    llama3.compile(
      inputs: [tokens2TensorGPU, rotaryTensorGPU, causalAttentionMaskLlama3GPU]
        + (injectedTextEmbedding.flatMap { [$0] } ?? []))
    // Move Llama3 8B to on-demand.
    TensorData.makeExternalData(for: filePaths[0], graph: graph)
    graph.openStore(
      filePaths[0], flags: .readOnly,
      externalStore: TensorData.externalStore(filePath: filePaths[0])
    ) {
      $0.read("llava", model: llama3, codec: [.q8p, .q6p, .q4p, .ezm7, .jit, .externalOnDemand])
    }
    let c2 = llama3(
      inputs: tokens2TensorGPU,
      [rotaryTensorGPU, causalAttentionMaskLlama3GPU]
        + (injectedTextEmbedding.flatMap { [$0] } ?? []))[0].as(
        of: FloatType.self
      ).reshaped(.HWC(batchSize, additionalTokenLength, 4096))[
        0..<batchSize, 95..<additionalTokenLength, 0..<4096
      ].copied()
    return ([c2, pooled], [textModel])
  }

  public func encode(
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
      return encodeLlama3(
        tokens: tokens, positions: positions, mask: mask, injectedEmbeddings: injectedEmbeddings,
        lengthsOfUncond: lengthsOfUncond, lengthsOfCond: lengthsOfCond,
        injectedTextEmbeddings: injectedTextEmbeddings, textModels: existingTextModels)
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
      externalOnDemand ? .externalOnDemand : .externalData
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
        .ssd1b, .svdI2v, .wurstchenStageC, .wurstchenStageB, .hunyuanVideo:
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
          LoRALoader<FloatType>.openStore(graph, lora: lora) { loader in
            if clipSkip > 1 {
              store.read(
                "text_model", model: textModel, codec: [.jit, .q6p, .q8p, .ezm7, externalData]
              ) { name, _, _, shape in
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
                  .sdxlRefiner, .ssd1b, .svdI2v, .wurstchenStageC, .wurstchenStageB, .hunyuanVideo:
                  fatalError()
                }
                return loader.mergeLoRA(graph, name: name, store: store, shape: shape)
              }
            } else {
              store.read(
                "text_model", model: textModel, codec: [.jit, .q6p, .q8p, .ezm7, externalData]
              ) { name, _, _, shape in
                return loader.mergeLoRA(graph, name: name, store: store, shape: shape)
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
                .sdxlRefiner, .ssd1b, .svdI2v, .wurstchenStageC, .wurstchenStageB, .hunyuanVideo:
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
