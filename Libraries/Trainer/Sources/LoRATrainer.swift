import DataModels
import Diffusion
import Foundation
import ModelZoo
import NNC
import SFMT
import TensorBoard

public struct LoRATrainer {

  public enum WeightsMemoryManagement: Int {
    case cached = 0
    case justInTime
  }

  public enum MemorySaver: Int {
    case minimal = 0
    case balanced
    case speed
    case turbo
  }

  public struct Input {
    public var identifier: String
    public var imageUrl: URL
    public var caption: String
    public init(identifier: String, imageUrl: URL, caption: String) {
      self.identifier = identifier
      self.imageUrl = imageUrl
      self.caption = caption
    }
  }

  public struct ProcessedInput {
    var imagePath: String
    var tokens: [Int32]
    var CLIPTokens: [Int32]
    var originalSize: (width: Int, height: Int)
    var cropTopLeft: (top: Int, left: Int)
    var targetSize: (width: Int, height: Int)
    var loadedImagePath: String
  }

  public let version: ModelVersion
  public let textEncoderVersion: TextEncoderVersion?
  public let session: String
  public let summaryWriter: SummaryWriter?
  private let model: String
  private let paddedTextEncodingLength: Int
  private let scale: DeviceCapability.Scale
  private let additionalScales: [DeviceCapability.Scale]
  private let useImageAspectRatio: Bool
  private let cotrainTextModel: Bool
  private let cotrainCustomEmbedding: Bool
  private let clipSkip: Int
  private let autoencoder: String
  private let textEncoder: String
  private let CLIPEncoder: String?
  private let resumeIfPossible: Bool
  private let imageInspector: (URL) -> (width: Int, height: Int)?
  private let imageLoader:
    (URL, Int, Int) -> (
      Tensor<FloatType>, (width: Int, height: Int), (top: Int, left: Int), (width: Int, height: Int)
    )?

  public init(
    tensorBoard: String?, comment: String,
    model: String, scale: DeviceCapability.Scale, additionalScales: [DeviceCapability.Scale],
    useImageAspectRatio: Bool, cotrainTextModel: Bool,
    cotrainCustomEmbedding: Bool, clipSkip: Int, maxTextLength: Int, session: String,
    resumeIfPossible: Bool,
    imageInspector: @escaping (URL) -> (width: Int, height: Int)?,
    imageLoader: @escaping (URL, Int, Int) -> (
      Tensor<FloatType>, (width: Int, height: Int), (top: Int, left: Int), (width: Int, height: Int)
    )?
  ) {  // The model identifier.
    summaryWriter = tensorBoard.map { SummaryWriter(logDirectory: $0, comment: comment) }
    self.model = model
    var scale = scale
    for additionalScale in additionalScales {
      if additionalScale.widthScale * additionalScale.heightScale > scale.widthScale
        * scale.heightScale
      {
        scale = additionalScale
      }
    }
    self.scale = scale
    self.additionalScales = additionalScales.filter {  // Only retains the scale that is smaller than the scale we selected.
      return $0.widthScale * $0.heightScale < scale.widthScale * scale.heightScale
    }
    self.useImageAspectRatio = useImageAspectRatio
    let version = ModelZoo.versionForModel(model)
    let textEncoderVersion = ModelZoo.textEncoderVersionForModel(model)
    // Cannot co-train a model if text encoder is available.
    self.cotrainTextModel = textEncoderVersion == nil ? cotrainTextModel : false
    self.cotrainCustomEmbedding = textEncoderVersion == nil ? cotrainCustomEmbedding : false
    self.clipSkip = clipSkip
    self.imageInspector = imageInspector
    self.imageLoader = imageLoader
    autoencoder =
      ModelZoo.autoencoderForModel(model)
      ?? ({
        switch version {
        case .v1, .v2, .svdI2v:
          return "vae_ft_mse_840000_f16.ckpt"
        case .sdxlBase, .sdxlRefiner, .ssd1b, .pixart, .auraflow:
          return "sdxl_vae_v1.0_f16.ckpt"
        case .kandinsky21:
          return "kandinsky_movq_f16.ckpt"
        case .wurstchenStageC, .wurstchenStageB:
          return "wurstchen_3.0_stage_a_hq_f16.ckpt"
        case .sd3, .sd3Large:
          return "sd3_vae_f16.ckpt"
        case .flux1, .hiDreamI1:
          return "flux_1_vae_f16.ckpt"
        case .hunyuanVideo:
          return "hunyuan_video_vae_f16.ckpt"
        case .wan21_1_3b, .wan21_14b:
          fatalError()
        }
      })()
    textEncoder =
      ModelZoo.textEncoderForModel(model)
      ?? ({
        switch version {
        case .v1:
          return "clip_vit_l14_f16.ckpt"
        case .v2:
          return "open_clip_vit_h14_f16.ckpt"
        case .svdI2v:
          fatalError()
        case .kandinsky21:
          return "xlm_roberta_f16.ckpt"
        case .pixart, .flux1:
          return "t5_xxl_encoder_q6p.ckpt"
        case .auraflow:
          return "pile_t5_xl_encoder_q8p.ckpt"
        case .sd3, .sd3Large, .sdxlBase, .sdxlRefiner, .ssd1b, .wurstchenStageC, .wurstchenStageB:
          return "open_clip_vit_bigg14_f16.ckpt"
        case .hunyuanVideo:
          return "llava_llama_3_8b_v1.1_q8p.ckpt"
        case .wan21_1_3b, .wan21_14b:
          fatalError()
        case .hiDreamI1:
          fatalError()
        }
      })()
    CLIPEncoder = ModelZoo.CLIPEncoderForModel(model)
    paddedTextEncodingLength = min(maxTextLength, ModelZoo.paddedTextEncodingLengthForModel(model))
    self.version = version
    self.textEncoderVersion = textEncoderVersion
    self.session = session
    self.resumeIfPossible = resumeIfPossible
  }

  public enum PrepareState {
    case imageEncoding
    case conditionalEncoding
  }

  private static func sizeThatFits(size: (width: Int, height: Int), scale: DeviceCapability.Scale)
    -> (width: Int, height: Int)
  {
    let aspectRatio = Double(size.width) / Double(size.height)
    // Set width / height to around 1.5M pixels.
    let imageWidth = (Double(scale.widthScale) * Double(scale.heightScale) * aspectRatio)
      .squareRoot()
    let imageHeight = (Double(scale.widthScale) * Double(scale.heightScale) / aspectRatio)
      .squareRoot()
    let roundedWidth = Int(imageWidth.rounded())
    let roundedHeight = Int(imageHeight.rounded())
    // If the width * height still fits into scale, we can return now.
    if roundedWidth * roundedHeight <= Int(scale.widthScale) * Int(scale.heightScale) {
      return (width: roundedWidth, height: roundedHeight)
    }
    let roundedDownWidth = Int(imageWidth.rounded(.down))
    let roundedDownHeight = Int(imageHeight.rounded(.down))
    // Otherwise, see which one is closer to the rounded value, and see use rounded down for the other would meet the goal.
    if abs(imageWidth - Double(roundedWidth)) < abs(imageHeight - Double(roundedHeight)) {
      if roundedWidth * roundedDownHeight <= Int(scale.widthScale) * Int(scale.heightScale) {
        return (width: roundedWidth, height: roundedDownHeight)
      }
    } else {
      if roundedDownWidth * roundedHeight <= Int(scale.widthScale) * Int(scale.heightScale) {
        return (width: roundedDownWidth, height: roundedHeight)
      }
    }
    return (width: roundedDownWidth, height: roundedDownHeight)  // This is guranteed to be smaller than scale.width * scale.height.
  }

  public func prepareDatasetForFlux1(
    inputs: [Input], tokenizers: [TextualInversionPoweredTokenizer & Tokenizer],
    customEmbeddingLength: Int, progressHandler: (PrepareState, Int) -> Bool
  ) -> (DataFrame, ProcessedInput)? {
    let latentsScaling = ModelZoo.latentsScalingForModel(model)
    let firstStage = FirstStage<FloatType>(
      filePath: ModelZoo.filePathForModelDownloaded(autoencoder), version: version,
      latentsScaling: latentsScaling, highPrecisionKeysAndValues: false,
      highPrecisionFallback: true,
      tiledDecoding: TiledConfiguration(
        isEnabled: false, tileSize: TiledConfiguration.Size(width: 0, height: 0),
        tileOverlap: 0),
      tiledDiffusion: TiledConfiguration(
        isEnabled: false, tileSize: TiledConfiguration.Size(width: 0, height: 0),
        tileOverlap: 0),
      externalOnDemand: false, alternativeUsesFlashAttention: false, alternativeFilePath: nil,
      alternativeDecoderVersion: nil, memoryCapacity: DeviceCapability.memoryCapacity,
      isNHWCPreferred: DeviceCapability.isNHWCPreferred)
    let graph = DynamicGraph()
    var processedInputs = [ProcessedInput]()
    let (_, zeroCLIPTokens, _, _, _) = tokenizers[0].tokenize(
      text: "", truncation: true, maxLength: 77)
    let (_, zeroTokens, _, _, _) = tokenizers[1].tokenize(
      text: "", truncation: true, maxLength: paddedTextEncodingLength)
    let zeroCaptionInput = ProcessedInput(
      imagePath: "", tokens: zeroTokens, CLIPTokens: zeroCLIPTokens, originalSize: (0, 0),
      cropTopLeft: (0, 0), targetSize: (0, 0), loadedImagePath: "")
    var stopped = false
    graph.openStore(session) { store in
      // First, center crop and encode the image.
      graph.withNoGrad {
        var previousImageWidth: Int? = nil
        var previousImageHeight: Int? = nil
        var encoder: Model? = nil
        for input in inputs {
          guard let originalSize = imageInspector(input.imageUrl),
            originalSize.width > 0 && originalSize.height > 0
          else { continue }
          let imageWidth: Int
          let imageHeight: Int
          if useImageAspectRatio {
            let imageSize = Self.sizeThatFits(size: originalSize, scale: scale)
            imageWidth = imageSize.width * 64
            imageHeight = imageSize.height * 64
          } else {
            imageWidth = Int(scale.widthScale) * 64
            imageHeight = Int(scale.heightScale) * 64
          }
          guard
            let (tensor, originalSize, cropTopLeft, targetSize) = imageLoader(
              input.imageUrl, imageWidth, imageHeight)
          else { continue }
          if previousImageWidth != imageWidth || previousImageHeight != imageHeight {
            encoder = nil  // Reload encoder.
          }
          previousImageWidth = imageWidth
          previousImageHeight = imageHeight
          // Keep this in the database.
          let tuple = firstStage.encode(
            graph.constant(tensor.toGPU(0)), encoder: encoder, cancellation: { _ in })
          let sample = tuple.0.rawValue.toCPU()
          encoder = tuple.1
          let imagePath = input.imageUrl.path
          store.write(imagePath, tensor: sample)
          let (_, CLIPTokens, _, _, _) = tokenizers[0].tokenize(
            text: input.caption, truncation: true, maxLength: 77)
          let (_, tokens, _, _, _) = tokenizers[1].tokenize(
            text: input.caption, truncation: true, maxLength: paddedTextEncodingLength)
          // No embedding support.
          processedInputs.append(
            ProcessedInput(
              imagePath: imagePath, tokens: tokens, CLIPTokens: CLIPTokens,
              originalSize: originalSize, cropTopLeft: cropTopLeft, targetSize: targetSize,
              loadedImagePath: imagePath))
          guard progressHandler(.imageEncoding, processedInputs.count) else {
            stopped = true
            break
          }
        }
        guard !stopped else { return }
        let imageWidth = Int(scale.widthScale) * 64
        let imageHeight = Int(scale.heightScale) * 64
        let zeros = graph.variable(
          .GPU(0), .NHWC(1, imageHeight, imageWidth, 3), of: FloatType.self)
        zeros.full(0)
        if previousImageWidth != imageWidth || previousImageHeight != imageHeight {
          encoder = nil  // Reload encoder.
        }
        let latentZeros = firstStage.sample(zeros, encoder: encoder, cancellation: { _ in }).1
          .copied().rawValue.toCPU()
        store.write("latent_zeros", tensor: latentZeros)
        if useImageAspectRatio && !additionalScales.isEmpty {
          var inputMap = [String: ProcessedInput]()
          for input in processedInputs {
            inputMap[input.imagePath] = input
          }
          for (i, scale) in additionalScales.enumerated() {
            for input in inputs {
              guard let originalSize = imageInspector(input.imageUrl),
                originalSize.width > 0 && originalSize.height > 0
              else { continue }
              let imageSize = Self.sizeThatFits(size: originalSize, scale: scale)
              let imageWidth = imageSize.width * 64
              let imageHeight = imageSize.height * 64
              guard
                let (tensor, _, _, _) = imageLoader(
                  input.imageUrl, imageWidth, imageHeight)
              else { continue }
              if previousImageWidth != imageWidth || previousImageHeight != imageHeight {
                encoder = nil  // Reload encoder.
              }
              previousImageWidth = imageWidth
              previousImageHeight = imageHeight
              // Keep this in the database.
              let tuple = firstStage.encode(
                graph.constant(tensor.toGPU(0)), encoder: encoder, cancellation: { _ in })
              let sample = tuple.0.rawValue.toCPU()
              encoder = tuple.1
              let imagePath = input.imageUrl.path
              store.write(imagePath + ":\(i)", tensor: sample)
              if var processedInput = inputMap[imagePath] {
                processedInput.loadedImagePath = imagePath + ":\(i)"
                processedInputs.append(processedInput)
              }
              guard progressHandler(.imageEncoding, processedInputs.count) else {
                stopped = true
                break
              }
            }
            guard !stopped else { break }
          }
          guard !stopped else { return }
        }
      }
      graph.withNoGrad {
        let textModel: [Model] = [
          CLIPTextModel(
            FloatType.self, injectEmbeddings: false,
            vocabularySize: 49408, maxLength: 77, maxTokenLength: 77, embeddingSize: 768,
            numLayers: 12, numHeads: 12, batchSize: 1,
            intermediateSize: 3072, usesFlashAttention: true, outputPenultimate: true
          ).0,
          T5ForConditionalGeneration(b: 1, t: paddedTextEncodingLength, of: FloatType.self).1,
        ]
        let tokensTensor = graph.variable(.CPU, format: .NHWC, shape: [77], of: Int32.self)
        tokensTensor.full(0)
        let tokensTensorGPU = tokensTensor.toGPU(0)
        let positionTensor = graph.variable(.CPU, format: .NHWC, shape: [77], of: Int32.self)
        for i in 0..<77 {
          positionTensor[i] = Int32(i)
        }
        let positionTensorGPU = positionTensor.toGPU(0)
        let causalAttentionMask = graph.variable(.CPU, .NHWC(1, 1, 77, 77), of: FloatType.self)
        causalAttentionMask.full(0)
        for i in 0..<76 {
          for j in (i + 1)..<77 {
            causalAttentionMask[0, 0, i, j] = -FloatType.greatestFiniteMagnitude
          }
        }
        let causalAttentionMaskGPU = causalAttentionMask.toGPU(0)
        textModel[0].maxConcurrency = .limit(4)
        textModel[0].compile(inputs: tokensTensorGPU, positionTensorGPU, causalAttentionMaskGPU)
        if let CLIPEncoder = CLIPEncoder {
          graph.openStore(
            ModelZoo.filePathForModelDownloaded(CLIPEncoder), flags: .readOnly,
            externalStore: TensorData.externalStore(
              filePath: ModelZoo.filePathForModelDownloaded(CLIPEncoder))
          ) {
            $0.read(
              "text_model", model: textModel[0],
              codec: [.q6p, .q8p, .ezm7, .fpzip, .jit, .externalData]
            )
          }
        }
        for (index, input) in ([zeroCaptionInput] + processedInputs).enumerated() {
          guard input.imagePath == input.loadedImagePath else { continue }
          for i in 0..<77 {
            tokensTensor[i] = input.CLIPTokens[i]
          }
          let tokensTensorGPU = tokensTensor.toGPU(0)
          let c = textModel[0](
            inputs: tokensTensorGPU, positionTensorGPU, causalAttentionMaskGPU
          ).map {
            $0.as(of: FloatType.self)
          }
          var tokenEnd: Int? = nil
          for i in 0..<77 {
            if input.CLIPTokens[i] == 49407 && tokenEnd == nil {
              tokenEnd = i
            }
          }
          if let tokenEnd = tokenEnd {
            store.write(
              "cond_\(input.imagePath)",
              tensor: c[1][tokenEnd..<(tokenEnd + 1), 0..<768].copied().rawValue.toCPU())
          }
          guard progressHandler(.conditionalEncoding, index) else {
            stopped = true
            return
          }
        }
        let relativePositionBuckets = relativePositionBuckets(
          sequenceLength: paddedTextEncodingLength, numBuckets: 32, maxDistance: 128)
        let tokens2Tensor = graph.variable(
          .CPU, format: .NHWC, shape: [paddedTextEncodingLength], of: Int32.self)
        tokens2Tensor.full(0)
        let tokens2TensorGPU = tokens2Tensor.toGPU(0)
        let relativePositionBucketsGPU = graph.variable(relativePositionBuckets.toGPU(0))
        textModel[1].maxConcurrency = .limit(4)
        textModel[1].compile(inputs: tokens2TensorGPU, relativePositionBucketsGPU)
        graph.openStore(
          ModelZoo.filePathForModelDownloaded(textEncoder), flags: .readOnly,
          externalStore: TensorData.externalStore(
            filePath: ModelZoo.filePathForModelDownloaded(textEncoder))
        ) {
          $0.read(
            "text_model", model: textModel[1],
            codec: [.q8p, .q6p, .q4p, .ezm7, .jit, .externalData])
        }
        for (index, input) in ([zeroCaptionInput] + processedInputs).enumerated() {
          guard input.imagePath == input.loadedImagePath else { continue }
          for i in 0..<paddedTextEncodingLength {
            tokens2Tensor[i] = input.tokens[i]
          }
          let tokens2TensorGPU = tokens2Tensor.toGPU(0)
          let c = textModel[1](inputs: tokens2TensorGPU, relativePositionBucketsGPU)[0].as(
            of: FloatType.self
          ).reshaped(.HWC(1, paddedTextEncodingLength, 4096))
          store.write("cond_te2_\(input.imagePath)", tensor: c.rawValue.toCPU())
          guard progressHandler(.conditionalEncoding, index) else {
            stopped = true
            return
          }
        }
      }
    }
    if stopped {
      return nil
    }
    var dataFrame = DataFrame(from: processedInputs)
    dataFrame["imagePath"] = dataFrame["0", ProcessedInput.self].map(\.imagePath)
    // No need to include tokens.
    return (dataFrame, zeroCaptionInput)
  }

  public func prepareKolorsTextEncodings(
    graph: DynamicGraph, store: DynamicGraph.Store, zeroCaptionInput: ProcessedInput,
    processedInputs: [ProcessedInput], progressHandler: (PrepareState, Int) -> Bool
  ) -> Bool {
    return graph.withNoGrad {
      var rightAlignedTokens = Tensor<Int32>(.CPU, format: .NHWC, shape: [256])
      let causalAttentionMask = graph.variable(
        .CPU, .NHWC(1, 1, 256, 256), of: FloatType.self)
      let rotaryEmbedding = GLMRotaryEmbedding(sequenceLength: 256, of: FloatType.self)
      var rightAlignedRotaryEmbedding = Tensor<FloatType>(.CPU, .NHWC(1, 256, 1, 128))
      let textModel = GLMTransformer(
        FloatType.self, vocabularySize: 65_024, width: 4_096, tokenLength: 256,
        layers: 29 - min(max(clipSkip - 1, 1), 27), MLP: 13_696, heads: 32, batchSize: 1,
        outputPenultimate: true, applyFinalNorm: false, usesFlashAttention: true
      ).0
      textModel.maxConcurrency = .limit(4)
      let rightAlignedTokensTensorGPU = graph.variable(rightAlignedTokens.toGPU(0))
      let rightAlignedRotaryEmbeddingGPU = graph.variable(rightAlignedRotaryEmbedding.toGPU(0))
      let causalAttentionMaskGPU = causalAttentionMask.toGPU(0)
      textModel.compile(
        inputs: rightAlignedTokensTensorGPU, rightAlignedRotaryEmbeddingGPU, causalAttentionMaskGPU)
      graph.openStore(
        ModelZoo.filePathForModelDownloaded(textEncoder), flags: .readOnly,
        externalStore: TensorData.externalStore(
          filePath: ModelZoo.filePathForModelDownloaded(textEncoder))
      ) {
        $0.read(
          "text_model", model: textModel, codec: [.jit, .q6p, .q8p, .ezm7, .fpzip, .externalData]
        )
      }
      let encoderHidProj = Dense(count: 2_048)
      let textEncoding = graph.variable(.GPU(0), .HWC(1, 256, 4096), of: FloatType.self)
      encoderHidProj.compile(inputs: textEncoding)
      graph.openStore(
        ModelZoo.filePathForModelDownloaded(model), flags: .readOnly,
        externalStore: TensorData.externalStore(
          filePath: ModelZoo.filePathForModelDownloaded(model))
      ) {
        $0.read(
          "encoder_hid_proj", model: encoderHidProj,
          codec: [.jit, .q6p, .q8p, .ezm7, .externalData])
      }
      for (index, input) in ([zeroCaptionInput] + processedInputs).enumerated() {
        guard input.imagePath == input.loadedImagePath else { continue }
        causalAttentionMask.full(0)
        let lengthOfCond = input.tokens.count
        for i in 0..<256 - lengthOfCond - 2 {
          rightAlignedTokens[i] = 0
        }
        rightAlignedTokens[256 - lengthOfCond - 2] = 64_790
        rightAlignedTokens[256 - lengthOfCond - 1] = 64_792
        for i in (256 - lengthOfCond)..<256 {
          rightAlignedTokens[i] = input.tokens[i - (256 - lengthOfCond)]
        }
        for i in (256 - lengthOfCond - 2)..<(256 - 1) {
          for j in (i + 1)..<256 {
            causalAttentionMask[0, 0, i, j] = -FloatType.greatestFiniteMagnitude
          }
        }
        for i in (256 - lengthOfCond - 2)..<256 {
          for j in 0..<(256 - lengthOfCond - 2) {
            causalAttentionMask[0, 0, i, j] = -FloatType.greatestFiniteMagnitude
          }
        }
        for i in 0..<(256 - lengthOfCond - 2) {
          rightAlignedRotaryEmbedding[0..<1, i..<(i + 1), 0..<1, 0..<128] =
            rotaryEmbedding[0..<1, 0..<1, 0..<1, 0..<128]
        }
        rightAlignedRotaryEmbedding[
          0..<1, (256 - lengthOfCond - 2)..<256, 0..<1, 0..<128] =
          rotaryEmbedding[0..<1, 0..<(lengthOfCond + 2), 0..<1, 0..<128]
        let rightAlignedTokensTensorGPU = graph.variable(rightAlignedTokens.toGPU(0))
        let rightAlignedRotaryEmbeddingGPU = graph.variable(rightAlignedRotaryEmbedding.toGPU(0))
        let causalAttentionMaskGPU = causalAttentionMask.toGPU(0)
        let c = textModel(
          inputs: rightAlignedTokensTensorGPU, rightAlignedRotaryEmbeddingGPU,
          causalAttentionMaskGPU
        ).map {
          $0.as(of: FloatType.self)
        }
        var textEncoding = c[0].reshaped(.HWC(1, 256, 4096))
        textEncoding = encoderHidProj(inputs: textEncoding)[0].as(of: FloatType.self)
        store.write(
          "cond_\(input.imagePath)",
          tensor: textEncoding.rawValue.toCPU())
        let pooled = c[1][255..<256, 0..<4096].copied()
        store.write("pool_\(input.imagePath)", tensor: pooled.rawValue.toCPU())
        guard progressHandler(.conditionalEncoding, index) else {
          return false
        }
      }
      return true
    }
  }

  public func prepareSDTextEncodings(
    graph: DynamicGraph, store: DynamicGraph.Store, zeroCaptionInput: ProcessedInput,
    processedInputs: [ProcessedInput], progressHandler: (PrepareState, Int) -> Bool
  ) -> Bool {
    return graph.withNoGrad {
      let textModel: [Model]
      let embeddingSize: Int
      switch version {
      case .v1:
        embeddingSize = 768
        textModel = [
          CLIPTextModel(
            FloatType.self, injectEmbeddings: false,
            vocabularySize: 49408, maxLength: 77, maxTokenLength: 77,
            embeddingSize: embeddingSize,
            numLayers: 13 - min(max(clipSkip, 1), 12),
            numHeads: 12, batchSize: 1, intermediateSize: 3072, usesFlashAttention: false
          ).0
        ]
      case .v2:
        embeddingSize = 1024
        textModel = [
          OpenCLIPTextModel(
            FloatType.self, injectEmbeddings: false,
            vocabularySize: 49408, maxLength: 77, maxTokenLength: 77,
            embeddingSize: embeddingSize,
            numLayers: 24 - min(max(clipSkip, 1), 23),
            numHeads: 16, batchSize: 1, intermediateSize: 4096,
            usesFlashAttention: false
          ).0
        ]
      case .sdxlBase, .ssd1b:
        embeddingSize = 1280
        textModel = [
          OpenCLIPTextModel(
            FloatType.self, injectEmbeddings: false,
            vocabularySize: 49408, maxLength: 77, maxTokenLength: 77, embeddingSize: 1280,
            numLayers: 32 - min(max(clipSkip - 1, 0), 30), numHeads: 20, batchSize: 1,
            intermediateSize: 5120, usesFlashAttention: false, outputPenultimate: true
          ).0,
          CLIPTextModel(
            FloatType.self, injectEmbeddings: false,
            vocabularySize: 49408, maxLength: 77, maxTokenLength: 77, embeddingSize: 768,
            numLayers: 13 - min(max(clipSkip, 1), 12), numHeads: 12, batchSize: 1,
            intermediateSize: 3072, usesFlashAttention: false, noFinalLayerNorm: true
          ).0,
        ]
      case .sdxlRefiner:
        embeddingSize = 1280
        textModel = [
          OpenCLIPTextModel(
            FloatType.self, injectEmbeddings: false,
            vocabularySize: 49408, maxLength: 77, maxTokenLength: 77, embeddingSize: 1280,
            numLayers: 32 - min(max(clipSkip - 1, 0), 30), numHeads: 20, batchSize: 1,
            intermediateSize: 5120, usesFlashAttention: false, outputPenultimate: true
          ).0
        ]
      case .sd3, .sd3Large, .pixart, .auraflow, .flux1, .kandinsky21, .svdI2v, .wurstchenStageC,
        .wurstchenStageB, .hunyuanVideo, .wan21_1_3b, .wan21_14b, .hiDreamI1:
        fatalError()
      }
      let tokensTensor = graph.variable(.CPU, format: .NHWC, shape: [77], of: Int32.self)
      tokensTensor.full(0)
      let tokensTensorGPU = tokensTensor.toGPU(0)
      let positionTensor = graph.variable(.CPU, format: .NHWC, shape: [77], of: Int32.self)
      for i in 0..<77 {
        positionTensor[i] = Int32(i)
      }
      let positionTensorGPU = positionTensor.toGPU(0)
      let causalAttentionMask = graph.variable(.CPU, .NHWC(1, 1, 77, 77), of: FloatType.self)
      causalAttentionMask.full(0)
      for i in 0..<76 {
        for j in (i + 1)..<77 {
          causalAttentionMask[0, 0, i, j] = -FloatType.greatestFiniteMagnitude
        }
      }
      let causalAttentionMaskGPU = causalAttentionMask.toGPU(0)
      textModel[0].maxConcurrency = .limit(4)
      textModel[0].compile(inputs: tokensTensorGPU, positionTensorGPU, causalAttentionMaskGPU)
      let textProjection: DynamicGraph.Tensor<FloatType>?
      if version == .sdxlBase || version == .sdxlRefiner {
        textProjection = graph.constant(.GPU(0), .WC(1280, 1280), of: FloatType.self)
      } else {
        textProjection = nil
      }
      graph.openStore(
        ModelZoo.filePathForModelDownloaded(textEncoder), flags: .readOnly,
        externalStore: TensorData.externalStore(
          filePath: ModelZoo.filePathForModelDownloaded(textEncoder))
      ) {
        store in
        store.read(
          "text_model", model: textModel[0],
          codec: [.jit, .q6p, .q8p, .ezm7, .fpzip, .externalData]
        ) {
          name, dataType, format, shape in
          // Need to handle clip skip.
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
          case .sdxlBase, .sdxlRefiner, .ssd1b:
            if name == "__text_model__[t-\(258 - (min(clipSkip, 31) - 1) * 8)-0]" {
              name = "__text_model__[t-258-0]"
            } else if name == "__text_model__[t-\(258 - (min(clipSkip, 31) - 1) * 8)-1]" {
              name = "__text_model__[t-258-1]"
            }
          case .sd3, .sd3Large, .pixart, .auraflow, .flux1, .kandinsky21, .svdI2v,
            .wurstchenStageC, .wurstchenStageB, .hunyuanVideo, .wan21_1_3b, .wan21_14b, .hiDreamI1:
            fatalError()
          }
          return .continue(name)
        }
        if let textProjection = textProjection {
          store.read(
            "text_projection", variable: textProjection,
            codec: [.q6p, .q8p, .ezm7, .fpzip, .externalData])
        }
      }
      for (index, input) in ([zeroCaptionInput] + processedInputs).enumerated() {
        guard input.imagePath == input.loadedImagePath else { continue }
        for i in 0..<77 {
          tokensTensor[i] = input.tokens[i]
        }
        let tokensTensorGPU = tokensTensor.toGPU(0)
        let c = textModel[0](
          inputs: tokensTensorGPU, positionTensorGPU, causalAttentionMaskGPU
        ).map {
          $0.as(of: FloatType.self)
        }
        store.write(
          "cond_\(input.imagePath)",
          tensor: c[0].reshaped(.HWC(1, 77, embeddingSize)).rawValue.toCPU())
        if c.count > 1, let textProjection = textProjection {
          var pooled = graph.variable(.GPU(0), .WC(1, embeddingSize), of: FloatType.self)
          var tokenEnd: Int? = nil
          for i in 0..<77 {
            if input.tokens[i] == 49407 && tokenEnd == nil {
              tokenEnd = i
            }
          }
          if let tokenEnd = tokenEnd {
            pooled[0..<1, 0..<embeddingSize] =
              c[1][(tokenEnd)..<(tokenEnd + 1), 0..<embeddingSize] * textProjection
          }
          store.write("pool_\(input.imagePath)", tensor: pooled.rawValue.toCPU())
        }
        guard progressHandler(.conditionalEncoding, index) else {
          return false
        }
      }
      if textModel.count > 1, let CLIPEncoder = CLIPEncoder {
        textModel[1].maxConcurrency = .limit(4)
        textModel[1].compile(inputs: tokensTensorGPU, positionTensorGPU, causalAttentionMaskGPU)
        graph.openStore(
          ModelZoo.filePathForModelDownloaded(CLIPEncoder), flags: .readOnly,
          externalStore: TensorData.externalStore(
            filePath: ModelZoo.filePathForModelDownloaded(CLIPEncoder))
        ) {
          store in
          store.read(
            "text_model", model: textModel[1],
            codec: [.jit, .q6p, .q8p, .ezm7, .fpzip, .externalData]
          ) {
            name, dataType, format, shape in
            // Need to handle clip skip.
            var name = name
            if name == "__text_model__[t-\(98 - (min(clipSkip, 12) - 1) * 8)-0]" {
              name = "__text_model__[t-98-0]"
            } else if name == "__text_model__[t-\(98 - (min(clipSkip, 12) - 1) * 8)-1]" {
              name = "__text_model__[t-98-1]"
            }
            return .continue(name)
          }
        }
        for (index, input) in ([zeroCaptionInput] + processedInputs).enumerated() {
          guard input.imagePath == input.loadedImagePath else { continue }
          for i in 0..<77 {
            tokensTensor[i] = input.tokens[i]
          }
          let tokensTensorGPU = tokensTensor.toGPU(0)
          let c = textModel[1](
            inputs: tokensTensorGPU, positionTensorGPU, causalAttentionMaskGPU)[0]
            .as(of: FloatType.self).reshaped(.HWC(1, 77, 768))
          store.write("cond_te2_\(input.imagePath)", tensor: c.rawValue.toCPU())
          guard progressHandler(.conditionalEncoding, index) else {
            return false
          }
        }
      }
      return true
    }
  }

  public func prepareDataset(
    inputs: [Input], tokenizers: [TextualInversionPoweredTokenizer & Tokenizer],
    customEmbeddingLength: Int, progressHandler: (PrepareState, Int) -> Bool
  ) -> (DataFrame, ProcessedInput)? {
    guard let tokenizer = tokenizers.first, version != .flux1 else {
      return prepareDatasetForFlux1(
        inputs: inputs, tokenizers: tokenizers, customEmbeddingLength: customEmbeddingLength,
        progressHandler: progressHandler)
    }
    // Load each of them, resize, center crop
    // Use VAE to encode the image.
    let latentsScaling = ModelZoo.latentsScalingForModel(model)
    let firstStage = FirstStage<FloatType>(
      filePath: ModelZoo.filePathForModelDownloaded(autoencoder), version: version,
      latentsScaling: latentsScaling, highPrecisionKeysAndValues: false,
      highPrecisionFallback: true,
      tiledDecoding: TiledConfiguration(
        isEnabled: false, tileSize: TiledConfiguration.Size(width: 0, height: 0),
        tileOverlap: 0),
      tiledDiffusion: TiledConfiguration(
        isEnabled: false, tileSize: TiledConfiguration.Size(width: 0, height: 0),
        tileOverlap: 0),
      externalOnDemand: false, alternativeUsesFlashAttention: false, alternativeFilePath: nil,
      alternativeDecoderVersion: nil, memoryCapacity: DeviceCapability.memoryCapacity,
      isNHWCPreferred: DeviceCapability.isNHWCPreferred)
    let graph = DynamicGraph()
    let imageWidth = Int(scale.widthScale) * 64
    let imageHeight = Int(scale.heightScale) * 64
    var processedInputs = [ProcessedInput]()
    let tokenMaxLength: Int
    let truncation: Bool
    switch textEncoderVersion {
    case .chatglm3_6b:
      truncation = false
      tokenMaxLength = 0
    case nil:
      truncation = true
      tokenMaxLength = 77
    }
    let (_, zeroTokens, _, _, _) = tokenizer.tokenize(
      text: "", truncation: truncation, maxLength: tokenMaxLength)
    let zeroCaptionInput = ProcessedInput(
      imagePath: "", tokens: zeroTokens, CLIPTokens: [], originalSize: (0, 0), cropTopLeft: (0, 0),
      targetSize: (0, 0), loadedImagePath: "")
    var stopped = false
    graph.openStore(session) { store in
      // First, center crop and encode the image.
      graph.withNoGrad {
        var previousImageWidth: Int? = nil
        var previousImageHeight: Int? = nil
        var encoder: Model? = nil
        for input in inputs {
          guard let originalSize = imageInspector(input.imageUrl),
            originalSize.width > 0 && originalSize.height > 0
          else { continue }
          let imageWidth: Int
          let imageHeight: Int
          if useImageAspectRatio {
            let imageSize = Self.sizeThatFits(size: originalSize, scale: scale)
            imageWidth = imageSize.width * 64
            imageHeight = imageSize.height * 64
          } else {
            imageWidth = Int(scale.widthScale) * 64
            imageHeight = Int(scale.heightScale) * 64
          }
          guard
            let (tensor, originalSize, cropTopLeft, targetSize) = imageLoader(
              input.imageUrl, imageWidth, imageHeight)
          else { continue }
          if previousImageWidth != imageWidth || previousImageHeight != imageHeight {
            encoder = nil  // Reload encoder.
          }
          // Keep this in the database.
          let tuple = firstStage.encode(
            graph.constant(tensor.toGPU(0)), encoder: encoder, cancellation: { _ in })
          let sample = tuple.0.rawValue.toCPU()
          encoder = tuple.1
          let imagePath = input.imageUrl.path
          store.write(imagePath, tensor: sample)
          let (_, tokens, _, _, _) = tokenizer.tokenize(
            text: input.caption, truncation: truncation, maxLength: tokenMaxLength)
          var updatedTokens = [Int32]()
          for token in tokens {
            if tokenizer.isTextualInversion(token) {
              for _ in 0..<customEmbeddingLength {
                updatedTokens.append(token)
              }
            } else {
              updatedTokens.append(token)
            }
          }
          if tokenMaxLength > 0 && truncation && updatedTokens.count > tokenMaxLength {
            updatedTokens = Array(updatedTokens.dropLast(updatedTokens.count - tokenMaxLength))
          }
          processedInputs.append(
            ProcessedInput(
              imagePath: imagePath, tokens: updatedTokens, CLIPTokens: [],
              originalSize: originalSize,
              cropTopLeft: cropTopLeft, targetSize: targetSize, loadedImagePath: imagePath))
          guard progressHandler(.imageEncoding, processedInputs.count) else {
            stopped = true
            break
          }
        }
        guard !stopped else { return }
        let zeros = graph.variable(
          .GPU(0), .NHWC(1, imageHeight, imageWidth, 3), of: FloatType.self)
        zeros.full(0)
        let latentZeros = firstStage.sample(zeros, encoder: encoder, cancellation: { _ in }).1
          .copied().rawValue.toCPU()
        store.write("latent_zeros", tensor: latentZeros)
        if useImageAspectRatio && !additionalScales.isEmpty {
          var inputMap = [String: ProcessedInput]()
          for input in processedInputs {
            inputMap[input.imagePath] = input
          }
          for (i, scale) in additionalScales.enumerated() {
            for input in inputs {
              guard let originalSize = imageInspector(input.imageUrl),
                originalSize.width > 0 && originalSize.height > 0
              else { continue }
              let imageSize = Self.sizeThatFits(size: originalSize, scale: scale)
              let imageWidth = imageSize.width * 64
              let imageHeight = imageSize.height * 64
              guard
                let (tensor, originalSize, cropTopLeft, targetSize) = imageLoader(
                  input.imageUrl, imageWidth, imageHeight)
              else { continue }
              if previousImageWidth != imageWidth || previousImageHeight != imageHeight {
                encoder = nil  // Reload encoder.
              }
              previousImageWidth = imageWidth
              previousImageHeight = imageHeight
              // Keep this in the database.
              let tuple = firstStage.encode(
                graph.constant(tensor.toGPU(0)), encoder: encoder, cancellation: { _ in })
              let sample = tuple.0.rawValue.toCPU()
              encoder = tuple.1
              let imagePath = input.imageUrl.path
              store.write(imagePath + ":\(i)", tensor: sample)
              if var processedInput = inputMap[imagePath] {
                processedInput.originalSize = originalSize
                processedInput.cropTopLeft = cropTopLeft
                processedInput.targetSize = targetSize
                processedInput.loadedImagePath = imagePath + ":\(i)"
                processedInputs.append(processedInput)
              }
              guard progressHandler(.imageEncoding, processedInputs.count) else {
                stopped = true
                break
              }
            }
            guard !stopped else { break }
          }
          guard !stopped else { return }
        }
      }
      if stopped {
        return
      }
      // Now, check if we have cotrain turned on, if we don't, save the text encoding directly so we don't have to load the text model.
      if !cotrainTextModel && !cotrainCustomEmbedding {
        switch textEncoderVersion {
        case .chatglm3_6b:
          guard
            prepareKolorsTextEncodings(
              graph: graph, store: store, zeroCaptionInput: zeroCaptionInput,
              processedInputs: processedInputs, progressHandler: progressHandler)
          else {
            stopped = true
            return
          }
        case nil:
          guard
            prepareSDTextEncodings(
              graph: graph, store: store, zeroCaptionInput: zeroCaptionInput,
              processedInputs: processedInputs, progressHandler: progressHandler)
          else {
            stopped = true
            return
          }
        }
      }
    }
    if stopped {
      return nil
    }
    var dataFrame = DataFrame(from: processedInputs)
    dataFrame["imagePath"] = dataFrame["0", ProcessedInput.self].map(\.imagePath)
    dataFrame["tokens"] = dataFrame["0", ProcessedInput.self].map(\.tokens)
    return (dataFrame, zeroCaptionInput)
  }

  public static func originalLoRA(name: String, LoRAMapping: [Int: Int]?) -> String {
    let components = name.split(separator: "-")
    guard components.count >= 3, let index = Int(components[2])
    else { return name }
    let originalIndex: Int
    if let LoRAMapping = LoRAMapping {
      guard let index = LoRAMapping[index] else { return name }
      originalIndex = index
    } else {
      originalIndex = index
    }
    let isUp = name.contains("lora_up")
    var infix = components[1].replacingOccurrences(of: "lora_up", with: "")
      .replacingOccurrences(
        of: "lora_down", with: "")
    // In case infix has _, remove them.
    if infix.hasSuffix("_") {
      infix = String(infix.prefix(upTo: infix.index(before: infix.endIndex)))
    }
    let originalPrefix: String
    if infix.isEmpty {
      originalPrefix = "\(components[0])-\(originalIndex)-0]"
    } else {
      originalPrefix = "\(components[0])-\(infix)-\(originalIndex)-0]"
    }
    return originalPrefix + (isUp ? "__up__" : "__down__")
  }

  // This is heavily based on https://github.com/thuanz123/realfill/blob/main/train_realfill.py
  private func makeMask<T: RandomNumberGenerator>(
    resolution: (width: Int, height: Int), times: Int = 30, using generator: inout T
  ) -> Tensor<FloatType> {
    var mask = Tensor<FloatType>(.CPU, .NHWC(1, resolution.height, resolution.width, 1))
    for y in 0..<resolution.height {
      for x in 0..<resolution.width {
        mask[0, y, x, 0] = 1
      }
    }
    var sfmt = SFMT(seed: UInt64.random(in: UInt64.min...UInt64.max, using: &generator))
    // 1 out of 10 chance to mask out everything.
    if Int.random(in: 0..<10, using: &sfmt) == 0 {
      return mask
    }
    let times = Int.random(in: 1...times, using: &sfmt)
    let minWidth = Int((0.03 * Float(resolution.width)).rounded())
    var maxWidth = Int((0.25 * Float(resolution.width)).rounded())
    let marginX = Int((0.01 * Float(resolution.width)).rounded())
    let minHeight = Int((0.03 * Float(resolution.height)).rounded())
    var maxHeight = Int((0.25 * Float(resolution.height)).rounded())
    let marginY = Int((0.01 * Float(resolution.height)).rounded())
    maxWidth = min(maxWidth, resolution.width - marginX * 2)
    maxHeight = min(maxHeight, resolution.height - marginY * 2)
    for _ in 0..<times {
      let width = Int.random(in: minWidth..<maxWidth, using: &sfmt)
      let height = Int.random(in: minHeight..<maxHeight, using: &sfmt)
      let xStart = Int.random(in: marginX..<(resolution.width - marginX - width + 1), using: &sfmt)
      let yStart = Int.random(
        in: marginY..<(resolution.height - marginY - height + 1), using: &sfmt)
      for y in yStart..<(yStart + height) {
        for x in xStart..<(xStart + width) {
          mask[0, y, x, 0] = 0
        }
      }
    }
    if Int.random(in: 0...1, using: &sfmt) == 1 {
      for y in 0..<resolution.height {
        for x in 0..<resolution.width {
          mask[0, y, x, 0] = 1 - mask[0, y, x, 0]
        }
      }
    }
    return mask
  }

  private func encodeFlux1Fixed(
    graph: DynamicGraph,
    externalData: DynamicGraph.Store.Codec,
    batch: [(
      pooled: Tensor<FloatType>, timestep: Float, guidance: Float
    )]
  ) -> [Tensor<FloatType>] {
    return graph.withNoGrad {
      let filePath = ModelZoo.filePathForModelDownloaded(model)
      let isGuidanceEmbedSupported =
        (try?
          (graph.openStore(
            filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
          ) {
            return $0.read(like: "__dit__[t-guidance_embedder_0-0-1]") != nil
          }).get()) ?? false
      let unetFixed = Flux1Fixed(
        batchSize: (batch.count, batch.count), channels: 3072, layers: (19, 38),
        contextPreloaded: false, guidanceEmbed: isGuidanceEmbedSupported
      ).1
      var timeEmbeds = graph.variable(
        .GPU(0), .WC(batch.count, 256), of: FloatType.self)
      var pooleds = graph.variable(
        .GPU(0), .WC(batch.count, 768), of: FloatType.self)
      var guidanceEmbeds: DynamicGraph.Tensor<FloatType>?
      if isGuidanceEmbedSupported {
        guidanceEmbeds = graph.variable(
          .GPU(0), .WC(batch.count, 256), of: FloatType.self)
      } else {
        guidanceEmbeds = nil
      }
      for (i, item) in batch.enumerated() {
        let timeEmbed = graph.variable(
          Tensor<FloatType>(
            from: timeEmbedding(
              timestep: item.timestep * 1_000, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000)
          ).toGPU(0))
        timeEmbeds[i..<(i + 1), 0..<256] = timeEmbed
        pooleds[i..<(i + 1), 0..<768] = graph.variable(item.pooled.toGPU(0))
        if var guidanceEmbeds = guidanceEmbeds {
          let guidanceScale = item.guidance
          let guidanceEmbed = graph.variable(
            Tensor<FloatType>(
              from: timeEmbedding(
                timestep: guidanceScale * 1_000, batchSize: 1, embeddingSize: 256,
                maxPeriod: 10_000)
            ).toGPU(0))
          guidanceEmbeds[i..<(i + 1), 0..<256] = guidanceEmbed
        }
      }
      unetFixed.maxConcurrency = .limit(4)
      unetFixed.compile(
        inputs: [timeEmbeds, pooleds] + (guidanceEmbeds.map { [$0] } ?? []))
      graph.openStore(
        filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
      ) {
        $0.read("dit", model: unetFixed, codec: [.jit, .q6p, .q8p, .ezm7, externalData])
      }
      return unetFixed(
        inputs: timeEmbeds, [pooleds] + (guidanceEmbeds.map { [$0] } ?? [])
      ).map { $0.as(of: FloatType.self).rawValue }
    }
  }

  public enum TrainingState {
    case compile
    case step(Int)
  }

  private static func randomOrthonormalMatrix<FloatType: TensorNumeric & BinaryFloatingPoint>(
    graph: DynamicGraph, M: Int, N: Int, of: FloatType.Type = FloatType.self
  ) -> Tensor<FloatType> {
    return graph.withNoGrad {
      let A = graph.variable(.GPU(0), .WC(M, N), of: Float.self)
      var Q = (0..<M).map { _ in graph.variable(.GPU(0), .WC(1, N), of: Float.self) }
      A.randn()
      Q.forEach { $0.full(0) }
      var O = graph.variable(.GPU(0), .WC(M, N), of: Float.self)
      for i in 0..<M {
        var v = A[i..<(i + 1), 0..<N].copied()
        for j in 0..<i {
          let proj =
            Functional.matmul(left: A[i..<(i + 1), 0..<N], right: Q[j], rightTranspose: (0, 1))
            .* Q[j]
          v = v - proj
        }
        let norm = v.reduced(.norm2, axis: [0, 1])
        Q[i] = (1 / norm) .* v
        O[i..<(i + 1), 0..<N] = Q[i]
      }
      return Tensor<FloatType>(from: O.rawValue).toCPU()
    }
  }

  // Using Newton method to compute gamma based on https://arxiv.org/pdf/2312.02696v1 where
  // theta_rel = (gamma + 1)^(1/2)*(gamma + 2)^(-1)*(gamma + 3)^(-0.5)
  // theta_rel is the width (std), and gamma is the gamma for power function EMA.
  private static func powerFunctionExponentialMovingAverageFromWidthToGamma(
    _ y: Double, tolerance: Double = 1e-10, maxIterations: Int = 100
  ) -> Double? {
    // Validate input
    guard 0 < y && y <= 1 else {
      return nil
    }
    // Main function evaluation
    func evaluate(_ x: Double) -> Double {
      (x + 1).squareRoot() / (x + 2) / (x + 3).squareRoot()
    }

    // Derivative evaluation
    func derivative(_ x: Double) -> Double {
      let term1 = 0.5 / (x + 1).squareRoot() / (x + 2) / (x + 3).squareRoot()
      let term2 = -1 * (x + 1).squareRoot() / ((x + 2) * (x + 2)) / (x + 3).squareRoot()
      let term3 = -0.5 * (x + 1).squareRoot() / (x + 2) / (x + 3) / (x + 3).squareRoot()
      return term1 + term2 + term3
    }

    var x = 1.0  // Initial guess

    for _ in 0..<maxIterations {
      let fx = evaluate(x) - y
      let dx = derivative(x)

      // Check if we're close enough to the solution
      if abs(fx) < tolerance {
        return x
      }

      // Check if derivative is too small
      guard abs(dx) >= 1e-10 else {
        return nil
      }

      // Newton's method step
      x = x - fx / dx

      // Reset if x goes negative (invalid domain)
      if x < 0 {
        x = 0.1
      }
    }
    return nil
  }

  struct CachedRotaryPositionEmbedder {
    private struct Size: Equatable, Hashable {
      var height: Int
      var width: Int
      var tokenLength: Int
    }
    private var rotaryEmbeddings = [Size: Tensor<FloatType>]()
    mutating func Flux1RotaryPositionEmbedding(
      graph: DynamicGraph, height: Int, width: Int, tokenLength: Int
    ) -> DynamicGraph.Tensor<FloatType> {
      // Only save rotary for one head, and generate full rotary embedding on-the-fly.
      // This way, it is 24x smaller to save, more comfortable for cases you have a few multi-scales and a few different aspect ratios (such that 200MiB RAM would be enough to save ~200 different rotary embeddings).
      let size = Size(height: height, width: width, tokenLength: tokenLength)
      if let embedding = rotaryEmbeddings[size] {
        var fullRotary = Tensor<FloatType>(.GPU(0), .NHWC(1, height * width + tokenLength, 24, 128))
        for i in 0..<24 {
          fullRotary[0..<1, 0..<(height * width + tokenLength), i..<(i + 1), 0..<128] = embedding
        }
        return graph.constant(fullRotary)
      }
      let rotary = Tensor<FloatType>(
        from: Diffusion.Flux1RotaryPositionEmbedding(
          height: height, width: width, tokenLength: tokenLength, channels: 128, heads: 1)
      ).toGPU(0)
      rotaryEmbeddings[size] = rotary
      var fullRotary = Tensor<FloatType>(.GPU(0), .NHWC(1, height * width + tokenLength, 24, 128))
      for i in 0..<24 {
        fullRotary[0..<1, 0..<(height * width + tokenLength), i..<(i + 1), 0..<128] = rotary
      }
      return graph.constant(fullRotary)
    }
  }

  private func trainFlux1(
    graph: DynamicGraph, firstStage: FirstStage<FloatType>, sessionStore: DynamicGraph.Store,
    resumingLoRAFile: (String, Int)?,
    dataFrame: DataFrame, trainingSteps: Int, warmupSteps: Int, gradientAccumulationSteps: Int,
    rankOfLoRA: Int, scaleOfLoRA: Float, unetLearningRate: ClosedRange<Float>,
    stepsBetweenRestarts: Int, seed: UInt32, trainableKeys: [String],
    resolutionDependentShift: Bool, shift: Float, noiseOffset: Float,
    guidanceEmbed: ClosedRange<Float>, denoisingTimesteps: ClosedRange<Int>,
    captionDropoutRate: Float, orthonormalLoRADown: Bool, powerEMA: ClosedRange<Float>?,
    memorySaver: MemorySaver, weightsMemory: WeightsMemoryManagement,
    progressHandler: (TrainingState, Float, LoRATrainerCheckpoint)
      -> Bool
  ) {
    guard unetLearningRate.upperBound > 0 else {
      return
    }
    let gammas: (lowerBound: Double, upperBound: Double)? = powerEMA.flatMap {
      guard
        let gammaLowerBound = Self.powerFunctionExponentialMovingAverageFromWidthToGamma(
          Double($0.lowerBound)),
        let gammaUpperBound = Self.powerFunctionExponentialMovingAverageFromWidthToGamma(
          Double($0.upperBound))
      else { return nil }
      return (lowerBound: gammaLowerBound, upperBound: gammaUpperBound)
    }
    let queueWatermark = DynamicGraph.queueWatermark
    if #unavailable(iOS 18.0, macOS 15.0) {  // It seems that for OS lower than 18 / 15, there are some MPSGraph synchronization issue that is solved under 18 / 15.
      DynamicGraph.queueWatermark = min(2, queueWatermark)
    }
    defer {
      DynamicGraph.queueWatermark = queueWatermark
    }
    DynamicGraph.setSeed(seed)
    var dataFrame = dataFrame
    let configuration: LoRANetworkConfiguration
    switch memorySaver {
    case .minimal:
      configuration = LoRANetworkConfiguration(
        rank: rankOfLoRA, scale: scaleOfLoRA, highPrecision: true, testing: false,
        gradientCheckpointingFeedForward: true, gradientCheckpointingTransformerLayer: true,
        keys: trainableKeys, orthonormalDown: orthonormalLoRADown)
    case .balanced:
      configuration = LoRANetworkConfiguration(
        rank: rankOfLoRA, scale: scaleOfLoRA, highPrecision: true, testing: false,
        gradientCheckpointingFeedForward: true, keys: trainableKeys,
        orthonormalDown: orthonormalLoRADown)
    case .speed, .turbo:
      configuration = LoRANetworkConfiguration(
        rank: rankOfLoRA, scale: scaleOfLoRA, highPrecision: true, testing: false,
        keys: trainableKeys, orthonormalDown: orthonormalLoRADown)
    }
    let latentsWidth = Int(scale.widthScale) * 8
    let latentsHeight = Int(scale.heightScale) * 8
    let dit: ModelBuilder<(width: Int, height: Int)> = ModelBuilder { size, _ in
      return LoRAFlux1(
        batchSize: 1, tokenLength: paddedTextEncodingLength, height: size.height,
        width: size.width, channels: 3072,
        layers: (19, 38), usesFlashAttention: .scaleMerged, contextPreloaded: false,
        injectControls: false, injectIPAdapterLengths: [:], outputResidual: false,
        inputResidual: false, LoRAConfiguration: configuration,
        useConvolutionForPatchify: false
      ).1
    }
    dit.maxConcurrency = .limit(1)
    dit.memoryReduction = (memorySaver != .turbo)
    let latents = graph.variable(
      .GPU(0), .HWC(1, (latentsHeight / 2) * (latentsWidth / 2), 16 * 2 * 2), of: FloatType.self)
    var cachedRotaryPositionEmbedder = CachedRotaryPositionEmbedder()
    let rotaryConstant = cachedRotaryPositionEmbedder.Flux1RotaryPositionEmbedding(
      graph: graph, height: latentsHeight / 2, width: latentsWidth / 2,
      tokenLength: paddedTextEncodingLength)
    let cArr =
      [rotaryConstant] + [
        graph.constant(.GPU(0), .HWC(1, paddedTextEncodingLength, 4096), of: FloatType.self)
      ]
      + Flux1FixedOutputShapes(
        batchSize: (1, 1), tokenLength: paddedTextEncodingLength, channels: 3072, layers: (19, 38),
        contextPreloaded: false
      ).map {
        graph.constant(.GPU(0), format: .NHWC, shape: $0, of: FloatType.self)
      }
    guard progressHandler(.compile, 0, LoRATrainerCheckpoint(version: version, unet: dit, step: 0))
    else {
      return
    }
    dit.compile((width: latentsWidth, height: latentsHeight), inputs: [latents] + cArr)
    let externalData: DynamicGraph.Store.Codec =
      weightsMemory == .justInTime ? .externalOnDemand : .externalData
    graph.openStore(
      ModelZoo.filePathForModelDownloaded(model), flags: .readOnly,
      externalStore: TensorData.externalStore(filePath: ModelZoo.filePathForModelDownloaded(model))
    ) { store in
      store.read("dit", model: dit, codec: [.jit, .q5p, .q6p, .q8p, .ezm7, .fpzip, externalData]) {
        name, dataType, format, shape in
        if resumeIfPossible && (name.contains("[i-") || name.contains("lora")) {
          if let resumingLoRAFile = resumingLoRAFile?.0, name.contains("lora") {
            if let tensor = try?
              (graph.openStore(
                LoRAZoo.filePathForModelDownloaded(resumingLoRAFile), flags: .readOnly
              ) {
                return $0.read(
                  Self.originalLoRA(name: name, LoRAMapping: nil),
                  codec: [.q8p, .ezm7, .fpzip, .externalData])
              }).get()
            {
              return .final(Tensor<Float>(from: tensor).toCPU())
            }
          } else if let tensor = sessionStore.read(name) {
            return .final(Tensor<Float>(tensor).toCPU())
          }
        }
        if name.contains("lora_up") {
          switch dataType {
          case .Float16:
            #if !((os(macOS) || (os(iOS) && targetEnvironment(macCatalyst))) && (arch(i386) || arch(x86_64)))
              var tensor = Tensor<Float16>(.CPU, format: format, shape: shape)
              tensor.withUnsafeMutableBytes {
                let size = shape.reduce(MemoryLayout<Float16>.size, *)
                memset($0.baseAddress, 0, size)
              }
              return .final(tensor)
            #else
              break
            #endif
          case .Float32:
            var tensor = Tensor<Float32>(.CPU, format: format, shape: shape)
            tensor.withUnsafeMutableBytes {
              let size = shape.reduce(MemoryLayout<Float32>.size, *)
              memset($0.baseAddress, 0, size)
            }
            return .final(tensor)
          case .Float64, .Int32, .Int64, .UInt8:
            fatalError()
          }
        } else if orthonormalLoRADown && name.contains("lora_down") {
          switch dataType {
          case .Float16:
            #if !((os(macOS) || (os(iOS) && targetEnvironment(macCatalyst))) && (arch(i386) || arch(x86_64)))
              let tensor = Self.randomOrthonormalMatrix(
                graph: graph, M: shape[0], N: shape[1..<shape.count].reduce(1, *), of: Float16.self)
              return .final(tensor)
            #else
              break
            #endif
          case .Float32:
            let tensor = Self.randomOrthonormalMatrix(
              graph: graph, M: shape[0], N: shape[1..<shape.count].reduce(1, *), of: Float32.self)
            return .final(tensor)
          case .Float64, .Int32, .Int64, .UInt8:
            fatalError()
          }
        }
        // Patch for bias weights which missing a 1/8 scale. Note that this is not needed if we merge this into the model import step like we do for Hunyuan.
        if name.hasSuffix("_out_proj-17-1]") || name.hasSuffix("_out_proj-18-1]"),
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
        return .continue(name)
      }
    }
    var optimizer = AdamWOptimizer(
      graph, rate: unetLearningRate.upperBound, betas: (0.9, 0.999), decay: 0.001, epsilon: 1e-8)
    optimizer.parameters = [dit.parameters]
    var optimizers = [optimizer]
    var scaler = GradScaler(scale: 32_768)
    var i: Int
    if resumeIfPossible {
      if let resumingStep = resumingLoRAFile?.1 {
        i = resumingStep
      } else {
        let stepTensor = sessionStore.read("current_step").flatMap { Tensor<Int32>(from: $0) }
        i = stepTensor.map { max(Int($0[0]), 0) } ?? 0
      }
    } else {
      i = 0
    }
    var stopped = false
    var batchCount = 0
    var batch = [
      (
        loadedImagePath: String, textEncodingPath: String, pooled: Tensor<FloatType>,
        timestep: Float, guidance: Float, captionDropout: Bool
      )
    ]()
    var sfmt = SFMT(seed: UInt64(seed))
    // Do a dry-run to allocate everything properly. This will make sure we allocate largest RAM so no reallocation later.
    let _ = dit((width: latentsWidth, height: latentsHeight), inputs: latents, cArr)
    guard progressHandler(.step(0), 0, LoRATrainerCheckpoint(version: version, unet: dit, step: 0))
    else {
      return
    }
    var ditEMALowerBoundWeights = [String: Tensor<Float>]()
    var ditEMAUpperBoundWeights = [String: Tensor<Float>]()
    while i < trainingSteps && !stopped {
      dataFrame.shuffle()
      for value in dataFrame["0", "imagePath"] {
        guard let input = value[0] as? ProcessedInput, let imagePath = value[1] as? String else {
          continue
        }
        let loadedImagePath = input.loadedImagePath
        guard let tensor = sessionStore.read(like: loadedImagePath) else { continue }
        let shape = tensor.shape
        let latentsHeight = tensor.shape[1]
        let latentsWidth = tensor.shape[2]
        guard shape[3] == 32 else {
          continue
        }
        let captionDropout = Float.random(in: 0...1, using: &sfmt) < captionDropoutRate
        let textEncodingPath = captionDropout ? "" : imagePath
        guard
          let _ = sessionStore.read(like: "cond_te2_\(textEncodingPath)"),
          let pooled = sessionStore.read("cond_\(textEncodingPath)").map({
            Tensor<FloatType>(from: $0)
          })
        else { continue }
        var timestep: Double = Double.random(
          in: (Double(denoisingTimesteps.lowerBound) / 999)...(Double(
            denoisingTimesteps.upperBound) / 999), using: &sfmt)
        var shift = shift
        if resolutionDependentShift {
          shift = Float(
            ModelZoo.shiftFor((width: UInt16(latentsWidth / 8), height: UInt16(latentsHeight / 8))))
        }
        timestep = Double(shift) * timestep / (1 + (Double(shift) - 1) * timestep)
        let guidanceEmbed = (Float.random(in: guidanceEmbed, using: &sfmt) * 10).rounded() / 10
        batch.append(
          (
            loadedImagePath, textEncodingPath, pooled, Float(timestep), guidanceEmbed,
            captionDropout
          ))
        if batch.count == 32 {
          let conditions = encodeFlux1Fixed(
            graph: graph, externalData: externalData,
            batch: batch.map {
              ($0.pooled, $0.timestep, $0.guidance)
            })
          for (j, item) in batch.enumerated() {
            guard let tensor = sessionStore.read(item.loadedImagePath) else { continue }
            let textEncodingPath = item.textEncodingPath
            guard
              let textEncoding = sessionStore.read("cond_te2_\(textEncodingPath)").map({
                Tensor<FloatType>(from: $0)
              })
            else { continue }
            let latentsHeight = tensor.shape[1]
            let latentsWidth = tensor.shape[2]
            let (zt, target) = graph.withNoGrad {
              let parameters = graph.variable(Tensor<FloatType>(from: tensor).toGPU(0))
              let noise = graph.variable(
                .CPU, .NHWC(1, latentsHeight, latentsWidth, 16), of: Float.self)
              noise.randn(std: 1, mean: 0)
              let noiseGPU = DynamicGraph.Tensor<FloatType>(from: noise.toGPU(0))
              var latents = firstStage.sampleFromDistribution(parameters, noise: noiseGPU).0
              latents = latents.reshaped(
                format: .NHWC, shape: [1, latentsHeight / 2, 2, latentsWidth / 2, 2, 16]
              ).permuted(0, 1, 3, 5, 2, 4).contiguous().reshaped(
                format: .NHWC,
                shape: [
                  1, (latentsHeight / 2) * (latentsWidth / 2), 16 * 2 * 2,
                ])
              let z1 = graph.variable(like: latents)
              z1.randn()
              let zt = Functional.add(
                left: latents, right: z1, leftScalar: 1 - item.timestep, rightScalar: item.timestep)
              return (zt, z1 - latents)
            }
            let rotaryConstant = cachedRotaryPositionEmbedder.Flux1RotaryPositionEmbedding(
              graph: graph, height: latentsHeight / 2, width: latentsWidth / 2,
              tokenLength: paddedTextEncodingLength)
            let context = graph.constant(textEncoding.toGPU(0))
            let condition1 = conditions.map {
              let shape = $0.shape
              return graph.constant($0[j..<(j + 1), 0..<shape[1], 0..<shape[2]].copied())
            }
            let vtheta = dit(
              (width: latentsWidth, height: latentsHeight), inputs: zt,
              [rotaryConstant, context] + condition1)[0].as(
                of: FloatType.self)
            let d = target - vtheta
            let loss = (d .* d).reduced(.mean, axis: [1, 2])
            scaler.scale(loss).backward(to: [zt])
            let value = loss.toCPU()[0, 0, 0]
            print(
              "loss \(value), scale \(scaler.scale), step \(i), timestep \(item.timestep), image \(latentsWidth)x\(latentsHeight)"
            )
            batchCount += 1
            let learningRate: Float
            if stepsBetweenRestarts > 1 {
              learningRate =
                unetLearningRate.lowerBound + 0.5
                * (unetLearningRate.upperBound - unetLearningRate.lowerBound)
                * (1
                  + cos(
                    (Float(i % (stepsBetweenRestarts - 1)) / Float(stepsBetweenRestarts - 1)) * .pi))
            } else {
              learningRate = unetLearningRate.upperBound
            }
            if (i + 1) < warmupSteps {
              optimizers[0].rate = learningRate * (Float(i + 1) / Float(warmupSteps))
            } else {
              optimizers[0].rate = learningRate
            }
            if let summaryWriter = summaryWriter {
              summaryWriter.addScalar("loss", value, step: i)
              summaryWriter.addScalar("scale", scaler.scale, step: i)
              summaryWriter.addScalar("timestep", item.timestep, step: i)
              summaryWriter.addScalar("latents_width", Float(latentsWidth), step: i)
              summaryWriter.addScalar("latents_height", Float(latentsHeight), step: i)
              summaryWriter.addScalar("learning_rate", optimizers[0].rate, step: i)
            }
            if batchCount == gradientAccumulationSteps {
              // Update the LoRA.
              scaler.step(&optimizers)
              batchCount = 0
              if let gammas = gammas, let optimizer = optimizers.first {  // If we need to compute ema.
                let betaLowerBound = Float(
                  pow((1 - 1 / Double(optimizer.step)), gammas.lowerBound + 1))
                let betaUpperBound = Float(
                  pow((1 - 1 / Double(optimizer.step)), gammas.upperBound + 1))
                graph.withNoGrad {
                  let loraUps = dit.parameters.filter(where: { $0.contains("lora_up") })
                  let loraDowns = dit.parameters.filter(where: { $0.contains("lora_down") })
                  let parameters = loraUps + loraDowns
                  for parameter in parameters {
                    let name = parameter.name
                    let value = parameter.copied(Float.self).toGPU(0)
                    if let oldEma1 = ditEMALowerBoundWeights[name],
                      let oldEma2 = ditEMAUpperBoundWeights[name]
                    {
                      let value = graph.variable(value)
                      ditEMALowerBoundWeights[name] =
                        Functional.add(
                          left: graph.variable(oldEma1), right: value, leftScalar: betaLowerBound,
                          rightScalar: 1 - betaLowerBound
                        ).rawValue
                      ditEMAUpperBoundWeights[name] =
                        Functional.add(
                          left: graph.variable(oldEma2), right: value, leftScalar: betaUpperBound,
                          rightScalar: 1 - betaUpperBound
                        ).rawValue
                    } else {
                      ditEMALowerBoundWeights[name] = value
                      ditEMAUpperBoundWeights[name] = value
                    }
                  }
                }
              }
            }
            i += 1
            guard
              progressHandler(
                .step(i), Float(value),
                LoRATrainerCheckpoint(
                  version: version, unet: dit, step: i,
                  exponentialMovingAverageLowerBound:
                    LoRATrainerCheckpoint.ExponentialMovingAverage(
                      textModel1: [:], textModel2: [:], unetFixed: [:],
                      unet: ditEMALowerBoundWeights),
                  exponentialMovingAverageUpperBound:
                    LoRATrainerCheckpoint.ExponentialMovingAverage(
                      textModel1: [:], textModel2: [:], unetFixed: [:],
                      unet: ditEMAUpperBoundWeights)))
            else {
              stopped = true
              break
            }
            if i >= trainingSteps {
              break
            }
          }
          if stopped {
            break
          }
          batch = []
          if i >= trainingSteps {
            break
          }
        }
      }
    }
  }

  public func train(
    resumingLoRAFile: (String, Int)?, tokenizers: [TextualInversionPoweredTokenizer & Tokenizer],
    dataFrame: DataFrame, zeroCaption: ProcessedInput, trainingSteps: Int, warmupSteps: Int,
    gradientAccumulationSteps: Int,
    rankOfLoRA: Int, scaleOfLoRA: Float, textModelLearningRate: Float,
    unetLearningRate: ClosedRange<Float>,
    stepsBetweenRestarts: Int, seed: UInt32, trainableKeys: [String],
    customEmbeddingLength: Int, customEmbeddingLearningRate: Float,
    stopEmbeddingTrainingAtStep: Int, resolutionDependentShift: Bool, shift: Float,
    noiseOffset: Float, guidanceEmbed: ClosedRange<Float>, denoisingTimesteps: ClosedRange<Int>,
    captionDropoutRate: Float, orthonormalLoRADown: Bool, powerEMA: ClosedRange<Float>?,
    memorySaver: MemorySaver, weightsMemory: WeightsMemoryManagement,
    progressHandler: (
      TrainingState, Float, LoRATrainerCheckpoint
    )
      -> Bool
  ) {
    let graph = DynamicGraph()
    // To make sure we triggered a clean-up so there are just a little bit more RAM available.
    if !DeviceCapability.isMaxPerformance {
      graph.garbageCollect()
    }
    let latentsScaling = ModelZoo.latentsScalingForModel(model)
    let firstStage = FirstStage<FloatType>(
      filePath: ModelZoo.filePathForModelDownloaded(autoencoder), version: version,
      latentsScaling: latentsScaling, highPrecisionKeysAndValues: false,
      highPrecisionFallback: true,
      tiledDecoding: TiledConfiguration(
        isEnabled: false, tileSize: TiledConfiguration.Size(width: 0, height: 0),
        tileOverlap: 0),
      tiledDiffusion: TiledConfiguration(
        isEnabled: false, tileSize: TiledConfiguration.Size(width: 0, height: 0),
        tileOverlap: 0),
      externalOnDemand: false, alternativeUsesFlashAttention: false, alternativeFilePath: nil,
      alternativeDecoderVersion: nil, memoryCapacity: DeviceCapability.memoryCapacity,
      isNHWCPreferred: DeviceCapability.isNHWCPreferred)
    graph.maxConcurrency = .limit(1)
    var dataFrame = dataFrame
    let cotrainUNet = unetLearningRate.upperBound > 0
    let cotrainTextModel = cotrainTextModel && textModelLearningRate > 0
    if version == .v2 || version == .sdxlBase || version == .sdxlRefiner || version == .sd3
      || version == .pixart
    {
      DynamicGraph.flags = .disableMixedMPSGEMM
    }
    if !DeviceCapability.isMemoryMapBufferSupported {
      DynamicGraph.flags.insert(.disableMmapMTLBuffer)
    }
    let isMFAEnabled = DeviceCapability.isMFAEnabled.load(ordering: .acquiring)
    if !isMFAEnabled {
      DynamicGraph.flags.insert(.disableMetalFlashAttention)
    } else {
      DynamicGraph.flags.remove(.disableMetalFlashAttention)
      if !DeviceCapability.isMFAGEMMFaster {
        DynamicGraph.flags.insert(.disableMFAGEMM)
      }
    }
    graph.openStore(session, flags: .readOnly) { sessionStore in
      guard let tokenizer = tokenizers.first, version != .flux1 else {
        trainFlux1(
          graph: graph, firstStage: firstStage, sessionStore: sessionStore,
          resumingLoRAFile: resumingLoRAFile, dataFrame: dataFrame, trainingSteps: trainingSteps,
          warmupSteps: warmupSteps, gradientAccumulationSteps: gradientAccumulationSteps,
          rankOfLoRA: rankOfLoRA, scaleOfLoRA: scaleOfLoRA, unetLearningRate: unetLearningRate,
          stepsBetweenRestarts: stepsBetweenRestarts, seed: seed, trainableKeys: trainableKeys,
          resolutionDependentShift: resolutionDependentShift, shift: shift,
          noiseOffset: noiseOffset, guidanceEmbed: guidanceEmbed,
          denoisingTimesteps: denoisingTimesteps, captionDropoutRate: captionDropoutRate,
          orthonormalLoRADown: orthonormalLoRADown, powerEMA: powerEMA, memorySaver: memorySaver,
          weightsMemory: weightsMemory, progressHandler: progressHandler)
        return
      }
      let gammas: (lowerBound: Double, upperBound: Double)? = powerEMA.flatMap {
        guard
          let gammaLowerBound = Self.powerFunctionExponentialMovingAverageFromWidthToGamma(
            Double($0.lowerBound)),
          let gammaUpperBound = Self.powerFunctionExponentialMovingAverageFromWidthToGamma(
            Double($0.upperBound))
        else { return nil }
        return (lowerBound: gammaLowerBound, upperBound: gammaUpperBound)
      }
      DynamicGraph.setSeed(seed)
      let positionTensor = graph.variable(.CPU, format: .NHWC, shape: [77], of: Int32.self)
      for i in 0..<77 {
        positionTensor[i] = Int32(i)
      }
      let positionTensorGPU = positionTensor.toGPU(0)
      let causalAttentionMask = graph.variable(.CPU, .NHWC(1, 1, 77, 77), of: FloatType.self)
      causalAttentionMask.full(0)
      for i in 0..<76 {
        for j in (i + 1)..<77 {
          causalAttentionMask[0, 0, i, j] = -FloatType.greatestFiniteMagnitude
        }
      }
      let latentsWidth = Int(scale.widthScale) * 8
      let latentsHeight = Int(scale.heightScale) * 8
      let causalAttentionMaskGPU = causalAttentionMask.toGPU(0)
      let LoRAConfiguration: LoRANetworkConfiguration
      switch memorySaver {
      case .minimal, .balanced:
        LoRAConfiguration = LoRANetworkConfiguration(
          rank: rankOfLoRA, scale: scaleOfLoRA, highPrecision: true, testing: false,
          gradientCheckpointingFeedForward: true, orthonormalDown: orthonormalLoRADown)
      case .speed, .turbo:
        LoRAConfiguration = LoRANetworkConfiguration(
          rank: rankOfLoRA, scale: scaleOfLoRA, highPrecision: true, testing: false,
          orthonormalDown: orthonormalLoRADown)
      }
      let textModel: [Model]
      let textLoRAMapping: [[Int: Int]]
      let unetFixed: Model?
      let unet: ModelBuilder<(width: Int, height: Int)>
      let unetLoRAMapping: [Int: Int]
      let embeddingSize: (Int, Int)
      let timeEmbeddingSize: Int
      let tokenLength: Int
      switch version {
      case .v1:
        embeddingSize = (768, 768)
        timeEmbeddingSize = 320
        tokenLength = 77
        if cotrainTextModel {
          textModel = [
            LoRACLIPTextModel(
              FloatType.self, injectEmbeddings: cotrainCustomEmbedding,
              vocabularySize: 49408, maxLength: 77, maxTokenLength: 77,
              embeddingSize: embeddingSize.0,
              numLayers: 13 - min(max(clipSkip, 1), 12),
              numHeads: 12, batchSize: 1, intermediateSize: 3072, usesFlashAttention: false,
              LoRAConfiguration: LoRAConfiguration)
          ]
        } else {
          textModel = [
            CLIPTextModel(
              FloatType.self, injectEmbeddings: cotrainCustomEmbedding,
              vocabularySize: 49408, maxLength: 77, maxTokenLength: 77,
              embeddingSize: embeddingSize.0,
              numLayers: 13 - min(max(clipSkip, 1), 12),
              numHeads: 12, batchSize: 1, intermediateSize: 3072, usesFlashAttention: false,
              trainable: false
            ).0
          ]
        }
        textLoRAMapping = [LoRAMapping.CLIPTextModel]
        unetFixed = nil
        var configuration = LoRAConfiguration
        if !cotrainUNet {
          configuration.rank = 0
        }
        unet = ModelBuilder { size, _ in
          return LoRAUNet(
            batchSize: 1, embeddingLength: (77, 77), startWidth: size.width,
            startHeight: size.height,
            usesFlashAttention: isMFAEnabled ? .scaleMerged : .none, injectControls: false,
            injectT2IAdapters: false,
            injectIPAdapterLengths: [], LoRAConfiguration: configuration)
        }
        unetLoRAMapping = LoRAMapping.SDUNet
      case .v2:
        embeddingSize = (1024, 1024)
        timeEmbeddingSize = 320
        tokenLength = 77
        if cotrainTextModel {
          textModel = [
            LoRAOpenCLIPTextModel(
              FloatType.self, injectEmbeddings: cotrainCustomEmbedding,
              vocabularySize: 49408, maxLength: 77, maxTokenLength: 77,
              embeddingSize: embeddingSize.0,
              numLayers: 24 - min(max(clipSkip, 1), 23),
              numHeads: 16, batchSize: 1, intermediateSize: 4096,
              usesFlashAttention: false, LoRAConfiguration: LoRAConfiguration)
          ]
        } else {
          textModel = [
            OpenCLIPTextModel(
              FloatType.self, injectEmbeddings: cotrainCustomEmbedding,
              vocabularySize: 49408, maxLength: 77, maxTokenLength: 77,
              embeddingSize: embeddingSize.0,
              numLayers: 24 - min(max(clipSkip, 1), 23),
              numHeads: 16, batchSize: 1, intermediateSize: 4096,
              usesFlashAttention: false, trainable: false
            ).0
          ]
        }
        textLoRAMapping = [LoRAMapping.OpenCLIPTextModel]
        unetFixed = nil
        var configuration = LoRAConfiguration
        if !cotrainUNet {
          configuration.rank = 0
        }
        unet = ModelBuilder { size, _ in
          return LoRAUNetv2(
            batchSize: 1, embeddingLength: (77, 77), startWidth: size.width,
            startHeight: size.height,
            upcastAttention: false, usesFlashAttention: isMFAEnabled ? .scaleMerged : .none,
            injectControls: false,
            LoRAConfiguration: configuration)
        }
        unetLoRAMapping = LoRAMapping.SDUNet
      case .sdxlBase:
        embeddingSize = (1280, 768)
        timeEmbeddingSize = 320
        switch textEncoderVersion {
        case .chatglm3_6b:
          tokenLength = 256
        case nil:
          tokenLength = 77
        }
        if cotrainTextModel {
          textModel = [
            LoRAOpenCLIPTextModel(
              FloatType.self, injectEmbeddings: cotrainCustomEmbedding,
              vocabularySize: 49408, maxLength: 77, maxTokenLength: 77,
              embeddingSize: embeddingSize.0,
              numLayers: 32 - min(max(clipSkip - 1, 0), 30), numHeads: 20, batchSize: 1,
              intermediateSize: 5120, usesFlashAttention: false,
              LoRAConfiguration: LoRAConfiguration,
              outputPenultimate: true
            ),
            LoRACLIPTextModel(
              FloatType.self, injectEmbeddings: cotrainCustomEmbedding,
              vocabularySize: 49408, maxLength: 77, maxTokenLength: 77,
              embeddingSize: embeddingSize.1,
              numLayers: 13 - min(max(clipSkip, 1), 12), numHeads: 12, batchSize: 1,
              intermediateSize: 3072, usesFlashAttention: false,
              LoRAConfiguration: LoRAConfiguration,
              noFinalLayerNorm: true
            ),
          ]
        } else {
          textModel = [
            OpenCLIPTextModel(
              FloatType.self, injectEmbeddings: cotrainCustomEmbedding,
              vocabularySize: 49408, maxLength: 77, maxTokenLength: 77,
              embeddingSize: embeddingSize.0,
              numLayers: 32 - min(max(clipSkip - 1, 0), 30), numHeads: 20, batchSize: 1,
              intermediateSize: 5120, usesFlashAttention: false,
              outputPenultimate: true, trainable: false
            ).0,
            CLIPTextModel(
              FloatType.self, injectEmbeddings: cotrainCustomEmbedding,
              vocabularySize: 49408, maxLength: 77, maxTokenLength: 77,
              embeddingSize: embeddingSize.1,
              numLayers: 13 - min(max(clipSkip, 1), 12), numHeads: 12, batchSize: 1,
              intermediateSize: 3072, usesFlashAttention: false,
              noFinalLayerNorm: true, trainable: false
            ).0,
          ]
        }
        textLoRAMapping = [LoRAMapping.OpenCLIPTextModelG, LoRAMapping.CLIPTextModel]
        var configuration = LoRAConfiguration
        if !cotrainUNet {
          configuration.rank = 0
        }
        unetFixed = LoRAUNetXLFixed(
          batchSize: 1, startHeight: latentsHeight, startWidth: latentsWidth,
          channels: [320, 640, 1280], embeddingLength: (tokenLength, tokenLength),
          inputAttentionRes: [2: [2, 2], 4: [10, 10]], middleAttentionBlocks: 10,
          outputAttentionRes: [2: [2, 2, 2], 4: [10, 10, 10]],
          usesFlashAttention: isMFAEnabled ? .scaleMerged : .none, LoRAConfiguration: configuration)
        unet = ModelBuilder { size, _ in
          return LoRAUNetXL(
            batchSize: 1, startHeight: size.height, startWidth: size.width,
            channels: [320, 640, 1280], inputAttentionRes: [2: [2, 2], 4: [10, 10]],
            middleAttentionBlocks: 10, outputAttentionRes: [2: [2, 2, 2], 4: [10, 10, 10]],
            embeddingLength: (tokenLength, tokenLength), injectIPAdapterLengths: [],
            upcastAttention: ([:], false, [:]),
            usesFlashAttention: isMFAEnabled ? .scaleMerged : .none,
            injectControls: false, LoRAConfiguration: configuration
          ).0
        }
        unetLoRAMapping = LoRAMapping.SDUNetXLBase
      case .sdxlRefiner:
        embeddingSize = (1280, 1280)
        timeEmbeddingSize = 384
        tokenLength = 77
        if cotrainTextModel {
          textModel = [
            LoRAOpenCLIPTextModel(
              FloatType.self, injectEmbeddings: cotrainCustomEmbedding,
              vocabularySize: 49408, maxLength: 77, maxTokenLength: 77,
              embeddingSize: embeddingSize.0,
              numLayers: 32 - min(max(clipSkip - 1, 0), 30), numHeads: 20, batchSize: 1,
              intermediateSize: 5120, usesFlashAttention: false,
              LoRAConfiguration: LoRAConfiguration,
              outputPenultimate: true
            )
          ]
        } else {
          textModel = [
            OpenCLIPTextModel(
              FloatType.self, injectEmbeddings: cotrainCustomEmbedding,
              vocabularySize: 49408, maxLength: 77, maxTokenLength: 77,
              embeddingSize: embeddingSize.0,
              numLayers: 32 - min(max(clipSkip - 1, 0), 30), numHeads: 20, batchSize: 1,
              intermediateSize: 5120, usesFlashAttention: false,
              outputPenultimate: true, trainable: false
            ).0
          ]
        }
        textLoRAMapping = [LoRAMapping.OpenCLIPTextModelG]
        var configuration = LoRAConfiguration
        if !cotrainUNet {
          configuration.rank = 0
        }
        unetFixed = LoRAUNetXLFixed(
          batchSize: 1, startHeight: latentsHeight, startWidth: latentsWidth,
          channels: [384, 768, 1536, 1536], embeddingLength: (77, 77),
          inputAttentionRes: [2: [4, 4], 4: [4, 4]], middleAttentionBlocks: 4,
          outputAttentionRes: [2: [4, 4, 4], 4: [4, 4, 4]],
          usesFlashAttention: isMFAEnabled ? .scaleMerged : .none,
          LoRAConfiguration: configuration)
        unet = ModelBuilder { size, _ in
          return LoRAUNetXL(
            batchSize: 1, startHeight: size.height, startWidth: size.width,
            channels: [384, 768, 1536, 1536], inputAttentionRes: [2: [4, 4], 4: [4, 4]],
            middleAttentionBlocks: 4, outputAttentionRes: [2: [4, 4, 4], 4: [4, 4, 4]],
            embeddingLength: (77, 77), injectIPAdapterLengths: [],
            upcastAttention: ([:], false, [:]),
            usesFlashAttention: isMFAEnabled ? .scaleMerged : .none,
            injectControls: false, LoRAConfiguration: configuration
          ).0
        }
        unetLoRAMapping = LoRAMapping.SDUNetXLRefiner
      case .ssd1b:
        embeddingSize = (1280, 768)
        timeEmbeddingSize = 320
        tokenLength = 77
        if cotrainTextModel {
          textModel = [
            LoRAOpenCLIPTextModel(
              FloatType.self, injectEmbeddings: cotrainCustomEmbedding,
              vocabularySize: 49408, maxLength: 77, maxTokenLength: 77,
              embeddingSize: embeddingSize.0,
              numLayers: 32 - min(max(clipSkip - 1, 0), 30), numHeads: 20, batchSize: 1,
              intermediateSize: 5120, usesFlashAttention: false,
              LoRAConfiguration: LoRAConfiguration,
              outputPenultimate: true
            ),
            LoRACLIPTextModel(
              FloatType.self, injectEmbeddings: cotrainCustomEmbedding,
              vocabularySize: 49408, maxLength: 77, maxTokenLength: 77,
              embeddingSize: embeddingSize.1,
              numLayers: 13 - min(max(clipSkip, 1), 12), numHeads: 12, batchSize: 1,
              intermediateSize: 3072, usesFlashAttention: false,
              LoRAConfiguration: LoRAConfiguration,
              noFinalLayerNorm: true
            ),
          ]
        } else {
          textModel = [
            OpenCLIPTextModel(
              FloatType.self, injectEmbeddings: cotrainCustomEmbedding,
              vocabularySize: 49408, maxLength: 77, maxTokenLength: 77,
              embeddingSize: embeddingSize.0,
              numLayers: 32 - min(max(clipSkip - 1, 0), 30), numHeads: 20, batchSize: 1,
              intermediateSize: 5120, usesFlashAttention: false,
              outputPenultimate: true, trainable: false
            ).0,
            CLIPTextModel(
              FloatType.self, injectEmbeddings: cotrainCustomEmbedding,
              vocabularySize: 49408, maxLength: 77, maxTokenLength: 77,
              embeddingSize: embeddingSize.1,
              numLayers: 13 - min(max(clipSkip, 1), 12), numHeads: 12, batchSize: 1,
              intermediateSize: 3072, usesFlashAttention: false,
              noFinalLayerNorm: true, trainable: false
            ).0,
          ]
        }
        textLoRAMapping = [LoRAMapping.OpenCLIPTextModelG, LoRAMapping.CLIPTextModel]
        var configuration = LoRAConfiguration
        if !cotrainUNet {
          configuration.rank = 0
        }
        unetFixed = LoRAUNetXLFixed(
          batchSize: 1, startHeight: latentsHeight, startWidth: latentsWidth,
          channels: [320, 640, 1280], embeddingLength: (77, 77),
          inputAttentionRes: [2: [2, 2], 4: [4, 4]], middleAttentionBlocks: 0,
          outputAttentionRes: [2: [2, 1, 1], 4: [4, 4, 10]],
          usesFlashAttention: isMFAEnabled ? .scale1 : .none, LoRAConfiguration: configuration)
        unet = ModelBuilder { size, _ in
          return LoRAUNetXL(
            batchSize: 1, startHeight: size.height, startWidth: size.width,
            channels: [320, 640, 1280], inputAttentionRes: [2: [2, 2], 4: [4, 4]],
            middleAttentionBlocks: 0, outputAttentionRes: [2: [2, 1, 1], 4: [4, 4, 10]],
            embeddingLength: (77, 77), injectIPAdapterLengths: [],
            upcastAttention: ([:], false, [:]), usesFlashAttention: isMFAEnabled ? .scale1 : .none,
            injectControls: false,
            LoRAConfiguration: configuration
          ).0
        }
        unetLoRAMapping = LoRAMapping.SDUNetXLSSD1B
      case .sd3, .sd3Large, .pixart, .auraflow, .flux1, .kandinsky21, .svdI2v, .wurstchenStageC,
        .wurstchenStageB, .hunyuanVideo, .wan21_1_3b, .wan21_14b, .hiDreamI1:
        fatalError()
      }
      let externalData: DynamicGraph.Store.Codec =
        weightsMemory == .justInTime ? .externalOnDemand : .externalData
      var textProjection: DynamicGraph.Tensor<FloatType>? = nil
      if cotrainTextModel || cotrainCustomEmbedding {
        if version == .sdxlBase || version == .sdxlRefiner || version == .ssd1b {
          textProjection = graph.constant(.GPU(0), .WC(1280, 1280), of: FloatType.self)
        }
        // Only load text model if it is cotrained.
        textModel[0].maxConcurrency = .limit(1)
        let tokensTensor = graph.variable(.CPU, format: .NHWC, shape: [77], of: Int32.self)
        tokensTensor.full(0)
        let tokensTensorGPU = tokensTensor.toGPU(0)
        if cotrainCustomEmbedding {
          let maskGPU = graph.variable(.GPU(0), .WC(77, 1), of: FloatType.self)
          let injectedEmbeddingsGPU = graph.variable(
            .GPU(0), .WC(77, embeddingSize.0), of: FloatType.self)
          textModel[0].compile(
            inputs: tokensTensorGPU, positionTensorGPU, causalAttentionMaskGPU, maskGPU,
            injectedEmbeddingsGPU)
        } else {
          textModel[0].compile(inputs: tokensTensorGPU, positionTensorGPU, causalAttentionMaskGPU)
        }
        graph.openStore(
          ModelZoo.filePathForModelDownloaded(textEncoder), flags: .readOnly,
          externalStore: TensorData.externalStore(
            filePath: ModelZoo.filePathForModelDownloaded(textEncoder))
        ) {
          store in
          if let textProjection = textProjection {
            store.read(
              "text_projection", variable: textProjection,
              codec: [.q6p, .q8p, .ezm7, .fpzip, .externalData])
          }
          store.read(
            "text_model", model: textModel[0],
            codec: [.jit, .q6p, .q8p, .ezm7, .fpzip, externalData]
          ) {
            name, dataType, format, shape in
            if resumeIfPossible && (name.contains("[i-") || name.contains("lora")) {
              if let resumingLoRAFile = resumingLoRAFile?.0, name.contains("lora") {
                if let tensor = try?
                  (graph.openStore(
                    LoRAZoo.filePathForModelDownloaded(resumingLoRAFile), flags: .readOnly
                  ) {
                    return $0.read(
                      Self.originalLoRA(name: name, LoRAMapping: textLoRAMapping[0]),
                      codec: [.q8p, .ezm7, .fpzip, .externalData])
                  }).get()
                {
                  return .final(Tensor<Float>(from: tensor).toCPU())
                }
              } else if let tensor = sessionStore.read("__te0" + name) {
                return .final(Tensor<Float>(tensor).toCPU())
              }
            }
            // Need to handle clip skip.
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
            case .sdxlBase, .sdxlRefiner, .ssd1b:
              if name == "__text_model__[t-\(258 - (min(clipSkip, 31) - 1) * 8)-0]" {
                name = "__text_model__[t-258-0]"
              } else if name == "__text_model__[t-\(258 - (min(clipSkip, 31) - 1) * 8)-1]" {
                name = "__text_model__[t-258-1]"
              }
            case .sd3, .sd3Large, .pixart, .auraflow, .flux1, .kandinsky21, .svdI2v,
              .wurstchenStageC, .wurstchenStageB, .hunyuanVideo, .wan21_1_3b, .wan21_14b,
              .hiDreamI1:
              fatalError()
            }
            if name.contains("lora_up") {
              switch dataType {
              case .Float16:
                #if !((os(macOS) || (os(iOS) && targetEnvironment(macCatalyst))) && (arch(i386) || arch(x86_64)))
                  var tensor = Tensor<Float16>(.CPU, format: format, shape: shape)
                  tensor.withUnsafeMutableBytes {
                    let size = shape.reduce(MemoryLayout<Float16>.size, *)
                    memset($0.baseAddress, 0, size)
                  }
                  return .final(tensor)
                #else
                  break
                #endif
              case .Float32:
                var tensor = Tensor<Float32>(.CPU, format: format, shape: shape)
                tensor.withUnsafeMutableBytes {
                  let size = shape.reduce(MemoryLayout<Float32>.size, *)
                  memset($0.baseAddress, 0, size)
                }
                return .final(tensor)
              case .Float64, .Int32, .Int64, .UInt8:
                fatalError()
              }
            } else if orthonormalLoRADown && name.contains("lora_down") {
              switch dataType {
              case .Float16:
                #if !((os(macOS) || (os(iOS) && targetEnvironment(macCatalyst))) && (arch(i386) || arch(x86_64)))
                  let tensor = Self.randomOrthonormalMatrix(
                    graph: graph, M: shape[0], N: shape[1..<shape.count].reduce(1, *),
                    of: Float16.self)
                  return .final(tensor)
                #else
                  break
                #endif
              case .Float32:
                let tensor = Self.randomOrthonormalMatrix(
                  graph: graph, M: shape[0], N: shape[1..<shape.count].reduce(1, *),
                  of: Float32.self)
                return .final(tensor)
              case .Float64, .Int32, .Int64, .UInt8:
                fatalError()
              }
            }
            return .continue(name)
          }
        }
        if textModel.count > 1, let CLIPEncoder = CLIPEncoder {
          textModel[1].maxConcurrency = .limit(1)
          if cotrainCustomEmbedding {
            let maskGPU = graph.variable(.GPU(0), .WC(77, 1), of: FloatType.self)
            let injectedEmbeddingsGPU = graph.variable(
              .GPU(0), .WC(77, embeddingSize.1), of: FloatType.self)
            textModel[1].compile(
              inputs: tokensTensorGPU, positionTensorGPU, causalAttentionMaskGPU, maskGPU,
              injectedEmbeddingsGPU)
          } else {
            textModel[1].compile(inputs: tokensTensorGPU, positionTensorGPU, causalAttentionMaskGPU)
          }
          graph.openStore(
            ModelZoo.filePathForModelDownloaded(CLIPEncoder), flags: .readOnly,
            externalStore: TensorData.externalStore(
              filePath: ModelZoo.filePathForModelDownloaded(CLIPEncoder))
          ) {
            store in
            store.read(
              "text_model", model: textModel[1],
              codec: [.jit, .q6p, .q8p, .ezm7, .fpzip, externalData]
            ) {
              name, dataType, format, shape in
              if resumeIfPossible && (name.contains("[i-") || name.contains("lora")) {
                if let resumingLoRAFile = resumingLoRAFile?.0, name.contains("lora") {
                  if let tensor = try?
                    (graph.openStore(
                      LoRAZoo.filePathForModelDownloaded(resumingLoRAFile), flags: .readOnly
                    ) {
                      return $0.read(
                        Self.originalLoRA(name: name, LoRAMapping: textLoRAMapping[1]),
                        codec: [.q8p, .ezm7, .fpzip, .externalData])
                    }).get()
                  {
                    return .final(Tensor<Float>(from: tensor).toCPU())
                  }
                } else if let tensor = sessionStore.read("__te1" + name) {
                  return .final(Tensor<Float>(tensor).toCPU())
                }
              }
              // Need to handle clip skip.
              var name = name
              if name == "__text_model__[t-\(98 - (min(clipSkip, 12) - 1) * 8)-0]" {
                name = "__text_model__[t-98-0]"
              } else if name == "__text_model__[t-\(98 - (min(clipSkip, 12) - 1) * 8)-1]" {
                name = "__text_model__[t-98-1]"
              }
              if name.contains("lora_up") {
                switch dataType {
                case .Float16:
                  #if !((os(macOS) || (os(iOS) && targetEnvironment(macCatalyst))) && (arch(i386) || arch(x86_64)))
                    var tensor = Tensor<Float16>(.CPU, format: format, shape: shape)
                    tensor.withUnsafeMutableBytes {
                      let size = shape.reduce(MemoryLayout<Float16>.size, *)
                      memset($0.baseAddress, 0, size)
                    }
                    return .final(tensor)
                  #else
                    break
                  #endif
                case .Float32:
                  var tensor = Tensor<Float32>(.CPU, format: format, shape: shape)
                  tensor.withUnsafeMutableBytes {
                    let size = shape.reduce(MemoryLayout<Float32>.size, *)
                    memset($0.baseAddress, 0, size)
                  }
                  return .final(tensor)
                case .Float64, .Int32, .Int64, .UInt8:
                  fatalError()
                }
              } else if orthonormalLoRADown && name.contains("lora_down") {
                switch dataType {
                case .Float16:
                  #if !((os(macOS) || (os(iOS) && targetEnvironment(macCatalyst))) && (arch(i386) || arch(x86_64)))
                    let tensor = Self.randomOrthonormalMatrix(
                      graph: graph, M: shape[0], N: shape[1..<shape.count].reduce(1, *),
                      of: Float16.self)
                    return .final(tensor)
                  #else
                    break
                  #endif
                case .Float32:
                  let tensor = Self.randomOrthonormalMatrix(
                    graph: graph, M: shape[0], N: shape[1..<shape.count].reduce(1, *),
                    of: Float32.self)
                  return .final(tensor)
                case .Float64, .Int32, .Int64, .UInt8:
                  fatalError()
                }
              }
              return .continue(name)
            }
          }
        }
      }
      let kvs: [DynamicGraph.Tensor<FloatType>]
      if let unetFixed = unetFixed {
        unetFixed.maxConcurrency = .limit(1)
        let crossattn: DynamicGraph.Tensor<FloatType>
        if version == .sdxlBase || version == .ssd1b {
          crossattn = graph.variable(.GPU(0), .HWC(1, tokenLength, 2048), of: FloatType.self)
        } else {
          precondition(version == .sdxlRefiner)
          crossattn = graph.variable(.GPU(0), .HWC(1, tokenLength, 1280), of: FloatType.self)
        }
        crossattn.full(0)
        kvs = unetFixed(inputs: crossattn).map { $0.as(of: FloatType.self) }
      } else {
        kvs = []
      }
      unet.maxConcurrency = .limit(1)
      let ts = timeEmbedding(
        timestep: 0, batchSize: 1, embeddingSize: timeEmbeddingSize, maxPeriod: 10_000
      )
      .toGPU(0)
      let latents: DynamicGraph.Tensor<FloatType>
      let modifier = ModelZoo.modifierForModel(model)
      switch modifier {
      case .inpainting:
        latents = graph.variable(
          .GPU(0), .NHWC(1, latentsHeight, latentsWidth, 9), of: FloatType.self)
      case .editing, .double:
        latents = graph.variable(
          .GPU(0), .NHWC(1, latentsHeight, latentsWidth, 8), of: FloatType.self)
      case .depth, .canny:
        latents = graph.variable(
          .GPU(0), .NHWC(1, latentsHeight, latentsWidth, 5), of: FloatType.self)
      case .none:
        latents = graph.variable(
          .GPU(0), .NHWC(1, latentsHeight, latentsWidth, 4), of: FloatType.self)
      }
      latents.full(0)
      let c: DynamicGraph.Tensor<FloatType>
      switch version {
      case .v1, .v2:
        c = graph.variable(.GPU(0), .HWC(1, 77, embeddingSize.0), of: FloatType.self)
      case .sdxlBase, .ssd1b:
        switch textEncoderVersion {
        case .chatglm3_6b:
          c = graph.variable(.GPU(0), .WC(1, 4096 + 1536), of: FloatType.self)
        case nil:
          c = graph.variable(.GPU(0), .WC(1, 1280 + 1536), of: FloatType.self)
        }
      case .sdxlRefiner:
        c = graph.variable(.GPU(0), .WC(1, 2560), of: FloatType.self)
      case .sd3, .sd3Large, .pixart, .auraflow, .flux1, .kandinsky21, .svdI2v, .wurstchenStageC,
        .wurstchenStageB, .hunyuanVideo, .wan21_1_3b, .wan21_14b, .hiDreamI1:
        fatalError()
      }
      c.full(0)
      switch memorySaver {
      case .minimal, .speed:
        unet.memoryReduction = true
      case .balanced, .turbo:  // For Balanced, we only do gradient checkpointing.
        unet.memoryReduction = false
      }
      guard
        progressHandler(
          .compile, 0,
          LoRATrainerCheckpoint(
            version: version, textModel1: cotrainTextModel ? textModel.first : nil,
            textModel2: cotrainTextModel && textModel.count > 1 ? textModel[1] : nil,
            unetFixed: unetFixed, unet: unet, step: 0))
      else {
        return
      }
      unet.compile(
        (width: latentsWidth, height: latentsHeight),
        inputs: [latents, graph.variable(Tensor<FloatType>(from: ts)), c] + kvs)
      graph.openStore(
        ModelZoo.filePathForModelDownloaded(model), flags: .readOnly,
        externalStore: TensorData.externalStore(
          filePath: ModelZoo.filePathForModelDownloaded(model))
      ) { store in
        if let unetFixed = unetFixed {
          store.read(
            "unet_fixed", model: unetFixed, codec: [.jit, .q6p, .q8p, .ezm7, .fpzip, externalData]
          ) {
            name, dataType, format, shape in
            if resumeIfPossible && (name.contains("[i-") || name.contains("lora")) {
              if let resumingLoRAFile = resumingLoRAFile?.0, name.contains("lora") {
                if let tensor = try?
                  (graph.openStore(
                    LoRAZoo.filePathForModelDownloaded(resumingLoRAFile), flags: .readOnly
                  ) {
                    return $0.read(
                      Self.originalLoRA(name: name, LoRAMapping: nil),
                      codec: [.q8p, .ezm7, .fpzip, .externalData])
                  }).get()
                {
                  return .final(Tensor<Float>(from: tensor).toCPU())
                }
              } else if let tensor = sessionStore.read(name) {
                return .final(Tensor<Float>(tensor).toCPU())
              }
            }
            if name.contains("lora_up") {
              switch dataType {
              case .Float16:
                #if !((os(macOS) || (os(iOS) && targetEnvironment(macCatalyst))) && (arch(i386) || arch(x86_64)))
                  var tensor = Tensor<Float16>(.CPU, format: format, shape: shape)
                  tensor.withUnsafeMutableBytes {
                    let size = shape.reduce(MemoryLayout<Float16>.size, *)
                    memset($0.baseAddress, 0, size)
                  }
                  return .final(tensor)
                #else
                  break
                #endif
              case .Float32:
                var tensor = Tensor<Float32>(.CPU, format: format, shape: shape)
                tensor.withUnsafeMutableBytes {
                  let size = shape.reduce(MemoryLayout<Float32>.size, *)
                  memset($0.baseAddress, 0, size)
                }
                return .final(tensor)
              case .Float64, .Int32, .Int64, .UInt8:
                fatalError()
              }
            } else if orthonormalLoRADown && name.contains("lora_down") {
              switch dataType {
              case .Float16:
                #if !((os(macOS) || (os(iOS) && targetEnvironment(macCatalyst))) && (arch(i386) || arch(x86_64)))
                  let tensor = Self.randomOrthonormalMatrix(
                    graph: graph, M: shape[0], N: shape[1..<shape.count].reduce(1, *),
                    of: Float16.self)
                  return .final(tensor)
                #else
                  break
                #endif
              case .Float32:
                let tensor = Self.randomOrthonormalMatrix(
                  graph: graph, M: shape[0], N: shape[1..<shape.count].reduce(1, *),
                  of: Float32.self)
                return .final(tensor)
              case .Float64, .Int32, .Int64, .UInt8:
                fatalError()
              }
            }
            return .continue(name)
          }
        }
        store.read("unet", model: unet, codec: [.jit, .q6p, .q8p, .ezm7, .fpzip, externalData]) {
          name, dataType, format, shape in
          if resumeIfPossible && (name.contains("[i-") || name.contains("lora")) {
            if let resumingLoRAFile = resumingLoRAFile?.0, name.contains("lora") {
              if let tensor = try?
                (graph.openStore(
                  LoRAZoo.filePathForModelDownloaded(resumingLoRAFile), flags: .readOnly
                ) {
                  return $0.read(
                    Self.originalLoRA(name: name, LoRAMapping: unetLoRAMapping),
                    codec: [.q8p, .ezm7, .fpzip, .externalData])
                }).get()
              {
                return .final(Tensor<Float>(from: tensor).toCPU())
              }
            } else if let tensor = sessionStore.read(name) {
              return .final(Tensor<Float>(tensor).toCPU())
            }
          }
          if name.contains("lora_up") {
            switch dataType {
            case .Float16:
              #if !((os(macOS) || (os(iOS) && targetEnvironment(macCatalyst))) && (arch(i386) || arch(x86_64)))
                var tensor = Tensor<Float16>(.CPU, format: format, shape: shape)
                tensor.withUnsafeMutableBytes {
                  let size = shape.reduce(MemoryLayout<Float16>.size, *)
                  memset($0.baseAddress, 0, size)
                }
                return .final(tensor)
              #else
                break
              #endif
            case .Float32:
              var tensor = Tensor<Float32>(.CPU, format: format, shape: shape)
              tensor.withUnsafeMutableBytes {
                let size = shape.reduce(MemoryLayout<Float32>.size, *)
                memset($0.baseAddress, 0, size)
              }
              return .final(tensor)
            case .Float64, .Int32, .Int64, .UInt8:
              fatalError()
            }
          } else if orthonormalLoRADown && name.contains("lora_down") {
            switch dataType {
            case .Float16:
              #if !((os(macOS) || (os(iOS) && targetEnvironment(macCatalyst))) && (arch(i386) || arch(x86_64)))
                let tensor = Self.randomOrthonormalMatrix(
                  graph: graph, M: shape[0], N: shape[1..<shape.count].reduce(1, *),
                  of: Float16.self)
                return .final(tensor)
              #else
                break
              #endif
            case .Float32:
              let tensor = Self.randomOrthonormalMatrix(
                graph: graph, M: shape[0], N: shape[1..<shape.count].reduce(1, *), of: Float32.self)
              return .final(tensor)
            case .Float64, .Int32, .Int64, .UInt8:
              fatalError()
            }
          }
          return .continue(name)
        }
      }
      var i: Int
      if resumeIfPossible {
        if let resumingStep = resumingLoRAFile?.1 {
          i = resumingStep
        } else {
          let stepTensor = sessionStore.read("current_step").flatMap { Tensor<Int32>(from: $0) }
          i = stepTensor.map { max(Int($0[0]), 0) } ?? 0
        }
      } else {
        i = 0
      }
      var optimizers = [AdamWOptimizer]()
      if cotrainUNet {
        var unetOptimizer = AdamWOptimizer(
          graph, rate: unetLearningRate.upperBound, betas: (0.9, 0.999), decay: 0.001, epsilon: 1e-8
        )
        unetOptimizer.parameters = [unet.parameters]
        optimizers.append(unetOptimizer)
      }
      if cotrainTextModel {
        var textModelOptimizer = AdamWOptimizer(
          graph, rate: textModelLearningRate, betas: (0.9, 0.999), decay: 0.001, epsilon: 1e-8)
        textModelOptimizer.parameters = [textModel[0].parameters]
        optimizers.append(textModelOptimizer)
      }
      let cotrainCustomEmbedding = cotrainCustomEmbedding && customEmbeddingLearningRate > 0
      var cotrainCustomEmbeddingStopped = !cotrainCustomEmbedding
      var customEmbeddings = [DynamicGraph.Tensor<Float>]()
      if cotrainCustomEmbedding {
        let embeddingName: (String, String)
        let firstStd: Float
        switch version {
        case .v1, .v2, .kandinsky21, .svdI2v, .pixart, .auraflow, .flux1, .hunyuanVideo,
          .wan21_1_3b, .wan21_14b, .hiDreamI1:
          embeddingName = ("string_to_param", "string_to_param")
          firstStd = 0.02
        case .sd3, .sd3Large, .sdxlBase, .ssd1b:
          embeddingName = ("string_to_param_clip_g", "string_to_param_clip_l")
          firstStd = 0.01
        case .sdxlRefiner, .wurstchenStageC, .wurstchenStageB:
          embeddingName = ("string_to_param_clip_g", "string_to_param_clip_g")
          firstStd = 0.01
        }
        let embedding0 = graph.variable(
          .GPU(0), .WC(customEmbeddingLength, embeddingSize.0), of: Float.self)
        if resumeIfPossible {
          if let resumingLoRAFile = resumingLoRAFile?.0 {
            graph.openStore(
              LoRAZoo.filePathForModelDownloaded(resumingLoRAFile), flags: .readOnly,
              externalStore: TensorData.externalStore(
                filePath: LoRAZoo.filePathForModelDownloaded(resumingLoRAFile))
            ) {
              if !$0.read(
                embeddingName.0, variable: embedding0, codec: [.q8p, .ezm7, .fpzip, .externalData])
              {
                if !sessionStore.read(
                  embeddingName.0, variable: embedding0,
                  codec: [.q8p, .ezm7, .fpzip, .externalData])
                {
                  embedding0.randn(std: firstStd)
                  print("Initialize custom embedding due to no available one found!")
                }
              }
            }
          } else {
            if !sessionStore.read(
              embeddingName.0, variable: embedding0, codec: [.q8p, .ezm7, .fpzip, .externalData])
            {
              embedding0.randn(std: firstStd)
              print("Initialize custom embedding due to no available one found!")
            }
          }
        } else {
          embedding0.randn(std: firstStd)
        }
        customEmbeddings.append(embedding0)
        if version == .sdxlBase || version == .ssd1b {
          let embedding1 = graph.variable(
            .GPU(0), .WC(customEmbeddingLength, embeddingSize.1), of: Float.self)
          if resumeIfPossible {
            if let resumingLoRAFile = resumingLoRAFile?.0 {
              graph.openStore(
                LoRAZoo.filePathForModelDownloaded(resumingLoRAFile), flags: .readOnly,
                externalStore: TensorData.externalStore(
                  filePath: LoRAZoo.filePathForModelDownloaded(resumingLoRAFile))
              ) {
                if !$0.read(
                  embeddingName.1, variable: embedding1,
                  codec: [.q8p, .ezm7, .fpzip, .externalData])
                {
                  if !sessionStore.read(
                    embeddingName.1, variable: embedding1,
                    codec: [.q8p, .ezm7, .fpzip, .externalData])
                  {
                    embedding1.randn(std: 0.02)
                    print("Initialize custom embedding due to no available one found!")
                  }
                }
              }
            } else {
              if !sessionStore.read(
                embeddingName.1, variable: embedding1, codec: [.q8p, .ezm7, .fpzip, .externalData])
              {
                embedding1.randn(std: 0.02)
                print("Initialize custom embedding due to no available one found!")
              }
            }
          } else {
            embedding1.randn(std: 0.02)
          }
          customEmbeddings.append(embedding1)
        }
        if i >= stopEmbeddingTrainingAtStep {
          // Not going to optimize this.
          customEmbeddings.forEach { $0.detach() }
          cotrainCustomEmbeddingStopped = true
        } else {
          var customEmbeddingOptimizer = AdamWOptimizer(
            graph, rate: customEmbeddingLearningRate, betas: (0.9, 0.999), decay: 0.001,
            epsilon: 1e-8
          )
          customEmbeddingOptimizer.parameters = customEmbeddings
          optimizers.append(customEmbeddingOptimizer)
        }
      }
      let discretization = Denoiser.LinearDiscretization(
        Denoiser.Parameterization.ddpm(
          .init(linearStart: 0.00085, linearEnd: 0.012, timesteps: 1_000, linspace: .linearWrtSigma)
        ), objective: .epsilon)
      let alphasCumprod = discretization.alphasCumprod
      let minSNRGamma: Double = 1
      var scaler = GradScaler(scale: 32_768)
      var stopped = false
      var batchCount = 0
      let latentZeros: DynamicGraph.Tensor<FloatType>?
      if modifier == .inpainting {
        latentZeros = sessionStore.read("latent_zeros").map {
          graph.constant(Tensor<FloatType>(from: $0).toGPU(0))
        }
      } else {
        latentZeros = nil
      }
      var sfmt = SFMT(seed: UInt64(seed))
      // Do a dry-run to allocate everything properly. This will make sure we allocate largest RAM so no reallocation later.
      let _ = unet(
        (width: latentsWidth, height: latentsHeight), inputs: latents,
        [graph.variable(Tensor<FloatType>(from: ts)), c] + kvs)
      guard
        progressHandler(
          .step(0), 0,
          LoRATrainerCheckpoint(
            version: version, textModel1: cotrainTextModel ? textModel.first : nil,
            textModel2: cotrainTextModel && textModel.count > 1 ? textModel[1] : nil,
            unetFixed: unetFixed, unet: unet, step: 0))
      else {
        return
      }
      var textModel1EMALowerBoundWeights = [String: Tensor<Float>]()
      var textModel1EMAUpperBoundWeights = [String: Tensor<Float>]()
      var textModel2EMALowerBoundWeights = [String: Tensor<Float>]()
      var textModel2EMAUpperBoundWeights = [String: Tensor<Float>]()
      var unetFixedEMALowerBoundWeights = [String: Tensor<Float>]()
      var unetFixedEMAUpperBoundWeights = [String: Tensor<Float>]()
      var unetEMALowerBoundWeights = [String: Tensor<Float>]()
      var unetEMAUpperBoundWeights = [String: Tensor<Float>]()
      var textEmbedding1LowerBound: Tensor<Float>? = nil
      var textEmbedding1UpperBound: Tensor<Float>? = nil
      var textEmbedding2LowerBound: Tensor<Float>? = nil
      var textEmbedding2UpperBound: Tensor<Float>? = nil
      while i < trainingSteps && !stopped && !optimizers.isEmpty {
        dataFrame.shuffle()
        for value in dataFrame["0", "imagePath", "tokens"] {
          guard let input = value[0] as? ProcessedInput, let imagePath = value[1] as? String,
            var tokens = value[2] as? [Int32]
          else {
            continue
          }
          let loadedImagePath = input.loadedImagePath
          guard let tensor = sessionStore.read(loadedImagePath) else { continue }
          let shape = tensor.shape
          let latentsHeight = shape[1]
          let latentsWidth = shape[2]
          let captionDropout = Float.random(in: 0...1, using: &sfmt) < captionDropoutRate
          if captionDropout {
            tokens = zeroCaption.tokens
          }
          let (latents, _) = graph.withNoGrad {
            let parameters = graph.variable(Tensor<FloatType>(from: tensor).toGPU(0))
            let noise = graph.variable(
              .CPU, .NHWC(1, latentsHeight, latentsWidth, 4), of: Float.self)
            noise.randn(std: 1, mean: 0)
            let noiseGPU = DynamicGraph.Tensor<FloatType>(from: noise.toGPU(0))
            return firstStage.sampleFromDistribution(parameters, noise: noiseGPU)
          }
          let noiseGPU = graph.withNoGrad {
            var noise = graph.variable(
              .CPU, .NHWC(1, latentsHeight, latentsWidth, 4), of: Float.self)
            noise.randn(std: 1, mean: 0)
            if noiseOffset > 0 {
              let offsetNoise = graph.variable(.CPU, .NHWC(1, 1, 1, 1), of: Float.self)
              offsetNoise.randn(std: noiseOffset, mean: 0)
              noise = noise + offsetNoise
            }
            return DynamicGraph.Tensor<FloatType>(from: noise.toGPU(0))
          }
          let timestep = Int.random(in: denoisingTimesteps, using: &sfmt)
          let sqrtAlphasCumprod = Float(alphasCumprod[timestep].squareRoot())
          let sqrtOneMinusAlphasCumprod = Float((1 - alphasCumprod[timestep]).squareRoot())
          let snr = alphasCumprod[timestep] / (1 - alphasCumprod[timestep])
          let gammaOverSNR = minSNRGamma / snr
          let snrWeight = Float(min(gammaOverSNR, 1))
          let noisyLatents = graph.withNoGrad {
            let noisyLatents = sqrtAlphasCumprod * latents + sqrtOneMinusAlphasCumprod * noiseGPU
            guard modifier == .inpainting else {
              return noisyLatents
            }
            var noisyLatentsPlusMaskAndImage = graph.variable(
              .GPU(0), .NHWC(1, latentsHeight, latentsWidth, 9), of: FloatType.self)
            // Need to do more in case of inpainting model.
            noisyLatentsPlusMaskAndImage[0..<1, 0..<latentsHeight, 0..<latentsWidth, 0..<4] =
              noisyLatents
            let mask = graph.variable(
              makeMask(resolution: (width: latentsWidth, height: latentsHeight), using: &sfmt)
                .toGPU(
                  0))
            noisyLatentsPlusMaskAndImage[0..<1, 0..<latentsHeight, 0..<latentsWidth, 4..<5] = mask
            // This is an approximation. The mask should be applied at image space, but that means we need to keep VAE in RAM. This allows us to mask out and fill in some zero latents (zero in image space).
            var conditioningImage = (1 - mask) .* latents
            if let latentZeros = latentZeros {
              conditioningImage = conditioningImage + (mask .* latentZeros)
            }
            noisyLatentsPlusMaskAndImage[0..<1, 0..<latentsHeight, 0..<latentsWidth, 5..<9] =
              conditioningImage
            return noisyLatentsPlusMaskAndImage
          }
          let ts = timeEmbedding(
            timestep: Float(timestep), batchSize: 1, embeddingSize: timeEmbeddingSize,
            maxPeriod: 10_000
          ).toGPU(0)
          let t = graph.variable(Tensor<FloatType>(from: ts))
          let condTokensTensorGPU: DynamicGraph.Tensor<Int32>?
          let kvs: [DynamicGraph.Tensor<FloatType>]
          let c: DynamicGraph.Tensor<FloatType>
          var trainableEmbeddings = cotrainCustomEmbeddingStopped ? [] : customEmbeddings
          var injectedEmbeddings = [DynamicGraph.Tensor<FloatType>]()
          if cotrainTextModel || cotrainCustomEmbedding {
            let tokensTensor = graph.variable(.CPU, format: .NHWC, shape: [77], of: Int32.self)
            for i in 0..<77 {
              tokensTensor[i] =
                tokenizer.isTextualInversion(tokens[i]) ? tokenizer.unknownToken : tokens[i]
            }
            let tokenMaskGPU: DynamicGraph.Tensor<FloatType>?
            if cotrainCustomEmbedding {
              let tokenMask = graph.variable(.CPU, .WC(77, 1), of: FloatType.self)
              for i in 0..<77 {
                tokenMask[i, 0] = tokenizer.isTextualInversion(tokens[i]) ? 0 : 1
              }
              tokenMaskGPU = tokenMask.toGPU(0)
              var hasTrainableEmbeddings = false
              switch version {
              case .v1, .v2, .sdxlRefiner:
                var injectedEmbedding = graph.variable(
                  .GPU(0), .WC(77, embeddingSize.0), of: FloatType.self)
                injectedEmbedding.full(0)
                var i = 0
                while i < 77 {
                  guard tokenizer.isTextualInversion(tokens[i]) else {
                    i += 1
                    continue
                  }
                  if i + customEmbeddingLength <= 77 {
                    injectedEmbedding[i..<(i + customEmbeddingLength), 0..<embeddingSize.0] =
                      DynamicGraph.Tensor<FloatType>(from: customEmbeddings[0])
                  } else {
                    injectedEmbedding[i..<77, 0..<embeddingSize.0] = DynamicGraph.Tensor<FloatType>(
                      from: customEmbeddings[0][
                        0..<(i + customEmbeddingLength - 77), 0..<embeddingSize.0])
                  }
                  i += customEmbeddingLength
                  hasTrainableEmbeddings = true
                }
                injectedEmbeddings.append(injectedEmbedding)
              case .sdxlBase, .ssd1b:
                var injectedEmbedding0 = graph.variable(
                  .GPU(0), .WC(77, embeddingSize.0), of: FloatType.self)
                injectedEmbedding0.full(0)
                var injectedEmbedding1 = graph.variable(
                  .GPU(0), .WC(77, embeddingSize.1), of: FloatType.self)
                injectedEmbedding1.full(0)
                var i = 0
                while i < 77 {
                  guard tokenizer.isTextualInversion(tokens[i]) else {
                    i += 1
                    continue
                  }
                  if i + customEmbeddingLength <= 77 {
                    injectedEmbedding0[i..<(i + customEmbeddingLength), 0..<embeddingSize.0] =
                      DynamicGraph.Tensor<FloatType>(from: customEmbeddings[0])
                    injectedEmbedding1[i..<(i + customEmbeddingLength), 0..<embeddingSize.1] =
                      DynamicGraph.Tensor<FloatType>(from: customEmbeddings[1])
                  } else {
                    injectedEmbedding0[i..<77, 0..<embeddingSize.0] = DynamicGraph.Tensor<
                      FloatType
                    >(
                      from: customEmbeddings[0][
                        0..<(i + customEmbeddingLength - 77), 0..<embeddingSize.0])
                    injectedEmbedding1[i..<77, 0..<embeddingSize.1] = DynamicGraph.Tensor<
                      FloatType
                    >(
                      from: customEmbeddings[1][
                        0..<(i + customEmbeddingLength - 77), 0..<embeddingSize.1])
                  }
                  i += customEmbeddingLength
                  hasTrainableEmbeddings = true
                }
                injectedEmbeddings.append(injectedEmbedding0)
                injectedEmbeddings.append(injectedEmbedding1)
              case .sd3, .sd3Large, .pixart, .auraflow, .flux1, .kandinsky21, .svdI2v,
                .wurstchenStageC, .wurstchenStageB, .hunyuanVideo, .wan21_1_3b, .wan21_14b,
                .hiDreamI1:
                fatalError()
              }
              if !hasTrainableEmbeddings {
                trainableEmbeddings.removeAll()
              }
            } else {
              tokenMaskGPU = nil
            }
            let tokensTensorGPU = tokensTensor.toGPU(0)
            switch version {
            case .v1, .v2:
              if let tokenMaskGPU = tokenMaskGPU, let injectedEmbedding = injectedEmbeddings.first {
                c = textModel[0](
                  inputs: tokensTensorGPU, positionTensorGPU, causalAttentionMaskGPU, tokenMaskGPU,
                  injectedEmbedding)[
                    0
                  ]
                  .as(of: FloatType.self).reshaped(.HWC(1, 77, embeddingSize.0))
              } else {
                c = textModel[0](
                  inputs: tokensTensorGPU, positionTensorGPU, causalAttentionMaskGPU)[
                    0
                  ]
                  .as(of: FloatType.self).reshaped(.HWC(1, 77, embeddingSize.0))
              }
              kvs = []
            case .sdxlBase, .ssd1b:
              let textEncoding: [DynamicGraph.Tensor<FloatType>]
              if let tokenMaskGPU = tokenMaskGPU, let injectedEmbedding = injectedEmbeddings.first {
                textEncoding = textModel[0](
                  inputs: tokensTensorGPU, positionTensorGPU, causalAttentionMaskGPU, tokenMaskGPU,
                  injectedEmbedding
                ).map { $0.as(of: FloatType.self) }
              } else {
                textEncoding = textModel[0](
                  inputs: tokensTensorGPU, positionTensorGPU, causalAttentionMaskGPU
                ).map { $0.as(of: FloatType.self) }
              }
              var pooled = graph.variable(.GPU(0), .WC(1, embeddingSize.0), of: FloatType.self)
              var tokenEnd: Int? = nil
              for i in 0..<77 {
                if tokens[i] == 49407 && tokenEnd == nil {
                  tokenEnd = i
                }
              }
              if let tokenEnd = tokenEnd, let textProjection = textProjection {
                pooled[0..<1, 0..<embeddingSize.0] =
                  textEncoding[1][(tokenEnd)..<(tokenEnd + 1), 0..<embeddingSize.0] * textProjection
              }
              let CLIPEncoding: DynamicGraph.Tensor<FloatType>
              if let tokenMaskGPU = tokenMaskGPU, let injectedEmbedding = injectedEmbeddings.last {
                CLIPEncoding = textModel[1](
                  inputs: tokensTensorGPU, positionTensorGPU, causalAttentionMaskGPU, tokenMaskGPU,
                  injectedEmbedding)[0].as(
                    of: FloatType.self
                  ).reshaped(.HWC(1, 77, 768))
              } else {
                CLIPEncoding = textModel[1](
                  inputs: tokensTensorGPU, positionTensorGPU, causalAttentionMaskGPU)[0].as(
                    of: FloatType.self
                  ).reshaped(.HWC(1, 77, 768))
              }
              let unetFixEncoder = UNetFixedEncoder<FloatType>(
                filePath: "", version: version, dualAttentionLayers: [], usesFlashAttention: true,
                zeroNegativePrompt: false, isQuantizedModel: false, canRunLoRASeparately: false,
                externalOnDemand: false)
              c =
                unetFixEncoder.vector(
                  textEmbedding: pooled, originalSize: input.originalSize,
                  cropTopLeft: input.cropTopLeft, targetSize: input.targetSize, aestheticScore: 6,
                  negativeOriginalSize: input.originalSize, negativeAestheticScore: 2.5, fpsId: 5,
                  motionBucketId: 127, condAug: 0.02)[0]
              var crossattn = graph.variable(.GPU(0), .HWC(1, 77, 2048), of: FloatType.self)
              crossattn[0..<1, 0..<77, 0..<768] = CLIPEncoding
              crossattn[0..<1, 0..<77, 768..<2048] = textEncoding[0].reshaped(.HWC(1, 77, 1280))
              if let unetFixed = unetFixed {
                kvs = unetFixed(inputs: crossattn).map { $0.as(of: FloatType.self) }
              } else {
                kvs = []
              }
            case .sdxlRefiner:
              let textEncoding: [DynamicGraph.Tensor<FloatType>]
              if let tokenMaskGPU = tokenMaskGPU, let injectedEmbedding = injectedEmbeddings.first {
                textEncoding = textModel[0](
                  inputs: tokensTensorGPU, positionTensorGPU, causalAttentionMaskGPU, tokenMaskGPU,
                  injectedEmbedding
                ).map { $0.as(of: FloatType.self) }
              } else {
                textEncoding = textModel[0](
                  inputs: tokensTensorGPU, positionTensorGPU, causalAttentionMaskGPU
                ).map { $0.as(of: FloatType.self) }
              }
              var pooled = graph.variable(.GPU(0), .WC(1, embeddingSize.0), of: FloatType.self)
              var tokenEnd: Int? = nil
              for i in 0..<77 {
                if tokens[i] == 49407 && tokenEnd == nil {
                  tokenEnd = i
                }
              }
              if let tokenEnd = tokenEnd, let textProjection = textProjection {
                pooled[0..<1, 0..<embeddingSize.0] =
                  textEncoding[1][(tokenEnd)..<(tokenEnd + 1), 0..<embeddingSize.0] * textProjection
              }
              let unetFixEncoder = UNetFixedEncoder<FloatType>(
                filePath: "", version: .sdxlRefiner, dualAttentionLayers: [],
                usesFlashAttention: true,
                zeroNegativePrompt: false, isQuantizedModel: false, canRunLoRASeparately: false,
                externalOnDemand: false)
              c =
                unetFixEncoder.vector(
                  textEmbedding: pooled, originalSize: input.originalSize,
                  cropTopLeft: input.cropTopLeft, targetSize: input.targetSize, aestheticScore: 6,
                  negativeOriginalSize: input.originalSize, negativeAestheticScore: 2.5, fpsId: 5,
                  motionBucketId: 127, condAug: 0.02)[0]
              let crossattn = textEncoding[0].reshaped(.HWC(1, 77, 1280))
              if let unetFixed = unetFixed {
                kvs = unetFixed(inputs: crossattn).map { $0.as(of: FloatType.self) }
              } else {
                kvs = []
              }
            case .sd3, .sd3Large, .pixart, .auraflow, .flux1, .kandinsky21, .svdI2v,
              .wurstchenStageC, .wurstchenStageB, .hunyuanVideo, .wan21_1_3b, .wan21_14b,
              .hiDreamI1:
              fatalError()
            }
            condTokensTensorGPU = tokensTensorGPU
          } else {
            let textEncodingPath = captionDropout ? "" : imagePath
            guard let tensor = sessionStore.read("cond_\(textEncodingPath)") else { continue }
            switch version {
            case .v1, .v2:
              c = graph.variable(Tensor<FloatType>(from: tensor).toGPU(0))
              kvs = []
            case .sdxlBase, .ssd1b:
              guard let pooled = sessionStore.read("pool_\(textEncodingPath)") else { continue }
              switch textEncoderVersion {
              case .chatglm3_6b:
                let unetFixEncoder = UNetFixedEncoder<FloatType>(
                  filePath: "", version: version, dualAttentionLayers: [], usesFlashAttention: true,
                  zeroNegativePrompt: false, isQuantizedModel: false, canRunLoRASeparately: false,
                  externalOnDemand: false)
                c =
                  unetFixEncoder.vector(
                    textEmbedding: graph.variable(Tensor<FloatType>(from: pooled).toGPU(0)),
                    originalSize: input.originalSize, cropTopLeft: input.cropTopLeft,
                    targetSize: input.targetSize, aestheticScore: 6,
                    negativeOriginalSize: input.originalSize, negativeAestheticScore: 2.5, fpsId: 5,
                    motionBucketId: 127, condAug: 0.02)[0]
                let crossattn = graph.variable(Tensor<FloatType>(from: tensor).toGPU(0))
                if let unetFixed = unetFixed {
                  kvs = unetFixed(inputs: crossattn).map { $0.as(of: FloatType.self) }
                } else {
                  kvs = []
                }
              case nil:
                guard let te2 = sessionStore.read("cond_te2_\(textEncodingPath)")
                else { continue }
                let unetFixEncoder = UNetFixedEncoder<FloatType>(
                  filePath: "", version: version, dualAttentionLayers: [], usesFlashAttention: true,
                  zeroNegativePrompt: false, isQuantizedModel: false, canRunLoRASeparately: false,
                  externalOnDemand: false)
                c =
                  unetFixEncoder.vector(
                    textEmbedding: graph.variable(Tensor<FloatType>(from: pooled).toGPU(0)),
                    originalSize: input.originalSize, cropTopLeft: input.cropTopLeft,
                    targetSize: input.targetSize, aestheticScore: 6,
                    negativeOriginalSize: input.originalSize, negativeAestheticScore: 2.5, fpsId: 5,
                    motionBucketId: 127, condAug: 0.02)[0]
                var crossattn = graph.variable(.GPU(0), .HWC(1, 77, 2048), of: FloatType.self)
                crossattn[0..<1, 0..<77, 0..<768] = graph.variable(
                  Tensor<FloatType>(from: te2).toGPU(0))
                crossattn[0..<1, 0..<77, 768..<2048] = graph.variable(
                  Tensor<FloatType>(from: tensor).toGPU(0))
                if let unetFixed = unetFixed {
                  kvs = unetFixed(inputs: crossattn).map { $0.as(of: FloatType.self) }
                } else {
                  kvs = []
                }
              }
            case .sdxlRefiner:
              guard let pooled = sessionStore.read("pool_\(textEncodingPath)") else { continue }
              let unetFixEncoder = UNetFixedEncoder<FloatType>(
                filePath: "", version: .sdxlRefiner, dualAttentionLayers: [],
                usesFlashAttention: true,
                zeroNegativePrompt: false, isQuantizedModel: false, canRunLoRASeparately: false,
                externalOnDemand: false)
              c =
                unetFixEncoder.vector(
                  textEmbedding: graph.variable(Tensor<FloatType>(from: pooled).toGPU(0)),
                  originalSize: input.originalSize, cropTopLeft: input.cropTopLeft,
                  targetSize: input.targetSize, aestheticScore: 6,
                  negativeOriginalSize: input.originalSize, negativeAestheticScore: 2.5, fpsId: 5,
                  motionBucketId: 127, condAug: 0.02)[0]
              let crossattn = graph.variable(Tensor<FloatType>(from: tensor).toGPU(0))
              if let unetFixed = unetFixed {
                kvs = unetFixed(inputs: crossattn).map { $0.as(of: FloatType.self) }
              } else {
                kvs = []
              }
            case .sd3, .sd3Large, .pixart, .auraflow, .flux1, .kandinsky21, .svdI2v,
              .wurstchenStageC, .wurstchenStageB, .hunyuanVideo, .wan21_1_3b, .wan21_14b,
              .hiDreamI1:
              fatalError()
            }
            condTokensTensorGPU = nil
          }
          let et = unet(
            (width: latentsWidth, height: latentsHeight), inputs: noisyLatents, [t, c] + kvs)[0].as(
              of: FloatType.self)
          let d = et - noiseGPU
          let loss = snrWeight * (d .* d).reduced(.mean, axis: [1, 2, 3])
          if let condTokensTensorGPU = condTokensTensorGPU {
            if trainableEmbeddings.isEmpty {
              scaler.scale(loss).backward(to: [noisyLatents, condTokensTensorGPU])
            } else {
              scaler.scale(loss).backward(to: [noisyLatents] + trainableEmbeddings)
            }
          } else {
            scaler.scale(loss).backward(to: [noisyLatents])
          }
          let value = loss.toCPU()[0, 0, 0, 0]
          print(
            "loss \(value), scale \(scaler.scale), step \(i), image \(latentsWidth)x\(latentsHeight)"
          )
          batchCount += 1
          let learningRate: Float
          if stepsBetweenRestarts > 1 {
            learningRate =
              unetLearningRate.lowerBound + 0.5
              * (unetLearningRate.upperBound - unetLearningRate.lowerBound)
              * (1
                + cos(
                  (Float(i % (stepsBetweenRestarts - 1)) / Float(stepsBetweenRestarts - 1)) * .pi))
          } else {
            learningRate = unetLearningRate.upperBound
          }
          if (i + 1) < warmupSteps {
            if cotrainUNet {
              optimizers[0].rate = learningRate * (Float(i + 1) / Float(warmupSteps))
              if optimizers.count > 1 {
                if cotrainTextModel {
                  optimizers[1].rate = textModelLearningRate * (Float(i + 1) / Float(warmupSteps))
                } else {
                  optimizers[1].rate =
                    customEmbeddingLearningRate * (Float(i + 1) / Float(warmupSteps))
                }
              }
              if optimizers.count > 2 {
                optimizers[2].rate =
                  customEmbeddingLearningRate * (Float(i + 1) / Float(warmupSteps))
              }
            } else {
              if cotrainTextModel {
                optimizers[0].rate = textModelLearningRate * (Float(i + 1) / Float(warmupSteps))
              } else {
                optimizers[0].rate =
                  customEmbeddingLearningRate * (Float(i + 1) / Float(warmupSteps))
              }
              if optimizers.count > 1 {
                optimizers[1].rate =
                  customEmbeddingLearningRate * (Float(i + 1) / Float(warmupSteps))
              }
            }
          } else {
            if cotrainUNet {
              optimizers[0].rate = learningRate
              if optimizers.count > 1 {
                if cotrainTextModel {
                  optimizers[1].rate = textModelLearningRate
                } else {
                  optimizers[1].rate = customEmbeddingLearningRate
                }
              }
              if optimizers.count > 2 {
                optimizers[2].rate = customEmbeddingLearningRate
              }
            } else {
              if cotrainTextModel {
                optimizers[0].rate = textModelLearningRate
              } else {
                optimizers[0].rate = customEmbeddingLearningRate
              }
              if optimizers.count > 1 {
                optimizers[1].rate = customEmbeddingLearningRate
              }
            }
          }
          if let summaryWriter = summaryWriter {
            summaryWriter.addScalar("loss", value, step: i)
            summaryWriter.addScalar("scale", scaler.scale, step: i)
            summaryWriter.addScalar("timestep", Float(timestep), step: i)
            summaryWriter.addScalar("latents_width", Float(latentsWidth), step: i)
            summaryWriter.addScalar("latents_height", Float(latentsHeight), step: i)
            if cotrainUNet {
              summaryWriter.addScalar("unet_learning_rate", optimizers[0].rate, step: i)
              if optimizers.count > 1 {
                if cotrainTextModel {
                  summaryWriter.addScalar("text_model_learning_rate", optimizers[1].rate, step: i)
                } else {
                  summaryWriter.addScalar(
                    "text_embedding_learning_rate", optimizers[1].rate, step: i)
                }
              }
              if optimizers.count > 2 {
                summaryWriter.addScalar("text_embedding_learning_rate", optimizers[2].rate, step: i)
              }
            } else {
              if cotrainTextModel {
                summaryWriter.addScalar("text_model_learning_rate", optimizers[0].rate, step: i)
              } else {
                summaryWriter.addScalar("text_embedding_learning_rate", optimizers[0].rate, step: i)
              }
              if optimizers.count > 1 {
                summaryWriter.addScalar("text_embedding_learning_rate", optimizers[1].rate, step: i)
              }
            }
          }
          if batchCount == gradientAccumulationSteps {
            // Update the LoRA.
            scaler.step(&optimizers)
            batchCount = 0
            if let gammas = gammas, let optimizer = optimizers.first {  // If we need to compute ema.
              let betaLowerBound = Float(
                pow((1 - 1 / Double(optimizer.step)), gammas.lowerBound + 1))
              let betaUpperBound = Float(
                pow((1 - 1 / Double(optimizer.step)), gammas.upperBound + 1))
              graph.withNoGrad {
                func computeEMA(
                  _ model: AnyModel, emaLowerBoundWeights: inout [String: Tensor<Float>],
                  emaUpperBoundWeights: inout [String: Tensor<Float>]
                ) {
                  let loraUps = model.parameters.filter(where: { $0.contains("lora_up") })
                  let loraDowns = model.parameters.filter(where: { $0.contains("lora_down") })
                  let parameters = loraUps + loraDowns
                  for parameter in parameters {
                    let name = parameter.name
                    let value = parameter.copied(Float.self).toGPU(0)
                    if let oldEma1 = emaLowerBoundWeights[name],
                      let oldEma2 = emaUpperBoundWeights[name]
                    {
                      let value = graph.variable(value)
                      emaLowerBoundWeights[name] =
                        Functional.add(
                          left: graph.variable(oldEma1), right: value, leftScalar: betaLowerBound,
                          rightScalar: 1 - betaLowerBound
                        ).rawValue
                      emaUpperBoundWeights[name] =
                        Functional.add(
                          left: graph.variable(oldEma2), right: value, leftScalar: betaUpperBound,
                          rightScalar: 1 - betaUpperBound
                        ).rawValue
                    } else {
                      emaLowerBoundWeights[name] = value
                      emaUpperBoundWeights[name] = value
                    }
                  }
                }
                if cotrainUNet {
                  computeEMA(
                    unet, emaLowerBoundWeights: &unetEMALowerBoundWeights,
                    emaUpperBoundWeights: &unetEMAUpperBoundWeights)
                  if let unetFixed = unetFixed {
                    computeEMA(
                      unetFixed, emaLowerBoundWeights: &unetFixedEMALowerBoundWeights,
                      emaUpperBoundWeights: &unetFixedEMAUpperBoundWeights)
                  }
                }
                if cotrainTextModel, let textModel1 = textModel.first {
                  computeEMA(
                    textModel1, emaLowerBoundWeights: &textModel1EMALowerBoundWeights,
                    emaUpperBoundWeights: &textModel1EMAUpperBoundWeights)
                  if textModel.count > 1 {
                    let textModel2 = textModel[1]
                    computeEMA(
                      textModel2, emaLowerBoundWeights: &textModel2EMALowerBoundWeights,
                      emaUpperBoundWeights: &textModel2EMAUpperBoundWeights)
                  }
                }
                if cotrainCustomEmbedding && !cotrainCustomEmbeddingStopped,
                  let textEmbedding1 = customEmbeddings.first
                {
                  if let oldEma1 = textEmbedding1LowerBound, let oldEma2 = textEmbedding1UpperBound
                  {
                    textEmbedding1LowerBound =
                      Functional.add(
                        left: graph.variable(oldEma1), right: textEmbedding1,
                        leftScalar: betaLowerBound,
                        rightScalar: 1 - betaLowerBound
                      ).rawValue
                    textEmbedding1UpperBound =
                      Functional.add(
                        left: graph.variable(oldEma2), right: textEmbedding1,
                        leftScalar: betaUpperBound,
                        rightScalar: 1 - betaUpperBound
                      ).rawValue
                  } else {
                    textEmbedding1LowerBound = textEmbedding1.rawValue
                    textEmbedding1UpperBound = textEmbedding1LowerBound
                  }
                  if customEmbeddings.count > 1 {
                    let textEmbedding2 = customEmbeddings[1]
                    if let oldEma1 = textEmbedding2LowerBound,
                      let oldEma2 = textEmbedding2UpperBound
                    {
                      textEmbedding2LowerBound =
                        Functional.add(
                          left: graph.variable(oldEma1), right: textEmbedding2,
                          leftScalar: betaLowerBound,
                          rightScalar: 1 - betaLowerBound
                        ).rawValue
                      textEmbedding2UpperBound =
                        Functional.add(
                          left: graph.variable(oldEma2), right: textEmbedding2,
                          leftScalar: betaUpperBound,
                          rightScalar: 1 - betaUpperBound
                        ).rawValue
                    } else {
                      textEmbedding2LowerBound = textEmbedding2.rawValue
                      textEmbedding2UpperBound = textEmbedding2LowerBound
                    }
                  }
                }
              }
            }
            if cotrainCustomEmbedding && !cotrainCustomEmbeddingStopped
              && i + 1 >= stopEmbeddingTrainingAtStep
            {
              customEmbeddings.forEach { $0.detach() }
              // Drop the optimizer.
              optimizers = optimizers.dropLast()
              cotrainCustomEmbeddingStopped = true
              if optimizers.isEmpty {
                let _ = progressHandler(
                  .step(i + 1), Float(value),
                  LoRATrainerCheckpoint(
                    version: version, textModel1: cotrainTextModel ? textModel.first : nil,
                    textModel2: cotrainTextModel && textModel.count > 1 ? textModel[1] : nil,
                    unetFixed: unetFixed, unet: unet, textEmbedding1: customEmbeddings.first,
                    textEmbedding2: customEmbeddings.count > 1 ? customEmbeddings[1] : nil,
                    step: i + 1,
                    exponentialMovingAverageLowerBound:
                      LoRATrainerCheckpoint.ExponentialMovingAverage(
                        textModel1: textModel1EMALowerBoundWeights,
                        textModel2: textModel2EMALowerBoundWeights,
                        unetFixed: unetFixedEMALowerBoundWeights, unet: unetEMALowerBoundWeights,
                        textEmbedding1: textEmbedding1LowerBound,
                        textEmbedding2: textEmbedding2LowerBound),
                    exponentialMovingAverageUpperBound:
                      LoRATrainerCheckpoint.ExponentialMovingAverage(
                        textModel1: textModel1EMAUpperBoundWeights,
                        textModel2: textModel2EMAUpperBoundWeights,
                        unetFixed: unetFixedEMAUpperBoundWeights, unet: unetEMAUpperBoundWeights,
                        textEmbedding1: textEmbedding1UpperBound,
                        textEmbedding2: textEmbedding2UpperBound)))
                break
              }
            }
          }
          i += 1
          guard
            progressHandler(
              .step(i), Float(value),
              LoRATrainerCheckpoint(
                version: version, textModel1: cotrainTextModel ? textModel.first : nil,
                textModel2: cotrainTextModel && textModel.count > 1 ? textModel[1] : nil,
                unetFixed: unetFixed, unet: unet, textEmbedding1: customEmbeddings.first,
                textEmbedding2: customEmbeddings.count > 1 ? customEmbeddings[1] : nil, step: i,
                exponentialMovingAverageLowerBound: LoRATrainerCheckpoint.ExponentialMovingAverage(
                  textModel1: textModel1EMALowerBoundWeights,
                  textModel2: textModel2EMALowerBoundWeights,
                  unetFixed: unetFixedEMALowerBoundWeights, unet: unetEMALowerBoundWeights,
                  textEmbedding1: textEmbedding1LowerBound, textEmbedding2: textEmbedding2LowerBound
                ),
                exponentialMovingAverageUpperBound: LoRATrainerCheckpoint.ExponentialMovingAverage(
                  textModel1: textModel1EMAUpperBoundWeights,
                  textModel2: textModel2EMAUpperBoundWeights,
                  unetFixed: unetFixedEMAUpperBoundWeights, unet: unetEMAUpperBoundWeights,
                  textEmbedding1: textEmbedding1UpperBound, textEmbedding2: textEmbedding2UpperBound
                )))
          else {
            stopped = true
            break
          }
          if i >= trainingSteps {
            break
          }
        }
      }
    }
  }
}
