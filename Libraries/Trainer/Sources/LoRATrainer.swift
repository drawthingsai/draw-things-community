import DataModels
import Diffusion
import Foundation
import ModelZoo
import NNC
import SFMT

public struct LoRATrainer {

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

  private struct ProcessedInput {
    var imagePath: String
    var tokens: [Int32]
    var originalSize: (width: Int, height: Int)
    var cropTopLeft: (top: Int, left: Int)
    var targetSize: (width: Int, height: Int)
  }

  public let version: ModelVersion
  private let model: String
  private let scale: DeviceCapability.Scale
  private let cotrainTextModel: Bool
  private let cotrainCustomEmbedding: Bool
  private let clipSkip: Int
  private let autoencoder: String
  private let textEncoder: String
  private let CLIPEncoder: String?
  private let session: String
  private let resumeIfPossible: Bool
  private let imageLoader:
    (URL, Int, Int) -> (
      Tensor<FloatType>, (width: Int, height: Int), (top: Int, left: Int), (width: Int, height: Int)
    )?

  public init(
    model: String, scale: DeviceCapability.Scale, cotrainTextModel: Bool,
    cotrainCustomEmbedding: Bool, clipSkip: Int, session: String, resumeIfPossible: Bool,
    imageLoader: @escaping (URL, Int, Int) -> (
      Tensor<FloatType>, (width: Int, height: Int), (top: Int, left: Int), (width: Int, height: Int)
    )?
  ) {  // The model identifier.
    self.model = model
    self.scale = scale
    self.cotrainTextModel = cotrainTextModel
    self.cotrainCustomEmbedding = cotrainCustomEmbedding
    self.clipSkip = clipSkip
    self.imageLoader = imageLoader
    let version = ModelZoo.versionForModel(model)
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
        case .flux1:
          return "flux_1_vae_f16.ckpt"
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
        }
      })()
    CLIPEncoder = ModelZoo.CLIPEncoderForModel(model)
    self.version = version
    self.session = session
    self.resumeIfPossible = resumeIfPossible
  }

  public enum PrepareState {
    case imageEncoding
    case conditionalEncoding
  }

  public func prepareDataset(
    inputs: [Input], tokenizer: TextualInversionPoweredTokenizer & Tokenizer,
    customEmbeddingLength: Int, progressHandler: (PrepareState, Int) -> Bool
  ) -> DataFrame? {
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
      alternativeDecoderVersion: nil)
    let graph = DynamicGraph()
    let imageWidth = Int(scale.widthScale) * 64
    let imageHeight = Int(scale.heightScale) * 64
    var processedInputs = [ProcessedInput]()
    var stopped = false
    graph.openStore(session) { store in
      // First, center crop and encode the image.
      graph.withNoGrad {
        var encoder: Model? = nil
        for input in inputs {
          guard
            let (tensor, originalSize, cropTopLeft, targetSize) = imageLoader(
              input.imageUrl, imageWidth, imageHeight)
          else { continue }
          // Keep this in the database.
          let tuple = firstStage.encode(graph.constant(tensor.toGPU(0)), encoder: encoder)
          let sample = tuple.0.rawValue.toCPU()
          encoder = tuple.1
          let imagePath = input.imageUrl.path
          store.write(imagePath, tensor: sample)
          let (_, tokens, _, _, _) = tokenizer.tokenize(
            text: input.caption, truncation: true, maxLength: 77)
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
          if updatedTokens.count > 77 {
            updatedTokens = Array(updatedTokens.dropLast(updatedTokens.count - 77))
          }
          processedInputs.append(
            ProcessedInput(
              imagePath: imagePath, tokens: updatedTokens, originalSize: originalSize,
              cropTopLeft: cropTopLeft, targetSize: targetSize))
          guard progressHandler(.imageEncoding, processedInputs.count) else {
            stopped = true
            break
          }
        }
        guard !stopped else { return }
        let zeros = graph.variable(
          .GPU(0), .NHWC(1, imageHeight, imageWidth, 3), of: FloatType.self)
        zeros.full(0)
        let latentZeros = firstStage.sample(zeros, encoder: encoder).1.copied().rawValue.toCPU()
        store.write("latent_zeros", tensor: latentZeros)
      }
      if stopped {
        return
      }
      // Now, check if we have cotrain turned on, if we don't, save the text encoding directly so we don't have to load the text model.
      if !cotrainTextModel && !cotrainCustomEmbedding {
        graph.withNoGrad {
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
            .wurstchenStageB:
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
              "text_model", model: textModel[0], codec: [.q6p, .q8p, .ezm7, .fpzip, .externalData]
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
                .wurstchenStageC,
                .wurstchenStageB:
                fatalError()
              }
              return .continue(name)
            }
            if let textProjection = textProjection {
              store.read("text_projection", variable: textProjection)
            }
          }
          for (index, input) in processedInputs.enumerated() {
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
            guard progressHandler(.conditionalEncoding, index + 1) else {
              stopped = true
              return
            }
          }
          if textModel.count > 1, let CLIPEncoder = CLIPEncoder {
            textModel[1].compile(inputs: tokensTensorGPU, positionTensorGPU, causalAttentionMaskGPU)
            graph.openStore(
              ModelZoo.filePathForModelDownloaded(CLIPEncoder), flags: .readOnly,
              externalStore: TensorData.externalStore(
                filePath: ModelZoo.filePathForModelDownloaded(CLIPEncoder))
            ) {
              store in
              store.read(
                "text_model", model: textModel[1],
                codec: [.q6p, .q8p, .ezm7, .fpzip, .externalData]
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
            for (index, input) in processedInputs.enumerated() {
              for i in 0..<77 {
                tokensTensor[i] = input.tokens[i]
              }
              let tokensTensorGPU = tokensTensor.toGPU(0)
              let c = textModel[1](
                inputs: tokensTensorGPU, positionTensorGPU, causalAttentionMaskGPU)[0]
                .as(of: FloatType.self).reshaped(.HWC(1, 77, 768))
              store.write("cond_te2_\(input.imagePath)", tensor: c.rawValue.toCPU())
              guard progressHandler(.conditionalEncoding, index + 1) else {
                stopped = true
                return
              }
            }
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
    return dataFrame
  }

  public func saveSession(
    textModel: [Model]?, unetFixed: Model?, unet: Model, embeddings: [DynamicGraph.Tensor<Float>],
    step: Int
  ) {
    let graph = embeddings.first?.graph ?? DynamicGraph()
    graph.openStore(session) {
      if !embeddings.isEmpty {
        switch version {
        case .v1, .v2, .kandinsky21, .svdI2v, .pixart, .auraflow, .flux1:
          $0.write("string_to_param", variable: embeddings[0])
        case .sd3, .sd3Large, .sdxlBase, .sdxlRefiner, .ssd1b, .wurstchenStageC, .wurstchenStageB:
          $0.write("string_to_param_clip_g", variable: embeddings[0])
          if embeddings.count > 1 {
            $0.write("string_to_param_clip_l", variable: embeddings[1])
          }
        }
      }
      if let textModel = textModel {
        for (i, textModel) in textModel.enumerated() {
          $0.write("te\(i)__text_model", model: textModel) { name, _ in
            guard name.contains("[i-") || name.contains("lora") else {
              return .skip
            }
            return .continue(name)
          }
        }
      }
      if let unetFixed = unetFixed {
        $0.write("unet_fixed", model: unetFixed) { name, _ in
          guard name.contains("[i-") || name.contains("lora") else {
            return .skip
          }
          return .continue(name)
        }
      }
      $0.write("unet", model: unet) { name, _ in
        guard name.contains("[i-") || name.contains("lora") else {
          return .skip
        }
        return .continue(name)
      }
      var stepTensor = Tensor<Int32>(.CPU, .C(1))
      stepTensor[0] = Int32(step)
      $0.write("current_step", tensor: stepTensor)
    }
  }

  public func saveAsLoRA(
    textModel: [Model]?, unetFixed: Model?, unet: Model, embeddings: [DynamicGraph.Tensor<Float>],
    scaleOfLoRA: Float, name: String
  ) {
    let graph = embeddings.first?.graph ?? DynamicGraph()
    graph.openStore(LoRAZoo.filePathForModelDownloaded(name)) { store in
      // Remove all values first.
      store.removeAll()
      if !embeddings.isEmpty {
        switch version {
        case .v1, .v2, .kandinsky21, .svdI2v, .pixart, .auraflow, .flux1:
          store.write("string_to_param", variable: embeddings[0])
        case .sd3, .sd3Large, .sdxlBase, .sdxlRefiner, .ssd1b, .wurstchenStageC, .wurstchenStageB:
          store.write("string_to_param_clip_g", variable: embeddings[0])
          if embeddings.count > 1 {
            store.write("string_to_param_clip_l", variable: embeddings[1])
          }
        }
      }
      if cotrainTextModel, let textModel = textModel {
        let textModelMapping: [Int: Int]
        switch version {
        case .v1:
          textModelMapping = LoRAMapping.CLIPTextModel
        case .v2:
          textModelMapping = LoRAMapping.OpenCLIPTextModel
        case .sd3, .sd3Large, .pixart, .auraflow, .flux1, .kandinsky21, .svdI2v, .wurstchenStageC,
          .wurstchenStageB:
          fatalError()
        case .sdxlBase, .ssd1b, .sdxlRefiner:
          textModelMapping = LoRAMapping.OpenCLIPTextModelG
        }
        store.write((textModel.count > 1 ? "te2__" : "") + "text_model", model: textModel[0]) {
          name, tensor in
          guard name.contains("lora") else { return .skip }
          let components = name.split(separator: "-")
          guard components.count >= 3, let index = Int(components[2]),
            let originalIndex = textModelMapping[index]
          else { return .skip }
          let isUp = name.contains("lora_up")
          let updatedName = "\(components[0])-\(originalIndex)-0]" + (isUp ? "__up__" : "__down__")
          if scaleOfLoRA != 1 && !isUp {
            let tensor = graph.withNoGrad {
              (scaleOfLoRA * graph.variable(Tensor<Float>(from: tensor))).rawValue
            }
            store.write(updatedName, tensor: tensor)
            return .skip
          }
          return .continue(updatedName)
        }
        if textModel.count > 1 {
          let textModelMapping = LoRAMapping.CLIPTextModel
          store.write("text_model", model: textModel[1]) { name, tensor in
            guard name.contains("lora") else { return .skip }
            let components = name.split(separator: "-")
            guard components.count >= 3, let index = Int(components[2]),
              let originalIndex = textModelMapping[index]
            else { return .skip }
            let isUp = name.contains("lora_up")
            let updatedName =
              "\(components[0])-\(originalIndex)-0]" + (isUp ? "__up__" : "__down__")
            if scaleOfLoRA != 1 && !isUp {
              let tensor = graph.withNoGrad {
                (scaleOfLoRA * graph.variable(Tensor<Float>(from: tensor))).rawValue
              }
              store.write(updatedName, tensor: tensor)
              return .skip
            }
            return .continue(updatedName)
          }
        }
      }
      if let unetFixed = unetFixed {
        store.write("unet_fixed", model: unetFixed) { name, tensor in
          guard name.contains("lora") else { return .skip }
          let components = name.split(separator: "-")
          // Every parameter in unetFixed is trainable.
          guard components.count >= 3, let originalIndex = Int(components[2])
          else { return .skip }
          let isUp = name.contains("lora_up")
          let updatedName = "\(components[0])-\(originalIndex)-0]" + (isUp ? "__up__" : "__down__")
          if scaleOfLoRA != 1 && !isUp {
            let tensor = graph.withNoGrad {
              (scaleOfLoRA * graph.variable(Tensor<Float>(from: tensor))).rawValue
            }
            store.write(updatedName, tensor: tensor)
            return .skip
          }
          return .continue(updatedName)
        }
      }
      let UNetMapping: [Int: Int]
      switch version {
      case .v1, .v2:
        UNetMapping = LoRAMapping.SDUNet
      case .sd3, .sd3Large, .pixart, .auraflow, .flux1, .kandinsky21, .svdI2v, .wurstchenStageC,
        .wurstchenStageB:
        fatalError()
      case .ssd1b:
        UNetMapping = LoRAMapping.SDUNetXLSSD1B
      case .sdxlBase:
        UNetMapping = LoRAMapping.SDUNetXLBase
      case .sdxlRefiner:
        UNetMapping = LoRAMapping.SDUNetXLRefiner
      }
      store.write("unet", model: unet) { name, tensor in
        guard name.contains("lora") else { return .skip }
        let components = name.split(separator: "-")
        guard components.count >= 3, let index = Int(components[2]),
          let originalIndex = UNetMapping[index]
        else { return .skip }
        let isUp = name.contains("lora_up")
        let updatedName = "\(components[0])-\(originalIndex)-0]" + (isUp ? "__up__" : "__down__")
        if scaleOfLoRA != 1 && !isUp {
          let tensor = graph.withNoGrad {
            (scaleOfLoRA * graph.variable(Tensor<Float>(from: tensor))).rawValue
          }
          store.write(updatedName, tensor: tensor)
          return .skip
        }
        return .continue(updatedName)
      }
    }
  }

  public func originalLoRA(name: String, LoRAMapping: [Int: Int]?) -> String {
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
    return "\(components[0])-\(originalIndex)-0]" + (isUp ? "__up__" : "__down__")
  }

  // This is heavily based on https://github.com/thuanz123/realfill/blob/main/train_realfill.py
  private func makeMask(resolution: (width: Int, height: Int), times: Int = 30) -> Tensor<FloatType>
  {
    var mask = Tensor<FloatType>(.CPU, .NHWC(1, resolution.height, resolution.width, 1))
    for y in 0..<resolution.height {
      for x in 0..<resolution.width {
        mask[0, y, x, 0] = 1
      }
    }
    var sfmt = SFMT(seed: UInt64.random(in: 0...UInt64.max))
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

  public func train(
    resumingLoRAFile: (String, Int)?, tokenizer: TextualInversionPoweredTokenizer & Tokenizer,
    dataFrame: DataFrame, trainingSteps: Int, warmupSteps: Int, gradientAccumulationSteps: Int,
    rankOfLoRA: Int, scaleOfLoRA: Float, textModelLearningRate: Float, unetLearningRate: Float,
    customEmbeddingLength: Int, customEmbeddingLearningRate: Float,
    stopEmbeddingTrainingAtStep: Int,
    noiseOffset: Float, denoisingTimesteps: ClosedRange<Int>, memorySaver: MemorySaver,
    progressHandler: (Int, Float, [Model]?, Model?, Model, [DynamicGraph.Tensor<Float>]) -> Bool
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
      alternativeDecoderVersion: nil)
    graph.maxConcurrency = .limit(1)
    var dataFrame = dataFrame
    let cotrainUNet = unetLearningRate > 0
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
      let latentWidth = Int(scale.widthScale) * 8
      let latentHeight = Int(scale.heightScale) * 8
      let causalAttentionMaskGPU = causalAttentionMask.toGPU(0)
      let LoRAConfiguration: LoRANetworkConfiguration
      switch memorySaver {
      case .minimal, .balanced:
        LoRAConfiguration = LoRANetworkConfiguration(
          rank: rankOfLoRA, scale: scaleOfLoRA, highPrecision: true, testing: false,
          gradientCheckpointingFeedForward: true)
      case .speed, .turbo:
        LoRAConfiguration = LoRANetworkConfiguration(
          rank: rankOfLoRA, scale: scaleOfLoRA, highPrecision: true, testing: false)
      }
      let textModel: [Model]
      let textLoRAMapping: [[Int: Int]]
      let unetFixed: Model?
      let unet: Model
      let unetLoRAMapping: [Int: Int]
      let embeddingSize: (Int, Int)
      let timeEmbeddingSize: Int
      switch version {
      case .v1:
        embeddingSize = (768, 768)
        timeEmbeddingSize = 320
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
        unet = LoRAUNet(
          batchSize: 1, embeddingLength: (77, 77), startWidth: latentWidth,
          startHeight: latentHeight,
          usesFlashAttention: isMFAEnabled ? .scaleMerged : .none, injectControls: false,
          injectT2IAdapters: false,
          injectIPAdapterLengths: [], LoRAConfiguration: configuration)
        unetLoRAMapping = LoRAMapping.SDUNet
      case .v2:
        embeddingSize = (1024, 1024)
        timeEmbeddingSize = 320
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
        unet = LoRAUNetv2(
          batchSize: 1, embeddingLength: (77, 77), startWidth: latentWidth,
          startHeight: latentHeight,
          upcastAttention: false, usesFlashAttention: isMFAEnabled ? .scaleMerged : .none,
          injectControls: false,
          LoRAConfiguration: configuration)
        unetLoRAMapping = LoRAMapping.SDUNet
      case .sdxlBase:
        embeddingSize = (1280, 768)
        timeEmbeddingSize = 320
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
          batchSize: 1, startHeight: latentHeight, startWidth: latentWidth,
          channels: [320, 640, 1280], embeddingLength: (77, 77),
          inputAttentionRes: [2: [2, 2], 4: [10, 10]], middleAttentionBlocks: 10,
          outputAttentionRes: [2: [2, 2, 2], 4: [10, 10, 10]],
          usesFlashAttention: isMFAEnabled ? .scaleMerged : .none, LoRAConfiguration: configuration)
        unet =
          LoRAUNetXL(
            batchSize: 1, startHeight: latentHeight, startWidth: latentWidth,
            channels: [320, 640, 1280], inputAttentionRes: [2: [2, 2], 4: [10, 10]],
            middleAttentionBlocks: 10, outputAttentionRes: [2: [2, 2, 2], 4: [10, 10, 10]],
            embeddingLength: (77, 77), injectIPAdapterLengths: [],
            upcastAttention: ([:], false, [:]),
            usesFlashAttention: isMFAEnabled ? .scaleMerged : .none,
            injectControls: false, LoRAConfiguration: configuration
          ).0
        unetLoRAMapping = LoRAMapping.SDUNetXLBase
      case .sdxlRefiner:
        embeddingSize = (1280, 1280)
        timeEmbeddingSize = 384
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
          batchSize: 1, startHeight: latentHeight, startWidth: latentWidth,
          channels: [384, 768, 1536, 1536], embeddingLength: (77, 77),
          inputAttentionRes: [2: [4, 4], 4: [4, 4]], middleAttentionBlocks: 4,
          outputAttentionRes: [2: [4, 4, 4], 4: [4, 4, 4]],
          usesFlashAttention: isMFAEnabled ? .scaleMerged : .none,
          LoRAConfiguration: configuration)
        unet =
          LoRAUNetXL(
            batchSize: 1, startHeight: latentHeight, startWidth: latentWidth,
            channels: [384, 768, 1536, 1536], inputAttentionRes: [2: [4, 4], 4: [4, 4]],
            middleAttentionBlocks: 4, outputAttentionRes: [2: [4, 4, 4], 4: [4, 4, 4]],
            embeddingLength: (77, 77), injectIPAdapterLengths: [],
            upcastAttention: ([:], false, [:]),
            usesFlashAttention: isMFAEnabled ? .scaleMerged : .none,
            injectControls: false, LoRAConfiguration: configuration
          ).0
        unetLoRAMapping = LoRAMapping.SDUNetXLRefiner
      case .ssd1b:
        embeddingSize = (1280, 768)
        timeEmbeddingSize = 320
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
          batchSize: 1, startHeight: latentHeight, startWidth: latentWidth,
          channels: [320, 640, 1280], embeddingLength: (77, 77),
          inputAttentionRes: [2: [2, 2], 4: [4, 4]], middleAttentionBlocks: 0,
          outputAttentionRes: [2: [2, 1, 1], 4: [4, 4, 10]],
          usesFlashAttention: isMFAEnabled ? .scale1 : .none, LoRAConfiguration: configuration)
        unet =
          LoRAUNetXL(
            batchSize: 1, startHeight: latentHeight, startWidth: latentWidth,
            channels: [320, 640, 1280], inputAttentionRes: [2: [2, 2], 4: [4, 4]],
            middleAttentionBlocks: 0, outputAttentionRes: [2: [2, 1, 1], 4: [4, 4, 10]],
            embeddingLength: (77, 77), injectIPAdapterLengths: [],
            upcastAttention: ([:], false, [:]), usesFlashAttention: isMFAEnabled ? .scale1 : .none,
            injectControls: false,
            LoRAConfiguration: configuration
          ).0
        unetLoRAMapping = LoRAMapping.SDUNetXLSSD1B
      case .sd3, .sd3Large, .pixart, .auraflow, .flux1, .kandinsky21, .svdI2v, .wurstchenStageC,
        .wurstchenStageB:
        fatalError()
      }
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
            store.read("text_projection", variable: textProjection)
          }
          store.read(
            "text_model", model: textModel[0], codec: [.q6p, .q8p, .ezm7, .fpzip, .externalData]
          ) {
            name, dataType, format, shape in
            if resumeIfPossible && (name.contains("[i-") || name.contains("lora")) {
              if let resumingLoRAFile = resumingLoRAFile?.0, name.contains("lora") {
                if let tensor = try?
                  (graph.openStore(
                    LoRAZoo.filePathForModelDownloaded(resumingLoRAFile), flags: .readOnly
                  ) {
                    return $0.read(
                      originalLoRA(name: name, LoRAMapping: textLoRAMapping[0]),
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
              .wurstchenStageC,
              .wurstchenStageB:
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
              "text_model", model: textModel[1], codec: [.q6p, .q8p, .ezm7, .fpzip, .externalData]
            ) {
              name, dataType, format, shape in
              if resumeIfPossible && (name.contains("[i-") || name.contains("lora")) {
                if let resumingLoRAFile = resumingLoRAFile?.0, name.contains("lora") {
                  if let tensor = try?
                    (graph.openStore(
                      LoRAZoo.filePathForModelDownloaded(resumingLoRAFile), flags: .readOnly
                    ) {
                      return $0.read(
                        originalLoRA(name: name, LoRAMapping: textLoRAMapping[1]),
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
          crossattn = graph.variable(.GPU(0), .HWC(1, 77, 2048), of: FloatType.self)
        } else {
          precondition(version == .sdxlRefiner)
          crossattn = graph.variable(.GPU(0), .HWC(1, 77, 1280), of: FloatType.self)
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
          .GPU(0), .NHWC(1, latentHeight, latentWidth, 9), of: FloatType.self)
      case .editing, .double:
        latents = graph.variable(
          .GPU(0), .NHWC(1, latentHeight, latentWidth, 8), of: FloatType.self)
      case .depth:
        latents = graph.variable(
          .GPU(0), .NHWC(1, latentHeight, latentWidth, 5), of: FloatType.self)
      case .none:
        latents = graph.variable(
          .GPU(0), .NHWC(1, latentHeight, latentWidth, 4), of: FloatType.self)
      }
      latents.full(0)
      let c: DynamicGraph.Tensor<FloatType>
      switch version {
      case .v1, .v2:
        c = graph.variable(.GPU(0), .HWC(1, 77, embeddingSize.0), of: FloatType.self)
      case .sdxlBase, .ssd1b:
        c = graph.variable(.GPU(0), .WC(1, 2816), of: FloatType.self)
      case .sdxlRefiner:
        c = graph.variable(.GPU(0), .WC(1, 2560), of: FloatType.self)
      case .sd3, .sd3Large, .pixart, .auraflow, .flux1, .kandinsky21, .svdI2v, .wurstchenStageC,
        .wurstchenStageB:
        fatalError()
      }
      c.full(0)
      switch memorySaver {
      case .minimal, .speed:
        unet.memoryReduction = true
      case .balanced, .turbo:  // For Balanced, we only do gradient checkpointing.
        unet.memoryReduction = false
      }
      unet.compile(inputs: [latents, graph.variable(Tensor<FloatType>(from: ts)), c] + kvs)
      graph.openStore(
        ModelZoo.filePathForModelDownloaded(model), flags: .readOnly,
        externalStore: TensorData.externalStore(
          filePath: ModelZoo.filePathForModelDownloaded(model))
      ) { store in
        if let unetFixed = unetFixed {
          store.read(
            "unet_fixed", model: unetFixed, codec: [.jit, .q6p, .q8p, .ezm7, .fpzip, .externalData]
          ) {
            name, dataType, format, shape in
            if resumeIfPossible && (name.contains("[i-") || name.contains("lora")) {
              if let resumingLoRAFile = resumingLoRAFile?.0, name.contains("lora") {
                if let tensor = try?
                  (graph.openStore(
                    LoRAZoo.filePathForModelDownloaded(resumingLoRAFile), flags: .readOnly
                  ) {
                    return $0.read(
                      originalLoRA(name: name, LoRAMapping: nil),
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
            }
            return .continue(name)
          }
        }
        store.read("unet", model: unet, codec: [.jit, .q6p, .q8p, .ezm7, .fpzip, .externalData]) {
          name, dataType, format, shape in
          if resumeIfPossible && (name.contains("[i-") || name.contains("lora")) {
            if let resumingLoRAFile = resumingLoRAFile?.0, name.contains("lora") {
              if let tensor = try?
                (graph.openStore(
                  LoRAZoo.filePathForModelDownloaded(resumingLoRAFile), flags: .readOnly
                ) {
                  return $0.read(
                    originalLoRA(name: name, LoRAMapping: unetLoRAMapping),
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
          graph, rate: unetLearningRate, betas: (0.9, 0.999), decay: 0.001, epsilon: 1e-8)
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
        case .v1, .v2, .kandinsky21, .svdI2v, .pixart, .auraflow, .flux1:
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
      while i < trainingSteps && !stopped && !optimizers.isEmpty {
        dataFrame.shuffle()
        for value in dataFrame["0", "imagePath", "tokens"] {
          guard let input = value[0] as? ProcessedInput, let imagePath = value[1] as? String,
            let tokens = value[2] as? [Int32]
          else {
            continue
          }
          guard let tensor = sessionStore.read(imagePath) else { continue }
          let shape = tensor.shape
          guard shape[1] == latentHeight && shape[2] == latentWidth else { continue }
          let (latents, _) = graph.withNoGrad {
            let parameters = graph.variable(Tensor<FloatType>(from: tensor).toGPU(0))
            let noise = graph.variable(.CPU, .NHWC(1, latentHeight, latentWidth, 4), of: Float.self)
            noise.randn(std: 1, mean: 0)
            let noiseGPU = DynamicGraph.Tensor<FloatType>(from: noise.toGPU(0))
            return firstStage.sampleFromDistribution(parameters, noise: noiseGPU)
          }
          let noiseGPU = graph.withNoGrad {
            var noise = graph.variable(.CPU, .NHWC(1, latentHeight, latentWidth, 4), of: Float.self)
            noise.randn(std: 1, mean: 0)
            if noiseOffset > 0 {
              let offsetNoise = graph.variable(.CPU, .NHWC(1, 1, 1, 1), of: Float.self)
              offsetNoise.randn(std: noiseOffset, mean: 0)
              noise = noise + offsetNoise
            }
            return DynamicGraph.Tensor<FloatType>(from: noise.toGPU(0))
          }
          let timestep = Int.random(in: denoisingTimesteps)
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
              .GPU(0), .NHWC(1, latentHeight, latentWidth, 9), of: FloatType.self)
            // Need to do more in case of inpainting model.
            noisyLatentsPlusMaskAndImage[0..<1, 0..<latentHeight, 0..<latentWidth, 0..<4] =
              noisyLatents
            let mask = graph.variable(
              makeMask(resolution: (width: latentWidth, height: latentHeight)).toGPU(0))
            noisyLatentsPlusMaskAndImage[0..<1, 0..<latentHeight, 0..<latentWidth, 4..<5] = mask
            // This is an approximation. The mask should be applied at image space, but that means we need to keep VAE in RAM. This allows us to mask out and fill in some zero latents (zero in image space).
            var conditioningImage = (1 - mask) .* latents
            if let latentZeros = latentZeros {
              conditioningImage = conditioningImage + (mask .* latentZeros)
            }
            noisyLatentsPlusMaskAndImage[0..<1, 0..<latentHeight, 0..<latentWidth, 5..<9] =
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
                .wurstchenStageC,
                .wurstchenStageB:
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
                filePath: "", version: version, usesFlashAttention: true,
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
                filePath: "", version: .sdxlRefiner, usesFlashAttention: true,
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
              .wurstchenStageC,
              .wurstchenStageB:
              fatalError()
            }
            condTokensTensorGPU = tokensTensorGPU
          } else {
            guard let tensor = sessionStore.read("cond_\(imagePath)") else { continue }
            switch version {
            case .v1, .v2:
              c = graph.variable(Tensor<FloatType>(from: tensor).toGPU(0))
              kvs = []
            case .sdxlBase, .ssd1b:
              guard let te2 = sessionStore.read("cond_te2_\(imagePath)"),
                let pooled = sessionStore.read("pool_\(imagePath)")
              else { continue }
              let unetFixEncoder = UNetFixedEncoder<FloatType>(
                filePath: "", version: version, usesFlashAttention: true,
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
            case .sdxlRefiner:
              guard let pooled = sessionStore.read("pool_\(imagePath)") else { continue }
              let unetFixEncoder = UNetFixedEncoder<FloatType>(
                filePath: "", version: .sdxlRefiner, usesFlashAttention: true,
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
              .wurstchenStageC,
              .wurstchenStageB:
              fatalError()
            }
            condTokensTensorGPU = nil
          }
          let et = unet(inputs: noisyLatents, [t, c] + kvs)[0].as(of: FloatType.self)
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
          print("loss \(value), scale \(scaler.scale), step \(i)")
          batchCount += 1
          if (i + 1) < warmupSteps {
            if cotrainUNet {
              optimizers[0].rate = unetLearningRate * (Float(i + 1) / Float(warmupSteps))
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
          } else if (i + 1) == warmupSteps {
            if cotrainUNet {
              optimizers[0].rate = unetLearningRate
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
          if batchCount == gradientAccumulationSteps {
            // Update the LoRA.
            scaler.step(&optimizers)
            batchCount = 0
            if cotrainCustomEmbedding && !cotrainCustomEmbeddingStopped
              && i + 1 >= stopEmbeddingTrainingAtStep
            {
              customEmbeddings.forEach { $0.detach() }
              // Drop the optimizer.
              optimizers = optimizers.dropLast()
              cotrainCustomEmbeddingStopped = true
              if optimizers.isEmpty {
                let _ = progressHandler(
                  i + 1, Float(value), cotrainTextModel ? textModel : nil, unetFixed, unet,
                  customEmbeddings
                )
                break
              }
            }
          }
          i += 1
          guard
            progressHandler(
              i, Float(value), cotrainTextModel ? textModel : nil, unetFixed, unet, customEmbeddings
            )
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
