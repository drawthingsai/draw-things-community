import DataModels
import Diffusion
import Foundation
import ModelZoo
import NNC

public struct ImageGeneratorUtils {

  public static var isDepthModelAvailable: Bool {
    EverythingZoo.isModelDownloaded("depth_anything_v2.0_f16.ckpt")
  }

  public static let defaultTextEncoder = "clip_vit_l14_f16.ckpt"

  public static let defaultAutoencoder = "vae_ft_mse_840000_f16.ckpt"

  public static let defaultSoftEdgePreprocessor = "hed_f16.ckpt"

  public static func convertTensorToData(
    tensor: AnyTensor, using codec: DynamicGraph.Store.Codec = []
  )
    -> Data
  {
    switch tensor.dataType {
    case .Float64:
      return Tensor<Float64>(tensor).data(using: codec)
    case .Float32:
      return Tensor<Float32>(tensor).data(using: codec)
    case .Int64:
      return Tensor<Int64>(tensor).data(using: codec)
    case .Int32:
      return Tensor<Int32>(tensor).data(using: codec)
    case .Float16:
      #if !((os(macOS) || (os(iOS) && targetEnvironment(macCatalyst))) && (arch(i386) || arch(x86_64)))
        return Tensor<Float16>(tensor).data(using: codec)
      #else
        fatalError()
      #endif
    case .UInt8:
      return Tensor<UInt8>(tensor).data(using: codec)
    }
  }

  public static func canInjectControls(
    hasImage: Bool, hasDepth: Bool, hasHints: Set<ControlHintType>, hasCustom: Bool,
    shuffleCount: Int, controls: [Control], version: ModelVersion
  ) -> (
    injectControls: Bool, injectT2IAdapters: Bool, injectAttentionKVs: Bool,
    injectIPAdapterLengths: [Int],
    injectedControls: Int
  ) {
    var injectControls = false
    var injectT2IAdapters = false
    var injectIPAdapterLengths = [Int]()
    var injectedControls = 0
    var injectAttentionKVs = false
    for control in controls {
      guard let file = control.file, let specification = ControlNetZoo.specificationForModel(file),
        ControlNetZoo.isModelDownloaded(specification)
      else { continue }
      guard ControlNetZoo.versionForModel(file) == version else { continue }
      guard control.weight > 0 else { continue }
      guard
        let modifier = ControlNetZoo.modifierForModel(file)
          ?? ControlHintType(from: control.inputOverride)
      else { continue }
      let type = ControlNetZoo.typeForModel(file)
      let isPreprocessorDownloaded = ControlNetZoo.preprocessorForModel(file).map {
        ControlNetZoo.isModelDownloaded($0)
      }
      switch type {
      case .controlnet, .controlnetunion, .controlnetlora:
        switch modifier {
        case .canny, .mlsd, .tile, .blur, .gray, .lowquality:
          injectControls = injectControls || hasImage || hasCustom
          injectedControls += hasImage || hasCustom ? 1 : 0
        case .depth:
          let hasDepth = (hasImage && Self.isDepthModelAvailable) || hasDepth
          injectControls = injectControls || hasDepth
          injectedControls += hasDepth ? 1 : 0
        case .pose:
          injectControls = injectControls || hasHints.contains(modifier) || hasCustom
          injectedControls += hasHints.contains(modifier) || hasCustom ? 1 : 0
        case .scribble:
          let isPreprocessorDownloaded =
            isPreprocessorDownloaded
            ?? ControlNetZoo.isModelDownloaded(Self.defaultSoftEdgePreprocessor)
          injectControls =
            injectControls || hasHints.contains(modifier) || (isPreprocessorDownloaded && hasImage)
          injectedControls +=
            hasHints.contains(modifier) || (isPreprocessorDownloaded && hasImage) ? 1 : 0
        case .color:
          break  // Not supported. See generateInjectedControls.
        case .softedge:
          let isPreprocessorDownloaded =
            isPreprocessorDownloaded
            ?? ControlNetZoo.isModelDownloaded(Self.defaultSoftEdgePreprocessor)
          injectControls = injectControls || (isPreprocessorDownloaded && hasImage) || hasCustom
          injectedControls += (isPreprocessorDownloaded && hasImage) || hasCustom ? 1 : 0
        case .normalbae, .lineart, .seg, .custom:
          injectControls = injectControls || hasCustom
          injectedControls += hasCustom ? 1 : 0
        case .shuffle:
          injectControls = injectControls || shuffleCount > 0 || hasCustom
          injectedControls += shuffleCount > 0 || hasCustom ? 1 : 0
        case .inpaint, .ip2p:
          injectControls = injectControls || hasImage
          injectedControls += hasImage ? 1 : 0
        }
      case .injectKV:
        injectAttentionKVs = injectAttentionKVs || shuffleCount > 0 || hasCustom
      case .ipadapterplus:
        if hasCustom || shuffleCount > 0 {
          injectIPAdapterLengths.append((shuffleCount > 0 ? shuffleCount : 1) * 16)
        }
      case .ipadapterfull:
        if hasCustom || shuffleCount > 0 {
          injectIPAdapterLengths.append((shuffleCount > 0 ? shuffleCount : 1) * 257)
        }
      case .ipadapterfaceidplus:
        if hasCustom || shuffleCount > 0 {
          injectIPAdapterLengths.append((shuffleCount > 0 ? shuffleCount : 1) * 6)
        }
      case .pulid:
        if hasCustom || shuffleCount > 0 {
          injectIPAdapterLengths.append((shuffleCount > 0 ? shuffleCount : 1) * 32)
        }
      case .t2iadapter:
        switch modifier {
        case .canny, .mlsd, .tile, .blur, .gray, .lowquality:
          injectT2IAdapters = injectT2IAdapters || hasImage || hasCustom
        case .depth:
          injectT2IAdapters =
            injectT2IAdapters || (hasImage && Self.isDepthModelAvailable) || hasDepth
        case .pose:
          injectT2IAdapters =
            injectT2IAdapters || hasHints.contains(modifier) || hasCustom
        case .scribble:
          injectT2IAdapters = injectT2IAdapters || hasHints.contains(modifier)
        case .color:
          injectT2IAdapters = injectT2IAdapters || hasHints.contains(modifier) || hasImage
        case .normalbae, .lineart, .softedge, .seg, .inpaint, .ip2p, .shuffle, .custom:
          break
        }
      }
    }
    return (
      injectControls, injectT2IAdapters, injectAttentionKVs, injectIPAdapterLengths,
      injectedControls
    )
  }

  public static func isInpainting(
    for binaryMask: Tensor<UInt8>?, configuration: GenerationConfiguration
  ) -> Bool {
    guard let binaryMask = binaryMask else { return false }
    let modelVersion = ModelZoo.versionForModel(configuration.model ?? "")
    let refinerVersion: ModelVersion? = configuration.refinerModel.flatMap {
      guard $0 != configuration.model, ModelZoo.isModelDownloaded($0) else { return nil }
      let version = ModelZoo.versionForModel($0)
      guard
        version == modelVersion
          || ([.sdxlBase, .sdxlRefiner, .ssd1b].contains(version)
            && [.sdxlBase, .sdxlRefiner, .ssd1b].contains(modelVersion))
      else { return nil }
      return ModelZoo.versionForModel($0)
    }
    var alternativeDecoderVersion: AlternativeDecoderVersion? = nil
    for lora in configuration.loras {
      guard let file = lora.file else { continue }
      let loraVersion = LoRAZoo.versionForModel(file)

      guard LoRAZoo.isModelDownloaded(file),
        modelVersion == loraVersion || refinerVersion == loraVersion
          || (modelVersion == .kandinsky21 && loraVersion == .v1)
      else { continue }
      if let alternativeDecoder = LoRAZoo.alternativeDecoderForModel(file) {
        alternativeDecoderVersion = alternativeDecoder.1
      }
    }
    let imageWidth = binaryMask.shape[1]
    let imageHeight = binaryMask.shape[0]
    // See detail explanation below.
    // This effectively tells whether we have any skip all (3, or in some conditions 0 can skip all).
    // It should be exists3 || (!(exists0 && exists1 && exists2 && exists3) && exists2) but can be simplified to below.
    for y in 0..<imageHeight {
      for x in 0..<imageWidth {
        let byteMask = (binaryMask[y, x] & 7)
        if byteMask == 3 && alternativeDecoderVersion != .transparent {
          return true
        } else if byteMask == 2 || byteMask == 4 {  // 4 is the same as 2.
          return true
        }
      }
    }
    return false
  }

  public static func expectedSignposts(
    _ image: Bool, mask: Bool, text: String, negativeText: String,
    configuration: GenerationConfiguration, version: ModelVersion
  ) -> Set<ImageGeneratorSignpost> {
    var signposts = Set<ImageGeneratorSignpost>([.textEncoded, .imageDecoded])
    if let faceRestoration = configuration.faceRestoration,
      EverythingZoo.isModelDownloaded(faceRestoration)
        && EverythingZoo.isModelDownloaded(EverythingZoo.parsenetForModel(faceRestoration))
    {
      signposts.insert(.faceRestored)
    }
    if let upscaler = configuration.upscaler, UpscalerZoo.isModelDownloaded(upscaler) {
      signposts.insert(.imageUpscaled)
    }
    guard image else {
      signposts.insert(.sampling(Int(configuration.steps)))
      if configuration.hiresFix || version == .wurstchenStageC {
        signposts.insert(.secondPassImageEncoded)
        if version == .wurstchenStageC {
          let tEnc = Int(configuration.stage2Steps)
          signposts.insert(.secondPassSampling(tEnc))
        } else {
          let tEnc = Int(configuration.hiresFixStrength * Float(Int(configuration.steps)))
          signposts.insert(.secondPassSampling(tEnc))
        }
        signposts.insert(.secondPassImageDecoded)
      }
      return signposts
    }
    signposts.insert(.imageEncoded)
    let tEnc = Int(configuration.strength * Float(configuration.steps))
    signposts.insert(.sampling(tEnc))
    if version == .wurstchenStageC {
      signposts.insert(.secondPassImageEncoded)
      let tEnc = Int(configuration.strength * Float(configuration.stage2Steps))
      signposts.insert(.secondPassSampling(tEnc))
      signposts.insert(.secondPassImageDecoded)
    }
    return signposts
  }

  public static func extractDepthMap(
    _ image: DynamicGraph.Tensor<FloatType>, imageWidth: Int, imageHeight: Int,
    usesFlashAttention: Bool
  ) -> Tensor<FloatType> {
    let depthEstimator = DepthEstimator<FloatType>(
      filePaths: (
        EverythingZoo.filePathForModelDownloaded("depth_anything_v2.0_f16.ckpt"),
        EverythingZoo.filePathForModelDownloaded("depth_anything_v2.0_f16.ckpt")
      ), usesFlashAttention: usesFlashAttention)
    var depthMap = depthEstimator.estimate(image)
    let shape = depthMap.shape
    if shape[1] != imageHeight || shape[2] != imageWidth {
      depthMap = Upsample(
        .bilinear, widthScale: Float(imageWidth) / Float(shape[2]),
        heightScale: Float(imageHeight) / Float(shape[1]))(depthMap)
    }
    var depthMapRawValue = depthMap.rawValue.toCPU()
    var maxVal = depthMapRawValue[0, 0, 0, 0]
    var minVal = maxVal
    depthMapRawValue.withUnsafeBytes {
      guard let f16 = $0.baseAddress?.assumingMemoryBound(to: FloatType.self) else { return }
      for i in 0..<(imageHeight * imageWidth) {
        let val = f16[i]
        maxVal = max(maxVal, val)
        minVal = min(minVal, val)
      }
    }
    if maxVal - minVal > 0 {
      let scale = (maxVal - minVal) * 0.5
      let midVal = scale + minVal
      depthMapRawValue.withUnsafeMutableBytes {
        guard let f16 = $0.baseAddress?.assumingMemoryBound(to: FloatType.self) else { return }
        for i in 0..<(imageHeight * imageWidth) {
          f16[i] = (f16[i] - midVal) / scale
        }
      }
    }
    return depthMapRawValue
  }
}

extension ImageGeneratorUtils {
  public static func metadataOverride(_ configuration: GenerationConfiguration) -> (
    models: [ModelZoo.Specification], loras: [LoRAZoo.Specification],
    controlNets: [ControlNetZoo.Specification]
  ) {
    var models = [ModelZoo.Specification]()
    func appendModel(_ model: String?) {
      guard let model = model else { return }
      guard let specification = ModelZoo.specificationForModel(model) else { return }
      models.append(specification)
      if let stageModels = specification.stageModels {
        for stageModel in stageModels {
          if let specification = ModelZoo.specificationForModel(stageModel) {
            models.append(specification)
          }
        }
      }
    }
    appendModel(configuration.model)
    appendModel(configuration.refinerModel)
    var loras = [LoRAZoo.Specification]()
    for lora in configuration.loras {
      guard let file = lora.file else { continue }
      guard let specification = LoRAZoo.specificationForModel(file) else { continue }
      loras.append(specification)
    }
    var controlNets = [ControlNetZoo.Specification]()
    for control in configuration.controls {
      guard let file = control.file else { continue }
      guard let specification = ControlNetZoo.specificationForModel(file) else { continue }
      controlNets.append(specification)
    }
    return (models, loras, controlNets)
  }

  public static func allModelFiles(_ configuration: GenerationConfiguration) -> [(
    name: String, subtitle: String, file: String, fileUrl: URL
  )] {

    var allModelFiles = [(name: String, subtitle: String, file: String, fileUrl: URL)]()

    if let model = configuration.model,
      let specification = ModelZoo.specificationForModel(model)
    {
      let files = ModelZoo.filesToDownload(specification).map { (name, subtitle, file, _) in
        return (
          file, subtitle, file, URL(fileURLWithPath: ModelZoo.filePathForModelDownloaded(file))
        )
      }
      allModelFiles.append(contentsOf: files)
    }

    if let refinerModel = configuration.refinerModel,
      let specification = ModelZoo.specificationForModel(refinerModel)
    {
      let files = ModelZoo.filesToDownload(specification).map { (name, subtitle, file, _) in
        return (
          file, subtitle, file, URL(fileURLWithPath: ModelZoo.filePathForModelDownloaded(file))
        )
      }
      allModelFiles.append(contentsOf: files)
    }

    for lora in configuration.loras {
      if let model = lora.file,
        let specification = LoRAZoo.specificationForModel(model)
      {
        let files = LoRAZoo.filesToDownload(specification).map {
          (name, subtitle, file, _) in
          return (
            file, subtitle, file, URL(fileURLWithPath: LoRAZoo.filePathForModelDownloaded(file))
          )
        }
        allModelFiles.append(contentsOf: files)
      }
    }

    for control in configuration.controls {
      if let model = control.file,
        let specification = ControlNetZoo.specificationForModel(model)
      {
        let files = ControlNetZoo.filesToDownload(specification).map { (name, subtitle, file, _) in
          return (
            file, subtitle, file,
            URL(fileURLWithPath: ControlNetZoo.filePathForModelDownloaded(file))
          )
        }
        allModelFiles.append(contentsOf: files)
      }
    }
    return allModelFiles
  }

}
