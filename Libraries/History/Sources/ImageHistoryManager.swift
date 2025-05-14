import DataModels
import Dflat
import Diffusion
import NNC
import UIKit

extension SamplerType {
  public init(from sampler: DataModels.SamplerType) {
    switch sampler {
    case .dPMPP2MKarras:
      self = .dPMPP2MKarras
    case .eulerA:
      self = .eulerA
    case .DDIM:
      self = .DDIM
    case .PLMS:
      self = .PLMS
    case .dPMPPSDEKarras:
      self = .dPMPPSDEKarras
    case .uniPC:
      self = .uniPC
    case .LCM:
      self = .LCM
    case .eulerASubstep:
      self = .eulerASubstep
    case .dPMPPSDESubstep:
      self = .dPMPPSDESubstep
    case .TCD:
      self = .TCD
    case .eulerATrailing:
      self = .eulerATrailing
    case .dPMPPSDETrailing:
      self = .dPMPPSDETrailing
    case .DPMPP2MAYS:
      self = .DPMPP2MAYS
    case .eulerAAYS:
      self = .eulerAAYS
    case .DPMPPSDEAYS:
      self = .DPMPPSDEAYS
    case .dPMPP2MTrailing:
      self = .dPMPP2MTrailing
    case .dDIMTrailing:
      self = .dDIMTrailing
    }
  }
}

extension DataModels.SamplerType {
  public init(from sampler: SamplerType) {
    switch sampler {
    case .dPMPP2MKarras:
      self = .dPMPP2MKarras
    case .eulerA:
      self = .eulerA
    case .DDIM:
      self = .DDIM
    case .PLMS:
      self = .PLMS
    case .dPMPPSDEKarras:
      self = .dPMPPSDEKarras
    case .uniPC:
      self = .uniPC
    case .LCM:
      self = .LCM
    case .eulerASubstep:
      self = .eulerASubstep
    case .dPMPPSDESubstep:
      self = .dPMPPSDESubstep
    case .TCD:
      self = .TCD
    case .eulerATrailing:
      self = .eulerATrailing
    case .dPMPPSDETrailing:
      self = .dPMPPSDETrailing
    case .DPMPP2MAYS:
      self = .DPMPP2MAYS
    case .eulerAAYS:
      self = .eulerAAYS
    case .DPMPPSDEAYS:
      self = .DPMPPSDEAYS
    case .dPMPP2MTrailing:
      self = .dPMPP2MTrailing
    case .dDIMTrailing:
      self = .dDIMTrailing
    }
  }
}

extension SeedMode {
  public init(from seedMode: DataModels.SeedMode) {
    switch seedMode {
    case .legacy:
      self = .legacy
    case .torchCpuCompatible:
      self = .torchCpuCompatible
    case .scaleAlike:
      self = .scaleAlike
    case .nvidiaGpuCompatible:
      self = .nvidiaGpuCompatible
    }
  }
}

extension DataModels.SeedMode {
  public init(from seedMode: SeedMode) {
    switch seedMode {
    case .legacy:
      self = .legacy
    case .torchCpuCompatible:
      self = .torchCpuCompatible
    case .scaleAlike:
      self = .scaleAlike
    case .nvidiaGpuCompatible:
      self = .nvidiaGpuCompatible
    }
  }
}

extension ControlMode {
  public init(from controlMode: DataModels.ControlMode) {
    switch controlMode {
    case .balanced:
      self = .balanced
    case .prompt:
      self = .prompt
    case .control:
      self = .control
    }
  }
}

extension DataModels.ControlMode {
  public init(from controlMode: ControlMode) {
    switch controlMode {
    case .balanced:
      self = .balanced
    case .prompt:
      self = .prompt
    case .control:
      self = .control
    }
  }
}

extension ControlInputType {
  public init(from controlInputType: DataModels.ControlInputType) {
    switch controlInputType {
    case .blur:
      self = .blur
    case .canny:
      self = .canny
    case .color:
      self = .color
    case .custom:
      self = .custom
    case .depth:
      self = .depth
    case .gray:
      self = .gray
    case .inpaint:
      self = .inpaint
    case .ip2p:
      self = .ip2p
    case .lineart:
      self = .lineart
    case .lowquality:
      self = .lowquality
    case .mlsd:
      self = .mlsd
    case .unspecified:
      self = .unspecified
    case .normalbae:
      self = .normalbae
    case .pose:
      self = .pose
    case .scribble:
      self = .scribble
    case .seg:
      self = .seg
    case .shuffle:
      self = .shuffle
    case .softedge:
      self = .softedge
    case .tile:
      self = .tile
    }
  }
}

extension DataModels.ControlInputType {
  public init(from controlInputType: ControlInputType) {
    switch controlInputType {
    case .blur:
      self = .blur
    case .canny:
      self = .canny
    case .color:
      self = .color
    case .custom:
      self = .custom
    case .depth:
      self = .depth
    case .gray:
      self = .gray
    case .inpaint:
      self = .inpaint
    case .ip2p:
      self = .ip2p
    case .lineart:
      self = .lineart
    case .lowquality:
      self = .lowquality
    case .mlsd:
      self = .mlsd
    case .unspecified:
      self = .unspecified
    case .normalbae:
      self = .normalbae
    case .pose:
      self = .pose
    case .scribble:
      self = .scribble
    case .seg:
      self = .seg
    case .shuffle:
      self = .shuffle
    case .softedge:
      self = .softedge
    case .tile:
      self = .tile
    }
  }
}

extension Control {
  public init(from control: DataModels.Control) {
    self.init(
      file: control.file, weight: control.weight, guidanceStart: control.guidanceStart,
      guidanceEnd: control.guidanceEnd, noPrompt: control.noPrompt,
      globalAveragePooling: control.globalAveragePooling,
      downSamplingRate: control.downSamplingRate, controlMode: .init(from: control.controlMode),
      targetBlocks: control.targetBlocks, inputOverride: .init(from: control.inputOverride)
    )
  }
}

extension DataModels.Control {
  public init(from control: Control) {
    self.init(
      file: control.file, weight: control.weight, guidanceStart: control.guidanceStart,
      guidanceEnd: control.guidanceEnd, noPrompt: control.noPrompt,
      globalAveragePooling: control.globalAveragePooling,
      downSamplingRate: control.downSamplingRate, controlMode: .init(from: control.controlMode),
      targetBlocks: control.targetBlocks, inputOverride: .init(from: control.inputOverride)
    )
  }
}

extension LoRA {
  public init(from lora: DataModels.LoRA) {
    self.init(file: lora.file, weight: lora.weight)
  }
}

extension DataModels.LoRA {
  public init(from lora: LoRA) {
    self.init(file: lora.file, weight: lora.weight)
  }
}

func downsampleImage(_ image: UIImage?) -> UIImage? {
  guard let image = image else { return nil }
  let imageSize = image.size
  // At least scale it down by 2. It could be more.
  let scale = max(Int(floor(CGFloat(min(image.size.width, image.size.height) / 384))), 2)
  let imageWidth = Int(imageSize.width) / scale
  let imageHeight = Int(imageSize.height) / scale
  guard
    let bitmapContext = CGContext(
      data: nil, width: imageWidth, height: imageHeight, bitsPerComponent: 8,
      bytesPerRow: imageWidth * 4, space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGBitmapInfo.byteOrderDefault.rawValue
        | CGImageAlphaInfo.premultipliedLast.rawValue, releaseCallback: nil, releaseInfo: nil)
  else {
    return nil
  }
  if let cgImage = image.cgImage {
    bitmapContext.draw(
      cgImage, in: CGRect(x: 0, y: 0, width: CGFloat(imageWidth), height: CGFloat(imageHeight)))
  }
  return bitmapContext.makeImage().flatMap { UIImage(cgImage: $0) }
}

extension ImageHistoryManager {
  public struct ImageData: Equatable & Hashable {
    public var x: Int32
    public var y: Int32
    public var width: Int32
    public var height: Int32
    public var scaleFactorBy120: Int32
    public var tensorId: Int64?
    public var maskId: Int64?
    public var depthMapId: Int64?
    public var scribbleId: Int64?
    public var poseId: Int64?
    public var colorPaletteId: Int64?
    public var customId: Int64?
    public init(
      x: Int32, y: Int32, width: Int32, height: Int32, scaleFactorBy120: Int32,
      tensorId: Int64? = nil, maskId: Int64? = nil,
      depthMapId: Int64? = nil, scribbleId: Int64? = nil, poseId: Int64? = nil,
      colorPaletteId: Int64? = nil, customId: Int64? = nil
    ) {
      self.x = x
      self.y = y
      self.width = width
      self.height = height
      self.scaleFactorBy120 = scaleFactorBy120
      self.tensorId = tensorId
      self.maskId = maskId
      self.depthMapId = depthMapId
      self.scribbleId = scribbleId
      self.poseId = poseId
      self.colorPaletteId = colorPaletteId
      self.customId = customId
    }
  }

  public var imageData: [ImageData] {
    // Compose tensor data with the image history together.
    guard dataStored.count == 0 else { return dataStored }
    var imageData = [ImageData]()
    guard let configuration = configuration else { return imageData }
    guard
      tensorId != nil || maskId != nil || depthMapId != nil || scribbleId != nil || poseId != nil
        || colorPaletteId != nil || customId != nil
    else { return imageData }
    let scaleFactor = Int32(scaleFactor)
    imageData.append(
      ImageData(
        x: 0, y: 0, width: Int32(configuration.startWidth) * 64 * scaleFactor,
        height: Int32(configuration.startHeight) * 64 * scaleFactor,
        scaleFactorBy120: 120 * scaleFactor, tensorId: tensorId, maskId: maskId,
        depthMapId: depthMapId, scribbleId: scribbleId, poseId: poseId,
        colorPaletteId: colorPaletteId, customId: customId))
    return imageData
  }
}

extension ImageHistoryManager {
  public struct ShuffleData: Equatable & Hashable {
    public var shuffleId: Int64
    public var weight: Float
    public init(shuffleId: Int64, weight: Float) {
      self.shuffleId = shuffleId
      self.weight = weight
    }
  }
}

public final class ImageHistoryManager {
  private struct LogicalTimeAndLineage: Equatable & Hashable {
    var logicalTime: Int64
    var lineage: Int64
  }
  private var dataStored: [ImageData] = []
  private var maxLineage: Int64 = 0
  public private(set) var lineage: Int64 = 0
  public private(set) var maxLogicalTime: Int64 = 0
  public private(set) var logicalTime: Int64 = 0
  public private(set) var configuration: GenerationConfiguration? = nil
  public private(set) var tensorId: Int64? = nil
  public private(set) var maskId: Int64? = nil
  public private(set) var depthMapId: Int64? = nil
  public private(set) var scribbleId: Int64? = nil
  public private(set) var poseId: Int64? = nil
  public private(set) var colorPaletteId: Int64? = nil
  public private(set) var customId: Int64? = nil
  public private(set) var shuffleData: [ShuffleData] = []
  public private(set) var previewId: Int64? = nil
  public private(set) var textEdits: Int? = nil
  public private(set) var textLineage: Int64? = nil
  public private(set) var scaleFactor: Int = 1
  public private(set) var isGenerated: Bool = false
  public private(set) var contentOffset: (x: Int32, y: Int32) = (x: 0, y: 0)
  public private(set) var scaleFactorBy120: Int32 = 120
  private let filePath: String
  private var project: Workspace
  private var previewCache = [Int64: UIImage]()  // Keyed by tensorId + maskId
  private var logicalVersion: Int = 0
  private var imageDataCache = [LogicalTimeAndLineage: ([TensorData], Int)]()  // Keyed by logical time and lineage
  private var shuffleDataCache = [LogicalTimeAndLineage: ([TensorMoodboardData], Int)]()  // Keyed by logical time and lineage
  private var nodeLineageCache = [LogicalTimeAndLineage: (TensorHistoryNode, Int)]()  // Keyed by logical time and lineage.
  private var nodeCache = [Int64: (TensorHistoryNode, Int)]()  // Keyed by logical time.
  private var maxLogicalTimeForLineage = [Int64: Int64]()  // This never cleared up and will grow, but it is OK, because we probably at around 10k lineage or somewhere around that.
  private var dynamicGraph = DynamicGraph()  // We don't do computations on this graph, it is just used to write tensors to disk.
  public init(project: Workspace, filePath: String) {
    self.project = project
    self.filePath = filePath
    guard
      let (imageHistory, imageData, shuffleData) =
        (project.fetchWithinASnapshot {
          () -> (TensorHistoryNode, [TensorData], [TensorMoodboardData])? in
          let imageHistories = project.fetch(for: TensorHistoryNode.self).all(
            limit: .limit(1),
            orderBy: [
              TensorHistoryNode.lineage.descending, TensorHistoryNode.logicalTime.descending,
            ])
          guard let imageHistory = imageHistories.first else {
            return nil
          }
          let imageData: [TensorData]
          if imageHistory.dataStored > 0 {
            imageData = Array(
              project.fetch(for: TensorData.self).where(
                TensorData.lineage == imageHistory.lineage
                  && TensorData.logicalTime == imageHistory.logicalTime))
          } else {
            imageData = []
          }
          let shuffleData: [TensorMoodboardData]
          if imageHistory.shuffleDataStored > 0 {
            shuffleData = Array(
              project.fetch(for: TensorMoodboardData.self).where(
                TensorMoodboardData.lineage == imageHistory.lineage
                  && TensorMoodboardData.logicalTime == imageHistory.logicalTime))
          } else {
            shuffleData = []
          }
          return (imageHistory, imageData, shuffleData)
        })
    else { return }
    setImageHistory(imageHistory, imageData: imageData, shuffleData: shuffleData)
    maxLineage = lineage
    maxLogicalTime = logicalTime
    maxLogicalTimeForLineage[lineage] = maxLogicalTime
    if var seekTo = project.dictionary["image_seek_to", Int.self] {
      seekTo = min(max(0, seekTo), Int(maxLogicalTime))
      seek(to: Int64(seekTo))
    }
  }

  private func setImageHistory(
    _ imageHistory: TensorHistoryNode, imageData: [TensorData], shuffleData: [TensorMoodboardData]
  ) {
    lineage = imageHistory.lineage
    logicalTime = imageHistory.logicalTime
    tensorId = imageHistory.tensorId == 0 ? nil : imageHistory.tensorId
    maskId = imageHistory.maskId == 0 ? nil : imageHistory.maskId
    depthMapId = imageHistory.depthMapId == 0 ? nil : imageHistory.depthMapId
    scribbleId = imageHistory.scribbleId == 0 ? nil : imageHistory.scribbleId
    poseId = imageHistory.poseId == 0 ? nil : imageHistory.poseId
    colorPaletteId = imageHistory.colorPaletteId == 0 ? nil : imageHistory.colorPaletteId
    customId = imageHistory.customId == 0 ? nil : imageHistory.customId
    if imageHistory.previewId == 0 {
      var previewId: Int64 = 0
      previewId +=
        Int64(imageHistory.tensorId) + Int64(imageHistory.maskId) + Int64(imageHistory.depthMapId)
      previewId +=
        Int64(imageHistory.scribbleId) + Int64(imageHistory.poseId)
        + Int64(imageHistory.colorPaletteId)
      previewId += Int64(imageHistory.customId)
      self.previewId = previewId
    } else {
      previewId = imageHistory.previewId
    }
    textEdits = imageHistory.textEdits == -1 ? nil : Int(imageHistory.textEdits)
    textLineage = imageHistory.textLineage == -1 ? nil : imageHistory.textLineage
    scaleFactor = max(Int(imageHistory.scaleFactor), 1)
    isGenerated = imageHistory.generated
    contentOffset = (x: imageHistory.contentOffsetX, y: imageHistory.contentOffsetY)
    scaleFactorBy120 = imageHistory.scaleFactorBy120
    configuration = GenerationConfiguration(
      id: 0, startWidth: imageHistory.startWidth, startHeight: imageHistory.startHeight,
      seed: imageHistory.seed, steps: imageHistory.steps, guidanceScale: imageHistory.guidanceScale,
      strength: imageHistory.strength, model: imageHistory.model,
      sampler: DataModels.SamplerType(from: imageHistory.sampler),
      batchSize: imageHistory.batchSize,
      hiresFix: imageHistory.hiresFix, hiresFixStartWidth: imageHistory.hiresFixStartWidth,
      hiresFixStartHeight: imageHistory.hiresFixStartHeight,
      hiresFixStrength: imageHistory.hiresFixStrength, upscaler: imageHistory.upscaler,
      imageGuidanceScale: imageHistory.imageGuidanceScale,
      seedMode: DataModels.SeedMode(from: imageHistory.seedMode),
      clipSkip: imageHistory.clipSkip,
      controls: imageHistory.controls.map { DataModels.Control(from: $0) },
      loras: imageHistory.loras.map { DataModels.LoRA(from: $0) },
      maskBlur: imageHistory.maskBlur, faceRestoration: imageHistory.faceRestoration,
      clipWeight: imageHistory.clipWeight,
      negativePromptForImagePrior: imageHistory.negativePromptForImagePrior,
      imagePriorSteps: imageHistory.imagePriorSteps, refinerModel: imageHistory.refinerModel,
      originalImageHeight: imageHistory.originalImageHeight,
      originalImageWidth: imageHistory.originalImageWidth, cropTop: imageHistory.cropTop,
      cropLeft: imageHistory.cropLeft, targetImageHeight: imageHistory.targetImageHeight,
      targetImageWidth: imageHistory.targetImageWidth, aestheticScore: imageHistory.aestheticScore,
      negativeAestheticScore: imageHistory.negativeAestheticScore,
      zeroNegativePrompt: imageHistory.zeroNegativePrompt, refinerStart: imageHistory.refinerStart,
      negativeOriginalImageHeight: imageHistory.negativeOriginalImageHeight,
      negativeOriginalImageWidth: imageHistory.negativeOriginalImageWidth,
      fpsId: imageHistory.fpsId, motionBucketId: imageHistory.motionBucketId,
      condAug: imageHistory.condAug, startFrameCfg: imageHistory.startFrameCfg,
      numFrames: imageHistory.numFrames, maskBlurOutset: imageHistory.maskBlurOutset,
      sharpness: imageHistory.sharpness, shift: imageHistory.shift,
      stage2Steps: imageHistory.stage2Steps, stage2Cfg: imageHistory.stage2Cfg,
      stage2Shift: imageHistory.stage2Shift, tiledDecoding: imageHistory.tiledDecoding,
      decodingTileWidth: imageHistory.decodingTileWidth,
      decodingTileHeight: imageHistory.decodingTileHeight,
      decodingTileOverlap: imageHistory.decodingTileOverlap,
      stochasticSamplingGamma: imageHistory.stochasticSamplingGamma,
      preserveOriginalAfterInpaint: imageHistory.preserveOriginalAfterInpaint,
      tiledDiffusion: imageHistory.tiledDiffusion,
      diffusionTileWidth: imageHistory.diffusionTileWidth,
      diffusionTileHeight: imageHistory.diffusionTileHeight,
      diffusionTileOverlap: imageHistory.diffusionTileOverlap,
      upscalerScaleFactor: imageHistory.upscalerScaleFactor,
      t5TextEncoder: imageHistory.t5TextEncoder,
      separateClipL: imageHistory.separateClipL,
      clipLText: imageHistory.clipLText,
      separateOpenClipG: imageHistory.separateOpenClipG,
      openClipGText: imageHistory.openClipGText,
      speedUpWithGuidanceEmbed: imageHistory.speedUpWithGuidanceEmbed,
      guidanceEmbed: imageHistory.guidanceEmbed,
      resolutionDependentShift: imageHistory.resolutionDependentShift,
      teaCacheStart: imageHistory.teaCacheStart,
      teaCacheEnd: imageHistory.teaCacheEnd,
      teaCacheThreshold: imageHistory.teaCacheThreshold,
      teaCache: imageHistory.teaCache,
      separateT5: imageHistory.separateT5,
      t5Text: imageHistory.t5Text,
      teaCacheMaxSkipSteps: imageHistory.teaCacheMaxSkipSteps
    )
    _profileData = imageHistory.profileData
    dataStored = imageData.sorted(by: { $0.index < $1.index }).map {
      let tensorId = $0.tensorId == 0 ? nil : $0.tensorId
      let maskId = $0.maskId == 0 ? nil : $0.maskId
      let depthMapId = $0.depthMapId == 0 ? nil : $0.depthMapId
      let scribbleId = $0.scribbleId == 0 ? nil : $0.scribbleId
      let poseId = $0.poseId == 0 ? nil : $0.poseId
      let colorPaletteId = $0.colorPaletteId == 0 ? nil : $0.colorPaletteId
      let customId = $0.customId == 0 ? nil : $0.customId
      return ImageData(
        x: $0.x, y: $0.y, width: $0.width, height: $0.height, scaleFactorBy120: $0.scaleFactorBy120,
        tensorId: tensorId, maskId: maskId,
        depthMapId: depthMapId, scribbleId: scribbleId, poseId: poseId,
        colorPaletteId: colorPaletteId, customId: customId)
    }
    self.shuffleData = shuffleData.sorted(by: { $0.index < $1.index }).map {
      return ShuffleData(shuffleId: $0.shuffleId, weight: $0.weight)
    }
  }

  private func uniqueVersion() -> Int {
    let uniqueVersion = logicalVersion
    logicalVersion += 1
    return uniqueVersion
  }

  public struct History {
    public var imageData: [ImageData]
    public var preview: UIImage?
    public var textEdits: Int?
    public var textLineage: Int64?
    public var configuration: GenerationConfiguration
    public var isGenerated: Bool
    public var contentOffset: (x: Int32, y: Int32)
    public var scaleFactorBy120: Int32
    public var scriptSessionId: UInt64?
    public var shuffleData: [ShuffleData]?
    public var profile: GenerationProfile?
    public var textPrompt: String?
    public var negativeTextPrompt: String?
    public init(
      imageData: [ImageData], preview: UIImage?, textEdits: Int?, textLineage: Int64?,
      configuration: GenerationConfiguration, isGenerated: Bool,
      contentOffset: (x: Int32, y: Int32), scaleFactorBy120: Int32, scriptSessionId: UInt64?,
      shuffleData: [ShuffleData]? = nil, profile: GenerationProfile? = nil,
      textPrompt: String? = nil, negativeTextPrompt: String? = nil
    ) {
      self.imageData = imageData
      self.preview = preview
      self.textEdits = textEdits
      self.textLineage = textLineage
      self.configuration = configuration
      self.isGenerated = isGenerated
      self.contentOffset = contentOffset
      self.scaleFactorBy120 = scaleFactorBy120
      self.scriptSessionId = scriptSessionId
      self.shuffleData = shuffleData
      self.profile = profile
      self.textPrompt = textPrompt
      self.negativeTextPrompt = negativeTextPrompt
    }
  }

  public func pushHistory(_ history: History) {
    dispatchPrecondition(condition: .onQueue(.main))
    // We need to fork this history.
    precondition(lineage <= maxLineage)
    // If logicalTime is 0, we don't need to deal with fork.
    if logicalTime != maxLogicalTime || lineage != maxLineage {
      if logicalTime > 0 {
        // If we are not the sacred lineage (as whether I am the maxLineage at this particular time)
        // Below: fetch the image history at this particular logical time from the sacred lineage.
        let minLogicalTimeImageHistory =
          nodeCache[min(logicalTime, maxLogicalTime)]?.0 ?? project.fetch(
            for: TensorHistoryNode.self
          )
          .where(
            TensorHistoryNode.logicalTime == min(logicalTime, maxLogicalTime),
            limit: .limit(1), orderBy: [TensorHistoryNode.lineage.descending]
          ).first!
        // If is smaller than the sacred one.
        if lineage < minLogicalTimeImageHistory.lineage {
          let imageHistories = project.fetch(for: TensorHistoryNode.self).where(
            TensorHistoryNode.lineage == lineage)
          let newLineage = maxLineage + 1
          maxLineage = newLineage
          var imageVersions = [Int]()
          var imageData = [TensorData]()
          var shuffleData = [TensorMoodboardData]()
          for imageHistory in imageHistories {
            let uniqueVersion = uniqueVersion()
            imageVersions.append(uniqueVersion)
            var builder = TensorHistoryNodeBuilder(from: imageHistory)
            builder.lineage = newLineage
            let node = (builder.build(), uniqueVersion)
            nodeCache[imageHistory.logicalTime] = node
            let logicalTimeAndLineage = LogicalTimeAndLineage(
              logicalTime: imageHistory.logicalTime, lineage: newLineage)
            nodeLineageCache[logicalTimeAndLineage] = node
            let imageDataForHistory = Array(
              project.fetch(for: TensorData.self).where(
                TensorData.lineage == imageHistory.lineage
                  && TensorData.logicalTime == imageHistory.logicalTime))
            imageDataCache[logicalTimeAndLineage] = (imageDataForHistory, uniqueVersion)
            imageData.append(contentsOf: imageDataForHistory)
            let shuffleDataForHistory = Array(
              project.fetch(for: TensorMoodboardData.self).where(
                TensorMoodboardData.lineage == imageHistory.lineage
                  && TensorMoodboardData.logicalTime == imageHistory.logicalTime))
            shuffleDataCache[logicalTimeAndLineage] = (shuffleDataForHistory, uniqueVersion)
            shuffleData.append(contentsOf: shuffleDataForHistory)
          }
          project.performChanges([
            TensorHistoryNode.self, TensorData.self, TensorMoodboardData.self,
          ]) { transactionContext in
            for imageHistory in imageHistories {
              guard let changeRequest = TensorHistoryNodeChangeRequest.changeRequest(imageHistory)
              else { continue }
              changeRequest.lineage = newLineage
              transactionContext.try(submit: changeRequest)
            }
            for item in imageData {
              guard let changeRequest = TensorDataChangeRequest.changeRequest(item) else {
                continue
              }
              changeRequest.lineage = newLineage
              transactionContext.try(submit: changeRequest)
            }
            for shuffleItem in shuffleData {
              guard let changeRequest = TensorMoodboardDataChangeRequest.changeRequest(shuffleItem)
              else {
                continue
              }
              changeRequest.lineage = newLineage
              transactionContext.try(submit: changeRequest)
            }
          } completionHandler: { _ in
            DispatchQueue.main.async { [weak self] in
              guard let self = self else { return }
              for (i, imageHistory) in imageHistories.enumerated() {
                if let node = self.nodeCache[imageHistory.logicalTime], node.1 == imageVersions[i] {
                  self.nodeCache[imageHistory.logicalTime] = nil
                }
                let logicalTimeAndLineage = LogicalTimeAndLineage(
                  logicalTime: imageHistory.logicalTime, lineage: newLineage)
                if let node = self.nodeLineageCache[logicalTimeAndLineage],
                  node.1 == imageVersions[i]
                {
                  self.nodeLineageCache[logicalTimeAndLineage] = nil
                }
                if let node = self.imageDataCache[logicalTimeAndLineage],
                  node.1 == imageVersions[i]
                {
                  self.imageDataCache[logicalTimeAndLineage] = nil
                }
                if let node = self.shuffleDataCache[logicalTimeAndLineage],
                  node.1 == imageVersions[i]
                {
                  self.shuffleDataCache[logicalTimeAndLineage] = nil
                }
              }
            }
          }
        }
      }
      lineage = maxLineage + 1
      maxLineage = lineage
      maxLogicalTime = logicalTime
      maxLogicalTimeForLineage[lineage] = maxLogicalTime
    }
    let imageData = history.imageData
    let shuffleData = history.shuffleData ?? self.shuffleData
    // Only moving forward if we are not empty. Otherwise just update the empty state.
    if !imageData.isEmpty || !self.imageData.isEmpty || !shuffleData.isEmpty
      || !self.shuffleData.isEmpty
    {
      logicalTime += 1
    }
    maxLogicalTime = logicalTime
    maxLogicalTimeForLineage[lineage] = maxLogicalTime
    self.tensorId = nil
    self.maskId = nil
    self.depthMapId = nil
    self.scribbleId = nil
    self.poseId = nil
    self.colorPaletteId = nil
    self.customId = nil
    self.scaleFactor = 1
    self.isGenerated = history.isGenerated
    self.configuration = history.configuration
    self.dataStored = history.imageData
    self.contentOffset = history.contentOffset
    self.scaleFactorBy120 = history.scaleFactorBy120
    self.shuffleData = shuffleData
    let profileData: [UInt8]? = {
      guard let profile = history.profile else { return nil }
      let jsonEncoder = JSONEncoder()
      jsonEncoder.keyEncodingStrategy = .convertToSnakeCase
      return (try? jsonEncoder.encode(profile)).map { [UInt8]($0) }
    }()
    self._profileData = profileData
    let previewId: Int64?
    if let preview = history.preview {
      var id: Int64 = 0
      for item in imageData {
        id += Int64(item.tensorId ?? 0) + Int64(item.maskId ?? 0) + Int64(item.depthMapId ?? 0)
        id +=
          Int64(item.scribbleId ?? 0) + Int64(item.poseId ?? 0) + Int64(item.colorPaletteId ?? 0)
        id += Int64(item.customId ?? 0)
      }
      for shuffleData in shuffleData {
        id += Int64(shuffleData.shuffleId) * Int64((shuffleData.weight * 1000).rounded())
      }
      previewCache[id] = preview
      previewId = id
    } else {
      previewId = nil
    }
    self.previewId = previewId
    let configuration = history.configuration
    let tensorHistoryNode = TensorHistoryNode(
      lineage: lineage, logicalTime: logicalTime, startWidth: configuration.startWidth,
      startHeight: configuration.startHeight, seed: configuration.seed, steps: configuration.steps,
      guidanceScale: configuration.guidanceScale, strength: configuration.strength,
      model: configuration.model, tensorId: nil, maskId: nil,
      wallClock: Int64(Date().timeIntervalSince1970),
      textEdits: history.textEdits.map { Int64($0) },
      textLineage: history.textLineage, batchSize: configuration.batchSize,
      sampler: SamplerType(from: configuration.sampler), hiresFix: configuration.hiresFix,
      hiresFixStartWidth: configuration.hiresFixStartWidth,
      hiresFixStartHeight: configuration.hiresFixStartHeight,
      hiresFixStrength: configuration.hiresFixStrength,
      upscaler: configuration.upscaler, scaleFactor: 1,
      depthMapId: nil, generated: isGenerated,
      imageGuidanceScale: configuration.imageGuidanceScale,
      seedMode: SeedMode(from: configuration.seedMode), clipSkip: configuration.clipSkip,
      controls: configuration.controls.map { Control(from: $0) }, scribbleId: nil,
      poseId: nil, loras: configuration.loras.map { LoRA(from: $0) },
      colorPaletteId: nil, maskBlur: configuration.maskBlur, customId: nil,
      faceRestoration: configuration.faceRestoration,
      clipWeight: configuration.clipWeight,
      negativePromptForImagePrior: configuration.negativePromptForImagePrior,
      imagePriorSteps: configuration.imagePriorSteps, dataStored: Int32(imageData.count),
      previewId: previewId, contentOffsetX: contentOffset.x, contentOffsetY: contentOffset.y,
      scaleFactorBy120: scaleFactorBy120, refinerModel: configuration.refinerModel,
      originalImageHeight: configuration.originalImageHeight,
      originalImageWidth: configuration.originalImageWidth, cropTop: configuration.cropTop,
      cropLeft: configuration.cropLeft, targetImageHeight: configuration.targetImageHeight,
      targetImageWidth: configuration.targetImageWidth,
      aestheticScore: configuration.aestheticScore,
      negativeAestheticScore: configuration.negativeAestheticScore,
      zeroNegativePrompt: configuration.zeroNegativePrompt,
      refinerStart: configuration.refinerStart,
      negativeOriginalImageHeight: configuration.negativeOriginalImageHeight,
      negativeOriginalImageWidth: configuration.negativeOriginalImageWidth,
      shuffleDataStored: Int32(shuffleData.count), fpsId: configuration.fpsId,
      motionBucketId: configuration.motionBucketId, condAug: configuration.condAug,
      startFrameCfg: configuration.startFrameCfg, numFrames: configuration.numFrames,
      maskBlurOutset: configuration.maskBlurOutset, sharpness: configuration.sharpness,
      shift: configuration.shift, stage2Steps: configuration.stage2Steps,
      stage2Cfg: configuration.stage2Cfg, stage2Shift: configuration.stage2Shift,
      tiledDecoding: configuration.tiledDecoding,
      decodingTileWidth: configuration.decodingTileWidth,
      decodingTileHeight: configuration.decodingTileHeight,
      decodingTileOverlap: configuration.decodingTileOverlap,
      stochasticSamplingGamma: configuration.stochasticSamplingGamma,
      preserveOriginalAfterInpaint: configuration.preserveOriginalAfterInpaint,
      tiledDiffusion: configuration.tiledDiffusion,
      diffusionTileWidth: configuration.diffusionTileWidth,
      diffusionTileHeight: configuration.diffusionTileHeight,
      diffusionTileOverlap: configuration.diffusionTileOverlap,
      upscalerScaleFactor: configuration.upscalerScaleFactor,
      scriptSessionId: history.scriptSessionId,
      t5TextEncoder: configuration.t5TextEncoder,
      separateClipL: configuration.separateClipL,
      clipLText: configuration.clipLText,
      separateOpenClipG: configuration.separateOpenClipG,
      openClipGText: configuration.openClipGText,
      speedUpWithGuidanceEmbed: configuration.speedUpWithGuidanceEmbed,
      guidanceEmbed: configuration.guidanceEmbed,
      resolutionDependentShift: configuration.resolutionDependentShift,
      teaCacheStart: configuration.teaCacheStart,
      teaCacheEnd: configuration.teaCacheEnd,
      teaCacheThreshold: configuration.teaCacheThreshold,
      teaCache: configuration.teaCache,
      separateT5: configuration.separateT5,
      t5Text: configuration.t5Text,
      teaCacheMaxSkipSteps: configuration.teaCacheMaxSkipSteps,
      textPrompt: history.textPrompt,
      negativeTextPrompt: history.negativeTextPrompt
    )
    let imageVersion = uniqueVersion()
    nodeCache[logicalTime] = (tensorHistoryNode, imageVersion)
    let logicalTimeAndLineage = LogicalTimeAndLineage(logicalTime: logicalTime, lineage: lineage)
    nodeLineageCache[logicalTimeAndLineage] = (tensorHistoryNode, imageVersion)
    let tensorData = imageData.enumerated().map {
      TensorData(
        lineage: lineage, logicalTime: logicalTime, index: Int64($0), x: $1.x, y: $1.y,
        width: $1.width, height: $1.height, scaleFactorBy120: $1.scaleFactorBy120,
        tensorId: $1.tensorId, maskId: $1.maskId, depthMapId: $1.depthMapId,
        scribbleId: $1.scribbleId, poseId: $1.poseId, colorPaletteId: $1.colorPaletteId,
        customId: $1.customId)
    }
    let tensorMoodboardData = shuffleData.enumerated().map {
      TensorMoodboardData(
        lineage: lineage, logicalTime: logicalTime, index: Int64($0), shuffleId: $1.shuffleId,
        weight: $1.weight)
    }
    shuffleDataCache[logicalTimeAndLineage] = (tensorMoodboardData, imageVersion)
    imageDataCache[logicalTimeAndLineage] = (tensorData, imageVersion)
    project.dictionary["image_seek_to", Int.self] = nil
    project.performChanges([
      TensorHistoryNode.self, ThumbnailHistoryNode.self, ThumbnailHistoryHalfNode.self,
      TensorData.self, TensorMoodboardData.self,
    ]) { transactionContext in
      // It is OK if this is already inserted.
      let upsertRequest = TensorHistoryNodeChangeRequest.upsertRequest(tensorHistoryNode)
      if let preview = history.preview, let previewId = previewId {
        let previewData = preview.jpegData(compressionQuality: 0.75)
        let downsampleData = downsampleImage(preview)?.jpegData(compressionQuality: 0.5)
        let thumbnailHistoryNodeChangeRequest = ThumbnailHistoryNodeChangeRequest.creationRequest()
        thumbnailHistoryNodeChangeRequest.id = previewId
        thumbnailHistoryNodeChangeRequest.data = previewData.map { [UInt8]($0) } ?? []
        let thumbnailHistoryHalfNodeChangeRequest =
          ThumbnailHistoryHalfNodeChangeRequest.creationRequest()
        thumbnailHistoryHalfNodeChangeRequest.id = previewId
        thumbnailHistoryHalfNodeChangeRequest.data = downsampleData.map { [UInt8]($0) } ?? []
        if thumbnailHistoryNodeChangeRequest.data.count > 0 {
          let _ = try? transactionContext.submit(thumbnailHistoryNodeChangeRequest)
        }
        if thumbnailHistoryHalfNodeChangeRequest.data.count > 0 {
          let _ = try? transactionContext.submit(thumbnailHistoryHalfNodeChangeRequest)
        }
      }
      if let profileData = profileData {
        upsertRequest.profileData = profileData
      }
      transactionContext.try(submit: upsertRequest)
      for item in tensorData {
        let upsertRequest = TensorDataChangeRequest.upsertRequest(item)
        transactionContext.try(submit: upsertRequest)
      }
      for moodItem in tensorMoodboardData {
        let upsertRequest = TensorMoodboardDataChangeRequest.upsertRequest(moodItem)
        transactionContext.try(submit: upsertRequest)
      }
    } completionHandler: { _ in
      DispatchQueue.main.async { [weak self] in
        guard let self = self else { return }
        if let previewId = previewId {
          self.previewCache[previewId] = nil
        }
        if let node = self.nodeCache[tensorHistoryNode.logicalTime], node.1 == imageVersion {
          self.nodeCache[tensorHistoryNode.logicalTime] = nil
        }
        if let node = self.nodeLineageCache[logicalTimeAndLineage], node.1 == imageVersion {
          self.nodeLineageCache[logicalTimeAndLineage] = nil
        }
        if let node = self.imageDataCache[logicalTimeAndLineage], node.1 == imageVersion {
          self.imageDataCache[logicalTimeAndLineage] = nil
        }
        if let node = self.shuffleDataCache[logicalTimeAndLineage], node.1 == imageVersion {
          self.shuffleDataCache[logicalTimeAndLineage] = nil
        }
      }
    }
  }

  public func image(
    tensorId: Int64? = nil, maskId: Int64? = nil, depthMapId: Int64? = nil,
    scribbleId: Int64? = nil, poseId: Int64? = nil, colorPaletteId: Int64? = nil,
    customId: Int64? = nil, shuffleId: Int64? = nil
  ) -> (
    Tensor<FloatType>?, Tensor<UInt8>?, Tensor<FloatType>?, Tensor<UInt8>?, Tensor<Float>?,
    Tensor<FloatType>?, Tensor<FloatType>?, Tensor<FloatType>?
  ) {
    var image: Tensor<FloatType>? = nil
    var binaryMask: Tensor<UInt8>? = nil
    var depthMap: Tensor<FloatType>? = nil
    var scribble: Tensor<UInt8>? = nil
    var pose: Tensor<Float>? = nil
    var colorPalette: Tensor<FloatType>? = nil
    var custom: Tensor<FloatType>? = nil
    var shuffle: Tensor<FloatType>? = nil
    dynamicGraph.openStore(filePath, flags: [.readOnly]) {
      if let tensorId = tensorId {
        image = $0.read("tensor_history_\(tensorId)", codec: [.fpzip, .zip]).map {
          Tensor<FloatType>(from: $0)
        }
      }
      if let maskId = maskId {
        binaryMask = $0.read("binary_mask_\(maskId)", codec: [.fpzip, .zip]).map {
          Tensor<UInt8>($0)
        }
      }
      if let depthMapId = depthMapId {
        depthMap = $0.read("depth_map_\(depthMapId)", codec: [.fpzip, .zip]).map {
          Tensor<FloatType>(from: $0)
        }
      }
      if let scribbleId = scribbleId {
        scribble = $0.read("scribble_\(scribbleId)", codec: [.fpzip, .zip]).map {
          Tensor<UInt8>($0)
        }
      }
      if let poseId = poseId {
        pose = $0.read("pose_\(poseId)", codec: [.fpzip, .zip]).map { Tensor<Float>($0) }
      }
      if let colorPaletteId = colorPaletteId {
        colorPalette = $0.read("color_palette_\(colorPaletteId)", codec: [.fpzip, .zip]).map {
          Tensor<FloatType>(from: $0)
        }
      }
      if let customId = customId {
        custom = $0.read("custom_\(customId)", codec: [.fpzip, .zip]).map {
          Tensor<FloatType>(from: $0)
        }
      }
      if let shuffleId = shuffleId {
        shuffle = $0.read("shuffle_\(shuffleId)", codec: [.fpzip, .zip]).map {
          Tensor<FloatType>(from: $0)
        }
      }
    }
    return (image, binaryMask, depthMap, scribble, pose, colorPalette, custom, shuffle)
  }

  public func addImage(
    _ image: Tensor<FloatType>?, binaryMask: Tensor<UInt8>?, depthMap: Tensor<FloatType>?,
    scribble: Tensor<UInt8>?, pose: Tensor<Float>?, colorPalette: Tensor<FloatType>?,
    custom: Tensor<FloatType>?, mood: Tensor<FloatType>? = nil
  ) -> (Int64?, Int64?, Int64?, Int64?, Int64?, Int64?, Int64?, Int64?) {
    let tensorId: Int64? = image != nil ? Int64.random(in: 100_000_000...299_999_999) : nil
    let maskId: Int64? = binaryMask != nil ? Int64.random(in: 300_000_000...599_999_999) : nil
    let depthMapId: Int64? = depthMap != nil ? Int64.random(in: 600_000_000...799_999_999) : nil
    let scribbleId: Int64? = scribble != nil ? Int64.random(in: 800_000_000...999_999_999) : nil
    let poseId: Int64? = pose != nil ? Int64.random(in: 1_000_000_000...1_199_999_999) : nil
    let colorPaletteId: Int64? =
      colorPalette != nil ? Int64.random(in: 1_200_000_000...1_399_999_999) : nil
    let customId: Int64? = custom != nil ? Int64.random(in: 1_400_000_000...1_599_999_999) : nil
    let shuffleId: Int64? = mood != nil ? Int64.random(in: 1_600_000_000...1_799_999_999) : nil
    dynamicGraph.openStore(filePath, flags: []) {
      if let image = image, let tensorId = tensorId {
        $0.write("tensor_history_\(tensorId)", tensor: image, codec: [.fpzip, .zip])
      }
      if let binaryMask = binaryMask, let maskId = maskId {
        $0.write("binary_mask_\(maskId)", tensor: binaryMask, codec: [.fpzip, .zip])
      }
      if let depthMap = depthMap, let depthMapId = depthMapId {
        $0.write("depth_map_\(depthMapId)", tensor: depthMap, codec: [.fpzip, .zip])
      }
      if let scribble = scribble, let scribbleId = scribbleId {
        $0.write("scribble_\(scribbleId)", tensor: scribble, codec: [.fpzip, .zip])
      }
      if let pose = pose, let poseId = poseId {
        $0.write("pose_\(poseId)", tensor: pose, codec: [.fpzip, .zip])
      }
      if let colorPalette = colorPalette, let colorPaletteId = colorPaletteId {
        $0.write("color_palette_\(colorPaletteId)", tensor: colorPalette, codec: [.fpzip, .zip])
      }
      if let custom = custom, let customId = customId {
        $0.write("custom_\(customId)", tensor: custom, codec: [.fpzip, .zip])
      }
      if let mood = mood, let shuffleId = shuffleId {
        $0.write("shuffle_\(shuffleId)", tensor: mood, codec: [.fpzip, .zip])
      }
    }
    return (tensorId, maskId, depthMapId, scribbleId, poseId, colorPaletteId, customId, shuffleId)
  }

  public func allImageHistories(_ filterByIsGenerated: Bool) -> FetchedResult<TensorHistoryNode> {
    if filterByIsGenerated {
      return project.fetch(for: TensorHistoryNode.self).where(
        TensorHistoryNode.generated == true,
        orderBy: [
          TensorHistoryNode.lineage.descending, TensorHistoryNode.logicalTime.descending,
        ])
    } else {
      return project.fetch(for: TensorHistoryNode.self).all(orderBy: [
        TensorHistoryNode.lineage.descending, TensorHistoryNode.logicalTime.descending,
      ])
    }
  }

  public func seek(to logicalTime: Int64) {
    guard logicalTime != self.logicalTime else {
      return
    }
    let _ = seek(to: logicalTime, lineage: nil)
  }

  // Return whether we are on the sacred lineage, which is the lineage can be represented by the slider.
  public func seek(to logicalTime: Int64, lineage: Int64?) -> Bool {
    dispatchPrecondition(condition: .onQueue(.main))
    if lineage == nil {
      precondition(logicalTime >= 0 && logicalTime <= maxLogicalTime)
    }
    if let imageHistory =
      nodeCache[logicalTime]?.0
      ?? project.fetch(for: TensorHistoryNode.self).where(
        TensorHistoryNode.logicalTime == logicalTime,
        limit: .limit(1), orderBy: [TensorHistoryNode.lineage.descending]
      ).first
    {
      precondition(imageHistory.lineage <= maxLineage)
      if let lineage = lineage, imageHistory.lineage != lineage {
        // Need to find the imageHistory this particular lineage reference to, now we cannot record
        // which image it seek to, and we cannot update the image slider.
        guard
          let imageHistory =
            nodeLineageCache[LogicalTimeAndLineage(logicalTime: logicalTime, lineage: lineage)]?.0
            ?? project.fetch(for: TensorHistoryNode.self).where(
              TensorHistoryNode.logicalTime == logicalTime && TensorHistoryNode.lineage == lineage,
              limit: .limit(1)
            ).first
        else { return false }
        let imageData =
          imageDataCache[LogicalTimeAndLineage(logicalTime: logicalTime, lineage: lineage)]?.0
          ?? Array(
            project.fetch(for: TensorData.self).where(
              TensorData.logicalTime == logicalTime && TensorData.lineage == lineage))
        let shuffleData =
          shuffleDataCache[LogicalTimeAndLineage(logicalTime: logicalTime, lineage: lineage)]?.0
          ?? Array(
            project.fetch(for: TensorMoodboardData.self).where(
              TensorMoodboardData.logicalTime == logicalTime
                && TensorMoodboardData.lineage == lineage))
        setImageHistory(imageHistory, imageData: imageData, shuffleData: shuffleData)
        if let maxLogicalTime = maxLogicalTimeForLineage[lineage] {
          self.maxLogicalTime = maxLogicalTime
        } else {
          let maxLogicalTimeImageHistory = project.fetch(for: TensorHistoryNode.self).where(
            TensorHistoryNode.lineage == lineage,
            limit: .limit(1),
            orderBy: [TensorHistoryNode.logicalTime.descending]
          ).first!
          maxLogicalTime = maxLogicalTimeImageHistory.logicalTime
          maxLogicalTimeForLineage[lineage] = maxLogicalTime
        }
        return false
      }
      let imageData =
        imageDataCache[
          LogicalTimeAndLineage(logicalTime: logicalTime, lineage: imageHistory.lineage)]?.0
        ?? Array(
          project.fetch(for: TensorData.self).where(
            TensorData.logicalTime == logicalTime && TensorData.lineage == imageHistory.lineage))
      let shuffleData =
        shuffleDataCache[
          LogicalTimeAndLineage(logicalTime: logicalTime, lineage: imageHistory.lineage)]?.0
        ?? Array(
          project.fetch(for: TensorMoodboardData.self).where(
            TensorMoodboardData.logicalTime == logicalTime
              && TensorMoodboardData.lineage == imageHistory.lineage))
      // Even if lineage matches the requested, we may not be on the sacred lineage because  the
      // sacred lineage is shorter. Checking if the requested logicalTime is smaller than maxLogicalTime.
      guard logicalTime <= maxLogicalTime else {
        setImageHistory(imageHistory, imageData: imageData, shuffleData: shuffleData)
        return false
      }
      setImageHistory(imageHistory, imageData: imageData, shuffleData: shuffleData)
    } else {
      assert(logicalTime == 0)
      configuration = nil
      scaleFactor = 1
      isGenerated = false
      tensorId = nil
      maskId = nil
      depthMapId = nil
      scribbleId = nil
      poseId = nil
      colorPaletteId = nil
      customId = nil
      previewId = nil
      textEdits = nil
      textLineage = nil
      self.logicalTime = 0
      self.lineage = 0
      dataStored = []
      shuffleData = []
      _profileData = nil
    }
    project.dictionary["image_seek_to", Int.self] = Int(self.logicalTime)
    return true
  }

  private var _profileData: [UInt8]?
  public var profileData: GenerationProfile? {
    guard let profileData = _profileData else { return nil }
    let jsonDecoder = JSONDecoder()
    jsonDecoder.keyDecodingStrategy = .convertFromSnakeCase
    return try? jsonDecoder.decode(GenerationProfile.self, from: Data(profileData))
  }

  public func preview(
    for previewId: Int64, lowestResolutionAvailable: Bool
  )
    -> UIImage?
  {
    dispatchPrecondition(condition: .onQueue(.main))
    if let preview = previewCache[previewId] {
      return preview
    }
    if lowestResolutionAvailable {
      if let halfNode = project.fetch(for: ThumbnailHistoryHalfNode.self).where(
        ThumbnailHistoryHalfNode.id == previewId, limit: .limit(1)
      ).first {
        return UIImage(data: Data(halfNode.data))
      }
    }
    guard
      let node = project.fetch(for: ThumbnailHistoryNode.self).where(
        ThumbnailHistoryNode.id == previewId, limit: .limit(1)
      ).first
    else {
      return nil
    }
    return UIImage(data: Data(node.data))
  }

  public func popImage(
    image: inout Tensor<FloatType>?, binaryMask: inout Tensor<UInt8>?,
    depthMap: inout Tensor<FloatType>?, scribble: inout Tensor<UInt8>?, pose: inout Tensor<Float>?,
    colorPalette: inout Tensor<FloatType>?, custom: inout Tensor<FloatType>?
  ) {
    dynamicGraph.openStore(filePath, flags: []) {
      if let tensorId = tensorId {
        image = $0.read("tensor_history_\(tensorId)", codec: [.fpzip, .zip]).map {
          Tensor<FloatType>(from: $0)
        }
      } else {
        image = nil
      }
      if let maskId = maskId {
        binaryMask = $0.read("binary_mask_\(maskId)", codec: [.fpzip, .zip]).map {
          Tensor<UInt8>($0)
        }
      } else {
        binaryMask = nil
      }
      if let depthMapId = depthMapId {
        depthMap = $0.read("depth_map_\(depthMapId)", codec: [.fpzip, .zip]).map {
          Tensor<FloatType>(from: $0)
        }
      } else {
        depthMap = nil
      }
      if let scribbleId = scribbleId {
        scribble = $0.read("scribble_\(scribbleId)", codec: [.fpzip, .zip]).map {
          Tensor<UInt8>($0)
        }
      } else {
        scribble = nil
      }
      if let poseId = poseId {
        pose = $0.read("pose_\(poseId)", codec: [.fpzip, .zip]).map { Tensor<Float>($0) }
      } else {
        pose = nil
      }
      if let colorPaletteId = colorPaletteId {
        colorPalette = $0.read("color_palette_\(colorPaletteId)", codec: [.fpzip, .zip]).map {
          Tensor<FloatType>(from: $0)
        }
      } else {
        colorPalette = nil
      }
      if let customId = customId {
        custom = $0.read("custom_\(customId)", codec: [.fpzip, .zip]).map {
          Tensor<FloatType>(from: $0)
        }
      } else {
        custom = nil
      }
    }
  }

  public func deleteHistory(
    _ imageHistory: TensorHistoryNode, completionHandler: @escaping (Int64, Int64) -> Void
  ) {
    dispatchPrecondition(condition: .onQueue(.main))
    let project = project
    let oldLineage = lineage
    var oldLogicalTime = logicalTime
    var isDeleted = false
    var imageData = [TensorData]()
    var moodboardData = [TensorMoodboardData]()
    project.performChanges([
      TensorHistoryNode.self, ThumbnailHistoryNode.self, ThumbnailHistoryHalfNode.self,
      TensorData.self, TensorMoodboardData.self,
    ]) { transactionContext in
      guard let deletionRequest = TensorHistoryNodeChangeRequest.deletionRequest(imageHistory)
      else { return }
      let imageHistoryLogicalTime = deletionRequest.logicalTime
      let imageHistoryLineage = deletionRequest.lineage
      isDeleted = (oldLineage == imageHistoryLineage && oldLogicalTime == imageHistoryLogicalTime)
      imageData = Array(
        project.fetch(for: TensorData.self).where(
          TensorData.logicalTime == imageHistoryLogicalTime
            && TensorData.lineage == imageHistoryLineage))
      moodboardData = Array(
        project.fetch(for: TensorMoodboardData.self).where(
          TensorMoodboardData.logicalTime == imageHistory.logicalTime
            && TensorMoodboardData.lineage == imageHistory.lineage))
      let imageDataDeletionRequests = imageData.compactMap {
        TensorDataChangeRequest.deletionRequest($0)
      }
      let moodboardDataDeletionRequests = moodboardData.compactMap {
        TensorMoodboardDataChangeRequest.deletionRequest($0)
      }
      var affectedImageHistories: [TensorHistoryNode]
      var affectedImageData: [TensorData]
      var affectedMoodboardData: [TensorMoodboardData]
      // Now need to fix up history from this and up. If this is the only one at this time with max lineage.
      if let maxLineageImageHistory = project.fetch(for: TensorHistoryNode.self).where(
        TensorHistoryNode.logicalTime == imageHistoryLogicalTime,
        limit: .limit(1), orderBy: [TensorHistoryNode.lineage.descending]
      ).first {
        if maxLineageImageHistory.lineage > imageHistoryLineage {
          affectedImageHistories = Array(
            project.fetch(for: TensorHistoryNode.self).where(
              TensorHistoryNode.logicalTime >= imageHistoryLogicalTime
                && TensorHistoryNode.lineage >= imageHistoryLineage
                && TensorHistoryNode.lineage < maxLineageImageHistory.lineage))
          affectedImageData = Array(
            project.fetch(for: TensorData.self).where(
              TensorData.logicalTime > imageHistoryLogicalTime
                && TensorData.lineage >= imageHistoryLineage
                && TensorData.lineage < maxLineageImageHistory.lineage))
          affectedMoodboardData = Array(
            project.fetch(for: TensorMoodboardData.self).where(
              TensorMoodboardData.logicalTime > imageHistoryLogicalTime
                && TensorMoodboardData.lineage >= imageHistoryLineage
                && TensorMoodboardData.lineage < maxLineageImageHistory.lineage))
        } else {
          affectedImageHistories = Array(
            project.fetch(for: TensorHistoryNode.self).where(
              TensorHistoryNode.logicalTime >= imageHistoryLogicalTime
                && TensorHistoryNode.lineage >= imageHistoryLineage))
          affectedImageData = Array(
            project.fetch(for: TensorData.self).where(
              TensorData.logicalTime > imageHistoryLogicalTime
                && TensorData.lineage >= imageHistoryLineage))
          affectedMoodboardData = Array(
            project.fetch(for: TensorMoodboardData.self).where(
              TensorMoodboardData.logicalTime > imageHistoryLogicalTime
                && TensorMoodboardData.lineage >= imageHistoryLineage))
        }
      } else {
        affectedImageHistories = Array(
          project.fetch(for: TensorHistoryNode.self).where(
            TensorHistoryNode.logicalTime >= imageHistoryLogicalTime
              && TensorHistoryNode.lineage >= imageHistoryLineage))
        affectedImageData = Array(
          project.fetch(for: TensorData.self).where(
            TensorData.logicalTime > imageHistoryLogicalTime
              && TensorData.lineage >= imageHistoryLineage))
        affectedMoodboardData = Array(
          project.fetch(for: TensorMoodboardData.self).where(
            TensorMoodboardData.logicalTime > imageHistoryLogicalTime
              && TensorMoodboardData.lineage >= imageHistoryLineage))
      }
      // Remove the ones with logicalTime == imageHistoryLogicalTime from affectedImageHistories.
      // Then mark these have different imageHistoryLineage as unrelated.
      let unrelatedLineage = Set<Int64>(
        affectedImageHistories.compactMap {
          return $0.logicalTime == imageHistoryLogicalTime && $0.lineage != imageHistoryLineage
            ? $0.lineage : nil
        })
      affectedImageHistories = affectedImageHistories.filter {
        $0.logicalTime != imageHistoryLogicalTime && !unrelatedLineage.contains($0.lineage)
      }
      affectedImageData = affectedImageData.filter { !unrelatedLineage.contains($0.lineage) }
      affectedMoodboardData = affectedMoodboardData.filter {
        !unrelatedLineage.contains($0.lineage)
      }
      for imageHistory in affectedImageHistories {
        if imageHistory.lineage == oldLineage && imageHistory.logicalTime == oldLogicalTime {
          oldLogicalTime -= 1
          break
        }
      }
      var previewId: Int64
      if imageHistory.previewId == 0 {
        previewId = imageHistory.tensorId + imageHistory.maskId + imageHistory.depthMapId
        previewId += imageHistory.scribbleId + imageHistory.poseId + imageHistory.colorPaletteId
        previewId += imageHistory.customId
      } else {
        previewId = imageHistory.previewId
      }
      let thumbnailHistoryNode = project.fetch(for: ThumbnailHistoryNode.self).where(
        ThumbnailHistoryNode.id == previewId, limit: .limit(1)
      ).first
      let thumbnailHistoryHalfNode = project.fetch(for: ThumbnailHistoryHalfNode.self).where(
        ThumbnailHistoryHalfNode.id == previewId, limit: .limit(1)
      ).first
      // Delete this one.
      transactionContext.try(submit: deletionRequest)
      let imageHistoriesChangeRequests = affectedImageHistories.compactMap {
        TensorHistoryNodeChangeRequest.changeRequest($0)
      }
      for changeRequest in imageHistoriesChangeRequests {
        changeRequest.logicalTime -= 1
        transactionContext.try(submit: changeRequest)
      }
      let imageDataChangeRequests = affectedImageData.compactMap {
        TensorDataChangeRequest.changeRequest($0)
      }
      for changeRequest in imageDataChangeRequests {
        changeRequest.logicalTime -= 1
        transactionContext.try(submit: changeRequest)
      }
      for deletionRequest in imageDataDeletionRequests {
        transactionContext.try(submit: deletionRequest)
      }
      let moodboardDataChangeRequests = affectedMoodboardData.compactMap {
        TensorMoodboardDataChangeRequest.changeRequest($0)
      }
      for changeRequest in moodboardDataChangeRequests {
        changeRequest.logicalTime -= 1
        transactionContext.try(submit: changeRequest)
      }
      for deletionRequest in moodboardDataDeletionRequests {
        transactionContext.try(submit: deletionRequest)
      }
      if project.fetch(for: TensorHistoryNode.self).where(
        TensorHistoryNode.previewId == imageHistory.previewId, limit: .limit(1)
      ).count == 0 {  // No reference to this thumbnail any more.
        if let thumbnailHistoryNode = thumbnailHistoryNode,
          let deletionRequest = ThumbnailHistoryNodeChangeRequest.deletionRequest(
            thumbnailHistoryNode)
        {
          transactionContext.try(submit: deletionRequest)
        }
        if let thumbnailHistoryHalfNode = thumbnailHistoryHalfNode,
          let deletionRequest = ThumbnailHistoryHalfNodeChangeRequest.deletionRequest(
            thumbnailHistoryHalfNode)
        {
          transactionContext.try(submit: deletionRequest)
        }
      }
    } completionHandler: { [weak self] success in
      guard let self = self else { return }
      if success {
        self.dynamicGraph.openStore(self.filePath, flags: []) {
          let tensorIds = [imageHistory.tensorId] + imageData.map { $0.tensorId }
          for tensorId in tensorIds {
            if tensorId > 0
              && project.fetch(for: TensorHistoryNode.self).where(
                TensorHistoryNode.tensorId == tensorId, limit: .limit(1)
              ).count == 0
              && project.fetch(for: TensorData.self).where(
                TensorData.tensorId == tensorId, limit: .limit(1)
              ).count == 0
            {
              $0.remove("tensor_history_\(tensorId)")
            }
          }
          let maskIds = [imageHistory.maskId] + imageData.map { $0.maskId }
          for maskId in maskIds {
            if maskId > 0
              && project.fetch(for: TensorHistoryNode.self).where(
                TensorHistoryNode.maskId == maskId, limit: .limit(1)
              ).count == 0
              && project.fetch(for: TensorData.self).where(
                TensorData.maskId == maskId, limit: .limit(1)
              ).count == 0
            {
              $0.remove("binary_mask_\(maskId)")
            }
          }
          let depthMapIds = [imageHistory.depthMapId] + imageData.map { $0.depthMapId }
          for depthMapId in depthMapIds {
            if depthMapId > 0
              && project.fetch(for: TensorHistoryNode.self).where(
                TensorHistoryNode.depthMapId == depthMapId, limit: .limit(1)
              ).count == 0
              && project.fetch(for: TensorData.self).where(
                TensorData.depthMapId == depthMapId, limit: .limit(1)
              ).count == 0
            {
              $0.remove("depth_map_\(depthMapId)")
            }
          }
          let scribbleIds = [imageHistory.scribbleId] + imageData.map { $0.scribbleId }
          for scribbleId in scribbleIds {
            if imageHistory.scribbleId > 0
              && project.fetch(for: TensorHistoryNode.self).where(
                TensorHistoryNode.scribbleId == scribbleId, limit: .limit(1)
              ).count == 0
              && project.fetch(for: TensorData.self).where(
                TensorData.scribbleId == scribbleId, limit: .limit(1)
              ).count == 0
            {
              $0.remove("scribble_\(scribbleId)")
            }
          }
          let poseIds = [imageHistory.poseId] + imageData.map { $0.poseId }
          for poseId in poseIds {
            if poseId > 0
              && project.fetch(for: TensorHistoryNode.self).where(
                TensorHistoryNode.poseId == poseId, limit: .limit(1)
              ).count == 0
              && project.fetch(for: TensorData.self).where(
                TensorData.poseId == poseId, limit: .limit(1)
              ).count == 0
            {
              $0.remove("pose_\(poseId)")
            }
          }
          let colorPaletteIds = [imageHistory.colorPaletteId] + imageData.map { $0.colorPaletteId }
          for colorPaletteId in colorPaletteIds {
            if colorPaletteId > 0
              && project.fetch(for: TensorHistoryNode.self).where(
                TensorHistoryNode.colorPaletteId == colorPaletteId, limit: .limit(1)
              ).count == 0
              && project.fetch(for: TensorData.self).where(
                TensorData.colorPaletteId == colorPaletteId, limit: .limit(1)
              ).count == 0
            {
              $0.remove("color_palette_\(colorPaletteId)")
            }
          }
          let customIds = [imageHistory.customId] + imageData.map { $0.customId }
          for customId in customIds {
            if customId > 0
              && project.fetch(for: TensorHistoryNode.self).where(
                TensorHistoryNode.customId == customId, limit: .limit(1)
              ).count == 0
              && project.fetch(for: TensorData.self).where(
                TensorData.customId == customId, limit: .limit(1)
              ).count == 0
            {
              $0.remove("custom_\(customId)")
            }
          }
          let shuffleIds = moodboardData.map { $0.shuffleId }
          for shuffleId in shuffleIds {
            if shuffleId > 0
              && project.fetch(for: TensorMoodboardData.self).where(
                TensorMoodboardData.shuffleId == shuffleId, limit: .limit(1)
              ).count == 0
            {
              $0.remove("shuffle_\(shuffleId)")
            }
          }
        }
      }
      DispatchQueue.main.async { [weak self] in
        guard let self = self else { return }
        if success {
          if let (imageHistory, imageData, shuffleData) =
            (project.fetchWithinASnapshot {
              () -> (TensorHistoryNode, [TensorData], [TensorMoodboardData])? in
              let imageHistories = project.fetch(for: TensorHistoryNode.self).all(
                limit: .limit(1),
                orderBy: [
                  TensorHistoryNode.lineage.descending, TensorHistoryNode.logicalTime.descending,
                ])
              guard let imageHistory = imageHistories.first else {
                return nil
              }
              let imageData: [TensorData]
              if imageHistory.dataStored > 0 {
                imageData = Array(
                  project.fetch(for: TensorData.self).where(
                    TensorData.lineage == imageHistory.lineage
                      && TensorData.logicalTime == imageHistory.logicalTime))
              } else {
                imageData = []
              }
              let shuffleData: [TensorMoodboardData]
              if imageHistory.shuffleDataStored > 0 {
                shuffleData = Array(
                  project.fetch(for: TensorMoodboardData.self).where(
                    TensorMoodboardData.lineage == imageHistory.lineage
                      && TensorMoodboardData.logicalTime == imageHistory.logicalTime))
              } else {
                shuffleData = []
              }
              return (imageHistory, imageData, shuffleData)
            })
          {
            self.maxLineage = imageHistory.lineage
            self.maxLogicalTime = imageHistory.logicalTime
            self.maxLogicalTimeForLineage[self.lineage] = self.maxLogicalTime
            if isDeleted {
              // Just seek to the very beginning.
              self.setImageHistory(imageHistory, imageData: imageData, shuffleData: shuffleData)
              self.project.dictionary["image_seek_to", Int.self] = Int(imageHistory.logicalTime)
            } else {
              let _ = self.seek(to: oldLogicalTime, lineage: oldLineage)
            }
          } else {
            self.maxLineage = 0
            self.maxLogicalTime = 0
            self.lineage = 0
            self.logicalTime = 0
            self.dataStored = []
            self.shuffleData = []
            self.maxLogicalTimeForLineage[self.lineage] = self.maxLogicalTime
            self.project.dictionary["image_seek_to", Int.self] = 0
          }
        }
        completionHandler(self.logicalTime, self.lineage)
      }
    }
  }

  public func garbageCollectAndZip(completion: @escaping () -> Void) {
    let imageHistories = project.fetch(for: TensorHistoryNode.self).all()
    let imageData = project.fetch(for: TensorData.self).all()
    let shuffleData = project.fetch(for: TensorMoodboardData.self).all()
    var keys = Set<String>()
    var previewIds = Set<Int64>()
    for item in imageData {
      if item.tensorId > 0 {
        keys.insert("tensor_history_\(item.tensorId)")
      }
      if item.maskId > 0 {
        keys.insert("binary_mask_\(item.maskId)")
      }
      if item.depthMapId > 0 {
        keys.insert("depth_map_\(item.depthMapId)")
      }
      if item.scribbleId > 0 {
        keys.insert("scribble_\(item.scribbleId)")
      }
      if item.poseId > 0 {
        keys.insert("pose_\(item.poseId)")
      }
      if item.colorPaletteId > 0 {
        keys.insert("color_palette_\(item.colorPaletteId)")
      }
      if item.customId > 0 {
        keys.insert("custom_\(item.customId)")
      }
    }
    for item in shuffleData {
      keys.insert("shuffle_\(item.shuffleId)")
    }
    for imageHistory in imageHistories {
      if imageHistory.tensorId > 0 {
        keys.insert("tensor_history_\(imageHistory.tensorId)")
      }
      if imageHistory.maskId > 0 {
        keys.insert("binary_mask_\(imageHistory.maskId)")
      }
      if imageHistory.depthMapId > 0 {
        keys.insert("depth_map_\(imageHistory.depthMapId)")
      }
      if imageHistory.scribbleId > 0 {
        keys.insert("scribble_\(imageHistory.scribbleId)")
      }
      if imageHistory.poseId > 0 {
        keys.insert("pose_\(imageHistory.poseId)")
      }
      if imageHistory.colorPaletteId > 0 {
        keys.insert("color_palette_\(imageHistory.colorPaletteId)")
      }
      if imageHistory.customId > 0 {
        keys.insert("custom_\(imageHistory.customId)")
      }
      var previewId: Int64
      if imageHistory.previewId == 0 {
        previewId = imageHistory.tensorId + imageHistory.maskId + imageHistory.depthMapId
        previewId += imageHistory.scribbleId + imageHistory.poseId + imageHistory.colorPaletteId
        previewId += imageHistory.customId
      } else {
        previewId = imageHistory.previewId
      }
      previewIds.insert(previewId)
    }
    let project = project
    let dynamicGraph = dynamicGraph
    let filePath = filePath
    project.performChanges([ThumbnailHistoryNode.self, ThumbnailHistoryHalfNode.self]) {
      transactionContext in
      let thumbnailHistoryNodes = project.fetch(for: ThumbnailHistoryNode.self).where(
        ThumbnailHistoryNode.id.notIn(previewIds))
      let thumbnailHistoryHalfNodes = project.fetch(for: ThumbnailHistoryHalfNode.self).where(
        ThumbnailHistoryHalfNode.id.notIn(previewIds))
      let deletionRequestsForThumbnailHistoryNode = thumbnailHistoryNodes.compactMap {
        ThumbnailHistoryNodeChangeRequest.deletionRequest($0)
      }
      let deletionRequestsForThumbnailHistoryHalfNode = thumbnailHistoryHalfNodes.compactMap {
        ThumbnailHistoryHalfNodeChangeRequest.deletionRequest($0)
      }
      for deletionRequest in deletionRequestsForThumbnailHistoryNode {
        transactionContext.try(submit: deletionRequest)
      }
      for deletionRequest in deletionRequestsForThumbnailHistoryHalfNode {
        transactionContext.try(submit: deletionRequest)
      }
    } completionHandler: { _ in
      dynamicGraph.openStore(filePath) {
        for key in $0.keys {
          guard !keys.contains(key) else { continue }
          $0.remove(key)
        }
        for key in $0.keys {
          guard let codec = $0.codec(for: key), codec.isEmpty else { continue }
          guard let tensor = $0.read(key, codec: [.fpzip, .zip]) else { continue }
          $0.write(key, tensor: tensor, codec: [.fpzip, .zip])
        }
      }
      completion()
    }
  }
}
