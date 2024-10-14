import AVFoundation
import Diffusion
import NNC
import UIKit

public protocol VideoExporterDelegate: AnyObject {
  func didProcessedFrameCount(processedFrameCount: Int)
}

public final class VideoExporter {
  public weak var delegate: VideoExporterDelegate? = nil
  private var pendingFrames = [CVPixelBuffer]()
  private lazy var videoExportQueue = DispatchQueue(
    label: "com.draw-things.video-exporter", qos: .userInteractive)
  public private(set) var isExportingInProgress = false  // main queue
  private var frameAppendingFinished = false  // videoExportQueue
  private var pixelBufferPool: CVPixelBufferPool? = nil
  private var exportingImageSize: CGSize? = nil
  private var lastAppendedFrame: Tensor<FloatType>? = nil
  private var subPyramidExtractorModel: Model? = nil
  private var predictors: [Model]? = nil
  private var fusion: Model? = nil
  private var graph: DynamicGraph? = nil
  private var interpolateRounds = 0
  private var processedFrameCount = 0
  private var aborted = false

  public func abortVideoExporting() {
    videoExportQueue.async {
      self.aborted = true
    }
  }
  private let FILMPath: String?
  public init(FILMPath: String?) {
    self.FILMPath = FILMPath
  }

  public func startVideoExporting(
    frameDuration: CMTime, outputFileURL: URL, imageSize: CGSize, interpolateRounds: Int,
    completion: @escaping (Bool) -> Void
  ) {
    self.isExportingInProgress = true
    self.exportingImageSize = imageSize
    self.interpolateRounds = interpolateRounds

    videoExportQueue.async {
      self.pendingFrames.removeAll()
      self.frameAppendingFinished = false
      self.aborted = false
      self.graph = DynamicGraph()
      self.lastAppendedFrame = nil
      self.processedFrameCount = 0
      self.initialVideoExportingOnExporterQueue(
        frameDuration: frameDuration, outputFileURL: outputFileURL, imageSize: imageSize
      ) { finish in
        DispatchQueue.main.async {
          self.isExportingInProgress = false
          self.exportingImageSize = nil
          self.resetInterpolateModels()
          self.lastAppendedFrame = nil
          self.pixelBufferPool = nil
          self.interpolateRounds = 0

          completion(finish)
        }
      }
    }
  }

  private func resetInterpolateModels() {
    self.subPyramidExtractorModel = nil
    self.predictors = nil
    self.fusion = nil
    self.graph = nil
  }

  public func markFrameAppendingEnd() {
    videoExportQueue.async {
      self.frameAppendingFinished = true
    }
  }

  public func appendFrame(
    frame: Tensor<FloatType>, completion: @escaping () -> Void
  ) {
    videoExportQueue.async {
      guard !self.aborted else { return }
      self.processedFrameCount += 1  // for the original frame
      self.delegate?.didProcessedFrameCount(processedFrameCount: self.processedFrameCount)
      if let lastAppendedFrame = self.lastAppendedFrame, let graph = self.graph,
        let modelFilePath = self.FILMPath
      {
        var frames = [lastAppendedFrame, frame]
        for _ in 0..<self.interpolateRounds {
          frames = self.interpolateFrames(from: frames, modelFilePath: modelFilePath, graph: graph)
        }

        // lastAppendedFrame is already exported, we only use it to build the frames, no need export it again
        frames.removeFirst()
        for frame in frames {
          self.appendFrameToExport(frame: Tensor<FloatType>(from: frame))
        }
      } else {
        self.appendFrameToExport(frame: frame)
      }

      self.lastAppendedFrame = frame
      completion()
    }
  }

  private func interpolateFrames(
    from frames: [Tensor<FloatType>], modelFilePath: String, graph: DynamicGraph
  ) -> [Tensor<
    FloatType
  >] {
    if frames.count <= 1 {
      return frames
    }
    var frames = frames
    var previousFrame = frames.removeFirst()
    var interpolatedFrames = [Tensor<FloatType>]()
    interpolatedFrames.append(previousFrame)
    while frames.count > 0 && !self.aborted {
      let curFrame = frames.removeFirst()
      let midFrame = generateMidFrameFrom(
        previousFrame, curFrame, modelFilePath: modelFilePath, graph: graph)
      interpolatedFrames.append(midFrame)
      interpolatedFrames.append(curFrame)
      previousFrame = curFrame
    }
    return interpolatedFrames
  }

  private func generateMidFrameFrom(
    _ frame1: Tensor<FloatType>, _ frame2: Tensor<FloatType>, modelFilePath: String,
    graph: DynamicGraph
  ) -> Tensor<FloatType> {
    let midFrame: Tensor<FloatType>
    (midFrame, subPyramidExtractorModel, predictors, fusion) =
      generateFILMIntermediateImage(
        imageTensor: frame1, imageTensor2: frame2,
        modelPath: modelFilePath, presetSubPyramidExtractorModel: subPyramidExtractorModel,
        presetPredictors: predictors, presetFusion: fusion, graph: graph)
    self.processedFrameCount += 1
    self.delegate?.didProcessedFrameCount(processedFrameCount: self.processedFrameCount)
    return midFrame
  }

  private func appendFrameToExport(frame: Tensor<FloatType>) {
    guard let exportingImageSize = exportingImageSize, !self.aborted else {
      return
    }
    let imageHeight = Int(frame.shape[1])
    let imageWidth = Int(frame.shape[2])
    // skip append frame size doesn't match origin export setup
    guard imageHeight == Int(exportingImageSize.height), imageWidth == Int(exportingImageSize.width)
    else { return }
    if let pixelBufferPool = pixelBufferPool,
      let buffer = newPixelBufferFrom(
        imageTensor: frame, pixelBufferPool: pixelBufferPool)
    {
      pendingFrames.append(buffer)
    }
  }

  private func initialVideoExportingOnExporterQueue(
    frameDuration: CMTime, outputFileURL: URL, imageSize: CGSize,
    completion: @escaping (Bool) -> Void
  ) {
    dispatchPrecondition(condition: .onQueue(videoExportQueue))

    let imageHeight = imageSize.height
    let imageWidth = imageSize.width

    let videoSettings: [String: Any] = [
      AVVideoCodecKey: AVVideoCodecType.h264.rawValue,
      AVVideoWidthKey: NSNumber(value: Float(imageWidth)),
      AVVideoHeightKey: NSNumber(value: Float(imageHeight)),
    ]

    let pixelBufferAttributes = [
      kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
    ]

    let videoWriterInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
    let pixelBufferAdaptor = AVAssetWriterInputPixelBufferAdaptor(
      assetWriterInput: videoWriterInput, sourcePixelBufferAttributes: pixelBufferAttributes)

    let videoWriter = try! AVAssetWriter(outputURL: outputFileURL, fileType: .mp4)
    videoWriter.add(videoWriterInput)
    videoWriter.startWriting()
    videoWriter.startSession(atSourceTime: .zero)
    pixelBufferPool = pixelBufferAdaptor.pixelBufferPool
    guard pixelBufferPool != nil else {
      completion(false)
      return
    }

    var time = CMTime.zero
    videoWriterInput.requestMediaDataWhenReady(on: videoExportQueue) { [weak self] in
      guard let self = self else { return }
      while videoWriterInput.isReadyForMoreMediaData {
        if let buffer = self.pendingFrames.first {
          pixelBufferAdaptor.append(buffer, withPresentationTime: time)
          time = CMTimeAdd(time, frameDuration)
          self.pendingFrames.removeFirst()
        } else {
          if self.pendingFrames.isEmpty, self.frameAppendingFinished {
            videoWriterInput.markAsFinished()
            videoWriter.finishWriting {
              completion(videoWriter.status == .completed)
            }
          } else if self.aborted {
            videoWriter.cancelWriting()
            completion(false)
          } else {
            break
          }
        }

      }
      Thread.sleep(forTimeInterval: 0.01)
    }
  }

  private func newPixelBufferFrom(
    imageTensor: Tensor<FloatType>, pixelBufferPool: CVPixelBufferPool
  )
    -> CVPixelBuffer?
  {
    var newPixelBuffer: CVPixelBuffer?
    CVPixelBufferPoolCreatePixelBuffer(nil, pixelBufferPool, &newPixelBuffer)
    guard let pixelBuffer = newPixelBuffer else { return nil }
    CVPixelBufferLockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 0))

    guard let pixelBaseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
      return nil
    }
    let imageHeight = imageTensor.shape[1]
    let imageWidth = imageTensor.shape[2]
    imageTensor.withUnsafeBytes {
      guard let fp16 = $0.baseAddress?.assumingMemoryBound(to: FloatType.self) else { return }
      let pixelBaseAddressTyped = pixelBaseAddress.assumingMemoryBound(to: UInt8.self)

      for i in 0..<imageHeight * imageWidth {
        // BGRA
        pixelBaseAddressTyped[i * 4] = UInt8(min(max(Int((fp16[i * 3 + 2] + 1) * 127.5), 0), 255))
        pixelBaseAddressTyped[i * 4 + 1] = UInt8(
          min(max(Int((fp16[i * 3 + 1] + 1) * 127.5), 0), 255))
        pixelBaseAddressTyped[i * 4 + 2] = UInt8(min(max(Int((fp16[i * 3] + 1) * 127.5), 0), 255))
        pixelBaseAddressTyped[i * 4 + 3] = 255
      }
    }

    CVPixelBufferUnlockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 0))

    return pixelBuffer
  }
}