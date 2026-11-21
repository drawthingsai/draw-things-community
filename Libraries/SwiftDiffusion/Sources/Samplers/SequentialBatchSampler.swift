import Foundation
import NNC

/// Samples a batch of N images with the unchanged sampler loop (one [N, ...] latent, the same
/// noise and the same sampler math as a tensor batch), but runs the model forward of every step
/// image by image. Peak activation memory then stays at the level of batch size 1.
///
/// How it works: while the wrapped sampler runs, `SequentialBatchScope` is active. Inside it
///   - `UNetFixedEncoder.encode` / `vector` compute the fixed conditioning (text, reference images,
///     timestep tables) once for a batch of 1 instead of N, so nothing is repeated N times,
///   - `UNetExtractConditions` and `UNetProtocol.timeEmbed` produce per-step inputs for one image,
///   - `UNetWrapper` compiles the model for one image and splits every forward call into N calls
///     (rows j * N + i of the input belong to image i, CFG branch j), then writes the results back
///     into one [cfgChannels * N, ...] output. CFG, the sampler update and the noise stay batched.
///
/// This requires that every image of the batch gets the same conditioning (true for every request
/// that goes through `repeatConditionsToMatchBatchSize`, checked once per batch). Whenever this or
/// another precondition does not hold (control models, TeaCache, video and multi-stage models,
/// SD 1.x/2.x and Kandinsky whose UNet takes the raw text conditioning), the original tensor batch
/// runs and the reason is logged.
public struct SequentialBatchSampler<
  FloatType: TensorNumeric & BinaryFloatingPoint, UNet: UNetProtocol
>: Sampler where UNet.FloatType == FloatType {
  public let base: any Sampler<FloatType, UNet>
  public let batchSize: Int

  public init(base: any Sampler<FloatType, UNet>, batchSize: Int) {
    self.base = base
    self.batchSize = batchSize
  }

  public var filePath: String { base.filePath }
  public var modifier: SamplerModifier { base.modifier }
  public var version: ModelVersion { base.version }
  public var audioShiftRatio: Float { base.audioShiftRatio }
  public var upcastAttention: Bool { base.upcastAttention }
  public var usesFlashAttention: UseFlashAttention { base.usesFlashAttention }
  public var usesSolAttention: Bool { base.usesSolAttention }
  public var externalOnDemand: Bool { base.externalOnDemand }
  public var injectControls: Bool { base.injectControls }
  public var injectT2IAdapters: Bool { base.injectT2IAdapters }
  public var injectIPAdapterLengths: [Int] { base.injectIPAdapterLengths }
  public var injectAttentionKV: Bool { base.injectAttentionKV }
  public var lora: [LoRAConfiguration] { base.lora }
  public var tiledDiffusion: TiledConfiguration { base.tiledDiffusion }
  public var teaCache: TeaCacheConfiguration { base.teaCache }
  public var isGuidanceEmbedEnabled: Bool { base.isGuidanceEmbedEnabled }

  public func timestep(for strength: Float, sampling: Sampling) -> (
    timestep: Float, startStep: Float, roundedDownStartStep: Int, roundedUpStartStep: Int
  ) {
    base.timestep(for: strength, sampling: sampling)
  }

  public func noiseScaleFactor(at step: Float, sampling: Sampling) -> Float {
    base.noiseScaleFactor(at: step, sampling: sampling)
  }

  public func sampleScaleFactor(at step: Float, sampling: Sampling) -> Float {
    base.sampleScaleFactor(at: step, sampling: sampling)
  }

  public func sample(
    _ x_T: DynamicGraph.Tensor<FloatType>, unets: [UNet?], sample: DynamicGraph.Tensor<FloatType>?,
    conditionImage: DynamicGraph.Tensor<FloatType>?,
    referenceImages: [DynamicGraph.Tensor<FloatType>],
    referenceAudios: [DynamicGraph.Tensor<FloatType>],
    mask: DynamicGraph.Tensor<FloatType>?, negMask: DynamicGraph.Tensor<FloatType>?,
    conditioning c: [DynamicGraph.Tensor<FloatType>], tokenLengthUncond: Int, tokenLengthCond: Int,
    extraProjection: DynamicGraph.Tensor<FloatType>?,
    injectedControls: [(
      model: ControlModel<FloatType>, hints: [([DynamicGraph.Tensor<FloatType>], Float)]
    )],
    textGuidanceScale: Float, imageGuidanceScale: Float, guidanceEmbed: Float,
    startStep: (integral: Int, fractional: Float), endStep: (integral: Int, fractional: Float),
    originalSize: (width: Int, height: Int), cropTopLeft: (top: Int, left: Int),
    targetSize: (width: Int, height: Int), aestheticScore: Float,
    negativeOriginalSize: (width: Int, height: Int), negativeAestheticScore: Float,
    zeroNegativePrompt: Bool, refiner: Refiner?, fpsId: Int, motionBucketId: Int, condAug: Float,
    startFrameCfg: Float, sharpness: Float, sampling: Sampling,
    cancellation: (@escaping () -> Void) -> Void, feedback: (Int, Tensor<FloatType>?) -> Bool
  ) -> Result<SamplerOutput<FloatType, UNet>, Error> {
    func run(
      cancellation: (@escaping () -> Void) -> Void
    ) -> Result<SamplerOutput<FloatType, UNet>, Error> {
      base.sample(
        x_T, unets: unets, sample: sample, conditionImage: conditionImage,
        referenceImages: referenceImages, referenceAudios: referenceAudios, mask: mask,
        negMask: negMask, conditioning: c, tokenLengthUncond: tokenLengthUncond,
        tokenLengthCond: tokenLengthCond, extraProjection: extraProjection,
        injectedControls: injectedControls, textGuidanceScale: textGuidanceScale,
        imageGuidanceScale: imageGuidanceScale, guidanceEmbed: guidanceEmbed,
        startStep: startStep, endStep: endStep, originalSize: originalSize,
        cropTopLeft: cropTopLeft, targetSize: targetSize, aestheticScore: aestheticScore,
        negativeOriginalSize: negativeOriginalSize,
        negativeAestheticScore: negativeAestheticScore, zeroNegativePrompt: zeroNegativePrompt,
        refiner: refiner, fpsId: fpsId, motionBucketId: motionBucketId, condAug: condAug,
        startFrameCfg: startFrameCfg, sharpness: sharpness, sampling: sampling,
        cancellation: cancellation, feedback: feedback)
    }
    let b = batchSize
    if let reason = reasonToRunAsTensorBatch(
      x_T, conditioning: c, extraProjection: extraProjection,
      injectedControlsCount: injectedControls.count, refiner: refiner)
    {
      if b > 1 {
        SequentialBatchScope.log("batch of \(b) runs as one tensor batch: \(reason)")
      }
      return run(cancellation: cancellation)
    }
    let started = Date()
    let result = SequentialBatchScope.withBatchSize(b) { scope -> Result<
      SamplerOutput<FloatType, UNet>, Error
    > in
      // A cancel stops only the forward that is running (Model.cancel), the next one would run in
      // full: the scope remembers it, so the loop over the images ends after the current image.
      let result = run(cancellation: { handler in
        cancellation {
          scope.cancel()
          handler()
        }
      })
      SequentialBatchScope.log(
        "batch of \(b) sampled step by step, model forward image by image: "
          + "\(scope.forwardCalls) forward calls in "
          + String(format: "%.1f", Date().timeIntervalSince(started)) + "s")
      return result
    }
    return result
  }
}

extension SequentialBatchSampler {
  /// Returns nil if the model forward can be split image by image, otherwise why not.
  func reasonToRunAsTensorBatch(
    _ x_T: DynamicGraph.Tensor<FloatType>, conditioning c: [DynamicGraph.Tensor<FloatType>],
    extraProjection: DynamicGraph.Tensor<FloatType>?, injectedControlsCount: Int,
    refiner: Refiner?
  ) -> String? {
    let b = batchSize
    guard b > 1 else { return "batch size 1" }
    let shape = x_T.shape
    guard shape.count == 4, shape[0] == b else {
      return "latent shape \(Array(shape)) does not match batch size \(b)"
    }
    guard Self.isSupported(version) else {
      return "model version \(version.rawValue) is not supported"
    }
    if let refiner, !Self.isSupported(refiner.version) {
      return "refiner model version \(refiner.version.rawValue) is not supported"
    }
    guard injectedControlsCount == 0 else { return "control models are not supported" }
    guard base.teaCache.threshold <= 0 else {
      return "TeaCache keeps per-step state for the whole batch"
    }
    for tensor in c + (extraProjection.map { [$0] } ?? []) {
      guard SequentialBatchScope.rowsAreIdenticalAcrossBatch(tensor, batchSize: b) else {
        return "conditioning differs between the images of the batch"
      }
    }
    return nil
  }

  /// Image models whose model inputs come entirely from `UNetFixedEncoder` (so they can be
  /// computed for one image) and that sample one frame per batch element.
  static func isSupported(_ version: ModelVersion) -> Bool {
    switch version {
    case .sdxlBase, .sdxlRefiner, .ssd1b, .sd3, .sd3Large, .pixart, .auraflow, .flux1, .hiDreamI1,
      .hiDreamO1, .qwenImage, .qwenImage2_1, .zImage, .ernieImage, .flux2, .flux2_9b, .flux2_4b,
      .cosmos2_5_2b, .ideogram4, .krea2:
      return true
    case .v1, .v2, .kandinsky21, .wurstchenStageB, .wurstchenStageC, .svdI2v, .hunyuanVideo,
      .wan21_1_3b, .wan21_14b, .wan22_5b, .ltx2, .ltx2_3, .seedvr2_3b, .seedvr2_7b,
      .longcatVideoAvatar1_5, .minimaxH3:
      return false
    case .flux3_5b, .cosmos2_5_2_9b, .mossImage:
      // not checked yet: keep the tensor batch until their inputs are known to come from the fixed encoder
      return false
    }
  }
}

/// Active while `SequentialBatchSampler` runs a batch step by step with the model forward split
/// image by image (thread-local, the sampler runs synchronously on one thread). The functions that
/// size model inputs by batch (`UNetFixedEncoder.encode` / `vector`, `UNetExtractConditions`,
/// `UNetProtocol.timeEmbed`) and `UNetWrapper` consult it.
public final class SequentialBatchScope {
  private static let threadKey = "com.draw-things.SequentialBatchScope"
  public let batchSize: Int
  public var forwardCalls = 0
  private let lock = NSLock()
  private var cancelled = false

  /// Set from the thread that cancels the generation, read by the sampling thread.
  public var isCancelled: Bool { lock.withLock { cancelled } }

  public func cancel() { lock.withLock { cancelled = true } }

  private init(batchSize: Int) {
    self.batchSize = batchSize
  }

  public static var current: SequentialBatchScope? {
    Thread.current.threadDictionary[threadKey] as? SequentialBatchScope
  }

  public static func withBatchSize<R>(_ batchSize: Int, _ body: (SequentialBatchScope) throws -> R)
    rethrows -> R
  {
    let previous = Thread.current.threadDictionary[threadKey]
    let scope = SequentialBatchScope(batchSize: batchSize)
    Thread.current.threadDictionary[threadKey] = scope
    defer { Thread.current.threadDictionary[threadKey] = previous }
    return try body(scope)
  }

  /// The batch size a model input of `batchSize` rows has for one image: `batchSize / N` inside
  /// an active scope, `batchSize` otherwise.
  public static func perImageBatchSize(_ batchSize: Int) -> Int {
    guard let scope = current, scope.batchSize > 1, batchSize % scope.batchSize == 0,
      batchSize >= scope.batchSize
    else { return batchSize }
    return batchSize / scope.batchSize
  }

  /// The text conditioning of the first image and batch size 1 when a scope for `batchSize`
  /// images is active (every image has the same rows), the inputs unchanged otherwise.
  public static func firstImage<FloatType: TensorNumeric & BinaryFloatingPoint>(
    textEncoding: [DynamicGraph.Tensor<FloatType>], batchSize: Int
  ) -> ([DynamicGraph.Tensor<FloatType>], Int) {
    guard let scope = current, scope.batchSize > 1, batchSize == scope.batchSize else {
      return (textEncoding, batchSize)
    }
    return (
      textEncoding.map {
        $0.shape.count == 2 || $0.shape.count == 3 ? rows($0, image: 0, batchSize: batchSize) : $0
      }, 1
    )
  }

  public static func log(_ message: String) {
    FileHandle.standardError.write(Data("[sequential-batch] \(message)\n".utf8))
  }

  /// Rows j * batchSize + i (j over all groups) of a tensor laid out as `repeatConditionsToMatchBatchSize`
  /// and the samplers lay out conditioning and model inputs. Tensors whose leading dimension is not
  /// a multiple of the batch size are shared by all images and returned unchanged.
  public static func rows<FloatType: TensorNumeric & BinaryFloatingPoint>(
    _ tensor: DynamicGraph.Tensor<FloatType>, image i: Int, batchSize b: Int
  ) -> DynamicGraph.Tensor<FloatType> {
    let shape = tensor.shape
    guard b > 1, shape.count >= 1, shape.count <= 4, shape[0] >= b, shape[0] % b == 0 else {
      return tensor
    }
    let groups = shape[0] / b
    let graph = tensor.graph
    switch shape.count {
    case 1:
      var result = graph.variable(.GPU(0), .C(groups), of: FloatType.self)
      for j in 0..<groups {
        result[j..<(j + 1)] = tensor[(j * b + i)..<(j * b + i + 1)]
      }
      return result
    case 2:
      var result = graph.variable(.GPU(0), .WC(groups, shape[1]), of: FloatType.self)
      for j in 0..<groups {
        result[j..<(j + 1), 0..<shape[1]] = tensor[(j * b + i)..<(j * b + i + 1), 0..<shape[1]]
      }
      return result
    case 3:
      var result = graph.variable(.GPU(0), .HWC(groups, shape[1], shape[2]), of: FloatType.self)
      for j in 0..<groups {
        result[j..<(j + 1), 0..<shape[1], 0..<shape[2]] =
          tensor[(j * b + i)..<(j * b + i + 1), 0..<shape[1], 0..<shape[2]]
      }
      return result
    default:
      var result = graph.variable(
        .GPU(0), .NHWC(groups, shape[1], shape[2], shape[3]), of: FloatType.self)
      for j in 0..<groups {
        result[j..<(j + 1), 0..<shape[1], 0..<shape[2], 0..<shape[3]] =
          tensor[(j * b + i)..<(j * b + i + 1), 0..<shape[1], 0..<shape[2], 0..<shape[3]]
      }
      return result
    }
  }

  /// Writes the rows of `part` (one image, `groups` rows) back to rows j * batchSize + i.
  public static func scatter<FloatType: TensorNumeric & BinaryFloatingPoint>(
    _ part: DynamicGraph.Tensor<FloatType>, into result: inout DynamicGraph.Tensor<FloatType>,
    image i: Int, batchSize b: Int
  ) {
    let shape = part.shape
    precondition(
      shape.count == 4 && result.shape.count == 4 && result.shape[0] == shape[0] * b,
      "[sequential-batch] model output shape \(Array(shape)) does not match \(Array(result.shape))")
    for j in 0..<shape[0] {
      result[(j * b + i)..<(j * b + i + 1), 0..<shape[1], 0..<shape[2], 0..<shape[3]] =
        part[j..<(j + 1), 0..<shape[1], 0..<shape[2], 0..<shape[3]]
    }
  }

  /// Whether every image of the batch gets the same rows (checked on the CPU, once per batch).
  /// Rank 2 and 3 tensors are the ones `repeatConditionsToMatchBatchSize` repeats per image.
  public static func rowsAreIdenticalAcrossBatch<FloatType: TensorNumeric & BinaryFloatingPoint>(
    _ tensor: DynamicGraph.Tensor<FloatType>, batchSize b: Int
  ) -> Bool {
    let shape = tensor.shape
    guard shape.count == 2 || shape.count == 3, shape[0] >= b, shape[0] % b == 0 else {
      return true
    }
    let groups = shape[0] / b
    let cpu = tensor.rawValue.toCPU()
    return cpu.withUnsafeBytes { bytes in
      let rowBytes = bytes.count / (groups * b)
      guard rowBytes > 0, let base = bytes.baseAddress else { return false }
      for j in 0..<groups {
        let first = base + (j * b) * rowBytes
        for i in 1..<b {
          guard memcmp(first, base + (j * b + i) * rowBytes, rowBytes) == 0 else { return false }
        }
      }
      return true
    }
  }
}
