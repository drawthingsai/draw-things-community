import Foundation
import NNC

public enum Qwen3_5TextGenerationError: LocalizedError {
  case missingStoreTensor(String)
  case invalidLogits

  public var errorDescription: String? {
    switch self {
    case .missingStoreTensor(let name):
      return "Missing tensor '\(name)' in Qwen3.5 store."
    case .invalidLogits:
      return "Qwen3.5 produced NaN logits."
    }
  }
}

public struct Qwen3_5GenerationTiming: Sendable {
  public var loadAndCompileMilliseconds: Double
  public var prefillMilliseconds: Double
  public var decodeCompileMilliseconds: Double
  public var decodeLoopMilliseconds: Double
  public var decodeLoopTokens: Int

  public var decodeLoopTokensPerSecond: Double {
    decodeLoopMilliseconds > 0
      ? Double(decodeLoopTokens) / (decodeLoopMilliseconds / 1_000) : 0
  }
}

public struct Qwen3_5TextGeneration<FloatType: TensorNumeric & BinaryFloatingPoint> {
  let filePath: String
  public let configuration: Qwen3_5ModelConfiguration
  public let eosTokenIds: Set<Int32>
  public let tieEmbedding: Bool

  public init(
    filePath: String,
    configuration: Qwen3_5ModelConfiguration,
    eosTokenIds: Set<Int32> = [248_046, 248_044], tieEmbedding: Bool
  ) {
    self.filePath = filePath
    self.configuration = configuration
    self.eosTokenIds = eosTokenIds
    self.tieEmbedding = tieEmbedding
  }

  public func generate(
    graph: DynamicGraph, promptTokenIds: [Int32], maxTokens: Int,
    partialHandler: ([Int32]) -> Bool,
    timingHandler: ((Qwen3_5GenerationTiming) -> Void)? = nil
  ) throws -> [Int32] {
    precondition(!promptTokenIds.isEmpty)
    guard maxTokens > 0 else { return [] }
    let streamContext = StreamContext(.GPU(0))
    return try graph.withStream(streamContext) {
      try graph.withNoGrad {
        let hasFullAttentionLayer = (0..<configuration.layers).contains {
          !configuration.isLinearAttentionLayer($0)
        }
        let cacheCapacity = promptTokenIds.count + maxTokens + 1
        var caches = Self.makeCaches(
          graph: graph, capacity: cacheCapacity, configuration: configuration)

        let decoder: ModelBuilder<(cachedTokenLength: Int, tokenLength: Int)> = ModelBuilder {
          (tokenLengths: (cachedTokenLength: Int, tokenLength: Int), _) in
          Qwen3_5CausalLM(
            FloatType.self, tokenLength: tokenLengths.tokenLength,
            cachedTokenLength: tokenLengths.cachedTokenLength, configuration: configuration,
            includeLogits: true, outputCacheStates: true, tieEmbedding: tieEmbedding)
        }
        decoder.maxConcurrency = .limit(4)
        let promptTokens = graph.variable(
          Tensor<Int32>(promptTokenIds, .CPU, .C(promptTokenIds.count)).toGPU(0))
        let prefillAttentionInputs: [DynamicGraph.AnyTensor]
        if hasFullAttentionLayer {
          let rotary = Qwen3_5RotaryEmbedding(
            sequenceLength: promptTokenIds.count, configuration: configuration, of: FloatType.self)
          prefillAttentionInputs = [graph.variable(rotary.toGPU(0))]
        } else {
          prefillAttentionInputs = []
        }
        let prefillCacheInputs = Self.cacheInputs(
          caches, currentTokenLength: promptTokenIds.count, configuration: configuration)
        let maxDecodeCachedTokenLength = cacheCapacity - 1
        let decodeRotaryLength = max(maxTokens - 1, 0)
        let decodeRotaryGPU: DynamicGraph.Tensor<FloatType>?
        if maxTokens > 1 && hasFullAttentionLayer {
          let rotary = Qwen3_5RotaryEmbedding(
            sequenceLength: decodeRotaryLength, cachedTokenLength: promptTokenIds.count,
            configuration: configuration, of: FloatType.self)
          decodeRotaryGPU = graph.variable(rotary.toGPU(0))
        } else {
          decodeRotaryGPU = nil
        }
        let decodeCompileAttentionInputs: [DynamicGraph.AnyTensor]
        if maxTokens > 1 && hasFullAttentionLayer {
          let decodeCompileRotary = Qwen3_5RotaryEmbedding(
            sequenceLength: 1, cachedTokenLength: maxDecodeCachedTokenLength,
            configuration: configuration, of: FloatType.self)
          decodeCompileAttentionInputs = [graph.variable(decodeCompileRotary.toGPU(0))]
        } else {
          decodeCompileAttentionInputs = []
        }
        var loadAndCompileMilliseconds = 0.0
        var prefillMilliseconds = 0.0
        do {
          let inputs: [DynamicGraph.AnyTensor] =
            [promptTokens] + prefillAttentionInputs + prefillCacheInputs
          let loadStart = Date.timeIntervalSinceReferenceDate
          try graph.openStore(
            filePath, flags: .readOnly,
            externalStore: TensorData.externalStore(filePath: filePath)
          ) { store in
            decoder.compile(
              (cachedTokenLength: 0, tokenLength: promptTokenIds.count),
              inputs: inputs)
            try store.read(
              "text_model", model: decoder, strict: true, codec: [.jit, .i8x, .ezm7, .externalData])
          }
          loadAndCompileMilliseconds = (Date.timeIntervalSinceReferenceDate - loadStart) * 1_000
          let prefillStart = Date.timeIntervalSinceReferenceDate
          var prefillOutputs = decoder(
            (cachedTokenLength: 0, tokenLength: promptTokenIds.count), inputs: promptTokens,
            prefillAttentionInputs + prefillCacheInputs)
          Self.updateLinearCaches(
            &caches, from: prefillOutputs, outputOffset: 1, configuration: configuration)
          let logits = prefillOutputs[0].as(of: FloatType.self)
          // On macOS 26.4.1, native FP16/BF16 MPSGraph argmax was unreliable on Qwen vocab-sized logits.
          let prefillTokenGPU = Functional.argmax(
            DynamicGraph.Tensor<Float>(from: logits), axis: 1
          ).reshaped(.C(1))
          let prefillTokenCPU = prefillTokenGPU.toCPU()
          graph.joined()
          prefillMilliseconds = (Date.timeIntervalSinceReferenceDate - prefillStart) * 1_000
          prefillOutputs.removeAll(keepingCapacity: false)
          let prefillToken = Int32(prefillTokenCPU[0])
          var generated = [prefillToken]
          if eosTokenIds.contains(prefillToken) {
            timingHandler?(
              Qwen3_5GenerationTiming(
                loadAndCompileMilliseconds: loadAndCompileMilliseconds,
                prefillMilliseconds: prefillMilliseconds,
                decodeCompileMilliseconds: 0,
                decodeLoopMilliseconds: 0,
                decodeLoopTokens: 0))
            _ = partialHandler(generated)
            return generated
          }
          var nextTokenGPU = prefillTokenGPU
          var decodeCompileMilliseconds = 0.0
          var decodeLoopMilliseconds = 0.0
          var decodeLoopTokens = 0
          if maxTokens > 1 {
            let decodeCompileCacheInputs = Self.cacheInputs(
              caches, currentTokenLength: cacheCapacity, configuration: configuration)
            let decodeCompileInputs: [DynamicGraph.AnyTensor] =
              [nextTokenGPU] + decodeCompileAttentionInputs + decodeCompileCacheInputs
            let decodeCompileStart = Date.timeIntervalSinceReferenceDate
            decoder.compile(
              (cachedTokenLength: maxDecodeCachedTokenLength, tokenLength: 1),
              inputs: decodeCompileInputs, isEager: true)
            graph.joined()
            decodeCompileMilliseconds =
              (Date.timeIntervalSinceReferenceDate - decodeCompileStart) * 1_000
            let queueWatermark = DynamicGraph.queueWatermark
            DynamicGraph.queueWatermark = queueWatermark * 8
            defer { DynamicGraph.queueWatermark = queueWatermark }
            let decodeLoopStart = Date.timeIntervalSinceReferenceDate
            var shouldAppendPendingToken = true
            for decodeIndex in 0..<(maxTokens - 1) {
              let cachedTokenLength = promptTokenIds.count + decodeIndex
              var oneAttentionInputs = [DynamicGraph.AnyTensor]()
              if let decodeRotaryGPU = decodeRotaryGPU {
                oneAttentionInputs.append(
                  decodeRotaryGPU.reshaped(
                    .NHWC(1, 1, 1, configuration.attentionHeadDim),
                    offset: [0, decodeIndex, 0, 0],
                    strides: [
                      decodeRotaryLength * configuration.attentionHeadDim,
                      configuration.attentionHeadDim, configuration.attentionHeadDim, 1,
                    ]))
              }
              let cacheInputs = Self.cacheInputs(
                caches, currentTokenLength: cachedTokenLength + 1, configuration: configuration)
              let oldDecodedTokenCPU = nextTokenGPU.toCPU()
              do {
                var decodeOutputs = decoder(
                  (cachedTokenLength: cachedTokenLength, tokenLength: 1), inputs: nextTokenGPU,
                  oneAttentionInputs + cacheInputs)
                Self.updateLinearCaches(
                  &caches, from: decodeOutputs, outputOffset: 1, configuration: configuration)
                let logits = decodeOutputs[0].as(of: FloatType.self)
                // On macOS 26.4.1, native FP16/BF16 MPSGraph argmax was unreliable on Qwen vocab-sized logits.
                nextTokenGPU = Functional.argmax(
                  DynamicGraph.Tensor<Float>(from: logits), axis: 1
                ).reshaped(.C(1))
                decodeOutputs.removeAll(keepingCapacity: false)
              }
              decodeLoopTokens += 1
              if decodeIndex > 0 {
                let decodedToken = Int32(oldDecodedTokenCPU[0])
                generated.append(decodedToken)
                if eosTokenIds.contains(decodedToken) {
                  shouldAppendPendingToken = false
                  break
                }
              }
            }
            if shouldAppendPendingToken && generated.count < maxTokens {
              let finalTokenCPU = nextTokenGPU.toCPU()
              graph.joined()
              let decodedToken = Int32(finalTokenCPU[0])
              generated.append(decodedToken)
            } else {
              graph.joined()
            }
            decodeLoopMilliseconds =
              (Date.timeIntervalSinceReferenceDate - decodeLoopStart) * 1_000
          }
          timingHandler?(
            Qwen3_5GenerationTiming(
              loadAndCompileMilliseconds: loadAndCompileMilliseconds,
              prefillMilliseconds: prefillMilliseconds,
              decodeCompileMilliseconds: decodeCompileMilliseconds,
              decodeLoopMilliseconds: decodeLoopMilliseconds,
              decodeLoopTokens: decodeLoopTokens))
          _ = partialHandler(generated)
          return generated
        }
      }
    }
  }

  public func benchmarkTTFT(
    graph: DynamicGraph, promptTokenIds: [Int32], runs: Int, warmups: Int = 1
  ) throws -> [Double] {
    precondition(!promptTokenIds.isEmpty)
    guard runs > 0 else { return [] }
    let streamContext = StreamContext(.GPU(0))
    return try graph.withStream(streamContext) {
      try graph.withNoGrad {
        let hasFullAttentionLayer = (0..<configuration.layers).contains {
          !configuration.isLinearAttentionLayer($0)
        }
        let tokenLength = promptTokenIds.count
        var caches = Self.makeCaches(
          graph: graph, capacity: tokenLength, configuration: configuration)

        let decoder: ModelBuilder<(cachedTokenLength: Int, tokenLength: Int)> = ModelBuilder {
          (tokenLengths: (cachedTokenLength: Int, tokenLength: Int), _) in
          Qwen3_5CausalLM(
            FloatType.self, tokenLength: tokenLengths.tokenLength,
            cachedTokenLength: tokenLengths.cachedTokenLength, configuration: configuration,
            includeLogits: true, outputCacheStates: true, tieEmbedding: tieEmbedding)
        }
        decoder.maxConcurrency = .limit(4)

        func promptInputs() -> (
          tokens: DynamicGraph.Tensor<Int32>, attentionInputs: [DynamicGraph.AnyTensor],
          cacheInputs: [DynamicGraph.AnyTensor]
        ) {
          let tokens = graph.variable(
            Tensor<Int32>(promptTokenIds, .CPU, .C(tokenLength)).toGPU(0))
          let attentionInputs: [DynamicGraph.AnyTensor]
          if hasFullAttentionLayer {
            let rotary = Qwen3_5RotaryEmbedding(
              sequenceLength: tokenLength, configuration: configuration, of: FloatType.self)
            attentionInputs = [graph.variable(rotary.toGPU(0))]
          } else {
            attentionInputs = []
          }
          let cacheInputs = Self.cacheInputs(
            caches, currentTokenLength: tokenLength, configuration: configuration)
          return (tokens, attentionInputs, cacheInputs)
        }

        try graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) { store in
          let inputs = promptInputs()
          decoder.compile(
            (cachedTokenLength: 0, tokenLength: tokenLength),
            inputs: [inputs.tokens] + inputs.attentionInputs + inputs.cacheInputs)
          try store.read(
            "text_model", model: decoder, strict: true, codec: [.jit, .i8x, .ezm7, .externalData])
        }

        func runPrefill() {
          Self.resetLinearCaches(&caches, configuration: configuration)
          graph.joined()
          let inputs = promptInputs()
          var outputs = decoder(
            (cachedTokenLength: 0, tokenLength: tokenLength), inputs: inputs.tokens,
            inputs.attentionInputs + inputs.cacheInputs)
          Self.updateLinearCaches(
            &caches, from: outputs, outputOffset: 1, configuration: configuration)
          graph.joined()
          outputs.removeAll(keepingCapacity: false)
        }

        for _ in 0..<max(0, warmups) {
          runPrefill()
        }

        var durations = [Double]()
        durations.reserveCapacity(runs)
        for _ in 0..<runs {
          Self.resetLinearCaches(&caches, configuration: configuration)
          graph.joined()
          let inputs = promptInputs()
          let start = Date.timeIntervalSinceReferenceDate
          var outputs = decoder(
            (cachedTokenLength: 0, tokenLength: tokenLength), inputs: inputs.tokens,
            inputs.attentionInputs + inputs.cacheInputs)
          Self.updateLinearCaches(
            &caches, from: outputs, outputOffset: 1, configuration: configuration)
          graph.joined()
          let end = Date.timeIntervalSinceReferenceDate
          durations.append((end - start) * 1_000)
          if outputs[0].isNaN {
            throw Qwen3_5TextGenerationError.invalidLogits
          }
          outputs.removeAll(keepingCapacity: false)
        }
        return durations
      }
    }
  }

  public func generateMultimodal(
    graph: DynamicGraph, promptTokenIds: [Int32], tokenTypeIds: [Int32],
    imagePatches: Tensor<Float>, imageGridThw: [(t: Int, h: Int, w: Int)],
    visionConfiguration: Qwen3_5VisionConfiguration = .qwen3_5_4B, maxTokens: Int,
    partialHandler: ([Int32]) -> Bool
  ) throws -> [Int32] {
    precondition(!promptTokenIds.isEmpty)
    precondition(promptTokenIds.count == tokenTypeIds.count)
    guard maxTokens > 0 else { return [] }
    let streamContext = StreamContext(.GPU(0))
    return try graph.withStream(streamContext) {
      try graph.withNoGrad {
        let imageTokenPositions = promptTokenIds.enumerated().compactMap {
          $0.element == 248_056 ? Int32($0.offset) : nil
        }
        let expectedImageTokenCount = imageGridThw.reduce(0) {
          $0 + $1.t * $1.h * $1.w
            / (visionConfiguration.spatialMergeSize * visionConfiguration.spatialMergeSize)
        }
        precondition(imageTokenPositions.count == expectedImageTokenCount)

        let hasFullAttentionLayer = (0..<configuration.layers).contains {
          !configuration.isLinearAttentionLayer($0)
        }
        let cacheCapacity = promptTokenIds.count + maxTokens + 1
        var caches = Self.makeCaches(
          graph: graph, capacity: cacheCapacity, configuration: configuration)

        let rotary = Qwen3_5VisionRotaryEmbedding(
          gridThw: imageGridThw, configuration: visionConfiguration, of: FloatType.self)
        let sequenceOffsets = Qwen3_5VisionSequenceOffsets(gridThw: imageGridThw).offsets
        let rotaryGPU = graph.variable(rotary.toGPU(0))
        let sequenceOffsetsGPU = graph.variable(sequenceOffsets.toGPU(0))
        var positionEmbedding: Tensor<Float>?
        try graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) { store in
          guard
            let weight = store.read(
              "model.visual.pos_embed.weight", codec: [.jit, .i8x, .ezm7, .externalData])
          else {
            throw Qwen3_5TextGenerationError.missingStoreTensor(
              "model.visual.pos_embed.weight")
          }
          positionEmbedding = Qwen3_5VisionPositionEmbedding(
            weight: weight, gridThw: imageGridThw, configuration: visionConfiguration,
            of: Float.self)
        }
        let patchesGPU = graph.variable(imagePatches.toGPU(0))
        let positionGPU = graph.variable(positionEmbedding!.toGPU(0))
        let visionModel = Qwen3_5VisionTransformer(
          Float.self, gridThw: imageGridThw, configuration: visionConfiguration)
        visionModel.maxConcurrency = .limit(4)
        visionModel.compile(inputs: patchesGPU, positionGPU, rotaryGPU, sequenceOffsetsGPU)
        try graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) { store in
          try store.read(
            "vision_model", model: visionModel, strict: true,
            codec: [.jit, .i8x, .ezm7, .externalData])
        }
        let imageEmbeds = visionModel(
          inputs: patchesGPU, positionGPU, rotaryGPU, sequenceOffsetsGPU
        )[0].as(of: Float.self)
        let imageEmbedsTyped = DynamicGraph.Tensor<FloatType>(from: imageEmbeds)

        let promptTokens = graph.variable(
          Tensor<Int32>(promptTokenIds, .CPU, .C(promptTokenIds.count)).toGPU(0))
        var tokenMask = Tensor<Float>(
          Array(repeating: 1, count: promptTokenIds.count), .CPU, .WC(promptTokenIds.count, 1))
        for tokenPosition in imageTokenPositions {
          tokenMask[Int(tokenPosition), 0] = 0
        }
        let tokenMaskGPU = graph.variable(Tensor<FloatType>(from: tokenMask).toGPU(0))
        var injectedEmbeddings = graph.variable(
          .GPU(0), .WC(promptTokenIds.count, configuration.hiddenSize), of: FloatType.self)
        injectedEmbeddings.full(0)
        for (imageIndex, tokenPosition) in imageTokenPositions.enumerated() {
          injectedEmbeddings[
            Int(tokenPosition)..<Int(tokenPosition + 1), 0..<configuration.hiddenSize
          ] =
            imageEmbedsTyped[
              imageIndex..<(imageIndex + 1), 0..<configuration.hiddenSize
            ]
        }

        let decoder: ModelBuilder<(cachedTokenLength: Int, tokenLength: Int)> = ModelBuilder {
          (tokenLengths: (cachedTokenLength: Int, tokenLength: Int), _) in
          Qwen3_5CausalLM(
            FloatType.self, tokenLength: tokenLengths.tokenLength,
            cachedTokenLength: tokenLengths.cachedTokenLength, configuration: configuration,
            includeLogits: true, outputCacheStates: true, tieEmbedding: tieEmbedding,
            injectEmbeddings: true)
        }
        decoder.maxConcurrency = .limit(4)

        let positionIDs = Qwen3_5MakeMultimodalPositionIDs(
          tokenTypeIDs: tokenTypeIds, imageGridThw: imageGridThw,
          configuration: visionConfiguration)
        let decodePositionOffset = positionIDs.ropeDelta
        let prefillAttentionInputs: [DynamicGraph.AnyTensor]
        if hasFullAttentionLayer {
          let prefillRotary = Qwen3_5RotaryEmbedding(
            positionIDs: positionIDs, configuration: configuration, of: FloatType.self)
          prefillAttentionInputs = [graph.variable(prefillRotary.toGPU(0))]
        } else {
          prefillAttentionInputs = []
        }
        let prefillCacheInputs = Self.cacheInputs(
          caches, currentTokenLength: promptTokenIds.count, configuration: configuration)
        let nextTokenFromPrefill: Int32
        do {
          let inputs: [DynamicGraph.AnyTensor] =
            [promptTokens, tokenMaskGPU, injectedEmbeddings] + prefillAttentionInputs
            + prefillCacheInputs
          try graph.openStore(
            filePath, flags: .readOnly,
            externalStore: TensorData.externalStore(filePath: filePath)
          ) { store in
            decoder.compile(
              (cachedTokenLength: 0, tokenLength: promptTokenIds.count),
              inputs: inputs)
            try store.read(
              "text_model", model: decoder, strict: true, codec: [.jit, .i8x, .ezm7, .externalData])
          }
          var prefillOutputs = decoder(
            (cachedTokenLength: 0, tokenLength: promptTokenIds.count), inputs: promptTokens,
            [tokenMaskGPU, injectedEmbeddings] + prefillAttentionInputs + prefillCacheInputs)
          Self.updateLinearCaches(
            &caches, from: prefillOutputs, outputOffset: 1, configuration: configuration)
          let logits = prefillOutputs[0].as(of: FloatType.self)
          let token = Functional.argmax(logits, axis: 1).toCPU()
          logits.graph.joined()
          nextTokenFromPrefill = Int32(token[0, 0])
          prefillOutputs.removeAll(keepingCapacity: false)
        }

        let nextToken = nextTokenFromPrefill
        var generated = [nextToken]
        guard partialHandler(generated), !eosTokenIds.contains(nextToken) else {
          return generated
        }
        guard maxTokens > 1 else {
          return generated
        }

        let maxDecodeCachedTokenLength = cacheCapacity - 1
        let decodeCompileToken = graph.variable(Tensor<Int32>([nextToken], .CPU, .C(1)).toGPU(0))
        let decodeTokenMask = graph.variable(
          Tensor<FloatType>(from: Tensor<Float>([1], .CPU, .WC(1, 1))).toGPU(0))
        let decodeInjectedEmbeddings = graph.variable(
          .GPU(0), .WC(1, configuration.hiddenSize), of: FloatType.self)
        decodeInjectedEmbeddings.full(0)
        var decodeCompileAttentionInputs = [DynamicGraph.AnyTensor]()
        if hasFullAttentionLayer {
          let decodeCompileRotary = Qwen3_5RotaryEmbedding(
            sequenceLength: 1,
            cachedTokenLength: maxDecodeCachedTokenLength + decodePositionOffset,
            configuration: configuration, of: FloatType.self)
          decodeCompileAttentionInputs = [graph.variable(decodeCompileRotary.toGPU(0))]
        }
        let decodeCompileCacheInputs = Self.cacheInputs(
          caches, currentTokenLength: cacheCapacity, configuration: configuration)
        decoder.compile(
          (cachedTokenLength: maxDecodeCachedTokenLength, tokenLength: 1),
          inputs: [decodeCompileToken, decodeTokenMask, decodeInjectedEmbeddings]
            + decodeCompileAttentionInputs + decodeCompileCacheInputs,
          isEager: true)
        graph.joined()
        let queueWatermark = DynamicGraph.queueWatermark
        DynamicGraph.queueWatermark = queueWatermark * 8  // Expanding queue watermark to support longer generation.
        defer {
          DynamicGraph.queueWatermark = queueWatermark
        }
        var nextTokenGPU = graph.variable(Tensor<Int32>([nextToken], .CPU, .C(1)).toGPU(0))
        let cachedTokenLength = promptTokenIds.count + generated.count - 1
        let rotaryEmbeddingForMaxTokens = graph.variable(
          Qwen3_5RotaryEmbedding(
            sequenceLength: maxTokens, cachedTokenLength: cachedTokenLength + decodePositionOffset,
            configuration: configuration, of: FloatType.self
          ).toGPU(0))
        for i in 1..<maxTokens {
          let cachedTokenLength = promptTokenIds.count + generated.count - 1
          var oneAttentionInputs = [DynamicGraph.AnyTensor]()
          if hasFullAttentionLayer {
            let oneRotary = rotaryEmbeddingForMaxTokens.reshaped(
              format: .NHWC, shape: [1, 1, 1, 256], offset: [0, 0, i - 1, 0],
              strides: [256, 256, 256, 1])
            oneAttentionInputs = [oneRotary.copied()]
          }
          let cacheInputs = Self.cacheInputs(
            caches, currentTokenLength: cachedTokenLength + 1, configuration: configuration)
          let oldDecodedTokenCPU = nextTokenGPU.toCPU()
          do {
            var decodeOutputs = decoder(
              (cachedTokenLength: cachedTokenLength, tokenLength: 1), inputs: nextTokenGPU,
              [decodeTokenMask, decodeInjectedEmbeddings] + oneAttentionInputs + cacheInputs)
            Self.updateLinearCaches(
              &caches, from: decodeOutputs, outputOffset: 1, configuration: configuration)
            let logits = decodeOutputs[0].as(of: FloatType.self)
            nextTokenGPU = Functional.argmax(logits, axis: 1).reshaped(.C(1))
            decodeOutputs.removeAll(keepingCapacity: false)
          }
          if i > 1 {
            let decodedToken = oldDecodedTokenCPU[0]
            generated.append(decodedToken)
            if eosTokenIds.contains(decodedToken) || !partialHandler(generated) {
              break
            }
          }
        }
        graph.joined()
        return generated
      }
    }
  }

  private static func makeCaches(
    graph: DynamicGraph, capacity: Int, configuration: Qwen3_5ModelConfiguration
  ) -> [DynamicGraph.AnyTensor] {
    var caches = [DynamicGraph.AnyTensor]()
    for layerIndex in 0..<configuration.layers {
      if configuration.isLinearAttentionLayer(layerIndex) {
        let convState = graph.variable(
          Tensor<FloatType>(
            Array(
              repeating: FloatType.zero,
              count: (configuration.linearConvKernel - 1) * configuration.linearConvDim),
            .CPU,
            .NHWC(1, configuration.linearConvKernel - 1, 1, configuration.linearConvDim)
          ).toGPU(0))
        let recurrentState = graph.variable(
          Tensor<Float>(
            Array(
              repeating: Float.zero,
              count: configuration.linearNumValueHeads * configuration.linearKeyHeadDim
                * configuration.linearValueHeadDim),
            .CPU,
            .NHWC(
              1, configuration.linearNumValueHeads, configuration.linearValueHeadDim,
              configuration.linearKeyHeadDim)
          ).toGPU(0))
        caches.append(contentsOf: [convState, recurrentState])
      } else {
        let k = graph.variable(
          Tensor<FloatType>(
            Array(
              repeating: FloatType.zero,
              count: capacity * configuration.keyValueHeads * configuration.attentionHeadDim),
            .CPU,
            .NHWC(1, capacity, configuration.keyValueHeads, configuration.attentionHeadDim)
          ).toGPU(0))
        let v = graph.variable(
          Tensor<FloatType>(
            Array(
              repeating: FloatType.zero,
              count: capacity * configuration.keyValueHeads * configuration.attentionHeadDim),
            .CPU,
            .NHWC(1, capacity, configuration.keyValueHeads, configuration.attentionHeadDim)
          ).toGPU(0))
        caches.append(contentsOf: [k, v])
      }
    }
    return caches
  }

  private static func cacheInputs(
    _ caches: [DynamicGraph.AnyTensor], currentTokenLength: Int,
    configuration: Qwen3_5ModelConfiguration
  ) -> [DynamicGraph.AnyTensor] {
    var result = [DynamicGraph.AnyTensor]()
    var cursor = 0
    for layerIndex in 0..<configuration.layers {
      if configuration.isLinearAttentionLayer(layerIndex) {
        result.append(caches[cursor])
        result.append(caches[cursor + 1])
      } else {
        result.append(
          caches[cursor].as(of: FloatType.self).reshaped(
            .NHWC(
              1, currentTokenLength, configuration.keyValueHeads, configuration.attentionHeadDim)))
        result.append(
          caches[cursor + 1].as(of: FloatType.self).reshaped(
            .NHWC(
              1, currentTokenLength, configuration.keyValueHeads, configuration.attentionHeadDim)))
      }
      cursor += 2
    }
    return result
  }

  private static func updateLinearCaches(
    _ caches: inout [DynamicGraph.AnyTensor], from outputs: [DynamicGraph.AnyTensor],
    outputOffset: Int, configuration: Qwen3_5ModelConfiguration
  ) {
    var cacheCursor = 0
    var outputCursor = outputOffset
    for layerIndex in 0..<configuration.layers {
      if configuration.isLinearAttentionLayer(layerIndex) {
        let convState = outputs[outputCursor].as(of: FloatType.self)
        let recurrentState = outputs[outputCursor + 1].as(of: Float.self)
        var convStateCache = caches[cacheCursor].as(of: FloatType.self)
        var recurrentStateCache = caches[cacheCursor + 1].as(of: Float.self)
        convStateCache[
          0..<1, 0..<(configuration.linearConvKernel - 1), 0..<1, 0..<configuration.linearConvDim
        ] = convState
        recurrentStateCache[
          0..<1, 0..<configuration.linearNumValueHeads, 0..<configuration.linearValueHeadDim,
          0..<configuration.linearKeyHeadDim
        ] = recurrentState
        outputCursor += 2
      }
      cacheCursor += 2
    }
  }

  private static func resetLinearCaches(
    _ caches: inout [DynamicGraph.AnyTensor], configuration: Qwen3_5ModelConfiguration
  ) {
    var cursor = 0
    for layerIndex in 0..<configuration.layers {
      if configuration.isLinearAttentionLayer(layerIndex) {
        caches[cursor].as(of: FloatType.self).full(0)
        caches[cursor + 1].as(of: Float.self).full(0)
      }
      cursor += 2
    }
  }

}
