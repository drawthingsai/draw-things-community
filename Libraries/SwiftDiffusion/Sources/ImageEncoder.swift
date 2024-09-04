import NNC

public struct ImageEncoder<FloatType: TensorNumeric & BinaryFloatingPoint> {
  let filePath: String
  let version: ImageEncoderVersion
  public init(filePath: String, version: ImageEncoderVersion) {
    self.filePath = filePath
    self.version = version
  }
}

extension ImageEncoder {
  public func encode(_ x: [DynamicGraph.Tensor<FloatType>]) -> [DynamicGraph.Tensor<FloatType>] {
    precondition(x.count > 0)
    let graph = x[0].graph
    return graph.withNoGrad {
      let vit: Model
      switch version {
      case .clipL14_336:
        vit = CLIPVisionTransformer(
          FloatType.self, grid: 24, width: 1024, layers: 23, heads: 16,
          batchSize: 1, noFinalLayerNorm: true)
      case .openClipH14:
        vit = VisionTransformer(
          FloatType.self, grid: 16, width: 1280, layers: 31, heads: 16, batchSize: 1,
          noFinalLayerNorm: true)
      }
      vit.compile(inputs: x[0])
      graph.openStore(
        filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
      ) {
        $0.read("vision_model", model: vit)
      }
      return x.map {
        let output = vit(inputs: $0)[0].as(of: FloatType.self)
        switch version {
        case .clipL14_336:
          return output.reshaped(.HWC(1, 577, 1024))
        case .openClipH14:
          return output.reshaped(.HWC(1, 257, 1280))
        }
      }
    }
  }
}
