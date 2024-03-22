import NNC

public struct ImageEncoder<FloatType: TensorNumeric & BinaryFloatingPoint> {
  let filePath: String
  init(filePath: String) {
    self.filePath = filePath
  }
}

extension ImageEncoder {
  public func encode(_ x: [DynamicGraph.Tensor<FloatType>]) -> [DynamicGraph.Tensor<FloatType>] {
    precondition(x.count > 0)
    let graph = x[0].graph
    return graph.withNoGrad {
      let vit = VisionTransformer(
        FloatType.self,
        grid: 16, width: 1280, outputDim: 1024, layers: 31, heads: 16, batchSize: 1,
        noFinalLayerNorm: true)
      vit.compile(inputs: x[0])
      graph.openStore(
        filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
      ) {
        $0.read("vision_model", model: vit)
      }
      return x.map { vit(inputs: $0)[0].as(of: FloatType.self).reshaped(.HWC(1, 257, 1280)) }
    }
  }
}
