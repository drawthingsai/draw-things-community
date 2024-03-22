import NNC

public struct MoondreamEncode<T: TensorNumeric & BinaryFloatingPoint> {
  public enum Version {
    case moondream1
    case moondream2
  }
  let filePaths: [String]
  let usesFlashAttention: Bool
  let version: Version
  public init(filePaths: [String], usesFlashAttention: Bool, version: Version) {
    self.filePaths = filePaths
    self.usesFlashAttention = usesFlashAttention
    self.version = version
  }
}

extension MoondreamEncode {
  public func encode(
    _ x: DynamicGraph.Tensor<T>, vit existingVit: Model? = nil,
    visionProj existingVisionProj: Model? = nil
  ) -> (DynamicGraph.Tensor<T>, Model, Model) {
    // We only support 378x378.
    precondition(x.shape[1] == 378)
    precondition(x.shape[2] == 378)
    let graph = x.graph
    return graph.withNoGrad {
      let vit =
        existingVit
        ?? SigLIPVisionTransformer(
          T.self, gridX: 27, gridY: 27, width: 1152, layers: 27, heads: 16, MLP: 4304, batchSize: 1,
          usesFlashAttention: usesFlashAttention)
      if existingVit == nil {
        vit.compile(inputs: x)
        graph.openStore(
          filePaths[0], flags: .readOnly,
          externalStore: TensorData.externalStore(filePath: filePaths[0])
        ) {
          $0.read("vit", model: vit, codec: [.jit, .q8p, .ezm7, .externalData])
        }
      }
      var out = vit(inputs: x)[0].as(of: T.self)
      let visionProj =
        existingVisionProj ?? MoondreamVisionProjection(layers: version == .moondream2 ? 1 : 2)
      if existingVisionProj == nil {
        visionProj.compile(inputs: out)
        graph.openStore(
          filePaths[1], flags: .readOnly,
          externalStore: TensorData.externalStore(filePath: filePaths[1])
        ) {
          $0.read("vision_proj", model: visionProj, codec: [.q8p, .ezm7, .externalData])
        }
      }
      out = visionProj(inputs: out)[0].as(of: T.self)
      return (out, vit, visionProj)
    }
  }
}
