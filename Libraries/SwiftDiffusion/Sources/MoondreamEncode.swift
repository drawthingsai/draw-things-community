import NNC

public struct MoondreamEncode<T: TensorNumeric & BinaryFloatingPoint> {
  public enum Version {
    case moondream1
    case moondream2_240306
    case moondream2_240520
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
      let vit: Model
      switch version {
      case .moondream1, .moondream2_240306:
        vit =
          existingVit
          ?? SigLIPVisionTransformer(
            T.self, gridX: 27, gridY: 27, width: 1152, layers: 27, heads: 16, MLP: 4304,
            batchSize: 1,
            usesFlashAttention: usesFlashAttention, approximate: .none)
      case .moondream2_240520:
        vit =
          existingVit
          ?? SigLIPVisionTransformer(
            T.self, gridX: 27, gridY: 27, width: 1152, layers: 27, heads: 16, MLP: 4304,
            batchSize: 1,
            usesFlashAttention: usesFlashAttention, approximate: .tanh)
      }
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
      let visionProj: Model
      switch version {
      case .moondream1:
        visionProj =
          existingVisionProj
          ?? MoondreamVisionProjection(layers: 2, approximate: .none)
      case .moondream2_240306:
        visionProj =
          existingVisionProj
          ?? MoondreamVisionProjection(layers: 1, approximate: .none)
      case .moondream2_240520:
        visionProj =
          existingVisionProj
          ?? MoondreamVisionProjection(layers: 1, approximate: .tanh)
        out = Functional.concat(axis: 1, out, out)
      }
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
