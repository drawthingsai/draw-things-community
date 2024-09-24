import NNC

public struct ControlModelLoader<FloatType: BinaryFloatingPoint & TensorNumeric> {
  private var stores: [DynamicGraph.Store]
  private init(stores: [DynamicGraph.Store]) {
    self.stores = stores
  }
  public func loadMergedWeight(name: String) -> DynamicGraph.Store.ModelReaderResult? {
    guard stores.count > 0 else { return nil }
    guard let last = name.components(separatedBy: "[").last, last.count > 2 else { return nil }
    // Remove t-.
    let name = last.suffix(from: last.index(last.startIndex, offsetBy: 2))
    let parts = name.components(separatedBy: ".")
    guard parts.count >= 2, let index = Int(parts[0]), index >= 0 && index < stores.count else {
      return nil
    }
    guard
      let tensor = stores[index].read(
        "__pulid__[t-" + parts[1], codec: [.ezm7, .externalData, .q6p, .q8p])
    else { return nil }
    if parts[1].contains("norm2") {
      return .final(Tensor<Float>(from: tensor))
    } else {
      return .final(tensor)
    }
  }
}

extension ControlModelLoader {
  private static func _openStore(
    _ graph: DynamicGraph, index: Int, injectControlModels: [ControlModel<FloatType>],
    stores: [DynamicGraph.Store], version: ModelVersion, handler: (ControlModelLoader) -> Void
  ) {
    guard index < injectControlModels.count else {
      handler(ControlModelLoader(stores: stores))
      return
    }
    for i in index..<injectControlModels.count {
      // Only open for PuLID of FLUX.1.
      if injectControlModels[i].type == .pulid && version == .flux1 {
        graph.openStore(
          injectControlModels[i].filePaths[0], flags: .readOnly,
          externalStore: TensorData.externalStore(filePath: injectControlModels[i].filePaths[0])
        ) { store in
          _openStore(
            graph, index: i + 1, injectControlModels: injectControlModels, stores: stores + [store],
            version: version, handler: handler)
        }
        break
      }
    }
  }
  public static func openStore(
    _ graph: DynamicGraph, injectControlModels: [ControlModel<FloatType>], version: ModelVersion,
    handler: (ControlModelLoader) -> Void
  ) {
    _openStore(
      graph, index: 0, injectControlModels: injectControlModels, stores: [], version: version,
      handler: handler)
  }
}
