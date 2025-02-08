import DataModels
import Diffusion
import Fickling
import Foundation
import ModelZoo
import NNC
import Upscaler
import ZIPFoundation

public final class UpscalerImporter {
  final class DataArchive: TensorDataArchive {
    let data: Data
    let bufferStart: Int
    init(data: Data, bufferStart: Int) {
      self.data = data
      self.bufferStart = bufferStart
    }
  }
  private let filePath: String
  public init(filePath: String) {
    self.filePath = filePath
  }
  public func `import`(
    progress: @escaping (Float) -> Void
  ) throws -> (file: String, numberOfBlocks: Int) {
    let archive: TensorArchive
    var stateDict: [String: TensorDescriptor]
    if let safeTensors = SafeTensors(url: URL(fileURLWithPath: filePath)) {
      archive = safeTensors
      let states = safeTensors.states
      stateDict = states
    } else if let zipArchive = Archive(url: URL(fileURLWithPath: filePath), accessMode: .read) {
      archive = zipArchive
      let rootObject = try Interpreter.unpickle(zip: zipArchive)
      let originalStateDict =
        rootObject["state_dict"] as? Interpreter.Dictionary ?? rootObject["params_ema"]
        as? Interpreter.Dictionary ?? rootObject["params"] as? Interpreter.Dictionary ?? rootObject
      stateDict = [String: TensorDescriptor]()
      originalStateDict.forEach { key, value in
        guard let value = value as? TensorDescriptor else { return }
        stateDict[key] = value
      }
    } else if let data = try? Data(
      contentsOf: URL(fileURLWithPath: filePath), options: .mappedIfSafe)
    {
      archive = DataArchive(data: data, bufferStart: 0)
      let rootObject = try Interpreter.unpickle(data: data, fileReadDirectly: true)
      let originalStateDict =
        rootObject["state_dict"] as? Interpreter.Dictionary ?? rootObject["params_ema"]
        as? Interpreter.Dictionary ?? rootObject["params"] as? Interpreter.Dictionary ?? rootObject
      stateDict = [String: TensorDescriptor]()
      originalStateDict.forEach { key, value in
        guard let value = value as? TensorDescriptor else { return }
        stateDict[key] = value
      }
    } else {
      throw UnpickleError.dataNotFound
    }
    return ("", 0)
  }
}
