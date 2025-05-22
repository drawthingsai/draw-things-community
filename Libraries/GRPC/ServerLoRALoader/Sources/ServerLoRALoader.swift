import DataModels
import Foundation
import Logging
import ModelZoo
import ServerConfigurationRewriter

public struct ServerLoRALoader: ServerConfigurationRewriter {

  private let localLoRAManager: LocalLoRAManager
  private let logger = Logger(label: "com.draw-things.image-generation-service")

  public init(localLoRAManager: LocalLoRAManager) {
    self.localLoRAManager = localLoRAManager
  }

  public func newConfiguration(
    configuration: GenerationConfiguration,
    completion: @escaping (Result<GenerationConfiguration, Error>) -> Void
  ) {

    let configLoras: [String] = configuration.loras.compactMap { $0.file }
    self.logger.info("loras: \(configLoras)")
    let loRAsNeedToLoad = configLoras.filter { file in
      let sha256 = sha256HashName(loraName: file)
      if ModelZoo.isModelDownloaded(sha256) {
        return false
      }
      return true
    }

    guard loRAsNeedToLoad.count > 0 else {
      self.logger.info("no loRAs NeedToLoad")
      completion(.success(configurationLorasWithoutSuffix(configuration)))
      return
    }

    self.logger.info("loRAsNeedToLoad: \(loRAsNeedToLoad)")
    localLoRAManager.downloadRemoteLoRAs(loRAsNeedToLoad) { results in
      // Check if any model failed to download
      if let failedModel = results.first(where: { !$0.value })?.key {
        self.logger.info("fail to load custom model: \(failedModel)")
        completion(.failure(ServerConfigurationRewriteError.canNotLoadModel(failedModel)))
        return
      }

      self.logger.info("load custom model: \(results)")
      completion(.success(configurationLorasWithoutSuffix(configuration)))
    }
  }

  func configurationLorasWithoutSuffix(_ configuration: GenerationConfiguration)
    -> GenerationConfiguration
  {
    var configurationBuilder = GenerationConfigurationBuilder(from: configuration)
    configurationBuilder.loras = configuration.loras.compactMap { lora in
      guard let file = lora.file else {
        return nil
      }
      let sha256 = sha256HashName(loraName: file)
      return DataModels.LoRA(file: sha256, weight: lora.weight)
    }
    return configurationBuilder.build()
  }

  func sha256HashName(loraName: String) -> String {
    guard isSHA256LoRA(loraName: loraName) else {
      return loraName
    }
    if let sha256LoRA = loraName.split(separator: "_", maxSplits: 1).first {
      return String(sha256LoRA)
    }
    return loraName
  }

  func isSHA256LoRA(loraName: String) -> Bool {
    // Check if the string is empty
    guard !loraName.isEmpty else {
      return false
    }

    guard !loraName.hasSuffix(".ckpt") else {
      return false
    }

    // Split the name by the first dash or underscore
    let components: [String]
    if loraName.contains("_") {
      components = loraName.split(separator: "_", maxSplits: 1).map { String($0) }
    } else {
      // If there's no separator, check if the whole string is a valid hash
      components = [loraName]
    }

    // Extract the first component (potential hash)
    guard let potentialHash = components.first else {
      return false
    }

    // Check if the potential hash is 64 characters long (SHA256 length)
    guard potentialHash.count == 64, potentialHash.allSatisfy({ $0.isHexDigit }) else {
      return false
    }

    return true
  }

}
