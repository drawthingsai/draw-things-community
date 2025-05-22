import Foundation
import Logging

#if os(Linux)
  import FoundationNetworking
#endif

/// Utility functions for managing custom model directories with size limits
public final class LocalLoRAManager {

  private let logger = Logger(label: "com.draw-things.local-lora-manager")
  private let r2Client: R2Client
  private let localDirectory: String
  public init(r2Client: R2Client, localDirectory: String) {
    self.r2Client = r2Client
    self.localDirectory = localDirectory
  }

  /// Download a model without performing cleanup
  /// - Parameters:
  ///   - modelName: Name of the model to download
  /// - Returns: A tuple with success status and the downloaded file size
  private func downloadRemoteLoRA(
    _ modelNames: [String], index: Int, results: [String: Bool],
    cancellation: @escaping (@escaping () -> Void) -> Void,
    completion: @escaping ([String: Bool]) -> Void
  ) {
    guard index < modelNames.count else {
      completion(results)
      return
    }
    var results = results
    let modelName = modelNames[index]
    let dirURL = URL(fileURLWithPath: localDirectory)
    let logger = logger
    logger.info("Downloading LoRA \(modelName)")
    let task = r2Client.downloadObject(key: modelName) { result in
      switch result {
      case .success(let tempUrl):
        logger.info("Downloaded LoRA \(modelName) at \(tempUrl)")
        do {
          // Get the file size from the downloaded temp file
          let fileAttributes = try FileManager.default.attributesOfItem(atPath: tempUrl.path)
          guard let _ = fileAttributes[.size] as? Int64 else {
            logger.info("Failed to determine size of downloaded file \(modelName)")
            return
          }
          // only using the prefix hash as the file name, for example
          // "072ef94e15252e963a0bc77702f8db329ef2ce0e2245ed487ee61aeca1cdb69d_d71b5bbc-0a6b-4b50-8c6f-3691b80bc2ee" --> "072ef94e15252e963a0bc77702f8db329ef2ce0e2245ed487ee61aeca1cdb69d"
          let modelName = modelName.components(separatedBy: "_").first ?? modelName
          // Create destination URL
          let destinationUrl = dirURL.appendingPathComponent(modelName)
          // Create parent directories if needed
          try FileManager.default.createDirectory(
            at: destinationUrl.deletingLastPathComponent(),
            withIntermediateDirectories: true
          )
          // Move downloaded file to destination
          try FileManager.default.moveItem(at: tempUrl, to: destinationUrl)
          logger.info(
            "Successfully moved model \(modelName) to custom directory \(self.localDirectory)")
          results[modelName] = true
        } catch {
          logger.info("Failed to save model \(modelName): \(error.localizedDescription)")
          results[modelName] = false
        }
        self.downloadRemoteLoRA(
          modelNames, index: index + 1, results: results, cancellation: cancellation,
          completion: completion)
      case .failure(let error):
        logger.info("Error downloading model \(modelName): \(error.localizedDescription)")
        results[modelName] = false
      }
    }
    if let task = task {
      cancellation({
        task.cancel()
      })
    }
  }

  /// Download multiple models and perform cleanup only once after all downloads
  /// - Parameters:
  ///   - modelNames: Array of model names to download
  /// - Returns: Dictionary mapping model names to download success status
  public func downloadRemoteLoRAs(
    _ modelNames: [String], cancellation: @escaping (@escaping () -> Void) -> Void,
    completion: @escaping ([String: Bool]) -> Void
  ) {
    downloadRemoteLoRA(
      modelNames, index: 0, results: [:], cancellation: cancellation, completion: completion)
  }
}
