import Foundation
import Logging

#if os(Linux)
  import FoundationNetworking
#endif

/// Utility functions for managing custom model directories with size limits
public final class CustomModelManager {

  /// Maximum size allowed for the custom model directory (500GB in bytes)
  public let maxDirectorySize: Int64 = 500 * 1024 * 1024 * 1024
  private let logger = Logger(label: "com.draw-things.image-generation-service")
  private let r2Client: R2Client
  private let customDirectory: String
  public init(r2Client: R2Client, customDirectory: String) {
    self.r2Client = r2Client
    self.customDirectory = customDirectory
  }

  /// Get a list of model files in the directory
  /// - Parameter directoryPath: Path to the directory
  /// - Returns: Array of file URLs and their sizes
  func getModelFiles(in directoryPath: String) -> [(url: URL, size: Int64)] {
    let fileManager = FileManager.default
    guard
      let urls = try? fileManager.contentsOfDirectory(
        at: URL(fileURLWithPath: directoryPath),
        includingPropertiesForKeys: [.fileSizeKey],
        options: [.skipsHiddenFiles, .skipsPackageDescendants])
    else {
      return []
    }

    return urls.compactMap { url in
      guard let attributes = try? url.resourceValues(forKeys: [.fileSizeKey]),
        let fileSize = attributes.fileSize
      else {
        return nil
      }
      return (url, Int64(fileSize))
    }
  }

  /// Remove random 10 models from the custom directory to keep it under the size limit
  /// - Parameters:
  ///   - directoryPath: Path to the custom model directory
  ///   - excludedFiles: Array of filenames to exclude from deletion
  /// - Returns: True if cleanup was successful
  public func cleanupRandomFilesCustomModelsDirectory(
    _ directoryPath: String, excludedFiles: [String]
  ) -> Bool {

    // Get all model files that can be deleted
    let modelFiles = getModelFiles(in: directoryPath)
      .filter { !excludedFiles.contains($0.url.lastPathComponent) }

    guard !modelFiles.isEmpty else {
      return false  // No files to delete
    }

    // Shuffle the files for random selection
    let shuffledFiles = modelFiles.shuffled()
    var filesToDelete: [URL] = []

    // Select files to delete until we have enough space
    for file in shuffledFiles {
      filesToDelete.append(file.url)

      if filesToDelete.count >= 10 {
        break
      }
    }

    // Delete the selected files
    let fileManager = FileManager.default
    var success = true

    for fileURL in filesToDelete {
      do {
        try fileManager.removeItem(at: fileURL)
        self.logger.info("Removed model file: \(fileURL.lastPathComponent) to free space")
      } catch {
        self.logger.info(
          "Failed to remove file \(fileURL.lastPathComponent): \(error.localizedDescription)")
        success = false
      }
    }

    return success
  }

  /// Download a model without performing cleanup
  /// - Parameters:
  ///   - modelName: Name of the model to download
  ///   - customDirectory: Directory to save the model to
  /// - Returns: A tuple with success status and the downloaded file size
  public func downloadCustomModel(
    _ modelName: String
  ) -> (success: Bool, size: Int64?) {
    let dirURL = URL(fileURLWithPath: customDirectory)
    // Use a single download task with semaphore to make it synchronous
    let downloadSemaphore = DispatchSemaphore(value: 0)
    var downloadResult: (success: Bool, size: Int64?) = (false, nil)
    self.logger.info("downloading custom model \(modelName)")

    r2Client.downloadObject(key: modelName) { result in
      defer { downloadSemaphore.signal() }

      switch result {
      case .success(let tempUrl):
        self.logger.info("Download model \(modelName) at \(tempUrl)")

        do {
          // Get the file size from the downloaded temp file
          let fileAttributes = try FileManager.default.attributesOfItem(atPath: tempUrl.path)
          guard let fileSize = fileAttributes[.size] as? Int64 else {
            self.logger.info("Failed to determine size of downloaded file \(modelName)")
            return
          }

          // only using the prefix hash as the file name, for example
          // "072ef94e15252e963a0bc77702f8db329ef2ce0e2245ed487ee61aeca1cdb69d-d71b5bbc-0a6b-4b50-8c6f-3691b80bc2ee" --> "072ef94e15252e963a0bc77702f8db329ef2ce0e2245ed487ee61aeca1cdb69d"

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
          self.logger.info(
            "Successfully downloaded model \(modelName) to custom directory\(self.customDirectory)")
          downloadResult = (true, fileSize)
        } catch {
          self.logger.info("Failed to save model \(modelName): \(error.localizedDescription)")
        }

      case .failure(let error):
        self.logger.info("Error downloading model \(modelName): \(error.localizedDescription)")

      }
    }

    downloadSemaphore.wait()
    return downloadResult
  }

  /// Download multiple models and perform cleanup only once after all downloads
  /// - Parameters:
  ///   - modelNames: Array of model names to download
  /// - Returns: Dictionary mapping model names to download success status
  public func batchDownloadModels(
    _ modelNames: [String]
  ) -> [String: Bool] {
    var results: [String: Bool] = [:]

    // First, download all models
    for modelName in modelNames {
      let result = downloadCustomModel(
        modelName
      )
      results[modelName] = result.success
    }

    return results
  }
}
