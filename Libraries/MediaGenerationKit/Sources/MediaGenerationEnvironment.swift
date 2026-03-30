import DataModels
import Foundation
import ModelZoo

public struct MediaGenerationEnvironment: Sendable {
  public enum EnsureState: Sendable, Equatable {
    case resolving
    case verifying(file: String, fileIndex: Int, totalFiles: Int)
    case downloading(
      file: String,
      fileIndex: Int,
      totalFiles: Int,
      bytesWritten: Int64,
      totalBytesExpected: Int64
    )
  }

  internal final class Storage: @unchecked Sendable {
    private let lock = NSLock()
    private var currentExternalURLs: [URL]
    private var cachedLocalResources: MediaGenerationLocalResources?

    init(externalUrls: [URL]) {
      self.currentExternalURLs = externalUrls
      if !externalUrls.isEmpty {
        ModelZoo.externalUrls = externalUrls
      }
    }

    var externalUrls: [URL] {
      get {
        lock.lock()
        defer { lock.unlock() }
        return currentExternalURLs
      }
      set {
        lock.lock()
        currentExternalURLs = newValue
        lock.unlock()
        if !newValue.isEmpty {
          ModelZoo.externalUrls = newValue
        }
      }
    }

    func localResources() -> MediaGenerationLocalResources {
      lock.lock()
      if let cachedLocalResources {
        lock.unlock()
        DeviceCapability.cacheUri = URL(fileURLWithPath: cachedLocalResources.tempDir)
        return cachedLocalResources
      }
      lock.unlock()

      let localResources = MediaGenerationExecutionUtilities.createLocalResources()

      lock.lock()
      self.cachedLocalResources = localResources
      lock.unlock()

      DeviceCapability.cacheUri = URL(fileURLWithPath: localResources.tempDir)
      return localResources
    }

    func ensure(
      _ model: String,
      offline: Bool,
      stateHandler: (@Sendable (EnsureState) -> Void)?
    ) async throws -> ModelResolver.Model {
      guard let resolvedModel = try ModelResolver.resolve(model, offline: offline) else {
        throw MediaGenerationKitError.unresolvedModelReference(
          query: model,
          suggestions: ModelResolver.suggestions(model, limit: 5, offline: offline).map(\.file)
        )
      }

      stateHandler?(.resolving)
      return try await withCheckedThrowingContinuation { continuation in
        DispatchQueue.global(qos: .userInitiated).async {
          do {
            try self.ensureModelReady(
              file: resolvedModel.file,
              includeDependencies: true,
              verification: { file, fileIndex, totalFiles in
                stateHandler?(
                  .verifying(
                    file: file,
                    fileIndex: fileIndex,
                    totalFiles: totalFiles
                  )
                )
              },
              progress: { progress in
                stateHandler?(
                  .downloading(
                    file: progress.file,
                    fileIndex: progress.fileIndex,
                    totalFiles: progress.totalFiles,
                    bytesWritten: progress.bytesWritten,
                    totalBytesExpected: progress.totalBytesExpected
                  )
                )
              }
            )
            continuation.resume(returning: resolvedModel)
          } catch {
            continuation.resume(throwing: error)
          }
        }
      }
    }

    func modelsDirectoryURL() throws -> URL {
      let externalUrls = self.externalUrls
      guard let primaryURL = externalUrls.first else {
        throw MediaGenerationKitError.invalidModelsDirectory
      }
      return try MediaGenerationDefaults.ensureDirectory(primaryURL)
    }
  }

  public static var `default` = MediaGenerationEnvironment(
    storage: Storage(externalUrls: MediaGenerationDefaults.defaultExternalURLs())
  )

  internal let storage: Storage

  public var externalUrls: [URL] {
    get {
      storage.externalUrls
    }
    nonmutating set {
      storage.externalUrls = newValue
    }
  }

  public func ensure(
    _ model: String,
    offline: Bool = false,
    stateHandler: (@Sendable (EnsureState) -> Void)? = nil
  ) async throws -> MediaGenerationResolvedModel {
    try await storage.ensure(model, offline: offline, stateHandler: stateHandler)
  }

  internal init(storage: Storage) {
    self.storage = storage
  }

  internal static func local(_ directory: String?) throws -> MediaGenerationEnvironment {
    if let directory, !directory.isEmpty {
      let url = try MediaGenerationDefaults.ensureDirectory(
        URL(fileURLWithPath: directory, isDirectory: true)
      )
      return MediaGenerationEnvironment(storage: Storage(externalUrls: [url]))
    }
    return .default
  }
}

internal enum MediaGenerationDefaults {
  private static let modelsDirectoryInfoKeys = [
    "MediaGenerationKitModelsDirectory",
    "DrawThingsModelsDirectory",
  ]
  private static let apiKeyInfoKeys = [
    "MediaGenerationKitAPIKey",
    "DrawThingsAPIKey",
  ]

  static func defaultExternalURLs() -> [URL] {
    if let envPath = ProcessInfo.processInfo.environment["DRAWTHINGS_MODELS_DIR"], !envPath.isEmpty
    {
      return [URL(fileURLWithPath: envPath, isDirectory: true)]
    }
    if let bundlePath = infoPlistValue(for: modelsDirectoryInfoKeys) {
      return [URL(fileURLWithPath: bundlePath, isDirectory: true)]
    }
    if let documentsURL = FileManager.default.urls(
      for: .documentDirectory, in: .userDomainMask
    ).first {
      return [documentsURL.appendingPathComponent("Models", isDirectory: true)]
    }
    return []
  }

  static func resolveAPIKey(explicitAPIKey: String?) throws -> String {
    if let explicitAPIKey, !explicitAPIKey.isEmpty {
      return explicitAPIKey
    }
    if let envValue = ProcessInfo.processInfo.environment["DRAWTHINGS_API_KEY"], !envValue.isEmpty {
      return envValue
    }
    if let bundleValue = infoPlistValue(for: apiKeyInfoKeys) {
      return bundleValue
    }
    throw MediaGenerationKitError.notConfigured
  }

  static func ensureDirectory(_ url: URL) throws -> URL {
    let normalized = url.standardizedFileURL
    var isDirectory: ObjCBool = false
    let exists = FileManager.default.fileExists(atPath: normalized.path, isDirectory: &isDirectory)
    if exists && !isDirectory.boolValue {
      throw MediaGenerationKitError.invalidModelsDirectory
    }
    if !exists {
      try FileManager.default.createDirectory(
        at: normalized,
        withIntermediateDirectories: true
      )
    }
    return normalized
  }

  private static func infoPlistValue(for keys: [String]) -> String? {
    guard let dictionary = Bundle.main.infoDictionary else {
      return nil
    }
    for key in keys {
      if let value = dictionary[key] as? String, !value.isEmpty {
        return value
      }
    }
    return nil
  }
}
