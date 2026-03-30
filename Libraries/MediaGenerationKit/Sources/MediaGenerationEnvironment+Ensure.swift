import CryptoKit
import Downloader
import Foundation
import ModelZoo

private final class MediaGenerationVerifiedSHACache {
  static let shared = MediaGenerationVerifiedSHACache()

  private var cache: [String: String] = [:]
  private var cacheURL: URL?
  private let lock = NSLock()

  private init() {}

  func configure(directory: URL) {
    lock.lock()
    defer { lock.unlock() }
    let url = directory.appendingPathComponent(".verified_sha256.json")
    guard cacheURL != url else { return }
    cacheURL = url
    if let data = try? Data(contentsOf: url),
      let decoded = try? JSONDecoder().decode([String: String].self, from: data)
    {
      cache = decoded
    } else {
      cache = [:]
    }
  }

  func cached(for file: String) -> String? {
    lock.lock()
    defer { lock.unlock() }
    return cache[file]
  }

  func set(_ sha: String, for file: String) {
    lock.lock()
    defer { lock.unlock() }
    cache[file] = sha
    persist()
  }

  func remove(file: String) {
    lock.lock()
    defer { lock.unlock() }
    cache.removeValue(forKey: file)
    persist()
  }

  private func persist() {
    guard let cacheURL, let data = try? JSONEncoder().encode(cache) else { return }
    try? data.write(to: cacheURL, options: .atomic)
  }
}

struct MediaGenerationModelDownloadProgress {
  let file: String
  let fileIndex: Int
  let totalFiles: Int
  let bytesWritten: Int64
  let totalBytesExpected: Int64
}

extension MediaGenerationEnvironment.Storage {
  func ensureModelReady(
    file: String,
    includeDependencies: Bool = true,
    verification: ((String, Int, Int) -> Void)? = nil,
    progress: ((MediaGenerationModelDownloadProgress) -> Void)? = nil
  ) throws {
    let modelsDirectory = try modelsDirectoryURL()
    let externalUrls = self.externalUrls
    if !externalUrls.isEmpty {
      ModelZoo.externalUrls = externalUrls
    }

    let allFiles: [String]
    if let specification = ModelZoo.specificationForModel(file) {
      if specification.remoteApiModelConfig != nil {
        throw MediaGenerationKitError.modelNotFoundInCatalog(file)
      }
      allFiles =
        includeDependencies
        ? ModelZoo.filesToDownload(specification).map(\.file)
        : [specification.file]
    } else {
      allFiles = [file]
    }

    guard !allFiles.isEmpty else { return }

    MediaGenerationVerifiedSHACache.shared.configure(directory: modelsDirectory)

    let largeSHAThresholdBytes: Int64 = 300 * 1024 * 1024
    var filesToDownload: [String] = []

    for (index, fileName) in allFiles.enumerated() {
      verification?(fileName, index + 1, allFiles.count)
      if try Self.fileNeedsDownload(
        fileName,
        largeSHAThresholdBytes: largeSHAThresholdBytes
      ) {
        filesToDownload.append(fileName)
      }
    }

    for (index, fileName) in filesToDownload.enumerated() {
      try Self.downloadFile(
        fileName: fileName,
        fileIndex: index,
        totalFiles: filesToDownload.count,
        progress: progress
      )
      if let expectedSHA = ModelZoo.fileSHA256ForModelDownloaded(fileName) {
        MediaGenerationVerifiedSHACache.shared.set(expectedSHA, for: fileName)
      }
    }
  }

  private static func fileNeedsDownload(
    _ fileName: String,
    largeSHAThresholdBytes: Int64
  ) throws -> Bool {
    guard ModelZoo.isModelDownloaded(fileName) else {
      return true
    }

    let expectedSHA = ModelZoo.fileSHA256ForModelDownloaded(fileName)
    let filePath = ModelZoo.filePathForModelDownloaded(fileName)
    let fileURL = URL(fileURLWithPath: filePath)

    if let cachedSHA = MediaGenerationVerifiedSHACache.shared.cached(for: fileName) {
      if let expectedSHA {
        if cachedSHA == expectedSHA {
          return false
        }
      } else {
        return false
      }
    }

    let fileSizeBytes =
      (try? FileManager.default.attributesOfItem(atPath: filePath))?[.size] as? Int64 ?? 0
    if fileSizeBytes > largeSHAThresholdBytes {
      if let expectedSHA {
        MediaGenerationVerifiedSHACache.shared.set(expectedSHA, for: fileName)
      }
      return false
    }

    guard let actualSHA = try? sha256Hex(fileURL: fileURL) else {
      return false
    }

    if let expectedSHA {
      if actualSHA == expectedSHA {
        MediaGenerationVerifiedSHACache.shared.set(actualSHA, for: fileName)
        return false
      }
      try? FileManager.default.removeItem(at: fileURL)
      MediaGenerationVerifiedSHACache.shared.remove(file: fileName)
      return true
    }

    MediaGenerationVerifiedSHACache.shared.set(actualSHA, for: fileName)
    return false
  }

  private static func downloadFile(
    fileName: String,
    fileIndex: Int,
    totalFiles: Int,
    progress: ((MediaGenerationModelDownloadProgress) -> Void)?
  ) throws {
    let localURL = URL(fileURLWithPath: ModelZoo.filePathForModelDownloaded(fileName))
    let remoteURL = URL(string: "https://static.libnnc.org/\(fileName)")!
    let expectedSHA = ModelZoo.fileSHA256ForModelDownloaded(fileName)
    try FileManager.default.createDirectory(
      at: localURL.deletingLastPathComponent(),
      withIntermediateDirectories: true
    )

    let semaphore = DispatchSemaphore(value: 0)
    var downloadError: Error?

    let downloader = ResumableDownloader(
      remoteUrl: remoteURL,
      localUrl: localURL,
      sha256: expectedSHA
    )
    downloader.resume { totalBytesWritten, totalBytesExpectedToWrite, isComplete, error in
      if let error {
        downloadError = error
        semaphore.signal()
        return
      }
      progress?(
        MediaGenerationModelDownloadProgress(
          file: fileName,
          fileIndex: fileIndex + 1,
          totalFiles: totalFiles,
          bytesWritten: totalBytesWritten,
          totalBytesExpected: totalBytesExpectedToWrite
        )
      )
      if isComplete {
        semaphore.signal()
      }
    }
    semaphore.wait()

    if let error = downloadError {
      if let downloaderError = error as? ResumableDownloader.DownloadError,
        case .fileMismatch = downloaderError
      {
        throw MediaGenerationKitError.hashMismatch(fileName)
      }
      let nsError = error as NSError
      if nsError.domain == NSCocoaErrorDomain && nsError.code == NSFileWriteOutOfSpaceError {
        throw MediaGenerationKitError.insufficientStorage
      }
      throw MediaGenerationKitError.downloadFailed(error.localizedDescription)
    }
  }

  private static func sha256Hex(fileURL: URL) throws -> String {
    let handle = try FileHandle(forReadingFrom: fileURL)
    defer { try? handle.close() }

    var hasher = SHA256()
    while true {
      let data = try handle.read(upToCount: 1024 * 1024) ?? Data()
      if data.isEmpty {
        break
      }
      hasher.update(data: data)
    }
    return hasher.finalize().map { String(format: "%02x", $0) }.joined()
  }
}
