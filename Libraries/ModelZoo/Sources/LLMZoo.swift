import Foundation
import LLM

public struct LLMZoo: DownloadZoo {
  public struct Specification: Codable, Hashable {
    public let name: String
    public let file: String
    public let version: LLMVersion

    public init(name: String, file: String, version: LLMVersion) {
      self.name = name
      self.file = file
      self.version = version
    }
  }

  public static let defaultSpecification = Specification(
    name: "Qwen 3.6 27B (4-bit S)", file: "qwen_3.6_27b_i4x.ckpt",
    version: .qwen_3_5_27b)

  public static let builtinSpecifications: [Specification] = [
    defaultSpecification,
    Specification(
      name: "Qwen 3.6 27B (8-bit S)", file: "qwen_3.6_27b_i8x.ckpt",
      version: .qwen_3_5_27b),
    Specification(
      name: "Qwen 3.5 9B (5-bit S)", file: "qwen_3.5_9b_i5x.ckpt",
      version: .qwen_3_5_9b),
  ]

  private static let fileSHA256: [String: String] = [
    "qwen_3.6_27b_i4x.ckpt": "20a28fb30af7d1d5228c314fa76080ec0dae7c9519fd69e5ad54c44462eb7dbc",
    "qwen_3.6_27b_i8x.ckpt": "1cd4b97a358a11dd795326e549e210d88f0aff6d91ebee8adc40eda6f36f7f4d",
    "qwen_3.5_9b_i5x.ckpt": "559d41f5e6721b4edb5a661c42b6f328a243876c2637cad36cf9937fb0d453c1",
  ]

  private static var specificationMapping: [String: Specification] = {
    var mapping = [String: Specification]()
    for specification in builtinSpecifications {
      mapping[specification.file] = specification
    }
    return mapping
  }()

  public static var overrideMapping: [String: Specification] = [:]

  public static var fallbackMapping: [String: Specification] = [:]

  public static func specificationForModel(_ name: String) -> Specification? {
    if let override = overrideMapping[name] {
      return override
    }
    return specificationMapping[name] ?? fallbackMapping[name]
  }

  public static func filePathForModelDownloaded(_ name: String) -> String {
    return ModelZoo.filePathForModelDownloaded(name)
  }

  public static func isModelDownloaded(_ name: String, memorizedBy: Set<String>) -> Bool {
    return ModelZoo.isModelDownloaded(name, memorizedBy: memorizedBy)
  }

  public static func humanReadableNameForModel(_ name: String) -> String {
    guard let specification = specificationForModel(name) else { return name }
    return specification.name
  }

  public static func versionForModel(_ name: String) -> LLMVersion {
    guard let specification = specificationForModel(name) else {
      return defaultSpecification.version
    }
    return specification.version
  }

  public static func fileSHA256ForModelDownloaded(_ name: String) -> String? {
    return fileSHA256[name]
  }

  public static func availableFiles(excluding file: String?) -> Set<String> {
    var files = Set<String>()
    for specification in builtinSpecifications {
      guard specification.file != file, LLMZoo.isModelDownloaded(specification.file)
      else { continue }
      files.insert(specification.file)
    }
    return files
  }
}
