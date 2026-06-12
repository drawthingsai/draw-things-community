import Foundation

public struct LLMZoo: DownloadZoo {
  public struct Specification: Codable, Hashable {
    public let name: String
    public let file: String

    public init(name: String, file: String) {
      self.name = name
      self.file = file
    }
  }

  public static let defaultSpecification = Specification(
    name: "Qwen 3.6 27B (4-bit S)", file: "qwen_3.6_27b_i4x.ckpt")

  public static let builtinSpecifications: [Specification] = [
    defaultSpecification,
    Specification(name: "Qwen 3.6 27B (2-bit S)", file: "qwen_3.6_27b_i2x.ckpt"),
    Specification(name: "Qwen 3.6 27B (8-bit S)", file: "qwen_3.6_27b_i8x.ckpt"),
  ]

  private static let fileSHA256: [String: String] = [:]

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
