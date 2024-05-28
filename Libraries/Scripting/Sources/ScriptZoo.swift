import Foundation
import Utils

public struct ScriptZoo {

  public enum ScriptType: String, Codable, Equatable, Hashable {
    case user
    case sample
    case community
  }

  public enum ThumbnailImage: Codable, Equatable, Hashable {
    case comparableImage(Image, Image)
    case plainImage(Image)
  }

  public struct Image: Codable, Equatable, Hashable {
    public var url: String
    public var tags: [String]

    public init(
      url: String, tags: [String]
    ) {
      self.url = url
      self.tags = tags
    }
  }

  public struct Script: Codable, Equatable, Hashable {
    public var name: String
    public var file: String
    public var filePath: String?
    public var isSampleDuplicate: Bool?
    public var type: ScriptType?
    public var description: String?
    public var author: String?
    public var tags: [String]?
    public var images: [ThumbnailImage]?
    public init(
      name: String, file: String, filePath: String?, isSampleDuplicate: Bool? = nil,
      type: ScriptType? = nil, description: String? = nil, author: String? = nil,
      tags: [String]? = nil, images: [ThumbnailImage]? = nil
    ) {
      self.name = name
      self.file = file
      self.filePath = filePath
      self.isSampleDuplicate = isSampleDuplicate
      self.type = type
      self.description = description
      self.author = author
      self.tags = tags
      self.images = images
    }
  }

  public static let scriptsUrl: URL = {
    let urls = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
    return urls.first!.appendingPathComponent("Scripts")
  }()

  public static var availableScripts: [Script] {
    var userScriptNames = [String: Int]()
    var scripts =
      ((try? FileManager.default.contentsOfDirectory(
        at: scriptsUrl,
        includingPropertiesForKeys: [],
        options: .skipsHiddenFiles)) ?? [])
      .enumerated().map {
        let filePath = $1.path
        let file = $1.lastPathComponent
        let script = Script(
          name: file, file: file, filePath: filePath, type: .user)
        userScriptNames[file] = $0
        return script
      }
    // Sort file list case-insensitive for user display.
    scripts = scripts.sorted {
      $0.name.localizedStandardCompare($1.name) == .orderedAscending
    }
    for path in Bundle.main.paths(forResourcesOfType: ".sample.js", inDirectory: nil) {
      if let filename = path.split(separator: "/").last {
        let name = filename.replacingOccurrences(of: ".sample", with: "")
        guard !userScriptNames.keys.contains(name) else {
          scripts[userScriptNames[name]!].isSampleDuplicate = true
          continue
        }
        scripts.append(
          Script(name: String(name), file: String(filename), filePath: path, type: .sample)
        )
      }
    }
    return scripts
  }

  private static var jsHeaderDoc: String = {
    return
      (try? String(
        contentsOfFile: Bundle.main.path(forResource: "js-doc-header", ofType: ".js")!,
        encoding: .utf8)) ?? ""
  }()

  public static func removeScript(name: String) {
    try? FileManager.default.removeItem(at: scriptsUrl.appendingPathComponent(name))
  }

  public static func duplicateScript(_ script: Script) {
    var counter = 1
    var filename = script.name
    let insertionIndex = filename.index(filename.endIndex, offsetBy: -3)
    filename.insert(contentsOf: "-\(counter)", at: insertionIndex)
    while FileManager.default.fileExists(atPath: scriptsUrl.appendingPathComponent(filename).path) {
      counter += 1
      filename = script.name
      let insertionIndex = filename.index(filename.endIndex, offsetBy: -3)
      filename.insert(contentsOf: "-\(counter)", at: insertionIndex)
    }
    if let filePath = script.filePath {
      try? FileManager.default.copyItem(
        at: URL(fileURLWithPath: filePath), to: scriptsUrl.appendingPathComponent(filename))
    }
  }

  public static func save(_ content: String, to file: String) {
    try? FileManager.default.createDirectory(at: scriptsUrl, withIntermediateDirectories: true)
    // TODO: is it best to crash here if writing fails?
    var presetContent = content
    if presetContent == "" {
      presetContent = jsHeaderDoc
    }
    try? presetContent.write(
      to: scriptsUrl.appendingPathComponent(file), atomically: true, encoding: .utf8)
  }

  public static func contentOf(_ path: String) -> String? {
    guard let data = FileManager.default.contents(atPath: path) else { return nil }
    return String(data: data, encoding: .utf8)
  }
}

extension ScriptZoo.Script {
  public var isImageToImage: Bool {
    tags?.first(where: { $0 == "image-to-image" }) != nil
  }
}
