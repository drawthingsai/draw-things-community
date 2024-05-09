import Foundation
import Utils

public struct ScriptZoo {

  public enum ScriptType: String, Codable, Equatable, Hashable {
    case user
    case sample
    case community
  }

  public struct Script: Codable, Equatable, Hashable {
    public var name: String
    public var file: String
    public var filePath: String?
    public var isSampleDuplicate: Bool?
    public var type: ScriptType?
    public var description: String?
    public var author: String?
    public init(
      name: String, file: String, filePath: String?, isSampleDuplicate: Bool? = nil,
      type: ScriptType? = nil, description: String? = nil, author: String? = nil
    ) {
      self.name = name
      self.file = file
      self.filePath = filePath
      self.isSampleDuplicate = isSampleDuplicate
      self.type = type
      self.description = description
      self.author = author
    }
  }

  public static let scriptsUrl: URL = {
    let urls = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
    return urls.first!.appendingPathComponent("Scripts")
  }()
  public static let localCommunityScriptsURL: URL = {
    let urls = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
    return urls.first!.appendingPathComponent("Scripts").appendingPathComponent("Community")
  }()
  public static var availableCommunityScripts: [Script] {
    ((try? FileManager.default.contentsOfDirectory(
      at: localCommunityScriptsURL,
      includingPropertiesForKeys: [],
      options: .skipsHiddenFiles)) ?? [])
      .enumerated().map {
        let filePath = $1.path
        let file = $1.lastPathComponent
        let script = Script(
          name: file, file: file, filePath: filePath, type: .community)
        return script
      }
  }
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

  public static func saveCommunityScript(_ localFile: URL, metadata: Script) {
    try? FileManager.default.createDirectory(
      at: localCommunityScriptsURL, withIntermediateDirectories: true)
    try? FileManager.default.moveItem(
      atPath: localFile.path,
      toPath: localCommunityScriptsURL.appendingPathComponent(metadata.file).path)
    try? FileManager.default.removeItem(at: localFile)
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

// remote scripts stuff
extension ScriptZoo {
  public static func fetch(completion: @escaping ([Script]) -> Void) {
    guard let url = URL(string: "https://scripts.drawthings.ai/scripts.json") else { return }
    URLSession.shared.dataTask(with: url) { data, response, error in
      guard let data, error == nil else {
        print("Error downloading data: \(error?.localizedDescription ?? "Unknown error")")
        return
      }
      let group = DispatchGroup()
      // Parse the JSON data
      do {
        let remoteScripts = try JSONDecoder().decode([Script].self, from: data).filter {
          wizRemoteList.contains($0.file)
        }
        remoteScripts.forEach { metadata in
          group.enter()
          URLSession.shared.downloadTask(
            with: URL(string: "https://scripts.drawthings.ai/scripts/\(metadata.file)")!,
            completionHandler: { localUrl, response, error in
              guard let localUrl else {
                return group.leave()
              }
              saveCommunityScript(localUrl, metadata: metadata)
              group.leave()
            }
          ).resume()
        }
        group.notify(queue: .main) {
          completion(availableCommunityScripts)
        }
      } catch {
        group.leave()
        print("Failed to parse JSON: \(error)")
      }
    }.resume()
  }

  static var wizRemoteList: Set<String> = [
    "creative-upscale.js"
  ]
}
