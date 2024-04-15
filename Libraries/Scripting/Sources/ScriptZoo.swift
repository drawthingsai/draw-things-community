import Foundation
import Utils

public struct ScriptZoo {

  public enum ScriptType {
    case user
    case sample
  }

  public struct Script: Equatable, Hashable {
    public let type: ScriptType
    public let name: String
    public let filename: String
    public let path: String
    public var isSampleDuplicate: Bool = false
  }

  public static let scriptsUrl: URL = {
    let urls = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
    return urls.first!.appendingPathComponent("Scripts")
  }()
  public static var availableScripts: [Script] {
    var userScriptNames = [String: Int]()
    var scripts = ((try? FileManager.default.contentsOfDirectory(atPath: scriptsUrl.path)) ?? [])
      .enumerated().map {
        let script = Script(
          type: .user, name: $1, filename: $1, path: scriptsUrl.appendingPathComponent($1).path)
        userScriptNames[$1] = $0
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
          Script(type: .sample, name: String(name), filename: String(filename), path: path)
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
    try? FileManager.default.copyItem(
      at: URL(fileURLWithPath: script.path), to: scriptsUrl.appendingPathComponent(filename))
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
