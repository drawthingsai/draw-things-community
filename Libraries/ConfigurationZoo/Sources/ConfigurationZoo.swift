import DataModels
import Dflat
import Foundation
import ModelZoo
import ScriptDataModels

public struct ConfigurationZoo {
  public struct Specification {
    public var name: String
    public var negative: String?
    public var configuration: [String: Any]
    public init(
      name: String, negative: String?, configuration: [String: Any]
    ) {
      self.name = name
      self.negative = negative
      self.configuration = configuration
    }
  }

  public static var availableSpecifications: [Specification] = {
    let jsonFile = filePathForModelDownloaded("custom_configs.json")
    guard let jsonData = try? Data(contentsOf: URL(fileURLWithPath: jsonFile)) else {
      return []
    }

    guard
      let jsonSpecifications = try? JSONSerialization.jsonObject(with: jsonData) as? [[String: Any]]
    else {
      return []
    }
    var availableSpecifications = [Specification]()
    for specification in jsonSpecifications {
      guard let name = specification["name"] as? String,
        let configuration = specification["configuration"] as? [String: Any]
      else { continue }
      availableSpecifications.append(
        Specification(
          name: name, negative: specification["negative"] as? String, configuration: configuration))
    }
    return availableSpecifications
  }()

  private static var specificationMapping: [String: Specification] = {
    var mapping = [String: Specification]()
    for specification in availableSpecifications {
      mapping[specification.name] = specification
    }
    return mapping
  }()

  public static func read(from workspace: Workspace) {
    let configurations = workspace.fetch(for: GenerationConfiguration.self).where(
      GenerationConfiguration.id != 0,
      orderBy: [
        GenerationConfiguration.name.descending
      ])
    guard !configurations.isEmpty else { return }
    var specifications = [Specification]()
    for configuration in configurations {
      guard let name = configuration.name else { continue }
      guard
        let jsonData = try? JSONEncoder().encode(
          JSGenerationConfiguration(configuration: configuration)),
        var configurationDictionary = try? JSONSerialization.jsonObject(
          with: jsonData) as? [String: Any]
      else { continue }
      configurationDictionary["id"] = nil
      specifications.append(
        Specification(name: name, negative: nil, configuration: configurationDictionary))
    }
    var availableSpecifications = availableSpecifications
    for specification in specifications {
      guard !availableSpecifications.contains(where: { $0.name == specification.name }) else {
        continue
      }
      availableSpecifications.append(specification)
    }
    var specificationMapping = specificationMapping
    var customSpecifications = [[String: Any]]()
    for specification in availableSpecifications {
      specificationMapping[specification.name] = specification
      customSpecifications.append([
        "name": specification.name, "configuration": specification.configuration,
      ])
    }
    guard
      let jsonData = try? JSONSerialization.data(
        withJSONObject: customSpecifications, options: [.prettyPrinted, .sortedKeys])
    else { return }
    let jsonFile = filePathForModelDownloaded("custom_configs.json")
    do {
      try jsonData.write(to: URL(fileURLWithPath: jsonFile), options: .atomic)
      // If succeed. We can remove all configurations from the workspace.
      workspace.performChanges([GenerationConfiguration.self]) { transactionContext in
        let configurations = workspace.fetch(for: GenerationConfiguration.self).where(
          GenerationConfiguration.id != 0)
        for configuration in configurations {
          guard
            let deletionRequest = GenerationConfigurationChangeRequest.deletionRequest(
              configuration)
          else { continue }
          transactionContext.try(submit: deletionRequest)
        }
      }
    } catch {
      // Do nothing.
    }
    self.availableSpecifications = availableSpecifications
    self.specificationMapping = specificationMapping
  }

  public static func specification(_ name: String) -> Specification? {
    return specificationMapping[name]
  }

  private static func filePathForModelDownloaded(_ name: String) -> String {
    return ModelZoo.filePathForModelDownloaded(name)
  }

  public static func appendCustomSpecification(_ specification: Specification) {
    dispatchPrecondition(condition: .onQueue(.main))
    var customSpecifications = [[String: Any]]()
    let jsonFile = filePathForModelDownloaded("custom_configs.json")
    if let jsonData = try? Data(contentsOf: URL(fileURLWithPath: jsonFile)) {
      if let jsonSpecifications = try? JSONSerialization.jsonObject(with: jsonData)
        as? [[String: Any]]
      {
        customSpecifications.append(contentsOf: jsonSpecifications)
      }
    }
    customSpecifications = customSpecifications.filter {
      guard let name = $0["name"] as? String else { return false }
      return name != specification.name
    }
    var dictionary: [String: Any] = [
      "name": specification.name, "configuration": specification.configuration,
    ]
    dictionary["negative"] = specification.negative
    customSpecifications.append(dictionary)
    guard
      let jsonData = try? JSONSerialization.data(
        withJSONObject: customSpecifications, options: [.prettyPrinted, .sortedKeys])
    else { return }
    try? jsonData.write(to: URL(fileURLWithPath: jsonFile), options: .atomic)
    // Modify these two are not thread safe. availableSpecifications are OK. specificationMapping is particularly problematic (as it is access on both main thread and a background thread).
    var availableSpecifications = availableSpecifications
    availableSpecifications = availableSpecifications.filter { $0.name != specification.name }
    availableSpecifications.append(specification)
    self.availableSpecifications = availableSpecifications
    specificationMapping[specification.name] = specification
  }

  public static func remove(by nameToRemove: String) {
    dispatchPrecondition(condition: .onQueue(.main))
    var customSpecifications = [[String: Any]]()
    let jsonFile = filePathForModelDownloaded("custom_configs.json")
    if let jsonData = try? Data(contentsOf: URL(fileURLWithPath: jsonFile)) {
      if let jsonSpecifications = try? JSONSerialization.jsonObject(with: jsonData)
        as? [[String: Any]]
      {
        customSpecifications.append(contentsOf: jsonSpecifications)
      }
    }
    customSpecifications = customSpecifications.filter {
      guard let name = $0["name"] as? String else { return false }
      return name != nameToRemove
    }
    guard
      let jsonData = try? JSONSerialization.data(
        withJSONObject: customSpecifications, options: [.prettyPrinted, .sortedKeys])
    else { return }
    try? jsonData.write(to: URL(fileURLWithPath: jsonFile), options: .atomic)
    // Modify these two are not thread safe. availableSpecifications are OK. specificationMapping is particularly problematic (as it is access on both main thread and a background thread).
    var availableSpecifications = availableSpecifications
    availableSpecifications = availableSpecifications.filter { $0.name != nameToRemove }
    self.availableSpecifications = availableSpecifications
    specificationMapping[nameToRemove] = nil
  }
}
