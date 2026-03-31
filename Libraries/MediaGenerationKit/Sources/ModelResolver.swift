import Foundation
import ModelZoo

public struct MediaGenerationResolvedModel: Sendable, Hashable {
  public var file: String
  public var name: String
  public var description: String
  public var version: String?
  public var huggingFaceLink: String?
  public var isDownloaded: Bool

  public init(
    file: String,
    name: String,
    description: String,
    version: String? = nil,
    huggingFaceLink: String? = nil,
    isDownloaded: Bool = false
  ) {
    self.file = file
    self.name = name
    self.description = description
    self.version = version
    self.huggingFaceLink = huggingFaceLink
    self.isDownloaded = isDownloaded
  }
}

internal enum ModelZooSource {
  case builtin
  case bundled
  case remote
}

internal enum ModelZooLoader {
  private static let remoteURL = URL(string: "https://models.drawthings.ai/models.json")!
  private static let lock = NSLock()
  private static var bundledCache: [ModelZoo.Specification]?
  private static var remoteCache: [ModelZoo.Specification]?

  static func specifications(from source: ModelZooSource, offline: Bool) -> [ModelZoo.Specification] {
    switch source {
    case .builtin:
      return ModelZoo.availableSpecifications.filter { $0.remoteApiModelConfig == nil }
    case .bundled:
      lock.lock()
      if let bundledCache {
        lock.unlock()
        return bundledCache
      }
      lock.unlock()
      let loaded = MediaGenerationResourceLoader.bundledData(resource: "models").flatMap(parse) ?? []
      lock.lock()
      bundledCache = loaded
      lock.unlock()
      return loaded
    case .remote:
      guard !offline else { return [] }
      lock.lock()
      if let remoteCache {
        lock.unlock()
        return remoteCache
      }
      lock.unlock()
      let loaded = MediaGenerationResourceLoader.fetchRemoteData(url: remoteURL).flatMap(parse) ?? []
      lock.lock()
      remoteCache = loaded
      lock.unlock()
      return loaded
    }
  }

  private static func parse(_ data: Data) -> [ModelZoo.Specification] {
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    guard
      let specifications = try? decoder.decode(
        [FailableDecodable<ModelZoo.Specification>].self,
        from: data
      ).compactMap(\.value)
    else {
      return []
    }
    return specifications.filter { $0.remoteApiModelConfig == nil }
  }
}

internal enum ModelResolver {
  typealias Model = MediaGenerationResolvedModel

  static func resolve(_ input: String, offline: Bool) throws -> Model? {
    if let specification = specification(for: input, offline: offline) {
      return model(from: specification)
    }
    return nil
  }

  static func specification(for input: String, offline: Bool) -> ModelZoo.Specification? {
    if let specification = ModelZoo.resolveModelReference(input)?.specification {
      return specification
    }

    if let bundled = matchingSpecification(
      for: input,
      in: ModelZooLoader.specifications(from: .bundled, offline: offline)
    ) {
      primeOverrideMapping(with: bundled)
      return bundled
    }

    if let remote = matchingSpecification(
      for: input,
      in: ModelZooLoader.specifications(from: .remote, offline: offline)
    ) {
      primeOverrideMapping(with: remote)
      return remote
    }

    return nil
  }

  static func suggestions(_ input: String, limit: Int = 5, offline: Bool) -> [Model] {
    var results = [Model]()
    var seen = Set<String>()

    for specification in ModelZoo.candidateSpecifications(forModelReference: input, limit: limit) {
      let candidate = model(from: specification)
      if seen.insert(candidate.file).inserted {
        results.append(candidate)
      }
    }

    for source in [ModelZooSource.bundled, .remote] {
      guard results.count < limit else { break }
      let candidates = candidateSpecifications(
        for: input,
        in: ModelZooLoader.specifications(from: source, offline: offline),
        limit: limit - results.count
      )
      for specification in candidates {
        let candidate = model(from: specification)
        if seen.insert(candidate.file).inserted {
          results.append(candidate)
        }
      }
    }

    return Array(results.prefix(limit))
  }

  static func model(from specification: ModelZoo.Specification) -> Model {
    Model(
      file: specification.file,
      name: specification.name,
      description: specification.note ?? "",
      version: ModelZoo.humanReadableNameForVersion(specification.version),
      huggingFaceLink: specification.huggingFaceLink,
      isDownloaded: ModelZoo.isModelDownloaded(specification.file)
    )
  }

  static func catalogModels(includeDownloaded: Bool, offline: Bool) -> [MediaGenerationResolvedModel] {
    var deduplicated = [String: ModelZoo.Specification]()
    for source in [ModelZooSource.builtin, .bundled, .remote] {
      for specification in ModelZooLoader.specifications(from: source, offline: offline) {
        deduplicated[specification.file] = specification
      }
    }

    return deduplicated.values
      .map(model(from:))
      .filter { includeDownloaded || !$0.isDownloaded }
      .sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
  }

  private static func matchingSpecification(
    for input: String,
    in specifications: [ModelZoo.Specification]
  ) -> ModelZoo.Specification? {
    if let exactFileMatch = specifications.first(where: { $0.file == input }) {
      return exactFileMatch
    }
    if let exactNameMatch = specifications.first(where: { $0.name == input }) {
      return exactNameMatch
    }
    guard let canonicalRepo = ModelZoo.normalizeHuggingFaceRepo(input) else {
      return nil
    }
    return specifications.first { specification in
      guard let huggingFaceLink = specification.huggingFaceLink,
        let specificationRepo = ModelZoo.normalizeHuggingFaceRepo(huggingFaceLink)
      else {
        return false
      }
      return specificationRepo == canonicalRepo
    }
  }

  private static func candidateSpecifications(
    for input: String,
    in specifications: [ModelZoo.Specification],
    limit: Int
  ) -> [ModelZoo.Specification] {
    let query = input.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    guard !query.isEmpty else { return [] }
    return specifications
      .filter {
        $0.file.lowercased().contains(query)
          || $0.name.lowercased().contains(query)
          || ($0.huggingFaceLink?.lowercased().contains(query) ?? false)
      }
      .sorted {
        let lhsExact = $0.file.caseInsensitiveCompare(input) == .orderedSame
          || $0.name.caseInsensitiveCompare(input) == .orderedSame
        let rhsExact = $1.file.caseInsensitiveCompare(input) == .orderedSame
          || $1.name.caseInsensitiveCompare(input) == .orderedSame
        if lhsExact != rhsExact {
          return lhsExact
        }
        return $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending
      }
      .prefix(max(0, limit))
      .map { $0 }
  }

  private static func primeOverrideMapping(with specification: ModelZoo.Specification) {
    ModelZoo.overrideMapping[specification.file] = specification
    if let huggingFaceLink = specification.huggingFaceLink,
      let repo = ModelZoo.normalizeHuggingFaceRepo(huggingFaceLink)
    {
      ModelZoo.huggingFaceRepoOverrideMapping[repo] = specification
    }
  }
}
