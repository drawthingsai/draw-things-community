import Foundation
import SwiftSoup

#if canImport(FoundationNetworking)
  import FoundationNetworking
#endif

/// Searches DuckDuckGo's HTML endpoint.
public struct DuckDuckGoSearch {
  private let httpTransport: HttpTransport
  private static let endpoint = URL(string: "https://html.duckduckgo.com/html/")!

  /// Creates a DuckDuckGo search tool.
  public init(httpTransport: HttpTransport = URLSessionHttpTransport()) {
    self.httpTransport = httpTransport
  }

  /// Searches DuckDuckGo and calls `completion` with normalized, de-duplicated results.
  public func search(
    query: String,
    options: DuckDuckGoSearchOptions = DuckDuckGoSearchOptions(),
    completion: @escaping (Result<[SearchResult], Error>) -> Void
  ) {
    Task {
      do {
        completion(.success(try await searchAsync(query: query, options: options)))
      } catch {
        completion(.failure(error))
      }
    }
  }

  /// Searches DuckDuckGo with async/await by wrapping the completion-handler API.
  public func search(query: String, options: DuckDuckGoSearchOptions = DuckDuckGoSearchOptions())
    async throws -> [SearchResult]
  {
    try await withCheckedThrowingContinuation { continuation in
      search(query: query, options: options) { result in
        continuation.resume(with: result)
      }
    }
  }

  private func searchAsync(
    query: String, options: DuckDuckGoSearchOptions = DuckDuckGoSearchOptions()
  ) async throws -> [SearchResult] {
    let normalizedQuery = query.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !normalizedQuery.isEmpty else {
      return []
    }

    var results = [SearchResult]()
    var seenURLs = Set<String>()
    var nextParameters: [(String, String)]? = nil
    var page = 0

    while page < options.pages && results.count < options.maxResults {
      let request: URLRequest
      if page == 0 {
        request = try Self.makeInitialRequest(
          endpoint: Self.endpoint, query: normalizedQuery, options: options)
      } else if let parameters = nextParameters {
        request = Self.makeNextRequest(
          endpoint: Self.endpoint, parameters: parameters, options: options)
      } else {
        break
      }

      let (data, response) = try await httpTransport.data(for: request)
      let decodedBody = Self.decodeBody(data)
      if Self.isAccessChallenge(statusCode: response.statusCode, body: decodedBody) {
        throw WebSearchError.searchBlocked(
          "DuckDuckGo returned an access challenge instead of search results.", response.url)
      }
      guard (200..<300).contains(response.statusCode) else {
        throw WebSearchError.httpStatus(
          response.statusCode, response.url, body: decodedBody,
          headers: response.allHeaderFields.reduce(into: [String: String]()) {
            guard let key = $1.key as? String else { return }
            $0[key] = String(describing: $1.value)
          }, byteCount: data.count)
      }
      guard let html = decodedBody else {
        throw WebSearchError.bodyDecodingFailed(response.url)
      }

      let parsed = try DuckDuckGoHTMLParser.parse(html: html, baseURL: Self.endpoint)
      for result in parsed.results {
        let key = result.url.absoluteString
        guard !seenURLs.contains(key) else {
          continue
        }
        seenURLs.insert(key)
        results.append(
          SearchResult(
            rank: results.count + 1,
            title: result.title,
            url: result.url,
            displayURL: result.displayURL,
            snippet: result.snippet,
            source: result.source))
        if results.count >= options.maxResults {
          break
        }
      }
      nextParameters = parsed.nextParameters
      page += 1
    }

    return results
  }

  static func makeInitialRequest(
    endpoint: URL, query: String, options: DuckDuckGoSearchOptions
  ) throws -> URLRequest {
    guard var components = URLComponents(url: endpoint, resolvingAgainstBaseURL: false) else {
      throw WebSearchError.invalidURL(endpoint.absoluteString)
    }
    var queryItems = [
      URLQueryItem(name: "q", value: query),
      URLQueryItem(name: "kl", value: options.region),
      URLQueryItem(name: "kp", value: options.safeSearch.duckDuckGoValue),
    ]
    if let timeFilter = options.timeFilter {
      queryItems.append(URLQueryItem(name: "df", value: timeFilter.duckDuckGoValue))
    }
    components.queryItems = queryItems
    guard let url = components.url else {
      throw WebSearchError.invalidURL(endpoint.absoluteString)
    }
    var request = URLRequest(url: url)
    request.httpMethod = "GET"
    request.timeoutInterval = options.timeout
    request.setValue(options.userAgent, forHTTPHeaderField: "User-Agent")
    request.setValue("text/html,application/xhtml+xml", forHTTPHeaderField: "Accept")
    return request
  }

  static func makeNextRequest(
    endpoint: URL, parameters: [(String, String)], options: DuckDuckGoSearchOptions
  ) -> URLRequest {
    var request = URLRequest(url: endpoint)
    request.httpMethod = "POST"
    request.timeoutInterval = options.timeout
    request.setValue(options.userAgent, forHTTPHeaderField: "User-Agent")
    request.setValue("text/html,application/xhtml+xml", forHTTPHeaderField: "Accept")
    request.setValue(
      "application/x-www-form-urlencoded; charset=utf-8", forHTTPHeaderField: "Content-Type")
    request.httpBody = Self.formURLEncodedData(parameters)
    return request
  }

  static func formURLEncodedData(_ parameters: [(String, String)]) -> Data {
    var components = URLComponents()
    components.queryItems = parameters.map { URLQueryItem(name: $0.0, value: $0.1) }
    let encoded = components.percentEncodedQuery?.replacingOccurrences(of: "%20", with: "+") ?? ""
    return Data(encoded.utf8)
  }

  static func decodeBody(_ data: Data) -> String? {
    String(data: data, encoding: .utf8) ?? String(data: data, encoding: .isoLatin1)
  }

  static func isAccessChallenge(statusCode: Int, body: String?) -> Bool {
    guard let body else {
      return false
    }
    let lowercased = body.lowercased()
    if lowercased.contains("unfortunately, bots use duckduckgo too")
      || lowercased.contains("please complete the following challenge")
      || lowercased.contains("select all squares containing a duck")
    {
      return true
    }
    return statusCode == 202 && lowercased.contains("duckduckgo")
      && lowercased.contains("challenge")
  }
}

struct DuckDuckGoParsedPage {
  var results: [SearchResult]
  var nextParameters: [(String, String)]?
}

enum DuckDuckGoHTMLParser {
  static func parse(html: String, baseURL: URL) throws -> DuckDuckGoParsedPage {
    let document = try SwiftSoup.parse(html, baseURL.absoluteString)
    let elements = try document.select("div.result.web-result")
    var results = [SearchResult]()
    for element in elements {
      guard let titleElement = try element.select("a.result__a").first() else {
        continue
      }
      let rawHref = try titleElement.attr("href")
      guard let url = decodeDuckDuckGoURL(rawHref, baseURL: baseURL) else {
        continue
      }
      let title = normalizeWhitespace(try titleElement.text())
      guard !title.isEmpty else {
        continue
      }
      let snippet = normalizeWhitespace(
        try element.select("a.result__snippet").first()?.text() ?? "")
      let displayURL = normalizeWhitespace(
        try element.select("a.result__url").first()?.text() ?? "")
      results.append(
        SearchResult(
          rank: results.count + 1,
          title: title,
          url: url,
          displayURL: displayURL,
          snippet: snippet,
          source: "duckduckgo"))
    }
    return DuckDuckGoParsedPage(
      results: results, nextParameters: try nextFormParameters(from: document))
  }

  static func decodeDuckDuckGoURL(_ rawHref: String, baseURL: URL) -> URL? {
    let trimmed = rawHref.trimmingCharacters(in: .whitespacesAndNewlines)
    let candidate: URL?
    if trimmed.hasPrefix("//") {
      candidate = URL(string: "https:" + trimmed)
    } else {
      candidate = URL(string: trimmed, relativeTo: baseURL)?.absoluteURL
    }
    guard let url = candidate else {
      return nil
    }
    guard
      url.host?.lowercased().contains("duckduckgo.com") == true,
      url.path == "/l/" || url.path == "/l",
      let components = URLComponents(url: url, resolvingAgainstBaseURL: false),
      let encoded = components.queryItems?.first(where: { $0.name == "uddg" })?.value,
      let decoded = URL(string: encoded)
    else {
      return url
    }
    return decoded
  }

  private static func nextFormParameters(from document: Document) throws -> [(String, String)]? {
    guard let form = try document.select("div.nav-link form").first() else {
      return nil
    }
    let inputs = try form.select("input[type=hidden]")
    var parameters = [(String, String)]()
    for input in inputs {
      let name = try input.attr("name")
      guard !name.isEmpty else {
        continue
      }
      parameters.append((name, try input.attr("value")))
    }
    return parameters.isEmpty ? nil : parameters
  }
}

func normalizeWhitespace(_ value: String) -> String {
  value
    .components(separatedBy: .whitespacesAndNewlines)
    .filter { !$0.isEmpty }
    .joined(separator: " ")
}
