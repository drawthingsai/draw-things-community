import Foundation
import SwiftSoup
import XCTest

@testable import WebSearch

#if canImport(FoundationNetworking)
  import FoundationNetworking
#endif

private struct StubHttpTransport: HttpTransport {
  var handler: (URLRequest) -> Result<(Data, HTTPURLResponse), Error>

  func data(
    for request: URLRequest,
    completion: @escaping (Result<(Data, HTTPURLResponse), Error>) -> Void
  ) {
    completion(handler(request))
  }
}

final class WebSearchTests: XCTestCase {
  func testWebSearchProviderCodableIncludesDisabled() throws {
    let encoded = try JSONEncoder().encode(WebSearchProvider.disabled)
    XCTAssertEqual(String(decoding: encoded, as: UTF8.self), "\"disabled\"")
    XCTAssertEqual(try JSONDecoder().decode(WebSearchProvider.self, from: encoded), .disabled)
  }

  func testDuckDuckGoRedirectDecoding() {
    let raw =
      "//duckduckgo.com/l/?uddg=https%3A%2F%2Fgithub.com%2Fscinfu%2FSwiftSoup&rut=abc"
    let decoded = DuckDuckGoHTMLParser.decodeDuckDuckGoURL(
      raw, baseURL: URL(string: "https://html.duckduckgo.com/html/")!)
    XCTAssertEqual(decoded?.absoluteString, "https://github.com/scinfu/SwiftSoup")
  }

  func testSearchResultParsing() throws {
    let html = """
      <html><body>
        <div class="result results_links results_links_deep web-result">
          <h2 class="result__title">
            <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fdoc"> Example Doc </a>
          </h2>
          <a class="result__url"> example.com/doc </a>
          <a class="result__snippet"> A <b>useful</b> result. </a>
        </div>
        <div class="nav-link">
          <form action="/html/" method="post">
            <input type="hidden" name="q" value="example" />
            <input type="hidden" name="s" value="10" />
          </form>
        </div>
      </body></html>
      """
    let parsed = try DuckDuckGoHTMLParser.parse(
      html: html, baseURL: URL(string: "https://html.duckduckgo.com/html/")!)
    XCTAssertEqual(parsed.results.count, 1)
    XCTAssertEqual(parsed.results[0].title, "Example Doc")
    XCTAssertEqual(parsed.results[0].url.absoluteString, "https://example.com/doc")
    XCTAssertEqual(parsed.results[0].displayURL, "example.com/doc")
    XCTAssertEqual(parsed.results[0].snippet, "A useful result.")
    XCTAssertEqual(parsed.nextParameters?.count, 2)
  }

  func testSearchRequestParameters() throws {
    let request = try DuckDuckGoSearch.makeInitialRequest(
      endpoint: URL(string: "https://html.duckduckgo.com/html/")!,
      query: "swift urlsession",
      options: DuckDuckGoSearchOptions(
        region: "us-en", safeSearch: .off, timeFilter: .week, maxResults: 10, pages: 1))
    let components = URLComponents(url: request.url!, resolvingAgainstBaseURL: false)
    let queryItems = Dictionary(
      uniqueKeysWithValues: components!.queryItems!.map { ($0.name, $0.value ?? "") })
    XCTAssertEqual(queryItems["q"], "swift urlsession")
    XCTAssertEqual(queryItems["kl"], "us-en")
    XCTAssertEqual(queryItems["kp"], "-2")
    XCTAssertEqual(queryItems["df"], "w")
    XCTAssertEqual(request.value(forHTTPHeaderField: "User-Agent"), WebSearchDefaultUserAgent)
    XCTAssertTrue(request.value(forHTTPHeaderField: "User-Agent")?.contains("Mozilla/5.0") == true)
    XCTAssertTrue(request.value(forHTTPHeaderField: "User-Agent")?.contains("Chrome/") == true)
  }

  func testDuckDuckGoAccessChallengeDetection() async throws {
    let html = """
      <html><body>
        <h1>DuckDuckGo</h1>
        <p>Unfortunately, bots use DuckDuckGo too.</p>
        <p>Please complete the following challenge to confirm this search was made by a human.</p>
        <p>Select all squares containing a duck:</p>
      </body></html>
      """
    let response = HTTPURLResponse(
      url: URL(string: "https://html.duckduckgo.com/html/")!,
      statusCode: 202,
      httpVersion: nil,
      headerFields: ["Content-Type": "text/html"])!
    let search = DuckDuckGoSearch(
      httpTransport: StubHttpTransport { _ in .success((Data(html.utf8), response)) })

    do {
      _ = try await search.search(query: "site:github.com ios_system")
      XCTFail("Expected searchBlocked")
    } catch WebSearchError.searchBlocked(let message, let url) {
      XCTAssertEqual(message, "DuckDuckGo returned an access challenge instead of search results.")
      XCTAssertEqual(url?.absoluteString, "https://html.duckduckgo.com/html/")
    } catch {
      XCTFail("Unexpected error: \(error)")
    }
  }

  func testSogouSearchRequestParameters() throws {
    let request = try SogouSearch.makeRequest(
      endpoint: URL(string: "https://www.sogou.com/web")!,
      query: "SwiftSoup GitHub",
      page: 2,
      options: SogouSearchOptions(
        timeFilter: .week, maxResults: 10, pages: 2, timeout: 12))
    let components = URLComponents(url: request.url!, resolvingAgainstBaseURL: false)
    let queryItems = Dictionary(
      uniqueKeysWithValues: components!.queryItems!.map { ($0.name, $0.value ?? "") })
    XCTAssertEqual(queryItems["query"], "SwiftSoup GitHub")
    XCTAssertEqual(queryItems["page"], "2")
    XCTAssertEqual(queryItems["ie"], "utf8")
    XCTAssertEqual(queryItems["tsn"], "2")
    XCTAssertEqual(request.value(forHTTPHeaderField: "User-Agent"), WebSearchDefaultUserAgent)
    XCTAssertEqual(
      request.value(forHTTPHeaderField: "Accept-Language"), "zh-CN,zh;q=0.9,en;q=0.8")
  }

  func testSogouSearchResultParsingPrefersMetadataURL() throws {
    let html = """
      <html><body>
        <div class="vrwrap" id="sogou_vr_30000000_wrap_4">
          <h3 class="vr-title">
            <a target="_blank" href="/link?url=wrapped">GitHub - holzschu/<em>ios_system</em></a>
          </h3>
          <div class="fz-mid space-txt base-ellipsis clamp2">
            Drop-in replacement for system() in iOS.
          </div>
          <a class="citeLinkClass" target="_blank" href="/link?url=wrapped">
            <span>GitHub</span>
            <span>https://github.com/h...</span>
            <span>2024-03-23</span>
          </a>
          <div class="r-sech ext_query" data-url="https://github.com/holzschu/ios_system"></div>
        </div>
      </body></html>
      """
    let parsed = try SogouHTMLParser.parse(
      html: html, baseURL: URL(string: "https://www.sogou.com/web")!)
    XCTAssertEqual(parsed.results.count, 1)
    XCTAssertEqual(parsed.results[0].title, "GitHub - holzschu/ios_system")
    XCTAssertEqual(parsed.results[0].url.absoluteString, "https://github.com/holzschu/ios_system")
    XCTAssertEqual(parsed.results[0].displayURL, "https://github.com/h...")
    XCTAssertEqual(parsed.results[0].snippet, "Drop-in replacement for system() in iOS.")
    XCTAssertEqual(parsed.results[0].source, "sogou")
  }

  func testMarkdownConversionPreservesCommonShapes() throws {
    let document = try SwiftSoup.parse(
      """
      <main>
        <h1>Title</h1>
        <p>See <a href="https://example.com">Example</a>.</p>
        <ul><li>One</li><li>Two</li></ul>
        <pre><code>let x = 1</code></pre>
        <table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>
      </main>
      """, "https://example.com")
    let markdown = try HTMLMarkdownConverter.convert(try document.select("main").first()!)
    XCTAssertTrue(markdown.contains("# Title"))
    XCTAssertTrue(markdown.contains("[Example](https://example.com)"))
    XCTAssertTrue(markdown.contains("- One"))
    XCTAssertTrue(markdown.contains("```"))
    XCTAssertTrue(markdown.contains("| A | B |"))
  }

  func testWebFetchRenderOutputFormatsHTMLLikeOpenCodeFetch() throws {
    let html = """
      <html>
        <head><title>Ignored</title><script>bad()</script></head>
        <body><main><h1>Title</h1><p>Body <a href="/doc">link</a>.</p></main></body>
      </html>
      """
    let baseURL = URL(string: "https://example.com/root/")!

    let markdown = try WebFetch.renderOutput(
      body: html, contentType: "text/html; charset=utf-8", format: .markdown, baseURL: baseURL)
    XCTAssertTrue(markdown.contains("# Title"))
    XCTAssertTrue(markdown.contains("[link](https://example.com/doc)"))
    XCTAssertFalse(markdown.contains("bad()"))

    let text = try WebFetch.renderOutput(
      body: html, contentType: "text/html; charset=utf-8", format: .text, baseURL: baseURL)
    XCTAssertEqual(text, "Title Body link.")

    let rawHTML = try WebFetch.renderOutput(
      body: html, contentType: "text/html; charset=utf-8", format: .html, baseURL: baseURL)
    XCTAssertTrue(rawHTML.contains("<script>bad()</script>"))
  }

  func testWebFetchResultEncodesOpenCodeToolFields() throws {
    let url = URL(string: "https://example.com")!
    let result = WebFetchResult(
      title: "https://example.com (text/html)",
      metadata: WebFetchMetadata(
        url: url,
        finalURL: url,
        contentType: "text/html",
        statusCode: 200,
        elapsedSeconds: 0.1,
        byteCount: 12,
        format: .markdown),
      output: "Body")
    let data = try JSONEncoder().encode(result)
    let json = String(decoding: data, as: UTF8.self)
    XCTAssertTrue(json.contains("\"title\""))
    XCTAssertTrue(json.contains("\"metadata\""))
    XCTAssertTrue(json.contains("\"output\""))
  }

  func testDuckDuckGoSearchCompletionAPIUsesAsyncImplementation() {
    let html = """
      <html><body>
        <div class="result results_links results_links_deep web-result">
          <h2 class="result__title">
            <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fdoc"> Example Doc </a>
          </h2>
          <a class="result__url"> example.com/doc </a>
          <a class="result__snippet"> A useful result. </a>
        </div>
      </body></html>
      """
    let response = HTTPURLResponse(
      url: URL(string: "https://html.duckduckgo.com/html/")!,
      statusCode: 200,
      httpVersion: nil,
      headerFields: ["Content-Type": "text/html"])!
    let search = DuckDuckGoSearch(
      httpTransport: StubHttpTransport { _ in .success((Data(html.utf8), response)) })
    let finished = expectation(description: "search completion")

    search.search(query: "example") { result in
      switch result {
      case .success(let results):
        XCTAssertEqual(results.first?.title, "Example Doc")
        XCTAssertEqual(results.first?.url.absoluteString, "https://example.com/doc")
      case .failure(let error):
        XCTFail("Unexpected error: \(error)")
      }
      finished.fulfill()
    }

    wait(for: [finished], timeout: 1)
  }

  func testWebFetchAsyncAPIWrapsCompletionTransport() async throws {
    let html = "<html><body><h1>Title</h1><p>Body</p></body></html>"
    let url = URL(string: "https://example.com")!
    let response = HTTPURLResponse(
      url: url,
      statusCode: 200,
      httpVersion: nil,
      headerFields: ["Content-Type": "text/html"])!
    let fetch = WebFetch(
      httpTransport: StubHttpTransport { request in
        XCTAssertEqual(request.value(forHTTPHeaderField: "User-Agent"), WebSearchDefaultUserAgent)
        XCTAssertEqual(
          request.value(forHTTPHeaderField: "Accept"),
          WebFetchFormat.markdown.acceptHeader)
        return .success((Data(html.utf8), response))
      })

    let result = try await fetch.fetch(url: url)

    XCTAssertEqual(result.title, "https://example.com (text/html)")
    XCTAssertTrue(result.output.contains("# Title"))
    XCTAssertEqual(result.metadata.format, .markdown)
  }

  func testWebFetchFallsBackToDefaultUserAgentWhenEmpty() async throws {
    let html = "<html><body><p>Body</p></body></html>"
    let url = URL(string: "https://example.com")!
    let response = HTTPURLResponse(
      url: url,
      statusCode: 200,
      httpVersion: nil,
      headerFields: ["Content-Type": "text/html"])!
    let fetch = WebFetch(
      httpTransport: StubHttpTransport { request in
        XCTAssertEqual(request.value(forHTTPHeaderField: "User-Agent"), WebSearchDefaultUserAgent)
        return .success((Data(html.utf8), response))
      })

    _ = try await fetch.fetch(url: url, options: WebFetchOptions(userAgent: " "))
  }
}
