#if canImport(WebKit)
  import Foundation
  import WebKit
  import XCTest

  @testable import WebSearch

  private final class SearchWebView: WKWebView {
    var currentURL: URL?
    var html = "<html>Loading</html>"
    var onLoad: ((URLRequest) -> Void)?
    override var url: URL? { currentURL }

    override func load(_ request: URLRequest) -> WKNavigation? {
      currentURL = request.url
      onLoad?(request)
      navigationDelegate?.webView?(self, didCommit: nil)
      navigationDelegate?.webView?(self, didFinish: nil)
      return nil
    }

    override func evaluateJavaScript(
      _ javaScriptString: String,
      completionHandler: (@MainActor @Sendable (Any?, Error?) -> Void)? = nil
    ) {
      completionHandler?(javaScriptString.contains("button.click") ? false : html, nil)
    }
  }

  final class WebKitDuckDuckGoSearchTests: XCTestCase {
    private let results = """
      <article data-testid="result"><a data-testid="result-title-a" href="https://example.com/doc">Example</a><div data-result="snippet">Snippet</div></article>
      """

    func testChallengeUsesSameViewAndResumesAfterConfirmation() {
      let done = expectation(description: "search resumed")
      var presented = 0
      var dismissed = 0
      var loads = 0
      let browser = WebKitDuckDuckGoSearch(
        challengeHandler: { webView, _ in
          presented += 1
          let webView = webView as! SearchWebView
          webView.html = "<p>Verification complete</p>"
          webView.currentURL = URL(string: "https://duckduckgo.com/anomaly.js")!
          return { dismissed += 1 }
        },
        makeWebView: { configuration in
          XCTAssertTrue(configuration.websiteDataStore.isPersistent)
          let webView = SearchWebView(frame: .zero, configuration: configuration)
          webView.html = "<form id='challenge-form'></form>"
          webView.onLoad = { [weak webView] request in
            loads += 1
            XCTAssertNil(request.value(forHTTPHeaderField: "User-Agent"))
            XCTAssertEqual(request.cachePolicy, .reloadIgnoringLocalCacheData)
            if loads == 2 { webView?.html = self.results }
          }
          return webView
        })
      browser.search(query: "example", options: DuckDuckGoSearchOptions()) { result in
        XCTAssertEqual(try? result.get().count, 1)
        done.fulfill()
      }
      wait(for: [done], timeout: 5)
      XCTAssertEqual(presented, 1)
      XCTAssertEqual(dismissed, 1)
      XCTAssertEqual(loads, 2)
    }

    func testChallengeCancellationCompletesOnceAndNextSearchRuns() {
      let cancelled = expectation(description: "cancelled")
      cancelled.assertForOverFulfill = true
      let next = expectation(description: "next search")
      var dismissed = 0
      let browser = WebKitDuckDuckGoSearch(
        challengeHandler: { _, cancel in
          cancel()
          cancel()
          return { dismissed += 1 }
        },
        makeWebView: { configuration in
          let webView = SearchWebView(frame: .zero, configuration: configuration)
          webView.onLoad = { [weak webView] request in
            webView?.html =
              request.url!.absoluteString.contains("cancel")
              ? "<form id='challenge-form'></form>" : self.results
          }
          return webView
        })
      browser.search(query: "cancel", options: DuckDuckGoSearchOptions()) { result in
        guard case .failure(let error) = result else { return XCTFail("Expected cancellation") }
        XCTAssertEqual((error as? URLError)?.code, .cancelled)
        cancelled.fulfill()
      }
      browser.search(query: "next", options: DuckDuckGoSearchOptions()) { result in
        XCTAssertEqual(try? result.get().count, 1)
        next.fulfill()
      }
      wait(for: [cancelled, next], timeout: 5)
      XCTAssertEqual(dismissed, 1)
    }

    func testChallengeWithoutPresenterFailsPromptly() {
      let done = expectation(description: "blocked")
      let browser = WebKitDuckDuckGoSearch(makeWebView: { configuration in
        let webView = SearchWebView(frame: .zero, configuration: configuration)
        webView.html = "<form id='challenge-form'></form>"
        return webView
      })
      browser.search(query: "example", options: DuckDuckGoSearchOptions()) { result in
        guard case .failure(WebSearchError.searchBlocked) = result else {
          return XCTFail("Expected explicit challenge error")
        }
        done.fulfill()
      }
      wait(for: [done], timeout: 3)
    }

    func testChallengeTimeoutDismissesView() {
      let done = expectation(description: "challenge timeout")
      var dismissed = false
      let browser = WebKitDuckDuckGoSearch(
        challengeTimeout: 1,
        challengeHandler: { _, _ in
          return { dismissed = true }
        },
        makeWebView: { configuration in
          let webView = SearchWebView(frame: .zero, configuration: configuration)
          webView.html = "<form id='challenge-form'></form>"
          return webView
        })
      browser.search(query: "example", options: DuckDuckGoSearchOptions()) { result in
        guard case .failure(WebSearchError.searchBlocked) = result else {
          return XCTFail("Expected challenge timeout")
        }
        XCTAssertTrue(dismissed)
        done.fulfill()
      }
      wait(for: [done], timeout: 4)
    }

    func testLoadingPageRetriesOnceThenTimesOutInsteadOfReturningEmpty() {
      let done = expectation(description: "loading timeout")
      var loads = 0
      let browser = WebKitDuckDuckGoSearch(makeWebView: { configuration in
        let webView = SearchWebView(frame: .zero, configuration: configuration)
        webView.onLoad = { _ in loads += 1 }
        return webView
      })
      browser.search(query: "example", options: DuckDuckGoSearchOptions(timeout: 1)) { result in
        guard case .failure(let error) = result else { return XCTFail("Expected timeout") }
        XCTAssertEqual((error as? URLError)?.code, .timedOut)
        done.fulfill()
      }
      wait(for: [done], timeout: 4)
      XCTAssertEqual(loads, 2)
    }

    func testBrowserFormEncodedQueryMatchesOriginal() {
      let done = expectation(description: "form encoded query")
      let browser = WebKitDuckDuckGoSearch(makeWebView: { configuration in
        let webView = SearchWebView(frame: .zero, configuration: configuration)
        webView.html = self.results
        webView.onLoad = { [weak webView] _ in
          webView?.currentURL = URL(string: "https://duckduckgo.com/?q=C%2B%2B+reference")
        }
        return webView
      })
      browser.search(query: "C++ reference", options: DuckDuckGoSearchOptions()) { result in
        XCTAssertEqual(try? result.get().count, 1)
        done.fulfill()
      }
      wait(for: [done], timeout: 3)
    }

    func testConcurrentBrowserQueriesRemainIndependent() {
      let done = expectation(description: "all queries")
      done.expectedFulfillmentCount = 3
      var createdViews = 0
      let browser = WebKitDuckDuckGoSearch(makeWebView: { configuration in
        createdViews += 1
        let webView = SearchWebView(frame: .zero, configuration: configuration)
        webView.onLoad = { [weak webView] request in
          let query = URLComponents(url: request.url!, resolvingAgainstBaseURL: false)!.queryItems!
            .first(where: { $0.name == "q" })!.value!
          webView?.html = self.results.replacingOccurrences(of: "Example", with: query)
        }
        return webView
      })
      for query in ["first", "second", "third"] {
        browser.search(query: query, options: DuckDuckGoSearchOptions()) { result in
          XCTAssertEqual(try? result.get().first?.title, query)
          done.fulfill()
        }
      }
      wait(for: [done], timeout: 5)
      XCTAssertEqual(createdViews, 3)
    }
  }
#endif
