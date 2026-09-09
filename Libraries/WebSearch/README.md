# DuckDuckGo browser fallback

`DuckDuckGoSearch()` first uses the HTML endpoint and falls back to `WebKitDuckDuckGoSearch` on Apple platforms when it receives a challenge, an unrecognized page, HTTP 403/408/429/5xx, or a transient timeout/connection loss. A successful HTTP search stays on that path. Offline, cancellation and TLS errors do not launch a browser. Pass `browserSearch: nil` to explicitly use HTTP only.

## Why this change

On the requesting Mac, the original HTTP implementation returned results for 4 of 10 benchmark queries. The other six returned DuckDuckGo's HTTP 202 duck-image challenge. The transport deliberately disables cookie storage; changing its user agent does not provide JavaScript execution or browser state.

Loading DuckDuckGo's regular site in WKWebView returned usable responses. The fallback uses the native WebKit user agent and `WKWebsiteDataStore.default()`, which persists cookies and website data. It reads the rendered DOM using semantic attributes rather than obfuscated CSS classes. [Apple documents the persistent data-store behavior](https://developer.apple.com/documentation/webkit/wkwebsitedatastore).

## Flow and limits

1. Attempt the existing HTML request. An empty parse is an error unless the document explicitly says there are no results.
2. On a recoverable failure, restart the original query in WebKit, preserving region, safe search, freshness, page limit and result limit. A failure on a later HTTP page also restarts the complete query in the browser.
3. Serialize browser searches on the main run loop. Each request owns its web view and shares the persistent website data store. Wait for a stable result count, normalize/deduplicate URLs, and use the site's “More results” button for pagination.
4. If the browser displays a challenge, Local Code presents that same live view. Completing verification resumes the search, including a single return to the original query if verification ends on a confirmation URL. Cancel and timeout dismiss the view and complete the request once. A CLI without a presenter returns `searchBlocked`.
5. If the page shell never produces results, allow one ordinary reload. No reload loop runs around a detected challenge. Main-frame navigation is limited to DuckDuckGo HTTPS pages; external result links and `!bang` redirects are not followed.

The HTTP timeout and browser timeout apply to their individual attempts; a browser reload or additional page can take another timeout interval. Human verification has a separate 120-second default timeout. Headless callers must keep the main run loop available to use WebKit, or explicitly select HTTP. Non-WebKit platforms retain HTTP-only behavior.

If an additional browser page produces no new unique results before its timeout, the search returns the results already collected. An explicit no-results page also preserves earlier results. A pending verification challenge still fails on timeout. Restarting the original query after verification resets pagination so later pages can be requested again.

The parser distinguishes genuine no-results pages from incomplete pages, including pages with related-search suggestions before the no-results message. Query matching decodes form-style `+` spaces while preserving literal `+` characters. Challenge wording inside result snippets or scripts does not by itself trigger verification.

## Alternatives considered

- **Cookie-enabled URLSession or the Lite endpoint:** potentially useful, but neither executes the regular site's JavaScript. DuckDuckGo describes HTML and Lite as non-JavaScript alternatives; another lightweight endpoint does not establish a reliable browser session. The working browser path avoids adding another scrape/retry chain. [DuckDuckGo documentation](https://safe.duckduckgo.com/duckduckgo-help-pages/features/non-javascript).
- **Model solving the image challenge:** adds vision-model dependencies and challenge-specific interaction logic. The browser fallback recovered these failures without solving a CAPTCHA. The implemented final step lets a user complete a persistent challenge directly in the provider's UI.
- **A supported search API:** the appropriate option when contractual reliability and sustained automated volume are required. Kagi is already supported by this library and provides a programmable search API. Brave also provides an independent search API requiring an API key. Neither is silently selected, since this changes provider/account/billing requirements. [Kagi Search API](https://help.kagi.com/kagi/api/search.html), [Brave Search API](https://brave.com/search/api/).

## Validation on 2026-09-08

Measured responses on this computer:

| Path | Valid responses | Nonempty responses | Median latency | P95 latency |
| --- | ---: | ---: | ---: | ---: |
| HTTP baseline | 4/10 | 4/10 | 0.64 s | 0.72 s |
| Browser | 10/10 | 9/10 | 1.38 s | 1.69 s |
| Automatic, batch 1 | 10/10 | 9/10 | 1.46 s | 1.78 s |
| Automatic, batch 2 | 10/10 | 9/10 | 1.18 s | 1.76 s |
| Automatic, batch 3 | 10/10 | 9/10 | 1.47 s | 1.77 s |
| Automatic, final (document cache bypassed) | 10/10 | 9/10 | 1.46 s | 1.76 s |

The exact query `"Swift ArgumentParser" GitHub` displayed a genuine no-results page in the regular browser; it is counted separately from nonempty results. Latency percentiles exclude errors. Additional browser checks returned 25 results over three pages, and five results each for a literal-plus query, a Chinese query, and a month-filtered strict-safe-search query. Raw benchmark metrics are in [Validation/2026-09-08.json](Validation/2026-09-08.json).

These are repeated local observations, not a guarantee against future provider outages, markup changes, rate limits or challenges. The live tests did not require user verification; challenge resume, cancellation, timeout and serialized searches are covered by deterministic WKWebView test doubles.

The final verification passed 35 tests, skipped the optional live Kagi test without credentials, and built both WebSearchCLI and the affected Local Code iOS library.

Reproduce with:

```sh
bazel test //Libraries/WebSearch:WebSearchTests
bazel build //Apps:WebSearchCLI
bazel-bin/Apps/WebSearchCLI benchmark --transport http --pretty
bazel-bin/Apps/WebSearchCLI benchmark --transport browser --pretty
bazel-bin/Apps/WebSearchCLI benchmark --transport automatic --pretty
bazel-bin/Apps/WebSearchCLI search 'SwiftSoup GitHub' --transport browser --pages 3 --max-results 25
bazel build //Apps/LocalCode:LocalCodeLib --platforms=@build_bazel_apple_support//platforms:ios_arm64
```

The full `//Apps/LocalCode:LocalCode` bundle cannot link in this checkout because the NodeMobile and Python iOS archives are Git LFS pointers rather than binary archives. This is independent of the WebSearch code.
