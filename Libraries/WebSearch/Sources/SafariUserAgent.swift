import Foundation

#if canImport(FoundationNetworking)
  import FoundationNetworking
#endif
#if canImport(WebKit)
  import WebKit
#endif
#if canImport(UIKit)
  import UIKit
#endif

/// Safari's user agent on this system. An app's web view reports WebKit's user agent, which
/// stops before Safari's `Version/… Safari/…` tokens, and sites read that as an embedded browser.
public enum SafariUserAgent {
  /// The Safari version the user agent reports after `Version/`.
  public static let version: String = {
    #if os(macOS) || targetEnvironment(macCatalyst)
      // Safari on macOS updates apart from the system, so its bundle has the version it reports.
      if let version = Bundle(path: "/Applications/Safari.app")?.infoDictionary?[
        "CFBundleShortVersionString"] as? String
      {
        return version
      }
    #endif
    // Safari on iOS ships with the system and reports its major and minor version.
    let version = ProcessInfo.processInfo.operatingSystemVersion
    return "\(version.majorVersion).\(version.minorVersion)"
  }()

  /// What Safari appends to WebKit's desktop user agent, as on macOS and iPadOS.
  public static var desktopApplicationName: String {
    "Version/\(version) Safari/605.1.15"
  }

  /// What Safari appends to WebKit's iPhone user agent.
  public static var mobileApplicationName: String {
    "Version/\(version) Mobile/15E148 Safari/604.1"
  }

  /// Safari on macOS, for requests made without WebKit. iPad Safari sends the same by default.
  public static var desktop: String {
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) "
      + desktopApplicationName
  }

  #if canImport(WebKit)
    /// Whether Safari shows this device desktop sites: on iPad and Mac, not iPhone.
    public static var prefersDesktop: Bool {
      #if os(iOS) && !targetEnvironment(macCatalyst)
        return UIDevice.current.userInterfaceIdiom != .phone
      #else
        return true
      #endif
    }

    /// Makes web views from `configuration` identify as Safari. WebKit picks the desktop or mobile
    /// user agent per navigation, so the content mode is pinned to keep the tokens appended here
    /// in step with it.
    public static func apply(
      to configuration: WKWebViewConfiguration, desktop: Bool = prefersDesktop
    ) {
      #if os(iOS)
        configuration.defaultWebpagePreferences.preferredContentMode = desktop ? .desktop : .mobile
      #endif
      configuration.applicationNameForUserAgent =
        desktop ? desktopApplicationName : mobileApplicationName
    }
  #endif
}

extension URLRequest {
  private static let safariNavigationHeaderNames = [
    "Sec-Fetch-Dest", "User-Agent", "Upgrade-Insecure-Requests", "Accept", "Sec-Fetch-Site",
    "Sec-Fetch-Mode", "Priority", "Origin", "Referer",
  ]

  /// Sets the headers WebKit adds when Safari navigates to a page. URLSession shares Safari's
  /// network stack and adds the same Accept-Language and Accept-Encoding, so the request looks
  /// like Safari's, except that URLSession orders headers its own way. Without `referrer` the
  /// navigation is typed; with one, it comes from a link or form on that same-origin page, whose
  /// referrer policy has already trimmed it. Set `httpMethod` first: posts carry an Origin.
  mutating func setSafariNavigationHeaders(userAgent: String, referrer: URL? = nil) {
    setValue("document", forHTTPHeaderField: "Sec-Fetch-Dest")
    setValue(userAgent, forHTTPHeaderField: "User-Agent")
    setValue("1", forHTTPHeaderField: "Upgrade-Insecure-Requests")
    setValue(
      "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
      forHTTPHeaderField: "Accept")
    setValue(referrer == nil ? "none" : "same-origin", forHTTPHeaderField: "Sec-Fetch-Site")
    setValue("navigate", forHTTPHeaderField: "Sec-Fetch-Mode")
    // URLSession sends u=3 over HTTP/2 and later otherwise.
    setValue("u=0, i", forHTTPHeaderField: "Priority")
    if let referrer {
      setValue(referrer.absoluteString, forHTTPHeaderField: "Referer")
      if httpMethod == "POST", let scheme = referrer.scheme, let host = referrer.host {
        setValue(
          "\(scheme)://\(host)\(referrer.port.map { ":\($0)" } ?? "")", forHTTPHeaderField: "Origin"
        )
      }
    }
  }

  /// Clears those headers before WebKit loads the request, so WebKit sends Safari's own.
  mutating func removeSafariNavigationHeaders() {
    for name in Self.safariNavigationHeaderNames {
      setValue(nil, forHTTPHeaderField: name)
    }
  }
}
