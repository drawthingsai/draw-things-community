import Foundation

extension String: Error {}

extension String {
  public func removingSuffix(_ suffix: String) -> String {
    if hasSuffix(suffix) {
      var copy = String(self)
      let range = copy.range(of: suffix, options: .backwards)!
      copy.removeSubrange(range)
      return copy
    } else {
      return self
    }
  }
}
