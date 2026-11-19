import Foundation
import ServerLoRALoader
import XCTest

final class LocalLoRAManagerTests: XCTestCase {
  func testNameThatIsNotContentHashIsRejected() throws {
    // rootDirectory/LoRAs is the local directory. Whatever else is under rootDirectory is what a
    // name must not be able to reach.
    let rootDirectory = FileManager.default.temporaryDirectory.appendingPathComponent(
      UUID().uuidString)
    let localDirectory = rootDirectory.appendingPathComponent("LoRAs")
    try FileManager.default.createDirectory(at: localDirectory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: rootDirectory) }
    // Nothing listens at this endpoint, and a rejected name does not get as far as a request.
    let localLoRAManager = LocalLoRAManager(
      r2Client: R2Client(
        accessKey: "", secret: "", endpoint: "http://127.0.0.1:9", bucket: "bucket"),
      localDirectory: localDirectory.path)
    let contentHash = String(repeating: "a", count: 64)
    for modelNames in [
      ["../evil/x"], ["x/../../evil/y_z"], ["\(contentHash)/../../evil/x"],
      [contentHash.uppercased()], [String(contentHash.dropLast())], [""],
      // The request fails on the first of these, however many follow.
      Array(repeating: "not a content hash", count: 1_000_000),
    ] {
      var results: [String: Bool]? = nil
      localLoRAManager.downloadRemoteLoRAs(modelNames) { _, _, _, _ in
      } cancellation: { _ in
      } completion: {
        results = $0
      }
      XCTAssertEqual(results, [modelNames[0]: false], modelNames[0].debugDescription)
    }
    XCTAssertEqual(
      try FileManager.default.contentsOfDirectory(atPath: rootDirectory.path), ["LoRAs"])
    XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: localDirectory.path), [])
  }
}
