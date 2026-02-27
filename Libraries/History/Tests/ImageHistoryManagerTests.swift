import Foundation
import History
import SQLiteDflat
import XCTest

final class ImageHistoryManagerTests: XCTestCase {
  private var workspace: SQLiteWorkspace!
  private var temporaryDirectory: URL!

  override func setUpWithError() throws {
    try super.setUpWithError()
    temporaryDirectory = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(
      "ImageHistoryManagerTests-\(UUID().uuidString)")
    try FileManager.default.createDirectory(
      at: temporaryDirectory, withIntermediateDirectories: true, attributes: nil)
    workspace = SQLiteWorkspace(
      filePath: temporaryDirectory.appendingPathComponent("history.sqlite3").path,
      fileProtectionLevel: .noProtection)
  }

  override func tearDownWithError() throws {
    workspace.shutdown(flags: .truncate)
    workspace = nil
    try? FileManager.default.removeItem(at: temporaryDirectory)
    temporaryDirectory = nil
    try super.tearDownWithError()
  }

  private func onMain<T>(_ block: () -> T) -> T {
    if Thread.isMainThread {
      return block()
    }
    return DispatchQueue.main.sync(execute: block)
  }

  private func seedHistory(nodes: [TensorHistoryNode]) {
    let expectation = expectation(description: "seed-history")
    workspace.performChanges([TensorHistoryNode.self]) { transactionContext in
      for node in nodes {
        transactionContext.try(submit: TensorHistoryNodeChangeRequest.upsertRequest(node))
      }
    } completionHandler: { _ in
      DispatchQueue.main.async {
        expectation.fulfill()
      }
    }
    wait(for: [expectation], timeout: 5)
  }

  func testUnsacredSeekPersistsResolvedLineage() {
    seedHistory(
      nodes: [
        TensorHistoryNode(lineage: 2, logicalTime: 0, seed: 2),
        TensorHistoryNode(lineage: 3, logicalTime: 0, seed: 3),
      ])

    let manager = onMain {
      ImageHistoryManager(
        project: workspace, filePath: temporaryDirectory.appendingPathComponent("store.ckpt").path)
    }

    XCTAssertEqual(manager.lineage, 3)

    let isSacred = onMain {
      manager.seek(to: 0, lineage: 2)
    }

    XCTAssertFalse(isSacred)
    XCTAssertEqual(manager.lineage, 2)
    XCTAssertEqual(workspace.dictionary["image_seek_to_lineage", Int.self], 2)
  }

  func testSacredSeekClearsStoredLineage() {
    seedHistory(
      nodes: [
        TensorHistoryNode(lineage: 2, logicalTime: 0, seed: 2),
        TensorHistoryNode(lineage: 3, logicalTime: 0, seed: 3),
      ])

    let manager = onMain {
      ImageHistoryManager(
        project: workspace, filePath: temporaryDirectory.appendingPathComponent("store.ckpt").path)
    }

    _ = onMain {
      manager.seek(to: 0, lineage: 2)
    }
    XCTAssertEqual(workspace.dictionary["image_seek_to_lineage", Int.self], 2)

    _ = onMain {
      manager.seek(to: 0, lineage: nil)
    }

    XCTAssertNil(workspace.dictionary["image_seek_to_lineage", Int.self])
  }
}
