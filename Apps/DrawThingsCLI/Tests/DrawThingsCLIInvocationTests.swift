import Foundation
import ModelZoo
import XCTest

@testable import DrawThingsCLILib

final class DrawThingsCLIInvocationTests: XCTestCase {

  private func withContext(
    resolveModelsDirectory: ((URL?, @escaping (Result<URL, Error>) -> Void) -> Void)? = nil,
    unloadTextGenerator: (() -> Void)? = nil,
    _ body: (DrawThingsCLIContext, URL, () -> (String, String)) throws -> Void
  ) throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
      UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let input = try XCTUnwrap(tmpfile())
    let output = try XCTUnwrap(tmpfile())
    let error = try XCTUnwrap(tmpfile())
    defer {
      fclose(input)
      fclose(output)
      fclose(error)
      try? FileManager.default.removeItem(at: directory)
    }
    let context = DrawThingsCLIContext(
      input: input, output: output, error: error,
      environment: ["DRAWTHINGS_MODELS_DIR": "Models"], isStandardOutputTTY: false,
      resolvePath: { URL(fileURLWithPath: $0, relativeTo: directory).standardizedFileURL.path },
      resolveModelsDirectory: resolveModelsDirectory, unloadTextGenerator: unloadTextGenerator)
    func contents(_ stream: UnsafeMutablePointer<FILE>) -> String {
      fflush(stream)
      rewind(stream)
      var bytes = [UInt8](repeating: 0, count: 4096)
      var data = Data()
      while true {
        let count = fread(&bytes, 1, bytes.count, stream)
        if count == 0 { break }
        data.append(contentsOf: bytes.prefix(count))
      }
      return String(decoding: data, as: UTF8.self)
    }
    try body(context, directory, { (contents(output), contents(error)) })
  }

  func testHelpAndErrorsReturnWithoutExiting() throws {
    try withContext { context, _, contents in
      XCTAssertEqual(DrawThingsCLI.run(arguments: ["generate", "--help"], context: context), 0)
      XCTAssertTrue(contents().0.contains("--model"))
      XCTAssertTrue(contents().1.isEmpty)
      XCTAssertNotEqual(DrawThingsCLI.run(arguments: ["--not-a-flag"], context: context), 0)
      XCTAssertTrue(contents().1.contains("--not-a-flag"))
      XCTAssertEqual(DrawThingsCLI.run(arguments: ["models", "--help"], context: context), 0)
    }
  }

  func testCompletionUsesInvocationDirectoryAndMatchesStdout() throws {
    XCTAssertEqual(
      DrawThingsCLIContext.process().path("~/completion.sh"),
      URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("completion.sh")
        .standardizedFileURL.path)
    try withContext { context, directory, contents in
      XCTAssertEqual(DrawThingsCLI.run(arguments: ["completion", "bash"], context: context), 0)
      let expected = contents().0
      XCTAssertFalse(expected.isEmpty)
      XCTAssertEqual(
        DrawThingsCLI.run(
          arguments: ["completion", "bash", "--output", "completion.sh"], context: context), 0)
      XCTAssertEqual(
        try String(contentsOf: directory.appendingPathComponent("completion.sh"), encoding: .utf8),
        expected)
    }
  }

  func testEnvironmentAndModelSettingsAreInvocationScoped() throws {
    try withContext { context, directory, contents in
      let originalURLs = ModelZoo.externalUrls
      let originalPreference = ModelZoo.isExternalUrlsPreferred
      XCTAssertEqual(
        DrawThingsCLI.run(arguments: ["models", "list", "--offline"], context: context), 0)
      XCTAssertTrue(
        FileManager.default.fileExists(atPath: directory.appendingPathComponent("Models").path))
      XCTAssertTrue(contents().0.contains("Models directory:"))
      XCTAssertEqual(ModelZoo.externalUrls, originalURLs)
      XCTAssertEqual(ModelZoo.isExternalUrlsPreferred, originalPreference)
      XCTAssertEqual(DrawThingsCLI.run(arguments: ["--help"], context: context), 0)
      XCTAssertFalse(context.offline)
    }
  }

  func testCapturedInputAndCancellation() throws {
    try withContext { context, _, _ in
      fputs("a prompt with 🐈\n", context.input)
      fflush(context.input)
      rewind(context.input)
      XCTAssertEqual(String(decoding: try context.readInput(), as: UTF8.self), "a prompt with 🐈\n")
      context.cancel()
      XCTAssertEqual(DrawThingsCLI.run(arguments: ["--help"], context: context), 130)
      XCTAssertThrowsError(try context.readInput())
    }
  }

  func testHostCallbacksAreLazyAndInvocationScoped() throws {
    var resolutions = 0
    var unloads = 0
    try withContext(
      resolveModelsDirectory: { requested, completion in
        resolutions += 1
        completion(.success(requested!.appendingPathComponent("Authorized")))
      }, unloadTextGenerator: { unloads += 1 }
    ) { context, directory, contents in
      XCTAssertEqual(DrawThingsCLI.run(arguments: ["--help"], context: context), 0)
      XCTAssertEqual(resolutions, 0)
      XCTAssertEqual(unloads, 0)
      for invocation in 1...2 {
        XCTAssertEqual(
          DrawThingsCLI.run(arguments: ["models", "list", "--offline"], context: context), 0)
        XCTAssertEqual(resolutions, invocation)
        XCTAssertEqual(unloads, 0)
      }
      XCTAssertTrue(
        contents().0.contains(directory.appendingPathComponent("Models/Authorized").path))
      try context.prepareForLocalModelExecution()
      try context.prepareForLocalModelExecution()
      XCTAssertEqual(unloads, 1)
      context.finishInvocation()
      try context.prepareForLocalModelExecution()
      XCTAssertEqual(unloads, 2)
      context.finishInvocation()
    }
  }

  func testPermissionCancellationDoesNotUnload() throws {
    enum PermissionError: Error { case denied }
    try withContext(
      resolveModelsDirectory: { _, completion in completion(.failure(PermissionError.denied)) },
      unloadTextGenerator: { XCTFail("Permission failure must not unload the text generator") }
    ) { context, _, _ in
      XCTAssertNotEqual(
        DrawThingsCLI.run(arguments: ["models", "list", "--offline"], context: context), 0)
    }
    var pendingCompletion: ((Result<URL, Error>) -> Void)?
    try withContext(resolveModelsDirectory: { _, completion in pendingCompletion = completion }) {
      context, directory, _ in
      DispatchQueue.global().asyncAfter(deadline: .now() + 0.1) { context.cancel() }
      XCTAssertEqual(
        DrawThingsCLI.run(arguments: ["models", "list", "--offline"], context: context), 130)
      // A dismissed command may still receive the picker result; it owns no
      // folder access and must not resume execution or touch closed streams.
      pendingCompletion?(.success(directory))
    }
  }
}
