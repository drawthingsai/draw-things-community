import Foundation
import Logging
import XCTest
import UniformTypeIdentifiers

@testable import MediaGenerationKit

final class MediaGenerationKitExampleCompileTests: XCTestCase {
  func testCurrentPipelineExampleSurfaceCompilesAndBasicSetupWorks() throws {
    let modelsDirectory = try makeTemporaryDirectory()
    let outputDirectory = try makeTemporaryDirectory()
    let onePixelPNGData = Data(base64Encoded: Self.onePixelPNGBase64)!

    var pipeline = try MediaGenerationPipeline.fromPretrained(
      "Stable Diffusion v1.5",
      backend: .local(directory: modelsDirectory.path)
    )

    pipeline.configuration.width = 768
    pipeline.configuration.height = 768
    pipeline.configuration.seed = 42
    pipeline.configuration.steps = 28
    pipeline.configuration.guidanceScale = 7.5
    pipeline.configuration.strength = 0.75
    pipeline.logger = Logger(label: "com.draw-things.tests.example")

    let generationTask: Task<[MediaGenerationPipeline.Result], Error> = Task {
      try await pipeline.generate(
        prompt: "cinematic portrait, rain-soaked street, neon reflections",
        negativePrompt: "",
        inputs: [
          MediaGenerationPipeline.data(onePixelPNGData),
          MediaGenerationPipeline.data(onePixelPNGData).mask(),
          MediaGenerationPipeline.data(onePixelPNGData).moodboard(),
        ]
      ) { _ in
      }
    }
    generationTask.cancel()

    XCTAssertEqual(pipeline.model, "sd_v1.5_f16.ckpt")
    XCTAssertEqual(pipeline.configuration.model, "sd_v1.5_f16.ckpt")

    let result = try MediaGenerationPipeline.Result(encodedData: onePixelPNGData)
    let outputURL = outputDirectory.appendingPathComponent("output.png")
    try result.write(to: outputURL, type: .png)
    XCTAssertTrue(FileManager.default.fileExists(atPath: outputURL.path))

    let environment = MediaGenerationEnvironment.default
    environment.externalUrls = [modelsDirectory]
    let ensureOperation: () async throws -> MediaGenerationResolvedModel = {
      try await environment.ensure("Stable Diffusion v1.5", offline: true) { _ in
      }
    }
    _ = ensureOperation
  }

  func testCurrentLoRAExamplesCompileAndNonNetworkOperationsWork() async throws {
    let importerExample: (URL, URL) throws -> Void = { sourceFileURL, destinationFileURL in
      var importer = LoRAImporter(file: sourceFileURL)
      try importer.inspect()
      try importer.import(to: destinationFileURL, scaleFactor: 1.0) { _ in
      }
    }
    _ = importerExample

    let store = try LoRAStore(backend: .cloudCompute(apiKey: "API_KEY"))
    let listOperation: () async throws -> [LoRAStore.File] = {
      try await store.list()
    }
    _ = listOperation

    let uploadOperation: (Data, String) async throws -> LoRAStore.File = { data, file in
      try await store.upload(data, file: file)
    }
    _ = uploadOperation

    try await store.delete(keys: [])
  }

  private func makeTemporaryDirectory() throws -> URL {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
      "mediagenerationkit-example-tests-\(UUID().uuidString)",
      isDirectory: true
    )
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    addTeardownBlock {
      try? FileManager.default.removeItem(at: directory)
    }
    return directory
  }

  private static let onePixelPNGBase64 =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO2p2ioAAAAASUVORK5CYII="
}
