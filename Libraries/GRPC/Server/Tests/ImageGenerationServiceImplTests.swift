import Crypto
import DataModels
import Diffusion
import Foundation
import GRPC
import GRPCImageServiceModels
import ImageGenerator
import ModelZoo
import NIOCore
import NIOPosix
import NNC
import XCTest

@testable import GRPCServer

private struct UnusedImageGenerator: ImageGenerator {
  func generate(
    trace: ImageGeneratorTrace,
    image: Tensor<FloatType>?, scaleFactor: Int, mask: Tensor<UInt8>?,
    hints: [(ControlHintType, [(AnyTensor, Float)])], text: String, negativeText: String,
    configuration: GenerationConfiguration, fileMapping: [String: String], keywords: [String],
    cancellation: (@escaping () -> Void) -> Void,
    feedback: @escaping (ImageGeneratorSignpost, Set<ImageGeneratorSignpost>, Tensor<FloatType>?)
      -> Bool
  ) throws -> ([Tensor<FloatType>]?, [Tensor<Float>]?, Int) {
    return (nil, nil, 0)
  }
}

final class ImageGenerationServiceImplTests: XCTestCase {
  // rootDirectory/Models is the only models directory the server is given. Whatever else is under
  // rootDirectory is what a file name must not be able to reach.
  private var rootDirectory: URL!
  private var modelsDirectory: URL!
  private var externalUrls = [URL]()
  private var serverGroup: MultiThreadedEventLoopGroup!
  private var server: Server?
  private let client = ImageGenerationClientWrapper()
  // ModelZoo also looks into ~/Documents/Models, so use names nothing else can have.
  private let model = "\(UUID().uuidString.lowercased())_lora_f16.ckpt"

  override func setUpWithError() throws {
    rootDirectory = FileManager.default.temporaryDirectory.appendingPathComponent(
      UUID().uuidString)
    modelsDirectory = rootDirectory.appendingPathComponent("Models")
    try FileManager.default.createDirectory(at: modelsDirectory, withIntermediateDirectories: true)
    externalUrls = ModelZoo.externalUrls
    ModelZoo.externalUrls = [modelsDirectory]
    serverGroup = MultiThreadedEventLoopGroup(numberOfThreads: 1)
  }

  override func tearDownWithError() throws {
    try? client.disconnect()
    try server?.close().wait()
    try serverGroup.syncShutdownGracefully()
    ModelZoo.externalUrls = externalUrls
    try FileManager.default.removeItem(at: rootDirectory)
  }

  private func startServer(enableFileUpload: Bool = true) throws {
    let queue = DispatchQueue(label: "com.draw-things.image-generation-service-tests")
    let service = ImageGenerationServiceImpl(
      imageGenerator: UnusedImageGenerator(), queue: queue, backupQueue: queue,
      enableFileUpload: enableFileUpload)
    let server = try Server.insecure(group: serverGroup).withServiceProviders([service]).bind(
      host: "127.0.0.1", port: 0
    ).wait()
    self.server = server
    try client.connect(
      host: "127.0.0.1", port: try XCTUnwrap(server.channel.localAddress?.port), TLS: false,
      hostnameVerification: false, sharedSecret: nil)
  }

  // Sends what a client sends for one file, as is. GRPCFileUploader would never send most of the
  // names and sizes below.
  private func upload(
    _ content: Data, as filename: String, totalSize: Int64? = nil, sha256: Data? = nil,
    sendsEnd: Bool = true
  ) throws -> GRPCStatus.Code? {
    let call = try XCTUnwrap(client.client).uploadFile { _ in }
    call.sendMessage(
      .with {
        $0.initRequest = .with {
          $0.filename = filename
          $0.sha256 = sha256 ?? Data(SHA256.hash(data: content))
          $0.totalSize = totalSize ?? Int64(content.count)
        }
      }, promise: nil)
    call.sendMessage(
      .with {
        $0.chunk = .with {
          $0.content = content
          $0.offset = 0
        }
      }, promise: nil)
    if sendsEnd {
      call.sendEnd(promise: nil)
    }
    var code: GRPCStatus.Code? = nil
    let completed = expectation(description: "uploadFile")
    call.status.whenSuccess {
      code = $0.code
      completed.fulfill()
    }
    // Without the end, there is only a status if the server ended the call by itself.
    wait(for: [completed], timeout: sendsEnd ? 60 : 10)
    // The server removes its partial file when it fails the call. Let its event loop finish that
    // before anything looks at the directory.
    try serverGroup.next().submit {}.wait()
    return code
  }

  private func contents(of directory: URL) throws -> [String] {
    return try FileManager.default.contentsOfDirectory(atPath: directory.path).sorted()
  }

  func testFileNameValidation() {
    for name in [
      "foo_lora_f16.ckpt", "foo.ckpt-tensordata", "Foo-2.1_q8p.ckpt", "_foo_.ckpt",
      "072ef94e15252e963a0bc77702f8db329ef2ce0e2245ed487ee61aeca1cdb69d",
      String(repeating: "a", count: 255),
    ] {
      XCTAssertTrue(ImageGenerationServiceImpl.isValidFileName(name), name)
    }
    for name in [
      "", ".", "..", "../x", "a/../../b", "/etc/x", "a/b", "a\\b", "a\0b", "a.ckpt\0", ".hidden",
      "-rf", "a b.ckpt", "a\nb.ckpt", "é.ckpt", "a:b", "a*.ckpt",
      String(repeating: "a", count: 256), String(repeating: "a", count: 300),
    ] {
      XCTAssertFalse(ImageGenerationServiceImpl.isValidFileName(name), name.debugDescription)
    }
  }

  func testUploadRejectsNameOutsideOfModelsDirectory() throws {
    try startServer()
    for filename in [
      "../evil.ckpt", "a/../../evil.ckpt", "Models/../../evil.ckpt", "..\\evil.ckpt",
      rootDirectory.appendingPathComponent("evil.ckpt").path, "evil.ckpt/", "evil\0.ckpt", "..",
      ".", "",
    ] {
      XCTAssertEqual(
        try upload(Data("evil".utf8), as: filename), .invalidArgument, filename.debugDescription)
    }
    XCTAssertEqual(try contents(of: rootDirectory), ["Models"])
    XCTAssertEqual(try contents(of: modelsDirectory), [])
  }

  func testUploadRejectsFileThatIsNotModel() throws {
    try startServer()
    for filename in ["custom.json", "evil.sh", "evil", "\(model).part", ".\(model)"] {
      XCTAssertEqual(try upload(Data("evil".utf8), as: filename), .invalidArgument, filename)
    }
    XCTAssertEqual(try contents(of: modelsDirectory), [])
  }

  func testUploadAddsFile() throws {
    try startServer()
    // More than GRPCFileUploader sends in one chunk.
    let content = Data((0..<2_500_000).map { UInt8(truncatingIfNeeded: $0 &* 31) })
    let clientDirectory = rootDirectory.appendingPathComponent("Client")
    try FileManager.default.createDirectory(at: clientDirectory, withIntermediateDirectories: true)
    let filenames = [model, "\(model)-tensordata"]
    for filename in filenames {
      let fileUrl = clientDirectory.appendingPathComponent(filename)
      try content.write(to: fileUrl)
      let uploaded = expectation(description: filename)
      let _ = try GRPCFileUploader(client: client).resume(fileUrl: fileUrl) { _, _ in
      } completionHandler: { success in
        XCTAssertTrue(success)
        uploaded.fulfill()
      }
      wait(for: [uploaded], timeout: 60)
      XCTAssertEqual(
        try Data(contentsOf: modelsDirectory.appendingPathComponent(filename)), content)
    }
    XCTAssertEqual(try contents(of: modelsDirectory), filenames)
  }

  func testUploadReplacesExistingFile() throws {
    try startServer()
    let fileUrl = modelsDirectory.appendingPathComponent(model)
    try Data("existing".utf8).write(to: fileUrl)
    XCTAssertEqual(try upload(Data("uploaded".utf8), as: model), .ok)
    XCTAssertEqual(try Data(contentsOf: fileUrl), Data("uploaded".utf8))
    XCTAssertEqual(try contents(of: modelsDirectory), [model])
  }

  func testUploadDisabled() throws {
    try startServer(enableFileUpload: false)
    XCTAssertEqual(try upload(Data("uploaded".utf8), as: model), .unimplemented)
    XCTAssertEqual(try contents(of: modelsDirectory), [])
  }

  func testUploadRejectsUnexpectedFileSize() throws {
    try startServer()
    let content = Data("uploaded".utf8)
    // More than announced is refused when it arrives, not only at the end. Then less than
    // announced, and sizes nothing should announce.
    XCTAssertEqual(
      try upload(content, as: model, totalSize: 4, sendsEnd: false), .invalidArgument)
    XCTAssertEqual(try upload(content, as: model, totalSize: 16), .invalidArgument)
    XCTAssertEqual(try upload(content, as: model, totalSize: -1), .invalidArgument)
    XCTAssertEqual(try upload(content, as: model, totalSize: .max), .invalidArgument)
    XCTAssertEqual(try contents(of: modelsDirectory), [])
  }

  func testUploadRejectsHashMismatch() throws {
    try startServer()
    XCTAssertEqual(
      try upload(Data("uploaded".utf8), as: model, sha256: Data(repeating: 0, count: 32)),
      .dataLoss)
    XCTAssertEqual(try contents(of: modelsDirectory), [])
  }

  func testFilesExistOnlyAnswersForModelsDirectory() throws {
    try startServer(enableFileUpload: false)
    try Data("model".utf8).write(to: modelsDirectory.appendingPathComponent(model))
    let outsideUrl = rootDirectory.appendingPathComponent("outside.ckpt")
    try Data("outside".utf8).write(to: outsideUrl)
    let files = [model, "../outside.ckpt", outsideUrl.path, "missing_\(model)"]
    var results = [(String, Bool, Data)]()
    let answered = expectation(description: "filesExist")
    let _ = client.filesExists(files: files, filesToMatch: files) { success, payload in
      XCTAssertTrue(success)
      results = payload
      answered.fulfill()
    }
    wait(for: [answered], timeout: 60)
    XCTAssertEqual(results.map { $0.0 }, files)
    XCTAssertEqual(results.map { $0.1 }, [true, false, false, false])
    XCTAssertEqual(
      results.map { $0.2 }, [Data(SHA256.hash(data: Data("model".utf8))), Data(), Data(), Data()])
  }
}
