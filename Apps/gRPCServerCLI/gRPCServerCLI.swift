import ArgumentParser
import BinaryResources
import DataDogLog
import DataModels
import Dflat
import Diffusion
import Foundation
import GRPC
import GRPCControlPanelModels
import GRPCImageServiceModels
import GRPCServer
import ImageGenerator
import LocalImageGenerator
import Logging
import ModelZoo
import NIO
import NIOSSL
import NNC
import ProxyControlClient
import SQLiteDflat
import Utils

private func createTemporaryDirectory() -> String {
  let fileManager = FileManager.default
  let tempDirURL = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(
    UUID().uuidString)
  try! fileManager.createDirectory(
    at: tempDirURL, withIntermediateDirectories: true, attributes: nil)
  return tempDirURL.path
}

private func createLocalImageGenerator(queue: DispatchQueue) -> LocalImageGenerator {
  let tempDir = createTemporaryDirectory()
  let workspace = SQLiteWorkspace(
    filePath: "\(tempDir)/config.sqlite3", fileProtectionLevel: .noProtection)
  // Just have an empty query result for now
  let configurations = workspace.fetch(for: GenerationConfiguration.self).where(
    GenerationConfiguration.id == 0, limit: .limit(0))

  let tokenizerV1 = TextualInversionAttentionCLIPTokenizer(
    vocabulary: BinaryResources.vocab_json,
    merges: BinaryResources.merges_txt,
    textualInversions: [])
  let tokenizerV2 = TextualInversionAttentionCLIPTokenizer(
    vocabulary: BinaryResources.vocab_16e6_json,
    merges: BinaryResources.bpe_simple_vocab_16e6_txt,
    textualInversions: [])
  let tokenizerKandinsky = SentencePieceTokenizer(
    data: BinaryResources.xlmroberta_bpe_model, startToken: 0,
    endToken: 2, tokenShift: 1)

  let tokenizerXL = tokenizerV2

  let tokenizerT5 = SentencePieceTokenizer(
    data: BinaryResources.t5_spiece_model, startToken: nil,
    endToken: 1, tokenShift: 0)
  let tokenizerPileT5 = SentencePieceTokenizer(
    data: BinaryResources.pile_t5_spiece_model, startToken: nil,
    endToken: 2, tokenShift: 0)
  let tokenizerChatGLM3 = SentencePieceTokenizer(
    data: BinaryResources.chatglm3_spiece_model, startToken: nil,
    endToken: nil, tokenShift: 0)
  let tokenizerLlama3 = GPT2Tokenizer(
    vocabulary: BinaryResources.vocab_llama3_json, merges: BinaryResources.merges_llama3_txt,
    specialTokens: [
      "<|start_header_id|>": 128006, "<|end_header_id|>": 128007, "<|eot_id|>": 128009,
      "<|begin_of_text|>": 128000, "<|end_of_text|>": 128001,
    ], unknownToken: "<|end_of_text|>", startToken: "<|begin_of_text|>",
    endToken: "<|end_of_text|>")

  return LocalImageGenerator(
    queue: queue, configurations: configurations, workspace: workspace, tokenizerV1: tokenizerV1,
    tokenizerV2: tokenizerV2, tokenizerXL: tokenizerXL, tokenizerKandinsky: tokenizerKandinsky,
    tokenizerT5: tokenizerT5, tokenizerPileT5: tokenizerPileT5,
    tokenizerChatGLM3: tokenizerChatGLM3, tokenizerLlama3: tokenizerLlama3)
}

#if os(Linux)
  private func localhost() -> String? {
    let maxLength = Int(HOST_NAME_MAX) + 1
    var buffer = [CChar](repeating: 0, count: maxLength)

    if gethostname(&buffer, maxLength) == 0 {
      return String(cString: buffer)
    } else {
      return nil
    }
  }

  private func checkWIFEXITED(_ status: Int32) -> Bool {
    return (status & 0x7f) == 0
  }

  private func getWEXITSTATUS(_ status: Int32) -> Int32 {
    return (status >> 8) & 0xff
  }

  private func checkWIFSIGNALED(_ status: Int32) -> Bool {
    return ((status & 0x7f) + 1) >> 1 > 0
  }

  private func getWTERMSIG(_ status: Int32) -> Int32 {
    return status & 0x7f
  }

  private func supervise(childWork: () -> Void) {
    var restartCount = 0
    while true {
      let pid = fork()
      guard pid != 0 else {
        childWork()
        return
      }
      guard pid > 0 else {
        print("Fork failed")
        return
      }
      // Parent process
      var status: Int32 = 0
      waitpid(pid, &status, 0)

      if checkWIFEXITED(status) {
        let exitStatus = getWEXITSTATUS(status)
        if exitStatus == 0 {
          print("Child process exited normally")
          return
        }
        print("Child process exited with status: \(exitStatus)")
      } else if checkWIFSIGNALED(status) {
        print("Child process terminated by signal: \(getWTERMSIG(status))")
      }
      restartCount += 1
      print("Restarting...  (Attempt \(restartCount) restarting")
    }
  }
#endif

enum CLIError: Error {
  case invalidModelPath
}

// Server info structure matching ServerArgu
struct ServerInfo: Codable {
  let address: String
  let port: Int
  let priority: Int
}

// Main configuration structure
struct AddServersConfiguration: Codable {
  var host: String
  var port: Int
  // Server management
  var servers: [ServerInfo]
}

extension AddServersConfiguration {
  static func parse(_ jsonString: String) throws -> AddServersConfiguration {
    guard let jsonData = jsonString.data(using: .utf8) else {
      throw ConfigurationError.invalidJSON
    }
    let decoder = JSONDecoder()
    return try decoder.decode(AddServersConfiguration.self, from: jsonData)
  }

}

enum ConfigurationError: Error {
  case invalidJSON
  case encodingError
}

@main
struct gRPCServerCLI: ParsableCommand {
  static let configuration: CommandConfiguration = CommandConfiguration(
    commandName: "gRPCServerCLI")
  @Argument(help: "The directory path to load models.")
  var modelsDirectory: String

  @Option(name: .shortAndLong, help: "The name you exposes in the local network.")
  var name: String = {
    #if os(Linux)
      if let localhost = localhost() {
        return localhost
      }
    #else
      if let handle = dlopen(
        "/System/Library/Frameworks/SystemConfiguration.framework/SystemConfiguration", RTLD_LAZY)
      {
        defer {
          dlclose(handle)
        }
        if let sym = dlsym(handle, "SCDynamicStoreCopyComputerName") {
          let SCDynamicStoreCopyComputerName = unsafeBitCast(
            sym,
            to: (@convention(c) (OpaquePointer?, UnsafeMutablePointer<Unmanaged<CFString>?>?) ->
              Unmanaged<CFString>?).self)
          let name = SCDynamicStoreCopyComputerName(nil, nil)?.takeRetainedValue() as String? ?? ""
          return name
        }
      }
    #endif
    return "Unknown Computer"
  }()

  @Option(name: .shortAndLong, help: "The address in your local network that you want to expose.")
  var address: String = "0.0.0.0"

  @Option(name: .shortAndLong, help: "The port in your local network that you want to expose.")
  var port: Int = 7859

  @Option(name: .shortAndLong, help: "The GPU to use for image generation.")
  var gpu: Int = 0

  @Option(name: .shortAndLong, help: "Use Datadog as the logging backend.")
  var datadogAPIKey: String = ""

  @Option(name: .shortAndLong, help: "The shared secret that can help to secure the server.")
  var sharedSecret: String = ""

  @Flag(help: "Disable TLS for the connection.")
  var noTLS = false

  @Flag(help: "Disable response compression.")
  var noResponseCompression = false

  @Flag(help: "Enable model browsing.")
  var modelBrowser = false

  @Flag(help: "Disable FlashAttention.")
  var noFlashAttention = false

  @Flag(help: "Debug flag for the verbose model inference logging.")
  var debug = false

  #if os(Linux)
    @Flag(help: "Supervise the server so it restarts upon a internal crash.")
    var supervised = false

  #endif

  @Option(
    name: .long,
    help:
      "Inline JSON configuration string, example  --join '{\"host\":\"proxy ip\" , \"port\": proxy port, \"servers\": [{\"address\":\"gpu ip\", \"port\":gpu port, \"priority\":1 1 = high, 2 = low}]}' "
  )
  var join: String?

  mutating func run() throws {
    #if os(Linux)
      if supervised {
        // Run in supervised mode
        supervise {
          try? startGRPCServer()
        }
      } else {
        try startGRPCServer()
      }
    #else
      try startGRPCServer()
    #endif
  }

  func startGRPCServer() throws {
    if debug {
      DynamicGraph.logLevel = .verbose
    }

    let fileManager = FileManager.default
    var isDirectory: ObjCBool = false
    let exists = fileManager.fileExists(
      atPath: URL(fileURLWithPath: modelsDirectory).path, isDirectory: &isDirectory)

    if exists && isDirectory.boolValue {
    } else {
      throw CLIError.invalidModelPath
    }

    if !datadogAPIKey.isEmpty {
      let gpu = gpu
      let datadogAPIKey = datadogAPIKey
      let name = name

      LoggingSystem.bootstrap { label in
        // Create a multiplexing log handler that will send logs to multiple destinations
        var handlers: [LogHandler] = []

        // Always add the console log handler for local visibility
        handlers.append(StreamLogHandler.standardOutput(label: label))

        // If Datadog API key is provided, add the Datadog handler
        var datadogHandler = DataDogLogHandler(
          label: label,
          key: datadogAPIKey,
          hostname: "\(name) GPU \(gpu)",
          region: .US5
        )
        datadogHandler.metadata = ["gpu": "\(gpu)"]
        handlers.append(datadogHandler)

        // Return a multiplexing handler that writes to all configured handlers
        return MultiplexLogHandler(handlers)
      }
      // LoggingSystem.bootstrap {
      //   var handler = DataDogLogHandler(
      //     label: $0, key: datadogAPIKey, hostname: "\(name) GPU \(gpu)", region: .US5)
      //   handler.metadata = ["gpu": "\(gpu)"]
      //   return handler
      // }
    }

    ModelZoo.externalUrl = URL(fileURLWithPath: modelsDirectory)
    if noFlashAttention {
      DeviceCapability.isMFAEnabled.store(false, ordering: .releasing)
    }
    if gpu > 0 && gpu < DeviceKind.GPUs.count {
      DeviceKind.GPUs.permute(gpu)
    }
    try self.runAndBlock(name: name, address: address, port: port, TLS: !noTLS)
  }

  func runAndBlock(name: String, address: String, port: Int, TLS: Bool) throws {

    let group = MultiThreadedEventLoopGroup(numberOfThreads: 1)
    defer {
      try! group.syncShutdownGracefully()
    }
    let queue = DispatchQueue(label: "com.draw-things.edit", qos: .userInteractive)
    let localImageGenerator = createLocalImageGenerator(queue: queue)
    let imageGenerationServiceImpl = ImageGenerationServiceImpl(
      imageGenerator: localImageGenerator, queue: queue, backupQueue: queue)
    if noResponseCompression {
      imageGenerationServiceImpl.responseCompression.store(false, ordering: .releasing)
    } else {
      imageGenerationServiceImpl.responseCompression.store(true, ordering: .releasing)
    }
    if modelBrowser {
      imageGenerationServiceImpl.enableModelBrowsing.store(true, ordering: .releasing)
    } else {
      imageGenerationServiceImpl.enableModelBrowsing.store(false, ordering: .releasing)
    }
    imageGenerationServiceImpl.sharedSecret = sharedSecret.isEmpty ? nil : sharedSecret

    // Bind the server and get an `EventLoopFuture<Server>`
    let serverFuture: EventLoopFuture<GRPC.Server>
    let certificate = try? NIOSSLCertificate(
      bytes: [UInt8](BinaryResources.server_crt_crt), format: .pem)
    let privateKey = try? NIOSSLPrivateKey(
      bytes: [UInt8](BinaryResources.server_key_key), format: .pem)
    var TLS = TLS
    if TLS, let certificate = certificate, let privateKey = privateKey {
      serverFuture = GRPC.Server.usingTLS(
        with: GRPCTLSConfiguration.makeServerConfigurationBackedByNIOSSL(
          certificateChain: [.certificate(certificate)], privateKey: .privateKey(privateKey)),
        on: group
      )
      .withServiceProviders([imageGenerationServiceImpl])
      .withMaximumReceiveMessageLength(1024 * 1024 * 1024)
      .bind(host: address, port: port)
    } else {
      serverFuture = GRPC.Server.insecure(group: group)
        .withServiceProviders([imageGenerationServiceImpl])
        .withMaximumReceiveMessageLength(1024 * 1024 * 1024)
        .bind(host: address, port: port)
      TLS = false
    }

    let server: Server = try serverFuture.wait()

    if let localAddress = server.channel.localAddress, let port = localAddress.port {
      print("Server started on local address: \(localAddress), port \(port)")
    } else {
      print("Server started, but port is unknown")
    }
    let advertiser: GRPCServerAdvertiser = GRPCServerAdvertiser(name: name)
    advertiser.startAdvertising(port: Int32(port), TLS: TLS)

    if let json = join {
      try addServers(try AddServersConfiguration.parse(json))
    }

    // Block the current thread until the server closes
    try server.onClose.wait()
    advertiser.stopAdvertising()
  }

  func addServers(_ addServers: AddServersConfiguration) throws {
    guard !addServers.servers.isEmpty else {
      return
    }
    let client = ProxyControlClient()
    try client.connect(host: addServers.host, port: addServers.port)
    let group = DispatchGroup()  // TODO: I would prefer not use DispatchGroup here.
    for server in addServers.servers {
      group.enter()
      client.addGPUServer(
        address: server.address, port: server.port,
        isHighPriority: server.priority == 1
      ) {
        _ in
        group.leave()
      }
    }
    group.wait()
  }
}
