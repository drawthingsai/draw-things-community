import Foundation

#if canImport(Darwin)
  import Darwin
#else
  import Glibc
#endif

/// The host owns these streams until `DrawThingsCLI.run` returns. Callbacks use
/// the captured streams, including when generation runs on another thread.
public final class DrawThingsCLIContext {
  public enum Output {
    case standardOutput
    case standardError
  }

  public let input: UnsafeMutablePointer<FILE>
  public let output: UnsafeMutablePointer<FILE>
  public let error: UnsafeMutablePointer<FILE>
  public let environment: [String: String]
  public let executablePath: String?
  public let isStandardOutputTTY: Bool
  private let resolvePath: (String) -> String
  private let cancellationRequested: () -> Bool
  private let resolveModelsDirectory: ((URL?, @escaping (Result<URL, Error>) -> Void) -> Void)?
  private let unloadTextGenerator: (() -> Void)?
  private var modelsDirectory: URL?
  private var isAccessingModelsDirectory = false
  private var didUnloadTextGenerator = false
  private let cancellationLock = NSLock()
  private var cancellation: (() -> Void)?
  private var cancelled = false
  var offline = false

  public init(
    input: UnsafeMutablePointer<FILE>, output: UnsafeMutablePointer<FILE>,
    error: UnsafeMutablePointer<FILE>, environment: [String: String],
    executablePath: String? = nil, isStandardOutputTTY: Bool,
    resolvePath: @escaping (String) -> String,
    cancellationRequested: @escaping () -> Bool = { false },
    resolveModelsDirectory: ((URL?, @escaping (Result<URL, Error>) -> Void) -> Void)? = nil,
    unloadTextGenerator: (() -> Void)? = nil
  ) {
    self.input = input
    self.output = output
    self.error = error
    self.environment = environment
    self.executablePath = executablePath
    self.isStandardOutputTTY = isStandardOutputTTY
    self.resolvePath = resolvePath
    self.cancellationRequested = cancellationRequested
    self.resolveModelsDirectory = resolveModelsDirectory
    self.unloadTextGenerator = unloadTextGenerator
  }

  public static func process() -> DrawThingsCLIContext {
    let directory = URL(
      fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
    return DrawThingsCLIContext(
      input: stdin, output: stdout, error: stderr,
      environment: ProcessInfo.processInfo.environment,
      executablePath: CommandLine.arguments.first,
      isStandardOutputTTY: isatty(STDOUT_FILENO) != 0,
      resolvePath: {
        URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath, relativeTo: directory)
          .standardizedFileURL.path
      })
  }

  var isCancelled: Bool {
    cancellationLock.lock()
    let value = cancelled
    cancellationLock.unlock()
    return value || cancellationRequested()
  }

  public func cancel() {
    cancellationLock.lock()
    if cancelled {
      cancellationLock.unlock()
      return
    }
    cancelled = true
    let cancellation = self.cancellation
    cancellationLock.unlock()
    cancellation?()
  }

  func setCancellation(_ cancellation: (() -> Void)?) {
    cancellationLock.lock()
    self.cancellation = cancellation
    let cancelled = self.cancelled
    cancellationLock.unlock()
    if cancelled { cancellation?() }
  }

  func checkCancellation() throws {
    if isCancelled { throw DrawThingsCLIInvocationError.cancelled }
  }

  func path(_ value: String) -> String { resolvePath(value) }

  func resolveModelsDirectoryAccess(_ requested: URL?) throws -> URL? {
    guard let resolveModelsDirectory else { return requested }
    if let modelsDirectory { return modelsDirectory }
    try checkCancellation()
    // Folder permission is asynchronous UI. Wait only on the command thread;
    // cancellation can return even if the picker is still being presented.
    let condition = NSCondition()
    var result: Result<URL, Error>?
    resolveModelsDirectory(requested) { value in
      condition.lock()
      result = value
      condition.broadcast()
      condition.unlock()
    }
    condition.lock()
    while result == nil && !isCancelled {
      condition.wait(until: Date().addingTimeInterval(0.1))
    }
    let resolved = result
    condition.unlock()
    try checkCancellation()
    guard let resolved else { throw DrawThingsCLIInvocationError.cancelled }
    let url = try resolved.get()
    #if canImport(Darwin)
      isAccessingModelsDirectory = url.startAccessingSecurityScopedResource()
    #endif
    modelsDirectory = url
    return url
  }

  func prepareForLocalModelExecution() throws {
    try checkCancellation()
    guard !didUnloadTextGenerator else { return }
    unloadTextGenerator?()
    didUnloadTextGenerator = true
    try checkCancellation()
  }

  func finishInvocation() {
    #if canImport(Darwin)
      if isAccessingModelsDirectory { modelsDirectory?.stopAccessingSecurityScopedResource() }
    #endif
    modelsDirectory = nil
    isAccessingModelsDirectory = false
    didUnloadTextGenerator = false
  }

  func write(_ value: String, to destination: Output = .standardOutput) {
    let stream = destination == .standardOutput ? output : error
    let data = Data(value.utf8)
    data.withUnsafeBytes { bytes in
      if let address = bytes.baseAddress { _ = fwrite(address, 1, bytes.count, stream) }
    }
    fflush(stream)
  }

  func print(_ items: Any..., separator: String = " ", terminator: String = "\n") {
    write(items.map { String(describing: $0) }.joined(separator: separator) + terminator)
  }

  func readInput() throws -> Data {
    var data = Data()
    var bytes = [UInt8](repeating: 0, count: 4096)
    while true {
      try checkCancellation()
      var descriptor = pollfd(fd: fileno(input), events: Int16(POLLIN), revents: 0)
      let ready = poll(&descriptor, 1, 100)
      if ready == 0 { continue }
      if ready < 0 {
        if errno == EINTR { continue }
        throw DrawThingsCLIInvocationError.inputFailed
      }
      let count = read(fileno(input), &bytes, bytes.count)
      if count == 0 { return data }
      if count < 0 {
        if errno == EINTR { continue }
        throw DrawThingsCLIInvocationError.inputFailed
      }
      data.append(contentsOf: bytes.prefix(count))
    }
  }
}

enum DrawThingsCLIInvocationError: Error, LocalizedError {
  case cancelled
  case inputFailed

  var errorDescription: String? {
    switch self {
    case .cancelled: return "Cancelled."
    case .inputFailed: return "Unable to read standard input."
    }
  }
}
