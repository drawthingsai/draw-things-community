import BashToolContext
import Darwin
import DrawThingsCLILib
import Foundation
import ios_system

public func draw_things_main(
  _ argc: Int32, _ argv: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
  // Like Z3 and Lean, unwind native and Swift objects before Darwin is allowed
  // to cancel this thread. Generation polls the session cancellation context.
  var previous: Int32 = 0
  pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &previous)
  let status: Int32 = autoreleasepool {
    guard let argv, argc > 0 else { return 1 }
    let arguments = (1..<Int(argc)).compactMap { argv[$0].map { String(cString: $0) } }
    var environment = [String: String]()
    for entry in environmentAsArray() as? [String] ?? [] {
      guard let separator = entry.firstIndex(of: "=") else { continue }
      environment[String(entry[..<separator])] = String(entry[entry.index(after: separator)...])
    }
    let directory =
      ios_getenv("PWD").map { String(cString: $0) }
      ?? FileManager.default.currentDirectoryPath
    let aliases = ["/home", "/tmp", "/app_bundle"].map { alias -> (String, String) in
      var buffer = [CChar](repeating: 0, count: Int(PATH_MAX))
      let resolved = ios_resolveDirectoryAlias(alias, &buffer, buffer.count)
      return (alias, resolved > 0 ? String(cString: buffer) : alias)
    }
    let cancellation = ios_getCommandCancellationContext()
    let output = ios_stdout() ?? stdout
    let configuration = BashToolContext.current
    let context = DrawThingsCLIContext(
      input: ios_stdin() ?? stdin, output: output, error: ios_stderr() ?? stderr,
      environment: environment, isStandardOutputTTY: ios_isatty(fileno(output)) != 0,
      resolvePath: { value in
        var value = value
        if value == "~" || value.hasPrefix("~/") {
          value = (environment["HOME"] ?? directory) + value.dropFirst()
        }
        for (alias, target) in aliases where value == alias || value.hasPrefix(alias + "/") {
          value = target + value.dropFirst(alias.count)
          break
        }
        return URL(
          fileURLWithPath: value, relativeTo: URL(fileURLWithPath: directory, isDirectory: true)
        ).standardizedFileURL.path
      }, cancellationRequested: { ios_commandCancellationRequested(cancellation) != 0 },
      resolveModelsDirectory: configuration?.resloveDrawThingsModelsDirectory,
      unloadTextGenerator: configuration?.unloadTextGenerator)
    // ios_system exposes cancellation by polling. Keep that polling at the host
    // boundary so the synchronous generation utilities need only a callback.
    let cancellationQueue = DispatchQueue(label: "com.drawthings.cli.cancellation")
    let timer = DispatchSource.makeTimerSource(queue: cancellationQueue)
    timer.schedule(deadline: .now(), repeating: .milliseconds(100))
    timer.setEventHandler {
      if ios_commandCancellationRequested(cancellation) != 0 { context.cancel() }
    }
    timer.resume()
    defer {
      timer.cancel()
      cancellationQueue.sync {}
    }
    return DrawThingsCLI.run(arguments: arguments, context: context)
  }
  pthread_setcancelstate(previous, nil)
  return status
}
