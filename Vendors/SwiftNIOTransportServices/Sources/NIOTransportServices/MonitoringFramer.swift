#if canImport(Network)
  import Network
  import Security
  import Atomics

  public final class MonitoringFramer: NWProtocolFramerImplementation {
    public func wakeup(framer: NWProtocolFramer.Instance) {}
    public func stop(framer: NWProtocolFramer.Instance) -> Bool {
      return true
    }
    public func cleanup(framer: NWProtocolFramer.Instance) {}

    public struct Statistics {
      public var bytesSent: Int
      public var bytesReceived: Int
      public init(bytesSent: Int, bytesReceived: Int) {
        self.bytesSent = bytesSent
        self.bytesReceived = bytesReceived
      }
    }
    public static var statistics: Statistics {
      return Statistics(
        bytesSent: bytesSent.load(ordering: .acquiring),
        bytesReceived: bytesReceived.load(ordering: .acquiring))
    }
    private static let bytesSent = ManagedAtomic<Int>(0)
    private static let bytesReceived = ManagedAtomic<Int>(0)

    // Protocol definition
    static let definition = NWProtocolFramer.Definition(implementation: MonitoringFramer.self)

    // Required protocol identifier
    public static var label: String { return "MonitoringFramer" }

    // Each connection needs its own instance of the framer
    public required init(framer: NWProtocolFramer.Instance) {
      self.framerInstance = framer
    }

    private let framerInstance: NWProtocolFramer.Instance

    // Start point for monitoring outgoing messages
    public func start(framer: NWProtocolFramer.Instance) -> NWProtocolFramer.StartResult {
      return .ready
    }

    // Handle incoming messages
    public func handleInput(framer: NWProtocolFramer.Instance) -> Int {
      _ = framer.parseInput(minimumIncompleteLength: 1, maximumLength: 65535) {
        buffer, endOfMessage in
        if let buffer = buffer {
          _ = framer.deliverInputNoCopy(
            length: buffer.count, message: .init(instance: framer), isComplete: endOfMessage)
          logReceived(messageLength: buffer.count)
        }
        return 0
      }
      return 0
    }

    // Handle outgoing messages
    public func handleOutput(
      framer: NWProtocolFramer.Instance, message: NWProtocolFramer.Message, messageLength: Int,
      isComplete: Bool
    ) {
      // Get data the application wants to send
      try? framer.writeOutputNoCopy(length: messageLength)
      logSent(messageLength: messageLength)
    }

    // MARK: - Logging Methods

    private func logSent(messageLength: Int) {
      Self.bytesSent.wrappingIncrement(by: messageLength, ordering: .acquiringAndReleasing)
    }

    private func logReceived(messageLength: Int) {
      Self.bytesReceived.wrappingIncrement(by: messageLength, ordering: .acquiringAndReleasing)
    }
  }

#endif
