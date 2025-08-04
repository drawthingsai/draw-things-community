import BinaryResources
import Foundation
import GRPC
import GRPCImageServiceModels
import Logging
import NIO
import NIOSSL

public final class ProxyGPUClientWrapper {
  public private(set) var deviceName: String? = nil
  private var eventLoopGroup: EventLoopGroup? = nil
  private var channel: GRPCChannel? = nil
  public private(set) var client: ImageGenerationServiceNIOClient? = nil
  private let logger = Logger(label: "com.draw-things.image-generation-proxy-service")
  public init(deviceName: String? = nil) {
    self.deviceName = deviceName
  }

  public func connect(host: String, port: Int) throws {
    try? eventLoopGroup?.syncShutdownGracefully()
    let eventLoopGroup = PlatformSupport.makeEventLoopGroup(loopCount: 1)
    let transportSecurity: GRPCChannelPool.Configuration.TransportSecurity = .plaintext

    var configuration = GRPCChannelPool.Configuration.with(
      target: .host(host, port: port),
      transportSecurity: transportSecurity,
      eventLoopGroup: eventLoopGroup)
    configuration.maximumReceiveMessageLength = 1024 * 1024 * 1024
    let channel = try GRPCChannelPool.with(configuration: configuration)
    let client = ImageGenerationServiceNIOClient(channel: channel)
    self.eventLoopGroup = eventLoopGroup
    self.channel = channel
    self.client = client
  }

  public func disconnect() throws {
    try channel?.close().wait()
    try eventLoopGroup?.syncShutdownGracefully()
    client = nil
    channel = nil
    eventLoopGroup = nil
  }

  deinit {
    guard let eventLoopGroup = eventLoopGroup else { return }
    // Do async shutdown in deinit, make sure we don't hold any of self reference.
    let _ = channel?.close().always { _ in
      eventLoopGroup.shutdownGracefully { _ in }
    }
  }

  public func echo(timeout: TimeInterval = 60.0, attempts: Int = 3) async -> (Bool, [String]) {
    guard let client = client else {
      return (false, [])
    }

    var request = EchoRequest()
    let name = deviceName ?? ""
    request.name = "Proxy Server connect \(name)"

    // Create call options with timeout
    let callOptions = CallOptions(
      timeLimit: .timeout(.seconds(Int64(timeout)))
    )

    // Retry logic
    for attempt in 1...attempts {
      do {
        if attempt > 1 {
          logger.info("Echo retry attempt \(attempt) for device: \(String(describing: deviceName))")
        }

        let result = try await client.echo(request, callOptions: callOptions).response.get()

        if attempt > 1 {
          logger.info(
            "Echo succeeded on attempt \(attempt) for device: \(String(describing: deviceName))")
        }

        return (true, result.files)

      } catch let error as GRPCStatus {
        logger.error(
          "DeviceName \(String(describing: deviceName)) Echo failed with gRPC status: \(error.code) - \(error.message ?? "")"
        )

        if attempt < attempts {
          // Exponential backoff with jitter
          let baseDelay = 2.0
          let delay = baseDelay * pow(2.0, Double(attempt - 1))
          let jitter = Double.random(in: 0...0.3) * delay
          let totalDelay = min(delay + jitter, 10.0)  // Cap at 10 seconds

          logger.warning(
            "Echo failed on attempt \(attempt) for device: \(String(describing: deviceName)), retrying in \(totalDelay) seconds"
          )

          try? await Task.sleep(nanoseconds: UInt64(totalDelay * 1_000_000_000))
        } else {
          logger.error(
            "Echo failed after \(attempts) attempts for device: \(String(describing: deviceName))")
        }

      } catch {
        logger.error(
          "DeviceName \(String(describing: deviceName)) Echo failed with error: \(error)")

        if attempt < attempts {
          // Same retry logic
          let baseDelay = 2.0
          let delay = baseDelay * pow(2.0, Double(attempt - 1))
          let jitter = Double.random(in: 0...0.3) * delay
          let totalDelay = min(delay + jitter, 10.0)  // Cap at 10 seconds

          logger.warning(
            "Echo failed on attempt \(attempt) for device: \(String(describing: deviceName)), retrying in \(totalDelay) seconds"
          )

          try? await Task.sleep(nanoseconds: UInt64(totalDelay * 1_000_000_000))
        } else {
          logger.error(
            "Echo failed after \(attempts) attempts for device: \(String(describing: deviceName))")
        }
      }
    }

    return (false, [])
  }

}
