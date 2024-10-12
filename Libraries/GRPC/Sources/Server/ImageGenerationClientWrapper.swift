import GRPC
import GRPCModels
import NIO

public final class ImageGenerationClientWrapper {
  private var eventLoopGroup: EventLoopGroup? = nil
  private var channel: GRPCChannel? = nil
  public private(set) var client: ImageGenerationServiceNIOClient? = nil

  public init() {}

  public func connect(host: String, port: Int) throws {
    try? eventLoopGroup?.syncShutdownGracefully()
    let eventLoopGroup = PlatformSupport.makeEventLoopGroup(loopCount: 1)
    var configuration = GRPCChannelPool.Configuration.with(
      target: .host(host, port: port),
      transportSecurity: .plaintext,
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

  public func echo(callback: @escaping (Bool) -> Void) {
    guard let client = client else {
      callback(false)
      return
    }

    var request = EchoRequest()
    request.name = ""
    let _ = client.echo(request).response.always {
      switch $0 {
      case .success(_):
        callback(true)
      case .failure(_):
        callback(false)
      }
    }
  }

}
