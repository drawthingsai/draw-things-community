import Foundation
import GRPC
import GRPCModels
import NIO
import NIOSSL

public final class ImageGenerationClientWrapper {
  public enum Error: Swift.Error {
    case invalidRootCA
  }

  private var eventLoopGroup: EventLoopGroup? = nil
  private var channel: GRPCChannel? = nil
  public private(set) var client: ImageGenerationServiceNIOClient? = nil

  public init() {}

  public func connect(host: String, port: Int, TLS: Bool) throws {
    try? eventLoopGroup?.syncShutdownGracefully()
    let eventLoopGroup = PlatformSupport.makeEventLoopGroup(loopCount: 1)
    let transportSecurity: GRPCChannelPool.Configuration.TransportSecurity
    if TLS {
      guard let rootCA = Bundle.main.path(forResource: "root_ca", ofType: "crt") else {
        throw ImageGenerationClientWrapper.Error.invalidRootCA
      }
      let certificate = try NIOSSLCertificate(file: rootCA, format: .pem)
      transportSecurity = .tls(
        .makeClientConfigurationBackedByNIOSSL(
          trustRoots: .certificates([certificate]), certificateVerification: .noHostnameVerification
        ))
    } else {
      transportSecurity = .plaintext
    }
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

  public typealias FileExistsCall = UnaryCall<FileListRequest, FileExistenceResponse>

  public func filesExists(
    files: [String], callback: @escaping (Bool, [(String, Bool)]) -> Void
  ) -> UnaryCall<FileListRequest, FileExistenceResponse>? {
    guard let client = client else {
      callback(false, [])
      return nil
    }

    var request = FileListRequest()
    request.files = files
    let call = client.filesExist(request)
    let _ = call.response.always { result in
      switch result {
      case .success(let response):
        callback(true, Array(zip(response.files, response.existences)))
      case .failure(_):
        callback(false, [])
      }
    }
    return call
  }
}
