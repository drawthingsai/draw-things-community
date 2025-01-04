import BinaryResources
import Foundation
import GRPC
import GRPCControlPanelModels
import NIO
import NIOSSL

public final class ProxyControlClient {
  public private(set) var deviceName: String? = nil
  private var eventLoopGroup: EventLoopGroup? = nil
  private var channel: GRPCChannel? = nil
  public private(set) var client: ControlPanelServiceNIOClient? = nil

  public init(deviceName: String? = nil) {
    self.deviceName = deviceName
  }

  public func connect(host: String, port: Int) throws {
    print("connect to proxy server \(host):\(port)")
    try? eventLoopGroup?.syncShutdownGracefully()
    let eventLoopGroup = PlatformSupport.makeEventLoopGroup(loopCount: 1)

    var configuration = GRPCChannelPool.Configuration.with(
      target: .host(host, port: port),
      transportSecurity: .plaintext,
      eventLoopGroup: eventLoopGroup)
    configuration.maximumReceiveMessageLength = 1024 * 1024 * 1024
    let channel = try GRPCChannelPool.with(configuration: configuration)
    let client = ControlPanelServiceNIOClient(channel: channel)
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

  public func addGPUServer(
    address: String, port: Int, isHighPriority: Bool, completion: @escaping (Bool) -> Void
  ) {
    guard let client = client else {
      print("addGPUServer can not connect to proxy server")
      completion(false)
      return
    }

    var request = GPUServerRequest()
    request.serverConfig.address = address
    request.serverConfig.port = Int32(port)
    request.serverConfig.isHighPriority = isHighPriority
    request.operation = .add

    let _ = client.manageGPUServer(request).response.always {
      switch $0 {
      case .success(_):
        print("added GPU Server \(address):\(port) to Proxy Server")
        completion(true)
      case .failure(_):
        print("can not add GPU Server \(address):\(port) to Proxy Server")
        completion(false)
      }
    }

  }

  public func removeGPUServer(address: String, port: Int, completion: @escaping (Bool) -> Void) {
    guard let client = client else {
      print("removeGPUServer can not connect to proxy server")
      completion(false)
      return
    }

    var request = GPUServerRequest()
    request.serverConfig.address = address
    request.serverConfig.port = Int32(port)
    request.operation = .remove

    let _ = client.manageGPUServer(request).response.always {
      switch $0 {
      case .success(_):
        print("remove GPU Server \(address):\(port) from Proxy Server")
        completion(true)
      case .failure(_):
        print("can not remove GPU Server \(address):\(port) from Proxy Server")
        completion(false)
      }
    }
  }

  public func updateThrottlingPolicy(policies: [String: Int], completion: @escaping (Bool) -> Void)
  {
    guard let client = client else {
      print("updateThrottlingPolicy can not connect to proxy server")
      completion(false)
      return
    }

    var request = ThrottlingRequest()
    request.limitConfig = policies.mapValues { Int32($0) }

    let _ = client.updateThrottlingConfig(request).response.always {
      switch $0 {
      case .success(_):
        print("update ThrottlingConfig succees")
        completion(true)

      case .failure(_):
        print("can not update ThrottlingConfig succees")
        completion(false)
      }
    }
  }

  public func updatePem(completion: @escaping (Bool) -> Void) {
    guard let client = client else {
      print("Update PEM can not connect to proxy server")
      completion(false)
      return
    }

    var request = UpdatePemRequest()

    let _ = client.updatePem(request).response.always {
      switch $0 {
      case .success(let response):
        print("\(response.message)")
        completion(true)

      case .failure(_):
        print("can not update PEM succees on Server")
        completion(false)
      }
    }
  }

  public func updateModelList(address: String, port: Int, completion: @escaping (Bool) -> Void) {
    guard let client = client else {
      print("can not connect to proxy server")
      completion(false)
      return
    }

    var request = UpdateModelListRequest()
    request.address = address
    request.port = Int32(port)
    let _ = client.updateModelList(request).response.always {
      switch $0 {
      case .success(let response):
        print(
          "update Model List success, updated :\(response.files.count) models in Proxy model-list file"
        )
        completion(true)

      case .failure(_):
        print("can not update Model List")
        completion(false)
      }
    }
  }
}
