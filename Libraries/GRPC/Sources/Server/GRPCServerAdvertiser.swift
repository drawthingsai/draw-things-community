import Foundation
import Network

public class GRPCServerAdvertiser {
  private var netService: NetService?
  private let objCResponder: ObjCResponder
  private let name: String

  final class ObjCResponder: NSObject, NetServiceDelegate {
    weak var advertiser: GRPCServerAdvertiser? = nil
    public func netServiceDidPublish(_ sender: NetService) {
      advertiser?.netServiceDidPublish(sender)
    }
    public func netService(_ sender: NetService, didNotPublish errorDict: [String: NSNumber]) {
      advertiser?.netService(sender, didNotPublish: errorDict)
    }
  }

  public init(name: String) {
    self.name = name
    objCResponder = ObjCResponder()
    objCResponder.advertiser = self
  }

  public func startAdvertising(port: Int32) {
    let serviceType = "_dt-grpc._tcp."
    let serviceDomain = "local."

    let netService = NetService(domain: serviceDomain, type: serviceType, name: name, port: port)
    self.netService = netService
    netService.delegate = objCResponder
    netService.publish()  // Removed the options parameter
  }

  public func stopAdvertising() {
    netService?.stop()
    netService = nil
  }

}

// NetServiceDelegate methods
extension GRPCServerAdvertiser {
  public func netServiceDidPublish(_ sender: NetService) {
    print(
      "Service published: domain=\(sender.domain) type=\(sender.type) name=\(sender.name) port=\(sender.port)"
    )
  }

  public func netService(_ sender: NetService, didNotPublish errorDict: [String: NSNumber]) {
    print("Failed to publish service: \(errorDict)")
  }
}
