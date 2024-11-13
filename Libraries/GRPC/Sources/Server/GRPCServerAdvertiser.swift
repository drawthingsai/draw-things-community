import Foundation

#if canImport(Network)
  import Network
#endif
#if canImport(C_DNSAdvertiser)
  import C_DNSAdvertiser

  private func localhost() -> String? {
    let maxLength = Int(HOST_NAME_MAX) + 1
    var buffer = [CChar](repeating: 0, count: maxLength)

    if gethostname(&buffer, maxLength) == 0 {
      return String(cString: buffer)
    } else {
      return nil
    }
  }

  private func IP() -> String? {
    guard let string = get_current_ip_address() else {
      return nil
    }
    let result = String(cString: string)
    string.deallocate()
    return result
  }
#endif

public class GRPCServerAdvertiser {
  private let name: String

  #if canImport(C_DNSAdvertiser)
    private let queue: DispatchQueue
  #else
    private var netService: NetService?
    private let objCResponder: ObjCResponder

    final class ObjCResponder: NSObject, NetServiceDelegate {
      weak var advertiser: GRPCServerAdvertiser? = nil
      public func netServiceDidPublish(_ sender: NetService) {
        advertiser?.netServiceDidPublish(sender)
      }
      public func netService(_ sender: NetService, didNotPublish errorDict: [String: NSNumber]) {
        advertiser?.netService(sender, didNotPublish: errorDict)
      }
    }
  #endif

  public init(name: String) {
    self.name = name
    #if canImport(C_DNSAdvertiser)
      queue = DispatchQueue(label: "com.draw-things.advertise", qos: .default)
    #else
      objCResponder = ObjCResponder()
      objCResponder.advertiser = self
    #endif
  }

  public func startAdvertising(port: Int32, TLS: Bool) {
    let name = TLS ? "\(name)_tls" : "\(name)_notls"
    #if canImport(C_DNSAdvertiser)
      if let IP = IP(), let localhost = localhost() {
        let hostname = localhost + ".local"
        send_mdns_packet(hostname, IP)
        advertise(name, "_dt-grpc._tcp.local", hostname, UInt16(port), 120)
      }
    #else
      let serviceType = "_dt-grpc._tcp."
      let serviceDomain = "local."
      let netService = NetService(domain: serviceDomain, type: serviceType, name: name, port: port)
      self.netService = netService
      netService.delegate = objCResponder
      netService.publish()  // Removed the options parameter
    #endif
  }

  public func stopAdvertising() {
    #if canImport(C_DNSAdvertiser)
    #else
      netService?.stop()
      netService = nil
    #endif
  }

}

#if canImport(C_DNSAdvertiser)
#else
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
#endif
