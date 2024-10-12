import Foundation

public protocol GRPCServiceBrowserDelegate: AnyObject {
  func didFindService(_: GRPCServiceBrowser.ServiceDescriptor)
  func didRemoveService(_: GRPCServiceBrowser.ServiceDescriptor)
}

public class GRPCServiceBrowser {
  public struct ServiceDescriptor: Equatable & Hashable {
    public let name: String
    public let host: String
    public let port: Int
    public init(name: String, host: String, port: Int) {
      self.name = name
      self.host = host
      self.port = port
    }
  }

  private var serviceBrowser: NetServiceBrowser
  private var discoveredServices = [NetService]()
  private var discoveredDescriptors = [ServiceDescriptor]()
  public weak var delegate: GRPCServiceBrowserDelegate?

  private let objCResponder: ObjCResponder

  final class ObjCResponder: NSObject, NetServiceBrowserDelegate, NetServiceDelegate {
    weak var serviceBrowser: GRPCServiceBrowser? = nil
    public func netServiceBrowser(
      _ browser: NetServiceBrowser, didFind service: NetService, moreComing: Bool
    ) {
      serviceBrowser?.netServiceBrowser(browser, didFind: service, moreComing: moreComing)
    }
    public func netServiceBrowser(
      _ browser: NetServiceBrowser, didRemove service: NetService, moreComing: Bool
    ) {
      serviceBrowser?.netServiceBrowser(browser, didRemove: service, moreComing: moreComing)
    }
    public func netService(_ sender: NetService, didNotResolve errorDict: [String: NSNumber]) {
      serviceBrowser?.netService(sender, didNotResolve: errorDict)
    }
    public func netServiceDidResolveAddress(_ sender: NetService) {
      serviceBrowser?.netServiceDidResolveAddress(sender)
    }
  }
  public init() {
    serviceBrowser = NetServiceBrowser()
    objCResponder = ObjCResponder()
    objCResponder.serviceBrowser = self
    serviceBrowser.delegate = objCResponder
    serviceBrowser.searchForServices(ofType: "_dt-grpc._tcp.", inDomain: "local.")
  }

  // NetServiceBrowserDelegate method
  public func netServiceBrowser(
    _ browser: NetServiceBrowser, didFind service: NetService, moreComing: Bool
  ) {
    dispatchPrecondition(condition: .onQueue(.main))
    discoveredServices.append(service)
    discoveredDescriptors.append(
      ServiceDescriptor(name: service.name, host: service.hostName ?? "", port: service.port))
    service.delegate = objCResponder
    service.resolve(withTimeout: 5.0)
  }

  public func netServiceBrowser(
    _ browser: NetServiceBrowser, didRemove service: NetService, moreComing: Bool
  ) {
    dispatchPrecondition(condition: .onQueue(.main))
    guard let firstIndex = discoveredServices.firstIndex(where: { $0 == service }) else {
      return
    }
    let descriptor = discoveredDescriptors[firstIndex]
    discoveredServices.remove(at: firstIndex)
    discoveredDescriptors.remove(at: firstIndex)
    delegate?.didRemoveService(descriptor)
  }

  // NetServiceDelegate method
  public func netServiceDidResolveAddress(_ sender: NetService) {
    dispatchPrecondition(condition: .onQueue(.main))
    if let addresses = sender.addresses, !addresses.isEmpty {
      if let hostname = sender.hostName, sender.port != 0 {
        let descriptor = ServiceDescriptor(name: sender.name, host: hostname, port: sender.port)
        if let firstIndex = discoveredServices.firstIndex(where: { $0 == sender }) {
          discoveredDescriptors[firstIndex] = descriptor  // Update the descriptor.
        }
        delegate?.didFindService(descriptor)
      }
    }
  }

  public func netService(_ sender: NetService, didNotResolve errorDict: [String: NSNumber]) {
    dispatchPrecondition(condition: .onQueue(.main))
    print("Failed to resolve service: \(errorDict)")
  }
}
