import DataModels
import Diffusion
import Foundation
import JavaScriptCore
import NNC
import Utils
import Vision

public func jsonObject(from object: Encodable) throws -> Any {
  let data = try JSONEncoder().encode(object)
  return try JSONSerialization.jsonObject(with: data)
}

public func object<T: Decodable>(jsonObject: Any, type: T.Type) throws -> T {
  let data = try JSONSerialization.data(withJSONObject: jsonObject)
  return try JSONDecoder().decode(type, from: data)
}

public enum OpenPurpose: String {
  case image
  case mask
  case depthMap
  case extractDepthMap
  case scribble
  case pose
  case color
  case custom
  case shuffle
  case addImage
}

public enum ModelType: String {
  case LoRA = "LoRA"
  case ControlNet = "Control Net"
  case notAvailable = "N/A"
}

public enum SupportedPrefix: String {
  case base64Png = "data:image/png;base64,"
  case base64Jpg = "data:image/jpeg;base64,"
  case base64Webp = "data:image/webp;base64,"
  case file = "file://"
  case notAvailable = "N/A"
}

extension SupportedPrefix {
  public static func supportedPrefix(from srcContent: String) -> (SupportedPrefix, String) {
    if srcContent.hasPrefix(SupportedPrefix.base64Png.rawValue) {
      return (
        .base64Png,
        srcContent.replacingOccurrences(of: SupportedPrefix.base64Png.rawValue, with: "")
      )
    }
    if srcContent.hasPrefix(SupportedPrefix.base64Jpg.rawValue) {
      return (
        .base64Jpg,
        srcContent.replacingOccurrences(of: SupportedPrefix.base64Jpg.rawValue, with: "")
      )
    }
    if srcContent.hasPrefix(SupportedPrefix.base64Webp.rawValue) {
      return (
        .base64Webp,
        srcContent.replacingOccurrences(of: SupportedPrefix.base64Webp.rawValue, with: "")
      )
    }
    if srcContent.hasPrefix(SupportedPrefix.file.rawValue) {
      return (.file, srcContent.replacingOccurrences(of: SupportedPrefix.file.rawValue, with: ""))
    }
    return (.notAvailable, "")
  }
}

@objc protocol JSInterop: JSExport {
  func log(_ message: String, _ type: Int)
  func generateImage(_ args: [String: Any])
  func createControl(_ name: String) -> [String: Any]?
  func createLoRA(_ name: String) -> [String: Any]?
  func fillMaskRectangle(
    _ maskDictionary: [String: Any], _ rect: CGRect, _ value: UInt8)
  func moveCanvas(_ x: Double, _ y: Double)
  func updateCanvasSize(_ configurationDictionary: [String: Any])
  func createMask(_ width: Double, _ height: Double, _ value: UInt8) -> Int
  func createForegroundMask() -> Int
  func createBackgroundMask() -> Int
  func setCanvasZoom(_ zoomScale: Double)
  func canvasZoom() -> Double
  func existingConfiguration() -> [String: Any]
  func existingPrompts() -> [String: String]
  func currentMask() -> [String: Any]?
  func topLeftCorner() -> [String: Any]
  func boundingBox() -> CGRect
  func clearCanvas()
  func listFilesWithinDirectory(_ directory: String) -> [String]
  func listFilesUnderPicturesWithinDirectory(_ directory: String) -> [String]
  func listFilesUnderPictures() -> [String]
  func picturesPath() -> String?
  func loadImageFileToCanvas(_ file: String)
  func saveImageFileFromCanvas(_ file: String, _ visibleRegionOnly: Bool)
  func saveImageSrcFromCanvas(_ visibleRegionOnly: Bool) -> String?
  func loadLayerFromPhotos(_ type: String)
  func loadLayerFromFiles(_ type: String)
  func loadLayerFromSrc(_ srcContent: String, _ type: String)
  func loadLayerFromJson(_ json: String, _ type: String)
  func detectFaces() -> [[String: Any]]
  func detectHands() -> [[String: Any]]
  func downloadBuiltins(_ files: [String])
  func requestFromUser(_ title: String, _ confirm: String, _ config: [[String: Any]]) -> [Any]
}

public protocol ScriptExecutorDelegate: AnyObject {
  func createControl(_ name: String) throws -> Control
  func createLoRA(_ name: String) throws -> LoRA
  func topLeftCorner() -> CGPoint
  func currentMask() -> Tensor<UInt8>?
  func foregroundMask() -> Tensor<UInt8>?
  func backgroundMask() -> Tensor<UInt8>?
  func generateImage(
    prompt: String?, negativePrompt: String?, configuration: GenerationConfiguration,
    mask: Tensor<UInt8>?
  ) -> Bool
  func setZoomScale(_ zoomScale: CGFloat)
  func zoomScale() -> CGFloat
  func updateCanvasSize(_ configuration: GenerationConfiguration)
  func moveCanvas(x: Int, y: Int)
  func existingConfiguration() -> GenerationConfiguration
  func existingPrompts() -> (prompt: String, negativePrompt: String)
  func boundingBox() -> CGRect
  func log(_ message: String, type: Int)
  func logException(_ message: String, line: Int, stack: String)
  func evaluateScriptBegan()
  func evaluateScriptEnded()
  func clearCanvas()
  func loadImageFileToCanvas(_ file: String)
  func saveImageFileFromCanvas(_ file: String, _ visibleRegionOnly: Bool)
  func saveImageDataFromCanvas(_ visibleRegionOnly: Bool) -> Data?
  func loadLayerFromPhotos(type: String) throws
  func loadLayerFromFiles(type: String) throws
  func loadLayerFromSrc(_ srcContent: String, type: String) throws
  func loadLayerFromJson(_ json: String, type: String) throws
  func detectFaces() throws -> [CGRect]
  func detectHands() throws -> [CGRect]
  func downloadBuiltins(_ files: [String]) throws
  func requestFromUser(title: String, confirm: String, _ config: [[String: Any]]) throws -> (
    lifetimeObjects: [AnyObject], result: [Any]
  )
}

@objc public final class ScriptExecutor: NSObject {
  private static let queue = DispatchQueue(label: "com.draw-things.script", qos: .userInteractive)
  var hasExecuted = false
  var hasCancelled = false
  var lifetimeObjects = [AnyObject]()
  let maskManager = MaskManager()
  weak var delegate: ScriptExecutorDelegate?
  let script: String?
  var context: JSContext?

  public init(script: ScriptZoo.Script, delegate: ScriptExecutorDelegate) {
    if let filePath = script.filePath {
      self.script = ScriptZoo.contentOf(filePath)
    } else {
      self.script = ""
    }
    self.delegate = delegate
  }
}

extension ScriptExecutor: JSInterop {
  public enum CancellationError: Error {
    case cancelled
  }
  func forwardExceptionsToJS<T: DefaultCreatable>(_ block: () throws -> T) -> T {
    do {
      return try block()
    } catch CancellationError.cancelled {
      context?.exception = JSValue(newErrorFromMessage: "cancelled", in: context)
      hasCancelled = true
      return T.defaultInstance()
    } catch let error {
      context?.exception = JSValue(newErrorFromMessage: "\(error)", in: context)
      return T.defaultInstance()
    }
  }

  func forwardExceptionsToJS(_ block: () throws -> Void) {
    let _ = forwardExceptionsToJS {
      try block()
      return 0
    }
  }

  func updateCanvasSize(_ configurationDictionary: [String: Any]) {
    forwardExceptionsToJS {
      let configuration = try object(
        jsonObject: configurationDictionary, type: JSGenerationConfiguration.self
      ).createGenerationConfiguration()
      delegate?.updateCanvasSize(configuration)
    }
  }

  func fixDimensionIfNecessary(_ dimension: inout UInt32) {
    if dimension % 64 != 0 {
      log(
        "All dimensions (width, height, hi-res fix width, hi-res fix height must be divisible by 64. Rounding \(dimension) down to \(dimension - dimension % 64)..."
      )
      dimension -= dimension % 64
    }
  }

  func generateImage(_ args: [String: Any]) {  // configurationDictionary: [String: Any], _ maskDictionary: [String: Any]?) {
    forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      let mask: Tensor<UInt8>?
      if let maskDictionary = args["mask"] as? [String: Any] {
        let jsMask = try object(jsonObject: maskDictionary, type: JSMask.self)
        guard let storedMask = maskManager.mask(forJSMask: jsMask) else {
          throw "Unrecognized handle: \(jsMask.handle)"
        }
        mask = storedMask
      } else {
        mask = nil
      }
      let jsConfiguration = try! object(
        jsonObject: args["configuration"] as! [String: Any], type: JSGenerationConfiguration.self)

      fixDimensionIfNecessary(&jsConfiguration.width)
      fixDimensionIfNecessary(&jsConfiguration.height)
      fixDimensionIfNecessary(&jsConfiguration.hiresFixWidth)
      fixDimensionIfNecessary(&jsConfiguration.hiresFixHeight)
      fixDimensionIfNecessary(&jsConfiguration.decodingTileWidth)
      fixDimensionIfNecessary(&jsConfiguration.decodingTileHeight)
      fixDimensionIfNecessary(&jsConfiguration.decodingTileOverlap)
      let configuration = jsConfiguration.createGenerationConfiguration()
      let prompt = (args["prompt"] as? NSString) as? String
      let negativePrompt = (args["negativePrompt"] as? NSString) as? String
      let notAborted = delegate.generateImage(
        prompt: prompt, negativePrompt: negativePrompt, configuration: configuration, mask: mask)
      if !notAborted {
        throw CancellationError.cancelled
      }
    }
  }

  func setCanvasZoom(_ zoomScale: Double) {
    if (zoomScale * 60).rounded() != zoomScale * 60 {
      log("Warning: setCanvasZoom parameter should be divisible by 60")
    }
    delegate?.setZoomScale(CGFloat(zoomScale))
  }

  func canvasZoom() -> Double {
    return (delegate?.zoomScale()).map({ Double($0) }) ?? 1
  }

  func moveCanvas(_ x: Double, _ y: Double) {
    if x != x.rounded() || y != y.rounded() {
      log("Warning: x and y coordinates to moveCanvas should be integers")
    }
    delegate?.moveCanvas(x: Int(x), y: Int(y))
  }

  func fillMaskRectangle(
    _ maskDictionary: [String: Any], _ rect: CGRect, _ value: UInt8
  ) {
    forwardExceptionsToJS {
      let jsMask = try object(jsonObject: maskDictionary, type: JSMask.self)
      guard var mask = maskManager.mask(forJSMask: jsMask) else {
        throw "Unrecognized handle: \(jsMask.handle)"
      }
      let xStart = Int(rect.origin.x)
      let yStart = Int(rect.origin.y)
      let yEnd = yStart + Int(rect.size.height)
      let xEnd = xStart + Int(rect.size.width)
      guard yEnd <= mask.shape[0] else {
        throw "Mask y larger than mask dimension: y \(yEnd), dimension: \(mask.shape[0])"
      }
      guard xEnd <= mask.shape[1] else {
        throw "Mask x larger than mask dimension: y \(xEnd), dimension: \(mask.shape[1])"
      }
      guard xStart >= 0 else {
        throw "Mask x must be greater than or equal to 0: \(xStart)"
      }
      guard yStart >= 0 else {
        throw "Mask y must be greater than or equal to 0: \(yStart))"
      }
      for y in yStart..<yEnd {
        for x in xStart..<xEnd {
          mask[y, x] = value
        }
      }
      maskManager.setMask(mask, forJSMask: jsMask)
    }
  }

  // console.log is variadic, but we only handle the first string
  func log(_ message: String, _ type: Int) {
    print("JavaScriptCore: \(message)")
    return forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      delegate.log(message, type: type)
    }
  }

  func log(_ message: String) {
    log(message, 0)
  }

  func createLoRA(_ name: String) -> [String: Any]? {
    return forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      let lora = try delegate.createLoRA(name)
      let jsLoRA = JSLoRA(lora: lora)
      let data = try JSONEncoder().encode(jsLoRA)
      return ((try JSONSerialization.jsonObject(with: data)) as! [String: Any])
    }
  }

  func clearCanvas() {
    return forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      delegate.clearCanvas()
    }
  }

  func createControl(_ name: String) -> [String: Any]? {
    return forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      let control = try delegate.createControl(name)
      let jsControl = JSControl(control: control)
      let data = try JSONEncoder().encode(jsControl)
      return ((try JSONSerialization.jsonObject(with: data)) as! [String: Any])
    }
  }

  func createMask(_ width: Double, _ height: Double, _ value: UInt8) -> Int {
    return forwardExceptionsToJS {
      if width != width.rounded() || height != height.rounded() {
        throw "Width and height must be integers"
      }
      var mask = Tensor<UInt8>(.CPU, .NC(Int(height), Int(width)))
      for y in 0..<Int(height) {
        for x in 0..<Int(width) {
          mask[y, x] = value
        }
      }
      let jsMask = maskManager.createNewMask()
      maskManager.setMask(mask, forJSMask: jsMask)
      // Just return the handle and not the whole mask object, because in JS,
      // `new Mask(handle)` is different from `{"handle": handle}` (with the latter, `mask.func(...)` won't work)
      return jsMask.handle
    }
  }

  func createForegroundMask() -> Int {
    return forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      guard let foregroundMask = delegate.foregroundMask() else { return 0 }
      let jsMask = maskManager.createNewMask()
      maskManager.setMask(foregroundMask, forJSMask: jsMask)
      return jsMask.handle
    }
  }

  func createBackgroundMask() -> Int {
    return forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      guard let backgroundMask = delegate.backgroundMask() else { return 0 }
      let jsMask = maskManager.createNewMask()
      maskManager.setMask(backgroundMask, forJSMask: jsMask)
      return jsMask.handle
    }
  }

  func topLeftCorner() -> [String: Any] {
    return forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      let point = delegate.topLeftCorner()
      let jsPoint = JSPoint(point: point)
      return try jsonObject(from: jsPoint) as! [String: Any]
    }
  }

  func boundingBox() -> CGRect {
    guard let delegate = delegate else { return .zero }
    return delegate.boundingBox()
  }

  func currentMask() -> [String: Any]? {
    return forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      guard let mask = delegate.currentMask() else {
        return nil
      }
      let jsMask = maskManager.createNewMask()
      maskManager.setMask(mask, forJSMask: jsMask)
      return try jsonObject(from: jsMask) as! [String: Any]?
    }
  }

  func handleException(value: JSValue?) {
    let message = value?.objectForKeyedSubscript("message")?.toString() ?? ""
    let lineNumber = Int(value?.objectForKeyedSubscript("line")?.toInt32() ?? -1)
    let stackString = value?.objectForKeyedSubscript("stack")?.toString() ?? ""
    delegate?.logException(message, line: lineNumber, stack: stackString)
  }

  public func run() {
    Self.queue.async {
      if self.hasExecuted {
        // It's one-shot, e.g. to avoid having to manually let go of all masks stored in maskManager
        fatalError(
          "This object is one-shot, create another instance if you need to run execute() again")
      }
      guard let script = self.script else { return }
      self.context = JSContext()
      self.hasExecuted = true

      self.context?.exceptionHandler = { [weak self] context, value in
        guard let self = self else { return }
        // If it is not cancellation error, we handle it.
        guard !self.hasCancelled else { return }
        self.handleException(value: value)
      }
      self.context?.setObject(self, forKeyedSubscript: "__dtHooks" as NSCopying & NSObjectProtocol)
      // It's good to evaluate these as two separate scripts, rather than pasting them together into one script,
      // so that stack trace line numbers are accurate for the user for errors in their code
      self.delegate?.evaluateScriptBegan()
      self.context?.evaluateScript(SharedScript)
      self.context?.evaluateScript(script)
      self.context = nil  // In case this will help prevent a retain cycle
      self.lifetimeObjects.removeAll()
      self.delegate?.evaluateScriptEnded()
      // TODO: prevent network and file access
    }
  }

  // TODO: should we add "Copy" to the end of this method so people know that mutating it won't change the existing one?
  func existingConfiguration() -> [String: Any] {
    return forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      let configuration = delegate.existingConfiguration()
      let jsConfiguration = JSGenerationConfiguration(configuration: configuration)
      return try jsonObject(from: jsConfiguration) as! [String: Any]
    }
  }

  public static func ensurePicturesDirectoryExists() {
    #if targetEnvironment(macCatalyst)
      return
    #else
      let fileManager = FileManager.default
      guard let documentUrl = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first
      else { return }
      let systemPicturesUrl = documentUrl.appendingPathComponent("Pictures")
      try? FileManager.default.createDirectory(
        at: systemPicturesUrl, withIntermediateDirectories: true)
    #endif
  }

  func listFilesWithinDirectory(_ directory: String) -> [String] {
    return forwardExceptionsToJS { () -> [String] in
      let fileManager = FileManager.default
      let directoryUrl = URL(fileURLWithPath: directory)
      let fileUrls = try fileManager.contentsOfDirectory(
        at: directoryUrl, includingPropertiesForKeys: nil)
      return fileUrls.map { $0.path }
    }
  }

  func listFilesUnderPicturesWithinDirectory(_ directory: String) -> [String] {
    return forwardExceptionsToJS { () -> [String] in
      let fileManager = FileManager.default
      #if targetEnvironment(macCatalyst)
        guard
          let systemPicturesUrl = fileManager.urls(for: .picturesDirectory, in: .userDomainMask)
            .first
        else { return [] }
      #else
        guard let documentUrl = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first
        else { return [] }
        let systemPicturesUrl = documentUrl.appendingPathComponent("Pictures")
      #endif
      let directoryUrl =
        directory.isEmpty ? systemPicturesUrl : systemPicturesUrl.appendingPathComponent(directory)
      let fileUrls = try fileManager.contentsOfDirectory(
        at: directoryUrl, includingPropertiesForKeys: nil)
      return fileUrls.map { $0.path }
    }
  }

  func listFilesUnderPictures() -> [String] {
    return forwardExceptionsToJS { () -> [String] in
      let fileManager = FileManager.default
      #if targetEnvironment(macCatalyst)
        guard
          let systemPicturesUrl = fileManager.urls(for: .picturesDirectory, in: .userDomainMask)
            .first
        else { return [] }
      #else
        guard let documentUrl = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first
        else { return [] }
        let systemPicturesUrl = documentUrl.appendingPathComponent("Pictures")
      #endif
      let fileUrls = try fileManager.contentsOfDirectory(
        at: systemPicturesUrl, includingPropertiesForKeys: nil)
      return fileUrls.map { $0.path }
    }
  }

  func picturesPath() -> String? {
    return forwardExceptionsToJS {
      let fileManager = FileManager.default
      #if targetEnvironment(macCatalyst)
        guard
          let systemPicturesUrl = fileManager.urls(for: .picturesDirectory, in: .userDomainMask)
            .first
        else { return nil }
      #else
        guard let documentUrl = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first
        else { return nil }
        let systemPicturesUrl = documentUrl.appendingPathComponent("Pictures")
      #endif
      return systemPicturesUrl.path
    }
  }

  func loadImageFileToCanvas(_ file: String) {
    forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      delegate.loadImageFileToCanvas(file)
    }
  }

  func saveImageFileFromCanvas(_ file: String, _ visibleRegionOnly: Bool) {
    forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      delegate.saveImageFileFromCanvas(file, visibleRegionOnly)
    }
  }

  func saveImageSrcFromCanvas(_ visibleRegionOnly: Bool) -> String? {
    return forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      guard let data = delegate.saveImageDataFromCanvas(visibleRegionOnly) else { return nil }
      return SupportedPrefix.base64Png.rawValue + data.base64EncodedString()
    }
  }

  func existingPrompts() -> [String: String] {
    return forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      let prompts = delegate.existingPrompts()
      return ["prompt": prompts.prompt, "negativePrompt": prompts.negativePrompt]
    }
  }

  func loadLayerFromPhotos(_ type: String) {
    forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      try delegate.loadLayerFromPhotos(type: type)
    }
  }

  func loadLayerFromFiles(_ type: String) {
    forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      try delegate.loadLayerFromFiles(type: type)
    }
  }

  func loadLayerFromSrc(_ srcContent: String, _ type: String) {
    forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      try delegate.loadLayerFromSrc(srcContent, type: type)
    }
  }

  func loadLayerFromJson(_ json: String, _ type: String) {
    forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      try delegate.loadLayerFromJson(json, type: type)
    }
  }

  func downloadBuiltins(_ files: [String]) {
    return forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      try delegate.downloadBuiltins(files)
    }
  }

  func detectFaces() -> [[String: Any]] {
    return forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      let rects = try delegate.detectFaces().map { JSRect(rect: $0) }
      let data = try JSONEncoder().encode(["faces": rects])
      let jsonObject = try JSONSerialization.jsonObject(with: data) as! [String: Any]
      return jsonObject["faces"] as! [[String: Any]]
    }
  }

  func detectHands() -> [[String: Any]] {
    return forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      let rects = try delegate.detectHands().map { JSRect(rect: $0) }
      let data = try JSONEncoder().encode(["hands": rects])
      let jsonObject = try JSONSerialization.jsonObject(with: data) as! [String: Any]
      return jsonObject["hands"] as! [[String: Any]]
    }
  }

  func requestFromUser(_ title: String, _ confirm: String, _ config: [[String: Any]]) -> [Any] {
    return forwardExceptionsToJS {
      guard let delegate = delegate else { throw "No delegate" }
      let (lifetimeObjects, result) = try delegate.requestFromUser(
        title: title, confirm: confirm, config)
      self.lifetimeObjects.append(contentsOf: lifetimeObjects)
      // Do a round-trip copy so it is "clean" JSON object to JS execution engine.
      let data = try JSONSerialization.data(withJSONObject: result)
      let jsonObject = try JSONSerialization.jsonObject(with: data) as! [Any]
      return jsonObject
    }
  }
}
