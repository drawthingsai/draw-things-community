import Foundation
import Upscaler

public struct UpscalerZoo: DownloadZoo {
  public struct Specification {
    public var name: String
    public var file: String
    public var scaleFactor: UpscaleFactor = .x4
    public var blocks: Int = 23
    public init(name: String, file: String, scaleFactor: UpscaleFactor = .x4, blocks: Int = 23) {
      self.name = name
      self.file = file
      self.scaleFactor = scaleFactor
      self.blocks = blocks
    }
  }

  private static let fileSHA256: [String: String] = [
    "realesrgan_x2plus_f16.ckpt":
      "98ce77870b5ca059ec004fe8572182dc67ac8d6a2bba8a938df0ba44fbaccc66",
    "realesrgan_x4plus_f16.ckpt":
      "3db00086d999e590e313dbf45f0701cdf0e3bca3a66a201a3078423501cb58fd",
    "realesrgan_x4plus_anime_6b_f16.ckpt":
      "3ad598b21e888590d1bd239dc55675de11b245c691728b56859aa05038c69099",
    "esrgan_4x_universal_upscaler_v2_sharp_f16.ckpt":
      "05a94d4b3c165f58915f5fafba31512ef5f393011450a40e4437c01d2e33c080",
    "remacri_4x_f16.ckpt":
      "88d7ae8ecce57de2ad3cb67bdee9937ea320fbaa6319b0d7eb78ea1730b70671",
    "4x_ultrasharp_f16.ckpt":
      "c8e9a1ee8bf5bc71cef7204bf1cf8cb120dc8b578189d33fd94025a6cfa9f0ec",
  ]

  static let builtinSpecifications: [Specification] = [
    Specification(name: "Real-ESRGAN X2+", file: "realesrgan_x2plus_f16.ckpt", scaleFactor: .x2),
    Specification(name: "Real-ESRGAN X4+", file: "realesrgan_x4plus_f16.ckpt"),
    Specification(
      name: "Real-ESRGAN X4+ Anime", file: "realesrgan_x4plus_anime_6b_f16.ckpt", blocks: 6),
    Specification(
      name: "UniversalUpscaler V2 Sharp", file: "esrgan_4x_universal_upscaler_v2_sharp_f16.ckpt"),
    Specification(
      name: "Remacri", file: "remacri_4x_f16.ckpt"),
    Specification(
      name: "4x UltraSharp", file: "4x_ultrasharp_f16.ckpt"),
  ]

  public static var availableSpecifications: [Specification] { builtinSpecifications }

  private static var specificationMapping: [String: Specification] = {
    var mapping = [String: Specification]()
    for specification in availableSpecifications {
      mapping[specification.file] = specification
    }
    return mapping
  }()

  public static func specificationForModel(_ name: String) -> Specification? {
    return specificationMapping[name]
  }

  public static func filePathForModelDownloaded(_ name: String) -> String {
    return ModelZoo.filePathForModelDownloaded(name)
  }

  public static func isModelDownloaded(_ name: String) -> Bool {
    return ModelZoo.isModelDownloaded(name)
  }

  public static func humanReadableNameForModel(_ name: String) -> String {
    guard let specification = specificationForModel(name) else { return name }
    return specification.name
  }

  public static func scaleFactorForModel(_ name: String) -> UpscaleFactor {
    guard let specification = specificationForModel(name) else { return .x4 }
    return specification.scaleFactor
  }

  public static func numberOfBlocksForModel(_ name: String) -> Int {
    guard let specification = specificationForModel(name) else { return 23 }
    return specification.blocks
  }

  public static func fileSHA256ForModelDownloaded(_ name: String) -> String? {
    return fileSHA256[name]
  }

  public static func availableFiles(excluding file: String?) -> Set<String> {
    var files = Set<String>()
    for specification in availableSpecifications {
      guard specification.file != file, UpscalerZoo.isModelDownloaded(specification.file) else {
        continue
      }
      files.insert(specification.file)
    }
    return files
  }
}
