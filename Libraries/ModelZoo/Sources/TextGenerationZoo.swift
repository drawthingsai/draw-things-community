import Diffusion
import Foundation
import NNC

public struct TextGenerationZoo: DownloadZoo {
  struct Specification: Codable {
    var name: String
    var file: String
  }

  private static let fileSHA256: [String: String] = [
    "blip2_eva_vit_q8p.ckpt": "98e66dd447218e497718014d555c15083e65c5452a5eafa720ad48ce864ebbe0",
    "blip2_qformer_f16.ckpt": "7b810d531a72f38a061412ac117be1fe9a2786ac4d25f19a700f561ed338f028",
    "opt_2.7b_q6p.ckpt": "08026e46ed55691d61aeb5740519e227a0ac17cce88dde399b4824dabec09ef2",
    "siglip_384_q8p.ckpt": "ff41f7ef2281cf4ad1b6523bf87cfea4595a5ff3327a9fce94660f96164a316c",
    "moondream1_q6p.ckpt": "25373519fac6dddfcbdb4cf0bca63372cde1bc3a46894d86fbe00aa0277be41b",
    "moondream2_q6p.ckpt": "58121af12e0506e8edb970a42103317ad749b7d4a05a4f4ac715df4df82dcbc6",
  ]

  static let builtinSpecifications: [Specification] = [
    Specification(name: "BLIP2 EVA Vision Transformer (8-bit)", file: "blip2_eva_vit_q8p.ckpt"),
    Specification(name: "BLIP2 QFormer", file: "blip2_qformer_f16.ckpt"),
    Specification(name: "OPT 2.7B (6-bit)", file: "opt_2.7b_q6p.ckpt"),
    Specification(name: "SigLIP Vision Transformer (8-bit)", file: "siglip_384_q8p.ckpt"),
    Specification(name: "Phi Moondream1 FT (6-bit)", file: "moondream1_q6p.ckpt"),
    Specification(name: "Phi Moondream2 FT (6-bit)", file: "moondream2_q6p.ckpt"),
  ]

  private static var specificationMapping: [String: Specification] = {
    var mapping = [String: Specification]()
    for specification in builtinSpecifications {
      mapping[specification.file] = specification
    }
    return mapping
  }()

  public static func filePathForModelDownloaded(_ name: String) -> String {
    return ModelZoo.filePathForModelDownloaded(name)
  }

  public static func isModelDownloaded(_ name: String) -> Bool {
    return ModelZoo.isModelDownloaded(name)
  }

  public static func humanReadableNameForModel(_ name: String) -> String {
    guard let specification = specificationMapping[name] else { return name }
    return specification.name
  }

  public static func fileSHA256ForModelDownloaded(_ name: String) -> String? {
    return fileSHA256[name]
  }

  public static func availableFiles(excluding file: String?) -> Set<String> {
    var files = Set<String>()
    for specification in builtinSpecifications {
      guard specification.file != file, TextGenerationZoo.isModelDownloaded(specification.file)
      else { continue }
      files.insert(specification.file)
    }
    return files
  }
}
