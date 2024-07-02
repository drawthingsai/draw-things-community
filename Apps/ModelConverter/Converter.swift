import ArgumentParser
import Diffusion
import Foundation
import ModelOp
import ModelZoo
import NNC

@main
struct Converter: ParsableCommand {
  @Option(
    name: .shortAndLong,
    help: "The model file that is either the safetensors or the PyTorch checkpoint.")
  var file: String
  @Option(name: .shortAndLong, help: "The name of the model.")
  var name: String
  @Option(help: "The autoencoder file.")
  var autoencoderFile: String? = nil
  @Option(name: .shortAndLong, help: "The directory to write the output files to.")
  var outputDirectory: String
  @Flag(help: "Whether to convert the text encoder(s).")
  var textEncoders = false

  private struct Specification: Codable {
    var name: String
    var file: String
    var version: ModelVersion
    var modifier: SamplerModifier
    var textEncoder: String?
    var autoencoder: String?
    var clipEncoder: String?
    var t5Encoder: String?
  }

  mutating func run() throws {
    ModelZoo.externalUrl = URL(fileURLWithPath: outputDirectory)
    let fileName = Importer.cleanup(filename: name)
    let importer = ModelImporter(
      filePath: file, modelName: fileName,
      isTextEncoderCustomized: textEncoders,
      autoencoderFilePath: autoencoderFile, textEncoderFilePath: nil, textEncoder2FilePath: nil)
    let (filePaths, modelVersion, modifier) = try importer.import { _ in
    } progress: { _ in
    }
    let fileNames = filePaths.map { ($0 as NSString).lastPathComponent }
    var autoencoder = fileNames.first {
      $0.hasSuffix("_vae_f16.ckpt")
    }
    var clipEncoder: String? = nil
    let textEncoder: String?
    var t5Encoder: String? = nil
    switch modelVersion {
    case .v1:
      textEncoder = fileNames.first {
        $0.hasSuffix("_clip_vit_l14_f16.ckpt")
      }
    case .v2:
      textEncoder = fileNames.first {
        $0.hasSuffix("_open_clip_vit_h14_f16.ckpt")
      }
    case .sdxlBase, .ssd1b:
      clipEncoder = fileNames.first {
        $0.hasSuffix("_clip_vit_l14_f16.ckpt")
      }
      textEncoder = fileNames.first {
        $0.hasSuffix("_open_clip_vit_bigg14_f16.ckpt")
      }
      if autoencoder == nil {
        autoencoder = "sdxl_vae_v1.0_f16.ckpt"
      }
    case .sd3:
      clipEncoder = fileNames.first {
        $0.hasSuffix("_clip_vit_l14_f16.ckpt")
      }
      textEncoder = fileNames.first {
        $0.hasSuffix("_open_clip_vit_bigg14_f16.ckpt")
      }
      t5Encoder = fileNames.first {
        $0.hasSuffix("_t5_xxl_encoder_f16.ckpt")
      }
      if autoencoder == nil {
        autoencoder = "sd3_vae_f16.ckpt"
      }
    case .pixart:
      textEncoder = fileNames.first {
        $0.hasSuffix("_t5_xxl_encoder_f16.ckpt")
      }
      if autoencoder == nil {
        autoencoder = "sdxl_vae_v1.0_f16.ckpt"
      }
    case .sdxlRefiner:
      textEncoder =
        fileNames.first {
          $0.hasSuffix("_open_clip_vit_bigg14_f16.ckpt")
        } ?? "open_clip_vit_bigg14_f16.ckpt"
      if autoencoder == nil {
        autoencoder = "sdxl_vae_v1.0_f16.ckpt"
      }
    case .svdI2v:
      textEncoder =
        fileNames.first {
          $0.hasSuffix("_open_clip_vit_h14_f16.ckpt")
        } ?? "open_clip_vit_h14_vision_model_f16.ckpt"
      if clipEncoder == nil {
        clipEncoder = "open_clip_vit_h14_visual_proj_f16.ckpt"
      }
    case .kandinsky21, .wurstchenStageC, .wurstchenStageB:
      fatalError()
    }
    let specification = Specification(
      name: name, file: "\(fileName)_f16.ckpt", version: modelVersion, modifier: modifier,
      textEncoder: textEncoder, autoencoder: autoencoder, clipEncoder: clipEncoder,
      t5Encoder: t5Encoder)
    let jsonEncoder = JSONEncoder()
    jsonEncoder.keyEncodingStrategy = .convertToSnakeCase
    jsonEncoder.outputFormatting = .prettyPrinted
    let jsonData = try jsonEncoder.encode(specification)
    print(String(decoding: jsonData, as: UTF8.self))
  }
}
