import ArgumentParser
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

  mutating func run() throws {
    ModelZoo.externalUrl = URL(fileURLWithPath: outputDirectory)
    let importer = ModelImporter(
      filePath: file, modelName: Importer.cleanup(filename: name),
      isTextEncoderCustomized: textEncoders,
      autoencoderFilePath: autoencoderFile)
    let result = try importer.import { _ in
    } progress: { _ in
    }
    print(result)
  }
}
