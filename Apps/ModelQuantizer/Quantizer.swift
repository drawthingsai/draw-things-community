import ArgumentParser
import Diffusion
import NNC

@main
struct Quantizer: ParsableCommand {
  @Option(
    name: .shortAndLong,
    help: "The input file to be converted.")
  var inputFile: String
  @Option(name: .shortAndLong, help: "The directory to write the output files to.")
  var outputFile: String

  mutating func run() throws {
    let graph = DynamicGraph()
    graph.openStore(
      inputFile, flags: .readOnly, externalStore: TensorData.externalStore(filePath: inputFile)
    ) { store in
      let keys = store.keys
      graph.openStore(outputFile) {
        for key in keys {
          guard let tensor = store.read(key, codec: [.q6p, .q8p, .ezm7, .externalData]) else {
            continue
          }
          // First convert the tensor to FP16, and then to q8p.
          let fp16 = Tensor<Float16>(from: tensor)
          let shape = fp16.shape
          let squeezedDims = shape.reduce(0) { $1 > 1 ? 1 + $0 : $0 }
          if shape.count == 2 && squeezedDims > 1 {
            $0.write(key, tensor: fp16, codec: [.q6p, .ezm7])
          } else if shape.count == 4 && squeezedDims > 1 {
            $0.write(key, tensor: fp16, codec: [.q8p, .ezm7])
          } else {
            $0.write(key, tensor: fp16, codec: .ezm7)
          }
        }
      }
    }
  }
}
