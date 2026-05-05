// swift-tools-version:5.9
import Foundation
import PackageDescription

let package = Package(
  name: "DrawThings",
  platforms: [.macOS(.v13), .iOS(.v16), .tvOS(.v16), .visionOS(.v1)],
  products: [
    .executable(name: "gRPCServerCLI", targets: ["gRPCServerCLI"]),
    .executable(name: "draw-things-cli", targets: ["DrawThingsCLI"]),
    .library(name: "_MediaGenerationKit", targets: ["_MediaGenerationKit"]),
  ],
  dependencies: [
    .package(
      url: "https://github.com/liuliu/ccv.git", revision: "a12d7f7c9e400a17696881adad4a5ee552583595"
    ),
    .package(
      url: "https://github.com/liuliu/s4nnc.git",
      revision: "1ca180b54bbefe8119604e1ec454b58f5d73a3a3"),
    .package(
      url: "https://github.com/liuliu/dflat.git",
      revision: "73925e51e4f44add842177a229f9990cb13711ff"),
    .package(
      url: "https://github.com/liuliu/swift-fickling.git",
      revision: "5c982bf479c4cdf8c7f72002cd79ec88b553ab34"),
    .package(
      url: "https://github.com/liuliu/swift-sentencepiece",
      revision: "8d17bf2e017c97563e8805545d676be9739b6c0e"),
    .package(url: "https://github.com/apple/swift-log.git", from: "1.4.4"),
    .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.3.1"),
    .package(url: "https://github.com/apple/swift-crypto.git", from: "3.7.1"),
    .package(url: "https://github.com/apple/swift-atomics.git", from: "1.2.0"),
    .package(url: "https://github.com/apple/swift-docc-plugin", from: "1.4.5"),
    .package(url: "https://github.com/apple/swift-nio-ssl.git", from: "2.23.1"),
    .package(url: "https://github.com/jagreenwood/swift-log-datadog.git", from: "0.3.0"),

    .package(url: "https://github.com/grpc/grpc-swift.git", from: "1.16.0"),
    .package(url: "https://github.com/apple/swift-protobuf.git", from: "1.27.0"),
    .package(url: "https://github.com/apple/swift-collections.git", from: "1.1.3"),
    .package(url: "https://github.com/apple/swift-algorithms.git", from: "1.1.0"),
    .package(url: "https://github.com/apple/swift-numerics.git", from: "1.0.0"),
    .package(
      url: "https://github.com/kelvin13/swift-png",
      revision: "075dfb248ae327822635370e9d4f94a5d3fe93b2"),
  ],
  targets: [
    .target(
      name: "Utils",
      path: "Libraries/Utils/Sources"
    ),
    .target(
      name: "Tokenizer",
      dependencies: [
        .product(name: "SentencePiece", package: "swift-sentencepiece")
      ],
      path: "Libraries/Tokenizer/Sources"
    ),
    .target(
      name: "WeightsCache",
      dependencies: [
        .product(name: "Collections", package: "swift-collections"),
        .product(name: "Numerics", package: "swift-numerics"),
        .product(name: "Atomics", package: "swift-atomics"),
        .product(name: "ccv", package: "ccv"),
        .product(name: "NNC", package: "s4nnc"),
      ],
      path: "Libraries/WeightsCache/Sources"
    ),
    .target(
      name: "SFMT",
      dependencies: [
        .product(name: "sfmt", package: "ccv")
      ],
      path: "Libraries/SFMT/Sources"
    ),
    .target(
      name: "C_Resources",
      path: "Libraries/BinaryResources/GeneratedC",
      publicHeadersPath: "."
    ),
    .target(
      name: "BinaryResources",
      dependencies: ["C_Resources"],
      path: "Libraries/BinaryResources",
      exclude: [
        "BUILD",
        "Resources",
        "GeneratedC",
      ],
      sources: ["Sources/BinaryResources.swift"]
    ),
    .target(
      name: "ZIPFoundation",
      path: "Vendors/ZIPFoundation/Sources/ZIPFoundation"
    ),

    .target(
      name: "DiffusionMappings",
      path: "Libraries/SwiftDiffusion/Sources/Mappings"
    ),
    .target(
      name: "LLM",
      dependencies: [
        "Tokenizer",
        .product(name: "ccv", package: "ccv"),
        .product(name: "NNC", package: "s4nnc"),
      ],
      path: "Libraries/SwiftLLM/Sources"
    ),
    .target(
      name: "Diffusion",
      dependencies: [
        "DiffusionMappings",
        "LLM",
        "Tokenizer",
        "WeightsCache",
        "ZIPFoundation",
        .product(name: "Numerics", package: "swift-numerics"),
        .product(name: "Atomics", package: "swift-atomics"),
        .product(name: "Fickling", package: "swift-fickling"),
        .product(name: "ccv", package: "ccv"),
        .product(name: "NNC", package: "s4nnc"),
      ],
      path: "Libraries/SwiftDiffusion/Sources",
      exclude: [
        "CoreML",
        "Preprocessors",
        "CoreMLModelManager",
        "UNetWrapper",
        "Mappings",
      ]
    ),
    .target(
      name: "DiffusionPreprocessors",
      dependencies: [
        "Diffusion",
        .product(name: "NNC", package: "s4nnc"),
        .product(name: "NNCCoreMLConversion", package: "s4nnc"),
      ],
      path: "Libraries/SwiftDiffusion/Sources/Preprocessors"
    ),
    .target(
      name: "DiffusionCoreMLModelManager",
      dependencies: [
        "DataModels",
        .product(name: "Algorithms", package: "swift-algorithms"),
        .product(name: "Atomics", package: "swift-atomics"),
      ],
      path: "Libraries/SwiftDiffusion/Sources/CoreMLModelManager"
    ),
    .target(
      name: "DiffusionCoreML",
      dependencies: [
        "Diffusion",
        "DiffusionCoreMLModelManager",
        "WeightsCache",
        "ZIPFoundation",
        .product(name: "Algorithms", package: "swift-algorithms"),
        .product(name: "Atomics", package: "swift-atomics"),
        .product(name: "NNC", package: "s4nnc"),
        .product(name: "NNCCoreMLConversion", package: "s4nnc"),
      ],
      path: "Libraries/SwiftDiffusion/Sources/CoreML"
    ),
    .target(
      name: "DiffusionUNetWrapper",
      dependencies: [
        "Diffusion",
        "DiffusionCoreML",
        "DataModels",
        .product(name: "Algorithms", package: "swift-algorithms"),
        .product(name: "Atomics", package: "swift-atomics"),
        .product(name: "NNC", package: "s4nnc"),
      ],
      path: "Libraries/SwiftDiffusion/Sources/UNetWrapper"
    ),

    .target(
      name: "DataModels",
      dependencies: [
        "Diffusion",
        "Utils",
        .product(name: "SQLiteDflat", package: "dflat"),
      ],
      path: "Libraries/DataModels",
      exclude: [
        "BUILD",
        "Sources/config.fbs",
        "Sources/estimation.fbs",
        "Sources/mixing.fbs",
        "Sources/lora.fbs",
        "Sources/dataset.fbs",
        "Sources/paint_color.fbs",
        "Sources/peer_connection_id.fbs",
      ],
      sources: ["Sources", "PreGeneratedSPM"]
    ),
    .target(
      name: "ScriptDataModels",
      dependencies: [
        "DataModels",
        .product(name: "NNC", package: "s4nnc"),
      ],
      path: "Libraries/Scripting",
      exclude: [
        "BUILD",
        "Sources/ScriptExecutor.swift",
        "Sources/ScriptZoo.swift",
        "Sources/SharedScript.swift",
      ],
      sources: ["Sources/ScriptModels.swift"]
    ),
    .target(
      name: "Scripting",
      dependencies: [
        "ScriptDataModels",
        "DataModels",
        "Diffusion",
        "ImageSegmentation",
        "Utils",
        .product(name: "NNC", package: "s4nnc"),
      ],
      path: "Libraries/Scripting",
      exclude: [
        "BUILD",
        "Sources/ScriptModels.swift",
      ],
      sources: [
        "Sources/ScriptExecutor.swift",
        "Sources/ScriptZoo.swift",
        "Sources/SharedScript.swift",
      ]
    ),

    .target(
      name: "Upscaler",
      dependencies: [
        .product(name: "NNC", package: "s4nnc")
      ],
      path: "Libraries/Upscaler/Sources"
    ),
    .target(
      name: "ImageSegmentation",
      dependencies: [
        .product(name: "ccv", package: "ccv"),
        .product(name: "NNC", package: "s4nnc"),
      ],
      path: "Libraries/ImageSegmentation/Sources"
    ),
    .target(
      name: "FaceRestorer",
      dependencies: [
        .product(name: "NNC", package: "s4nnc"),
        .product(name: "ccv", package: "ccv"),
      ],
      path: "Libraries/FaceRestorer/Sources"
    ),

    .target(
      name: "ModelZoo",
      dependencies: [
        "DataModels",
        "Diffusion",
        "Upscaler",
        .product(name: "NNC", package: "s4nnc"),
      ],
      path: "Libraries/ModelZoo/Sources"
    ),
    .target(
      name: "Downloader",
      dependencies: [
        "ModelZoo"
      ],
      path: "Libraries/Downloader/Sources"
    ),
    .target(
      name: "Trainer",
      dependencies: [
        "DataModels",
        "Diffusion",
        "ModelZoo",
        "Tokenizer",
        "WeightsCache",
        "SFMT",
        .product(name: "NNC", package: "s4nnc"),
        .product(name: "TensorBoard", package: "s4nnc"),
        .product(name: "SQLiteDflat", package: "dflat"),
      ],
      path: "Libraries/Trainer/Sources"
    ),
    .target(
      name: "ModelOp",
      dependencies: [
        "DataModels",
        "Diffusion",
        "ModelZoo",
        "Upscaler",
        "WeightsCache",
        "ZIPFoundation",
        .product(name: "Fickling", package: "swift-fickling"),
        .product(name: "NNC", package: "s4nnc"),
      ],
      path: "Libraries/ModelOp/Sources"
    ),
    .target(
      name: "ConfigurationZoo",
      dependencies: [
        "DataModels",
        "ModelZoo",
        "ScriptDataModels",
      ],
      path: "Libraries/ConfigurationZoo/Sources"
    ),

    .target(
      name: "ImageGenerator",
      dependencies: [
        "DataModels",
        "ModelZoo",
        "Diffusion",
        .product(name: "ccv", package: "ccv"),
        .product(name: "NNC", package: "s4nnc"),
      ],
      path: "Libraries/ImageGenerator/Sources"
    ),
    .target(
      name: "LocalImageGenerator",
      dependencies: [
        "DataModels",
        "ImageGenerator",
        "ModelZoo",
        "ScriptDataModels",
        "Diffusion",
        "DiffusionCoreMLModelManager",
        "DiffusionPreprocessors",
        "DiffusionUNetWrapper",
        "Upscaler",
        "FaceRestorer",
        .product(name: "Logging", package: "swift-log"),
        .product(name: "ccv", package: "ccv"),
        .product(name: "NNC", package: "s4nnc"),
      ],
      path: "Libraries/LocalImageGenerator/Sources"
    ),
    .target(
      name: "RemoteImageGenerator",
      dependencies: [
        "DataModels",
        "ImageGenerator",
        "ModelZoo",
        "GRPCImageServiceModels",
        "GRPCServer",
        "Diffusion",
        .product(name: "Crypto", package: "swift-crypto"),
        .product(name: "Logging", package: "swift-log"),
        .product(name: "GRPC", package: "grpc-swift"),
        .product(name: "NNC", package: "s4nnc"),
        .product(name: "Collections", package: "swift-collections"),
      ],
      path: "Libraries/RemoteImageGenerator/Sources"
    ),
    .target(
      name: "GRPCImageServiceModels",
      dependencies: [
        .product(name: "GRPC", package: "grpc-swift")
      ],
      path: "Libraries/GRPC/Models/Sources/imageService",
      exclude: ["imageService.proto"]
    ),
    .target(
      name: "GRPCControlPanelModels",
      dependencies: [
        .product(name: "GRPC", package: "grpc-swift")
      ],
      path: "Libraries/GRPC/Models/Sources/controlPanel",
      exclude: ["controlPanel.proto"]
    ),
    .target(
      name: "ServerConfigurationRewriter",
      dependencies: [
        "DataModels",
        .product(name: "GRPC", package: "grpc-swift"),
      ],
      path: "Libraries/GRPC/ServerConfigurationRewriter/Sources"
    ),
    .target(
      name: "GRPCServer",
      dependencies: [
        "GRPCImageServiceModels",
        "ServerConfigurationRewriter",
        "BinaryResources",
        "DataModels",
        "ImageGenerator",
        "ModelZoo",
        "ScriptDataModels",
        "Diffusion",
        "Utils",
        .product(name: "ArgumentParser", package: "swift-argument-parser"),
        .product(name: "Crypto", package: "swift-crypto"),
        .product(name: "Logging", package: "swift-log"),
        .product(name: "NNC", package: "s4nnc"),
      ],
      path: "Libraries/GRPC/Server/Sources",
      sources: [
        "GRPCFileUploader.swift",
        "GRPCHostnameUtils.swift",
        "GRPCServerAdvertiser.swift",
        "ImageGenerationClientWrapper.swift",
        "ImageGenerationServiceImpl.swift",
        "ProtectedValue.swift",
        "GRPCServiceBrowser.swift",
      ]
    ),
    .target(
      name: "ProxyControlClient",
      dependencies: [
        "GRPCControlPanelModels",
        "GRPCImageServiceModels",
        "BinaryResources",
        "DataModels",
        "ModelZoo",
        .product(name: "ArgumentParser", package: "swift-argument-parser"),
        .product(name: "Crypto", package: "swift-crypto"),
        .product(name: "Logging", package: "swift-log"),
      ],
      path: "Libraries/GRPC/ProxyControlClient/Sources"
    ),
    .target(
      name: "ServerLoRALoader",
      dependencies: [
        "ServerConfigurationRewriter",
        "DataModels",
        "ModelZoo",
        .product(name: "Crypto", package: "swift-crypto"),
        .product(name: "Logging", package: "swift-log"),
        .product(name: "GRPC", package: "grpc-swift"),
      ],
      path: "Libraries/GRPC/ServerLoRALoader/Sources"
    ),

    .executableTarget(
      name: "gRPCServerCLI",
      dependencies: [
        "BinaryResources",
        "DataModels",
        "GRPCControlPanelModels",
        "GRPCImageServiceModels",
        "GRPCServer",
        "ProxyControlClient",
        "ServerLoRALoader",
        "ImageGenerator",
        "LocalImageGenerator",
        "Diffusion",
        "Utils",
        .product(name: "ArgumentParser", package: "swift-argument-parser"),
        .product(name: "GRPC", package: "grpc-swift"),
        .product(name: "DataDogLog", package: "swift-log-datadog"),
      ],
      path: "Apps/gRPCServerCLI",
      exclude: ["SupportingFiles"],
      sources: ["gRPCServerCLI.swift"]
    ),
    .executableTarget(
      name: "DrawThingsCLI",
      dependencies: [
        "BinaryResources",
        "ConfigurationZoo",
        "DataModels",
        "Downloader",
        "ImageGenerator",
        "LocalImageGenerator",
        "ModelOp",
        "ModelZoo",
        "ScriptDataModels",
        "Diffusion",
        "Trainer",
        "Tokenizer",
        .product(name: "ArgumentParser", package: "swift-argument-parser"),
        .product(name: "PNG", package: "swift-png"),
      ],
      path: "Apps/DrawThingsCLI",
      sources: ["DrawThingsCLI.swift"]
    ),
    .target(
      name: "DeviceAttestation",
      path: "Libraries/DeviceAttestation/Sources"
    ),
    .target(
      name: "_MediaGenerationKit",
      dependencies: [
        "BinaryResources",
        "ConfigurationZoo",
        "DataModels",
        "Diffusion",
        "Downloader",
        "GRPCImageServiceModels",
        "ImageGenerator",
        "LocalImageGenerator",
        "ModelOp",
        "RemoteImageGenerator",
        "ModelZoo",
        "ScriptDataModels",
        "Tokenizer",
        "GRPCServer",
        .product(name: "Atomics", package: "swift-atomics"),
        .product(name: "Dflat", package: "dflat"),
        .product(name: "GRPC", package: "grpc-swift"),
        .product(name: "Logging", package: "swift-log"),
        .product(name: "NNC", package: "s4nnc"),
        .product(name: "SQLiteDflat", package: "dflat"),
        .product(name: "Crypto", package: "swift-crypto"),
      ],
      path: "Libraries/MediaGenerationKit/Sources"
    ),
  ]
)
