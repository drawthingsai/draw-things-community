load("@build_bazel_rules_swift//swift:swift.bzl", "swift_binary", "swift_library")

swift_binary(
    name = "ModelConverter",
    srcs = glob(["ModelConverter/*.swift"]),
    deps = [
        "//Libraries/ModelOp",
        "//Libraries/SwiftDiffusion",
        "@SwiftArgumentParser//:ArgumentParser",
        "@s4nnc//nnc",
    ],
)

swift_binary(
    name = "LoRAConverter",
    srcs = glob(["LoRAConverter/*.swift"]),
    deps = [
        "//Libraries/ModelOp",
        "//Libraries/SwiftDiffusion",
        "@SwiftArgumentParser//:ArgumentParser",
        "@s4nnc//nnc",
    ],
)

swift_binary(
    name = "EmbeddingConverter",
    srcs = glob(["EmbeddingConverter/*.swift"]),
    deps = [
        "//Libraries/ModelOp",
        "//Libraries/SwiftDiffusion",
        "@SwiftArgumentParser//:ArgumentParser",
        "@s4nnc//nnc",
    ],
)

swift_binary(
    name = "ModelQuantizer",
    srcs = glob(["ModelQuantizer/*.swift"]),
    deps = [
        "//Libraries/SwiftDiffusion",
        "@SwiftArgumentParser//:ArgumentParser",
        "@s4nnc//nnc",
    ],
)

swift_library(
    name = "gRPCServerCLILib",
    srcs = glob(["gRPCServerCLI/gRPCServerCLI.swift"]),
    visibility = ["//Apps:__subpackages__"],
    deps = [
        "//Libraries/BinaryResources",
        "//Libraries/DataModels",
        "//Libraries/GRPC:GRPCControlPanelModels",
        "//Libraries/GRPC:GRPCImageServiceModels",
        "//Libraries/GRPC:GRPCServer",
        "//Libraries/GRPC:ProxyControlClient",
        "//Libraries/GRPC:ServerLoRALoader",
        "//Libraries/ImageGenerator",
        "//Libraries/LocalImageGenerator",
        "//Libraries/SwiftDiffusion",
        "//Libraries/Utils",
        "@SwiftArgumentParser//:ArgumentParser",
        "@grpc-swift//:GRPC",
        "@swift-log-datadog//:DataDogLog",
    ],
)

swift_binary(
    name = "gRPCServerCLI",
    srcs = [],
    visibility = ["//visibility:public"],
    deps = [
        ":gRPCServerCLILib",
    ],
)
