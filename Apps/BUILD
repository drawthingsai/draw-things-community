load("@build_bazel_rules_swift//swift:swift.bzl", "swift_binary", "swift_library")
load("@build_bazel_rules_apple//apple:macos.bzl", "macos_command_line_application")

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
    deps = [
        "//Libraries/BinaryResources",
        "//Libraries/DataModels",
        "//Libraries/GRPC:GRPCModels",
        "//Libraries/GRPC:GRPCServer",
        "//Libraries/ImageGenerator",
        "//Libraries/LocalImageGenerator",
        "//Libraries/SwiftDiffusion",
        "//Libraries/Utils",
        "@SwiftArgumentParser//:ArgumentParser",
        "@grpc-swift//:GRPC",
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

macos_command_line_application(
    name = "gRPCServerCLI-macOS",
    bundle_id = "com.liuliu.draw-things-grpc-server-cli",
    minimum_os_version = "13.0",
    visibility = ["//visibility:public"],
    deps = [
        ":gRPCServerCLILib",
    ],
)
