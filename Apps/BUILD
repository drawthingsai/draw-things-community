load("@build_bazel_rules_swift//swift:swift.bzl", "swift_binary")

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

swift_binary(
    name = "gRPCServerCLI",
    srcs = glob(["gRPCServerCLI/gRPCServerCLI.swift"]),
    visibility = ["//visibility:public"],
    deps = [
        "//Libraries/BinaryResources",
        "//Libraries/DataModels",
        "//Libraries/GRPC:C_DNSAdvertiser",
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
