load("@build_bazel_rules_swift//swift:swift.bzl", "swift_binary", "swift_library")
load("@bazel_skylib//lib:selects.bzl", "selects")

config_setting(
    name = "macos_build",
    constraint_values = [
        "@platforms//os:osx",
    ],
)

config_setting(
    name = "ios_build",
    constraint_values = [
        "@platforms//os:ios",
    ],
)

selects.config_setting_group(
    name = "ios_or_macos_build",
    match_any = [
        ":macos_build",
        ":ios_build",
    ],
)

swift_library(
    name = "GRPCServer",
    srcs = [
        "Sources/Server/GRPCFileUploader.swift",
        "Sources/Server/ImageGenerationClientWrapper.swift",
        "Sources/Server/ImageGenerationServiceImpl.swift",
    ] + select({
        ":ios_or_macos_build": [
            "Sources/Server/GRPCServerAdvertiser.swift",
            "Sources/Server/GRPCServiceBrowser.swift",
        ],
        "//conditions:default": [],
    }),
    module_name = "GRPCServer",
    visibility = ["//visibility:public"],
    deps = [
        ":GRPCModels",
        "//Libraries/DataModels",
        "//Libraries/ImageGenerator",
        "//Libraries/ModelZoo",
        "//Libraries/SwiftDiffusion",
        "//Libraries/Utils",
        "@SwiftArgumentParser//:ArgumentParser",
        "@SwiftCrypto//:CCryptoBoringSSL",
        "@SwiftCrypto//:CCryptoBoringSSLShims",
        "@SwiftCrypto//:Crypto",
        "@SwiftCrypto//:CryptoBoringWrapper",
        "@SwiftLog//:Logging",
        "@s4nnc//nnc",
    ],
)

swift_library(
    name = "GRPCModels",
    srcs = glob(["Sources/Models/*.swift"]),
    module_name = "GRPCModels",
    visibility = ["//visibility:public"],
    deps = [
        "//Libraries/DataModels",
        "@grpc-swift//:GRPC",
    ],
)
