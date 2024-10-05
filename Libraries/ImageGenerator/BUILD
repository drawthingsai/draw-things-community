load("@bazel_skylib//lib:selects.bzl", "selects")
load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

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
    name = "LocalImageGenerator",
    srcs = glob(
               ["Sources/**/*.swift"],
               exclude = [
                   "Sources/RemoteImageGenerator.swift",
                   "Sources/ImageConverter.swift",
                   "Sources/ImageGeneratorProtocol.swift",
                   "Sources/ImageGeneratorUtils.swift",
                   "Sources/GenerationEstimator.swift",
               ],
           ) +
           select({
               ":ios_or_macos_build": ["Sources/ImageConverter.swift"],
               "//conditions:default": [],
           }),
    module_name = "LocalImageGenerator",
    visibility = ["//visibility:public"],
    deps = [
        ":ImageGenerator",
        "//Libraries/DataModels",
        "//Libraries/ModelZoo",
        "//Libraries/Scripting:ScriptDataModels",
        "//Libraries/SwiftDiffusion",
        "//Libraries/SwiftDiffusion:SwiftDiffusionCoreMLModelManager",
        "//Libraries/SwiftDiffusion:SwiftDiffusionPreprocessors",
        "//Libraries/SwiftDiffusion:SwiftDiffusionWrapper",
        "//Libraries/Upscaler",
        "@SwiftLog//:Logging",
        "@ccv//lib:ccv",
        "@s4nnc//nnc",
    ] + select({
        ":ios_or_macos_build": [
            "//Libraries/FaceRestorer",
            "//Libraries/Scripting",
            "//Libraries/SwiftDiffusion:SwiftDiffusionCoreML",
        ],
        "//conditions:default": [],
    }),
)

swift_library(
    name = "RemoteImageGenerator",
    srcs = ["Sources/RemoteImageGenerator.swift"],
    module_name = "RemoteImageGenerator",
    visibility = ["//visibility:public"],
    deps = [
        ":ImageGenerator",
        "//Libraries/GRPC:GRPCModels",
        "//Libraries/GRPC:GRPCServer",
        "@grpc-swift//:GRPC",
    ],
)

swift_library(
    name = "ImageGenerator",
    srcs = glob(
        [
            "Sources/ImageGeneratorProtocol.swift",
            "Sources/ImageGeneratorUtils.swift",
            "Sources/GenerationEstimator.swift",
        ],
    ),
    module_name = "ImageGenerator",
    visibility = ["//visibility:public"],
    deps = [
        "//Libraries/DataModels",
        "//Libraries/ModelZoo",
        "//Libraries/SwiftDiffusion",
        "@ccv//lib:ccv",
        "@s4nnc//nnc",
    ],
)
