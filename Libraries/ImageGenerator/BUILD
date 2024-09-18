load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

config_setting(
    name = "macos_build",
    constraint_values = [
        "@platforms//os:osx",
    ],
)

swift_library(
    name = "LocalImageGenerator",
    srcs = glob(
               ["Sources/**/*.swift"],
               exclude = [
                   "Sources/ImageConverter.swift",
                   "Sources/ImageGeneratorProtocol.swift",
                   "Sources/ImageGeneratorUtils.swift",
               ],
           ) +
           select({
               ":macos_build": ["Sources/ImageConverter.swift"],
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
        "//Libraries/SwiftDiffusion:SwiftDiffusionPreprocessors",
        "//Libraries/SwiftDiffusion:SwiftDiffusionCoreMLModelManager",
        "//Libraries/SwiftDiffusion:SwiftDiffusionWrapper",
        "//Libraries/Upscaler",
        "@ccv//lib:ccv",
        "@s4nnc//nnc",
        "//Libraries/GRPC:GRPCModels",
        "//Libraries/GRPC:GRPCServer",
        "@grpc-swift//:GRPC",
        "@SwiftLog//:Logging",
    ] + select({
        ":macos_build": [
            "//Libraries/FaceRestorer",
            "//Libraries/Scripting",
            "//Libraries/SwiftDiffusion:SwiftDiffusionCoreML",
        ],
        "//conditions:default": [],
    }),
)

swift_library(
    name = "ImageGenerator",
    srcs = glob(
        [
            "Sources/ImageGeneratorProtocol.swift",
            "Sources/ImageGeneratorUtils.swift",
        ],
    ),
    module_name = "ImageGenerator",
    visibility = ["//visibility:public"],
    deps = [
        "//Libraries/DataModels",
        "//Libraries/GRPC:GRPCModels",
        "//Libraries/ModelZoo",
        "//Libraries/SwiftDiffusion",
        "@ccv//lib:ccv",
        "@s4nnc//nnc",
    ],
)
