load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

config_setting(
    name = "macos_build",
    constraint_values = [
        "@platforms//os:osx",
    ],
)

swift_library(
    name = "ImageGenerator",
    srcs = glob(
               ["Sources/**/*.swift"],
               exclude = ["Sources/ImageConverter.swift"],
           ) +
           select({
               ":macos_build": ["Sources/ImageConverter.swift"],
               "//conditions:default": [],
           }),
    module_name = "ImageGenerator",
    visibility = ["//visibility:public"],
    deps = [
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
    ] + select({
        ":macos_build": [
            "//Libraries/FaceRestorer",
            "//Libraries/Scripting",
            "//Libraries/SwiftDiffusion:SwiftDiffusionCoreML",
        ],
        "//conditions:default": [],
    }),
)
