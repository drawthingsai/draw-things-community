load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")
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
    name = "SwiftDiffusion",
    srcs = glob(
        ["Sources/**/*.swift"],
        exclude = [
            "Sources/CoreML/**/*.swift",
            "Sources/Preprocessors/**/*.swift",
            "Sources/CoreMLModelManager/**/*.swift",
            "Sources/UNetWrapper/**/*.swift",
        ],
    ),
    module_name = "Diffusion",
    visibility = ["//visibility:public"],
    deps = [
        "//Vendors/ZIPFoundation",
        "@SwiftNumerics//:Numerics",
        "@ccv//lib:ccv",
        "@s4nnc//nnc",
        "@swift-atomics//:SwiftAtomics",
        "@swift-fickling//:fickling",
        "@swift-sentencepiece//:swift-sentencepiece",
    ],
)

swift_library(
    name = "SwiftDiffusionCoreML",
    srcs = glob(["Sources/CoreML/**/*.swift"]),
    module_name = "DiffusionCoreML",
    visibility = ["//visibility:public"],
    deps = [
        ":SwiftDiffusion",
        ":SwiftDiffusionCoreMLModelManager",
        "@SwiftAlgorithms//:Algorithms",
        "@s4nnc//nnc:nnc_coreml",
    ],
)

swift_library(
    name = "SwiftDiffusionCoreMLModelManager",
    srcs = glob(["Sources/CoreMLModelManager/CoreMLModelManager.swift"]),
    module_name = "DiffusionCoreMLModelManager",
    visibility = ["//visibility:public"],
    deps = [
        "//Libraries/DataModels",
        "@SwiftAlgorithms//:Algorithms",
        "@swift-atomics//:SwiftAtomics",
    ],
)

swift_library(
    name = "SwiftDiffusionWrapper",
    srcs = glob(["Sources/UNetWrapper/UNetWrapper.swift"]),
    module_name = "DiffusionUNetWrapper",
    visibility = ["//visibility:public"],
    deps = [
        ":SwiftDiffusion",
        "//Libraries/DataModels",
        "@SwiftAlgorithms//:Algorithms",
        "@s4nnc//nnc:nnc_coreml",
        "@swift-atomics//:SwiftAtomics",
    ] + select({
        ":ios_or_macos_build": [
            ":SwiftDiffusionCoreML",
        ],
        "//conditions:default": [],
    }),
)

swift_library(
    name = "SwiftDiffusionPreprocessors",
    srcs = glob(
        ["Sources/Preprocessors/**/*.swift"],
    ),
    module_name = "DiffusionPreprocessors",
    visibility = ["//visibility:public"],
    deps = [
        ":SwiftDiffusion",
        "@s4nnc//nnc:nnc_coreml",
    ],
)
