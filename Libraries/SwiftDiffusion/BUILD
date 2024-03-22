load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

swift_library(
    name = "SwiftDiffusion",
    srcs = glob(
        ["Sources/**/*.swift"],
        exclude = [
            "Sources/CoreML/**/*.swift",
            "Sources/Preprocessors/**/*.swift",
        ],
    ),
    module_name = "Diffusion",
    visibility = ["//visibility:public"],
    deps = [
        "//Vendors/ZIPFoundation",
        "@SwiftNumerics//:Numerics",
        "@ccv//lib:ccv",
        "@s4nnc//nnc",
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
        "@SwiftAlgorithms//:Algorithms",
        "@s4nnc//nnc:nnc_coreml",
        "@swift-atomics//:SwiftAtomics",
    ],
)

swift_library(
    name = "SwiftDiffusionPreprocessors",
    srcs = glob(["Sources/Preprocessors/**/*.swift"]),
    module_name = "DiffusionPreprocessors",
    visibility = ["//visibility:public"],
    deps = [
        ":SwiftDiffusion",
        "@s4nnc//nnc:nnc_coreml",
    ],
)
