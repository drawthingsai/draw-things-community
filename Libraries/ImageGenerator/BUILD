load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

swift_library(
    name = "ImageGenerator",
    srcs = glob(["Sources/**/*.swift"]),
    module_name = "ImageGenerator",
    visibility = ["//visibility:public"],
    deps = [
        "//Libraries/DataModels",
        "//Libraries/FaceRestorer",
        "//Libraries/ModelZoo",
        "//Libraries/Scripting",
        "//Libraries/SwiftDiffusion",
        "//Libraries/SwiftDiffusion:SwiftDiffusionCoreML",
        "//Libraries/SwiftDiffusion:SwiftDiffusionPreprocessors",
        "//Libraries/Upscaler",
        "@ccv//lib:ccv",
        "@s4nnc//nnc",
    ],
)
