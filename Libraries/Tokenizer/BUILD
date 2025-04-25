load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")
load("@bazel_skylib//lib:selects.bzl", "selects")

swift_library(
    name = "Tokenizer",
    srcs = glob(
        ["Sources/**/*.swift"],
        exclude = [
            "Sources/CoreML/**/*.swift",
            "Sources/Preprocessors/**/*.swift",
            "Sources/CoreMLModelManager/**/*.swift",
            "Sources/UNetWrapper/**/*.swift",
        ],
    ),
    module_name = "Tokenizer",
    visibility = ["//visibility:public"],
    deps = [
        "@swift-sentencepiece//:swift-sentencepiece",
    ],
)
