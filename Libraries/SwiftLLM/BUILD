load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library", "swift_test")

# Shared model metadata is also compiled into the standalone weight converter.
exports_files(["Sources/Models/DeepSeek4_1Configuration.swift"])

swift_library(
    name = "SwiftLLM",
    srcs = glob(["Sources/**/*.swift"]),
    module_name = "LLM",
    visibility = ["//visibility:public"],
    deps = [
        "//Libraries/Tokenizer",
        "@ccv//lib:ccv",
        "@s4nnc//nnc",
    ],
)

swift_test(
    name = "DeepSeek4RotaryEmbeddingTests",
    srcs = ["Tests/DeepSeek4RotaryEmbeddingTests.swift"],
    deps = [
        ":SwiftLLM",
        "@s4nnc//nnc",
    ],
)

swift_test(
    name = "Qwen3_5TextGenerationTests",
    srcs = ["Tests/Qwen3_5TextGenerationTests.swift"],
    tags = [
        "manual",
        "requires-gpu",
    ],
    deps = [
        ":SwiftLLM",
        "@s4nnc//nnc",
    ],
)
