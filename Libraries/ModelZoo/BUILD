load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

swift_library(
    name = "ModelZoo",
    srcs = glob(["Sources/**/*.swift"]),
    module_name = "ModelZoo",
    visibility = ["//visibility:public"],
    deps = [
        "//Libraries/DataModels",
        "//Libraries/SwiftDiffusion",
        "//Libraries/Upscaler",
        "@s4nnc//nnc",
    ],
)
