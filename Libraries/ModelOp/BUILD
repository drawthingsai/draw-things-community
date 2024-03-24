load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

swift_library(
    name = "ModelOp",
    srcs = glob(["Sources/**/*.swift"]),
    module_name = "ModelOp",
    visibility = ["//visibility:public"],
    deps = [
        "//Libraries/DataModels",
        "//Libraries/ModelZoo",
        "//Libraries/SwiftDiffusion",
        "@s4nnc//nnc",
    ],
)
