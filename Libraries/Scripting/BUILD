load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

swift_library(
    name = "Scripting",
    srcs = glob(["Sources/**/*.swift"]),
    module_name = "Scripting",
    visibility = ["//visibility:public"],
    deps = [
        "//Libraries/DataModels",
        "//Libraries/ImageSegmentation",
        "//Libraries/SwiftDiffusion",
        "//Libraries/Utils",
        "@s4nnc//nnc",
    ],
)
