load("@build_bazel_rules_swift//swift:swift.bzl", "swift_binary")

swift_binary(
    name = "ModelConverter",
    srcs = glob(["ModelConverter/*.swift"]),
    deps = [
        "//Libraries/ModelOp",
        "//Libraries/SwiftDiffusion",
        "@SwiftArgumentParser//:ArgumentParser",
        "@s4nnc//nnc",
    ],
)
