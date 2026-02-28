load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

swift_library(
    name = "Downloader",
    srcs = glob(["Sources/**/*.swift"]),
    module_name = "Downloader",
    visibility = ["//visibility:public"],
    deps = [
        "//Libraries/ModelZoo",
    ],
)
