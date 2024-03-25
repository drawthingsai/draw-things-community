load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

cc_library(
    name = "CAtomic",
    srcs = glob(["Sources/CAtomic/*.c"]),
    hdrs = glob([
        "Sources/CAtomic/include/*.h",
    ]),
    includes = [
        "Sources/CAtomic/include/",
    ],
    tags = ["swift_module=CAtomic"],
)

swift_library(
    name = "Markdown",
    srcs = glob([
        "Sources/Markdown/**/*.swift",
    ]),
    module_name = "Markdown",
    visibility = ["//visibility:public"],
    deps = [
        ":CAtomic",
        "@SwiftCMark//:cmark-gfm",
        "@SwiftCMark//:cmark-gfm-extensions",
    ],
)
