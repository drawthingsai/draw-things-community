load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

cc_library(
    name = "C_zlib",
    hdrs = ["Sources/CZLib/shim.h"],
    defines = ["_GNU_SOURCE"],
    linkopts = ["-lz"],
    tags = ["swift_module=CZlib"],
)

swift_library(
    name = "ZIPFoundation",
    srcs = glob([
        "Sources/ZIPFoundation/**/*.swift",
    ]),
    module_name = "ZIPFoundation",
    visibility = ["//visibility:public"],
    deps = [
        ":C_zlib",
    ],
)
