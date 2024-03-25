load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

cc_library(
    name = "TSCclibc",
    srcs = glob(["Sources/TSCclibc/*.c"]),
    hdrs = glob([
        "Sources/TSCclibc/include/*.h",
    ]),
    includes = [
        "Sources/TSCclibc/include/",
    ],
    tags = ["swift_module=TSCclibc"],
)

swift_library(
    name = "TSCLibc",
    srcs = glob([
        "Sources/TSCLibc/**/*.swift",
    ]),
    module_name = "TSCLibc",
    deps = [],
)

swift_library(
    name = "TSCBasic",
    srcs = glob([
        "Sources/TSCBasic/**/*.swift",
    ]),
    module_name = "TSCBasic",
    visibility = ["//visibility:public"],
    deps = [
        ":TSCLibc",
        ":TSCclibc",
        "@SwiftSystem//:SystemPackage",
    ],
)
