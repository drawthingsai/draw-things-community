load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

cc_library(
    name = "CSystem",
    srcs = ["Sources/CSystem/shims.c"],
    hdrs = glob([
        "Sources/CSystem/include/*.h",
    ]),
    includes = [
        "Sources/CSystem/include/",
    ],
    tags = ["swift_module=CSystem"],
)

swift_library(
    name = "SystemPackage",
    srcs = glob([
        "Sources/System/**/*.swift",
    ]),
    defines = [
        "_CRT_SECURE_NO_WARNINGS",
        "SYSTEM_PACKAGE",
    ],
    module_name = "SystemPackage",
    visibility = ["//visibility:public"],
    deps = [
        ":CSystem",
    ],
)
