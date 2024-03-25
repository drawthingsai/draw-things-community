load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

config_setting(
    name = "osx_build",
    constraint_values = [
        "@platforms//os:osx",
    ],
)

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
    ] + select({
        ":osx_build": ["SYSTEM_PACKAGE_DARWIN"],
        "//conditions:default": [],
    }),
    module_name = "SystemPackage",
    visibility = ["//visibility:public"],
    deps = [
        ":CSystem",
    ],
)
