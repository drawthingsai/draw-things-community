load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

cc_library(
    name = "_NumericsShims",
    srcs = ["Sources/_NumericsShims/_NumericsShims.c"],
    hdrs = ["Sources/_NumericsShims/include/_NumericsShims.h"],
    includes = [
        "Sources/_NumericsShims/include/",
    ],
    tags = ["swift_module=_NumericsShims"],
)

swift_library(
    name = "RealModule",
    srcs = glob([
        "Sources/RealModule/**/*.swift",
    ]),
    module_name = "RealModule",
    visibility = ["//visibility:public"],
    deps = [
        ":_NumericsShims",
    ],
)

swift_library(
    name = "ComplexModule",
    srcs = glob([
        "Sources/ComplexModule/**/*.swift",
    ]),
    module_name = "ComplexModule",
    visibility = ["//visibility:public"],
    deps = [
        ":RealModule",
    ],
)

swift_library(
    name = "Numerics",
    srcs = glob([
        "Sources/Numerics/**/*.swift",
    ]),
    module_name = "Numerics",
    visibility = ["//visibility:public"],
    deps = [
        ":ComplexModule",
        ":RealModule",
    ],
)
