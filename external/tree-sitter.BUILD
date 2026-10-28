load("@build_bazel_rules_swift//swift:swift.bzl", "swift_interop_hint")

# The tree-sitter runtime, built from its single-file amalgamation.
cc_library(
    name = "TreeSitter",
    srcs = ["lib/src/lib.c"],
    hdrs = ["lib/include/tree_sitter/api.h"],
    aspect_hints = [":TreeSitter_swift_interop"],
    copts = [
        "-std=c11",
        "-D_POSIX_C_SOURCE=200112L",
        "-D_DEFAULT_SOURCE",
        "-D_DARWIN_C_SOURCE",
        "-Wno-unused-but-set-variable",
    ],
    includes = [
        "lib/include",
        "lib/src",
    ],
    tags = ["swift_module=CTreeSitter"],
    textual_hdrs = glob(
        [
            "lib/src/**/*.c",
            "lib/src/**/*.h",
        ],
        exclude = ["lib/src/lib.c"],
    ),
    visibility = ["//visibility:public"],
)

swift_interop_hint(
    name = "TreeSitter_swift_interop",
    module_name = "CTreeSitter",
)
