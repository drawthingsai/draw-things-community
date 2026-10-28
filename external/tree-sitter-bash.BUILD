load("@build_bazel_rules_swift//swift:swift.bzl", "swift_interop_hint")

# The tree-sitter-bash grammar: the generated parser and its external scanner.
cc_library(
    name = "TreeSitterBash",
    srcs = [
        "src/parser.c",
        "src/scanner.c",
    ],
    hdrs = ["bindings/c/tree_sitter/tree-sitter-bash.h"],
    aspect_hints = [":TreeSitterBash_swift_interop"],
    copts = [
        "-std=c11",
        "-Wno-unused-but-set-variable",
    ],
    includes = [
        "bindings/c",
        "src",
    ],
    tags = ["swift_module=CTreeSitterBash"],
    textual_hdrs = glob(["src/tree_sitter/*.h"]),
    visibility = ["//visibility:public"],
)

swift_interop_hint(
    name = "TreeSitterBash_swift_interop",
    module_name = "CTreeSitterBash",
)
