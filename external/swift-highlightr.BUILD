load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

swift_library(
    name = "Highlightr",
    srcs = glob([
        "Pod/Classes/**/*.swift",
    ]),
    data = glob([
        "Pod/Assets/styles/atom-one-*.css",
        "Pod/Assets/styles/pojoaque.min.css",
        "Pod/Assets/Highlighter/highlight.min.js",
    ]),
    module_name = "Highlightr",
    visibility = ["//visibility:public"],
)
