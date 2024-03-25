load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")
load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

swift_library(
    name = "SwiftSyntax",
    srcs = glob([
        "Sources/SwiftSyntax/**/*.swift",
    ]),
    module_name = "SwiftSyntax",
    visibility = ["//visibility:public"],
    deps = [],
)

swift_library(
    name = "SwiftBasicFormat",
    srcs = glob([
        "Sources/SwiftBasicFormat/**/*.swift",
    ]),
    module_name = "SwiftBasicFormat",
    visibility = ["//visibility:public"],
    deps = [
        ":SwiftSyntax",
    ],
)

swift_library(
    name = "SwiftDiagnostics",
    srcs = glob([
        "Sources/SwiftDiagnostics/**/*.swift",
    ]),
    module_name = "SwiftDiagnostics",
    visibility = ["//visibility:public"],
    deps = [
        ":SwiftSyntax",
    ],
)

swift_library(
    name = "SwiftParser",
    srcs = glob([
        "Sources/SwiftParser/**/*.swift",
    ]),
    module_name = "SwiftParser",
    visibility = ["//visibility:public"],
    deps = [
        ":SwiftDiagnostics",
        ":SwiftSyntax",
    ],
)

swift_library(
    name = "SwiftParserDiagnostics",
    srcs = glob([
        "Sources/SwiftParserDiagnostics/**/*.swift",
    ]),
    module_name = "SwiftParserDiagnostics",
    visibility = ["//visibility:public"],
    deps = [
        ":SwiftBasicFormat",
        ":SwiftSyntax",
        ":SwiftParser",
        ":SwiftDiagnostics",
    ],
)

swift_library(
    name = "SwiftSyntaxBuilder",
    srcs = glob([
        "Sources/SwiftSyntaxBuilder/**/*.swift",
    ]),
    module_name = "SwiftSyntaxBuilder",
    visibility = ["//visibility:public"],
    deps = [
        ":SwiftBasicFormat",
        ":SwiftSyntax",
        ":SwiftParser",
        ":SwiftParserDiagnostics",
    ],
)

swift_library(
    name = "SwiftSyntaxParser",
    srcs = glob([
        "Sources/SwiftSyntaxParser/**/*.swift",
    ]),
    module_name = "SwiftSyntaxParser",
    visibility = ["//visibility:public"],
    deps = [
        ":SwiftSyntax",
        ":SwiftParser",
    ],
)

swift_library(
    name = "SwiftOperators",
    srcs = glob([
        "Sources/SwiftOperators/**/*.swift",
    ]),
    module_name = "SwiftOperators",
    visibility = ["//visibility:public"],
    deps = [
        ":SwiftSyntax",
        ":SwiftParser",
        ":SwiftDiagnostics",
    ],
)
