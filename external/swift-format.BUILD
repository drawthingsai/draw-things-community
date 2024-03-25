load("@build_bazel_rules_swift//swift:swift.bzl", "swift_binary", "swift_library")

swift_library(
    name = "SwiftFormatConfiguration",
    srcs = glob([
        "Sources/SwiftFormatConfiguration/**/*.swift",
    ]),
    module_name = "SwiftFormatConfiguration",
)

swift_library(
    name = "SwiftFormatCore",
    srcs = glob([
        "Sources/SwiftFormatCore/**/*.swift",
    ]),
    module_name = "SwiftFormatCore",
    deps = [
        ":SwiftFormatConfiguration",
        "@SwiftSyntax",
        "@SwiftSyntax//:SwiftOperators",
    ],
)

swift_library(
    name = "SwiftFormatRules",
    srcs = glob([
        "Sources/SwiftFormatRules/**/*.swift",
    ]),
    module_name = "SwiftFormatRules",
    deps = [
        ":SwiftFormatCore",
    ],
)

swift_library(
    name = "SwiftFormatPrettyPrint",
    srcs = glob([
        "Sources/SwiftFormatPrettyPrint/**/*.swift",
    ]),
    module_name = "SwiftFormatPrettyPrint",
    deps = [
        ":SwiftFormatCore",
    ],
)

swift_library(
    name = "SwiftFormatWhitespaceLinter",
    srcs = glob([
        "Sources/SwiftFormatWhitespaceLinter/**/*.swift",
    ]),
    module_name = "SwiftFormatWhitespaceLinter",
    deps = [
        ":SwiftFormatCore",
    ],
)

swift_library(
    name = "SwiftFormat",
    srcs = glob([
        "Sources/SwiftFormat/**/*.swift",
    ]),
    module_name = "SwiftFormat",
    deps = [
        ":SwiftFormatCore",
        ":SwiftFormatPrettyPrint",
        ":SwiftFormatRules",
        ":SwiftFormatWhitespaceLinter",
        "@SwiftSyntax//:SwiftSyntaxParser",
        "@SwiftSyntax//:SwiftParserDiagnostics",
    ],
)

swift_binary(
    name = "swift-format",
    srcs = glob([
        "Sources/swift-format/**/*.swift",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        ":SwiftFormat",
        "@SwiftArgumentParser//:ArgumentParser",
        "@SwiftToolsSupportCore//:TSCBasic",
    ],
)
