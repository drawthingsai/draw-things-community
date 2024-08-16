load(
    "@build_bazel_rules_swift//swift:swift.bzl",
    "swift_library",
)

swift_library(
    name = "NIOSSL",
    srcs = glob([
        "Sources/NIOSSL/**/*.swift",
    ]),
    module_name = "NIOSSL",
    visibility = ["//visibility:public"],
    deps = [
        ":CNIOBoringSSL",
        ":CNIOBoringSSLShims",
        "@SwiftNIO//:NIO",
        "@SwiftNIO//:NIOConcurrencyHelpers",
        "@SwiftNIO//:NIOCore",
        "@SwiftNIO//:NIOTLS",
    ],
)

cc_library(
    name = "CNIOBoringSSLShims",
    srcs = glob([
        "Sources/CNIOBoringSSLShims/**/*.c",
    ]),
    hdrs = glob([
        "Sources/CNIOBoringSSLShims/include/**/*.h",
    ]),
    copts = [],
    includes = ["Sources/CNIOBoringSSLShims/include"],
    tags = ["swift_module=CNIOBoringSSLShims"],
    visibility = ["//visibility:public"],
    deps = [":CNIOBoringSSL"],
)

cc_library(
    name = "CNIOBoringSSL",
    srcs = glob([
        "Sources/CNIOBoringSSL/**/*.h",
        "Sources/CNIOBoringSSL/**/*.c",
        "Sources/CNIOBoringSSL/**/*.cc",
        "Sources/CNIOBoringSSL/**/*.S",
    ]),
    hdrs = glob([
        "Sources/CNIOBoringSSL/include/**/*.h",
        "Sources/CNIOBoringSSL/include/**/*.inc",
    ]),
    copts = [],
    includes = ["Sources/CNIOBoringSSL/include"],
    tags = ["swift_module=CNIOBoringSSL"],
    visibility = ["//visibility:public"],
    deps = [],
)
