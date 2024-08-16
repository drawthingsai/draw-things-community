load(
    "@build_bazel_rules_swift//swift:swift.bzl",
    "swift_library",
)

cc_library(
    name = "CNIOAtomics",
    srcs = glob([
        "Sources/CNIOAtomics/**/*.c",
    ]),
    hdrs = glob([
        "Sources/CNIOAtomics/**/*.h",
    ]),
    copts = [],
    includes = ["Sources/CNIOAtomics/include"],
    tags = ["swift_module=CNIOAtomics"],
)

cc_library(
    name = "CNIODarwin",
    srcs = glob([
        "Sources/CNIODarwin/**/*.c",
    ]),
    hdrs = glob([
        "Sources/CNIODarwin/**/*.h",
    ]),
    defines = [
        "__APPLE_USE_RFC_3542",
    ],
    includes = ["Sources/CNIODarwin/include"],
    tags = ["swift_module=CNIODarwin"],
)

cc_library(
    name = "CNIOLLHTTP",
    srcs = glob([
        "Sources/CNIOLLHTTP/**/*.c",
    ]),
    hdrs = glob([
        "Sources/CNIOLLHTTP/**/*.h",
    ]),
    copts = [],
    defines = [
        "LLHTTP_STRICT_MODE",
    ],
    includes = ["Sources/CNIOLLHTTP/include"],
    tags = ["swift_module=CNIOLLHTTP"],
)

cc_library(
    name = "CNIOLinux",
    srcs = glob([
        "Sources/CNIOLinux/**/*.c",
    ]),
    hdrs = glob([
        "Sources/CNIOLinux/**/*.h",
    ]),
    copts = [],
    includes = ["Sources/CNIOLinux/include"],
    tags = ["swift_module=CNIOLinux"],
)

cc_library(
    name = "CNIOSHA1",
    srcs = glob([
        "Sources/CNIOSHA1/**/*.c",
    ]),
    hdrs = [
        "Sources/CNIOSHA1/**/*.h",
    ],
    copts = [],
    includes = ["Sources/CNIOSHA1/include"],
    tags = ["swift_module=CNIOSHA1"],
)

cc_library(
    name = "CNIOWindows",
    srcs = glob([
        "Sources/CNIOWindows/**/*.c",
    ]),
    hdrs = glob([
        "Sources/CNIOWindows/**/*.h",
    ]),
    copts = [],
    includes = ["Sources/CNIOWindows/include"],
    tags = ["swift_module=CNIOWindows"],
)

swift_library(
    name = "NIO",
    srcs = glob([
        "Sources/NIO/*.swift",
    ]),
    module_name = "NIO",
    visibility = ["//visibility:public"],
    deps = [
        ":NIOCore",
        ":NIOEmbedded",
        ":NIOPosix",
    ],
)

swift_library(
    name = "NIOConcurrencyHelpers",
    srcs = glob([
        "Sources/NIOConcurrencyHelpers/*.swift",
    ]),
    module_name = "NIOConcurrencyHelpers",
    visibility = ["//visibility:public"],
    deps = [
        ":CNIOAtomics",
    ],
)

swift_library(
    name = "NIOCore",
    srcs = glob([
        "Sources/NIOCore/**/*.swift",
    ]),
    copts = [],
    module_name = "NIOCore",
    visibility = ["//visibility:public"],
    deps = [
        ":CNIOLinux",
        ":CNIOWindows",
        ":NIOConcurrencyHelpers",
        "@SwiftCollections//:Collections",
        "@swift-atomics//:SwiftAtomics",
    ],
)

swift_library(
    name = "NIOEmbedded",
    srcs = glob([
        "Sources/NIOEmbedded/*.swift",
    ]),
    copts = [],
    module_name = "NIOEmbedded",
    visibility = ["//visibility:public"],
    deps = [
        ":NIOConcurrencyHelpers",
        ":NIOCore",
        ":_NIODataStructures",
        "@swift-atomics//:SwiftAtomics",
    ],
)

swift_library(
    name = "NIOFoundationCompat",
    srcs = glob([
        "Sources/NIOFoundationCompat/*.swift",
    ]),
    module_name = "NIOFoundationCompat",
    visibility = ["//visibility:public"],
    deps = [
        ":NIO",
        ":NIOCore",
    ],
)

swift_library(
    name = "NIOHTTP1",
    srcs = glob([
        "Sources/NIOHTTP1/*.swift",
    ]),
    module_name = "NIOHTTP1",
    visibility = ["//visibility:public"],
    deps = [
        ":CNIOLLHTTP",
        ":NIO",
        ":NIOConcurrencyHelpers",
        ":NIOCore",
    ],
)

swift_library(
    name = "NIOPosix",
    srcs = glob([
        "Sources/NIOPosix/*.swift",
    ]),
    copts = [],
    module_name = "NIOPosix",
    visibility = ["//visibility:public"],
    deps = [
        ":CNIODarwin",
        ":CNIOLinux",
        ":CNIOWindows",
        ":NIOConcurrencyHelpers",
        ":NIOCore",
        ":_NIODataStructures",
        "@swift-atomics//:SwiftAtomics",
    ],
)

swift_library(
    name = "NIOTLS",
    srcs = glob([
        "Sources/NIOTLS/*.swift",
    ]),
    module_name = "NIOTLS",
    visibility = ["//visibility:public"],
    deps = [
        ":NIO",
        ":NIOCore",
        "@SwiftCollections//:Collections",
    ],
)

swift_library(
    name = "_NIODataStructures",
    srcs = glob([
        "Sources/_NIODataStructures/*.swift",
    ]),
    module_name = "_NIODataStructures",
    visibility = ["//visibility:public"],
    deps = [],
)
