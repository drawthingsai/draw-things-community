load(
    "@build_bazel_rules_swift//swift:swift.bzl",
    "swift_interop_hint",
    "swift_library",
)

SWIFT_NIO_COPTS = [
    "-package-name",
    "swift_nio",
    "-enable-experimental-feature",
    "Lifetimes",
    "-enable-upcoming-feature",
    "MemberImportVisibility",
]

cc_library(
    name = "CNIOAtomics",
    srcs = glob([
        "Sources/CNIOAtomics/**/*.c",
    ]),
    hdrs = glob([
        "Sources/CNIOAtomics/**/*.h",
    ]),
    aspect_hints = [":CNIOAtomics_swift_interop"],
    copts = [],
    defines = [
        "_GNU_SOURCE",
    ],
    includes = ["Sources/CNIOAtomics/include"],
    tags = ["swift_module=CNIOAtomics"],
)

swift_interop_hint(
    name = "CNIOAtomics_swift_interop",
    module_name = "CNIOAtomics",
)

cc_library(
    name = "CNIODarwin",
    srcs = glob([
        "Sources/CNIODarwin/**/*.c",
    ]),
    hdrs = glob([
        "Sources/CNIODarwin/**/*.h",
    ]),
    aspect_hints = [":CNIODarwin_swift_interop"],
    defines = [
        "__APPLE_USE_RFC_3542",
    ],
    includes = ["Sources/CNIODarwin/include"],
    tags = ["swift_module=CNIODarwin"],
)

swift_interop_hint(
    name = "CNIODarwin_swift_interop",
    module_name = "CNIODarwin",
)

cc_library(
    name = "CNIOFreeBSD",
    srcs = glob([
        "Sources/CNIOFreeBSD/**/*.c",
    ]),
    hdrs = glob([
        "Sources/CNIOFreeBSD/**/*.h",
    ]),
    aspect_hints = [":CNIOFreeBSD_swift_interop"],
    copts = [],
    includes = ["Sources/CNIOFreeBSD/include"],
    tags = ["swift_module=CNIOFreeBSD"],
)

swift_interop_hint(
    name = "CNIOFreeBSD_swift_interop",
    module_name = "CNIOFreeBSD",
)

cc_library(
    name = "CNIOLLHTTP",
    srcs = glob([
        "Sources/CNIOLLHTTP/**/*.c",
    ]),
    hdrs = glob([
        "Sources/CNIOLLHTTP/**/*.h",
    ]),
    aspect_hints = [":CNIOLLHTTP_swift_interop"],
    copts = [],
    defines = [
        "_GNU_SOURCE",
        "LLHTTP_STRICT_MODE",
    ],
    includes = ["Sources/CNIOLLHTTP/include"],
    tags = ["swift_module=CNIOLLHTTP"],
)

swift_interop_hint(
    name = "CNIOLLHTTP_swift_interop",
    module_name = "CNIOLLHTTP",
)

cc_library(
    name = "CNIOLinux",
    srcs = glob([
        "Sources/CNIOLinux/**/*.c",
    ]),
    hdrs = glob([
        "Sources/CNIOLinux/**/*.h",
    ]),
    aspect_hints = [":CNIOLinux_swift_interop"],
    copts = [],
    defines = [
        "_GNU_SOURCE",
    ],
    includes = ["Sources/CNIOLinux/include"],
    tags = ["swift_module=CNIOLinux"],
)

swift_interop_hint(
    name = "CNIOLinux_swift_interop",
    module_name = "CNIOLinux",
)

cc_library(
    name = "CNIOOpenBSD",
    srcs = glob([
        "Sources/CNIOOpenBSD/**/*.c",
    ]),
    hdrs = glob([
        "Sources/CNIOOpenBSD/**/*.h",
    ]),
    aspect_hints = [":CNIOOpenBSD_swift_interop"],
    copts = [],
    includes = ["Sources/CNIOOpenBSD/include"],
    tags = ["swift_module=CNIOOpenBSD"],
)

swift_interop_hint(
    name = "CNIOOpenBSD_swift_interop",
    module_name = "CNIOOpenBSD",
)

cc_library(
    name = "CNIOPosix",
    srcs = glob([
        "Sources/CNIOPosix/**/*.c",
    ]),
    hdrs = glob([
        "Sources/CNIOPosix/**/*.h",
    ]),
    aspect_hints = [":CNIOPosix_swift_interop"],
    copts = [],
    defines = [
        "_GNU_SOURCE",
    ],
    includes = ["Sources/CNIOPosix/include"],
    tags = ["swift_module=CNIOPosix"],
)

swift_interop_hint(
    name = "CNIOPosix_swift_interop",
    module_name = "CNIOPosix",
)

cc_library(
    name = "CNIOSHA1",
    srcs = glob([
        "Sources/CNIOSHA1/**/*.c",
    ]),
    hdrs = [
        "Sources/CNIOSHA1/**/*.h",
    ],
    aspect_hints = [":CNIOSHA1_swift_interop"],
    copts = [],
    includes = ["Sources/CNIOSHA1/include"],
    tags = ["swift_module=CNIOSHA1"],
)

swift_interop_hint(
    name = "CNIOSHA1_swift_interop",
    module_name = "CNIOSHA1",
)

cc_library(
    name = "CNIOWASI",
    srcs = glob([
        "Sources/CNIOWASI/**/*.c",
    ]),
    hdrs = glob([
        "Sources/CNIOWASI/**/*.h",
    ]),
    aspect_hints = [":CNIOWASI_swift_interop"],
    copts = [],
    includes = ["Sources/CNIOWASI/include"],
    tags = ["swift_module=CNIOWASI"],
)

swift_interop_hint(
    name = "CNIOWASI_swift_interop",
    module_name = "CNIOWASI",
)

cc_library(
    name = "CNIOWindows",
    srcs = glob([
        "Sources/CNIOWindows/**/*.c",
    ]),
    hdrs = glob([
        "Sources/CNIOWindows/**/*.h",
    ]),
    aspect_hints = [":CNIOWindows_swift_interop"],
    copts = [],
    includes = ["Sources/CNIOWindows/include"],
    tags = ["swift_module=CNIOWindows"],
)

swift_interop_hint(
    name = "CNIOWindows_swift_interop",
    module_name = "CNIOWindows",
)

swift_library(
    name = "NIO",
    srcs = glob([
        "Sources/NIO/*.swift",
    ]),
    copts = SWIFT_NIO_COPTS,
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
    copts = SWIFT_NIO_COPTS,
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
    copts = SWIFT_NIO_COPTS,
    module_name = "NIOCore",
    visibility = ["//visibility:public"],
    deps = [
        ":CNIODarwin",
        ":CNIOFreeBSD",
        ":CNIOLinux",
        ":CNIOOpenBSD",
        ":CNIOWASI",
        ":CNIOWindows",
        ":NIOConcurrencyHelpers",
        ":_NIOBase64",
        ":_NIODataStructures",
        "@SwiftCollections//:Collections",
        "@swift-atomics//:SwiftAtomics",
    ],
)

swift_library(
    name = "NIOEmbedded",
    srcs = glob([
        "Sources/NIOEmbedded/*.swift",
    ]),
    copts = SWIFT_NIO_COPTS,
    module_name = "NIOEmbedded",
    visibility = ["//visibility:public"],
    deps = [
        ":NIOConcurrencyHelpers",
        ":NIOCore",
        ":_NIODataStructures",
        "@SwiftCollections//:Collections",
        "@swift-atomics//:SwiftAtomics",
    ],
)

swift_library(
    name = "NIOFoundationEssentialsCompat",
    srcs = glob([
        "Sources/NIOFoundationEssentialsCompat/*.swift",
    ]),
    copts = SWIFT_NIO_COPTS,
    module_name = "NIOFoundationEssentialsCompat",
    visibility = ["//visibility:public"],
    deps = [
        ":NIOCore",
    ],
)

swift_library(
    name = "NIOFoundationCompat",
    srcs = glob([
        "Sources/NIOFoundationCompat/*.swift",
    ]),
    copts = SWIFT_NIO_COPTS,
    module_name = "NIOFoundationCompat",
    visibility = ["//visibility:public"],
    deps = [
        ":NIO",
        ":NIOFoundationEssentialsCompat",
    ],
)

swift_library(
    name = "NIOHTTP1",
    srcs = glob([
        "Sources/NIOHTTP1/*.swift",
    ]),
    copts = SWIFT_NIO_COPTS,
    module_name = "NIOHTTP1",
    visibility = ["//visibility:public"],
    deps = [
        ":CNIOLLHTTP",
        ":NIO",
        ":NIOConcurrencyHelpers",
        ":NIOCore",
        "@SwiftCollections//:Collections",
    ],
)

swift_library(
    name = "NIOPosix",
    srcs = glob([
        "Sources/NIOPosix/*.swift",
    ]),
    copts = SWIFT_NIO_COPTS,
    module_name = "NIOPosix",
    visibility = ["//visibility:public"],
    deps = [
        ":CNIODarwin",
        ":CNIOLinux",
        ":CNIOOpenBSD",
        ":CNIOPosix",
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
    copts = SWIFT_NIO_COPTS,
    module_name = "NIOTLS",
    visibility = ["//visibility:public"],
    deps = [
        ":NIO",
        ":NIOCore",
        "@SwiftCollections//:Collections",
    ],
)

swift_library(
    name = "_NIOBase64",
    srcs = glob([
        "Sources/_NIOBase64/*.swift",
    ]),
    copts = SWIFT_NIO_COPTS,
    module_name = "_NIOBase64",
    visibility = ["//visibility:public"],
    deps = [],
)

swift_library(
    name = "_NIODataStructures",
    srcs = glob([
        "Sources/_NIODataStructures/*.swift",
    ]),
    copts = SWIFT_NIO_COPTS,
    module_name = "_NIODataStructures",
    visibility = ["//visibility:public"],
    deps = [],
)
