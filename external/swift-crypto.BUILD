load(
    "@build_bazel_rules_swift//swift:swift.bzl",
    "swift_library",
)

cc_library(
    name = "CCryptoBoringSSL",
    srcs = glob([
        "Sources/CCryptoBoringSSL/**/*.h",
        "Sources/CCryptoBoringSSL/**/*.c",
        "Sources/CCryptoBoringSSL/**/*.cc",
        "Sources/CCryptoBoringSSL/**/*.S",
        "Sources/CCryptoBoringSSL/**/*.inc",
    ]),
    hdrs = glob([
        "Sources/CCryptoBoringSSL/include/**/*.h",
    ]),
    copts = [],
    includes = ["Sources/CCryptoBoringSSL/include"],
    tags = ["swift_module=CCryptoBoringSSL"],
    visibility = ["//visibility:public"],
    deps = [],
)

cc_library(
    name = "CCryptoBoringSSLShims",
    srcs = glob([
        "Sources/CCryptoBoringSSLShims/**/*.c",
    ]),
    hdrs = glob([
        "Sources/CCryptoBoringSSLShims/include/**/*.h",
    ]),
    copts = [],
    includes = ["Sources/CCryptoBoringSSLShims/include"],
    tags = ["swift_module=CCryptoBoringSSLShims"],
    visibility = ["//visibility:public"],
    deps = [":CCryptoBoringSSL"],
)

swift_library(
    name = "CryptoBoringWrapper",
    srcs = glob(
        [
            "Sources/CryptoBoringWrapper/**/*.swift",
        ],
        exclude = [
            "Sources/CryptoBoringWrapper/CMakeLists.txt",
            "Sources/CryptoBoringWrapper/PrivacyInfo.xcprivacy",
        ],
    ),
    defines = [
        "CRYPTO_IN_SWIFTPM",
        "CRYPTO_IN_SWIFTPM_FORCE_BUILD_API",
    ],
    module_name = "CryptoBoringWrapper",
    visibility = ["//visibility:public"],
    deps = [
        ":CCryptoBoringSSL",
        ":CCryptoBoringSSLShims",
    ],
)

swift_library(
    name = "Crypto",
    srcs = glob(
        [
            "Sources/Crypto/**/*.swift",
        ],
        exclude = [
            "Sources/Crypto/CMakeLists.txt",
            "Sources/Crypto/AEADs/Nonces.swift.gyb",
            "Sources/Crypto/Digests/Digests.swift.gyb",
            "Sources/Crypto/Key Agreement/ECDH.swift.gyb",
            "Sources/Crypto/Signatures/ECDSA.swift.gyb",
            "Sources/Crypto/PrivacyInfo.xcprivacy",
        ],
    ),
    defines = [
        "MODULE_IS_CRYPTO",
    ],
    module_name = "Crypto",
    visibility = ["//visibility:public"],
    deps = [
        ":CCryptoBoringSSL",
        ":CCryptoBoringSSLShims",
        ":CryptoBoringWrapper",
    ],
)
