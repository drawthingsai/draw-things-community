package(default_visibility = ["//visibility:public"])

# Vendors/curl/CurlPort.h is forced into every source: ios_system streams,
# environment and session-relative paths. Configuration and the BoringSSL
# header shims come from Vendors/curl as well.
COPTS = [
    "-DHAVE_CONFIG_H",
    "-DCURL_STATICLIB",
    "-DIOS_SYSTEM_CURL=1",
    "-include",
    "CurlPort.h",
    "-Wno-deprecated-declarations",
    "-Wno-unused-function",
    "-Wno-unused-variable",
]

# TLS is swift-nio-ssl's BoringSSL, already in the app, with certificates
# verified by Apple's trust store (USE_APPLE_SECTRUST).
objc_library(
    name = "libcurl",
    srcs = glob([
        "lib/**/*.c",
        "lib/**/*.h",
    ]),
    hdrs = glob(["include/curl/*.h"]),
    copts = COPTS + [
        "-DBUILDING_LIBCURL",
        "-DCURL_HIDDEN_SYMBOLS",
    ],
    includes = [
        "include",
        "lib",
    ],
    linkopts = ["-lz"],
    sdk_frameworks = [
        "CoreFoundation",
        "Security",
    ] + select({
        # macOS reads the system proxy settings (SCDynamicStoreCopyProxies).
        "@platforms//os:macos": ["SystemConfiguration"],
        "//conditions:default": [],
    }),
    deps = [
        "@//Vendors/curl:boringssl_headers",
        "@//Vendors/curl:config",
        "@//Vendors/curl:port",
        "@//Vendors/ios_system:ios_system_support",
        "@SwiftNIOSSL//:CNIOBoringSSL",
    ],
)

# The curl command line tool. Vendors/curl:curl wraps its main as curl_main.
objc_library(
    name = "tool",
    srcs = glob(
        [
            "src/**/*.c",
            "src/**/*.h",
        ],
        exclude = ["src/curlinfo.c"],
    ),
    copts = COPTS + [
        "-DCURL_PORT_TOOL",
        "-DUSE_MANUAL",
        "-Dmain=curl_tool_main",
    ],
    includes = ["src"],
    deps = [
        ":libcurl",
        "@//Vendors/curl:port",
        "@//Vendors/ios_system:api",
    ],
)
