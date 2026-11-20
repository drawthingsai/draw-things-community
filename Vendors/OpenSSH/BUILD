package(default_visibility = ["//visibility:public"])

exports_files(["ios.patch"])

# OpenSSH includes <openssl/...>; these map it to swift-nio-ssl's prefixed
# BoringSSL, the same copy curl uses.
cc_library(
    name = "boringssl_headers",
    hdrs = glob(["BoringSSL/openssl/*.h"]),
    includes = ["BoringSSL"],
    deps = ["@SwiftNIOSSL//:CNIOBoringSSL"],
)

cc_library(
    name = "config",
    hdrs = [
        "Config/config.h",
        "Config/ios_compat_config.h",
    ],
    includes = ["Config"],
)

cc_library(
    name = "compat_headers",
    includes = ["CompatInclude"],
    textual_hdrs = [
        "CompatInclude/OpenSSHPort.h",
        "CompatInclude/endian.h",
        "CompatInclude/glob.h",
        "CompatInclude/netgroup.h",
        "CompatInclude/nlist.h",
        "CompatInclude/readpassphrase.h",
        "CompatInclude/sys/queue.h",
        "CompatInclude/sys/tree.h",
    ],
    deps = [
        "//Vendors/ios_system:api",
        "//Vendors/ios_system:ios_system_support",
    ],
)
