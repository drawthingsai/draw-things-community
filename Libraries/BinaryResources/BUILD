load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")
load("//Tools:Packager/packager.bzl", "resource_library")

resource_library(
    name = "Resources",
    srcs = [
        "Resources/bpe_simple_vocab_16e6.txt",
        "Resources/chatglm3.spiece.model",
        "Resources/isrgrootx1.pem",
        "Resources/merges.txt",
        "Resources/pile_t5.spiece.model",
        "Resources/root_ca.crt",
        "Resources/server_crt.crt",
        "Resources/server_key.key",
        "Resources/t5.spiece.model",
        "Resources/vocab.json",
        "Resources/vocab_16e6.json",
        "Resources/xlmroberta.bpe.model",
    ],
)

cc_library(
    name = "CResources",
    hdrs = ["Sources/Module.h"],
    tags = ["swift_module=C_Resources"],
    deps = [":Resources"],
)

swift_library(
    name = "BinaryResources",
    srcs = glob(["Sources/**/*.swift"]),
    module_name = "BinaryResources",
    visibility = ["//visibility:public"],
    deps = [":CResources"],
)
