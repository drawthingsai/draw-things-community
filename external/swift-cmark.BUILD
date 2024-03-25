cc_library(
    name = "cmark-gfm",
    srcs = glob([
        "src/*.c",
        "src/*.h",
        "src/*.inc",
    ]),
    hdrs = glob([
        "src/include/*.h",
    ]),
    includes = [
        "src/include/",
    ],
    tags = ["swift_module=cmark_gfm"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cmark-gfm-extensions",
    srcs = glob([
        "extensions/*.c",
        "extensions/*.h",
    ]),
    hdrs = glob([
        "extensions/include/*.h",
    ]),
    includes = [
        "extensions/include/",
    ],
    tags = ["swift_module=cmark_gfm_extensions"],
    visibility = ["//visibility:public"],
    deps = [":cmark-gfm"],
)
