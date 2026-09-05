package(default_visibility = ["//visibility:public"])

objc_library(
    name = "Z3Bridge",
    srcs = ["Z3Bridge.cpp"],
    hdrs = ["Z3Bridge.h"],
    copts = ["-std=c++20"],
    module_name = "Z3Bridge",
    deps = ["@z3"],
)

cc_test(
    name = "Z3BridgeTests",
    srcs = ["Z3BridgeTests.cpp"],
    deps = [":Z3Bridge"],
)

exports_files([
    "generate.py",
    "repositories.bzl",
    "z3.BUILD.bazel",
])
