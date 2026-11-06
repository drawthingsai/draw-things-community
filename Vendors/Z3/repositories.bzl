load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")


def z3_repositories():
    http_archive(
        name = "z3",
        build_file = "//Vendors/Z3:z3.BUILD.bazel",
        sha256 = "c68c3e5e4810b16126b8cb4c47eee85c1ac3e24a81914c8e371b40de9dd33ac7",
        strip_prefix = "z3-z3-4.16.0",
        urls = ["https://github.com/Z3Prover/z3/archive/refs/tags/z3-4.16.0.tar.gz"],
    )
