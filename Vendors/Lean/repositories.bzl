load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def lean_repositories():
    http_archive(
        name = "lean4",
        build_file = "//Vendors/Lean:lean4.BUILD.bazel",
        patch_args = ["-p1"],
        patches = ["//Vendors/Lean:apple.patch"],
        sha256 = "6747b099d03d3675f2716e7cf5924a16bfbf75822b13a3d36f51f933ffcc307b",
        strip_prefix = "lean4-4.35.0",
        urls = ["https://github.com/leanprover/lean4/archive/refs/tags/v4.35.0.tar.gz"],
    )

    http_archive(
        name = "lean4_toolchain",
        build_file = "//Vendors/Lean:lean4_toolchain.BUILD.bazel",
        sha256 = "c0c6f34f35ba1dcc5c4a5fd106fefab9eb5e8469712624dd2c868d51b63177c2",
        strip_prefix = "lean-4.35.0-darwin_aarch64",
        type = "tar.zst",
        urls = [
            "https://github.com/leanprover/lean4/releases/download/v4.35.0/lean-4.35.0-darwin_aarch64.tar.zst",
        ],
    )

    http_archive(
        name = "lean_gmp",
        build_file = "//Vendors/Lean:gmp.BUILD.bazel",
        sha256 = "a3c2b80201b89e68616f4ad30bc66aee4927c3ce50e33929ca819d5c43538898",
        strip_prefix = "gmp-6.3.0",
        urls = ["https://gmplib.org/download/gmp/gmp-6.3.0.tar.xz"],
    )

    http_archive(
        name = "lean_libuv",
        build_file = "//Vendors/Lean:libuv.BUILD.bazel",
        sha256 = "8c253adb0f800926a6cbd1c6576abae0bc8eb86a4f891049b72f9e5b7dc58f33",
        strip_prefix = "libuv-1.48.0",
        urls = ["https://github.com/libuv/libuv/archive/refs/tags/v1.48.0.tar.gz"],
    )

    http_archive(
        name = "lean_mimalloc",
        build_file = "//Vendors/Lean:mimalloc.BUILD.bazel",
        sha256 = "19a43af0645c57d348e729d5b31e23e912582911bb1047f795790834d3416221",
        strip_prefix = "mimalloc-3.4.5",
        urls = ["https://github.com/microsoft/mimalloc/archive/refs/tags/v3.4.5.tar.gz"],
    )
