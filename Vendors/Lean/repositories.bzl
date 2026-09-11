load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def lean_repositories():
    http_archive(
        name = "lean4",
        build_file = "//Vendors/Lean:lean4.BUILD.bazel",
        patch_args = ["-p1"],
        patches = ["//Vendors/Lean:apple.patch"],
        sha256 = "561b500da16419aec0baeba5dc47ad11979b5bdcd30a02aafc88c3eed599b766",
        strip_prefix = "lean4-4.33.1",
        urls = ["https://github.com/leanprover/lean4/archive/refs/tags/v4.33.1.tar.gz"],
    )

    http_archive(
        name = "lean4_toolchain",
        build_file = "//Vendors/Lean:lean4_toolchain.BUILD.bazel",
        sha256 = "88c45aad985b5d2a8d925fe10bd1296bd35f66f408480ab182d3facccd065a9d",
        strip_prefix = "lean-4.33.1-darwin_aarch64",
        type = "tar.zst",
        urls = [
            "https://github.com/leanprover/lean4/releases/download/v4.33.1/lean-4.33.1-darwin_aarch64.tar.zst",
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
        sha256 = "ac5ba94172b60823215a22b87ae923c5b05ef0cdd9047df2a832c16da02a6447",
        strip_prefix = "mimalloc-2.2.3",
        urls = ["https://github.com/microsoft/mimalloc/archive/refs/tags/v2.2.3.tar.gz"],
    )
