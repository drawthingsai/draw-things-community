load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@rules_rust//crate_universe:defs.bzl", "crates_repository")

def pdf_inspector_repositories():
    crates_repository(
        name = "pdf_inspector_crates",
        cargo_lockfile = "//Vendors/PDFInspector:Cargo.lock",
        lockfile = "//Vendors/PDFInspector:cargo-bazel-lock.json",
        manifests = ["//Vendors/PDFInspector:Cargo.toml"],
        rust_version = "1.89.0",
        supported_platform_triples = [
            "aarch64-apple-darwin",
            "aarch64-apple-ios",
            "aarch64-apple-ios-sim",
            "x86_64-apple-darwin",
            "x86_64-apple-ios",
        ],
    )
    http_archive(
        name = "pdf_inspector",
        build_file = "//Vendors/PDFInspector:pdf-inspector.BUILD.bazel",
        patch_args = ["-p1"],
        patches = ["//Vendors/PDFInspector:password.patch"],
        sha256 = "814d801ce6032f47a68e758c643b983b07e5949c674249e2a0ed463c6246264b",
        strip_prefix = "pdf-inspector-2dbd16b3ea4ea943af2802bd72e45dd264eacb9f",
        urls = ["https://github.com/firecrawl/pdf-inspector/archive/2dbd16b3ea4ea943af2802bd72e45dd264eacb9f.tar.gz"],
    )
