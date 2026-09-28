load("@rules_rust//rust:defs.bzl", "rust_static_library")

package(default_visibility = ["//visibility:public"])

objc_library(
    name = "api",
    hdrs = ["include/pdf_inspector.h"],
    includes = ["include"],
    module_name = "CPDFInspector",
    deps = [":bridge"],
)

rust_static_library(
    name = "bridge",
    srcs = ["bridge.rs"],
    crate_name = "pdf_inspector_bridge",
    crate_root = "bridge.rs",
    edition = "2021",
    deps = ["@pdf_inspector"],
)

exports_files([
    "Cargo.lock",
    "Cargo.toml",
    "cargo-bazel-lock.json",
    "fake.rs",
    "password.patch",
    "pdf-inspector.BUILD.bazel",
])
