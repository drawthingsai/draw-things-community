load("@build_bazel_rules_swift//swift:swift.bzl", "swift_binary", "swift_library")
load("@rules_oci//oci:defs.bzl", "oci_image", "oci_push", "oci_tarball")
load("@rules_pkg//:pkg.bzl", "pkg_tar")

pkg_tar(
    name = "gRPCServerCLI_layer",
    srcs = [
        "//Apps:gRPCServerCLI",
    ],
    package_dir = "/usr/local/bin",
)

oci_image(
    name = "gRPCServerCLI_image",
    base = "@cuda12.4.1_swift5.10.1_ubuntu22.04",
    tars = [
        ":gRPCServerCLI_layer",
    ],
)

oci_tarball(
    name = "gRPCServerCLI_tar",
    image = ":gRPCServerCLI_image",
    repo_tags = ["draw-things-grpc-server-cli:latest"],
)

oci_push(
    name = "push_gRPCServerCLI_server",
    image = ":gRPCServerCLI_image",
    remote_tags = ["latest"],
    repository = "docker.io/drawthingsai/grpc-server",  # adjust to your org/repo
)
