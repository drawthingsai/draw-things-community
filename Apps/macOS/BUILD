load("@build_bazel_rules_apple//apple:macos.bzl", "macos_command_line_application")

macos_command_line_application(
    name = "gRPCServerCLI-macOS",
    bundle_id = "com.liuliu.draw-things-grpc-server-cli",
    linkopts = [
        "-framework",
        "Accelerate",
        "-framework",
        "Metal",
        "-framework",
        "MetalPerformanceShaders",
        "-framework",
        "MetalPerformanceShadersGraph",
    ],
    minimum_os_version = "13.0",
    visibility = ["//visibility:public"],
    deps = [
        "//Apps:gRPCServerCLILib",
    ],
)
