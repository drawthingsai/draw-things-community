load("@build_bazel_rules_apple//apple/testing/default_runner:ios_xctestrun_runner.bzl", "ios_xctestrun_runner")
load("@build_bazel_rules_swift//swift:swift.bzl", "swift_binary")

swift_binary(
    name = "Packager",
    srcs = glob(["Packager/*.swift"]),
    visibility = ["//visibility:public"],
    deps = [
    ],
)

# Runs ios_unit_test targets on a simulator. The default runner asks for an iPhone 6s Plus, which
# current simulator runtimes don't offer.
ios_xctestrun_runner(
    name = "IOSSimulatorRunner",
    device_type = "iPhone 17 Pro",
    visibility = ["//visibility:public"],
)
