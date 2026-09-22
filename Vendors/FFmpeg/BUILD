load("@build_bazel_rules_apple//apple:macos.bzl", "macos_command_line_application")
load("@build_bazel_rules_apple//apple:resources.bzl", "apple_resource_group")
load(":archive.bzl", "ffmpeg_archive")

package(default_visibility = ["//visibility:public"])

apple_resource_group(
    name = "notices",
    resources = [
        "COPYING.LGPLv2.1",
        "LICENSE.md",
        "README.md",
    ],
)

filegroup(
    name = "support",
    srcs = [
        "FFmpegCommand.h",
        "ffmpeg_host.c",
        "ffmpeg_host.h",
    ],
)

ffmpeg_archive(name = "archive")

cc_import(
    name = "private_archive",
    static_library = ":archive",
)

objc_library(
    name = "FFmpeg",
    hdrs = ["FFmpegCommand.h"],
    sdk_dylibs = [
        "z",
        "iconv",
        "bz2",
    ],
    sdk_frameworks = [
        "AudioToolbox",
        "CoreFoundation",
        "CoreMedia",
        "CoreVideo",
        "Security",
        "VideoToolbox",
    ],
    deps = [":private_archive"],
)

objc_library(
    name = "command",
    srcs = ["ios_system_ffmpeg.m"],
    deps = [
        ":FFmpeg",
        "//Vendors/ios_system:api",
    ],
)

cc_test(
    name = "FFmpegCommandTests",
    size = "medium",
    srcs = ["FFmpegCommandTests.c"],
    deps = [":FFmpeg"],
)

objc_library(
    name = "ios_system_validation_main",
    srcs = ["IOSSystemFFmpegTests.m"],
    sdk_frameworks = ["Foundation"],
    deps = ["//Vendors/ios_system"],
)

macos_command_line_application(
    name = "ios_system_validation",
    bundle_id = "com.liuliu.ffmpeg-validation",
    minimum_os_version = "13.5",
    deps = [
        ":ios_system_validation_main",
        "//Vendors/NodeJS:NodeMobile",
        "//Vendors/PythonIOS:PythonRuntime",
    ],
)

sh_test(
    name = "IOSSystemFFmpegTests",
    size = "medium",
    srcs = ["run-ios-system-tests.sh"],
    data = [":ios_system_validation"],
)
