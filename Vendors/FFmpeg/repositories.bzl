load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def ffmpeg_repositories():
    http_archive(
        name = "ffmpeg",
        urls = ["https://ffmpeg.org/releases/ffmpeg-9.0.2.tar.xz"],
        sha256 = "8c3850283eb25fa026482078a04051e0be17347b09ef81a0849bec15a96e002e",
        strip_prefix = "ffmpeg-9.0.2",
        build_file_content = 'filegroup(name = "sources", srcs = glob(["**"], exclude = ["BUILD", "BUILD.bazel"]), visibility = ["//visibility:public"])',
    )
