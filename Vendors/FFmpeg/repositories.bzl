load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

load(":codec_sources.bzl", "CODEC_SOURCES")

def _host_tools_impl(ctx):
    cmake = ctx.which("cmake")
    if not cmake:
        fail("FFmpeg codec builds require cmake on PATH")
    ctx.symlink(cmake, "cmake")

    python = None
    for name in ["python3", "python3.14", "python3.13", "python3.12", "python3.11", "python3.10"]:
        candidate = ctx.which(name)
        if candidate and ctx.execute([
            candidate,
            "-c",
            "import sys; raise SystemExit(sys.version_info < (3, 10))",
        ]).return_code == 0:
            python = candidate
            break
    if not python:
        fail("FFmpeg codec builds require Python 3.10+ on PATH")
    ctx.symlink(python, "python3")
    ctx.file("BUILD.bazel", 'exports_files(["cmake", "python3"], visibility = ["//visibility:public"])')

_ffmpeg_host_tools = repository_rule(implementation = _host_tools_impl, local = True)

def _vmaf_source_impl(ctx):
    source = CODEC_SOURCES["libvmaf"]
    ctx.download_and_extract(source["url"], sha256 = source["sha256"], stripPrefix = source["strip_prefix"])
    # Upstream has a test-data directory named WORKSPACE.
    ctx.execute(["/bin/mv", "WORKSPACE", "vmaf_test_workspace"])
    ctx.file("BUILD.bazel", 'filegroup(name = "sources", srcs = glob(["**"], exclude = ["BUILD", "BUILD.bazel"]), visibility = ["//visibility:public"])')

_vmaf_source = repository_rule(implementation = _vmaf_source_impl)

def ffmpeg_repositories():
    _ffmpeg_host_tools(name = "ffmpeg_host_tools")
    for name, source in CODEC_SOURCES.items():
        if name == "libvmaf":
            _vmaf_source(name = "ffmpeg_libvmaf")
            continue
        http_archive(
            name = "ffmpeg_" + name.replace("-", "_"),
            urls = [source["url"]],
            sha256 = source["sha256"],
            strip_prefix = source["strip_prefix"],
            patches = ["//Vendors/FFmpeg:x265-neon.patch"] if name == "x265" else [],
            patch_args = ["-p1"],
            build_file_content = 'filegroup(name = "sources", srcs = glob(["**"], exclude = ["BUILD", "BUILD.bazel"]), visibility = ["//visibility:public"])',
        )
    http_archive(
        name = "ffmpeg",
        urls = ["https://ffmpeg.org/releases/ffmpeg-9.0.2.tar.xz"],
        sha256 = "8c3850283eb25fa026482078a04051e0be17347b09ef81a0849bec15a96e002e",
        strip_prefix = "ffmpeg-9.0.2",
        build_file_content = 'filegroup(name = "sources", srcs = glob(["**"], exclude = ["BUILD", "BUILD.bazel"]), visibility = ["//visibility:public"])',
    )
