# FFmpeg and FFprobe command integration

FFmpeg 9.0.2 is downloaded from the upstream release archive with a SHA-256 pin
in `repositories.bzl`. Bazel builds it from source for the selected Apple SDK,
architecture, and deployment target. No downloaded executable is launched.

`//Vendors/FFmpeg:FFmpeg` exposes the synchronous C functions `FFmpegCommandRun`
and `FFprobeCommandRun`, sharing one context type and one libav build.
`//Vendors/FFmpeg:command` registers conventional `ffmpeg_main(argc, argv)` and
`ffprobe_main(argc, argv)` adapters through the existing ios_system registry.
Local Code schedules both commands in the FFmpeg operation queue. A mutex inside
the library enforces serialization even for calls from OSH, Python, or other hosts.

## Command syntax and components

The command uses FFmpeg's original `ffmpeg_parse_options`, option tables, stream
specifiers, and filter parser. FFprobe uses its original `parse_options`, option
tables, `-show_entries`/stream selection handlers, and output formatters (including
JSON and CSV). The adapter does not parse or rewrite arguments.
The build enables the built-in components available without external codec
dependencies, plus VideoToolbox, AudioToolbox, SecureTransport, and zlib. Native
ARM assembly and FFmpeg worker threads remain enabled. Device capture/output is
disabled, except the `lavfi` input device. `ffplay` is not built.
Use `ffmpeg -codecs`, `-filters`, `-formats`, and `-protocols` to inspect the
compiled capabilities. GPL and nonfree components are not enabled.

Examples:

```sh
ffmpeg -i input.mov -c:v h264_videotoolbox -c:a aac output.mp4
ffmpeg -i input.mp4 -vf scale=640:-2 -frames:v 1 preview.png
ffmpeg -i input.wav -ar 16000 -ac 1 -f s16le pipe:1 > audio.raw
ffprobe -v error -show_format -show_streams -of json input.mp4
ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of csv=p=0 input.mp4
```

## Invocation ownership

The execution lock covers option initialization, file opening, transcoding,
worker shutdown, and cleanup. The host context and its strings/streams are
borrowed until return. Worker threads access the same active invocation rather
than ios_system's command-thread TLS. The adapter snapshots the environment and
directory aliases; relative filesystem paths use the session directory.

`ffmpeg-ios.patch` resets mutable frontend options and statistics, clears closed
handles, closes progress/report streams on error paths, and releases embedded
resources. It preserves upstream parsing and media execution. The archive binds
and hides all implementation symbols except `FFmpegCommandRun` and
`FFprobeCommandRun`, so libav logging, CPU overrides, and allocation settings
cannot affect another libav consumer.

FFprobe resets its section selections, format options, counters, and logging
state between calls. Decoder setup errors return through normal cleanup instead
of exiting the process. Failed formatter initialization closes the already-open
writer. Shared help dispatch and program metadata select the active frontend
under the same execution lock.

Cancellation is cooperative through the host token, the existing FFmpeg I/O
interrupt callback, and scheduler shutdown. Cancellation while queued does not
enter either frontend. FFprobe uses the host cancellation token in its input I/O
interrupt callback and packet loop. Pipe reads/writes poll for cancellation.
Neither frontend installs process signal handlers or alters terminal modes; keyboard interaction is
disabled, while media on stdin remains supported. `-timelimit` returns an
unsupported error rather than changing the application's CPU resource limit.

The current OSH shell buffers pipeline stages with its existing 16 MiB limit.
Sequential FFmpeg/FFprobe stages therefore work within that limit; this is not an
unbounded streaming shell. Hosts must not connect simultaneous FFmpeg/FFprobe
invocations with a live bounded pipe: serialization would prevent both stages
from making progress.

## Validation and upgrades

```sh
bazel test //Vendors/FFmpeg:FFmpegCommandTests //Vendors/FFmpeg:IOSSystemFFmpegTests --macos_minimum_os=13.5
bazel test //Vendors/FFmpeg:FFmpegCommandTests --config=asan --macos_minimum_os=13.5
bazel build //Apps/LocalCode:LocalCode --ios_multi_cpus=arm64
```

The tests exercise the actual upstream parser, audio/video sample counts,
repeated success and partial initialization failure, option reset, descriptor
and live-allocation growth, serialized admission, cancellation, signal
dispositions, FFprobe JSON metadata and frame counts, and real ios_system/OSH
streams, environment and paths.

When upgrading, update the release checksum and rebase the patch. Re-audit
file-scope and function-static mutable state in fftools, including shared
command utilities, logging, resources and newly introduced frontend modules.
Retain the same upstream parser and run both test suites and the iOS app build.

## Notices

Upstream: https://ffmpeg.org/
Source: https://ffmpeg.org/releases/ffmpeg-9.0.2.tar.xz
SHA-256: `8c3850283eb25fa026482078a04051e0be17347b09ef81a0849bec15a96e002e`

FFmpeg is copyright its respective contributors. The build uses the LGPL 2.1 or
later configuration; see `COPYING.LGPLv2.1` and upstream `LICENSE.md` for the
complete license and component notices. This software is based in part on the
work of the Independent JPEG Group. The three IJG-derived files identified in
`LICENSE.md` are unmodified. Local modifications are recorded in
`ffmpeg-ios.patch` and the host adapter sources alongside it.
