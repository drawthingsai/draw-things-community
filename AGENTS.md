# Session Learnings

## FrameCompression (VideoToolbox + s4nnc)
- Prefer a synchronous public API for single-frame artifact injection:
  - `FrameCompression.applyCompressionArtifacts(to:codec:quality:) throws -> Tensor<FloatType>`
  - Keep codec selection as an enum parameter (`h264`, `h265`, `jpeg`) instead of separate functions.
- For VideoToolbox single-frame encode/decode, avoid semaphores:
  - Encode with callback capture, then block using `VTCompressionSessionCompleteFrames(...)`.
  - Decode with callback capture, then block using `VTDecompressionSessionWaitForAsynchronousFrames(...)`.
- Validate tensor contract strictly before compression:
  - `.CPU`, `.NHWC(1, height, width, 3)`, quality in `0...100`, and even width/height for H264/H265.
- Keep tensor/pixel conversions explicit and stable:
  - Tensor range `[-1, 1]` <-> `CVPixelBuffer` BGRA8.

## Integration + Verification
- Keep artifact injection opportunistic at app call-sites:
  - use `try?` and fall back to original tensor if compression fails.
- Verify behavior with Bazel CLI tests using measurable assertions:
  - shape preserved, output range remains in `[-1, 1]`, non-zero artifacts, lower quality increases error.
- After API refactors, validate both:
  - focused target (`bazel run //Libraries/FrameCompression:FrameCompressionTestCLI`)
  - integration target (`bazel build //Apps/DrawThings:DrawThings`).

## LTX/PyAV Parameter Parity Notes
- To better match LTX `media_io.py` behavior (x264 CRF flow), prefer constant-quality style over explicit bitrate limits in VideoToolbox:
  - do not set `AverageBitRate` / `DataRateLimits`.
  - set `ExpectedFrameRate = 1`, `MaxKeyFrameInterval = 1`, `AllowFrameReordering = false`.
  - set `RealTime = false` and, when supported, `PrioritizeEncodingSpeedOverQuality = true` to approximate `preset=veryfast`.
- Quality mapping now follows CRF semantics:
  - `crf = round((100 - quality) / 100 * 51)`.
  - inverse: `quality ≈ 100 - (crf / 51 * 100)`.
  - example: `crf = 29` corresponds to quality around `43` to `44`.

## Utility Function Style
- Prefer synchronous utility functions whenever possible.
- If an operation is inherently callback-driven and cannot be made truly synchronous with native APIs, expose an async completion-handler API directly.
- Do not introduce `DispatchQueue`, `DispatchGroup`, or `DispatchSemaphore` in utility functions to simulate sync/async behavior unless absolutely necessary.
- Keep utility functions thread-agnostic and leave threading/queue decisions to upper-level call sites.
