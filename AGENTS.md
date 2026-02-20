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
