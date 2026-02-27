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

## Instruction Count Utilities (s4nnc Models)
- For model compute-estimation helpers, prefer exact, boring names derived from the original builder:
  - use `{ORIGINAL_FUNC}InstructionCount` (for example `Flux1InstructionCount`, `Flux1FixedInstructionCount`).
  - avoid creative wrapper names when the goal is to mirror existing model-builder functions.
- Put shared counting primitives in a separate reusable file so multiple models can reuse them:
  - `DenseInstructionCount(...)`
  - `ConvolutionInstructionCount(...)`
  - `ScaledDotProductAttentionInstructionCount(...)` (use full name, not `SDPA` abbreviation in the function name).
- Keep per-model instruction count functions separate by builder path (for example main denoiser vs fixed encoder), instead of forcing a combined API.
- For the simplified estimator, return a single total `Int` unless a detailed breakdown is explicitly requested.
- Validate Swift syntax/type correctness with Bazel on the affected library target after refactors:
  - `bazel build //Libraries/SwiftDiffusion:SwiftDiffusion`

## Flux2 LoRA Layer Indexing
- In `Flux2` LoRA single blocks, do not assume one shared LoRA layer index for all projections if the exported layer naming hierarchy differs.
- `LoRASingleTransformerBlock` needs separate indices for:
  - attention/output projections (`x_q`, `x_k`, `x_v`, `x_o`) using the global single-block layer index (`i + layers.0`)
  - MLP projections (`x_w1`, `x_w2`, `x_w3`) using the single-block-local MLP index (`i`)
- Reason: the single-block MLP layer names start from `0`, so reusing the global index causes LoRA weight lookup mismatches.
- When reviewing LoRA parity, check not only operator replacement (`LoRADense`/`LoRAConvolution`) but also index mapping semantics against the target naming scheme.

## LTX2 Stateful TAE Integration (ImageConverter)
- `ImageConverter.imagesWithTAESD(fromLatent:version:)` for `.ltx2` should use the **stateful Core ML API** (`prediction(from:)`) rather than the batch API used by image TAEs.
- LTX2 stateful decoder activation tensors are fixed and known in app integration:
  - inputs: `act_0 ... act_8`
  - outputs: `act_0_out ... act_8_out`
  - no parsing/sorting of activation names is needed.
- Keep the app-side LTX2 TAE path **FP16-only**:
  - require FP16 `latent`, `image`, and `act_*` tensors; return `[]` on mismatch instead of adding FP32 handling.
- Preferred latent chunk construction style (match image path ergonomics):
  - build a padded CPU tensor with explicit zero init using `Array(repeating: 0, ...)`
  - copy source frames via tensor slicing/subscript assignment
  - reshape to rank-5 `NTHWC` (`[1, chunkT, H, W, C]`) before `MLShapedArray`
  - convert with `MLMultiArray(MLShapedArray(...))`
- LTX2 stateful output semantics:
  - outputs are raw chunk frames; apply a **global trim** after concatenation
  - trim count is `tUpscale - 1` (preview `dtu000` naturally trims `0`)
- Verify compileability of the integration path with:
  - `bazel build Libraries/LocalImageGenerator --ios_multi_cpus=arm64`

## SwiftPM Support (gRPCServerCLI + Generated Sources)
- The `swift-package-build` workflow validates the SwiftPM path by:
  - running `./Scripts/swift-package/generate_binary_resources.sh`
  - running `./Scripts/swift-package/generate_datamodels.sh`
  - building `swift build --target gRPCServerCLI`
- Keep `Package.swift` in sync with Bazel module splits for `Libraries/SwiftDiffusion`:
  - when `Libraries/SwiftDiffusion/BUILD` splits out `Sources/Mappings/**/*.swift` into module `DiffusionMappings`, mirror that in SPM by adding a `DiffusionMappings` target in `Package.swift`
  - make `Diffusion` depend on `DiffusionMappings`
  - exclude `Mappings` from the `Diffusion` target sources to avoid duplicate compilation
- Symptom of missing SPM mirror for the mappings split:
  - `swift build --target gRPCServerCLI` fails with `no such module 'DiffusionMappings'` from `Libraries/SwiftDiffusion/Sources/...`
- `Libraries/DataModels/PreGeneratedSPM` output can differ from committed files if generated code is copied without formatting:
  - `Scripts/swift-package/generate_datamodels.sh` should run `swift-format` on generated `*.swift` files after copying
  - use repo config `.swift-format.json`
  - use the repo's Bazel formatter target `@SwiftFormat//:swift-format` (same formatting path as the pre-commit hook)
- `Libraries/BinaryResources/GeneratedC` is SPM-generated too, but may already be byte-for-byte up to date when regenerated; check diffs separately from `PreGeneratedSPM`.

## LTX2 LoRA Complementary Models + Integration
- When adding LoRA support for an LTX2 model builder in `Libraries/SwiftDiffusion/Sources/Models/LTX2.swift`, mirror the `Flux2.swift` pattern exactly:
  - public APIs should be `LoRA{Original}` (`LoRALTX2`, `LoRALTX2Fixed`, `LoRAEmbedding1DConnector`)
  - private helpers should also use `LoRA{Helper}` names (for example `LoRALTX2TransformerBlock`, `LoRABasicTransformerBlock1D`)
- Preserve original weight mapping keys and behavior; the LoRA version should primarily swap `Dense` / `Convolution` with `LoRADense` / `LoRAConvolution`.
- For repeated LTX2 transformer blocks, pass explicit LoRA `layerIndex` into helper functions so exported LoRA keys are stable and non-colliding across layers.
- LTX2 LoRA integration is not just the main denoiser:
  - `UNetProtocol.swift` `.ltx2` branch needs separate-LoRA gating and should switch builder between `LTX2` and `LoRALTX2`
  - `UNetFixedEncoder.swift` `.ltx2` branch must cover **both** connectors (`videoConnector` / `audioConnector`) and `LTX2Fixed`
  - `TextEncoder.encodeLTX2(...)` also needs LoRA support for `text_feature_extractor` (the post-text-model projection), not just the Gemma text model
- In `UNetFixedEncoder.swift`, canonicalize/dedupe `lora` once near the top of `encode(...)` and reuse that list in every branch:
  - do not re-merge/re-dedupe LoRA files inside individual model branches (including `.ltx2`)
- For LTX2 separate-LoRA loading in `UNetFixedEncoder.swift`, use per-submodel identity layer mappings that match the component depth:
  - connectors (`Embedding1DConnector`): `0..<2`
  - `LTX2Fixed` transformer blocks: `0..<48`
- When adding separate-LoRA load paths, keep weights-cache logic aligned with existing patterns:
  - use `concatenateLoRA(...)` only for separate execution path
  - use `mergeLoRA(...)` for merged path
  - preserve existing non-LoRA `store.read(...)` behavior and cache attach/detach flow
- Validate all LTX2 LoRA integration changes with:
  - `bazel build //Libraries/SwiftDiffusion:SwiftDiffusion`

## SamplerType Additions (TCDTrailing)
- For FlatBuffers sampler enums, append new values **at the end** in both schemas for backward compatibility:
  - `Libraries/DataModels/Sources/config.fbs`
  - `Libraries/History/Sources/tensor_history.fbs`
- Do not hand-edit `Libraries/DataModels/PreGeneratedSPM/*`; regenerate via:
  - `./Scripts/swift-package/generate_datamodels.sh`
- Generated Swift enum case naming may not preserve acronym casing exactly:
  - `TCDTrailing` in `.fbs` becomes `.tCDTrailing` in generated Swift (similar to `.dDIMTrailing`)
  - use the generated case spelling everywhere in Swift switch/mapping logic.
- Adding a sampler variant requires cross-layer updates, not just enum/schema:
  - DataModels parser/display (`Defaults.swift`)
  - History/DataModels bridging (`ImageHistoryManager.swift`)
  - CLI/display (`Libraries/Invocation/Sources/Parameters.swift`)
  - scripting constants (`Libraries/Scripting/Sources/SharedScript.swift`)
  - runtime sampler construction (`Libraries/LocalImageGenerator/Sources/LocalImageGenerator.swift`)
  - compatibility gates (`Libraries/ModelZoo/Sources/ModelZoo.swift`, `SettingsWorkflow.swift`)
  - UI/metadata fields gated on sampler type (for TCD gamma): settings sections, edit summary, serialization/export paths.
- For `TCDTrailing`, behavior should mirror other trailing samplers:
  - use `TCDSampler` with `Denoiser.LinearDiscretization(..., timestepSpacing: .trailing)`.
  - keep regular `.TCD` path unchanged.
- Validation checklist for sampler additions:
  - `bazel build //Libraries/LocalImageGenerator --ios_multi_cpus=arm64`
  - `bazel build //Apps/DrawThings:DrawThings`

## TextHistoryManager Edge Cases (NSRange + Lineage)
- `NSRange` must be treated as UTF-16 indexed when used with `NSString` APIs:
  - for full-text replacements, use `.utf16.count` (not `.count`) at call-sites.
  - affected app-side text history call-sites are in `EditWorkflow.swift`.
- Keep a shared range validator in `TextHistoryManager`:
  - use `private static func normalizedRange(...) -> NSRange?` for bounds checking only.
  - do not include migration/rewriting behavior in the validator; validation only.
- In `setTextHistory`, if a persisted modification range is invalid:
  - skip applying that modification and continue replay safely (avoid crash from invalid ranges).
- In `pushChange`, validate range before any fork/lineage/history state mutation:
  - use explicit `switch textType` validation branches against the current text.
- For non-sacred seeks, persist the resolved lineage:
  - set `project.dictionary["text_seek_to_lineage"]` from `self.lineage` (resolved target), not requested alias lineage.
- During lineage remap/fork caching:
  - key temporary `nodeLineageCache` entries by the remapped lineage (`newLineage`) consistently for insert/cleanup.
- Focused test strategy for these edge cases:
  - use a dedicated `swift_test` target (`//Libraries/History:TextHistoryManagerTests`) backed by a text-only library target (`TextHistory`) to avoid unrelated UIKit dependencies from `History`.
  - include regressions for UTF-16 replacement behavior, invalid persisted range replay safety, segmentation sanity, and unsacred-lineage seek persistence.
- Verification checklist after refactors:
  - `bazel test //Libraries/History:TextHistoryManagerTests`
  - `bazel build //Apps/DrawThings:DrawThings`
