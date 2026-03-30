# AGENTS.md

## Purpose

This guide is for agents working on the current `MediaGenerationKit` surface.

The public API is centered on:

- `MediaGenerationPipeline`
- `LoRAImporter`
- `LoRAStore`
- `MediaGenerationKitError`
- `AppCheckConfiguration`

Do not reintroduce the removed legacy façade or the old internal runtime naming scheme.

## Canonical Generation Flow

```swift
import Foundation
import MediaGenerationKit

var pipeline = try MediaGenerationPipeline.fromPretrained(
  "sd_v1.5_f16.ckpt",
  backend: .local("/tmp")
)

pipeline.configuration.width = 768
pipeline.configuration.height = 768
pipeline.configuration.steps = 28

let results = try await pipeline.generate(
  prompt: "a cat in studio lighting",
  negativePrompt: "",
  inputs: []
)

try results[0].write(
  to: URL(fileURLWithPath: "/tmp/cat.png"),
  type: .png
)
```

## Input Model

Use `MediaGenerationPipeline.generate(prompt:negativePrompt:inputs:stateHandler:)`.

Inputs are direct values:

- `CIImage`
- `UIImage`
- `MediaGenerationPipeline.data(_:)`
- `MediaGenerationPipeline.file(_:)`
- role wrappers such as `.mask()`, `.moodboard()`, `.depth()`

There is no standalone request/options/assets object.

## Configuration Model

- configuration lives on `pipeline.configuration`
- model identity is bound by `fromPretrained(...)`
- copied pipelines should not share mutable configuration
- async cancellation is `Swift.Task` cancellation

## LoRA Flows

Local conversion:

- `LoRAImporter(file:version:)`
- optional `inspect()`
- `import(to:scaleFactor:progressHandler:)`

Cloud storage:

- `LoRAStore(backend:)`
- `list()`
- `upload(_:file:)`
- `delete(_:)`
- `delete(keys:)`

## Verification

Minimum checks after API or internal runtime changes:

1. `bazel build //Libraries/MediaGenerationKit:MediaGenerationKit`
2. `bazel test //Libraries/MediaGenerationKit:MediaGenerationKitTests`
