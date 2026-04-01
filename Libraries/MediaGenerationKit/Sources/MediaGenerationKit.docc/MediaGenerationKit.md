# MediaGenerationKit

@Metadata {
  @TechnologyRoot
}

Generate images from Swift using local models, a remote Draw Things server, or Draw Things cloud compute.

## Overview

`MediaGenerationKit` is centered around ``_MediaGenerationKit/MediaGenerationPipeline``:

1. Create a pipeline with ``_MediaGenerationKit/MediaGenerationPipeline/fromPretrained(_:backend:)``.
2. Adjust ``_MediaGenerationKit/MediaGenerationPipeline/Configuration`` when you need to override recommended defaults.
3. Call ``_MediaGenerationKit/MediaGenerationPipeline/generate(prompt:negativePrompt:inputs:stateHandler:)``.
4. Save each ``_MediaGenerationKit/MediaGenerationPipeline/Result`` to disk.

The package also exposes:

- ``_MediaGenerationKit/MediaGenerationEnvironment`` for model resolution, catalog inspection, and download/ensure workflows.
- ``_MediaGenerationKit/LoRAImporter`` for local LoRA conversion.
- ``_MediaGenerationKit/LoRAStore`` for cloud-side LoRA upload and deletion.

## Topics

### Essentials

- <doc:GettingStarted>
- <doc:RemoteGeneration>
- <doc:CloudCompute>
- <doc:ModelResolutionAndDownloads>
- <doc:LoRAWorkflows>

### Main Symbols

- ``_MediaGenerationKit/MediaGenerationPipeline``
- ``_MediaGenerationKit/MediaGenerationEnvironment``
- ``_MediaGenerationKit/MediaGenerationResolvedModel``
- ``_MediaGenerationKit/LoRAImporter``
- ``_MediaGenerationKit/LoRAStore``
- ``_MediaGenerationKit/AppCheckConfiguration``
