# AGENTS.md

## Purpose

This guide is for agents working on `Libraries/MediaGenerationKitPy`.

The Python package mirrors the current `MediaGenerationKit` public shape where it applies to:

- remote Draw Things gRPC generation
- Draw Things cloud compute generation
- model catalog helpers on `MediaGenerationEnvironment`

Out of scope today:

- local generation
- LoRA conversion
- cloud LoRA storage
- Pipecat integration

## Package Shape

- `MediaGenerationPipeline` is the main generation entry point.
- `MediaGenerationPipeline.from_pretrained(...)` is async.
- `pipeline.configuration` is the mutable generation configuration.
- `MediaGenerationEnvironment.default` owns catalog helpers.
- `RemoteBackend` is for direct gRPC servers.
- `CloudComputeBackend` is for Draw Things cloud compute.

Keep the public surface small and value-oriented. Do not add a separate request/options wrapper object.

## Canonical Python Flow

```python
from media_generation_kit import CloudComputeBackend, MediaGenerationPipeline

backend = CloudComputeBackend(api_key="dk_xxx")

async with await MediaGenerationPipeline.from_pretrained(
    "flux_2_klein_4b_f16.ckpt",
    backend=backend,
) as pipeline:
    pipeline.configuration.width = 1024
    pipeline.configuration.height = 1024
    pipeline.configuration.steps = 4

    results = await pipeline.generate(
        "a cat in studio lighting",
        negative_prompt="",
        inputs=[],
    )

    await results[0].write("/tmp/cat.png")
```

## Backends

- Remote:
  - `RemoteBackend("127.0.0.1", port=7859, use_tls=False)`
  - TLS defaults to `True` unless explicitly disabled.
  - `shared_secret` is only for plain remote servers that require it.
- Cloud:
  - `CloudComputeBackend(api_key="dk_xxx")`
  - if `api_key` is omitted, resolve it from `DRAWTHINGS_API_KEY`

Cloud execution always targets:

- host: `compute.drawthings.ai`
- port: `443`
- base API URL for auth/token flow: `https://api.drawthings.ai`

## Model References

The Python resolver supports:

- exact file ids, for example `flux_2_klein_4b_f16.ckpt`
- display names, for example `FLUX.2 [klein] 4B`
- Hugging Face references:
  - `hf://black-forest-labs/FLUX.2-klein-4B`
  - `black-forest-labs/FLUX.2-klein-4B`
  - `https://huggingface.co/black-forest-labs/FLUX.2-klein-4B`

When possible, prefer exact file ids in tests and debugging because they are the least ambiguous.

## Environment Helpers

`MediaGenerationEnvironment.default` is the catalog entry point.

Supported helpers:

- `resolve_model(...)` and `resolveModel(...)`
- `suggested_models(...)` and `suggestedModels(...)`
- `inspect_model(...)` and `inspectModel(...)`
- `downloadable_models(...)` and `downloadableModels(...)`

Important:

- `offline=True` only means "use bundled catalog data and do not fetch remote catalog JSON".
- `offline=True` does **not** mean local generation.
- Execution is still remote/cloud-only in Python.

## Request Parity Requirements

For remote/cloud generation, Python must stay aligned with the Swift remote path.

The important parity points are:

1. `request.override.models` must be populated from the model catalog.
   - This is required systematically, not just for one example model.
   - Exact models such as `flux_2_klein_4b_f16.ckpt` can degrade badly if override model metadata is missing.
2. Cloud auth must follow the Swift flow:
   - `/sdk/token`
   - `/billing/stripe/payg`
   - `/authenticate`
   - inject bearer token into the gRPC metadata
   - MediaGenerationKitPy intentionally uses `appCheckType = "none"`
3. Tensor decoding must support:
   - raw tensors
   - zlib/zip tensors
   - `fpzip` tensors

Do not regress these.

## Cloud Gotchas

The current cloud path had two concrete production issues that were fixed here:

1. Cloudflare rejected the default `Python-urllib/3.11` user agent with HTTP `403` / error `1010`.
   - The cloud HTTP helper must send an explicit user agent.
2. Cloud responses can be `fpzip`-compressed.
   - `fpzip` decode support is required for end-to-end cloud generation.

If cloud smoke starts failing again, check those first.

## Dependencies

Runtime dependencies in `pyproject.toml`:

- `grpcio`
- `protobuf`
- `flatbuffers`
- `numpy`
- `Pillow`
- `fpzip`

If remote/cloud generation fails at import time, verify the active Python environment has those installed.

## Examples

Remote smoke example:

- file: `examples/remote_generate.py`
- expected usage:

```bash
cd /Users/weiyan/DrawThings/draw-things/Libraries/MediaGenerationKitPy
PYTHONPATH=src \
DT_REMOTE_HOST=127.0.0.1 \
DT_REMOTE_PORT=7859 \
DT_REMOTE_USE_TLS=false \
/Users/weiyan/miniconda3/bin/python3 examples/remote_generate.py
```

Cloud smoke example:

- file: `examples/cloud_generate.py`
- current baked-in example values:
  - model: `flux_2_klein_4b_f16.ckpt`
  - width/height: `1024x1024`
  - steps: `4`
  - prompt: `a cat in studio lighting`
- expected usage:

```bash
cd /Users/weiyan/DrawThings/draw-things/Libraries/MediaGenerationKitPy
PYTHONPATH=src \
DRAWTHINGS_API_KEY='dk_xxx' \
/Users/weiyan/miniconda3/bin/python3 examples/cloud_generate.py
```

## Testing

Use the conda Python that already has the package dependencies installed:

```bash
cd /Users/weiyan/DrawThings/draw-things/Libraries/MediaGenerationKitPy
PYTHONPATH=src \
/Users/weiyan/miniconda3/bin/python3 -B -m unittest discover -s tests
```

Current unit coverage includes:

- bundled model resolution
- Hugging Face model resolution
- recommended configuration lookup
- flatbuffer validation and serialization
- raw tensor decode
- `fpzip` tensor decode
- cloud token caching
- cloud `/authenticate` request flow
- cloud bearer metadata injection
- model override metadata population
- async context cleanup

## Current Validation Status

Known successful live cloud smoke:

- backend: `CloudComputeBackend`
- model: `flux_2_klein_4b_f16.ckpt`
- size: `1024x1024`
- steps: `4`
- prompt: `a cat in studio lighting`

That run completed and wrote `/tmp/cat.png` after fixing:

- explicit cloud HTTP user agent
- `fpzip` decode support
- `override.models` metadata population

## Guardrails

- Prefer the existing Swift-facing names and behavior over inventing Python-specific abstractions.
- Keep remote/cloud execution behavior aligned with the Swift package.
- Do not remove the snake_case API, but it is acceptable to add camelCase aliases when mirroring Swift surface helps.
- Do not claim local generation support in Python.
- If a change touches request construction, rerun the Python unit suite and at least one live smoke path when feasible.
