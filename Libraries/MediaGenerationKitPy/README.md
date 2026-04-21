# MediaGenerationKitPy

Python client for Draw Things `MediaGenerationKit` remote and cloud generation.

This package mirrors the Swift `Libraries/MediaGenerationKit` public shape where it applies to
remote generation. It is intentionally small: create a remote backend, load a model reference,
mutate the pipeline configuration, then call `generate`.

```python
from media_generation_kit import MediaGenerationPipeline, RemoteBackend

backend = RemoteBackend("127.0.0.1", port=7859, use_tls=False)

async with await MediaGenerationPipeline.from_pretrained(
    "flux_2_klein_4b_f16.ckpt",
    backend=backend,
) as pipeline:
    pipeline.configuration.width = 1024
    pipeline.configuration.height = 1024
    pipeline.configuration.steps = 4

    results = await pipeline.generate("a red cube on a table")
    await results[0].write("/tmp/remote-output.png")
```

Remote backend defaults match the Swift package: TLS is enabled unless you opt out. For a local
plain gRPC server, pass `use_tls=False`.

```python
backend = RemoteBackend(
    "127.0.0.1",
    port=7859,
    use_tls=False,
    shared_secret=None,
)
```

Generation accepts the same image-role shape as the Swift pipeline:

```python
init_image = MediaGenerationPipeline.file("input.png")
mask = MediaGenerationPipeline.file("mask.png").mask()
depth = MediaGenerationPipeline.file("depth.png").depth()

results = await pipeline.generate(
    "replace the background with a forest",
    negative_prompt="low quality",
    inputs=[init_image, mask, depth],
)
```

Progress is delivered through typed state objects:

```python
from media_generation_kit import Generating

def on_state(state, preview):
    if isinstance(state, Generating):
        print(state.step, state.total_steps)

results = await pipeline.generate("a watercolor fox", state_handler=on_state)
```

The `examples/remote_generate.py` script is a gated smoke test for a running Draw Things remote
server:

```bash
DT_REMOTE_HOST=127.0.0.1 \
DT_REMOTE_PORT=7859 \
DT_REMOTE_USE_TLS=false \
PYTHONPATH=src \
python examples/remote_generate.py
```

Run the local unit tests with:

```bash
PYTHONPATH=src python -B -m unittest discover -s tests
```

Cloud compute is also supported through the same pipeline shape:

```python
from media_generation_kit import CloudComputeBackend, MediaGenerationPipeline

backend = CloudComputeBackend(api_key="dk_xxx")

async with await MediaGenerationPipeline.from_pretrained(
    "flux_2_klein_4b_f16.ckpt",
    backend=backend,
) as pipeline:
    results = await pipeline.generate("a red cube on a table")
```

`CloudComputeBackend` resolves `api_key` from `DRAWTHINGS_API_KEY` when omitted.

The `examples/cloud_generate.py` script uses the same flow with `DRAWTHINGS_API_KEY`.

Local generation, LoRA conversion, cloud LoRA storage, and Pipecat integration remain out of scope
for this package layer.
