from __future__ import annotations

import asyncio
import os
from pathlib import Path

from media_generation_kit import Generating, MediaGenerationPipeline, RemoteBackend


async def main() -> None:
    host = os.environ.get("DT_REMOTE_HOST", "127.0.0.1")
    port = int(os.environ.get("DT_REMOTE_PORT", "7859"))
    use_tls = os.environ.get("DT_REMOTE_USE_TLS", "false").lower() in {"1", "true", "yes"}
    shared_secret = os.environ.get("DT_REMOTE_SHARED_SECRET") or None
    model = os.environ.get("DT_MODEL", "flux_2_klein_4b_f16.ckpt")
    output = Path(os.environ.get("DT_OUTPUT", "remote-output.png"))

    backend = RemoteBackend(
        host,
        port=port,
        use_tls=use_tls,
        shared_secret=shared_secret,
    )

    def on_state(state, preview) -> None:
        del preview
        if isinstance(state, Generating):
            print(f"step {state.step}/{state.total_steps}")
        else:
            print(type(state).__name__)

    async with await MediaGenerationPipeline.from_pretrained(model, backend=backend) as pipeline:
        pipeline.configuration.steps = int(os.environ.get("DT_STEPS", "4"))
        results = await pipeline.generate(
            os.environ.get("DT_PROMPT", "a red cube on a white table"),
            state_handler=on_state,
        )
        await results[0].write(output)
        print(f"wrote {output}")


if __name__ == "__main__":
    asyncio.run(main())
