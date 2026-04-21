from __future__ import annotations

import asyncio
import os
from pathlib import Path

from media_generation_kit import CloudComputeBackend, Generating, MediaGenerationPipeline

MODEL = "flux_2_klein_4b_f16.ckpt"
OUTPUT_PATH = Path("/tmp/cat_f16.png")
WIDTH = 1024
HEIGHT = 1024
STEPS = 4
PROMPT = "a cat in studio lighting"
NEGATIVE_PROMPT = ""


async def main() -> None:
    api_key = os.environ.get("DRAWTHINGS_API_KEY") or os.environ.get("DT_API_KEY")

    backend = CloudComputeBackend(api_key=api_key)

    def on_state(state, preview) -> None:
        del preview
        if isinstance(state, Generating):
            print(f"step {state.step}/{state.total_steps}")
        else:
            print(type(state).__name__)

    async with await MediaGenerationPipeline.from_pretrained(MODEL, backend=backend) as pipeline:
        pipeline.configuration.width = WIDTH
        pipeline.configuration.height = HEIGHT
        pipeline.configuration.steps = STEPS
        results = await pipeline.generate(
            PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            inputs=[],
            state_handler=on_state,
        )
        await results[0].write(OUTPUT_PATH)
        print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
