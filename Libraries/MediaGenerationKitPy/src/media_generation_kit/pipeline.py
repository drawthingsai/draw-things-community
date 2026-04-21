from __future__ import annotations

import asyncio
import math
from collections.abc import Callable
from pathlib import Path

from .backend import CloudComputeBackend, RemoteBackend
from .catalog import recommended_configuration, resolve_model_file, suggested_models
from .configuration import Configuration
from .errors import MediaGenerationKitError
from .input import DataInput, FileInput, MediaGenerationInput
from .remote import RemoteExecutor
from .result import Preview, Result
from .states import Cancelled, Completed, PipelineState, Preparing

StateHandler = Callable[[PipelineState, Preview | None], None]


class MediaGenerationPipeline:
    def __init__(
        self,
        *,
        backend: RemoteBackend,
        configuration: Configuration,
        model: str,
        remote_executor: RemoteExecutor,
    ):
        self.backend = backend
        self.configuration = configuration
        self.model = model
        self._remote_executor = remote_executor

    @staticmethod
    def data(data: bytes | bytearray | memoryview) -> DataInput:
        return DataInput(data)

    @staticmethod
    def file(path: str | Path) -> FileInput:
        return FileInput(path)

    @classmethod
    async def from_pretrained(
        cls,
        model: str,
        *,
        backend: RemoteBackend | CloudComputeBackend,
    ) -> "MediaGenerationPipeline":
        if not isinstance(backend, (RemoteBackend, CloudComputeBackend)):
            raise MediaGenerationKitError.generation_failed(
                "MediaGenerationKitPy currently supports RemoteBackend and CloudComputeBackend"
            )

        resolved_model = await resolve_model_file(model, offline=False)
        if resolved_model is None:
            if backend.allows_unresolved_model_reference:
                resolved_model = model
            else:
                suggestions = await suggested_models(model, limit=5, offline=False)
                raise MediaGenerationKitError.unresolved_model_reference(
                    model,
                    [suggestion.file for suggestion in suggestions],
                )

        configuration = await recommended_configuration(resolved_model, offline=False)
        remote_executor = await RemoteExecutor.create(backend)
        return cls(
            backend=backend,
            configuration=configuration,
            model=configuration.model,
            remote_executor=remote_executor,
        )

    async def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        inputs: list[MediaGenerationInput] | None = None,
        state_handler: StateHandler | None = None,
    ) -> list[Result]:
        state_handler and state_handler(Preparing(), None)
        try:
            await self._remote_executor.prepare_for_generation()
            results = await self._remote_executor.generate(
                prompt,
                negative_prompt,
                self.configuration,
                inputs or [],
                state_handler,
            )
            state_handler and state_handler(Completed(), None)
            return results
        except asyncio.CancelledError:
            state_handler and state_handler(Cancelled(), None)
            raise
        except MediaGenerationKitError as error:
            if error.code == "cancelled":
                state_handler and state_handler(Cancelled(), None)
            raise

    def estimated_compute_units(self, inputs: list[MediaGenerationInput] | None = None) -> int:
        inputs = inputs or []
        has_image = any(input_value.role == "image" for input_value in inputs)
        image_factor = max(self.configuration.strength, 0.05) if has_image else 1
        megapixels = (self.configuration.width * self.configuration.height) / (1024 * 1024)
        return max(
            1,
            math.ceil(
                megapixels
                * self.configuration.steps
                * self.configuration.batch_size
                * image_factor
                * 12
            ),
        )

    async def aclose(self) -> None:
        await self._remote_executor.aclose()

    async def __aenter__(self) -> "MediaGenerationPipeline":
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        await self.aclose()
