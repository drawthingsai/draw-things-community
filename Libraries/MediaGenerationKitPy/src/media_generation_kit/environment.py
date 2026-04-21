from __future__ import annotations

from .catalog import (
    MediaGenerationResolvedModel,
    downloadable_models,
    inspect_model,
    resolve_model,
    suggested_models,
)


class MediaGenerationEnvironment:
    """Process-scoped model catalog helpers."""

    async def resolve_model(
        self, model: str, *, offline: bool = False
    ) -> MediaGenerationResolvedModel | None:
        return await resolve_model(model, offline=offline)

    async def resolveModel(
        self, model: str, *, offline: bool = False
    ) -> MediaGenerationResolvedModel | None:
        return await self.resolve_model(model, offline=offline)

    async def suggested_models(
        self, model: str, *, limit: int = 5, offline: bool = False
    ) -> list[MediaGenerationResolvedModel]:
        return await suggested_models(model, limit=limit, offline=offline)

    async def suggestedModels(
        self, model: str, *, limit: int = 5, offline: bool = False
    ) -> list[MediaGenerationResolvedModel]:
        return await self.suggested_models(model, limit=limit, offline=offline)

    async def inspect_model(
        self, model: str, *, offline: bool = False
    ) -> MediaGenerationResolvedModel:
        return await inspect_model(model, offline=offline)

    async def inspectModel(
        self, model: str, *, offline: bool = False
    ) -> MediaGenerationResolvedModel:
        return await self.inspect_model(model, offline=offline)

    async def downloadable_models(
        self, *, include_downloaded: bool = True, offline: bool = False
    ) -> list[MediaGenerationResolvedModel]:
        return await downloadable_models(include_downloaded=include_downloaded, offline=offline)

    async def downloadableModels(
        self, *, include_downloaded: bool = True, offline: bool = False
    ) -> list[MediaGenerationResolvedModel]:
        return await self.downloadable_models(
            include_downloaded=include_downloaded,
            offline=offline,
        )


MediaGenerationEnvironment.default = MediaGenerationEnvironment()  # type: ignore[attr-defined]
