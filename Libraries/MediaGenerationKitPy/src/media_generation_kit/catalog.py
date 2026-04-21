from __future__ import annotations

import asyncio
import json
import re
import urllib.request
from dataclasses import dataclass
from importlib import resources
from typing import Any

from .configuration import Configuration
from .errors import MediaGenerationKitError


@dataclass(frozen=True, slots=True)
class MediaGenerationResolvedModel:
    file: str
    name: str
    description: str
    version: str | None = None
    hugging_face_link: str | None = None
    is_downloaded: bool = False


_REMOTE_MODELS_URL = "https://models.drawthings.ai/models.json"
_REMOTE_CONFIGS_URL = "https://models.drawthings.ai/configs.json"
_bundled_models: list[dict[str, Any]] | None = None
_bundled_configs: list[dict[str, Any]] | None = None
_remote_models: list[dict[str, Any]] | None = None
_remote_configs: list[dict[str, Any]] | None = None


async def resolve_model(input_model: str, *, offline: bool = False) -> MediaGenerationResolvedModel | None:
    spec = await resolve_model_specification(input_model, offline=offline)
    return _resolved_model(spec) if spec else None


async def resolve_model_file(input_model: str, *, offline: bool = False) -> str | None:
    spec = await resolve_model_specification(input_model, offline=offline)
    return spec.get("file") if spec else None


async def resolve_model_specification(
    input_model: str, *, offline: bool = False
) -> dict[str, Any] | None:
    return _matching_specification(input_model, await all_model_specifications(offline=offline))


async def metadata_override_models(
    model: str | None,
    *,
    refiner_model: str | None = None,
    offline: bool = False,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    seen_files: set[str] = set()
    for candidate in (model, refiner_model):
        if not candidate:
            continue
        spec = await resolve_model_specification(candidate, offline=offline)
        if spec is None:
            continue
        file = spec.get("file")
        if isinstance(file, str) and file and file not in seen_files:
            specs.append(spec)
            seen_files.add(file)
        for stage_model in spec.get("stage_models", []) or spec.get("stageModels", []) or []:
            stage_spec = await resolve_model_specification(str(stage_model), offline=offline)
            if stage_spec is None:
                continue
            stage_file = stage_spec.get("file")
            if isinstance(stage_file, str) and stage_file and stage_file not in seen_files:
                specs.append(stage_spec)
                seen_files.add(stage_file)
    return specs


async def suggested_models(
    input_model: str, *, limit: int = 5, offline: bool = False
) -> list[MediaGenerationResolvedModel]:
    query = input_model.strip().lower()
    if not query:
        return []
    specs = await all_model_specifications(offline=offline)
    results = [
        spec
        for spec in specs
        if query in str(spec.get("file", "")).lower()
        or query in str(spec.get("name", "")).lower()
        or query in str(_hugging_face_link(spec) or "").lower()
    ]
    results.sort(
        key=lambda spec: (
            not (
                str(spec.get("file", "")).lower() == query
                or str(spec.get("name", "")).lower() == query
            ),
            str(spec.get("name", "")).lower(),
        )
    )
    return [_resolved_model(spec) for spec in results[: max(0, limit)]]


async def inspect_model(input_model: str, *, offline: bool = False) -> MediaGenerationResolvedModel:
    spec = await resolve_model_specification(input_model, offline=offline)
    if spec is None:
        raise MediaGenerationKitError.model_not_found_in_catalog(input_model)
    return _resolved_model(spec)


async def downloadable_models(
    *, include_downloaded: bool = True, offline: bool = False
) -> list[MediaGenerationResolvedModel]:
    del include_downloaded
    deduped: dict[str, dict[str, Any]] = {}
    for spec in await all_model_specifications(offline=offline):
        file = spec.get("file")
        if isinstance(file, str) and file:
            deduped[file] = spec
    return sorted(
        (_resolved_model(spec) for spec in deduped.values()),
        key=lambda item: item.name.lower(),
    )


async def recommended_configuration(model: str, *, offline: bool = False) -> Configuration:
    spec = await resolve_model_specification(model, offline=offline)
    resolved_model = str(spec.get("file")) if spec and spec.get("file") else model
    default_scale = int(spec.get("default_scale") or spec.get("defaultScale") or 8) if spec else 8
    best = _matching_configuration(
        resolved_model,
        str(spec.get("version")) if spec and spec.get("version") else None,
        await all_config_specifications(offline=offline),
    )
    overrides = dict(best.get("configuration", {})) if best else {}
    return Configuration.from_model(resolved_model, default_scale=default_scale, overrides=overrides)


async def all_model_specifications(*, offline: bool) -> list[dict[str, Any]]:
    bundled = _filter_model_specs(_load_bundled_models())
    if offline:
        return bundled
    return bundled + _filter_model_specs(await _load_remote_models())


async def all_config_specifications(*, offline: bool) -> list[dict[str, Any]]:
    bundled = _load_bundled_configs()
    if offline:
        return bundled
    return bundled + await _load_remote_configs()


def _load_bundled_models() -> list[dict[str, Any]]:
    global _bundled_models
    if _bundled_models is None:
        _bundled_models = json.loads(
            resources.files("media_generation_kit.resources").joinpath("models.json").read_text()
        )
    return _bundled_models


def _load_bundled_configs() -> list[dict[str, Any]]:
    global _bundled_configs
    if _bundled_configs is None:
        _bundled_configs = json.loads(
            resources.files("media_generation_kit.resources").joinpath("configs.json").read_text()
        )
    return _bundled_configs


async def _load_remote_models() -> list[dict[str, Any]]:
    global _remote_models
    if _remote_models is None:
        _remote_models = await asyncio.to_thread(_fetch_json_list, _REMOTE_MODELS_URL)
    return _remote_models


async def _load_remote_configs() -> list[dict[str, Any]]:
    global _remote_configs
    if _remote_configs is None:
        _remote_configs = await asyncio.to_thread(_fetch_json_list, _REMOTE_CONFIGS_URL)
    return _remote_configs


def _fetch_json_list(url: str) -> list[dict[str, Any]]:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload if isinstance(payload, list) else []
    except Exception:
        return []


def _filter_model_specs(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        spec
        for spec in specs
        if not (spec.get("remote_api_model_config") or spec.get("remoteApiModelConfig"))
    ]


def _matching_specification(input_model: str, specs: list[dict[str, Any]]) -> dict[str, Any] | None:
    for key in ("file", "name"):
        for spec in specs:
            if spec.get(key) == input_model:
                return spec
    normalized_name = _normalized_display_name(input_model)
    for spec in specs:
        if _normalized_display_name(str(spec.get("name") or "")) == normalized_name:
            return spec
    canonical_repo = _normalize_hugging_face_repo(input_model)
    if canonical_repo is None:
        return None
    for spec in specs:
        if _normalize_hugging_face_repo(_hugging_face_link(spec)) == canonical_repo:
            return spec
    return None


def _matching_configuration(
    model: str,
    version: str | None,
    configs: list[dict[str, Any]],
) -> dict[str, Any] | None:
    prefix = _model_prefix(model)
    for config in configs:
        if config.get("configuration", {}).get("model") == model:
            return config
    for config in configs:
        config_model = config.get("configuration", {}).get("model")
        if prefix and isinstance(config_model, str) and config_model.startswith(prefix):
            return config
    if version:
        for config in configs:
            if config.get("version") == version:
                return config
    return None


def _model_prefix(model: str) -> str:
    stem = model.split(".")[0]
    parts = stem.split("_")
    while parts and parts[-1] in {"f16", "svd", "q5p", "q6p", "q8p", "i8x"}:
        parts.pop()
    return "_".join(parts)


def _resolved_model(spec: dict[str, Any]) -> MediaGenerationResolvedModel:
    return MediaGenerationResolvedModel(
        file=str(spec.get("file") or ""),
        name=str(spec.get("name") or ""),
        description=str(spec.get("note") or ""),
        version=str(spec.get("version")) if spec.get("version") else None,
        hugging_face_link=_hugging_face_link(spec),
        is_downloaded=False,
    )


def _hugging_face_link(spec: dict[str, Any]) -> str | None:
    value = spec.get("hugging_face_link") or spec.get("huggingFaceLink")
    if value:
        return str(value)
    note = spec.get("note")
    if not isinstance(note, str):
        return None
    match = re.search(r"https://huggingface\.co/([^\s)]+)", note)
    if match is None:
        return None
    return match.group(1)


def _normalize_hugging_face_repo(input_value: str | None) -> str | None:
    if not input_value:
        return None
    trimmed = input_value.strip()
    if trimmed.startswith("hf://"):
        trimmed = trimmed[len("hf://") :]
    if trimmed.startswith("https://huggingface.co/"):
        trimmed = trimmed[len("https://huggingface.co/") :]
    parts = [part for part in trimmed.split("/") if part]
    if len(parts) < 2:
        return None
    return f"{parts[0]}/{parts[1]}".lower()


def _normalized_display_name(name: str) -> str:
    while "(" in name and ")" in name and name.rfind("(") < name.rfind(")"):
        start = name.rfind("(")
        end = name.rfind(")")
        name = (name[:start] + name[end + 1 :]).strip()
    return " ".join(name.lower().split())
