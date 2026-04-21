from __future__ import annotations

from dataclasses import dataclass

from .cloud import DEFAULT_CLOUD_BASE_URL


@dataclass(frozen=True, slots=True)
class RemoteBackend:
    """Network location of a remote Draw Things generation server."""

    host: str
    port: int = 7859
    use_tls: bool = True
    shared_secret: str | None = None
    device_name: str = "MediaGenerationKitPy"

    @property
    def allows_unresolved_model_reference(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class CloudComputeBackend:
    """Draw Things cloud compute backend."""

    api_key: str | None = None
    base_url: str = DEFAULT_CLOUD_BASE_URL
    device_name: str = "MediaGenerationKitPy"

    @property
    def allows_unresolved_model_reference(self) -> bool:
        return True
