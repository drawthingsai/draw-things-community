from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from .backend import RemoteBackend


@dataclass(frozen=True, slots=True)
class ResolvingBackend:
    backend: RemoteBackend


@dataclass(frozen=True, slots=True)
class ResolvingModel:
    model: str


@dataclass(frozen=True, slots=True)
class Preparing:
    pass


@dataclass(frozen=True, slots=True)
class EnsuringResources:
    pass


@dataclass(frozen=True, slots=True)
class Uploading:
    bytes_sent: int
    total_bytes: int


@dataclass(frozen=True, slots=True)
class Downloading:
    bytes_received: int
    total_bytes: int


@dataclass(frozen=True, slots=True)
class EncodingText:
    pass


@dataclass(frozen=True, slots=True)
class EncodingInputs:
    pass


@dataclass(frozen=True, slots=True)
class Generating:
    step: int
    total_steps: int


@dataclass(frozen=True, slots=True)
class Decoding:
    pass


@dataclass(frozen=True, slots=True)
class Postprocessing:
    pass


@dataclass(frozen=True, slots=True)
class Cancelling:
    pass


@dataclass(frozen=True, slots=True)
class Completed:
    pass


@dataclass(frozen=True, slots=True)
class Cancelled:
    pass


PipelineState = Union[
    ResolvingBackend,
    ResolvingModel,
    Preparing,
    EnsuringResources,
    Uploading,
    Downloading,
    EncodingText,
    EncodingInputs,
    Generating,
    Decoding,
    Postprocessing,
    Cancelling,
    Completed,
    Cancelled,
]
