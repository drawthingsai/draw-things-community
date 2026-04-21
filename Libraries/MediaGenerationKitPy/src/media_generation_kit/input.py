from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

InputRole = Literal["image", "mask", "moodboard", "depth"]


class MediaGenerationInput(Protocol):
    role: InputRole

    async def encoded_data(self) -> bytes:
        ...


class MediaGenerationImageInput(MediaGenerationInput, Protocol):
    def mask(self) -> MediaGenerationInput:
        ...

    def moodboard(self) -> MediaGenerationInput:
        ...

    def depth(self) -> MediaGenerationInput:
        ...


@dataclass(frozen=True, slots=True)
class DataInput:
    data: bytes | bytearray | memoryview
    role: InputRole = "image"

    async def encoded_data(self) -> bytes:
        return bytes(self.data)

    def mask(self) -> MediaGenerationInput:
        return RoleInput("mask", self)

    def moodboard(self) -> MediaGenerationInput:
        return RoleInput("moodboard", self)

    def depth(self) -> MediaGenerationInput:
        return RoleInput("depth", self)


@dataclass(frozen=True, slots=True)
class FileInput:
    path: str | Path
    role: InputRole = "image"

    async def encoded_data(self) -> bytes:
        return Path(self.path).read_bytes()

    def mask(self) -> MediaGenerationInput:
        return RoleInput("mask", self)

    def moodboard(self) -> MediaGenerationInput:
        return RoleInput("moodboard", self)

    def depth(self) -> MediaGenerationInput:
        return RoleInput("depth", self)


@dataclass(frozen=True, slots=True)
class RoleInput:
    role: InputRole
    source: MediaGenerationImageInput

    async def encoded_data(self) -> bytes:
        return await self.source.encoded_data()
