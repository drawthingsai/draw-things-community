from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .errors import MediaGenerationKitError
from .image_codec import tensor_to_png, tensor_to_rgb_bytes
from .tensor import Tensor


@dataclass(frozen=True, slots=True)
class Result:
    tensor: Tensor

    @property
    def width(self) -> int:
        return self.tensor.width

    @property
    def height(self) -> int:
        return self.tensor.height

    def data(self, type: str = "png") -> bytes:
        if type.lower() != "png":
            raise MediaGenerationKitError.generation_failed("only PNG result encoding is supported")
        return tensor_to_png(self.tensor)

    def rgb_bytes(self) -> bytes:
        return tensor_to_rgb_bytes(self.tensor)

    def numpy(self) -> np.ndarray:
        channels = self.tensor.channels
        return self.tensor.data.reshape((-1, self.height, self.width, channels))

    async def write(self, path: str | Path, type: str = "png") -> None:
        Path(path).write_bytes(self.data(type))


class Preview:
    def __init__(self, tensors: list[Tensor]):
        self._results = [Result(tensor) for tensor in tensors]

    @property
    def count(self) -> int:
        return len(self._results)

    def __len__(self) -> int:
        return len(self._results)

    def __iter__(self) -> Iterator[Result]:
        return iter(self._results)

    def __getitem__(self, index: int) -> Result:
        return self._results[index]
