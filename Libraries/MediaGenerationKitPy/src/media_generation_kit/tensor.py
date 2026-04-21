from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Literal

import numpy as np

from .errors import MediaGenerationKitError

TensorDataType = Literal["float16", "float32", "uint8"]

CCV_TENSOR_CPU_MEMORY = 0x1
CCV_TENSOR_FORMAT_NHWC = 0x02
CCV_8U = 0x01000
CCV_32F = 0x04000
CCV_16F = 0x20000

TENSOR_PARAM_SIZE = 64
TENSOR_HEADER_SIZE = 4 + TENSOR_PARAM_SIZE
ZIP_IDENTIFIER = 0x217
FPZIP_IDENTIFIER = 0xF7217


@dataclass(frozen=True, slots=True)
class Tensor:
    shape: tuple[int, ...]
    data_type: TensorDataType
    data: np.ndarray

    @property
    def width(self) -> int:
        return self.shape[-2] if len(self.shape) >= 3 else 0

    @property
    def height(self) -> int:
        return self.shape[-3] if len(self.shape) >= 3 else 0

    @property
    def channels(self) -> int:
        return self.shape[-1] if self.shape else 0


def encode_float_tensor_raw(shape: tuple[int, ...] | list[int], values: np.ndarray) -> bytes:
    values = np.asarray(values, dtype=np.float16).reshape(-1)
    return _encode_raw_tensor_header(tuple(shape), CCV_16F, values.tobytes())


def encode_uint8_tensor_raw(shape: tuple[int, ...] | list[int], values: np.ndarray) -> bytes:
    values = np.asarray(values, dtype=np.uint8).reshape(-1)
    return _encode_raw_tensor_header(tuple(shape), CCV_8U, values.tobytes())


def decode_tensor(data: bytes | bytearray | memoryview) -> Tensor:
    buffer = bytes(data)
    if len(buffer) < TENSOR_HEADER_SIZE:
        raise MediaGenerationKitError.generation_failed("tensor payload is too small")

    identifier = struct.unpack_from("<I", buffer, 0)[0]
    datatype = struct.unpack_from("<i", buffer, 12)[0]
    shape = tuple(
        value
        for value in struct.unpack_from("<12i", buffer, 20)
        if value > 0
    )
    payload = buffer[TENSOR_HEADER_SIZE:]
    expected_bytes = _element_count(shape) * _bytes_per_element(datatype)

    if identifier == ZIP_IDENTIFIER:
        payload = zlib.decompress(payload)
    elif identifier == FPZIP_IDENTIFIER:
        return _decode_fpzip_tensor(shape, datatype, payload)
    elif identifier != 0:
        raise MediaGenerationKitError.generation_failed(
            f"unsupported tensor codec identifier 0x{identifier:x}"
        )

    if len(payload) < expected_bytes:
        raise MediaGenerationKitError.generation_failed(
            "tensor payload ended before expected data size"
        )

    payload = payload[:expected_bytes]
    if datatype == CCV_16F:
        array = np.frombuffer(payload, dtype=np.float16).astype(np.float32)
        return Tensor(shape, "float16", array)
    if datatype == CCV_32F:
        array = np.frombuffer(payload, dtype="<f4").astype(np.float32)
        return Tensor(shape, "float32", array)
    if datatype == CCV_8U:
        array = np.frombuffer(payload, dtype=np.uint8).copy()
        return Tensor(shape, "uint8", array)
    raise MediaGenerationKitError.generation_failed(
        f"unsupported tensor datatype 0x{datatype:x}"
    )


def _decode_fpzip_tensor(shape: tuple[int, ...], datatype: int, payload: bytes) -> Tensor:
    try:
        import fpzip
    except ModuleNotFoundError as error:
        raise MediaGenerationKitError.generation_failed(
            "fpzip-compressed tensor payloads require the fpzip package"
        ) from error

    try:
        array = fpzip.decompress(payload, order="C")
    except Exception as error:
        raise MediaGenerationKitError.generation_failed(
            f"failed to decode fpzip tensor payload: {error}"
        ) from error

    if datatype == CCV_16F:
        return Tensor(shape, "float16", np.asarray(array, dtype=np.float32).reshape(-1))
    if datatype == CCV_32F:
        return Tensor(shape, "float32", np.asarray(array, dtype=np.float32).reshape(-1))
    if datatype == CCV_8U:
        return Tensor(shape, "uint8", np.asarray(array, dtype=np.uint8).reshape(-1))
    raise MediaGenerationKitError.generation_failed(
        f"unsupported fpzip tensor datatype 0x{datatype:x}"
    )


def _encode_raw_tensor_header(shape: tuple[int, ...], datatype: int, payload: bytes) -> bytes:
    if len(shape) > 12:
        raise MediaGenerationKitError.generation_failed("tensor rank must be <= 12")
    header = bytearray(TENSOR_HEADER_SIZE)
    struct.pack_into("<I", header, 0, 0)
    struct.pack_into("<i", header, 4, CCV_TENSOR_CPU_MEMORY)
    struct.pack_into("<i", header, 8, CCV_TENSOR_FORMAT_NHWC)
    struct.pack_into("<i", header, 12, datatype)
    struct.pack_into("<i", header, 16, 0)
    for index, value in enumerate(shape):
        struct.pack_into("<i", header, 20 + index * 4, int(value))
    return bytes(header) + payload


def _element_count(shape: tuple[int, ...]) -> int:
    return reduce(mul, shape, 1)


def _bytes_per_element(datatype: int) -> int:
    if datatype == CCV_8U:
        return 1
    if datatype == CCV_16F:
        return 2
    if datatype == CCV_32F:
        return 4
    raise MediaGenerationKitError.generation_failed(
        f"unsupported tensor datatype 0x{datatype:x}"
    )
