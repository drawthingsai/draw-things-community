from __future__ import annotations

import io

import numpy as np
from PIL import Image

from .errors import MediaGenerationKitError
from .tensor import Tensor, encode_float_tensor_raw, encode_uint8_tensor_raw


def image_to_tensor_data(
    data: bytes | bytearray | memoryview,
    target_width: int,
    target_height: int,
) -> tuple[bytes, bytes | None]:
    image = _decode_image(data).resize((target_width, target_height), Image.Resampling.NEAREST)
    rgba = np.asarray(image, dtype=np.uint8)
    rgb = rgba[:, :, :3].astype(np.float32)
    values = (rgb * 2.0 / 255.0) - 1.0
    alpha = rgba[:, :, 3]
    auto_mask = None
    if np.any(alpha < 255):
        mask = (255 - alpha).astype(np.uint8)
        auto_mask = encode_uint8_tensor_raw((1, target_height, target_width, 1), mask.reshape(-1))
    return (
        encode_float_tensor_raw((1, target_height, target_width, 3), values.reshape(-1)),
        auto_mask,
    )


def mask_to_tensor_data(
    data: bytes | bytearray | memoryview,
    target_width: int,
    target_height: int,
) -> bytes:
    image = _decode_image(data).resize((target_width, target_height), Image.Resampling.NEAREST)
    rgba = np.asarray(image, dtype=np.uint8)
    rgb = rgba[:, :, :3].astype(np.float32)
    alpha = rgba[:, :, 3].astype(np.float32) / 255.0
    mask = np.rint(rgb.mean(axis=2) * alpha).astype(np.uint8)
    return encode_uint8_tensor_raw((1, target_height, target_width, 1), mask.reshape(-1))


def hint_to_tensor_data(data: bytes | bytearray | memoryview) -> bytes:
    image = _decode_image(data)
    rgba = np.asarray(image, dtype=np.uint8)
    rgb = rgba[:, :, :3].astype(np.float32)
    values = (rgb * 2.0 / 255.0) - 1.0
    return encode_float_tensor_raw((1, image.height, image.width, 3), values.reshape(-1))


def tensor_to_png(tensor: Tensor) -> bytes:
    if tensor.data_type not in {"float16", "float32"}:
        raise MediaGenerationKitError.generation_failed("result tensor is not a floating point image")
    width = tensor.width
    height = tensor.height
    channels = tensor.channels
    if width <= 0 or height <= 0 or channels < 3:
        raise MediaGenerationKitError.generation_failed("result tensor is not an RGB image tensor")
    values = tensor.data.reshape((-1, channels))[: width * height, :3]
    rgb = np.clip((values + 1.0) * 127.5, 0, 255).astype(np.uint8)
    image = Image.fromarray(rgb.reshape((height, width, 3)), mode="RGB")
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def tensor_to_rgb_bytes(tensor: Tensor) -> bytes:
    if tensor.data_type not in {"float16", "float32"}:
        raise MediaGenerationKitError.generation_failed("result tensor is not a floating point image")
    width = tensor.width
    height = tensor.height
    channels = tensor.channels
    values = tensor.data.reshape((-1, channels))[: width * height, :3]
    return np.clip((values + 1.0) * 127.5, 0, 255).astype(np.uint8).tobytes()


def _decode_image(data: bytes | bytearray | memoryview) -> Image.Image:
    try:
        return Image.open(io.BytesIO(bytes(data))).convert("RGBA")
    except Exception as error:
        raise MediaGenerationKitError.generation_failed(f"failed to decode input image: {error}") from error
