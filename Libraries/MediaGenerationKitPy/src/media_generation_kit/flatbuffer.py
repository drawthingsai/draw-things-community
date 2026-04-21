from __future__ import annotations

import flatbuffers

from .configuration import Configuration, Control, LoRA
from .enums import (
    normalize_compression,
    normalize_control_input,
    normalize_control_mode,
    normalize_lora_mode,
    normalize_sampler,
    normalize_seed_mode,
)
from .errors import MediaGenerationKitError


def serialize_configuration(configuration: Configuration) -> bytes:
    config = configuration.normalized()
    _validate_configuration(config)

    builder = flatbuffers.Builder(1024)
    model = _optional_string(builder, config.model)
    upscaler = _optional_string(builder, config.upscaler)
    face_restoration = _optional_string(builder, config.face_restoration)
    refiner_model = _optional_string(builder, config.refiner_model)
    name = builder.CreateString("")
    clip_l_text = _optional_string(builder, config.clip_l_text)
    open_clip_g_text = _optional_string(builder, config.open_clip_g_text)
    t5_text = _optional_string(builder, config.t5_text)

    controls = [_create_control(builder, control) for control in config.controls]
    controls_vector = _create_offset_vector(builder, controls)
    loras = [_create_lora(builder, lora) for lora in config.loras]
    loras_vector = _create_offset_vector(builder, loras)

    builder.StartObject(86)
    builder.PrependFloat32Slot(85, _clamp(config.compression_artifacts_quality or 43.1, 0, 100), 43.1)
    builder.PrependInt8Slot(84, int(normalize_compression(config.compression_artifacts)), 0)
    builder.PrependInt32Slot(83, _int32(config.cfg_zero_init_steps, "cfgZeroInitSteps"), 0)
    builder.PrependBoolSlot(82, bool(config.cfg_zero_star), False)
    builder.PrependInt32Slot(81, _int32(config.causal_inference_pad, "causalInferencePad"), 0)
    builder.PrependInt32Slot(80, _int32(config.causal_inference, "causalInference"), 3)
    builder.PrependBoolSlot(79, config.causal_inference > 0, False)
    builder.PrependInt32Slot(78, _int32(config.tea_cache_max_skip_steps, "teaCacheMaxSkipSteps"), 3)
    builder.PrependUOffsetTRelativeSlot(77, t5_text, 0)
    builder.PrependBoolSlot(76, bool(config.separate_t5), False)
    builder.PrependBoolSlot(75, bool(config.tea_cache), False)
    builder.PrependFloat32Slot(74, float(config.tea_cache_threshold), 0.06)
    builder.PrependInt32Slot(73, _int32(config.tea_cache_end, "teaCacheEnd"), -1)
    builder.PrependInt32Slot(72, _int32(config.tea_cache_start, "teaCacheStart"), 5)
    builder.PrependBoolSlot(71, bool(config.resolution_dependent_shift), True)
    builder.PrependFloat32Slot(70, float(config.guidance_embed), 3.5)
    builder.PrependBoolSlot(69, bool(config.speed_up_with_guidance_embed), True)
    builder.PrependUOffsetTRelativeSlot(68, open_clip_g_text, 0)
    builder.PrependBoolSlot(67, bool(config.separate_open_clip_g), False)
    builder.PrependUOffsetTRelativeSlot(66, clip_l_text, 0)
    builder.PrependBoolSlot(65, bool(config.separate_clip_l), False)
    builder.PrependBoolSlot(64, bool(config.t5_text_encoder), True)
    builder.PrependUint8Slot(63, _uint8(config.upscaler_scale_factor, "upscalerScaleFactor"), 0)
    builder.PrependUint16Slot(62, _to_scale(config.diffusion_tile_overlap, "diffusionTileOverlap", allow_zero=True), 2)
    builder.PrependUint16Slot(61, _to_scale(config.diffusion_tile_height, "diffusionTileHeight"), 16)
    builder.PrependUint16Slot(60, _to_scale(config.diffusion_tile_width, "diffusionTileWidth"), 16)
    builder.PrependBoolSlot(59, bool(config.tiled_diffusion), False)
    builder.PrependBoolSlot(58, bool(config.preserve_original_after_inpaint), True)
    builder.PrependFloat32Slot(57, float(config.stochastic_sampling_gamma), 0.3)
    builder.PrependUint16Slot(56, _to_scale(config.decoding_tile_overlap, "decodingTileOverlap", allow_zero=True), 2)
    builder.PrependUint16Slot(55, _to_scale(config.decoding_tile_height, "decodingTileHeight"), 10)
    builder.PrependUint16Slot(54, _to_scale(config.decoding_tile_width, "decodingTileWidth"), 10)
    builder.PrependBoolSlot(53, bool(config.tiled_decoding), False)
    builder.PrependFloat32Slot(52, float(config.stage2_shift), 1.0)
    builder.PrependFloat32Slot(51, float(config.stage2_guidance), 1.0)
    builder.PrependUint32Slot(50, _uint32(config.stage2_steps, "stage2Steps"), 10)
    builder.PrependFloat32Slot(49, float(config.shift), 1.0)
    builder.PrependFloat32Slot(48, float(config.sharpness), 0.0)
    builder.PrependInt32Slot(47, _int32(config.mask_blur_outset, "maskBlurOutset"), 0)
    builder.PrependUint32Slot(46, _uint32(config.num_frames, "numFrames"), 14)
    builder.PrependFloat32Slot(45, float(config.start_frame_guidance), 1.0)
    builder.PrependFloat32Slot(44, float(config.guiding_frame_noise), 0.02)
    builder.PrependUint32Slot(43, _uint32(config.motion_scale, "motionScale"), 127)
    builder.PrependUint32Slot(42, _uint32(config.fps, "fps"), 5)
    builder.PrependUOffsetTRelativeSlot(41, name, 0)
    builder.PrependUint32Slot(40, _uint32(config.negative_original_image_width, "negativeOriginalImageWidth"), 0)
    builder.PrependUint32Slot(39, _uint32(config.negative_original_image_height, "negativeOriginalImageHeight"), 0)
    builder.PrependFloat32Slot(38, float(config.refiner_start), 0.7)
    builder.PrependBoolSlot(37, bool(config.zero_negative_prompt), False)
    builder.PrependFloat32Slot(36, float(config.negative_aesthetic_score), 2.5)
    builder.PrependFloat32Slot(35, float(config.aesthetic_score), 6.0)
    builder.PrependUint32Slot(34, _uint32(config.target_image_width, "targetImageWidth"), 0)
    builder.PrependUint32Slot(33, _uint32(config.target_image_height, "targetImageHeight"), 0)
    builder.PrependInt32Slot(32, _int32(config.crop_left, "cropLeft"), 0)
    builder.PrependInt32Slot(31, _int32(config.crop_top, "cropTop"), 0)
    builder.PrependUint32Slot(30, _uint32(config.original_image_width, "originalImageWidth"), 0)
    builder.PrependUint32Slot(29, _uint32(config.original_image_height, "originalImageHeight"), 0)
    builder.PrependUOffsetTRelativeSlot(28, refiner_model, 0)
    builder.PrependUint32Slot(27, _uint32(config.image_prior_steps, "imagePriorSteps"), 5)
    builder.PrependBoolSlot(26, bool(config.negative_prompt_for_image_prior), True)
    builder.PrependFloat32Slot(25, float(config.clip_weight), 1.0)
    builder.PrependUOffsetTRelativeSlot(22, face_restoration, 0)
    builder.PrependFloat32Slot(21, float(config.mask_blur), 0.0)
    builder.PrependUOffsetTRelativeSlot(20, loras_vector, 0)
    builder.PrependUOffsetTRelativeSlot(19, controls_vector, 0)
    builder.PrependUint32Slot(18, _uint32(config.clip_skip, "clipSkip"), 1)
    builder.PrependInt8Slot(17, int(normalize_seed_mode(config.seed_mode)), 0)
    builder.PrependFloat32Slot(16, float(config.image_guidance_scale), 1.5)
    builder.PrependUOffsetTRelativeSlot(15, upscaler, 0)
    builder.PrependFloat32Slot(14, float(config.hires_fix_strength), 0.7)
    builder.PrependUint16Slot(13, _to_scale(config.hires_fix_height, "hiresFixHeight"), 0)
    builder.PrependUint16Slot(12, _to_scale(config.hires_fix_width, "hiresFixWidth"), 0)
    builder.PrependBoolSlot(11, bool(config.hires_fix), False)
    builder.PrependUint32Slot(10, min(max(_uint32(config.batch_size, "batchSize"), 1), 4), 1)
    builder.PrependUint32Slot(9, max(_uint32(config.batch_count, "batchCount"), 1), 1)
    builder.PrependInt8Slot(8, int(normalize_sampler(config.sampler)), 0)
    builder.PrependUOffsetTRelativeSlot(7, model, 0)
    builder.PrependFloat32Slot(6, float(config.strength), 0.0)
    builder.PrependFloat32Slot(5, float(config.guidance_scale), 0.0)
    builder.PrependUint32Slot(4, _uint32(config.steps, "steps"), 0)
    builder.PrependUint32Slot(3, _uint32(config.seed, "seed"), 0)
    builder.PrependUint16Slot(2, _to_scale(config.height, "height"), 0)
    builder.PrependUint16Slot(1, _to_scale(config.width, "width"), 0)
    builder.PrependInt64Slot(0, 0, 0)
    root = builder.EndObject()
    builder.Finish(root)
    return bytes(builder.Output())


def _create_control(builder: flatbuffers.Builder, value: Control) -> int:
    control = value.normalized()
    file = _optional_string(builder, control.file)
    target_block_offsets = [builder.CreateString(block) for block in control.target_blocks]
    target_blocks = _create_offset_vector(builder, target_block_offsets)

    builder.StartObject(10)
    builder.PrependInt8Slot(9, int(normalize_control_input(control.input_override)), 0)
    builder.PrependUOffsetTRelativeSlot(8, target_blocks, 0)
    builder.PrependInt8Slot(7, int(normalize_control_mode(control.control_mode)), 0)
    builder.PrependFloat32Slot(6, float(control.down_sampling_rate), 1.0)
    builder.PrependBoolSlot(5, bool(control.global_average_pooling), True)
    builder.PrependBoolSlot(4, bool(control.no_prompt), False)
    builder.PrependFloat32Slot(3, float(control.guidance_end), 1.0)
    builder.PrependFloat32Slot(2, float(control.guidance_start), 0.0)
    builder.PrependFloat32Slot(1, float(control.weight), 1.0)
    builder.PrependUOffsetTRelativeSlot(0, file, 0)
    return builder.EndObject()


def _create_lora(builder: flatbuffers.Builder, value: LoRA) -> int:
    lora = value.normalized()
    file = _optional_string(builder, lora.file)
    builder.StartObject(3)
    builder.PrependInt8Slot(2, int(normalize_lora_mode(lora.mode)), 0)
    builder.PrependFloat32Slot(1, float(lora.weight), 0.6)
    builder.PrependUOffsetTRelativeSlot(0, file, 0)
    return builder.EndObject()


def _create_offset_vector(builder: flatbuffers.Builder, offsets: list[int]) -> int:
    builder.StartVector(4, len(offsets), 4)
    for offset in reversed(offsets):
        builder.PrependUOffsetTRelative(offset)
    return builder.EndVector()


def _optional_string(builder: flatbuffers.Builder, value: str | None) -> int:
    return builder.CreateString(value) if value else 0


def _validate_configuration(configuration: Configuration) -> None:
    if configuration.width <= 0 or configuration.height <= 0:
        raise MediaGenerationKitError.generation_failed(
            "invalid request: width and height must be greater than 0"
        )
    if configuration.width % 64 != 0 or configuration.height % 64 != 0:
        raise MediaGenerationKitError.generation_failed(
            "invalid request: width and height must be multiples of 64"
        )
    if configuration.steps <= 0:
        raise MediaGenerationKitError.generation_failed("invalid request: steps must be greater than 0")
    if configuration.batch_count <= 0 or configuration.batch_size <= 0:
        raise MediaGenerationKitError.generation_failed(
            "invalid request: batchCount and batchSize must be greater than 0"
        )


def _to_scale(value: int, name: str, *, allow_zero: bool = False) -> int:
    if (not allow_zero and value <= 0) or value < 0:
        raise MediaGenerationKitError.generation_failed(
            f"invalid request: {name} must be greater than 0"
        )
    if value % 64 != 0:
        raise MediaGenerationKitError.generation_failed(
            f"invalid request: {name} must be a multiple of 64"
        )
    return _uint16(value // 64, name)


def _uint8(value: int, name: str) -> int:
    integer = _uint32(value, name)
    if integer > 0xFF:
        raise MediaGenerationKitError.generation_failed(f"invalid request: {name} must be <= 255")
    return integer


def _uint16(value: int, name: str) -> int:
    integer = _uint32(value, name)
    if integer > 0xFFFF:
        raise MediaGenerationKitError.generation_failed(f"invalid request: {name} must be <= 65535")
    return integer


def _uint32(value: int, name: str) -> int:
    if not isinstance(value, int) or value < 0 or value > 0xFFFFFFFF:
        raise MediaGenerationKitError.generation_failed(
            f"invalid request: {name} must be an unsigned 32-bit integer"
        )
    return value


def _int32(value: int, name: str) -> int:
    if not isinstance(value, int) or value < -0x80000000 or value > 0x7FFFFFFF:
        raise MediaGenerationKitError.generation_failed(
            f"invalid request: {name} must be a signed 32-bit integer"
        )
    return value


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), lower), upper)
