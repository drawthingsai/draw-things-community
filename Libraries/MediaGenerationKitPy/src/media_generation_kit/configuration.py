from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from typing import Any

from .enums import (
    CompressionMethod,
    ControlInputType,
    ControlMode,
    LoRAMode,
    SamplerType,
    SeedMode,
    normalize_compression,
    normalize_control_input,
    normalize_control_mode,
    normalize_lora_mode,
    normalize_sampler,
    normalize_seed_mode,
)


@dataclass(slots=True)
class LoRA:
    file: str | None = None
    weight: float = 0.6
    mode: LoRAMode | int | str = LoRAMode.All

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "LoRA":
        return cls(
            file=_empty_to_none(value.get("file")),
            weight=float(value.get("weight", 0.6)),
            mode=normalize_lora_mode(value.get("mode", LoRAMode.All)),
        )

    def normalized(self) -> "LoRA":
        return replace(self, mode=normalize_lora_mode(self.mode))


@dataclass(slots=True)
class Control:
    file: str | None = None
    weight: float = 1.0
    guidance_start: float = 0.0
    guidance_end: float = 1.0
    no_prompt: bool = False
    global_average_pooling: bool = True
    down_sampling_rate: float = 1.0
    control_mode: ControlMode | int | str = ControlMode.Balanced
    target_blocks: list[str] = field(default_factory=list)
    input_override: ControlInputType | int | str = ControlInputType.Unspecified

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "Control":
        control_mode = value.get("controlMode", value.get("controlImportance", ControlMode.Balanced))
        return cls(
            file=_empty_to_none(value.get("file")),
            weight=float(value.get("weight", 1.0)),
            guidance_start=float(value.get("guidanceStart", value.get("guidance_start", 0.0))),
            guidance_end=float(value.get("guidanceEnd", value.get("guidance_end", 1.0))),
            no_prompt=bool(value.get("noPrompt", value.get("no_prompt", False))),
            global_average_pooling=bool(
                value.get("globalAveragePooling", value.get("global_average_pooling", True))
            ),
            down_sampling_rate=float(
                value.get("downSamplingRate", value.get("down_sampling_rate", 1.0))
            ),
            control_mode=normalize_control_mode(control_mode),
            target_blocks=list(value.get("targetBlocks", value.get("target_blocks", [])) or []),
            input_override=normalize_control_input(
                value.get("inputOverride", value.get("input_override", ControlInputType.Unspecified))
            ),
        )

    def normalized(self) -> "Control":
        return replace(
            self,
            control_mode=normalize_control_mode(self.control_mode),
            input_override=normalize_control_input(self.input_override),
        )


@dataclass(slots=True)
class Configuration:
    """Mutable generation settings for the next ``MediaGenerationPipeline.generate`` call."""

    model: str
    width: int = 512
    height: int = 512
    seed: int = field(default_factory=lambda: random.randint(0, 0xFFFFFFFF))
    steps: int = 20
    guidance_scale: float = 4.5
    strength: float = 1.0
    seed_mode: SeedMode | int | str = SeedMode.ScaleAlike
    clip_skip: int = 1
    batch_count: int = 1
    batch_size: int = 1
    num_frames: int = 14
    fps: int = 5
    motion_scale: int = 127
    sampler: SamplerType | int | str = SamplerType.UniPCAYS
    hires_fix: bool = False
    hires_fix_width: int = 448
    hires_fix_height: int = 448
    hires_fix_strength: float = 0.7
    tiled_decoding: bool = False
    decoding_tile_width: int = 640
    decoding_tile_height: int = 640
    decoding_tile_overlap: int = 128
    tiled_diffusion: bool = False
    diffusion_tile_width: int = 1024
    diffusion_tile_height: int = 1024
    diffusion_tile_overlap: int = 128
    upscaler: str | None = None
    upscaler_scale_factor: int = 0
    image_guidance_scale: float = 1.5
    loras: list[LoRA] = field(default_factory=list)
    controls: list[Control] = field(default_factory=list)
    mask_blur: float = 1.5
    mask_blur_outset: int = 0
    sharpness: float = 0.0
    face_restoration: str | None = None
    clip_weight: float = 1.0
    negative_prompt_for_image_prior: bool = True
    image_prior_steps: int = 5
    refiner_model: str | None = None
    original_image_height: int = 0
    original_image_width: int = 0
    crop_top: int = 0
    crop_left: int = 0
    target_image_height: int = 0
    target_image_width: int = 0
    aesthetic_score: float = 6.0
    negative_aesthetic_score: float = 2.5
    zero_negative_prompt: bool = False
    refiner_start: float = 0.85
    negative_original_image_height: int = 0
    negative_original_image_width: int = 0
    guiding_frame_noise: float = 0.02
    start_frame_guidance: float = 1.0
    shift: float = 1.0
    stage2_steps: int = 10
    stage2_guidance: float = 1.0
    stage2_shift: float = 1.0
    stochastic_sampling_gamma: float = 0.3
    preserve_original_after_inpaint: bool = True
    t5_text_encoder: bool = True
    separate_clip_l: bool = False
    clip_l_text: str | None = None
    separate_open_clip_g: bool = False
    open_clip_g_text: str | None = None
    speed_up_with_guidance_embed: bool = True
    guidance_embed: float = 3.5
    resolution_dependent_shift: bool = True
    tea_cache: bool = False
    tea_cache_start: int = 5
    tea_cache_end: int = -1
    tea_cache_threshold: float = 0.06
    tea_cache_max_skip_steps: int = 3
    separate_t5: bool = False
    t5_text: str | None = None
    causal_inference: int = 0
    causal_inference_pad: int = 0
    cfg_zero_star: bool = False
    cfg_zero_init_steps: int = 0
    compression_artifacts: CompressionMethod | int | str = CompressionMethod.Disabled
    compression_artifacts_quality: float | None = None

    @classmethod
    def from_model(
        cls,
        model: str,
        *,
        default_scale: int = 8,
        overrides: dict[str, Any] | None = None,
    ) -> "Configuration":
        configuration = cls(
            model=model,
            width=default_scale * 64,
            height=default_scale * 64,
        )
        if overrides:
            configuration.apply_mapping({**overrides, "model": model})
        return configuration

    def clone(self) -> "Configuration":
        return replace(
            self,
            loras=[lora.normalized() for lora in self.loras],
            controls=[control.normalized() for control in self.controls],
        )

    def apply_mapping(self, values: dict[str, Any]) -> None:
        for source_key, value in values.items():
            attr = _CONFIG_KEY_MAP.get(source_key, source_key)
            if not hasattr(self, attr):
                continue
            setattr(self, attr, self._coerce_value(attr, value))

    def normalized(self) -> "Configuration":
        copy = self.clone()
        copy.seed_mode = normalize_seed_mode(copy.seed_mode)
        copy.sampler = normalize_sampler(copy.sampler)
        copy.compression_artifacts = normalize_compression(copy.compression_artifacts)
        copy.upscaler = _empty_to_none(copy.upscaler)
        copy.face_restoration = _empty_to_none(copy.face_restoration)
        copy.refiner_model = _empty_to_none(copy.refiner_model)
        copy.clip_l_text = _empty_to_none(copy.clip_l_text)
        copy.open_clip_g_text = _empty_to_none(copy.open_clip_g_text)
        copy.t5_text = _empty_to_none(copy.t5_text)
        return copy

    def _coerce_value(self, attr: str, value: Any) -> Any:
        if attr == "loras":
            return [item if isinstance(item, LoRA) else LoRA.from_mapping(item) for item in value or []]
        if attr == "controls":
            return [
                item if isinstance(item, Control) else Control.from_mapping(item)
                for item in value or []
            ]
        if attr == "seed_mode":
            return normalize_seed_mode(value)
        if attr == "sampler":
            return normalize_sampler(value)
        if attr == "compression_artifacts":
            return normalize_compression(value)
        if attr in _OPTIONAL_STRING_ATTRS:
            return _empty_to_none(value)
        return value


_CONFIG_KEY_MAP = {
    "guidanceScale": "guidance_scale",
    "seedMode": "seed_mode",
    "clipSkip": "clip_skip",
    "batchCount": "batch_count",
    "batchSize": "batch_size",
    "numFrames": "num_frames",
    "motionScale": "motion_scale",
    "hiresFix": "hires_fix",
    "hiresFixWidth": "hires_fix_width",
    "hiresFixHeight": "hires_fix_height",
    "hiresFixStrength": "hires_fix_strength",
    "tiledDecoding": "tiled_decoding",
    "decodingTileWidth": "decoding_tile_width",
    "decodingTileHeight": "decoding_tile_height",
    "decodingTileOverlap": "decoding_tile_overlap",
    "tiledDiffusion": "tiled_diffusion",
    "diffusionTileWidth": "diffusion_tile_width",
    "diffusionTileHeight": "diffusion_tile_height",
    "diffusionTileOverlap": "diffusion_tile_overlap",
    "upscalerScaleFactor": "upscaler_scale_factor",
    "imageGuidanceScale": "image_guidance_scale",
    "maskBlur": "mask_blur",
    "maskBlurOutset": "mask_blur_outset",
    "faceRestoration": "face_restoration",
    "clipWeight": "clip_weight",
    "negativePromptForImagePrior": "negative_prompt_for_image_prior",
    "imagePriorSteps": "image_prior_steps",
    "refinerModel": "refiner_model",
    "originalImageHeight": "original_image_height",
    "originalImageWidth": "original_image_width",
    "cropTop": "crop_top",
    "cropLeft": "crop_left",
    "targetImageHeight": "target_image_height",
    "targetImageWidth": "target_image_width",
    "aestheticScore": "aesthetic_score",
    "negativeAestheticScore": "negative_aesthetic_score",
    "zeroNegativePrompt": "zero_negative_prompt",
    "refinerStart": "refiner_start",
    "negativeOriginalImageHeight": "negative_original_image_height",
    "negativeOriginalImageWidth": "negative_original_image_width",
    "guidingFrameNoise": "guiding_frame_noise",
    "startFrameGuidance": "start_frame_guidance",
    "stage2Steps": "stage2_steps",
    "stage2Guidance": "stage2_guidance",
    "stage2Shift": "stage2_shift",
    "stochasticSamplingGamma": "stochastic_sampling_gamma",
    "preserveOriginalAfterInpaint": "preserve_original_after_inpaint",
    "t5TextEncoder": "t5_text_encoder",
    "separateClipL": "separate_clip_l",
    "clipLText": "clip_l_text",
    "separateOpenClipG": "separate_open_clip_g",
    "openClipGText": "open_clip_g_text",
    "speedUpWithGuidanceEmbed": "speed_up_with_guidance_embed",
    "guidanceEmbed": "guidance_embed",
    "resolutionDependentShift": "resolution_dependent_shift",
    "teaCache": "tea_cache",
    "teaCacheStart": "tea_cache_start",
    "teaCacheEnd": "tea_cache_end",
    "teaCacheThreshold": "tea_cache_threshold",
    "teaCacheMaxSkipSteps": "tea_cache_max_skip_steps",
    "separateT5": "separate_t5",
    "t5Text": "t5_text",
    "causalInference": "causal_inference",
    "causalInferencePad": "causal_inference_pad",
    "cfgZeroStar": "cfg_zero_star",
    "cfgZeroInitSteps": "cfg_zero_init_steps",
    "compressionArtifacts": "compression_artifacts",
    "compressionArtifactsQuality": "compression_artifacts_quality",
}

_OPTIONAL_STRING_ATTRS = {
    "upscaler",
    "face_restoration",
    "refiner_model",
    "clip_l_text",
    "open_clip_g_text",
    "t5_text",
}


def _empty_to_none(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value == "":
        return None
    return str(value)
