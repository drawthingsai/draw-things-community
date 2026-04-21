from __future__ import annotations

from enum import IntEnum


class SamplerType(IntEnum):
    DPMPP2MKarras = 0
    EulerA = 1
    DDIM = 2
    PLMS = 3
    DPMPPSDEKarras = 4
    UniPC = 5
    LCM = 6
    EulerASubstep = 7
    DPMPPSDESubstep = 8
    TCD = 9
    EulerATrailing = 10
    DPMPPSDETrailing = 11
    DPMPP2MAYS = 12
    EulerAAYS = 13
    DPMPPSDEAYS = 14
    DPMPP2MTrailing = 15
    DDIMTrailing = 16
    UniPCTrailing = 17
    UniPCAYS = 18
    TCDTrailing = 19


class SeedMode(IntEnum):
    Legacy = 0
    TorchCpuCompatible = 1
    ScaleAlike = 2
    NvidiaGpuCompatible = 3


class ControlMode(IntEnum):
    Balanced = 0
    Prompt = 1
    Control = 2


class ControlInputType(IntEnum):
    Unspecified = 0
    Custom = 1
    Depth = 2
    Canny = 3
    Scribble = 4
    Pose = 5
    Normalbae = 6
    Color = 7
    Lineart = 8
    Softedge = 9
    Seg = 10
    Inpaint = 11
    Ip2p = 12
    Shuffle = 13
    Mlsd = 14
    Tile = 15
    Blur = 16
    Lowquality = 17
    Gray = 18


class LoRAMode(IntEnum):
    All = 0
    Base = 1
    Refiner = 2


class CompressionMethod(IntEnum):
    Disabled = 0
    H264 = 1
    H265 = 2
    Jpeg = 3


_SAMPLER_ALIASES = {
    "dpmpp2mkarras": SamplerType.DPMPP2MKarras,
    "dpm++ 2m karras": SamplerType.DPMPP2MKarras,
    "eulera": SamplerType.EulerA,
    "euler ancestral": SamplerType.EulerA,
    "ddim": SamplerType.DDIM,
    "plms": SamplerType.PLMS,
    "dpmppsdekarras": SamplerType.DPMPPSDEKarras,
    "dpm++ sde karras": SamplerType.DPMPPSDEKarras,
    "unipc": SamplerType.UniPC,
    "lcm": SamplerType.LCM,
    "tcd": SamplerType.TCD,
    "unipctrailing": SamplerType.UniPCTrailing,
    "unipc trailing": SamplerType.UniPCTrailing,
    "unipcays": SamplerType.UniPCAYS,
    "unipc ays": SamplerType.UniPCAYS,
    "tcdtrailing": SamplerType.TCDTrailing,
    "tcd trailing": SamplerType.TCDTrailing,
}

_CONTROL_MODE_ALIASES = {
    "balanced": ControlMode.Balanced,
    "prompt": ControlMode.Prompt,
    "control": ControlMode.Control,
}

_LORA_MODE_ALIASES = {
    "all": LoRAMode.All,
    "base": LoRAMode.Base,
    "refiner": LoRAMode.Refiner,
}

_CONTROL_INPUT_ALIASES = {
    "": ControlInputType.Unspecified,
    "unspecified": ControlInputType.Unspecified,
    "custom": ControlInputType.Custom,
    "depth": ControlInputType.Depth,
    "canny": ControlInputType.Canny,
    "scribble": ControlInputType.Scribble,
    "pose": ControlInputType.Pose,
    "normal bae": ControlInputType.Normalbae,
    "normalbae": ControlInputType.Normalbae,
    "color": ControlInputType.Color,
    "lineart": ControlInputType.Lineart,
    "line art": ControlInputType.Lineart,
    "softedge": ControlInputType.Softedge,
    "soft edge": ControlInputType.Softedge,
    "segmentation": ControlInputType.Seg,
    "seg": ControlInputType.Seg,
    "inpaint": ControlInputType.Inpaint,
    "instruct pix2pix": ControlInputType.Ip2p,
    "ip2p": ControlInputType.Ip2p,
    "shuffle": ControlInputType.Shuffle,
    "mlsd": ControlInputType.Mlsd,
    "tile": ControlInputType.Tile,
    "blur": ControlInputType.Blur,
    "lowquality": ControlInputType.Lowquality,
    "low quality": ControlInputType.Lowquality,
    "gray": ControlInputType.Gray,
}

_COMPRESSION_ALIASES = {
    "disabled": CompressionMethod.Disabled,
    "none": CompressionMethod.Disabled,
    "h264": CompressionMethod.H264,
    "h265": CompressionMethod.H265,
    "jpeg": CompressionMethod.Jpeg,
    "jpg": CompressionMethod.Jpeg,
}


def normalize_sampler(value: SamplerType | int | str) -> SamplerType:
    if isinstance(value, SamplerType):
        return value
    if isinstance(value, int):
        return SamplerType(value)
    alias = _SAMPLER_ALIASES.get(_key(value))
    return alias if alias is not None else SamplerType[_enum_key(value)]


def normalize_seed_mode(value: SeedMode | int | str) -> SeedMode:
    if isinstance(value, SeedMode):
        return value
    if isinstance(value, int):
        return SeedMode(value)
    key = _key(value)
    aliases = {
        "legacy": SeedMode.Legacy,
        "torch cpu compatible": SeedMode.TorchCpuCompatible,
        "scale alike": SeedMode.ScaleAlike,
        "nvidia gpu compatible": SeedMode.NvidiaGpuCompatible,
    }
    alias = aliases.get(key)
    return alias if alias is not None else SeedMode[_enum_key(value)]


def normalize_control_mode(value: ControlMode | int | str) -> ControlMode:
    if isinstance(value, ControlMode):
        return value
    if isinstance(value, int):
        return ControlMode(value)
    alias = _CONTROL_MODE_ALIASES.get(_key(value))
    return alias if alias is not None else ControlMode[_enum_key(value)]


def normalize_control_input(value: ControlInputType | int | str) -> ControlInputType:
    if isinstance(value, ControlInputType):
        return value
    if isinstance(value, int):
        return ControlInputType(value)
    alias = _CONTROL_INPUT_ALIASES.get(_key(value))
    return alias if alias is not None else ControlInputType[_enum_key(value)]


def normalize_lora_mode(value: LoRAMode | int | str) -> LoRAMode:
    if isinstance(value, LoRAMode):
        return value
    if isinstance(value, int):
        return LoRAMode(value)
    alias = _LORA_MODE_ALIASES.get(_key(value))
    return alias if alias is not None else LoRAMode[_enum_key(value)]


def normalize_compression(value: CompressionMethod | int | str | None) -> CompressionMethod:
    if value is None:
        return CompressionMethod.Disabled
    if isinstance(value, CompressionMethod):
        return value
    if isinstance(value, int):
        return CompressionMethod(value)
    alias = _COMPRESSION_ALIASES.get(_key(value))
    return alias if alias is not None else CompressionMethod[_enum_key(value)]


def _key(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").strip().lower()


def _enum_key(value: str) -> str:
    return value.replace("_", "").replace("-", "").replace(" ", "")
