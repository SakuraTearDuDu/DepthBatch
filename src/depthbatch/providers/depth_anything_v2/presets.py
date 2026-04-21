from __future__ import annotations

from depthbatch.errors import ConfigError
from depthbatch.types import ModelPreset

MODEL_PRESETS = {
    "da-v2-small": ModelPreset(
        alias="da-v2-small",
        encoder="vits",
        features=64,
        out_channels=(48, 96, 192, 384),
        license_name="Apache-2.0",
        checkpoint_filename="depth_anything_v2_vits.pth",
        hf_model_id="depth-anything/Depth-Anything-V2-Small-hf",
    ),
    "da-v2-base": ModelPreset(
        alias="da-v2-base",
        encoder="vitb",
        features=128,
        out_channels=(96, 192, 384, 768),
        license_name="CC-BY-NC-4.0",
        checkpoint_filename="depth_anything_v2_vitb.pth",
        hf_model_id="depth-anything/Depth-Anything-V2-Base-hf",
    ),
    "da-v2-large": ModelPreset(
        alias="da-v2-large",
        encoder="vitl",
        features=256,
        out_channels=(256, 512, 1024, 1024),
        license_name="CC-BY-NC-4.0",
        checkpoint_filename="depth_anything_v2_vitl.pth",
        hf_model_id="depth-anything/Depth-Anything-V2-Large-hf",
    ),
    "da-v2-giant": ModelPreset(
        alias="da-v2-giant",
        encoder="vitg",
        features=384,
        out_channels=(1536, 1536, 1536, 1536),
        license_name="CC-BY-NC-4.0",
        checkpoint_filename="depth_anything_v2_vitg.pth",
        hf_model_id=None,
    ),
}


def resolve_preset(alias: str) -> ModelPreset:
    try:
        return MODEL_PRESETS[alias]
    except KeyError as exc:
        raise ConfigError(f"Unknown model preset: {alias}") from exc
