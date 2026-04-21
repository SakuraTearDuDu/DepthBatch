from __future__ import annotations

from typing import Any, cast

import cv2
import numpy as np

from depthbatch.constants import DEFAULT_INPUT_SIZE
from depthbatch.types import ModelPreset, PreparedSample

from .presets import resolve_preset


class DepthAnythingV2Provider:
    name = "depth_anything_v2"

    def resolve_preset(self, model_name: str) -> ModelPreset:
        return resolve_preset(model_name)

    def apply_preset(self, config: Any) -> ModelPreset:
        preset = self.resolve_preset(config.model.name)
        config.model.encoder = preset.encoder
        config.model.license_name = preset.license_name
        config.model.checkpoint_filename = preset.checkpoint_filename
        if config.model.hf_model_id is None:
            config.model.hf_model_id = preset.hf_model_id
        if config.model.input_size <= 0:
            config.model.input_size = DEFAULT_INPUT_SIZE
        return preset

    def preprocess_image(
        self, raw_bgr: np.ndarray, *, input_size: int, keep_aspect_ratio: bool = True
    ) -> tuple[np.ndarray, tuple[int, int]]:
        from depthbatch.providers.depth_anything_v2._vendor.util.transform import (
            NormalizeImage,
            PrepareForNet,
            Resize,
        )

        original_size = (raw_bgr.shape[0], raw_bgr.shape[1])
        image = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        sample = {"image": image}
        for transform in (
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=keep_aspect_ratio,
                ensure_multiple_of=14,
                resize_method="lower_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ):
            sample = transform(sample)
        tensor = sample["image"].astype(np.float32)
        return tensor, original_size

    def prepare_sample(
        self,
        item: Any,
        raw_bgr: np.ndarray,
        *,
        input_size: int,
        keep_aspect_ratio: bool = True,
    ) -> PreparedSample:
        tensor, original_size = self.preprocess_image(
            raw_bgr, input_size=input_size, keep_aspect_ratio=keep_aspect_ratio
        )
        return PreparedSample(
            item=item,
            original_bgr=raw_bgr,
            tensor=tensor,
            original_size=original_size,
            processed_size=(tensor.shape[1], tensor.shape[2]),
        )

    def postprocess_depth(self, depth: np.ndarray, original_size: tuple[int, int]) -> np.ndarray:
        height, width = original_size
        if depth.shape == (height, width):
            return depth.astype(np.float32)
        try:
            import torch
            from torch.nn import functional as torch_functional
        except ImportError:
            resized = cv2.resize(
                depth.astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC
            )
            return resized.astype(np.float32)
        tensor = torch.from_numpy(depth.astype(np.float32)).view(
            1, 1, depth.shape[0], depth.shape[1]
        )
        resized_tensor = torch_functional.interpolate(
            tensor, (height, width), mode="bilinear", align_corners=True
        )
        return cast(np.ndarray, resized_tensor[0, 0].cpu().numpy().astype(np.float32))
