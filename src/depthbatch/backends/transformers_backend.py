from __future__ import annotations

from typing import Any

import numpy as np

from depthbatch.backends.base import BackendSession
from depthbatch.errors import BackendError
from depthbatch.types import BackendCapabilities, BackendOutput, ModelPreset


class TransformersBackend(BackendSession):
    def __init__(self, *, preset: ModelPreset, model_id: str | None, device: str = "auto") -> None:
        self._preset = preset
        self._model_id = model_id or preset.hf_model_id
        if self._model_id is None:
            raise BackendError("A Hugging Face model id is required for the transformers backend.")
        self._torch, self._model, self._device = self._load(device)

    def _load(self, device: str) -> tuple[Any, Any, str]:
        try:
            import torch
            from transformers import AutoModelForDepthEstimation
        except ImportError as exc:
            raise BackendError(
                "transformers and torch are required for the transformers backend."
            ) from exc
        resolved_device = (
            "cuda"
            if device == "auto" and torch.cuda.is_available()
            else "cpu"
            if device == "auto"
            else device
        )
        model = (
            AutoModelForDepthEstimation.from_pretrained(self._model_id).to(resolved_device).eval()
        )
        return torch, model, resolved_device

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name="transformers",
            supports_export=False,
            notes=("Experimental compatibility backend",),
        )

    def infer(self, batch: np.ndarray) -> BackendOutput:
        with self._torch.no_grad():
            outputs = self._model(
                pixel_values=self._torch.from_numpy(batch.astype(np.float32)).to(self._device)
            )
            depth = outputs.predicted_depth.detach().cpu().numpy()
        return BackendOutput(
            depths=[sample.astype(np.float32) for sample in depth], metadata=self.inspect()
        )

    def inspect(self) -> dict[str, Any]:
        return {"backend": "transformers", "device": self._device, "model_id": self._model_id}
