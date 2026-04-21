from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from depthbatch.backends.base import ExportCapableSession
from depthbatch.errors import BackendError
from depthbatch.types import BackendCapabilities, BackendOutput, ModelPreset


class PytorchBackend(ExportCapableSession):
    def __init__(self, *, weights_path: Path, preset: ModelPreset, device: str = "auto") -> None:
        self._weights_path = weights_path
        self._preset = preset
        self._torch = self._import_torch()
        self._device = self._select_device(device)
        self._model = self._load_model()

    def _import_torch(self) -> Any:
        try:
            import torch
        except ImportError as exc:
            raise BackendError("torch is required for the PyTorch backend.") from exc
        return torch

    def _select_device(self, device: str) -> str:
        if device == "auto":
            if self._torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return device

    def _load_model(self) -> Any:
        from depthbatch.providers.depth_anything_v2._vendor.dpt import DepthAnythingV2

        model = DepthAnythingV2(
            encoder=self._preset.encoder,
            features=self._preset.features,
            out_channels=list(self._preset.out_channels),
        )
        state_dict = self._load_state_dict()
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as exc:
            raise BackendError(
                "Failed to load the checkpoint into the Depth Anything V2 model. "
                "Confirm that the checkpoint matches the selected model preset."
            ) from exc
        model = model.to(self._device).eval()
        return model

    def _load_state_dict(self) -> dict[str, Any]:
        checkpoint = self._torch.load(self._weights_path, map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise BackendError(
                "Unsupported checkpoint format. Expected a state_dict-style PyTorch checkpoint."
            )
        for key in ("state_dict", "model"):
            nested = checkpoint.get(key)
            if isinstance(nested, dict):
                checkpoint = nested
                break
        normalized: dict[str, Any] = {}
        for key, value in checkpoint.items():
            normalized_key = key[7:] if key.startswith("module.") else key
            normalized[normalized_key] = value
        return normalized

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(name="pytorch", supports_export=True)

    def infer(self, batch: np.ndarray) -> BackendOutput:
        tensor = self._torch.from_numpy(batch.astype(np.float32)).to(self._device)
        with self._torch.no_grad():
            depth = self._model(tensor).detach().cpu().numpy()
        return BackendOutput(
            depths=[sample.astype(np.float32) for sample in depth], metadata=self.inspect()
        )

    def inspect(self) -> dict[str, Any]:
        return {
            "backend": "pytorch",
            "device": self._device,
            "weights_path": str(self._weights_path),
            "encoder": self._preset.encoder,
        }

    def export_onnx(
        self,
        output_path: Path,
        *,
        input_size: int,
        opset: int,
        dynamic: bool,
    ) -> dict[str, Any]:
        if dynamic:
            raise BackendError(
                "Dynamic ONNX export is not release-validated for Depth Anything V2 in v0.1.0-alpha. "
                "Use the default static square export path."
            )
        dummy = self._torch.randn(1, 3, input_size, input_size, device="cpu")
        model = self._model.to("cpu").eval()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._torch.onnx.export(
                model,
                dummy,
                str(output_path),
                input_names=["input"],
                output_names=["depth"],
                opset_version=opset,
                dynamo=False,
                verbose=False,
            )
        except Exception as exc:
            raise BackendError(
                "Failed to export the PyTorch model to ONNX. "
                "Review the selected opset and dynamic export settings."
            ) from exc
        finally:
            model.to(self._device)
        return {
            "backend": "pytorch",
            "model_path": str(output_path),
            "dynamic": dynamic,
            "opset": opset,
            "input_size": input_size,
            "model_alias": self._preset.alias,
            "encoder": self._preset.encoder,
            "input_names": ["input"],
            "output_names": ["depth"],
            "keep_aspect_ratio": False,
        }
