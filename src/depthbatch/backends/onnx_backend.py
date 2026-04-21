from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from depthbatch.backends.base import BackendSession
from depthbatch.errors import BackendError
from depthbatch.types import BackendCapabilities, BackendOutput


class OnnxRuntimeBackend(BackendSession):
    def __init__(self, model_path: Path, *, device: str = "auto") -> None:
        self._model_path = model_path
        self._device = device
        self._session = self._build_session()

    def _build_session(self) -> Any:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise BackendError("onnxruntime is required for the ONNXRuntime backend.") from exc
        providers = ["CPUExecutionProvider"]
        if self._device in {"auto", "cuda"}:
            available = set(ort.get_available_providers())
            if "CUDAExecutionProvider" in available:
                providers.insert(0, "CUDAExecutionProvider")
        return ort.InferenceSession(str(self._model_path), providers=providers)

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(name="onnxruntime", supports_export=False)

    def infer(self, batch: np.ndarray) -> BackendOutput:
        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: batch.astype(np.float32)})
        depth = outputs[0]
        if depth.ndim == 4 and depth.shape[1] == 1:
            depth = depth[:, 0]
        if depth.ndim != 3:
            raise BackendError(f"Unexpected ONNX output shape: {depth.shape}")
        return BackendOutput(
            depths=[sample.astype(np.float32) for sample in depth], metadata=self.inspect()
        )

    def inspect(self) -> dict[str, Any]:
        return {
            "backend": "onnxruntime",
            "model_path": str(self._model_path),
            "providers": self._session.get_providers(),
        }
