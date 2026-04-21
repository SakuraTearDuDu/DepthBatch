from typing import Any

from depthbatch.backends.base import BackendCapabilities, BackendSession, ExportCapableSession
from depthbatch.backends.fake_backend import FakeBackend
from depthbatch.backends.onnx_backend import OnnxRuntimeBackend
from depthbatch.backends.pytorch_backend import PytorchBackend
from depthbatch.backends.transformers_backend import TransformersBackend

BACKEND_REGISTRY = {
    "fake": FakeBackend,
    "onnxruntime": OnnxRuntimeBackend,
    "pytorch": PytorchBackend,
    "transformers": TransformersBackend,
}


def get_backend(name: str) -> Any:
    try:
        return BACKEND_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown backend: {name}") from exc


__all__ = [
    "BackendCapabilities",
    "BackendSession",
    "ExportCapableSession",
    "FakeBackend",
    "OnnxRuntimeBackend",
    "PytorchBackend",
    "TransformersBackend",
    "get_backend",
]
