from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from depthbatch.backends.base import ExportCapableSession
from depthbatch.errors import BackendError
from depthbatch.types import BackendCapabilities, BackendOutput


class FakeBackend(ExportCapableSession):
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name="fake", supports_export=True, notes=("CI-safe deterministic backend",)
        )

    def infer(self, batch: np.ndarray) -> BackendOutput:
        if batch.ndim != 4:
            raise BackendError("Fake backend expects an NCHW batch.")
        depth = batch.mean(axis=1)
        depth = depth - depth.min(axis=(1, 2), keepdims=True)
        return BackendOutput(
            depths=[sample.astype(np.float32) for sample in depth], metadata={"backend": "fake"}
        )

    def inspect(self) -> dict[str, Any]:
        return {"backend": "fake", "verified": True}

    def export_onnx(
        self,
        output_path: Path,
        *,
        input_size: int,
        opset: int,
        dynamic: bool,
    ) -> dict[str, Any]:
        try:
            import onnx
            from onnx import TensorProto, helper
        except ImportError as exc:
            raise BackendError("onnx is required to export the fake ONNX model.") from exc

        shape = [
            None if dynamic else 1,
            3,
            None if dynamic else input_size,
            None if dynamic else input_size,
        ]
        output_shape = [
            None if dynamic else 1,
            None if dynamic else input_size,
            None if dynamic else input_size,
        ]
        reduce_node = helper.make_node(
            "ReduceMean",
            inputs=["input"],
            outputs=["depth"],
            axes=[1],
            keepdims=0,
        )
        graph = helper.make_graph(
            [reduce_node],
            "depthbatch_fake_depth",
            [
                helper.make_tensor_value_info("input", TensorProto.FLOAT, shape),
            ],
            [
                helper.make_tensor_value_info("depth", TensorProto.FLOAT, output_shape),
            ],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, output_path)
        return {
            "backend": "fake",
            "model_path": str(output_path),
            "dynamic": dynamic,
            "opset": opset,
        }
