from __future__ import annotations

from pathlib import Path

from depthbatch.pipelines import benchmark as benchmark_pipeline
from depthbatch.pipelines import export_onnx as export_onnx_pipeline
from depthbatch.pipelines import infer_images as infer_images_pipeline
from depthbatch.pipelines import infer_onnx as infer_onnx_pipeline
from depthbatch.pipelines import infer_video as infer_video_pipeline
from depthbatch.pipelines import inspect_run as inspect_run_pipeline
from depthbatch.types import AppConfig, RunResult


def infer_images(
    *,
    input_path: Path,
    output_root: Path,
    backend_name: str = "pytorch",
    model_name: str = "da-v2-small",
    weights: Path | None = None,
    onnx_path: Path | None = None,
    input_size: int = 518,
    batch_size: int = 1,
    save_raw: bool = False,
) -> RunResult:
    config = AppConfig()
    config.run.command = "infer-images"
    config.run.output = output_root
    config.inputs.input = input_path
    config.model.name = model_name
    config.model.weights = weights
    config.model.onnx_path = onnx_path
    config.model.input_size = input_size
    config.backend.name = backend_name
    config.backend.batch_size = batch_size
    config.artifacts.save_raw = save_raw
    return infer_images_pipeline(config)


def infer_video(
    *,
    input_path: Path,
    output_root: Path,
    backend_name: str = "pytorch",
    model_name: str = "da-v2-small",
    weights: Path | None = None,
    input_size: int = 518,
    stride: int = 1,
) -> RunResult:
    config = AppConfig()
    config.run.command = "infer-video"
    config.run.output = output_root
    config.inputs.input = input_path
    config.inputs.stride = stride
    config.model.name = model_name
    config.model.weights = weights
    config.model.input_size = input_size
    config.backend.name = backend_name
    return infer_video_pipeline(config)


def export_onnx(
    *,
    output_root: Path,
    backend_name: str = "pytorch",
    model_name: str = "da-v2-small",
    weights: Path | None = None,
    input_size: int = 518,
    dynamic: bool = True,
    opset: int = 17,
) -> RunResult:
    config = AppConfig()
    config.run.command = "export-onnx"
    config.run.output = output_root
    config.model.name = model_name
    config.model.weights = weights
    config.model.input_size = input_size
    config.model.dynamic = dynamic
    config.model.opset = opset
    config.backend.name = backend_name
    return export_onnx_pipeline(config)


def infer_onnx(
    *,
    input_path: Path,
    output_root: Path,
    onnx_path: Path,
    input_size: int = 518,
    batch_size: int = 1,
) -> RunResult:
    config = AppConfig()
    config.run.command = "infer-onnx"
    config.run.output = output_root
    config.inputs.input = input_path
    config.model.onnx_path = onnx_path
    config.model.input_size = input_size
    config.backend.name = "onnxruntime"
    config.backend.batch_size = batch_size
    return infer_onnx_pipeline(config)


def benchmark(
    *,
    input_path: Path,
    output_root: Path,
    compare_backends: list[str],
    weights: Path | None = None,
    onnx_path: Path | None = None,
) -> RunResult:
    config = AppConfig()
    config.run.command = "benchmark"
    config.run.output = output_root
    config.inputs.input = input_path
    config.model.weights = weights
    config.model.onnx_path = onnx_path
    config.benchmark.compare_backends = compare_backends
    return benchmark_pipeline(config)


def inspect_run(*, run_root: Path) -> RunResult:
    return inspect_run_pipeline(run_root)
