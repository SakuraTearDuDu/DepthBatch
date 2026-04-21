from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from depthbatch.backends import get_backend
from depthbatch.config import config_to_dict
from depthbatch.errors import ArtifactError, BackendError
from depthbatch.io import expected_primary_output
from depthbatch.manifests import ManifestRecorder
from depthbatch.providers import DepthAnythingV2Provider
from depthbatch.runtime import build_environment_snapshot, build_run_paths
from depthbatch.runtime.workspace import write_run_prologue
from depthbatch.types import AppConfig, ModelPreset, PreparedSample, RunPaths


def clone_config(config: AppConfig) -> AppConfig:
    return copy.deepcopy(config)


def create_run_context(config: AppConfig) -> tuple[RunPaths, dict[str, Any], ManifestRecorder]:
    paths = build_run_paths(config)
    environment = build_environment_snapshot(Path.cwd())
    write_run_prologue(paths, config, environment)
    recorder = ManifestRecorder(
        paths=paths,
        command=config.run.command,
        config_summary=config_to_dict(config),
        environment=environment,
    )
    return paths, environment, recorder


def resolve_provider(config: AppConfig) -> tuple[DepthAnythingV2Provider, ModelPreset]:
    provider = DepthAnythingV2Provider()
    preset = provider.apply_preset(config)
    return provider, preset


def open_backend_session(config: AppConfig, preset: ModelPreset) -> Any:
    backend_name = config.backend.name
    backend_cls = get_backend(backend_name)
    if backend_name == "pytorch":
        if config.model.weights is None:
            raise BackendError("The PyTorch backend requires --weights.")
        return backend_cls(
            weights_path=config.model.weights, preset=preset, device=config.backend.device
        )
    if backend_name == "onnxruntime":
        if config.model.onnx_path is None:
            raise BackendError("The ONNXRuntime backend requires --onnx-path.")
        return backend_cls(model_path=config.model.onnx_path, device=config.backend.device)
    if backend_name == "transformers":
        return backend_cls(
            preset=preset, model_id=config.model.hf_model_id, device=config.backend.device
        )
    return backend_cls()


def read_image(path: Path) -> np.ndarray:
    buffer = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ArtifactError(f"Failed to read image: {path}")
    return image


def should_skip_existing(
    config: AppConfig, paths: RunPaths, relative_path: str
) -> tuple[bool, str | None]:
    primary_output = expected_primary_output(paths, relative_path, config.artifacts)
    if not primary_output.exists():
        return False, None
    if config.run.skip_existing and not config.run.overwrite:
        return True, "existing outputs skipped"
    if not config.run.overwrite:
        return True, "existing outputs present; use --overwrite or --skip-existing"
    return False, None


def flush_prepared_batch(
    *,
    pending: list[PreparedSample],
    provider: DepthAnythingV2Provider,
    session: Any,
    config: AppConfig,
    paths: RunPaths,
    recorder: ManifestRecorder,
) -> None:
    if not pending:
        return
    batch = np.stack([sample.tensor for sample in pending], axis=0).astype(np.float32)
    started = time.perf_counter()
    output = session.infer(batch)
    infer_seconds = time.perf_counter() - started
    for sample, predicted_depth in zip(pending, output.depths, strict=True):
        post_started = time.perf_counter()
        raw_depth = provider.postprocess_depth(predicted_depth, sample.original_size)
        post_seconds = time.perf_counter() - post_started
        write_started = time.perf_counter()
        artifact_paths = {}
        if paths is not None:
            from depthbatch.io import write_depth_outputs

            artifact_paths = write_depth_outputs(
                paths=paths,
                relative_path=sample.item.relative_path,
                raw_bgr=sample.original_bgr,
                depth=raw_depth,
                artifacts=config.artifacts,
            )
        write_seconds = time.perf_counter() - write_started
        recorder.record_item(
            {
                "input_path": str(sample.item.source_path),
                "relative_path": sample.item.relative_path,
                "status": "completed",
                "timing": {
                    **sample.timing,
                    "infer_seconds": infer_seconds / len(pending),
                    "postprocess_seconds": post_seconds,
                    "write_seconds": write_seconds,
                },
                "backend": output.metadata,
                "artifacts": artifact_paths,
                "original_size": list(sample.original_size),
                "processed_size": list(sample.processed_size),
            }
        )
