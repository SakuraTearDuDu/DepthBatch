from __future__ import annotations

import json
from pathlib import Path

from depthbatch.pipelines.infer_images import infer_images
from depthbatch.types import AppConfig, RunResult


def infer_onnx(config: AppConfig) -> RunResult:
    config.backend.name = "onnxruntime"
    _apply_export_metadata(config)
    return infer_images(config)


def _apply_export_metadata(config: AppConfig) -> None:
    if config.model.onnx_path is None:
        return
    export_json = Path(config.model.onnx_path).with_name("export.json")
    if not export_json.exists():
        config.backend.session_options.setdefault("keep_aspect_ratio", False)
        return
    payload = json.loads(export_json.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    if isinstance(metadata.get("input_size"), int):
        config.model.input_size = int(metadata["input_size"])
    if isinstance(metadata.get("dynamic"), bool):
        config.model.dynamic = bool(metadata["dynamic"])
    config.backend.session_options["keep_aspect_ratio"] = bool(
        metadata.get("keep_aspect_ratio", False)
    )
