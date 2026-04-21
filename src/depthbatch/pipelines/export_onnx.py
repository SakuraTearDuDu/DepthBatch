from __future__ import annotations

import time

import numpy as np

from depthbatch.backends.base import ExportCapableSession
from depthbatch.backends.onnx_backend import OnnxRuntimeBackend
from depthbatch.errors import BackendError
from depthbatch.pipelines.common import create_run_context, open_backend_session, resolve_provider
from depthbatch.types import AppConfig, RunResult
from depthbatch.utils import compute_sha256, dump_json, relative_to


def export_onnx(config: AppConfig) -> RunResult:
    provider, preset = resolve_provider(config)
    paths, _environment, recorder = create_run_context(config)
    session = open_backend_session(config, preset)
    if not isinstance(session, ExportCapableSession):
        session.close()
        raise BackendError(f"Backend '{config.backend.name}' does not support ONNX export.")
    model_path = paths.export_dir / "model.onnx"
    export_started = time.perf_counter()
    export_metadata = session.export_onnx(
        model_path,
        input_size=config.model.input_size,
        opset=config.model.opset,
        dynamic=config.model.dynamic,
    )
    export_seconds = time.perf_counter() - export_started
    validation: dict[str, object] = {"attempted": False}
    if config.backend.verify_export:
        validator = OnnxRuntimeBackend(model_path=model_path, device="cpu")
        dummy = np.zeros((1, 3, config.model.input_size, config.model.input_size), dtype=np.float32)
        validator.infer(dummy)
        validation = {"attempted": True, "backend": validator.inspect()}
        validator.close()
    export_summary = {
        "model_path": relative_to(model_path, paths.root),
        "sha256": compute_sha256(model_path),
        "metadata": export_metadata,
        "validation": validation,
    }
    dump_json(paths.export_dir / "export.json", export_summary)
    recorder.record_item(
        {
            "status": "completed",
            "relative_path": "model.onnx",
            "timing": {"export_seconds": export_seconds},
            "artifacts": {"onnx_model": relative_to(model_path, paths.root)},
            "metadata": export_summary,
        }
    )
    session.close()
    summary = recorder.finalize({"mode": "export", "export": export_summary})
    return RunResult(run_root=paths.root, manifest_path=paths.manifest_path, summary=summary)
