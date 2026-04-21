from __future__ import annotations

import time
from typing import Any, cast

import numpy as np

from depthbatch.io import resolve_input_items
from depthbatch.pipelines.common import (
    clone_config,
    open_backend_session,
    read_image,
    resolve_provider,
)
from depthbatch.types import AppConfig, RunResult
from depthbatch.utils import dump_json


def benchmark(config: AppConfig) -> RunResult:
    if config.inputs.input is None:
        raise ValueError("benchmark requires an input path.")
    provider, preset = resolve_provider(config)
    items = resolve_input_items(config.inputs.input, "images")
    raw_images = [(item, read_image(item.source_path)) for item in items]
    from depthbatch.pipelines.common import create_run_context

    paths, _environment, recorder = create_run_context(config)
    results: list[dict[str, object]] = []
    for backend_name in config.benchmark.compare_backends:
        trial_config = clone_config(config)
        trial_config.backend.name = backend_name
        if backend_name == "onnxruntime":
            from depthbatch.pipelines.infer_onnx import _apply_export_metadata

            _apply_export_metadata(trial_config)
        started = time.perf_counter()
        session = open_backend_session(trial_config, preset)
        try:
            for _ in range(max(config.benchmark.warmup_runs, 0)):
                _run_once(raw_images, provider, session, trial_config)
            run_times = []
            for _ in range(max(config.benchmark.repeat_runs, 1)):
                run_started = time.perf_counter()
                _run_once(raw_images, provider, session, trial_config)
                run_times.append(time.perf_counter() - run_started)
        finally:
            session.close()
        total_seconds = sum(run_times)
        result = {
            "backend": backend_name,
            "image_count": len(items),
            "total_seconds": total_seconds,
            "avg_seconds_per_image": total_seconds / max(len(items) * len(run_times), 1),
            "input_size": config.model.input_size,
            "model": config.model.name,
            "inspection": session.inspect(),
            "wall_seconds": time.perf_counter() - started,
        }
        results.append(result)
        recorder.record_item(
            {"status": "completed", "relative_path": backend_name, "metadata": result}
        )
    report_path = paths.reports_dir / "benchmark.json"
    dump_json(report_path, {"results": results})
    if config.benchmark.output_markdown:
        markdown_path = paths.reports_dir / "benchmark.md"
        markdown_path.write_text(_render_markdown(results), encoding="utf-8")
    summary = recorder.finalize(
        {"mode": "benchmark", "report": str(report_path.name), "results": results}
    )
    return RunResult(run_root=paths.root, manifest_path=paths.manifest_path, summary=summary)


def _run_once(
    raw_images: list[tuple[Any, np.ndarray]], provider: Any, session: Any, config: AppConfig
) -> None:
    pending: list[Any] = []
    keep_aspect_ratio = True
    if config.backend.name == "onnxruntime":
        keep_aspect_ratio = bool(config.backend.session_options.get("keep_aspect_ratio", False))
    for item, raw in raw_images:
        prepared = provider.prepare_sample(
            item,
            raw,
            input_size=config.model.input_size,
            keep_aspect_ratio=keep_aspect_ratio,
        )
        if pending and (
            len(pending) >= config.backend.batch_size
            or pending[0].tensor.shape != prepared.tensor.shape
        ):
            _run_batch(pending, provider, session)
            pending = []
        pending.append(prepared)
    if pending:
        _run_batch(pending, provider, session)


def _run_batch(pending: list[Any], provider: Any, session: Any) -> None:
    batch = np.stack([sample.tensor for sample in pending], axis=0).astype(np.float32)
    output = session.infer(batch)
    for sample, predicted in zip(pending, output.depths, strict=True):
        provider.postprocess_depth(predicted, sample.original_size)


def _render_markdown(results: list[dict[str, object]]) -> str:
    lines = [
        "| Backend | Images | Total Seconds | Avg Seconds / Image | Input Size | Model |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        backend = str(result["backend"])
        image_count = int(cast(int, result["image_count"]))
        total_seconds = float(cast(float, result["total_seconds"]))
        avg_seconds = float(cast(float, result["avg_seconds_per_image"]))
        input_size = int(cast(int, result["input_size"]))
        model = str(result["model"])
        lines.append(
            f"| {backend} | {image_count} | {total_seconds:.4f} | {avg_seconds:.4f} | {input_size} | {model} |"
        )
    return "\n".join(lines) + "\n"
