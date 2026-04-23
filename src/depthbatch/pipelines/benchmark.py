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
    comparison = None
    if _should_compare_outputs(config):
        comparison = _compare_backend_outputs(raw_images, provider, preset, config)
    report_path = paths.reports_dir / "benchmark.json"
    report_payload: dict[str, object] = {"results": results}
    if comparison is not None:
        report_payload["comparison"] = comparison
    dump_json(report_path, report_payload)
    if config.benchmark.output_markdown:
        markdown_path = paths.reports_dir / "benchmark.md"
        markdown_path.write_text(_render_markdown(results, comparison), encoding="utf-8")
    summary = recorder.finalize(
        {
            "mode": "benchmark",
            "report": str(report_path.name),
            "results": results,
            "comparison_summary": None if comparison is None else comparison["summary"],
        }
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


def _should_compare_outputs(config: AppConfig) -> bool:
    backends = set(config.benchmark.compare_backends)
    return (
        {"pytorch", "onnxruntime"}.issubset(backends)
        and config.model.weights is not None
        and config.model.onnx_path is not None
    )


def _compare_backend_outputs(
    raw_images: list[tuple[Any, np.ndarray]],
    provider: Any,
    preset: Any,
    config: AppConfig,
) -> dict[str, Any]:
    pytorch_config = clone_config(config)
    pytorch_config.backend.name = "pytorch"
    onnx_config = clone_config(config)
    onnx_config.backend.name = "onnxruntime"
    from depthbatch.pipelines.infer_onnx import _apply_export_metadata

    _apply_export_metadata(onnx_config)
    pytorch_depths, pytorch_inspection = _capture_backend_depths(
        raw_images, provider, preset, pytorch_config
    )
    onnx_depths, onnx_inspection = _capture_backend_depths(
        raw_images, provider, preset, onnx_config
    )
    return _build_comparison_report(
        pytorch_depths,
        onnx_depths,
        baseline_backend="pytorch",
        candidate_backend="onnxruntime",
        baseline_inspection=pytorch_inspection,
        candidate_inspection=onnx_inspection,
    )


def _capture_backend_depths(
    raw_images: list[tuple[Any, np.ndarray]],
    provider: Any,
    preset: Any,
    config: AppConfig,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    keep_aspect_ratio = True
    if config.backend.name == "onnxruntime":
        keep_aspect_ratio = bool(config.backend.session_options.get("keep_aspect_ratio", False))
    session = open_backend_session(config, preset)
    try:
        depths: dict[str, np.ndarray] = {}
        for item, raw in raw_images:
            prepared = provider.prepare_sample(
                item,
                raw,
                input_size=config.model.input_size,
                keep_aspect_ratio=keep_aspect_ratio,
            )
            batch = np.expand_dims(prepared.tensor.astype(np.float32), axis=0)
            output = session.infer(batch)
            depths[item.relative_path] = provider.postprocess_depth(
                output.depths[0], prepared.original_size
            )
        return depths, session.inspect()
    finally:
        session.close()


def _build_comparison_report(
    baseline_depths: dict[str, np.ndarray],
    candidate_depths: dict[str, np.ndarray],
    *,
    baseline_backend: str,
    candidate_backend: str,
    baseline_inspection: dict[str, Any],
    candidate_inspection: dict[str, Any],
) -> dict[str, Any]:
    shared_paths = sorted(set(baseline_depths) & set(candidate_depths))
    missing_from_candidate = sorted(set(baseline_depths) - set(candidate_depths))
    missing_from_baseline = sorted(set(candidate_depths) - set(baseline_depths))
    items: list[dict[str, Any]] = []
    mae_values: list[float] = []
    rmse_values: list[float] = []
    max_abs_values: list[float] = []
    shape_mismatches = 0
    for relative_path in shared_paths:
        baseline = baseline_depths[relative_path]
        candidate = candidate_depths[relative_path]
        shape_match = baseline.shape == candidate.shape
        item_result: dict[str, Any] = {
            "relative_path": relative_path,
            "shape_match": shape_match,
            baseline_backend: _describe_depth(baseline),
            candidate_backend: _describe_depth(candidate),
        }
        if shape_match:
            normalized_error = _normalized_error(baseline, candidate)
            item_result["normalized_error"] = normalized_error
            mae_values.append(float(normalized_error["mae"]))
            rmse_values.append(float(normalized_error["rmse"]))
            max_abs_values.append(float(normalized_error["max_abs_error"]))
        else:
            shape_mismatches += 1
        items.append(item_result)
    return {
        "baseline_backend": baseline_backend,
        "candidate_backend": candidate_backend,
        "baseline_inspection": baseline_inspection,
        "candidate_inspection": candidate_inspection,
        "items": items,
        "summary": {
            "item_count": len(shared_paths),
            "shape_mismatches": shape_mismatches,
            "missing_from_candidate": missing_from_candidate,
            "missing_from_baseline": missing_from_baseline,
            "mean_mae": None if not mae_values else float(np.mean(mae_values)),
            "mean_rmse": None if not rmse_values else float(np.mean(rmse_values)),
            "max_abs_error": None if not max_abs_values else float(np.max(max_abs_values)),
        },
    }


def _describe_depth(depth: np.ndarray) -> dict[str, Any]:
    array = depth.astype(np.float32)
    return {
        "shape": list(array.shape),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
    }


def _normalized_error(baseline: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    baseline_normalized = _normalize_depth(baseline)
    candidate_normalized = _normalize_depth(candidate)
    delta = baseline_normalized - candidate_normalized
    abs_delta = np.abs(delta)
    return {
        "mae": float(abs_delta.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
        "max_abs_error": float(abs_delta.max()),
    }


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    array = depth.astype(np.float32)
    minimum = float(array.min())
    maximum = float(array.max())
    scale = maximum - minimum
    if scale <= 1e-12:
        return np.zeros_like(array, dtype=np.float32)
    return (array - minimum) / scale


def _render_markdown(
    results: list[dict[str, object]], comparison: dict[str, Any] | None = None
) -> str:
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
    if comparison is not None:
        summary = comparison["summary"]
        baseline_backend = str(comparison["baseline_backend"])
        candidate_backend = str(comparison["candidate_backend"])
        lines.extend(
            [
                "",
                f"## {baseline_backend} vs {candidate_backend} comparison",
                "",
                f"- Compared items: {int(summary['item_count'])}",
                f"- Shape mismatches: {int(summary['shape_mismatches'])}",
                f"- Mean normalized MAE: {_format_optional_float(summary['mean_mae'])}",
                f"- Mean normalized RMSE: {_format_optional_float(summary['mean_rmse'])}",
                f"- Max normalized absolute error: {_format_optional_float(summary['max_abs_error'])}",
                "",
                "| Item | Shape Match | PyTorch Mean | ONNX Mean | MAE | RMSE | Max Abs |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for item in comparison["items"]:
            normalized_error = item.get("normalized_error", {})
            lines.append(
                "| "
                f"{item['relative_path']} | "
                f"{'yes' if item['shape_match'] else 'no'} | "
                f"{item[baseline_backend]['mean']:.4f} | "
                f"{item[candidate_backend]['mean']:.4f} | "
                f"{_format_optional_float(normalized_error.get('mae'))} | "
                f"{_format_optional_float(normalized_error.get('rmse'))} | "
                f"{_format_optional_float(normalized_error.get('max_abs_error'))} |"
            )
    return "\n".join(lines) + "\n"


def _format_optional_float(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(cast(float, value)):.6f}"
