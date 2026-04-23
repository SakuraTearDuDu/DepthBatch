from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from depthbatch.api import benchmark, export_onnx, infer_images, inspect_run
from depthbatch.pipelines.benchmark import _build_comparison_report


def test_benchmark_writes_reports(images_dir: Path, tmp_path: Path) -> None:
    export_result = export_onnx(
        backend_name="fake",
        output_root=tmp_path / "export-benchmark",
    )
    model_path = export_result.run_root / "artifacts" / "export" / "model.onnx"
    result = benchmark(
        input_path=images_dir,
        output_root=tmp_path / "benchmark",
        compare_backends=["fake", "onnxruntime"],
        onnx_path=model_path,
    )
    assert (result.run_root / "reports" / "benchmark.json").exists()
    assert (result.run_root / "reports" / "benchmark.md").exists()
    report = json.loads(
        (result.run_root / "reports" / "benchmark.json").read_text(encoding="utf-8")
    )
    assert "comparison" not in report


def test_build_comparison_report_tracks_normalized_error() -> None:
    report = _build_comparison_report(
        {
            "sample_a": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            "sample_b": np.array([[5.0, 7.0], [9.0, 11.0]], dtype=np.float32),
        },
        {
            "sample_a": np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32),
            "sample_b": np.array([[6.0, 8.0], [10.0, 12.0]], dtype=np.float32),
        },
        baseline_backend="pytorch",
        candidate_backend="onnxruntime",
        baseline_inspection={"backend": "pytorch"},
        candidate_inspection={"backend": "onnxruntime"},
    )
    assert report["summary"]["item_count"] == 2
    assert report["summary"]["shape_mismatches"] == 0
    assert report["summary"]["mean_mae"] == 0.0
    assert report["summary"]["mean_rmse"] == 0.0
    assert report["summary"]["max_abs_error"] == 0.0
    assert report["items"][0]["normalized_error"]["mae"] == 0.0


def test_inspect_reads_existing_run(images_dir: Path, tmp_path: Path) -> None:
    infer_result = infer_images(
        backend_name="fake",
        input_path=images_dir,
        output_root=tmp_path / "inspect-target",
        save_raw=True,
    )
    result = inspect_run(run_root=infer_result.run_root)
    assert result.summary["counts"]["completed"] == 2
    assert (infer_result.run_root / "reports" / "inspect.json").exists()
