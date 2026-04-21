from __future__ import annotations

from pathlib import Path

from depthbatch.api import benchmark, export_onnx, infer_images, inspect_run


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
