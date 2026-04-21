from __future__ import annotations

import json
from pathlib import Path

from depthbatch.api import export_onnx, infer_onnx


def test_export_onnx_fake_backend(tmp_path: Path) -> None:
    result = export_onnx(
        backend_name="fake",
        output_root=tmp_path / "export-onnx",
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    model_path = result.run_root / "artifacts" / "export" / "model.onnx"
    assert manifest["counts"]["completed"] == 1
    assert model_path.exists()


def test_infer_onnx_fake_export(images_dir: Path, tmp_path: Path) -> None:
    export_result = export_onnx(
        backend_name="fake",
        output_root=tmp_path / "export-onnx",
    )
    model_path = export_result.run_root / "artifacts" / "export" / "model.onnx"
    infer_result = infer_onnx(
        input_path=images_dir,
        onnx_path=model_path,
        output_root=tmp_path / "infer-onnx",
    )
    manifest = json.loads(infer_result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["counts"]["completed"] == 2
