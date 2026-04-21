from __future__ import annotations

import json
from pathlib import Path

from depthbatch.api import infer_images


def test_infer_images_fake_backend(images_dir: Path, tmp_path: Path) -> None:
    result = infer_images(
        backend_name="fake",
        input_path=images_dir,
        output_root=tmp_path / "infer-images",
        save_raw=True,
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["counts"]["completed"] == 2
    assert (result.run_root / "artifacts" / "raw" / "sample_a.npy").exists()
    assert (result.run_root / "artifacts" / "depth" / "sample_a.png").exists()
