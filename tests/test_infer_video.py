from __future__ import annotations

import json

from depthbatch.api import infer_video


def test_infer_video_fake_backend(sample_video, tmp_path) -> None:
    result = infer_video(
        backend_name="fake",
        input_path=sample_video,
        output_root=tmp_path / "infer-video",
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["counts"]["completed"] == 1
    assert any((result.run_root / "artifacts" / "video").glob("*.mp4"))
