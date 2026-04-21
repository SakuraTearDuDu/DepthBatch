from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture()
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture()
def fixtures_dir(repo_root: Path) -> Path:
    return repo_root / "tests" / "fixtures"


@pytest.fixture()
def images_dir(fixtures_dir: Path) -> Path:
    return fixtures_dir / "images"


@pytest.fixture()
def sample_video(tmp_path: Path) -> Path:
    video_path = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 6.0, (32, 24))
    for idx in range(6):
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        frame[:, :, 0] = idx * 20
        frame[:, :, 1] = 255 - idx * 20
        frame[:, :, 2] = 80
        writer.write(frame)
    writer.release()
    return video_path
