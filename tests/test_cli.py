from __future__ import annotations

from pathlib import Path

import pytest

from depthbatch.cli import main


def test_cli_infer_images(images_dir: Path, tmp_path: Path) -> None:
    exit_code = main(
        [
            "infer-images",
            "--backend",
            "fake",
            "--input",
            str(images_dir),
            "--output",
            str(tmp_path / "cli-infer-images"),
            "--save-raw",
        ]
    )
    assert exit_code == 0


def test_cli_inspect(images_dir: Path, tmp_path: Path) -> None:
    infer_root = tmp_path / "cli-inspect-source"
    main(
        [
            "infer-images",
            "--backend",
            "fake",
            "--input",
            str(images_dir),
            "--output",
            str(infer_root),
        ]
    )
    exit_code = main(["inspect", "--run", str(infer_root), "--stdout-json"])
    assert exit_code == 0


def test_cli_version() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
