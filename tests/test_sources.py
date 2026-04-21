from __future__ import annotations

from pathlib import Path

from depthbatch.io import resolve_input_items


def test_resolve_directory_images(images_dir: Path) -> None:
    items = resolve_input_items(images_dir, "images")
    assert len(items) == 2
    assert items[0].relative_path.endswith(".ppm")


def test_resolve_txt_list(images_dir: Path, tmp_path: Path) -> None:
    list_path = tmp_path / "images.txt"
    list_path.write_text(
        "\n".join(
            [
                str(images_dir / "sample_a.ppm"),
                str(images_dir / "sample_b.ppm"),
            ]
        ),
        encoding="utf-8",
    )
    items = resolve_input_items(list_path, "images")
    assert len(items) == 2
