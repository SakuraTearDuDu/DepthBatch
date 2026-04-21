from __future__ import annotations

from pathlib import Path

from depthbatch.config import build_config


def test_config_merge_precedence(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model:",
                "  input_size: 320",
                "backend:",
                "  batch_size: 2",
                "profiles:",
                "  fast:",
                "    backend:",
                "      batch_size: 3",
            ]
        ),
        encoding="utf-8",
    )
    config = build_config(
        command="infer-images",
        config_path=config_path,
        profile="fast",
        cli_overrides={"model": {"input_size": 640}},
        set_overrides={"backend": {"batch_size": 5}},
    )
    assert config.model.input_size == 640
    assert config.backend.batch_size == 5
    assert config.run.command == "infer-images"
