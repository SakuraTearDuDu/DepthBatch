from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from depthbatch.types import AppConfig, RunPaths
from depthbatch.utils import dump_json, dump_yaml, sanitize_name, utc_now_iso


def default_run_root(command: str, base_dir: Path | None = None) -> Path:
    base = base_dir or Path.cwd() / "runs"
    stamp = utc_now_iso().replace(":", "-")
    return base / f"{stamp}-{sanitize_name(command)}"


def build_run_paths(config: AppConfig) -> RunPaths:
    root = (config.run.output or default_run_root(config.run.command)).resolve()
    root.mkdir(parents=True, exist_ok=True)
    depth_dir = root / "artifacts" / "depth"
    raw_dir = root / "artifacts" / "raw"
    preview_dir = root / "artifacts" / "preview"
    video_dir = root / "artifacts" / "video"
    export_dir = root / "artifacts" / "export"
    reports_dir = root / "reports"
    for path in [depth_dir, raw_dir, preview_dir, video_dir, export_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        root=root,
        depth_dir=depth_dir,
        raw_dir=raw_dir,
        preview_dir=preview_dir,
        video_dir=video_dir,
        export_dir=export_dir,
        reports_dir=reports_dir,
        manifest_path=root / "manifest.json",
        items_path=root / "items.jsonl",
        resolved_config_path=root / "resolved-config.yaml",
        environment_path=root / "environment.json",
    )


def write_run_prologue(paths: RunPaths, config: AppConfig, environment: dict[str, object]) -> None:
    dump_yaml(paths.resolved_config_path, asdict(config))
    dump_json(paths.environment_path, environment)
