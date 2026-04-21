from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from depthbatch.constants import (
    DEFAULT_BACKEND,
    DEFAULT_COLORMAP,
    DEFAULT_INPUT_SIZE,
    DEFAULT_MODEL_ALIAS,
    DEFAULT_PROVIDER,
)


@dataclass
class RunSection:
    command: str = ""
    output: Path | None = None
    run_name: str | None = None
    config_path: Path | None = None
    stdout_json: bool = False
    overwrite: bool = False
    skip_existing: bool = False
    log_level: str = "INFO"


@dataclass
class ModelSection:
    provider: str = DEFAULT_PROVIDER
    name: str = DEFAULT_MODEL_ALIAS
    encoder: str = "vits"
    weights: Path | None = None
    onnx_path: Path | None = None
    input_size: int = DEFAULT_INPUT_SIZE
    opset: int = 17
    dynamic: bool = False
    hf_model_id: str | None = None
    license_name: str | None = None
    checkpoint_filename: str | None = None


@dataclass
class BackendSection:
    name: str = DEFAULT_BACKEND
    device: str = "auto"
    batch_size: int = 1
    verify_export: bool = True
    session_options: dict[str, Any] = field(default_factory=dict)


@dataclass
class InputsSection:
    input: Path | None = None
    mode: str = "images"
    stride: int = 1


@dataclass
class ArtifactsSection:
    pred_only: bool = False
    grayscale: bool = False
    colormap: str = DEFAULT_COLORMAP
    save_raw: bool = False
    save_uint16: bool = True
    save_visualization: bool = True
    save_side_by_side: bool = True
    output_frames: bool = False


@dataclass
class RuntimeSection:
    profile: str | None = None


@dataclass
class BenchmarkSection:
    compare_backends: list[str] = field(default_factory=lambda: ["pytorch", "onnxruntime"])
    warmup_runs: int = 1
    repeat_runs: int = 3
    output_markdown: bool = True


@dataclass
class AppConfig:
    run: RunSection = field(default_factory=RunSection)
    model: ModelSection = field(default_factory=ModelSection)
    backend: BackendSection = field(default_factory=BackendSection)
    inputs: InputsSection = field(default_factory=InputsSection)
    artifacts: ArtifactsSection = field(default_factory=ArtifactsSection)
    runtime: RuntimeSection = field(default_factory=RuntimeSection)
    benchmark: BenchmarkSection = field(default_factory=BenchmarkSection)


@dataclass
class InputItem:
    source_path: Path
    relative_path: str
    source_group: str


@dataclass
class PreparedSample:
    item: InputItem
    original_bgr: Any
    tensor: Any
    original_size: tuple[int, int]
    processed_size: tuple[int, int]
    timing: dict[str, float] = field(default_factory=dict)


@dataclass
class BackendCapabilities:
    name: str
    supports_export: bool = False
    notes: tuple[str, ...] = ()


@dataclass
class BackendOutput:
    depths: list[Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunPaths:
    root: Path
    depth_dir: Path
    raw_dir: Path
    preview_dir: Path
    video_dir: Path
    export_dir: Path
    reports_dir: Path
    manifest_path: Path
    items_path: Path
    resolved_config_path: Path
    environment_path: Path


@dataclass
class RunResult:
    run_root: Path
    manifest_path: Path
    summary: dict[str, Any]


@dataclass
class ModelPreset:
    alias: str
    encoder: str
    features: int
    out_channels: tuple[int, int, int, int]
    license_name: str
    checkpoint_filename: str
    hf_model_id: str | None = None
