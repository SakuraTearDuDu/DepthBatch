from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from depthbatch import __version__
from depthbatch.api import inspect_run as inspect_run_api
from depthbatch.config import build_config, parse_set_overrides, set_nested
from depthbatch.errors import ConfigError, DepthBatchError
from depthbatch.pipelines import benchmark, export_onnx, infer_images, infer_onnx, infer_video
from depthbatch.types import AppConfig, RunResult


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="depthbatch", description="Mono depth batch and deployment tooling."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    infer_images_parser = subparsers.add_parser("infer-images", help="Run image inference.")
    _add_common_run_args(infer_images_parser)
    _add_inference_args(infer_images_parser)

    infer_video_parser = subparsers.add_parser("infer-video", help="Run video inference.")
    _add_common_run_args(infer_video_parser)
    _add_inference_args(infer_video_parser)
    infer_video_parser.add_argument("--stride", type=int, default=None)
    infer_video_parser.add_argument("--output-frames", action="store_true")

    export_parser = subparsers.add_parser("export-onnx", help="Export an ONNX model.")
    _add_common_run_args(export_parser)
    _add_model_args(export_parser)
    export_parser.add_argument("--backend", choices=["pytorch", "fake"], default=None)
    export_parser.add_argument("--dynamic", action="store_true")
    export_parser.add_argument("--opset", type=int, default=None)
    export_parser.add_argument("--skip-smoke", action="store_true")

    infer_onnx_parser = subparsers.add_parser("infer-onnx", help="Run ONNXRuntime image inference.")
    _add_common_run_args(infer_onnx_parser)
    _add_model_args(infer_onnx_parser)
    infer_onnx_parser.add_argument("--input", required=True, type=Path)
    infer_onnx_parser.add_argument("--onnx-path", required=True, type=Path)
    infer_onnx_parser.add_argument("--device", default=None)
    infer_onnx_parser.add_argument("--batch-size", type=int, default=None)
    infer_onnx_parser.add_argument("--pred-only", action="store_true")
    infer_onnx_parser.add_argument("--grayscale", action="store_true")
    infer_onnx_parser.add_argument("--save-raw", action="store_true")
    infer_onnx_parser.add_argument(
        "--no-save-uint16", dest="save_uint16", action="store_false", default=None
    )
    infer_onnx_parser.add_argument(
        "--no-save-side-by-side", dest="save_side_by_side", action="store_false", default=None
    )
    infer_onnx_parser.add_argument("--colormap", default=None)

    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark one or more backends.")
    _add_common_run_args(benchmark_parser)
    _add_model_args(benchmark_parser)
    benchmark_parser.add_argument("--input", required=True, type=Path)
    benchmark_parser.add_argument("--onnx-path", type=Path, default=None)
    benchmark_parser.add_argument("--device", default=None)
    benchmark_parser.add_argument("--batch-size", type=int, default=None)
    benchmark_parser.add_argument(
        "--compare-backend", dest="compare_backends", action="append", default=None
    )
    benchmark_parser.add_argument("--warmup-runs", type=int, default=None)
    benchmark_parser.add_argument("--repeat-runs", type=int, default=None)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect an existing run directory.")
    inspect_parser.add_argument("--run", required=True, type=Path)
    inspect_parser.add_argument("--stdout-json", action="store_true")

    return parser


def _add_common_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--set", dest="set_values", action="append", default=[])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--stdout-json", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--log-level", default=None)


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=None)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--input-size", type=int, default=None)


def _add_inference_args(parser: argparse.ArgumentParser) -> None:
    _add_model_args(parser)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--backend", choices=["fake", "onnxruntime", "pytorch", "transformers"], default=None
    )
    parser.add_argument("--onnx-path", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--pred-only", action="store_true")
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--save-raw", action="store_true")
    parser.add_argument("--no-save-uint16", dest="save_uint16", action="store_false", default=None)
    parser.add_argument(
        "--no-save-visualization", dest="save_visualization", action="store_false", default=None
    )
    parser.add_argument(
        "--no-save-side-by-side", dest="save_side_by_side", action="store_false", default=None
    )
    parser.add_argument("--colormap", default=None)


def _collect_cli_overrides(args: argparse.Namespace) -> tuple[dict[str, Any], set[str]]:
    overrides: dict[str, Any] = {}
    explicit: set[str] = set()
    mapping = {
        "output": "run.output",
        "run_name": "run.run_name",
        "stdout_json": "run.stdout_json",
        "overwrite": "run.overwrite",
        "skip_existing": "run.skip_existing",
        "log_level": "run.log_level",
        "model": "model.name",
        "weights": "model.weights",
        "onnx_path": "model.onnx_path",
        "input_size": "model.input_size",
        "backend": "backend.name",
        "device": "backend.device",
        "batch_size": "backend.batch_size",
        "input": "inputs.input",
        "stride": "inputs.stride",
        "pred_only": "artifacts.pred_only",
        "grayscale": "artifacts.grayscale",
        "save_raw": "artifacts.save_raw",
        "save_uint16": "artifacts.save_uint16",
        "save_visualization": "artifacts.save_visualization",
        "save_side_by_side": "artifacts.save_side_by_side",
        "output_frames": "artifacts.output_frames",
        "colormap": "artifacts.colormap",
        "dynamic": "model.dynamic",
        "opset": "model.opset",
        "warmup_runs": "benchmark.warmup_runs",
        "repeat_runs": "benchmark.repeat_runs",
    }
    for attr, dotted in mapping.items():
        if not hasattr(args, attr):
            continue
        value = getattr(args, attr)
        if value is None:
            continue
        if isinstance(value, bool) and value is False:
            continue
        set_nested(overrides, dotted, value)
        explicit.add(dotted)
    if getattr(args, "compare_backends", None):
        set_nested(overrides, "benchmark.compare_backends", args.compare_backends)
        explicit.add("benchmark.compare_backends")
    if getattr(args, "skip_smoke", False):
        set_nested(overrides, "backend.verify_export", False)
        explicit.add("backend.verify_export")
    return overrides, explicit


def _check_override_conflicts(explicit: set[str], set_values: list[str]) -> dict[str, Any]:
    parsed = parse_set_overrides(set_values)
    conflicts = sorted(explicit & {value.split("=", 1)[0].strip() for value in set_values})
    if conflicts:
        raise ConfigError(
            f"Conflicting overrides provided by CLI flag and --set: {', '.join(conflicts)}"
        )
    return parsed


def _emit_summary(result: RunResult, stdout_json: bool) -> None:
    if stdout_json:
        print(json.dumps(result.summary, indent=2, sort_keys=True, default=str))
    else:
        print(result.manifest_path)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            result = inspect_run_api(run_root=args.run)
            _emit_summary(result, args.stdout_json)
            return 0

        cli_overrides, explicit = _collect_cli_overrides(args)
        set_overrides = _check_override_conflicts(explicit, args.set_values)
        config = build_config(
            command=args.command,
            config_path=args.config,
            profile=args.profile,
            cli_overrides=cli_overrides,
            set_overrides=set_overrides,
        )
        result = _dispatch(config)
        _emit_summary(result, config.run.stdout_json)
        return 0
    except (ConfigError, DepthBatchError) as exc:
        print(f"depthbatch: {exc}", file=sys.stderr)
        return 1


def _dispatch(config: AppConfig) -> RunResult:
    if config.run.command == "infer-images":
        return infer_images(config)
    if config.run.command == "infer-video":
        return infer_video(config)
    if config.run.command == "export-onnx":
        return export_onnx(config)
    if config.run.command == "infer-onnx":
        return infer_onnx(config)
    if config.run.command == "benchmark":
        return benchmark(config)
    raise ConfigError(f"Unsupported command: {config.run.command}")
