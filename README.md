<p align="center">
  <img src="docs/assets/depthbatch-banner.svg" alt="DepthBatch banner" width="100%" />
</p>

<p align="center">
  <strong>English</strong> · <a href="README.zh-CN.md">简体中文</a>
</p>

<p align="center">
  <a href="https://github.com/SakuraTearDuDu/DepthBatch/actions/workflows/ci.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/SakuraTearDuDu/DepthBatch/ci.yml?branch=main&label=CI" alt="CI status" />
  </a>
  <a href="https://github.com/SakuraTearDuDu/DepthBatch/releases/tag/v0.1.0-alpha">
    <img src="https://img.shields.io/github/v/release/SakuraTearDuDu/DepthBatch?include_prereleases&label=release" alt="Release" />
  </a>
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/license-Apache--2.0-2EA043" alt="Apache-2.0" />
</p>

<p align="center">
  <a href="docs/quickstart.md">Quick Start (EN)</a> ·
  <a href="docs/quickstart.zh-CN.md">Quick Start (ZH-CN)</a> ·
  <a href="docs/license_notes.md">License Notes</a> ·
  <a href="docs/deployment_notes.md">Deployment Notes</a> ·
  <a href="https://github.com/SakuraTearDuDu/DepthBatch/releases/tag/v0.1.0-alpha">v0.1.0-alpha</a>
</p>

# DepthBatch

DepthBatch is a batch-processing and deployment-oriented monocular depth estimation toolchain built around the inference semantics of [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2).

It is not a training framework, model zoo, or GUI product. The project turns the upstream single-image and single-video workflows into a reusable engineering package with a consistent CLI, Python API, manifests, ONNX export hooks, and structured run outputs.

> Current release scope: `v0.1.0-alpha` records local DA-V2 Small evidence for the canonical PyTorch path on CPU and CUDA, plus the static-square ONNXRuntime validation path on CPU and the current Windows/NVIDIA CUDA setup. Fake backend coverage remains the CI-safe smoke path.

## Why This Exists

The upstream Depth Anything V2 repository is a strong reference for model behavior, preprocessing, and inference usage. DepthBatch extends that baseline into a tool layer that is better suited for:

- image, directory, and `.txt` list batch runs
- output artifact management
- ONNX export and validation
- benchmarking and run inspection
- GitHub-friendly packaging and CI
- Windows-first command-line usage

## Positioning

DepthBatch intentionally keeps the upstream PyTorch path as the canonical semantic baseline.

- `pytorch`: canonical backend, locally validated for DA-V2 Small with a user-provided local checkpoint
- `onnxruntime`: deployment validation backend, locally validated for the static square ONNX export path, with recorded CUDAExecutionProvider evidence on the current Windows/NVIDIA setup
- `transformers`: experimental compatibility backend
- `fake`: deterministic smoke-test backend for CI and examples

TensorRT is documented as a future extension point, not a V1 implementation.

## Install

### Base package

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

### Development

```powershell
pip install -e .[dev]
```

### Real backend extras

```powershell
pip install -e .[pytorch]
pip install -e .[pytorch,onnx]
pip install -e .[transformers]
```

> GPU wheels remain environment-specific and are intentionally not hard-coded in `pyproject.toml`. The recorded Windows 11 + RTX 4080 SUPER validation commands are documented in [Deployment Notes](docs/deployment_notes.md).

## Quick Start

### 1. Smoke-test the CLI with the fake backend

```powershell
depthbatch infer-images `
  --backend fake `
  --input tests/fixtures/images `
  --output runs/fake-smoke `
  --save-raw `
  --stdout-json
```

### 2. Download the official DA-V2 Small checkpoint into the repository

```powershell
python scripts/download_da_v2_small.py
```

The default target is `artifacts/weights/depth_anything_v2_vits.pth`. The script validates the file against the recorded SHA256 in `checksums/depth_anything_v2_vits.sha256`.

For the current alpha release, the same official checkpoint is also mirrored as a GitHub pre-release asset on the [DepthBatch `v0.1.0-alpha` release page](https://github.com/SakuraTearDuDu/DepthBatch/releases/tag/v0.1.0-alpha). The repository still does not bundle the checkpoint in Git history.

### 3. Real PyTorch inference with Depth Anything V2 Small

```powershell
depthbatch infer-images `
  --backend pytorch `
  --model da-v2-small `
  --weights artifacts\weights\depth_anything_v2_vits.pth `
  --input tests\fixtures\images `
  --output runs\real-small-pytorch `
  --save-raw `
  --stdout-json
```

On the recorded Windows/NVIDIA machine, the same path was also validated with `--device cuda`.

### 4. Export ONNX from the local checkpoint

```powershell
pip install -e .[pytorch,onnx]

depthbatch export-onnx `
  --backend pytorch `
  --model da-v2-small `
  --weights artifacts\weights\depth_anything_v2_vits.pth `
  --output artifacts\onnx `
  --stdout-json
```

### 5. Validate the exported ONNX model with ONNXRuntime

DepthBatch `0.1.0a0` validates the ONNXRuntime path with a static square export (`input_size=518`, `keep_aspect_ratio=false` in the export metadata). This is an engineering deployment path, not a claim of numerical equivalence with the canonical PyTorch preprocessing contract.

```powershell
depthbatch infer-onnx `
  --model da-v2-small `
  --onnx-path artifacts\onnx\artifacts\export\model.onnx `
  --input tests/fixtures/images `
  --output runs\real-small-onnx `
  --save-raw `
  --stdout-json
```

The recorded local GPU validation also succeeded with `--device cuda`, using `CUDAExecutionProvider` plus CPU fallback.

### 6. Run a small backend benchmark

```powershell
depthbatch benchmark `
  --model da-v2-small `
  --weights artifacts\weights\depth_anything_v2_vits.pth `
  --onnx-path artifacts\onnx\artifacts\export\model.onnx `
  --device cuda `
  --input tests\fixtures\images `
  --output runs\real-small-benchmark `
  --stdout-json
```

When both `pytorch` and `onnxruntime` are present, `benchmark` now also writes a statistical comparison section to `reports/benchmark.json` and `reports/benchmark.md`. These normalized-error metrics are engineering evidence, not a claim of full preprocessing equivalence.

## CLI

```text
depthbatch infer-images ...
depthbatch infer-video ...
depthbatch export-onnx ...
depthbatch infer-onnx ...
depthbatch benchmark ...
depthbatch inspect ...
```

Shared conventions:

- `--input` accepts a file, directory, or `.txt` list
- `--config` accepts YAML or JSON
- `--set key=value` overrides config fields
- `--output` points at the run root
- `--stdout-json` prints a compact run summary

## Python API

```python
from pathlib import Path

from depthbatch.api import infer_images

result = infer_images(
    backend_name="fake",
    input_path=Path("tests/fixtures/images"),
    output_root=Path("runs/api-smoke"),
    save_raw=True,
)

print(result.manifest_path)
```

## Inputs and Outputs

Supported inputs:

- single image
- image directory
- `.txt` file containing image paths
- single video
- video directory
- `.txt` file containing video paths

Run outputs:

- `manifest.json`
- `items.jsonl`
- `resolved-config.yaml`
- `environment.json`
- `artifacts/depth/`
- `artifacts/raw/`
- `artifacts/preview/`
- `artifacts/video/`
- `artifacts/export/`
- `reports/`

## Backends

| Backend | Status | Notes |
| --- | --- | --- |
| `fake` | verified | Deterministic smoke-test backend used in CI |
| `pytorch` | alpha-supported | Locally validated for DA-V2 Small on Windows 11 / Python 3.12.7 with `torch 2.11.0+cu128`; CPU and CUDA evidence are both recorded |
| `onnxruntime` | alpha-supported | Locally validated for the static square export path on Windows 11 / Python 3.12.7 with `onnxruntime-gpu 1.25.0`; CPU smoke and CUDAExecutionProvider evidence are both recorded |
| `transformers` | experimental | Compatibility path, not canonical |

## Limitations

- No bundled model weights or exported ONNX files
- No training, fine-tuning, distillation, or metric-depth research workflows
- No TensorRT backend implementation in V1
- Current GPU evidence is limited to the recorded Windows 11 + RTX 4080 SUPER environment
- Dynamic ONNX export is not release-validated in `v0.1.0-alpha`
- ONNXRuntime validation in this alpha uses a static square preprocessing contract
- Benchmark comparison reports statistical consistency, not numerical equivalence
- Export-time ONNX smoke validation remains CPU-based for deterministic release checks
- Real backend validation depends on local user environment and user-provided artifacts

## License

The repository is Apache-2.0. Vendored runtime files derived from Depth Anything V2 and DINOv2 are attributed in `NOTICE`.

Model licenses are separate from repository code licenses. The validated default release path is Depth Anything V2 Small with a user-provided local checkpoint. See `docs/license_notes.md` before using non-small checkpoints or redistributing derived artifacts.

## Roadmap

- expand validation beyond DA-V2 Small and beyond the recorded local environment
- add richer ONNX benchmarking presets
- add TensorRT backend contract implementation
- improve multi-run comparison reports
