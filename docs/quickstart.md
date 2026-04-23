# Quick Start

**English** | [简体中文](quickstart.zh-CN.md)

## Smoke Path

Use the fake backend first to verify packaging, CLI wiring, manifest writing, and output layout.

```powershell
depthbatch infer-images `
  --backend fake `
  --input tests/fixtures/images `
  --output runs/fake-smoke `
  --save-raw
```

Inspect the run:

```powershell
depthbatch inspect --run runs/fake-smoke --stdout-json
```

## Real PyTorch Path

Install optional dependencies and point DepthBatch at a local Depth Anything V2 Small checkpoint.

```powershell
pip install -e .[pytorch]
python scripts/download_da_v2_small.py

depthbatch infer-images `
  --backend pytorch `
  --model da-v2-small `
  --weights artifacts\weights\depth_anything_v2_vits.pth `
  --input tests\fixtures\images `
  --output runs\da-v2-small `
  --save-raw `
  --stdout-json
```

Recorded local GPU install commands for the current Windows/NVIDIA environment:

```powershell
pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu128 torch torchvision
pip uninstall -y onnxruntime
pip install onnxruntime-gpu
```

These GPU wheels remain environment-specific and are intentionally not pinned in `pyproject.toml`.

## ONNX Validation Path

```powershell
pip install -e .[pytorch,onnx]

depthbatch export-onnx `
  --backend pytorch `
  --model da-v2-small `
  --weights artifacts\weights\depth_anything_v2_vits.pth `
  --output artifacts\onnx `
  --overwrite `
  --stdout-json

depthbatch infer-onnx `
  --model da-v2-small `
  --onnx-path artifacts\onnx\artifacts\export\model.onnx `
  --input tests\fixtures\images `
  --output runs\onnx-infer `
  --overwrite `
  --save-raw `
  --stdout-json
```

Current alpha note: the validated ONNXRuntime path is static square export at `input_size=518`.

The recorded local CUDA path also succeeded with `--device cuda`.
