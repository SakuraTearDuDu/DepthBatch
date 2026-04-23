# Deployment Notes

## ONNXRuntime

DepthBatch treats ONNXRuntime as the first deployment validation target after native PyTorch.

Recorded `0.1.0-alpha` validation behavior:

- export from the canonical PyTorch checkpoint path
- write `artifacts/export/model.onnx` plus `artifacts/export/export.json`
- validate the exported model with ONNXRuntime on CPU during `export-onnx`
- reuse the standard run directory layout for `infer-onnx`

Recorded CPU release-validation environment:

- Windows 11 `10.0.26200`
- Python `3.12.7`
- `torch 2.11.0+cpu`
- `onnxruntime 1.24.4`
- `opencv-python-headless 4.13.0.92`

Recorded release-validated command chain:

```powershell
python scripts/download_da_v2_small.py

depthbatch infer-images `
  --backend pytorch `
  --model da-v2-small `
  --weights artifacts\weights\depth_anything_v2_vits.pth `
  --input tests\fixtures\images `
  --output runs\real-small-pytorch-folder `
  --save-raw `
  --stdout-json

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
  --output runs\real-small-onnx-folder `
  --overwrite `
  --save-raw `
  --stdout-json
```

For convenience, the current `v0.1.0-alpha` GitHub pre-release also mirrors the official DA-V2 Small checkpoint as a release asset. The repository source tree still does not bundle the checkpoint.

Recorded local CUDA environment on `2026-04-23`:

- Windows 11 `10.0.26200`
- Python `3.12.7`
- `torch 2.11.0+cu128`
- `onnxruntime-gpu 1.25.0`
- GPU: `NVIDIA GeForce RTX 4080 SUPER`
- ONNXRuntime providers: `CUDAExecutionProvider`, `CPUExecutionProvider`

Recorded local GPU install commands:

```powershell
pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu128 torch torchvision
pip uninstall -y onnxruntime
pip install onnxruntime-gpu
```

Recorded local CUDA command chain:

```powershell
depthbatch infer-images `
  --backend pytorch `
  --device cuda `
  --model da-v2-small `
  --weights artifacts\weights\depth_anything_v2_vits.pth `
  --input tests\fixtures\images `
  --output runs\gpu-small-pytorch-folder `
  --overwrite `
  --save-raw `
  --stdout-json

depthbatch export-onnx `
  --backend pytorch `
  --model da-v2-small `
  --weights artifacts\weights\depth_anything_v2_vits.pth `
  --output artifacts\onnx-gpu `
  --overwrite `
  --stdout-json

depthbatch infer-onnx `
  --model da-v2-small `
  --onnx-path artifacts\onnx-gpu\artifacts\export\model.onnx `
  --device cuda `
  --input tests\fixtures\images `
  --output runs\gpu-small-onnx-folder `
  --overwrite `
  --save-raw `
  --stdout-json

depthbatch benchmark `
  --model da-v2-small `
  --weights artifacts\weights\depth_anything_v2_vits.pth `
  --onnx-path artifacts\onnx-gpu\artifacts\export\model.onnx `
  --device cuda `
  --input tests\fixtures\images `
  --output runs\gpu-small-benchmark `
  --overwrite `
  --stdout-json
```

Recorded local CUDA benchmark evidence on `tests/fixtures/images`:

- PyTorch CUDA: `0.17991710000205785` total seconds, `0.02998618333367631` average seconds per image
- ONNXRuntime CUDA: `0.11646099998324644` total seconds, `0.019410166663874406` average seconds per image
- Statistical comparison summary:
  - mean normalized MAE: `0.02774494375626091`
  - mean normalized RMSE: `0.03597149583220016`
  - max normalized absolute error: `0.16140271723270416`

Important boundary for this alpha:

- the release-validated ONNX path is static square export with `input_size=518`
- `infer-onnx` reads the sibling `export.json` and uses `keep_aspect_ratio=false`
- dynamic ONNX export is not release-validated for DA-V2 Small in this version
- `benchmark` comparison reports statistical consistency, not numerical equivalence with canonical PyTorch preprocessing
- export-time ONNX smoke remains CPU-based for deterministic release checks
- recorded ORT CUDA evidence is local to the environment above and should not be read as broad cross-machine support

## Release Automation

Tag pushes that match `v*` now trigger `.github/workflows/release.yml`.

The workflow:

- installs the package with `.[dev]`
- runs `ruff`, `mypy`, `pytest`, and `python -m build`
- uploads `dist/*`
- creates or updates a GitHub release
- marks tags containing `alpha`, `beta`, or `rc` as pre-releases
- uses `docs/release_notes_<tag>.md` when present, otherwise falls back to generated notes

The workflow does not auto-download or auto-upload model checkpoints. If a release should mirror the official DA-V2 Small checkpoint, upload the checkpoint asset and checksum as a manual post-step after policy review.

## TensorRT

TensorRT is a documented extension point only.

V1 does not ship:

- TensorRT backend implementation
- engine build automation
- TensorRT CI coverage

The architecture keeps a backend capability boundary so TensorRT can be added later without changing CLI or manifest contracts.
