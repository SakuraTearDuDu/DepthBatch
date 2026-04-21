# Deployment Notes

## ONNXRuntime

DepthBatch treats ONNXRuntime as the first deployment validation target after native PyTorch.

Recorded `0.1.0-alpha` validation behavior:

- export from the canonical PyTorch checkpoint path
- write `artifacts/export/model.onnx` plus `artifacts/export/export.json`
- validate the exported model with ONNXRuntime on CPU
- reuse the standard run directory layout for `infer-onnx`

Recorded environment:

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

Important boundary for this alpha:

- the release-validated ONNX path is static square export with `input_size=518`
- `infer-onnx` reads the sibling `export.json` and uses `keep_aspect_ratio=false`
- dynamic ONNX export is not release-validated for DA-V2 Small in this version
- benchmark evidence is CPU-only

## TensorRT

TensorRT is a documented extension point only.

V1 does not ship:

- TensorRT backend implementation
- engine build automation
- TensorRT CI coverage

The architecture keeps a backend capability boundary so TensorRT can be added later without changing CLI or manifest contracts.
