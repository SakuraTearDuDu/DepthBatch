# DepthBatch v0.1.0-alpha

## Highlights

- GitHub-ready alpha release of the DepthBatch engineering wrapper around Depth Anything V2
- CI-safe fake backend plus recorded local DA-V2 Small validation
- PyTorch image inference, ONNX export, ONNXRuntime validation, benchmark, manifests, and inspection workflows

## Recorded Validation

- Environment: Windows 11 `10.0.26200`, Python `3.12.7`
- Dependencies: `torch 2.11.0+cpu`, `onnxruntime 1.24.4`, `opencv-python-headless 4.13.0.92`
- Official DA-V2 Small checkpoint SHA256: `715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378`
- Verified command chain:
  - real PyTorch inference on a single image and a directory
  - ONNX export from the local DA-V2 Small checkpoint
  - ONNXRuntime inference on a single image and a directory
  - backend benchmark on the repository fixtures
- Additional local CUDA evidence recorded on `2026-04-23`:
  - `torch 2.11.0+cu128`
  - `onnxruntime-gpu 1.25.0`
  - GPU: `NVIDIA GeForce RTX 4080 SUPER`
  - PyTorch CUDA inference succeeded
  - ONNXRuntime CUDAExecutionProvider inference succeeded
  - benchmark comparison now records normalized MAE / RMSE / max abs error between PyTorch and ONNXRuntime outputs

## Current Support Boundary

- `fake`: verified CI/smoke backend
- `pytorch`: alpha-supported for DA-V2 Small with a user-provided local checkpoint
- `onnxruntime`: alpha-supported for the static square export path
- `transformers`: experimental compatibility backend
- TensorRT: planned/documented only

## Weight Strategy

- The repository does not bundle the DA-V2 Small checkpoint
- Default distribution path is the official source plus `scripts/download_da_v2_small.py`
- The current `v0.1.0-alpha` pre-release mirrors the official checkpoint as a release asset, together with the SHA256 file
- GitHub Actions release automation only publishes code artifacts; checkpoint asset uploads remain manual

## Known Limits

- broad cross-machine GPU support is still unverified; current CUDA evidence is local to one Windows/NVIDIA environment
- dynamic ONNX export is not release-validated
- ONNX benchmark comparison is an engineering consistency signal, not a numerical-equivalence guarantee
- DA-V2 Base / Large / Giant remain outside the validated alpha path
