# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0-alpha] - 2026-04-22

### Added
- Scaffolded the DepthBatch V1 package, CLI, configuration system, manifests, and docs.
- Added canonical provider/backend/pipeline architecture centered on Depth Anything V2 workflows.
- Added fake backend and smoke-test fixtures for CI-safe verification.
- Added `depthbatch --version`.
- Added a repository-local DA-V2 Small downloader script and checksum record.

### Changed
- Established Apache-2.0 licensing and NOTICE-based attribution for vendored upstream runtime code.
- Hardened Windows image IO so failed writes no longer pass silently.
- Tightened ONNX export for DA-V2 Small to the release-validated static square path.

### Validated
- Real DA-V2 Small checkpoint source: `https://huggingface.co/depth-anything/Depth-Anything-V2-Small`
- Real checkpoint file: `depth_anything_v2_vits.pth`
- Real checkpoint SHA256: `715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378`
- Recorded environment: Windows 11 `10.0.26200`, Python `3.12.7`, `torch 2.11.0+cpu`, `onnxruntime 1.24.4`, `opencv-python-headless 4.13.0.92`
- Verified commands:
  - `depthbatch infer-images --backend pytorch --model da-v2-small --weights artifacts\\weights\\depth_anything_v2_vits.pth --input tests\\fixtures\\images --output runs\\real-small-pytorch-folder --save-raw --stdout-json`
  - `depthbatch export-onnx --backend pytorch --model da-v2-small --weights artifacts\\weights\\depth_anything_v2_vits.pth --output artifacts\\onnx --overwrite --stdout-json`
  - `depthbatch infer-onnx --model da-v2-small --onnx-path artifacts\\onnx\\artifacts\\export\\model.onnx --input tests\\fixtures\\images --output runs\\real-small-onnx-folder --overwrite --save-raw --stdout-json`
  - `depthbatch benchmark --model da-v2-small --weights artifacts\\weights\\depth_anything_v2_vits.pth --onnx-path artifacts\\onnx\\artifacts\\export\\model.onnx --input tests\\fixtures\\images --output runs\\real-small-benchmark --overwrite --stdout-json`
- Recorded benchmark on `tests/fixtures/images` (2 images, CPU-only):
  - PyTorch: `2.2837s` total, `0.3806s/image`
  - ONNXRuntime: `3.5217s` total, `0.5869s/image`

### Known
- Real PyTorch and ONNXRuntime paths require user-provided local weights and optional dependencies.
- ONNXRuntime validation in `0.1.0-alpha` is limited to the static square export path (`input_size=518`, `keep_aspect_ratio=false`).
- Dynamic ONNX export is intentionally not release-validated for DA-V2 Small.
- Transformers remains compatibility-only and is not part of the validated release path.
- DA-V2 Base / Large / Giant remain outside the validated alpha path.
- TensorRT remains a documented future extension point only.
