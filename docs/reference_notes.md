# Reference Notes

## Primary Reference

DepthBatch is primarily based on the inference semantics and usage patterns of `DepthAnything/Depth-Anything-V2`.

Key reference points adopted:

- `DepthAnythingV2(...).infer_image(raw_img, input_size)` style native inference
- image and video input patterns: single file, directory, or `.txt` list
- default `input_size=518`
- official preprocessing semantics:
  - BGR to RGB conversion
  - keep aspect ratio
  - resize by lower bound
  - ensure multiple of 14
  - OpenCV interpolation
  - standard normalization
  - resize prediction back to original resolution

## What DepthBatch Reuses

- Native preprocessing and postprocessing expectations
- Encoder preset mapping
- A minimal vendored runtime required to load official-style checkpoints

## What DepthBatch Does Not Copy

- Training or research workflow layout
- standalone script duplication
- visualization-only output assumptions
- permissive recursive file enumeration without media filtering
- community patch flows that require copying files into the upstream repository

## Secondary References

- `fabio-sim/Depth-Anything-ONNX`: ONNX export and inference workflow ideas
- `spacewalk01/depth-anything-tensorrt`: deployment CLI and engine-oriented organization
- `zhujiajian98/Depth-Anythingv2-TensorRT-python`: lightweight ONNX-to-TensorRT bridge ideas
- Hugging Face Transformers docs for Depth Anything V2: compatibility reference only

## Canonical Backend Rule

PyTorch native remains the canonical semantic baseline.

- PyTorch defines expected behavior.
- ONNXRuntime validates deployment compatibility, but in `0.1.0-alpha` the recorded path is a static square export contract rather than full aspect-ratio-preserving parity.
- Transformers is convenience-oriented and may differ numerically because of preprocessing and interpolation differences.

## Release-Validation Notes

- DA-V2 Small PyTorch with a local `.pth` checkpoint is the validated default path for this alpha.
- Dynamic ONNX export was investigated and intentionally not released as validated because traced DA-V2 export did not hold across varying non-square inputs in local testing.
- The release-validated ONNXRuntime path therefore uses static square export and square preprocessing derived from `export.json`.
