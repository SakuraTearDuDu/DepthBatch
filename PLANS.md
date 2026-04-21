# DepthBatch V1 Plan Snapshot

This repository implements the approved V1 plan for:

- batch image inference
- batch video inference
- ONNX export
- ONNXRuntime inference validation
- benchmark reporting
- run inspection and manifest management

Core constraints:

- canonical provider: Depth Anything V2 native preprocessing/postprocessing semantics
- default model path: user-provided Depth Anything V2 Small weights
- no bundled weights or exported deployment artifacts
- provider/backend/pipeline separation with shared CLI and Python API contracts
