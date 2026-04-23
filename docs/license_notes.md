# License Notes

## Repository Code

- DepthBatch repository code: Apache-2.0
- Vendored Depth Anything V2 runtime subset: Apache-2.0

## Model and Artifact Boundary

Repository code licenses do not automatically apply to model weights or exported artifacts.

Default supported guidance:

- Depth Anything V2 Small: validated default release path with user-provided local weights
- Base / Large / Giant: user-supplied, user-verified, license-sensitive, and outside the validated alpha path

## Distribution Policy

- Do not commit model weights
- Do not bundle exported ONNX files in the repository
- Do not bundle TensorRT engines
- GitHub Actions release automation publishes code artifacts only; checkpoint mirroring remains a manual policy-reviewed step
- Record artifact source, checksum, and license notes in run metadata when available

## Artifact Matrix

| Artifact | Default repo policy | Notes |
| --- | --- | --- |
| DepthBatch source | distributable | Apache-2.0 |
| Vendored upstream runtime subset | distributable | Apache-2.0 attribution retained |
| DA-V2 Small local checkpoint | user supplied | validated default release path; official source recorded below |
| DA-V2 Base/Large/Giant checkpoint | user supplied | verify license before use |
| Exported ONNX from user checkpoint | user generated | treat as derived artifact tied to upstream model license |
| TensorRT engine | user generated | not a V1 repository deliverable |

## Recorded DA-V2 Small Checkpoint

- official source page: `https://huggingface.co/depth-anything/Depth-Anything-V2-Small`
- expected filename: `depth_anything_v2_vits.pth`
- recorded SHA256: `715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378`
- repository policy:
  - do not commit the checkpoint into Git history
  - prefer external distribution such as a GitHub release asset only if release tooling and policy checks are available
  - otherwise use the downloader script plus checksum verification
- current pre-release mirror:
  - release page: `https://github.com/SakuraTearDuDu/DepthBatch/releases/tag/v0.1.0-alpha`
  - checkpoint asset: `depth_anything_v2_vits.pth`
  - checksum asset: `depth_anything_v2_vits.sha256`
  - upload mode: manual post-step after the code release workflow completes

## Important Caveat

Public pages for different Depth Anything V2 artifacts may not all present identical license language. Treat license checks as artifact-specific, not family-wide.

This document is engineering guidance, not legal advice.
