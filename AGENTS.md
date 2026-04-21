# AGENTS

## Repository Boundaries

- Only operate inside `D:\github_test_DepthBatch`.
- The only allowed virtual environment location is `D:\github_test_DepthBatch\.venv`.
- Temporary files, caches, logs, and generated outputs must remain under this repository root.

## Engineering Rules

- DepthBatch is an engineering wrapper around Depth Anything V2 workflows, not a training framework.
- Keep provider-specific logic under `src/depthbatch/providers/`.
- Keep CLI thin; business logic belongs in pipelines, backends, and IO/manifests.
- Treat PyTorch native as the canonical semantic baseline for preprocessing and postprocessing.

## Verification Rules

- Never invent performance, export, or accuracy results.
- Mark fake backend paths explicitly in tests, examples, and docs.
- Record significant decisions and verification gaps in `docs/devlog.md`.

## Release Rules

- Do not bundle model weights or exported artifacts.
- Default public guidance must stay centered on Depth Anything V2 Small and user-provided local weights.
- If evidence is limited to fake backend coverage, keep releases in pre-release/alpha positioning.
