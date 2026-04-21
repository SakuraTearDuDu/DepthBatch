# Contributing

## Ground Rules

- All file operations must stay inside `D:\github_test_DepthBatch`.
- Do not add bundled model weights, ONNX exports, or TensorRT engines to the repository.
- Do not claim a backend is verified unless the PR includes reproducible commands and evidence.
- Fake backend results must be labeled as fake or smoke-only in code, tests, examples, and docs.

## Local Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .[dev]
```

Optional runtime extras:

```powershell
pip install -e .[pytorch]
pip install -e .[onnx]
pip install -e .[transformers]
```

## Required Checks

```powershell
ruff check .
ruff format --check .
mypy src
pytest
python -m build
```

## PR Expectations

- Explain whether the change targets fake backend coverage, real backend support, or docs only.
- Include updated tests for behavior changes.
- Update `docs/devlog.md` for significant architectural or verification decisions.
- Update `README.md`, `docs/license_notes.md`, or `docs/reference_notes.md` when public behavior changes.

## Backend Claims

- `supported`: requires reproducible evidence in the repo or documented manual validation.
- `experimental`: implementation exists, but validation is limited or environment-dependent.
- `planned`: interface or docs exist, but no implementation should be implied.
