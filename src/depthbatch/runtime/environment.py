from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from depthbatch._version import __version__


def _optional_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit(cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def build_environment_snapshot(cwd: Path) -> dict[str, Any]:
    return {
        "depthbatch_version": __version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": str(cwd),
        "git_commit": _git_commit(cwd),
        "dependencies": {
            "numpy": _optional_version("numpy"),
            "opencv-python-headless": _optional_version("opencv-python-headless"),
            "onnxruntime": _optional_version("onnxruntime"),
            "torch": _optional_version("torch"),
            "transformers": _optional_version("transformers"),
        },
    }
