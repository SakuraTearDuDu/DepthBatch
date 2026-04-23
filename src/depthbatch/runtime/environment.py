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


def _torch_accelerator_snapshot() -> dict[str, Any]:
    version = _optional_version("torch")
    snapshot: dict[str, Any] = {
        "version": version,
        "cuda_available": False,
        "cuda_version": None,
        "device_count": 0,
        "devices": [],
    }
    try:
        import torch
    except ImportError:
        return snapshot
    snapshot["version"] = torch.__version__
    cuda_available = bool(torch.cuda.is_available())
    snapshot["cuda_available"] = cuda_available
    snapshot["cuda_version"] = getattr(torch.version, "cuda", None)
    if not cuda_available:
        return snapshot
    device_count = int(torch.cuda.device_count())
    snapshot["device_count"] = device_count
    snapshot["devices"] = [
        {"index": index, "name": torch.cuda.get_device_name(index)} for index in range(device_count)
    ]
    return snapshot


def _onnxruntime_providers_snapshot() -> list[str]:
    try:
        import onnxruntime as ort
    except ImportError:
        return []
    try:
        return [str(provider) for provider in ort.get_available_providers()]
    except Exception:
        return []


def _system_gpu_snapshot() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return {"names": [], "driver_version": None, "source": None}
    if completed.returncode != 0:
        return {"names": [], "driver_version": None, "source": "nvidia-smi"}
    names: list[str] = []
    driver_version: str | None = None
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if not parts or not parts[0]:
            continue
        names.append(parts[0])
        if driver_version is None and len(parts) > 1:
            driver_version = parts[1] or None
    return {"names": names, "driver_version": driver_version, "source": "nvidia-smi"}


def build_environment_snapshot(cwd: Path) -> dict[str, Any]:
    torch_snapshot = _torch_accelerator_snapshot()
    system_gpu = _system_gpu_snapshot()
    onnxruntime_version = _optional_version("onnxruntime") or _optional_version("onnxruntime-gpu")
    return {
        "depthbatch_version": __version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": str(cwd),
        "git_commit": _git_commit(cwd),
        "dependencies": {
            "numpy": _optional_version("numpy"),
            "opencv-python-headless": _optional_version("opencv-python-headless"),
            "onnxruntime": onnxruntime_version,
            "onnxruntime-gpu": _optional_version("onnxruntime-gpu"),
            "torch": _optional_version("torch"),
            "transformers": _optional_version("transformers"),
        },
        "accelerators": {
            "torch": torch_snapshot,
            "cuda": {
                "available": bool(torch_snapshot["cuda_available"]),
                "version": torch_snapshot["cuda_version"],
                "device_count": int(torch_snapshot["device_count"]),
            },
            "gpu": system_gpu,
            "onnxruntime_providers": _onnxruntime_providers_snapshot(),
        },
    }
