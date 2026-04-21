from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, cast

import yaml


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._/-]+", "-", value.replace("\\", "/")).strip("-")
    return cleaned or "item"


def short_hash(value: str, length: int = 8) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return to_serializable(asdict(cast(Any, value)))
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_serializable(item) for item in value]
    return value


def dump_json(path: Path, payload: Any) -> None:
    ensure_parent(path)
    path.write_text(
        json.dumps(to_serializable(payload), indent=2, sort_keys=True), encoding="utf-8"
    )


def dump_yaml(path: Path, payload: Any) -> None:
    ensure_parent(path)
    path.write_text(yaml.safe_dump(to_serializable(payload), sort_keys=False), encoding="utf-8")


def relative_to(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def path_stem(relative_path: str) -> str:
    return str(PurePosixPath(relative_path).with_suffix(""))


def parse_key_value(value: str) -> tuple[str, Any]:
    if "=" not in value:
        raise ValueError(f"Expected key=value syntax, received: {value}")
    key, raw = value.split("=", 1)
    return key.strip(), yaml.safe_load(raw)
