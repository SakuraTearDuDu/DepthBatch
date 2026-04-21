from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import yaml

from depthbatch.errors import ConfigError
from depthbatch.types import (
    AppConfig,
    ArtifactsSection,
    BackendSection,
    BenchmarkSection,
    InputsSection,
    ModelSection,
    RunSection,
    RuntimeSection,
)
from depthbatch.utils import parse_key_value, to_serializable

SECTION_TYPES = {
    "run": RunSection,
    "model": ModelSection,
    "backend": BackendSection,
    "inputs": InputsSection,
    "artifacts": ArtifactsSection,
    "runtime": RuntimeSection,
    "benchmark": BenchmarkSection,
}

PATH_FIELDS = {
    "run": {"output", "config_path"},
    "model": {"weights", "onnx_path"},
    "inputs": {"input"},
}


def default_config_dict() -> dict[str, Any]:
    return cast(dict[str, Any], to_serializable(asdict(AppConfig())))


def load_config_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ConfigError("Config file must contain a mapping at the top level.")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def set_nested(mapping: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    current = mapping
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def parse_set_overrides(values: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for value in values:
        key, parsed = parse_key_value(value)
        set_nested(overrides, key, parsed)
    return overrides


def build_config(
    *,
    command: str,
    config_path: Path | None = None,
    profile: str | None = None,
    cli_overrides: dict[str, Any] | None = None,
    set_overrides: dict[str, Any] | None = None,
) -> AppConfig:
    raw = default_config_dict()
    raw["run"]["command"] = command
    if config_path is not None:
        file_mapping = load_config_file(config_path)
        if profile is not None:
            profiles = file_mapping.get("profiles", {})
            if profile not in profiles:
                raise ConfigError(f"Profile not found in config: {profile}")
            file_mapping = deep_merge(file_mapping, profiles[profile])
        raw = deep_merge(raw, file_mapping)
        raw.setdefault("run", {})["config_path"] = str(config_path)
    if cli_overrides:
        raw = deep_merge(raw, cli_overrides)
    if set_overrides:
        raw = deep_merge(raw, set_overrides)
    return config_from_mapping(raw)


def _coerce_section(section_name: str, mapping: dict[str, Any]) -> Any:
    section_type = SECTION_TYPES[section_name]
    section_mapping = dict(mapping)
    for field_name in PATH_FIELDS.get(section_name, set()):
        if section_mapping.get(field_name) is not None:
            section_mapping[field_name] = Path(section_mapping[field_name])
    return section_type(**section_mapping)


def config_from_mapping(mapping: dict[str, Any]) -> AppConfig:
    kwargs: dict[str, Any] = {}
    for name, _section_type in SECTION_TYPES.items():
        section_mapping = mapping.get(name, {})
        if not isinstance(section_mapping, dict):
            raise ConfigError(f"Config section '{name}' must be a mapping.")
        kwargs[name] = _coerce_section(name, section_mapping)
    return AppConfig(**kwargs)


def config_to_dict(config: AppConfig) -> dict[str, Any]:
    return cast(dict[str, Any], to_serializable(asdict(config)))
