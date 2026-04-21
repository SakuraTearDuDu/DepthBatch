from __future__ import annotations

from pathlib import Path

from depthbatch.constants import IMAGE_SUFFIXES, VIDEO_SUFFIXES
from depthbatch.errors import InputResolutionError
from depthbatch.types import InputItem
from depthbatch.utils import sanitize_name, short_hash


def _resolve_list_file(path: Path, mode: str) -> list[InputItem]:
    items: list[InputItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        item_path = Path(line)
        if not item_path.is_absolute():
            item_path = (path.parent / item_path).resolve()
        relative = item_path.name
        items.append(InputItem(source_path=item_path, relative_path=relative, source_group="list"))
    return _filter_items(items, mode)


def _suffixes_for_mode(mode: str) -> set[str]:
    if mode == "images":
        return IMAGE_SUFFIXES
    if mode == "video":
        return VIDEO_SUFFIXES
    raise InputResolutionError(f"Unsupported input mode: {mode}")


def _filter_items(items: list[InputItem], mode: str) -> list[InputItem]:
    suffixes = _suffixes_for_mode(mode)
    filtered = [item for item in items if item.source_path.suffix.lower() in suffixes]
    if not filtered:
        raise InputResolutionError(f"No supported {mode} inputs were found.")
    return _dedupe_items(filtered)


def _dedupe_items(items: list[InputItem]) -> list[InputItem]:
    seen: dict[str, int] = {}
    deduped: list[InputItem] = []
    for item in items:
        relative = sanitize_name(item.relative_path)
        if relative in seen:
            relative_path = (
                f"{Path(relative).stem}-{short_hash(str(item.source_path))}{Path(relative).suffix}"
            )
        else:
            relative_path = relative
        seen[relative] = seen.get(relative, 0) + 1
        deduped.append(
            InputItem(
                source_path=item.source_path.resolve(),
                relative_path=relative_path,
                source_group=item.source_group,
            )
        )
    return deduped


def resolve_input_items(input_path: Path, mode: str) -> list[InputItem]:
    path = input_path.resolve()
    if not path.exists():
        raise InputResolutionError(f"Input path does not exist: {path}")
    if path.is_file() and path.suffix.lower() == ".txt":
        return _resolve_list_file(path, mode)
    if path.is_file():
        item = InputItem(source_path=path, relative_path=path.name, source_group="file")
        return _filter_items([item], mode)
    if not path.is_dir():
        raise InputResolutionError(f"Unsupported input path: {path}")
    suffixes = _suffixes_for_mode(mode)
    items = [
        InputItem(
            source_path=file_path.resolve(),
            relative_path=str(file_path.relative_to(path)).replace("\\", "/"),
            source_group="directory",
        )
        for file_path in sorted(path.rglob("*"))
        if file_path.is_file() and file_path.suffix.lower() in suffixes
    ]
    return _filter_items(items, mode)
