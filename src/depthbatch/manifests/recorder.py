from __future__ import annotations

import json
from typing import Any

from depthbatch.types import RunPaths
from depthbatch.utils import dump_json, utc_now_iso


class ManifestRecorder:
    def __init__(
        self,
        *,
        paths: RunPaths,
        command: str,
        config_summary: dict[str, Any],
        environment: dict[str, Any],
    ) -> None:
        self._paths = paths
        self._command = command
        self._config_summary = config_summary
        self._environment = environment
        self._counts = {"completed": 0, "failed": 0, "skipped": 0}
        self._items: list[dict[str, Any]] = []
        self._started_at = utc_now_iso()
        self._paths.items_path.write_text("", encoding="utf-8")

    def record_item(self, record: dict[str, Any]) -> None:
        status = str(record.get("status", "failed"))
        if status in self._counts:
            self._counts[status] += 1
        self._items.append(record)
        with self._paths.items_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def finalize(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        summary = {
            "command": self._command,
            "started_at": self._started_at,
            "finished_at": utc_now_iso(),
            "counts": self._counts,
            "config": self._config_summary,
            "environment": self._environment,
            "items_path": self._paths.items_path.name,
        }
        if extra:
            summary.update(extra)
        dump_json(self._paths.manifest_path, summary)
        return summary
