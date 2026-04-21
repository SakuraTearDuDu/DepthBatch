from __future__ import annotations

import json
from pathlib import Path

from depthbatch.types import RunResult
from depthbatch.utils import dump_json


def inspect_run(run_root: Path) -> RunResult:
    root = run_root.resolve()
    manifest_path = root / "manifest.json"
    items_path = root / "items.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = []
    if items_path.exists():
        items = [
            json.loads(line)
            for line in items_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    counts = {"completed": 0, "failed": 0, "skipped": 0}
    for item in items:
        status = item.get("status")
        if status in counts:
            counts[status] += 1
    summary = {
        "run_root": str(root),
        "manifest": manifest,
        "counts": counts,
        "item_count": len(items),
    }
    report_path = root / "reports" / "inspect.json"
    dump_json(report_path, summary)
    return RunResult(run_root=root, manifest_path=manifest_path, summary=summary)
