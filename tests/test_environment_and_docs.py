from __future__ import annotations

import re
from pathlib import Path

import yaml

from depthbatch.runtime.environment import build_environment_snapshot


def test_environment_snapshot_includes_accelerator_keys(repo_root: Path) -> None:
    snapshot = build_environment_snapshot(repo_root)
    accelerators = snapshot["accelerators"]
    assert "torch" in accelerators
    assert "cuda" in accelerators
    assert "gpu" in accelerators
    assert "onnxruntime_providers" in accelerators
    assert isinstance(accelerators["onnxruntime_providers"], list)
    assert isinstance(accelerators["gpu"]["names"], list)


def test_release_workflow_configures_prerelease_rules(repo_root: Path) -> None:
    workflow_path = repo_root / ".github" / "workflows" / "release.yml"
    workflow_text = workflow_path.read_text(encoding="utf-8")
    workflow = yaml.safe_load(workflow_text)
    assert workflow["on"]["push"]["tags"] == ["v*"]
    assert workflow["permissions"]["contents"] == "write"
    assert "alpha" in workflow_text
    assert "beta" in workflow_text
    assert "rc" in workflow_text
    assert "action-gh-release" in workflow_text


def test_bilingual_readme_and_quickstart_links_resolve(repo_root: Path) -> None:
    for relative_path in ("README.md", "README.zh-CN.md", "docs/quickstart.zh-CN.md"):
        document = repo_root / relative_path
        content = document.read_text(encoding="utf-8")
        for target in _extract_local_links(content):
            resolved = (document.parent / target).resolve()
            assert resolved.exists(), f"Broken link in {relative_path}: {target}"


def _extract_local_links(markdown: str) -> list[str]:
    links: list[str] = []
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", markdown):
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        if target.startswith("<") and target.endswith(">"):
            target = target[1:-1]
        if not target:
            continue
        links.append(target)
    return links
