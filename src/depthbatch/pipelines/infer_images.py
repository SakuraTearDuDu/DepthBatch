from __future__ import annotations

import time

from depthbatch.io import resolve_input_items
from depthbatch.pipelines.common import (
    create_run_context,
    flush_prepared_batch,
    open_backend_session,
    read_image,
    resolve_provider,
    should_skip_existing,
)
from depthbatch.types import AppConfig, PreparedSample, RunResult


def infer_images(config: AppConfig) -> RunResult:
    config.inputs.mode = "images"
    if config.inputs.input is None:
        raise ValueError("infer-images requires an input path.")
    provider, preset = resolve_provider(config)
    paths, _environment, recorder = create_run_context(config)
    session = open_backend_session(config, preset)
    items = resolve_input_items(config.inputs.input, "images")
    pending: list[PreparedSample] = []
    try:
        for index, item in enumerate(items, start=1):
            print(f"[infer-images] {index}/{len(items)} {item.source_path}")
            skip, reason = should_skip_existing(config, paths, item.relative_path)
            if skip:
                status = (
                    "skipped" if config.run.skip_existing and not config.run.overwrite else "failed"
                )
                recorder.record_item(
                    {
                        "input_path": str(item.source_path),
                        "relative_path": item.relative_path,
                        "status": status,
                        "reason": reason,
                    }
                )
                continue
            try:
                read_started = time.perf_counter()
                raw_image = read_image(item.source_path)
                read_seconds = time.perf_counter() - read_started
                prep_started = time.perf_counter()
                prepared = provider.prepare_sample(
                    item,
                    raw_image,
                    input_size=config.model.input_size,
                    keep_aspect_ratio=_keep_aspect_ratio(config),
                )
                preprocess_seconds = time.perf_counter() - prep_started
                prepared.timing = {
                    "read_seconds": read_seconds,
                    "preprocess_seconds": preprocess_seconds,
                }
                if pending and (
                    len(pending) >= config.backend.batch_size
                    or pending[0].tensor.shape != prepared.tensor.shape
                ):
                    flush_prepared_batch(
                        pending=pending,
                        provider=provider,
                        session=session,
                        config=config,
                        paths=paths,
                        recorder=recorder,
                    )
                    pending = []
                pending.append(prepared)
            except Exception as exc:  # noqa: BLE001
                recorder.record_item(
                    {
                        "input_path": str(item.source_path),
                        "relative_path": item.relative_path,
                        "status": "failed",
                        "reason": str(exc),
                    }
                )
        flush_prepared_batch(
            pending=pending,
            provider=provider,
            session=session,
            config=config,
            paths=paths,
            recorder=recorder,
        )
    finally:
        session.close()
    summary = recorder.finalize(
        {
            "mode": "images",
            "item_count": len(items),
            "backend": session.inspect(),
        }
    )
    return RunResult(run_root=paths.root, manifest_path=paths.manifest_path, summary=summary)


def _keep_aspect_ratio(config: AppConfig) -> bool:
    if config.backend.name == "onnxruntime":
        return bool(config.backend.session_options.get("keep_aspect_ratio", False))
    return True
