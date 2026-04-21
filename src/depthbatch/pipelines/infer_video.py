from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from depthbatch.io import colorize_depth, make_side_by_side, resolve_input_items
from depthbatch.pipelines.common import create_run_context, open_backend_session, resolve_provider
from depthbatch.types import AppConfig, PreparedSample, RunResult
from depthbatch.utils import ensure_parent, path_stem, relative_to


def infer_video(config: AppConfig) -> RunResult:
    config.inputs.mode = "video"
    if config.inputs.input is None:
        raise ValueError("infer-video requires an input path.")
    provider, preset = resolve_provider(config)
    paths, _environment, recorder = create_run_context(config)
    session = open_backend_session(config, preset)
    items = resolve_input_items(config.inputs.input, "video")
    try:
        for index, item in enumerate(items, start=1):
            print(f"[infer-video] {index}/{len(items)} {item.source_path}")
            capture = cv2.VideoCapture(str(item.source_path))
            if not capture.isOpened():
                recorder.record_item(
                    {
                        "input_path": str(item.source_path),
                        "relative_path": item.relative_path,
                        "status": "failed",
                        "reason": "Could not open video.",
                    }
                )
                continue
            frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
            relative_stem = path_stem(item.relative_path)
            output_video_path = paths.video_dir / f"{relative_stem}.mp4"
            ensure_parent(output_video_path)
            render_width = frame_width if config.artifacts.pred_only else frame_width * 2 + 24
            writer = cv2.VideoWriter(
                str(output_video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore[attr-defined]
                fps,
                (render_width, frame_height),
            )
            pending: list[PreparedSample] = []
            frame_count = 0
            processed_frames = 0
            frame_dir = paths.video_dir / relative_stem / "frames"
            if config.artifacts.output_frames:
                frame_dir.mkdir(parents=True, exist_ok=True)
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if frame_count % max(config.inputs.stride, 1) != 0:
                        frame_count += 1
                        continue
                    prepared = provider.prepare_sample(
                        item,
                        frame,
                        input_size=config.model.input_size,
                    )
                    pending.append(prepared)
                    if len(pending) >= config.backend.batch_size:
                        _flush_video_batch(
                            pending=pending,
                            provider=provider,
                            session=session,
                            config=config,
                            writer=writer,
                            frame_dir=frame_dir if config.artifacts.output_frames else None,
                            frame_start_index=processed_frames,
                        )
                        processed_frames += len(pending)
                        pending = []
                    frame_count += 1
                if pending:
                    _flush_video_batch(
                        pending=pending,
                        provider=provider,
                        session=session,
                        config=config,
                        writer=writer,
                        frame_dir=frame_dir if config.artifacts.output_frames else None,
                        frame_start_index=processed_frames,
                    )
                    processed_frames += len(pending)
            finally:
                capture.release()
                writer.release()
            recorder.record_item(
                {
                    "input_path": str(item.source_path),
                    "relative_path": item.relative_path,
                    "status": "completed",
                    "frames_total": frame_count,
                    "frames_processed": processed_frames,
                    "artifacts": {"video": relative_to(output_video_path, paths.root)},
                }
            )
    finally:
        session.close()
    summary = recorder.finalize(
        {
            "mode": "video",
            "item_count": len(items),
            "backend": session.inspect(),
        }
    )
    return RunResult(run_root=paths.root, manifest_path=paths.manifest_path, summary=summary)


def _flush_video_batch(
    *,
    pending: list[PreparedSample],
    provider: Any,
    session: Any,
    config: AppConfig,
    writer: Any,
    frame_dir: Path | None,
    frame_start_index: int,
) -> None:
    batch = np.stack([sample.tensor for sample in pending], axis=0).astype(np.float32)
    output = session.infer(batch)
    for offset, (sample, predicted_depth) in enumerate(zip(pending, output.depths, strict=True)):
        depth = provider.postprocess_depth(predicted_depth, sample.original_size)
        rendered = colorize_depth(
            depth,
            grayscale=config.artifacts.grayscale,
            colormap=config.artifacts.colormap,
        )
        if config.artifacts.pred_only:
            frame_to_write = rendered
        else:
            frame_to_write = make_side_by_side(sample.original_bgr, rendered)
        writer.write(frame_to_write)
        if frame_dir is not None:
            frame_path = frame_dir / f"frame_{frame_start_index + offset:06d}.png"
            ensure_parent(frame_path)
            cv2.imwrite(str(frame_path), frame_to_write)
