from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from depthbatch.errors import ArtifactError
from depthbatch.types import ArtifactsSection, RunPaths
from depthbatch.utils import ensure_parent, path_stem, relative_to

COLORMAP_NAMES = {
    "inferno": cv2.COLORMAP_INFERNO,
    "magma": cv2.COLORMAP_MAGMA,
    "plasma": cv2.COLORMAP_PLASMA,
    "turbo": cv2.COLORMAP_TURBO,
    "viridis": cv2.COLORMAP_VIRIDIS,
}


def normalize_depth(depth: np.ndarray, max_value: int) -> np.ndarray:
    depth = depth.astype(np.float32)
    min_value = float(depth.min())
    max_depth = float(depth.max())
    if max_depth <= min_value:
        return np.zeros(depth.shape, dtype=np.uint16 if max_value > 255 else np.uint8)
    scaled = (depth - min_value) / (max_depth - min_value)
    scaled *= float(max_value)
    dtype = np.uint16 if max_value > 255 else np.uint8
    return scaled.astype(dtype)


def colorize_depth(depth: np.ndarray, *, grayscale: bool, colormap: str) -> np.ndarray:
    depth8 = normalize_depth(depth, 255).astype(np.uint8)
    if grayscale:
        return np.repeat(depth8[..., None], 3, axis=2)
    cmap = COLORMAP_NAMES.get(colormap)
    if cmap is None:
        raise ArtifactError(f"Unsupported colormap: {colormap}")
    return cv2.applyColorMap(depth8, cmap)


def make_side_by_side(
    raw_bgr: np.ndarray, rendered_bgr: np.ndarray, margin: int = 24
) -> np.ndarray:
    separator = np.full((raw_bgr.shape[0], margin, 3), 255, dtype=np.uint8)
    return cv2.hconcat([raw_bgr, separator, rendered_bgr])


def _artifact_path(base_dir: Path, relative_stem: str, suffix: str) -> Path:
    path = base_dir / f"{relative_stem}{suffix}"
    ensure_parent(path)
    return path


def _write_image(path: Path, image: np.ndarray) -> None:
    success, encoded = cv2.imencode(path.suffix, image)
    if not success:
        raise ArtifactError(f"Failed to encode image artifact: {path}")
    path.write_bytes(encoded.tobytes())


def expected_primary_output(
    paths: RunPaths,
    relative_path: str,
    artifacts: ArtifactsSection,
) -> Path:
    relative_stem = path_stem(relative_path)
    if artifacts.save_raw:
        return paths.raw_dir / f"{relative_stem}.npy"
    if artifacts.save_uint16:
        return paths.depth_dir / f"{relative_stem}.png"
    if artifacts.save_visualization:
        return paths.preview_dir / f"{relative_stem}_color.png"
    return paths.preview_dir / f"{relative_stem}_side.png"


def write_depth_outputs(
    *,
    paths: RunPaths,
    relative_path: str,
    raw_bgr: np.ndarray,
    depth: np.ndarray,
    artifacts: ArtifactsSection,
) -> dict[str, str]:
    relative_stem = path_stem(relative_path)
    outputs: dict[str, str] = {}
    if artifacts.save_raw:
        raw_path = _artifact_path(paths.raw_dir, relative_stem, ".npy")
        np.save(raw_path, depth.astype(np.float32))
        outputs["raw_depth"] = relative_to(raw_path, paths.root)
    if artifacts.save_uint16:
        depth16 = normalize_depth(depth, 65535)
        depth16_path = _artifact_path(paths.depth_dir, relative_stem, ".png")
        _write_image(depth16_path, depth16)
        outputs["depth_uint16"] = relative_to(depth16_path, paths.root)
    rendered = colorize_depth(depth, grayscale=artifacts.grayscale, colormap=artifacts.colormap)
    if artifacts.save_visualization:
        color_path = _artifact_path(paths.preview_dir, relative_stem, "_color.png")
        _write_image(color_path, rendered)
        outputs["visualization"] = relative_to(color_path, paths.root)
    if artifacts.save_side_by_side and not artifacts.pred_only:
        side = make_side_by_side(raw_bgr, rendered)
        side_path = _artifact_path(paths.preview_dir, relative_stem, "_side.png")
        _write_image(side_path, side)
        outputs["side_by_side"] = relative_to(side_path, paths.root)
    return outputs
