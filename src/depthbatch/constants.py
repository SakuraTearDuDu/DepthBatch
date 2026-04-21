from __future__ import annotations

DEFAULT_BACKEND = "pytorch"
DEFAULT_COLORMAP = "turbo"
DEFAULT_INPUT_SIZE = 518
DEFAULT_MODEL_ALIAS = "da-v2-small"
DEFAULT_PROVIDER = "depth_anything_v2"

IMAGE_SUFFIXES = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".ppm",
    ".tif",
    ".tiff",
    ".webp",
}

VIDEO_SUFFIXES = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
}

RUN_ARTIFACT_DIRS = {
    "depth": "artifacts/depth",
    "raw": "artifacts/raw",
    "preview": "artifacts/preview",
    "video": "artifacts/video",
    "export": "artifacts/export",
    "reports": "reports",
}
