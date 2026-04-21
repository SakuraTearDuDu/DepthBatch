from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from depthbatch.types import BackendCapabilities, BackendOutput


class BackendSession(ABC):
    @abstractmethod
    def capabilities(self) -> BackendCapabilities:
        raise NotImplementedError

    @abstractmethod
    def infer(self, batch: np.ndarray) -> BackendOutput:
        raise NotImplementedError

    @abstractmethod
    def inspect(self) -> dict[str, Any]:
        raise NotImplementedError

    def close(self) -> None:
        return None


class ExportCapableSession(BackendSession):
    @abstractmethod
    def export_onnx(
        self,
        output_path: Path,
        *,
        input_size: int,
        opset: int,
        dynamic: bool,
    ) -> dict[str, Any]:
        raise NotImplementedError
