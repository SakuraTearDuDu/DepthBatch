class DepthBatchError(Exception):
    """Base exception for DepthBatch."""


class ConfigError(DepthBatchError):
    """Raised when configuration is invalid."""


class InputResolutionError(DepthBatchError):
    """Raised when an input path cannot be resolved."""


class BackendError(DepthBatchError):
    """Raised when a backend fails."""


class ArtifactError(DepthBatchError):
    """Raised when artifact writing fails."""
