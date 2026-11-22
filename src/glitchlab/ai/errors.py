"""Error types for the optional AI features (all are user-facing)."""

from __future__ import annotations

from ..engine.errors import DependencyError


class AIError(DependencyError):
    """Base class for AI problems that the user can fix (install, download, hardware)."""


class AIUnavailable(AIError):
    """The AI engine (PyTorch / transformers / diffusers) is not installed."""

    def __init__(self, message: str = "", *, detail: str = ""):
        super().__init__(message or (
            "The AI engine is not installed. Use AI › Install AI engine… (about 165 MB, downloaded once; "
            "everything then runs offline on this computer)."), detail=detail)


class ModelMissing(AIError):
    """Model weights needed by an effect have not been downloaded yet."""

    def __init__(self, model_ids: list[str], message: str = "", *, detail: str = ""):
        from .models import MODELS

        self.model_ids = list(model_ids)
        if not message:
            parts = [f"{MODELS[m].name} ({MODELS[m].size_text}, {MODELS[m].licence})" for m in self.model_ids]
            message = ("Model weights are missing: " + "; ".join(parts) +
                       ". Use AI › Models… to download them (needed once; they are stored on this computer).")
        super().__init__(message, detail=detail)


class AIHardwareError(AIError):
    """The AI engine cannot run on this computer as configured."""
