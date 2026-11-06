"""Exception types.  Every user-facing error carries a short, understandable
message; technical details go to the log."""


class GlitchLabError(Exception):
    """Base class for errors that can be shown to the user directly."""

    def __init__(self, message: str, *, detail: str = ""):
        super().__init__(message)
        self.message = message
        self.detail = detail

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


class ImageLoadError(GlitchLabError):
    pass


class UnsupportedFormatError(ImageLoadError):
    pass


class ImageTooLargeError(ImageLoadError):
    def __init__(self, message: str, *, width: int, height: int, limit: int):
        super().__init__(message)
        self.width = width
        self.height = height
        self.limit = limit


class EffectError(GlitchLabError):
    """An effect could not produce a valid result (e.g. undecodable JPEG)."""


class DependencyError(GlitchLabError):
    """A required external dependency (FFmpeg) is missing or incompatible."""


class ExportError(GlitchLabError):
    pass


class ProjectError(GlitchLabError):
    pass


class Cancelled(Exception):
    """Raised inside workers when the user cancels a render."""
