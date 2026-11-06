"""Headless processing engine for Glitch Lab.

Nothing in this package imports Qt, so every effect can be tested and run
without a display (see ``glitchlab.cli`` and ``tests/``).
"""

from .errors import (  # noqa: F401
    Cancelled,
    DependencyError,
    EffectError,
    ExportError,
    GlitchLabError,
    ImageLoadError,
    ImageTooLargeError,
    ProjectError,
    UnsupportedFormatError,
)
