"""Optional local AI support for Glitch Lab.

Nothing in this package imports PyTorch at import time.  The AI engine
(PyTorch, transformers, diffusers) is either installed in the Python
environment (development) or downloaded by the program into the user's data
folder (Windows build, see :mod:`glitchlab.ai.runtime`).  Model weights are
downloaded on request from pinned Hugging Face revisions
(:mod:`glitchlab.ai.models`) and every inference result is cached
(:mod:`glitchlab.ai.jobs`).  All processing is local.
"""

from .errors import AIError, AIHardwareError, AIUnavailable, ModelMissing

__all__ = ["AIError", "AIHardwareError", "AIUnavailable", "ModelMissing"]
