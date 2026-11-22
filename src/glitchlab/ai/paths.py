"""Where the AI engine, model weights and the inference cache live."""

from __future__ import annotations

import os
from pathlib import Path

from ..engine.settings import data_dir


def ai_root() -> Path:
    env = os.environ.get("GLITCHLAB_AI_DIR")
    p = Path(env) if env else data_dir() / "ai"
    p.mkdir(parents=True, exist_ok=True)
    return p


def models_dir() -> Path:
    p = ai_root() / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def runtime_root() -> Path:
    p = ai_root() / "engine"
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_dir() -> Path:
    p = ai_root() / "cache"
    p.mkdir(parents=True, exist_ok=True)
    return p
