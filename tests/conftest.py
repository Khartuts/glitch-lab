import os
import sys
import tempfile

import numpy as np
import pytest

# Isolate user configuration for the whole test session.
_CFG = tempfile.mkdtemp(prefix="glitchlab-test-config-")
os.environ["GLITCHLAB_CONFIG_DIR"] = _CFG
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from glitchlab.engine import effects  # noqa: E402,F401  (registers effects)
from glitchlab.engine import fixtures  # noqa: E402
from glitchlab.engine.services import EngineServices  # noqa: E402


@pytest.fixture
def photo():
    return fixtures.photo_like(160, 120)


@pytest.fixture
def transparent():
    return fixtures.transparent_circle(128, 96)


@pytest.fixture(scope="session")
def services():
    return EngineServices()


def ffmpeg_available() -> bool:
    ok, _ = EngineServices().ffmpeg_status()
    return ok


requires_ffmpeg = pytest.mark.skipif(not ffmpeg_available(), reason="FFmpeg not available")
