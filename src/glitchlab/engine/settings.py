"""Per-user configuration and log locations, plus a small JSON settings store."""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def config_dir() -> Path:
    env = os.environ.get("GLITCHLAB_CONFIG_DIR")
    if env:
        p = Path(env)
    elif sys.platform == "win32":
        p = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "GlitchLab"
    elif sys.platform == "darwin":
        p = Path.home() / "Library" / "Application Support" / "GlitchLab"
    else:
        p = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "glitchlab"
    p.mkdir(parents=True, exist_ok=True)
    return p


def data_dir() -> Path:
    env = os.environ.get("GLITCHLAB_CONFIG_DIR")
    if env:
        p = Path(env) / "data"
    elif sys.platform == "win32":
        p = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "GlitchLab"
    else:
        p = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "glitchlab"
    p.mkdir(parents=True, exist_ok=True)
    return p


def log_dir() -> Path:
    p = data_dir() / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def setup_logging(level: int = logging.INFO, *, to_file: bool = True) -> Path | None:
    root = logging.getLogger()
    if getattr(root, "_glitchlab_configured", False):
        return None
    root.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    path = None
    if to_file:
        try:
            path = log_dir() / "glitchlab.log"
            fh = logging.handlers.RotatingFileHandler(path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
            fh.setFormatter(fmt)
            fh.setLevel(logging.DEBUG)
            root.addHandler(fh)
        except OSError:
            path = None
    if sys.stderr is not None:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        sh.setLevel(level)
        root.addHandler(sh)
    root._glitchlab_configured = True  # type: ignore[attr-defined]
    return path


class Settings:
    """Tiny JSON-backed settings store (thread-unsafe; UI thread only)."""

    DEFAULTS = {
        "ffmpeg_path": "",
        "preview_max_side": 1024,
        "max_import_megapixels": 64,
        "cache_megabytes": 768,
        "recent_files": [],
        "recent_projects": [],
        "browser_folder": "",
        "export_folder": "",
        "export_pattern": "{name}_glitch_{seed}",
        "show_checkerboard": True,
        "window_geometry": "",
        "window_state_v2": "",
        "center_split": "",
    }

    def __init__(self, path: Path | None = None):
        self.path = path or (config_dir() / "settings.json")
        self.data = dict(self.DEFAULTS)
        try:
            if self.path.exists():
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self.data.update({k: v for k, v in loaded.items() if k in self.DEFAULTS})
        except Exception as exc:
            log.warning("Settings unreadable, using defaults: %s", exc)

    def get(self, key):
        return self.data.get(key, self.DEFAULTS.get(key))

    def set(self, key, value) -> None:
        self.data[key] = value
        self.save()

    def add_recent(self, key: str, path: str, limit: int = 10) -> None:
        items = [p for p in self.data.get(key, []) if p != path]
        items.insert(0, path)
        self.data[key] = items[:limit]
        self.save()

    def save(self) -> None:
        try:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as exc:
            log.warning("Could not save settings: %s", exc)
