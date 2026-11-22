"""Font discovery, identification and Unicode fallback.

Fonts come from three places: the fonts bundled with Glitch Lab (DejaVu,
Bitstream Vera licence; STIX General, SIL Open Font License -- both allow
redistribution), the fonts installed on the computer, and TTF/OTF files the
user imports (copied into the user's data folder; they are referenced, never
embedded in projects, to respect font licences).

A font is identified by ``"Family|Style"`` (e.g. ``"DejaVu Sans|Bold"``).
Characters a font cannot draw fall back to the bundled DejaVu Sans, which
covers Latin Extended (č ć š ž đ ...), Greek and Cyrillic.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sys
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..settings import data_dir

log = logging.getLogger(__name__)

BUNDLED_DIR = Path(__file__).resolve().parents[2] / "resources" / "fonts"
FALLBACK_ID = "DejaVu Sans|Book"
FONT_EXTS = (".ttf", ".otf", ".ttc")
_lock = threading.Lock()


@dataclass(frozen=True)
class FontInfo:
    id: str
    family: str
    style: str
    path: str
    origin: str          # "bundled" | "system" | "imported"

    @property
    def label(self) -> str:
        return f"{self.family} {self.style}".strip()


def system_font_dirs() -> list[Path]:
    home = Path.home()
    if sys.platform == "win32":
        windir = Path(os.environ.get("WINDIR", r"C:\Windows"))
        dirs = [windir / "Fonts", Path(os.environ.get("LOCALAPPDATA", home / "AppData" / "Local")) / "Microsoft" /
                "Windows" / "Fonts"]
    elif sys.platform == "darwin":
        dirs = [Path("/System/Library/Fonts"), Path("/Library/Fonts"), home / "Library" / "Fonts"]
    else:
        dirs = [Path("/usr/share/fonts"), Path("/usr/local/share/fonts"), home / ".fonts", home / ".local" / "share" /
                "fonts"]
    return [d for d in dirs if d.is_dir()]


def imported_dir() -> Path:
    d = data_dir() / "fonts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _describe(path: Path) -> tuple[str, str] | None:
    try:
        f = ImageFont.truetype(str(path), 12)
        fam, sty = f.getname()
        return (fam or path.stem), (sty or "Regular")
    except Exception:  # unreadable or unsupported file
        return None


def _scan(dirs: list[Path], origin: str, cache: dict) -> list[FontInfo]:
    out = []
    for d in dirs:
        for root, _sub, files in os.walk(d):
            for fn in files:
                if not fn.lower().endswith(FONT_EXTS):
                    continue
                p = Path(root) / fn
                try:
                    st = p.stat()
                except OSError:
                    continue
                key = f"{p}|{st.st_mtime_ns}|{st.st_size}"
                desc = cache.get(key)
                if desc is None:
                    desc = _describe(p)
                    cache[key] = desc or ["", ""]
                if desc and desc[0]:
                    out.append(FontInfo(f"{desc[0]}|{desc[1]}", desc[0], desc[1], str(p), origin))
    return out


class FontRegistry:
    def __init__(self):
        self._fonts: dict[str, FontInfo] | None = None

    def _load(self) -> dict[str, FontInfo]:
        with _lock:
            if self._fonts is not None:
                return self._fonts
            cache_path = data_dir() / "font-index.json"
            try:
                cache = json.loads(cache_path.read_text())
            except (OSError, ValueError):
                cache = {}
            fonts: dict[str, FontInfo] = {}
            # later sources do not replace earlier ones: bundled < imported < system order of preference
            for info in (_scan([BUNDLED_DIR], "bundled", cache) + _scan([imported_dir()], "imported", cache)
                         + _scan(system_font_dirs(), "system", cache)):
                fonts.setdefault(info.id, info)
            try:
                cache_path.write_text(json.dumps(cache))
            except OSError:
                pass
            self._fonts = fonts
            return fonts

    def refresh(self) -> None:
        with _lock:
            self._fonts = None

    def all(self) -> list[FontInfo]:
        return sorted(self._load().values(), key=lambda f: (f.family.lower(), f.style.lower()))

    def families(self) -> list[str]:
        return sorted({f.family for f in self._load().values()}, key=str.lower)

    def styles(self, family: str) -> list[str]:
        return sorted(f.style for f in self._load().values() if f.family == family)

    def get(self, font_id: str) -> FontInfo | None:
        return self._load().get(font_id)

    def bundled(self) -> list[FontInfo]:
        return [f for f in self.all() if f.origin == "bundled"]

    def import_font(self, path: str) -> FontInfo:
        """Copy a TTF/OTF into the user's font folder and register it."""
        src = Path(path)
        if src.suffix.lower() not in FONT_EXTS:
            raise ValueError("Only TrueType/OpenType font files (.ttf, .otf, .ttc) can be imported.")
        desc = _describe(src)
        if desc is None:
            raise ValueError(f"{src.name} is not a readable font file.")
        dest = imported_dir() / src.name
        if not dest.exists() or dest.read_bytes() != src.read_bytes():
            shutil.copyfile(src, dest)
        self.refresh()
        info = self.get(f"{desc[0]}|{desc[1]}")
        assert info is not None
        return info


REGISTRY = FontRegistry()


def resolve(font_id: str) -> tuple[FontInfo, bool]:
    """(font, found). Missing fonts resolve to the bundled fallback."""
    info = REGISTRY.get(font_id)
    if info is not None:
        return info, True
    fam = (font_id or "").split("|")[0]
    same = [f for f in REGISTRY.all() if f.family == fam]
    if same:  # family exists, style does not: closest available style
        return same[0], False
    return REGISTRY.get(FALLBACK_ID) or REGISTRY.bundled()[0], False


def missing_fonts(font_ids) -> list[str]:
    return sorted({fid for fid in font_ids if fid and REGISTRY.get(fid) is None})


@lru_cache(maxsize=256)
def load(path: str, size_px: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, max(1, int(size_px)))


def file_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------------ coverage
_MISSING_PROBE = "\U0010fffd"


@lru_cache(maxsize=65536)
def has_glyph(path: str, ch: str) -> bool:
    """True when the font has a real glyph for ``ch`` (not its 'missing' box)."""
    if ch.isspace() or ord(ch) < 32:
        return True
    font = load(path, 48)
    try:
        a = np.asarray(font.getmask(ch))
        b = np.asarray(font.getmask(_MISSING_PROBE))
    except Exception:
        return False
    if a.size == 0 or not a.any():
        return False
    return not (a.shape == b.shape and np.array_equal(a, b))


def fallback_path() -> str:
    info = REGISTRY.get(FALLBACK_ID) or REGISTRY.bundled()[0]
    return info.path


def font_for_char(path: str, ch: str) -> str:
    """The font file that will draw ``ch``: the requested font or the fallback."""
    if has_glyph(path, ch):
        return path
    fb = fallback_path()
    if fb != path and has_glyph(fb, ch):
        return fb
    for info in REGISTRY.bundled():
        if has_glyph(info.path, ch):
            return info.path
    return path


def render_probe(path: str, text: str, size: int = 40) -> np.ndarray:
    """Rendering of ``text`` (for tests and previews)."""
    font = load(path, size)
    im = Image.new("L", (size * (len(text) + 2), size * 2), 0)
    ImageDraw.Draw(im).text((size // 2, size // 2), text, font=font, fill=255)
    return np.asarray(im)
