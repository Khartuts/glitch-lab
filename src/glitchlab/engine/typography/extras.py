"""Typography helpers: presets on disk, document fonts, typography-only
export and the style contact sheet."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..settings import config_dir
from . import fonts
from .model import STYLE_IDS, STYLE_NAMES, apply_typography_preset, new_box, preset_from_box, style_preset

#: Short original sample lines (written for Glitch Lab; not quoted from any song).
SAMPLE_TEXT = "static in the bloodstream\nwe glow louder — čćšžđ"


# ------------------------------------------------------------------ presets on disk
def preset_dir() -> Path:
    d = config_dir() / "typography-presets"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe(name: str) -> str:
    return (re.sub(r"[^A-Za-z0-9 _-]+", "", name).strip() or "preset")[:60]


def save_typography_preset(box: dict, name: str, folder: Path | None = None) -> Path:
    folder = Path(folder or preset_dir())
    p = folder / f"{_safe(name)}.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(preset_from_box(box, name), indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)
    return p


def load_typography_presets(folder: Path | None = None) -> list[dict]:
    out = []
    for p in sorted(Path(folder or preset_dir()).glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            if d.get("kind") == "typography" and isinstance(d.get("box"), dict):
                out.append(d)
        except (OSError, ValueError):
            continue
    return out


def apply_preset(box: dict, preset: dict) -> dict:
    return apply_typography_preset(box, preset)


# ------------------------------------------------------------------ documents
def document_fonts(doc) -> set[str]:
    return {b.get("font", "") for layer in doc.layers if layer.kind == "text" for b in layer.boxes}


def missing_document_fonts(doc) -> list[str]:
    return fonts.missing_fonts(document_fonts(doc))


def substitute_font(doc, old: str, new: str) -> int:
    """Replace a (missing) font everywhere in the document. Returns the number of boxes changed."""
    n = 0
    for layer in doc.layers:
        for b in layer.boxes:
            if b.get("font") == old:
                b["font"] = new
                n += 1
    return n


def render_typography(source: np.ndarray, doc, **kw):
    """Only the text layers over transparency (for a transparent PNG)."""
    from ..pipeline import render_full  # noqa: PLC0415

    return render_full(source, doc, text_only=True, **kw)


def export_typography_png(source: np.ndarray, doc, path: str, *, overwrite: bool = False) -> str:
    from ..errors import ExportError  # noqa: PLC0415
    from ..image_io import save_png_bytes  # noqa: PLC0415

    if os.path.exists(path) and not overwrite:
        raise ExportError(f"{os.path.basename(path)} already exists.")
    res = render_typography(source, doc)
    with open(path, "wb") as f:
        f.write(save_png_bytes(res.image))
    return path


# ------------------------------------------------------------------ contact sheet
def type_contact_sheet(path: str, photo: np.ndarray | None = None, text: str = SAMPLE_TEXT,
                       tile_w: int = 640) -> str:
    """All eight styles on the same photograph (plus the clean reference)."""
    from .. import fixtures  # noqa: PLC0415
    from ..image_io import save_png_bytes  # noqa: PLC0415
    from ..model import Document, Layer  # noqa: PLC0415
    from ..pipeline import render_full  # noqa: PLC0415

    if photo is None:
        photo = fixtures.photo_like(tile_w, int(tile_w * 0.75))
    h, w = photo.shape[:2]
    styles = [s for s in STYLE_IDS if s != "clean"]
    tiles = []
    for st in ["clean", *styles]:
        box = style_preset(st, new_box(text, x=0.5, y=0.5, w=0.9, h=0.62, size=0.1, align="center", valign="middle",
                                       fit=True, seed=11))
        doc = Document.new(1)
        doc.layers.append(Layer(name="Text", kind="text", boxes=[box]))
        img = render_full(photo, doc).image
        lab = Image.fromarray(img[..., :3].copy())
        d = ImageDraw.Draw(lab)
        font = fonts.load(fonts.fallback_path(), max(12, w // 32))
        d.rectangle([0, h - w // 20 - 10, w, h], fill=(0, 0, 0))
        d.text((10, h - w // 20 - 6), STYLE_NAMES[st], font=font, fill=(255, 255, 255))
        tiles.append(np.asarray(lab))
    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * h, cols * w, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = t
    rgba = np.dstack([sheet, np.full(sheet.shape[:2], 255, np.uint8)])
    with open(path, "wb") as f:
        f.write(save_png_bytes(rgba))
    return path
