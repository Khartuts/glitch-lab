"""Contact sheet: every effect (default parameters) and every built-in preset
applied to generated fixtures, with per-effect status labels."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .. import __version__
from .effects.base import REGISTRY
from .errors import DependencyError, GlitchLabError
from .fixtures import checkerboard, color_bars, gradient, photo_like, transparent_circle
from .image_io import ExportOptions, save_image
from .model import Document, EffectInstance
from .pipeline import render_full
from .presets import BUILTIN_PRESETS, apply_preset
from .services import EngineServices, default_services

TILE_W, TILE_H = 208, 156
LABEL_W = 250
PAD = 6
BG = (20, 21, 25)
FG = (228, 229, 234)
MUTED = (141, 144, 160)
OK = (94, 224, 138)
WARN = (255, 179, 71)
ERR = (255, 92, 92)


@dataclass
class SheetResult:
    key: str
    name: str
    status: str      # ok | failed | skipped | unchanged
    message: str
    seconds: float


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # pragma: no cover - very old Pillow
        return ImageFont.load_default()


def _on_checker(rgba: np.ndarray) -> np.ndarray:
    h, w = rgba.shape[:2]
    bg = checkerboard(w, h, cell=8)[..., :3].astype(np.float32) * 0.5 + 60
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    return np.clip(rgba[..., :3] * a + bg * (1 - a), 0, 255).astype(np.uint8)


def fixtures_for_sheet() -> list[tuple[str, np.ndarray]]:
    return [("photo", photo_like(TILE_W, TILE_H)), ("colour bars", color_bars(TILE_W, TILE_H)),
            ("gradient", gradient(TILE_W, TILE_H)), ("transparent", transparent_circle(TILE_W, TILE_H)),
            ("checkerboard", checkerboard(TILE_W, TILE_H, 13))]


def build_contact_sheet(services: EngineServices | None = None, progress=None) -> tuple[np.ndarray, list[SheetResult]]:
    services = services or default_services()
    fx = fixtures_for_sheet()
    effects = sorted(REGISTRY.values(), key=lambda e: (e.category, e.name))
    presets = [p for p in BUILTIN_PRESETS if all(f["effect"] in REGISTRY for f in p["stack"])]
    rows: list[tuple[str, str, Document | None]] = [("__orig__", "Original fixtures", None)]
    for e in effects:
        rows.append((e.id, e.name, Document.single([EffectInstance.create(e.id, seed=1234)], 77)))
    for p in presets:
        rows.append(("preset:" + p["name"], "Preset: " + p["name"], apply_preset(Document.single([], 77), p)))
    ncols = len(fx)
    width = LABEL_W + ncols * (TILE_W + PAD) + PAD
    header = 86
    height = header + len(rows) * (TILE_H + PAD) + PAD + 30
    sheet = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(sheet)
    f_title, f_name, f_small = _font(22), _font(15), _font(12)
    draw.text((PAD * 2, 12), f"GLITCH LAB {__version__} - CONTACT SHEET", fill=(255, 61, 127), font=f_title)
    draw.text((PAD * 2, 40), "Default parameters, fixed seeds, generated fixtures (transparent fixture shown over a checkerboard).",
              fill=MUTED, font=f_small)
    for c, (fname, _) in enumerate(fx):
        draw.text((LABEL_W + c * (TILE_W + PAD) + PAD, header - 18), fname.upper(), fill=MUTED, font=f_small)
    results: list[SheetResult] = []
    for r, (key, name, doc) in enumerate(rows):
        if progress:
            progress(r / len(rows), name)
        y = header + r * (TILE_H + PAD) + PAD
        status, message, t0 = "ok", "", time.perf_counter()
        tiles = []
        for _fname, img in fx:
            if doc is None:
                tiles.append(img)
                continue
            try:
                out = render_full(img, doc, services=services).image
                tiles.append(out)
            except DependencyError as exc:
                status, message = "skipped", exc.message
                tiles.append(None)
            except GlitchLabError as exc:
                status, message = "failed", exc.message
                tiles.append(None)
            except Exception as exc:  # pragma: no cover - reported on the sheet
                status, message = "failed", repr(exc)
                tiles.append(None)
        if doc is not None and status == "ok" and all(t is not None and np.array_equal(t, img)
                                                       for t, (_, img) in zip(tiles, fx)):
            status, message = "unchanged", "default parameters left every fixture unchanged"
        secs = time.perf_counter() - t0
        if doc is not None:
            results.append(SheetResult(key, name, status, message, secs))
        color = {"ok": OK, "skipped": WARN, "unchanged": WARN}.get(status, ERR)
        draw.text((PAD * 2, y + 6), name, fill=FG, font=f_name)
        if doc is not None:
            label = {"ok": "OK", "failed": "FAILED", "skipped": "SKIPPED", "unchanged": "NO CHANGE"}[status]
            draw.text((PAD * 2, y + 28), f"{label}  ·  {secs:.2f} s", fill=color, font=f_small)
            if message:
                words, line, ly = message.split(), "", y + 46
                for wd in words:
                    if len(line) + len(wd) > 34:
                        draw.text((PAD * 2, ly), line, fill=color, font=f_small)
                        ly += 15
                        line = ""
                        if ly > y + TILE_H - 14:
                            break
                    line += wd + " "
                if ly <= y + TILE_H - 14:
                    draw.text((PAD * 2, ly), line, fill=color, font=f_small)
        for c, t in enumerate(tiles):
            x = LABEL_W + c * (TILE_W + PAD) + PAD
            if t is None:
                draw.rectangle([x, y, x + TILE_W - 1, y + TILE_H - 1], outline=color, width=2)
                draw.text((x + 10, y + TILE_H // 2 - 8), label, fill=color, font=f_name)
                continue
            sheet.paste(Image.fromarray(_on_checker(t)), (x, y))
    n_ok = sum(1 for r in results if r.status == "ok")
    ff_ok, ff_text = services.ffmpeg_status()
    draw.text((PAD * 2, height - 24), f"{n_ok}/{len(results)} rows OK. FFmpeg: {ff_text if ff_ok else 'not available - ' + ff_text}",
              fill=MUTED, font=f_small)
    if progress:
        progress(1.0, "done")
    return np.array(sheet.convert("RGBA")), results


def write_contact_sheet(path: str, services: EngineServices | None = None, overwrite: bool = False,
                        progress=None) -> tuple[str, list[SheetResult]]:
    img, results = build_contact_sheet(services, progress)
    if not os.path.splitext(path)[1]:
        path += ".png"
    ext = os.path.splitext(path)[1].lower()
    fmt = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG", ".webp": "WEBP"}.get(ext, "PNG")
    save_image(img, path, ExportOptions(format=fmt, quality=90, overwrite=overwrite))
    return path, results
