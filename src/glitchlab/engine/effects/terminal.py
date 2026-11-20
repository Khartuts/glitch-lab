"""Hacker-terminal looks: ASCII art, digital code rain and hex-dump mosaics.

Glyphs are rendered once per (character set, cell size) into an atlas with
Pillow's bundled font (identical on every platform) and assembled with NumPy
indexing, so whole images of text render in milliseconds.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..ops import luminance, to_float, to_uint8
from ..params import TEXT, Param, bparam, color_to_rgb, cparam, fparam, iparam
from .base import Effect, register

CHARSETS = {
    "ramp": " .:-=+*#%@",
    "dense": " .'`^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$",
    "binary": " 01",
    "hex": " 0123456789ABCDEF",
    "code": " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ$#%&*+=<>:;{}[]|/\\",
}
PRINTABLE = "".join(chr(c) for c in range(32, 127))
PHOSPHOR = {"green": "#33ff66", "amber": "#ffb000", "cyan": "#2fe0ff", "white": "#e8eef2", "red": "#ff3344"}

_ATLAS: OrderedDict = OrderedDict()
_ATLAS_LOCK = threading.Lock()


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # pragma: no cover - Pillow without FreeType
        return ImageFont.load_default()


def glyph_atlas(chars: str, cw: int, ch: int) -> np.ndarray:
    """(len(chars), ch, cw) float32 glyph coverage tiles (cached)."""
    key = (chars, cw, ch)
    with _ATLAS_LOCK:
        if key in _ATLAS:
            return _ATLAS[key]
    font = _font(max(5, int(round(ch * 0.92))))
    tiles = np.zeros((len(chars), ch, cw), np.float32)
    for i, c in enumerate(chars):
        if c == " ":
            continue
        im = Image.new("L", (cw * 3, ch * 3), 0)
        ImageDraw.Draw(im).text((cw * 1.5, ch * 1.5), c, fill=255, font=font, anchor="mm")
        a = np.asarray(im, np.float32)[ch:2 * ch, cw:2 * cw] / 255.0
        tiles[i] = a
    with _ATLAS_LOCK:
        _ATLAS[key] = tiles
        while len(_ATLAS) > 16:
            _ATLAS.popitem(last=False)
    return tiles


def compose_glyphs(atlas: np.ndarray, idx: np.ndarray, h: int, w: int) -> np.ndarray:
    """Assemble glyph tiles for an index grid (gh, gw) into an (h, w) coverage map."""
    gh, gw = idx.shape
    n, ch, cw = atlas.shape
    tiles = atlas[np.clip(idx, 0, n - 1)]                     # (gh, gw, ch, cw)
    big = tiles.transpose(0, 2, 1, 3).reshape(gh * ch, gw * cw)
    return big[:h, :w]


def expand_cells(grid: np.ndarray, ch: int, cw: int, h: int, w: int) -> np.ndarray:
    """Repeat a per-cell array (gh, gw[, c]) to pixel resolution (h, w[, c])."""
    return np.repeat(np.repeat(grid, ch, axis=0), cw, axis=1)[:h, :w]


def hexdump_lines(data: bytes, width_chars: int, rows: int) -> list[str]:
    per = max(1, (width_chars - 12) // 4)
    lines = []
    off = 0
    n = len(data)
    for r in range(rows):
        chunk = bytes(data[(off + i) % n] for i in range(per)) if n else b""
        hexs = " ".join(f"{b:02x}" for b in chunk)
        asc = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        line = f"{off:08x}  {hexs}  |{asc}|"
        lines.append((line + " " * width_chars)[:width_chars])
        off += per
    return lines


@register
class HackerTerminal(Effect):
    id = "hacker_terminal"
    name = "Hacker Terminal"
    category = "Abstract & Style"
    description = ("Hacker aesthetic: the photo rebuilt from glowing terminal characters - ASCII art, falling "
                   "digital code rain, or a hex dump of your text (or of the image's own bytes) whose brightness "
                   "draws the picture. Green, amber, cyan phosphor or the photo's own colours.")
    supports_channels = False
    preview_note = "Characters are scaled with the preview; random glyph choices differ from the export."
    params = (
        cparam("mode", "Mode", "ascii", (("ascii", "ASCII art"), ("matrix", "Code rain"), ("hexdump", "Hex dump mosaic"))),
        iparam("cell", "Character size", 12, 4, 160, spatial=True, unit="px"),
        cparam("charset", "Characters", "ramp", (("ramp", "Classic ramp  .:-=+*#%@"), ("dense", "Dense ramp"),
                                                 ("binary", "Binary 0 1"), ("hex", "Hex 0-F"), ("code", "Code symbols")),
               show_if=("mode", ("ascii", "matrix"))),
        Param("text", "Text", TEXT, "", show_if=("mode", ("hexdump",)), randomizable=False,
              tooltip="Bytes to dump. Empty = the image's own pixel bytes."),
        cparam("colour", "Colour", "green", (("green", "Phosphor green"), ("amber", "Amber"), ("cyan", "Cyan"),
                                             ("white", "White"), ("red", "Red alert"), ("original", "Photo colours"))),
        fparam("contrast", "Contrast", 1.3, 0.3, 4.0),
        bparam("invert", "Invert brightness", False),
        fparam("background", "Photo behind text", 0.12, 0.0, 1.0),
        fparam("density", "Rain density", 0.75, 0.05, 1.0, show_if=("mode", ("matrix",))),
        iparam("trail", "Rain trail length", 16, 2, 120, show_if=("mode", ("matrix",)), tooltip="In characters."),
        fparam("glow", "Glow", 0.45, 0.0, 1.5),
    )

    def process(self, img, p, ctx):
        h, w = img.shape[:2]
        ch = max(4, int(p["cell"]))
        cw = max(3, int(round(ch * 0.62)))
        gh, gw = -(-h // ch), -(-w // cw)
        lum = luminance(img)
        cell_lum = cv2.resize(lum, (gw, gh), interpolation=cv2.INTER_AREA)
        cell_lum = np.clip(0.5 + (cell_lum - 0.5) * p["contrast"], 0.0, 1.0)
        if p["invert"]:
            cell_lum = 1.0 - cell_lum
        mode = p["mode"]
        rng = ctx.rng
        if mode == "ascii":
            chars = CHARSETS[p["charset"]]
            idx = np.clip(np.rint(cell_lum * (len(chars) - 1)), 0, len(chars) - 1).astype(np.int64)
            cov = compose_glyphs(glyph_atlas(chars, cw, ch), idx, h, w)
            bright = expand_cells(0.35 + 0.65 * cell_lum, ch, cw, h, w)
            layer = cov * bright
        elif mode == "matrix":
            chars = CHARSETS[p["charset"]].strip() or "01"
            idx = rng.integers(0, len(chars), size=(gh, gw))
            active = rng.random(gw) < p["density"]
            trail = float(p["trail"])
            heads = rng.integers(0, gh + int(trail), size=gw).astype(np.float32)
            lens = (trail * rng.uniform(0.6, 1.4, size=gw)).astype(np.float32)
            rows = np.arange(gh, dtype=np.float32)[:, None]
            dist = heads[None, :] - rows
            inten = np.where((dist >= 0) & (dist < lens[None, :]), 1.0 - dist / lens[None, :], 0.0)
            inten = inten * active[None, :]
            head_cell = (dist >= 0) & (dist < 1) & active[None, :]
            cell_b = inten * (0.15 + 0.85 * cell_lum) + head_cell * 0.6
            cov = compose_glyphs(glyph_atlas(chars, cw, ch), idx, h, w)
            layer = cov * expand_cells(np.clip(cell_b, 0, 1.4), ch, cw, h, w)
            # the faint image itself as dim static characters where there is no rain
            static = cov * expand_cells(cell_lum * 0.3, ch, cw, h, w)
            layer = np.maximum(layer, static)
        else:  # hexdump
            data = p["text"].encode("utf-8") if p["text"] else np.ascontiguousarray(img[..., :3]).tobytes()[: 65536]
            lines = hexdump_lines(data, gw, gh)
            table = np.frombuffer("".join(lines).encode("ascii", "replace"), np.uint8).reshape(gh, gw)
            idx = np.clip(table.astype(np.int64) - 32, 0, len(PRINTABLE) - 1)
            cov = compose_glyphs(glyph_atlas(PRINTABLE, cw, ch), idx, h, w)
            layer = cov * expand_cells(0.15 + 0.85 * cell_lum, ch, cw, h, w)
        layer = layer.astype(np.float32)
        if p["colour"] == "original":
            cell_col = cv2.resize(to_float(np.ascontiguousarray(img[..., :3])), (gw, gh), interpolation=cv2.INTER_AREA)
            colour = expand_cells(np.clip(cell_col * 1.3 + 0.1, 0, 1), ch, cw, h, w)
        else:
            colour = np.array(color_to_rgb(PHOSPHOR[p["colour"]]), np.float32)[None, None, :] / 255.0
        text_rgb = layer[..., None] * colour
        if p["glow"] > 0:
            glow = cv2.GaussianBlur(text_rgb, (0, 0), max(0.8, ch * 0.35))
            text_rgb = text_rgb + glow * np.float32(p["glow"])
        base = to_float(img[..., :3])
        if p["colour"] != "original":
            base = luminance(img)[..., None] * colour
        out_rgb = base * np.float32(p["background"]) * 0.6 + text_rgb
        out = img.copy()
        out[..., :3] = to_uint8(out_rgb)
        return out
