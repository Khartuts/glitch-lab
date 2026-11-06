"""Colour and bit-level effects: bit-plane surgery, palette damage, error-diffusion trails."""

from __future__ import annotations

import os
import re

import numpy as np
from PIL import Image

from ..errors import EffectError
from ..ops import luminance, to_uint8
from ..params import BITS, FILE, Param, bparam, cparam, fparam, iparam
from .base import Effect, register


# ---------------------------------------------------------------------------
# Bit-plane surgery
# ---------------------------------------------------------------------------

@register
class BitPlane(Effect):
    id = "bitplane"
    name = "Bit-Plane Surgery"
    category = "Colour & Bits"
    description = ("Drops, sets, inverts, isolates, swaps, rotates or shuffles selected bit planes of the "
                   "8-bit channel values (optionally reduced to a lower bit depth first).")
    uses_seed = True
    params = (
        cparam("operation", "Operation", "invert",
               (("drop", "Drop (clear to 0)"), ("set", "Set to 1"), ("invert", "Invert (XOR)"),
                ("keep", "Keep only selected"), ("swap", "Swap two planes"), ("rotate", "Rotate bits"),
                ("shuffle", "Shuffle planes (seeded)"))),
        Param("planes", "Bit planes", BITS, 0b00110000, nbits=8,
              tooltip="Selected planes (7 = most significant). Used by drop/set/invert/keep.",
              show_if=("operation", ("drop", "set", "invert", "keep"))),
        iparam("depth", "Bit depth", 8, 1, 8, tooltip="Values are first reduced to this many bits."),
        iparam("swap_a", "Swap plane A", 7, 0, 7, show_if=("operation", ("swap",))),
        iparam("swap_b", "Swap plane B", 0, 0, 7, show_if=("operation", ("swap",))),
        iparam("rotate_by", "Rotate by", 3, -7, 7, show_if=("operation", ("rotate",))),
    )

    def process(self, img, p, ctx):
        depth = int(p["depth"])
        full = (1 << depth) - 1
        v = img[..., :3].astype(np.uint16)
        q = v >> (8 - depth)
        op = p["operation"]
        m = int(p["planes"]) & full
        if op == "drop":
            q = q & (~m & full)
        elif op == "set":
            q = q | m
        elif op == "invert":
            q = q ^ m
        elif op == "keep":
            q = q & m
        elif op == "swap":
            a, b = min(p["swap_a"], depth - 1), min(p["swap_b"], depth - 1)
            ba, bb = (q >> a) & 1, (q >> b) & 1
            keep = full & ~((1 << a) | (1 << b))  # non-negative Python int mask
            q = (q & keep) | (ba << b) | (bb << a)
        elif op == "rotate":
            k = int(p["rotate_by"]) % depth
            q = ((q << k) | (q >> (depth - k))) & full if k else q
        elif op == "shuffle":
            perm = ctx.rng.permutation(depth)
            newq = np.zeros_like(q)
            for src_bit, dst_bit in enumerate(perm):
                newq |= ((q >> src_bit) & 1) << int(dst_bit)
            q = newq
        out = img.copy()
        if depth == 8:
            out[..., :3] = q.astype(np.uint8)
        else:
            out[..., :3] = np.rint(q.astype(np.float32) * (255.0 / full)).astype(np.uint8)
        return out


# ---------------------------------------------------------------------------
# Palette damage
# ---------------------------------------------------------------------------

GRADIENTS = {
    "thermal": ["#000000", "#2b0f6b", "#b3206b", "#ff7a00", "#ffe600", "#ffffff"],
    "acid": ["#0b0221", "#00ff9c", "#ff00e1", "#fff200", "#00b3ff"],
    "mono": ["#000000", "#ffffff"],
    "sunset": ["#140021", "#5d0d6e", "#e2316b", "#ff9a3c", "#ffe8a8"],
    "ice": ["#00030f", "#08306b", "#2b8cbe", "#a6e3ff", "#ffffff"],
    "toxic": ["#001a00", "#1f7a00", "#c6ff00", "#000000", "#ff2bd6"],
}

FIXED_PALETTES = {
    "gameboy": ["#0f380f", "#306230", "#8bac0f", "#9bbc0f"],
    "cga": ["#000000", "#55ffff", "#ff55ff", "#ffffff"],
    "vapor": ["#ff71ce", "#01cdfe", "#05ffa1", "#b967ff", "#fffb96", "#1a1a2e"],
    "zx": ["#000000", "#0000d7", "#d70000", "#d700d7", "#00d700", "#00d7d7", "#d7d700", "#d7d7d7",
           "#0000ff", "#ff0000", "#ff00ff", "#00ff00", "#00ffff", "#ffff00", "#ffffff"],
    "rgbcmy": ["#000000", "#ff0000", "#00ff00", "#0000ff", "#00ffff", "#ff00ff", "#ffff00", "#ffffff"],
}

_BAYER2 = np.array([[0, 2], [3, 1]], np.float32)


def bayer_matrix(n: int) -> np.ndarray:
    """Normalised ordered-dither threshold matrix in (0, 1)."""
    m = _BAYER2
    while m.shape[0] < n:
        m = np.block([[4 * m, 4 * m + 2], [4 * m + 3, 4 * m + 1]])
    return (m + 0.5) / m.size


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def gradient_lut(stops: list[str]) -> np.ndarray:
    cols = np.array([_hex_to_rgb(s) for s in stops], np.float32)
    x = np.linspace(0, 1, len(cols))
    t = np.linspace(0, 1, 256)
    return np.stack([np.interp(t, x, cols[:, c]) for c in range(3)], -1).astype(np.float32)


def load_palette_file(path: str, max_colors: int = 256) -> np.ndarray:
    """Read a palette: GIMP .gpl, Adobe .act, .hex/.txt (one hex colour per line) or any image."""
    if not path or not os.path.exists(path):
        raise EffectError(f"Palette file not found: {path or '(none)'}")
    ext = os.path.splitext(path)[1].lower()
    colors: list[tuple[int, int, int]] = []
    try:
        if ext == ".gpl":
            for line in open(path, encoding="utf-8", errors="replace"):
                m = re.match(r"^\s*(\d{1,3})\s+(\d{1,3})\s+(\d{1,3})", line)
                if m:
                    colors.append(tuple(min(255, int(g)) for g in m.groups()))
        elif ext == ".act":
            data = open(path, "rb").read()
            n = 256
            if len(data) >= 772:
                n = int.from_bytes(data[768:770], "big") or 256
            for i in range(min(n, len(data) // 3, 256)):
                colors.append(tuple(data[i * 3:i * 3 + 3]))
        elif ext in (".hex", ".txt", ".pal", ".csv"):
            for m in re.finditer(r"#?\b([0-9A-Fa-f]{6})\b", open(path, encoding="utf-8", errors="replace").read()):
                colors.append(_hex_to_rgb(m.group(1)))
        else:
            with Image.open(path) as im:
                im = im.convert("RGB")
                if im.width * im.height > 1_000_000:
                    im.thumbnail((1000, 1000))
                q = im.quantize(colors=max_colors, method=Image.Quantize.MEDIANCUT)
                pal = q.getpalette()[: 3 * len(q.getcolors(max_colors) or [])]
                colors = [tuple(pal[i:i + 3]) for i in range(0, len(pal), 3)]
    except EffectError:
        raise
    except Exception as exc:
        raise EffectError(f"Could not read palette '{os.path.basename(path)}': {exc}") from exc
    colors = list(dict.fromkeys(colors))[:max_colors]
    if len(colors) < 2:
        raise EffectError(f"Palette '{os.path.basename(path)}' has fewer than two colours.")
    return np.array(colors, np.uint8)


def quantize_to_palette(rgb: np.ndarray, palette: np.ndarray, dither: str, spread: float) -> np.ndarray:
    """Map RGB to the nearest palette colours (Pillow), with optional ordered or FS dithering."""
    work = rgb
    if dither.startswith("ordered"):
        n = int(dither[len("ordered"):])
        bm = bayer_matrix(n)
        h, w = rgb.shape[:2]
        tile = np.tile(bm, (h // n + 1, w // n + 1))[:h, :w]
        step = 255.0 / max(1, len(palette) ** (1 / 3))
        work = to_uint8(rgb.astype(np.float32) + ((tile - 0.5) * spread * step)[..., None], scale=1.0)
    pal_img = Image.new("P", (1, 1))
    flat = palette.reshape(-1).tolist()
    flat += flat[-3:] * (256 - len(palette))
    pal_img.putpalette(flat)
    im = Image.fromarray(np.ascontiguousarray(work), "RGB")
    dmode = Image.Dither.FLOYDSTEINBERG if dither == "floyd" else Image.Dither.NONE
    q = im.quantize(palette=pal_img, dither=dmode)
    return np.array(q.convert("RGB"), np.uint8)


def permute_palette_assignment(rgb: np.ndarray, palette: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Re-map every palette entry to a different (seeded) palette entry."""
    pal = np.unique(np.asarray(palette, np.uint8).reshape(-1, 3), axis=0)
    keys = (pal[:, 0].astype(np.int64) << 16) | (pal[:, 1].astype(np.int64) << 8) | pal[:, 2]
    order = np.argsort(keys)
    keys, pal = keys[order], pal[order]
    flat = rgb.reshape(-1, 3)
    pk = (flat[:, 0].astype(np.int64) << 16) | (flat[:, 1].astype(np.int64) << 8) | flat[:, 2]
    idx = np.clip(np.searchsorted(keys, pk), 0, len(keys) - 1)
    n = len(pal)
    if n < 2:
        return rgb.copy()
    # cyclic shift of a random order: every entry maps to a different colour
    order = rng.permutation(n)
    mapping = np.empty(n, np.int64)
    mapping[order] = np.roll(order, 1)
    return pal[mapping[idx]].reshape(rgb.shape)


_DITHERS = (("none", "None"), ("ordered2", "Ordered 2x2"), ("ordered4", "Ordered 4x4"), ("ordered8", "Ordered 8x8"),
            ("floyd", "Floyd–Steinberg (error diffusion)"))


@register
class PaletteDamage(Effect):
    id = "palette"
    name = "Palette Damage"
    category = "Colour & Bits"
    description = ("Posterisation, adaptive or fixed palettes, imported palettes (.gpl/.act/.hex/image), seeded "
                   "palette permutation and false-colour gradient maps, with ordered or error-diffusion dithering.")
    params = (
        cparam("mode", "Mode", "posterize",
               (("posterize", "Posterise"), ("adaptive", "Adaptive palette"), ("fixed", "Fixed palette"),
                ("file", "Imported palette"), ("false_colour", "False colour (gradient map)"))),
        iparam("levels", "Levels / colours", 5, 2, 64, tooltip="Levels per channel (posterise/false colour) or colours (adaptive)."),
        cparam("fixed_palette", "Palette", "vapor", tuple((k, k.upper() if len(k) < 4 else k.title()) for k in FIXED_PALETTES),
               show_if=("mode", ("fixed",))),
        Param("palette_file", "Palette file", FILE, "", file_filter="Palettes (*.gpl *.act *.hex *.txt *.png *.jpg *.gif *.bmp);;All files (*)",
              show_if=("mode", ("file",)), randomizable=False),
        cparam("gradient", "Gradient", "thermal", tuple((k, k.title()) for k in GRADIENTS),
               show_if=("mode", ("false_colour",))),
        Param("dither", "Dither", "choice", "none", choices=_DITHERS),
        fparam("spread", "Dither strength", 1.0, 0.0, 2.0, show_if=("dither", ("ordered2", "ordered4", "ordered8"))),
        bparam("permute", "Shuffle palette (seeded)", False, tooltip="Randomly reassigns palette entries: wild false colours."),
    )

    def validate(self, p):
        if p["mode"] == "file" and not p["palette_file"]:
            return ["choose a palette file (GIMP .gpl, Adobe .act, .hex text or an image)."]
        return []

    def process(self, img, p, ctx):
        rgb = np.ascontiguousarray(img[..., :3])
        h, w = rgb.shape[:2]
        mode = p["mode"]
        levels = int(p["levels"])
        dither = p["dither"]
        if mode in ("posterize", "false_colour"):
            src = luminance(img)[..., None] if mode == "false_colour" else rgb.astype(np.float32) / 255.0
            if dither == "floyd":
                L = max(2, levels)
                pal = np.array([[round(i * 255 / (L - 1))] * 3 for i in range(L)], np.uint8)
                if mode == "posterize":
                    grid = np.array(np.meshgrid(*[np.linspace(0, 255, min(L, 6))] * 3)).reshape(3, -1).T
                    pal = np.rint(grid).astype(np.uint8)
                    q = quantize_to_palette(rgb, pal, "floyd", 1.0).astype(np.float32) / 255.0
                else:
                    g = to_uint8(src[..., 0])
                    q = quantize_to_palette(np.dstack([g, g, g]), pal, "floyd", 1.0)[..., :1].astype(np.float32) / 255.0
            else:
                x = src * (levels - 1)
                if dither.startswith("ordered"):
                    n = int(dither[len("ordered"):])
                    bm = bayer_matrix(n)
                    tile = np.tile(bm, (h // n + 1, w // n + 1))[:h, :w][..., None]
                    x = x + (tile - 0.5) * p["spread"]
                q = np.clip(np.rint(x), 0, levels - 1) / (levels - 1)
            if mode == "false_colour":
                lut = gradient_lut(GRADIENTS[p["gradient"]])
                if p["permute"]:
                    idx = np.rint(np.linspace(0, 255, levels)).astype(int)
                    perm = ctx.rng.permutation(levels)
                    lut = np.repeat(lut[idx[perm]], int(np.ceil(256 / levels)), axis=0)[:256]
                res = lut[np.clip(np.rint(q[..., 0] * 255), 0, 255).astype(np.int64)]
                out_rgb = to_uint8(res, scale=1.0)
            else:
                if p["permute"]:
                    perm = ctx.rng.permutation(levels)
                    qi = np.rint(q * (levels - 1)).astype(np.int64)
                    q = perm[qi] / (levels - 1)
                out_rgb = to_uint8(q)
        else:
            if mode == "adaptive":
                im = Image.fromarray(rgb, "RGB")
                qimg = im.quantize(colors=levels, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
                ncol = len(qimg.getcolors(256) or []) or levels
                palette = np.array(qimg.getpalette()[: 3 * ncol], np.uint8).reshape(-1, 3)
            elif mode == "fixed":
                palette = np.array([_hex_to_rgb(c) for c in FIXED_PALETTES[p["fixed_palette"]]], np.uint8)
            else:
                palette = load_palette_file(p["palette_file"])
            out_rgb = quantize_to_palette(rgb, palette, dither, p["spread"])
            if p["permute"]:
                out_rgb = permute_palette_assignment(out_rgb, palette, ctx.rng)
        out = img.copy()
        out[..., :3] = out_rgb
        return out


# ---------------------------------------------------------------------------
# Error-diffusion trails
# ---------------------------------------------------------------------------

@register
class ErrorTrails(Effect):
    id = "error_trails"
    name = "Error Diffusion Trails"
    category = "Colour & Bits"
    description = ("Quantises colours while pushing each line's quantisation error into the next line with "
                   "adjustable direction, weights, drift and gain. Error is clamped so feedback stays bounded.")
    params = (
        iparam("levels", "Levels", 4, 2, 32),
        cparam("direction", "Error flows", "down", (("down", "Down"), ("up", "Up"), ("right", "Right"), ("left", "Left"))),
        fparam("w_down", "Weight straight", 0.5, 0.0, 1.0),
        fparam("w_down_left", "Weight diagonal −", 0.25, 0.0, 1.0),
        fparam("w_down_right", "Weight diagonal +", 0.25, 0.0, 1.0),
        fparam("gain", "Feedback gain", 1.0, 0.0, 1.5, tooltip="Above 1 the error grows into long trails (bounded by the clamp)."),
        iparam("drift", "Drift", 0, -16, 16, unit="px/line", tooltip="Shifts the carried error sideways each line (slanted trails)."),
        iparam("clamp", "Error clamp", 64, 1, 255, tooltip="Maximum carried error per channel (prevents overflow)."),
        cparam("mode", "Colour", "per_channel", (("per_channel", "Per channel"), ("luminance", "Luminance only"))),
    )
    uses_seed = False

    def process(self, img, p, ctx):
        d = p["direction"]
        rgb = img[..., :3].astype(np.float32)
        if d == "up":
            rgb = rgb[::-1]
        elif d in ("right", "left"):
            rgb = np.swapaxes(rgb, 0, 1)
            if d == "left":
                rgb = rgb[::-1]
        if p["mode"] == "luminance":
            y = (rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114)[..., None]
            chroma = rgb - y
            work = y
        else:
            work = rgb
            chroma = None
        h, w, c = work.shape
        L = int(p["levels"])
        step = 255.0 / (L - 1)
        clampv = float(p["clamp"])
        wd, wl, wr = p["w_down"], p["w_down_left"], p["w_down_right"]
        gain = float(p["gain"])
        drift = int(p["drift"])
        carry = np.zeros((w, c), np.float32)
        out = np.empty_like(work)
        for yy in range(h):
            if yy % 128 == 0:
                ctx.check()
                ctx.progress(yy / h)
            v = work[yy] + carry
            q = np.clip(np.rint(v / step), 0, L - 1) * step
            out[yy] = q
            err = np.clip((v - q) * gain, -clampv, clampv)
            nxt = wd * err
            nxt[1:] += wl * err[:-1]
            nxt[:-1] += wr * err[1:]
            if drift:
                nxt = np.roll(nxt, drift, axis=0)
            carry = np.clip(nxt, -clampv, clampv)
        if chroma is not None:
            out = out + chroma
        if d == "up":
            out = out[::-1]
        elif d in ("right", "left"):
            if d == "left":
                out = out[::-1]
            out = np.swapaxes(out, 0, 1)
        res = img.copy()
        res[..., :3] = to_uint8(np.nan_to_num(out), scale=1.0)
        return res
