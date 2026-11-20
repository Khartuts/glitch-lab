"""Aesthetic stylisation effects: psychedelic pop, dark deco, luminous gradient
fields and moody corporate futurism.

Procedural line work (deco rays, arcs, chevrons, perspective grids) is drawn
from scalar fields: a line sits wherever a field crosses an integer, and its
pixel distance is estimated as |F - round(F)| / |grad F|, giving crisp,
anti-aliased lines of a constant pixel width at any resolution.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ..ops import luminance, to_float, to_uint8
from ..params import COLOR, Param, bparam, color_to_rgb, cparam, fparam, iparam
from .base import Effect, register


def _hex(c: str) -> np.ndarray:
    return np.array(color_to_rgb(c), np.float32) / 255.0


def gradient_map(t: np.ndarray, stops: list[str], positions: list[float] | None = None) -> np.ndarray:
    """Map t (0..1) through colour stops -> float RGB (H, W, 3)."""
    cols = np.array([_hex(s) for s in stops], np.float32)
    xp = np.linspace(0.0, 1.0, len(stops)) if positions is None else np.asarray(positions, np.float32)
    t = np.clip(t, 0.0, 1.0)
    return np.stack([np.interp(t, xp, cols[:, c]).astype(np.float32) for c in range(3)], -1)


def field_lines(F: np.ndarray, width_px: float) -> np.ndarray:
    """Anti-aliased lines where the scalar field F crosses integers (0..1 mask)."""
    gy, gx = np.gradient(F.astype(np.float32))
    grad = np.sqrt(gx * gx + gy * gy) + 1e-6
    d = np.abs(F - np.rint(F)) / grad
    return np.clip(width_px / 2.0 - d + 0.5, 0.0, 1.0).astype(np.float32)


def saturate(rgb: np.ndarray, amount: float) -> np.ndarray:
    g = rgb.mean(axis=-1, keepdims=True)
    return np.clip(g + (rgb - g) * np.float32(amount), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Psychedelic pop
# ---------------------------------------------------------------------------

PSY_PALETTES = {
    "rainbow": ["#ff2d55", "#ff9500", "#ffe600", "#34c759", "#00c7ff", "#5856d6", "#ff2dd4", "#ff2d55"],
    "candy": ["#ff5fa2", "#ffd1e8", "#7ee8fa", "#b388ff", "#fff07a", "#ff5fa2"],
    "seventies": ["#e85d04", "#f48c06", "#faa307", "#6a994e", "#bc4749", "#7b2cbf", "#e85d04"],
    "acid": ["#39ff14", "#ff00ff", "#00ffff", "#ffff00", "#ff3131", "#39ff14"],
    "lava": ["#2b0a3d", "#ff2e63", "#ff9a00", "#ffe066", "#ff2e63", "#2b0a3d"],
}


def swirl_maps(h: int, w: int, cx: float, cy: float, angle_deg: float, radius: float):
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32) - cx, np.arange(h, dtype=np.float32) - cy)
    r = np.sqrt(xs * xs + ys * ys)
    k = np.clip(1.0 - r / max(radius, 1.0), 0.0, 1.0) ** 2
    a = np.float32(math.radians(angle_deg)) * k
    c, s = np.cos(a), np.sin(a)
    return (cx + xs * c - ys * s).astype(np.float32), (cy + xs * s + ys * c).astype(np.float32)


@register
class PsychedelicPop(Effect):
    id = "psychedelic_pop"
    name = "Psychedelic Pop"
    category = "Abstract & Style"
    description = ("60s/70s psychedelic poster look: the photo's brightness becomes bands of cycling rainbow, candy, "
                   "acid or seventies colours with echoing contour outlines and an optional liquid swirl.")
    uses_seed = False
    supports_channels = False
    params = (
        cparam("palette", "Palette", "rainbow", tuple((k, k.title()) for k in PSY_PALETTES)),
        iparam("bands", "Colour bands", 12, 2, 64),
        fparam("cycles", "Colour cycles", 2.0, 0.25, 12.0, tooltip="How many times the palette repeats from dark to light."),
        fparam("offset", "Colour offset", 0.0, 0.0, 1.0),
        iparam("smooth", "Smoothing", 6, 0, 200, spatial=True, unit="px", tooltip="Rounder, poster-like shapes."),
        fparam("saturation", "Saturation", 1.15, 0.0, 2.0),
        fparam("detail", "Keep shading", 0.25, 0.0, 1.0, tooltip="Blend the photo's light and shade into the colours."),
        fparam("outline", "Echo outlines", 0.7, 0.0, 1.0),
        iparam("outline_px", "Outline width", 2, 1, 40, spatial=True, unit="px"),
        Param("outline_color", "Outline colour", COLOR, "#1b0b2e"),
        fparam("swirl", "Swirl", 0.0, -1080.0, 1080.0, decimals=0, unit="°", group="Swirl"),
        fparam("swirl_radius", "Swirl radius", 0.6, 0.05, 1.5, group="Swirl"),
        fparam("cx", "Swirl centre X", 0.5, 0.0, 1.0, group="Swirl"),
        fparam("cy", "Swirl centre Y", 0.5, 0.0, 1.0, group="Swirl"),
    )

    def process(self, img, p, ctx):
        h, w = img.shape[:2]
        lum = luminance(img)
        if p["smooth"] > 0:
            lum = cv2.GaussianBlur(lum, (0, 0), max(0.5, p["smooth"] / 2.0))
        if abs(p["swirl"]) > 0.01:
            mx, my = swirl_maps(h, w, p["cx"] * w, p["cy"] * h, p["swirl"], p["swirl_radius"] * math.hypot(w, h) / 2)
            lum = cv2.remap(lum, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        n = int(p["bands"])
        band = np.clip((lum * n).astype(np.int32), 0, n - 1)
        t = np.mod(band.astype(np.float32) / n * p["cycles"] + p["offset"], 1.0)
        col = gradient_map(t, PSY_PALETTES[p["palette"]])
        if p["detail"] > 0:
            col = col * ((1.0 - p["detail"]) + p["detail"] * (0.45 + lum))[..., None]
        col = saturate(col, p["saturation"])
        if p["outline"] > 0:
            edge = np.zeros((h, w), np.uint8)
            edge[:, 1:] |= (band[:, 1:] != band[:, :-1]).astype(np.uint8)
            edge[1:, :] |= (band[1:, :] != band[:-1, :]).astype(np.uint8)
            k = int(p["outline_px"])
            if k > 1:
                edge = cv2.dilate(edge, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
            a = (edge.astype(np.float32) * p["outline"])[..., None]
            col = col * (1 - a) + _hex(p["outline_color"]) * a
        out = img.copy()
        out[..., :3] = to_uint8(col)
        return out


# ---------------------------------------------------------------------------
# Dark deco
# ---------------------------------------------------------------------------

DECO_PALETTES = {
    "noir_gold": ("#060608", "#22262c", "#d4af37", "#f1d77a"),
    "emerald": ("#020d0a", "#0c4232", "#d9b65d", "#f3dc8f"),
    "navy_brass": ("#03060e", "#14294d", "#c9a227", "#ead07a"),
    "burgundy": ("#0c0305", "#4b0f1e", "#e0b46b", "#f7dca6"),
    "silver_jet": ("#040405", "#1c1d22", "#b9bec8", "#f2f4f8"),
}
_DECO_NAMES = {"noir_gold": "Noir & gold", "emerald": "Emerald & gold", "navy_brass": "Navy & brass",
               "burgundy": "Burgundy & champagne", "silver_jet": "Jet & silver"}


@register
class DarkDeco(Effect):
    id = "dark_deco"
    name = "Dark Deco"
    category = "Abstract & Style"
    description = ("Moody Art Deco: crushed blacks, a deep jewel tone and metallic gold highlights, overlaid with "
                   "gold sunbursts, fans, arches or chevrons and a double Deco frame.")
    uses_seed = False
    supports_channels = False
    params = (
        cparam("palette", "Palette", "noir_gold", tuple(_DECO_NAMES.items())),
        fparam("contrast", "Contrast", 1.35, 0.5, 3.0),
        fparam("gold_level", "Gold threshold", 0.68, 0.3, 0.95, tooltip="Brightness above which tones turn metallic."),
        cparam("pattern", "Pattern", "sunburst", (("sunburst", "Sunburst rays"), ("fan", "Fan"), ("arches", "Nested arches"),
                                                  ("chevron", "Chevrons"), ("none", "None")), group="Linework"),
        iparam("density", "Density", 18, 2, 120, group="Linework"),
        fparam("origin_y", "Origin height", 1.0, -0.5, 1.5, group="Linework",
               tooltip="Vertical position the rays/fans spring from (1 = bottom edge)."),
        fparam("pattern_opacity", "Pattern opacity", 0.55, 0.0, 1.0, group="Linework"),
        iparam("line_px", "Line width", 2, 1, 40, spatial=True, unit="px", group="Linework"),
        bparam("panels", "Alternate panels", True, group="Linework", tooltip="Shade every other sunburst/fan segment."),
        cparam("frame", "Frame", "double", (("none", "None"), ("single", "Single"), ("double", "Double")), group="Frame"),
        fparam("frame_inset", "Frame inset", 0.04, 0.0, 0.25, group="Frame"),
        fparam("vignette", "Vignette", 0.5, 0.0, 1.0),
    )

    def process(self, img, p, ctx):
        h, w = img.shape[:2]
        L = float(max(h, w))
        shadow, mid, gold, gold_hi = DECO_PALETTES[p["palette"]]
        lum = luminance(img)
        lum = np.clip(0.5 + (lum - 0.5) * p["contrast"], 0.0, 1.0)
        g = p["gold_level"]
        tone = gradient_map(lum, [shadow, mid, mid, gold, gold_hi], [0.0, 0.45, g, min(0.99, g + 0.18), 1.0])
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        sheen = 0.9 + 0.1 * np.cos(yy / L * 40.0 + xx / L * 7.0)
        metal = np.clip((lum - g) / max(1e-3, 1 - g), 0, 1)[..., None]
        tone = tone * (1 - metal) + tone * sheen[..., None] * metal
        ox, oy = 0.5 * w, p["origin_y"] * h
        X, Y = xx - ox, yy - oy
        r = np.sqrt(X * X + Y * Y) / L
        pat = p["pattern"]
        lines = np.zeros((h, w), np.float32)
        panel = np.zeros((h, w), np.float32)
        n = float(p["density"])
        if pat == "sunburst":
            th = np.arctan2(Y, X)
            F = th / (2 * np.pi) * n * 2
            lines = field_lines(F, p["line_px"])
            panel = (np.floor(F) % 2).astype(np.float32)
        elif pat == "fan":
            th = np.arctan2(Y, X)
            Fa = r * n
            Fr = th / np.pi * (n / 2)
            lines = np.maximum(field_lines(Fa, p["line_px"]), field_lines(Fr, p["line_px"]) * (r > 0.15))
            above = (Y <= 0).astype(np.float32)
            lines *= above
            panel = (np.floor(Fr) % 2).astype(np.float32) * above
        elif pat == "arches":
            a, b = 0.5 * w / L, h / L
            rho = np.sqrt((X / L / a) ** 2 + (Y / L / b) ** 2)
            lines = field_lines(rho * n * 0.5, p["line_px"])
            panel = (np.floor(rho * n * 0.5) % 2).astype(np.float32)
        elif pat == "chevron":
            F = (Y / L + np.abs(X / L) * 0.6) * n
            lines = field_lines(F, p["line_px"])
            panel = (np.floor(F) % 2).astype(np.float32)
        if p["frame"] != "none":
            m = p["frame_inset"] * min(h, w)
            hw, hh = w / 2 - m, h / 2 - m
            cxx, cyy = xx - w / 2, yy - h / 2
            sdf = np.maximum(np.abs(cxx) - hw, np.abs(cyy) - hh)
            fr = np.clip(p["line_px"] / 2.0 - np.abs(sdf) + 0.5, 0, 1)
            if p["frame"] == "double":
                gap = max(3.0, 3.0 * p["line_px"])
                fr = np.maximum(fr, np.clip(p["line_px"] / 2.0 - np.abs(sdf + gap) + 0.5, 0, 1))
            lines = np.maximum(lines, fr)
        out = tone
        if p["panels"] and pat in ("sunburst", "fan", "arches", "chevron"):
            out = out * (1.0 - 0.12 * panel[..., None] * p["pattern_opacity"])
        if pat != "none" or p["frame"] != "none":
            gold_line = (_hex(gold) * 0.6 + _hex(gold_hi) * 0.4) * sheen[..., None]
            a = (lines * p["pattern_opacity"])[..., None]
            if p["frame"] != "none":
                a = np.maximum(a, (fr * min(1.0, p["pattern_opacity"] + 0.35))[..., None])
            out = out * (1 - a) + gold_line * a
        if p["vignette"] > 0:
            nx, ny = (xx / w * 2 - 1), (yy / h * 2 - 1)
            out = out * np.clip(1 - p["vignette"] * (nx * nx + ny * ny) * 0.45, 0, 1)[..., None]
        res = img.copy()
        res[..., :3] = to_uint8(out)
        return res


# ---------------------------------------------------------------------------
# Luminous gradient field (in the spirit of Koichi Sato's glowing posters)
# ---------------------------------------------------------------------------

GLOW_PALETTES = {
    "deep_blue": ["#02030f", "#0b1a5c", "#3a2c9e", "#8f7bf0", "#eceaff"],
    "dawn": ["#120214", "#5b1647", "#d6456f", "#f7a072", "#fff3dc"],
    "aurora": ["#010806", "#063b3a", "#0f8f7a", "#7ef0c4", "#f2fff9"],
    "gold_mist": ["#0c0703", "#4a2a0c", "#b8721d", "#f2c66d", "#fff9e8"],
    "ultraviolet": ["#05010a", "#2a0845", "#6f1bc4", "#ff5ad8", "#ffe9fc"],
    "ink_moon": ["#000000", "#0f1418", "#3b4a55", "#a9bcc7", "#ffffff"],
}
_GLOW_NAMES = {"deep_blue": "Deep blue / violet", "dawn": "Dawn rose", "aurora": "Aurora green", "gold_mist": "Gold mist",
               "ultraviolet": "Ultraviolet", "ink_moon": "Ink & moonlight"}


@register
class LuminousGradient(Effect):
    id = "luminous_gradient"
    name = "Luminous Gradient Field"
    category = "Abstract & Style"
    description = ("Dissolves the photo into soft, glowing colour fields around a radiant light source - quiet, "
                   "mystical gradients in the spirit of Japanese poster design such as Koichi Sato's.")
    supports_channels = False
    params = (
        cparam("palette", "Palette", "deep_blue", tuple(_GLOW_NAMES.items())),
        fparam("abstraction", "Abstraction", 0.55, 0.0, 1.0, tooltip="How much of the photo dissolves into gradient."),
        fparam("glow_x", "Light X", 0.5, -0.2, 1.2, group="Light"),
        fparam("glow_y", "Light Y", 0.42, -0.2, 1.2, group="Light"),
        fparam("glow_radius", "Light radius", 0.35, 0.03, 1.5, group="Light"),
        fparam("glow", "Light strength", 0.9, 0.0, 2.0, group="Light"),
        fparam("halo", "Halo bloom", 0.5, 0.0, 1.5, group="Light"),
        iparam("steps", "Soft banding", 0, 0, 24, tooltip="0 = perfectly smooth; otherwise gentle tonal steps."),
        fparam("detail", "Keep detail", 0.15, 0.0, 1.0),
        fparam("grain", "Grain", 0.03, 0.0, 0.2, decimals=3),
        fparam("vignette", "Vignette", 0.55, 0.0, 1.0),
    )

    def process(self, img, p, ctx):
        h, w = img.shape[:2]
        L = float(max(h, w))
        lum = luminance(img)
        base = lum
        if p["abstraction"] > 0:
            base = cv2.GaussianBlur(lum, (0, 0), max(0.5, p["abstraction"] * 0.07 * L))
        lo, hi = np.percentile(base, [2, 98])
        base = np.clip((base - lo) / max(1e-4, hi - lo), 0, 1)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        R = max(1.0, p["glow_radius"] * L)
        d2 = ((xx - p["glow_x"] * w) ** 2 + (yy - p["glow_y"] * h) ** 2) / (R * R)
        light = np.exp(-d2 * 2.2).astype(np.float32)
        F = base * (1.0 - 0.45 * min(1.0, p["glow"])) * 0.85 + light * p["glow"] * 0.75
        if p["detail"] > 0:
            hp = lum - cv2.GaussianBlur(lum, (0, 0), max(0.6, 0.004 * L))
            F = F + hp * p["detail"] * 1.5
        F = np.clip(F, 0, 1)
        if p["steps"] > 0:
            s = int(p["steps"])
            F = np.floor(F * s) / s
            F = cv2.GaussianBlur(F.astype(np.float32), (0, 0), max(0.6, 0.0025 * L))
        col = gradient_map(F, GLOW_PALETTES[p["palette"]])
        if p["halo"] > 0:
            bright = np.clip(col - 0.55, 0, 1)
            bloom = cv2.GaussianBlur(bright, (0, 0), max(1.0, 0.03 * L))
            col = 1.0 - (1.0 - col) * (1.0 - np.clip(bloom * p["halo"] * 1.6, 0, 1))
        if p["vignette"] > 0:
            nx, ny = (xx / w * 2 - 1), (yy / h * 2 - 1)
            col = col * np.clip(1 - p["vignette"] * (nx * nx + ny * ny) * 0.5, 0, 1)[..., None]
        if p["grain"] > 0:
            col = col + ctx.rng.normal(0, p["grain"], (h, w, 1)).astype(np.float32)
        out = img.copy()
        out[..., :3] = to_uint8(col)
        return out


# ---------------------------------------------------------------------------
# Moody corporate futurism
# ---------------------------------------------------------------------------

@register
class CorporateFuturism(Effect):
    id = "corporate_futurism"
    name = "Corporate Futurism"
    category = "Abstract & Style"
    description = ("Moody corporate-future look: a cold, dark grade with one accent colour, a perspective grid "
                   "floor, HUD brackets and frosted glass panels, a horizon haze and a scanning light band.")
    supports_channels = False
    params = (
        fparam("grade", "Grade strength", 0.85, 0.0, 1.0, group="Grade"),
        Param("shadow", "Shadow colour", COLOR, "#04070d", group="Grade"),
        Param("mid", "Mid colour", COLOR, "#18293d", group="Grade"),
        Param("highlight", "Highlight colour", COLOR, "#d9ecff", group="Grade"),
        Param("accent", "Accent colour", COLOR, "#22e0ff", group="Grade"),
        fparam("accent_amount", "Accent on edges", 0.45, 0.0, 1.0, group="Grade"),
        bparam("grid", "Perspective grid", True, group="Grid"),
        fparam("horizon", "Horizon height", 0.62, 0.05, 0.95, group="Grid"),
        iparam("grid_density", "Grid density", 14, 2, 80, group="Grid"),
        fparam("grid_opacity", "Grid opacity", 0.45, 0.0, 1.0, group="Grid"),
        fparam("haze", "Horizon haze", 0.4, 0.0, 1.0, group="Grid"),
        iparam("hud", "HUD elements", 9, 0, 60, group="Interface"),
        iparam("panels", "Glass panels", 2, 0, 12, group="Interface"),
        iparam("panel_blur", "Glass blur", 14, 0, 400, spatial=True, unit="px", group="Interface"),
        fparam("line_opacity", "Interface opacity", 0.7, 0.0, 1.0, group="Interface"),
        iparam("line_px", "Line width", 1, 1, 20, spatial=True, unit="px", group="Interface"),
        fparam("scan", "Scan light", 0.25, 0.0, 1.0),
    )

    def process(self, img, p, ctx):
        h, w = img.shape[:2]
        L = float(max(h, w))
        f = to_float(img[..., :3])
        lum = luminance(img)
        graded = gradient_map(lum, [p["shadow"], p["mid"], p["highlight"]], [0.0, 0.55, 1.0])
        out = f * (1 - p["grade"]) + graded * p["grade"]
        accent = _hex(p["accent"])
        if p["accent_amount"] > 0:
            e = np.hypot(cv2.Sobel(lum, cv2.CV_32F, 1, 0), cv2.Sobel(lum, cv2.CV_32F, 0, 1))
            e = np.clip(e / max(1e-4, float(np.percentile(e, 99))), 0, 1)
            out = out + (e * p["accent_amount"])[..., None] * accent * 0.8
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        hy = p["horizon"] * h
        if p["grid"]:
            below = yy > hy + 1
            depth = np.where(below, (h - hy) / np.maximum(yy - hy, 1.0), 0).astype(np.float32)
            n = float(p["grid_density"])
            Fz = depth * n * 0.25
            Fx = (xx - w / 2) / L * depth * n
            g = np.maximum(field_lines(Fx, p["line_px"] * 1.2), field_lines(Fz, p["line_px"] * 1.2))
            fade = np.clip((yy - hy) / max(1.0, h - hy), 0, 1) ** 0.7
            g = g * below * fade
            out = out * (1 - (g * p["grid_opacity"])[..., None]) + accent * (g * p["grid_opacity"])[..., None]
        if p["haze"] > 0:
            band = np.exp(-((yy - hy) / (0.05 * h)) ** 2)
            out = out + (band * p["haze"] * 0.5)[..., None] * (accent * 0.5 + 0.5 * _hex(p["highlight"]))
        rng = ctx.rng
        for _ in range(int(p["panels"])):
            pw, ph = rng.uniform(0.15, 0.35) * w, rng.uniform(0.12, 0.3) * h
            px, py = rng.uniform(0.03, 0.97) * w - pw / 2, rng.uniform(0.05, 0.9) * h - ph / 2
            x0, y0 = int(max(0, px)), int(max(0, py))
            x1, y1 = int(min(w, px + pw)), int(min(h, py + ph))
            if x1 - x0 < 3 or y1 - y0 < 3:
                continue
            region = np.ascontiguousarray(out[y0:y1, x0:x1])
            if p["panel_blur"] > 0:
                region = cv2.GaussianBlur(region, (0, 0), max(0.5, p["panel_blur"] / 2.0))
            region = region * 0.75 + 0.12 + accent * 0.04
            out[y0:y1, x0:x1] = region
            ln = max(1, int(p["line_px"]))
            out[y0:y0 + ln, x0:x1] = out[y0:y0 + ln, x0:x1] * 0.3 + 0.7 * (accent * 0.3 + 0.7)
        overlay = np.zeros((h, w, 3), np.float32)
        lw = max(1, int(p["line_px"]))
        for _ in range(int(p["hud"])):
            kind = rng.integers(0, 4)
            cx, cy = rng.uniform(0.05, 0.95) * w, rng.uniform(0.05, 0.95) * h
            s = rng.uniform(0.03, 0.12) * L
            col = tuple(float(v) for v in accent)
            if kind == 0:  # corner brackets
                x0, y0, x1, y1 = int(cx - s), int(cy - s * 0.7), int(cx + s), int(cy + s * 0.7)
                b = int(s * 0.3)
                for (ax, ay, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
                    cv2.line(overlay, (ax, ay), (ax + dx * b, ay), col, lw, cv2.LINE_AA)
                    cv2.line(overlay, (ax, ay), (ax, ay + dy * b), col, lw, cv2.LINE_AA)
            elif kind == 1:  # crosshair
                cv2.circle(overlay, (int(cx), int(cy)), int(s * 0.5), col, lw, cv2.LINE_AA)
                cv2.line(overlay, (int(cx - s * 0.8), int(cy)), (int(cx - s * 0.2), int(cy)), col, lw, cv2.LINE_AA)
                cv2.line(overlay, (int(cx + s * 0.2), int(cy)), (int(cx + s * 0.8), int(cy)), col, lw, cv2.LINE_AA)
            elif kind == 2:  # tick ruler
                n = 12
                for i in range(n):
                    x = int(cx - s + 2 * s * i / (n - 1))
                    tl = int(s * (0.25 if i % 4 == 0 else 0.1))
                    cv2.line(overlay, (x, int(cy)), (x, int(cy) - tl), col, lw, cv2.LINE_AA)
                cv2.line(overlay, (int(cx - s), int(cy)), (int(cx + s), int(cy)), col, lw, cv2.LINE_AA)
            else:  # data bars
                for i in range(5):
                    bw = int(s * rng.uniform(0.3, 1.6))
                    y = int(cy + i * max(3, lw * 4))
                    cv2.line(overlay, (int(cx), y), (int(cx) + bw, y), col, lw, cv2.LINE_AA)
        a = np.clip(overlay.max(axis=-1, keepdims=True) / max(1e-4, float(accent.max())), 0, 1) * p["line_opacity"]
        out = out * (1 - a) + overlay / max(1e-4, float(accent.max())) * accent * a
        if p["scan"] > 0:
            sy = rng.uniform(0.15, 0.85) * h
            band = np.exp(-((yy - sy) / (0.025 * h)) ** 2)
            out = out + (band * p["scan"] * 0.35)[..., None] * (accent * 0.6 + 0.4)
        res = img.copy()
        res[..., :3] = to_uint8(out)
        return res
