"""Print-process effects: halftone screening, CMYK halftone drift (with
misregistered / sliced plates) and Ben-Day dots.

A halftone *plate* is rendered with a classic spot function on a rotated
screen: every pixel finds the centre of its screen cell, samples the
coverage (ink amount) there, and is inked when its spot-function value
exceeds a threshold derived from that coverage.  Round dots use the
Euclidean spot function, so dots grow, touch at 50 % and invert into white
holes in dark tones exactly like printed halftones.  Screen geometry is in
full-resolution pixels and scaled for previews.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ..ops import luminance, shift_plane, shift_rows, to_float, to_uint8
from ..params import COLOR, Param, color_to_rgb, cparam, fparam, iparam
from .base import Effect, register

SHAPES = (("round", "Round (Euclidean)"), ("square", "Square"), ("line", "Line"), ("diamond", "Diamond"))
#: Traditional CMYK screen angles (degrees).
CMYK_ANGLES = {"c": 15.0, "m": 75.0, "y": 0.0, "k": 45.0}


def rgb_to_cmyk(rgb: np.ndarray, gcr: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Naive device CMYK separation with adjustable grey-component replacement."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    k = (1.0 - np.maximum(np.maximum(r, g), b)) * np.float32(gcr)
    denom = np.maximum(1.0 - k, 1e-4)
    c = np.clip((1.0 - r - k) / denom, 0.0, 1.0)
    m = np.clip((1.0 - g - k) / denom, 0.0, 1.0)
    y = np.clip((1.0 - b - k) / denom, 0.0, 1.0)
    return c.astype(np.float32), m.astype(np.float32), y.astype(np.float32), np.clip(k, 0, 1).astype(np.float32)


def _spot(fu: np.ndarray, fv: np.ndarray, shape: str) -> np.ndarray:
    """Spot function value (higher = inked first) for cell positions in [-1, 1]."""
    ax, ay = np.abs(fu), np.abs(fv)
    if shape == "square":
        return 1.0 - np.maximum(ax, ay)
    if shape == "line":
        return 1.0 - ay
    if shape == "diamond":
        return 1.0 - (ax + ay) * 0.5
    # Euclidean round dot: round in highlights, checkerboard at 50 %, round holes in shadows
    return np.where(ax + ay <= 1.0, 1.0 - (ax * ax + ay * ay), (ax - 1.0) ** 2 + (ay - 1.0) ** 2 - 1.0)


_SPOT_TABLES: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def _spot_threshold_table(shape: str) -> tuple[np.ndarray, np.ndarray]:
    """Linearisation: coverage c -> threshold t such that exactly c of the cell
    area has spot >= t (computed once per shape from a dense sample)."""
    if shape not in _SPOT_TABLES:
        g = (np.arange(512, dtype=np.float32) + 0.5) / 256.0 - 1.0
        fu, fv = np.meshgrid(g, g)
        vals = np.sort(_spot(fu, fv, shape).ravel())
        cov = np.linspace(0.0, 1.0, 1025, dtype=np.float32)
        idx = np.clip(((1.0 - cov) * (vals.size - 1)).round().astype(np.int64), 0, vals.size - 1)
        _SPOT_TABLES[shape] = (cov, vals[idx].astype(np.float32))
    return _SPOT_TABLES[shape]


def _ink_from_spot(fu: np.ndarray, fv: np.ndarray, cov: np.ndarray, shape: str, aa: float) -> np.ndarray:
    """fu, fv: position inside the cell in [-1, 1]; cov: coverage at the cell centre."""
    cov = np.clip(cov, 0.0, 1.0)
    s = _spot(fu, fv, shape)
    cgrid, tgrid = _spot_threshold_table(shape)
    t = np.interp(cov, cgrid, tgrid).astype(np.float32)
    w = aa * (2.0 if shape == "round" else (0.5 if shape == "diamond" else 1.0))
    ink = np.clip((s - t) / np.maximum(w, 1e-4) + 0.5, 0.0, 1.0)
    ink = np.where(cov <= 0.002, 0.0, np.where(cov >= 0.998, 1.0, ink))
    return ink.astype(np.float32)


def screen_plate(cov: np.ndarray, cell: float, angle: float, shape: str = "round", *, softness: float = 1.0,
                 warp=None, check=None, chunk: int = 384) -> np.ndarray:
    """Halftone a coverage map (H, W float32, 0..1) -> ink map (H, W float32, 0..1).

    ``warp(X, Y) -> (X', Y')`` displaces the *screen* (dot grid) coordinates;
    the image content is still sampled at the undisplaced cell centres.
    """
    h, w = cov.shape
    cell = max(2.0, float(cell))
    th = math.radians(angle)
    ca, sa = np.float32(math.cos(th)), np.float32(math.sin(th))
    aa = np.float32(2.0 * softness / cell)
    cov = np.ascontiguousarray(cov, dtype=np.float32)
    out = np.empty((h, w), np.float32)
    xs = np.arange(w, dtype=np.float32) + 0.5
    for y0 in range(0, h, chunk):
        if check:
            check()
        y1 = min(h, y0 + chunk)
        X, Y = np.meshgrid(xs, np.arange(y0, y1, dtype=np.float32) + 0.5)
        if warp is not None:
            X, Y = warp(X, Y)
        u = (X * ca + Y * sa) / cell
        v = (-X * sa + Y * ca) / cell
        cu = np.floor(u) + 0.5
        cv_ = np.floor(v) + 0.5
        fu = (u - cu) * 2.0
        fv = (v - cv_) * 2.0
        cx = ((cu * ca - cv_ * sa) * cell - 0.5).astype(np.float32)
        cy = ((cu * sa + cv_ * ca) * cell - 0.5).astype(np.float32)
        c = cv2.remap(cov, cx, cy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        out[y0:y1] = _ink_from_spot(fu, fv, c, shape, aa)
    return out


def wave_warp(amp: float, wavelength: float, direction: str, phase: float):
    """Screen warp producing the drifting/vibrating dot grids."""
    if amp == 0:
        return None
    lam = max(4.0, float(wavelength))
    ph = 2 * math.pi * phase

    def warp(X, Y):
        if direction == "horizontal":
            return X + amp * np.sin(2 * np.pi * Y / lam + ph), Y
        if direction == "vertical":
            return X, Y + amp * np.sin(2 * np.pi * X / lam + ph)
        if direction == "ripple":
            cx, cy = X.mean(), Y.mean()
            r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            return X + amp * np.sin(2 * np.pi * r / lam + ph), Y + amp * np.cos(2 * np.pi * r / lam + ph)
        return X + amp * np.sin(2 * np.pi * Y / lam + ph), Y + amp * np.sin(2 * np.pi * X / lam + ph)

    return warp


def compose_cmyk(c, m, y, k, paper=(1.0, 1.0, 1.0)) -> np.ndarray:
    """Subtractive composite of ink coverages on paper -> float RGB."""
    pr, pg, pb = paper
    one_k = 1.0 - k
    return np.stack([(1.0 - c) * one_k * pr, (1.0 - m) * one_k * pg, (1.0 - y) * one_k * pb], axis=-1)


def _paper(p) -> tuple[float, float, float]:
    r, g, b = color_to_rgb(p)
    return r / 255.0, g / 255.0, b / 255.0


@register
class HalftoneScreen(Effect):
    id = "halftone"
    name = "Halftone Screen"
    category = "Print & Halftone"
    description = ("Converts tones into printed dots: large dots for shadows, tiny dots for highlights. "
                   "Mono, CMYK colour halftone (classic screen angles) or additive RGB dots.")
    uses_seed = False
    supports_channels = False
    params = (
        cparam("mode", "Mode", "mono", (("mono", "Mono (one ink)"), ("cmyk", "Colour halftone (CMYK)"),
                                        ("rgb", "RGB dots (additive)"))),
        iparam("cell", "Dot spacing", 9, 3, 256, spatial=True, unit="px", tooltip="Screen cell size (lines per pixel)."),
        fparam("angle", "Screen angle", 45.0, 0.0, 90.0, decimals=1, unit="°",
               tooltip="Mono: screen angle. Colour: rotates the whole CMYK angle set."),
        Param("shape", "Dot shape", "choice", "round", choices=SHAPES),
        fparam("dot_gain", "Dot gain", 1.0, 0.3, 3.0, tooltip="Above 1 = darker, fuller dots (ink spread)."),
        fparam("softness", "Edge softness", 1.0, 0.3, 4.0),
        fparam("gcr", "Black generation", 0.8, 0.0, 1.0, show_if=("mode", ("cmyk",)),
               tooltip="How much grey is printed with black ink instead of C+M+Y."),
        Param("ink", "Ink colour", COLOR, "#111111", show_if=("mode", ("mono",))),
        Param("paper", "Paper colour", COLOR, "#f4f1ea", show_if=("mode", ("mono", "cmyk"))),
    )

    def process(self, img, p, ctx):
        f = to_float(img[..., :3])
        gain = 1.0 / float(p["dot_gain"])
        cell, sh, soft = p["cell"], p["shape"], p["softness"]
        mode = p["mode"]
        if mode == "mono":
            cov = np.clip(1.0 - luminance(img), 0, 1) ** gain
            ink = screen_plate(cov, cell, p["angle"], sh, softness=soft, check=ctx.check)[..., None]
            paper = np.array(_paper(p["paper"]), np.float32)
            inkc = np.array(_paper(p["ink"]), np.float32)
            res = paper * (1.0 - ink) + inkc * ink
        elif mode == "cmyk":
            plates = rgb_to_cmyk(f, p["gcr"])
            inks = []
            for name, cov in zip("cmyk", plates):
                inks.append(screen_plate(cov ** gain, cell, CMYK_ANGLES[name] + p["angle"] - 45.0, sh, softness=soft,
                                         check=ctx.check))
            res = compose_cmyk(*inks, paper=_paper(p["paper"]))
        else:
            chans = [screen_plate(f[..., i] ** gain, cell, a + p["angle"] - 45.0, sh, softness=soft, check=ctx.check)
                     for i, a in enumerate((15.0, 45.0, 75.0))]
            res = np.stack(chans, -1)
        out = img.copy()
        out[..., :3] = to_uint8(res)
        return out


_DRIFT_PLATES = (("cm", "Cyan + Magenta"), ("c", "Cyan"), ("m", "Magenta"), ("cmy", "Cyan + Magenta + Yellow"),
                 ("cmyk", "All four plates"))


@register
class CMYKHalftoneDrift(Effect):
    id = "cmyk_drift"
    name = "CMYK Halftone Drift"
    category = "Print & Halftone"
    description = ("Separates the photo into cyan, magenta, yellow and black halftone plates, then warps the "
                   "dot grids of chosen plates with a wave. Overlapping screens create vibrating moiré, like a "
                   "misprinted comic. Plate misregistration and plate slicing break the print alignment.")
    preview_note = "Dot screens are scaled with the preview; moiré patterns differ slightly from the export."
    params = (
        iparam("cell", "Dot spacing", 8, 3, 256, spatial=True, unit="px"),
        Param("shape", "Dot shape", "choice", "round", choices=SHAPES),
        fparam("gcr", "Black generation", 0.7, 0.0, 1.0),
        fparam("angle_offset", "Angle offset", 0.0, -45.0, 45.0, decimals=1, unit="°",
               tooltip="Rotates the classic C 15° / M 75° / Y 0° / K 45° screen set."),
        Param("drift_plates", "Drifting plates", "choice", "cm", choices=_DRIFT_PLATES, group="Wave drift"),
        iparam("drift_amp", "Drift amplitude", 6, 0, 400, spatial=True, unit="px", group="Wave drift"),
        iparam("drift_wavelength", "Drift wavelength", 140, 8, 4000, spatial=True, unit="px", group="Wave drift"),
        cparam("drift_direction", "Drift direction", "horizontal",
               (("horizontal", "Horizontal waves"), ("vertical", "Vertical waves"), ("both", "Both"),
                ("ripple", "Ripple")), group="Wave drift"),
        fparam("drift_phase", "Drift phase", 0.0, 0.0, 1.0, group="Wave drift",
               tooltip="Magenta drifts with this phase, cyan with the opposite phase."),
        iparam("c_dx", "Cyan offset X", 2, -400, 400, spatial=True, unit="px", group="Misregistration"),
        iparam("c_dy", "Cyan offset Y", 0, -400, 400, spatial=True, unit="px", group="Misregistration"),
        iparam("m_dx", "Magenta offset X", -2, -400, 400, spatial=True, unit="px", group="Misregistration"),
        iparam("m_dy", "Magenta offset Y", 1, -400, 400, spatial=True, unit="px", group="Misregistration"),
        iparam("y_dx", "Yellow offset X", 0, -400, 400, spatial=True, unit="px", group="Misregistration"),
        iparam("y_dy", "Yellow offset Y", 0, -400, 400, spatial=True, unit="px", group="Misregistration"),
        iparam("k_dx", "Black offset X", 0, -400, 400, spatial=True, unit="px", group="Misregistration"),
        iparam("k_dy", "Black offset Y", 0, -400, 400, spatial=True, unit="px", group="Misregistration"),
        iparam("slice_bands", "Sliced bands", 0, 0, 80, group="Plate slicing (glitch hybrid)",
               tooltip="Seeded horizontal bands in which the sliced plates are shifted sideways."),
        iparam("slice_shift", "Slice shift", 24, 0, 2000, spatial=True, unit="px", group="Plate slicing (glitch hybrid)"),
        Param("slice_plates", "Sliced plates", "choice", "cm", choices=_DRIFT_PLATES,
              group="Plate slicing (glitch hybrid)"),
        Param("paper", "Paper colour", COLOR, "#f7f3e8"),
    )

    def process(self, img, p, ctx):
        h, w = img.shape[:2]
        f = to_float(img[..., :3])
        plates = dict(zip("cmyk", rgb_to_cmyk(f, p["gcr"])))
        # seeded slicing bands (normalised positions -> resolution independent)
        rng = ctx.rng
        n = int(p["slice_bands"])
        centers, thick, sign = rng.random(n), rng.uniform(0.01, 0.06, n), rng.uniform(-1, 1, n)
        slice_off = np.zeros(h, np.float64)
        for c0, t0, s0 in zip(centers, thick, sign):
            y0, y1 = int((c0 - t0 / 2) * h), int((c0 + t0 / 2) * h)
            slice_off[max(0, y0):max(0, min(h, y1))] = s0 * p["slice_shift"]
        inks = {}
        for name in "cmyk":
            drift = None
            if name in p["drift_plates"]:
                phase = p["drift_phase"] if name != "c" else 0.5 - p["drift_phase"]
                drift = wave_warp(p["drift_amp"], p["drift_wavelength"], p["drift_direction"], phase)
            ink = screen_plate(plates[name], p["cell"], CMYK_ANGLES[name] + p["angle_offset"], p["shape"],
                               warp=drift, check=ctx.check)
            dx, dy = p[f"{name}_dx"], p[f"{name}_dy"]
            if dx or dy:
                ink = shift_plane(ink, dx, dy, "clamp")
            if n and name in p["slice_plates"]:
                ink = shift_rows(ink, slice_off, "wrap")
            inks[name] = ink
            ctx.progress(("cmyk".index(name) + 1) / 4)
        res = compose_cmyk(inks["c"], inks["m"], inks["y"], inks["k"], paper=_paper(p["paper"]))
        out = img.copy()
        out[..., :3] = to_uint8(res)
        return out


_BENDAY_PALETTES = (("cmyk", "Process CMY + black"), ("pop", "Pop art blue / red / yellow"))
#: ink colours replacing the cyan, magenta and yellow plates
_BENDAY_INKS = {
    "cmyk": np.array([[0, 174, 239], [236, 0, 140], [255, 242, 0]], np.float32) / 255.0,
    "pop": np.array([[26, 84, 196], [222, 32, 40], [250, 214, 0]], np.float32) / 255.0,
}


@register
class BenDayDots(Effect):
    id = "benday"
    name = "Ben-Day Dots"
    category = "Print & Halftone"
    description = ("Pop-art comic printing: flat colour areas filled with uniform, evenly spaced dots in "
                   "process or pop colours, solid blacks and inked outlines.")
    uses_seed = False
    supports_channels = False
    params = (
        iparam("spacing", "Dot spacing", 10, 3, 256, spatial=True, unit="px"),
        fparam("dot_size", "Dot size", 0.7, 0.15, 1.0, tooltip="Dot diameter as a fraction of the spacing."),
        fparam("angle", "Grid angle", 45.0, 0.0, 90.0, decimals=1, unit="°"),
        cparam("palette", "Palette", "pop", _BENDAY_PALETTES),
        fparam("threshold", "Colour threshold", 0.35, 0.05, 0.95, tooltip="CMY amount needed before an area gets dots."),
        fparam("black_level", "Solid black below", 0.18, 0.0, 0.6, tooltip="Luminance under which areas print solid black."),
        iparam("flatten", "Flatten", 5, 0, 60, spatial=True, unit="px", tooltip="Smooths the photo into flat areas first."),
        fparam("outline", "Outline", 0.8, 0.0, 1.0),
        iparam("outline_px", "Outline width", 2, 1, 40, spatial=True, unit="px"),
        Param("paper", "Paper colour", COLOR, "#f6efdc"),
    )

    def process(self, img, p, ctx):
        h, w = img.shape[:2]
        rgb = np.ascontiguousarray(img[..., :3])
        k = int(p["flatten"])
        flat = rgb
        if k >= 3:
            flat = cv2.medianBlur(rgb, 5 if k >= 5 else 3)
        if k > 5:
            flat = cv2.bilateralFilter(flat, d=-1, sigmaColor=40, sigmaSpace=k * 0.6)
        f = to_float(flat)
        lum = luminance(flat)
        paper = np.array(_paper(p["paper"]), np.float32)
        dot_cov = np.float32(min(0.95, math.pi * (p["dot_size"] / 2.0) ** 2))
        const = np.full((h, w), dot_cov, np.float32)
        black = lum < p["black_level"]
        # Three uniform dot plates; an area gets a plate's dots when its CMY
        # separation needs that ink, so secondary colours appear as two
        # overlapping dot screens (green = blue dots + yellow dots).
        c, m, y, _k = rgb_to_cmyk(f, 1.0)
        inks = _BENDAY_INKS[p["palette"]]
        res = np.broadcast_to(paper, (h, w, 3)).copy()
        for (plate, angle), ink in zip(((c, 15.0), (m, 75.0), (y, 0.0)), inks):
            present = (plate > p["threshold"]).astype(np.float32)
            dots = screen_plate(const, p["spacing"], angle - 45.0 + p["angle"], "round", check=ctx.check)
            cover = (dots * present)[..., None]
            res *= 1.0 - cover * (1.0 - ink)
        res = np.where(black[..., None], np.float32(0.07), res)
        if p["outline"] > 0:
            g = to_uint8(lum)
            edges = cv2.Canny(g, 40, 110)
            r = max(1, int(p["outline_px"]))
            edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r)))
            e = (edges.astype(np.float32) / 255.0 * p["outline"])[..., None]
            res = res * (1 - e) + np.float32(0.05) * e
        out = img.copy()
        out[..., :3] = to_uint8(res)
        return out

