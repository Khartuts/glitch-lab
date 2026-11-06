"""Geometric displacement effects: channel displacement, scanline tearing,
row/column drift, displacement maps and channel cross-modulation."""

from __future__ import annotations

import cv2
import numpy as np

from ..ops import (
    edge_index,
    luminance,
    rgb_to_hsv,
    rgb_to_ycbcr,
    shift_cols,
    shift_plane,
    shift_rows,
    to_float,
    to_uint8,
    value_noise,
    ycbcr_to_rgb,
)
from ..errors import EffectError
from ..params import EDGE_MODES, FILE, Param, bparam, cparam, fparam, iparam
from .base import Effect, register


@register
class ChannelShift(Effect):
    id = "channel_shift"
    name = "RGB Channel Displacement"
    category = "Displacement"
    description = ("Offsets each colour channel independently. In YCbCr space the luma and the two "
                   "chroma channels are shifted instead, giving softer colour fringing.")
    supports_channels = False
    supports_alpha = True
    uses_seed = False
    params = (
        cparam("space", "Channel space", "rgb", (("rgb", "RGB"), ("ycbcr", "YCbCr (Y, Cb, Cr)")),
               tooltip="Which channel triplet to displace."),
        iparam("r_dx", "Red / Y  X", 12, -1000, 1000, spatial=True, unit="px", group="Channel 1"),
        iparam("r_dy", "Red / Y  Y", 0, -1000, 1000, spatial=True, unit="px", group="Channel 1"),
        iparam("g_dx", "Green / Cb  X", 0, -1000, 1000, spatial=True, unit="px", group="Channel 2"),
        iparam("g_dy", "Green / Cb  Y", 0, -1000, 1000, spatial=True, unit="px", group="Channel 2"),
        iparam("b_dx", "Blue / Cr  X", -12, -1000, 1000, spatial=True, unit="px", group="Channel 3"),
        iparam("b_dy", "Blue / Cr  Y", 4, -1000, 1000, spatial=True, unit="px", group="Channel 3"),
        iparam("a_dx", "Alpha X", 0, -1000, 1000, spatial=True, unit="px", group="Alpha",
               tooltip="Only used when 'Affect alpha' is enabled."),
        iparam("a_dy", "Alpha Y", 0, -1000, 1000, spatial=True, unit="px", group="Alpha",
               tooltip="Only used when 'Affect alpha' is enabled."),
        Param("edge", "Edges", "choice", "wrap", choices=EDGE_MODES,
              tooltip="What appears where a channel is pushed past the border."),
    )

    def process(self, img, p, ctx):
        mode = p["edge"]
        offs = [(p["r_dx"], p["r_dy"]), (p["g_dx"], p["g_dy"]), (p["b_dx"], p["b_dy"])]
        out = np.empty_like(img)
        if p["space"] == "ycbcr":
            ycc = rgb_to_ycbcr(img)
            shifted = np.stack([shift_plane(ycc[..., i], dx, dy, mode) for i, (dx, dy) in enumerate(offs)], -1)
            out[..., :3] = ycbcr_to_rgb(shifted)
        else:
            for i, (dx, dy) in enumerate(offs):
                out[..., i] = shift_plane(img[..., i], dx, dy, mode)
        out[..., 3] = shift_plane(img[..., 3], p["a_dx"], p["a_dy"], mode)
        return out


@register
class ScanlineTear(Effect):
    id = "scanline_tear"
    name = "Scanline Tearing"
    category = "Displacement"
    description = "Seeded horizontal bands are torn sideways by different amounts, with per-row jitter."
    supports_alpha = True
    params = (
        iparam("bands", "Band count", 14, 1, 300, tooltip="Number of torn bands."),
        iparam("min_thickness", "Min thickness", 3, 1, 1000, spatial=True, unit="px"),
        iparam("max_thickness", "Max thickness", 48, 1, 2000, spatial=True, unit="px"),
        iparam("displacement", "Displacement", 90, 0, 4000, spatial=True, unit="px",
               tooltip="Maximum sideways shift of a band."),
        iparam("jitter", "Row jitter", 3, 0, 400, spatial=True, unit="px",
               tooltip="Extra random shift for every row inside a band."),
        cparam("direction", "Direction", "both", (("both", "Both ways"), ("left", "Left only"), ("right", "Right only"))),
        iparam("rgb_split", "Colour split", 0, 0, 400, spatial=True, unit="px",
               tooltip="Push red and blue apart inside torn bands."),
        Param("edge", "Edges", "choice", "wrap", choices=EDGE_MODES),
    )

    def process(self, img, p, ctx):
        h, w = img.shape[:2]
        rng = ctx.rng
        n = p["bands"]
        tmin, tmax = sorted((p["min_thickness"], p["max_thickness"]))
        centers = rng.random(n)
        thick_u = rng.random(n)
        if p["direction"] == "left":
            amount = -rng.random(n)
        elif p["direction"] == "right":
            amount = rng.random(n)
        else:
            amount = rng.uniform(-1.0, 1.0, n)
        full_h = max(1, int(ctx.full_size[1]))
        jit_full = rng.normal(0.0, 1.0, full_h)
        offsets = np.zeros(h, np.float64)
        in_band = np.zeros(h, bool)
        for c, tu, a in zip(centers, thick_u, amount):
            t = max(1, int(round(tmin + tu * (tmax - tmin))))
            y0 = int(round(c * h - t / 2))
            y0c, y1c = max(0, y0), min(h, y0 + t)
            if y1c <= y0c:
                continue
            offsets[y0c:y1c] = a * p["displacement"]
            in_band[y0c:y1c] = True
        if p["jitter"] > 0:
            rows = np.minimum(full_h - 1, (np.arange(h) / max(ctx.scale, 1e-9)).astype(np.int64))
            offsets = offsets + np.where(in_band, jit_full[rows] * p["jitter"], 0.0)
        ctx.check()
        out = shift_rows(img, offsets, p["edge"])
        if p["rgb_split"] > 0:
            split = np.where(in_band, float(p["rgb_split"]), 0.0)
            out[..., 0] = shift_rows(img[..., 0], offsets + split, p["edge"])
            out[..., 2] = shift_rows(img[..., 2], offsets - split, p["edge"])
        return out


# ---------------------------------------------------------------------------
# Row / column drift
# ---------------------------------------------------------------------------

_WAVES = (("sine", "Sine"), ("saw", "Sawtooth"), ("triangle", "Triangle"), ("square", "Square"),
          ("steps", "Random steps"), ("noise", "Smooth noise"))


def wave_profile(t: np.ndarray, waveform: str, frequency: float, phase: float,
                 rng_values: np.ndarray | None = None) -> np.ndarray:
    """Waveform in [-1, 1] sampled at normalised positions ``t`` (0..1)."""
    x = t * frequency + phase
    if waveform == "sine":
        return np.sin(2 * np.pi * x)
    if waveform == "saw":
        return 2.0 * (x - np.floor(x + 0.5))
    if waveform == "triangle":
        return 2.0 * np.abs(2.0 * (x - np.floor(x + 0.5))) - 1.0
    if waveform == "square":
        return np.where((x - np.floor(x)) < 0.5, 1.0, -1.0)
    # lattice-based: values[k] at integer lattice points of x
    vals = rng_values
    n = len(vals)
    xi = np.floor(x).astype(np.int64)
    if waveform == "steps":
        return vals[np.mod(xi, n)]
    f = x - np.floor(x)
    f = f * f * (3 - 2 * f)
    return vals[np.mod(xi, n)] * (1 - f) + vals[np.mod(xi + 1, n)] * f


@register
class RowDrift(Effect):
    id = "row_drift"
    name = "Row & Column Drift"
    category = "Displacement"
    description = "Shifts every row (or column) by a sine, sawtooth, triangle, square, stepped or noise waveform."
    supports_alpha = True
    params = (
        cparam("axis", "Shift", "rows", (("rows", "Rows sideways"), ("columns", "Columns up/down"))),
        Param("waveform", "Waveform", "choice", "sine", choices=_WAVES),
        iparam("amplitude", "Amplitude", 24, 0, 3000, spatial=True, unit="px"),
        fparam("frequency", "Frequency", 3.0, 0.1, 200.0, decimals=2, tooltip="Cycles (or steps) across the image."),
        fparam("phase", "Phase", 0.0, 0.0, 1.0),
        fparam("chroma_phase", "Chroma phase split", 0.0, 0.0, 0.5,
               tooltip="Offsets the red and blue waves in opposite directions for colour fringing."),
        Param("edge", "Edges", "choice", "wrap", choices=EDGE_MODES),
    )

    def process(self, img, p, ctx):
        h, w = img.shape[:2]
        n_lines = h if p["axis"] == "rows" else w
        lattice = ctx.rng.uniform(-1.0, 1.0, int(np.ceil(p["frequency"])) + 3)
        t = (np.arange(n_lines) + 0.5) / n_lines
        amp = float(p["amplitude"])

        def offsets(phase_shift: float) -> np.ndarray:
            return wave_profile(t, p["waveform"], p["frequency"], p["phase"] + phase_shift, lattice) * amp

        shift = shift_rows if p["axis"] == "rows" else shift_cols
        base = offsets(0.0)
        out = shift(img, base, p["edge"])
        if p["chroma_phase"] > 0:
            out[..., 0] = shift(img[..., 0], offsets(p["chroma_phase"]), p["edge"])
            out[..., 2] = shift(img[..., 2], offsets(-p["chroma_phase"]), p["edge"])
        return out


# ---------------------------------------------------------------------------
# Displacement maps
# ---------------------------------------------------------------------------

_BORDER = {"wrap": cv2.BORDER_WRAP, "clamp": cv2.BORDER_REPLICATE, "mirror": cv2.BORDER_REFLECT}


@register
class DisplacementMap(Effect):
    id = "displace_map"
    name = "Displacement Map"
    category = "Displacement"
    description = ("Warps the image with a map taken from its own luminance, its edges, seeded fractal noise "
                   "or a second imported image (red = X, green = Y).")
    supports_alpha = True
    params = (
        cparam("source", "Map source", "noise",
               (("luminance", "Own luminance"), ("edges", "Own edges (gradient)"), ("noise", "Procedural noise"),
                ("image", "Imported image"))),
        Param("map_file", "Map image", FILE, "", file_filter="Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)",
              show_if=("source", ("image",)), randomizable=False,
              tooltip="Red channel displaces horizontally, green vertically (stretched to fit)."),
        iparam("amount_x", "Amount X", 30, -1000, 1000, spatial=True, unit="px"),
        iparam("amount_y", "Amount Y", 30, -1000, 1000, spatial=True, unit="px"),
        fparam("noise_scale", "Noise scale", 6.0, 1.0, 64.0, decimals=1, show_if=("source", ("noise",)),
               tooltip="Noise cells across the long side."),
        iparam("smooth", "Map blur", 2, 0, 200, spatial=True, unit="px"),
        Param("edge", "Edges", "choice", "mirror", choices=EDGE_MODES),
    )

    def validate(self, p):
        if p["source"] == "image" and not p["map_file"]:
            return ["choose a map image (or another map source)."]
        return []

    def process(self, img, p, ctx):
        h, w = img.shape[:2]
        src = p["source"]
        if src == "noise":
            mx = value_noise(h, w, ctx.sub_rng(1), p["noise_scale"]) * 2 - 1
            my = value_noise(h, w, ctx.sub_rng(2), p["noise_scale"]) * 2 - 1
        elif src == "image":
            aux = ctx.services.load_aux_image(p["map_file"]) if ctx.services else None
            if aux is None:
                raise EffectError("the map image service is unavailable")
            aux = cv2.resize(np.ascontiguousarray(aux[..., :3]), (w, h), interpolation=cv2.INTER_AREA)
            mx = aux[..., 0].astype(np.float32) / 127.5 - 1
            my = aux[..., 1].astype(np.float32) / 127.5 - 1
        else:
            lum = luminance(img)
            if p["smooth"] > 0:
                lum = cv2.GaussianBlur(lum, (0, 0), max(0.5, p["smooth"]))
            if src == "luminance":
                mx = my = lum * 2 - 1
            else:
                gx = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3)
                m = float(max(np.abs(gx).max(), np.abs(gy).max(), 1e-6))
                mx, my = gx / m, gy / m
        if p["smooth"] > 0 and src in ("noise", "image"):
            mx = cv2.GaussianBlur(mx.astype(np.float32), (0, 0), max(0.5, p["smooth"]))
            my = cv2.GaussianBlur(my.astype(np.float32), (0, 0), max(0.5, p["smooth"]))
        ctx.check()
        xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        map_x = (xs + mx.astype(np.float32) * float(p["amount_x"])).astype(np.float32)
        map_y = (ys + my.astype(np.float32) * float(p["amount_y"])).astype(np.float32)
        return cv2.remap(np.ascontiguousarray(img), map_x, map_y, interpolation=cv2.INTER_LINEAR,
                         borderMode=_BORDER[p["edge"]])


# ---------------------------------------------------------------------------
# Channel cross-modulation
# ---------------------------------------------------------------------------

_SOURCES = (("r", "Red"), ("g", "Green"), ("b", "Blue"), ("luma", "Luminance"), ("hue", "Hue"),
            ("saturation", "Saturation"))


@register
class CrossModulation(Effect):
    id = "cross_mod"
    name = "Channel Cross-Modulation"
    category = "Colour & Bits"
    description = "Uses one channel to displace, threshold, remap or ring-modulate another channel."
    supports_channels = False
    uses_seed = False
    params = (
        Param("source", "Source channel", "choice", "luma", choices=_SOURCES),
        cparam("destination", "Destination", "r", (("r", "Red"), ("g", "Green"), ("b", "Blue"))),
        cparam("mode", "Modulation", "displace_x",
               (("displace_x", "Displace horizontally"), ("displace_y", "Displace vertically"),
                ("threshold", "Threshold invert"), ("remap", "Remap (wrap-around)"), ("ring", "Ring modulate"))),
        fparam("amount", "Amount", 0.6, 0.0, 1.0),
        iparam("distance", "Displace distance", 40, 0, 2000, spatial=True, unit="px",
               show_if=("mode", ("displace_x", "displace_y"))),
        fparam("threshold", "Threshold", 0.5, 0.0, 1.0, show_if=("mode", ("threshold",))),
        fparam("frequency", "Frequency", 2.0, 0.25, 32.0, show_if=("mode", ("remap", "ring"))),
    )

    def process(self, img, p, ctx):
        h, w = img.shape[:2]
        sname = p["source"]
        if sname in ("r", "g", "b"):
            s = img[..., "rgb".index(sname)].astype(np.float32) / 255.0
        elif sname == "luma":
            s = luminance(img)
        else:
            hh, ss, _ = rgb_to_hsv(img)
            s = hh if sname == "hue" else ss
        di = "rgb".index(p["destination"])
        d = img[..., di]
        a = float(p["amount"])
        mode = p["mode"]
        if mode in ("displace_x", "displace_y"):
            off = np.rint((s - 0.5) * 2.0 * a * p["distance"]).astype(np.int64)
            if mode == "displace_x":
                xs = edge_index(np.arange(w)[None, :] + off, w, "wrap")
                new = np.take_along_axis(d, xs, axis=1)
            else:
                ys = edge_index(np.arange(h)[:, None] + off, h, "wrap")
                new = np.take_along_axis(d, ys, axis=0)
        else:
            df = d.astype(np.float32)
            if mode == "threshold":
                res = np.where(s > p["threshold"], 255.0 - df, df)
            elif mode == "remap":
                res = np.mod(df + s * 255.0 * p["frequency"], 256.0)
            else:
                res = df * (0.5 + 0.5 * np.sin(2 * np.pi * p["frequency"] * s))
            new = to_uint8(df + (res - df) * a, scale=1.0)
        out = img.copy()
        out[..., di] = new
        return out
