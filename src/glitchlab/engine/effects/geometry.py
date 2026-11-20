"""Symmetry effects: kaleidoscope (radial wedges, hexagonal and square mirror
tiles) and mirror symmetry.

All modes build a sampling map with NumPy and resample the image once with
``cv2.remap``.  Positions are normalised to the image, tile sizes are in
full-resolution pixels, so the reduced preview matches the export.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ..params import Param, bparam, cparam, fparam, iparam
from .base import Effect, register

_BORDERS = {"mirror": cv2.BORDER_REFLECT_101, "wrap": cv2.BORDER_WRAP, "clamp": cv2.BORDER_REPLICATE}


def fold_wedge(dx: np.ndarray, dy: np.ndarray, segments: int, mirror: bool) -> tuple[np.ndarray, np.ndarray]:
    """Fold offsets (dx, dy) from a centre into the first wedge of ``segments``.

    Returns the folded (radius, angle) with angle in [0, 2*pi/segments)."""
    r = np.sqrt(dx * dx + dy * dy)
    th = np.arctan2(dy, dx)
    alpha = 2.0 * np.pi / segments
    k = np.floor(th / alpha)
    t = th - k * alpha
    if mirror:
        odd = (k.astype(np.int64) % 2) == 1
        t = np.where(odd, alpha - t, t)
    return r, t


def hex_local(x: np.ndarray, y: np.ndarray, size: float) -> tuple[np.ndarray, np.ndarray]:
    """Offsets of every point from the centre of its pointy-top hexagon."""
    q = (math.sqrt(3) / 3 * x - y / 3) / size
    r = (2.0 / 3 * y) / size
    # cube rounding
    cx, cz = q, r
    cy = -cx - cz
    rx, ry, rz = np.rint(cx), np.rint(cy), np.rint(cz)
    dx_, dy_, dz_ = np.abs(rx - cx), np.abs(ry - cy), np.abs(rz - cz)
    fix_x = (dx_ > dy_) & (dx_ > dz_)
    fix_y = ~fix_x & (dy_ > dz_)
    rx = np.where(fix_x, -ry - rz, rx)
    ry = np.where(fix_y, -rx - rz, ry)
    rz = np.where(~fix_x & ~fix_y, -rx - ry, rz)
    hx = size * math.sqrt(3) * (rx + rz / 2.0)
    hy = size * 1.5 * rz
    return x - hx, y - hy


@register
class Kaleidoscope(Effect):
    id = "kaleidoscope"
    name = "Kaleidoscope"
    category = "Abstract & Style"
    description = ("Mirror symmetry like a real kaleidoscope: radial wedges (any number of segments), repeating "
                   "hexagonal or square mirror tiles, or simple left/right and top/bottom mirroring. Choose which "
                   "part of the photo feeds the mirrors with the source position and angle.")
    supports_alpha = True
    uses_seed = False
    params = (
        cparam("mode", "Mode", "wedge", (("wedge", "Radial wedges"), ("hex", "Hexagon tiles"),
                                         ("square", "Square mirror tiles"), ("mirror", "Mirror halves"))),
        iparam("segments", "Segments", 8, 2, 48, show_if=("mode", ("wedge",)),
               tooltip="Number of wedges around the centre (mirrored pairs look like a real kaleidoscope)."),
        bparam("mirrored", "Mirror alternate wedges", True, show_if=("mode", ("wedge",))),
        cparam("axis", "Mirror", "vertical", (("vertical", "Left ↔ right"), ("horizontal", "Top ↔ bottom"),
                                              ("both", "Four quadrants")), show_if=("mode", ("mirror",))),
        bparam("flip_side", "Use other side", False, show_if=("mode", ("mirror",))),
        iparam("tile", "Tile size", 160, 8, 4000, spatial=True, unit="px", show_if=("mode", ("hex", "square"))),
        fparam("cx", "Centre X", 0.5, 0.0, 1.0),
        fparam("cy", "Centre Y", 0.5, 0.0, 1.0),
        fparam("rotation", "Rotation", 0.0, -180.0, 180.0, decimals=1, unit="°"),
        fparam("zoom", "Zoom", 1.0, 0.1, 8.0, tooltip="Magnifies the sampled part of the photo."),
        fparam("source_angle", "Source angle", 0.0, -180.0, 180.0, decimals=1, unit="°",
               tooltip="Which direction of the photo the wedge looks at."),
        fparam("source_x", "Source shift X", 0.0, -0.5, 0.5, tooltip="Moves the sampled area (fraction of the width)."),
        fparam("source_y", "Source shift Y", 0.0, -0.5, 0.5),
        Param("edge", "Outside the photo", "choice", "mirror", choices=(("mirror", "Mirror"), ("wrap", "Wrap"),
                                                                       ("clamp", "Clamp"))),
    )

    def process(self, img, p, ctx):
        h, w = img.shape[:2]
        cx, cy = p["cx"] * w, p["cy"] * h
        sx, sy = cx + p["source_x"] * w, cy + p["source_y"] * h
        xs, ys = np.meshgrid(np.arange(w, dtype=np.float32) - cx, np.arange(h, dtype=np.float32) - cy)
        rot = math.radians(p["rotation"])
        if rot:
            c, s = math.cos(rot), math.sin(rot)
            xs, ys = xs * c + ys * s, -xs * s + ys * c
        mode = p["mode"]
        zoom = float(p["zoom"])
        src_rot = math.radians(p["source_angle"])
        if mode == "wedge":
            r, t = fold_wedge(xs, ys, int(p["segments"]), p["mirrored"])
            phi = t + src_rot
            mx = sx + (r / zoom) * np.cos(phi)
            my = sy + (r / zoom) * np.sin(phi)
        elif mode == "hex":
            lx, ly = hex_local(xs, ys, max(4.0, float(p["tile"]) / 2.0))
            r, t = fold_wedge(lx, ly, 6, True)
            phi = t + src_rot
            mx = sx + (r / zoom) * np.cos(phi)
            my = sy + (r / zoom) * np.sin(phi)
        elif mode == "square":
            tile = max(4.0, float(p["tile"]))
            u = np.mod(xs / tile, 2.0)
            v = np.mod(ys / tile, 2.0)
            u = np.where(u > 1.0, 2.0 - u, u) * tile
            v = np.where(v > 1.0, 2.0 - v, v) * tile
            c, s = math.cos(src_rot), math.sin(src_rot)
            mx = sx + (u * c - v * s) / zoom
            my = sy + (u * s + v * c) / zoom
        else:  # mirror halves
            ax = p["axis"]
            fx = -1.0 if p["flip_side"] else 1.0
            mxx = np.abs(xs) * fx if ax in ("vertical", "both") else xs
            myy = np.abs(ys) * fx if ax in ("horizontal", "both") else ys
            mx = sx + mxx / zoom
            my = sy + myy / zoom
        return cv2.remap(np.ascontiguousarray(img), mx.astype(np.float32), my.astype(np.float32),
                         interpolation=cv2.INTER_LINEAR, borderMode=_BORDERS[p["edge"]])
