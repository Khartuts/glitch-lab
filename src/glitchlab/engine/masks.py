"""Rasterisation of resolution-independent masks (rectangles, brush strokes,
luminance range, feather, invert)."""

from __future__ import annotations

import cv2
import numpy as np

from .model import MaskSpec
from .ops import luminance, smoothstep


def rasterize_mask(spec: MaskSpec, width: int, height: int, source: np.ndarray | None = None) -> np.ndarray | None:
    """Return a float32 (H, W) mask in [0, 1], or ``None`` for "everything".

    ``source`` (uint8 RGBA at the same resolution) is required for the
    luminance-range component, which is measured on the *original* image so
    that it does not change as effects are adjusted.
    """
    if spec.is_identity():
        return None
    long_side = float(max(width, height))
    if spec.shapes:
        # Draw at 8x sub-pixel precision for smooth anti-aliased edges.
        base = np.zeros((height, width), np.float32)
        shift = 3
        mul = float(1 << shift)
        for shp in spec.shapes:
            layer = np.zeros((height, width), np.uint8)
            if shp["type"] == "rect":
                x0, x1 = sorted((shp["x0"], shp["x1"]))
                y0, y1 = sorted((shp["y0"], shp["y1"]))
                px0 = int(round(x0 * width))
                px1 = int(round(x1 * width))
                py0 = int(round(y0 * height))
                py1 = int(round(y1 * height))
                layer[max(0, py0):max(0, py1), max(0, px0):max(0, px1)] = 255
            elif shp["type"] == "stroke":
                radius = max(0.5, shp["radius"] * long_side)
                thick = max(1, int(round(radius * 2)))
                pts = np.array([[p[0] * width * mul, p[1] * height * mul] for p in shp["points"]], np.float64)
                pts = np.rint(pts).astype(np.int32)
                if len(pts) == 1:
                    cv2.circle(layer, tuple(int(v) for v in pts[0]), int(round(radius * mul)), 255, -1,
                               lineType=cv2.LINE_AA, shift=shift)
                else:
                    cv2.polylines(layer, [pts.reshape(-1, 1, 2)], False, 255, thickness=thick,
                                  lineType=cv2.LINE_AA, shift=shift)
                    for p in (pts[0], pts[-1]):
                        cv2.circle(layer, (int(p[0]), int(p[1])), int(round(radius * mul)), 255, -1,
                                   lineType=cv2.LINE_AA, shift=shift)
            lf = layer.astype(np.float32) / 255.0
            if shp["mode"] == "subtract":
                base = np.minimum(base, 1.0 - lf)
            else:
                base = np.maximum(base, lf)
    else:
        base = np.ones((height, width), np.float32)
    if spec.feather > 0 and spec.shapes:
        sigma = spec.feather * long_side
        if sigma >= 0.3:
            base = cv2.GaussianBlur(base, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    if spec.luma_enabled:
        if source is None:
            raise ValueError("A luminance mask needs the source image")
        lum = luminance(source)
        lo, hi = sorted((spec.luma_low, spec.luma_high))
        soft = max(spec.luma_softness, 1e-4)
        lm = smoothstep(lo - soft, lo, lum) * (1.0 - smoothstep(hi, hi + soft, lum))
        base = base * lm
    if spec.invert:
        base = 1.0 - base
    return np.clip(base, 0.0, 1.0).astype(np.float32)
