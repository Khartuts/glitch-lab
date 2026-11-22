"""Helpers shared by the AI effects (pure NumPy/OpenCV, no PyTorch)."""

from __future__ import annotations

import math

import cv2
import numpy as np

from ..params import color_to_rgb

GAP_FILLS = (("transparent", "Transparent"), ("stretch", "Stretched edge pixels"), ("colour", "Colour"))


def smooth_noise(rng: np.random.Generator, w: int, h: int, full_size: tuple[int, int], cell_full: float) -> np.ndarray:
    """Smooth noise in [-1, 1] whose pattern depends only on the full-resolution
    size and cell, so the reduced preview and the export look alike."""
    fw, fh = full_size
    cell = max(1.0, float(cell_full))
    gw = min(512, int(math.ceil(fw / cell)) + 2)
    gh = min(512, int(math.ceil(fh / cell)) + 2)
    g = rng.standard_normal((gh, gw)).astype(np.float32)
    n = cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(n * 0.6, -1.0, 1.0)


def nearest_fill(img: np.ndarray, holes: np.ndarray) -> np.ndarray:
    """Replace hole pixels by the nearest non-hole pixel ("stretched edge pixels")."""
    if not holes.any():
        return img
    if holes.all():
        return img
    src = (~holes).astype(np.uint8)
    # distanceTransform measures distance to the nearest ZERO pixel: make the valid pixels zero.
    _, labels = cv2.distanceTransformWithLabels(1 - src, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    valid_idx = np.flatnonzero(src.ravel())
    lut = np.concatenate([[0], valid_idx])
    nearest = lut[labels.ravel()]
    out = img.reshape(-1, img.shape[-1]).copy() if img.ndim == 3 else img.ravel().copy()
    flat_src = img.reshape(-1, img.shape[-1]) if img.ndim == 3 else img.ravel()
    hidx = np.flatnonzero(holes.ravel())
    out[hidx] = flat_src[nearest[hidx]]
    return out.reshape(img.shape)


def apply_gap_fill(out: np.ndarray, holes: np.ndarray, mode: str, colour: str, ctx, *,
                   coverage: np.ndarray | None = None) -> np.ndarray:
    """Fill ``holes`` of a uint8/float RGBA image with transparency, nearest pixels or a colour.

    Transparency needs "Affect alpha"; without it the colour is used and the
    user is told why."""
    if not holes.any():
        return out
    if mode == "transparent" and not ctx.affect_alpha:
        ctx.warn("Transparent gaps need 'Affect alpha' to be switched on; the gap colour is shown instead.")
        mode = "colour"
    if mode == "stretch":
        return nearest_fill(out, holes)
    res = out.copy()
    if mode == "transparent":
        res[holes, 3] = 0
        return res
    res[holes, :3] = color_to_rgb(colour)
    return res


def hue_matrix(degrees: float, saturation: float = 1.0) -> np.ndarray:
    """3x3 RGB matrix rotating hue (YIQ) and scaling saturation."""
    a = math.radians(degrees)
    c, s = math.cos(a) * saturation, math.sin(a) * saturation
    to_yiq = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]], np.float32)
    rot = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float32)
    return np.linalg.inv(to_yiq).astype(np.float32) @ rot @ to_yiq


def overlay_map(img: np.ndarray, m: np.ndarray, colour=(255, 40, 160), strength: float = 0.65) -> np.ndarray:
    """Tint ``img`` by a 0..1 map (for showing analysis maps on the canvas)."""
    out = img.copy()
    a = (np.clip(m, 0, 1) * strength)[..., None]
    out[..., :3] = np.clip(img[..., :3] * (1 - a) + np.asarray(colour, np.float32) * a, 0, 255).astype(np.uint8)
    return out


def heat(m: np.ndarray) -> np.ndarray:
    """0..1 map -> RGB heat colours (uint8)."""
    u8 = (np.clip(m, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


def label_colours(n: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cols = rng.integers(40, 255, (max(n, 1), 3)).astype(np.uint8)
    return cols
