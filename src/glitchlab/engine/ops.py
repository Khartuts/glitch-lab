"""Vectorised pixel helpers shared by effects.

Conventions
-----------
* The engine's inter-effect image format is ``uint8`` RGBA, shape (H, W, 4),
  in the sRGB-encoded working space.  All conversions to/from float are
  explicit and go through :func:`to_float` / :func:`to_uint8`, which guard
  against NaN, infinity and overflow.
* Luminance uses Rec. 601 luma weights on sRGB-encoded values (the same
  weights JPEG uses), which is what most glitch tools sort by.
"""

from __future__ import annotations

import numpy as np

LUMA_R, LUMA_G, LUMA_B = 0.299, 0.587, 0.114

CHANNEL_CHOICES = (
    ("rgb", "RGB (all)"),
    ("r", "Red"),
    ("g", "Green"),
    ("b", "Blue"),
    ("luma", "Luminance"),
)


def to_float(img: np.ndarray) -> np.ndarray:
    """uint8 -> float32 in [0, 1]."""
    return img.astype(np.float32) * np.float32(1.0 / 255.0)


def to_uint8(arr: np.ndarray, *, scale: float = 255.0) -> np.ndarray:
    """float -> uint8 with rounding, clipping and NaN/inf guards."""
    a = np.asarray(arr, dtype=np.float32)
    if scale != 1.0:
        a = a * np.float32(scale)
    a = np.nan_to_num(a, nan=0.0, posinf=255.0, neginf=0.0)
    np.clip(a, 0.0, 255.0, out=a)
    return np.rint(a).astype(np.uint8)


def clip_u8(arr: np.ndarray) -> np.ndarray:
    """Integer/float array -> uint8 with saturation (no wrap-around)."""
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        return to_uint8(arr, scale=1.0)
    return np.clip(arr, 0, 255).astype(np.uint8)


def ensure_rgba(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        a = np.full(img.shape, 255, np.uint8)
        return np.dstack([img, img, img, a])
    if img.shape[2] == 3:
        a = np.full(img.shape[:2], 255, np.uint8)
        return np.dstack([img, a])
    if img.shape[2] == 4:
        return img
    raise ValueError(f"Unsupported channel count {img.shape[2]}")


def luminance(img: np.ndarray) -> np.ndarray:
    """Rec. 601 luma of an RGB(A) uint8 image as float32 in [0, 1]."""
    rgb = img[..., :3].astype(np.float32)
    y = rgb[..., 0] * LUMA_R + rgb[..., 1] * LUMA_G + rgb[..., 2] * LUMA_B
    return y * np.float32(1.0 / 255.0)


def luminance_u8(img: np.ndarray) -> np.ndarray:
    return to_uint8(luminance(img))


def rgb_to_hsv(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised RGB -> (hue, saturation, value), each float32 in [0, 1]."""
    rgb = to_float(img[..., :3])
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    d = mx - mn
    safe = np.where(d > 0, d, 1.0).astype(np.float32)
    h = np.zeros_like(mx)
    rc = (mx == r) & (d > 0)
    gc = (mx == g) & (d > 0) & ~rc
    bc = (d > 0) & ~rc & ~gc
    h = np.where(rc, ((g - b) / safe) % 6.0, h)
    h = np.where(gc, (b - r) / safe + 2.0, h)
    h = np.where(bc, (r - g) / safe + 4.0, h)
    h = (h / 6.0).astype(np.float32)
    s = np.where(mx > 0, d / np.where(mx > 0, mx, 1.0), 0.0).astype(np.float32)
    return h, s, mx.astype(np.float32)


def hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectorised HSV (floats in [0,1]) -> float32 RGB (H, W, 3) in [0,1]."""
    h = (np.asarray(h, np.float32) % 1.0) * 6.0
    s = np.clip(np.asarray(s, np.float32), 0, 1)
    v = np.clip(np.asarray(v, np.float32), 0, 1)
    i = np.floor(h).astype(np.int32) % 6
    f = h - np.floor(h)
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1).astype(np.float32)


# YCbCr (BT.601 full range, as used by JFIF/JPEG) -----------------------------

def rgb_to_ycbcr(img: np.ndarray) -> np.ndarray:
    """uint8 RGB(A) -> float32 YCbCr (H, W, 3) with values in 0..255."""
    rgb = img[..., :3].astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return np.stack([y, cb, cr], axis=-1)


def ycbcr_to_rgb(ycc: np.ndarray) -> np.ndarray:
    """float YCbCr (0..255) -> uint8 RGB."""
    y = ycc[..., 0].astype(np.float32)
    cb = ycc[..., 1].astype(np.float32) - 128.0
    cr = ycc[..., 2].astype(np.float32) - 128.0
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return to_uint8(np.stack([r, g, b], axis=-1), scale=1.0)


# Edge handling --------------------------------------------------------------

def edge_index(idx: np.ndarray, n: int, mode: str) -> np.ndarray:
    """Map possibly out-of-range integer indices into [0, n) using an edge mode.

    ``wrap`` repeats the image, ``clamp`` repeats the border pixel and
    ``mirror`` reflects (symmetric, the edge pixel is repeated once).
    """
    idx = np.asarray(idx, dtype=np.int64)
    if n <= 1:
        return np.zeros_like(idx)
    if mode == "wrap":
        return np.mod(idx, n)
    if mode == "clamp":
        return np.clip(idx, 0, n - 1)
    if mode == "mirror":
        period = 2 * n
        m = np.mod(idx, period)
        return np.where(m >= n, period - 1 - m, m)
    raise ValueError(f"Unknown edge mode {mode!r}")


def shift_plane(plane: np.ndarray, dx: int, dy: int, mode: str) -> np.ndarray:
    """Translate a 2-D (or 3-D, channel-last) array by integer (dx, dy)."""
    h, w = plane.shape[:2]
    if dx == 0 and dy == 0:
        return plane.copy()
    ys = edge_index(np.arange(h) - int(dy), h, mode)
    xs = edge_index(np.arange(w) - int(dx), w, mode)
    return np.take(np.take(plane, ys, axis=0), xs, axis=1)


def shift_rows(img: np.ndarray, offsets: np.ndarray, mode: str, *, chunk: int = 256) -> np.ndarray:
    """Shift every row horizontally by its own integer offset.

    Processed in row chunks so the temporary index array stays bounded
    (``chunk * W`` entries) regardless of image size.
    """
    h, w = img.shape[:2]
    offs = np.rint(np.asarray(offsets, dtype=np.float64)).astype(np.int64)
    out = np.empty_like(img)
    xs = np.arange(w, dtype=np.int64)
    for y0 in range(0, h, chunk):
        y1 = min(h, y0 + chunk)
        idx = edge_index(xs[None, :] - offs[y0:y1, None], w, mode)
        block = img[y0:y1]
        if img.ndim == 3:
            out[y0:y1] = np.take_along_axis(block, idx[:, :, None], axis=1)
        else:
            out[y0:y1] = np.take_along_axis(block, idx, axis=1)
    return out


def shift_cols(img: np.ndarray, offsets: np.ndarray, mode: str) -> np.ndarray:
    """Shift every column vertically by its own integer offset."""
    t = np.swapaxes(img, 0, 1)
    return np.ascontiguousarray(np.swapaxes(shift_rows(t, offsets, mode), 0, 1))


# Blending ---------------------------------------------------------------------

def lerp_u8(a: np.ndarray, b: np.ndarray, t: float | np.ndarray) -> np.ndarray:
    """Linear mix of two uint8 images: a*(1-t) + b*t (t scalar or HxW map)."""
    if np.isscalar(t):
        if t <= 0:
            return a.copy()
        if t >= 1:
            return b.copy()
        tt = np.float32(t)
    else:
        tt = np.asarray(t, np.float32)
        if tt.ndim == 2 and a.ndim == 3:
            tt = tt[..., None]
    fa = a.astype(np.float32)
    fb = b.astype(np.float32)
    return to_uint8(fa + (fb - fa) * tt, scale=1.0)


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    if edge1 == edge0:
        return (x >= edge0).astype(np.float32)
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def value_noise(h: int, w: int, rng: np.random.Generator, cells: float, octaves: int = 3) -> np.ndarray:
    """Smooth fractal value noise in [0,1] with shape (h, w).

    ``cells`` is the number of lattice cells across the long side for the
    first octave, so the pattern scales with the image (preview == export).
    """
    import cv2

    h = max(1, int(h))
    w = max(1, int(w))
    total = np.zeros((h, w), np.float32)
    amp = 1.0
    norm = 0.0
    long_side = max(h, w)
    for o in range(max(1, octaves)):
        c = max(1.0, cells * (2 ** o))
        gh = int(np.ceil(c * h / long_side)) + 2
        gw = int(np.ceil(c * w / long_side)) + 2
        lattice = rng.random((gh, gw)).astype(np.float32)
        layer = cv2.resize(lattice, (w, h), interpolation=cv2.INTER_CUBIC)
        total += layer * amp
        norm += amp
        amp *= 0.5
    total /= norm
    lo, hi = float(total.min()), float(total.max())
    if hi - lo > 1e-6:
        total = (total - lo) / (hi - lo)
    return np.clip(total, 0, 1).astype(np.float32)


def resize_rgba(img: np.ndarray, width: int, height: int, *, interpolation: str = "area") -> np.ndarray:
    """Resize uint8 RGBA with premultiplied alpha (avoids dark fringes)."""
    import cv2

    width = max(1, int(width))
    height = max(1, int(height))
    if img.shape[1] == width and img.shape[0] == height:
        return img.copy()
    interp = {
        "area": cv2.INTER_AREA,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
        "nearest": cv2.INTER_NEAREST,
    }[interpolation]
    alpha = img[..., 3]
    if alpha.min() == 255:
        rgb = cv2.resize(np.ascontiguousarray(img[..., :3]), (width, height), interpolation=interp)
        return np.dstack([rgb, np.full((height, width), 255, np.uint8)])
    f = to_float(img)
    f[..., :3] *= f[..., 3:4]
    r = cv2.resize(f, (width, height), interpolation=interp)
    r = np.clip(r, 0, 1)
    a = r[..., 3:4]
    rgb = np.where(a > 1e-6, r[..., :3] / np.maximum(a, 1e-6), 0.0)
    return to_uint8(np.concatenate([rgb, a], axis=-1))
