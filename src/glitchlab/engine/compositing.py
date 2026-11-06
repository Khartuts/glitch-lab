"""Layer compositing.

One compositing model is used for preview and export alike: the W3C
Compositing and Blending Level 1 formula for *source-over* with separable
blend modes, computed in float32 on sRGB-encoded values:

    Cs' = (1 - ab) * Cs + ab * B(Cb, Cs)
    ao  = as + ab * (1 - as)
    Co  = (as * Cs' + ab * (1 - as) * Cb) / ao

where ``as`` already includes layer opacity and mask.
"""

from __future__ import annotations

import numpy as np

from .ops import to_float, to_uint8


def blend_function(cb: np.ndarray, cs: np.ndarray, mode: str) -> np.ndarray:
    if mode == "normal":
        return cs
    if mode == "multiply":
        return cb * cs
    if mode == "screen":
        return cb + cs - cb * cs
    if mode == "difference":
        return np.abs(cb - cs)
    if mode == "overlay":
        # overlay(Cb, Cs) = hardlight(Cs, Cb)
        return np.where(cb <= 0.5, 2.0 * cb * cs, 1.0 - 2.0 * (1.0 - cb) * (1.0 - cs))
    raise ValueError(f"Unknown blend mode {mode!r}")


def composite_over(backdrop: np.ndarray | None, source: np.ndarray, *, mode: str = "normal",
                   opacity: float = 1.0, mask: np.ndarray | None = None) -> np.ndarray:
    """Composite float32 RGBA ``source`` over float32 RGBA ``backdrop``.

    Both arrays are straight (non-premultiplied) alpha in [0, 1].  Returns a
    new float32 RGBA array.
    """
    cs = source[..., :3]
    a_s = source[..., 3] * np.float32(opacity)
    if mask is not None:
        a_s = a_s * mask.astype(np.float32)
    if backdrop is None:
        out = np.empty_like(source)
        out[..., :3] = cs
        out[..., 3] = a_s
        return out
    cb = backdrop[..., :3]
    a_b = backdrop[..., 3]
    mixed = cs if mode == "normal" else (1.0 - a_b)[..., None] * cs + a_b[..., None] * blend_function(cb, cs, mode)
    a_o = a_s + a_b * (1.0 - a_s)
    num = a_s[..., None] * mixed + (a_b * (1.0 - a_s))[..., None] * cb
    safe = np.where(a_o > 1e-7, a_o, 1.0)
    co = np.where(a_o[..., None] > 1e-7, num / safe[..., None], 0.0)
    out = np.empty_like(source)
    out[..., :3] = np.clip(co, 0.0, 1.0)
    out[..., 3] = np.clip(a_o, 0.0, 1.0)
    return out


def composite_layers(images: list[np.ndarray], modes: list[str], opacities: list[float],
                     masks: list[np.ndarray | None]) -> np.ndarray:
    """Composite uint8 RGBA layer images bottom-to-top -> uint8 RGBA."""
    if not images:
        raise ValueError("No visible layers")
    # Fast path: a single opaque normal layer is returned untouched.
    if len(images) == 1 and opacities[0] >= 1.0 and masks[0] is None:
        return images[0].copy()
    acc = None
    for img, mode, op, mask in zip(images, modes, opacities, masks):
        acc = composite_over(acc, to_float(img), mode=mode, opacity=op, mask=mask)
    return to_uint8(acc)
