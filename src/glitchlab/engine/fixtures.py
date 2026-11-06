"""Small generated test images (no external files needed)."""

from __future__ import annotations

import cv2
import numpy as np


def _rgba(rgb: np.ndarray, alpha: np.ndarray | None = None) -> np.ndarray:
    h, w = rgb.shape[:2]
    a = np.full((h, w), 255, np.uint8) if alpha is None else alpha.astype(np.uint8)
    return np.ascontiguousarray(np.dstack([rgb.astype(np.uint8), a]))


def gradient(w: int = 256, h: int = 192) -> np.ndarray:
    x = np.linspace(0, 255, w, dtype=np.float32)[None, :].repeat(h, 0)
    y = np.linspace(0, 255, h, dtype=np.float32)[:, None].repeat(w, 1)
    rgb = np.stack([x, y, 255 - x * 0.5 - y * 0.5], -1)
    return _rgba(np.rint(rgb))


def color_bars(w: int = 256, h: int = 192) -> np.ndarray:
    colors = np.array([[192, 192, 192], [192, 192, 0], [0, 192, 192], [0, 192, 0],
                       [192, 0, 192], [192, 0, 0], [0, 0, 192], [16, 16, 16]], np.uint8)
    idx = (np.arange(w) * len(colors) // w)
    rgb = colors[idx][None, :, :].repeat(h, 0)
    ramp = np.linspace(0, 255, w).astype(np.uint8)
    rgb[int(h * 0.8):] = ramp[None, :, None]
    return _rgba(rgb)


def checkerboard(w: int = 256, h: int = 192, cell: int = 16) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    c = (((yy // cell) + (xx // cell)) % 2).astype(np.uint8)
    rgb = np.where(c[..., None] == 1, np.array([235, 235, 235], np.uint8), np.array([30, 30, 30], np.uint8))
    return _rgba(rgb)


def transparent_circle(w: int = 256, h: int = 192) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy, r = w / 2, h / 2, min(w, h) * 0.4
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    alpha = np.clip((r - d) * 2.0 + 0.5, 0, 1) * 255
    alpha[: h // 6] = np.clip(alpha[: h // 6], 0, 128)
    rgb = np.stack([np.full_like(d, 230), xx / w * 255, yy / h * 255], -1)
    return _rgba(np.rint(rgb), np.rint(alpha))


def photo_like(w: int = 320, h: int = 240, seed: int = 7) -> np.ndarray:
    """A deterministic synthetic 'photograph': sky gradient, sun, hills,
    shapes and fine noise, so effects have edges, textures and gradients."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    sky = np.stack([40 + 120 * (yy / h), 90 + 110 * (yy / h), 200 - 60 * (yy / h)], -1)
    img = sky.copy()
    cx, cy, r = w * 0.72, h * 0.3, min(w, h) * 0.12
    sun = np.clip(1.5 - np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r, 0, 1)[..., None]
    img = img * (1 - sun) + np.array([255, 220, 120], np.float32) * sun
    hill1 = h * 0.62 + np.sin(xx / w * 6.0) * h * 0.06
    hill2 = h * 0.75 + np.sin(xx / w * 11.0 + 1.3) * h * 0.04
    img = np.where((yy > hill1)[..., None], np.array([50, 110, 60], np.float32) * (0.7 + 0.3 * yy / h)[..., None], img)
    img = np.where((yy > hill2)[..., None], np.array([120, 85, 50], np.float32) * (0.8 + 0.2 * xx / w)[..., None], img)
    out = np.clip(img, 0, 255).astype(np.uint8)
    out = np.ascontiguousarray(out)
    cv2.rectangle(out, (int(w * 0.1), int(h * 0.45)), (int(w * 0.25), int(h * 0.8)), (200, 60, 60), -1)
    cv2.rectangle(out, (int(w * 0.13), int(h * 0.5)), (int(w * 0.18), int(h * 0.58)), (250, 240, 200), -1)
    cv2.circle(out, (int(w * 0.4), int(h * 0.7)), int(min(w, h) * 0.07), (230, 230, 240), -1, cv2.LINE_AA)
    cv2.line(out, (0, int(h * 0.92)), (w, int(h * 0.88)), (20, 20, 30), max(1, h // 60), cv2.LINE_AA)
    noise = rng.normal(0, 6, (h, w, 3))
    out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return _rgba(out)


ALL_FIXTURES = {
    "gradient": gradient,
    "color_bars": color_bars,
    "checkerboard": checkerboard,
    "transparent": transparent_circle,
    "photo": photo_like,
}
