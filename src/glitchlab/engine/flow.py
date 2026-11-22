"""Artistic flow advection (used by Acid Flow and Acid / Liquid Type).

This is *not* a physical fluid solver.  A static 2-D velocity field is built
from seeded multiscale curl noise, smoothed image contours, gravity, vortex
centres and painted direction strokes; image coordinates are then advected
through it for a bounded number of semi-Lagrangian steps:

    phi_0(x)     = x
    phi_{k+1}(x) = phi_k(x - dt * v_k(x))        (bilinear sampling)

so that ``phi_N(x)`` is the source position whose colour arrives at ``x``.
Maps are composed by sampling the previous *displacement* ``phi_k(x) - x``,
which keeps the composition exact for the identity and well behaved at the
borders.  The field lives on a small simulation grid whose size depends
only on the image aspect ratio, so a reduced preview and the full-resolution
export see the same flow, scaled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

GRID_LONG = 384


def grid_shape(w: int, h: int, long_side: int = GRID_LONG) -> tuple[int, int]:
    s = long_side / float(max(w, h))
    return max(16, int(round(w * s))), max(16, int(round(h * s)))


def value_noise(rng: np.random.Generator, gw: int, gh: int, cell: float) -> np.ndarray:
    """Smooth noise on the (gw, gh) grid; ``cell`` is the feature size as a fraction of the long side."""
    long_cells = max(1.0, 1.0 / max(cell, 1e-3))
    nx = int(math.ceil(long_cells * gw / max(gw, gh))) + 3
    ny = int(math.ceil(long_cells * gh / max(gw, gh))) + 3
    lattice = rng.standard_normal((ny, nx)).astype(np.float32)
    return cv2.resize(lattice, (gw, gh), interpolation=cv2.INTER_CUBIC)


def curl_noise(rng: np.random.Generator, gw: int, gh: int, scale: float, octaves: int, detail: float,
               anisotropy: tuple[float, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Divergence-free multiscale curl noise, normalised to unit RMS speed."""
    psi = np.zeros((gh, gw), np.float32)
    amp, cell = 1.0, scale
    for _ in range(max(1, octaves)):
        psi += amp * value_noise(rng, gw, gh, cell)
        amp *= 0.35 + 0.5 * detail
        cell *= 0.5
    if anisotropy is not None:  # stretch features along a direction (melting streams, droplets)
        ux, uy, length = anisotropy
        psi = directional_blur(psi, ux, uy, length)
    gy, gx = np.gradient(psi)
    vx, vy = gy, -gx
    rms = float(np.sqrt(np.mean(vx * vx + vy * vy))) or 1.0
    return vx / rms, vy / rms


def directional_blur(arr: np.ndarray, ux: float, uy: float, length: float) -> np.ndarray:
    """Motion blur of ``length`` grid pixels along (ux, uy)."""
    n = int(max(3, min(151, round(length)))) | 1
    k = np.zeros((n, n), np.float32)
    c = n // 2
    for t in np.linspace(-c, c, n * 2):
        x, y = int(round(c + t * ux)), int(round(c + t * uy))
        if 0 <= x < n and 0 <= y < n:
            k[y, x] = 1.0
    k /= k.sum()
    return cv2.filter2D(arr, -1, k, borderType=cv2.BORDER_REFLECT_101)


def contour_field(lum: np.ndarray, smooth: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit flow along image contours (perpendicular to the gradient), the edge
    strength (0..1) and the attraction field towards edges."""
    gh, gw = lum.shape
    sig = max(0.6, smooth * max(gw, gh))
    L = cv2.GaussianBlur(lum.astype(np.float32), (0, 0), sig)
    gy, gx = np.gradient(L)
    mag = np.sqrt(gx * gx + gy * gy)
    edge = mag / (np.percentile(mag, 98) + 1e-6)
    edge = np.clip(edge, 0, 1).astype(np.float32)
    n = mag + 1e-6
    tx, ty = -gy / n, gx / n           # tangent: liquid travels along edges
    ey, ex = np.gradient(cv2.GaussianBlur(edge, (0, 0), sig * 2))
    em = np.sqrt(ex * ex + ey * ey) + 1e-6
    return np.stack([tx * edge, ty * edge]), edge, np.stack([ex / em * np.clip(em * 40, 0, 1),
                                                             ey / em * np.clip(em * 40, 0, 1)])


def vortex_field(gw: int, gh: int, centres: list[tuple[float, float, float]], size: float, pull: float
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian vortices. ``centres`` are (x, y, spin) in 0..1 image coordinates, spin = +/-1 * strength."""
    vx = np.zeros((gh, gw), np.float32)
    vy = np.zeros((gh, gw), np.float32)
    if not centres:
        return vx, vy
    long_px = float(max(gw, gh))
    xs, ys = np.meshgrid(np.arange(gw, dtype=np.float32), np.arange(gh, dtype=np.float32))
    r0 = max(1.0, size * long_px)
    for cx, cy, spin in centres:
        dx, dy = xs - cx * (gw - 1), ys - cy * (gh - 1)
        r2 = dx * dx + dy * dy
        fall = np.exp(-r2 / (r0 * r0)).astype(np.float32)
        r = np.sqrt(r2) + 1e-3
        vx += (-dy / r0 * spin - dx / r * pull * 0.5) * fall
        vy += (dx / r0 * spin - dy / r * pull * 0.5) * fall
    return vx, vy


def stroke_field(gw: int, gh: int, strokes: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Directional flow along painted strokes (direction = drawing order), fading with the brush radius."""
    vx = np.zeros((gh, gw), np.float32)
    vy = np.zeros((gh, gw), np.float32)
    wsum = np.zeros((gh, gw), np.float32)
    long_px = float(max(gw, gh))
    for s in strokes:
        if s.get("type") != "stroke" or s.get("mode") == "subtract" or len(s.get("points", [])) < 2:
            continue
        pts = np.asarray(s["points"], np.float32) * [gw - 1, gh - 1]
        rad = max(1.5, float(s.get("radius", 0.03)) * long_px * 2.0)
        for a, b in zip(pts[:-1], pts[1:]):
            d = b - a
            n = float(np.hypot(*d))
            if n < 1e-3:
                continue
            ux, uy = d / n
            m = np.zeros((gh, gw), np.uint8)
            cv2.line(m, tuple(int(round(v)) for v in a), tuple(int(round(v)) for v in b), 255,
                     thickness=max(1, int(rad)), lineType=cv2.LINE_AA)
            mf = m.astype(np.float32) / 255.0
            vx += mf * ux
            vy += mf * uy
            wsum += mf
    ok = wsum > 1e-3
    vx[ok] /= wsum[ok]
    vy[ok] /= wsum[ok]
    sig = max(1.0, long_px * 0.01)
    return cv2.GaussianBlur(vx, (0, 0), sig), cv2.GaussianBlur(vy, (0, 0), sig)


@dataclass
class FlowResult:
    dx: np.ndarray          # backward displacement on the grid, grid pixels
    dy: np.ndarray
    vx: np.ndarray          # velocity used (grid pixels per unit time)
    vy: np.ndarray
    grid: tuple[int, int]


def advect(vx: np.ndarray, vy: np.ndarray, distance: float, steps: int, viscosity: float,
           cancel=None) -> FlowResult:
    """Integrate the field. ``distance`` is the travel of a unit-speed particle,
    in grid pixels; ``viscosity`` (0..1) smooths and damps the velocity."""
    gh, gw = vx.shape
    sig = viscosity * 0.03 * max(gw, gh)
    if sig >= 0.5:
        vx = cv2.GaussianBlur(vx, (0, 0), sig)
        vy = cv2.GaussianBlur(vy, (0, 0), sig)
    steps = int(min(max(steps, 1), 96))
    damping = 1.0 - 0.6 * viscosity / steps * 4.0
    damping = float(min(max(damping, 0.5), 1.0))
    weights = damping ** np.arange(steps)
    dt = distance / float(weights.sum())
    xs, ys = np.meshgrid(np.arange(gw, dtype=np.float32), np.arange(gh, dtype=np.float32))
    dx = np.zeros((gh, gw), np.float32)
    dy = np.zeros((gh, gw), np.float32)
    for k in range(steps):
        if cancel is not None and k % 4 == 0:
            cancel.check()
        f = np.float32(dt * weights[k])
        bx, by = xs - f * vx, ys - f * vy
        # phi_{k+1}(x) = phi_k(b) = b + D_k(b)
        dxs = cv2.remap(dx, bx, by, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        dys = cv2.remap(dy, bx, by, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        dx = (bx - xs) + dxs
        dy = (by - ys) + dys
    return FlowResult(dx, dy, vx, vy, (gw, gh))


EDGE_MODES = {"mirror": cv2.BORDER_REFLECT_101, "clamp": cv2.BORDER_REPLICATE, "wrap": cv2.BORDER_WRAP}


def sample_grid(arr: np.ndarray, w: int, h: int, y0: int, y1: int) -> np.ndarray:
    """Bilinear sample of a grid map for render rows [y0, y1) of a (w, h) image."""
    gh, gw = arr.shape[:2]
    xs = (np.arange(w, dtype=np.float32) + 0.5) * gw / w - 0.5
    ys = (np.arange(y0, y1, dtype=np.float32) + 0.5) * gh / h - 0.5
    mx, my = np.meshgrid(xs, ys)
    return cv2.remap(np.ascontiguousarray(arr), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def premultiplied_remap(img: np.ndarray, mx: np.ndarray, my: np.ndarray, border: int) -> np.ndarray:
    """Sample uint8 RGBA with premultiplied alpha (no dark fringes); returns straight float RGBA 0..255."""
    a = img[..., 3:4].astype(np.float32) / 255.0
    pm = np.concatenate([img[..., :3].astype(np.float32) * a, a * 255.0], -1)
    out = cv2.remap(pm, mx, my, cv2.INTER_LINEAR, borderMode=border)
    alpha = out[..., 3:4] / 255.0
    rgb = np.where(alpha > 1e-4, out[..., :3] / np.maximum(alpha, 1e-4), 0.0)
    return np.concatenate([np.clip(rgb, 0, 255), out[..., 3:4]], -1)


def deformation(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Area change of the backward map (det of its Jacobian); < 1 = compressed (bright caustics)."""
    dxy, dxx = np.gradient(dx)
    dyy, dyx = np.gradient(dy)
    return (1.0 + dxx) * (1.0 + dyy) - dxy * dyx
