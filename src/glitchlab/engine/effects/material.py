"""Material looks: Y2K translucent clear plastic."""

from __future__ import annotations

import math

import cv2
import numpy as np

from ..ops import luminance, to_float, to_uint8
from ..params import COLOR, Param, color_to_rgb, cparam, fparam, iparam
from .base import Effect, register

PLASTIC_TINTS = {
    "bondi": ("Bondi blue", "#00a3c4"),
    "tangerine": ("Tangerine", "#ff7a1a"),
    "grape": ("Grape", "#7b3fbf"),
    "lime": ("Lime", "#7ed321"),
    "strawberry": ("Strawberry", "#ff3b6b"),
    "ice": ("Ice", "#cfe9ff"),
    "graphite": ("Graphite smoke", "#5d6470"),
    "custom": ("Custom colour", None),
}


@register
class ClearPlastic(Effect):
    id = "clear_plastic"
    name = "Y2K Clear Plastic"
    category = "Abstract & Style"
    description = ("Turns the image into glossy, translucent candy-coloured plastic: the photo's light and shade "
                   "become a smooth jelly surface that refracts what is behind it, with a tinted body, specular "
                   "hotspots, rim light, caustic glints and optional bubbles.")
    supports_channels = False
    preview_note = "Bubbles are placed in normalised positions; surface detail follows the scaled smoothing."
    params = (
        cparam("tint", "Plastic colour", "bondi", tuple((k, v[0]) for k, v in PLASTIC_TINTS.items())),
        Param("custom_tint", "Custom colour", COLOR, "#9ad0ff", show_if=("tint", ("custom",))),
        fparam("translucency", "Translucency", 0.6, 0.0, 1.0, tooltip="How clearly the photo shows through the plastic."),
        iparam("smooth", "Smoothness", 14, 1, 400, spatial=True, unit="px", tooltip="Blur of the surface: bigger = blobbier jelly."),
        fparam("thickness", "Bulge", 1.0, 0.0, 4.0, tooltip="How strongly the surface bulges."),
        iparam("refraction", "Refraction", 18, 0, 400, spatial=True, unit="px"),
        iparam("frost", "Frost", 2, 0, 100, spatial=True, unit="px", tooltip="Blur of what is seen through the plastic."),
        fparam("gloss", "Gloss", 1.0, 0.0, 2.5, group="Light"),
        fparam("shininess", "Shininess", 48.0, 4.0, 256.0, decimals=0, group="Light"),
        fparam("light_angle", "Light direction", 135.0, 0.0, 360.0, decimals=0, unit="°", group="Light"),
        fparam("rim", "Rim light", 0.5, 0.0, 1.5, group="Light"),
        fparam("caustics", "Caustic glints", 0.35, 0.0, 1.5, group="Light"),
        iparam("bubbles", "Bubbles", 14, 0, 300),
    )

    def process(self, img, p, ctx):
        h, w = img.shape[:2]
        L = float(max(h, w))
        f = to_float(img[..., :3])
        tint_hex = PLASTIC_TINTS[p["tint"]][1] or p["custom_tint"]
        tint = np.array(color_to_rgb(tint_hex), np.float32) / 255.0
        lum = luminance(img)
        sm = max(1.0, float(p["smooth"]))
        height = cv2.GaussianBlur(lum, (0, 0), sm / 2.0)
        k = np.float32(p["thickness"] * sm * 2.5)
        gx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        gy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        nx, ny, nz = -gx * k, -gy * k, np.ones_like(gx)
        norm = np.sqrt(nx * nx + ny * ny + nz * nz)
        nx, ny, nz = nx / norm, ny / norm, nz / norm
        xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        refr = float(p["refraction"])
        seen = cv2.remap(np.ascontiguousarray(f), xs + nx * refr, ys + ny * refr, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)
        if p["frost"] > 0:
            seen = cv2.GaussianBlur(seen, (0, 0), max(0.5, p["frost"] / 2.0))
        seen_lum = seen.mean(axis=-1, keepdims=True)
        t = np.float32(p["translucency"])
        transmitted = seen * (tint * 0.65 + 0.35)
        body = tint * (0.45 + 0.55 * seen_lum)
        col = transmitted * t + body * (1 - t)
        a = math.radians(p["light_angle"])
        lx, ly, lz = math.cos(a), -math.sin(a), 0.9
        ln = math.sqrt(lx * lx + ly * ly + lz * lz)
        lx, ly, lz = lx / ln, ly / ln, lz / ln
        ndotl = np.clip(nx * lx + ny * ly + nz * lz, 0.0, 1.0)
        hx, hy, hz = lx, ly, lz + 1.0
        hn = math.sqrt(hx * hx + hy * hy + hz * hz)
        ndoth = np.clip((nx * hx + ny * hy + nz * hz) / hn, 0.0, 1.0)
        spec = (ndoth ** np.float32(p["shininess"])) * np.float32(p["gloss"])
        rim = ((1.0 - nz) ** 2) * np.float32(p["rim"]) * 3.0
        col = col * (0.72 + 0.28 * ndotl)[..., None]
        col = col + spec[..., None] + (rim[..., None] * (0.6 + 0.4 * tint))
        if p["caustics"] > 0:
            lap = cv2.Laplacian(height, cv2.CV_32F, ksize=3)
            c = np.clip(-lap * k * 6.0, 0, 1) * np.float32(p["caustics"])
            col = col + c[..., None] * (tint * 0.5 + 0.5)
        n_b = int(p["bubbles"])
        if n_b:
            rng = ctx.rng
            ring = np.zeros((h, w), np.float32)
            dots = np.zeros((h, w), np.float32)
            for _ in range(n_b):
                bx, by = rng.random() * w, rng.random() * h
                br = rng.uniform(0.004, 0.022) * L
                thick = max(1, int(round(br * 0.12)))
                cv2.circle(ring, (int(bx), int(by)), max(1, int(br)), 1.0, thick, cv2.LINE_AA)
                cv2.circle(dots, (int(bx - br * 0.35), int(by - br * 0.35)), max(1, int(br * 0.22)), 1.0, -1,
                           cv2.LINE_AA)
            col = col * (1 - ring[..., None] * 0.35) + ring[..., None] * 0.25 * tint + dots[..., None] * 0.9
        out = img.copy()
        out[..., :3] = to_uint8(col)
        return out
