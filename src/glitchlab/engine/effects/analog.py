"""Analog video looks: VHS tape and CRT screen."""

from __future__ import annotations

import cv2
import numpy as np

from ..ops import shift_rows, to_float, to_uint8
from ..params import bparam, cparam, fparam, iparam
from .base import Effect, register

# RGB <-> YIQ (NTSC) - the colour space VHS records chroma in
_RGB2YIQ = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]], np.float32)
_YIQ2RGB = np.linalg.inv(_RGB2YIQ).astype(np.float32)


def _hblur(plane: np.ndarray, px: float) -> np.ndarray:
    if px < 0.5:
        return plane
    k = max(3, int(px * 2) | 1)
    return cv2.GaussianBlur(plane, (k, 1), sigmaX=px / 1.5, sigmaY=0.1)


@register
class VHSCRT(Effect):
    id = "vhs_crt"
    name = "VHS & CRT"
    category = "Analog & Lens"
    description = ("Paused-VHS / old-TV look: smeared and shifted colour, tape tracking errors, seeded static "
                   "noise, scanlines, phosphor masks, bloom, vignette and screen curvature.")
    supports_channels = False
    preview_note = "Noise grain and phosphor masks are pixel-sized and look finer in the export."
    params = (
        iparam("chroma_blur", "Colour bleed", 6, 0, 200, spatial=True, unit="px", group="Tape colour"),
        iparam("chroma_shift", "Colour shift", 3, -200, 200, spatial=True, unit="px", group="Tape colour"),
        fparam("saturation", "Saturation", 1.15, 0.0, 2.5, group="Tape colour"),
        fparam("sharpen", "Luma ringing", 0.4, 0.0, 2.0, group="Tape colour",
               tooltip="VHS-style edge sharpening halos."),
        fparam("tracking", "Tracking error", 0.35, 0.0, 1.0, group="Tracking & noise"),
        iparam("tracking_bands", "Tracking bands", 3, 0, 30, group="Tracking & noise"),
        fparam("noise", "Static noise", 0.12, 0.0, 1.0, group="Tracking & noise"),
        iparam("streaks", "Noise streaks", 8, 0, 200, group="Tracking & noise", tooltip="Bright horizontal dropout lines."),
        fparam("scanlines", "Scanline strength", 0.35, 0.0, 1.0, group="CRT screen"),
        iparam("scan_period", "Scanline spacing", 3, 2, 40, spatial=True, unit="px", group="CRT screen"),
        cparam("mask", "Phosphor mask", "aperture", (("none", "None"), ("aperture", "Aperture grille"),
                                                     ("shadow", "Shadow mask"), ("slot", "Slot mask")), group="CRT screen"),
        fparam("mask_strength", "Mask strength", 0.25, 0.0, 1.0, group="CRT screen"),
        fparam("bloom", "Bloom", 0.25, 0.0, 1.0, group="CRT screen"),
        fparam("vignette", "Vignette", 0.35, 0.0, 1.0, group="CRT screen"),
        fparam("curvature", "Curvature", 0.08, 0.0, 0.4, group="CRT screen"),
        bparam("border", "Black screen border", True, group="CRT screen"),
    )

    def process(self, img, p, ctx):
        h, w = img.shape[:2]
        rng = ctx.rng
        f = to_float(img[..., :3])
        yiq = f @ _RGB2YIQ.T
        y, iq = yiq[..., 0], yiq[..., 1:]
        # --- tape colour: chroma is low-bandwidth and lags behind luma
        iq = np.stack([_hblur(np.ascontiguousarray(iq[..., i]), p["chroma_blur"]) for i in range(2)], -1)
        if p["chroma_shift"]:
            iq = np.roll(iq, int(p["chroma_shift"]), axis=1)
        iq *= np.float32(p["saturation"])
        if p["sharpen"] > 0:
            soft = _hblur(y, 2.0 + p["chroma_blur"] * 0.15)
            y = y + (y - soft) * np.float32(p["sharpen"])
        # --- noise
        full_h = max(1, int(ctx.full_size[1]))
        rows_full = np.minimum(full_h - 1, (np.arange(h) / max(ctx.scale, 1e-9)).astype(np.int64))
        if p["noise"] > 0:
            y = y + ctx.sub_rng(1).normal(0.0, 0.12 * p["noise"], (h, w)).astype(np.float32)
            iq = iq + ctx.sub_rng(2).normal(0.0, 0.05 * p["noise"], (h, w, 2)).astype(np.float32)
        out = np.dstack([y, iq]) @ _YIQ2RGB.T
        # --- tracking errors: bands near the bottom jump sideways and get noisy
        offsets = np.zeros(h, np.float64)
        if p["tracking"] > 0 and p["tracking_bands"] > 0:
            for _ in range(int(p["tracking_bands"])):
                c = rng.uniform(0.6, 1.0) if rng.random() < 0.6 else rng.random()
                t = rng.uniform(0.005, 0.03)
                y0, y1 = int((c - t) * h), int(c * h)
                y0, y1 = max(0, y0), min(h, max(y1, y0 + 1))
                ramp = np.linspace(0, 1, max(1, y1 - y0)) ** 2
                offsets[y0:y1] += rng.uniform(-1, 1) * p["tracking"] * 0.08 * w * ramp
            jitter = rng.normal(0, 1, full_h)[rows_full] * p["tracking"] * 0.002 * w
            offsets += jitter
            out = shift_rows(out.astype(np.float32), offsets, "clamp")
        if p["streaks"] > 0:
            for _ in range(int(p["streaks"])):
                row = int(rng.random() * h)
                x0 = int(rng.random() * w)
                ln = int(rng.uniform(0.05, 0.5) * w)
                hh = max(1, int(round(ctx.scale)))
                out[row:row + hh, x0:x0 + ln] = np.clip(out[row:row + hh, x0:x0 + ln] + rng.uniform(0.3, 0.8), 0, 1)
        out = np.clip(out, 0.0, 1.0)
        # --- CRT: bloom, scanlines, phosphor mask
        if p["bloom"] > 0:
            bright = np.clip(out - 0.6, 0, 1)
            glow = cv2.GaussianBlur(bright, (0, 0), max(1.0, 0.01 * max(h, w)))
            out = out + glow * np.float32(p["bloom"] * 1.5)
        if p["scanlines"] > 0:
            per = max(2.0, float(p["scan_period"]))
            yy = np.arange(h, dtype=np.float32)
            line = 1.0 - p["scanlines"] * (0.5 - 0.5 * np.cos(2 * np.pi * yy / per))
            out = out * line[:, None, None]
        if p["mask"] != "none" and p["mask_strength"] > 0:
            cell = max(3, int(round(3 * max(1.0, ctx.scale * 1.0))))
            xx = np.arange(w)
            phase = (xx // max(1, cell // 3)) % 3
            m = np.full((h, w, 3), 1.0 - p["mask_strength"], np.float32)
            for ch in range(3):
                m[:, :, ch][:, phase == ch] = 1.0
            if p["mask"] in ("shadow", "slot"):
                yy = np.arange(h)
                if p["mask"] == "shadow":
                    stagger = ((yy // cell) % 2)[:, None] * (cell // 2)
                    ph2 = ((xx[None, :] + stagger) // max(1, cell // 3)) % 3
                    m = np.where(ph2[..., None] == np.arange(3)[None, None, :], 1.0, 1.0 - p["mask_strength"]).astype(np.float32)
                else:
                    gap = (yy % (cell * 2)) == 0
                    m[gap] *= 1.0 - p["mask_strength"]
            out = out * m * np.float32(1.0 + p["mask_strength"] * 0.6)
        # --- curvature + vignette
        if p["curvature"] > 0 or p["vignette"] > 0:
            nx = (np.arange(w, dtype=np.float32) + 0.5) / w * 2 - 1
            ny = (np.arange(h, dtype=np.float32) + 0.5) / h * 2 - 1
            X, Y = np.meshgrid(nx, ny)
            r2 = X * X + Y * Y
            if p["curvature"] > 0:
                k = np.float32(p["curvature"])
                sx = X * (1 + k * r2)
                sy = Y * (1 + k * r2)
                mx = ((sx + 1) / 2 * w - 0.5).astype(np.float32)
                my = ((sy + 1) / 2 * h - 0.5).astype(np.float32)
                border = cv2.BORDER_CONSTANT if p["border"] else cv2.BORDER_REPLICATE
                out = cv2.remap(out.astype(np.float32), mx, my, interpolation=cv2.INTER_LINEAR, borderMode=border,
                                borderValue=(0, 0, 0))
            if p["vignette"] > 0:
                out = out * np.clip(1.0 - p["vignette"] * r2 * 0.5, 0, 1)[..., None]
        res = img.copy()
        res[..., :3] = to_uint8(out)
        return res
