"""Frequency-domain damage and feedback accumulation."""

from __future__ import annotations

import cv2
import numpy as np

from ..ops import to_float, to_uint8
from ..params import cparam, fparam, iparam
from .base import Effect, register


def _smooth_band(r: np.ndarray, lo: float, hi: float, soft: float) -> np.ndarray:
    soft = max(soft, 1e-4)
    a = np.clip((r - (lo - soft)) / soft, 0.0, 1.0)
    b = np.clip(((hi + soft) - r) / soft, 0.0, 1.0)
    m = np.minimum(a, b)
    return (m * m * (3 - 2 * m)).astype(np.float32)


def antisymmetrize_columns(noise: np.ndarray, width: int) -> np.ndarray:
    """Make phase noise odd-symmetric on the self-conjugate rfft2 columns.

    In the half spectrum produced by ``rfft2`` the DC column (and the Nyquist
    column for even widths) contain both F[ky] and F[-ky], which must stay
    complex conjugates for the inverse to be real.  Odd phase noise
    (n[ky] = -n[-ky]) preserves that, so no energy is silently discarded.
    """
    h = noise.shape[0]
    cols = [0] + ([width // 2] if width % 2 == 0 else [])
    neg = (-np.arange(h)) % h
    for c in cols:
        n = noise[:, c]
        noise[:, c] = (n - n[neg]) * 0.5
    return noise


@register
class SpectralDamage(Effect):
    id = "spectral"
    name = "Spectral Damage"
    category = "Frequency & Feedback"
    description = ("2-D FFT per channel: boosts or attenuates a radial frequency band (optionally only one "
                   "orientation), perturbs its phase with seeded noise and reconstructs a finite real image.")
    params = (
        fparam("band_low", "Band low", 0.04, 0.0, 1.4, decimals=3, tooltip="Fraction of the full-resolution Nyquist frequency."),
        fparam("band_high", "Band high", 0.3, 0.0, 1.4, decimals=3),
        fparam("softness", "Band softness", 0.02, 0.0, 0.3, decimals=3),
        fparam("gain", "Band gain", 2.0, 0.0, 6.0, tooltip="Magnitude multiplier inside the band."),
        fparam("outside_gain", "Outside gain", 1.0, 0.0, 2.0, tooltip="Magnitude multiplier outside the band (DC kept)."),
        fparam("phase_noise", "Phase noise", 0.4, 0.0, 1.0, tooltip="Random phase offset inside the band, as a fraction of π."),
        fparam("orientation", "Orientation", 0.0, -90.0, 90.0, decimals=0, unit="°"),
        fparam("sector", "Angular width", 180.0, 1.0, 180.0, decimals=0, unit="°",
               tooltip="180° = all orientations; smaller values only damage stripes near the orientation."),
        cparam("channels_mode", "Phase noise", "per_channel", (("per_channel", "Independent per channel"), ("shared", "Shared by channels"))),
        cparam("normalize", "Range", "clip", (("clip", "Clip to 0..255"), ("stretch", "Stretch to full range"))),
    )

    def process(self, img, p, ctx):
        h, w = img.shape[:2]
        f = to_float(img[..., :3])
        fy = np.fft.fftfreq(h).astype(np.float32)[:, None]
        fx = np.fft.rfftfreq(w).astype(np.float32)[None, :]
        # radius relative to the *full-resolution* Nyquist (preview-consistent)
        r = np.sqrt(fx * fx + fy * fy) / 0.5 * np.float32(ctx.scale)
        band = _smooth_band(r, min(p["band_low"], p["band_high"]), max(p["band_low"], p["band_high"]), p["softness"])
        if p["sector"] < 180.0:
            ang = np.degrees(np.arctan2(fy, np.where(fx == 0, 1e-9, fx)))
            dif = np.abs(((ang - p["orientation"]) + 90.0) % 180.0 - 90.0)
            band = band * np.clip((p["sector"] / 2.0 - dif) / 3.0 + 0.5, 0.0, 1.0).astype(np.float32)
        gain = p["outside_gain"] + (p["gain"] - p["outside_gain"]) * band
        gain[0, 0] = 1.0  # keep the mean brightness
        shared = None
        out = np.empty_like(f)
        for c in range(3):
            ctx.check()
            F = np.fft.rfft2(f[..., c])
            if p["phase_noise"] > 0:
                if p["channels_mode"] == "shared":
                    if shared is None:
                        shared = ctx.sub_rng(99).uniform(-np.pi, np.pi, F.shape).astype(np.float32)
                    noise = shared.copy()
                else:
                    noise = ctx.sub_rng(c + 1).uniform(-np.pi, np.pi, F.shape).astype(np.float32)
                noise *= p["phase_noise"] * band
                noise[0, 0] = 0.0
                noise = antisymmetrize_columns(noise, w)
                F = F * (gain * np.exp(1j * noise))
            else:
                F = F * gain
            rec = np.fft.irfft2(F, s=(h, w)).astype(np.float32)
            out[..., c] = np.nan_to_num(rec, nan=0.0, posinf=1.0, neginf=0.0)
            ctx.progress((c + 1) / 3)
        if p["normalize"] == "stretch":
            lo, hi = np.percentile(out, [0.5, 99.5])
            out = (out - lo) / max(1e-6, hi - lo)
        res = img.copy()
        res[..., :3] = to_uint8(out)
        return res


def _hue_rotation_matrix(deg: float) -> np.ndarray:
    """RGB hue rotation about the grey axis (luminance-preserving approximation)."""
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    k = 1.0 / 3.0
    sq = np.sqrt(k)
    return np.array([
        [c + (1 - c) * k, k * (1 - c) - sq * s, k * (1 - c) + sq * s],
        [k * (1 - c) + sq * s, c + k * (1 - c), k * (1 - c) - sq * s],
        [k * (1 - c) - sq * s, k * (1 - c) + sq * s, c + k * (1 - c)],
    ], np.float32)


_FB_BLENDS = (("normal", "Normal"), ("screen", "Screen"), ("add", "Add"), ("multiply", "Multiply"),
              ("difference", "Difference"), ("lighten", "Lighten"), ("darken", "Darken"))


@register
class FeedbackAccumulation(Effect):
    id = "feedback"
    name = "Feedback Accumulation"
    category = "Frequency & Feedback"
    description = ("Repeatedly zooms, rotates and translates the previous iteration and blends it back over "
                   "the source with decay - a still-image analogue of video feedback.")
    uses_seed = False
    supports_alpha = True
    params = (
        iparam("iterations", "Iterations", 8, 1, 64),
        fparam("zoom", "Zoom", 1.04, 0.8, 1.25, decimals=3),
        fparam("rotation", "Rotation", 1.5, -20.0, 20.0, unit="°"),
        iparam("dx", "Translate X", 0, -400, 400, spatial=True, unit="px"),
        iparam("dy", "Translate Y", 0, -400, 400, spatial=True, unit="px"),
        fparam("decay", "Decay", 0.75, 0.0, 0.99, tooltip="Weight of the fed-back image each iteration."),
        cparam("blend", "Blend", "screen", _FB_BLENDS),
        fparam("hue_drift", "Hue drift", 0.0, 0.0, 90.0, unit="°/iter"),
        fparam("cx", "Centre X", 0.5, 0.0, 1.0),
        fparam("cy", "Centre Y", 0.5, 0.0, 1.0),
    )

    def process(self, img, p, ctx):
        h, w = img.shape[:2]
        src = to_float(img)
        acc = src.copy()
        M = cv2.getRotationMatrix2D((p["cx"] * w, p["cy"] * h), p["rotation"], p["zoom"])
        M[0, 2] += p["dx"]
        M[1, 2] += p["dy"]
        hue = _hue_rotation_matrix(p["hue_drift"]) if p["hue_drift"] > 0 else None
        d = np.float32(p["decay"])
        mode = p["blend"]
        for i in range(int(p["iterations"])):
            ctx.check()
            ctx.progress(i / p["iterations"])
            t = cv2.warpAffine(acc, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            if hue is not None:
                t[..., :3] = np.clip(t[..., :3] @ hue.T, 0.0, 1.0)
            s = src
            if mode == "normal":
                fed = t
            elif mode == "screen":
                fed = s + t - s * t
            elif mode == "add":
                fed = s + t
            elif mode == "multiply":
                fed = s * t
            elif mode == "difference":
                fed = np.abs(s - t)
            elif mode == "lighten":
                fed = np.maximum(s, t)
            else:
                fed = np.minimum(s, t)
            acc = np.clip(s * (1 - d) + fed * d, 0.0, 1.0)
        return to_uint8(acc)
