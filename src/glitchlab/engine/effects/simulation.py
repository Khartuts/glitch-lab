"""Simulations seeded from the image: cellular automata and Gray-Scott reaction-diffusion.

Both run on a grid whose size is defined relative to the image (cells /
simulation size across the long side), so the reduced preview and the
full-resolution export evolve the same pattern.  Iterations are bounded and
values are clamped every step, so the simulations can never overflow or
produce NaN.
"""

from __future__ import annotations

import re

import cv2
import numpy as np

from ..ops import luminance, to_float, to_uint8
from ..params import COLOR, TEXT, Param, bparam, color_to_rgb, cparam, fparam, iparam
from .base import Effect, register

NAMED_RULES = {
    "life": "B3/S23", "highlife": "B36/S23", "seeds": "B2/S", "daynight": "B3678/S34678",
    "maze": "B3/S12345", "anneal": "B4678/S35678", "replicator": "B1357/S1357", "coral": "B3/S45678",
    "diamoeba": "B35678/S5678", "morley": "B368/S245",
}
_RULE_RE = re.compile(r"^\s*B([0-8]*)\s*/\s*S([0-8]*)\s*$", re.IGNORECASE)


def parse_rule(text: str) -> tuple[np.ndarray, np.ndarray]:
    t = (text or "").strip()
    t = NAMED_RULES.get(t.lower(), t)
    m = _RULE_RE.match(t)
    if not m:
        raise ValueError(f"'{text}' is not a rule like B3/S23 (or a name: {', '.join(NAMED_RULES)}).")
    born = np.zeros(9, bool)
    surv = np.zeros(9, bool)
    for ch in m.group(1):
        born[int(ch)] = True
    for ch in m.group(2):
        surv[int(ch)] = True
    return born, surv


def life_step(alive: np.ndarray, born: np.ndarray, surv: np.ndarray) -> np.ndarray:
    p = np.pad(alive.astype(np.uint8), 1, mode="wrap")
    n = (p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] + p[1:-1, :-2] + p[1:-1, 2:] +
         p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:])
    return np.where(alive, surv[n], born[n])


def _grid_size(h: int, w: int, cells: int) -> tuple[int, int]:
    s = cells / float(max(h, w))
    return max(2, int(round(h * s))), max(2, int(round(w * s)))


@register
class CellularAutomata(Effect):
    id = "cellular"
    name = "Cellular Automata"
    category = "Simulation"
    description = ("Seeds a life-like cellular automaton from luminance, a bit plane or noise, evolves it with a "
                   "B/S rule, then blends, inverts, darkens or displaces the image with the result.")
    params = (
        cparam("seed_mode", "Seed from", "luminance", (("luminance", "Luminance threshold"), ("bitplane", "Bit plane"),
                                                         ("random", "Random noise"))),
        fparam("threshold", "Threshold", 0.5, 0.0, 1.0, show_if=("seed_mode", ("luminance",))),
        iparam("bit", "Bit plane", 5, 0, 7, show_if=("seed_mode", ("bitplane",))),
        fparam("density", "Noise density", 0.35, 0.01, 0.99, show_if=("seed_mode", ("random",))),
        Param("rule", "Rule", TEXT, "B3678/S34678", randomizable=False,
              tooltip="Birth/Survival rule, e.g. B3/S23, or a name: " + ", ".join(NAMED_RULES)),
        iparam("steps", "Generations", 40, 1, 400),
        iparam("cells", "Cells across", 256, 16, 1024, tooltip="Grid resolution across the long side."),
        cparam("output", "Output", "overlay", (("overlay", "Colour overlay"), ("darken", "Darken"), ("invert", "Invert"),
                                               ("trails", "Age trails"), ("displace", "Displace"))),
        fparam("amount", "Amount", 0.7, 0.0, 1.0),
        Param("color", "Cell colour", COLOR, "#ff3d7f", show_if=("output", ("overlay", "trails"))),
    )

    def validate(self, p):
        try:
            parse_rule(p["rule"])
        except ValueError as exc:
            return [str(exc)]
        return []

    def process(self, img, p, ctx):
        h, w = img.shape[:2]
        gh, gw = _grid_size(h, w, p["cells"])
        small = cv2.resize(np.ascontiguousarray(img), (gw, gh), interpolation=cv2.INTER_AREA)
        mode = p["seed_mode"]
        if mode == "luminance":
            alive = luminance(small) > p["threshold"]
        elif mode == "bitplane":
            alive = ((np.rint(luminance(small) * 255).astype(np.uint8) >> p["bit"]) & 1).astype(bool)
        else:
            alive = ctx.rng.random((gh, gw)) < p["density"]
        born, surv = parse_rule(p["rule"])
        age = np.zeros((gh, gw), np.float32)
        steps = int(p["steps"])
        for i in range(steps):
            if i % 16 == 0:
                ctx.check()
                ctx.progress(i / steps)
            alive = life_step(alive, born, surv)
            age = np.where(alive, age + 1.0, age * 0.85)
        a = cv2.resize(alive.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        ag = age / max(1.0, float(age.max()))
        ag = cv2.resize(ag, (w, h), interpolation=cv2.INTER_NEAREST)
        amt = float(p["amount"])
        f = to_float(img[..., :3])
        out_mode = p["output"]
        col = np.array(color_to_rgb(p["color"]), np.float32) / 255.0
        if out_mode == "overlay":
            m = (a * amt)[..., None]
            res = f * (1 - m) + col * m
        elif out_mode == "darken":
            res = f * (1 - (a * amt)[..., None])
        elif out_mode == "invert":
            m = (a * amt)[..., None]
            res = f * (1 - m) + (1 - f) * m
        elif out_mode == "trails":
            res = np.clip(f + (ag * amt)[..., None] * col, 0, 1)
        else:
            cell_px = max(h, w) / float(p["cells"])
            dx = (ag - 0.5) * 2 * amt * cell_px * 6
            xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            moved = cv2.remap(np.ascontiguousarray(img), (xs + dx).astype(np.float32), ys,
                              interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP)
            return moved
        out = img.copy()
        out[..., :3] = to_uint8(res)
        return out


_LAPLACE = np.array([[0.05, 0.2, 0.05], [0.2, -1.0, 0.2], [0.05, 0.2, 0.05]], np.float32)


def gray_scott(u: np.ndarray, v: np.ndarray, feed: float, kill: float, iterations: int,
               du: float = 1.0, dv: float = 0.5, dt: float = 1.0, check=None, progress=None) -> tuple[np.ndarray, np.ndarray]:
    """Bounded explicit Gray-Scott integration (Karl Sims' formulation).

    U and V are diffused together with one 2-channel convolution; values are
    clamped to [0, 1] every step and tiny values are flushed to zero
    periodically so decaying regions never reach (very slow) denormal floats.
    """
    uv = np.ascontiguousarray(np.dstack([u, v]).astype(np.float32))
    u = uv[..., 0]
    v = uv[..., 1]
    uvv = np.empty_like(u)
    tmp = np.empty_like(u)
    fk = np.float32(feed + kill)
    feed = np.float32(feed)
    for i in range(iterations):
        if i % 100 == 0:
            if check:
                check()
            if progress:
                progress(i / max(1, iterations))
        if i % 20 == 0:
            uv[uv < 1e-20] = 0.0
        lap = cv2.filter2D(uv, -1, _LAPLACE, borderType=cv2.BORDER_REFLECT)
        np.multiply(v, v, out=uvv)
        uvv *= u
        # u += du*lap_u - uvv + feed*(1-u)
        np.subtract(1.0, u, out=tmp)
        tmp *= feed
        tmp -= uvv
        tmp += du * lap[..., 0]
        u += tmp * dt if dt != 1.0 else tmp
        # v += dv*lap_v + uvv - (feed+kill)*v
        np.multiply(v, fk, out=tmp)
        np.subtract(uvv, tmp, out=tmp)
        tmp += dv * lap[..., 1]
        v += tmp * dt if dt != 1.0 else tmp
        np.clip(uv, 0.0, 1.0, out=uv)
    u = uv[..., 0].copy()
    v = uv[..., 1].copy()
    if not (np.isfinite(u).all() and np.isfinite(v).all()):  # pragma: no cover - guarded by clipping
        u = np.nan_to_num(u)
        v = np.nan_to_num(v)
    return u, v


_RD_BLENDS = (("colorize", "Colourise"), ("multiply", "Multiply"), ("screen", "Screen"), ("overlay", "Overlay"),
              ("difference", "Difference"), ("displace", "Displace"))


@register
class ReactionDiffusion(Effect):
    id = "reaction_diffusion"
    name = "Reaction-Diffusion Texture"
    category = "Simulation"
    description = ("A real, bounded Gray-Scott simulation seeded from the image, grown for N iterations and "
                   "mapped back over the image. Feed/kill pick the pattern family (spots, worms, mazes, coral).")
    params = (
        iparam("iterations", "Iterations", 1500, 10, 10000),
        fparam("feed", "Feed (F)", 0.0545, 0.005, 0.1, decimals=4, step=0.0005,
               tooltip="Try F=0.0367,k=0.0649 (mitosis), 0.0545/0.062 (coral), 0.029/0.057 (mazes), 0.025/0.06 (spots)."),
        fparam("kill", "Kill (k)", 0.062, 0.03, 0.075, decimals=4, step=0.0005),
        iparam("sim_size", "Simulation size", 256, 64, 1024, tooltip="Grid long side; larger = finer pattern, slower."),
        cparam("seed_mode", "Seed from", "luminance", (("luminance", "Dark areas"), ("bright", "Bright areas"),
                                                         ("edges", "Edges"), ("random", "Random spots"))),
        fparam("seed_threshold", "Seed threshold", 0.45, 0.0, 1.0),
        cparam("blend", "Blend", "colorize", _RD_BLENDS),
        Param("color_a", "Colour A", COLOR, "#120018"),
        Param("color_b", "Colour B", COLOR, "#2fe0ff"),
        iparam("displace", "Displace distance", 20, 0, 500, spatial=True, unit="px", show_if=("blend", ("displace",))),
    )

    def process(self, img, p, ctx):
        h, w = img.shape[:2]
        gh, gw = _grid_size(h, w, p["sim_size"])
        small = cv2.resize(np.ascontiguousarray(img), (gw, gh), interpolation=cv2.INTER_AREA)
        lum = luminance(small)
        mode = p["seed_mode"]
        if mode == "luminance":
            seed = lum < p["seed_threshold"]
        elif mode == "bright":
            seed = lum > p["seed_threshold"]
        elif mode == "edges":
            e = np.hypot(cv2.Sobel(lum, cv2.CV_32F, 1, 0), cv2.Sobel(lum, cv2.CV_32F, 0, 1))
            seed = e > (p["seed_threshold"] * max(float(e.max()), 1e-6))
        else:
            seed = np.ones((gh, gw), bool)
        u = np.ones((gh, gw), np.float32)
        v = np.zeros((gh, gw), np.float32)
        # 3x3 seed blobs at seeded random positions inside the seed area
        # (single-cell seeds are dissolved by diffusion before they react)
        density = 0.02 if mode != "random" else 0.002 + 0.02 * p["seed_threshold"]
        spots = (seed & (ctx.rng.random((gh, gw)) < density)).astype(np.uint8)
        spots = cv2.dilate(spots, np.ones((3, 3), np.uint8)).astype(bool)
        if not spots.any():
            spots[gh // 2 - 1:gh // 2 + 2, gw // 2 - 1:gw // 2 + 2] = True
        u[spots] = 0.0
        v[spots] = 1.0
        u, v = gray_scott(u, v, p["feed"], p["kill"], int(p["iterations"]), check=ctx.check, progress=ctx.progress)
        pat = v / max(float(v.max()), 1e-6)
        pat = np.clip(cv2.resize(pat, (w, h), interpolation=cv2.INTER_CUBIC), 0.0, 1.0)
        f = to_float(img[..., :3])
        ca = np.array(color_to_rgb(p["color_a"]), np.float32) / 255.0
        cb = np.array(color_to_rgb(p["color_b"]), np.float32) / 255.0
        tex = ca + (cb - ca) * pat[..., None]
        mode = p["blend"]
        if mode == "displace":
            gx = cv2.Sobel(pat, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(pat, cv2.CV_32F, 0, 1, ksize=3)
            m = max(float(np.abs(gx).max()), float(np.abs(gy).max()), 1e-6)
            xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            return cv2.remap(np.ascontiguousarray(img), xs + gx / m * p["displace"], ys + gy / m * p["displace"],
                             interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        if mode == "colorize":
            res = tex
        elif mode == "multiply":
            res = f * tex
        elif mode == "screen":
            res = f + tex - f * tex
        elif mode == "overlay":
            res = np.where(f <= 0.5, 2 * f * tex, 1 - 2 * (1 - f) * (1 - tex))
        else:
            res = np.abs(f - tex)
        out = img.copy()
        out[..., :3] = to_uint8(res)
        return out
