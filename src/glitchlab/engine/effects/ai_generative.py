"""Generative AI effects: VAE latent bending and Stable Diffusion 1.5
inpainting / image-to-image, all running locally.  Generation results are
cached on disk; composition parameters only restyle cached results.  When a
model is missing the effect reports it -- it never substitutes a procedural
imitation.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ..params import COLOR, MASK, POINTS, TEXT, Param, bparam, cparam, fparam, iparam
from .ai_common import heat, hue_matrix, smooth_noise
from .base import Effect, register

CAT = "AI Experimental"
DEFAULT_NEGATIVE = "text, watermark, signature, frame, blurry, low quality"


def _gen():
    from ...ai import generate  # noqa: PLC0415

    return generate


def _analysis():
    from ...ai import analysis  # noqa: PLC0415

    return analysis


def _region(spec_dict: dict, w: int, h: int, source: np.ndarray) -> np.ndarray | None:
    from ..masks import rasterize_mask  # noqa: PLC0415 - avoids an import cycle with engine.model
    from ..model import MaskSpec  # noqa: PLC0415

    spec = MaskSpec.from_dict(spec_dict)
    return None if spec.is_identity() else rasterize_mask(spec, w, h, source)


def _res_param(default: int = 512):
    return iparam("resolution", "Generation size", default, 256, 1024, step=64, unit="px",
                  tooltip="Long side used by the model. Larger is slower (SD 1.5 works best at 512-768).")


def _auto_param():
    return bparam("auto", "Generate automatically", False,
                  tooltip="Off: the preview waits until you click 'Generate' (generation is slow on a processor).")


def _lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


# =====================================================================
# Latent Circuit Bending
# =====================================================================
MUTATIONS = (("tile_swap", "Latent tile swaps"), ("offset", "Spatial offsets"), ("permute", "Channel permutation"),
             ("amplify", "Local amplification"), ("quantise", "Quantisation"), ("noise", "Seeded noise"),
             ("mixed", "Mixed"))


def parse_channels(text: str, n: int) -> list[int]:
    t = (text or "").strip().lower()
    if t in ("", "all", "*"):
        return list(range(n))
    out = []
    for part in t.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    bad = [c for c in out if c < 0 or c >= n]
    if bad:
        raise ValueError(f"This model's latents have channels 0-{n - 1}; {bad} do not exist.")
    return sorted(set(out))


def mutate_latent(z: np.ndarray, kind: str, tile: int, channels: list[int], density: float, amplitude: float,
                  rng: np.random.Generator, region: np.ndarray | None) -> np.ndarray:
    """Apply a bounded, finite mutation to normalised latents z (C, H, W)."""
    c, h, w = z.shape
    t = max(1, int(tile))
    gh, gw = max(1, h // t), max(1, w // t)
    std = float(z.std()) or 1.0
    out = z.copy()
    chans = np.asarray(channels, int)
    sel = rng.random((gh, gw)) < density
    kinds = [kind] if kind != "mixed" else ["tile_swap", "offset", "permute", "amplify", "quantise", "noise"]
    for ti in range(gh):
        for tj in range(gw):
            if not sel[ti, tj]:
                continue
            k = kinds[int(rng.integers(0, len(kinds)))]
            y0, x0 = ti * t, tj * t
            y1, x1 = (h if ti == gh - 1 else y0 + t), (w if tj == gw - 1 else x0 + t)
            blk = out[chans, y0:y1, x0:x1]
            if k == "tile_swap":
                si, sj = int(rng.integers(0, gh)), int(rng.integers(0, gw))
                sy, sx = si * t, sj * t
                hh, ww = min(y1 - y0, h - sy), min(x1 - x0, w - sx)
                out[chans, y0:y0 + hh, x0:x0 + ww] = z[chans, sy:sy + hh, sx:sx + ww]
            elif k == "offset":
                d = max(1, int(round(amplitude * 3 * t)))
                dy, dx = int(rng.integers(-d, d + 1)), int(rng.integers(-d, d + 1))
                ys = np.clip(np.arange(y0, y1) + dy, 0, h - 1)
                xs = np.clip(np.arange(x0, x1) + dx, 0, w - 1)
                out[np.ix_(chans, ys - dy, xs - dx)] = z[np.ix_(chans, ys, xs)]
            elif k == "permute":
                perm = rng.permutation(len(chans))
                out[chans, y0:y1, x0:x1] = blk[perm]
            elif k == "amplify":
                gain = 1.0 + 3.0 * amplitude * (0.5 + rng.random())
                out[chans, y0:y1, x0:x1] = blk * gain
            elif k == "quantise":
                levels = max(2, int(round(24 - 21 * amplitude)))
                step = 6.0 * std / levels
                out[chans, y0:y1, x0:x1] = np.round(blk / step) * step
            elif k == "noise":
                out[chans, y0:y1, x0:x1] = blk + rng.standard_normal(blk.shape).astype(np.float32) * std * amplitude * 1.5
    if region is not None:
        m = region[None].astype(np.float32)
        out = z * (1 - m) + out * m
    lim = 8.0 * std
    return np.clip(np.nan_to_num(out, nan=0.0, posinf=lim, neginf=-lim), -lim, lim).astype(np.float32)


@register
class LatentCircuitBending(Effect):
    id = "latent_bending"
    name = "Latent Circuit Bending"
    category = CAT
    ai_models = ("vae",)
    supports_channels = False
    description = ("Genuine AI circuit bending (Stable Diffusion VAE, runs locally): the photo is encoded into the "
                   "model's latent space, the latent tensor is damaged -- tiles swapped or shifted, channels "
                   "permuted, regions amplified, quantised or injected with seeded noise -- and decoded again, "
                   "giving strange material transitions instead of pixel noise. The model weights are never "
                   "touched. 'Baseline' shows the undamaged reconstruction for comparison.")
    preview_note = "Latents are computed at the generation size; the decoded result is scaled to the image."
    params = (
        cparam("mutation", "Mutation type", "mixed", MUTATIONS),
        iparam("tile", "Latent tile size", 4, 1, 32, tooltip="In latent cells (one cell = 8 image pixels "
               "at the generation size for this model)."),
        Param("channels", "Affected channels", TEXT, "all", tooltip="'all', or a list such as 0,2 or 1-3. The "
              "number of latent channels comes from the model (4 for this VAE)."),
        fparam("density", "Mutation density", 0.3, 0.0, 1.0),
        fparam("amplitude", "Amplitude", 0.5, 0.0, 1.0),
        Param("region", "Mutation region", MASK, {}, tooltip="Optional: paint where the latents may be damaged."),
        cparam("composite", "Blend", "residual", (("decoded", "Decoded image"),
                                                  ("residual", "Damage only (keeps source detail)"))),
        _res_param(640),
        cparam("view", "Show", "result", (("result", "Result"), ("baseline", "Baseline VAE reconstruction"),
                                          ("damage", "Damage (difference to baseline)"))),
    )

    def validate(self, p):
        try:
            parse_channels(p["channels"], 64)
        except ValueError as exc:
            return [str(exc)]
        return []

    def process(self, img, p, ctx):
        h, w = img.shape[:2]
        gen = _gen()
        res = int(p["resolution"])
        enc = gen.vae_encode(ctx, img, res)
        z = enc["z"]
        try:
            chans = parse_channels(p["channels"], z.shape[0])
        except ValueError as exc:
            from ..errors import EffectError  # noqa: PLC0415

            raise EffectError(f"{self.name}: {exc}") from None
        base_rgb = enc["baseline"]
        if p["view"] == "baseline":
            out = img.copy()
            out[..., :3] = cv2.resize(base_rgb, (w, h), interpolation=cv2.INTER_CUBIC)
            return out
        reg = _region(p["region"], z.shape[2], z.shape[1], cv2.resize(img, (z.shape[2], z.shape[1]),
                                                                    interpolation=cv2.INTER_AREA))
        rng = ctx.sub_rng(1)
        zm = mutate_latent(z, p["mutation"], p["tile"], chans, p["density"], p["amplitude"], rng, reg)
        settings = {"m": p["mutation"], "t": p["tile"], "c": chans, "d": round(p["density"], 6),
                    "a": round(p["amplitude"], 6), "r": p["region"], "seed": int(ctx.sub_rng(2).integers(0, 2**31))}
        dec = gen.vae_decode(ctx, img, res, zm, settings)
        dec_up = cv2.resize(dec, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        base_up = cv2.resize(base_rgb, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        out = img.copy()
        if p["view"] == "damage":
            d = np.abs(dec_up - base_up).mean(-1) / 64.0
            out[..., :3] = heat(d)
            return out
        if p["composite"] == "decoded":
            rgb = dec_up
        else:
            rgb = img[..., :3].astype(np.float32) + (dec_up - base_up)
        out[..., :3] = np.clip(rgb + 0.5, 0, 255).astype(np.uint8)
        return out


# =====================================================================
# Hallucination Scars
# =====================================================================
def _walk(rng, start, heading, n_seg, step, jitter, aspect=1.0):
    """Random walk; ``step`` is a fraction of the image height (x is corrected by the aspect ratio)."""
    pts = [start]
    x, y = start
    for _ in range(n_seg):
        heading += rng.normal(0, jitter)
        x += math.cos(heading) * step * rng.uniform(0.6, 1.4) / max(aspect, 1e-6)
        y += math.sin(heading) * step * rng.uniform(0.6, 1.4)
        pts.append((x, y))
    return pts, heading


def scar_geometry(rng: np.random.Generator, shape: str, density: float, branching: float, aspect: float,
                  starts: np.ndarray | None = None, contours: list[np.ndarray] | None = None) -> dict:
    """Resolution-independent scar shapes in normalised coordinates.

    Returns {'lines': [(points (N, 2), width factor)], 'rects': [(cx, cy, w, h, angle)]}."""
    lines, rects = [], []
    kinds = [shape] if shape != "mixed" else ["cracks", "scanlines", "ribbons", "wounds"]
    share = 1.0 / len(kinds)

    def start_point():
        if starts is not None and len(starts):
            return tuple(starts[int(rng.integers(0, len(starts)))])
        return (float(rng.random()), float(rng.random()))

    for kind in kinds:
        if kind == "ribbons" and not contours:
            kind = "cracks"
        if kind == "cracks":
            n = max(1, int(round((2 + density * 16) * share)))
            stack = [(start_point(), float(rng.uniform(-math.pi, math.pi)), 0, rng.uniform(0.6, 1.4))
                     for _ in range(n)]
            budget = 400
            while stack and budget > 0:
                (sx, sy), head, depth, wf = stack.pop()
                pts, head = _walk(rng, (sx, sy), head, int(rng.integers(5, 16)), rng.uniform(0.015, 0.045), 0.45,
                                  aspect)
                budget -= len(pts)
                lines.append((np.asarray(pts, np.float32), float(wf)))
                if depth < 3:
                    for k in range(1, len(pts) - 1):
                        if rng.random() < branching * 0.3:
                            stack.append((pts[k], head + rng.choice([-1, 1]) * rng.uniform(0.4, 1.2), depth + 1,
                                          wf * rng.uniform(0.45, 0.8)))
        elif kind == "scanlines":
            n = max(1, int(round((3 + density * 24) * share)))
            for _ in range(n):
                y = float(rng.random())
                x = float(rng.uniform(-0.1, 0.6))
                end = x + rng.uniform(0.2, 1.0)
                wf = float(rng.uniform(0.5, 1.5))
                while x < end:
                    seg = rng.uniform(0.03, 0.25)
                    yy = y + rng.normal(0, 0.003)
                    lines.append((np.asarray([(x, yy), (min(x + seg, end), yy)], np.float32), wf))
                    x += seg + rng.uniform(0.005, 0.08)
        elif kind == "ribbons":
            n = max(1, int(round((2 + density * 10) * share)))
            for _ in range(n):
                c = contours[int(rng.integers(0, len(contours)))]
                if len(c) < 4:
                    continue
                L = int(max(4, len(c) * rng.uniform(0.1, 0.45)))
                s = int(rng.integers(0, max(1, len(c) - L)))
                lines.append((c[s:s + L].astype(np.float32), float(rng.uniform(0.8, 2.0))))
        elif kind == "wounds":
            n = max(1, int(round((2 + density * 10) * share)))
            for _ in range(n):
                cx, cy = start_point()
                ang = float(rng.uniform(0, 180))
                for _k in range(int(rng.integers(2, 7))):
                    rects.append((cx + rng.normal(0, 0.03), cy + rng.normal(0, 0.03), float(rng.uniform(0.01, 0.07)),
                                  float(rng.uniform(0.004, 0.03)), ang + rng.normal(0, 12)))
    return {"lines": lines, "rects": rects}


def rasterize_scars(geom: dict, w: int, h: int, width_px: float, *, min_width: float = 1.0) -> np.ndarray:
    """Anti-aliased scar mask (float32 0..1) at any resolution."""
    shift = 3
    mul = float(1 << shift)
    m = np.zeros((h, w), np.uint8)
    for pts, wf in geom["lines"]:
        thick = max(1, int(round(max(min_width, width_px * wf))))
        p = np.rint(np.asarray(pts, np.float64) * [w * mul, h * mul]).astype(np.int32)
        cv2.polylines(m, [p.reshape(-1, 1, 2)], False, 255, thickness=thick, lineType=cv2.LINE_AA, shift=shift)
    long_side = max(w, h)
    for cx, cy, rw, rh, ang in geom["rects"]:
        box = cv2.boxPoints(((cx * w, cy * h), (max(min_width, rw * long_side), max(min_width, rh * long_side)), ang))
        cv2.fillPoly(m, [np.rint(box * mul).astype(np.int32)], 255, lineType=cv2.LINE_AA, shift=shift)
    return m.astype(np.float32) / 255.0


def _contours_from_map(m: np.ndarray, levels: int = 5) -> list[np.ndarray]:
    h, w = m.shape
    out = []
    for k in range(1, levels):
        thr = (m > k / levels).astype(np.uint8)
        cs, _ = cv2.findContours(thr, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for c in cs:
            if len(c) >= 12:
                out.append(c[:, 0, :].astype(np.float32) / [w, h])
    return out


@register
class HallucinationScars(Effect):
    id = "hallucination_scars"
    name = "Hallucination Scars"
    category = CAT
    ai_models = ("sd15_inpaint",)
    heavy = True
    supports_channels = False
    description = ("Local AI inpainting (Stable Diffusion 1.5 Inpainting). Narrow tears -- branching cracks, broken "
                   "scanlines, contour ribbons or rectangular wounds -- reveal an impossible material described by "
                   "your prompt. Only the scars are replaced: outside them the photo stays exactly unchanged "
                   "(unless you add edge bleed or fragments).")
    preview_note = "Generation is slow on a processor: the preview waits for 'Generate' unless automatic."
    params = (
        Param("prompt", "Material prompt", TEXT, "oxidised copper circuitry embedded in cracked porcelain",
              flags=("long",), tooltip="Describe the material inside the scars, e.g. 'black liquid chrome with "
              "tangled luminous fibres' or 'photocopied botanical tissue and fractured glass'."),
        Param("negative", "Avoid", TEXT, DEFAULT_NEGATIVE, flags=("long",)),
        cparam("shape", "Scar shape", "cracks", (("cracks", "Branching cracks"), ("scanlines", "Broken scanlines"),
                                                 ("ribbons", "Contour-following ribbons"),
                                                 ("wounds", "Rectangular wounds"), ("mixed", "Mixed"))),
        cparam("guide", "Placement guide", "none", (("none", "Image contours / random"),
                                                    ("depth", "Depth discontinuities (Depth Anything)"),
                                                    ("objects", "Object boundaries (SAM 2)"))),
        Param("objects", "Objects", POINTS, [], show_if=("guide", ("objects",))),
        fparam("density", "Scar density", 0.4, 0.0, 1.0),
        iparam("width", "Width", 14, 1, 400, spatial=True, unit="px"),
        fparam("branching", "Branching", 0.5, 0.0, 1.0),
        cparam("fill", "Fill", "material", (("material", "Prompt material (two passes)"),
                                            ("context", "Context (heals the photo, one pass)")),
               tooltip="Material: the model first paints your material, then fits it into the scars. Context: "
                       "plain inpainting, which tends to continue the photo."),
        fparam("strength", "Denoising strength", 0.45, 0.1, 1.0,
               tooltip="Prompt material: how strongly the material is re-fitted to the photo (low keeps more of "
                       "the material). Context: use about 1.0."),
        fparam("guidance", "Guidance", 7.5, 1.0, 15.0, decimals=1),
        iparam("steps", "Steps", 20, 4, 50),
        fparam("interior", "Interior texture", 1.2, 0.3, 2.5, group="Scar"),
        fparam("saturation", "Interior colour", 1.1, 0.0, 2.0, group="Scar"),
        cparam("edge_style", "Scar edge", "burn", (("none", "None"), ("burn", "Burnt"), ("glow", "Glowing"),
                                                   ("chroma", "Chromatic")), group="Scar"),
        iparam("edge_width", "Edge width", 2, 0, 100, spatial=True, unit="px", group="Scar"),
        fparam("edge_strength", "Edge strength", 0.6, 0.0, 1.0, group="Scar"),
        Param("edge_colour", "Glow colour", COLOR, "#7af7ff", show_if=("edge_style", ("glow",)), group="Scar"),
        fparam("bleed", "Edge bleed", 0.0, 0.0, 1.0, group="Scar",
               tooltip="Lets the scar's colour bleed into the surrounding photo."),
        fparam("fragments", "Displaced fragments", 0.0, 0.0, 1.0, group="Scar",
               tooltip="Scatters tiny displaced chips of the material around the wounds."),
        _res_param(512),
        _auto_param(),
        cparam("view", "Show", "result", (("result", "Result"), ("mask", "Scar mask"), ("material", "Generated image"))),
    )

    def info(self, p):
        g = _gen()
        n = max(1, int(p["steps"] * p["strength"])) + (int(p["steps"]) if p["fill"] == "material" else 0)
        return g.estimate_text(n, int(p["resolution"]))

    def _geometry(self, ctx, img, p):
        rng = ctx.sub_rng(1)
        fw, fh = ctx.full_size
        starts, contours = None, None
        an = _analysis()
        if p["guide"] == "depth":
            d = an.robust_normalise(an.depth(ctx, img))
            gy, gx = np.gradient(cv2.GaussianBlur(d, (0, 0), 1.0))
            g = np.hypot(gx, gy)
            hh, ww = d.shape
            cand = np.argwhere(g > np.percentile(g, 92))
            starts = (cand[:, ::-1].astype(np.float32) + 0.5) / [ww, hh] if len(cand) else None
            contours = _contours_from_map(d, 7)
        elif p["guide"] == "objects":
            pts = p["objects"] or [[0.5, 0.5, 1, 1]]
            logits = an.object_logits(ctx, img, pts)
            m = (an.masks_from_logits(logits, 256, 256)).any(0).astype(np.uint8)
            cs, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            contours = [c[:, 0, :].astype(np.float32) / 256.0 for c in cs if len(c) >= 12]
            starts = np.vstack(contours) if contours else None
        else:
            small = cv2.resize(img, (256, max(1, int(256 * fh / max(fw, 1)))), interpolation=cv2.INTER_AREA)
            lum = cv2.GaussianBlur(small[..., :3].mean(-1).astype(np.float32) / 255.0, (0, 0), 2.0)
            contours = _contours_from_map(lum, 6)
        return scar_geometry(rng, p["shape"], p["density"], p["branching"], fw / max(fh, 1), starts, contours)

    def process(self, img, p, ctx):
        h, w = img.shape[:2]
        geom = self._geometry(ctx, img, p)
        width = float(p["width"])
        mask = rasterize_scars(geom, w, h, width)
        if p["view"] == "mask":
            out = img.copy()
            out[..., :3] = (img[..., :3] * (1 - mask[..., None] * 0.85) + np.float32([255, 40, 120]) * mask[..., None]
                            * 0.85).astype(np.uint8)
            return out
        width_full = width / max(ctx.scale, 1e-6)
        fw = ctx.full_size[0]

        def gen_mask(gw, gh):
            # The model needs room to invent a material: generate in a wide zone around each
            # scar; the result is composited through the exact (narrow) scar mask afterwards.
            s = gw / float(fw)
            zone = max(32.0, width_full * s * 3.0)
            m = rasterize_scars(geom, gw, gh, zone, min_width=zone)
            return (m > 0.1).astype(np.float32)

        settings = {"prompt": p["prompt"].strip(), "negative": p["negative"].strip(), "fill": p["fill"],
                    "strength": round(p["strength"], 4),
                    "guidance": round(p["guidance"], 3), "steps": int(p["steps"]), "res": int(p["resolution"]),
                    "seed": _gen().gen_seed(ctx), "geom": _geom_signature(geom), "width": round(width_full, 3)}
        res = _gen().inpaint(ctx, img, gen_mask, settings, auto=p["auto"])
        mat = cv2.resize(res["rgb"], (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        if p["view"] == "material":
            out = img.copy()
            out[..., :3] = np.clip(mat, 0, 255).astype(np.uint8)
            return out
        return _composite_scars(img, mat, mask, p, ctx)


def _geom_signature(geom: dict) -> str:
    import hashlib  # noqa: PLC0415

    h = hashlib.sha1()
    for pts, wf in geom["lines"]:
        h.update(np.round(np.asarray(pts, np.float32), 5).tobytes())
        h.update(np.float32(wf).tobytes())
    for r in geom["rects"]:
        h.update(np.round(np.asarray(r, np.float32), 5).tobytes())
    return h.hexdigest()


def _composite_scars(img, mat, mask, p, ctx):
    h, w = img.shape[:2]
    src = img[..., :3].astype(np.float32)
    # interior texture: local contrast and colour of the generated material
    mean = cv2.GaussianBlur(mat, (0, 0), max(1.0, p["width"] * 1.5))
    mat = mean + (mat - mean) * p["interior"]
    g = mat.mean(-1, keepdims=True)
    mat = g + (mat - g) * p["saturation"]
    ew = int(p["edge_width"])
    if p["edge_style"] != "none" and ew > 0:
        inner = cv2.erode(mask, np.ones((2 * ew + 1, 2 * ew + 1), np.uint8))
        edge = np.clip(mask - inner, 0, 1)[..., None]
        edge = edge * p["edge_strength"]
        if p["edge_style"] == "burn":
            mat = mat * (1 - 0.85 * edge)
        elif p["edge_style"] == "glow":
            from ..params import color_to_rgb  # noqa: PLC0415

            mat = mat + edge * (np.float32(color_to_rgb(p["edge_colour"])) - mat) * 0.85
        elif p["edge_style"] == "chroma":
            sh = max(1, ew)
            r = np.roll(mat[..., 0], sh, axis=1)
            b = np.roll(mat[..., 2], -sh, axis=1)
            mat = mat * (1 - edge) + np.dstack([r, mat[..., 1], b]) * edge
    m = mask[..., None]
    out = src * (1 - m) + mat * m      # exactly the original wherever the mask is 0
    if p["bleed"] > 0:
        sig = max(1.0, p["bleed"] * p["width"] * 4)
        num = cv2.GaussianBlur((mat - src) * m, (0, 0), sig)
        den = cv2.GaussianBlur(mask, (0, 0), sig)[..., None]
        spread = num / np.maximum(den, 1e-3) * np.clip(den * 3, 0, 1)
        out = out + spread * (1 - m) * p["bleed"] * 0.6
    if p["fragments"] > 0:
        rng = ctx.sub_rng(4)
        ys, xs = np.nonzero(mask > 0.5)
        if len(ys):
            n = int(p["fragments"] * 80)
            size = max(2, int(p["width"] * 1.2))
            for _ in range(n):
                k = int(rng.integers(0, len(ys)))
                y, x = int(ys[k]), int(xs[k])
                a = rng.uniform(0, 2 * math.pi)
                dist = p["width"] * rng.uniform(1.5, 6.0)
                ty, tx = int(y + math.sin(a) * dist), int(x + math.cos(a) * dist)
                s = int(max(1, size * rng.uniform(0.3, 1.0)))
                if 0 <= ty < h - s and 0 <= tx < w - s and y + s < h and x + s < w:
                    out[ty:ty + s, tx:tx + s] = mat[y:y + s, x:x + s]
    res = img.copy()
    res[..., :3] = np.clip(out + 0.5, 0, 255).astype(np.uint8)
    untouched = (mask <= 0) if (p["bleed"] <= 0 and p["fragments"] <= 0) else None
    if untouched is not None:
        res[untouched] = img[untouched]
    return res


# =====================================================================
# Counterfactual Residue
# =====================================================================
@register
class CounterfactualResidue(Effect):
    id = "counterfactual_residue"
    name = "Counterfactual Residue"
    category = CAT
    ai_models = ("sd15",)
    heavy = True
    supports_channels = False
    description = ("Local AI (Stable Diffusion 1.5 image-to-image). Two contradictory versions of the photo are "
                   "generated from the same noise -- one per prompt -- and fragments of both haunt the original "
                   "where they disagree. An artistic comparison, not an explanation of the model's reasoning.")
    preview_note = "Both interpretations are cached; composition changes are immediate."
    params = (
        Param("prompt_a", "Prompt A", TEXT, "frozen ceramic", flags=("long",)),
        Param("prompt_b", "Prompt B", TEXT, "molten organic metal", flags=("long",)),
        Param("negative", "Avoid", TEXT, DEFAULT_NEGATIVE, flags=("long",)),
        fparam("strength", "Generation strength", 0.55, 0.15, 0.95),
        fparam("guidance", "Guidance", 7.5, 1.0, 15.0, decimals=1),
        iparam("steps", "Steps", 20, 4, 50),
        fparam("feature_weight", "Feature difference", 0.0, 0.0, 1.0,
               tooltip="Mix learned-feature difference (DINOv2) into the colour difference map."),
        fparam("threshold", "Difference threshold", 0.35, 0.0, 1.0),
        fparam("visibility", "Residue visibility", 1.0, 0.0, 1.0),
        cparam("mode", "Composition", "threshold", (("threshold", "Hard threshold fragments"),
                                                    ("contour", "Contour bands"), ("strips", "Alternating strips"))),
        iparam("fragment", "Fragment size", 48, 2, 2000, spatial=True, unit="px"),
        iparam("contour_width", "Contour width", 12, 1, 400, spatial=True, unit="px", show_if=("mode", ("contour",))),
        iparam("displacement", "Displacement", 0, 0, 2000, spatial=True, unit="px"),
        Param("region", "Effect region", MASK, {}, tooltip="Optional: the original is kept outside this mask."),
        _res_param(512),
        _auto_param(),
        cparam("view", "Show", "result", (("result", "Result"), ("a", "Interpretation A"), ("b", "Interpretation B"),
                                          ("difference", "Difference map"))),
    )

    def info(self, p):
        return _gen().estimate_text(2 * max(1, int(p["steps"] * p["strength"])), int(p["resolution"]))

    def process(self, img, p, ctx):
        h, w = img.shape[:2]
        gen = _gen()
        settings = {"prompt_a": p["prompt_a"].strip(), "prompt_b": p["prompt_b"].strip(),
                    "negative": p["negative"].strip(), "strength": round(p["strength"], 4),
                    "guidance": round(p["guidance"], 3), "steps": int(p["steps"]), "res": int(p["resolution"]),
                    "seed": gen.gen_seed(ctx)}
        pair = gen.counterfactual_pair(ctx, img, settings, auto=p["auto"])
        A, B = pair["a"], pair["b"]
        dc = np.linalg.norm(_lab(A) - _lab(B), axis=-1)
        dc = dc / max(float(np.percentile(dc, 98)), 1e-3)
        D = np.clip(dc, 0, 1)
        if p["feature_weight"] > 0:
            D = (1 - p["feature_weight"]) * D + p["feature_weight"] * _feature_difference(ctx, img, A, B)
        D = cv2.GaussianBlur(D.astype(np.float32), (0, 0), 1.2)
        Du = cv2.resize(D, (w, h), interpolation=cv2.INTER_CUBIC)
        Au = cv2.resize(A, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        Bu = cv2.resize(B, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        out = img.copy()
        if p["view"] in ("a", "b"):
            out[..., :3] = np.clip(Au if p["view"] == "a" else Bu, 0, 255).astype(np.uint8)
            return out
        if p["view"] == "difference":
            out[..., :3] = heat(Du)
            return out
        thr = p["threshold"]
        reveal = np.clip((Du - thr) / 0.06 + 0.5, 0, 1)
        f = float(max(2, p["fragment"]))
        fw_full = ctx.full_size[0]
        xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        mode = p["mode"]
        if mode == "threshold":
            n1 = smooth_noise(ctx.sub_rng(1), w, h, ctx.full_size, f / max(ctx.scale, 1e-6) * 2)
            n2 = smooth_noise(ctx.sub_rng(2), w, h, ctx.full_size, f / max(ctx.scale, 1e-6) * 2)
            ci = np.floor((xs + n1 * f * 0.6) / f).astype(np.int64)
            cj = np.floor((ys + n2 * f * 0.6) / f).astype(np.int64)
            tbl = ctx.sub_rng(3).random(4096) < 0.5
            pick_a = tbl[(ci * 73856093 ^ cj * 19349663) % 4096]
        elif mode == "contour":
            L = max(3.0, (w / f) * 0.25)
            F = Du * L
            gy, gx = np.gradient(F)
            dist = np.abs(F - np.rint(F)) / (np.hypot(gx, gy) + 1e-6)
            line = np.clip(p["contour_width"] / 2.0 - dist + 0.5, 0, 1)
            reveal = reveal * line
            pick_a = (np.rint(F).astype(np.int64) % 2) == 0
        else:
            pick_a = (np.floor(xs / f).astype(np.int64) % 2) == 0
        res_rgb = np.where(pick_a[..., None], Au, Bu)
        if p["displacement"] > 0:
            gy, gx = np.gradient(cv2.GaussianBlur(Du, (0, 0), max(1.0, f * 0.25)))
            norm = np.hypot(gx, gy).max() or 1.0
            amp = p["displacement"] / norm
            res_rgb = cv2.remap(res_rgb.astype(np.float32), xs + gx * amp, ys + gy * amp, cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT_101)
        region = _region(p["region"], w, h, img)
        a = reveal * p["visibility"]
        if region is not None:
            a = a * region
        a3 = a[..., None]
        rgb = img[..., :3].astype(np.float32) * (1 - a3) + res_rgb * a3
        out[..., :3] = np.clip(rgb + 0.5, 0, 255).astype(np.uint8)
        keep = a <= 0
        out[keep] = img[keep]  # the original is preserved exactly outside the effect mask
        return out


def _feature_difference(ctx, img, A, B):
    from ...ai import analysis, support  # noqa: PLC0415

    def run(_full, job):
        model = support.load_dinov2()
        fa = analysis._dino_tokens(model, support.rgb_resized(A, *analysis.grid_for(A.shape[1], A.shape[0], 32)))
        fb = analysis._dino_tokens(model, support.rgb_resized(B, *analysis.grid_for(B.shape[1], B.shape[0], 32)))
        return {"d": (1.0 - np.sum(fa * fb, -1)).astype(np.float32)}

    d = support.cached_analysis(ctx, img, "residue-features", {"a": analysis.array_identity(A[::4, ::4]),
                                                               "b": analysis.array_identity(B[::4, ::4])},
                                run, model_ids=("dinov2",), label="Feature difference (DINOv2)")["d"]
    d = d / max(float(np.percentile(d, 98)), 1e-3)
    return np.clip(cv2.resize(d, (A.shape[1], A.shape[0]), interpolation=cv2.INTER_CUBIC), 0, 1)


# =====================================================================
# Recursive Misremembering
# =====================================================================
def parse_selection(pattern: str, custom: str, n: int) -> list[int]:
    if pattern == "last":
        return [n]
    if pattern == "every_other":
        return list(range(2, n + 1, 2)) or [n]
    if pattern == "first_last":
        return sorted({1, n})
    if pattern == "custom":
        out = []
        for part in (custom or "").replace(";", ",").split(","):
            part = part.strip()
            if part.isdigit() and 1 <= int(part) <= n:
                out.append(int(part))
        return sorted(set(out)) or [n]
    return list(range(1, n + 1))


@register
class RecursiveMisremembering(Effect):
    id = "recursive_misremembering"
    name = "Recursive Misremembering"
    category = CAT
    ai_models = ("sd15",)
    heavy = True
    supports_channels = False
    description = ("Local AI (Stable Diffusion 1.5 image-to-image, repeated). The image is remembered and "
                   "reconstructed again and again at low strength, each time from the previous memory, slowly "
                   "growing false details, drifting textures and a fragmented identity. Selected generations "
                   "are then combined into one still. Repeatable on the same software and hardware.")
    preview_note = "Every generation is cached; an interrupted run resumes from the last finished generation."
    params = (
        Param("prompts", "Prompts (one per line)", TEXT, "a faded photograph remembered wrongly",
              flags=("long",), tooltip="One prompt per line; the schedule decides which line each generation uses."),
        cparam("schedule", "Prompt schedule", "constant", (("constant", "Constant (first line)"),
                                                           ("alternate", "Alternating"),
                                                           ("progressive", "Progressive (spread over generations)"))),
        Param("negative", "Avoid", TEXT, DEFAULT_NEGATIVE, flags=("long",)),
        iparam("generations", "Generation count", 4, 1, 12),
        fparam("strength", "Per-generation strength", 0.3, 0.05, 0.7),
        fparam("anchor", "Original anchor", 0.15, 0.0, 0.9, tooltip="Blends the original back in before every "
               "generation."),
        fparam("guidance", "Guidance", 7.0, 1.0, 15.0, decimals=1),
        iparam("steps", "Steps", 20, 4, 50),
        cparam("seed_schedule", "Seed schedule", "increment", (("fixed", "Same seed"), ("increment", "Seed + 1"),
                                                               ("random", "Seeded random"))),
        Param("protect", "Protected region", MASK, {}, tooltip="Restored exactly after every generation."),
        cparam("memory", "Mode", "accumulated", (("accumulated", "Accumulated memory"),
                                                 ("collision", "Memory collision"))),
        cparam("combine", "Combine generations by", "contour", (("contour", "Contour bands"),
                                                                ("mosaic", "Patch mosaic"),
                                                                ("disagreement", "Disagreement masks"))),
        cparam("selection", "Generation selection", "all", (("all", "All"), ("last", "Last"),
                                                            ("every_other", "Every other"),
                                                            ("first_last", "First and last"), ("custom", "Custom"))),
        Param("custom", "Generations (e.g. 1,3,4)", TEXT, "", show_if=("selection", ("custom",))),
        iparam("fragment", "Fragment size", 64, 4, 2000, spatial=True, unit="px"),
        _res_param(512),
        _auto_param(),
        cparam("view", "Show", "result", (("result", "Result"), ("last", "Last generation"),
                                          ("strip", "All generations side by side"))),
    )

    def validate(self, p):
        total = _gen().total_steps(p["generations"], p["steps"], p["strength"])
        if total > 400:
            return [f"This would run {total} denoising steps; the limit is 400. Use fewer generations or steps."]
        return []

    def info(self, p):
        return _gen().estimate_text(_gen().total_steps(p["generations"], p["steps"], p["strength"]),
                                    int(p["resolution"]))

    def process(self, img, p, ctx):
        h, w = img.shape[:2]
        gen = _gen()
        prompts = [ln.strip() for ln in p["prompts"].splitlines() if ln.strip()] or [""]
        protect = p["protect"]
        settings = {"prompts": prompts, "schedule": p["schedule"], "negative": p["negative"].strip(),
                    "generations": int(p["generations"]), "strength": round(p["strength"], 4),
                    "anchor": round(p["anchor"], 4), "guidance": round(p["guidance"], 3), "steps": int(p["steps"]),
                    "seed_schedule": p["seed_schedule"], "seed": gen.gen_seed(ctx), "res": int(p["resolution"]),
                    "protect": protect}

        def protect_fn(gw, gh, rgb0):
            rgba = np.dstack([rgb0, np.full(rgb0.shape[:2], 255, np.uint8)])
            return _region(protect, gw, gh, rgba)

        gens = gen.misremember(ctx, img, settings, protect_fn, auto=p["auto"])
        n = len(gens) - 1
        out = img.copy()
        if p["view"] == "last":
            out[..., :3] = cv2.resize(gens[-1], (w, h), interpolation=cv2.INTER_CUBIC)
        elif p["view"] == "strip":
            k = len(gens)
            cols = [cv2.resize(g, (max(1, w // k), h), interpolation=cv2.INTER_AREA) for g in gens]
            strip = np.concatenate(cols, 1)
            out[:, :strip.shape[1], :3] = strip
        else:
            sel = parse_selection(p["selection"], p["custom"], n)
            comp = _combine(gens, sel, p, ctx, w, h)
            out[..., :3] = comp
        region = _region(protect, w, h, img)
        if region is not None:  # the protected region is the exact original
            keep = region > 0.5
            out[keep] = img[keep]
        return out


def _combine(gens, sel, p, ctx, w, h):
    G = [cv2.resize(gens[i], (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32) for i in sel]
    orig = cv2.resize(gens[0], (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    k = len(G)
    if k == 1:
        return np.clip(G[0] + 0.5, 0, 255).astype(np.uint8)
    acc = p["memory"] == "accumulated"
    f = float(max(4, p["fragment"]))
    if p["combine"] == "contour":
        drift = np.linalg.norm(_lab(np.clip(G[-1], 0, 255).astype(np.uint8)) - _lab(np.clip(orig, 0, 255).astype(
            np.uint8)), axis=-1)
        drift = cv2.GaussianBlur(drift / max(float(np.percentile(drift, 98)), 1e-3), (0, 0), max(1.0, f * 0.15))
        noise = smooth_noise(ctx.sub_rng(1), w, h, ctx.full_size, f / max(ctx.scale, 1e-6) * 3)
        bands = np.floor(np.clip(drift, 0, 1.5) * (w / f) * 0.15 + noise * 0.8).astype(np.int64)
        if acc:  # later memories dominate: index grows with drift
            idx = np.clip(bands, 0, k - 1)
        else:
            idx = np.mod(bands, k)
    elif p["combine"] == "mosaic":
        fw, fh = ctx.full_size
        f_full = f / max(ctx.scale, 1e-6)
        gw, gh = max(1, int(round(fw / f_full))), max(1, int(round(fh / f_full)))
        rng = ctx.sub_rng(2)
        weights = (np.arange(k) + 1.0) ** 2 if acc else np.ones(k)
        cells = rng.choice(k, size=(gh, gw), p=weights / weights.sum())
        idx = cv2.resize(cells.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int64)
    else:  # disagreement masks
        stack = np.stack(G)
        mean = stack.mean(0)
        dev = np.linalg.norm(stack - mean, axis=-1)
        dev = np.stack([cv2.GaussianBlur(d, (0, 0), max(1.0, f * 0.2)) for d in dev])
        if acc:
            dev = dev * (np.arange(k) + 1.0)[:, None, None]
        idx = dev.argmax(0)
    out = np.zeros((h, w, 3), np.float32)
    for i in range(k):
        m = idx == i
        out[m] = G[i][m]
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


_ = hue_matrix
