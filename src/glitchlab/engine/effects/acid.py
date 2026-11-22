"""Acid Flow: the photograph liquefies into flowing psychedelic material.

Artistic flow advection (see :mod:`glitchlab.engine.flow`), not a fluid
simulation: a velocity field built from curl noise, image contours, gravity,
vortices and painted strokes moves the image coordinates for a bounded
number of steps; finishing stages then add chromatic separation along the
flow, colour ribbons, iridescence, caustic highlights, liquid chrome and
corrosive rims.  All spatial controls are fractions of the image's long
side, so the reduced preview and the export look the same.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .. import flow
from ..ops import luminance
from ..params import COLOR, DATA, MASK, POINTS, Param, bparam, cparam, fparam, iparam
from .base import Effect, register

PALETTES = {
    "neon": ("Neon", ["#ff2d95", "#7a00ff", "#00f0ff", "#b8ff00", "#ffb000", "#ff2d95"]),
    "acid": ("Acid", ["#39ff14", "#ff00ff", "#00e5ff", "#fff200", "#ff3d00", "#39ff14"]),
    "toxic": ("Toxic ink", ["#020403", "#08140c", "#0f3d1c", "#7dff00", "#00ffa2", "#e6ff3d"]),
    "mercury": ("Mercury", ["#0e1014", "#3c434c", "#8a939d", "#d9dee5", "#ffffff", "#9aa3ad"]),
    "oil": ("Oil film", ["#1b1035", "#0b5d6e", "#2f8f5b", "#a6c34a", "#e0b84c", "#c03a83", "#1b1035"]),
}
MOVEMENT = ("mode", "strength", "direction", "gravity", "turbulence", "turbulence_scale", "turbulence_detail",
            "vortex_count", "vortex_size", "vortex_rotation", "vortices", "contour", "melt_threshold", "viscosity",
            "steps", "strokes")
GUIDES = (("none", "None (procedural only)"), ("contain", "Subject melts, background stays (SAM 2)"),
          ("spill", "Subject spills outward (SAM 2)"), ("depth", "Depth: foreground moves more"),
          ("depth_inverse", "Depth: background moves more"))


def _hex(c: str) -> np.ndarray:
    from ..params import color_to_rgb  # noqa: PLC0415

    return np.array(color_to_rgb(c), np.float32)


def _gradient_map(t: np.ndarray, stops: list[str]) -> np.ndarray:
    cols = np.array([_hex(s) for s in stops], np.float32)
    xp = np.linspace(0.0, 1.0, len(stops))
    t = np.mod(t, 1.0)
    return np.stack([np.interp(t, xp, cols[:, c]) for c in range(3)], -1).astype(np.float32)


@register
class AcidFlow(Effect):
    id = "acid_flow"
    name = "Acid Flow"
    category = "Abstract & Style"
    supports_alpha = True
    supports_channels = False
    description = ("The photograph liquefies: colours and textures stretch into ribbons, swirl into pools and "
                   "melt along their edges, with oily interference, corrosive rims and liquid chrome. Contour "
                   "Flow follows the subject's edges, Corrosive Melt dissolves bright (or painted) areas into "
                   "streams, Vortex Pools pull details into spirals, Liquid Acid does all of it at once. This is "
                   "artistic flow advection, not a fluid simulation.")
    preview_note = "All sizes are fractions of the image, so the preview and the export match."
    params = (
        # ------------------------------------------------------------ Movement
        cparam("mode", "Behaviour", "contour", (("contour", "Contour Flow"), ("melt", "Corrosive Melt"),
                                                ("vortex", "Vortex Pools"), ("liquid_acid", "Liquid Acid (intense)")),
               group="Movement"),
        fparam("strength", "Flow strength", 0.08, 0.0, 0.5, decimals=3, group="Movement",
               tooltip="How far the liquid travels, as a fraction of the image's long side."),
        fparam("direction", "Flow direction", 90.0, -180.0, 180.0, decimals=0, unit="°", group="Movement",
               tooltip="Direction of gravity and of melting drips (90° = down)."),
        fparam("gravity", "Gravity", 0.15, 0.0, 1.0, group="Movement"),
        fparam("turbulence", "Turbulence", 0.6, 0.0, 1.5, group="Movement"),
        fparam("turbulence_scale", "Turbulence scale", 0.25, 0.02, 1.0, decimals=3, group="Movement"),
        fparam("turbulence_detail", "Turbulence detail", 0.5, 0.0, 1.0, group="Movement"),
        iparam("vortex_count", "Vortex count", 3, 0, 24, group="Movement",
               tooltip="Seeded vortices, in addition to the ones you place."),
        fparam("vortex_size", "Vortex size", 0.16, 0.02, 0.6, decimals=3, group="Movement"),
        fparam("vortex_rotation", "Vortex rotation", 0.8, -2.0, 2.0, group="Movement",
               tooltip="Spin strength; negative turns the seeded vortices the other way."),
        Param("vortices", "Vortex centres", POINTS, [], group="Movement",
              tooltip="Pick on image: left click = anticlockwise vortex, right click = clockwise. Drag a centre "
                      "to move it."),
        fparam("contour", "Contour attraction", 0.6, 0.0, 1.5, group="Movement"),
        fparam("melt_threshold", "Melt threshold", 0.55, 0.0, 1.0, group="Movement",
               tooltip="Corrosive Melt: areas brighter than this dissolve (or paint a Liquid area)."),
        fparam("viscosity", "Viscosity", 0.35, 0.0, 1.0, group="Movement",
               tooltip="Apparent viscosity: smooths and damps the flow."),
        iparam("steps", "Advection steps", 16, 1, 64, group="Movement"),
        Param("strokes", "Flow strokes", MASK, {}, group="Movement", flags=("strokes",),
              tooltip="Paint strokes on the image: the liquid flows along them in the direction you drew."),
        Param("region", "Liquid area", MASK, {}, group="Movement",
              tooltip="Optional: paint where the image becomes liquid (default: everywhere)."),
        Param("protect", "Protected area", MASK, {}, group="Movement",
              tooltip="Painted areas stay exactly unchanged; the edge is feathered outside them."),
        fparam("feather", "Protection feather", 0.03, 0.0, 0.2, decimals=3, group="Movement"),
        cparam("edge", "Edges", "mirror", (("mirror", "Mirror"), ("clamp", "Clamp"), ("wrap", "Wrap")),
               group="Movement"),
        cparam("guide", "AI guidance (optional)", "none", GUIDES, group="Movement"),
        Param("objects", "Subject", POINTS, [], group="Movement", show_if=("guide", ("contain", "spill"))),
        bparam("freeze", "Freeze flow", False, group="Movement",
               tooltip="Keeps the current flow state; colour and material stay adjustable."),
        Param("frozen", "Frozen flow", DATA, {}, group="Movement"),
        # ------------------------------------------------------------ Material
        fparam("detail", "Source-detail retention", 0.35, 0.0, 1.0, group="Material"),
        fparam("fine", "Fine turbulence", 0.3, 0.0, 1.0, group="Material"),
        fparam("chrome", "Liquid chrome", 0.0, 0.0, 1.0, group="Material"),
        fparam("corrosion", "Corrosive edges", 0.2, 0.0, 1.0, group="Material"),
        fparam("highlights", "Caustic highlights", 0.35, 0.0, 1.5, group="Material"),
        # ------------------------------------------------------------ Colour
        cparam("colour_mode", "Colour", "original", (("original", "Original colours"),
                                                     ("neon", "Neon colourisation")), group="Colour"),
        cparam("palette", "Palette", "neon", tuple((k, v[0]) for k, v in PALETTES.items()), group="Colour",
               show_if=("colour_mode", ("neon",))),
        fparam("ribbons", "Colour ribbons", 0.25, 0.0, 1.0, group="Colour",
               tooltip="Palette compression: colours band into stretched ribbons."),
        iparam("ribbon_count", "Ribbon count", 10, 3, 32, group="Colour"),
        fparam("chroma", "Chromatic separation", 0.006, 0.0, 0.04, decimals=4, group="Colour",
               tooltip="RGB split sampled along the local flow direction (fraction of the long side)."),
        fparam("iridescence", "Iridescence", 0.25, 0.0, 1.0, group="Colour"),
        Param("highlight_colour", "Highlight colour", COLOR, "#f4fbff", group="Colour"),
        fparam("saturation", "Saturation", 1.1, 0.0, 2.0, group="Colour"),
    )

    def disabled_params(self, p):
        return set(MOVEMENT) if p.get("freeze") and p.get("frozen") else set()

    def movement_params(self, p: dict, ctx) -> tuple[dict, int]:
        """Movement settings and seed, from the frozen snapshot when frozen."""
        seed = int(ctx.sub_rng(1).integers(0, 2**31 - 1))
        if p["freeze"] and p["frozen"]:
            q = dict(p)
            fr = p["frozen"]
            for k in MOVEMENT:
                if k in fr:
                    q[k] = self.param(k).coerce(fr[k])
            return q, int(fr.get("seed", seed))
        return p, seed

    @staticmethod
    def snapshot(p: dict, effect_seed: int, global_seed: int) -> dict:
        """What 'Freeze flow' stores (called by the UI)."""
        from .base import make_seed_sequence  # noqa: PLC0415

        ss = make_seed_sequence(global_seed, effect_seed)
        rng = np.random.default_rng(np.random.SeedSequence(entropy=ss.entropy, spawn_key=(1,)))
        out = {k: p[k] for k in MOVEMENT}
        out["seed"] = int(rng.integers(0, 2**31 - 1))
        return out

    # ------------------------------------------------------------------
    def process(self, img, p, ctx):
        h, w = img.shape[:2]
        q, seed = self.movement_params(p, ctx)
        res, maps = self._flow(img, q, seed, ctx)
        return self._finish(img, p, res, maps, ctx)

    def _flow(self, img, q, seed, ctx):
        h, w = img.shape[:2]
        gw, gh = flow.grid_shape(w, h)
        long = float(max(gw, gh))
        rng = np.random.default_rng(seed)
        small = cv2.resize(img, (gw, gh), interpolation=cv2.INTER_AREA)
        lum = luminance(small)
        mode = q["mode"]
        acid = mode == "liquid_acid"
        turb = float(q["turbulence"])
        cvx, cvy = flow.curl_noise(rng, gw, gh, q["turbulence_scale"], 1 + int(round(q["turbulence_detail"] * 4)),
                                   q["turbulence_detail"])
        tang, edge, attract = flow.contour_field(lum, 0.004)
        a = math.radians(q["direction"])
        gx, gy = math.cos(a), math.sin(a)
        # vortices: user-placed + seeded
        centres = [(float(x), float(y), (1.0 if lab else -1.0) * max(0.2, abs(q["vortex_rotation"])))
                   for x, y, lab, _obj in q["vortices"]]
        for _ in range(int(q["vortex_count"])):
            centres.append((float(rng.uniform(0.08, 0.92)), float(rng.uniform(0.08, 0.92)),
                            float(q["vortex_rotation"]) * float(rng.choice([-1.0, 1.0]) if rng.random() < 0.35 else 1.0)
                            * float(rng.uniform(0.6, 1.3))))
        pull = 0.8 if mode in ("vortex", "liquid_acid") else 0.25
        vvx, vvy = flow.vortex_field(gw, gh, centres, q["vortex_size"], pull)
        svx, svy = flow.stroke_field(gw, gh, (q["strokes"] or {}).get("shapes", []))
        c = float(q["contour"])
        melt = None
        if mode == "contour":
            vx = turb * 0.35 * cvx + c * 1.2 * tang[0] + c * 0.4 * attract[0] + 0.5 * vvx
            vy = turb * 0.35 * cvy + c * 1.2 * tang[1] + c * 0.4 * attract[1] + 0.5 * vvy
        elif mode == "vortex":
            vx = turb * 0.45 * cvx + 1.7 * vvx + c * 0.3 * tang[0]
            vy = turb * 0.45 * cvy + 1.7 * vvy + c * 0.3 * tang[1]
        else:
            thr = float(q["melt_threshold"])
            melt = np.clip((lum - (thr - 0.1)) / 0.2, 0, 1).astype(np.float32)
            melt = cv2.GaussianBlur(melt, (0, 0), long * 0.01)
            svx_, svy_ = flow.curl_noise(rng, gw, gh, q["turbulence_scale"] * 0.6, 2, q["turbulence_detail"],
                                         anisotropy=(gx, gy, long * 0.12))
            drip = 1.0 + 0.6 * np.clip(flow.value_noise(rng, gw, gh, 0.05), -1, 1)
            mvx = melt * (gx * drip + 0.7 * svx_)
            mvy = melt * (gy * drip + 0.7 * svy_)
            if mode == "melt":
                vx = 1.5 * mvx + turb * 0.25 * cvx + c * 0.3 * tang[0]
                vy = 1.5 * mvy + turb * 0.25 * cvy + c * 0.3 * tang[1]
            else:  # liquid acid: everything, stronger
                vx = 0.9 * mvx + turb * 0.9 * cvx + c * 0.9 * tang[0] + 1.5 * vvx
                vy = 0.9 * mvy + turb * 0.9 * cvy + c * 0.9 * tang[1] + 1.5 * vvy
        vx = vx + q["gravity"] * gx + 1.6 * svx
        vy = vy + q["gravity"] * gy + 1.6 * svy
        # where the image may move
        region = _mask(q, "region", gw, gh, small)
        if region is not None:
            vx, vy = vx * region, vy * region
        protect = _mask(q, "protect", gw, gh, small)
        if protect is not None:
            vx, vy = vx * (1 - protect), vy * (1 - protect)
        guide = q.get("guide", "none") if "guide" in q else "none"
        ai_mask = None
        if guide in ("contain", "spill"):
            from ...ai import analysis  # noqa: PLC0415 - optional AI, cached inference

            pts = q.get("objects") or [[0.5, 0.5, 1, 1]]
            m = analysis.masks_from_logits(analysis.object_logits(ctx, img, pts), gw, gh).any(0).astype(np.float32)
            k = max(1, int(long * 0.02))
            if guide == "contain":
                m = cv2.erode(m, np.ones((k, k), np.uint8))
            else:
                m = cv2.dilate(m, np.ones((3 * k, 3 * k), np.uint8))
            m = cv2.GaussianBlur(m, (0, 0), k)
            vx, vy = vx * m, vy * m
            ai_mask = m
        elif guide in ("depth", "depth_inverse"):
            from ...ai import analysis  # noqa: PLC0415

            d = analysis.robust_normalise(cv2.resize(analysis.depth(ctx, img), (gw, gh),
                                                     interpolation=cv2.INTER_LINEAR))
            if guide == "depth_inverse":
                d = 1.0 - d
            vx, vy = vx * (0.2 + d), vy * (0.2 + d)
        speed = np.sqrt(vx * vx + vy * vy)
        norm = float(np.percentile(speed, 97)) or 1.0
        vx, vy = (vx / norm).astype(np.float32), (vy / norm).astype(np.float32)
        distance = float(q["strength"]) * long * (1.6 if acid else 1.0)
        res = flow.advect(vx, vy, distance, int(q["steps"]), float(q["viscosity"]), cancel=ctx.cancel)
        # analysis maps on the grid for the finishing stages
        mag = np.sqrt(res.dx ** 2 + res.dy ** 2) / long
        det = flow.deformation(res.dx, res.dy)
        # caustic-like glints where the deformation compresses the image (thin ridges, not blobs)
        comp = np.clip(1.0 / np.maximum(det, 0.05) - 1.0, 0, 4).astype(np.float32)
        comp = cv2.GaussianBlur(comp, (0, 0), 0.8)
        ridge = np.clip(comp - cv2.GaussianBlur(comp, (0, 0), 3.0), 0, None)
        caustic = np.clip(ridge * 1.6 + comp * 0.15, 0, 1.5)
        sp = np.sqrt(vx * vx + vy * vy) + 1e-6
        frng = np.random.default_rng(seed + 7)
        fvx, fvy = flow.curl_noise(frng, gw, gh, max(0.02, q["turbulence_scale"] * 0.12), 2, 0.6)
        speed_n = np.clip(sp, 0, 1.5)
        gyy, gxx = np.gradient(cv2.GaussianBlur(mag.astype(np.float32), (0, 0), 1.0))
        rim = np.sqrt(gxx * gxx + gyy * gyy)
        rim = np.clip(rim / (np.percentile(rim, 98) + 1e-6), 0, 1)
        maps = {"mag": mag.astype(np.float32), "caustic": caustic, "ux": (vx / sp).astype(np.float32),
                "uy": (vy / sp).astype(np.float32), "fx": (fvx * speed_n).astype(np.float32),
                "fy": (fvy * speed_n).astype(np.float32), "edge": edge, "rim": rim.astype(np.float32),
                "nx": gxx.astype(np.float32), "ny": gyy.astype(np.float32), "ai": ai_mask}
        return res, maps

    def _finish(self, img, p, res, maps, ctx):
        h, w = img.shape[:2]
        gw, gh = res.grid
        L = float(max(w, h))
        border = flow.EDGE_MODES[p["edge"]]
        sx, sy = w / float(gw), h / float(gh)
        out = np.empty_like(img)
        # source detail (high-pass of the original at the output position)
        sig = max(0.6, 0.0025 * L)
        blur = cv2.GaussianBlur(img, (0, 0), sig)
        region = _mask(p, "region", w, h, img)
        prot_w = _protect_weight(p, w, h, img)
        ai_r = None if maps["ai"] is None or p["guide"] != "contain" else \
            cv2.resize(maps["ai"], (w, h), interpolation=cv2.INTER_LINEAR)
        neon = p["colour_mode"] == "neon"
        stops = PALETTES[p["palette"]][1]
        n_rib = float(p["ribbon_count"])
        hl = _hex(p["highlight_colour"])
        chroma = float(p["chroma"]) * L
        fine = float(p["fine"]) * 0.004 * L
        strip = max(64, int(4_000_000 // max(w, 1)))
        for y0 in range(0, h, strip):
            ctx.check()
            y1 = min(h, y0 + strip)
            xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(y0, y1, dtype=np.float32))
            g = {k: flow.sample_grid(maps[k], w, h, y0, y1) for k in ("mag", "caustic", "ux", "uy", "fx", "fy",
                                                                      "edge", "rim", "nx", "ny")}
            mx = xs + flow.sample_grid(res.dx, w, h, y0, y1) * sx + g["fx"] * fine
            my = ys + flow.sample_grid(res.dy, w, h, y0, y1) * sy + g["fy"] * fine
            base = flow.premultiplied_remap(img, mx, my, border)
            rgb = base[..., :3].copy()
            # Colour that flowed in from fully transparent areas is undefined: keep the pixel's own colour
            # so that a preserved alpha channel never shows dark fringes.
            empty = base[..., 3] < 1.0
            if empty.any():
                rgb[empty] = img[y0:y1, :, :3][empty]
            if chroma > 0:
                for ch, sgn in ((0, 1.0), (2, -1.0)):
                    s = flow.premultiplied_remap(img, mx + sgn * chroma * g["ux"], my + sgn * chroma * g["uy"], border)
                    rgb[..., ch] = s[..., ch]
            lum = (rgb @ np.float32([0.299, 0.587, 0.114])) / 255.0
            mag = g["mag"]
            # colour: neon colourisation and/or palette compression ribbons
            if neon:
                t = lum * 0.85 + mag * 3.0
                if p["ribbons"] > 0:
                    tq = np.floor(t * n_rib) / n_rib
                    t = t * (1 - p["ribbons"]) + tq * p["ribbons"]
                col = _gradient_map(t, stops)
                rgb = rgb * 0.3 + col * (0.2 + 0.8 * np.clip(lum * 1.25, 0, 1))[..., None] * 0.75
            elif p["ribbons"] > 0:
                q = np.round(rgb / 255.0 * (n_rib / 2.0)) / (n_rib / 2.0) * 255.0
                bands = (np.floor(lum * n_rib + mag * n_rib * 2.0) % 2)[..., None]
                q = q * (0.88 + 0.24 * bands)
                rgb = rgb * (1 - p["ribbons"]) + q * p["ribbons"]
            if p["iridescence"] > 0:
                t = mag * 7.0 + g["edge"] * 1.5 + lum * 0.6
                film = 0.5 + 0.5 * np.cos(2 * np.pi * (t[..., None] + np.float32([0.0, 0.33, 0.67])))
                scr = 255.0 - (255.0 - rgb) * (1.0 - film * 0.9)
                k = p["iridescence"] * np.clip(0.35 + mag * 12.0, 0, 1)[..., None]
                rgb = rgb * (1 - k) + scr * k
            if p["chrome"] > 0:
                nx, ny = g["nx"] * 40.0, g["ny"] * 40.0
                env = 0.5 + 0.5 * np.tanh((ny * 0.8 + nx * 0.4) * 3.0 + (lum - 0.5) * 2.5)
                bands = 0.5 + 0.5 * np.cos((env * 3.0 + mag * 6.0) * np.pi)
                chrome = (0.15 + 0.85 * (env * 0.7 + bands * 0.3))[..., None] * np.float32([228, 234, 242])
                rgb = rgb * (1 - p["chrome"]) + chrome * p["chrome"]
            if p["corrosion"] > 0:
                rim = g["rim"][..., None] * p["corrosion"]
                tint = np.float32([150, 255, 40]) if neon else rgb * 0.25
                rgb = rgb * (1 - rim * 0.8) + tint * rim * 0.8
            if p["highlights"] > 0:
                c = np.clip((g["caustic"] * p["highlights"]) ** 1.5 * 0.8, 0, 0.85)[..., None]
                rgb = 255.0 - (255.0 - rgb) * (1.0 - c * hl / 255.0)
            if p["detail"] > 0:
                hp = img[y0:y1, :, :3].astype(np.float32) - blur[y0:y1, :, :3].astype(np.float32)
                rgb = rgb + hp * p["detail"] * 1.4
            if p["saturation"] != 1.0:
                gry = rgb.mean(-1, keepdims=True)
                rgb = gry + (rgb - gry) * p["saturation"]
            res_px = np.concatenate([np.clip(rgb, 0, 255), base[..., 3:4]], -1)
            orig = img[y0:y1].astype(np.float32)
            keep = None
            if region is not None:
                keep = 1.0 - region[y0:y1]
            if ai_r is not None:
                k2 = 1.0 - np.clip(ai_r[y0:y1] * 1.5, 0, 1)
                keep = k2 if keep is None else np.maximum(keep, k2)
            if prot_w is not None:
                pw = prot_w[y0:y1]
                keep = pw if keep is None else np.maximum(keep, pw)
            if keep is not None:
                res_px = res_px * (1 - keep[..., None]) + orig * keep[..., None]
            o = np.clip(res_px + 0.5, 0, 255).astype(np.uint8)
            if prot_w is not None:  # protected pixels: exactly the original
                hard = prot_w[y0:y1] >= 1.0
                o[hard] = img[y0:y1][hard]
            out[y0:y1] = o
        return out


def _mask(p: dict, name: str, w: int, h: int, src: np.ndarray) -> np.ndarray | None:
    from ..masks import rasterize_mask  # noqa: PLC0415 - avoids an import cycle with engine.model
    from ..model import MaskSpec  # noqa: PLC0415

    spec = MaskSpec.from_dict(p.get(name) or {})
    if spec.is_identity():
        return None
    return rasterize_mask(spec, w, h, src)


def _protect_weight(p: dict, w: int, h: int, img: np.ndarray) -> np.ndarray | None:
    """1 inside the protected area (exactly), feathered to 0 outside it."""
    m = _mask(p, "protect", w, h, img)
    if m is None:
        return None
    hard = (m >= 0.5).astype(np.uint8)
    if not hard.any():
        return None
    feather = float(p["feather"]) * max(w, h)
    if feather < 0.5:
        return hard.astype(np.float32)
    dist = cv2.distanceTransform(1 - hard, cv2.DIST_L2, 5)
    soft = np.clip(1.0 - dist / feather, 0, 1)
    soft = soft * soft * (3 - 2 * soft)
    return np.maximum(hard.astype(np.float32), soft.astype(np.float32))
