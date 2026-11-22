"""Typography style recipes.

Each style turns glyph *coverage masks* (rendered from real font glyphs)
into a premultiplied RGBA layer.  All lengths are in em (font size) and all
noise lattices are sized from em quantities, so a reduced preview and the
full-resolution export get the same damage pattern.  Damage is driven by a
seed; ``intensity`` 0 gives the clean letters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from .. import flow


@dataclass
class StyleInput:
    F: np.ndarray                      # glyph coverage 0..1 (h, w)
    em: float                          # font size in canvas pixels
    canvas_em: tuple[float, float]     # canvas size in em (scale independent)
    params: dict
    intensity: float
    damage: np.ndarray | None          # where damage is allowed (0..1) or None
    seed: int
    fill: np.ndarray                   # RGB 0..1
    outline: np.ndarray
    outline_px: float
    damage_colour: np.ndarray
    glyph_boxes: list                  # (x0, y0, x1, y1, ch, word, line) in canvas px
    render: Callable | None = None     # render(glyph_subset) -> coverage mask (cut-and-paste)
    glyphs: list | None = None


# ------------------------------------------------------------------ helpers
def noise(rng: np.random.Generator, shape: tuple[int, int], canvas_em: tuple[float, float], cell_em: float
          ) -> np.ndarray:
    """Smooth noise ~[-1, 1] whose lattice depends only on em quantities."""
    h, w = shape
    nx = int(math.ceil(canvas_em[0] / max(cell_em, 1e-3))) + 3
    ny = int(math.ceil(canvas_em[1] / max(cell_em, 1e-3))) + 3
    lat = rng.standard_normal((min(ny, 2048), min(nx, 2048))).astype(np.float32)
    return np.clip(cv2.resize(lat, (w, h), interpolation=cv2.INTER_CUBIC) * 0.7, -1.5, 1.5)


def dist_inside(F: np.ndarray) -> np.ndarray:
    return cv2.distanceTransform((F > 0.5).astype(np.uint8), cv2.DIST_L2, 5)


def ring(F: np.ndarray, width_px: float) -> np.ndarray:
    """Anti-aliased outline band of ``width_px`` around the glyphs."""
    if width_px <= 0:
        return np.zeros_like(F)
    d_out = cv2.distanceTransform((F < 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    return np.clip(width_px - d_out + 0.5, 0, 1) * (1 - F) + 0 * F


def layer(mask: np.ndarray, rgb: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    a = np.clip(mask, 0, 1).astype(np.float32) * np.float32(alpha)
    return np.dstack([a * rgb[0], a * rgb[1], a * rgb[2], a]).astype(np.float32)


def layer_rgb(mask: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """``rgb`` per pixel (h, w, 3) in 0..1."""
    a = np.clip(mask, 0, 1).astype(np.float32)
    return np.dstack([rgb * a[..., None], a]).astype(np.float32)


def over(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    return top + bottom * (1.0 - top[..., 3:4])


def shift(arr: np.ndarray, dx: float, dy: float) -> np.ndarray:
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(arr, m, (arr.shape[1], arr.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT)


def _dmg(si: StyleInput) -> np.ndarray | float:
    return 1.0 if si.damage is None else si.damage


def clean(si: StyleInput) -> np.ndarray:
    out = layer(si.F, si.fill)
    if si.outline_px > 0:
        out = over(out, layer(ring(si.F, si.outline_px), si.outline))
    return out


# ------------------------------------------------------------------ 1 thrashed
def thrashed(si: StyleInput) -> np.ndarray:
    p, k, em = si.params, si.intensity, si.em
    rng = np.random.default_rng(si.seed)
    F = si.F
    sh = F.shape
    d = dist_inside(F) / em
    where = np.exp(-d / 0.07) if p["distribution"] == "edges" else np.ones_like(F)
    dm = _dmg(si)
    loss = noise(rng, sh, si.canvas_em, 0.09) + 0.5 * noise(rng, sh, si.canvas_em, 0.035)
    holes = (loss > 1.3 - 1.6 * p["ink_loss"] * k * (0.4 + 0.6 * where)).astype(np.float32)
    speck = noise(rng, sh, si.canvas_em, 0.012)
    abr = np.clip((speck - (1.15 - 1.3 * p["abrasion"] * k)) * 4, 0, 1)
    chip_n = noise(rng, sh, si.canvas_em, 0.05)
    chip = ((d < 0.03 + 0.07 * p["chipping"] * k) & (chip_n > 0.55 - 0.8 * p["chipping"] * k)).astype(np.float32)
    scratch = np.zeros(sh, np.uint8)
    gb = [g for g in si.glyph_boxes if not g[4].isspace()]
    if gb:
        n = int(len(gb) * 1.6 * p["abrasion"] * k) + 1
        for _ in range(n):
            x0g, y0g, x1g, y1g = gb[int(rng.integers(0, len(gb)))][:4]
            px, py = x0g + rng.random() * (x1g - x0g), y0g + rng.random() * (y1g - y0g)
            L = em * p["scratch_scale"] * rng.uniform(0.4, 1.8)
            a = math.radians(rng.normal(0, 18) + (90 if rng.random() < 0.15 else 0))
            x0, y0 = px - math.cos(a) * L / 2, py - math.sin(a) * L / 2
            x1, y1 = px + math.cos(a) * L / 2, py + math.sin(a) * L / 2
            th = max(1, int(round(em * rng.uniform(0.012, 0.045))))
            cv2.line(scratch, (int(x0), int(y0)), (int(x1), int(y1)), 255, th, cv2.LINE_AA)
    scr = scratch.astype(np.float32) / 255.0
    damage = np.clip(holes + abr * 0.9 + chip + scr, 0, 1) * dm
    body = F * (1.0 - damage)
    out = layer(body, si.fill)
    if si.outline_px > 0:
        out = over(out, layer(ring(F, si.outline_px) * (1 - damage * 0.7), si.outline))
    # offset duplicate impression (misregistration) in the damage colour, under the letters
    m = p["misregistration"] * k * em
    if m > 0.3:
        ang = rng.uniform(0, 2 * math.pi)
        dup = shift(F * (1 - holes * 0.5), math.cos(ang) * m, math.sin(ang) * m)
        out = over(out, layer(dup * 0.8, si.damage_colour))
    # torn scratches show the damage colour faintly at their rims
    rim = np.clip(cv2.dilate(scr, np.ones((3, 3), np.uint8)) - scr, 0, 1) * F * 0.5 * (dm if isinstance(dm, float)
                                                                                       else dm)
    return over(layer(rim, si.damage_colour), out)


# ------------------------------------------------------------------ 2 industrial
STENCIL_COUNTERS = set("ABDOPQRabdegopq04689&@%")


def industrial(si: StyleInput) -> np.ndarray:
    p, k, em = si.params, si.intensity, si.em
    rng = np.random.default_rng(si.seed)
    F = si.F.copy()
    sh = F.shape
    dm = _dmg(si)
    cut = np.zeros(sh, np.float32)
    bw = max(1.0, p["stencil"] * em * min(k, 1.0))
    if p["stencil"] > 0 and k > 0:
        for x0, y0, x1, y1, ch, _w, _l in si.glyph_boxes:
            if ch.isspace():
                continue
            cx = (x0 + x1) / 2
            if ch in STENCIL_COUNTERS or (x1 - x0) > 0.55 * em:
                cv2.rectangle(cut, (int(cx - bw / 2), int(y0)), (int(math.ceil(cx + bw / 2)), int(y1)), 1.0, -1)
            if ch in "ABEFHPR8":
                cy = y0 + (y1 - y0) * 0.5
                cv2.rectangle(cut, (int(x0), int(cy - bw / 2)), (int(x1), int(math.ceil(cy + bw / 2))), 1.0, -1)
    F = F * (1 - cut * (dm if isinstance(dm, float) else dm))
    # horizontal slices displaced sideways
    if p["slices"] > 0 and k > 0:
        y = 0.0
        out_rows = F.copy()
        while y < sh[0]:
            hgt = em * rng.uniform(0.08, 0.3)
            if rng.random() < 0.35:
                dx = rng.uniform(-1, 1) * p["slices"] * k * em * 0.5
                a, b = int(y), int(min(sh[0], y + hgt))
                out_rows[a:b] = shift(F[a:b], dx, 0)
            y += hgt
        F = F * (1 - (1 if si.damage is None else si.damage)) + out_rows * (1 if si.damage is None else si.damage) \
            if si.damage is not None else out_rows
    # uneven stamp pressure
    pres = noise(rng, sh, si.canvas_em, 0.6) * 0.6 + noise(rng, sh, si.canvas_em, 0.1) * 0.4
    ink = np.clip(1.0 - p["pressure"] * k * np.clip(0.45 - pres * 0.6, 0, 1) * 1.6, 0.15, 1.0)
    speck = noise(rng, sh, si.canvas_em, 0.015)
    ink = ink * np.where(speck > 1.1 - 0.6 * p["pressure"] * k, 0.25, 1.0)
    body = F * ink
    out = layer(body, si.fill)
    if si.outline_px > 0:
        out = over(out, layer(ring(F, si.outline_px), si.outline))
    reg = p["registration"] * k * em
    if reg > 0.3:
        out = over(out, layer(shift(F, reg, -reg * 0.4) * 0.85, si.damage_colour))
    if p["plate"] > 0:
        plate = np.zeros(sh, np.float32)
        lines = {}
        for x0, y0, x1, y1, ch, _w, li in si.glyph_boxes:
            b = lines.setdefault(li, [x0, y0, x1, y1])
            b[0], b[1], b[2], b[3] = min(b[0], x0), min(b[1], y0), max(b[2], x1), max(b[3], y1)
        pad = 0.18 * em
        for x0, y0, x1, y1 in lines.values():
            cv2.rectangle(plate, (int(x0 - pad), int(y0 - pad)), (int(x1 + pad), int(y1 + pad)), 1.0, -1)
        grime = 0.85 + 0.15 * noise(rng, sh, si.canvas_em, 0.3)
        from ..params import color_to_rgb  # noqa: PLC0415

        pc = np.float32(color_to_rgb(p["plate_colour"])) / 255.0
        out = over(out, layer(plate * np.clip(grime, 0, 1), pc, p["plate"]))
    return out


# ------------------------------------------------------------------ 3 acid / liquid
def acid(si: StyleInput) -> np.ndarray:
    p, k, em = si.params, si.intensity, si.em
    rng = np.random.default_rng(si.seed)
    F = si.F
    h, w = F.shape
    cem_w, cem_h = si.canvas_em
    gl = 256.0 / max(cem_w, cem_h)                          # grid pixels per em (scale independent)
    gw, gh = max(16, int(round(cem_w * gl))), max(16, int(round(cem_h * gl)))
    gem = gl                                                # em in grid pixels
    cell = p["vortex_scale"] * gem / max(gw, gh)
    vx, vy = flow.curl_noise(rng, gw, gh, max(cell, 0.01), 3, 0.5)
    ang = rng.uniform(-0.4, 0.4)
    vx = vx + p["stretching"] * math.cos(ang) * 1.5
    vy = vy + p["stretching"] * math.sin(ang) * 0.6
    dist = p["liquidity"] * k * gem * 0.35
    res = flow.advect(vx.astype(np.float32), vy.astype(np.float32), dist, 12, 0.3)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    dxu = cv2.resize(res.dx, (w, h), interpolation=cv2.INTER_LINEAR) * (w / gw)
    dyu = cv2.resize(res.dy, (w, h), interpolation=cv2.INTER_LINEAR) * (h / gh)
    if si.damage is not None:
        dxu, dyu = dxu * si.damage, dyu * si.damage
    warp = cv2.remap(F, xs + dxu, ys + dyu, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    # liquid bridges between nearby characters (metaball merge)
    if p["bridges"] > 0 and k > 0:
        sig = em * (0.03 + 0.09 * p["bridges"] * min(k, 1.5))
        blob = cv2.GaussianBlur(warp, (0, 0), sig)
        bridge = np.clip((blob - (0.42 - 0.1 * p["bridges"])) * 8, 0, 1)
        warp = np.maximum(warp, bridge)
    mag = np.sqrt(dxu * dxu + dyu * dyu) / max(em, 1.0)
    if p["iridescent"]:
        t = mag * 2.5 + ys / max(em, 1) * 0.18 + xs / max(em, 1) * 0.07
        film = 0.5 + 0.5 * np.cos(2 * np.pi * (t[..., None] + np.float32([0.0, 0.33, 0.67])))
        col = si.fill[None, None, :] * 0.45 + film.astype(np.float32) * 0.65
        out = layer_rgb(warp, np.clip(col, 0, 1))
    else:
        out = layer(warp, si.fill)
    if si.outline_px > 0:
        out = over(out, layer(ring(warp, si.outline_px), si.outline))
    sep = p["separation"] * k * em
    if sep > 0.3:
        ux, uy = np.mean(dxu), np.mean(dyu)
        n = math.hypot(ux, uy) or 1.0
        ux, uy = (ux / n, uy / n) if n > 1e-3 else (1.0, 0.0)
        r = ring(warp, max(1.0, em * 0.03))
        mag_l = layer(shift(r, ux * sep, uy * sep), np.float32([1.0, 0.1, 0.85]))
        cyn_l = layer(shift(r, -ux * sep, -uy * sep), np.float32([0.0, 0.9, 1.0]))
        out = over(out, over(mag_l, cyn_l))
    return out


# ------------------------------------------------------------------ 4 rust
RUST = {"orange": (("#6e6e72", "#9aa0a6"), ("#5a2a12", "#a5491c", "#d9822f")),
        "iron": (("#3a3c40", "#6a6e74"), ("#1c1512", "#3b2a22", "#5c4433")),
        "verdigris": (("#b87333", "#e0a167"), ("#2c5e50", "#4fa58f", "#8fd3bd"))}


def rust(si: StyleInput) -> np.ndarray:
    from ..params import color_to_rgb  # noqa: PLC0415

    p, k, em = si.params, si.intensity, si.em
    rng = np.random.default_rng(si.seed)
    F = si.F
    sh = F.shape
    metal, corr = RUST[p["palette"]]
    m0, m1 = (np.float32(color_to_rgb(c)) / 255.0 for c in metal)
    c0, c1, c2 = (np.float32(color_to_rgb(c)) / 255.0 for c in corr)
    d = dist_inside(F) / em
    n = noise(rng, sh, si.canvas_em, 0.45) * 0.55 + noise(rng, sh, si.canvas_em, 0.14) * 0.3 + \
        noise(rng, sh, si.canvas_em, 0.045) * 0.15
    edge_bias = np.exp(-d / 0.08) * 0.6
    dm = _dmg(si)
    cov = np.clip((n + edge_bias - (0.9 - 1.6 * p["coverage"] * min(k, 1.5))) * 3.5, 0, 1) * dm
    # metallic shading from the distance field (bevel) and a directional highlight
    dd = cv2.GaussianBlur(np.minimum(d, 0.12), (0, 0), max(0.8, em * 0.02))
    gy, gx = np.gradient(dd)
    a = math.radians(p["light"])
    shade = np.clip(0.5 + (gx * math.cos(a) - gy * math.sin(a)) * em * 6.0 * p["metallic"], 0, 1)
    metal_rgb = m0 + (m1 - m0) * shade[..., None]
    rn = np.clip(0.5 + 0.5 * noise(rng, sh, si.canvas_em, 0.08), 0, 1)[..., None]
    rust_rgb = np.where(rn < 0.5, c0 + (c1 - c0) * (rn * 2), c1 + (c2 - c1) * (rn * 2 - 1))
    col = metal_rgb * (1 - cov[..., None]) + rust_rgb * cov[..., None]
    col = col * (0.75 + 0.25 * shade[..., None]) if p["metallic"] > 0 else col
    # pits and flaking edges remove material
    pits_n = noise(rng, sh, si.canvas_em, max(0.01, p["pit_size"]))
    pits = (pits_n > 1.25 - 0.5 * min(k, 1.5) * p["coverage"]).astype(np.float32) * dm
    flake_n = noise(rng, sh, si.canvas_em, 0.06)
    flake = ((d < 0.02 + 0.08 * p["flaking"] * k) & (flake_n > 0.4 - 0.7 * p["flaking"] * k)).astype(np.float32) * dm
    alpha = F * (1 - np.clip(pits + flake, 0, 1))
    col = col * (1 - 0.6 * cv2.dilate(pits, np.ones((3, 3), np.uint8))[..., None])
    col = col * (1 - 0.0) + si.fill * 0.0
    out = layer_rgb(alpha, np.clip(col, 0, 1))
    if si.outline_px > 0:
        out = over(out, layer(ring(F, si.outline_px) * (1 - flake), si.outline))
    return out


# ------------------------------------------------------------------ 5 xerox
def xerox(si: StyleInput) -> np.ndarray:
    p, k, em = si.params, si.intensity, si.em
    rng = np.random.default_rng(si.seed)
    F = si.F.astype(np.float32)
    sh = F.shape
    h, w = sh
    dm = _dmg(si)
    cur = F.copy()
    gens = max(1, int(round(p["generations"] * min(k, 2.0))))
    for g in range(gens):
        sig = max(0.3, em * 0.018 * (0.5 + p["bleed"]))
        cur = cv2.GaussianBlur(cur, (0, 0), sig)
        f = 1.0 / (1.06 + 0.04 * rng.random())       # down and up sampling loses detail
        small = cv2.resize(cur, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
        cur = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        drift = p["drift"] * em * 0.35
        cur = shift(cur, rng.uniform(-1, 1) * drift, rng.uniform(-1, 1) * drift * 0.4)
        thr = p["threshold"] + 0.08 * noise(rng, sh, si.canvas_em, 1.5) * (g + 1) / gens
        # threshold relative to the local ink peak: small marks (accents, dots) survive like real toner
        k_sz = max(3, int(em * 0.14) | 1)
        peak = cv2.dilate(cur, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_sz, k_sz)))
        cur = np.clip((cur - thr * np.maximum(peak, 0.35)) / 0.06 + 0.5, 0, 1)
    if si.damage is not None:
        cur = F * (1 - si.damage) + cur * si.damage
    drop = noise(rng, sh, si.canvas_em, 0.02) + 0.6 * noise(rng, sh, si.canvas_em, 0.2)
    toner = np.where(drop > 0.3 + 1.6 * p["toner"] - 0.6 * (k - 1), 0.0, 1.0) * dm + (1 - dm) * 1.0
    body = cur * toner
    # dust: tiny specks around the letters
    dust_n = noise(rng, sh, si.canvas_em, 0.01)
    near = cv2.GaussianBlur(F, (0, 0), em * 0.6)
    dust = ((dust_n > 1.42 - 0.1 * k) & (near > 0.02)).astype(np.float32) * 0.9
    ink = {"black": np.float32([0.05, 0.05, 0.05]), "white": np.float32([0.96, 0.96, 0.94])}.get(p["ink"], si.fill)
    copy2 = shift(cur, p["drift"] * em * 1.2, p["drift"] * em * 0.5) * 0.35 * min(k, 1.0)
    out = layer(np.clip(np.maximum(body, dust * dm if isinstance(dm, float) else dust * dm), 0, 1), ink)
    out = over(out, layer(copy2, ink))
    if si.outline_px > 0:
        out = over(out, layer(ring(cur, si.outline_px), si.outline))
    return out


# ------------------------------------------------------------------ 6 cut-and-paste
PAPERS = ("#efe9dc", "#fbfbf7", "#f3e3a0", "#f2c6d2", "#d9d9d9", "#151515", "#c9e3f2")


def cut(si: StyleInput) -> np.ndarray:
    """Paper fragments behind each word / character (fonts were varied during layout).

    All papers are laid down first and all ink on top, so overlapping pieces
    never hide letters; word order and punctuation stay intact."""
    from ..params import color_to_rgb  # noqa: PLC0415

    p, k, em = si.params, si.intensity, si.em
    h, w = si.F.shape
    papers = np.zeros((h, w, 4), np.float32)
    inks = np.zeros((h, w, 4), np.float32)
    groups: dict[tuple, list] = {}
    for g in si.glyphs or []:
        key = (g.word, g.line) if p.get("unit", "word") == "word" else (g.index, g.line)
        groups.setdefault(key, []).append(g)
    locks = p.get("locks") or {}
    pieces: dict[int, int] = {}
    for key in sorted(groups):
        gl = groups[key]
        lk = locks.get(str(key[0]), {})
        piece = pieces.get(key[0], 0)       # a word split over lines gives several pieces
        pieces[key[0]] = piece + 1
        frng = np.random.default_rng([si.seed, key[0], piece])
        paper_i = int(lk.get("paper", frng.integers(0, len(PAPERS))))
        rot = float(lk.get("rot", frng.normal(0, 1) * p["rotation"] * min(k, 2.0)))
        lift = float(lk.get("baseline", frng.normal(0, 1) * p["baseline"] * min(k, 2.0))) * em
        cover = shift(si.render(gl), 0, lift)
        ys, xs = np.nonzero(cover > 0.02)
        if not len(ys):
            continue
        mrg = p["paper_margin"] * em
        x0, x1 = xs.min() - mrg, xs.max() + mrg
        y0, y1 = ys.min() - mrg * 0.8, ys.max() + mrg * 0.8
        paper = np.zeros((h, w), np.float32)
        cv2.rectangle(paper, (int(x0), int(y0)), (int(x1), int(y1)), 1.0, -1)
        if p["tearing"] > 0:
            tn = noise(frng, (h, w), si.canvas_em, 0.04) * p["tearing"] * em * 0.12
            d_in = cv2.distanceTransform((paper > 0.5).astype(np.uint8), cv2.DIST_L2, 3)
            paper = np.clip((d_in + tn) / 1.5, 0, 1) * (paper > 0)
        pc = np.float32(color_to_rgb(PAPERS[paper_i])) / 255.0
        ink = si.fill if np.mean(pc) > 0.4 else np.float32([0.95, 0.95, 0.92])
        if paper_i == 1 and frng.random() < 0.3:
            ink = si.damage_colour
        shadow = cv2.GaussianBlur(shift(paper, em * 0.03, em * 0.05), (0, 0), max(0.5, em * 0.03)) * 0.35
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        M = cv2.getRotationMatrix2D((cx, cy), rot, 1.0)

        def rot_(a):
            return cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        papers = over(rot_(over(layer(paper, pc), layer(shadow, np.float32([0, 0, 0])))), papers)
        inks = over(rot_(layer(cover, ink)), inks)
    return over(inks, papers)


# ------------------------------------------------------------------ 7 transmission failure
def transmission(si: StyleInput) -> np.ndarray:
    p, k, em = si.params, si.intensity, si.em
    rng = np.random.default_rng(si.seed)
    F = si.F
    h, w = F.shape
    dm = _dmg(si)
    cur = F.copy()
    y = 0.0
    while y < h:
        hgt = em * rng.uniform(0.03, 0.22)
        a, b = int(y), int(min(h, y + hgt))
        r = rng.random()
        if r < p["slice_density"] * min(k, 1.5):
            cur[a:b] = shift(F[a:b], rng.uniform(-1, 1) * p["displacement"] * k * em, 0)
        elif r < p["slice_density"] * min(k, 1.5) + p["dropout"] * k * 0.5:
            cur[a:b] = 0.0
        elif r < p["slice_density"] * min(k, 1.5) + p["dropout"] * k * 0.5 + 0.06 * k and a > (b - a):
            cur[a:b] = cur[a - (b - a):a]           # duplicated rows
        y += hgt
    # pixel blocks
    bs = max(2, int(round(p["pixel_scale"] * em)))
    blk_n = noise(rng, (h, w), si.canvas_em, 0.6)
    small = cv2.resize(cur, (max(1, w // bs), max(1, h // bs)), interpolation=cv2.INTER_AREA)
    pix = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    sel = (blk_n > 0.75 - 0.4 * min(k, 1.5)).astype(np.float32)
    cur = cur * (1 - sel) + pix * sel
    if si.damage is not None:
        cur = F * (1 - si.damage) + cur * si.damage
    out = layer(cur, si.fill)
    sep = p["separation"] * k * em
    if sep > 0.3:
        r_l = layer(shift(cur, sep, 0) * 0.8, np.float32([1.0, 0.1, 0.25]))
        b_l = layer(shift(cur, -sep, 0) * 0.8, np.float32([0.1, 0.55, 1.0]))
        out = over(out, r_l + b_l * (1 - r_l[..., 3:4]))
    if si.outline_px > 0:
        out = over(out, layer(ring(cur, si.outline_px), si.outline))
    return out


# ------------------------------------------------------------------ 8 ghost carbon
def ghost(si: StyleInput) -> np.ndarray:
    p, k, em = si.params, si.intensity, si.em
    rng = np.random.default_rng(si.seed)
    F = si.F
    h, w = F.shape
    dm = _dmg(si)
    ys, xs = np.nonzero(F > 0.3)
    cx, cy = (xs.mean(), ys.mean()) if len(xs) else (w / 2, h / 2)
    out = np.zeros((h, w, 4), np.float32)
    n_e = int(p["echoes"])
    direction = rng.choice([-1.0, 1.0])
    for e in range(n_e, 0, -1):
        if p["spread"] == "drift":
            M = np.float32([[1, 0, direction * e * p["distance"] * k * em], [0, 1, rng.normal(0, 0.03) * em * e]])
        else:
            s = 1.0 + e * p["distance"] * k * 0.08
            M = np.float32([[s, 0, cx * (1 - s)], [0, s, cy * (1 - s)]])
        echo = cv2.warpAffine(F, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        if p["blur"] > 0:
            echo = cv2.GaussianBlur(echo, (0, 0), max(0.3, p["blur"] * em * 0.04 * e))
        er = noise(rng, (h, w), si.canvas_em, 0.25)
        echo = echo * np.clip((er + 1.0 - p["erosion"] * 1.6 * min(k, 1.5)) * 1.5, 0, 1)
        op = (1 - p["fading"]) ** e * min(k, 1.0) * 1.4
        out = over(out, layer(echo * (dm if isinstance(dm, float) else dm), si.fill, float(min(op, 1.0))))
    vanish = noise(rng, (h, w), si.canvas_em, 0.35)
    main = F * np.clip((vanish + 1.25 - p["erosion"] * 0.9 * k) * 1.5, 0.15, 1.0)
    fibre = 0.82 + 0.18 * noise(rng, (h, w), si.canvas_em, 0.015)
    if si.damage is not None:
        main = F * (1 - si.damage) + main * si.damage
    top = layer(np.clip(main * fibre, 0, 1), si.fill, 0.9)
    if si.outline_px > 0:
        top = over(top, layer(ring(F, si.outline_px), si.outline))
    return over(top, out)


STYLES: dict[str, Callable[[StyleInput], np.ndarray]] = {
    "clean": clean, "thrashed": thrashed, "industrial": industrial, "acid": acid, "rust": rust, "xerox": xerox,
    "cut": cut, "transmission": transmission, "ghost": ghost,
}
