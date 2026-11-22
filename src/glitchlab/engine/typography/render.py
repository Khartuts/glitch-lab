"""Rendering text layers: layout -> glyph masks -> style -> transform -> composite.

Everything is computed at the target render resolution (with 2x
supersampling for small text) from normalised geometry, so the preview and
the export have the same layout and the same seeded damage.  Layers are
composited with premultiplied alpha, which keeps outlines, shadows and
distressed edges free of dark fringes.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ..params import color_to_rgb
from . import fonts
from .layout import FontPlan, Glyph, Layout, layout_path, layout_text
from .model import STYLE_PARAMS, coerce_box
from .styles import STYLES, StyleInput, layer, over, shift

GLITCH_GLYPHS = "█▓▒░▚▞▙▟"


@dataclass
class BoxReport:
    uid: str
    overflow: bool = False
    size_px: float = 0.0
    warnings: list[str] = field(default_factory=list)
    missing_font: str = ""


def _rgb(c: str) -> np.ndarray:
    return np.float32(color_to_rgb(c)) / 255.0


def _font_pool() -> list[str]:
    """Fonts used by Cut-and-Paste (bundled ones, plus a few common system faces when present)."""
    pool = [f.path for f in fonts.REGISTRY.bundled()]
    for fam in ("Impact", "Arial Black", "Georgia", "Courier New", "Times New Roman", "Verdana", "Arial",
                "Trebuchet MS", "Liberation Serif", "Liberation Sans"):
        for st in ("Regular", "Bold", "Normal"):
            f = fonts.REGISTRY.get(f"{fam}|{st}")
            if f is not None:
                pool.append(f.path)
                break
    return pool


def _words(text: str) -> list[str]:
    return text.split()


def _plan(box: dict, path: str) -> FontPlan:
    if box["style"] != "cut":
        return FontPlan(path)
    sp = box["style_params"]["cut"]
    pool = _font_pool()
    per = {}
    locks = sp.get("locks") or {}
    for i, _w in enumerate(_words(box["text"])):
        rng = np.random.default_rng([box["seed"], i, 77])
        lk = locks.get(str(i), {})
        if "font" in lk and fonts.REGISTRY.get(lk["font"]):
            fp = fonts.REGISTRY.get(lk["font"]).path
        elif rng.random() < sp["font_variation"] * min(box["intensity"], 1.0):
            fp = pool[int(rng.integers(0, len(pool)))]
        else:
            fp = path
        k = float(lk.get("size", 1.0 + (rng.uniform(-0.22, 0.38) * sp["font_variation"] * min(box["intensity"], 1.0))))
        per[i] = (fp, k)
    return FontPlan(path, per)


class GlyphRaster:
    """Draws glyphs (with per-glyph fonts, rotation, compression) into coverage masks."""

    def __init__(self, shape: tuple[int, int], ox: float, oy: float, substitute: float = 0.0, seed: int = 0):
        self.h, self.w = shape
        self.ox, self.oy = ox, oy
        self.substitute = substitute
        self.seed = seed

    def char_for(self, g: Glyph) -> str:
        if self.substitute > 0 and not g.ch.isspace():
            r = np.random.default_rng([self.seed, g.index, 913]).random()
            if r < self.substitute:
                return GLITCH_GLYPHS[int(r * 997) % len(GLITCH_GLYPHS)]
        return g.ch

    def render(self, glyphs: list[Glyph]) -> np.ndarray:
        im = Image.new("L", (self.w, self.h), 0)
        draw = ImageDraw.Draw(im)
        extra = []
        for g in glyphs:
            ch = self.char_for(g)
            if ch.isspace():
                continue
            path = g.path if ch == g.ch else fonts.font_for_char(fonts.fallback_path(), ch)
            font = fonts.load(path, int(round(g.size)))
            if abs(g.angle) < 0.01 and abs(g.scale_x - 1.0) < 1e-3:
                draw.text((g.x + self.ox, g.y + self.oy), ch, font=font, fill=255, anchor="ls")
            else:
                extra.append((g, ch, font))
        m = np.asarray(im, np.float32) / 255.0
        if extra:
            m = m.copy()
            for g, ch, font in extra:
                t = int(g.size * 3)
                tile = Image.new("L", (t, t), 0)
                ImageDraw.Draw(tile).text((t / 2, t / 2), ch, font=font, fill=255, anchor="mm")
                arr = np.asarray(tile, np.float32) / 255.0
                cx, cy = g.x + self.ox + g.adv / 2, g.y + self.oy - g.size * 0.32
                if g.angle == 0.0:
                    cx = g.x + self.ox + g.adv / 2
                M = cv2.getRotationMatrix2D((t / 2, t / 2), -g.angle, 1.0)
                M[0, :] *= g.scale_x
                M[0, 2] += cx - t / 2 * g.scale_x
                M[1, 2] += cy - t / 2
                warped = cv2.warpAffine(arr, M, (self.w, self.h), flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT)
                np.maximum(m, warped, out=m)
        return m


def _scatter_layout(box: dict, plan: FontPlan, size: float, bw: float, bh: float) -> Layout:
    """Words placed at seeded positions inside the box (reading order kept along a loose diagonal)."""
    rng = np.random.default_rng([box["seed"], 4242])
    glyphs = []
    words = []
    idx = 0
    text = box["text"]
    for wi, w in enumerate(text.split()):
        start = text.find(w, idx)
        idx = start + len(w)
        words.append((wi, w, start))
    n = max(1, len(words))
    for k, (wi, w, start) in enumerate(words):
        path, f = plan.for_word(wi)
        s = size * f * rng.uniform(0.7, 1.5)
        font = fonts.load(fonts.font_for_char(path, w[0]), int(round(s)))
        wlen = font.getlength(w)
        t = (k + 0.5) / n
        x = (t * 0.7 + rng.uniform(0, 0.3)) * max(1.0, bw - wlen)
        y = s + (rng.uniform(0, 1)) * max(1.0, bh - s * 1.3)
        ang = rng.normal(0, 14)
        a = math.radians(ang)
        pen = 0.0
        for ci, ch in enumerate(w):
            p = fonts.font_for_char(path, ch)
            fnt = fonts.load(p, int(round(s)))
            adv = fnt.getlength(ch)
            gx = x + math.cos(a) * pen
            gy = y + math.sin(a) * pen
            glyphs.append(Glyph(ch, gx, gy, adv, p, s, start + ci, wi, wi, angle=ang))
            pen += adv
    return Layout(glyphs, [], size, bw, bh, False, [])


def _canvas_to_image(box: dict, W: int, H: int, x0: float, y0: float, cw: float, ch: float) -> np.ndarray:
    """Homography from canvas pixels to image pixels.

    The canvas' top-left corner sits at image pixel (x0, y0) (integers) before
    the box rotation / perspective, which pivot on the box centre; without
    them the mapping is an exact whole-pixel translation."""
    cx, cy = box["x"] * W, box["y"] * H
    hbw, hbh = max(box["w"] * W / 2, 1e-3), max(box["h"] * H / 2, 1e-3)
    src = np.float32([[0, 0], [cw, 0], [cw, ch], [0, ch]])
    pts = src + np.float32([x0, y0]) - np.float32([cx, cy])
    px, py = box["persp_x"], box["persp_y"]
    out = pts.copy()
    out[:, 0] = pts[:, 0] * (1 + px * pts[:, 1] / hbh)
    out[:, 1] = pts[:, 1] * (1 + py * pts[:, 0] / hbw)
    a = math.radians(box["rotation"])
    R = np.float32([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    dst = out @ R.T + np.float32([cx, cy])
    return cv2.getPerspectiveTransform(src, dst.astype(np.float32))


def _premul_to_straight_u8(pm: np.ndarray) -> np.ndarray:
    a = pm[..., 3:4]
    rgb = np.where(a > 1e-5, pm[..., :3] / np.maximum(a, 1e-5), 0.0)
    out = np.concatenate([rgb, a], -1)
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


def render_box(box: dict, W: int, H: int, *, full_size: tuple[int, int], source: np.ndarray | None = None,
               subject_mask: np.ndarray | None = None, report: BoxReport | None = None,
               decorative: bool = True) -> np.ndarray:
    """One text box -> premultiplied float RGBA (H, W, 4)."""
    from ..masks import rasterize_mask  # noqa: PLC0415 - avoids an import cycle
    from ..model import MaskSpec  # noqa: PLC0415

    b = coerce_box(box)
    report = report or BoxReport(b["uid"])
    info, found = fonts.resolve(b["font"])
    if not found:
        report.missing_font = b["font"]
        report.warnings.append(f"The font '{b['font'].replace('|', ' ')}' is not installed; '{info.label}' is used "
                               "instead. Choose a replacement in the Text panel.")
    style = b["style"]
    sp = dict(b["style_params"].get(style, {}))
    k = float(b["intensity"])
    # Layout is computed once in full-resolution pixels and scaled to the render size, so line breaks,
    # fitting and positions are identical in the preview and the export.
    Wf, Hf = full_size
    f = H / float(Hf)
    size_full = max(1.0, b["size"] * Hf)
    ss = 2 if size_full * f < 36 else 1
    fs = f * ss                                   # full-resolution pixels -> canvas pixels
    plan = _plan(b, info.path)
    condense = 1.0 - (sp.get("condense", 0.0) * min(k, 1.0) if style == "industrial" else 0.0)
    out = np.zeros((H, W, 4), np.float32)
    path_mode = b["layout"] == "path" and len(b["path"]) >= 2
    if path_mode:
        pts = np.float32([[p[0] * Wf, p[1] * Hf] for p in b["path"]])
        lay = layout_path(b["text"], plan, size_full, pts, tracking=b["tracking"], align=b["align"])
    elif b["layout"] == "scatter":
        lay = _scatter_layout(b, plan, size_full, b["w"] * Wf, b["h"] * Hf)
    else:
        lay = layout_text(b["text"], plan, size_full, b["w"] * Wf, b["h"] * Hf, tracking=b["tracking"],
                          line_spacing=b["line_spacing"], align=b["align"], wrap=b["wrap"],
                          preserve_breaks=b["preserve_breaks"], fit=b["fit"], min_size=b["min_size"] * Hf,
                          condense=condense)
    em_full = lay.size
    for g in lay.glyphs:
        g.x, g.y, g.adv, g.size = g.x * fs, g.y * fs, g.adv * fs, g.size * fs
    lay_h = lay.height * fs
    em = em_full * fs
    if path_mode:
        cw, ch = W * ss, H * ss
        ox = oy = 0.0
        Hm = np.eye(3, dtype=np.float32)
        transform_canvas = False
        canvas_em = (round(Wf / em_full, 3), round(Hf / em_full, 3))
    else:
        bw, bh = b["w"] * Wf * fs, b["h"] * Hf * fs
        margin_em = 2.6 + abs(b["shadow_dx"]) + abs(b["shadow_dy"]) + b["shadow_blur"] * 2
        if style == "ghost":
            margin_em += sp["echoes"] * sp["distance"] * 2.0
        margin = em * margin_em
        cx_s, cy_s = b["x"] * W * ss, b["y"] * H * ss
        bx0, by0 = cx_s - bw / 2, cy_s - bh / 2
        free = bh - lay_h
        voff = {"middle": free / 2, "bottom": free}.get(b["valign"], 0.0) if free > 0 else 0.0
        over_h = max(0.0, lay_h - bh)   # overflowing lines run past the bottom of the box
        canvas_em = (round(b["w"] * Wf / em_full + 2 * margin_em, 3),
                     round(max(b["h"] * Hf, lay.height) / em_full + 2 * margin_em, 3))
        ox_i = int(math.floor((bx0 - margin) / ss)) * ss     # canvas origin on the image pixel grid
        oy_i = int(math.floor((by0 - margin) / ss)) * ss
        ox, oy = bx0 - ox_i, by0 - oy_i + voff
        cw = int(math.ceil((bx0 + bw + margin - ox_i) / ss)) * ss
        ch = int(math.ceil((by0 + bh + over_h + margin - oy_i) / ss)) * ss
        Hm = _canvas_to_image(b, W, H, ox_i / ss, oy_i / ss, cw / ss, ch / ss)
        transform_canvas = True
    report.overflow = report.overflow or lay.overflow
    report.size_px = em_full
    report.warnings.extend(lay.warnings)
    if not lay.glyphs:
        return out
    sub = sp.get("substitution", 0.0) * k if style == "transmission" else 0.0
    raster = GlyphRaster((ch, cw), ox, oy, sub, b["seed"])
    # damage only where the distress brush allows it (mask painted in image coordinates)
    dmg = None
    spec = MaskSpec.from_dict(b["distress"])
    if not spec.is_identity():
        m_img = rasterize_mask(spec, W, H, source if source is not None else np.zeros((H, W, 4), np.uint8))
        to_img = Hm.astype(np.float64) @ np.diag([1.0 / ss, 1.0 / ss, 1.0])
        dmg = cv2.warpPerspective(m_img, np.linalg.inv(to_img), (cw, ch), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT) if m_img is not None else None
    # style groups (per-word / per-line overrides)
    groups: dict[tuple[str, float], list[Glyph]] = {}
    for g in lay.glyphs:
        st, inten = style, k
        for o in b["overrides"]:
            if (o["scope"] == "word" and g.word == o["index"]) or (o["scope"] == "line" and g.line == o["index"]):
                st, inten = o["style"], o["intensity"]
        groups.setdefault((st, inten), []).append(g)
    F_all = raster.render(lay.glyphs)
    boxes = []
    for g in lay.glyphs:
        boxes.append((g.x + ox, g.y + oy - g.size * 0.75, g.x + ox + g.adv, g.y + oy + g.size * 0.05, g.ch, g.word,
                      g.line))
    fill, outline, dcol = _rgb(b["fill"]), _rgb(b["outline"]), _rgb(b["damage_colour"])
    styled = np.zeros((ch, cw, 4), np.float32)
    for (st, inten), gl in groups.items():
        F = raster.render(gl) if len(groups) > 1 else F_all
        si = StyleInput(F, em, canvas_em, dict(b["style_params"].get(st, {})), inten, dmg,
                        int(b["seed"]) * 31 + hash(st) % 1000, fill, outline, b["outline_width"] * em, dcol,
                        [bx for bx, g in zip(boxes, lay.glyphs) if g in gl], render=raster.render, glyphs=gl)
        si.seed = int(np.random.default_rng([b["seed"], STYLE_IDX[st]]).integers(0, 2**31 - 1))
        res = STYLES[st](si) if inten > 0 or st == "clean" else STYLES["clean"](si)
        if dmg is not None and st != "clean":  # outside the distress brush the letters stay clean
            res = STYLES["clean"](si) * (1.0 - dmg[..., None]) + res * dmg[..., None]
        styled = over(res, styled)
    if b["readability"] > 0 and style != "clean":
        styled = over(styled, layer(F_all, fill, float(b["readability"])))
    alpha_all = styled[..., 3]
    under = np.zeros_like(styled)
    if b["backing"] != "none" and b["backing_opacity"] > 0:
        pad = b["backing_padding"] * em
        plate = np.zeros((ch, cw), np.float32)
        if b["backing"] == "box" and transform_canvas:
            x0, y0 = ox - pad * 0.5, oy - voff - pad * 0.5
            cv2.rectangle(plate, (int(x0), int(y0)), (int(ox + b["w"] * W * ss + pad * 0.5),
                                                      int(oy - voff + max(lay_h, b["h"] * H * ss) + pad * 0.5)),
                          1.0, -1)
        else:
            lines: dict[int, list[float]] = {}
            for bx in boxes:
                v = lines.setdefault(bx[6], [bx[0], bx[1], bx[2], bx[3]])
                v[0], v[1], v[2], v[3] = min(v[0], bx[0]), min(v[1], bx[1]), max(v[2], bx[2]), max(v[3], bx[3])
            for x0, y0, x1, y1 in lines.values():
                cv2.rectangle(plate, (int(x0 - pad), int(y0 - pad * 0.6)), (int(x1 + pad), int(y1 + pad * 0.6)), 1.0,
                              -1)
        under = over(under, layer(plate, _rgb(b["backing_colour"]), float(b["backing_opacity"])))
    if b["shadow_opacity"] > 0:
        sh = shift(alpha_all, b["shadow_dx"] * em, b["shadow_dy"] * em)
        if b["shadow_blur"] > 0:
            sh = cv2.GaussianBlur(sh, (0, 0), max(0.3, b["shadow_blur"] * em))
        under = over(layer(sh, _rgb(b["shadow"]), float(b["shadow_opacity"])), under)
    full = over(styled, under)
    if decorative and b["repeat"] != "none":
        full = over(full, _repetition(b, raster, lay, cw, ch, em, ox, oy, fill))
    # canvas -> image (premultiplied, so edges stay clean)
    if ss > 1:  # premultiplied area averaging = correct anti-aliasing
        full = cv2.resize(full, (cw // ss, ch // ss), interpolation=cv2.INTER_AREA)
    if transform_canvas:
        img = cv2.warpPerspective(full, Hm.astype(np.float64), (W, H), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT)
    else:
        img = full[:H, :W]
    img = img * np.float32(b["opacity"])
    if b["behind"]:
        occ = subject_mask
        mspec = MaskSpec.from_dict(b["subject_mask"])
        if not mspec.is_identity():
            mm = rasterize_mask(mspec, W, H, source if source is not None else np.zeros((H, W, 4), np.uint8))
            if mm is not None:
                occ = mm if occ is None else np.clip(np.maximum(occ, mm), 0, 1)
        if occ is not None:
            img = img * (1.0 - occ[..., None])
    return img


STYLE_IDX = {s: i for i, s in enumerate(STYLES)}


def _repetition(b, raster: GlyphRaster, lay: Layout, cw, ch, em, ox, oy, fill) -> np.ndarray:
    """Decorative repeated text (separate from the editable text, drawn under it)."""
    rng = np.random.default_rng([b["seed"], 5151])
    s = em * b["repeat_scale"]
    flat = " ".join(b["text"].split())
    if not flat:
        return np.zeros((ch, cw, 4), np.float32)
    glyphs = []
    path = lay.glyphs[0].path
    font = fonts.load(path, int(round(s)))
    if b["repeat"] == "texture":
        rows = int(ch / (s * 1.05)) + 1
        for r in range(rows):
            x = -rng.uniform(0, 1) * font.getlength(flat)
            y = (r + 1) * s * 1.05 - oy * 0
            i = 0
            while x < cw:
                c = flat[i % len(flat)]
                p = fonts.font_for_char(path, c)
                adv = fonts.load(p, int(round(s))).getlength(c)
                if not c.isspace():
                    glyphs.append(Glyph(c, x - ox, y - oy, adv, p, s, -1, -1, -1))
                x += adv
                i += 1
    else:
        words = flat.split()
        for _ in range(int(b["repeat_count"])):
            wd = words[int(rng.integers(0, len(words)))]
            x, y = rng.uniform(0, cw) - ox, rng.uniform(0, ch) - oy
            ang = rng.normal(0, 20)
            pen = 0.0
            a = math.radians(ang)
            sc = s * rng.uniform(0.6, 1.6)
            for c in wd:
                p = fonts.font_for_char(path, c)
                adv = fonts.load(p, int(round(sc))).getlength(c)
                glyphs.append(Glyph(c, x + math.cos(a) * pen, y + math.sin(a) * pen, adv, p, sc, -1, -1, -1,
                                    angle=ang))
                pen += adv
    m = raster.render(glyphs)
    return layer(m, fill, float(b["repeat_opacity"]))


def text_layer_signature(boxes: list[dict], W: int, H: int) -> str:
    ids = sorted({bx.get("font", "") for bx in boxes})
    paths = [fonts.resolve(i)[0].path for i in ids]
    return hashlib.sha1(json.dumps([boxes, W, H, paths], sort_keys=True, default=str).encode()).hexdigest()


def render_text_layer(boxes: list[dict], W: int, H: int, *, full_size: tuple[int, int],
                      source: np.ndarray | None = None, subject_fn=None, reports: list | None = None,
                      cancel=None) -> np.ndarray:
    """All boxes of a text layer -> straight-alpha uint8 RGBA (H, W, 4)."""
    acc = np.zeros((H, W, 4), np.float32)
    for bx in boxes:
        if cancel is not None:
            cancel.check()
        rep = BoxReport(str(bx.get("uid", "")))
        occ = None
        if bx.get("behind") and bx.get("subject") and subject_fn is not None:
            try:
                occ = subject_fn(bx["subject"])
            except Exception as exc:  # AI optional: the manual mask still works
                rep.warnings.append(f"Subject mask unavailable ({getattr(exc, 'message', exc)}); only the manual "
                                    "mask is used.")
        acc = over(render_box(bx, W, H, full_size=full_size, source=source, subject_mask=occ, report=rep), acc)
        if reports is not None:
            reports.append(rep)
    return _premul_to_straight_u8(acc)


def box_report(box: dict, W: int, H: int) -> BoxReport:
    """Layout check for the UI (overflow, missing font) without styling."""
    b = coerce_box(box)
    rep = BoxReport(b["uid"])
    info, found = fonts.resolve(b["font"])
    if not found:
        rep.missing_font = b["font"]
    size = max(1.0, b["size"] * H)
    if b["layout"] == "path":
        return rep
    lay = layout_text(b["text"], _plan(b, info.path), size, b["w"] * W, b["h"] * H, tracking=b["tracking"],
                      line_spacing=b["line_spacing"], align=b["align"], wrap=b["wrap"],
                      preserve_breaks=b["preserve_breaks"], fit=b["fit"], min_size=b["min_size"] * H)
    rep.overflow, rep.size_px, rep.warnings = lay.overflow, lay.size, list(lay.warnings)
    return rep


__all__ = ["render_text_layer", "render_box", "box_report", "BoxReport", "text_layer_signature", "STYLE_PARAMS"]
