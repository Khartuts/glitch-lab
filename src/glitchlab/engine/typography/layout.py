"""Text layout: line breaking, alignment, fit-to-box, overflow and paths.

Every character of the source text is placed (nothing is dropped); when the
text does not fit, ``Layout.overflow`` says so and the lines that do not fit
are still laid out (outside the box).  Characters missing from the chosen
font are drawn with the fallback font.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from . import fonts

_TOKEN = re.compile(r"(\s+)")


@dataclass
class Glyph:
    ch: str
    x: float              # pen position (left) in layout pixels
    y: float              # baseline
    adv: float
    path: str             # font file used for this character
    size: float           # pixel size
    index: int            # character index in the source text
    word: int             # word number in the text (0-based, whitespace separated)
    line: int             # laid-out line number
    angle: float = 0.0    # rotation in degrees (paths, fragments)
    scale_x: float = 1.0  # horizontal compression (industrial)
    frag: int = -1        # fragment number (cut-and-paste)


@dataclass
class Layout:
    glyphs: list[Glyph]
    lines: list[tuple[float, float, float, float]]   # x0, top, x1, bottom of each line
    size: float
    width: float
    height: float
    overflow: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass
class FontPlan:
    """How to pick the font (and size factor) of every word."""
    path: str
    per_word: dict[int, tuple[str, float]] | None = None   # word -> (font path, size factor)

    def for_word(self, w: int) -> tuple[str, float]:
        if self.per_word and w in self.per_word:
            return self.per_word[w]
        return self.path, 1.0


def _advance(path: str, size: float, ch: str, nxt: str | None) -> float:
    font = fonts.load(path, int(round(size)))
    if nxt:
        nf = fonts.load(fonts.font_for_char(path, nxt), int(round(size)))
        if nf is font:  # pair kerning within one font
            return font.getlength(ch + nxt) - font.getlength(nxt)
    return font.getlength(ch)


def _words(text: str, preserve_breaks: bool) -> list[list[tuple[str, int]]]:
    """Paragraphs -> tokens (text, start index). Whitespace runs are kept as tokens."""
    paras, pos = [], 0
    chunks = text.split("\n") if preserve_breaks else [text]
    for chunk in chunks:
        toks = []
        for t in _TOKEN.split(chunk):
            if t:
                toks.append((t if preserve_breaks else t.replace("\n", " "), pos))
            pos += len(t)
        paras.append(toks)
        pos += 1  # the newline
    return paras


def _measure(token: str, plan: FontPlan, word: int, size: float, tracking: float, start: int):
    path0, k = plan.for_word(word)
    s = size * k
    out, x = [], 0.0
    for i, ch in enumerate(token):
        p = fonts.font_for_char(path0, ch)
        nxt = token[i + 1] if i + 1 < len(token) else None
        adv = _advance(p, s, ch, nxt) + tracking * s
        out.append((ch, x, adv, p, s, start + i))
        x += adv
    return out, x


def layout_text(text: str, plan: FontPlan, size: float, box_w: float, box_h: float, *, tracking: float = 0.0,
                line_spacing: float = 1.15, align: str = "left", wrap: bool = True, preserve_breaks: bool = True,
                fit: bool = False, min_size: float = 6.0, condense: float = 1.0) -> Layout:
    """Lay out ``text`` in a box of ``box_w`` x ``box_h`` pixels."""
    if fit:
        hi = size
        lay = _layout(text, plan, hi, box_w, box_h, tracking, line_spacing, align, wrap, preserve_breaks, condense)
        if not lay.overflow:
            return lay
        lo = max(1.0, min_size)
        best = _layout(text, plan, lo, box_w, box_h, tracking, line_spacing, align, wrap, preserve_breaks, condense)
        if best.overflow:
            best.warnings.append("The text does not fit in the box even at the minimum size; some lines lie "
                                 "outside the box (nothing was removed).")
            return best
        for _ in range(14):
            mid = (lo + hi) / 2
            cand = _layout(text, plan, mid, box_w, box_h, tracking, line_spacing, align, wrap, preserve_breaks,
                           condense)
            if cand.overflow:
                hi = mid
            else:
                lo, best = mid, cand
            if hi - lo < 0.25:
                break
        return best
    lay = _layout(text, plan, size, box_w, box_h, tracking, line_spacing, align, wrap, preserve_breaks, condense)
    if lay.overflow:
        lay.warnings.append("The text is larger than its box; the overflowing lines are drawn outside the box "
                            "(nothing was removed). Enlarge the box, reduce the size or use 'Fit text to the box'.")
    return lay


def _layout(text, plan, size, box_w, box_h, tracking, line_spacing, align, wrap, preserve_breaks, condense) -> Layout:
    lines: list[list] = []          # each: list of (measured token, kind, word index)
    warnings: list[str] = []
    word_no = -1
    broke = False
    for para in _words(text, preserve_breaks):
        cur, cur_w = [], 0.0
        para_lines = []
        for tok, start in para:
            is_space = tok.isspace()
            if not is_space:
                word_no += 1
            meas, tw = _measure(tok, plan, max(word_no, 0), size, tracking, start)
            tw *= condense
            if is_space:
                if cur:
                    cur.append((meas, "space", word_no, tw))
                    cur_w += tw
                continue
            if wrap and cur and cur_w + tw > box_w:
                para_lines.append(cur)
                cur, cur_w = [], 0.0
            if wrap and tw > box_w and box_w > 0:  # a single word wider than the box: break it, keep every letter
                broke = True

                def rebased(pc):  # each piece starts at x = 0 on its own line
                    x0 = pc[0][1]
                    return [(c, gx - x0, adv, pth, sz, ix) for c, gx, adv, pth, sz, ix in pc]

                piece, pw = [], 0.0
                for g in meas:
                    if piece and pw + g[2] * condense > box_w:
                        if cur:
                            para_lines.append(cur)
                            cur, cur_w = [], 0.0
                        para_lines.append([(rebased(piece), "word", word_no, pw)])
                        piece, pw = [], 0.0
                    piece.append(g)
                    pw += g[2] * condense
                cur, cur_w = [(rebased(piece), "word", word_no, pw)], pw
                continue
            cur.append((meas, "word", word_no, tw))
            cur_w += tw
        para_lines.append(cur)
        for i, ln in enumerate(para_lines):
            lines.append((ln, i == len(para_lines) - 1))
    if broke:
        warnings.append("A word was wider than the box and was split across lines (no letters were removed).")
    font = fonts.load(plan.path, int(round(size)))
    asc, desc = font.getmetrics()
    lh = size * line_spacing
    glyphs: list[Glyph] = []
    boxes = []
    max_w = 0.0
    for li, (ln, last) in enumerate(lines):
        while ln and ln[-1][1] == "space":  # trailing spaces do not count for alignment
            ln = ln[:-1]
        width = sum(t[3] for t in ln)
        max_w = max(max_w, width)
        extra = max(0.0, box_w - width)
        gaps = [i for i, t in enumerate(ln) if t[1] == "space"]
        if align == "center":
            x = extra / 2
        elif align == "right":
            x = extra
        else:
            x = 0.0
        stretch = extra / len(gaps) if (align == "justify" and not last and gaps and wrap) else 0.0
        base = asc + li * lh
        for ti, (meas, kind, wno, tw) in enumerate(ln):
            for ch, gx, adv, p, s, idx in meas:
                if kind == "word":
                    glyphs.append(Glyph(ch, x + gx * condense, base, adv * condense, p, s, idx, wno, li,
                                        scale_x=condense))
            x += tw + (stretch if kind == "space" else 0.0)
        boxes.append((0.0, base - asc, max(x, width), base + desc))
    height = (len(lines) - 1) * lh + asc + desc if lines else 0.0
    overflow = height > box_h + 0.5 or (not wrap and max_w > box_w + 0.5) or (wrap and max_w > box_w + 0.5)
    return Layout(glyphs, boxes, size, max_w, height, overflow, warnings)


def layout_path(text: str, plan: FontPlan, size: float, pts: np.ndarray, *, tracking: float = 0.0,
                align: str = "left") -> Layout:
    """Glyphs following a polyline (image pixels). Line breaks become spaces on a path."""
    flat = text.replace("\n", " ")
    seg = np.diff(pts, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    total = float(seg_len.sum())
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    meas = []
    x = 0.0
    word = -1
    prev_space = True
    for i, ch in enumerate(flat):
        if not ch.isspace() and prev_space:
            word += 1
        prev_space = ch.isspace()
        p = fonts.font_for_char(plan.for_word(max(word, 0))[0], ch)
        adv = _advance(p, size, ch, flat[i + 1] if i + 1 < len(flat) else None) + tracking * size
        meas.append((ch, x, adv, p, i, max(word, 0)))
        x += adv
    start = {"center": (total - x) / 2, "right": total - x}.get(align, 0.0)
    glyphs = []
    for ch, gx, adv, p, i, wno in meas:
        s = start + gx + adv / 2
        s_c = min(max(s, 0.0), total)
        k = int(np.searchsorted(cum, s_c, side="right") - 1)
        k = min(max(k, 0), len(seg) - 1)
        t = (s_c - cum[k]) / max(seg_len[k], 1e-6)
        px, py = pts[k] + seg[k] * t
        ang = math.degrees(math.atan2(seg[k][1], seg[k][0]))
        # extend straight beyond the ends so no letter is lost
        if s < 0 or s > total:
            d = s - s_c
            px += math.cos(math.radians(ang)) * d
            py += math.sin(math.radians(ang)) * d
        if not ch.isspace():
            glyphs.append(Glyph(ch, px, py, adv, p, size, i, wno, 0, angle=ang))
    warnings = []
    overflow = x > total + 1
    if overflow:
        warnings.append("The text is longer than the path; it continues past the end of the path.")
    return Layout(glyphs, [], size, x, size, overflow, warnings)
