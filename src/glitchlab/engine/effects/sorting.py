"""Pixel sorting and tile/strip rearrangement effects."""

from __future__ import annotations

import cv2
import numpy as np

from ..ops import luminance, rgb_to_hsv, to_float, to_uint8
from ..params import COLOR, Param, bparam, color_to_rgb, cparam, fparam, iparam
from .base import Effect, register

_KEY_CHOICES = (("luminance", "Luminance"), ("hue", "Hue"), ("saturation", "Saturation"),
                ("red", "Red"), ("green", "Green"), ("blue", "Blue"))


def _key_map(img: np.ndarray, key: str) -> np.ndarray:
    if key == "luminance":
        return luminance(img)
    if key in ("hue", "saturation"):
        hh, ss, _ = rgb_to_hsv(img)
        return hh if key == "hue" else ss
    ci = {"red": 0, "green": 1, "blue": 2}[key]
    return img[..., ci].astype(np.float32) / 255.0


def _line_sequence(h: int, w: int, direction: str) -> tuple[np.ndarray, np.ndarray]:
    """Flat pixel indices ordered along lines, plus a 'first pixel of line' flag."""
    if direction == "horizontal":
        seq = np.arange(h * w, dtype=np.int64)
        starts = np.zeros(h * w, bool)
        starts[::w] = True
        return seq, starts
    if direction == "vertical":
        seq = np.arange(h * w, dtype=np.int64).reshape(h, w).T.ravel()
        starts = np.zeros(h * w, bool)
        starts[::h] = True
        return seq, starts
    # Diagonals: skew rows so each diagonal becomes one column of a padded
    # index grid, then read column by column (padding = -1 is dropped).
    dt = np.int32 if h * w < 2**31 - 1 else np.int64
    idx = np.arange(h * w, dtype=dt).reshape(h, w)
    pad = np.full((h, w + h - 1), -1, dtype=dt)
    for y in range(h):
        off = (h - 1 - y) if direction == "diag_down" else y
        pad[y, off:off + w] = idx[y]
    cols = pad.T  # (w+h-1, h)
    valid = cols >= 0
    seq = cols[valid].astype(np.int64)
    # first valid element of each column starts a new line
    first = np.zeros_like(valid)
    has = valid.any(axis=1)
    first_pos = valid.argmax(axis=1)
    first[np.flatnonzero(has), first_pos[has]] = True
    starts = first[valid]
    return seq, starts


@register
class PixelSort(Effect):
    id = "pixel_sort"
    name = "Pixel Sorting"
    category = "Sorting & Scrambling"
    description = ("Sorts pixels inside spans whose luminance, hue or saturation lies within a "
                   "threshold interval. Spans can be capped in length and randomised.")
    supports_alpha = True
    preview_note = "Span lengths scale with the preview, but sorting is resolution dependent."
    params = (
        cparam("direction", "Direction", "horizontal",
               (("horizontal", "Horizontal"), ("vertical", "Vertical"),
                ("diag_down", "Diagonal ↘"), ("diag_up", "Diagonal ↗"))),
        Param("interval", "Select spans by", "choice", "luminance",
              choices=_KEY_CHOICES[:3] + (("edges", "Between edges"),),
              tooltip="Pixels whose value lies inside the threshold interval are sorted."),
        fparam("lower", "Lower threshold", 0.25, 0.0, 1.0, show_if=("interval", ("luminance", "hue", "saturation"))),
        fparam("upper", "Upper threshold", 0.85, 0.0, 1.0, show_if=("interval", ("luminance", "hue", "saturation"))),
        fparam("edge_threshold", "Edge sensitivity", 0.5, 0.05, 1.0, show_if=("interval", ("edges",)),
               tooltip="Spans run between detected edges; higher = fewer edges = longer streaks."),
        bparam("invert", "Invert selection", False, tooltip="Sort pixels outside the interval instead."),
        Param("sort_by", "Sort by", "choice", "luminance", choices=_KEY_CHOICES),
        bparam("reverse", "Reverse order", False),
        iparam("max_span", "Max span length", 0, 0, 8000, spatial=True, unit="px",
               tooltip="0 = spans run until the threshold breaks them."),
        fparam("span_random", "Span randomness", 0.0, 0.0, 1.0,
               tooltip="Randomly shortens each span (seeded) when a max length is set."),
    )

    def process(self, img, p, ctx):
        h, w = img.shape[:2]
        if p["interval"] == "edges":
            lum8 = np.clip(np.rint(luminance(img) * 255), 0, 255).astype(np.uint8)
            t = 30 + p["edge_threshold"] * 220
            edges = cv2.Canny(cv2.GaussianBlur(lum8, (3, 3), 0), t * 0.5, t)
            crit = edges == 0
            crit_map = luminance(img)
        else:
            crit_map = _key_map(img, p["interval"])
            lo, hi = sorted((p["lower"], p["upper"]))
            crit = (crit_map >= lo) & (crit_map <= hi)
        if p["invert"]:
            crit = ~crit
        crit &= img[..., 3] > 0  # transparent pixels never take part
        keys = crit_map if p["sort_by"] == p["interval"] else _key_map(img, p["sort_by"])
        keys16 = np.clip(np.rint(keys * 65535.0), 0, 65535).astype(np.int64).ravel()
        if p["reverse"]:
            keys16 = 65535 - keys16
        ctx.check()
        seq, starts = _line_sequence(h, w, p["direction"])
        c = crit.ravel()[seq]
        m = np.flatnonzero(c)
        out = img.reshape(-1, 4).copy()
        if m.size == 0:
            return out.reshape(img.shape)
        prev = np.zeros_like(c)
        prev[1:] = c[:-1]
        run_start = (c & (~prev | starts))[m]
        run_id = np.cumsum(run_start) - 1
        first = np.flatnonzero(run_start)
        pos = np.arange(m.size) - first[run_id]
        max_span = int(p["max_span"])
        if max_span > 0:
            lengths = np.full(first.size, max_span, np.int64)
            if p["span_random"] > 0:
                u = ctx.rng.random(first.size)
                lengths = np.maximum(1, np.rint(max_span * (1.0 - p["span_random"] * u))).astype(np.int64)
            new_span = (pos % lengths[run_id]) == 0
        else:
            new_span = run_start
        span = np.cumsum(new_span).astype(np.int64)
        pix = seq[m]
        comp = span * 65536 + keys16[pix]
        ctx.check()
        order = np.argsort(comp, kind="stable")
        src = img.reshape(-1, 4)
        out[pix] = src[pix[order]]
        return out.reshape(img.shape)


def _tiles(img: np.ndarray, ts: int) -> tuple[np.ndarray, int, int]:
    h, w, c = img.shape
    ny, nx = h // ts, w // ts
    region = img[: ny * ts, : nx * ts]
    t = region.reshape(ny, ts, nx, ts, c).swapaxes(1, 2).reshape(ny * nx, ts, ts, c)
    return np.ascontiguousarray(t), ny, nx


def _untile(tiles: np.ndarray, ny: int, nx: int, ts: int, into: np.ndarray) -> None:
    c = tiles.shape[-1]
    region = tiles.reshape(ny, nx, ts, ts, c).swapaxes(1, 2).reshape(ny * ts, nx * ts, c)
    into[: ny * ts, : nx * ts] = region


@register
class BlockScramble(Effect):
    id = "block_scramble"
    name = "Block Scrambling"
    category = "Sorting & Scrambling"
    description = "Cuts the image into a grid of tiles and shuffles, duplicates, rotates or offsets a seeded selection of them."
    supports_alpha = True
    preview_note = "The preview tile grid is scaled; tile selection can differ slightly from the export."
    params = (
        iparam("tile", "Tile size", 48, 2, 2048, spatial=True, unit="px"),
        fparam("amount", "Affected tiles", 0.3, 0.0, 1.0, tooltip="Fraction of tiles that are altered."),
        cparam("mode", "Operation", "shuffle",
               (("shuffle", "Shuffle"), ("duplicate", "Duplicate"), ("rotate", "Rotate"),
                ("offset", "Offset contents"), ("mixed", "Mixed"))),
        fparam("offset_max", "Max offset", 0.5, 0.0, 1.0, tooltip="Offset mode: fraction of the tile size.",
               show_if=("mode", ("offset", "mixed"))),
    )

    def process(self, img, p, ctx):
        h, w = img.shape[:2]
        ts = int(max(2, min(p["tile"], h, w)))
        tiles, ny, nx = _tiles(img, ts)
        n = tiles.shape[0]
        out = img.copy()
        if n == 0 or p["amount"] <= 0:
            return out
        rng = ctx.rng
        sel = np.flatnonzero(rng.random(n) < p["amount"])
        if sel.size == 0:
            return out
        orig = tiles.copy()
        mode = p["mode"]
        modes = np.full(sel.size, mode, dtype=object)
        if mode == "mixed":
            modes = rng.choice(np.array(["shuffle", "duplicate", "rotate", "offset"], dtype=object), size=sel.size)
        # shuffle: permute selected tiles among themselves
        idx = sel[modes == "shuffle"]
        if idx.size == 1 and n >= 2:
            # shuffling one tile is a no-op: pair it with another random tile
            other = int(rng.integers(0, n - 1))
            other += other >= idx[0]
            idx = np.array([idx[0], other])
        if idx.size > 1:
            # cyclic shift of a random order = every selected tile moves
            order = idx[rng.permutation(idx.size)]
            tiles[order] = orig[np.roll(order, 1)]
        idx = sel[modes == "duplicate"]
        if idx.size:
            tiles[idx] = orig[rng.integers(0, n, size=idx.size)]
        idx = sel[modes == "rotate"]
        if idx.size:
            ks = rng.integers(1, 4, size=idx.size)
            for k in (1, 2, 3):
                kk = idx[ks == k]
                if kk.size:
                    tiles[kk] = np.rot90(orig[kk], k, axes=(1, 2))
        idx = sel[modes == "offset"]
        if idx.size:
            m = max(1, int(round(p["offset_max"] * ts)))
            dy = rng.integers(-m, m + 1, size=idx.size)
            dx = rng.integers(-m, m + 1, size=idx.size)
            ar = np.arange(ts)
            ys = (ar[None, :] - dy[:, None]) % ts
            xs = (ar[None, :] - dx[:, None]) % ts
            tiles[idx] = orig[idx[:, None, None], ys[:, :, None], xs[:, None, :]]
        _untile(tiles, ny, nx, ts, out)
        return out


# ---------------------------------------------------------------------------
# Slit-scan collage
# ---------------------------------------------------------------------------

@register
class SlitScan(Effect):
    id = "slitscan"
    name = "Slit-Scan Collage"
    category = "Sorting & Scrambling"
    description = ("Rebuilds the image from strips sampled at other positions of the same still: drifting, "
                   "waving, reversed, interleaved or seeded random order; strips or stretched 1-px slits.")
    supports_alpha = True
    params = (
        cparam("orientation", "Strips", "vertical", (("vertical", "Vertical strips"), ("horizontal", "Horizontal strips"))),
        iparam("strip", "Strip width", 8, 1, 1000, spatial=True, unit="px"),
        cparam("order", "Order", "drift",
               (("drift", "Drift"), ("wave", "Wave"), ("random", "Random (seeded)"), ("reverse", "Reverse"),
                ("interleave", "Interleave halves"), ("mirror", "Mirror halves"))),
        fparam("drift", "Drift", 0.25, -2.0, 2.0, tooltip="Extra sampling offset per strip, in strip widths.",
               show_if=("order", ("drift",))),
        iparam("wave_amount", "Wave amount", 40, 0, 4000, spatial=True, unit="px"),
        fparam("wave_cycles", "Wave cycles", 2.0, 0.0, 40.0),
        fparam("stretch", "Sampling stretch", 1.0, 0.1, 4.0, tooltip="Scales the source position of every strip."),
        cparam("sampling", "Sampling", "block", (("block", "Copy strip"), ("slit", "Stretch 1-px slit"))),
    )

    def process(self, img, p, ctx):
        work = img if p["orientation"] == "vertical" else np.swapaxes(img, 0, 1)
        h, w = work.shape[:2]
        sw = max(1, int(p["strip"]))
        n = int(np.ceil(w / sw))
        i = np.arange(n, dtype=np.float64)
        order = p["order"]
        if order == "drift":
            pos = i * sw * (1.0 + p["drift"])
        elif order == "random":
            pos = ctx.rng.permutation(n).astype(np.float64) * sw
        elif order == "reverse":
            pos = (n - 1 - i) * sw
        elif order == "interleave":
            half = (n + 1) // 2
            pos = np.where(i % 2 == 0, (i // 2), half + i // 2) * sw
        elif order == "mirror":
            pos = np.where(i < n / 2, i, n - 1 - (i - n // 2)) * sw
        else:  # wave
            pos = i * sw
        pos = pos * p["stretch"]
        if p["wave_amount"]:
            pos = pos + p["wave_amount"] * np.sin(2 * np.pi * p["wave_cycles"] * (i + 0.5) / max(n, 1))
        pos = np.rint(pos).astype(np.int64)
        within = np.arange(w) % sw
        strip_of = np.arange(w) // sw
        if p["sampling"] == "slit":
            cols = pos[strip_of] + sw // 2
        else:
            cols = pos[strip_of] + within
        cols = np.mod(cols, w)
        out = work[:, cols]
        if p["orientation"] != "vertical":
            out = np.swapaxes(out, 0, 1)
        return np.ascontiguousarray(out)


# ---------------------------------------------------------------------------
# Mosaic bleed
# ---------------------------------------------------------------------------

def _feather_window(ts: int, feather: float) -> np.ndarray:
    if feather <= 0:
        return np.ones((ts, ts), np.float32)
    fw = max(1.0, feather * ts / 2.0)
    r = np.arange(ts, dtype=np.float32)
    ramp = np.clip(np.minimum(r + 1, ts - r) / fw, 0.0, 1.0)
    return np.outer(ramp, ramp).astype(np.float32)


@register
class MosaicBleed(Effect):
    id = "mosaic_bleed"
    name = "Mosaic Bleed"
    category = "Sorting & Scrambling"
    description = ("Replaces seeded blocks with neighbouring or repeated blocks (stuck, smeared tiles) with "
                   "controllable chroma leakage and feathered edges. A still-image effect, not video datamoshing.")
    supports_alpha = True
    preview_note = "The block grid is scaled with the preview; block selection can differ slightly."
    params = (
        iparam("block", "Block size", 32, 4, 1024, spatial=True, unit="px"),
        fparam("amount", "Affected blocks", 0.2, 0.0, 1.0),
        cparam("source", "Replacement", "repeat",
               (("neighbour", "Random neighbour"), ("repeat", "Repeat (stuck run)"), ("smear", "Smear streak"))),
        cparam("direction", "Direction", "down", (("down", "Down"), ("up", "Up"), ("right", "Right"), ("left", "Left")),
               show_if=("source", ("repeat", "smear"))),
        iparam("run", "Run length", 4, 1, 64, show_if=("source", ("repeat", "smear")),
               tooltip="How many blocks a stuck block is copied into."),
        fparam("leakage", "Colour leakage", 0.35, 0.0, 1.0, tooltip="Chroma bleeds out of replaced blocks."),
        fparam("feather", "Edge feather", 0.2, 0.0, 1.0, tooltip="Soft edges of pasted blocks (fraction of block)."),
    )

    def process(self, img, p, ctx):
        from ..ops import rgb_to_ycbcr, ycbcr_to_rgb

        h, w = img.shape[:2]
        ts = int(max(2, min(p["block"], h, w)))
        ny, nx = h // ts, w // ts
        if ny == 0 or nx == 0 or p["amount"] <= 0:
            return img.copy()
        rng = ctx.rng
        sel = np.flatnonzero(rng.random(ny * nx) < p["amount"])
        pasted = img.copy()
        weight = np.zeros((h, w), np.float32)
        win = _feather_window(ts, p["feather"])
        dirs = {"down": (1, 0), "up": (-1, 0), "right": (0, 1), "left": (0, -1)}
        dy, dx = dirs[p["direction"]]
        for k, b in enumerate(sel):
            if k % 512 == 0:
                ctx.check()
            by, bx = divmod(int(b), nx)
            if p["source"] == "neighbour":
                sy = int(np.clip(by + rng.integers(-1, 2), 0, ny - 1))
                sx = int(np.clip(bx + rng.integers(-1, 2), 0, nx - 1))
                block = img[sy * ts:(sy + 1) * ts, sx * ts:(sx + 1) * ts]
                targets = [(by, bx)]
            else:
                block = img[by * ts:(by + 1) * ts, bx * ts:(bx + 1) * ts]
                run = int(p["run"]) if p["source"] == "smear" else max(1, int(rng.integers(1, p["run"] + 1)))
                targets = [(by + dy * r, bx + dx * r) for r in range(1, run + 1)]
            for (ty, tx) in targets:
                if not (0 <= ty < ny and 0 <= tx < nx):
                    break
                ys, xs = slice(ty * ts, (ty + 1) * ts), slice(tx * ts, (tx + 1) * ts)
                cur = weight[ys, xs]
                upd = win > cur
                if upd.all():
                    pasted[ys, xs] = block
                    weight[ys, xs] = win
                else:
                    pasted[ys, xs] = np.where(upd[..., None], block, pasted[ys, xs])
                    np.maximum(cur, win, out=cur)
        if not weight.any():
            return img.copy()
        # blend only the rows that were touched (bounded temporaries)
        out = img.copy()
        rows = np.flatnonzero(weight.any(axis=1))
        y0, y1 = int(rows[0]), int(rows[-1]) + 1
        wm = weight[y0:y1, :, None]
        a = img[y0:y1].astype(np.float32)
        out[y0:y1] = to_uint8(a + (pasted[y0:y1].astype(np.float32) - a) * wm, scale=1.0)
        if p["leakage"] > 0:
            ycc = rgb_to_ycbcr(out)
            k = max(3, int(ts * 1.5) | 1)
            blurred = cv2.blur(ycc[..., 1:], (k, k))
            spread = cv2.blur(weight, (k, k))[..., None]
            amt = np.clip(spread * 1.5, 0, 1) * float(p["leakage"])
            ycc[..., 1:] = ycc[..., 1:] * (1 - amt) + blurred * amt
            out[..., :3] = ycbcr_to_rgb(ycc)
        return out


# ---------------------------------------------------------------------------
# Voronoi shatter sorting
# ---------------------------------------------------------------------------

def voronoi_labels(h: int, w: int, points: np.ndarray) -> np.ndarray:
    """Label map (H, W int32) assigning every pixel to its nearest seed point."""
    src = np.full((h, w), 255, np.uint8)
    px = np.clip(points[:, 0].astype(np.int64), 0, w - 1)
    py = np.clip(points[:, 1].astype(np.int64), 0, h - 1)
    src[py, px] = 0
    _dist, labels = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    return labels.astype(np.int32)


@register
class VoronoiShatterSort(Effect):
    id = "voronoi_sort"
    name = "Voronoi Shatter Sorting"
    category = "Sorting & Scrambling"
    description = ("Breaks the image into irregular glass-like Voronoi cells and sorts the colours inside every "
                   "cell from dark to light: recognisable from afar, melting stained glass up close.")
    supports_alpha = True
    params = (
        iparam("cells", "Cells across", 40, 3, 600, tooltip="Number of cells across the long side."),
        Param("sort_by", "Sort by", "choice", "luminance", choices=_KEY_CHOICES),
        cparam("direction", "Sort direction", "vertical",
               (("vertical", "Top to bottom"), ("horizontal", "Left to right"), ("diagonal", "Diagonal"),
                ("radial", "Centre outwards"))),
        bparam("reverse", "Light to dark", False),
        iparam("lead", "Lead lines", 1, 0, 40, spatial=True, unit="px", tooltip="Width of the dark lines between cells."),
        Param("lead_color", "Lead colour", COLOR, "#161616"),
    )

    def process(self, img, p, ctx):
        h, w = img.shape[:2]
        long_side, short_side = max(h, w), min(h, w)
        n = max(2, int(round(p["cells"] ** 2 * short_side / long_side)))
        pts = ctx.rng.random((n, 2)) * np.array([w, h], np.float64)
        labels = voronoi_labels(h, w, pts).ravel().astype(np.int64)
        ctx.check()
        keys16 = np.clip(np.rint(_key_map(img, p["sort_by"]) * 65535.0), 0, 65535).astype(np.int64).ravel()
        if p["reverse"]:
            keys16 = 65535 - keys16
        yy, xx = np.divmod(np.arange(h * w, dtype=np.int64), w)
        d = p["direction"]
        if d == "vertical":
            pos = yy * w + xx
        elif d == "horizontal":
            pos = xx * h + yy
        elif d == "diagonal":
            pos = (xx + yy) * (w + 1) + xx
        else:
            cnt = np.bincount(labels).astype(np.float64)
            cnt[cnt == 0] = 1
            cx = np.bincount(labels, weights=xx) / cnt
            cy = np.bincount(labels, weights=yy) / cnt
            pos = np.rint(np.hypot(xx - cx[labels], yy - cy[labels]) * 16).astype(np.int64)
        big = int(pos.max()) + 1
        order_pos = np.argsort(labels * big + pos, kind="stable")
        ctx.check()
        order_col = np.argsort(labels * 65536 + keys16, kind="stable")
        src = img.reshape(-1, 4)
        out = np.empty_like(src)
        out[order_pos] = src[order_col]
        out = out.reshape(img.shape)
        if p["lead"] > 0:
            lab = labels.reshape(h, w)
            border = np.zeros((h, w), np.uint8)
            border[:, 1:] |= (lab[:, 1:] != lab[:, :-1])
            border[1:, :] |= (lab[1:, :] != lab[:-1, :])
            k = int(p["lead"])
            if k > 1:
                border = cv2.dilate(border, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
            col = np.array(color_to_rgb(p["lead_color"]), np.uint8)
            out[border.astype(bool), :3] = col
        return out
