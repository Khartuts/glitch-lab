"""AI-assisted effects driven by local vision models (segmentation, depth,
dense features).  Inference is cached (see glitchlab.ai.analysis); the
styling below is plain NumPy/OpenCV and reruns instantly when a styling
parameter changes.

The models' outputs are treated as artistic raw material: segmentation
masks are approximate, inferred depth is a relative guess rather than scene
geometry, and feature similarity is not object recognition.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ..params import COLOR, FILE, MASK, POINTS, Param, bparam, cparam, fparam, iparam
from .ai_common import GAP_FILLS, apply_gap_fill, heat, hue_matrix, label_colours, nearest_fill, overlay_map, smooth_noise
from .base import Effect, register

CAT = "AI Experimental"
IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff);;All files (*)"


def _analysis():
    from ...ai import analysis  # noqa: PLC0415 - keeps PyTorch-free imports cheap

    return analysis


# =====================================================================
# Semantic Faultlines
# =====================================================================
def _contour_arclength_labels(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """For every pixel: the normalised arc-length position (0..1) of the
    nearest point on the object outline, and the outline normal there."""
    h, w = mask.shape
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    pts = [c[:, 0, :] for c in contours if len(c) >= 3]
    if not pts:
        return np.zeros((h, w), np.float32), np.zeros((h, w, 2), np.float32), 0.0
    arcs = []
    total = 0.0
    for c in pts:
        seg = np.sqrt((np.diff(np.vstack([c, c[:1]]), axis=0) ** 2).sum(-1))
        arcs.append(total + np.concatenate([[0.0], np.cumsum(seg[:-1])]))
        total += float(seg.sum())
    allp = np.vstack(pts)
    alls = np.concatenate(arcs) / max(total, 1e-6)
    img = np.ones((h, w), np.uint8)
    img[allp[:, 1], allp[:, 0]] = 0
    s_at = np.zeros(h * w, np.float32)
    s_at[allp[:, 1] * w + allp[:, 0]] = alls
    _, labels = cv2.distanceTransformWithLabels(img, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    zero_idx = np.flatnonzero(img.ravel() == 0)
    lut = np.concatenate([[0.0], s_at[zero_idx]]).astype(np.float32)
    return lut[labels].astype(np.float32), None, total


@register
class SemanticFaultlines(Effect):
    id = "semantic_faultlines"
    name = "Semantic Faultlines"
    category = CAT
    ai_models = ("sam2",)
    supports_alpha = True
    supports_channels = False
    description = ("AI-assisted (SAM 2 segmentation, runs locally). Objects you select come loose from their own "
                   "outlines: the boundary is split into irregular concentric bands that crack into fragments, "
                   "each shifted, rotated and colour-separated on its own, exposing colour strata and gaps. "
                   "Skin slip moves the object's inside while its silhouette stays put.")
    preview_note = "The object mask is computed once (at about 1024 px) and reused for preview and export."
    params = (
        Param("objects", "Objects", POINTS, [], tooltip="Click 'Pick on image', then click on objects in the "
              "preview. Right-click marks areas to exclude. 'New object' starts another object."),
        iparam("width", "Fracture width", 48, 2, 3000, spatial=True, unit="px",
               tooltip="Width of the boundary zone that fractures."),
        fparam("position", "Band position", 0.0, -1.0, 1.0, tooltip="-1 = inside the object, +1 = outside it."),
        iparam("bands", "Band count", 6, 1, 24),
        fparam("irregularity", "Band irregularity", 0.5, 0.0, 1.0),
        iparam("displacement", "Displacement", 28, 0, 3000, spatial=True, unit="px"),
        fparam("rotation", "Fragment rotation", 4.0, 0.0, 45.0, decimals=1, unit="°"),
        fparam("discontinuity", "Discontinuity", 0.65, 0.0, 1.0,
               tooltip="How independently neighbouring fragments move: 0 = smooth halo, 1 = shattered."),
        iparam("fragments", "Fragments per band", 24, 2, 256),
        iparam("separation", "Colour separation", 6, 0, 600, spatial=True, unit="px"),
        fparam("strata", "Colour strata", 0.5, 0.0, 1.0, tooltip="Hue shift between the bands."),
        cparam("interior", "Object interior", "slip", (("keep", "Keep"), ("slip", "Skin slip"),
                                                       ("strata", "Colour strata"), ("transparent", "Remove (gap)"))),
        iparam("slip", "Interior slip", 40, 0, 6000, spatial=True, unit="px", show_if=("interior", ("slip",))),
        fparam("slip_angle", "Slip direction", 30.0, -180.0, 180.0, decimals=0, unit="°",
               show_if=("interior", ("slip",))),
        cparam("background", "Background", "keep", (("keep", "Keep"), ("dim", "Dim"), ("desaturate", "Desaturate"),
                                                    ("blur", "Blur"), ("transparent", "Remove (gap)"))),
        fparam("background_amount", "Background amount", 0.6, 0.0, 1.0,
               show_if=("background", ("dim", "desaturate", "blur"))),
        cparam("gap_fill", "Gap fill", "stretch", GAP_FILLS),
        Param("gap_colour", "Gap colour", COLOR, "#0b0b10", show_if=("gap_fill", ("colour", "transparent"))),
        bparam("boundary", "Fracture the boundary", True),
        cparam("view", "Show", "result", (("result", "Result"), ("mask", "Object mask"),
                                          ("bands", "Bands and fragments"))),
    )

    def process(self, img, p, ctx):
        h, w = img.shape[:2]
        pts = p["objects"]
        if not any(pt[2] for pt in pts):
            ctx.warn("No object selected yet: use 'Pick on image' and click an object. Using the image centre.")
            pts = [[0.5, 0.5, 1, 1]]
        an = _analysis()
        logits = an.object_logits(ctx, img, pts)
        masks = an.masks_from_logits(logits, w, h)
        M = masks.any(0)
        if p["view"] == "mask":
            cols = label_colours(len(masks))
            out = img.copy()
            for i, m in enumerate(masks):
                out = overlay_map(out, m.astype(np.float32), tuple(int(c) for c in cols[i]), 0.55)
            return out
        if not M.any():
            ctx.warn("The selected points did not produce an object mask; try clicking elsewhere.")
            return img.copy()
        rng = ctx.rng
        inside = M.astype(np.uint8)
        d_in = cv2.distanceTransform(inside, cv2.DIST_L2, 5)
        d_out = cv2.distanceTransform(1 - inside, cv2.DIST_L2, 5)
        sdf = d_out - d_in  # render pixels, > 0 outside the object
        W = float(max(1, p["width"]))
        N = int(p["bands"])
        cell_full = (W / max(ctx.scale, 1e-6)) * 1.5
        noise = smooth_noise(ctx.sub_rng(1), w, h, ctx.full_size, cell_full)
        u = (sdf - p["position"] * W / 2.0) / W + 0.5 + noise * p["irregularity"] * 0.35
        region = (u >= 0.0) & (u < 1.0) if p["boundary"] else np.zeros((h, w), bool)
        band = np.clip(np.floor(u * N), 0, N - 1).astype(np.int32)
        s_norm, _, _ = _contour_arclength_labels(M)

        # ---- per band: crack positions along the outline, per fragment: transform
        K = int(p["fragments"])
        disc = float(p["discontinuity"])
        brng = ctx.sub_rng(2)
        cuts, band_dir, band_mag, band_hue = [], [], [], []
        frag_t, frag_rot, frag_still = [], [], []
        for b in range(N):
            c = np.sort(brng.random(K - 1)) if K > 1 else np.zeros(0)
            # less discontinuity -> more regular cracks
            even = (np.arange(1, K) / K)
            cuts.append(np.sort(even * (1 - disc) + c * disc + brng.random() / K))
            ang = brng.uniform(-math.pi, math.pi)
            band_dir.append((math.cos(ang), math.sin(ang)))
            band_mag.append(brng.uniform(0.35, 1.0))
            band_hue.append((b / max(N - 1, 1) - 0.5) * 140.0 * p["strata"] + brng.normal(0, 12) * p["strata"])
            ts, rs, st = [], [], []
            for _k in range(K + 1):
                if brng.random() < disc:
                    a2 = brng.uniform(-math.pi, math.pi)
                    m2 = brng.uniform(0.2, 1.0)
                    t = (math.cos(a2) * m2, math.sin(a2) * m2)
                else:
                    t = (band_dir[-1][0] * band_mag[-1], band_dir[-1][1] * band_mag[-1])
                ts.append(t)
                rs.append(brng.uniform(-1, 1) * (0.3 + 0.7 * disc))
                st.append(brng.random() < 0.18 * disc)
            frag_t.append(ts)
            frag_rot.append(rs)
            frag_still.append(st)
        frag_cell = np.zeros((h, w), np.int32)
        for b in range(N):
            sel = region & (band == b)
            if sel.any():
                frag_cell[sel] = np.searchsorted(cuts[b], np.mod(s_norm[sel], 1.0))
        labels = np.where(region, band * (K + 1) + frag_cell, -1)

        if p["view"] == "bands":
            cols = label_colours(N * (K + 1), seed=int(ctx.sub_rng(6).integers(0, 2**31)))
            out = img.copy()
            vis = cols[np.clip(labels, 0, None)]
            out[region, :3] = (0.35 * img[region, :3] + 0.65 * vis[region]).astype(np.uint8)
            return out

        # ---- base: interior / background processed independently; the boundary zone is a gap
        src = img.astype(np.float32)
        out = src.copy()
        gaps = region.copy()
        interior = M & ~region
        back = ~M & ~region
        mode_i = p["interior"]
        if mode_i == "slip" and p["slip"] > 0:
            a = math.radians(p["slip_angle"])
            vx, vy = p["slip"] * math.cos(a), p["slip"] * math.sin(a)
            T = np.float32([[1, 0, vx], [0, 1, vy]])
            shifted = cv2.warpAffine(src, T, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            smask = cv2.warpAffine(M.astype(np.float32), T, (w, h), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT) > 0.5
            ok = interior & smask
            out[ok] = shifted[ok]
            gaps |= interior & ~smask
        elif mode_i == "strata":
            depth_in = np.clip(-sdf, 0, None)
            k = np.floor(depth_in / max(W, 1.0)).astype(np.int32) % 6
            for j in range(6):
                sel = interior & (k == j)
                if sel.any():
                    mat = hue_matrix(60.0 * j * p["strata"] + 20.0, 1.0 + 0.4 * p["strata"])
                    out[sel, :3] = out[sel, :3] @ mat.T
        elif mode_i == "transparent":
            gaps |= interior
        mode_b = p["background"]
        amt = float(p["background_amount"])
        if mode_b == "dim":
            out[back, :3] *= 1.0 - 0.85 * amt
        elif mode_b == "desaturate":
            g = out[..., :3].mean(-1, keepdims=True)
            out[back, :3] = out[back, :3] * (1 - amt) + g[back] * amt
        elif mode_b == "blur":
            sig = max(0.5, amt * W * 0.5)
            bl = cv2.GaussianBlur(src, (0, 0), sig)
            out[back] = bl[back]
        elif mode_b == "transparent":
            gaps |= back
        out = np.clip(out, 0, 255)
        base = apply_gap_fill(out.astype(np.uint8), gaps, p["gap_fill"], p["gap_colour"], ctx).astype(np.float32)

        # ---- fragments: each piece of the boundary zone moves on its own
        if region.any():
            disp = float(p["displacement"])
            sep = float(p["separation"])
            rot_max = float(p["rotation"])
            flat = labels.ravel()
            idx = np.flatnonzero(flat >= 0)
            labs = flat[idx]
            order = np.argsort(labs, kind="stable")
            idx, labs = idx[order], labs[order]
            starts = np.flatnonzero(np.r_[True, labs[1:] != labs[:-1]])
            ends = np.r_[starts[1:], len(labs)]
            paint = ctx.sub_rng(3).permutation(len(starts))
            gy, gx = np.gradient(sdf)
            margin = int(math.ceil(disp + sep + 4))
            for n_done, gi in enumerate(paint):
                if n_done % 64 == 0:
                    ctx.check()
                a0, a1 = starts[gi], ends[gi]
                lab = int(labs[a0])
                b, c = divmod(lab, K + 1)
                pix = idx[a0:a1]
                ys, xs = pix // w, pix % w
                cy, cx = float(ys.mean()), float(xs.mean())
                tx, ty = frag_t[b][c]
                nx, ny = float(gx[ys, xs].mean()), float(gy[ys, xs].mean())
                nn = math.hypot(nx, ny) or 1.0
                drift = 0.5 * (1.0 - 0.5 * disc)
                dx = disp * (tx * (1 - drift) + nx / nn * drift)
                dy = disp * (ty * (1 - drift) + ny / nn * drift)
                ang = rot_max * frag_rot[b][c]
                if frag_still[b][c]:
                    dx = dy = ang = 0.0
                rad = int(math.ceil(max(ys.max() - ys.min(), xs.max() - xs.min()) * abs(math.sin(math.radians(ang)))))
                m = margin + rad
                y0, y1 = max(0, ys.min() - m), min(h, ys.max() + 1 + m)
                x0, x1 = max(0, xs.min() - m), min(w, xs.max() + 1 + m)
                ch, cw = y1 - y0, x1 - x0
                fmask = np.zeros((ch, cw), np.float32)
                fmask[ys - y0, xs - x0] = 1.0
                crop = src[y0:y1, x0:x1]
                R = cv2.getRotationMatrix2D((cx - x0, cy - y0), ang, 1.0)
                R[0, 2] += dx
                R[1, 2] += dy
                hue = hue_matrix(band_hue[b])
                crop_h = np.dstack([np.clip(crop[..., :3] @ hue.T, 0, 255), crop[..., 3:4], fmask])
                warped = cv2.warpAffine(crop_h, R, (cw, ch), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                alpha = warped[..., 4]
                tile = base[y0:y1, x0:x1]
                chans = {1: (warped[..., 1], alpha)}
                if sep > 0:
                    bd = band_dir[b]
                    for chn, sgn in ((0, 1.0), (2, -1.0)):
                        Rc = R.copy()
                        Rc[0, 2] += sgn * sep * -bd[1]
                        Rc[1, 2] += sgn * sep * bd[0]
                        wc = cv2.warpAffine(np.dstack([crop_h[..., chn], fmask]), Rc, (cw, ch),
                                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                        chans[chn] = (wc[..., 0], wc[..., 1])
                else:
                    chans[0] = (warped[..., 0], alpha)
                    chans[2] = (warped[..., 2], alpha)
                for chn, (val, a) in chans.items():
                    tile[..., chn] = tile[..., chn] * (1 - a) + val * a
                tile[..., 3] = tile[..., 3] * (1 - alpha) + warped[..., 3] * alpha
        return np.clip(base, 0, 255).astype(np.uint8)


# =====================================================================
# Depth Mutiny
# =====================================================================
@register
class DepthMutiny(Effect):
    id = "depth_mutiny"
    name = "Depth Mutiny"
    category = CAT
    ai_models = ("depth",)
    supports_alpha = True
    supports_channels = False
    description = ("AI-assisted (Depth Anything V2 Small, runs locally). The photo's estimated depth is cut into "
                   "slabs that are folded, reversed and pushed around in 2.5D: background fragments punch "
                   "through the foreground and surfaces fold into ribbons. The depth is an artistic guess, "
                   "not real scene geometry.")
    preview_note = "Depth is estimated once (about 518 px) and reused for preview and export."
    params = (
        cparam("mode", "Mode", "accordion", (("accordion", "Accordion fold"), ("inside_out", "Inside-out perspective"),
                                             ("avalanche", "Depth avalanche"))),
        iparam("slabs", "Slab count", 8, 2, 48),
        iparam("amplitude", "Fold amplitude", 40, 0, 4000, spatial=True, unit="px"),
        fparam("direction", "Perspective direction", 0.0, -180.0, 180.0, decimals=0, unit="°"),
        fparam("inversion", "Inversion", 0.35, 0.0, 1.0, tooltip="Share of slabs whose depth order is reversed."),
        fparam("jitter", "Discontinuity", 0.3, 0.0, 1.0, tooltip="Random extra offsets per slab."),
        cparam("collision", "Collision", "correct", (("correct", "Nearer wins (correct occlusion)"),
                                                     ("inverted", "Farther wins (inverted occlusion)"),
                                                     ("slab", "Later slab wins"))),
        cparam("gap_fill", "Gap fill", "stretch", (("transparent", "Transparent"),
                                                   ("stretch", "Directional stretch"),
                                                   ("mirror", "Mirrored fragments"), ("colour", "Colour"))),
        Param("gap_colour", "Gap colour", COLOR, "#000000", show_if=("gap_fill", ("colour", "transparent"))),
        cparam("view", "Show", "result", (("result", "Result"), ("depth", "Depth map"), ("slabs", "Depth slabs"))),
    )

    def process(self, img, p, ctx):
        h, w = img.shape[:2]
        an = _analysis()
        raw = an.depth(ctx, img)
        d = an.robust_normalise(cv2.resize(raw, (w, h), interpolation=cv2.INTER_LINEAR))  # 1 = near
        N = int(p["slabs"])
        slab = np.clip(np.floor(d * N), 0, N - 1).astype(np.int32)
        if p["view"] == "depth":
            out = img.copy()
            out[..., :3] = heat(d)
            return out
        rng = ctx.sub_rng(1)
        n_inv = int(round(p["inversion"] * N))
        inverted = np.zeros(N, bool)
        inverted[rng.permutation(N)[:n_inv]] = True
        jit = rng.uniform(-1, 1, N).astype(np.float32) * p["jitter"]
        order = np.arange(N)
        perm = rng.permutation(N)
        order[inverted] = perm[np.isin(perm, np.flatnonzero(inverted))] if n_inv else order[inverted]
        if p["view"] == "slabs":
            cols = label_colours(N, 11)
            out = img.copy()
            out[..., :3] = (0.4 * img[..., :3] + 0.6 * cols[slab]).astype(np.uint8)
            return out
        t = d * N - slab  # position inside the slab 0..1
        mode = p["mode"]
        if mode == "accordion":
            z = np.where(slab % 2 == 0, t, 1.0 - t)
            z = np.where(inverted[slab], 1.0 - z, z)
            mag = (z - 0.5) * 2.0 + jit[slab]
            zprio = (slab + np.where(inverted[slab], 1.0 - t, t)) / N
        elif mode == "inside_out":
            z = np.where(inverted[slab], 1.0 - d, d)
            z = (order[slab] + (z * N - np.floor(z * N))) / N if n_inv else z
            mag = (z - 0.5) * 2.0 + jit[slab]
            zprio = z
        else:  # avalanche: slabs cascade, farther ones slide further
            rank = order[slab].astype(np.float32) / max(N - 1, 1)
            fall = (1.0 - rank) ** 1.5
            mag = fall + jit[slab] * 0.5
            zprio = np.where(inverted[slab], 1.0 - d, d)
        amp = float(p["amplitude"])
        a = math.radians(p["direction"])
        ux, uy = math.cos(a), math.sin(a)
        mag = np.clip(mag, -1.5, 1.5).astype(np.float32)
        dx, dy = mag * amp * ux, mag * amp * uy
        collision = p["collision"]
        prio = zprio if collision == "correct" else (1.0 - zprio if collision == "inverted" else slab / N + t * 1e-3)
        out, holes = _splat(img, dx, dy, prio.astype(np.float32))
        # thin cracks from rounding are closed by nearest pixels; real gaps get the chosen fill
        big = cv2.morphologyEx(holes.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
        small = holes & ~big
        if small.any():
            out = nearest_fill(out, small)
        fill = p["gap_fill"]
        if big.any():
            if fill == "stretch":
                out = _directional_fill(out, big, -ux, -uy, int(math.ceil(abs(amp) * 1.6)) + 2)
            elif fill == "mirror":
                xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
                mir = cv2.remap(img, xs + 2 * dx, ys + 2 * dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                mir = mir[:, ::-1] if ctx.sub_rng(5).random() < 0.5 else mir[::-1]
                out[big] = mir[big]
            else:
                out = apply_gap_fill(out, big, fill, p["gap_colour"], ctx)
        return out


def _splat(img: np.ndarray, dx: np.ndarray, dy: np.ndarray, prio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forward-warp every pixel by (dx, dy); collisions keep the highest priority. Returns (image, holes)."""
    h, w = img.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    tx = np.rint(xs + dx).astype(np.int64)
    ty = np.rint(ys + dy).astype(np.int64)
    ok = (tx >= 0) & (tx < w) & (ty >= 0) & (ty < h)
    src = np.flatnonzero(ok.ravel())
    tgt = (ty * w + tx).ravel()[src]
    pr = prio.ravel()[src].astype(np.float64)
    key = tgt.astype(np.float64) + np.clip(pr, 0, 1) * 0.999
    order = np.argsort(key, kind="stable")
    tgt_s = tgt[order]
    last = np.r_[tgt_s[1:] != tgt_s[:-1], True]
    winners = src[order[last]]
    out = np.zeros_like(img)
    flat_out = out.reshape(-1, img.shape[2])
    flat_out[tgt_s[last]] = img.reshape(-1, img.shape[2])[winners]
    holes = np.ones(h * w, bool)
    holes[tgt_s[last]] = False
    return out, holes.reshape(h, w)


def _directional_fill(img: np.ndarray, holes: np.ndarray, ux: float, uy: float, max_dist: int) -> np.ndarray:
    """Fill holes by stretching the nearest pixels found along direction (ux, uy)."""
    out = img.copy()
    todo = holes.copy()
    h, w = holes.shape
    step = 1
    while step <= max_dist and todo.any():
        ox, oy = int(round(ux * step)), int(round(uy * step))
        M = np.float32([[1, 0, -ox], [0, 1, -oy]])
        shifted = cv2.warpAffine(out, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
        valid = cv2.warpAffine((~todo).astype(np.uint8), M, (w, h), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT).astype(bool)
        take = todo & valid
        out[take] = shifted[take]
        todo &= ~take
        step *= 2
    if todo.any():
        out = nearest_fill(out, todo)
    return out


# =====================================================================
# Feature Contagion
# =====================================================================
@register
class FeatureContagion(Effect):
    id = "feature_contagion"
    name = "Feature Contagion"
    category = CAT
    ai_models = ("dinov2",)
    supports_channels = False
    description = ("AI-assisted (DINOv2 features, runs locally). Regions that look alike to the model infect "
                   "each other: patches are replaced by real pixels from distant, similar-looking patches "
                   "(or deliberately dissimilar ones), optionally spreading for several generations or taking "
                   "donors from a second image. Similarity of learned features is not object recognition.")
    preview_note = "Features are extracted once; patch size, density and blending are adjusted instantly."
    params = (
        cparam("mode", "Mode", "similar", (("similar", "Similarity infection"), ("dissimilar", "Dissimilarity invasion"))),
        iparam("patch", "Patch scale", 64, 8, 2000, spatial=True, unit="px"),
        fparam("distance", "Donor distance", 0.25, 0.0, 0.9, tooltip="Minimum distance to the donor, as a fraction "
               "of the image diagonal (ignored for a donor image)."),
        fparam("similarity", "Similarity range", 0.03, 0.001, 0.5, decimals=3,
               tooltip="Pick donors among this share of the best (or worst) matching candidates."),
        fparam("density", "Infection density", 0.25, 0.0, 1.0),
        iparam("generations", "Generations", 1, 1, 8),
        fparam("feather", "Boundary feathering", 0.35, 0.0, 1.0),
        fparam("colour_match", "Colour matching", 0.3, 0.0, 1.0),
        iparam("detail", "Feature detail", 40, 16, 64, tooltip="Feature grid along the long side (higher = finer "
               "but slower analysis)."),
        Param("donor_image", "Donor image (optional)", FILE, "", file_filter=IMAGE_FILTER),
        bparam("show_map", "Show infected patches", False),
    )

    def process(self, img, p, ctx):
        h, w = img.shape[:2]
        an = _analysis()
        feats = an.features(ctx, img, int(p["detail"]))
        fw_full, fh_full = ctx.full_size
        pw_full = max(4.0, p["patch"] / max(ctx.scale, 1e-6))
        gw = int(min(256, max(2, round(fw_full / pw_full))))
        gh = int(min(256, max(2, round(fh_full / pw_full))))
        F = _grid_features(feats, gw, gh)
        donor_rgba = None
        if p["donor_image"]:
            raw = ctx.services.load_aux_image(p["donor_image"]) if ctx.services else None
            if raw is None:
                from ...engine.services import default_services  # noqa: PLC0415

                raw = default_services().load_aux_image(p["donor_image"])
            donor_full = _cover(raw, fw_full, fh_full)
            Fd = _grid_features(an.features_of(donor_full, an.array_identity(donor_full[::7, ::7]) + str(donor_full.shape),
                                               int(p["detail"]), ctx), gw, gh)
            donor_rgba = _cover(raw, w, h)
        else:
            Fd = F
        xs = np.round(np.arange(gw + 1) * w / gw).astype(int)
        ys = np.round(np.arange(gh + 1) * h / gh).astype(int)
        rng = ctx.sub_rng(1)
        n = gw * gh
        cur = img.astype(np.float32)
        infected = np.zeros(n, bool)
        dests = np.flatnonzero(rng.random(n) < p["density"])
        diag = math.hypot(gw, gh)
        min_dist = 0.0 if donor_rgba is not None else max(1.5, p["distance"] * diag)
        for gen in range(int(p["generations"])):
            ctx.check()
            if gen > 0:
                nb = np.zeros((gh, gw), bool)
                inf2 = infected.reshape(gh, gw)
                nb[1:] |= inf2[:-1]
                nb[:-1] |= inf2[1:]
                nb[:, 1:] |= inf2[:, :-1]
                nb[:, :-1] |= inf2[:, 1:]
                cand = np.flatnonzero(nb.ravel() & (rng.random(n) < max(p["density"], 0.15)))
                dests = cand
            if len(dests) == 0:
                break
            donors = _choose_donors(F, Fd, dests, gw, gh, min_dist, p["similarity"], p["mode"] == "dissimilar", rng)
            pool = (donor_rgba.astype(np.float32) if donor_rgba is not None else cur.copy())
            cur = _transfer(cur, pool, dests, donors, xs, ys, gw, p["feather"], p["colour_match"])
            infected[dests[donors >= 0]] = True
        out = np.clip(cur, 0, 255).astype(np.uint8)
        out[..., 3] = img[..., 3]
        if p["show_map"]:
            m = np.zeros((h, w), np.float32)
            for k in np.flatnonzero(infected):
                i, j = divmod(k, gw)
                m[ys[i]:ys[i + 1], xs[j]:xs[j + 1]] = 1.0
            out = overlay_map(out, m, (60, 255, 140), 0.45)
        return out


def _cover(rgba: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resize to cover (w, h) keeping the aspect ratio, then centre-crop."""
    ih, iw = rgba.shape[:2]
    s = max(w / iw, h / ih)
    nw, nh = max(w, int(math.ceil(iw * s))), max(h, int(math.ceil(ih * s)))
    r = cv2.resize(rgba, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    x0, y0 = (nw - w) // 2, (nh - h) // 2
    return np.ascontiguousarray(r[y0:y0 + h, x0:x0 + w])


def _grid_features(feats: np.ndarray, gw: int, gh: int) -> np.ndarray:
    """Resample a (fh, fw, C) feature grid to (gh*gw, C) unit vectors."""
    fh, fw, c = feats.shape
    interp = cv2.INTER_AREA if (gw < fw or gh < fh) else cv2.INTER_LINEAR
    chans = [cv2.resize(np.ascontiguousarray(feats[..., k:k + 4]), (gw, gh), interpolation=interp).reshape(gh, gw, -1)
             for k in range(0, c, 4)]
    g = np.concatenate(chans, -1).reshape(gh * gw, c)
    g /= np.linalg.norm(g, axis=1, keepdims=True) + 1e-6
    return g.astype(np.float32)


def _choose_donors(F, Fd, dests, gw, gh, min_dist, window, dissimilar, rng, chunk: int = 256) -> np.ndarray:
    """Seeded donor choice among the top (or bottom) matches, in bounded-memory chunks."""
    n = Fd.shape[0]
    jj = np.arange(n) % gw
    ii = np.arange(n) // gw
    out = np.full(len(dests), -1, np.int64)
    k_win = max(1, int(round(window * n)))
    for s in range(0, len(dests), chunk):
        d = dests[s:s + chunk]
        sims = F[d] @ Fd.T                       # (chunk, n) -- bounded by `chunk`
        di, dj = d // gw, d % gw
        dist = np.hypot(ii[None, :] - di[:, None], jj[None, :] - dj[:, None])
        bad = dist < min_dist
        if min_dist <= 0:  # donor image: everything allowed
            bad[:] = False
        score = -sims if dissimilar else sims
        score = np.where(bad, -np.inf, score)
        k = min(k_win, n)
        top = np.argpartition(-score, k - 1, axis=1)[:, :k]
        pick = top[np.arange(len(d)), rng.integers(0, k, len(d))]
        ok = np.isfinite(score[np.arange(len(d)), pick])
        out[s:s + chunk] = np.where(ok, pick, -1)
    return out


def _window(n: int, pad: int) -> np.ndarray:
    total = n + 2 * pad
    wgt = np.ones(total, np.float32)
    if pad > 0:
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, math.pi, 2 * pad + 2)[1:-1])
        wgt[:2 * pad] = ramp
        wgt[-2 * pad:] = ramp[::-1]
    return wgt


def _transfer(cur, pool, dests, donors, xs, ys, gw, feather, colour_match):
    h, w = cur.shape[:2]
    num = np.zeros((h, w, 4), np.float32)
    den = np.zeros((h, w), np.float32)
    for dst, don in zip(dests, donors):
        if don < 0:
            continue
        di, dj = divmod(int(dst), gw)
        si, sj = divmod(int(don), gw)
        ph, pw = ys[di + 1] - ys[di], xs[dj + 1] - xs[dj]
        pad = int(round(feather * 0.5 * min(ph, pw)))
        y0, x0 = ys[di] - pad, xs[dj] - pad
        sy0, sx0 = ys[si] - pad, xs[sj] - pad
        hh, ww = ph + 2 * pad, pw + 2 * pad
        # clip both rectangles consistently
        cy0 = max(0, -y0, -sy0)
        cx0 = max(0, -x0, -sx0)
        cy1 = min(hh, h - y0, h - sy0)
        cx1 = min(ww, w - x0, w - sx0)
        if cy1 <= cy0 or cx1 <= cx0:
            continue
        wy = _window(ph, pad)[cy0:cy1]
        wx = _window(pw, pad)[cx0:cx1]
        wgt = np.outer(wy, wx)
        patch = pool[sy0 + cy0:sy0 + cy1, sx0 + cx0:sx0 + cx1].copy()
        if colour_match > 0:
            dst_px = cur[y0 + cy0:y0 + cy1, x0 + cx0:x0 + cx1, :3]
            ms, ss = patch[..., :3].mean((0, 1)), patch[..., :3].std((0, 1)) + 1e-3
            md, sd = dst_px.mean((0, 1)), dst_px.std((0, 1)) + 1e-3
            matched = (patch[..., :3] - ms) * (sd / ss) + md
            patch[..., :3] = patch[..., :3] * (1 - colour_match) + matched * colour_match
        num[y0 + cy0:y0 + cy1, x0 + cx0:x0 + cx1] += patch * wgt[..., None]
        den[y0 + cy0:y0 + cy1, x0 + cx0:x0 + cx1] += wgt
    a = np.clip(den, 0, 1)[..., None]
    blended = num / np.maximum(den, 1e-6)[..., None]
    return cur * (1 - a) + blended * a


# =====================================================================
# Perception Breakdown
# =====================================================================
@register
class PerceptionBreakdown(Effect):
    id = "perception_breakdown"
    name = "Perception Breakdown"
    category = CAT
    ai_models = ("dinov2",)
    supports_channels = False
    description = ("AI-assisted (DINOv2 or SAM 2, runs locally). The model looks at mildly altered copies of the "
                   "photo (mirrored, slightly zoomed, brighter, darker). Where its answers disagree, edges and "
                   "textures erupt into tearing, chromatic scars, repeats or sorting, while stable areas stay "
                   "recognisable. The map shows model disagreement, not calibrated confidence or uncertainty.")
    preview_note = "The disagreement analysis is cached; glitch settings change instantly."
    params = (
        cparam("model", "Analysis model", "features", (("features", "Dense features (DINOv2)"),
                                                       ("segmentation", "Segmentation (SAM 2, slower)"))),
        fparam("perturbation", "Perturbation amount", 0.5, 0.05, 1.0),
        fparam("threshold", "Disagreement threshold", 0.45, 0.0, 1.0),
        fparam("contrast", "Map contrast", 0.6, 0.0, 1.0),
        cparam("target", "Glitch", "high", (("high", "High-disagreement regions"),
                                            ("low", "Low-disagreement regions"))),
        cparam("glitch", "Glitch type", "tear", (("tear", "Local tearing"), ("channel", "Channel displacement"),
                                                 ("repeat", "Patch duplication"), ("sort", "Pixel sorting"),
                                                 ("all", "All together"))),
        iparam("displacement", "Displacement", 40, 0, 4000, spatial=True, unit="px"),
        fparam("softness", "Protected-region softness", 0.3, 0.0, 1.0),
        cparam("show_map", "Model disagreement map", "off", (("off", "Hidden"), ("overlay", "Overlay"),
                                                             ("map", "Map only"))),
    )

    def process(self, img, p, ctx):
        h, w = img.shape[:2]
        an = _analysis()
        if p["model"] == "segmentation":
            from ...ai import models as _m  # noqa: PLC0415

            _m.require("sam2")
            res = an.segmentation_disagreement(ctx, img, p["perturbation"])
        else:
            res = an.feature_disagreement(ctx, img, p["perturbation"])
        raw = res["map"].astype(np.float32)
        cov = res["coverage"].astype(np.float32)
        lo, hi = np.percentile(raw[cov > 0], [5, 95]) if (cov > 0).any() else (0.0, 1.0)
        m = np.clip((raw - lo) / max(hi - lo, 1e-6), 0, 1)
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_CUBIC)
        k = 2.0 + 18.0 * p["contrast"]
        m = 1.0 / (1.0 + np.exp(-k * (m - p["threshold"])))
        sig = p["softness"] * 0.04 * max(w, h)
        if sig > 0.3:
            m = cv2.GaussianBlur(m, (0, 0), sig)
        m = np.clip(m, 0, 1).astype(np.float32)
        if p["target"] == "low":
            m = 1.0 - m
        if p["show_map"] == "map":
            out = img.copy()
            out[..., :3] = heat(m if p["target"] == "high" else 1 - m)
            return out
        out = _breakdown(img, m, p, ctx)
        if p["show_map"] == "overlay":
            out = overlay_map(out, m, (255, 60, 170), 0.45)
        return out


def _breakdown(img, m, p, ctx):
    h, w = img.shape[:2]
    amp = float(p["displacement"])
    kind = p["glitch"]
    src = img
    out = img.astype(np.float32)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    full_h = ctx.full_size[1]
    if kind in ("tear", "all"):
        rows = smooth_noise(ctx.sub_rng(11), 1, h, (1, full_h), max(2.0, full_h / 180.0))[:, 0]
        jag = smooth_noise(ctx.sub_rng(12), 1, h, (1, full_h), max(2.0, full_h / 900.0))[:, 0]
        dx = (rows * 0.7 + np.sign(jag) * 0.3)[:, None] * amp * m
        torn = cv2.remap(src, xs + dx, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        out = torn.astype(np.float32)
        src = torn
    if kind in ("channel", "all"):
        res = out.copy()
        a = ctx.sub_rng(13).uniform(-math.pi, math.pi)
        for c, sgn in ((0, 1.0), (2, -1.0)):
            ch = cv2.remap(np.ascontiguousarray(src[..., c]), xs + sgn * amp * 0.6 * m * math.cos(a),
                           ys + sgn * amp * 0.6 * m * math.sin(a), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            res[..., c] = ch
        out = res
        src = np.clip(out, 0, 255).astype(np.uint8)
    if kind in ("repeat", "all"):
        rng = ctx.sub_rng(14)
        fw, fh = ctx.full_size
        cell_full = max(8.0, amp / max(ctx.scale, 1e-6))
        gw, gh = max(1, int(fw // cell_full)), max(1, int(fh // cell_full))
        offs = rng.integers(-3, 4, (gh, gw, 2))
        cw, chh = w / gw, h / gh
        rep = src.astype(np.float32).copy()
        for i in range(gh):
            for j in range(gw):
                y0, y1 = int(i * chh), int((i + 1) * chh)
                x0, x1 = int(j * cw), int((j + 1) * cw)
                if y1 <= y0 or x1 <= x0 or m[y0:y1, x0:x1].mean() < 0.15:
                    continue
                sy = int(np.clip(y0 + offs[i, j, 0] * (y1 - y0), 0, h - (y1 - y0)))
                sx = int(np.clip(x0 + offs[i, j, 1] * (x1 - x0), 0, w - (x1 - x0)))
                rep[y0:y1, x0:x1] = src[sy:sy + (y1 - y0), sx:sx + (x1 - x0)]
        a = m[..., None]
        out = out * (1 - a) + rep * a
        src = np.clip(out, 0, 255).astype(np.uint8)
    if kind in ("sort", "all"):
        L = max(4, int(round(max(amp, 4) * 2)))
        lum = src[..., :3].astype(np.float32) @ np.float32([0.299, 0.587, 0.114])
        pad = (-w) % L
        lp = np.pad(lum, ((0, 0), (0, pad)), mode="edge").reshape(h, -1, L)
        ip = np.pad(src, ((0, 0), (0, pad), (0, 0)), mode="edge").reshape(h, -1, L, src.shape[2])
        order = np.argsort(lp, axis=2, kind="stable")
        sorted_px = np.take_along_axis(ip, order[..., None], axis=2).reshape(h, -1, src.shape[2])[:, :w]
        a = (m > 0.5).astype(np.float32)[..., None] * m[..., None]
        out = out * (1 - a) + sorted_px.astype(np.float32) * a
    if kind in ("tear",):
        pass
    res = np.clip(out, 0, 255).astype(np.uint8)
    res[..., 3] = img[..., 3]
    return res


# Keep MASK import referenced for the generative module's shared param kinds.
_ = MASK
