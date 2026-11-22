"""Cached model analyses used by the AI effects: SAM 2 object masks, Depth
Anything relative depth, DINOv2 patch features and multi-view "model
disagreement".  Every function returns plain NumPy arrays at an analysis
resolution; the effects resample them to the render size.

Analyses never call each other from inside a job (there is one AI worker
thread); results needed by a later step are computed first and captured.
"""

from __future__ import annotations

import hashlib
import math

import cv2
import numpy as np

from ..engine.effects.base import EffectContext
from . import models
from .jobs import make_key, manager
from .support import (cached_analysis, device, ensure_ready, fit_size, inference_mode, load_depth, load_dinov2,
                      load_sam2, normalised_tensor, rgb_resized)

SAM_SIZE = 1024
DINO_PATCH = 14


# ------------------------------------------------------------------ SAM 2
def _sam_size() -> int:
    from .support import preprocessor_config  # noqa: PLC0415

    size = preprocessor_config("sam2").get("size", {})
    return int(size.get("height", SAM_SIZE)) if isinstance(size, dict) else SAM_SIZE


def sam_embedding(ctx: EffectContext, img: np.ndarray, *, view: str = "id") -> dict:
    """SAM 2 image embedding of the effect's input (computed once, cached)."""

    def run(full, job):
        n = _sam_size()
        rgb = transform_view(rgb_resized(full, n, n), view)
        model = load_sam2()
        job.progress(0.2, "Segmentation: encoding image")
        with inference_mode():
            emb = model.get_image_embeddings(normalised_tensor(rgb, "sam2").to(device()))
        return {f"e{i}": e.float().cpu().numpy().astype(np.float16) for i, e in enumerate(emb)}

    return cached_analysis(ctx, img, "sam2-embed", {"view": view}, run, model_ids=("sam2",),
                           label="Segmentation (SAM 2)")


def _decode(emb: dict, prompts: list[tuple[list, list]], multimask_single: bool = True) -> list[np.ndarray]:
    """Decode one mask per prompt (points in 0..1, labels 1/0). Returns 256x256 logits."""
    import torch  # noqa: PLC0415

    model = load_sam2()
    n = _sam_size()
    dev = device()
    embs = [torch.from_numpy(emb[f"e{i}"].astype(np.float32)).to(dev) for i in range(len(emb))]
    out = []
    with inference_mode():
        for pts, labels in prompts:
            p = torch.tensor([[[[x * n, y * n] for x, y in pts]]], dtype=torch.float32, device=dev)
            lab = torch.tensor([[labels]], dtype=torch.long, device=dev)
            multi = multimask_single and len(pts) == 1
            res = model(input_points=p, input_labels=lab, image_embeddings=embs, multimask_output=multi)
            masks = res.pred_masks[0, 0].float().cpu().numpy()
            k = int(res.iou_scores[0, 0].argmax()) if multi else 0
            out.append(masks[k])
    return out


def _decode_batch(emb: dict, points: np.ndarray) -> np.ndarray:
    """One best mask per single positive point, batched. points (K, 2) in 0..1 -> (K, 256, 256) logits."""
    import torch  # noqa: PLC0415

    model = load_sam2()
    n = _sam_size()
    dev = device()
    embs = [torch.from_numpy(emb[f"e{i}"].astype(np.float32)).to(dev) for i in range(len(emb))]
    res = []
    with inference_mode():
        for start in range(0, len(points), 32):
            chunk = points[start:start + 32]
            p = torch.tensor(chunk[None, :, None, :] * n, dtype=torch.float32, device=dev)
            lab = torch.ones((1, len(chunk), 1), dtype=torch.long, device=dev)
            r = model(input_points=p, input_labels=lab, image_embeddings=embs, multimask_output=True)
            masks = r.pred_masks[0].float().cpu().numpy()       # (k, 3, 256, 256)
            best = r.iou_scores[0].float().cpu().numpy().argmax(-1)
            res.append(masks[np.arange(len(chunk)), best])
    return np.concatenate(res, 0)


def group_points(points: list) -> list[tuple[int, list, list]]:
    objs: dict[int, tuple[list, list]] = {}
    for x, y, label, obj in points:
        pts, labs = objs.setdefault(int(obj), ([], []))
        pts.append((float(x), float(y)))
        labs.append(int(label))
    return [(k, v[0], v[1]) for k, v in sorted(objs.items()) if any(v[1])]


def object_logits(ctx: EffectContext, img: np.ndarray, points: list) -> np.ndarray:
    """(n_objects, 256, 256) SAM 2 mask logits for the user's object points.

    The logits live in SAM's square input frame, which maps onto the whole
    image; resize them straight to the render size."""
    emb = sam_embedding(ctx, img)
    groups = group_points(points)

    def run(_full, job):
        job.progress(0.6, "Segmentation: decoding objects")
        logits = _decode(emb, [(p, lab) for _, p, lab in groups])
        return {"logits": np.stack(logits).astype(np.float16)}

    res = cached_analysis(ctx, img, "sam2-objects", {"points": points}, run, model_ids=("sam2",),
                          label="Segmentation (SAM 2)")
    return res["logits"].astype(np.float32)


def masks_from_logits(logits: np.ndarray, w: int, h: int, threshold: float = 0.0) -> np.ndarray:
    out = np.zeros((len(logits), h, w), bool)
    for i, lg in enumerate(logits):
        out[i] = cv2.resize(lg, (w, h), interpolation=cv2.INTER_LINEAR) > threshold
    return out


# ------------------------------------------------------------------ depth
def depth(ctx: EffectContext, img: np.ndarray, max_side: int = 518) -> np.ndarray:
    """Relative inverse depth (larger = nearer) at analysis resolution, float32."""

    def run(full, job):
        h, w = full.shape[:2]
        dw, dh = fit_size(w, h, max_side, DINO_PATCH)
        model = load_depth()
        job.progress(0.3, "Estimating depth")
        with inference_mode():
            pred = model(pixel_values=normalised_tensor(rgb_resized(full, dw, dh), "depth").to(device()))
        d = pred.predicted_depth[0].float().cpu().numpy()
        return {"depth": d.astype(np.float32)}

    return cached_analysis(ctx, img, "depth", {"side": max_side}, run, model_ids=("depth",),
                           label="Depth (Depth Anything V2)")["depth"]


def robust_normalise(d: np.ndarray, lo_pct: float = 2.0, hi_pct: float = 98.0) -> np.ndarray:
    lo, hi = np.percentile(d, [lo_pct, hi_pct])
    return np.clip((d - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0).astype(np.float32)


# ------------------------------------------------------------------ DINOv2
def _dino_tokens(model, rgb: np.ndarray) -> np.ndarray:
    """Patch features (gh, gw, C) without the class and register tokens, L2-normalised."""
    h, w = rgb.shape[:2]
    gh, gw = h // DINO_PATCH, w // DINO_PATCH
    with inference_mode():
        out = model(pixel_values=normalised_tensor(rgb, "dinov2").to(device()))
    tok = out.last_hidden_state[0].float().cpu().numpy()
    n_special = tok.shape[0] - gh * gw
    expected = 1 + int(getattr(model.config, "num_register_tokens", 0) or 0)
    if n_special != expected:
        raise RuntimeError(f"Unexpected DINOv2 token layout ({tok.shape[0]} tokens for a {gh}x{gw} grid).")
    f = tok[n_special:].reshape(gh, gw, -1)
    f /= np.linalg.norm(f, axis=-1, keepdims=True) + 1e-6
    return f


def grid_for(w: int, h: int, grid_long: int) -> tuple[int, int]:
    """Input size (multiple of the 14-px patch) so the long side has ``grid_long`` patches."""
    s = grid_long / float(max(w, h))
    gw = max(2, int(round(w * s)))
    gh = max(2, int(round(h * s)))
    return gw * DINO_PATCH, gh * DINO_PATCH


def features(ctx: EffectContext, img: np.ndarray, grid_long: int = 40) -> np.ndarray:
    """DINOv2 patch features of the effect's input: (gh, gw, C) float32, unit length.

    The whole image is resized to (gw*14, gh*14), so patch (i, j) covers
    exactly the image region [i/gh, (i+1)/gh) x [j/gw, (j+1)/gw)."""

    def run(full, job):
        h, w = full.shape[:2]
        fw, fh = grid_for(w, h, grid_long)
        job.progress(0.3, "Extracting features")
        return {"f": _dino_tokens(load_dinov2(), rgb_resized(full, fw, fh)).astype(np.float16)}

    return cached_analysis(ctx, img, "dinov2", {"grid": grid_long}, run, model_ids=("dinov2",),
                           label="Features (DINOv2)")["f"].astype(np.float32)


def features_of(rgba: np.ndarray, identity: str, grid_long: int, ctx: EffectContext) -> np.ndarray:
    """Features of an arbitrary image (e.g. the donor image), cached by ``identity``."""
    ensure_ready("dinov2")
    key = make_key("dinov2-image", identity, grid_long, models.revision("dinov2"))

    def run(job):
        h, w = rgba.shape[:2]
        fw, fh = grid_for(w, h, grid_long)
        return {"f": _dino_tokens(load_dinov2(), rgb_resized(rgba, fw, fh)).astype(np.float16)}

    res = manager().run(key, "Features of donor image (DINOv2)", run, owner=ctx.owner + ":donor",
                        cancel=ctx.cancel, progress=ctx.progress)
    return res["f"].astype(np.float32)


def array_identity(a: np.ndarray) -> str:
    return hashlib.sha1(f"{a.shape}".encode() + np.ascontiguousarray(a).tobytes()).hexdigest()


# ------------------------------------------------------------ perturbations
def perturbations(amount: float) -> list[tuple[str, float]]:
    a = float(min(max(amount, 0.05), 1.0))
    return [("flip", 1.0), ("scale", 1.0 - 0.12 * a), ("scale", 1.0 + 0.12 * a),
            ("exposure", 2.0 ** (-0.8 * a)), ("exposure", 2.0 ** (0.8 * a))]


def transform_view(rgb: np.ndarray, view: str) -> np.ndarray:
    """Apply a named perturbation ('id', 'flip', 'scale:s', 'exposure:e') to an RGB image."""
    if view == "id":
        return rgb
    kind, _, val = view.partition(":")
    if kind == "flip":
        return np.ascontiguousarray(rgb[:, ::-1])
    v = float(val)
    if kind == "scale":
        h, w = rgb.shape[:2]
        m = np.float32([[v, 0, (1 - v) * (w - 1) / 2], [0, v, (1 - v) * (h - 1) / 2]])
        return cv2.warpAffine(rgb, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    if kind == "exposure":
        lin = (rgb.astype(np.float32) / 255.0) ** 2.2 * v
        return np.clip(255.0 * np.clip(lin, 0, 1) ** (1 / 2.2) + 0.5, 0, 255).astype(np.uint8)
    raise ValueError(view)


def view_names(amount: float) -> list[str]:
    out = []
    for kind, v in perturbations(amount):
        out.append("flip" if kind == "flip" else f"{kind}:{v:.4f}")
    return out


def align_grid(grid: np.ndarray, view: str, valid_margin: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Map a per-patch grid computed on a perturbed view back to the reference
    frame. Returns (aligned grid, validity mask). Patches whose aligned sample
    falls on the padded/reflected border or outside the view are invalid."""
    gh, gw = grid.shape[:2]
    valid = np.ones((gh, gw), bool)
    if view in ("id",) or view.startswith("exposure"):
        return grid, valid
    if view == "flip":
        return np.ascontiguousarray(grid[:, ::-1]), valid
    s = float(view.split(":")[1])
    # Patch centres in pixel units of the view: u = (p - c) * s + c.
    jj, ii = np.meshgrid(np.arange(gw, dtype=np.float32) + 0.5, np.arange(gh, dtype=np.float32) + 0.5)
    cu, cv = gw / 2.0, gh / 2.0
    u = (jj - cu) * s + cu - 0.5
    v = (ii - cv) * s + cv - 0.5
    m = valid_margin
    valid &= (u >= m - 0.5) & (u <= gw - 0.5 - m) & (v >= m - 0.5) & (v <= gh - 0.5 - m)
    if s < 1.0:  # zoomed out: the view's border band is reflected padding
        pad_u, pad_v = (1 - s) * gw / 2.0, (1 - s) * gh / 2.0
        valid &= (u >= pad_u + m - 0.5) & (u <= gw - pad_u - m - 0.5) & (v >= pad_v + m - 0.5) & \
                 (v <= gh - pad_v - m - 0.5)
    flat = grid.reshape(gh, gw, -1).astype(np.float32)
    chans = [cv2.remap(np.ascontiguousarray(flat[..., c:c + 4]), u, v, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE).reshape(gh, gw, -1) for c in range(0, flat.shape[-1], 4)]
    out = np.concatenate(chans, -1)
    return out.reshape(grid.shape), valid


def feature_disagreement(ctx: EffectContext, img: np.ndarray, amount: float, grid_long: int = 32) -> dict:
    """Mean feature disagreement (1 - cosine) between the image and mildly
    perturbed views of it, after undoing each spatial change. Returns
    {'map': (gh, gw) float32, 'coverage': (gh, gw) number of valid views}."""
    views = view_names(amount)

    def run(full, job):
        h, w = full.shape[:2]
        fw, fh = grid_for(w, h, grid_long)
        rgb = rgb_resized(full, fw, fh)
        model = load_dinov2()
        ref = _dino_tokens(model, rgb)
        acc = np.zeros(ref.shape[:2], np.float32)
        cnt = np.zeros(ref.shape[:2], np.float32)
        for k, view in enumerate(views):
            job.check()
            job.progress((k + 1) / (len(views) + 1), f"Model disagreement: view {k + 1}/{len(views)}")
            f = _dino_tokens(model, transform_view(rgb, view))
            fa, valid = align_grid(f, view)
            fa /= np.linalg.norm(fa, axis=-1, keepdims=True) + 1e-6
            d = 1.0 - np.sum(fa * ref, axis=-1)
            acc += np.where(valid, d, 0.0)
            cnt += valid
        return {"map": acc / np.maximum(cnt, 1.0), "coverage": cnt}

    return cached_analysis(ctx, img, "disagree-features", {"views": views, "grid": grid_long}, run,
                           model_ids=("dinov2",), label="Model disagreement (DINOv2)")


def segmentation_disagreement(ctx: EffectContext, img: np.ndarray, amount: float, grid: int = 8) -> dict:
    """Disagreement between SAM 2 masks of the image and of perturbed views.

    A grid of prompt points is placed on the image and, transformed, on each
    view, so mask k of a view corresponds to mask k of the image.  Each view
    mask is warped back to the reference frame and *matched* to the reference
    masks (its own prompt first, otherwise the best-overlapping mask), then
    the pixels where matched masks disagree are counted.  Returns
    {'map': (256, 256) float32 in SAM's square frame, 'coverage': same}."""
    views = view_names(amount)
    embs = {v: sam_embedding(ctx, img, view=v) for v in ["id", *views]}
    gx, gy = np.meshgrid((np.arange(grid) + 0.5) / grid, (np.arange(grid) + 0.5) / grid)
    ref_pts = np.stack([gx.ravel(), gy.ravel()], -1).astype(np.float32)

    def warp_points(pts: np.ndarray, view: str) -> np.ndarray:
        if view == "flip":
            return np.stack([1 - pts[:, 0], pts[:, 1]], -1)
        if view.startswith("scale"):
            s = float(view.split(":")[1])
            return (pts - 0.5) * s + 0.5
        return pts

    def unwarp_mask(m: np.ndarray, view: str) -> tuple[np.ndarray, np.ndarray]:
        n = m.shape[0]
        valid = np.ones_like(m, dtype=bool)
        if view == "flip":
            return m[:, ::-1], valid
        if view.startswith("scale"):
            s = float(view.split(":")[1])
            inv = cv2.invertAffineTransform(np.float32([[s, 0, (1 - s) * (n - 1) / 2], [0, s, (1 - s) * (n - 1) / 2]]))
            back = cv2.warpAffine(m.astype(np.float32), inv, (n, n), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=-1.0)
            valid = back > -0.5
            if s < 1.0:  # exclude the region that came from reflected padding
                pad = int(math.ceil((1 - s) * n / 2 / s)) + 2
                valid[:pad] = valid[-pad:] = False
                valid[:, :pad] = valid[:, -pad:] = False
            return back > 0.5, valid
        return m, valid

    def run(_full, job):
        ref = _decode_batch(embs["id"], ref_pts) > 0
        acc = np.zeros(ref.shape[1:], np.float32)
        cnt = np.zeros(ref.shape[1:], np.float32)
        for k, view in enumerate(views):
            job.check()
            job.progress((k + 1) / (len(views) + 1), f"Model disagreement: view {k + 1}/{len(views)}")
            vm = _decode_batch(embs[view], np.clip(warp_points(ref_pts, view), 0.01, 0.99)) > 0
            back = [unwarp_mask(m, view) for m in vm]
            masks = np.stack([b[0] for b in back])
            valid = np.stack([b[1] for b in back]).all(0)
            for i in range(len(ref)):
                a = ref[i]
                cand = masks[i]
                inter = np.logical_and(a, cand).sum()
                union = np.logical_or(a, cand).sum()
                if union and inter / union < 0.5:  # prompt drifted: match the best-overlapping view mask
                    ious = [np.logical_and(a, mm).sum() / max(1, np.logical_or(a, mm).sum()) for mm in masks]
                    cand = masks[int(np.argmax(ious))]
                acc += np.logical_xor(a, cand) & valid
            cnt += valid * len(ref)
        return {"map": acc / np.maximum(cnt, 1.0), "coverage": cnt}

    return cached_analysis(ctx, img, "disagree-sam", {"views": views, "grid": grid}, run, model_ids=("sam2",),
                           label="Model disagreement (SAM 2)")
