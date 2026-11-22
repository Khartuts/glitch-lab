"""Render pipeline: original image -> per-layer effect stacks -> masks ->
compositing.  The *same* function renders previews (on a reduced copy of the
original with ``scale < 1``) and full-resolution exports (``scale == 1``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .cancel import CancelToken
from .compositing import composite_layers, composite_over
from .effects.base import apply_effect, get_effect
from .errors import Cancelled, DependencyError, EffectError, GlitchLabError
from .masks import rasterize_mask
from .model import Document, EffectInstance
from .ops import resize_rgba, to_float, to_uint8
from .services import EngineServices, default_services

log = logging.getLogger(__name__)

ProgressFn = Callable[[float, str], None]


@dataclass
class EffectFailure:
    uid: str
    name: str
    message: str
    #: Exception class name (e.g. "ModelMissing", "AIUnavailable", "GenerationPending").
    code: str = ""


@dataclass
class RenderResult:
    image: np.ndarray
    scale: float
    warnings: list[str] = field(default_factory=list)
    failures: list[EffectFailure] = field(default_factory=list)
    elapsed: float = 0.0
    cache_hits: int = 0

    @property
    def size(self) -> tuple[int, int]:
        return int(self.image.shape[1]), int(self.image.shape[0])

    @property
    def ok(self) -> bool:
        return not self.failures


class RenderCache:
    """Thread-safe LRU cache of intermediate stack results, bounded in bytes."""

    def __init__(self, max_bytes: int = 768 * 1024 * 1024):
        self.max_bytes = int(max_bytes)
        self._data: OrderedDict[str, np.ndarray] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, key: str) -> np.ndarray | None:
        with self._lock:
            arr = self._data.get(key)
            if arr is not None:
                self._data.move_to_end(key)
            return arr

    def put(self, key: str, arr: np.ndarray) -> None:
        if arr.nbytes > self.max_bytes // 3:
            return  # never let one huge buffer evict everything
        arr = arr.view()
        arr.flags.writeable = False
        with self._lock:
            old = self._data.pop(key, None)
            if old is not None:
                self._bytes -= old.nbytes
            self._data[key] = arr
            self._bytes += arr.nbytes
            while self._bytes > self.max_bytes and self._data:
                _, ev = self._data.popitem(last=False)
                self._bytes -= ev.nbytes

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._bytes = 0

    @property
    def nbytes(self) -> int:
        return self._bytes


def make_preview_source(rgba: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Downscale the original for preview. Returns (image, scale)."""
    h, w = rgba.shape[:2]
    if max_side <= 0 or max(h, w) <= max_side:
        return rgba, 1.0
    s = max_side / float(max(h, w))
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    img = resize_rgba(rgba, nw, nh, interpolation="area")
    return img, nw / float(w)


def _file_signature(inst: EffectInstance) -> str:
    """Include modification times of referenced files in cache keys."""
    import os

    eff = get_effect(inst.effect_id)
    parts = []
    for prm in eff.params:
        if prm.kind == "file":
            path = inst.params.get(prm.name) or ""
            try:
                st = os.stat(path)
                parts.append(f"{path}:{st.st_mtime_ns}:{st.st_size}")
            except OSError:
                parts.append(f"{path}:missing")
    return "|".join(parts)


def source_fingerprint(img: np.ndarray) -> str:
    """Cheap, scale-independent identity of a full-resolution source image."""
    h, w = img.shape[:2]
    step = max(1, int(max(h, w) // 384))
    sample = np.ascontiguousarray(img[::step, ::step])
    return hashlib.sha1(f"{img.shape}|{step}".encode() + sample.tobytes()).hexdigest()


def render_stack(
    source: np.ndarray,
    stack: list[EffectInstance],
    *,
    global_seed: int,
    scale: float,
    full_size: tuple[int, int],
    services: EngineServices,
    cancel: CancelToken,
    cache: RenderCache | None = None,
    cache_prefix: str = "",
    on_error: str = "raise",
    failures: list[EffectFailure] | None = None,
    warnings: list[str] | None = None,
    progress: ProgressFn | None = None,
    progress_range: tuple[float, float] = (0.0, 1.0),
    full_source: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Render one effect stack from ``source``. Returns (image, cache_hits).

    ``full_source`` is the full-resolution original (when ``source`` is a
    reduced preview copy); AI effects analyse their full-resolution input so
    that preview and export share one analysis.
    """
    failures = failures if failures is not None else []
    warnings = warnings if warnings is not None else []
    active = [e for e in stack if e.enabled and e.mix > 0]
    if not active:
        return source, 0
    # Chain of cache keys: key_i identifies source + effects[0..i].
    keys = []
    h = hashlib.sha1(f"{cache_prefix}|{source.shape}|{global_seed}|{scale:.6f}".encode())
    for e in active:
        h.update(e.cache_signature().encode())
        h.update(_file_signature(e).encode())
        keys.append(h.hexdigest())
    # Scale-independent identities of every effect's input (for AI caches).
    if full_source is None and scale == 1.0:
        full_source = source
    base_id = source_fingerprint(full_source) if full_source is not None else \
        f"scaled:{scale:.6f}:{source_fingerprint(source)}"
    ih = hashlib.sha1(f"{base_id}|{full_size}|{global_seed}".encode())
    input_keys = []
    for e in active:
        input_keys.append(ih.hexdigest())
        ih.update(e.cache_signature().encode())
        ih.update(_file_signature(e).encode())

    def full_input_of(i: int, current: np.ndarray):
        if scale == 1.0 or full_source is None:
            return lambda: current

        def render_prefix():
            out, _ = render_stack(full_source, active[:i], global_seed=global_seed, scale=1.0,
                                  full_size=full_size, services=services, cancel=cancel, cache=cache,
                                  cache_prefix=cache_prefix + ":ai-full", on_error=on_error, failures=[],
                                  warnings=[], full_source=full_source)
            return out
        return render_prefix

    start_idx = 0
    img = source
    hits = 0
    if cache is not None:
        for i in range(len(active) - 1, -1, -1):
            hit = cache.get(keys[i])
            if hit is not None:
                img = hit
                start_idx = i + 1
                hits = 1
                break
    lo, hi = progress_range
    n = len(active)
    for i in range(start_idx, n):
        cancel.check()
        inst = active[i]
        name = inst.display_name()
        if progress:
            progress(lo + (hi - lo) * i / n, f"{name} ({i + 1}/{n})")

        def sub_progress(f, _i=i, _name=name):
            if progress:
                progress(lo + (hi - lo) * (_i + f) / n, f"{_name} ({_i + 1}/{n})")

        t0 = time.perf_counter()
        try:
            out = apply_effect(img, inst, global_seed=global_seed, scale=scale, full_size=full_size,
                               cancel=cancel, services=services, warnings=warnings, progress_cb=sub_progress,
                               input_key=input_keys[i], full_input=full_input_of(i, img),
                               interactive=(on_error == "skip"))
        except Cancelled:
            raise
        except (EffectError, DependencyError) as exc:
            log.warning("Effect %s failed: %s %s", name, exc.message, exc.detail)
            if on_error == "raise":
                raise
            failures.append(EffectFailure(inst.uid, name, exc.message, type(exc).__name__))
            continue
        except MemoryError as exc:
            msg = "Not enough memory for this effect at the current resolution."
            log.error("%s: %s", name, msg)
            if on_error == "raise":
                raise EffectError(f"{name}: {msg}") from exc
            failures.append(EffectFailure(inst.uid, name, msg))
            continue
        except Exception as exc:  # unexpected bug: log full traceback, show short text
            log.exception("Unexpected error in effect %s", name)
            if on_error == "raise":
                raise EffectError(f"{name} failed unexpectedly: {exc}", detail=repr(exc)) from exc
            failures.append(EffectFailure(inst.uid, name, f"Unexpected error: {exc}"))
            continue
        log.debug("%s took %.3f s", name, time.perf_counter() - t0)
        img = out
        if cache is not None:
            cache.put(keys[i], img)
    if progress:
        progress(hi, "")
    return img, hits


def render_document(
    source: np.ndarray,
    doc: Document,
    *,
    scale: float = 1.0,
    full_size: tuple[int, int] | None = None,
    services: EngineServices | None = None,
    cancel: CancelToken | None = None,
    cache: RenderCache | None = None,
    cache_prefix: str = "",
    on_error: str = "raise",
    progress: ProgressFn | None = None,
    full_source: np.ndarray | None = None,
    text_only: bool = False,
) -> RenderResult:
    """Render a whole document.

    ``source`` is the original at render resolution (uint8 RGBA). ``scale``
    is render width / full-resolution width and is used to scale spatial
    parameters.  ``on_error='skip'`` (used for previews) skips failing
    effects and reports them; ``'raise'`` (exports) aborts.
    """
    t0 = time.perf_counter()
    services = services or default_services()
    cancel = cancel or CancelToken()
    if source.dtype != np.uint8 or source.ndim != 3 or source.shape[2] != 4:
        raise GlitchLabError("Internal error: source must be uint8 RGBA.")
    if full_size is None:
        full_size = (int(round(source.shape[1] / scale)), int(round(source.shape[0] / scale)))
    failures: list[EffectFailure] = []
    warnings: list[str] = []
    visible = [layer for layer in doc.layers if layer.visible and layer.opacity > 0
               and (not text_only or layer.kind == "text")]
    hits = 0
    h, w = source.shape[:2]
    nl = max(1, len(visible))
    acc = None
    single = None
    for li, layer in enumerate(visible):
        base, prefix, fs = source, cache_prefix, full_source
        if layer.kind == "text":
            base, tsig = _text_raster(layer, source, doc, scale, full_size, full_source, services, cancel, cache,
                                      cache_prefix, on_error, warnings)
            prefix, fs = f"{cache_prefix}:text:{tsig}", None
        elif layer.source == "below" and acc is not None:
            base = to_uint8(acc)
            below = hashlib.sha1(json.dumps([lay.to_dict() for lay in visible[:li]], sort_keys=True,
                                            default=str).encode()).hexdigest()
            prefix, fs = f"{cache_prefix}:below:{below}", None
        img, hit = render_stack(
            base, layer.stack, global_seed=doc.global_seed, scale=scale, full_size=full_size,
            services=services, cancel=cancel, cache=cache, cache_prefix=prefix, on_error=on_error,
            failures=failures, warnings=warnings, progress=progress,
            progress_range=(li / nl * 0.95, (li + 1) / nl * 0.95), full_source=fs)
        hits += hit
        mask = rasterize_mask(layer.mask, w, h, source)
        if li == 0 and len(visible) == 1 and layer.opacity >= 1.0 and mask is None and layer.kind == "image":
            single = img
            break
        acc = composite_over(acc, to_float(img), mode=layer.blend, opacity=layer.opacity, mask=mask)
    cancel.check()
    if single is not None:
        out = single.copy()
    elif acc is None:
        out = np.zeros_like(source)
    else:
        out = to_uint8(acc)
    if doc.effective_alpha_mode() == "preserve" and not text_only:
        out[..., 3] = source[..., 3]
    if progress:
        progress(1.0, "Done")
    return RenderResult(image=out, scale=scale, warnings=warnings, failures=failures,
                        elapsed=time.perf_counter() - t0, cache_hits=hits)


def _text_raster(layer, source, doc, scale, full_size, full_source, services, cancel, cache, cache_prefix, on_error,
                 warnings):
    """Typography of a text layer at render resolution (cached)."""
    from .typography.render import render_text_layer, text_layer_signature  # noqa: PLC0415

    h, w = source.shape[:2]
    sig = text_layer_signature(layer.boxes, w, h)
    key = f"{cache_prefix}|textlayer|{sig}"
    hit = cache.get(key) if cache is not None else None
    reports: list = []
    if hit is not None:
        return hit, sig

    def subject_fn(points):
        from ..ai import analysis  # noqa: PLC0415 - optional, cached SAM 2 inference
        from .effects.base import EffectContext  # noqa: PLC0415

        fsrc = full_source if full_source is not None else (source if scale == 1.0 else None)
        base_id = source_fingerprint(fsrc) if fsrc is not None else f"scaled:{scale:.6f}:{source_fingerprint(source)}"
        key0 = hashlib.sha1(f"{base_id}|{full_size}|{doc.global_seed}".encode()).hexdigest()
        ss = np.random.SeedSequence(0)
        ctx = EffectContext(rng=np.random.default_rng(ss), seed_sequence=ss, scale=scale, full_size=full_size,
                            cancel=cancel, services=services, input_key=key0,
                            full_input=(lambda: fsrc if fsrc is not None else source), owner=f"text:{layer.uid}",
                            interactive=on_error == "skip", effect_label="Text behind subject")
        logits = analysis.object_logits(ctx, source, points)
        import cv2  # noqa: PLC0415

        m = np.zeros((h, w), np.float32)
        for lg in logits:
            m = np.maximum(m, 1.0 / (1.0 + np.exp(-cv2.resize(lg, (w, h), interpolation=cv2.INTER_LINEAR) * 2.0)))
        return m

    img = render_text_layer(layer.boxes, w, h, full_size=full_size, source=source, subject_fn=subject_fn,
                            reports=reports, cancel=cancel)
    for i, rep in enumerate(reports):
        if rep.overflow and not any("larger than its box" in x or "does not fit" in x for x in rep.warnings):
            rep.warnings.append("The text overflows its box.")
        for msg in rep.warnings:
            warnings.append(f"{layer.name}, text box {i + 1}: {msg}")
    if cache is not None:
        cache.put(key, img)
    return img, sig


def render_full(source: np.ndarray, doc: Document, **kw) -> RenderResult:
    """Full-resolution render (export pipeline)."""
    kw.setdefault("on_error", "raise")
    return render_document(source, doc, scale=1.0, full_size=(source.shape[1], source.shape[0]), **kw)


def image_hash(img: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest()
