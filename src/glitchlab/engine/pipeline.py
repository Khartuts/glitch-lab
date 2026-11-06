"""Render pipeline: original image -> per-layer effect stacks -> masks ->
compositing.  The *same* function renders previews (on a reduced copy of the
original with ``scale < 1``) and full-resolution exports (``scale == 1``).
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .cancel import CancelToken
from .compositing import composite_layers
from .effects.base import apply_effect, get_effect
from .errors import Cancelled, DependencyError, EffectError, GlitchLabError
from .masks import rasterize_mask
from .model import Document, EffectInstance
from .ops import resize_rgba
from .services import EngineServices, default_services

log = logging.getLogger(__name__)

ProgressFn = Callable[[float, str], None]


@dataclass
class EffectFailure:
    uid: str
    name: str
    message: str


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
) -> tuple[np.ndarray, int]:
    """Render one effect stack from ``source``. Returns (image, cache_hits)."""
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
                               cancel=cancel, services=services, warnings=warnings, progress_cb=sub_progress)
        except Cancelled:
            raise
        except (EffectError, DependencyError) as exc:
            log.warning("Effect %s failed: %s %s", name, exc.message, exc.detail)
            if on_error == "raise":
                raise
            failures.append(EffectFailure(inst.uid, name, exc.message))
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
    images, modes, opacities, masks = [], [], [], []
    visible = [layer for layer in doc.layers if layer.visible and layer.opacity > 0]
    hits = 0
    h, w = source.shape[:2]
    nl = max(1, len(visible))
    for li, layer in enumerate(visible):
        img, hit = render_stack(
            source, layer.stack, global_seed=doc.global_seed, scale=scale, full_size=full_size,
            services=services, cancel=cancel, cache=cache, cache_prefix=cache_prefix, on_error=on_error,
            failures=failures, warnings=warnings, progress=progress,
            progress_range=(li / nl * 0.95, (li + 1) / nl * 0.95))
        hits += hit
        images.append(img)
        modes.append(layer.blend)
        opacities.append(layer.opacity)
        masks.append(rasterize_mask(layer.mask, w, h, source))
    cancel.check()
    if not images:
        out = np.zeros_like(source)
    else:
        out = composite_layers(images, modes, opacities, masks)
    if doc.effective_alpha_mode() == "preserve":
        out[..., 3] = source[..., 3]
    if progress:
        progress(1.0, "Done")
    return RenderResult(image=out, scale=scale, warnings=warnings, failures=failures,
                        elapsed=time.perf_counter() - t0, cache_hits=hits)


def render_full(source: np.ndarray, doc: Document, **kw) -> RenderResult:
    """Full-resolution render (export pipeline)."""
    kw.setdefault("on_error", "raise")
    return render_document(source, doc, scale=1.0, full_size=(source.shape[1], source.shape[0]), **kw)


def image_hash(img: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest()
