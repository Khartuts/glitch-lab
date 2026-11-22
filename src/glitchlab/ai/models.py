"""Registry of the local AI models, their download and in-memory loading.

Every model is pinned to an exact Hugging Face revision (commit) and every
file to its size and checksum (``model_files.json``), so all computers use
byte-identical weights and the revision can be saved in projects.  Nothing is
downloaded without the user asking.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import shutil
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..engine.cancel import CancelToken
from .download import ByteProgress, download, verify_file
from .errors import ModelMissing
from .paths import models_dir

log = logging.getLogger(__name__)

_FILES = json.loads(Path(__file__).with_name("model_files.json").read_text())

_INFO = {
    "sam2": ("SAM 2.1 Hiera Tiny", "Object segmentation (Semantic Faultlines, Perception Breakdown, scar guides)",
             "Apache-2.0", 700),
    "depth": ("Depth Anything V2 Small", "Relative depth (Depth Mutiny, scar guides)", "Apache-2.0", 450),
    "dinov2": ("DINOv2 Small", "Learned image features (Feature Contagion, Perception Breakdown, residue maps)",
               "Apache-2.0", 400),
    "vae": ("Stable Diffusion VAE (ft-MSE)", "Image latents (Latent Circuit Bending)", "MIT", 900),
    "sd15": ("Stable Diffusion 1.5", "Image-to-image (Counterfactual Residue, Recursive Misremembering)",
             "CreativeML OpenRAIL-M", 4500),
    "sd15_inpaint": ("Stable Diffusion 1.5 Inpainting", "Inpainting (Hallucination Scars)",
                     "CreativeML OpenRAIL-M", 4500),
}


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    purpose: str
    repo: str
    revision: str
    files: tuple[dict, ...]
    licence: str
    ram_mb: int

    @property
    def size(self) -> int:
        return sum(int(f["size"]) for f in self.files)

    @property
    def size_text(self) -> str:
        mb = self.size / 2**20
        return f"{mb / 1024:.1f} GB" if mb >= 1024 else f"{mb:.0f} MB"

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}"

    @property
    def licence_url(self) -> str:
        return f"https://huggingface.co/{self.repo}/blob/{self.revision}/README.md"

    def file_url(self, path: str) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/{self.revision}/{path}"


MODELS: dict[str, ModelSpec] = {
    mid: ModelSpec(mid, _INFO[mid][0], _INFO[mid][1], d["repo"], d["revision"], tuple(d["files"]), _INFO[mid][2],
                   _INFO[mid][3])
    for mid, d in _FILES.items()
}


def model_dir(mid: str) -> Path:
    spec = MODELS[mid]
    return models_dir() / mid / spec.revision[:12]


def is_installed(mid: str) -> bool:
    override = os.environ.get(f"GLITCHLAB_MODEL_{mid.upper()}")
    if override:
        return Path(override).exists()
    return (model_dir(mid) / "COMPLETE").exists()


def path(mid: str) -> Path:
    """Folder with the model's files (``from_pretrained`` can load it)."""
    override = os.environ.get(f"GLITCHLAB_MODEL_{mid.upper()}")
    if override:
        return Path(override)
    return model_dir(mid)


def missing(mids) -> list[str]:
    return [m for m in mids if not is_installed(m)]


def require(*mids: str) -> None:
    miss = missing(mids)
    if miss:
        raise ModelMissing(miss)


def revision(mid: str) -> str:
    return MODELS[mid].revision


def _find_local_copy(sha256: str | None, exclude: str) -> Path | None:
    if not sha256:
        return None
    for other in MODELS.values():
        if other.id == exclude or not is_installed(other.id):
            continue
        for f in other.files:
            if f.get("sha256") == sha256:
                p = model_dir(other.id) / f["path"]
                if p.exists():
                    return p
    return None


def install(mids, progress: ByteProgress | None = None, cancel: CancelToken | None = None) -> None:
    """Download (and verify) the given models. Identical files are shared."""
    cancel = cancel or CancelToken()
    todo = [m for m in mids if not is_installed(m)]
    total = sum(MODELS[m].size for m in todo)
    done = 0
    for mid in todo:
        spec = MODELS[mid]
        root = model_dir(mid)
        for f in spec.files:
            cancel.check()
            dest = root / f["path"]
            label = f"{spec.name}: {f['path']}"
            if not verify_file(dest, size=f["size"], sha256=None):
                local = _find_local_copy(f.get("sha256"), mid)
                if local is not None:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        os.link(local, dest)
                    except OSError:
                        shutil.copyfile(local, dest)
                else:
                    download(spec.file_url(f["path"]), dest, size=f["size"], sha256=f.get("sha256"),
                             git_sha1=f.get("git_sha1"), cancel=cancel, progress=progress, label=label,
                             base_done=done, grand_total=total)
            done += f["size"]
            if progress:
                progress(done, total, label)
        (root / "COMPLETE").write_text(json.dumps({"repo": spec.repo, "revision": spec.revision}))
        log.info("Installed model %s (%s @ %s)", mid, spec.repo, spec.revision)


def delete(mid: str) -> None:
    unload(mid)
    shutil.rmtree(models_dir() / mid, ignore_errors=True)


# ---------------------------------------------------------------- loaded models
class _Loaded:
    """Thread-safe LRU of loaded models, bounded by their estimated RAM."""

    def __init__(self, budget_mb: int = 7000):
        self.budget_mb = budget_mb
        self._items: OrderedDict[str, tuple[Any, int]] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: str, loader: Callable[[], Any], ram_mb: int) -> Any:
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key][0]
            while self._items and sum(v[1] for v in self._items.values()) + ram_mb > self.budget_mb:
                old, _ = self._items.popitem(last=False)
                log.info("Unloading %s to make room", old)
            _collect()
            obj = loader()
            self._items[key] = (obj, ram_mb)
            return obj

    def unload(self, prefix: str | None = None) -> int:
        with self._lock:
            keys = [k for k in self._items if prefix is None or k.split(":")[0] == prefix]
            for k in keys:
                del self._items[k]
        _collect()
        return len(keys)

    def loaded(self) -> list[str]:
        with self._lock:
            return list(self._items)


def _collect() -> None:
    gc.collect()
    try:
        import sys  # noqa: PLC0415

        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover
        pass


LOADED = _Loaded()


def get_loaded(mid: str, variant: str, loader: Callable[[], Any]) -> Any:
    return LOADED.get(f"{mid}:{variant}", loader, MODELS[mid].ram_mb if mid in MODELS else 500)


def unload(mid: str | None = None) -> int:
    """Free loaded models (all when ``mid`` is None). Returns how many were unloaded."""
    return LOADED.unload(mid)
