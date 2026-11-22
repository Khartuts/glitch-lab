"""Helpers shared by the AI effects: cached inference, model loaders and
pre-processing.  Pre-processing follows each model's own
``preprocessor_config.json`` (read at load time, not hard-coded)."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Callable

import cv2
import numpy as np

from ..engine.effects.base import EffectContext
from . import models, runtime
from .jobs import JobContext, make_key, manager

#: "auto", "cpu", "cuda" or "mps"; set from Preferences / the command line.
DEVICE_PREFERENCE = os.environ.get("GLITCHLAB_AI_DEVICE", "auto")

for _k, _v in (("HF_HUB_OFFLINE", "1"), ("TRANSFORMERS_OFFLINE", "1"), ("HF_HUB_DISABLE_TELEMETRY", "1"),
               ("DIFFUSERS_VERBOSITY", "error"), ("TRANSFORMERS_VERBOSITY", "error"),
               ("HF_HUB_DISABLE_PROGRESS_BARS", "1")):
    os.environ.setdefault(_k, _v)


def device() -> str:
    return runtime.select_device(DEVICE_PREFERENCE)


def ensure_ready(*model_ids: str) -> None:
    """Raise a clear error when the engine or a model is missing."""
    if not runtime.engine_available():
        from .errors import AIUnavailable  # noqa: PLC0415

        raise AIUnavailable()
    models.require(*model_ids)


def input_identity(ctx: EffectContext, img: np.ndarray) -> str:
    if ctx.input_key:
        return ctx.input_key
    import hashlib  # noqa: PLC0415

    return hashlib.sha1(f"{img.shape}|{ctx.scale}".encode() + np.ascontiguousarray(img).tobytes()).hexdigest()


def cached_analysis(ctx: EffectContext, img: np.ndarray, name: str, settings: dict,
                    fn: Callable[[np.ndarray, JobContext], Any], *, model_ids: tuple[str, ...] = (),
                    label: str = "", persist: bool = False, extra_key: Any = None) -> Any:
    """Run ``fn(full_resolution_input_rgba, job)`` once per input/settings and cache it.

    The result is shared by the preview and the full-resolution export (the
    analysis always uses the full-resolution input) and survives styling
    changes.  The model revisions are part of the key.
    """
    ensure_ready(*model_ids)
    key = analysis_key(ctx, img, name, settings, model_ids, extra_key)
    mgr = manager()
    hit = mgr.cached(key, persist)
    if hit is not None:
        return hit
    full = ctx.full_input() if ctx.full_input is not None else img
    full = np.ascontiguousarray(full)
    return mgr.run(key, label or ctx.effect_label or name, lambda job: fn(full, job), owner=ctx.owner,
                   persist=persist, cancel=ctx.cancel, progress=ctx.progress)


def analysis_key(ctx: EffectContext, img: np.ndarray, name: str, settings: dict, model_ids=(), extra_key=None) -> str:
    return make_key(name, input_identity(ctx, img), settings, [models.revision(m) for m in model_ids], extra_key)


# ------------------------------------------------------------------ images
def fit_size(w: int, h: int, max_side: int, multiple: int = 1, min_side: int = 0) -> tuple[int, int]:
    s = min(1.0, max_side / float(max(w, h)))
    nw, nh = max(1, w * s), max(1, h * s)
    if min_side:
        k = max(1.0, min_side / min(nw, nh))
        nw, nh = nw * k, nh * k
    if multiple > 1:
        nw = max(multiple, int(round(nw / multiple)) * multiple)
        nh = max(multiple, int(round(nh / multiple)) * multiple)
    return int(round(nw)), int(round(nh))


def rgb_resized(img: np.ndarray, w: int, h: int) -> np.ndarray:
    rgb = np.ascontiguousarray(img[..., :3])
    if rgb.shape[1] == w and rgb.shape[0] == h:
        return rgb
    interp = cv2.INTER_AREA if w < rgb.shape[1] else cv2.INTER_CUBIC
    return cv2.resize(rgb, (w, h), interpolation=interp)


def upsample(arr: np.ndarray, w: int, h: int, *, nearest: bool = False) -> np.ndarray:
    """Resize an analysis map (H, W[, C]) to the render size."""
    if arr.shape[1] == w and arr.shape[0] == h:
        return arr
    interp = cv2.INTER_NEAREST if nearest else (cv2.INTER_AREA if w < arr.shape[1] else cv2.INTER_LINEAR)
    return cv2.resize(np.ascontiguousarray(arr), (w, h), interpolation=interp)


@lru_cache(maxsize=16)
def preprocessor_config(mid: str) -> dict:
    p = models.path(mid) / "preprocessor_config.json"
    try:
        return json.loads(p.read_text())
    except OSError:
        return {}


def normalised_tensor(rgb: np.ndarray, mid: str):
    """uint8 RGB (H, W, 3) -> float tensor (1, 3, H, W) normalised as the model expects."""
    torch = runtime.import_torch()
    cfg = preprocessor_config(mid)
    mean = np.asarray(cfg.get("image_mean", [0.485, 0.456, 0.406]), np.float32)
    std = np.asarray(cfg.get("image_std", [0.229, 0.224, 0.225]), np.float32)
    scale = float(cfg.get("rescale_factor", 1 / 255.0))
    x = (rgb.astype(np.float32) * scale - mean) / std
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None]


# ------------------------------------------------------------------ loaders
def _dtype_kw(lib: str) -> dict:
    torch = runtime.import_torch()
    return {"dtype": torch.float32}


def load_sam2():
    def load():
        from transformers import Sam2Model  # noqa: PLC0415

        return Sam2Model.from_pretrained(str(models.path("sam2")), **_dtype_kw("transformers")).eval().to(device())
    return models.get_loaded("sam2", device(), load)


def load_depth():
    def load():
        from transformers import AutoModelForDepthEstimation  # noqa: PLC0415

        return AutoModelForDepthEstimation.from_pretrained(str(models.path("depth")),
                                                           **_dtype_kw("transformers")).eval().to(device())
    return models.get_loaded("depth", device(), load)


def load_dinov2():
    def load():
        from transformers import AutoModel  # noqa: PLC0415

        return AutoModel.from_pretrained(str(models.path("dinov2")), **_dtype_kw("transformers")).eval().to(device())
    return models.get_loaded("dinov2", device(), load)


def load_vae():
    def load():
        from diffusers import AutoencoderKL  # noqa: PLC0415

        return AutoencoderKL.from_pretrained(str(models.path("vae")), **_dtype_kw("diffusers")).eval().to(device())
    return models.get_loaded("vae", device(), load)


def _sd_pipeline(cls_name: str, mid: str):
    def load():
        import diffusers  # noqa: PLC0415

        cls = getattr(diffusers, cls_name)
        pipe = cls.from_pretrained(str(models.path(mid)), variant="fp16", safety_checker=None,
                                   requires_safety_checker=False, feature_extractor=None, **_dtype_kw("diffusers"))
        pipe = pipe.to(device())
        pipe.set_progress_bar_config(disable=True)
        return pipe
    # Only one large diffusion pipeline is kept in memory at a time.
    for other in ("sd15", "sd15_inpaint"):
        if other != mid:
            models.unload(other)
    return models.get_loaded(mid, device(), load)


def load_img2img():
    return _sd_pipeline("StableDiffusionImg2ImgPipeline", "sd15")


def load_inpaint():
    return _sd_pipeline("StableDiffusionInpaintPipeline", "sd15_inpaint")


def torch_generator(seed: int):
    torch = runtime.import_torch()
    # CPU generator: the same seed gives the same initial noise on every device.
    return torch.Generator(device="cpu").manual_seed(int(seed) & 0xFFFFFFFF)


def inference_mode():
    torch = runtime.import_torch()
    return torch.inference_mode()
