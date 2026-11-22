"""Generative AI steps (VAE latents and Stable Diffusion 1.5), all local and cached.

Stable Diffusion results are stored on disk, keyed by the input image, every
generation setting, the seed and the pinned model revision, so composition
changes and reopened projects are immediate and an interrupted multi-step
generation resumes from the last finished step.  Repeatability is aimed at
the same software and hardware; diffusion output is not promised to be
identical on different devices.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any

import cv2
import numpy as np

from ..engine.effects.base import EffectContext
from ..engine.errors import Cancelled, EffectError
from . import models
from .jobs import JobContext, make_key, manager, take_approval
from .support import (analysis_key, cached_analysis, device, ensure_ready, fit_size, inference_mode, input_identity,
                      load_img2img, load_inpaint, load_vae, rgb_resized, torch_generator)

#: Measured seconds per denoising step (updated after every generation; used for estimates).
SECONDS_PER_STEP: dict[str, float] = {}
DEFAULT_SECONDS_PER_STEP = {"cpu": 1.6, "cuda": 0.06, "mps": 0.3}
_LAST_GENERATIONS: dict[str, list[str]] = {}
_LG_LOCK = threading.Lock()


class GenerationPending(EffectError):
    """Raised in interactive previews until the user asks for the (slow) generation."""


def seconds_per_step(res: int = 512) -> float:
    import sys  # noqa: PLC0415

    dev = "cpu"
    if "torch" in sys.modules:  # never import PyTorch just to show an estimate
        try:
            dev = device()
        except Exception:
            dev = "cpu"
    base = SECONDS_PER_STEP.get(dev, DEFAULT_SECONDS_PER_STEP.get(dev, 1.6))
    return base * (res / 512.0) ** 2


def estimate_text(steps: int, res: int) -> str:
    secs = steps * seconds_per_step(res)
    t = f"{secs:.0f} s" if secs < 90 else f"{secs / 60:.1f} min"
    return f"Estimated work: {steps} denoising steps at {res} px (about {t} on this computer)."


def gen_seed(ctx: EffectContext, salt: int = 0) -> int:
    return int(ctx.sub_rng(9000 + salt).integers(0, 2**31 - 1))


def _gate(ctx: EffectContext, auto: bool, key: str, persist: bool, label: str, run_key: str | None = None) -> None:
    """Interactive previews only generate after 'Generate' (or with automatic generation on)."""
    if not ctx.interactive or auto:
        return
    mgr = manager()
    if mgr.cached(key, persist) is not None or mgr.in_flight(run_key or key):
        return
    if take_approval(ctx.owner):
        return
    raise GenerationPending(f"{label} has not been generated for these settings yet. Click 'Generate' in the "
                            "effect panel (or tick 'Generate automatically').")


def _step_callback(job: JobContext, total: int, offset: float = 0.0, span: float = 1.0, msg: str = ""):
    state = {"t": time.monotonic(), "n": 0}

    def cb(pipe, step, timestep, kwargs):
        job.check()
        state["n"] += 1
        job.progress(offset + span * min(1.0, (step + 1) / max(total, 1)), msg or f"Denoising step {step + 1}/{total}")
        return kwargs
    return cb, state


def _record_speed(state: dict, res: int) -> None:
    n = state["n"]
    if n >= 2:
        dt = (time.monotonic() - state["t"]) / n
        try:
            SECONDS_PER_STEP[device()] = dt / (res / 512.0) ** 2
        except Exception:  # pragma: no cover
            pass


def _to_pil(rgb: np.ndarray):
    from PIL import Image  # noqa: PLC0415

    return Image.fromarray(np.ascontiguousarray(rgb))


def _np_image(out) -> np.ndarray:
    img = out.images[0]
    if not isinstance(img, np.ndarray):
        img = np.asarray(img, np.float32) / 255.0
    img = np.nan_to_num(np.asarray(img, np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


# =================================================================== VAE
def vae_info() -> dict:
    """Latent conventions read from the VAE's configuration."""
    vae = load_vae()
    cfg = vae.config
    factor = 2 ** (len(cfg.block_out_channels) - 1)
    shift = getattr(cfg, "shift_factor", None)
    return {"channels": int(cfg.latent_channels), "factor": int(factor),
            "scale": float(getattr(cfg, "scaling_factor", 1.0) or 1.0), "shift": float(shift or 0.0)}


def vae_encode(ctx: EffectContext, img: np.ndarray, res: int) -> dict:
    """Posterior-mean latents (normalised by the model's scale/shift) and the
    baseline reconstruction (decode of the unmodified latents)."""

    def run(full, job):
        import torch  # noqa: PLC0415

        info = vae_info()
        h, w = full.shape[:2]
        gw, gh = fit_size(w, h, res, info["factor"] * 2)
        rgb = rgb_resized(full, gw, gh)
        vae = load_vae()
        x = torch.from_numpy(rgb.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None].to(device())
        job.progress(0.15, "Encoding image into latents")
        with inference_mode():
            z = vae.encode(x).latent_dist.mean
            zn = (z - info["shift"]) * info["scale"]
            job.progress(0.55, "Decoding baseline reconstruction")
            base = vae.decode(zn / info["scale"] + info["shift"]).sample
        base = ((base[0].clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).round().byte().cpu().numpy()
        return {"z": zn[0].float().cpu().numpy(), "baseline": base, "rgb": rgb,
                "factor": np.int32(info["factor"])}

    return cached_analysis(ctx, img, "vae-encode", {"res": res}, run, model_ids=("vae",),
                           label="Latent encoding (VAE)", persist=True)


def vae_decode(ctx: EffectContext, img: np.ndarray, res: int, z: np.ndarray, settings: dict) -> np.ndarray:
    z = np.nan_to_num(z.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    def run(_full, job):
        import torch  # noqa: PLC0415

        info = vae_info()
        vae = load_vae()
        job.progress(0.3, "Decoding mutated latents")
        with inference_mode():
            t = torch.from_numpy(z)[None].to(device())
            out = vae.decode(t / info["scale"] + info["shift"]).sample
        out = torch.nan_to_num(out, nan=0.0).clamp(-1, 1)
        return {"rgb": ((out[0] + 1) * 127.5).permute(1, 2, 0).round().byte().cpu().numpy()}

    return cached_analysis(ctx, img, "vae-decode", {"res": res, **settings}, run, model_ids=("vae",),
                           label="Latent decoding (VAE)", persist=True)["rgb"]


# =================================================================== inpainting
def inpaint(ctx: EffectContext, img: np.ndarray, mask_fn, settings: dict, *, auto: bool) -> dict:
    """Inpaint the scar mask with SD 1.5 inpainting. ``mask_fn(w, h)`` returns
    the (dilated) generation mask at the generation size."""
    res = int(settings["res"])
    ensure_ready("sd15_inpaint")
    key_settings = dict(settings)
    key = analysis_key(ctx, img, "sd-inpaint", key_settings, ("sd15_inpaint",))
    _gate(ctx, auto, key, True, "Hallucination Scars")

    def run(full, job):
        h, w = full.shape[:2]
        gw, gh = fit_size(w, h, res, 8)
        rgb = rgb_resized(full, gw, gh)
        mask = mask_fn(gw, gh)
        pipe = load_inpaint()
        steps = int(settings["steps"])
        neg = settings["negative"] or None
        image = rgb
        span0 = 0.0
        out_plate = None
        if settings.get("fill", "material") == "material":
            # Stage 1: a pure material plate from the prompt (the whole frame is "missing").
            cb, state = _step_callback(job, steps, 0.02, 0.48, "Inventing the material")
            plate = pipe(prompt=settings["prompt"], negative_prompt=neg, image=_to_pil(rgb),
                         mask_image=_to_pil(np.full((gh, gw), 255, np.uint8)), strength=1.0,
                         guidance_scale=float(settings["guidance"]), num_inference_steps=steps, height=gh,
                         width=gw, generator=torch_generator(settings["seed"] + 1), callback_on_step_end=cb,
                         output_type="np")
            _record_speed(state, res)
            out_plate = _np_image(plate)
            # Stage 2: inpaint the scar zones, starting from the material, so it meets the photo.
            image = np.where(mask[..., None] > 0.5, out_plate, rgb).astype(np.uint8)
            span0 = 0.5
        total = max(1, int(steps * min(1.0, settings["strength"])))
        cb, state = _step_callback(job, total, span0 + 0.02, 0.98 - span0, "Inpainting scars")
        job.progress(span0 + 0.01, "Inpainting scars")
        out = pipe(prompt=settings["prompt"], negative_prompt=neg, image=_to_pil(image),
                   mask_image=_to_pil((mask * 255).astype(np.uint8)), strength=float(settings["strength"]),
                   guidance_scale=float(settings["guidance"]), num_inference_steps=steps, height=gh, width=gw,
                   generator=torch_generator(settings["seed"]), callback_on_step_end=cb, output_type="np")
        _record_speed(state, res)
        result = {"rgb": _np_image(out), "mask": (mask * 255).astype(np.uint8)}
        if out_plate is not None:
            result["plate"] = out_plate
        return result

    return cached_analysis(ctx, img, "sd-inpaint", key_settings, run, model_ids=("sd15_inpaint",),
                           label="Hallucination Scars (SD 1.5 inpainting)", persist=True)


# =================================================================== img2img
def _img2img(pipe, rgb: np.ndarray, prompt: str, negative: str, strength: float, guidance: float, steps: int,
             seed: int, job: JobContext, offset: float, span: float, msg: str) -> np.ndarray:
    total = max(1, int(steps * min(1.0, strength)))
    cb, state = _step_callback(job, total, offset, span, msg)
    out = pipe(prompt=prompt, negative_prompt=negative or None, image=_to_pil(rgb), strength=float(strength),
               guidance_scale=float(guidance), num_inference_steps=int(steps), generator=torch_generator(seed),
               callback_on_step_end=cb, output_type="np")
    _record_speed(state, max(rgb.shape[:2]))
    return _np_image(out)


def counterfactual_pair(ctx: EffectContext, img: np.ndarray, settings: dict, *, auto: bool) -> dict:
    """Two image-to-image interpretations with the same model, size, scheduler and initial noise."""
    ensure_ready("sd15")
    key = analysis_key(ctx, img, "sd-pair", settings, ("sd15",))
    _gate(ctx, auto, key, True, "Counterfactual Residue")

    def run(full, job):
        h, w = full.shape[:2]
        gw, gh = fit_size(w, h, int(settings["res"]), 8)
        rgb = rgb_resized(full, gw, gh)
        pipe = load_img2img()
        a = _img2img(pipe, rgb, settings["prompt_a"], settings["negative"], settings["strength"],
                     settings["guidance"], settings["steps"], settings["seed"], job, 0.0, 0.5, "Interpretation A")
        b = _img2img(pipe, rgb, settings["prompt_b"], settings["negative"], settings["strength"],
                     settings["guidance"], settings["steps"], settings["seed"], job, 0.5, 0.5, "Interpretation B")
        return {"a": a, "b": b, "rgb": rgb, "scheduler": np.array(type(pipe.scheduler).__name__)}

    return cached_analysis(ctx, img, "sd-pair", settings, run, model_ids=("sd15",),
                           label="Counterfactual Residue (SD 1.5)", persist=True)


# =================================================================== recursion
def prompt_for(prompts: list[str], schedule: str, g: int, n: int) -> str:
    if not prompts:
        return ""
    if schedule == "constant":
        return prompts[0]
    if schedule == "alternate":
        return prompts[g % len(prompts)]
    return prompts[min(len(prompts) - 1, int(g * len(prompts) / max(n, 1)))]


def seed_for(base: int, schedule: str, g: int) -> int:
    if schedule == "fixed":
        return base
    if schedule == "increment":
        return (base + g) & 0x7FFFFFFF
    return int(np.random.default_rng([base, g]).integers(0, 2**31 - 1))


def misremember(ctx: EffectContext, img: np.ndarray, settings: dict, protect_fn, *, auto: bool) -> list[np.ndarray]:
    """Run the bounded generation chain; every finished generation is cached on
    disk so a cancelled run resumes where it stopped. Returns [original, g1, g2, ...]."""
    ensure_ready("sd15")
    ident = input_identity(ctx, img)
    rev = models.revision("sd15")
    n = int(settings["generations"])
    keys = []
    prev = make_key("misremember-start", ident, settings["res"], rev)
    for g in range(n):
        step = {k: settings[k] for k in ("strength", "guidance", "steps", "anchor", "negative", "res", "protect")}
        step.update(prompt=prompt_for(settings["prompts"], settings["schedule"], g, n),
                    seed=seed_for(settings["seed"], settings["seed_schedule"], g))
        prev = make_key("misremember-gen", prev, step)
        keys.append((prev, step))
    mgr = manager()
    final_key = keys[-1][0]
    _gate(ctx, auto, final_key, True, "Recursive Misremembering", make_key("misremember-run", final_key))
    with _LG_LOCK:
        _LAST_GENERATIONS[ctx.owner] = [k for k, _ in keys]
    if all(mgr.cached(k, True) is not None for k, _ in keys):
        start = mgr.cached(make_key("misremember-src", ident, settings["res"]), True)
        if start is not None:
            return [start["rgb"]] + [mgr.cached(k, True)["rgb"] for k, _ in keys]
    full = np.ascontiguousarray(ctx.full_input() if ctx.full_input is not None else img)

    def run(job):
        h, w = full.shape[:2]
        gw, gh = fit_size(w, h, int(settings["res"]), 8)
        rgb0 = rgb_resized(full, gw, gh)
        mgr.store(make_key("misremember-src", ident, settings["res"]), {"rgb": rgb0}, True)
        prot = protect_fn(gw, gh, rgb0)
        cur = rgb0
        outs = [rgb0]
        pipe = None
        for g, (k, step) in enumerate(keys):
            job.check()
            hit = mgr.cached(k, True)
            if hit is not None:  # resume: this generation was finished before
                cur = hit["rgb"]
                outs.append(cur)
                continue
            if pipe is None:
                pipe = load_img2img()
            src = cur.astype(np.float32)
            if step["anchor"] > 0:
                src = src * (1 - step["anchor"]) + rgb0.astype(np.float32) * step["anchor"]
            res = _img2img(pipe, np.clip(src + 0.5, 0, 255).astype(np.uint8), step["prompt"], step["negative"],
                           step["strength"], step["guidance"], step["steps"], step["seed"], job,
                           g / n, 1.0 / n, f"Generation {g + 1}/{n}")
            if prot is not None:
                res = np.where(prot[..., None] > 0.5, rgb0, res)
            mgr.store(k, {"rgb": res}, True)
            cur = res
            outs.append(res)
        return {f"g{i}": o for i, o in enumerate(outs)}

    res = mgr.run(make_key("misremember-run", final_key), "Recursive Misremembering (SD 1.5)", run,
                  owner=ctx.owner, persist=False, cancel=ctx.cancel, progress=ctx.progress)
    return [res[f"g{i}"] for i in range(n + 1)]


def last_generations(owner: str) -> list[np.ndarray]:
    """The cached intermediate generations of the most recent run of an effect instance."""
    with _LG_LOCK:
        keys = list(_LAST_GENERATIONS.get(owner, []))
    out = []
    for k in keys:
        hit = manager().cached(k, True)
        if hit is None:
            break
        out.append(hit["rgb"])
    return out


def total_steps(generations: int, steps: int, strength: float) -> int:
    return int(generations) * max(1, int(math.floor(steps * min(1.0, strength))))


__all__ = ["GenerationPending", "vae_encode", "vae_decode", "inpaint", "counterfactual_pair", "misremember",
           "last_generations", "estimate_text", "total_steps", "Any", "Cancelled"]
