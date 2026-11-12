"""Effect base class, registry and the generic effect-application wrapper.

An effect implements :meth:`Effect.process` on a uint8 RGBA image.  The
wrapper :func:`apply_effect` adds, uniformly for every effect:

* enable/bypass and *mix* (strength) blending with the input,
* channel targeting (RGB, single channel, luminance),
* source-alpha preservation (unless the effect supports and the user
  enables *Affect alpha*),
* seeded random generators derived from the global and per-effect seeds,
* output validation (dtype, shape).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from ..cancel import CancelToken
from ..errors import EffectError
from ..ops import lerp_u8, rgb_to_ycbcr, ycbcr_to_rgb
from ..params import Param

if TYPE_CHECKING:  # pragma: no cover
    from ..model import EffectInstance
    from ..services import EngineServices

log = logging.getLogger(__name__)

SEED_MASK = 0xFFFFFFFF

CATEGORIES = (
    "Displacement",
    "Sorting & Scrambling",
    "Data Bending",
    "Audio Bending",
    "Print & Halftone",
    "Analog & Lens",
    "Colour & Bits",
    "Frequency & Feedback",
    "Simulation",
)


@dataclass
class EffectContext:
    rng: np.random.Generator
    seed_sequence: np.random.SeedSequence
    scale: float
    full_size: tuple[int, int]
    cancel: CancelToken
    services: "EngineServices | None" = None
    warnings: list[str] = field(default_factory=list)
    progress_cb: Callable[[float], None] | None = None
    effect_label: str = ""

    def check(self) -> None:
        self.cancel.check()

    def warn(self, message: str) -> None:
        self.warnings.append(f"{self.effect_label}: {message}" if self.effect_label else message)

    def progress(self, fraction: float) -> None:
        if self.progress_cb is not None:
            try:
                self.progress_cb(float(min(max(fraction, 0.0), 1.0)))
            except Exception:  # pragma: no cover - UI callback problems must not kill renders
                pass

    def sub_rng(self, *salt: int) -> np.random.Generator:
        """Independent deterministic generator for a named sub-task."""
        ss = np.random.SeedSequence(entropy=self.seed_sequence.entropy,
                                    spawn_key=tuple(int(s) & SEED_MASK for s in salt))
        return np.random.default_rng(ss)


class Effect:
    id: str = ""
    name: str = ""
    category: str = ""
    description: str = ""
    params: tuple[Param, ...] = ()
    uses_seed: bool = True
    supports_channels: bool = True
    supports_alpha: bool = False
    requires_ffmpeg: bool = False
    #: Short note shown when the reduced preview is only an approximation.
    preview_note: str = ""

    # ------------------------------------------------------------------
    def process(self, img: np.ndarray, p: dict[str, Any], ctx: EffectContext) -> np.ndarray:
        raise NotImplementedError

    def validate(self, p: dict[str, Any]) -> list[str]:
        """Return human readable problems with the parameter set (empty = OK)."""
        return []

    # ------------------------------------------------------------------
    def param(self, name: str) -> Param:
        for prm in self.params:
            if prm.name == name:
                return prm
        raise KeyError(name)

    def defaults(self) -> dict[str, Any]:
        return {prm.name: prm.default for prm in self.params}

    def coerce_params(self, values: dict[str, Any] | None) -> dict[str, Any]:
        values = values or {}
        return {prm.name: prm.coerce(values.get(prm.name, prm.default)) for prm in self.params}

    def resolve(self, values: dict[str, Any] | None, scale: float) -> dict[str, Any]:
        """Coerce and scale spatial parameters for the render resolution."""
        out = self.coerce_params(values)
        for prm in self.params:
            if prm.spatial:
                out[prm.name] = prm.scaled(out[prm.name], scale)
        return out

    def describe(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "params": [p.describe() for p in self.params],
        }


REGISTRY: dict[str, Effect] = {}


def register(cls: type[Effect]) -> type[Effect]:
    inst = cls()
    if not inst.id:
        raise ValueError(f"{cls.__name__} has no id")
    if inst.id in REGISTRY:
        raise ValueError(f"Duplicate effect id {inst.id}")
    names = [p.name for p in inst.params]
    if len(names) != len(set(names)):
        raise ValueError(f"{inst.id}: duplicate parameter names")
    REGISTRY[inst.id] = inst
    return cls


def get_effect(effect_id: str) -> Effect:
    try:
        return REGISTRY[effect_id]
    except KeyError:
        raise EffectError(f"Unknown effect '{effect_id}'. The project may come from a newer version.") from None


def make_seed_sequence(global_seed: int, effect_seed: int) -> np.random.SeedSequence:
    return np.random.SeedSequence([int(global_seed) & SEED_MASK, int(effect_seed) & SEED_MASK, 0x6C17C4])


def _validate_output(eff: Effect, img: np.ndarray, out: Any) -> np.ndarray:
    if not isinstance(out, np.ndarray):
        raise EffectError(f"{eff.name} produced no image.")
    if out.dtype != np.uint8:
        if np.issubdtype(out.dtype, np.floating):
            out = np.nan_to_num(out, nan=0.0, posinf=255.0, neginf=0.0)
        out = np.clip(out, 0, 255).astype(np.uint8)
    if out.shape != img.shape:
        raise EffectError(f"{eff.name} changed the image size from {img.shape} to {out.shape}.")
    return np.ascontiguousarray(out)


def apply_effect(
    img: np.ndarray,
    inst: "EffectInstance",
    *,
    global_seed: int,
    scale: float,
    full_size: tuple[int, int],
    cancel: CancelToken,
    services: "EngineServices | None" = None,
    warnings: list[str] | None = None,
    progress_cb: Callable[[float], None] | None = None,
) -> np.ndarray:
    """Apply one effect instance to ``img`` (uint8 RGBA). Returns a new array."""
    if not inst.enabled or inst.mix <= 0.0:
        return img
    eff = get_effect(inst.effect_id)
    p = eff.resolve(inst.params, scale)
    problems = eff.validate(p)
    if problems:
        raise EffectError(f"{eff.name}: " + " ".join(problems))
    ss = make_seed_sequence(global_seed, inst.seed)
    ctx = EffectContext(
        rng=np.random.default_rng(ss), seed_sequence=ss, scale=scale, full_size=full_size,
        cancel=cancel, services=services, warnings=warnings if warnings is not None else [],
        progress_cb=progress_cb, effect_label=inst.display_name())
    src = img
    if src.flags.writeable:
        src = img.view()
        src.flags.writeable = False  # effects must never modify their input
    channels = inst.channels if eff.supports_channels else "rgb"
    if channels == "luma":
        ycc = rgb_to_ycbcr(src)
        y = np.clip(np.rint(ycc[..., 0]), 0, 255).astype(np.uint8)
        gray = np.dstack([y, y, y, src[..., 3]])
        gray.flags.writeable = False
        res = _validate_output(eff, src, eff.process(gray, p, ctx))
        y2 = rgb_to_ycbcr(res)[..., 0]
        ycc[..., 0] = y2
        out = np.dstack([ycbcr_to_rgb(ycc), res[..., 3]])
    else:
        res = _validate_output(eff, src, eff.process(src, p, ctx))
        if channels in ("r", "g", "b"):
            ci = "rgb".index(channels)
            out = np.array(src, copy=True)
            out[..., ci] = res[..., ci]
            out[..., 3] = res[..., 3]
        else:
            out = res if res is not src else np.array(src, copy=True)
    if not (eff.supports_alpha and inst.affect_alpha):
        if not out.flags.writeable:
            out = np.array(out, copy=True)
        out[..., 3] = src[..., 3]
    if inst.mix < 1.0:
        out = lerp_u8(src, out, float(inst.mix))
    if not out.flags.writeable:
        out = np.array(out, copy=True)
    return out
