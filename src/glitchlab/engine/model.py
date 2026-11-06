"""Nondestructive document model: layers, effect stacks, masks and seeds.

The document is pure data (JSON-serialisable).  Rendering always starts from
the original image, so moving a slider never accumulates degradation, and
undo/redo only needs to store these small recipes - never image buffers.
"""

from __future__ import annotations

import copy
import json
import secrets
import uuid
from dataclasses import dataclass, field
from typing import Any

from .effects.base import SEED_MASK, get_effect

DOCUMENT_VERSION = 1
BLEND_MODES = ("normal", "multiply", "screen", "difference", "overlay")
ALPHA_MODES = ("auto", "preserve", "composite")
CHANNEL_TARGETS = ("rgb", "r", "g", "b", "luma")


def new_uid() -> str:
    return uuid.uuid4().hex[:10]


def random_seed() -> int:
    return secrets.randbits(32)


@dataclass
class EffectInstance:
    effect_id: str
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    mix: float = 1.0
    seed: int = 0
    channels: str = "rgb"
    affect_alpha: bool = False
    #: Parameters excluded from Variation Grid randomisation.
    locked_params: list[str] = field(default_factory=list)
    seed_locked: bool = False
    #: Whether the Variation Grid may randomise this effect at all.
    vary: bool = True
    #: Optional per-parameter variation ranges {name: [lo, hi]}.
    ranges: dict[str, list[float]] = field(default_factory=dict)
    label: str = ""
    uid: str = field(default_factory=new_uid)

    @classmethod
    def create(cls, effect_id: str, *, seed: int | None = None, **params: Any) -> "EffectInstance":
        eff = get_effect(effect_id)
        p = eff.defaults()
        p.update(params)
        return cls(effect_id=effect_id, params=eff.coerce_params(p),
                   seed=random_seed() if seed is None else int(seed) & SEED_MASK)

    def display_name(self) -> str:
        if self.label:
            return self.label
        try:
            return get_effect(self.effect_id).name
        except Exception:
            return self.effect_id

    def normalise(self) -> "EffectInstance":
        """Coerce every field to a valid value (used after loading files)."""
        eff = get_effect(self.effect_id)
        self.params = eff.coerce_params(self.params)
        self.mix = float(min(max(float(self.mix), 0.0), 1.0))
        self.seed = int(self.seed) & SEED_MASK
        if self.channels not in CHANNEL_TARGETS:
            self.channels = "rgb"
        self.locked_params = [n for n in self.locked_params if n in self.params]
        self.ranges = {k: [float(v[0]), float(v[1])] for k, v in self.ranges.items()
                       if k in self.params and isinstance(v, (list, tuple)) and len(v) == 2}
        return self

    def reset_params(self) -> None:
        self.params = get_effect(self.effect_id).defaults()

    def duplicate(self) -> "EffectInstance":
        d = copy.deepcopy(self)
        d.uid = new_uid()
        return d

    def to_dict(self) -> dict:
        return {
            "effect": self.effect_id,
            "uid": self.uid,
            "label": self.label,
            "enabled": self.enabled,
            "mix": self.mix,
            "seed": self.seed,
            "seed_locked": self.seed_locked,
            "channels": self.channels,
            "affect_alpha": self.affect_alpha,
            "vary": self.vary,
            "locked_params": list(self.locked_params),
            "ranges": {k: list(v) for k, v in self.ranges.items()},
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EffectInstance":
        inst = cls(
            effect_id=str(d["effect"]),
            params=dict(d.get("params", {})),
            enabled=bool(d.get("enabled", True)),
            mix=float(d.get("mix", 1.0)),
            seed=int(d.get("seed", 0)),
            channels=str(d.get("channels", "rgb")),
            affect_alpha=bool(d.get("affect_alpha", False)),
            locked_params=list(d.get("locked_params", [])),
            seed_locked=bool(d.get("seed_locked", False)),
            vary=bool(d.get("vary", True)),
            ranges=dict(d.get("ranges", {})),
            label=str(d.get("label", "")),
            uid=str(d.get("uid") or new_uid()),
        )
        return inst.normalise()

    def cache_signature(self) -> str:
        """Everything that influences the pixels produced by this instance."""
        return json.dumps({
            "e": self.effect_id, "m": round(self.mix, 6), "s": self.seed, "c": self.channels,
            "a": self.affect_alpha, "p": self.params,
        }, sort_keys=True, default=str)


@dataclass
class MaskSpec:
    """Resolution-independent selection mask.

    Shapes use normalised image coordinates (0..1).  Brush radius and
    feather are fractions of the image's long side, so the mask rasterises
    identically at preview and export resolution.
    """

    enabled: bool = False
    shapes: list[dict] = field(default_factory=list)
    feather: float = 0.0
    invert: bool = False
    luma_enabled: bool = False
    luma_low: float = 0.0
    luma_high: float = 1.0
    luma_softness: float = 0.05

    def is_identity(self) -> bool:
        return (not self.enabled) or (not self.shapes and not self.luma_enabled and not self.invert)

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled, "shapes": copy.deepcopy(self.shapes), "feather": self.feather,
            "invert": self.invert, "luma_enabled": self.luma_enabled, "luma_low": self.luma_low,
            "luma_high": self.luma_high, "luma_softness": self.luma_softness,
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> "MaskSpec":
        d = d or {}
        shapes = []
        for s in d.get("shapes", []):
            if not isinstance(s, dict):
                continue
            t = s.get("type")
            mode = "subtract" if s.get("mode") == "subtract" else "add"
            if t == "rect":
                try:
                    shapes.append({"type": "rect", "mode": mode,
                                   "x0": float(s["x0"]), "y0": float(s["y0"]),
                                   "x1": float(s["x1"]), "y1": float(s["y1"])})
                except (KeyError, TypeError, ValueError):
                    continue
            elif t == "stroke":
                pts = [[float(p[0]), float(p[1])] for p in s.get("points", []) if len(p) >= 2]
                if pts:
                    shapes.append({"type": "stroke", "mode": mode, "points": pts,
                                   "radius": float(max(1e-4, min(0.5, s.get("radius", 0.02))))})
        return cls(
            enabled=bool(d.get("enabled", False)), shapes=shapes,
            feather=float(max(0.0, min(0.25, d.get("feather", 0.0)))),
            invert=bool(d.get("invert", False)),
            luma_enabled=bool(d.get("luma_enabled", False)),
            luma_low=float(max(0.0, min(1.0, d.get("luma_low", 0.0)))),
            luma_high=float(max(0.0, min(1.0, d.get("luma_high", 1.0)))),
            luma_softness=float(max(0.0, min(0.5, d.get("luma_softness", 0.05)))),
        )

    def signature(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


@dataclass
class Layer:
    name: str = "Layer"
    stack: list[EffectInstance] = field(default_factory=list)
    visible: bool = True
    opacity: float = 1.0
    blend: str = "normal"
    mask: MaskSpec = field(default_factory=MaskSpec)
    uid: str = field(default_factory=new_uid)

    def find(self, uid: str) -> EffectInstance | None:
        for e in self.stack:
            if e.uid == uid:
                return e
        return None

    def index_of(self, uid: str) -> int:
        for i, e in enumerate(self.stack):
            if e.uid == uid:
                return i
        return -1

    def to_dict(self) -> dict:
        return {
            "uid": self.uid, "name": self.name, "visible": self.visible, "opacity": self.opacity,
            "blend": self.blend, "mask": self.mask.to_dict(),
            "stack": [e.to_dict() for e in self.stack],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Layer":
        blend = d.get("blend", "normal")
        return cls(
            name=str(d.get("name", "Layer")),
            stack=[EffectInstance.from_dict(e) for e in d.get("stack", [])],
            visible=bool(d.get("visible", True)),
            opacity=float(max(0.0, min(1.0, d.get("opacity", 1.0)))),
            blend=blend if blend in BLEND_MODES else "normal",
            mask=MaskSpec.from_dict(d.get("mask")),
            uid=str(d.get("uid") or new_uid()),
        )


@dataclass
class Document:
    layers: list[Layer] = field(default_factory=list)
    global_seed: int = 0
    active_layer: int = 1
    alpha_mode: str = "auto"
    preset_name: str = ""

    @classmethod
    def new(cls, global_seed: int | None = None) -> "Document":
        """Default document: the untouched original under a processing layer."""
        return cls(
            layers=[Layer(name="Original"), Layer(name="Glitch")],
            global_seed=random_seed() if global_seed is None else int(global_seed) & SEED_MASK,
            active_layer=1,
        )

    @classmethod
    def single(cls, stack: list[EffectInstance], global_seed: int = 0) -> "Document":
        """A document with one layer holding ``stack`` (useful for tests/CLI)."""
        return cls(layers=[Layer(name="Glitch", stack=stack)], global_seed=global_seed, active_layer=0)

    @property
    def layer(self) -> Layer:
        if not self.layers:
            self.layers.append(Layer(name="Glitch"))
        self.active_layer = max(0, min(self.active_layer, len(self.layers) - 1))
        return self.layers[self.active_layer]

    def all_effects(self):
        for layer in self.layers:
            yield from layer.stack

    def find_effect(self, uid: str) -> tuple[Layer, EffectInstance] | tuple[None, None]:
        for layer in self.layers:
            e = layer.find(uid)
            if e is not None:
                return layer, e
        return None, None

    def effective_alpha_mode(self) -> str:
        """``auto`` preserves the source alpha unless an enabled effect on a
        visible layer explicitly affects alpha."""
        if self.alpha_mode != "auto":
            return self.alpha_mode
        for layer in self.layers:
            if layer.visible and any(e.enabled and e.affect_alpha for e in layer.stack):
                return "composite"
        return "preserve"

    def copy(self) -> "Document":
        return Document.from_dict(self.to_dict())

    def to_dict(self) -> dict:
        return {
            "version": DOCUMENT_VERSION,
            "global_seed": self.global_seed,
            "active_layer": self.active_layer,
            "alpha_mode": self.alpha_mode,
            "preset_name": self.preset_name,
            "layers": [layer.to_dict() for layer in self.layers],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Document":
        version = int(d.get("version", 1))
        if version > DOCUMENT_VERSION:
            from .errors import ProjectError
            raise ProjectError(
                f"This recipe was made by a newer Glitch Lab (format {version}); "
                f"this version reads format {DOCUMENT_VERSION}.")
        alpha_mode = d.get("alpha_mode", "auto")
        doc = cls(
            layers=[Layer.from_dict(x) for x in d.get("layers", [])],
            global_seed=int(d.get("global_seed", 0)) & SEED_MASK,
            active_layer=int(d.get("active_layer", 0)),
            alpha_mode=alpha_mode if alpha_mode in ALPHA_MODES else "auto",
            preset_name=str(d.get("preset_name", "")),
        )
        if not doc.layers:
            doc.layers = [Layer(name="Glitch")]
        doc.active_layer = max(0, min(doc.active_layer, len(doc.layers) - 1))
        return doc

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_json(cls, text: str) -> "Document":
        return cls.from_dict(json.loads(text))
