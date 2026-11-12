"""Seeded variation generation for the Variation Grid.

Every variation is a complete :class:`Document`, so clicking a thumbnail
restores its exact recipe.  Randomisation is limited to

* effects whose *Include in variations* flag is on (and that are enabled),
* parameters that are not locked, within each parameter's optional custom
  range, moved at most ``intensity`` x (range width) from the current value,
* effect seeds that are not locked (if ``reroll_seeds``),
* the global seed only if ``vary_global_seed``.

Any count from 1 to 64 can be requested.

The same base document + settings (including ``batch_seed``) always yields
the same variations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .effects.base import SEED_MASK, get_effect
from .model import Document

MIN_VARIATIONS = 1
MAX_VARIATIONS = 64
#: Quick choices offered by older versions (kept for compatibility).
VARIATION_COUNTS = (4, 9, 16)


@dataclass
class VariationSettings:
    count: int = 9
    intensity: float = 0.35
    reroll_seeds: bool = True
    vary_global_seed: bool = False
    batch_seed: int = 0
    vary_mix: bool = False

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict) -> "VariationSettings":
        s = cls()
        for k, v in (d or {}).items():
            if hasattr(s, k):
                setattr(s, k, type(getattr(s, k))(v))
        return s


def variable_parameter_count(doc: Document) -> int:
    n = 0
    for layer in doc.layers:
        for inst in layer.stack:
            if not (inst.enabled and inst.vary):
                continue
            eff = get_effect(inst.effect_id)
            n += sum(1 for p in eff.params if p.randomizable and p.name not in inst.locked_params
                     and p.kind not in ("hex", "text", "file", "color"))
            if eff.uses_seed and not inst.seed_locked:
                n += 1
    return n


def make_variation(base: Document, settings: VariationSettings, index: int) -> Document:
    rng = np.random.default_rng(np.random.SeedSequence([int(settings.batch_seed) & SEED_MASK, int(index), 0x7A21]))
    doc = base.copy()
    if settings.vary_global_seed:
        doc.global_seed = int(rng.integers(0, 2 ** 32))
    for layer in doc.layers:
        for inst in layer.stack:
            if not (inst.enabled and inst.vary):
                continue
            eff = get_effect(inst.effect_id)
            for prm in eff.params:
                if prm.name in inst.locked_params:
                    continue
                lo, hi = inst.ranges.get(prm.name, (None, None))
                inst.params[prm.name] = prm.random_value(rng, inst.params.get(prm.name, prm.default),
                                                         settings.intensity, lo, hi)
            if settings.vary_mix and "mix" not in inst.locked_params:
                inst.mix = float(np.clip(inst.mix + rng.uniform(-0.5, 0.5) * settings.intensity, 0.2, 1.0))
            if eff.uses_seed and settings.reroll_seeds and not inst.seed_locked:
                inst.seed = int(rng.integers(0, 2 ** 32))
            inst.normalise()
    return doc


def generate_variations(base: Document, settings: VariationSettings) -> list[Document]:
    count = int(min(max(int(settings.count), MIN_VARIATIONS), MAX_VARIATIONS))
    return [make_variation(base, settings, i) for i in range(count)]
