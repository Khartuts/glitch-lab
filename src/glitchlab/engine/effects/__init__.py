"""Effect implementations.  Importing this package registers every effect."""

from . import base  # noqa: F401
from . import displace  # noqa: F401
from . import sorting  # noqa: F401
from . import databend  # noqa: F401
from . import sonify  # noqa: F401
from . import colour  # noqa: F401
from . import frequency  # noqa: F401
from . import simulation  # noqa: F401
from .base import CATEGORIES, REGISTRY, Effect, apply_effect, get_effect  # noqa: F401


def effects_by_category() -> dict[str, list[Effect]]:
    out: dict[str, list[Effect]] = {c: [] for c in CATEGORIES}
    for eff in REGISTRY.values():
        out.setdefault(eff.category, []).append(eff)
    for v in out.values():
        v.sort(key=lambda e: e.name)
    return {k: v for k, v in out.items() if v}
