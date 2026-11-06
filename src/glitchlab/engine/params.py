"""Effect parameter specifications.

A :class:`Param` describes one adjustable value: its type, range, default,
whether it is a *spatial* quantity (expressed in full-resolution pixels and
scaled for the reduced-resolution preview), and how it may be randomised by
the Variation Grid.  The UI builds its controls from these specifications,
so an effect never needs UI code.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

FLOAT = "float"
INT = "int"
CHOICE = "choice"
BOOL = "bool"
HEX = "hex"
TEXT = "text"
FILE = "file"
COLOR = "color"
BITS = "bits"

_HEX_PAIR = re.compile(r"^[0-9A-Fa-f]{2}$")
_COLOR_RE = re.compile(r"^#?([0-9A-Fa-f]{6})$")


def parse_hex_bytes(text: str, *, allow_wildcard: bool = False) -> list[int | None]:
    """Parse a human-typed hex byte string.

    Accepts ``"DA"``, ``"0E DA"``, ``"0eda"``, ``"0x0E 0xDA"`` and (when
    ``allow_wildcard``) ``"??"`` as a single-byte wildcard.  Raises
    :class:`ValueError` with an explanation for invalid input.
    """
    if text is None:
        raise ValueError("No hex bytes entered.")
    cleaned = text.replace(",", " ").replace(";", " ").strip()
    if not cleaned:
        raise ValueError("No hex bytes entered.")
    out: list[int | None] = []
    for token in cleaned.split():
        if token.lower().startswith("0x"):
            token = token[2:]
        if not token:
            raise ValueError("'0x' must be followed by hex digits.")
        if len(token) % 2:
            raise ValueError(
                f"'{token}' has an odd number of hex digits; bytes need two digits each (e.g. 0E)."
            )
        for i in range(0, len(token), 2):
            pair = token[i : i + 2]
            if pair == "??":
                if not allow_wildcard:
                    raise ValueError("Wildcards (??) are only allowed in the search pattern.")
                out.append(None)
            elif _HEX_PAIR.match(pair):
                out.append(int(pair, 16))
            else:
                raise ValueError(f"'{pair}' is not a hexadecimal byte (use 0-9 and A-F).")
    if len(out) > 64:
        raise ValueError("Hex patterns are limited to 64 bytes.")
    return out


def format_hex_bytes(values: list[int | None]) -> str:
    return " ".join("??" if v is None else f"{v:02X}" for v in values)


def parse_color(value: Any, default: str = "#000000") -> str:
    """Normalise a colour to ``#rrggbb``."""
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        r, g, b = (int(max(0, min(255, round(float(c))))) for c in value[:3])
        return f"#{r:02x}{g:02x}{b:02x}"
    if isinstance(value, str):
        m = _COLOR_RE.match(value.strip())
        if m:
            return "#" + m.group(1).lower()
    return default


def color_to_rgb(value: str) -> tuple[int, int, int]:
    v = parse_color(value)
    return int(v[1:3], 16), int(v[3:5], 16), int(v[5:7], 16)


@dataclass(frozen=True)
class Param:
    name: str
    label: str
    kind: str
    default: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: tuple[tuple[str, str], ...] = ()
    spatial: bool = False
    tooltip: str = ""
    group: str = ""
    randomizable: bool = True
    decimals: int = 2
    unit: str = ""
    show_if: tuple[str, tuple] | None = None
    file_filter: str = ""
    nbits: int = 8
    #: free-form flags, e.g. "wildcard" (hex search patterns may use ??).
    flags: tuple[str, ...] = ()

    # ------------------------------------------------------------------
    @property
    def choice_values(self) -> tuple[str, ...]:
        return tuple(c[0] for c in self.choices)

    def coerce(self, value: Any) -> Any:
        """Return a valid value of this parameter's type (clamped/normalised).

        Invalid input falls back to the default so that hand-edited or older
        project files can never crash the engine.
        """
        k = self.kind
        if k == FLOAT:
            try:
                f = float(value)
            except (TypeError, ValueError):
                f = float(self.default)
            if not math.isfinite(f):
                f = float(self.default)
            if self.min is not None:
                f = max(f, float(self.min))
            if self.max is not None:
                f = min(f, float(self.max))
            return f
        if k == INT:
            try:
                f = float(value)
                if not math.isfinite(f):
                    raise ValueError
                i = int(round(f))
            except (TypeError, ValueError, OverflowError):
                i = int(self.default)
            if self.min is not None:
                i = max(i, int(self.min))
            if self.max is not None:
                i = min(i, int(self.max))
            return i
        if k == BOOL:
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)
        if k == CHOICE:
            return value if value in self.choice_values else self.default
        if k in (HEX, TEXT):
            return "" if value is None else str(value)[:512]
        if k == FILE:
            return "" if not value else str(value)
        if k == COLOR:
            return parse_color(value, self.default)
        if k == BITS:
            try:
                i = int(value)
            except (TypeError, ValueError):
                i = int(self.default)
            return i & ((1 << self.nbits) - 1)
        raise ValueError(f"Unknown parameter kind {k!r}")

    def scaled(self, value: Any, scale: float) -> Any:
        """Scale a spatial value from full-resolution pixels to render pixels."""
        if not self.spatial or scale == 1.0:
            return value
        if self.kind == FLOAT:
            return float(value) * scale
        if self.kind == INT:
            v = int(value)
            s = int(round(v * scale))
            if v != 0 and s == 0:
                s = 1 if v > 0 else -1
            return s
        return value

    def random_value(
        self,
        rng: np.random.Generator,
        current: Any,
        intensity: float,
        lo: float | None = None,
        hi: float | None = None,
    ) -> Any:
        """Draw a new value for the Variation Grid.

        ``intensity`` (0..1) limits how far from ``current`` the value may
        move, as a fraction of the allowed range ``[lo, hi]`` (which defaults
        to the parameter's full range).
        """
        intensity = float(min(max(intensity, 0.0), 1.0))
        if not self.randomizable or intensity <= 0:
            return current
        k = self.kind
        if k in (FLOAT, INT):
            lo = float(self.min if lo is None else max(lo, self.min))
            hi = float(self.max if hi is None else min(hi, self.max))
            if hi < lo:
                lo, hi = hi, lo
            cur = float(min(max(float(current), lo), hi))
            half = (hi - lo) * intensity
            a, b = max(lo, cur - half), min(hi, cur + half)
            v = float(rng.uniform(a, b)) if b > a else cur
            return self.coerce(v)
        if k == CHOICE:
            if len(self.choices) > 1 and rng.random() < intensity:
                return self.choices[int(rng.integers(0, len(self.choices)))][0]
            return current
        if k == BOOL:
            return (not current) if rng.random() < intensity * 0.5 else current
        if k == BITS:
            flips = 0
            for bit in range(self.nbits):
                if rng.random() < intensity * 0.3:
                    flips |= 1 << bit
            v = int(current) ^ flips
            return v if v else int(current)
        return current  # hex, text, file, colour: never randomised

    def describe(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "kind": self.kind,
            "default": self.default,
            "min": self.min,
            "max": self.max,
            "choices": [list(c) for c in self.choices],
            "spatial": self.spatial,
        }


# Convenience constructors ---------------------------------------------------

def fparam(name, label, default, lo, hi, *, step=None, decimals=2, **kw) -> Param:
    return Param(name, label, FLOAT, float(default), float(lo), float(hi),
                 step=step if step is not None else (hi - lo) / 100.0, decimals=decimals, **kw)


def iparam(name, label, default, lo, hi, *, step=1, **kw) -> Param:
    return Param(name, label, INT, int(default), int(lo), int(hi), step=step, decimals=0, **kw)


def cparam(name, label, default, choices, **kw) -> Param:
    ch = tuple((c, c.replace("_", " ").title()) if isinstance(c, str) else tuple(c) for c in choices)
    return Param(name, label, CHOICE, default, choices=ch, **kw)


def bparam(name, label, default, **kw) -> Param:
    return Param(name, label, BOOL, bool(default), **kw)


EDGE_MODES = (("wrap", "Wrap"), ("clamp", "Clamp"), ("mirror", "Mirror"))
