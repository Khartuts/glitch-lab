"""Text boxes, typography styles and layout presets (data only).

A text box is a plain dict validated by :data:`BOX_PARAMS` (the UI builds
its controls from the same specifications, like effect parameters).  The
``text`` is stored exactly as typed -- wording, punctuation, Unicode and line
breaks are never rewritten; layout and styles only change how it is drawn.

Positions are fractions of the image (``x``/``y`` = box centre, ``w``/``h`` =
box size); the font size is a fraction of the image height and every style
length is in *em* (font size), so previews and exports have the same
layout and the same damage pattern.
"""

from __future__ import annotations

import copy
import secrets
import uuid

from ..params import (BOOL, COLOR, DATA, MASK, POINTS, TEXT, Param, bparam, cparam, fparam, iparam)

ALIGNS = (("left", "Left"), ("center", "Centre"), ("right", "Right"), ("justify", "Justified"))
LAYOUT_MODES = (("box", "Text box"), ("path", "Follow a drawn path"), ("scatter", "Scattered fragments"))

# ------------------------------------------------------------------ styles
STYLE_PARAMS: dict[str, tuple[Param, ...]] = {
    "clean": (),
    "thrashed": (
        fparam("abrasion", "Abrasion", 0.5, 0.0, 1.0),
        fparam("scratch_scale", "Scratch scale", 1.0, 0.1, 4.0),
        fparam("ink_loss", "Ink loss", 0.35, 0.0, 1.0),
        fparam("chipping", "Edge chipping", 0.5, 0.0, 1.0),
        fparam("misregistration", "Misregistration", 0.06, 0.0, 0.4, decimals=3),
        cparam("distribution", "Damage", "edges", (("edges", "At the edges"), ("throughout", "Throughout letters"))),
    ),
    "industrial": (
        fparam("stencil", "Stencil width", 0.08, 0.0, 0.3, decimals=3),
        fparam("pressure", "Stamp pressure", 0.6, 0.0, 1.0),
        fparam("slices", "Slice displacement", 0.12, 0.0, 1.0),
        fparam("condense", "Compression", 0.15, 0.0, 0.5, tooltip="Horizontal compression of the letters."),
        fparam("plate", "Plate opacity", 0.0, 0.0, 1.0),
        Param("plate_colour", "Plate colour", COLOR, "#d8b026"),
        fparam("registration", "Registration offset", 0.04, 0.0, 0.3, decimals=3),
    ),
    "acid": (
        fparam("liquidity", "Liquidity", 0.5, 0.0, 1.5),
        fparam("vortex_scale", "Vortex scale", 1.2, 0.2, 6.0, tooltip="In em (font size)."),
        fparam("stretching", "Stretching", 0.4, 0.0, 1.5),
        fparam("separation", "Colour separation", 0.06, 0.0, 0.4, decimals=3),
        fparam("bridges", "Bridge density", 0.4, 0.0, 1.0),
        bparam("iridescent", "Iridescent fill", True),
    ),
    "rust": (
        fparam("coverage", "Corrosion coverage", 0.5, 0.0, 1.0),
        fparam("pit_size", "Pit size", 0.05, 0.01, 0.3, decimals=3),
        fparam("flaking", "Flaking", 0.4, 0.0, 1.0),
        fparam("metallic", "Metallic contrast", 0.6, 0.0, 1.5),
        cparam("palette", "Palette", "orange", (("orange", "Oxidised orange"), ("iron", "Dark iron"),
                                                ("verdigris", "Verdigris"))),
        fparam("light", "Highlight direction", 135.0, 0.0, 360.0, decimals=0, unit="°"),
    ),
    "xerox": (
        iparam("generations", "Copy generations", 4, 1, 12),
        fparam("threshold", "Threshold", 0.4, 0.2, 0.8),
        fparam("toner", "Toner density", 0.8, 0.1, 1.0),
        fparam("bleed", "Bleed", 0.25, 0.0, 1.0),
        fparam("drift", "Registration drift", 0.04, 0.0, 0.3, decimals=3),
        cparam("ink", "Toner", "black", (("black", "Black toner"), ("white", "White toner"),
                                         ("spot", "Spot colour (fill colour)"))),
    ),
    "cut": (
        fparam("font_variation", "Font variation", 0.8, 0.0, 1.0),
        fparam("rotation", "Rotation", 6.0, 0.0, 30.0, decimals=1, unit="°"),
        fparam("paper_margin", "Paper margin", 0.12, 0.0, 0.5, decimals=3),
        fparam("tearing", "Tearing", 0.5, 0.0, 1.0),
        fparam("baseline", "Baseline irregularity", 0.12, 0.0, 0.6, decimals=3),
        cparam("unit", "Fragments", "word", (("word", "Words"), ("char", "Characters"))),
        Param("locks", "Locked fragments", DATA, {}),
    ),
    "transmission": (
        fparam("slice_density", "Slice density", 0.35, 0.0, 1.0),
        fparam("displacement", "Displacement", 0.25, 0.0, 1.5),
        fparam("dropout", "Dropout", 0.15, 0.0, 1.0),
        fparam("pixel_scale", "Pixel scale", 0.08, 0.02, 0.5, decimals=3),
        fparam("separation", "Colour separation", 0.06, 0.0, 0.4, decimals=3),
        fparam("substitution", "Decorative substitution", 0.0, 0.0, 0.5,
               tooltip="Off by default. Draws some characters as glitch blocks; the text itself is never changed."),
    ),
    "ghost": (
        iparam("echoes", "Echo count", 4, 1, 12),
        fparam("distance", "Distance", 0.12, 0.0, 1.0, decimals=3),
        fparam("fading", "Fading", 0.55, 0.05, 0.95),
        fparam("blur", "Blur", 0.4, 0.0, 1.5),
        fparam("erosion", "Erosion", 0.35, 0.0, 1.0),
        cparam("spread", "Spread", "drift", (("drift", "Horizontal drift"), ("radial", "Radial"))),
    ),
}
STYLE_NAMES = {"clean": "Clean", "thrashed": "Thrashed", "industrial": "Industrial", "acid": "Acid / Liquid Type",
               "rust": "Rust / Oxidised", "xerox": "Xerox / Photocopier Decay", "cut": "Cut-and-Paste",
               "transmission": "Transmission Failure", "ghost": "Ghost Carbon"}
STYLE_IDS = tuple(STYLE_NAMES)

# ------------------------------------------------------------------ boxes
BOX_PARAMS: tuple[Param, ...] = (
    Param("text", "Text", TEXT, "", flags=("long", "live", "raw")),
    Param("font", "Font", TEXT, "DejaVu Sans|Bold", flags=("font",)),
    fparam("size", "Size", 0.07, 0.004, 0.6, decimals=3, group="Layout", tooltip="Fraction of the image height."),
    fparam("tracking", "Tracking", 0.0, -0.3, 1.0, decimals=3, group="Layout", tooltip="Letter spacing in em."),
    fparam("line_spacing", "Line spacing", 1.15, 0.6, 3.0, group="Layout"),
    cparam("align", "Alignment", "left", ALIGNS, group="Layout"),
    cparam("valign", "Vertical alignment", "top", (("top", "Top"), ("middle", "Middle"), ("bottom", "Bottom")),
           group="Layout"),
    bparam("wrap", "Wrap lines automatically", True, group="Layout"),
    bparam("preserve_breaks", "Keep my line breaks", True, group="Layout"),
    bparam("fit", "Fit text to the box", False, group="Layout"),
    fparam("min_size", "Minimum size when fitting", 0.012, 0.003, 0.2, decimals=3, group="Layout"),
    cparam("layout", "Layout", "box", LAYOUT_MODES, group="Layout"),
    Param("path", "Path", POINTS, [], group="Layout", show_if=("layout", ("path",))),
    fparam("x", "Centre X", 0.5, -0.5, 1.5, decimals=3, group="Position"),
    fparam("y", "Centre Y", 0.5, -0.5, 1.5, decimals=3, group="Position"),
    fparam("w", "Width", 0.6, 0.02, 2.0, decimals=3, group="Position"),
    fparam("h", "Height", 0.3, 0.02, 2.0, decimals=3, group="Position"),
    fparam("rotation", "Rotation", 0.0, -180.0, 180.0, decimals=1, unit="°", group="Position"),
    fparam("persp_x", "Perspective (horizontal)", 0.0, -0.6, 0.6, group="Position"),
    fparam("persp_y", "Perspective (vertical)", 0.0, -0.6, 0.6, group="Position"),
    Param("fill", "Fill colour", COLOR, "#f4f1ea", group="Colour"),
    fparam("opacity", "Opacity", 1.0, 0.0, 1.0, group="Colour"),
    Param("outline", "Outline colour", COLOR, "#0d0d0d", group="Colour"),
    fparam("outline_width", "Outline width", 0.0, 0.0, 0.25, decimals=3, group="Colour", tooltip="In em."),
    Param("damage_colour", "Damage colour", COLOR, "#d12a2a", group="Colour"),
    Param("shadow", "Shadow colour", COLOR, "#000000", group="Colour"),
    fparam("shadow_opacity", "Shadow opacity", 0.0, 0.0, 1.0, group="Colour"),
    fparam("shadow_dx", "Shadow offset X", 0.04, -0.5, 0.5, decimals=3, group="Colour"),
    fparam("shadow_dy", "Shadow offset Y", 0.06, -0.5, 0.5, decimals=3, group="Colour"),
    fparam("shadow_blur", "Shadow blur", 0.06, 0.0, 0.6, decimals=3, group="Colour"),
    cparam("backing", "Contrast backing", "none", (("none", "None"), ("box", "Behind the box"),
                                                   ("lines", "Behind each line")), group="Colour"),
    Param("backing_colour", "Backing colour", COLOR, "#000000", group="Colour"),
    fparam("backing_opacity", "Backing opacity", 0.6, 0.0, 1.0, group="Colour"),
    fparam("backing_padding", "Backing padding", 0.25, 0.0, 2.0, group="Colour"),
    cparam("style", "Style", "clean", tuple(STYLE_NAMES.items()), group="Style"),
    fparam("intensity", "Style intensity", 1.0, 0.0, 2.0, group="Style"),
    fparam("readability", "Preserve readability", 0.0, 0.0, 1.0, group="Style",
           tooltip="Keeps a clean copy of the letters under the damaged layer. An artistic control, not a "
                   "guarantee of readability."),
    iparam("seed", "Style seed", 1, 0, 2**31 - 1, group="Style"),
    Param("distress", "Distress only here", MASK, {}, group="Style",
          tooltip="Optional: paint where the lettering may be damaged."),
    cparam("repeat", "Decorative repetition", "none", (("none", "None"), ("texture", "Repeated text texture"),
                                                        ("scatter", "Scattered echoes")), group="Repetition"),
    iparam("repeat_count", "Repetitions", 12, 1, 200, group="Repetition"),
    fparam("repeat_opacity", "Repetition opacity", 0.25, 0.0, 1.0, group="Repetition"),
    fparam("repeat_scale", "Repetition size", 0.5, 0.1, 3.0, group="Repetition"),
    bparam("behind", "Place behind the subject", False, group="Subject"),
    Param("subject", "Subject (SAM 2, optional)", POINTS, [], group="Subject", show_if=("behind", (True,))),
    Param("subject_mask", "Subject mask (manual)", MASK, {}, group="Subject", show_if=("behind", (True,))),
)
BOX_PARAM = {p.name: p for p in BOX_PARAMS}
_RAW_TEXT = {"text"}  # never coerced beyond str()


def style_param(style: str, name: str) -> Param:
    for p in STYLE_PARAMS.get(style, ()):
        if p.name == name:
            return p
    raise KeyError(name)


def default_style_params() -> dict:
    return {sid: {p.name: p.default for p in params} for sid, params in STYLE_PARAMS.items() if params}


def new_box(text: str = "", **values) -> dict:
    box = {p.name: copy.deepcopy(p.default) for p in BOX_PARAMS}
    box.update(uid=uuid.uuid4().hex[:10], style_params=default_style_params(), overrides=[])
    box["seed"] = secrets.randbelow(2**31 - 1)
    box["text"] = text
    box.update(values)
    return coerce_box(box)


def coerce_box(d: dict) -> dict:
    """Validate a box dict (unknown keys dropped, bad values replaced). The text is kept verbatim."""
    out = {"uid": str(d.get("uid") or uuid.uuid4().hex[:10])}
    for p in BOX_PARAMS:
        v = d.get(p.name, p.default)
        if p.name in _RAW_TEXT:
            out[p.name] = "" if v is None else str(v)   # no truncation, no normalisation
        else:
            out[p.name] = p.coerce(v)
    sp = default_style_params()
    for sid, vals in (d.get("style_params") or {}).items():
        if sid in sp and isinstance(vals, dict):
            for p in STYLE_PARAMS[sid]:
                if p.name in vals:
                    sp[sid][p.name] = p.coerce(vals[p.name])
    out["style_params"] = sp
    ovs = []
    for o in d.get("overrides") or []:
        if not isinstance(o, dict):
            continue
        scope = o.get("scope") if o.get("scope") in ("word", "line") else None
        style = o.get("style") if o.get("style") in STYLE_NAMES else None
        if scope is None or style is None:
            continue
        try:
            idx = int(o.get("index", 0))
        except (TypeError, ValueError):
            continue
        ovs.append({"scope": scope, "index": max(0, idx), "style": style,
                    "intensity": float(min(max(float(o.get("intensity", 1.0)), 0.0), 2.0))})
    out["overrides"] = ovs
    return out


def box_fonts(box: dict) -> set[str]:
    return {box.get("font", "")}


# ------------------------------------------------------------------ presets
def layout_preset(name: str, box: dict) -> dict:
    """Apply a named layout preset to a copy of ``box`` (text and style untouched)."""
    b = dict(box)
    if name == "corner":
        b.update(layout="box", x=0.2, y=0.84, w=0.34, h=0.22, size=0.028, align="left", valign="bottom", fit=True,
                 wrap=True, rotation=0.0, line_spacing=1.2)
    elif name == "statement":
        b.update(layout="box", x=0.5, y=0.5, w=0.86, h=0.44, size=0.14, align="center", valign="middle", fit=True,
                 wrap=True, rotation=0.0, line_spacing=1.0)
    elif name == "stacked":
        b.update(layout="box", x=0.5, y=0.5, w=0.36, h=0.88, size=0.11, align="center", valign="middle", fit=True,
                 wrap=True, rotation=0.0, line_spacing=0.92, tracking=-0.02)
    elif name == "sheet":
        b.update(layout="box", x=0.5, y=0.5, w=0.84, h=0.86, size=0.032, align="left", fit=True, wrap=True,
                 preserve_breaks=True, rotation=0.0, line_spacing=1.3)
    elif name == "scatter":
        b.update(layout="scatter", x=0.5, y=0.5, w=0.9, h=0.85, size=0.05, rotation=0.0)
    elif name == "texture":
        b.update(layout="box", x=0.5, y=0.5, w=0.8, h=0.3, size=0.06, align="center", valign="middle", repeat="texture",
                 repeat_count=30, repeat_opacity=0.22, repeat_scale=0.45)
    elif name == "path":
        b.update(layout="path", size=0.05, path=[[0.1, 0.75, 1, 1], [0.3, 0.45, 1, 1], [0.55, 0.35, 1, 1],
                                                 [0.8, 0.45, 1, 1], [0.92, 0.7, 1, 1]])
    return coerce_box(b)


LAYOUT_PRESETS = (("corner", "Small lyric block in a corner"), ("statement", "Large central statement"),
                  ("stacked", "Tall stacked lines"), ("sheet", "Full-page lyric sheet"),
                  ("scatter", "Scattered fragments"), ("texture", "Repeated text texture"),
                  ("path", "Words following a curved path"))

STYLE_PRESETS: dict[str, dict] = {
    "thrashed": {"font": "DejaVu Sans|Bold", "fill": "#f2efe6", "damage_colour": "#c41e1e", "outline_width": 0.0},
    "industrial": {"font": "DejaVu Sans Mono|Bold", "fill": "#111111", "tracking": -0.04, "outline_width": 0.0,
                   "damage_colour": "#b0451c", "style_params": {"industrial": {"plate": 0.9}}},
    "acid": {"font": "DejaVu Sans|Bold", "fill": "#ff3df2", "outline": "#00e5ff", "outline_width": 0.03},
    "rust": {"font": "DejaVu Serif|Bold", "fill": "#9a5a2e", "outline": "#140b06", "outline_width": 0.02},
    "xerox": {"font": "DejaVu Serif|Bold", "fill": "#111111", "outline_width": 0.0},
    "cut": {"font": "DejaVu Serif|Bold", "fill": "#111111", "outline_width": 0.0},
    "transmission": {"font": "DejaVu Sans Mono|Bold", "fill": "#e9f6ff", "outline_width": 0.0,
                     "damage_colour": "#ff2a6d"},
    "ghost": {"font": "DejaVu Serif|Book", "fill": "#1c2c6b", "outline_width": 0.0},
}


def style_preset(style: str, box: dict) -> dict:
    b = copy.deepcopy(box)
    pre = copy.deepcopy(STYLE_PRESETS.get(style, {}))
    sp = pre.pop("style_params", {})
    b.update(pre)
    for sid, vals in sp.items():
        b.setdefault("style_params", {}).setdefault(sid, {}).update(vals)
    b["style"] = style
    return coerce_box(b)


def preset_from_box(box: dict, name: str) -> dict:
    """A reusable typography preset: everything except the words and the position."""
    keep = {k: v for k, v in box.items() if k not in ("uid", "text", "x", "y", "path", "distress", "subject",
                                                       "subject_mask")}
    return {"name": name, "kind": "typography", "box": copy.deepcopy(keep)}


def apply_typography_preset(box: dict, preset: dict) -> dict:
    b = dict(box)
    b.update(copy.deepcopy(preset.get("box", {})))
    for k in ("uid", "text", "x", "y", "path", "distress", "subject", "subject_mask"):
        b[k] = box.get(k)
    return coerce_box(b)


__all__ = ["BOX_PARAMS", "BOX_PARAM", "STYLE_PARAMS", "STYLE_NAMES", "STYLE_IDS", "new_box", "coerce_box",
           "layout_preset", "LAYOUT_PRESETS", "style_preset", "STYLE_PRESETS", "preset_from_box",
           "apply_typography_preset", "BOOL"]
