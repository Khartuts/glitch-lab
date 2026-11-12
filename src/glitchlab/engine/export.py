"""Full-resolution export, filename patterns, JSON recipes and batch export."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import re
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from .. import __version__
from .cancel import CancelToken
from .errors import ExportError
from .image_io import ExportOptions, save_image
from .model import Document
from .pipeline import RenderCache, image_hash, render_full
from .services import EngineServices, default_services

RECIPE_FORMAT = "glitchlab-recipe"
RECIPE_VERSION = 1
PATTERN_TOKENS = ("name", "seed", "preset", "index", "date", "w", "h", "fmt")


def render_signature(doc: Document) -> str:
    """Hash of everything that affects rendered pixels (not UI state)."""
    d = {
        "g": doc.global_seed,
        "a": doc.effective_alpha_mode(),
        "l": [{
            "v": layer.visible, "o": layer.opacity, "b": layer.blend, "m": layer.mask.to_dict(),
            "s": [e.cache_signature() for e in layer.stack if e.enabled and e.mix > 0],
        } for layer in doc.layers],
    }
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()


def environment_info(services: EngineServices | None = None) -> dict:
    import cv2
    import PIL

    info = {
        "glitchlab": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "opencv": cv2.__version__,
    }
    if services is not None:
        ok, text = services.ffmpeg_status()
        info["ffmpeg"] = text if ok else f"unavailable: {text}"
    return info


def sanitize_filename(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", name).strip().strip(".")
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    if name.upper() in reserved:
        name = "_" + name
    return name[:180] or "glitch"


def format_filename(pattern: str, *, name: str, seed: int, preset: str = "", index: int = 0,
                    width: int = 0, height: int = 0, fmt: str = "png", when: _dt.datetime | None = None) -> str:
    """Expand ``{name}``, ``{seed}``, ``{preset}``, ``{index}``, ``{date}``, ``{w}``, ``{h}``, ``{fmt}``."""
    when = when or _dt.datetime.now()
    values = {
        "name": name, "seed": str(seed), "preset": re.sub(r"\s+", "-", preset.strip()) or "custom",
        "index": f"{index:02d}", "date": when.strftime("%Y%m%d-%H%M%S"), "w": str(width), "h": str(height),
        "fmt": fmt.lower(),
    }

    def repl(m):
        key = m.group(1)
        if key not in values:
            raise ExportError(f"Unknown filename token {{{key}}}. Use: " + ", ".join("{" + t + "}" for t in PATTERN_TOKENS))
        return values[key]

    out = re.sub(r"\{(\w+)\}", repl, pattern or "{name}")
    return sanitize_filename(out)


@dataclass
class SourceInfo:
    name: str = "image"
    path: str = ""
    sha256: str = ""
    width: int = 0
    height: int = 0
    notes: list[str] = field(default_factory=list)


def recipe_payload(doc: Document, source: SourceInfo, *, opts: ExportOptions | None = None,
                   output_size: tuple[int, int] | None = None, result_sha256: str | None = None,
                   services: EngineServices | None = None) -> dict:
    payload = {
        "format": RECIPE_FORMAT,
        "version": RECIPE_VERSION,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "source": asdict(source),
        "document": doc.to_dict(),
        "environment": environment_info(services),
    }
    if opts is not None:
        o = asdict(opts)
        o.pop("overwrite", None)
        payload["output"] = o
    if output_size is not None:
        payload["output_size"] = list(output_size)
    if result_sha256:
        payload["render_sha256"] = result_sha256
    return payload


def load_recipe(path: str) -> tuple[Document, dict]:
    from .errors import ProjectError

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        raise ProjectError(f"Could not read recipe: {exc}") from exc
    if data.get("format") != RECIPE_FORMAT:
        raise ProjectError("This JSON file is not a Glitch Lab recipe.")
    if int(data.get("version", 1)) > RECIPE_VERSION:
        raise ProjectError("This recipe was written by a newer Glitch Lab.")
    return Document.from_dict(data["document"]), data


def output_dimensions(opts: ExportOptions, render_size: tuple[int, int]) -> tuple[int, int]:
    w, h = render_size
    if opts.width and opts.height:
        return int(opts.width), int(opts.height)
    return w, h


@dataclass
class ExportReport:
    path: str
    recipe_path: str | None
    size: tuple[int, int]
    render_sha256: str
    warnings: list[str]


def write_json_atomic(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def export_rendered(image: np.ndarray, doc: Document, path: str, opts: ExportOptions, *,
                    source: SourceInfo, write_recipe: bool = False, services: EngineServices | None = None,
                    warnings: list[str] | None = None) -> ExportReport:
    """Save an already rendered full-resolution image (+ optional recipe)."""
    recipe_path = None
    if write_recipe:
        recipe_path = os.path.splitext(path)[0] + ".json"
        if os.path.exists(recipe_path) and not opts.overwrite:
            raise ExportError(f"'{os.path.basename(recipe_path)}' already exists.", detail="exists")
    save_image(image, path, opts, source_path=source.path or None)
    sha = image_hash(image)
    size = output_dimensions(opts, (image.shape[1], image.shape[0]))
    if recipe_path:
        write_json_atomic(recipe_path, recipe_payload(doc, source, opts=opts, output_size=size,
                                                      result_sha256=sha, services=services))
    return ExportReport(path, recipe_path, size, sha, list(warnings or []))


def export_document(source_img: np.ndarray, doc: Document, path: str, opts: ExportOptions, *,
                    source: SourceInfo, write_recipe: bool = False, services: EngineServices | None = None,
                    cancel: CancelToken | None = None, progress=None,
                    cache: RenderCache | None = None) -> ExportReport:
    """Render at full resolution (errors abort) and save."""
    services = services or default_services()
    res = render_full(source_img, doc, services=services, cancel=cancel or CancelToken(), progress=progress,
                      cache=cache, cache_prefix="full")
    return export_rendered(res.image, doc, path, opts, source=source, write_recipe=write_recipe,
                           services=services, warnings=res.warnings)


_NUMBER_SUFFIX = re.compile(r"^(.*)_(\d{2,4})$")


def sidecar_path(path: str) -> str:
    return os.path.splitext(path)[0] + ".json"


def path_taken(path: str, *, with_recipe: bool = False, reserved: set | None = None) -> bool:
    if reserved and os.path.normcase(os.path.abspath(path)) in reserved:
        return True
    return os.path.exists(path) or (with_recipe and os.path.exists(sidecar_path(path)))


def next_free_path(path: str, *, with_recipe: bool = False, reserved: set | None = None, limit: int = 9999) -> str:
    """First free numbered variant of ``path``: ``name_01.png``, ``name_02.png`` ...

    A name that already ends in ``_NN`` continues counting from there, so
    saving ``art_03.png`` again proposes ``art_04.png``.  When
    ``with_recipe`` is set, the JSON sidecar name must be free too.
    """
    folder, fname = os.path.split(path)
    stem, ext = os.path.splitext(fname)
    m = _NUMBER_SUFFIX.match(stem)
    base, start = (m.group(1), int(m.group(2)) + 1) if m else (stem, 1)
    for n in range(start, limit + 1):
        cand = os.path.join(folder, f"{base}_{n:02d}{ext}")
        if not path_taken(cand, with_recipe=with_recipe, reserved=reserved):
            return cand
    raise ExportError(f"No free file name left for '{fname}'.")


def plan_batch_paths(docs: list[Document], folder: str, pattern: str, opts: ExportOptions, *,
                     source: SourceInfo) -> list[str]:
    """Output paths for a batch (duplicates inside the batch get an index)."""
    paths: list[str] = []
    for i, doc in enumerate(docs):
        fn = format_filename(pattern, name=source.name, seed=doc.global_seed, preset=doc.preset_name,
                             index=i + 1, width=source.width, height=source.height,
                             fmt=opts.format) + opts.extension()
        p = os.path.join(folder, fn)
        if p in paths:
            base, ext = os.path.splitext(p)
            p = f"{base}_{i + 1:02d}{ext}"
        paths.append(p)
    return paths


CONFLICT_POLICIES = ("error", "overwrite", "number")


def resolve_batch_paths(paths: list[str], policy: str, *, with_recipe: bool) -> list[str]:
    """Apply a conflict policy: ``error`` (refuse), ``overwrite`` or ``number``
    (existing names get the next free ``_NN`` suffix)."""
    if policy not in CONFLICT_POLICIES:
        raise ValueError(policy)
    existing = [p for p in paths if path_taken(p, with_recipe=with_recipe)]
    if not existing or policy == "overwrite":
        return list(paths)
    if policy == "error":
        raise ExportError(f"{len(existing)} file(s) already exist, e.g. '{os.path.basename(existing[0])}'.",
                          detail="exists")
    out: list[str] = []
    reserved: set[str] = {os.path.normcase(os.path.abspath(p)) for p in paths}
    for p in paths:
        if path_taken(p, with_recipe=with_recipe):
            p = next_free_path(p, with_recipe=with_recipe, reserved=reserved)
            reserved.add(os.path.normcase(os.path.abspath(p)))
        out.append(p)
    return out


def batch_export(source_img: np.ndarray, docs: list[Document], folder: str, pattern: str,
                 opts: ExportOptions, *, source: SourceInfo, write_recipe: bool = True,
                 services: EngineServices | None = None, cancel: CancelToken | None = None,
                 progress=None, on_conflict: str | None = None) -> list[ExportReport]:
    """Export several recipes (e.g. selected variations) at full resolution.

    ``on_conflict``: ``error`` (default unless ``opts.overwrite``), ``overwrite``
    or ``number`` (save existing names as ``name_01``, ``name_02`` ...).
    """
    cancel = cancel or CancelToken()
    policy = on_conflict or ("overwrite" if opts.overwrite else "error")
    paths = resolve_batch_paths(plan_batch_paths(docs, folder, pattern, opts, source=source), policy,
                                with_recipe=write_recipe)
    opts = replace(opts, overwrite=(policy == "overwrite"))
    reports = []
    for i, (doc, p) in enumerate(zip(docs, paths)):
        cancel.check()

        def sub(f, msg="", _i=i):
            if progress:
                progress((_i + f) / len(docs), f"Variation {_i + 1}/{len(docs)}: {msg}")

        reports.append(export_document(source_img, doc, p, opts, source=source, write_recipe=write_recipe,
                                       services=services, cancel=cancel, progress=sub))
    return reports
