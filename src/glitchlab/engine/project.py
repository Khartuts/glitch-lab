"""Versioned project files (``.glab``).

A project is a ZIP archive::

    project.json          format/version, document (layers, stacks, seeds, masks),
                          favourites, embedded presets, selection, environment
    source/working.png    the exact 8-bit sRGB RGBA working image (managed copy)
    source/original.<ext> optional byte-identical copy of the imported file
    assets/<sha1>.<ext>   auxiliary files referenced by effects (palettes, maps)
    thumbs/fav_<n>.png    favourite thumbnails

Because the working image is stored losslessly and every seed is in the
recipe, reopening a project reproduces the full-resolution result exactly in
the same software environment.  ``render_sha256`` records the hash of the
last full-resolution render so this can be verified.
"""

from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import json
import os
import tempfile
import zipfile
from dataclasses import dataclass, field

import numpy as np

from .. import __version__
from .effects.base import get_effect
from .errors import ProjectError
from .export import SourceInfo, environment_info
from .image_io import LoadedImage, apply_default_permissions, decode_png_bytes, save_png_bytes
from .model import Document
from .settings import data_dir

PROJECT_FORMAT = "glitchlab-project"
PROJECT_VERSION = 1
ASSET_PREFIX = "asset:"


@dataclass
class Favourite:
    name: str
    document: Document
    thumb: np.ndarray | None = None
    created: str = field(default_factory=lambda: _dt.datetime.now().isoformat(timespec="seconds"))


@dataclass
class Project:
    document: Document
    source: LoadedImage
    original_bytes: bytes | None = None
    original_name: str = ""
    favourites: list[Favourite] = field(default_factory=list)
    presets: list[dict] = field(default_factory=list)
    selection: dict = field(default_factory=dict)
    render_sha256: str | None = None
    path: str | None = None

    def source_info(self) -> SourceInfo:
        name = os.path.splitext(self.original_name or "image")[0]
        return SourceInfo(name=name, path="", sha256=self.source.sha256(), width=self.source.width,
                          height=self.source.height, notes=list(self.source.notes))


def _file_params(doc_dict: dict):
    """Yield (effect_dict, param_name) for every file-typed parameter."""
    for layer in doc_dict.get("layers", []):
        for fx in layer.get("stack", []):
            try:
                eff = get_effect(fx["effect"])
            except Exception:
                continue
            for prm in eff.params:
                if prm.kind == "file" and fx.get("params", {}).get(prm.name):
                    yield fx, prm.name


def _pack_assets(doc_dict: dict, assets: dict[str, bytes]) -> dict:
    d = copy.deepcopy(doc_dict)
    for fx, name in _file_params(d):
        path = fx["params"][name]
        if path.startswith(ASSET_PREFIX):
            continue
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError as exc:
            raise ProjectError(f"Referenced file is missing: {path}") from exc
        key = hashlib.sha1(data).hexdigest()[:16] + os.path.splitext(path)[1].lower()
        assets[key] = data
        fx["params"][name] = ASSET_PREFIX + key
    return d


def _unpack_assets(doc_dict: dict, folder: str) -> dict:
    d = copy.deepcopy(doc_dict)
    for fx, name in _file_params(d):
        v = fx["params"][name]
        if v.startswith(ASSET_PREFIX):
            fx["params"][name] = os.path.join(folder, v[len(ASSET_PREFIX):])
    return d


def save_project(path: str, project: Project, *, include_original: bool = True) -> str:
    assets: dict[str, bytes] = {}
    doc = _pack_assets(project.document.to_dict(), assets)
    favs = []
    thumbs = {}
    for i, fav in enumerate(project.favourites):
        entry = {"name": fav.name, "created": fav.created, "document": _pack_assets(fav.document.to_dict(), assets)}
        if fav.thumb is not None:
            tn = f"thumbs/fav_{i:03d}.png"
            thumbs[tn] = save_png_bytes(fav.thumb)
            entry["thumb"] = tn
        favs.append(entry)
    src = project.source
    orig_entry = None
    if include_original and project.original_bytes:
        ext = os.path.splitext(project.original_name)[1].lower() or ".bin"
        orig_entry = f"source/original{ext}"
    meta = {
        "format": PROJECT_FORMAT,
        "version": PROJECT_VERSION,
        "app_version": __version__,
        "saved": _dt.datetime.now().isoformat(timespec="seconds"),
        "document": doc,
        "favourites": favs,
        "presets": project.presets,
        "selection": project.selection,
        "render_sha256": project.render_sha256,
        "source": {
            "working": "source/working.png",
            "working_sha256": src.sha256(),
            "width": src.width,
            "height": src.height,
            "had_alpha": src.had_alpha,
            "format": src.format,
            "original_mode": src.original_mode,
            "frame_index": src.frame_index,
            "n_frames": src.n_frames,
            "icc_description": src.icc_description,
            "notes": src.notes,
            "original_name": project.original_name,
            "original": orig_entry,
            "original_sha256": hashlib.sha256(project.original_bytes).hexdigest() if project.original_bytes else None,
        },
        "environment": environment_info(),
    }
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".glitchlab-", suffix=".glab.tmp", dir=directory)
    os.close(fd)
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("project.json", json.dumps(meta, indent=2))
            z.writestr("source/working.png", save_png_bytes(src.rgba), compress_type=zipfile.ZIP_STORED)
            if orig_entry:
                z.writestr(orig_entry, project.original_bytes, compress_type=zipfile.ZIP_STORED)
            for name, data in assets.items():
                z.writestr(f"assets/{name}", data)
            for name, data in thumbs.items():
                z.writestr(name, data, compress_type=zipfile.ZIP_STORED)
        apply_default_permissions(tmp)
        os.replace(tmp, path)
    except Exception as exc:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        if isinstance(exc, ProjectError):
            raise
        raise ProjectError(f"Could not save project: {exc}", detail=repr(exc)) from exc
    project.path = path
    return path


def load_project(path: str) -> Project:
    try:
        z = zipfile.ZipFile(path)
    except FileNotFoundError as exc:
        raise ProjectError(f"Project not found: {path}") from exc
    except (zipfile.BadZipFile, OSError) as exc:
        raise ProjectError("This file is not a Glitch Lab project (or it is damaged).", detail=repr(exc)) from exc
    with z:
        try:
            meta = json.loads(z.read("project.json").decode("utf-8"))
        except (KeyError, ValueError) as exc:
            raise ProjectError("The project is missing its project.json description.") from exc
        if meta.get("format") != PROJECT_FORMAT:
            raise ProjectError("This file is not a Glitch Lab project.")
        version = int(meta.get("version", 0))
        if version > PROJECT_VERSION:
            raise ProjectError(f"This project was saved by a newer Glitch Lab (format {version}).")
        if version < 1:
            raise ProjectError("Unsupported project format version.")
        s = meta["source"]
        try:
            rgba = decode_png_bytes(z.read(s["working"]))
        except Exception as exc:
            raise ProjectError("The project's managed source image is missing or damaged.", detail=repr(exc)) from exc
        li = LoadedImage(rgba=rgba, path=None, format=s.get("format", "PNG"), original_mode=s.get("original_mode", "RGBA"),
                         frame_index=int(s.get("frame_index", 0)), n_frames=int(s.get("n_frames", 1)),
                         had_alpha=bool(s.get("had_alpha", False)), icc_description=s.get("icc_description"),
                         notes=list(s.get("notes", [])), original_size=(int(s["width"]), int(s["height"])))
        if s.get("working_sha256") and li.sha256() != s["working_sha256"]:
            raise ProjectError("The project's source image does not match its checksum (file damaged?).")
        original = None
        if s.get("original"):
            try:
                original = z.read(s["original"])
            except KeyError:
                original = None
        names = [n for n in z.namelist() if n.startswith("assets/")]
        asset_dir = ""
        if names:
            key = hashlib.sha1(os.path.abspath(path).encode() + str(os.path.getmtime(path)).encode()).hexdigest()[:12]
            asset_dir = str(data_dir() / "project-assets" / key)
            os.makedirs(asset_dir, exist_ok=True)
            for n in names:
                target = os.path.join(asset_dir, os.path.basename(n))
                if not os.path.exists(target):
                    with open(target, "wb") as f:
                        f.write(z.read(n))
        doc = Document.from_dict(_unpack_assets(meta["document"], asset_dir))
        favs = []
        for f in meta.get("favourites", []):
            thumb = None
            if f.get("thumb"):
                try:
                    thumb = decode_png_bytes(z.read(f["thumb"]))
                except Exception:
                    thumb = None
            favs.append(Favourite(name=f.get("name", "Favourite"), created=f.get("created", ""),
                                  document=Document.from_dict(_unpack_assets(f["document"], asset_dir)), thumb=thumb))
    return Project(document=doc, source=li, original_bytes=original, original_name=s.get("original_name", ""),
                   favourites=favs, presets=list(meta.get("presets", [])), selection=dict(meta.get("selection", {})),
                   render_sha256=meta.get("render_sha256"), path=path)


def project_from_image(li: LoadedImage, doc: Document) -> Project:
    original = None
    name = "image.png"
    if li.path:
        name = os.path.basename(li.path)
        try:
            if os.path.getsize(li.path) <= 200 * 1024 * 1024:
                with open(li.path, "rb") as f:
                    original = f.read()
        except OSError:
            original = None
    return Project(document=doc, source=li, original_bytes=original, original_name=name)
