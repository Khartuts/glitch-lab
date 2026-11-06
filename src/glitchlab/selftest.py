"""Headless self-test for installed/packaged builds.

Run ``GlitchLab.exe --selftest report.txt`` (or ``glitchlab-cli --selftest``)
on the target machine to check that the bundled libraries, codecs, FFmpeg
and every effect work - without opening the user interface.
"""

from __future__ import annotations

import os
import platform
import sys
import tempfile
import time
import traceback


def run_selftest(services=None) -> tuple[bool, str]:
    from . import __version__
    from .engine import effects  # noqa: F401
    from .engine import fixtures
    from .engine.effects.base import REGISTRY
    from .engine.errors import DependencyError
    from .engine.export import environment_info
    from .engine.image_io import ExportOptions, load_image, save_image, webp_supported
    from .engine.model import Document, EffectInstance
    from .engine.pipeline import image_hash, render_full
    from .engine.presets import BUILTIN_PRESETS, apply_preset
    from .engine.project import load_project, project_from_image, save_project
    from .engine.services import EngineServices

    services = services or EngineServices()
    lines: list[str] = []
    failures = 0

    def check(name, fn):
        nonlocal failures
        t0 = time.perf_counter()
        try:
            msg = fn() or ""
            lines.append(f"PASS  {name}  ({time.perf_counter() - t0:.2f}s) {msg}")
        except DependencyError as exc:
            lines.append(f"SKIP  {name}: {exc.message}")
        except Exception as exc:
            failures += 1
            lines.append(f"FAIL  {name}: {exc!r}")
            lines.append("      " + traceback.format_exc().replace("\n", "\n      "))

    lines.append(f"Glitch Lab {__version__} self-test — {platform.platform()} — Python {platform.python_version()}")
    lines.append(f"Executable: {sys.executable}  frozen={getattr(sys, 'frozen', False)}")
    for k, v in environment_info(services).items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    photo = fixtures.photo_like(160, 120)
    trans = fixtures.transparent_circle(96, 72)
    tmp = tempfile.mkdtemp(prefix="glitchlab-selftest-")

    def io_roundtrip():
        for fmt in ("PNG", "JPEG") + (("WEBP",) if webp_supported() else ()):
            p = os.path.join(tmp, "t." + fmt.lower())
            save_image(trans, p, ExportOptions(format=fmt, webp_lossless=True, overwrite=True))
            li = load_image(p)
            assert li.rgba.shape == trans.shape, fmt
        return "PNG/JPEG" + ("/WEBP" if webp_supported() else " (no WebP)")

    check("image import/export", io_roundtrip)

    for eid in sorted(REGISTRY):
        def run(eid=eid):
            inst = EffectInstance.create(eid, seed=99)
            a = render_full(photo, Document.single([inst], 1), services=services).image
            b = render_full(photo, Document.single([inst], 1), services=services).image
            assert a.shape == photo.shape and a.dtype == photo.dtype
            assert (a != photo).any(), "no visible change"
            assert (a == b).all(), "not deterministic"
            t = render_full(trans, Document.single([inst], 1), services=services).image
            assert (t[..., 3] == trans[..., 3]).all(), "alpha not preserved"
            return image_hash(a)[:12]
        check(f"effect {eid}", run)

    for p in BUILTIN_PRESETS:
        if all(fx["effect"] in REGISTRY for fx in p["stack"]):
            check(f"preset {p['name']}", lambda p=p: image_hash(render_full(photo, apply_preset(Document.new(1), p),
                                                                            services=services).image)[:12])

    def project_roundtrip():
        from .engine.image_io import LoadedImage
        li = LoadedImage(rgba=photo.copy(), path=None, format="PNG", original_mode="RGBA")
        doc = apply_preset(Document.new(5), BUILTIN_PRESETS[0])
        proj = project_from_image(li, doc)
        before = render_full(photo, doc, services=services).image
        path = os.path.join(tmp, "p.glab")
        save_project(path, proj)
        loaded = load_project(path)
        after = render_full(loaded.source.rgba, loaded.document, services=services).image
        assert (before == after).all(), "project did not reproduce the render"

    check("project save/load reproduces render", project_roundtrip)
    ok, ff = services.ffmpeg_status()
    lines.append("")
    lines.append(f"FFmpeg: {'OK' if ok else 'NOT AVAILABLE'} — {ff}")
    try:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    except Exception:
        pass
    lines.append("")
    lines.append("RESULT: " + ("PASSED" if failures == 0 else f"FAILED ({failures} failure(s))"))
    return failures == 0, "\n".join(lines)
