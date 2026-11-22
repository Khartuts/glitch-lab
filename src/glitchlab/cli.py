"""Command-line interface (headless engine access).

Examples::

    glitchlab-cli --info
    glitchlab-cli --render photo.jpg --recipe recipe.json -o out.png
    glitchlab-cli --render project.glab -o out.png
    glitchlab-cli --preset "Broken Broadcast" --render photo.jpg -o out.png
    glitchlab-cli --contact-sheet sheet.png
    glitchlab-cli --selftest report.txt
    glitchlab-cli --ai-status
    glitchlab-cli --ai-install-engine
    glitchlab-cli --ai-download depth dinov2        (or: all)
    glitchlab-cli --ai-selftest ai-report.txt --ai-out results/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time


def _print(msg: str) -> None:
    try:
        print(msg, flush=True)
    except Exception:  # windowed builds have no console
        pass


def main(argv: list[str] | None = None) -> int:
    from . import __version__
    from .engine import effects  # noqa: F401
    from .engine.settings import setup_logging

    ap = argparse.ArgumentParser(prog="glitchlab-cli", description="Glitch Lab headless engine")
    ap.add_argument("--version", action="store_true", help="print version and exit")
    ap.add_argument("--info", action="store_true", help="print environment, FFmpeg and effect list")
    ap.add_argument("--render", metavar="INPUT", help="image or .glab project to render at full resolution")
    ap.add_argument("--recipe", metavar="JSON", help="recipe JSON (from 'Write JSON recipe') to apply")
    ap.add_argument("--preset", metavar="NAME", help="apply a built-in preset to the image")
    ap.add_argument("-o", "--output", metavar="FILE", help="output image path (.png/.jpg/.webp)")
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    ap.add_argument("--write-recipe", action="store_true", help="write a JSON recipe next to the output")
    ap.add_argument("--contact-sheet", metavar="FILE", help="render the effect contact sheet to FILE")
    ap.add_argument("--selftest", metavar="REPORT", nargs="?", const="-", help="run engine self-test")
    ap.add_argument("--ffmpeg", metavar="PATH", help="FFmpeg executable to use")
    ap.add_argument("--type-contact-sheet", metavar="FILE", help="render all typography styles on one sheet")
    ap.add_argument("--text-only", action="store_true", help="with --render: export only the text layers "
                    "(transparent PNG)")
    ap.add_argument("--ai-status", action="store_true", help="show the AI engine and model status")
    ap.add_argument("--ai-install-engine", action="store_true", help="download and install the AI engine")
    ap.add_argument("--ai-download", nargs="+", metavar="MODEL", help="download AI models (ids or 'all')")
    ap.add_argument("--ai-selftest", metavar="REPORT", nargs="?", const="-", help="test the AI effects with "
                    "the real models")
    ap.add_argument("--ai-image", metavar="FILE", help="image for --ai-selftest")
    ap.add_argument("--ai-out", metavar="DIR", help="save --ai-selftest results here")
    ap.add_argument("--ai-device", choices=("auto", "cpu", "cuda", "mps"), help="where AI effects run")
    args = ap.parse_args(argv)
    setup_logging(logging.WARNING, to_file=False)
    from .ai import runtime as ai_runtime

    ai_runtime.activate()
    if args.ai_device:
        from .ai import support as ai_support

        ai_support.DEVICE_PREFERENCE = args.ai_device
    if args.ai_status or args.ai_install_engine or args.ai_download or args.ai_selftest:
        return _ai(args)

    from .engine.services import EngineServices
    services = EngineServices(ffmpeg_path=args.ffmpeg)

    if args.version:
        _print(f"Glitch Lab {__version__}")
        return 0
    if args.info:
        from .engine.effects.base import REGISTRY
        from .engine.export import environment_info
        _print(json.dumps(environment_info(services), indent=2))
        _print(f"{len(REGISTRY)} effects: " + ", ".join(sorted(REGISTRY)))
        return 0
    if args.selftest:
        from .selftest import run_selftest
        ok, report = run_selftest(services)
        if args.selftest == "-":
            _print(report)
        else:
            with open(args.selftest, "w", encoding="utf-8") as f:
                f.write(report)
            _print(f"Self-test {'PASSED' if ok else 'FAILED'}; report written to {args.selftest}")
        return 0 if ok else 1
    if args.contact_sheet:
        from .engine.contact_sheet import write_contact_sheet
        path, results = write_contact_sheet(args.contact_sheet, services=services, overwrite=args.overwrite)
        failed = [r for r in results if r.status != "ok"]
        _print(f"Contact sheet written to {path} ({len(results)} effects, {len(failed)} not ok)")
        for r in failed:
            _print(f"  {r.status.upper()}: {r.name}: {r.message}")
        return 0
    if args.type_contact_sheet:
        from .engine.typography.extras import type_contact_sheet
        image = None
        if args.render:
            from .engine.image_io import load_image
            image = load_image(args.render).rgba
        path = type_contact_sheet(args.type_contact_sheet, image)
        _print(f"Typography contact sheet written to {path}")
        return 0
    if args.render:
        return _render(args, services)
    ap.print_help()
    return 0


def _progress_printer():
    state = {"t": 0.0}

    def prog(done, total, label):
        now = time.monotonic()
        if now - state["t"] > 2.0 or done >= total:
            state["t"] = now
            _print(f"  {done / 2**20:8.1f} / {total / 2**20:.1f} MB  {label}")
    return prog


def _ai(args) -> int:
    from .ai import models as ai_models
    from .ai import runtime as ai_runtime
    from .engine.errors import GlitchLabError

    try:
        if args.ai_install_engine:
            _print("Installing the AI engine…")
            ai_runtime.install(_progress_printer())
            _print("AI engine installed: " + ai_runtime.status().message)
        if args.ai_download:
            ids = list(ai_models.MODELS) if args.ai_download == ["all"] else args.ai_download
            unknown = [m for m in ids if m not in ai_models.MODELS]
            if unknown:
                _print("Unknown model(s): " + ", ".join(unknown) + ". Known: " + ", ".join(ai_models.MODELS))
                return 2
            ai_models.install(ids, _progress_printer())
            _print("Downloaded: " + ", ".join(ids))
        if args.ai_status:
            _print("AI engine: " + ai_runtime.status().message)
            for mid, spec in ai_models.MODELS.items():
                state = "installed" if ai_models.is_installed(mid) else "not downloaded"
                _print(f"  {mid:13s} {spec.name:34s} {spec.size_text:>8s}  {spec.licence:22s} {state}")
        if args.ai_selftest:
            from .ai.selftest import run_ai_selftest
            from .engine.image_io import load_image

            image = load_image(args.ai_image).rgba if args.ai_image else None
            ok, report = run_ai_selftest(image, args.ai_out)
            if args.ai_selftest == "-":
                _print(report)
            else:
                with open(args.ai_selftest, "w", encoding="utf-8") as f:
                    f.write(report)
                _print(report)
            return 0 if ok else 1
        return 0
    except GlitchLabError as exc:
        _print(f"error: {exc.message}")
        return 1


def _render(args, services) -> int:
    from .engine.errors import GlitchLabError
    from .engine.export import SourceInfo, export_document, load_recipe
    from .engine.image_io import ExportOptions, load_image
    from .engine.model import Document
    from .engine.presets import BUILTIN_PRESETS, apply_preset

    if not args.output:
        _print("--render needs -o/--output")
        return 2
    try:
        if args.render.lower().endswith(".glab"):
            from .engine.project import load_project
            proj = load_project(args.render)
            src, doc = proj.source.rgba, proj.document
            info = proj.source_info()
        else:
            li = load_image(args.render)
            src = li.rgba
            doc = Document.new(0)
            if args.recipe:
                doc, _ = load_recipe(args.recipe)
            if args.preset:
                match = [p for p in BUILTIN_PRESETS if p["name"].lower() == args.preset.lower()]
                if not match:
                    _print("Unknown preset. Available: " + ", ".join(p["name"] for p in BUILTIN_PRESETS))
                    return 2
                doc = apply_preset(doc, match[0])
            info = SourceInfo(name=os.path.splitext(os.path.basename(args.render))[0], path=os.path.abspath(args.render),
                              sha256=li.sha256(), width=li.width, height=li.height, notes=li.notes)
        ext = os.path.splitext(args.output)[1].lower()
        fmt = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG", ".webp": "WEBP"}.get(ext)
        if fmt is None:
            _print("Output must end in .png, .jpg or .webp")
            return 2
        t0 = time.perf_counter()
        if args.text_only:
            from .engine.typography.extras import export_typography_png
            if fmt != "PNG":
                _print("--text-only exports a transparent PNG; use an output ending in .png")
                return 2
            export_typography_png(src, doc, args.output, overwrite=args.overwrite)
            _print(f"Wrote typography {args.output} in {time.perf_counter() - t0:.2f}s")
            return 0
        rep = export_document(src, doc, args.output, ExportOptions(format=fmt, quality=args.quality,
                                                                    overwrite=args.overwrite),
                              source=info, write_recipe=args.write_recipe, services=services)
        _print(f"Wrote {rep.path} {rep.size[0]}x{rep.size[1]} in {time.perf_counter() - t0:.2f}s sha256={rep.render_sha256}")
        for w in rep.warnings:
            _print(f"warning: {w}")
        return 0
    except GlitchLabError as exc:
        _print(f"error: {exc.message}")
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
