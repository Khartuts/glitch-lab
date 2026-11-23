# Glitch Lab - progress log

Working copy: `/home/user/glitch_lab` (Linux build host; target Windows 10/11 x64).
Dev venv: `.venv` (Python 3.13, `requirements-dev.txt`, package installed with `pip install -e . --no-deps`).

* Tests: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q`  -> **335 passed** (engine + offscreen UI)
* UI screenshots: `.venv/bin/python scripts/ui_smoke.py OUT_DIR`
* Linux package build + verification: `bash packaging/build_linux.sh` -> PASSED
  (packaged self-test 33/33, contact sheet 28/28 OK, GUI smoke test with full render)

## Status

- [x] M1: engine core, 4 effects, presets, UI shell, undo/redo, workers, export
- [x] M2: HEX corruption (BMP/PPM/JPEG), Sonify, masks, layers, projects, variation grid, favourites, batch export
- [x] M3: effects 7-20, performance work (prefix cache, Gray-Scott 5x, mosaic bleed 1.8x),
      contact sheet, self-test, PyInstaller spec, Windows/Linux build scripts, installer script, CI workflow, docs

## Windows release

* Repo: https://github.com/Khartuts/glitch-lab (public). Every push to `main` rebuilds on windows-latest and
  creates/updates the release `v<version from pyproject.toml>` (use `[skip ci]` for repo-only changes).
* Stable download links (always the newest release):
  https://github.com/Khartuts/glitch-lab/releases/latest/download/GlitchLab-setup-win64.exe
  https://github.com/Khartuts/glitch-lab/releases/latest/download/GlitchLab-win64.zip
* Build results/log: `build-status` branch.

## Version 1.4.0 (Acid Flow + typography)

* engine/flow.py (artistic flow advection), effects/acid.py (Acid Flow) + 5 presets (seeds 9201-9601).
* engine/typography/: fonts (bundled DejaVu/STIX, system, imported, fallback), model (box/style schemas,
  layout and style presets), layout (wrap/justify/fit/overflow/path), styles (8 recipes), render (full-res
  layout scaled to render, pixel-aligned canvases, premultiplied compositing), extras (presets, export,
  contact sheet). Layers: kind "text" with boxes; image layers can process the layers below.
* UI: Text tab (text_panel.py), canvas text tool (move/resize/rotate, guides, snapping), point dragging,
  stroke-only masks, freeze snapshots, text-only preview, typography export, missing-font substitution.
* Tests: test_acid_flow.py, test_typography.py, test_ui_typography.py. Linux: 746 passed; AI environment with
  real models: 132 passed. Windows run 45124476674: 743 passed / 5 skipped, packaged self-test and AI self-test
  PASSED, release v1.4.0 published (installer 113 MB, zip 173 MB). The first 1.4.0 run failed only because the
  Text panel was too wide for the Windows fallback font (fixed: shorter labels, wrapping).

## Version 1.3.0 (local AI)

* `glitchlab/ai/`: runtime (in-app engine download from a pinned wheel manifest made by
  `packaging/ai_runtime/make_manifest.py`), models (pinned HF revisions + checksums), download, jobs
  (worker thread, approvals, memory/disk cache), support (loaders, preprocessing from model configs),
  analysis (SAM 2, depth, DINOv2, disagreement), generate (VAE, SD 1.5 inpaint/img2img/recursion), selftest.
* Effects: engine/effects/ai_vision.py (4) and ai_generative.py (4); 12 AI presets (seeds 8001-9101).
* Pipeline: scale-independent `ctx.input_key`, `ctx.full_input()`, `ctx.interactive`, failure codes.
* UI: object points + per-effect masks on the canvas, AI box (models, estimate, Generate), AI menu,
  models dialog, AI progress bar with cancel, AI device preference, project model revisions.
* CLI: --ai-status, --ai-install-engine, --ai-download, --ai-selftest (used by Windows CI).

## Version 1.2.0

* New category "Abstract & Style" (37 effects total): kaleidoscope (geometry.py), psychedelic_pop, dark_deco,
  luminous_gradient, corporate_futurism (stylize.py), clear_plastic (material.py), hacker_terminal
  (terminal.py; Pillow default-font glyph atlas, ASCII / code rain / hex dump).
* 14 presets (seeds 6001-7301): Kaleidoscope Bloom, Acid Kaleidoscope, Psychedelic Pop Ad, Candy Pop Ad,
  Dark Deco, Emerald Deco Fan, Luminous Poster, Dawn Gradient Poster, Y2K Clear Plastic, Tangerine Jelly,
  Moody Corporate Futurism, Hackers, Mainframe ASCII, Hex Dump Portrait.
* tests/test_v12_effects.py (symmetry, glyph atlas, hex dump, palettes, determinism, presets).
* Tests: 612 passed locally (Linux); Windows run 44801659803: 609 passed / 3 Linux-only skipped, packaged
  self-test PASSED (37 effects + presets, bundled FFmpeg n8.1.3), contact sheet 69/69; release v1.2.0
  published (installer 114 MB, portable zip 174 MB) and the stable `latest/download` links point to it.

## Version 1.1.0

* Bugs fixed: variations panel aligned under the preview (vertical splitter); window never wider than the
  screen (min width 766 px; elided status; wrapping toolbars; screen-fitted geometry); any variation
  count 1-64; export conflicts -> Overwrite / Save as new numbered (`_01`, `_02` ...) / Cancel (also batch).
* New effects (30 total): data_injection, audio_file_bend, audio_reverb, halftone, cmyk_drift, benday,
  voronoi_sort, contour_tear, chromatic, vhs_crt; pixel_sort "between edges"; 7 new presets.
* NumPy G.711 A-law codec bit-exact with FFmpeg.
* Tests: 505 passed locally (Linux); Windows run 43315964432: 502 passed / 3 Linux-only skipped,
  packaged self-test PASSED (30 effects, Sonify with bundled FFmpeg n8.1.3), contact sheet 48/48,
  installer built; release v1.1.0 published with stable names.
* Windows-only layout bug found by the runner's layout report: backslash folder paths cannot wrap, so the
  browser label widened the window -> now elided in the middle. Panel minimums scale with text width.
