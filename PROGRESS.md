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

## Version 1.1.0 (this session)

* Bugs fixed: variations panel aligned under the preview (vertical splitter); window never wider than the
  screen (min width 766 px; elided status; wrapping toolbars; screen-fitted geometry); any variation
  count 1-64; export conflicts -> Overwrite / Save as new numbered (`_01`, `_02` ...) / Cancel (also batch).
* New effects (30 total): data_injection, audio_file_bend, audio_reverb, halftone, cmyk_drift, benday,
  voronoi_sort, contour_tear, chromatic, vhs_crt; pixel_sort "between edges"; 7 new presets.
* NumPy G.711 A-law codec bit-exact with FFmpeg.
* Tests: 504 passed locally (Linux). Linux package pre-flight: self-test PASSED, contact sheet 48/48.
