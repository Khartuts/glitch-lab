# Glitch Lab - progress log

Working copy: `/home/user/glitch_lab` (Linux build host; target Windows 10/11 x64).
Dev venv: `.venv` (Python 3.13, `requirements-dev.txt`, package installed with `pip install -e . --no-deps`).

* Tests: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q`  -> **332 passed** (engine + offscreen UI)
* UI screenshots: `.venv/bin/python scripts/ui_smoke.py OUT_DIR`
* Linux package build + verification: `bash packaging/build_linux.sh` -> PASSED
  (packaged self-test 33/33, contact sheet 28/28 OK, GUI smoke test with full render)

## Status

- [x] M1: engine core, 4 effects, presets, UI shell, undo/redo, workers, export
- [x] M2: HEX corruption (BMP/PPM/JPEG), Sonify, masks, layers, projects, variation grid, favourites, batch export
- [x] M3: effects 7-20, performance work (prefix cache, Gray-Scott 5x, mosaic bleed 1.8x),
      contact sheet, self-test, PyInstaller spec, Windows/Linux build scripts, installer script, CI workflow, docs

## Not done on this host (needs Windows)

1. Run `packaging\build_windows.ps1 -FetchFFmpeg [-Installer]` on Windows 10/11 x64 with Python 3.13.
2. Confirm dist\selftest-report.txt says PASSED and the FFmpeg line is OK (FFmpeg 8.1 LGPL).
3. Manual checklist in README.md ("Manual verification checklist for a Windows build").
