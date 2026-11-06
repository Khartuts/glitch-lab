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

## Windows build & download link (next step)

The repo is committed locally (`main`, tag `v1.0.0`). No GitHub account is linked to the session yet.
Once the user connects GitHub and names an empty repository `owner/repo`:
1. attach it with add_repo (push access), `git remote add origin ...`, push `main` and tag `v1.0.0`;
2. `.github/workflows/windows-build.yml` builds on windows-latest, runs all tests (Sonify against the
   bundled FFmpeg 8.1 LGPL), verifies the packaged app, builds the installer, and publishes the release;
3. download links: `https://github.com/<owner>/<repo>/releases/latest/download/GlitchLab-1.0.0-setup-win64.exe`
   and `.../GlitchLab-1.0.0-win64.zip`;
4. if the Windows run fails, fix and re-tag (or re-run via workflow_dispatch).

Fallback without GitHub: on a Windows PC with Python 3.13 x64, run `packaging\build_windows.bat -Installer`.
