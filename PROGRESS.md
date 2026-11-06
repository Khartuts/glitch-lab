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

## Windows release (done)

* Repo: https://github.com/Khartuts/glitch-lab (public). Every push to `main` rebuilds on windows-latest
  and updates release v1.0.0 (use `[skip ci]` in the commit message for repo-only changes).
* Run 42212319268: 332 passed / 3 Linux-only skipped on Windows; packaged self-test PASSED incl. Sonify
  with bundled FFmpeg n8.1.3; installer + zip published.
* Downloads: https://github.com/Khartuts/glitch-lab/releases/latest/download/GlitchLab-1.0.0-setup-win64.exe
  and .../GlitchLab-1.0.0-win64.zip
* Build results/log: `build-status` branch.
* Not yet tested: interactive use on a real Windows 10/11 desktop; installer install/uninstall.
