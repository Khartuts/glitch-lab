# Glitch Lab

A Windows desktop application for **still-image glitch art**: import a
photograph, build a nondestructive stack of real data-bending and glitch
effects, explore seeded variations, and export finished artwork.
Everything runs locally and offline. No accounts, no cloud.

* 22 effects: genuine HEX/byte corruption of BMP/PPM/JPEG files, **Sonify**
  (pixels → raw A-law audio → FFmpeg filters → pixels), pixel sorting, channel
  displacement, scanline tearing, block scrambling, feedback, bit-plane
  surgery, palette damage, compression decay, data reinterpretation,
  row/column drift, displacement maps, spectral (FFT) damage, slit-scan,
  error-diffusion trails, cellular automata, Gray–Scott reaction-diffusion,
  mosaic bleed and channel cross-modulation.
* Layers with blend modes and opacity, rectangle/brush/luminance masks.
* Variation Grid (4/9/16 seeded variations, locks, favourites, batch export).
* Projects (`.glab`) that reproduce the full-resolution result exactly.

Beginners: start with **docs/USER_GUIDE.md**.
Feature status: **docs/FEATURES.md**. Technical notes: **docs/TECHNICAL.md**.

## Requirements

* Windows 10/11 64-bit (the code also runs on Linux/macOS for development).
* For development: **Python 3.13 (64-bit)**. Dependencies are pinned in
  `requirements.txt` / `requirements-dev.txt`.
* FFmpeg for the Sonify effect (bundled in the Windows package; see
  docs/FFMPEG.md). All other effects work without it.

## Development setup (Windows, PowerShell)

```
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pip install --no-deps -e .
```

Launch the app (development):

```
.venv\Scripts\python -m glitchlab
```

Run the tests (engine + offscreen UI):

```
$env:QT_QPA_PLATFORM="offscreen"; .venv\Scripts\python -m pytest -q
```

On Linux/macOS use `.venv/bin/python` and `QT_QPA_PLATFORM=offscreen`.

## Command line (headless engine)

```
python -m glitchlab.cli --info
python -m glitchlab.cli --render photo.jpg --preset "Broken Broadcast" -o out.png --write-recipe
python -m glitchlab.cli --render project.glab -o out.png
python -m glitchlab.cli --contact-sheet sheet.png
python -m glitchlab.cli --selftest report.txt
```

## Building the Windows package

On a Windows 10/11 x64 machine with Python 3.13 (64-bit) installed:

```
powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1 -FetchFFmpeg
```

This creates `.venv-win`, installs pinned dependencies, runs the tests,
downloads and SHA-256-verifies an LGPL FFmpeg build, runs PyInstaller, then
**tests the packaged application separately** (`glitchlab-cli.exe --selftest`,
`--contact-sheet`, and `GlitchLab.exe --smoke-test`) and writes
`dist\GlitchLab-1.0.0-win64.zip`. Add `-Installer` to build an Inno Setup
installer (`winget install JRSoftware.InnoSetup` first). The resulting
`dist\GlitchLab\GlitchLab.exe` runs without a separate Python installation.

### Building without a Windows PC (GitHub Actions)

`.github/workflows/windows-build.yml` runs the same script on a GitHub-hosted
Windows runner: full test suite (Sonify tested against the bundled FFmpeg),
PyInstaller build, packaged self-test / contact sheet / GUI smoke test,
installer. Pushing a tag such as `v1.0.0` publishes a **GitHub Release** with
`GlitchLab-1.0.0-setup-win64.exe`, `GlitchLab-1.0.0-win64.zip` and the
packaged self-test report. Download links then have the form
`https://github.com/<owner>/<repo>/releases/latest/download/GlitchLab-1.0.0-setup-win64.exe`.
The build can also be started by hand from the repository's *Actions* tab
(*windows-build → Run workflow*); that produces a downloadable workflow artifact.

### Manual verification checklist for a Windows build

1. Copy `dist\GlitchLab` (or the zip) to a machine **without Python**.
2. `glitchlab-cli.exe --selftest report.txt` → last line `RESULT: PASSED`;
   check the `FFmpeg:` line says OK.
3. `GlitchLab.exe --smoke-test smoke.png` → exit code 0, screenshot written.
4. Start `GlitchLab.exe`, import a JPEG with EXIF rotation, add Sonify and
   HEX Corruption, render (F5), export PNG and JPEG, save and reopen a project,
   generate 9 variations, and export two of them.

## Project layout

```
src/glitchlab/engine/     headless engine (no Qt): I/O, model, pipeline, effects, projects
src/glitchlab/engine/effects/   effect implementations
src/glitchlab/ui/         PySide6 user interface
src/glitchlab/cli.py      command-line tool;  selftest.py  packaged-build self-test
tests/                    pytest suite (engine + offscreen UI)
packaging/                PyInstaller spec, Windows/Linux build scripts, installer, FFmpeg fetcher
scripts/                  icon generator, UI screenshot script
docs/                     user guide, feature checklist, FFmpeg and technical notes
```
