# Third-party notices

Glitch Lab's own code is MIT licensed (see `LICENSE`). The packaged
application redistributes the following components. Their licence texts are
included in the package (PyInstaller copies the `*.dist-info` licence files of
each Python package; FFmpeg's licence is in `ffmpeg/FFMPEG_LICENSE.txt`).

| Component | Version | Licence | Notes |
|---|---|---|---|
| Python runtime | 3.13 | PSF License | embedded by PyInstaller |
| PySide6 / Shiboken6 / Qt 6 | 6.10.3 | LGPL-3.0 (also GPL/commercial) | Qt DLLs ship as separate, replaceable files in the onedir bundle, satisfying LGPL relinking. Source: https://download.qt.io/official_releases/QtForPython/ and https://code.qt.io |
| NumPy | 2.3.5 | BSD-3-Clause | bundles OpenBLAS (BSD-3-Clause) |
| Pillow | 12.3.0 | MIT-CMU (HPND) | bundles libjpeg-turbo (IJG/BSD), libpng (libpng), zlib, libwebp (BSD), libtiff, LittleCMS (MIT), FreeType (FTL) |
| OpenCV (opencv-python-headless) | 4.13.0.92 | Apache-2.0 | bundles its own third-party libs; see `cv2/LICENSE*.txt` in the package |
| FFmpeg (optional, bundled `ffmpeg.exe`) | 8.x LGPL build | LGPL-2.1-or-later | invoked as a separate program; never linked. See `docs/FFMPEG.md` for the source-code offer and obligations |
| PyInstaller bootloader | 6.19.0 | GPL-2.0 with bootloader exception | the exception allows distributing the bootloader with any program |

If you distribute a build that contains a **GPL** FFmpeg build instead of an
LGPL one, the combined distribution terms of that FFmpeg build apply to the
`ffmpeg.exe` you ship (Glitch Lab only executes it as a separate process).
