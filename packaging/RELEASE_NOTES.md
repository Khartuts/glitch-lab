# Glitch Lab for Windows 10/11 (64-bit)

**Download one of these:**

* `GlitchLab-1.0.0-setup-win64.exe` — installer (Start-menu shortcut, `.glab` file association, uninstaller).
* `GlitchLab-1.0.0-win64.zip` — portable: unzip anywhere and run `GlitchLab\GlitchLab.exe`.

No Python or other software is needed. FFmpeg (used by the Sonify effect) is included.

**First start:** the program is not code-signed, so Windows SmartScreen may show
"Windows protected your PC". Click **More info → Run anyway**.

**Check your copy:** `selftest-report.txt` is the self-test of exactly these
files, run on a Windows build machine. You can repeat it yourself:
open a terminal in the `GlitchLab` folder and run `glitchlab-cli.exe --selftest report.txt`.

User guide: `docs\USER_GUIDE.md` inside the program folder (also *Help › User guide*).

## Third-party software

Includes Qt/PySide6 (LGPL-3.0), NumPy, Pillow, OpenCV and FFmpeg
(LGPL-2.1-or-later, unmodified static LGPL build from
https://github.com/BtbN/FFmpeg-Builds; FFmpeg source code:
https://ffmpeg.org/download.html). Licence texts are in the program folder
(`THIRD_PARTY_NOTICES.md`, `ffmpeg\FFMPEG_LICENSE.txt`).
