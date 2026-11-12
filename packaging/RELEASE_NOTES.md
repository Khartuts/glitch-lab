# Glitch Lab for Windows 10/11 (64-bit)

**Download one of these** (the names without a version number always point to the newest release):

* `GlitchLab-setup-win64.exe` — installer (Start-menu shortcut, `.glab` file association, uninstaller).
* `GlitchLab-win64.zip` — portable: unzip anywhere and run `GlitchLab\GlitchLab.exe`.

The same files are also attached with the version in their name.
No Python or other software is needed. FFmpeg (used by Sonify and for decoding MP3/FLAC/OGG audio files) is included.

**First start:** the program is not code-signed, so Windows SmartScreen may show
"Windows protected your PC". Click **More info → Run anyway**.

## What's new in 1.1

* Fixed: the Variations panel now sits directly under the preview, aligned with it.
* Fixed: the window no longer grows wider than the screen (long messages, small or scaled screens).
* Any number of variations (1–64).
* Saving to an existing file name asks: overwrite, or save as a new numbered file (`name_01`, `name_02` …).
* New databending: **Text & File Injection** (type text or load any file - txt, wav, mp3, anything - and inject its
  bytes into BMP pixels, the BMP colour table, PPM, JPEG scan data or JPEG quantisation tables),
  **Audio File Bending** (bend the pixels with your own audio file) and **Audio Reverb Smearing**
  (cathedral / hall / room / your own impulse response, reverse reverb).
* New print effects: **Halftone Screen**, **CMYK Halftone Drift** (moiré, misregistration, plate slicing) and
  **Ben-Day Dots**.
* New: **Voronoi Shatter Sorting**, **Topographic Contour Tearing**, **VHS & CRT**, **Chromatic Aberration (lens)**,
  and pixel sorting "between edges".
* Seven new presets: Pop Art Misprint, Comic Moiré, VHS Night, Stained Glass Meltdown, Cathedral Ghosts,
  Lyric Injection, Tectonic Portrait.

**Check your copy:** `selftest-report.txt` is the self-test of exactly these files, run on a Windows build
machine. Repeat it yourself: open a terminal in the `GlitchLab` folder and run
`glitchlab-cli.exe --selftest report.txt`.

User guide: `docs\USER_GUIDE.md` inside the program folder (also *Help › User guide*).

## Third-party software

Includes Qt/PySide6 (LGPL-3.0), NumPy, Pillow, OpenCV and FFmpeg
(LGPL-2.1-or-later, unmodified static LGPL build from
https://github.com/BtbN/FFmpeg-Builds; FFmpeg source code:
https://ffmpeg.org/download.html). Licence texts are in the program folder
(`THIRD_PARTY_NOTICES.md`, `ffmpeg\FFMPEG_LICENSE.txt`).
