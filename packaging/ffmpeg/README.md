# FFmpeg for the Windows package

Place **ffmpeg.exe** (64-bit, static build) and its licence text in this folder
before running the build. The PyInstaller spec copies every file here into
`dist/GlitchLab/ffmpeg/`, where Glitch Lab finds it automatically.

Recommended: an **LGPL** build, e.g. `ffmpeg-n7.1-latest-win64-lgpl-7.1.zip`
from https://github.com/BtbN/FFmpeg-Builds/releases (download, verify the
SHA-256 against `checksums.sha256` from the same release, extract
`bin/ffmpeg.exe` and `LICENSE.txt` here). `packaging/fetch_ffmpeg.ps1` does
exactly this.

Required filters: acrusher, volume, asoftclip, bass, treble, highpass,
lowpass, bandpass, aecho, aphaser, flanger, tremolo (all part of FFmpeg's
LGPL core). Check with `ffmpeg -hide_banner -filters`, or run
`glitchlab-cli --info` after building.

See docs/FFMPEG.md for licensing obligations.
