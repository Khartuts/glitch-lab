# FFmpeg: bundling, verification and licence obligations

Glitch Lab uses FFmpeg **only** for the *Sonify* effect, and only as a
separate executable started with an argument list (never a shell string),
with a timeout, killed on cancel, and with temporary files removed
afterwards. No FFmpeg library is linked into Glitch Lab. Pixel-derived audio
is never played.

## Exact processing contract

```
ffmpeg -hide_banner -nostdin -loglevel error
       -f alaw -ar <rate> -ac 1 -i pixels_in.alaw      # raw 8-bit A-law, mono
       -af "<filter chain>"
       -c:a pcm_alaw -f alaw -ar <rate> -ac 1 -y pixels_out.alaw
```

`<rate>` is 8000/11025/22050/44100/48000 Hz (default 44100; optionally scaled
with the preview size). Filters used: `volume`, `bass`, `treble`, `highpass`,
`lowpass`, `bandpass`, `acrusher`, `aecho`, `aphaser`, `flanger`, `tremolo`,
`asoftclip`.

On first use Glitch Lab (1) runs `ffmpeg -version`, (2) parses
`ffmpeg -filters` and checks that every filter in the chain exists, and
(3) performs a functional A-law round trip with `anull`, which must be
byte-identical. Any failure is shown as an actionable message; *Help › Check
FFmpeg* shows the full report and `glitchlab-cli --info` prints it.

Lookup order: path set in *Edit › Preferences* (or `GLITCHLAB_FFMPEG`
environment variable) → `ffmpeg\ffmpeg.exe` next to `GlitchLab.exe` →
`PATH`.

Tested in this release with **FFmpeg 6.1.1** (Linux build host). The Windows
package recipe downloads an FFmpeg **8.1 LGPL** build; the arguments above are
standard since FFmpeg 4.x, and the runtime round-trip check guards against an
incompatible build. *The 8.1 build itself was not executed on the build host
used for this release* — run `glitchlab-cli.exe --selftest` after packaging.

## Which build to bundle

Use a 64-bit **static LGPL** build, so `ffmpeg.exe` has no extra DLLs and no
GPL-only components. Recommended source: BtbN FFmpeg-Builds,
asset `ffmpeg-n8.1-latest-win64-lgpl-8.1.zip`.

```
powershell -ExecutionPolicy Bypass -File packaging\fetch_ffmpeg.ps1
```

The script downloads the archive and the release's `checksums.sha256`,
verifies the SHA-256, and places `ffmpeg.exe`, `FFMPEG_LICENSE.txt` and
`FFMPEG_SOURCE.txt` (download URL + hash) in `packaging\ffmpeg\`. The
PyInstaller spec copies that folder to `dist\GlitchLab\ffmpeg\`. For
reproducible builds pass `-Release autobuild-YYYY-MM-DD-HH-MM` (a dated tag)
instead of the moving `latest` release and record the printed hash.

Manual alternative: download any trustworthy 64-bit FFmpeg ≥ 4.4 build,
put `ffmpeg.exe` and its licence file into `packaging\ffmpeg\`, then build.

## Licence obligations when you distribute the package

FFmpeg LGPL builds are licensed under LGPL-2.1-or-later. When you distribute
`ffmpeg.exe` with Glitch Lab you must:

1. Include the FFmpeg licence text (`ffmpeg\FFMPEG_LICENSE.txt`).
2. State that the software uses FFmpeg under the LGPL (see
   `THIRD_PARTY_NOTICES.md`).
3. Make the **corresponding source code** of the exact FFmpeg build available
   (e.g. link to the tagged FFmpeg source and the build scripts —
   https://ffmpeg.org/download.html and https://github.com/BtbN/FFmpeg-Builds —
   or host the source archive yourself, ideally next to your download).
4. Not prevent users from replacing `ffmpeg.exe` (Glitch Lab simply runs
   whichever executable is in the `ffmpeg` folder or configured path).

If you choose a **GPL** FFmpeg build instead, the GPL applies to that
executable; providing its complete corresponding source is then mandatory.
This guidance is not legal advice.
