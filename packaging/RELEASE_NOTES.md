# Glitch Lab for Windows 10/11 (64-bit)

**Download one of these** (the names without a version number always point to the newest release):

* `GlitchLab-setup-win64.exe` — installer (Start-menu shortcut, `.glab` file association, uninstaller).
* `GlitchLab-win64.zip` — portable: unzip anywhere and run `GlitchLab\GlitchLab.exe`.

The same files are also attached with the version in their name.
No Python or other software is needed. FFmpeg (used by Sonify and for decoding MP3/FLAC/OGG audio files) is included.

**First start:** the program is not code-signed, so Windows SmartScreen may show
"Windows protected your PC". Click **More info → Run anyway**.

## What's new in 1.4 — Acid Flow and lyrics typography

* **Acid Flow**: the photo liquefies into psychedelic flowing material — Contour Flow, Corrosive Melt,
  Vortex Pools and the intense Liquid Acid mode; place and drag vortex centres, draw flow strokes, paint the
  liquid area and protected areas, freeze a flow state; chromatic separation, colour ribbons, iridescence,
  caustic highlights, liquid chrome. Presets: Liquid Acid, Mercury Skin, Toxic Ink, Slow Dissolve, Oil Spill.
* **Lyrics / glitch typography**: text layers with editable text boxes (exact wording, punctuation, Unicode and
  line breaks are kept), installed / bundled / imported fonts with Unicode fallback, wrapping, fit-to-box,
  overflow warnings, guides, rotation and perspective, seven layout presets and eight distinct styles:
  Thrashed, Industrial, Acid / Liquid Type, Rust / Oxidised, Xerox, Cut-and-Paste, Transmission Failure and
  Ghost Carbon. Transparent typography export, typography presets, text behind the subject.

## What's new in 1.3 — local AI effects

* New category **AI Experimental** with eight effects that run on this computer:
  **Semantic Faultlines** (SAM 2: objects you click come loose from their outlines),
  **Depth Mutiny** (Depth Anything V2: depth slabs fold, invert and collide),
  **Feature Contagion** (DINOv2: similar-looking regions infect each other, optional donor image),
  **Perception Breakdown** (glitches where the model disagrees with itself),
  **Latent Circuit Bending** (damage inside the Stable Diffusion VAE's latent space),
  **Hallucination Scars** (SD 1.5 inpainting: tears reveal a material you describe),
  **Counterfactual Residue** (two contradictory generations haunt the photo) and
  **Recursive Misremembering** (the image is remembered again and again).
* The AI engine (about 165 MB) and each model are **downloaded only when you ask** (AI menu), pinned and
  checksum-verified; the installer itself stays small. Everything runs offline afterwards, on the processor
  (AMD/Intel graphics) or an NVIDIA card where supported.
* Twelve AI presets, object picking on the preview, per-effect painted masks, progress and *Cancel AI*,
  cached results, and AI model revisions saved in projects.

## What's new in 1.2

* New category **Abstract & Style** with seven effects:
  **Kaleidoscope** (radial wedges with any number of segments, hexagon or square mirror tiles, mirror halves),
  **Psychedelic Pop** (poster colour bands, echo outlines, swirl),
  **Dark Deco** (black-and-gold Art Deco: sunburst, fan, arches, chevrons, stepped panels, frames),
  **Luminous Gradient** (glowing gradient light fields, in the spirit of Koichi Sato's posters),
  **Y2K Clear Plastic** (glossy translucent jelly plastic with refraction, highlights and bubbles),
  **Corporate Futurism** (cold grade, grid floor, HUD brackets, glass panels) and
  **Hacker Terminal** (ASCII art, code rain or a hex dump of the image or your own text).
* Fourteen new presets: Kaleidoscope Bloom, Acid Kaleidoscope, Psychedelic Pop Ad, Candy Pop Ad, Dark Deco,
  Emerald Deco Fan, Luminous Poster, Dawn Gradient Poster, Y2K Clear Plastic, Tangerine Jelly,
  Moody Corporate Futurism, Hackers, Mainframe ASCII, Hex Dump Portrait.

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
