# Glitch Lab — User Guide

Glitch Lab turns photographs into glitch art. You never edit your photo
directly: you build a **recipe** (a stack of effects with settings and random
seeds) that is always re-applied to the untouched original. Your source file
is never overwritten.

## 1. The window

| Area | What it does |
|---|---|
| **Centre** | Large preview. Wheel = zoom, middle-drag or Space+drag = pan, double-click = fit. The label at the top left tells you whether you see a **reduced preview** (orange) or the **full resolution** render (cyan). |
| **Left: Library** | *Images*: thumbnails of the current folder (double-click to import). *Effects*: all effects by category (double-click to add). *Presets*: ready-made stacks. |
| **Right: Stack** | The effect stack of the selected layer (top runs first) and the controls of the selected effect. |
| **Right: Layers & Mask** | Layers, blend modes, opacity, and the selection mask tools. |
| **Under the preview** | Render status (progress, **Cancel**), the **Variations** grid, **Favourites** and the **Log**. Drag the divider between the preview and this panel to give the thumbnails more room. |

## 2. Your first glitch (2 minutes)

1. **Import** (Ctrl+O) a JPEG, PNG, WebP, BMP or TIFF — or drop it on the window.
   For animated/multipage files you are asked which frame/page to use.
2. Double-click **Pixel Sorting** in the Effects list. The preview updates.
3. Drag the **Lower threshold** slider. Only pixels whose brightness is inside
   the threshold interval are sorted.
4. Add **Scanline Tearing**, then press **R** to give it a new random seed.
   Press **C** to compare original and result; drag the divider.
5. Press **F5** (Render) to see the full-resolution result.
6. **Export** (Ctrl+E): choose PNG (keeps transparency), JPEG (pick a background
   colour for transparent areas) or WebP, the size and the file name pattern.
   Tick *Write JSON recipe* to save the exact settings next to the image.
   If a file with that name already exists you choose: **Overwrite** it, or
   **Save as new** — the new image gets the next free number (`name_01.png`,
   `name_02.png` …) and the old one is kept.

## 3. Working with the stack

* Each effect has **Enabled** (bypass, key **B**), **Mix** (strength; 0% = no
  effect), **Reset**, **Duplicate** (Ctrl+D) and **Remove** (Del).
* Drag effects in the list (or Ctrl+Up/Down) to reorder: order matters!
* **Channels**: apply an effect to all colours, one channel, or only
  luminance (keeps the colours, damages the brightness).
* **Affect alpha**: by default transparency is preserved. Tick this to let
  geometric effects move transparency too.
* **Seed**: effects with randomness store a seed. The same seed always gives
  the same result. 🎲 / **R** stores a new seed. The **global seed** in the
  toolbar (Ctrl+R to randomise) is combined with every effect's seed.
* Double-click a parameter's name to reset just that parameter.
* **Undo/Redo**: Ctrl+Z / Ctrl+Y. A slider drag is one undo step.

## 4. Data-bending effects explained

* **HEX / Byte Corruption** really encodes the image as a BMP, PPM or JPEG
  file and edits its bytes. *Search hex* `DA` and *Replace with* `0E DA`
  inserts a byte before every chosen `DA`, shifting everything after it — the
  classic diagonal colour tear. Headers are protected so the file still
  opens. JPEG mode is experimental; if every attempt produces an undecodable
  file you get a clear error (the effect is skipped in the preview) instead of
  a fake result.
* **Sonify** treats the pixel bytes as an audio recording and runs real
  FFmpeg audio filters on them: echo, phaser, flanger, tremolo, bit crusher,
  EQ and filters. Nothing is ever played. Needs FFmpeg (bundled; see
  *Help › Check FFmpeg*).
* **Text & File Injection** — the databending you would otherwise do in a
  hex or text editor. Type text (lyrics, GPS coordinates, a poem) *or* choose
  **any file** (a .txt, a .wav or .mp3, a PDF …): its raw bytes are written
  into the encoded image and the decoder misreads them. Choose where:
  - *BMP pixel data* with **Insert** → shifted scanlines and diagonal tears;
    **Overwrite / XOR / Add** → blocks of noise and neon colour.
  - *8-bit BMP colour table* → the whole palette is remapped (XOR gives neon).
  - *JPEG quantisation tables* → blocky, posterised, neon compression.
  - *JPEG scan data* (experimental) → smeared, broken blocks.
  - *PPM* or *raw RGB bytes* → stripes and colour-channel rotation.
  *Injections* sets how many pieces are pasted and *Bytes per injection* how
  big each piece is; the file is read piece by piece, so a long song spreads
  through the whole image.
* **Audio File Bending** — load your own sound (WAV directly; MP3, FLAC, OGG,
  M4A via the bundled FFmpeg) or use the built-in beat, and bend the pixels
  with it: *Mix*, *Add*, *Ring modulate*, *Envelope follower* (the music's
  loudness pumps the image) or *Gate* (loud parts replace pixels).
* **Audio Reverb Smearing** — the Audacity "reverb on a raw photo" trick:
  pixels pass through a cathedral, hall or room reverb (or your own
  impulse-response file), so highlights cast ghostly echoes. *Reverse reverb*
  makes the ghosts appear before the highlights.
* **Compression Decay** saves the image as JPEG over and over.
* **Data Reinterpretation** reads the raw bytes with the wrong row width,
  channel order or layout.

## 4b. Print, analog and geometry effects

* **Halftone Screen** — dots that are large in shadows and tiny in
  highlights: mono, CMYK colour halftone (classic screen angles) or RGB dots;
  round, square, line or diamond dots.
* **CMYK Halftone Drift** — separate cyan, magenta, yellow and black dot
  plates; a wave warps only the chosen plates (cyan + magenta by default) so
  the overlapping screens shimmer with moiré. *Misregistration* shifts plates
  out of alignment and *Plate slicing* tears bands of single plates sideways
  — the "Halftone Glitch Hybrid".
* **Ben-Day Dots** — Lichtenstein-style flat areas of uniform dots (pop
  red/yellow/blue or process CMY), solid blacks and inked outlines.
* **Voronoi Shatter Sorting** — the image breaks into glass-like cells whose
  colours are sorted dark to light; *Lead lines* draw the stained-glass frame.
* **Topographic Contour Tearing** — brightness becomes contour rings; every
  n-th ring slides sideways like a tectonic plate.
* **Pixel Sorting** — besides brightness/hue/saturation thresholds, *Between
  edges* sorts the runs between detected edges for long melting streaks.
* **RGB Channel Displacement** shifts colour channels; **Chromatic
  Aberration (lens)** adds lens-like colour fringes that grow towards the edges.
* **VHS & CRT** — smeared tape colour, tracking errors, static, scanlines,
  phosphor mask, bloom, vignette and screen curvature.

## 5. Selective processing: layers and masks

The default document has two layers: **Original** (bottom) and **Glitch**
(top, where your effects go). Each layer renders its own stack from the
original image.

* Lower the Glitch layer's **Opacity** to mix original and processed images,
  or choose a **Blend mode**: normal, multiply, screen, difference, overlay.
* **Add layers** (Ctrl+Shift+N) for different effects in different areas.
* **Mask tools** (Layers & Mask tab, or keys): **M** rectangle (right-drag
  subtracts), **P** brush, **E** eraser, **H** back to pan. `[` and `]`
  change the brush size. **Feather** softens edges, **Invert** flips the
  selection, **Luminance range** limits the layer to dark or bright parts of
  the original. The pink tint shows the *unselected* area.

## 6. Exploring: Variation Grid and favourites

1. Open the **Variations** tab, type any **Count** from 1 to 64 and choose an **Intensity**.
2. Click **Generate** (G). Each thumbnail is a complete recipe.
3. **Click** a thumbnail to restore exactly that recipe (undoable).
   Ctrl/Shift-click to select several, then **★ Favourite** or
   **Export selected…** (full resolution, with JSON recipes).
4. Control what may change: untick *Include in variations* on an effect, click
   **lock** next to a parameter, or lock an effect's seed. The *Batch seed*
   makes a variation set reproducible.

## 7. Presets

Double-click a preset to replace the selected layer's stack (right-click to
append). Built-in: Broken Broadcast, Acid Chromatic, Industrial Decay,
Corrupted Memory, Spectral Ghost, Scanner Failure, Digital Rust,
Photocopier Meltdown, Pop Art Misprint, Comic Moiré, VHS Night, Stained Glass
Meltdown, Cathedral Ghosts, Lyric Injection, Tectonic Portrait, plus a few
simple ones. *Save stack as preset…* stores
your own (marked ★).

## 8. Projects

**Save project** (Ctrl+S) writes a `.glab` file containing a managed copy of
the image, all layers, stacks, seeds, masks, favourites and view settings.
Reopening it (Ctrl+Shift+O) reproduces the same full-resolution result.
*Render › Verify project render* checks this against the saved hash.

## 9. Preview vs. final render

For speed the preview is computed on a reduced copy (choose its size in the
toolbar). Sizes such as offsets and tile sizes are scaled so it looks like the
final image, but byte-level effects (HEX, Sonify, pixel sorting) can differ in
detail. **Render (F5)** and **Export** always use full resolution.
Heavy renders run in the background — press **Esc** or *Cancel* at any time.

## 10. Troubleshooting

* *"FFmpeg is required…"* — Sonify needs `ffmpeg.exe`. The Windows package
  includes it in the `ffmpeg` folder; otherwise set it in *Edit › Preferences*.
* *Huge photos* — images over the working limit (64 MP by default) can be
  imported as a downscaled working copy. Change the limit in Preferences.
* *Something failed* — the effect is marked ⚠ in the stack with an
  explanation, and details are in the **Log** tab and in the log file
  (*Help › Open log folder*).
* *Contact sheet* — *Help › Generate effect contact sheet…* renders every
  effect and preset on test images, labelling anything that fails.
