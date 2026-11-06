# Technical notes

## Architecture

* `glitchlab.engine` is a pure-Python/NumPy package with **no Qt import**.
  It contains image I/O, the document model, the render pipeline, every
  effect, projects, presets, variations, export and the contact sheet. The UI
  (`glitchlab.ui`) and the CLI (`glitchlab.cli`) are thin clients; all tests
  of effects run headlessly.
* An effect is a class with `Param` specifications and a
  `process(img, params, ctx)` method on uint8 RGBA. The generic wrapper
  (`effects/base.py::apply_effect`) adds bypass, mix, channel targeting,
  alpha preservation, seeded RNGs and output validation uniformly.

## Colour and working space

* Working format: **8-bit sRGB-encoded RGBA** (`uint8`, H×W×4).
* Embedded non-sRGB ICC profiles are converted to sRGB with LittleCMS
  (relative colorimetric + black-point compensation). Untagged images are
  assumed to be sRGB. CMYK without a profile uses Pillow's simple formula
  (reported).
* 16-bit / 32-bit / float data is reduced to 8 bits explicitly
  (16-bit: value/257, rounded) and every conversion is listed in the import
  notes and the log.
* EXIF orientation is applied on import.
* Exports embed an sRGB ICC profile (optional).
* Luminance = Rec. 601 luma on sRGB-encoded values; the luminance channel
  target uses BT.601 full-range YCbCr (JFIF).
* Effects that need floating point convert explicitly and back with
  rounding, clipping and NaN/∞ guards (`ops.to_uint8`).

## Nondestructive rendering

* The document (layers → stacks → effect instances with parameters, mix,
  seeds, channel target, alpha flag, locks) is JSON. Rendering always starts
  from the original; undo history stores recipes only (≤ 200 entries,
  ≤ 64 MB).
* Seeds: each effect gets `np.random.SeedSequence([global_seed, effect_seed,
  salt])`. Effects draw spatial randomness in normalised coordinates where
  possible, so the reduced preview resembles the export.
* Spatial parameters are declared in full-resolution pixels and scaled by
  `scale = preview_width / full_width` for previews. Simulations use grids
  defined relative to the image; spectral bands are defined relative to the
  full-resolution Nyquist frequency.
* Cache: intermediate results per stack prefix are cached (LRU, default
  768 MB, configurable) so editing a late effect does not recompute earlier
  ones. Disabled effects do not invalidate the cache.

## Compositing model (preview and export are identical)

W3C Compositing Level 1 *source-over* with separable blend modes in float32:

```
Cs' = (1 − αb)·Cs + αb·B(Cb, Cs)
αo  = αs + αb·(1 − αs)          αs includes layer opacity × mask
Co  = (αs·Cs' + αb·(1 − αs)·Cb) / αo
```

Blend functions: normal, multiply, screen, difference, overlay.
Alpha output: *auto* (default) preserves the source alpha unless an enabled
effect on a visible layer has *Affect alpha* on; *preserve* and *composite*
force either behaviour.

## Masks

Vector shapes in normalised coordinates (rectangles, brush/eraser polylines
with radius as a fraction of the long side), anti-aliased rasterisation at
render resolution, Gaussian feather, optional luminance range measured on the
original, invert. Identical at preview and export resolution.

## HEX / byte corruption

* BMP: 24-bit BI_RGB bottom-up (54-byte header protected; rows padded to 4
  bytes). PPM: P6 maxval 255 (ASCII header protected).
* Search/replace supports `??` wildcards (search only); matches are
  non-overlapping, selected with a seeded probability and capped.
* Mutations: random value, bit flip, increment, swap, insert, delete;
  density × region length per iteration; seeded.
* Fixed-size payload policy: insertions/deletions shift following bytes; the
  result is truncated to the original length, or padded with the source
  tail / zeros / repetition (user choice).
* JPEG (experimental): only entropy-coded scan data (after SOS, before EOI)
  is edited; `FF` bytes and the byte after them are protected and `FF` is never
  written. Up to *Attempts* seeded attempts, each decoded with a 10 s limit;
  accepted only if it decodes at the right size and differs from the clean
  decode. Otherwise an `EffectError` is raised (preview: effect skipped and
  marked ⚠; export: aborted with the message).
* "Pattern not found" / "no bytes changed" are reported as warnings.

## Sonify

RGB → even-padded (edge replicate) → I420/YUV420p planar bytes (OpenCV,
BT.601) → raw A-law mono at the chosen rate → FFmpeg `-af` chain → raw A-law
→ YUV420p → RGB → crop; alpha restored. Output longer than the input is
truncated; shorter output is completed with the unprocessed source bytes.
Planes option: whole stream, luma only, chroma only. See docs/FFMPEG.md.

## Bounded resources

* Import limit 64 MP by default (offer to downscale), hard refusal at 250 MP
  (decompression-bomb guard).
* Bounded loops: CA ≤ 400 generations on ≤ 1024-cell grids; Gray–Scott ≤ 10 000
  iterations on ≤ 1024 grids with clamping every step and denormal flushing;
  feedback ≤ 64 iterations; JPEG decay ≤ 60 generations; HEX ≤ 20 iterations
  and ≤ 16 JPEG attempts; error trails clamp carried error (≤ 255).
* Row-chunked index arrays for row shifting; int32 index grids for diagonal
  sorting.
* Every long loop checks the cancel token; FFmpeg is killed on cancel and its
  temp directory removed.

## Determinism

Same image + recipe + seeds + resolution + software environment ⇒
identical pixels (tested). Projects store the exact working image (lossless
PNG) and optionally the hash of the last full render. Different library
versions (e.g. libjpeg, FFmpeg) may change JPEG/Sonify-based results.

## Known limitations

* Processing is 8-bit only (explicit conversion on import).
* Error Diffusion Trails diffuse error to the *next line* only (vectorised);
  classic Floyd–Steinberg is available in Palette Damage (Pillow).
* Preview of byte-level effects is an approximation (labelled).
* Windows packaging was configured and its Linux equivalent was built and
  verified; a Windows binary was not built on this build host.
