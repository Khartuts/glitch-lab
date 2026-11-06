# Feature checklist — Glitch Lab 1.0.0

Legend

| Mark | Meaning |
|---|---|
| ✅ | Implemented and covered by automated tests (`pytest`, 335 tests, all passing on the Linux build host) |
| 👁 | Implemented and verified manually / visually (offscreen screenshots, contact sheet), no dedicated automated assertion |
| ⚠️ | Implemented but **untested** in this release (cannot be exercised on the build host) |
| ❌ | Not implemented / known limitation |

Build hosts: Linux x86-64 (Python 3.13, FFmpeg 6.1.1) and the GitHub Actions
Windows runner (Windows Server 2025, Python 3.13.17, bundled FFmpeg 8.1.3):
332 tests passed there, 3 Linux-only tests skipped
([run 42212319268](https://github.com/Khartuts/glitch-lab/actions/runs/42212319268)).
The GUI was exercised headlessly (offscreen) on Windows; an interactive session
on a Windows 10/11 desktop has not been tested.

## Application & UI

| Feature | Status | Notes |
|---|---|---|
| Dark interface, central preview, left library, right stack/controls, bottom variations + render status | ✅ 👁 | `tests/test_ui.py`, screenshots |
| Import / Reset / Undo / Redo / Render / Export actions with shortcuts & tooltips | ✅ | |
| Zoom (wheel, Ctrl +/-), pan (middle/Space drag), fit (Ctrl+0), 100% (Ctrl+1) | 👁 | |
| Original/result draggable split (C) | 👁 | screenshot |
| Checkerboard behind transparency | 👁 | |
| Debounced preview (140 ms) | ✅ | |
| Clearly labelled reduced-resolution preview vs FULL RESOLUTION | ✅ 👁 | |
| Background workers, stale results ignored, cancellation then re-render | ✅ | `test_preview_matches_engine_and_ignores_stale`, `test_full_render_cancel_then_render` |
| Image browser with thumbnails, drag & drop import | 👁 | |
| Effect library by category, presets list, user presets | ✅ | |
| Preferences (FFmpeg path, working-size limit, cache size) | 👁 | |
| Help: user guide, shortcuts, contact sheet, self-test, FFmpeg check, log folder | 👁 | |

## Image support

| Feature | Status |
|---|---|
| Import JPEG, PNG, WebP, BMP, TIFF | ✅ |
| Animated / multipage: choose a single frame/page (dialog) | ✅ (engine) 👁 (dialog) |
| EXIF orientation | ✅ |
| Explicit sRGB working space; ICC → sRGB conversion (LittleCMS) | ✅ (synthetic Adobe RGB profile) |
| Untagged = sRGB, broken ICC fallback, CMYK, 16-bit, palette, 1-bit | ✅ |
| Explicit 8-bit conversion with notes | ✅ |
| Transparency preserved | ✅ |
| Source never overwritten; overwrite warning | ✅ (engine) 👁 (dialog) |
| Bounded memory: 64 MP working limit + downscale option, 250 MP refusal | ✅ |
| Invalid / truncated / unsupported files → clear errors | ✅ |

## Nondestructive stack

| Feature | Status |
|---|---|
| Enable/bypass, parameters, mix, reset, duplicate, remove, reorder | ✅ |
| Always rendered from the original (no cumulative degradation) | ✅ |
| Global + per-effect seeds; randomise stores the seed; previews never change seeds | ✅ |
| Channel targeting (RGB, R, G, B, luminance) | ✅ |
| Source alpha preserved by default; explicit *Affect alpha* | ✅ |
| Undo/redo of recipes only, bounded (200 entries / 64 MB), slider drags coalesced | ✅ |
| Zero strength returns input; bypass returns input; determinism | ✅ (all 20 effects) |
| Reordering changes processing order | ✅ |
| Prefix render cache (bounded LRU) | ✅ |

## Effects

| # | Effect | Status | Notes |
|---|---|---|---|
| 1 | HEX / Byte Corruption — BMP & PPM | ✅ | header protection, `DA → 0E DA`, wildcards, validation, seeded mutation (6 modes), density/region/iterations, length policies |
| 1 | HEX / Byte Corruption — JPEG scan data (experimental) | ✅ | marker/stuffing protection, bounded attempts, decode timeout, validation, failure reported (never disguised) |
| 2 | Sonify (YUV420p → A-law → FFmpeg → YUV420p) | ✅ | every filter verified to alter pixels; odd sizes; alpha; temp cleanup; cancel kills FFmpeg; missing/incompatible FFmpeg errors |
| 3 | Pixel sorting (luma/hue/sat; H, V, 2 diagonals; span length & randomness) | ✅ | |
| 4 | RGB channel displacement (wrap/clamp/mirror; RGB or YCbCr) | ✅ | |
| 5 | Scanline tearing | ✅ | |
| 6 | Block scrambling (shuffle/duplicate/rotate/offset/mixed) | ✅ | |
| 7 | Feedback accumulation | ✅ | |
| 8 | Bit-plane surgery (drop/set/invert/keep/swap/rotate/shuffle, bit depth) | ✅ | |
| 9 | Palette damage (posterise, adaptive, fixed, imported .gpl/.act/.hex/image, permutation, false colour, ordered & Floyd–Steinberg dithering) | ✅ | |
| 10 | Compression decay (real JPEG generations, downscale, shifts) | ✅ | |
| 11 | Data reinterpretation (channel order, row width, offset, planar/interleaved, gray/RGBA misreads) | ✅ | |
| 12 | Row & column drift (sine/saw/triangle/square/steps/noise) | ✅ | |
| 13 | Displacement maps (luminance, edges, noise, imported image) | ✅ | |
| 14 | Spectral damage (rfft2 bands, orientation sector, phase noise with Hermitian-safe symmetry) | ✅ | |
| 15 | Slit-scan collage | ✅ | |
| 16 | Error diffusion trails (direction, weights, gain, drift, clamp) | ✅ | next-line diffusion only (see limitations) |
| 17 | Cellular automata (B/S rules, luminance/bit-plane/noise seeds, 5 outputs) | ✅ | |
| 18 | Gray–Scott reaction-diffusion (bounded, seeded from image) | ✅ | |
| 19 | Mosaic bleed (neighbour/repeat/smear, chroma leakage, feather) | ✅ | |
| 20 | Channel cross-modulation (displace/threshold/remap/ring) | ✅ | |

## Selective processing & compositing

| Feature | Status |
|---|---|
| Rectangle selection (add / subtract) | ✅ |
| Brush & eraser with size and feather | ✅ (engine raster) 👁 (painting) |
| Mask inversion, luminance-range masks | ✅ |
| Multiple layers (≥ 2), opacity, visibility, reorder | ✅ |
| Blend modes normal, multiply, screen, difference, overlay | ✅ |
| One compositing model for preview and export | ✅ |
| Mask overlay on canvas | ✅ 👁 |

## Creative exploration

| Feature | Status |
|---|---|
| Variation Grid 4 / 9 / 16, seeded & reproducible (batch seed) | ✅ |
| Limit to selected effects (*Include in variations*), parameter locks, seed locks, intensity, per-parameter ranges (right-click a parameter label) | ✅ |
| Click restores exact recipe | ✅ |
| Favourites (saved in projects), export selected variations / favourites | ✅ |
| 8 named presets with real stacks + 3 simple presets | ✅ |

## Projects & export

| Feature | Status |
|---|---|
| Versioned `.glab` project: managed working image + original bytes, order, params, seeds, masks, layers, favourites, embedded presets, selection, render hash | ✅ |
| Reopen reproduces full-resolution result (hash) | ✅ |
| Auxiliary files (palettes, maps) embedded in projects | ✅ |
| Damaged / foreign / newer project files rejected with messages | ✅ |
| PNG with alpha, JPEG with quality & explicit background flattening, WebP (lossy/lossless) | ✅ |
| Output dimensions, resampling, filename patterns, JSON recipe sidecar, apply recipe | ✅ |
| Batch export of variations | ✅ |
| Full-resolution export matches the full-resolution pipeline | ✅ |

## Reliability

| Feature | Status |
|---|---|
| Vectorised NumPy/OpenCV processing; chunked index arrays | ✅ |
| NaN/∞/overflow guards; bounded iterations & attempts | ✅ |
| Temp files and FFmpeg child processes cleaned up on success, failure and cancel | ✅ |
| FFmpeg via argument lists, timeout, kill on cancel | ✅ |
| Rotating log file + Log tab; short UI messages | 👁 |

## Testing, tools & packaging

| Feature | Status | Notes |
|---|---|---|
| Automated engine tests on generated fixtures | ✅ | gradients, colour bars, checkerboards, transparent image, synthetic photo |
| Contact sheet (every effect + 8 presets on 5 fixtures, failures labelled) | ✅ | Help menu or `glitchlab-cli --contact-sheet`; sample in `docs/contact_sheet.png` |
| Headless self-test (`--selftest`) and GUI smoke test (`--smoke-test`) | ✅ | |
| PyInstaller onedir spec (GUI + CLI executables, FFmpeg folder, docs, licences) | ✅ on Linux | built and verified on Linux from a scrubbed environment |
| **Windows** package build (`packaging\build_windows.ps1`) | ✅ | GitHub Actions: tests, packaged self-test (incl. Sonify), contact sheet, headless GUI smoke test |
| Windows installer (Inno Setup `packaging\installer.iss`) | ⚠️ | built in CI; installing/uninstalling not tested |
| GitHub Actions Windows workflow (build + release) | ✅ | publishes the v1.0.0 release |
| `fetch_ffmpeg.ps1` (BtbN LGPL 8.1, SHA-256 check) | ✅ | run in CI (FFmpeg n8.1.3) |
| FFmpeg 6.x/8.x compatibility | ✅ | 6.1.1 (Linux) and 8.1.3 (Windows); 7.x untested |

## Known limitations

* ❌ No high-bit-depth or linear-light processing (8-bit sRGB working space by design for 1.0).
* ❌ Error Diffusion Trails propagate error to the next line only (same-line term omitted to keep it vectorised); classic Floyd–Steinberg lives in Palette Damage.
* ❌ Reduced previews of byte-level effects (HEX, Sonify, pixel sorting, block grids) approximate the export; this is labelled in the UI.
* ❌ No video, animation, GIF export, camera, audio playback or music features (out of scope by design).
* Package size: Windows installer 113 MB, portable zip 172 MB (Qt + OpenCV).
