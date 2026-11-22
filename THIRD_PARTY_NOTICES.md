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

## Bundled fonts

| Font | Licence |
|---|---|
| DejaVu Sans, Sans Mono, Serif (12 styles) | Bitstream Vera Fonts licence + public-domain DejaVu changes (`resources/fonts/LICENSE_DEJAVU`) |
| STIX General (4 styles) | SIL Open Font License 1.1 (`resources/fonts/LICENSE_STIX`) |

Both licences allow redistribution with the program. Fonts you import are copied to your own data folder and are
never embedded in projects.

## Optional AI engine and models (downloaded on request, not in the installer)

The AI effects use components that the program downloads only when you ask
(AI › Install AI engine…, AI › Models…). They are stored in your user data
folder, are never uploaded anywhere and run locally.

| Component | Version / revision | Licence |
|---|---|---|
| PyTorch (CPU build for Windows) | 2.15.0 | BSD-3-Clause (bundles further permissive licences, see its `LICENSE` in the downloaded engine) |
| Hugging Face transformers | 5.24.0 | Apache-2.0 |
| Hugging Face diffusers | 0.43.0 | Apache-2.0 |
| safetensors, tokenizers, huggingface_hub and their dependencies | see `src/glitchlab/ai/runtime_win_amd64.json` | Apache-2.0 / MIT / BSD (per package) |
| SAM 2.1 Hiera Tiny (facebook/sam2.1-hiera-tiny) | de431c40 | Apache-2.0 |
| Depth Anything V2 Small (depth-anything/Depth-Anything-V2-Small-hf) | 5426e4f0 | Apache-2.0 |
| DINOv2 Small (facebook/dinov2-small) | ed25f3a3 | Apache-2.0 |
| Stable Diffusion VAE ft-MSE (stabilityai/sd-vae-ft-mse) | 31f26fde | MIT |
| Stable Diffusion 1.5 (stable-diffusion-v1-5/stable-diffusion-v1-5) | 451f4fe1 | CreativeML OpenRAIL-M (use restrictions apply to the user) |
| Stable Diffusion 1.5 Inpainting (stable-diffusion-v1-5/stable-diffusion-inpainting) | 8a4288a7 | CreativeML OpenRAIL-M (use restrictions apply to the user) |

The Stable Diffusion safety checker is not downloaded; you are responsible for
using the generative effects within the OpenRAIL-M licence's use restrictions.

If you distribute a build that contains a **GPL** FFmpeg build instead of an
LGPL one, the combined distribution terms of that FFmpeg build apply to the
`ffmpeg.exe` you ship (Glitch Lab only executes it as a separate process).
