"""Image import and export.

Working space
-------------
Glitch Lab 1.0 processes images as **8-bit sRGB-encoded RGBA** (``uint8``,
shape H x W x 4).  On import:

1. The requested frame/page of animated or multipage files is selected.
2. EXIF orientation is applied (and recorded) so pixels are upright.
3. An embedded ICC profile that is not sRGB is converted to sRGB with
   LittleCMS (relative colorimetric intent, black-point compensation).
   Images without a profile are *assumed* to be sRGB.
4. Higher bit-depth data (16-bit, 32-bit integer, float) is reduced to 8 bits
   explicitly and the conversion is recorded in ``notes``.
5. Greyscale, palette and CMYK modes are expanded to RGBA; source
   transparency is kept in the alpha channel.

Every conversion that changes data is listed in :attr:`LoadedImage.notes`
so the UI can show it.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageOps, features

from .errors import ExportError, ImageLoadError, ImageTooLargeError, UnsupportedFormatError
from .ops import resize_rgba

log = logging.getLogger(__name__)

IMPORT_EXTENSIONS = (".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".webp", ".bmp", ".dib", ".tif", ".tiff")
#: Pillow format names accepted on import (MPO is the multi-picture JPEG variant
#: written by many cameras; it is treated as JPEG and frame 0 is the main image).
IMPORT_FORMATS = {"JPEG", "MPO", "PNG", "WEBP", "BMP", "DIB", "TIFF"}

#: Default working-size limit.  Larger images can be imported downscaled.
DEFAULT_MAX_PIXELS = 64_000_000
#: Absolute refusal limit (decompression-bomb protection).
HARD_MAX_PIXELS = 250_000_000

Image.MAX_IMAGE_PIXELS = HARD_MAX_PIXELS

EXPORT_FORMATS = ("PNG", "JPEG", "WEBP")

_ORIENTATION_TEXT = {
    2: "mirrored horizontally",
    3: "rotated 180°",
    4: "mirrored vertically",
    5: "transposed",
    6: "rotated 90° clockwise",
    7: "transversed",
    8: "rotated 90° counter-clockwise",
}


def file_filter_string() -> str:
    pats = " ".join(f"*{e}" for e in IMPORT_EXTENSIONS)
    return f"Images ({pats});;All files (*)"


def webp_supported() -> bool:
    return bool(features.check("webp"))


@dataclass
class ImageProbe:
    path: str
    format: str
    width: int
    height: int
    mode: str
    n_frames: int


@dataclass
class LoadedImage:
    rgba: np.ndarray
    path: str | None
    format: str
    original_mode: str
    frame_index: int = 0
    n_frames: int = 1
    had_alpha: bool = False
    icc_description: str | None = None
    notes: list[str] = field(default_factory=list)
    original_size: tuple[int, int] = (0, 0)

    @property
    def width(self) -> int:
        return int(self.rgba.shape[1])

    @property
    def height(self) -> int:
        return int(self.rgba.shape[0])

    def sha256(self) -> str:
        return hashlib.sha256(np.ascontiguousarray(self.rgba).tobytes()).hexdigest()


_SRGB_PROFILE = None


def srgb_profile() -> ImageCms.ImageCmsProfile:
    global _SRGB_PROFILE
    if _SRGB_PROFILE is None:
        _SRGB_PROFILE = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
    return _SRGB_PROFILE


def srgb_icc_bytes() -> bytes:
    return srgb_profile().tobytes()


def _open(source: str | os.PathLike | bytes) -> tuple[Image.Image, str | None]:
    try:
        if isinstance(source, (bytes, bytearray, memoryview)):
            return Image.open(io.BytesIO(bytes(source))), None
        path = os.fspath(source)
        if not os.path.exists(path):
            raise ImageLoadError(f"File not found: {path}")
        if os.path.isdir(path):
            raise ImageLoadError(f"'{path}' is a folder, not an image file.")
        return Image.open(path), path
    except ImageLoadError:
        raise
    except Image.DecompressionBombError as exc:
        raise ImageLoadError(
            "This image is too large to open safely (over "
            f"{HARD_MAX_PIXELS // 1_000_000} megapixels).", detail=str(exc)) from exc
    except Exception as exc:  # PIL.UnidentifiedImageError, OSError, ...
        raise UnsupportedFormatError(
            "The file could not be read as an image. It may be damaged or in an "
            "unsupported format (supported: JPEG, PNG, WebP, BMP, TIFF).",
            detail=repr(exc),
        ) from exc


def probe_image(source: str | os.PathLike | bytes) -> ImageProbe:
    im, path = _open(source)
    with im:
        fmt = (im.format or "").upper()
        if fmt not in IMPORT_FORMATS:
            raise UnsupportedFormatError(
                f"{fmt or 'This'} files are not supported. Use JPEG, PNG, WebP, BMP or TIFF.")
        n = int(getattr(im, "n_frames", 1) or 1)
        return ImageProbe(path or "<memory>", fmt, im.width, im.height, im.mode, n)


def _icc_description(icc: bytes) -> tuple[ImageCms.ImageCmsProfile | None, str | None]:
    try:
        prof = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        desc = (ImageCms.getProfileDescription(prof) or "").strip()
        return prof, desc or "unnamed profile"
    except Exception as exc:  # broken profile
        log.warning("Unreadable ICC profile: %r", exc)
        return None, None


def _to_rgba_8bit(im: Image.Image, notes: list[str]) -> tuple[Image.Image, bool]:
    """Convert any Pillow mode to 8-bit RGBA, recording lossy steps."""
    mode = im.mode
    had_alpha = mode in ("RGBA", "LA", "La", "RGBa", "PA") or (
        mode in ("P", "L", "RGB") and "transparency" in im.info)
    if mode in ("I;16", "I;16B", "I;16L", "I;16N", "I"):
        arr = np.asarray(im)
        arr = arr.astype(np.int64)
        mx = int(arr.max()) if arr.size else 0
        if mode == "I" and mx > 65535:
            scale = 255.0 / mx
            notes.append(f"32-bit integer greyscale scaled to 8-bit (max value {mx}).")
            arr8 = np.clip(np.rint(arr * scale), 0, 255).astype(np.uint8)
        else:
            notes.append("16-bit greyscale reduced to 8-bit for processing (value / 257, rounded).")
            arr8 = np.clip(np.rint(np.clip(arr, 0, 65535) / 257.0), 0, 255).astype(np.uint8)
        return Image.fromarray(arr8, "L").convert("RGBA"), False
    if mode == "F":
        arr = np.nan_to_num(np.asarray(im, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        mx = float(arr.max()) if arr.size else 0.0
        if mx <= 1.0:
            notes.append("32-bit float greyscale (0..1) converted to 8-bit.")
            arr8 = np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)
        else:
            notes.append("32-bit float greyscale clipped to 0..255 and converted to 8-bit.")
            arr8 = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
        return Image.fromarray(arr8, "L").convert("RGBA"), False
    if mode == "1":
        notes.append("1-bit image expanded to 8-bit RGBA.")
        return im.convert("L").convert("RGBA"), False
    if mode == "P":
        notes.append("Palette image expanded to RGBA.")
        return im.convert("RGBA"), had_alpha
    if mode == "CMYK":
        notes.append("CMYK without a usable ICC profile converted to RGB with a simple formula; colours may shift.")
        return im.convert("RGB").convert("RGBA"), False
    if mode in ("YCbCr", "LAB", "HSV"):
        notes.append(f"{mode} image converted to RGB.")
        return im.convert("RGB").convert("RGBA"), False
    if mode == "RGBA":
        return im, True
    if mode in ("RGBa", "La", "LA", "PA"):
        return im.convert("RGBA"), True
    if mode in ("RGB", "L"):
        return im.convert("RGBA"), had_alpha
    notes.append(f"Unusual mode {mode} converted to RGBA.")
    return im.convert("RGBA"), had_alpha


def load_image(
    source: str | os.PathLike | bytes,
    *,
    frame: int = 0,
    max_pixels: int | None = DEFAULT_MAX_PIXELS,
    downscale_to_fit: bool = False,
) -> LoadedImage:
    """Load an image file into the 8-bit sRGB RGBA working format.

    Raises :class:`ImageTooLargeError` if the image exceeds ``max_pixels`` and
    ``downscale_to_fit`` is false.
    """
    im, path = _open(source)
    notes: list[str] = []
    try:
        fmt = (im.format or "").upper()
        if fmt not in IMPORT_FORMATS:
            raise UnsupportedFormatError(
                f"{fmt or 'This'} files are not supported. Use JPEG, PNG, WebP, BMP or TIFF.")
        n_frames = int(getattr(im, "n_frames", 1) or 1)
        frame = int(frame)
        if not 0 <= frame < n_frames:
            raise ImageLoadError(f"Frame {frame + 1} requested but the file has {n_frames} frame(s).")
        if n_frames > 1:
            im.seek(frame)
            kind = "page" if fmt == "TIFF" else "frame"
            if fmt == "MPO":
                notes.append(f"Multi-picture JPEG: picture {frame + 1} of {n_frames} used.")
            else:
                notes.append(f"Multi-{kind} file: {kind} {frame + 1} of {n_frames} imported as a still image.")
        original_mode = im.mode
        w, h = im.size
        original_size = (w, h)
        if w * h > HARD_MAX_PIXELS:
            raise ImageLoadError(f"Image is {w}x{h}; images over {HARD_MAX_PIXELS // 1_000_000} MP are refused.")
        reduce_factor = 1.0
        if max_pixels and w * h > max_pixels:
            if not downscale_to_fit:
                raise ImageTooLargeError(
                    f"The image is {w}x{h} ({w * h / 1e6:.1f} MP), above the working limit of "
                    f"{max_pixels / 1e6:.0f} MP.", width=w, height=h, limit=max_pixels)
            reduce_factor = (max_pixels / float(w * h)) ** 0.5
            if fmt in ("JPEG", "MPO"):
                im.draft("RGB", (int(w * reduce_factor) + 1, int(h * reduce_factor) + 1))
        # Bit-depth detection (Pillow silently reduces 16-bit RGB to 8-bit).
        try:
            rawmode = im.tile[0][3] if im.tile else None
            if isinstance(rawmode, tuple):
                rawmode = rawmode[0]
            if isinstance(rawmode, str) and ";16" in rawmode and im.mode in ("RGB", "RGBA"):
                notes.append("16-bit-per-channel colour reduced to 8-bit by the decoder.")
        except Exception:  # pragma: no cover - tile layout is decoder specific
            pass
        icc = im.info.get("icc_profile")
        orientation = 1
        try:
            orientation = int(im.getexif().get(0x0112, 1) or 1)
        except Exception:
            orientation = 1
        im.load()
        work = im
        if orientation in _ORIENTATION_TEXT:
            work = ImageOps.exif_transpose(work)
            notes.append(f"EXIF orientation {orientation} applied ({_ORIENTATION_TEXT[orientation]}).")
        icc_desc = None
        converted_by_icc = False
        if icc:
            prof, icc_desc = _icc_description(icc)
            if prof is None:
                notes.append("Embedded ICC profile is unreadable; colours interpreted as sRGB.")
            elif "srgb" in icc_desc.lower():
                notes.append(f"Embedded profile '{icc_desc}' treated as sRGB (no conversion).")
            elif work.mode in ("RGB", "RGBA", "CMYK", "L", "LA"):
                try:
                    in_mode = work.mode
                    alpha = None
                    if in_mode == "LA":
                        alpha = work.getchannel("A")
                        work = work.convert("L")
                        in_mode = "L"
                    if in_mode == "RGBA":
                        alpha = work.getchannel("A")
                        work = work.convert("RGB")
                        in_mode = "RGB"
                    flags = ImageCms.Flags.BLACKPOINTCOMPENSATION
                    work = ImageCms.profileToProfile(
                        work, prof, srgb_profile(),
                        renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                        outputMode="RGB", flags=flags)
                    if alpha is not None:
                        work.putalpha(alpha)
                    converted_by_icc = True
                    notes.append(f"Converted from embedded profile '{icc_desc}' to sRGB "
                                 "(relative colorimetric, black-point compensation).")
                except Exception as exc:
                    log.warning("ICC conversion failed: %r", exc)
                    notes.append(f"ICC conversion from '{icc_desc}' failed; colours interpreted as sRGB.")
            else:
                notes.append(f"Embedded profile '{icc_desc}' ignored for mode {work.mode}; assumed sRGB.")
        else:
            notes.append("No embedded colour profile; pixels assumed to be sRGB.")
        rgba_im, had_alpha = _to_rgba_8bit(work, notes)
        if original_mode in ("RGBA", "LA", "PA", "RGBa", "La"):
            had_alpha = True
        arr = np.array(rgba_im, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 4:
            raise ImageLoadError("Decoded image has an unexpected layout.")
        if reduce_factor < 1.0 and max_pixels:
            ah, aw = arr.shape[:2]
            if aw * ah > max_pixels:
                f = (max_pixels / float(aw * ah)) ** 0.5
                arr = resize_rgba(arr, max(1, int(aw * f)), max(1, int(ah * f)), interpolation="area")
            notes.append(f"Downscaled on import from {original_size[0]}x{original_size[1]} to "
                         f"{arr.shape[1]}x{arr.shape[0]} to fit the {max_pixels / 1e6:.0f} MP working limit.")
        if not had_alpha:
            arr[..., 3] = 255
        return LoadedImage(
            rgba=np.ascontiguousarray(arr), path=path, format=fmt, original_mode=original_mode,
            frame_index=frame, n_frames=n_frames, had_alpha=had_alpha, icc_description=icc_desc,
            notes=notes, original_size=original_size)
    except (ImageLoadError, ImageTooLargeError):
        raise
    except Exception as exc:
        raise ImageLoadError("The image could not be decoded. The file may be truncated or damaged.",
                             detail=repr(exc)) from exc
    finally:
        try:
            im.close()
        except Exception:
            pass


def rgba_from_pil(im: Image.Image) -> np.ndarray:
    notes: list[str] = []
    rgba, had_alpha = _to_rgba_8bit(im, notes)
    arr = np.array(rgba, dtype=np.uint8)
    if not had_alpha and im.mode not in ("RGBA", "LA"):
        arr[..., 3] = 255
    return arr


# ----------------------------------------------------------------------------
# Export
# ----------------------------------------------------------------------------

@dataclass
class ExportOptions:
    format: str = "PNG"              # PNG | JPEG | WEBP
    quality: int = 92                # JPEG / lossy WebP
    webp_lossless: bool = False
    background: tuple[int, int, int] = (255, 255, 255)   # JPEG flattening colour
    width: int | None = None         # output size (None = render size)
    height: int | None = None
    resample: str = "lanczos"
    jpeg_subsampling: str = "4:2:0"  # 4:4:4 | 4:2:2 | 4:2:0
    embed_srgb: bool = True
    overwrite: bool = False
    keep_opaque_alpha: bool = False  # PNG: write RGBA even when fully opaque

    def extension(self) -> str:
        return {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}[self.format.upper()]


def flatten(rgba: np.ndarray, background: tuple[int, int, int]) -> np.ndarray:
    """Composite RGBA over an opaque background colour -> uint8 RGB."""
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    rgb = rgba[..., :3].astype(np.float32)
    bg = np.asarray(background, np.float32).reshape(1, 1, 3)
    out = rgb * a + bg * (1.0 - a)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _same_file(a: str, b: str | None) -> bool:
    if not b:
        return False
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def encode_image(rgba: np.ndarray, opts: ExportOptions) -> bytes:
    """Encode an RGBA working image to bytes according to ``opts``."""
    fmt = opts.format.upper()
    if fmt not in EXPORT_FORMATS:
        raise ExportError(f"Unsupported export format {opts.format}.")
    img = rgba
    if opts.width and opts.height and (opts.width != img.shape[1] or opts.height != img.shape[0]):
        img = resize_rgba(img, opts.width, opts.height, interpolation=opts.resample)
    params: dict = {}
    if opts.embed_srgb:
        params["icc_profile"] = srgb_icc_bytes()
    buf = io.BytesIO()
    if fmt == "PNG":
        opaque = bool(img[..., 3].min() == 255)
        pil = Image.fromarray(img[..., :3] if (opaque and not opts.keep_opaque_alpha) else img,
                              "RGB" if (opaque and not opts.keep_opaque_alpha) else "RGBA")
        pil.save(buf, "PNG", compress_level=6, **params)
    elif fmt == "JPEG":
        rgb = flatten(img, opts.background)
        sub = {"4:4:4": 0, "4:2:2": 1, "4:2:0": 2}.get(opts.jpeg_subsampling, 2)
        Image.fromarray(rgb, "RGB").save(buf, "JPEG", quality=int(max(1, min(100, opts.quality))),
                                         subsampling=sub, optimize=True, **params)
    else:
        if not webp_supported():
            raise ExportError("This Pillow build has no WebP support.")
        pil = Image.fromarray(img, "RGBA")
        if opts.webp_lossless:
            pil.save(buf, "WEBP", lossless=True, quality=100, method=4, exact=True, **params)
        else:
            pil.save(buf, "WEBP", quality=int(max(1, min(100, opts.quality))), method=4, **params)
    return buf.getvalue()


def apply_default_permissions(path: str) -> None:
    """mkstemp creates 0600 files; give finished files the user's normal umask permissions."""
    if os.name == "nt":
        return
    try:
        mask = os.umask(0)
        os.umask(mask)
        os.chmod(path, 0o666 & ~mask)
    except OSError:
        pass


def save_image(rgba: np.ndarray, path: str | os.PathLike, opts: ExportOptions,
               *, source_path: str | None = None) -> str:
    """Write an image atomically.  Never overwrites the source image and
    refuses to replace an existing file unless ``opts.overwrite`` is set."""
    path = os.fspath(path)
    if _same_file(path, source_path):
        raise ExportError("Refusing to overwrite the source image. Choose a different file name.")
    if os.path.exists(path) and not opts.overwrite:
        raise ExportError(f"'{os.path.basename(path)}' already exists.", detail="exists")
    data = encode_image(rgba, opts)
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".glitchlab-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        apply_default_permissions(tmp)
        os.replace(tmp, path)
    except Exception as exc:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise ExportError(f"Could not write '{path}': {exc}", detail=repr(exc)) from exc
    return path


def save_png_bytes(rgba: np.ndarray) -> bytes:
    return encode_image(rgba, ExportOptions(format="PNG", keep_opaque_alpha=True))


def decode_png_bytes(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        return np.array(im.convert("RGBA"), dtype=np.uint8)


def thumbnail_rgba(rgba: np.ndarray, max_side: int) -> np.ndarray:
    h, w = rgba.shape[:2]
    s = min(1.0, max_side / float(max(h, w)))
    if s >= 1.0:
        return rgba.copy()
    return resize_rgba(rgba, max(1, round(w * s)), max(1, round(h * s)), interpolation="area")


def load_thumbnail(path: str, max_side: int = 160) -> np.ndarray:
    """Fast thumbnail for the image browser (uses JPEG draft decoding)."""
    with Image.open(path) as im:
        im.draft("RGB", (max_side * 2, max_side * 2))
        im = ImageOps.exif_transpose(im)
        im.thumbnail((max_side, max_side))
        return rgba_from_pil(im)


def list_images(folder: str | os.PathLike) -> list[str]:
    try:
        entries = sorted(Path(folder).iterdir(), key=lambda p: p.name.lower())
    except OSError:
        return []
    return [str(p) for p in entries if p.is_file() and p.suffix.lower() in IMPORT_EXTENSIONS]
