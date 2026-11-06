"""Image import/export."""

import io
import os

import numpy as np
import pytest
from PIL import Image

from glitchlab.engine import fixtures
from glitchlab.engine.errors import ExportError, ImageTooLargeError, UnsupportedFormatError, ImageLoadError
from glitchlab.engine.image_io import ExportOptions, encode_image, flatten, load_image, probe_image, save_image


@pytest.mark.parametrize("fmt,ext", [("PNG", ".png"), ("BMP", ".bmp"), ("TIFF", ".tif"), ("WEBP", ".webp")])
def test_lossless_import_roundtrip(tmp_path, fmt, ext):
    img = fixtures.color_bars(64, 48)
    p = tmp_path / f"a{ext}"
    kw = {"lossless": True} if fmt == "WEBP" else {}
    Image.fromarray(img[..., :3]).save(p, fmt, **kw)
    li = load_image(p)
    assert li.rgba.shape == (48, 64, 4)
    assert np.array_equal(li.rgba[..., :3], img[..., :3])
    assert (li.rgba[..., 3] == 255).all()
    assert not li.had_alpha


def test_jpeg_import(tmp_path):
    img = fixtures.photo_like(64, 48)
    p = tmp_path / "a.jpg"
    Image.fromarray(img[..., :3]).save(p, "JPEG", quality=95)
    li = load_image(p)
    assert li.format == "JPEG"
    assert np.abs(li.rgba[..., :3].astype(int) - img[..., :3]).mean() < 6


def test_png_alpha_preserved(tmp_path):
    img = fixtures.transparent_circle(64, 48)
    p = tmp_path / "t.png"
    Image.fromarray(img, "RGBA").save(p)
    li = load_image(p)
    assert li.had_alpha
    assert np.array_equal(li.rgba, img)


def test_palette_transparency(tmp_path):
    im = Image.new("P", (8, 8), 0)
    im.putpalette([255, 0, 0, 0, 255, 0] + [0] * 762)
    im.info["transparency"] = 0
    p = tmp_path / "p.png"
    im.save(p, transparency=0)
    li = load_image(p)
    assert li.had_alpha and (li.rgba[..., 3] == 0).all()


def test_exif_orientation_applied(tmp_path):
    img = np.zeros((20, 40, 3), np.uint8)
    img[:, :10] = 255  # white strip on the left
    im = Image.fromarray(img)
    exif = im.getexif()
    exif[0x0112] = 6  # rotate 90 CW for display
    p = tmp_path / "o.jpg"
    im.save(p, "JPEG", exif=exif.tobytes(), quality=95)
    li = load_image(p)
    assert li.rgba.shape[:2] == (40, 20)
    assert li.rgba[:10, :, 0].mean() > 200  # strip now on top
    assert any("orientation 6" in n for n in li.notes)


def test_srgb_profile_not_converted(tmp_path):
    from glitchlab.engine.image_io import srgb_icc_bytes
    img = fixtures.gradient(32, 16)
    p = tmp_path / "s.png"
    Image.fromarray(img[..., :3]).save(p, icc_profile=srgb_icc_bytes())
    li = load_image(p)
    assert np.array_equal(li.rgba[..., :3], img[..., :3])
    assert any("treated as sRGB" in n for n in li.notes)


def test_no_profile_assumed_srgb(tmp_path):
    p = tmp_path / "n.png"
    Image.fromarray(fixtures.gradient(8, 8)[..., :3]).save(p)
    assert any("assumed to be sRGB" in n for n in load_image(p).notes)


def test_16bit_grey_reduced(tmp_path):
    arr = (np.arange(64 * 8, dtype=np.uint32).reshape(8, 64) * 128).astype(np.uint16)
    p = tmp_path / "g16.png"
    Image.fromarray(arr).save(p)
    li = load_image(p)
    assert any("16-bit" in n for n in li.notes)
    assert li.rgba[..., 0].max() == round(arr.max() / 257)


def test_multipage_tiff_frame_selection(tmp_path):
    frames = [Image.fromarray(np.full((10, 12, 3), v, np.uint8)) for v in (10, 120, 240)]
    p = tmp_path / "m.tif"
    frames[0].save(p, save_all=True, append_images=frames[1:])
    pr = probe_image(p)
    assert pr.n_frames == 3
    li = load_image(p, frame=2)
    assert li.rgba[0, 0, 0] == 240
    assert li.n_frames == 3 and any("page 3 of 3" in n for n in li.notes)
    with pytest.raises(ImageLoadError):
        load_image(p, frame=5)


def test_animated_webp_frame(tmp_path):
    frames = [Image.fromarray(np.full((10, 12, 3), v, np.uint8)) for v in (30, 200)]
    p = tmp_path / "a.webp"
    frames[0].save(p, save_all=True, append_images=frames[1:], lossless=True, duration=100)
    li = load_image(p, frame=1)
    assert abs(int(li.rgba[0, 0, 0]) - 200) <= 2


def test_unsupported_and_invalid(tmp_path):
    p = tmp_path / "x.gif"
    Image.new("RGB", (4, 4)).save(p)
    with pytest.raises(UnsupportedFormatError):
        load_image(p)
    q = tmp_path / "bad.png"
    q.write_bytes(b"this is not an image")
    with pytest.raises(UnsupportedFormatError):
        load_image(q)
    with pytest.raises(ImageLoadError):
        load_image(tmp_path / "missing.png")
    t = tmp_path / "trunc.png"
    buf = io.BytesIO()
    Image.fromarray(fixtures.photo_like(64, 64)[..., :3]).save(buf, "PNG")
    t.write_bytes(buf.getvalue()[:200])
    with pytest.raises(ImageLoadError):
        load_image(t)


def test_too_large_and_downscale(tmp_path):
    p = tmp_path / "big.png"
    Image.fromarray(fixtures.gradient(400, 300)[..., :3]).save(p)
    with pytest.raises(ImageTooLargeError):
        load_image(p, max_pixels=10_000)
    li = load_image(p, max_pixels=10_000, downscale_to_fit=True)
    assert li.width * li.height <= 10_000
    assert li.original_size == (400, 300)


def test_export_png_alpha_and_size(tmp_path):
    img = fixtures.transparent_circle(64, 48)
    out = tmp_path / "o.png"
    save_image(img, out, ExportOptions(format="PNG"))
    li = load_image(out)
    assert np.array_equal(li.rgba, img)
    out2 = tmp_path / "o2.png"
    save_image(img, out2, ExportOptions(format="PNG", width=32, height=24))
    assert load_image(out2).rgba.shape == (24, 32, 4)


def test_export_jpeg_flattens_with_background(tmp_path):
    img = np.zeros((16, 16, 4), np.uint8)  # fully transparent black
    out = tmp_path / "o.jpg"
    save_image(img, out, ExportOptions(format="JPEG", quality=95, background=(255, 0, 0)))
    li = load_image(out)
    assert li.rgba[..., 0].mean() > 240 and li.rgba[..., 1].mean() < 15
    assert np.array_equal(flatten(img, (1, 2, 3))[0, 0], [1, 2, 3])


def test_export_webp_alpha(tmp_path):
    img = fixtures.transparent_circle(64, 48)
    out = tmp_path / "o.webp"
    save_image(img, out, ExportOptions(format="WEBP", webp_lossless=True))
    li = load_image(out)
    assert np.array_equal(li.rgba[..., 3], img[..., 3])
    vis = img[..., 3] > 0
    assert np.array_equal(li.rgba[vis], img[vis])


def test_export_refuses_overwrite_and_source(tmp_path):
    img = fixtures.gradient(8, 8)
    out = tmp_path / "o.png"
    save_image(img, out, ExportOptions())
    with pytest.raises(ExportError):
        save_image(img, out, ExportOptions())
    save_image(img, out, ExportOptions(overwrite=True))
    with pytest.raises(ExportError):
        save_image(img, out, ExportOptions(overwrite=True), source_path=str(out))
    assert not any(n.startswith(".glitchlab-") for n in os.listdir(tmp_path))


def test_export_embeds_srgb(tmp_path):
    data = encode_image(fixtures.gradient(8, 8), ExportOptions(format="JPEG"))
    with Image.open(io.BytesIO(data)) as im:
        assert im.info.get("icc_profile")


def test_icc_profile_converted_to_srgb(tmp_path):
    from icc_util import make_rgb_profile
    icc = make_rgb_profile()
    arr = np.zeros((4, 3, 3), np.uint8)
    arr[:, 0] = (128, 128, 128)   # neutral grey
    arr[:, 1] = (90, 170, 90)     # mid green
    arr[:, 2] = (200, 60, 40)     # red-orange
    p = tmp_path / "adobe.png"
    Image.fromarray(arr).save(p, icc_profile=icc)
    li = load_image(p)
    assert li.icc_description == "Adobe RGB (1998) test"
    assert any("Converted from embedded profile" in n for n in li.notes)
    out = li.rgba[..., :3].astype(int)
    grey = out[0, 0]
    assert grey.max() - grey.min() <= 2 and abs(int(grey[0]) - 128) <= 4   # neutral stays neutral
    g_src, g_out = arr[0, 1].astype(int), out[0, 1]
    # Adobe RGB has a wider gamut: the same numbers mean a more saturated colour in sRGB
    assert (g_out[1] - g_out[0]) > (g_src[1] - g_src[0])
    # RGBA images keep their alpha through the conversion
    rgba = np.dstack([arr, np.full((4, 3), 77, np.uint8)])
    q = tmp_path / "adobe_rgba.png"
    Image.fromarray(rgba, "RGBA").save(q, icc_profile=icc)
    li2 = load_image(q)
    assert (li2.rgba[..., 3] == 77).all() and np.array_equal(li2.rgba[..., :3], li.rgba[..., :3])


def test_broken_icc_profile_falls_back(tmp_path):
    p = tmp_path / "broken.png"
    Image.fromarray(fixtures.gradient(8, 8)[..., :3]).save(p, icc_profile=b"not an icc profile at all")
    li = load_image(p)
    assert any("unreadable" in n for n in li.notes)
    assert li.rgba.shape == (8, 8, 4)


def test_cmyk_jpeg_import(tmp_path):
    p = tmp_path / "cmyk.jpg"
    Image.new("CMYK", (16, 8), (0, 255, 255, 0)).save(p, "JPEG", quality=95)
    li = load_image(p)
    assert li.original_mode == "CMYK"
    r, g, b = li.rgba[4, 8, :3].astype(int)
    assert r > 200 and g < 60 and b < 60   # cyan=0, magenta+yellow=full -> red
    assert any("CMYK" in n for n in li.notes)


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
def test_export_uses_normal_file_permissions(tmp_path):
    out = tmp_path / "perm.png"
    save_image(fixtures.gradient(8, 8), out, ExportOptions())
    mask = os.umask(0)
    os.umask(mask)
    assert (out.stat().st_mode & 0o777) == (0o666 & ~mask)
