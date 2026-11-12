"""Version 1.1 effects: halftone family, Voronoi, contour tear, lens/analog,
text & file injection, audio reverb and audio-file bending."""

import io
import os
import subprocess
import wave

import numpy as np
import pytest
from PIL import Image

from glitchlab.engine import fixtures
from glitchlab.engine.effects import databend
from glitchlab.engine.effects.audio import (
    ALAW_DECODE,
    PixelAudio,
    _read_wav,
    alaw_decode,
    alaw_encode,
    fft_convolve,
    load_audio,
    synthetic_ir,
)
from glitchlab.engine.effects.base import REGISTRY, make_seed_sequence
from glitchlab.engine.effects.databend import (
    encode_jpeg,
    inject,
    injection_chunks,
    jpeg_dqt_value_positions,
)
from glitchlab.engine.effects.print import compose_cmyk, rgb_to_cmyk, screen_plate
from glitchlab.engine.effects.sorting import voronoi_labels
from glitchlab.engine.errors import EffectError
from glitchlab.engine.model import Document, EffectInstance
from glitchlab.engine.ops import luminance, to_float
from glitchlab.engine.pipeline import render_full
from glitchlab.engine.variations import VariationSettings, generate_variations

from conftest import requires_ffmpeg

NEW_EFFECTS = ["halftone", "cmyk_drift", "benday", "voronoi_sort", "contour_tear", "chromatic", "vhs_crt",
               "data_injection", "audio_reverb", "audio_file_bend"]


def run(img, eid, seed=1, **params):
    return render_full(img, Document.single([EffectInstance.create(eid, seed=seed, **params)], 0)).image


def test_new_effects_registered():
    assert set(NEW_EFFECTS) <= set(REGISTRY)
    assert len(REGISTRY) == 30


# ----------------------------------------------------------------- halftone
@pytest.mark.parametrize("shape", ["round", "square", "line", "diamond"])
@pytest.mark.parametrize("cov", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_screen_plate_ink_matches_coverage(shape, cov):
    plate = screen_plate(np.full((240, 240), cov, np.float32), 12, 30.0, shape)
    assert abs(float(plate.mean()) - cov) < 0.08, (shape, cov, plate.mean())
    assert plate.min() >= 0 and plate.max() <= 1


def test_screen_plate_extremes():
    assert screen_plate(np.zeros((40, 40), np.float32), 8, 45, "round").max() == 0
    assert screen_plate(np.ones((40, 40), np.float32), 8, 45, "round").min() == 1


@pytest.mark.parametrize("gcr", [0.0, 0.5, 1.0])
def test_cmyk_separation_roundtrip(gcr, photo):
    f = to_float(photo[..., :3])
    back = compose_cmyk(*rgb_to_cmyk(f, gcr))
    assert np.abs(back - f).max() < 1e-3


def test_mono_halftone_tracks_tone():
    img = fixtures.gradient(256, 64)  # left dark-ish -> right bright in red
    out = run(img, "halftone", mode="mono", cell=6, ink="#000000", paper="#ffffff")
    lum_out = luminance(out)
    lum_in = luminance(img)
    left, right = lum_in[:, :64].mean() < lum_in[:, -64:].mean(), lum_out[:, :64].mean() < lum_out[:, -64:].mean()
    assert left == right
    assert set(np.unique(out[..., :3])) <= set(range(256))


def test_cmyk_drift_controls_change_result(photo):
    flat = run(photo, "cmyk_drift", drift_amp=0, c_dx=0, c_dy=0, m_dx=0, m_dy=0, slice_bands=0)
    drift = run(photo, "cmyk_drift", drift_amp=12, c_dx=0, c_dy=0, m_dx=0, m_dy=0, slice_bands=0)
    sliced = run(photo, "cmyk_drift", drift_amp=0, c_dx=0, c_dy=0, m_dx=0, m_dy=0, slice_bands=10, slice_shift=30)
    assert not np.array_equal(flat, drift) and not np.array_equal(flat, sliced)
    # drift only on cyan+magenta: the yellow plate (blue channel ink) is unchanged where cyan/magenta are absent
    a = run(photo, "cmyk_drift", seed=4, drift_plates="cm")
    b = run(photo, "cmyk_drift", seed=4, drift_plates="cmyk")
    assert not np.array_equal(a, b)


def test_benday_green_is_blue_plus_yellow_dots():
    img = np.zeros((96, 96, 4), np.uint8)
    img[..., :3] = (40, 160, 60)
    img[..., 3] = 255
    out = run(img, "benday", palette="pop", outline=0.0, flatten=0, black_level=0.0)
    cols = {tuple(c) for c in out[..., :3].reshape(-1, 3)[::7]}
    blueish = any(b > r + 60 and b > g for r, g, b in cols)
    yellowish = any(r > 180 and g > 150 and b < 120 for r, g, b in cols)
    assert blueish and yellowish


# ----------------------------------------------------------------- voronoi
def test_voronoi_sort_permutes_within_cells(photo):
    seed, gseed = 9, 0
    out = render_full(photo, Document.single([EffectInstance.create("voronoi_sort", seed=seed, cells=12, lead=0)],
                                             gseed)).image
    h, w = photo.shape[:2]
    rng = np.random.default_rng(make_seed_sequence(gseed, seed))
    n = max(2, int(round(12 ** 2 * min(h, w) / max(h, w))))
    labels = voronoi_labels(h, w, rng.random((n, 2)) * np.array([w, h]))
    a = photo.reshape(-1, 4).view(np.uint32).ravel()
    b = out.reshape(-1, 4).view(np.uint32).ravel()
    for lab in np.unique(labels)[:40]:
        m = labels.ravel() == lab
        assert np.array_equal(np.sort(a[m]), np.sort(b[m]))
    assert not np.array_equal(out, photo)


def test_voronoi_directions_and_leads(photo):
    for d in ("vertical", "horizontal", "diagonal", "radial"):
        assert run(photo, "voronoi_sort", direction=d, lead=2).shape == photo.shape


# --------------------------------------------------------- contour / lens
def test_contour_tear_zero_offset_is_identity(photo):
    assert np.array_equal(run(photo, "contour_tear", offset_x=0, offset_y=0, lines=0.0), photo)
    assert not np.array_equal(run(photo, "contour_tear", offset_x=15, lines=0.0), photo)


def test_chromatic_aberration(photo):
    assert np.array_equal(run(photo, "chromatic", strength=0.0, blur=0.0), photo)
    out = run(photo, "chromatic", strength=12.0, blur=0.0, green=0.0)
    assert np.array_equal(out[..., 1], photo[..., 1])
    assert not np.array_equal(out[..., 0], photo[..., 0])
    h, w = photo.shape[:2]  # the centre barely moves, the corners do
    centre = np.abs(out[h // 2 - 2:h // 2 + 2, w // 2 - 2:w // 2 + 2, 0].astype(int) - photo[h // 2 - 2:h // 2 + 2, w // 2 - 2:w // 2 + 2, 0]).mean()
    corner = np.abs(out[:12, :12, 0].astype(int) - photo[:12, :12, 0]).mean()
    assert centre <= corner


def test_vhs_crt_neutral_settings_are_near_identity(photo):
    out = run(photo, "vhs_crt", chroma_blur=0, chroma_shift=0, saturation=1.0, sharpen=0.0, tracking=0.0,
              noise=0.0, streaks=0, scanlines=0.0, mask="none", bloom=0.0, vignette=0.0, curvature=0.0)
    assert np.abs(out.astype(int) - photo).max() <= 1
    for mask in ("aperture", "shadow", "slot"):
        assert run(photo, "vhs_crt", mask=mask).shape == photo.shape


def test_pixel_sort_between_edges_preserves_rows(photo):
    out = run(photo, "pixel_sort", interval="edges", direction="horizontal", edge_threshold=0.3)
    assert not np.array_equal(out, photo)
    for y in range(0, photo.shape[0], 5):
        assert np.array_equal(np.sort(out[y].view(np.uint32).ravel()), np.sort(photo[y].view(np.uint32).ravel()))


# ----------------------------------------------------------- injection
def test_inject_modes():
    buf = np.arange(20, dtype=np.uint8)
    pos = np.array([2, 10])
    ch = [b"\xff\x01", b"\x05"]
    assert list(inject(buf, pos, ch, "overwrite")[:4]) == [0, 1, 255, 1]
    ins = inject(buf, pos, ch, "insert")
    assert len(ins) == 23 and list(ins[2:4]) == [255, 1] and ins[12] == 5
    assert inject(buf, pos, ch, "xor")[2] == 2 ^ 0xFF
    assert inject(buf, pos, ch, "add")[3] == 4
    prot = np.zeros(20, bool)
    prot[2] = True
    out = inject(buf, pos, ch, "overwrite", protected=prot, forbid_ff=True)
    assert out[2] == 2 and out[3] == 1
    assert 0xFF not in inject(buf, pos, ch, "insert", forbid_ff=True).tolist()


def test_injection_chunks_cycle():
    assert injection_chunks(b"abcdef", 4, 4) == [b"abcd", b"efab", b"cdef", b"abcd"]
    assert injection_chunks(b"ab", 2, 10) == [b"ab", b"ab"]


def test_jpeg_quant_table_positions(photo):
    blob = encode_jpeg(np.ascontiguousarray(photo[..., :3]), 80)
    both = jpeg_dqt_value_positions(blob, "both")
    assert len(both) == 128
    assert len(jpeg_dqt_value_positions(blob, "luma")) == 64
    assert len(jpeg_dqt_value_positions(blob, "chroma")) == 64


@pytest.mark.parametrize("target", ["bmp_pixels", "bmp_palette", "ppm", "jpeg_scan", "jpeg_tables", "raw"])
@pytest.mark.parametrize("mode", ["overwrite", "insert", "xor", "add"])
def test_injection_targets_and_modes(photo, target, mode):
    out = run(photo, "data_injection", seed=2, target=target, mode=mode, count=12, chunk=24)
    assert out.shape == photo.shape and not np.array_equal(out, photo)
    assert np.array_equal(out[..., 3], photo[..., 3])


def test_injection_from_any_file(photo, tmp_path):
    f = tmp_path / "blob.bin"
    f.write_bytes(os.urandom(5000))
    out = run(photo, "data_injection", source="file", data_file=str(f), target="bmp_pixels", mode="overwrite",
              count=5, chunk=300)
    assert not np.array_equal(out, photo)
    with pytest.raises(EffectError, match="choose a file"):
        run(photo, "data_injection", source="file", data_file="")
    with pytest.raises(EffectError, match="not found"):
        run(photo, "data_injection", source="file", data_file=str(tmp_path / "nope.txt"))
    with pytest.raises(EffectError):
        run(photo, "data_injection", source="text", text="")


def test_injection_bmp_palette_limits_colours(photo):
    out = run(photo, "data_injection", target="bmp_palette", palette_colors=16, mode="xor", count=4, chunk=8)
    assert len({tuple(c) for c in out[..., :3].reshape(-1, 3)}) <= 16


def test_injection_jpeg_scan_failure_is_reported(photo, monkeypatch):
    real = databend.decode_with_timeout
    calls = {"n": 0}

    def broken(blob, timeout=10.0):
        calls["n"] += 1
        if calls["n"] == 1:
            return real(blob, timeout)
        raise EffectError("decoder gave up")

    monkeypatch.setattr(databend, "decode_with_timeout", broken)
    with pytest.raises(EffectError, match="JPEG injection failed after 3"):
        run(photo, "data_injection", target="jpeg_scan", attempts=3)


# ------------------------------------------------------------------- audio
def test_alaw_roundtrip_and_monotonic():
    codes = np.arange(256, dtype=np.uint8)
    assert np.array_equal(alaw_encode(alaw_decode(codes)), codes)
    ramp = np.linspace(-1, 1, 5001, dtype=np.float32)
    dec = ALAW_DECODE[alaw_encode(ramp)]
    assert (np.diff(dec) >= 0).all()


@requires_ffmpeg
def test_alaw_codec_bit_exact_with_ffmpeg(tmp_path, services):
    ff = services.ffmpeg().path
    np.arange(256, dtype=np.uint8).tofile(tmp_path / "c.alaw")
    subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-f", "alaw", "-ar", "8000", "-ac", "1", "-i",
                    str(tmp_path / "c.alaw"), "-f", "s16le", "-c:a", "pcm_s16le", "-y", str(tmp_path / "c.s16")],
                   check=True)
    assert np.array_equal(np.fromfile(tmp_path / "c.s16", "<i2").astype(np.int32), ALAW_DECODE)
    np.arange(-32768, 32768, dtype=np.int16).tofile(tmp_path / "r.s16")
    subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-f", "s16le", "-ar", "8000", "-ac", "1", "-i",
                    str(tmp_path / "r.s16"), "-c:a", "pcm_alaw", "-f", "alaw", "-y", str(tmp_path / "r.alaw")],
                   check=True)
    mine = alaw_encode(np.arange(-32768, 32768, dtype=np.float32) / 32768.0)
    assert np.array_equal(np.fromfile(tmp_path / "r.alaw", np.uint8), mine)


def test_fft_convolve_matches_numpy():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000).astype(np.float32)
    k = rng.normal(size=300).astype(np.float32)
    assert np.allclose(fft_convolve(x, k, block=1024), np.convolve(x, k)[:5000], atol=1e-3)


def test_synthetic_ir_bounded():
    for kind in ("cathedral", "hall", "room", "plate"):
        ir = synthetic_ir(kind, 2.0, 20, 0.5, 8000, np.random.default_rng(1))
        assert np.isfinite(ir).all() and abs(float(np.sqrt((ir ** 2).sum())) - 1) < 1e-4


def test_audio_reverb_dry_only_is_pixel_audio_roundtrip(photo):
    out = run(photo, "audio_reverb", wet=0.0, dry=1.0)
    pa = PixelAudio(photo, "all")
    assert np.array_equal(out, pa.rebuild(pa.samples()))
    fwd = run(photo, "audio_reverb", reverse=False)
    rev = run(photo, "audio_reverb", reverse=True)
    assert not np.array_equal(fwd, rev) and not np.array_equal(fwd, photo)


def test_audio_bend_zero_amount_is_roundtrip(photo):
    pa = PixelAudio(photo, "all")
    base = pa.rebuild(pa.samples())
    for mode in ("mix", "add", "ring", "envelope", "gate"):
        assert np.array_equal(run(photo, "audio_file_bend", mode=mode, amount=0.0), base), mode


def _write_wav(path, data, sr=22050, width=2, channels=1):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(width)
        wf.setframerate(sr)
        wf.writeframes(data)


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_wav_reader_sample_widths(tmp_path, width):
    n = 1000
    ramp = np.linspace(-0.5, 0.5, n)
    if width == 1:
        raw = np.rint(ramp * 127 + 128).astype(np.uint8).tobytes()
    elif width == 2:
        raw = np.rint(ramp * 32767).astype("<i2").tobytes()
    elif width == 3:
        v = np.rint(ramp * (2 ** 23 - 1)).astype(np.int32)
        raw = b"".join(int(x).to_bytes(3, "little", signed=True) for x in v)
    else:
        raw = np.rint(ramp * (2 ** 31 - 1)).astype("<i4").tobytes()
    p = tmp_path / f"w{width}.wav"
    _write_wav(p, raw, width=width)
    a, sr = _read_wav(str(p))
    assert sr == 22050 and len(a) == n and np.abs(a - ramp).max() < 0.02


def test_audio_bend_with_wav_file(photo, tmp_path):
    t = np.arange(22050) / 22050.0
    stereo = np.stack([np.sin(2 * np.pi * 330 * t), np.sin(2 * np.pi * 440 * t)], -1)
    p = tmp_path / "s.wav"
    _write_wav(p, np.rint(stereo * 20000).astype("<i2").tobytes(), channels=2)
    a = load_audio(str(p), 11025, None, None)
    assert abs(len(a) - 11025) <= 1
    out = run(photo, "audio_file_bend", source="file", audio_file=str(p), mode="ring", amount=0.8)
    assert not np.array_equal(out, photo)
    with pytest.raises(EffectError, match="choose an audio file"):
        run(photo, "audio_file_bend", source="file", audio_file="")


@requires_ffmpeg
def test_audio_bend_decodes_other_formats_with_ffmpeg(photo, tmp_path, services):
    wav = tmp_path / "x.wav"
    t = np.arange(16000) / 16000.0
    _write_wav(wav, np.rint(np.sin(2 * np.pi * 200 * t) * 20000).astype("<i2").tobytes(), sr=16000)
    flac = tmp_path / "x.flac"
    subprocess.run([services.ffmpeg().path, "-hide_banner", "-loglevel", "error", "-i", str(wav), "-y", str(flac)],
                   check=True)
    doc = Document.single([EffectInstance.create("audio_file_bend", seed=1, source="file", audio_file=str(flac))], 0)
    out = render_full(photo, doc, services=services).image
    assert not np.array_equal(out, photo)


def test_audio_reverb_with_ir_file(photo, tmp_path):
    p = tmp_path / "ir.wav"
    noise = np.random.default_rng(3).normal(0, 0.3, 8000) * np.exp(-np.arange(8000) / 1500)
    _write_wav(p, np.rint(noise * 32767).astype("<i2").tobytes(), sr=8000)
    out = run(photo, "audio_reverb", space="file", ir_file=str(p))
    assert not np.array_equal(out, photo)


# ------------------------------------------------------------- variations
def test_variation_counts_any_number():
    doc = Document.single([EffectInstance.create("scanline_tear", seed=1)], 1)
    assert len(generate_variations(doc, VariationSettings(count=7))) == 7
    assert len(generate_variations(doc, VariationSettings(count=500))) == 64
    assert len(generate_variations(doc, VariationSettings(count=0))) == 1
