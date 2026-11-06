"""Milestone 2: HEX corruption, Sonify, projects, variations."""

import os
import stat
import sys
import zipfile

import numpy as np
import pytest

from glitchlab.engine import fixtures
from glitchlab.engine.effects import databend
from glitchlab.engine.effects.databend import (
    encode_bmp,
    encode_ppm,
    find_pattern,
    jpeg_scan_range,
    mutate,
    normalise_length,
    encode_jpeg,
)
from glitchlab.engine.effects.sonify import Sonify, fit_length, rgb_to_yuv420p, yuv420p_to_rgb
from glitchlab.engine.errors import DependencyError, EffectError, ProjectError
from glitchlab.engine.image_io import LoadedImage
from glitchlab.engine.model import Document, EffectInstance, MaskSpec
from glitchlab.engine.params import parse_hex_bytes
from glitchlab.engine.pipeline import image_hash, render_document, render_full
from glitchlab.engine.project import Favourite, load_project, project_from_image, save_project
from glitchlab.engine.services import EngineServices
from glitchlab.engine.variations import VariationSettings, generate_variations, make_variation

from conftest import requires_ffmpeg


# ----------------------------------------------------------------- hex input
def test_parse_hex_bytes_valid():
    assert parse_hex_bytes("DA") == [0xDA]
    assert parse_hex_bytes("0E DA") == [0x0E, 0xDA]
    assert parse_hex_bytes("0eda") == [0x0E, 0xDA]
    assert parse_hex_bytes("0x0E, 0xDA") == [0x0E, 0xDA]
    assert parse_hex_bytes("FF ?? 00", allow_wildcard=True) == [0xFF, None, 0x00]


@pytest.mark.parametrize("bad", ["", "  ", "D", "GG", "0E D", "??", "0x"])
def test_parse_hex_bytes_invalid(bad):
    with pytest.raises(ValueError):
        parse_hex_bytes(bad)


def test_hex_invalid_param_reports_error(photo):
    inst = EffectInstance.create("hex_corrupt", seed=1, search="XY")
    with pytest.raises(EffectError, match="Search hex"):
        render_full(photo, Document.single([inst]))


# ----------------------------------------------------------------- encoders
def test_bmp_encoder_decodes_exactly(photo):
    from PIL import Image
    import io
    for w in (160, 161, 163):  # exercise row padding
        rgb = np.ascontiguousarray(fixtures.photo_like(w, 50)[..., :3])
        data, hlen = encode_bmp(rgb)
        assert hlen == 54 and data[:2] == b"BM"
        assert np.array_equal(np.array(Image.open(io.BytesIO(data)).convert("RGB")), rgb)
        data, hlen = encode_ppm(rgb)
        assert np.array_equal(np.array(Image.open(io.BytesIO(data)).convert("RGB")), rgb)


def test_find_and_replace_inserts_bytes():
    buf = np.frombuffer(bytes([1, 0xDA, 2, 0xDA, 0xDA, 3]), np.uint8)
    starts = find_pattern(buf, [0xDA], 0, len(buf))
    assert list(starts) == [1, 3, 4]
    out = databend.replace_at(buf, starts, 1, bytes([0x0E, 0xDA]))
    assert out.tobytes() == bytes([1, 0x0E, 0xDA, 2, 0x0E, 0xDA, 0x0E, 0xDA, 3])
    wild = find_pattern(buf, [0xDA, None], 0, len(buf))
    assert list(wild) == [1, 3]  # non-overlapping


def test_length_policies():
    orig = np.arange(10, dtype=np.uint8)
    longer = np.arange(13, dtype=np.uint8) + 100
    assert np.array_equal(normalise_length(longer, orig, "truncate_pad_source"), longer[:10])
    short = np.array([7, 8, 9], np.uint8)
    assert list(normalise_length(short, orig, "truncate_pad_source")) == [7, 8, 9, 3, 4, 5, 6, 7, 8, 9]
    assert list(normalise_length(short, orig, "truncate_pad_zero")) == [7, 8, 9] + [0] * 7
    assert list(normalise_length(short, orig, "truncate_pad_repeat")) == [7, 8, 9, 7, 8, 9, 7, 8, 9, 7]


def test_hex_bmp_header_protected_and_replacement_applied(photo, monkeypatch):
    captured = {}
    real = databend.decode_with_timeout

    def spy(blob, timeout=10.0):
        captured["blob"] = blob
        return real(blob, timeout)

    monkeypatch.setattr(databend, "decode_with_timeout", spy)
    inst = EffectInstance.create("hex_corrupt", seed=2, container="bmp", search="DA", replace="0E DA",
                                 replace_prob=1.0, max_replacements=0, mutation="none")
    out = render_full(photo, Document.single([inst])).image
    clean, hlen = encode_bmp(np.ascontiguousarray(photo[..., :3]))
    blob = captured["blob"]
    assert blob[:hlen] == clean[:hlen]          # headers untouched
    assert len(blob) == len(clean)              # payload normalised
    pay = np.frombuffer(clean[hlen:], np.uint8)
    n_matches = int((pay == 0xDA).sum())
    assert n_matches > 0
    # first match: bytes before it are unchanged, then 0E DA is inserted
    first = int(np.flatnonzero(pay == 0xDA)[0])
    assert blob[hlen:hlen + first] == clean[hlen:hlen + first]
    assert blob[hlen + first:hlen + first + 2] == bytes([0x0E, 0xDA])
    assert not np.array_equal(out, photo)
    assert np.array_equal(out[..., 3], photo[..., 3])


def test_hex_pattern_not_found_is_reported_not_disguised(photo):
    inst = EffectInstance.create("hex_corrupt", seed=2, search="AB CD EF 01 23", replace="00",
                                 mutation="none")
    r = render_full(photo, Document.single([inst]))
    assert any("not found" in w for w in r.warnings)
    assert any("unaltered" in w for w in r.warnings)


def test_hex_random_mutation_seeded(photo):
    a = render_full(photo, Document.single([EffectInstance.create("hex_corrupt", seed=5, search="", mutation="random", density=0.002)])).image
    b = render_full(photo, Document.single([EffectInstance.create("hex_corrupt", seed=5, search="", mutation="random", density=0.002)])).image
    c = render_full(photo, Document.single([EffectInstance.create("hex_corrupt", seed=6, search="", mutation="random", density=0.002)])).image
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert not np.array_equal(a, photo)


@pytest.mark.parametrize("mode", ["random", "bitflip", "increment", "swap", "insert", "delete"])
def test_hex_mutation_modes(photo, mode):
    inst = EffectInstance.create("hex_corrupt", seed=1, search="", mutation=mode, density=0.003, iterations=2)
    out = render_full(photo, Document.single([inst])).image
    assert out.shape == photo.shape and not np.array_equal(out, photo)


def test_hex_region_limits_damage(photo):
    inst = EffectInstance.create("hex_corrupt", seed=1, container="ppm", search="", mutation="random",
                                 density=0.01, region_start=0.5, region_end=1.0)
    out = render_full(photo, Document.single([inst])).image
    h = photo.shape[0]
    assert np.array_equal(out[: h // 2 - 1], photo[: h // 2 - 1])  # PPM is top-down
    assert not np.array_equal(out[h // 2 + 1:], photo[h // 2 + 1:])


def test_jpeg_mutation_never_touches_markers():
    rgb = np.ascontiguousarray(fixtures.photo_like(128, 96)[..., :3])
    data = encode_jpeg(rgb, 75)
    s, e = jpeg_scan_range(data)
    scan = np.frombuffer(data[s:e], np.uint8)
    prot = databend.jpeg_protect_mask(scan)
    rng = np.random.default_rng(1)
    out = mutate(scan, "random", 500, 0, len(scan), rng, prot, forbid_ff=True)
    changed = np.flatnonzero(out != scan)
    assert changed.size > 0
    assert not prot[changed].any()
    assert not (out[changed] == 0xFF).any()


def test_jpeg_corruption_succeeds(photo):
    inst = EffectInstance.create("hex_corrupt", seed=3, container="jpeg", search="", mutation="random", density=0.002)
    out = render_full(photo, Document.single([inst])).image
    assert not np.array_equal(out, photo)


def test_jpeg_failure_reported_not_disguised(photo, monkeypatch):
    def broken(blob, timeout=10.0):
        broken.calls += 1
        if broken.calls == 1:
            return databend._decode(blob)  # the clean reference decode
        raise EffectError("decoder gave up")
    broken.calls = 0
    monkeypatch.setattr(databend, "decode_with_timeout", broken)
    inst = EffectInstance.create("hex_corrupt", seed=3, container="jpeg", search="", density=0.002, attempts=3)
    with pytest.raises(EffectError, match="failed after 3 attempt"):
        render_full(photo, Document.single([inst]))
    # preview mode skips the effect but reports the failure
    broken.calls = 0
    r = render_document(photo, Document.single([inst]), on_error="skip")
    assert r.failures and "failed after" in r.failures[0].message


def test_jpeg_nothing_to_change_is_error(photo):
    inst = EffectInstance.create("hex_corrupt", seed=3, container="jpeg", search="AB CD EF AB CD", replace="00",
                                 mutation="none")
    with pytest.raises(EffectError):
        render_full(photo, Document.single([inst]))


def test_jpeg_replacement_with_ff_rejected(photo):
    inst = EffectInstance.create("hex_corrupt", seed=3, container="jpeg", search="00", replace="FF")
    with pytest.raises(EffectError, match="FF"):
        render_full(photo, Document.single([inst]))


def test_decoder_timeout(monkeypatch):
    import time
    monkeypatch.setattr(databend, "_decode", lambda data: time.sleep(2))
    with pytest.raises(EffectError, match="longer than"):
        databend.decode_with_timeout(b"x", timeout=0.2)


# -------------------------------------------------------------------- sonify
def test_yuv_roundtrip_and_padding():
    rgb = np.ascontiguousarray(fixtures.photo_like(81, 57)[..., :3])
    yuv, padded = rgb_to_yuv420p(rgb)
    assert padded == (58, 82)
    assert yuv.shape == (58 * 3 // 2, 82)
    back = yuv420p_to_rgb(yuv, padded)[:57, :81]
    assert back.shape == rgb.shape
    assert np.abs(back.astype(int) - rgb).mean() < 6  # only chroma subsampling loss


def test_sonify_length_policy():
    src = np.arange(10, dtype=np.uint8)
    assert list(fit_length(bytes(range(100, 115)), src)) == list(range(100, 110))
    assert list(fit_length(bytes([200, 201]), src)) == [200, 201, 2, 3, 4, 5, 6, 7, 8, 9]


def test_sonify_filter_chain_strings():
    eff = Sonify()
    p = eff.coerce_params({"echo": True, "crush": True, "phaser": True, "flanger": True, "tremolo": True,
                           "hp": True, "lp": True, "bp": True, "gain_db": 3, "bass_gain": 2, "treble_gain": -2,
                           "clip": "tanh"})
    chain, used = eff.build_chain(p, 44100)
    assert used == ["volume", "bass", "treble", "highpass", "lowpass", "bandpass", "acrusher", "aecho",
                    "aphaser", "flanger", "tremolo", "asoftclip"]
    assert "aecho=in_gain=0.800" in chain
    # frequencies are clamped below Nyquist at low (preview) sample rates
    chain2, _ = eff.build_chain(eff.coerce_params({"lp": True, "lp_freq": 20000, "echo": False}), 8000)
    assert "lowpass=f=3600.000" in chain2


@requires_ffmpeg
@pytest.mark.parametrize("flt", ["echo", "crush", "phaser", "flanger", "tremolo", "hp", "lp", "bp", "gain", "bass",
                                 "treble", "clip"])
def test_sonify_each_filter_alters_pixels(flt, services):
    img = fixtures.photo_like(97, 63)  # odd size
    base = {"echo": False}
    extra = {"gain": {"gain_db": 9.0}, "bass": {"bass_gain": 12.0}, "treble": {"treble_gain": 12.0},
             "clip": {"gain_db": 12.0, "clip": "hard", "clip_threshold": 0.3}}.get(flt, {flt: True})
    base.update(extra)
    inst = EffectInstance.create("sonify", seed=0, **base)
    out = render_full(img, Document.single([inst]), services=services).image
    assert out.shape == img.shape
    diff = np.abs(out.astype(int) - img.astype(int))[..., :3].mean()
    assert diff > 1.0, f"{flt} barely changed the image ({diff:.3f})"


@requires_ffmpeg
def test_sonify_planes_and_alpha(services, transparent):
    inst = EffectInstance.create("sonify", seed=0, planes="luma")
    out = render_full(transparent, Document.single([inst]), services=services).image
    assert np.array_equal(out[..., 3], transparent[..., 3])
    assert not np.array_equal(out[..., :3], transparent[..., :3])


@requires_ffmpeg
def test_sonify_cleans_temp_files(services, tmp_path, photo):
    svc = EngineServices(temp_root=str(tmp_path))
    render_full(photo, Document.single([EffectInstance.create("sonify", seed=0)]), services=svc)
    assert os.listdir(tmp_path) == []


@requires_ffmpeg
def test_sonify_cancel_kills_ffmpeg_and_allows_next_render(tmp_path):
    from glitchlab.engine.cancel import CancelToken
    from glitchlab.engine.errors import Cancelled
    import threading
    svc = EngineServices(temp_root=str(tmp_path))
    big = fixtures.photo_like(2400, 1800)  # large enough that FFmpeg is still running when cancelled
    tok = CancelToken()
    threading.Timer(0.15, tok.cancel).start()
    inst = EffectInstance.create("sonify", seed=0, echo=True, flanger=True, phaser=True)
    with pytest.raises(Cancelled):
        render_full(big, Document.single([inst]), services=svc, cancel=tok)
    assert os.listdir(tmp_path) == []
    r = render_full(fixtures.photo_like(64, 48), Document.single([inst]), services=svc)
    assert r.image.shape == (48, 64, 4)


def test_sonify_missing_ffmpeg_actionable_error(photo, tmp_path):
    svc = EngineServices(ffmpeg_path=str(tmp_path / "no-such-ffmpeg"))
    inst = EffectInstance.create("sonify", seed=0)
    with pytest.raises(DependencyError, match="not found"):
        render_full(photo, Document.single([inst]), services=svc)
    r = render_document(photo, Document.single([inst]), services=svc, on_error="skip")
    assert r.failures and "FFmpeg" in r.failures[0].message


@pytest.mark.skipif(sys.platform == "win32", reason="shell script stub")
def test_sonify_incompatible_ffmpeg_lists_missing_filters(photo, tmp_path):
    fake = tmp_path / "ffmpeg"
    fake.write_text("#!/bin/sh\nif [ \"$2\" = \"-version\" ]; then echo 'ffmpeg version 0.0-fake'; fi\n"
                    "if [ \"$2\" = \"-filters\" ]; then echo ' ... volume A->A Change volume.'; fi\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    svc = EngineServices(ffmpeg_path=str(fake))
    with pytest.raises(DependencyError, match="aecho"):
        render_full(photo, Document.single([EffectInstance.create("sonify", seed=0)]), services=svc)


def test_ffmpeg_not_an_ffmpeg(tmp_path):
    from glitchlab.engine.services import probe_ffmpeg
    with pytest.raises(DependencyError):
        probe_ffmpeg(sys.executable)


# ------------------------------------------------------------------ projects
def _project(photo, tmp_path):
    p = tmp_path / "src.png"
    from PIL import Image
    Image.fromarray(photo, "RGBA").save(p)
    from glitchlab.engine.image_io import load_image
    li = load_image(p)
    doc = Document.new(1234)
    doc.layers[1].stack = [EffectInstance.create("pixel_sort", seed=1, max_span=40),
                           EffectInstance.create("hex_corrupt", seed=2),
                           EffectInstance.create("block_scramble", seed=3)]
    doc.layers[1].mask = MaskSpec(enabled=True, feather=0.01, shapes=[
        {"type": "rect", "mode": "add", "x0": 0.1, "y0": 0.1, "x1": 0.8, "y1": 0.9},
        {"type": "stroke", "mode": "subtract", "points": [[0.2, 0.2], [0.5, 0.6]], "radius": 0.04}])
    doc.layers[1].blend = "screen"
    doc.layers[1].opacity = 0.8
    proj = project_from_image(li, doc)
    return proj


def test_project_roundtrip_reproduces_full_render(photo, tmp_path):
    proj = _project(photo, tmp_path)
    before = render_full(proj.source.rgba, proj.document).image
    proj.render_sha256 = image_hash(before)
    proj.favourites.append(Favourite(name="fav", document=proj.document.copy(), thumb=before[:20, :20].copy()))
    proj.presets = [{"name": "x", "stack": []}]
    proj.selection = {"selected_uid": proj.document.layers[1].stack[0].uid}
    path = str(tmp_path / "p.glab")
    save_project(path, proj)
    loaded = load_project(path)
    assert loaded.document.to_dict() == proj.document.to_dict()
    assert np.array_equal(loaded.source.rgba, proj.source.rgba)
    after = render_full(loaded.source.rgba, loaded.document).image
    assert image_hash(after) == loaded.render_sha256
    assert loaded.favourites[0].name == "fav" and loaded.favourites[0].thumb.shape == (20, 20, 4)
    assert loaded.original_bytes == proj.original_bytes
    assert loaded.selection == proj.selection
    with zipfile.ZipFile(path) as z:
        assert "source/working.png" in z.namelist()


def test_project_embeds_aux_files(photo, tmp_path):
    from PIL import Image
    proj = _project(photo, tmp_path)
    if "displace_map" not in __import__("glitchlab.engine.effects.base", fromlist=["REGISTRY"]).REGISTRY:
        pytest.skip("displacement map effect not available")
    aux = tmp_path / "map.png"
    Image.fromarray(fixtures.gradient(50, 40)).save(aux)
    proj.document.layers[1].stack.append(EffectInstance.create("displace_map", seed=1, source="image", map_file=str(aux)))
    before = render_full(proj.source.rgba, proj.document).image
    path = str(tmp_path / "aux.glab")
    save_project(path, proj)
    os.remove(aux)
    loaded = load_project(path)
    assert np.array_equal(render_full(loaded.source.rgba, loaded.document).image, before)


def test_project_errors(tmp_path):
    bad = tmp_path / "bad.glab"
    bad.write_bytes(b"not a zip")
    with pytest.raises(ProjectError):
        load_project(str(bad))
    with pytest.raises(ProjectError):
        load_project(str(tmp_path / "missing.glab"))
    z = tmp_path / "other.glab"
    with zipfile.ZipFile(z, "w") as zz:
        zz.writestr("project.json", '{"format": "something-else"}')
    with pytest.raises(ProjectError):
        load_project(str(z))
    newer = tmp_path / "newer.glab"
    with zipfile.ZipFile(newer, "w") as zz:
        zz.writestr("project.json", '{"format": "glitchlab-project", "version": 99}')
    with pytest.raises(ProjectError, match="newer"):
        load_project(str(newer))


def test_project_detects_damaged_source(photo, tmp_path):
    proj = _project(photo, tmp_path)
    path = str(tmp_path / "p.glab")
    save_project(path, proj)
    damaged = str(tmp_path / "d.glab")
    from glitchlab.engine.image_io import save_png_bytes
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(damaged, "w") as zout:
        for item in zin.namelist():
            data = zin.read(item)
            if item == "source/working.png":
                data = save_png_bytes(np.zeros_like(photo))
            zout.writestr(item, data)
    with pytest.raises(ProjectError, match="checksum"):
        load_project(damaged)


# ---------------------------------------------------------------- variations
def _vdoc():
    doc = Document.new(5)
    doc.layers[1].stack = [EffectInstance.create("pixel_sort", seed=1), EffectInstance.create("scanline_tear", seed=2),
                           EffectInstance.create("channel_shift", seed=3)]
    return doc


def test_variations_deterministic_and_distinct(photo):
    doc = _vdoc()
    s = VariationSettings(count=9, intensity=0.5, batch_seed=42)
    a = generate_variations(doc, s)
    b = generate_variations(doc, s)
    assert len(a) == 9
    assert [d.to_dict() for d in a] == [d.to_dict() for d in b]
    assert len({d.to_json() for d in a}) == 9
    imgs = [render_full(photo, d).image for d in a[:4]]
    assert not np.array_equal(imgs[0], imgs[1])


def test_variations_respect_locks_and_flags():
    doc = _vdoc()
    ps, st, cs = doc.layers[1].stack
    ps.locked_params = [p for p in ps.params]
    ps.seed_locked = True
    st.vary = False
    for v in generate_variations(doc, VariationSettings(count=16, intensity=1.0, batch_seed=1)):
        vps, vst, vcs = v.layers[1].stack
        assert vps.params == ps.params and vps.seed == ps.seed
        assert vst.to_dict() == st.to_dict()
        assert v.global_seed == doc.global_seed
    changed = [v.layers[1].stack[2].params != cs.params for v in generate_variations(doc, VariationSettings(count=4, intensity=1.0, batch_seed=1))]
    assert any(changed)


def test_variations_respect_ranges_and_intensity():
    doc = _vdoc()
    ps = doc.layers[1].stack[0]
    ps.ranges = {"lower": [0.1, 0.2]}
    for v in generate_variations(doc, VariationSettings(count=16, intensity=1.0, batch_seed=3)):
        assert 0.1 <= v.layers[1].stack[0].params["lower"] <= 0.2
    small = generate_variations(_vdoc(), VariationSettings(count=16, intensity=0.05, batch_seed=3))
    for v in small:
        assert abs(v.layers[1].stack[1].params["displacement"] - 90) <= 0.05 * 4000 + 1


def test_variation_restore_exact_recipe(photo):
    doc = _vdoc()
    v = make_variation(doc, VariationSettings(batch_seed=9, intensity=0.6), 3)
    restored = Document.from_dict(v.to_dict())
    assert np.array_equal(render_full(photo, v).image, render_full(photo, restored).image)


@pytest.mark.skipif(sys.platform == "win32", reason="shell script stub")
def test_ffmpeg_failing_roundtrip_is_reported(photo, tmp_path):
    from glitchlab.engine.services import SONIFY_FILTERS
    fake = tmp_path / "ffmpeg"
    listing = "".join(f"echo ' ... {f} A->A x'\n" for f in SONIFY_FILTERS)
    fake.write_text("#!/bin/sh\nif [ \"$2\" = \"-version\" ]; then echo 'ffmpeg version 0.0-fake'; exit 0; fi\n"
                    f"if [ \"$2\" = \"-filters\" ]; then\n{listing}exit 0; fi\nexit 1\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    svc = EngineServices(ffmpeg_path=str(fake))
    with pytest.raises(DependencyError, match="round-trip"):
        render_full(photo, Document.single([EffectInstance.create("sonify", seed=0)]), services=svc)


@requires_ffmpeg
def test_real_ffmpeg_passes_roundtrip_probe(services):
    info = services.ffmpeg()
    assert info.raw_alaw_ok and not info.missing()


FILTERS_61 = """Filters:
  T.. = Timeline support
  .S. = Slice threading
  ..C = Command support
 T.C acrusher          A->A       Reduce audio bit resolution.
 ... aecho             A->A       Add echoing to the audio.
 TSC bandpass          A->A       Apply a two-pole Butterworth band-pass filter.
 ... abuffer           |->A       Buffer audio frames, and make them accessible to the filterchain.
 ... amix              N->A       Audio mixing.
"""

FILTERS_8X = """Filters:
  T. = Timeline support
  .S = Slice threading
  A = Audio input/output
 T. acrusher          A->A       Reduce audio bit resolution.
 .. aecho             A->A       Add echoing to the audio.
 TS bandpass          A->A       Apply a two-pole Butterworth band-pass filter.
 .. abuffer           |->A       Buffer audio frames, and make them accessible to the filterchain.
 .. amix              N->A       Audio mixing.
"""


@pytest.mark.parametrize("text", [FILTERS_61, FILTERS_8X, FILTERS_8X.replace("\n", "\r\n")])
def test_parse_filter_list_all_ffmpeg_layouts(text):
    from glitchlab.engine.services import parse_filter_list
    assert parse_filter_list(text) == {"acrusher", "aecho", "bandpass", "abuffer", "amix"}
