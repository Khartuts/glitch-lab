"""Milestone 3 effects, contact sheet and self-test."""

import numpy as np
import pytest
from PIL import Image

from glitchlab.engine import fixtures
from glitchlab.engine.effects.base import REGISTRY
from glitchlab.engine.effects.colour import bayer_matrix, load_palette_file
from glitchlab.engine.effects.frequency import antisymmetrize_columns
from glitchlab.engine.effects.simulation import gray_scott, life_step, parse_rule
from glitchlab.engine.errors import EffectError
from glitchlab.engine.model import Document, EffectInstance
from glitchlab.engine.pipeline import render_document, render_full
from glitchlab.engine.presets import BUILTIN_PRESETS, apply_preset

SPEC_EFFECTS = ["pixel_sort", "channel_shift", "scanline_tear", "block_scramble", "feedback", "bitplane", "palette",
                "jpeg_decay", "reinterpret", "row_drift", "displace_map", "spectral", "slitscan", "error_trails",
                "cellular", "reaction_diffusion", "mosaic_bleed", "cross_mod", "hex_corrupt", "sonify"]


def run(img, eid, seed=1, **params):
    return render_full(img, Document.single([EffectInstance.create(eid, seed=seed, **params)], 0)).image


def test_all_twenty_spec_effects_registered():
    assert set(SPEC_EFFECTS) <= set(REGISTRY)


def test_all_named_presets_available_and_valid(photo):
    names = {"Broken Broadcast", "Acid Chromatic", "Industrial Decay", "Corrupted Memory", "Spectral Ghost",
             "Scanner Failure", "Digital Rust", "Photocopier Meltdown"}
    have = {p["name"] for p in BUILTIN_PRESETS}
    assert names <= have
    for p in BUILTIN_PRESETS:
        for fx in p["stack"]:
            assert fx["effect"] in REGISTRY, (p["name"], fx["effect"])
            unknown = set(fx["params"]) - {q.name for q in REGISTRY[fx["effect"]].params}
            assert not unknown, (p["name"], fx["effect"], unknown)


# --- bit planes -------------------------------------------------------------
def test_bitplane_operations(photo):
    rgb = photo[..., :3]
    assert (run(photo, "bitplane", operation="drop", planes=0xFF)[..., :3] == 0).all()
    assert np.array_equal(run(photo, "bitplane", operation="invert", planes=0xFF)[..., :3], 255 - rgb)
    msb = run(photo, "bitplane", operation="keep", planes=0x80)[..., :3]
    assert set(np.unique(msb)) <= {0, 128}
    assert np.array_equal(run(photo, "bitplane", operation="rotate", rotate_by=0)[..., :3], rgb)
    one_bit = run(photo, "bitplane", operation="invert", planes=0, depth=1)[..., :3]
    assert set(np.unique(one_bit)) <= {0, 255}
    sw = run(photo, "bitplane", operation="swap", swap_a=7, swap_b=0)[..., :3]
    exp = (rgb & 0b01111110) | ((rgb >> 7) & 1) | ((rgb & 1) << 7)
    assert np.array_equal(sw, exp)


# --- palettes -------------------------------------------------------------
def test_posterize_levels(photo):
    out = run(photo, "palette", mode="posterize", levels=3)
    assert set(np.unique(out[..., :3])) <= {0, 128, 255}


def test_ordered_dither_matrix():
    m = bayer_matrix(8)
    assert m.shape == (8, 8) and len(np.unique(m)) == 64 and 0 < m.min() < m.max() < 1


@pytest.mark.parametrize("ext", [".gpl", ".hex", ".act", ".png"])
def test_imported_palettes(tmp_path, photo, ext):
    cols = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (20, 20, 20)]
    p = tmp_path / f"pal{ext}"
    if ext == ".gpl":
        p.write_text("GIMP Palette\nName: t\n#\n" + "\n".join(f"{r} {g} {b} c" for r, g, b in cols))
    elif ext == ".hex":
        p.write_text("\n".join(f"#{r:02x}{g:02x}{b:02x}" for r, g, b in cols))
    elif ext == ".act":
        data = bytes(sum(cols, ())) + bytes(768 - 12) + (4).to_bytes(2, "big") + b"\xff\xff"
        p.write_bytes(data)
    else:
        Image.fromarray(np.array([cols], np.uint8)).save(p)
    pal = load_palette_file(str(p))
    assert {tuple(c) for c in pal} == set(cols)
    out = run(photo, "palette", mode="file", palette_file=str(p), dither="none")
    assert {tuple(c) for c in out[..., :3].reshape(-1, 3)} <= set(cols)
    fs = run(photo, "palette", mode="file", palette_file=str(p), dither="floyd")
    assert {tuple(c) for c in fs[..., :3].reshape(-1, 3)} <= set(cols)


def test_palette_errors(tmp_path, photo):
    with pytest.raises(EffectError):
        run(photo, "palette", mode="file", palette_file="")
    bad = tmp_path / "one.hex"
    bad.write_text("#ffffff")
    with pytest.raises(EffectError, match="fewer than two"):
        run(photo, "palette", mode="file", palette_file=str(bad))


def test_palette_modes_and_permutation(photo):
    for mode in ("adaptive", "fixed", "false_colour", "posterize"):
        a = run(photo, "palette", seed=1, mode=mode, permute=True)
        b = run(photo, "palette", seed=2, mode=mode, permute=True)
        assert not np.array_equal(a, photo)
        assert not np.array_equal(a, b), mode
    fixed = run(photo, "palette", mode="fixed", fixed_palette="gameboy")
    assert len({tuple(c) for c in fixed[..., :3].reshape(-1, 3)}) <= 4


# --- error trails ---------------------------------------------------------
def test_error_trails_quantised_and_bounded(photo):
    out = run(photo, "error_trails", levels=2, gain=1.5, clamp=255, w_down=1, w_down_left=1, w_down_right=1)
    assert set(np.unique(out[..., :3])) <= {0, 255}
    for d in ("up", "left", "right"):
        assert run(photo, "error_trails", direction=d).shape == photo.shape
    lum = run(photo, "error_trails", mode="luminance", drift=3)
    assert np.isfinite(lum).all()


# --- compression / reinterpretation ---------------------------------------
def test_jpeg_decay_generations(photo):
    one = run(photo, "jpeg_decay", generations=1, quality_min=90, quality_max=90)
    many = run(photo, "jpeg_decay", generations=20, quality_min=10, quality_max=20, downscale=0.6)
    e1 = np.abs(one.astype(int) - photo).mean()
    e2 = np.abs(many.astype(int) - photo).mean()
    assert 0 < e1 < e2


def test_reinterpret_identity_and_swaps(photo):
    ident = run(photo, "reinterpret", width_delta=0, offset=0, channel_order="rgb", layout="interleaved")
    assert np.array_equal(ident, photo)
    bgr = run(photo, "reinterpret", width_delta=0, channel_order="bgr")
    assert np.array_equal(bgr[..., :3], photo[..., 2::-1])
    for layout in ("planar_as_interleaved", "interleaved_as_planar", "as_gray", "as_rgba"):
        for fill in ("wrap", "zero", "source"):
            o = run(photo, "reinterpret", layout=layout, width_delta=-3, offset=-7, fill=fill)
            assert o.shape == photo.shape
    shifted = run(photo, "reinterpret", width_delta=0, offset=3)
    assert np.array_equal(shifted[0, :-1, :3], photo[0, 1:, :3])


# --- drift / displacement / cross-mod -------------------------------------
def test_row_drift_rows_are_shifted_copies(photo):
    out = run(photo, "row_drift", waveform="saw", amplitude=20, frequency=2, edge="wrap")
    for y in range(0, photo.shape[0], 9):
        assert any(np.array_equal(out[y], np.roll(photo[y], k, axis=0)) for k in range(-21, 22))
    assert np.array_equal(run(photo, "row_drift", amplitude=0), photo)
    for wf in ("sine", "triangle", "square", "steps", "noise"):
        assert not np.array_equal(run(photo, "row_drift", waveform=wf, axis="columns"), photo)


def test_displace_map_sources(photo, tmp_path):
    assert np.array_equal(run(photo, "displace_map", amount_x=0, amount_y=0, smooth=0, source="luminance"), photo)
    for src in ("luminance", "edges", "noise"):
        assert not np.array_equal(run(photo, "displace_map", source=src), photo)
    with pytest.raises(EffectError, match="map image"):
        run(photo, "displace_map", source="image", map_file="")
    m = tmp_path / "m.png"
    Image.fromarray(fixtures.gradient(40, 30)).save(m)
    assert not np.array_equal(run(photo, "displace_map", source="image", map_file=str(m)), photo)


def test_cross_mod(photo):
    assert np.array_equal(run(photo, "cross_mod", amount=0.0, mode="remap"), photo)
    inv = run(photo, "cross_mod", mode="threshold", threshold=0.0, amount=1.0, source="luma", destination="b")
    lum_pos = (photo[..., :3].sum(-1) > 0)
    assert np.array_equal(inv[lum_pos][:, 2], 255 - photo[lum_pos][:, 2])
    assert np.array_equal(inv[..., :2], photo[..., :2])
    for mode in ("displace_x", "displace_y", "remap", "ring"):
        o = run(photo, "cross_mod", mode=mode, source="hue", destination="g")
        assert np.array_equal(o[..., [0, 2]], photo[..., [0, 2]])


# --- spectral ---------------------------------------------------------------
def test_spectral_identity_when_neutral(photo):
    out = run(photo, "spectral", gain=1.0, outside_gain=1.0, phase_noise=0.0)
    assert np.abs(out.astype(int) - photo).max() <= 1


def test_spectral_finite_and_symmetry():
    rng = np.random.default_rng(0)
    for w in (16, 17):
        noise = antisymmetrize_columns(rng.uniform(-3, 3, (12, w // 2 + 1)).astype(np.float32), w)
        neg = (-np.arange(12)) % 12
        assert np.allclose(noise[:, 0], -noise[neg, 0])
        if w % 2 == 0:
            assert np.allclose(noise[:, -1], -noise[neg, -1])
    # with odd-symmetric noise the modified half-spectrum is exactly representable:
    noise = antisymmetrize_columns(rng.uniform(-3, 3, (12, 9)).astype(np.float32), 16)
    img = rng.random((12, 16)).astype(np.float32)
    F = np.fft.rfft2(img) * np.exp(1j * noise)
    rec = np.fft.irfft2(F, s=(12, 16))
    assert np.allclose(np.fft.rfft2(rec), F, atol=1e-4)


def test_spectral_modes(photo):
    for kw in ({"channels_mode": "shared"}, {"normalize": "stretch"}, {"sector": 20, "orientation": 45},
               {"band_low": 0.5, "band_high": 0.1}):
        o = run(photo, "spectral", **kw)
        assert np.isfinite(o).all() and not np.array_equal(o, photo)


# --- slit-scan / mosaic ---------------------------------------------------
def test_slitscan(photo):
    ident = run(photo, "slitscan", order="drift", drift=0.0, wave_amount=0, stretch=1.0)
    assert np.array_equal(ident, photo)
    rev = run(photo, "slitscan", order="reverse", strip=1, wave_amount=0)
    assert np.array_equal(rev, photo[:, ::-1])
    for order in ("wave", "random", "interleave", "mirror"):
        for orient in ("vertical", "horizontal"):
            o = run(photo, "slitscan", order=order, orientation=orient, sampling="slit")
            assert o.shape == photo.shape


def test_mosaic_bleed(photo):
    assert np.array_equal(run(photo, "mosaic_bleed", amount=0.0), photo)
    for src in ("neighbour", "repeat", "smear"):
        for d in ("down", "up", "left", "right"):
            o = run(photo, "mosaic_bleed", source=src, direction=d, block=16, amount=0.4)
            assert o.shape == photo.shape


# --- simulations ----------------------------------------------------------
def test_rule_parsing():
    b, s = parse_rule("B3/S23")
    assert list(np.flatnonzero(b)) == [3] and list(np.flatnonzero(s)) == [2, 3]
    assert parse_rule("daynight")[0][6]
    for bad in ("", "B9/S2", "hello", "B3S23"):
        with pytest.raises(ValueError):
            parse_rule(bad)


def test_life_blinker():
    g = np.zeros((5, 5), bool)
    g[2, 1:4] = True
    b, s = parse_rule("B3/S23")
    g2 = life_step(g, b, s)
    assert g2[1:4, 2].all() and g2.sum() == 3
    assert np.array_equal(life_step(g2, b, s), g)


def test_cellular_outputs_and_invalid_rule(photo):
    for out in ("overlay", "darken", "invert", "trails", "displace"):
        for seed_mode in ("luminance", "bitplane", "random"):
            o = run(photo, "cellular", output=out, seed_mode=seed_mode, steps=10)
            assert o.shape == photo.shape
    with pytest.raises(EffectError):
        run(photo, "cellular", rule="X9")


def test_gray_scott_bounded():
    u = np.ones((40, 40), np.float32)
    v = np.zeros((40, 40), np.float32)
    v[18:22, 18:22] = 1.0
    u2, v2 = gray_scott(u, v, 0.0545, 0.062, 600)
    assert np.isfinite(u2).all() and np.isfinite(v2).all()
    assert u2.min() >= 0 and u2.max() <= 1 and v2.min() >= 0 and v2.max() <= 1
    assert v2.std() > 0.01  # a pattern grew


def test_reaction_diffusion_iterations_bounded_and_modes(photo):
    eff = REGISTRY["reaction_diffusion"]
    assert eff.coerce_params({"iterations": 10 ** 9})["iterations"] == 10000
    for blend in ("colorize", "multiply", "screen", "overlay", "difference", "displace"):
        o = run(photo, "reaction_diffusion", blend=blend, iterations=200, sim_size=96)
        assert o.shape == photo.shape


def test_simulations_preview_consistent():
    """Sim grids are defined relative to the image, so preview ~ downscaled export."""
    import cv2
    full = fixtures.photo_like(320, 240)
    prev = cv2.resize(full, (160, 120), interpolation=cv2.INTER_AREA)
    for eid, kw in (("reaction_diffusion", {"iterations": 400, "sim_size": 128}), ("cellular", {"cells": 80})):
        doc = Document.single([EffectInstance.create(eid, seed=3, **kw)], 1)
        a = render_full(full, doc).image
        b = render_document(prev, doc, scale=0.5, full_size=(320, 240)).image
        a_small = cv2.resize(a, (160, 120), interpolation=cv2.INTER_AREA)
        assert np.abs(a_small.astype(int) - b.astype(int)).mean() < 25, eid


# --- contact sheet & self-test ----------------------------------------------
def test_contact_sheet(tmp_path, services):
    from glitchlab.engine.contact_sheet import write_contact_sheet
    path, results = write_contact_sheet(str(tmp_path / "sheet.png"), services=services)
    assert len(results) >= len(REGISTRY)
    assert all(r.status in ("ok", "skipped") for r in results), [(r.key, r.status, r.message) for r in results]
    im = Image.open(path)
    assert im.width > 1000 and im.height > 3000


def test_contact_sheet_labels_failures(tmp_path, monkeypatch):
    from glitchlab.engine.contact_sheet import build_contact_sheet
    from glitchlab.engine.services import EngineServices
    svc = EngineServices(ffmpeg_path=str(tmp_path / "missing-ffmpeg"))
    _, results = build_contact_sheet(svc)
    son = [r for r in results if r.key == "sonify"][0]
    assert son.status == "skipped" and "FFmpeg" in son.message


def test_selftest_passes(services):
    from glitchlab.selftest import run_selftest
    ok, report = run_selftest(services)
    assert ok, report
    assert "RESULT: PASSED" in report
