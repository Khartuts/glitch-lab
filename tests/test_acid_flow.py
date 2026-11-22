"""Acid Flow: flow engine and the effect's guarantees."""

import cv2
import numpy as np
import pytest

from glitchlab.engine import fixtures, flow
from glitchlab.engine.effects.base import REGISTRY
from glitchlab.engine.model import Document, EffectInstance
from glitchlab.engine.pipeline import render_document, render_full
from glitchlab.engine.presets import ACID_PRESETS, apply_preset

MODES = ["contour", "melt", "vortex", "liquid_acid"]


def run(img, seed=3, mix=1.0, **kw):
    inst = EffectInstance.create("acid_flow", seed=seed, **kw)
    inst.mix = mix
    return render_full(img, Document.single([inst], 1)).image


@pytest.fixture
def photo():
    return fixtures.photo_like(240, 180)


# ------------------------------------------------------------------ engine
def test_advection_with_zero_velocity_is_identity():
    z = np.zeros((20, 30), np.float32)
    r = flow.advect(z, z, 10.0, 8, 0.3)
    assert np.abs(r.dx).max() == 0 and np.abs(r.dy).max() == 0


def test_advection_of_a_uniform_field_composes_exactly():
    vx = np.ones((40, 60), np.float32)
    vy = np.zeros((40, 60), np.float32)
    r = flow.advect(vx, vy, 12.0, 6, 0.0)
    # content moved 12 px right: every output samples 12 px to the left
    assert np.allclose(r.dx[:, 20:], -12.0, atol=1e-3) and np.allclose(r.dy, 0.0)


def test_curl_noise_is_divergence_free():
    vx, vy = flow.curl_noise(np.random.default_rng(1), 96, 64, 0.2, 3, 0.5)
    div = np.gradient(vx, axis=1) + np.gradient(vy, axis=0)
    assert np.abs(div[4:-4, 4:-4]).mean() < 0.05 * np.abs(vx).mean()


def test_premultiplied_sampling_has_no_dark_fringes():
    img = np.zeros((40, 40, 4), np.uint8)
    img[:, :20] = (255, 0, 0, 255)            # opaque red next to transparent black
    xs, ys = np.meshgrid(np.arange(40, dtype=np.float32), np.arange(40, dtype=np.float32))
    out = flow.premultiplied_remap(img, xs + 0.5, ys, cv2.BORDER_REFLECT_101)
    edge = out[:, 19]
    assert 100 < edge[0, 3] < 160                # half covered
    assert edge[0, 0] > 250                     # still fully red, not darkened


# ------------------------------------------------------------------ effect
def test_acid_flow_registered_with_grouped_controls():
    eff = REGISTRY["acid_flow"]
    groups = {p.group for p in eff.params}
    assert {"Movement", "Material", "Colour"} <= groups
    names = {p.name for p in eff.params}
    for n in ("strength", "direction", "vortex_size", "vortex_count", "vortex_rotation", "turbulence_scale",
              "turbulence_detail", "viscosity", "steps", "contour", "melt_threshold", "chroma", "iridescence",
              "highlights", "detail", "region", "protect", "vortices", "strokes", "freeze", "colour_mode"):
        assert n in names


@pytest.mark.parametrize("mode", MODES)
def test_modes_are_reproducible_and_valid(photo, mode):
    a = run(photo, mode=mode)
    b = run(photo, mode=mode)
    assert np.array_equal(a, b)
    assert a.dtype == np.uint8 and a.shape == photo.shape
    assert np.abs(a.astype(int) - photo.astype(int)).mean() > 3
    np.testing.assert_array_equal(a[..., 3], photo[..., 3])


def test_zero_mix_returns_the_original(photo):
    assert np.array_equal(run(photo, mix=0.0, mode="liquid_acid", strength=0.3), photo)


def test_protected_pixels_stay_exactly_unchanged(photo):
    prot = {"enabled": True, "shapes": [{"type": "rect", "mode": "add", "x0": 0.3, "y0": 0.3, "x1": 0.7, "y1": 0.7}]}
    out = run(photo, mode="liquid_acid", strength=0.3, colour_mode="neon", protect=prot, feather=0.05)
    h, w = photo.shape[:2]
    inner = (slice(int(0.3 * h) + 1, int(0.7 * h) - 1), slice(int(0.3 * w) + 1, int(0.7 * w) - 1))
    assert np.array_equal(out[inner], photo[inner])
    # the transition is feathered outside the protected area, not a hard seam
    ring = out[int(0.7 * h) + 2:int(0.7 * h) + 6, int(0.4 * w):int(0.6 * w)].astype(int)
    far = out[h - 6:h - 2, int(0.4 * w):int(0.6 * w)].astype(int)
    orig_ring = photo[int(0.7 * h) + 2:int(0.7 * h) + 6, int(0.4 * w):int(0.6 * w)].astype(int)
    orig_far = photo[h - 6:h - 2, int(0.4 * w):int(0.6 * w)].astype(int)
    assert np.abs(ring - orig_ring).mean() < np.abs(far - orig_far).mean()


def test_liquid_area_limits_the_effect(photo):
    region = {"enabled": True, "shapes": [{"type": "rect", "mode": "add", "x0": 0.0, "y0": 0.0, "x1": 0.5, "y1": 1.0}]}
    out = run(photo, mode="vortex", strength=0.2, region=region)
    w = photo.shape[1]
    assert np.array_equal(out[:, int(0.75 * w):], photo[:, int(0.75 * w):])
    assert (out[:, : int(0.4 * w)] != photo[:, : int(0.4 * w)]).any()


def test_full_resolution_has_no_invalid_pixels_or_borders():
    img = fixtures.photo_like(640, 480)
    for mode in MODES:
        out = run(img, mode=mode, strength=0.25, edge="mirror")
        assert (out[..., 3] == 255).all()
        for edge in (out[0], out[-1], out[:, 0], out[:, -1]):
            assert edge[..., :3].max() > 30 and edge[..., :3].std() > 2  # not a flat (black) border
        assert np.abs(out[0, :, :3].astype(int).mean() - out[3, :, :3].astype(int).mean()) < 40


def test_preview_and_export_match_in_scale():
    full = fixtures.photo_like(480, 360)
    doc = Document.single([EffectInstance.create("acid_flow", seed=5, mode="vortex", strength=0.15)], 1)
    a = render_full(full, doc).image
    small = cv2.resize(full, (240, 180), interpolation=cv2.INTER_AREA)
    b = render_document(small, doc, scale=0.5, full_size=(480, 360), full_source=full).image
    ref = cv2.resize(a, (240, 180), interpolation=cv2.INTER_AREA)
    assert np.abs(ref.astype(int) - b.astype(int)).mean() < 14


@pytest.mark.parametrize("affect_alpha", [False, True])
def test_transparent_input_keeps_clean_edges(affect_alpha):
    # an opaque magenta disc with a soft edge on transparent black
    yy, xx = np.mgrid[0:120, 0:160]
    r = np.hypot(xx - 80, yy - 60)
    img = np.zeros((120, 160, 4), np.uint8)
    img[..., 0], img[..., 2] = 230, 200
    img[..., 3] = np.clip((45 - r) * 25, 0, 255).astype(np.uint8)
    img[img[..., 3] == 0, :3] = 0
    inst = EffectInstance.create("acid_flow", seed=2, mode="vortex", strength=0.12, highlights=0.0,
                                 iridescence=0.0, detail=0.0, chroma=0.0, corrosion=0.0, ribbons=0.0,
                                 saturation=1.0)
    inst.affect_alpha = affect_alpha
    out = render_full(img, Document.single([inst], 1)).image
    vis = out[..., 3] > 20
    assert vis.any()
    assert out[vis, 0].min() > 150 and out[vis, 1].max() < 60  # magenta everywhere it is visible


def test_vortices_and_strokes_change_the_flow(photo):
    base = run(photo, mode="vortex", vortex_count=0)
    with_vortex = run(photo, mode="vortex", vortex_count=0, vortices=[[0.5, 0.5, 1, 1]])
    assert not np.array_equal(base, with_vortex)
    stroke = {"enabled": True, "shapes": [{"type": "stroke", "mode": "add", "radius": 0.05,
                                           "points": [[0.1, 0.5], [0.5, 0.5], [0.9, 0.5]]}]}
    assert not np.array_equal(base, run(photo, mode="vortex", vortex_count=0, strokes=stroke))


def test_freeze_keeps_the_flow_but_not_the_colours(photo):
    eff = REGISTRY["acid_flow"]
    p = eff.coerce_params({"mode": "vortex", "strength": 0.15})
    snap = eff.snapshot(p, 3, 1)
    frozen = run(photo, mode="vortex", strength=0.15, freeze=True, frozen=snap)
    assert np.array_equal(frozen, run(photo, mode="vortex", strength=0.15))
    moved = run(photo, mode="melt", strength=0.4, freeze=True, frozen=snap)
    assert np.array_equal(frozen, moved)  # movement settings are frozen
    recoloured = run(photo, mode="melt", strength=0.4, freeze=True, frozen=snap, colour_mode="neon")
    assert not np.array_equal(frozen, recoloured)
    assert eff.disabled_params(eff.coerce_params({"freeze": True, "frozen": snap})) >= {"strength", "mode"}


@pytest.mark.parametrize("preset", ACID_PRESETS, ids=lambda p: p["name"])
def test_acid_presets_render(photo, preset):
    out = render_full(photo, apply_preset(Document.new(1), preset)).image
    assert np.abs(out.astype(int) - photo.astype(int)).mean() > 2
    for fx in preset["stack"]:
        assert set(fx["params"]) <= {p.name for p in REGISTRY["acid_flow"].params}


def test_acid_presets_have_the_requested_names():
    assert [p["name"] for p in ACID_PRESETS] == ["Liquid Acid", "Mercury Skin", "Toxic Ink", "Slow Dissolve",
                                                 "Oil Spill"]
