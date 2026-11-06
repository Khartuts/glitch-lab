"""Properties every registered effect must satisfy."""

import numpy as np
import pytest

from glitchlab.engine.effects.base import REGISTRY
from glitchlab.engine.model import Document, EffectInstance
from glitchlab.engine.pipeline import render_document, render_full

from conftest import ffmpeg_available

ALL_EFFECTS = sorted(REGISTRY)


def _skip_if_needs_ffmpeg(eid):
    if REGISTRY[eid].requires_ffmpeg and not ffmpeg_available():
        pytest.skip("FFmpeg not available")


@pytest.mark.parametrize("eid", ALL_EFFECTS)
def test_zero_strength_returns_input(eid, photo):
    inst = EffectInstance.create(eid, seed=11)
    inst.mix = 0.0
    out = render_full(photo, Document.single([inst], 3)).image
    assert np.array_equal(out, photo)


@pytest.mark.parametrize("eid", ALL_EFFECTS)
def test_bypass_returns_input(eid, photo):
    inst = EffectInstance.create(eid, seed=11)
    inst.enabled = False
    out = render_full(photo, Document.single([inst], 3)).image
    assert np.array_equal(out, photo)


@pytest.mark.parametrize("eid", ALL_EFFECTS)
def test_default_changes_pixels_and_is_valid(eid, photo, services):
    _skip_if_needs_ffmpeg(eid)
    inst = EffectInstance.create(eid, seed=11)
    r = render_full(photo, Document.single([inst], 3), services=services)
    assert r.image.dtype == np.uint8
    assert r.image.shape == photo.shape
    assert not np.array_equal(r.image, photo), f"{eid} default parameters did nothing"
    assert np.array_equal(r.image[..., 3], photo[..., 3])


@pytest.mark.parametrize("eid", ALL_EFFECTS)
def test_deterministic(eid, photo, services):
    _skip_if_needs_ffmpeg(eid)
    inst = EffectInstance.create(eid, seed=12345)
    doc = Document.single([inst], 99)
    a = render_full(photo, doc, services=services).image
    b = render_full(photo, Document.from_dict(doc.to_dict()), services=services).image
    assert np.array_equal(a, b)


@pytest.mark.parametrize("eid", ALL_EFFECTS)
def test_source_alpha_preserved_by_default(eid, transparent, services):
    _skip_if_needs_ffmpeg(eid)
    inst = EffectInstance.create(eid, seed=5)
    out = render_full(transparent, Document.single([inst], 1), services=services).image
    assert np.array_equal(out[..., 3], transparent[..., 3])


@pytest.mark.parametrize("eid", ALL_EFFECTS)
def test_half_mix_is_between(eid, photo, services):
    _skip_if_needs_ffmpeg(eid)
    full = EffectInstance.create(eid, seed=8)
    half = EffectInstance.from_dict(full.to_dict())
    half.mix = 0.5
    a = render_full(photo, Document.single([full], 2), services=services).image.astype(int)
    m = render_full(photo, Document.single([half], 2), services=services).image.astype(int)
    expect = (photo.astype(int) + a) / 2.0
    assert np.abs(m - expect).max() <= 1.0


@pytest.mark.parametrize("eid", ALL_EFFECTS)
def test_preview_scale_runs(eid, services):
    """Effects run on odd, tiny and reduced-scale images without errors."""
    _skip_if_needs_ffmpeg(eid)
    from glitchlab.engine import fixtures

    img = fixtures.gradient(37, 23)
    inst = EffectInstance.create(eid, seed=4)
    r = render_document(img, Document.single([inst], 1), scale=0.25, full_size=(148, 92),
                        services=services, on_error="raise")
    assert r.image.shape == img.shape


@pytest.mark.parametrize("eid", ALL_EFFECTS)
def test_params_coerce_garbage(eid):
    eff = REGISTRY[eid]
    garbage = {p.name: "not-a-number" if p.kind in ("float", "int") else None for p in eff.params}
    out = eff.coerce_params(garbage)
    assert set(out) == {p.name for p in eff.params}
    out2 = eff.coerce_params({p.name: float("nan") for p in eff.params if p.kind in ("float", "int")})
    for p in eff.params:
        if p.kind in ("float", "int"):
            assert np.isfinite(out2[p.name])
