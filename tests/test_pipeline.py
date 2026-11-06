"""Pipeline, compositing, masks, history, presets."""

import numpy as np
import pytest

from glitchlab.engine import fixtures
from glitchlab.engine.cancel import CancelToken
from glitchlab.engine.compositing import blend_function, composite_layers
from glitchlab.engine.errors import Cancelled
from glitchlab.engine.history import History
from glitchlab.engine.masks import rasterize_mask
from glitchlab.engine.model import Document, EffectInstance, Layer, MaskSpec
from glitchlab.engine.pipeline import RenderCache, make_preview_source, render_document, render_full
from glitchlab.engine.presets import (
    BUILTIN_PRESETS,
    apply_preset,
    available_builtin_presets,
    load_user_presets,
    save_user_preset,
    stack_to_preset,
    validate_preset,
)
from glitchlab.engine.effects.base import REGISTRY


def test_blend_modes_formulas():
    cb = np.array([0.2, 0.5, 0.8], np.float32)
    cs = np.array([0.6, 0.6, 0.6], np.float32)
    assert np.allclose(blend_function(cb, cs, "multiply"), cb * cs)
    assert np.allclose(blend_function(cb, cs, "screen"), cb + cs - cb * cs)
    assert np.allclose(blend_function(cb, cs, "difference"), np.abs(cb - cs))
    ov = blend_function(cb, cs, "overlay")
    assert np.allclose(ov[0], 2 * 0.2 * 0.6)
    assert np.allclose(ov[2], 1 - 2 * (1 - 0.8) * (1 - 0.6))


def test_layer_opacity_mixes_original_and_processed(photo):
    fx = EffectInstance.create("channel_shift", seed=0, r_dx=10)
    processed = render_full(photo, Document.single([fx])).image
    doc = Document.new(1)
    doc.layers[1].stack = [fx]
    doc.layers[1].opacity = 0.5
    out = render_full(photo, doc).image.astype(int)
    expect = (photo.astype(int) + processed.astype(int)) / 2
    assert np.abs(out - expect).max() <= 1


@pytest.mark.parametrize("mode", ["normal", "multiply", "screen", "difference", "overlay"])
def test_blend_modes_render(photo, mode):
    doc = Document.new(1)
    doc.layers[1].stack = [EffectInstance.create("channel_shift", seed=0, r_dx=10, b_dx=-6)]
    doc.layers[1].blend = mode
    out = render_full(photo, doc).image
    assert out.shape == photo.shape
    if mode == "difference":
        a = photo[..., :3].astype(int)
        b = render_full(photo, Document.single(doc.layers[1].stack)).image[..., :3].astype(int)
        assert np.abs(out[..., :3].astype(int) - np.abs(a - b)).max() <= 1


def test_composite_transparent_source_over_layers():
    img = fixtures.transparent_circle(64, 48)
    out = composite_layers([img, img], ["normal", "normal"], [1.0, 1.0], [None, None])
    a = img[..., 3].astype(float) / 255
    expect = np.rint((a + a * (1 - a)) * 255)
    assert np.abs(out[..., 3].astype(float) - expect).max() <= 1


def test_alpha_mode_auto_preserves_source_alpha():
    img = fixtures.transparent_circle(64, 48)
    doc = Document.new(1)
    doc.layers[1].stack = [EffectInstance.create("pixel_sort", seed=0)]
    out = render_full(img, doc).image
    assert np.array_equal(out[..., 3], img[..., 3])
    doc.alpha_mode = "composite"
    out2 = render_full(img, doc).image
    assert not np.array_equal(out2[..., 3], img[..., 3])


def test_hidden_layer_ignored(photo):
    doc = Document.new(1)
    doc.layers[1].stack = [EffectInstance.create("channel_shift", seed=0, r_dx=10)]
    doc.layers[1].visible = False
    assert np.array_equal(render_full(photo, doc).image, photo)


def test_rect_mask_limits_effect(photo):
    doc = Document.new(1)
    doc.layers[1].stack = [EffectInstance.create("channel_shift", seed=0, r_dx=10, b_dx=-10)]
    doc.layers[1].mask = MaskSpec(enabled=True, shapes=[{"type": "rect", "mode": "add", "x0": 0.5, "y0": 0.0, "x1": 1.0, "y1": 1.0}])
    out = render_full(photo, doc).image
    w = photo.shape[1]
    assert np.array_equal(out[:, : w // 2 - 1], photo[:, : w // 2 - 1])
    assert not np.array_equal(out[:, w // 2 + 1:], photo[:, w // 2 + 1:])
    doc.layers[1].mask.invert = True
    out2 = render_full(photo, doc).image
    assert np.array_equal(out2[:, w // 2 + 1:], photo[:, w // 2 + 1:])


def test_mask_rasterisation_resolution_independent():
    spec = MaskSpec(enabled=True, feather=0.02, shapes=[
        {"type": "stroke", "mode": "add", "points": [[0.1, 0.1], [0.9, 0.8]], "radius": 0.05},
        {"type": "rect", "mode": "subtract", "x0": 0.4, "y0": 0.0, "x1": 0.6, "y1": 1.0}])
    big = rasterize_mask(spec, 400, 300)
    small = rasterize_mask(spec, 100, 75)
    import cv2
    down = cv2.resize(big, (100, 75), interpolation=cv2.INTER_AREA)
    assert np.abs(down - small).mean() < 0.03
    assert big.min() >= 0 and big.max() <= 1
    assert big[150, 200] < 0.05  # subtracted centre band


def test_luminance_mask(photo):
    spec = MaskSpec(enabled=True, luma_enabled=True, luma_low=0.6, luma_high=1.0, luma_softness=0.0)
    m = rasterize_mask(spec, photo.shape[1], photo.shape[0], photo)
    from glitchlab.engine.ops import luminance
    lum = luminance(photo)
    assert (m[lum > 0.62] > 0.99).all()
    assert (m[lum < 0.58] < 0.01).all()


def test_cache_reuses_prefix(photo):
    cache = RenderCache()
    a = EffectInstance.create("pixel_sort", seed=1)
    b = EffectInstance.create("channel_shift", seed=1, r_dx=4)
    doc = Document.single([a, b])
    r1 = render_document(photo, doc, cache=cache)
    assert r1.cache_hits == 0
    doc.layers[0].stack[1].params["r_dx"] = 8
    r2 = render_document(photo, doc, cache=cache)
    assert r2.cache_hits == 1
    r3 = render_document(photo, doc)
    assert np.array_equal(r2.image, r3.image)


def test_cache_is_bounded():
    cache = RenderCache(max_bytes=1000)
    for i in range(10):
        cache.put(str(i), np.zeros(300, np.uint8))
    assert cache.nbytes <= 1000


def test_cancellation_then_next_render_succeeds(photo):
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(Cancelled):
        render_document(photo, Document.single([EffectInstance.create("pixel_sort", seed=1)]), cancel=tok)
    r = render_document(photo, Document.single([EffectInstance.create("pixel_sort", seed=1)]), cancel=CancelToken())
    assert r.image.shape == photo.shape


def test_preview_source_scale():
    img = fixtures.gradient(400, 200)
    prev, s = make_preview_source(img, 100)
    assert prev.shape[:2] == (50, 100)
    assert abs(s - 0.25) < 1e-9
    same, s1 = make_preview_source(img, 1000)
    assert same is img and s1 == 1.0


def test_document_roundtrip_json(photo):
    doc = Document.new(77)
    doc.layers[1].stack = [EffectInstance.create("pixel_sort", seed=5, max_span=40),
                           EffectInstance.create("block_scramble", seed=6)]
    doc.layers[1].stack[0].locked_params = ["lower"]
    doc.layers[1].mask = MaskSpec(enabled=True, shapes=[{"type": "rect", "mode": "add", "x0": 0, "y0": 0, "x1": 0.5, "y1": 0.5}])
    doc2 = Document.from_json(doc.to_json())
    assert doc2.to_dict() == doc.to_dict()
    assert np.array_equal(render_full(photo, doc).image, render_full(photo, doc2).image)


def test_history_undo_redo_and_coalescing():
    h = History(limit=5)
    h.reset({"v": 0})
    h.push({"v": 1}, "a")
    h.push({"v": 2}, "slider", coalesce_key="k")
    h.push({"v": 3}, "slider", coalesce_key="k")
    assert h.undo() == {"v": 1}
    assert h.redo() == {"v": 3}
    assert not h.can_redo
    for i in range(10):
        h.push({"v": 10 + i}, f"x{i}")
    assert len(h) == 5
    assert h.current() == {"v": 19}
    h.undo()
    h.push({"v": 100}, "branch")
    assert not h.can_redo


def test_builtin_presets_valid_params():
    for preset in BUILTIN_PRESETS:
        for fx in preset["stack"]:
            eff = REGISTRY.get(fx["effect"])
            if eff is None:
                continue  # checked by test_all_presets_available once all effects exist
            names = {p.name for p in eff.params}
            unknown = set(fx["params"]) - names
            assert not unknown, f"{preset['name']}: {fx['effect']} has unknown params {unknown}"


def test_presets_apply_and_render(photo):
    presets = available_builtin_presets()
    assert len(presets) >= 3
    for preset in presets:
        doc = apply_preset(Document.new(1), preset)
        assert len(doc.layer.stack) == len(preset["stack"])
        assert [e.seed for e in doc.layer.stack] == [fx["seed"] for fx in preset["stack"]]
        a = render_full(photo, doc).image
        b = render_full(photo, apply_preset(Document.new(1), preset)).image
        assert np.array_equal(a, b), preset["name"]
        assert not np.array_equal(a, photo), preset["name"]


def test_user_preset_roundtrip(tmp_path, photo):
    stack = [EffectInstance.create("pixel_sort", seed=3, max_span=30),
             EffectInstance.create("channel_shift", seed=4)]
    preset = stack_to_preset("My Test", stack, "desc")
    save_user_preset(preset, tmp_path)
    loaded = load_user_presets(tmp_path)
    assert len(loaded) == 1 and loaded[0]["name"] == "My Test"
    validate_preset(loaded[0])
    d1 = Document.single(stack, 5)
    d2 = apply_preset(Document.single([], 5), loaded[0])
    assert np.array_equal(render_full(photo, d1).image, render_full(photo, d2).image)


def test_on_error_skip_reports_failure(photo, monkeypatch):
    from glitchlab.engine.errors import EffectError
    eff = REGISTRY["block_scramble"]

    def boom(*a, **k):
        raise EffectError("synthetic failure")

    monkeypatch.setattr(eff, "process", boom)
    doc = Document.single([EffectInstance.create("block_scramble", seed=1),
                           EffectInstance.create("channel_shift", seed=1, r_dx=3)])
    r = render_document(photo, doc, on_error="skip")
    assert len(r.failures) == 1 and "synthetic" in r.failures[0].message
    assert not np.array_equal(r.image, photo)
    with pytest.raises(EffectError):
        render_document(photo, doc, on_error="raise")
