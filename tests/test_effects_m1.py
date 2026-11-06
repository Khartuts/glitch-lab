"""Milestone 1 effects: specific behaviour."""

import numpy as np

from glitchlab.engine import fixtures
from glitchlab.engine.effects.base import REGISTRY
from glitchlab.engine.model import Document, EffectInstance
from glitchlab.engine.ops import edge_index, rgb_to_ycbcr
from glitchlab.engine.pipeline import render_document, render_full


def run(img, eid, seed=1, gseed=0, **params):
    inst = EffectInstance.create(eid, seed=seed, **params)
    return render_full(img, Document.single([inst], gseed)).image


def test_edge_index_modes():
    idx = np.arange(-3, 8)
    assert list(edge_index(idx, 5, "wrap")) == [2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2]
    assert list(edge_index(idx, 5, "clamp")) == [0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4]
    assert list(edge_index(idx, 5, "mirror")) == [2, 1, 0, 0, 1, 2, 3, 4, 4, 3, 2]


def test_channel_shift_wrap_is_roll(photo):
    out = run(photo, "channel_shift", r_dx=7, r_dy=-3, g_dx=0, g_dy=0, b_dx=0, b_dy=5, edge="wrap")
    assert np.array_equal(out[..., 0], np.roll(np.roll(photo[..., 0], 7, axis=1), -3, axis=0))
    assert np.array_equal(out[..., 1], photo[..., 1])
    assert np.array_equal(out[..., 2], np.roll(photo[..., 2], 5, axis=0))


def test_channel_shift_clamp_edges(photo):
    out = run(photo, "channel_shift", r_dx=10, r_dy=0, g_dx=0, b_dx=0, b_dy=0, edge="clamp")
    assert np.array_equal(out[:, :10, 0], np.repeat(photo[:, :1, 0], 10, axis=1))
    assert np.array_equal(out[:, 10:, 0], photo[:, :-10, 0])


def test_channel_shift_affects_alpha_only_when_enabled(transparent):
    inst = EffectInstance.create("channel_shift", seed=0, r_dx=0, b_dx=0, b_dy=0, a_dx=9, a_dy=0, edge="wrap")
    out = render_full(transparent, Document.single([inst])).image
    assert np.array_equal(out[..., 3], transparent[..., 3])
    inst.affect_alpha = True
    doc = Document.single([inst])
    assert doc.effective_alpha_mode() == "composite"
    out = render_full(transparent, doc).image
    assert np.array_equal(out[..., 3], np.roll(transparent[..., 3], 9, axis=1))


def test_spatial_params_scale_with_preview():
    full = fixtures.checkerboard(200, 100, cell=20)
    half = full[::2, ::2].copy()
    inst = EffectInstance.create("channel_shift", seed=0, r_dx=20, r_dy=0, g_dx=0, b_dx=0, b_dy=0)
    r = render_document(half, Document.single([inst]), scale=0.5, full_size=(200, 100)).image
    assert np.array_equal(r[..., 0], np.roll(half[..., 0], 10, axis=1))


def test_scanline_tear_rows_are_shifted_copies(photo):
    out = run(photo, "scanline_tear", seed=3, bands=20, displacement=40, jitter=0, rgb_split=0, edge="wrap")
    changed = 0
    for y in range(photo.shape[0]):
        row, src = out[y], photo[y]
        if np.array_equal(row, src):
            continue
        changed += 1
        assert any(np.array_equal(row, np.roll(src, k, axis=0)) for k in range(-40, 41)), y
    assert changed > 0


def test_scanline_seed_changes_result(photo):
    a = run(photo, "scanline_tear", seed=1)
    b = run(photo, "scanline_tear", seed=2)
    c = run(photo, "scanline_tear", seed=1)
    assert not np.array_equal(a, b)
    assert np.array_equal(a, c)


def test_global_seed_changes_result(photo):
    a = run(photo, "scanline_tear", seed=1, gseed=1)
    b = run(photo, "scanline_tear", seed=1, gseed=2)
    assert not np.array_equal(a, b)


def _multiset_equal(a, b):
    return np.array_equal(np.sort(a.reshape(-1, 4).view(np.uint32).ravel()),
                          np.sort(b.reshape(-1, 4).view(np.uint32).ravel()))


def test_pixel_sort_preserves_row_contents(photo):
    out = run(photo, "pixel_sort", direction="horizontal", lower=0.2, upper=0.9)
    assert not np.array_equal(out, photo)
    for y in range(photo.shape[0]):
        assert _multiset_equal(out[y], photo[y])


def test_pixel_sort_vertical_preserves_columns(photo):
    out = run(photo, "pixel_sort", direction="vertical", lower=0.0, upper=1.0)
    for x in range(photo.shape[1]):
        assert _multiset_equal(out[:, x], photo[:, x])
    # full-interval vertical sort makes every column monotonic in luminance
    from glitchlab.engine.ops import luminance
    lum = np.rint(luminance(out) * 65535)
    assert (np.diff(lum, axis=0) >= 0).all()


def test_pixel_sort_diagonal_preserves_image(photo):
    for d in ("diag_down", "diag_up"):
        out = run(photo, "pixel_sort", direction=d, lower=0.0, upper=1.0)
        assert _multiset_equal(out, photo)
        assert not np.array_equal(out, photo)


def test_pixel_sort_max_span_and_reverse(photo):
    a = run(photo, "pixel_sort", lower=0.0, upper=1.0, max_span=10)
    b = run(photo, "pixel_sort", lower=0.0, upper=1.0, max_span=10, reverse=True)
    assert not np.array_equal(a, b)
    for y in range(0, photo.shape[0], 7):
        for x0 in range(0, photo.shape[1] - 10, 10):
            assert _multiset_equal(a[y, x0:x0 + 10], photo[y, x0:x0 + 10])


def test_pixel_sort_skips_transparent(transparent):
    out = run(transparent, "pixel_sort", lower=0.0, upper=1.0)
    clear = transparent[..., 3] == 0
    assert np.array_equal(out[clear], transparent[clear])


def test_block_scramble_shuffle_permutes_tiles():
    img = fixtures.photo_like(128, 96)
    out = run(img, "block_scramble", seed=9, tile=16, amount=0.5, mode="shuffle")
    assert not np.array_equal(out, img)

    def tiles(a):
        return sorted(a[y:y + 16, x:x + 16].tobytes() for y in range(0, 96, 16) for x in range(0, 128, 16))

    assert tiles(out) == tiles(img)


def test_block_scramble_modes_run(photo):
    for mode in ("shuffle", "duplicate", "rotate", "offset", "mixed"):
        out = run(photo, "block_scramble", seed=2, tile=20, amount=0.6, mode=mode)
        assert not np.array_equal(out, photo), mode


def test_channel_target_single_channel(photo):
    inst = EffectInstance.create("pixel_sort", seed=0, lower=0.0, upper=1.0)
    inst.channels = "r"
    out = render_full(photo, Document.single([inst])).image
    assert np.array_equal(out[..., 1:], photo[..., 1:])
    assert not np.array_equal(out[..., 0], photo[..., 0])


def test_channel_target_luminance_keeps_chroma(photo):
    inst = EffectInstance.create("scanline_tear", seed=4, displacement=60)
    inst.channels = "luma"
    out = render_full(photo, Document.single([inst])).image
    a, b = rgb_to_ycbcr(out), rgb_to_ycbcr(photo)
    assert np.abs(a[..., 1:] - b[..., 1:]).mean() < 3.0
    assert np.abs(a[..., 0] - b[..., 0]).mean() > 3.0


def test_reordering_changes_processing_order(photo):
    a = EffectInstance.create("pixel_sort", seed=1, lower=0.3, upper=0.9)
    b = EffectInstance.create("block_scramble", seed=2, tile=16, amount=0.5)
    ab = render_full(photo, Document.single([a, b])).image
    ba = render_full(photo, Document.single([b, a])).image
    assert not np.array_equal(ab, ba)


def test_all_m1_effects_registered():
    for eid in ("channel_shift", "scanline_tear", "pixel_sort", "block_scramble"):
        assert eid in REGISTRY
