"""Version 1.2 effects: kaleidoscope, psychedelic pop, dark deco, luminous
gradients, Y2K clear plastic, corporate futurism and the hacker terminal."""

import math

import numpy as np
import pytest

from glitchlab.engine import fixtures
from glitchlab.engine.effects.base import REGISTRY
from glitchlab.engine.effects.geometry import fold_wedge, hex_local
from glitchlab.engine.effects.stylize import field_lines, gradient_map
from glitchlab.engine.effects.terminal import CHARSETS, compose_glyphs, glyph_atlas, hexdump_lines
from glitchlab.engine.model import Document, EffectInstance
from glitchlab.engine.pipeline import render_full
from glitchlab.engine.presets import BUILTIN_PRESETS, apply_preset

V12_EFFECTS = ["kaleidoscope", "psychedelic_pop", "dark_deco", "luminous_gradient", "clear_plastic",
               "corporate_futurism", "hacker_terminal"]
V12_PRESETS = ["Kaleidoscope Bloom", "Acid Kaleidoscope", "Psychedelic Pop Ad", "Candy Pop Ad", "Dark Deco",
               "Emerald Deco Fan", "Luminous Poster", "Dawn Gradient Poster", "Y2K Clear Plastic",
               "Tangerine Jelly", "Moody Corporate Futurism", "Hackers", "Mainframe ASCII", "Hex Dump Portrait"]


def run(img, eid, seed=1, **params):
    return render_full(img, Document.single([EffectInstance.create(eid, seed=seed, **params)], 0)).image


def diff(a, b):
    return np.abs(a.astype(np.int16) - b.astype(np.int16))


@pytest.fixture
def square():
    """Odd-sized square image so the centre falls exactly on a pixel."""
    return fixtures.photo_like(101, 101)


CENTRE = 50 / 101


def test_v12_effects_registered():
    assert set(V12_EFFECTS) <= set(REGISTRY)
    for eid in V12_EFFECTS:
        assert REGISTRY[eid].category == "Abstract & Style"


@pytest.mark.parametrize("eid", V12_EFFECTS)
def test_v12_effects_are_deterministic_and_change_the_image(photo, eid):
    a = run(photo, eid, seed=5)
    b = run(photo, eid, seed=5)
    assert a.shape == photo.shape and a.dtype == np.uint8
    assert np.array_equal(a, b)
    assert diff(a, photo).mean() > 2.0
    np.testing.assert_array_equal(a[..., 3], photo[..., 3])


# ---------------------------------------------------------------- kaleidoscope
def test_fold_wedge_is_invariant_under_two_wedge_rotation_and_reflection():
    # Exact dihedral symmetry needs an even number of mirrored wedges (as with real mirrors).
    rng = np.random.default_rng(0)
    seg = 8
    alpha = 2 * math.pi / seg
    r = rng.uniform(1, 50, 2000)
    th = rng.uniform(-math.pi, math.pi, 2000)
    r0, t0 = fold_wedge(r * np.cos(th), r * np.sin(th), seg, True)
    assert np.all((t0 >= -1e-9) & (t0 <= alpha + 1e-9))
    np.testing.assert_allclose(r0, r, rtol=1e-9)
    # Rotating by two wedges keeps the parity, so the fold is unchanged.
    th2 = th + 2 * alpha
    _, t2 = fold_wedge(r * np.cos(th2), r * np.sin(th2), seg, True)
    # Reflecting across the first wedge boundary (the x axis) mirrors into the same wedge.
    _, t3 = fold_wedge(r * np.cos(-th), r * np.sin(-th), seg, True)
    away = np.minimum(np.mod(th, alpha), alpha - np.mod(th, alpha)) > 1e-6
    np.testing.assert_allclose(t2[away], t0[away], atol=1e-7)
    np.testing.assert_allclose(t3[away], t0[away], atol=1e-7)


def test_kaleidoscope_mirrored_wedges_are_symmetric(square):
    out = run(square, "kaleidoscope", mode="wedge", segments=4, mirrored=True, cx=CENTRE, cy=CENTRE,
              source_x=0.12, source_y=-0.07, zoom=1.0)
    assert diff(out, np.flipud(out)).max() <= 1
    assert diff(out, np.fliplr(out)).max() <= 1
    assert diff(out, np.rot90(out, 2)).max() <= 1
    assert diff(out, square).mean() > 2.0


def test_kaleidoscope_plain_wedges_have_rotational_symmetry(square):
    out = run(square, "kaleidoscope", mode="wedge", segments=4, mirrored=False, cx=CENTRE, cy=CENTRE,
              source_x=0.1, zoom=1.2)
    assert diff(out, np.rot90(out, 1)).max() <= 1


@pytest.mark.parametrize("axis, flips", [("vertical", [np.fliplr]), ("horizontal", [np.flipud]),
                                         ("both", [np.fliplr, np.flipud])])
def test_kaleidoscope_mirror_halves(square, axis, flips):
    out = run(square, "kaleidoscope", mode="mirror", axis=axis, cx=CENTRE, cy=CENTRE)
    for flip in flips:
        assert diff(out, flip(out)).max() <= 1
    # The kept half is untouched.
    if axis == "vertical":
        assert diff(out[:, 50:], square[:, 50:]).max() <= 1


def test_kaleidoscope_identity_settings_keep_one_half(square):
    out = run(square, "kaleidoscope", mode="mirror", axis="vertical", flip_side=True, cx=CENTRE, cy=CENTRE)
    assert diff(out[:, :51], square[:, :51]).max() <= 1


def test_hex_local_stays_inside_the_hexagon():
    rng = np.random.default_rng(1)
    size = 17.0
    x = rng.uniform(-400, 400, 5000)
    y = rng.uniform(-400, 400, 5000)
    lx, ly = hex_local(x, y, size)
    assert np.all(np.hypot(lx, ly) <= size + 1e-6)
    # Hexagon centres map to zero offset.
    cx = size * math.sqrt(3) * (3 + 2 / 2.0)
    cy = size * 1.5 * 2
    zx, zy = hex_local(np.array([cx]), np.array([cy]), size)
    assert abs(zx[0]) < 1e-9 and abs(zy[0]) < 1e-9


@pytest.mark.parametrize("mode", ["hex", "square"])
def test_kaleidoscope_tiles_repeat(mode):
    img = fixtures.photo_like(240, 240)
    out = run(img, "kaleidoscope", mode=mode, tile=40, cx=0.5, cy=0.5)
    if mode == "square":
        # Mirror tiles repeat every two tiles.
        assert diff(out[:, 40:120], out[:, 120:200]).max() <= 1
    assert diff(out, img).mean() > 2.0


def test_kaleidoscope_keeps_alpha_shape(transparent):
    out = run(transparent, "kaleidoscope", mode="wedge", segments=6)
    assert out.shape == transparent.shape
    assert out[..., 3].min() < 255  # alpha is remapped with the colour, not discarded


# ------------------------------------------------------------------- stylize
def test_gradient_map_hits_its_stops():
    t = np.array([[0.0, 0.5, 1.0]], np.float32)
    col = gradient_map(t, ["#000000", "#ff0000", "#ffffff"])
    np.testing.assert_allclose(col[0, 0], [0, 0, 0], atol=1e-6)
    np.testing.assert_allclose(col[0, 1], [1, 0, 0], atol=1e-6)
    np.testing.assert_allclose(col[0, 2], [1, 1, 1], atol=1e-6)


def test_field_lines_fall_on_integer_crossings():
    xs = np.tile(np.arange(100, dtype=np.float32), (5, 1))
    m = field_lines(xs / 10.0, 1.0)
    cols = np.where(m[2] > 0.5)[0]
    assert set(cols) == set(range(0, 100, 10))


def test_psychedelic_pop_uses_at_most_bands_colours(photo):
    out = run(photo, "psychedelic_pop", bands=6, cycles=1.0, detail=0.0, saturation=1.0, outline=0.0, swirl=0.0)
    colours = np.unique(out[..., :3].reshape(-1, 3), axis=0)
    assert 2 <= len(colours) <= 6


def test_psychedelic_pop_palette_and_outline_matter(photo):
    a = run(photo, "psychedelic_pop", palette="rainbow")
    b = run(photo, "psychedelic_pop", palette="lava")
    c = run(photo, "psychedelic_pop", palette="rainbow", outline=0.0)
    assert diff(a, b).mean() > 10
    assert diff(a, c).max() > 50


@pytest.mark.parametrize("pattern", ["sunburst", "fan", "arches", "chevron", "none"])
def test_dark_deco_patterns_are_dark_with_metal_accents(photo, pattern):
    out = run(photo, "dark_deco", pattern=pattern)
    lum = out[..., :3].mean(-1)
    assert np.median(lum) < 110  # mostly dark
    assert lum.max() > 150  # metallic highlights


def test_dark_deco_frame_draws_a_border(photo):
    plain = run(photo, "dark_deco", pattern="none", frame="none", panels=False, vignette=0.0)
    framed = run(photo, "dark_deco", pattern="none", frame="double", panels=False, vignette=0.0)
    assert diff(plain, framed).max() > 60


def test_luminous_gradient_glow_is_brightest_near_its_centre():
    img = fixtures.photo_like(200, 160)
    out = run(img, "luminous_gradient", glow_x=0.25, glow_y=0.3, glow=1.0, halo=0.0, grain=0.0, detail=0.0,
              abstraction=1.0, vignette=0.0)
    lum = out[..., :3].astype(np.float32).mean(-1)
    near = lum[40:56, 42:58].mean()
    far = lum[120:150, 150:190].mean()
    assert near > far + 30


def test_luminous_gradient_grain_is_the_only_seeded_part(photo):
    a = run(photo, "luminous_gradient", seed=1, grain=0.0)
    b = run(photo, "luminous_gradient", seed=2, grain=0.0)
    c = run(photo, "luminous_gradient", seed=2, grain=0.05)
    assert np.array_equal(a, b)
    assert not np.array_equal(b, c)


def test_luminous_gradient_steps_posterise(photo):
    smooth = run(photo, "luminous_gradient", steps=0, grain=0.0, halo=0.0, vignette=0.0)
    stepped = run(photo, "luminous_gradient", steps=4, grain=0.0, halo=0.0, vignette=0.0)
    assert len(np.unique(stepped[..., :3].reshape(-1, 3), axis=0)) < len(
        np.unique(smooth[..., :3].reshape(-1, 3), axis=0))


def test_corporate_futurism_is_cool_and_dark(photo):
    out = run(photo, "corporate_futurism").astype(np.float32)
    assert out[..., 2].mean() > out[..., 0].mean()  # blue over red
    assert out[..., :3].mean() < photo[..., :3].astype(np.float32).mean()


def test_corporate_futurism_overlays_switch_off(photo):
    a = run(photo, "corporate_futurism", grid=False, hud=0, panels=0, scan=0.0, seed=1)
    b = run(photo, "corporate_futurism", grid=False, hud=0, panels=0, scan=0.0, seed=9)
    assert np.array_equal(a, b)
    c = run(photo, "corporate_futurism", grid=True, hud=0, panels=0, scan=0.0, seed=1)
    assert diff(a, c).max() > 20


# ------------------------------------------------------------- clear plastic
@pytest.mark.parametrize("tint, hot, cold", [("bondi", 2, 0), ("tangerine", 0, 2), ("lime", 1, 2)])
def test_clear_plastic_takes_the_tint_colour(photo, tint, hot, cold):
    out = run(photo, "clear_plastic", tint=tint, bubbles=0).astype(np.float32)
    assert out[..., hot].mean() > out[..., cold].mean() + 10


def test_clear_plastic_custom_tint_and_bubbles(photo):
    a = run(photo, "clear_plastic", tint="custom", custom_tint="#ff00ff", bubbles=0).astype(np.float32)
    assert a[..., 0].mean() > a[..., 1].mean() + 10 and a[..., 2].mean() > a[..., 1].mean() + 10
    b1 = run(photo, "clear_plastic", bubbles=30, seed=1)
    b2 = run(photo, "clear_plastic", bubbles=30, seed=2)
    assert not np.array_equal(b1, b2)


# ---------------------------------------------------------- hacker terminal
def test_glyph_atlas_is_deterministic_and_blank_for_space():
    a = glyph_atlas(CHARSETS["ramp"], 8, 13)
    b = glyph_atlas(CHARSETS["ramp"], 8, 13)
    assert a.shape == (len(CHARSETS["ramp"]), 13, 8)
    assert np.array_equal(a, b)
    assert a[0].max() == 0.0
    assert all(a[i].max() > 0.3 for i in range(1, len(a)))
    # The ramp gets denser towards '@'.
    assert a[-1].mean() > a[1].mean()


def test_compose_glyphs_places_tiles():
    atlas = glyph_atlas("01", 6, 10)
    idx = np.array([[0, 1], [1, 0]])
    m = compose_glyphs(atlas, idx, 20, 12)
    assert m.shape == (20, 12)
    assert m[:10, :6].max() == m[10:, 6:].max() == atlas[0].max()
    assert np.array_equal(m[:10, 6:], atlas[1])


def test_hexdump_lines_format():
    lines = hexdump_lines(b"Glitch Lab!", 60, 3)
    assert len(lines) == 3 and all(len(line) == 60 for line in lines)
    assert lines[0].startswith("00000000  47 6c 69 74")
    assert "|Glitch" in lines[0]


@pytest.mark.parametrize("colour, channel", [("green", 1), ("amber", 0), ("cyan", 2)])
def test_hacker_terminal_phosphor_colour(photo, colour, channel):
    out = run(photo, "hacker_terminal", mode="ascii", colour=colour).astype(np.float32)
    means = out[..., :3].mean(axis=(0, 1))
    assert means[channel] == means.max()


@pytest.mark.parametrize("mode", ["ascii", "matrix", "hexdump"])
def test_hacker_terminal_modes(photo, mode):
    a = run(photo, "hacker_terminal", mode=mode, seed=3)
    assert np.array_equal(a, run(photo, "hacker_terminal", mode=mode, seed=3))
    assert a[..., :3].mean() < photo[..., :3].mean()  # terminal screens are dark


def test_hacker_terminal_text_and_seed(photo):
    own = run(photo, "hacker_terminal", mode="hexdump")
    text = run(photo, "hacker_terminal", mode="hexdump", text="HACK THE PLANET")
    assert not np.array_equal(own, text)
    m1 = run(photo, "hacker_terminal", mode="matrix", seed=1)
    m2 = run(photo, "hacker_terminal", mode="matrix", seed=2)
    assert not np.array_equal(m1, m2)


def test_hacker_terminal_original_colours_follow_the_photo():
    img = fixtures.photo_like(160, 120)
    img[..., :3] = (255, 40, 40)
    out = run(img, "hacker_terminal", mode="ascii", colour="original", background=0.0).astype(np.float32)
    lit = out[..., :3].sum(-1) > 60
    assert lit.any()
    assert out[lit][:, 0].mean() > out[lit][:, 1].mean() * 2


# ------------------------------------------------------------------- presets
def test_v12_presets_exist_and_render():
    names = {p["name"]: p for p in BUILTIN_PRESETS}
    img = fixtures.photo_like(96, 72)
    for name in V12_PRESETS:
        assert name in names, name
        out = render_full(img, apply_preset(Document.new(1), names[name])).image
        assert out.shape == img.shape
        assert diff(out, img).mean() > 2.0, name


def test_v12_preset_seeds_are_unique():
    seeds = [fx["seed"] for p in BUILTIN_PRESETS for fx in p["stack"] if "seed" in fx]
    assert len(seeds) == len(set(seeds))
