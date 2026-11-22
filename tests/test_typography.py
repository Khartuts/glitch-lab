"""Lyrics / glitch typography: fonts, layout, styles, text layers, projects, exports."""

import cv2
import numpy as np
import pytest

from glitchlab.engine import fixtures
from glitchlab.engine.model import Document, EffectInstance, Layer
from glitchlab.engine.pipeline import render_document, render_full
from glitchlab.engine.typography import fonts
from glitchlab.engine.typography.extras import (SAMPLE_TEXT, apply_preset, document_fonts, export_typography_png,
                                                load_typography_presets, missing_document_fonts,
                                                save_typography_preset, substitute_font, type_contact_sheet)
from glitchlab.engine.typography.layout import FontPlan, layout_text
from glitchlab.engine.typography.model import (LAYOUT_PRESETS, STYLE_IDS, coerce_box, layout_preset, new_box,
                                               style_preset)
from glitchlab.engine.typography.render import box_report, render_text_layer

LYRIC = "Ne znam, ali — čujem đavola!\n\n  Two spaces, \"quotes\" & 'apostrophes'…\nčćšžđ ČĆŠŽĐ"
STYLES = [s for s in STYLE_IDS if s != "clean"]


@pytest.fixture
def photo():
    return fixtures.photo_like(320, 240)


def text_doc(*boxes, stack=None):
    doc = Document.new(1)
    doc.layers.append(Layer(name="Text", kind="text", boxes=[coerce_box(b) for b in boxes], stack=stack or []))
    doc.active_layer = len(doc.layers) - 1
    return doc


def box(text=LYRIC, **kw):
    base = dict(x=0.5, y=0.5, w=0.9, h=0.8, size=0.07, seed=5, align="center", valign="middle")
    base.update(kw)
    return new_box(text, **base)


# ------------------------------------------------------------------ fonts
def test_bundled_fonts_cover_south_slavic_letters():
    for info in fonts.REGISTRY.bundled():
        for ch in "čćšžđČĆŠŽĐ":
            assert fonts.has_glyph(fonts.font_for_char(info.path, ch), ch), (info.id, ch)


def test_missing_glyphs_fall_back():
    stix = fonts.REGISTRY.get("STIXGeneral|Regular").path
    ch = "Ж"
    if fonts.has_glyph(stix, ch):
        pytest.skip("STIX has this glyph")
    assert fonts.font_for_char(stix, ch) != stix


def test_missing_font_resolves_with_a_warning(photo):
    b = box(font="Imaginary Grotesk|Heavy")
    reps = []
    render_text_layer([b], 320, 240, full_size=(320, 240), reports=reps)
    assert reps[0].missing_font == "Imaginary Grotesk|Heavy"
    assert any("not installed" in w and "Text panel" in w for w in reps[0].warnings)
    doc = text_doc(b)
    assert missing_document_fonts(doc) == ["Imaginary Grotesk|Heavy"]
    assert substitute_font(doc, "Imaginary Grotesk|Heavy", "DejaVu Serif|Book") == 1
    assert missing_document_fonts(doc) == []


def test_imported_font_is_registered(tmp_path):
    src = fonts.REGISTRY.get("DejaVu Serif|Italic").path
    info = fonts.REGISTRY.import_font(src)
    assert info.id == "DejaVu Serif|Italic"
    with pytest.raises(ValueError):
        fonts.REGISTRY.import_font(str(tmp_path / "not-a-font.txt"))


# ------------------------------------------------------------------ layout
def test_layout_places_every_character_in_order():
    p = fonts.REGISTRY.get("DejaVu Sans|Bold").path
    lay = layout_text(LYRIC, FontPlan(p), 30, 280, 400, align="justify")
    placed = "".join(g.ch for g in lay.glyphs)
    assert placed == "".join(c for c in LYRIC if not c.isspace())
    assert [g.index for g in lay.glyphs] == sorted(g.index for g in lay.glyphs)
    assert all(LYRIC[g.index] == g.ch for g in lay.glyphs)


def test_manual_line_breaks_are_kept():
    p = fonts.REGISTRY.get("DejaVu Sans|Book").path
    lay = layout_text("a\nb\n\nc", FontPlan(p), 20, 500, 500)
    lines = {g.ch: g.line for g in lay.glyphs}
    assert lines == {"a": 0, "b": 1, "c": 3}  # the empty line is kept
    flat = layout_text("a\nb\n\nc", FontPlan(p), 20, 500, 500, preserve_breaks=False)
    assert {g.line for g in flat.glyphs} == {0}


def test_overflow_is_reported_not_truncated():
    p = fonts.REGISTRY.get("DejaVu Sans|Bold").path
    lay = layout_text(LYRIC * 3, FontPlan(p), 40, 200, 100)
    assert lay.overflow and lay.warnings
    assert len(lay.glyphs) == sum(1 for c in LYRIC * 3 if not c.isspace())
    fitted = layout_text(LYRIC, FontPlan(p), 40, 300, 300, fit=True, min_size=6)
    assert not fitted.overflow and fitted.size < 40
    tiny = layout_text(LYRIC * 6, FontPlan(p), 40, 100, 60, fit=True, min_size=12)
    assert tiny.overflow and tiny.size == 12 and any("minimum" in w for w in tiny.warnings)


def test_box_report_for_the_ui():
    rep = box_report(box(LYRIC * 4, size=0.2, fit=False), 320, 240)
    assert rep.overflow


@pytest.mark.parametrize("name", [n for n, _ in LAYOUT_PRESETS])
def test_layout_presets_keep_the_text(name, photo):
    b = layout_preset(name, box())
    assert b["text"] == LYRIC
    out = render_full(photo, text_doc(b)).image
    assert (out != photo).any()


# ------------------------------------------------------------------ styles
@pytest.mark.parametrize("style", STYLES)
def test_styles_are_seeded_and_distinct(style):
    a = render_text_layer([style_preset(style, box(seed=3))], 320, 240, full_size=(320, 240))
    b = render_text_layer([style_preset(style, box(seed=3))], 320, 240, full_size=(320, 240))
    c = render_text_layer([style_preset(style, box(seed=4))], 320, 240, full_size=(320, 240))
    clean = render_text_layer([box(seed=3)], 320, 240, full_size=(320, 240))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert not np.array_equal(a, clean)


@pytest.mark.parametrize("style", [s for s in STYLES if s not in ("cut",)])
def test_zero_damage_gives_clean_typography(style):
    b = box(style=style, intensity=0.0)
    clean = box(style="clean")
    assert np.array_equal(render_text_layer([b], 320, 240, full_size=(320, 240)),
                          render_text_layer([clean], 320, 240, full_size=(320, 240)))


def test_styles_are_pairwise_different():
    outs = {s: render_text_layer([style_preset(s, box(seed=3))], 320, 240, full_size=(320, 240)) for s in STYLES}
    for i, s1 in enumerate(STYLES):
        for s2 in STYLES[i + 1:]:
            assert np.abs(outs[s1].astype(int) - outs[s2].astype(int)).mean() > 1.0, (s1, s2)


def test_distress_mask_limits_damage():
    region = {"enabled": True, "shapes": [{"type": "rect", "mode": "add", "x0": 0.0, "y0": 0.0, "x1": 0.5,
                                           "y1": 1.0}]}
    dam = render_text_layer([box(style="thrashed", distress=region)], 320, 240, full_size=(320, 240))
    clean = render_text_layer([box(style="clean")], 320, 240, full_size=(320, 240))
    right = slice(200, 320)
    assert np.abs(dam[:, right].astype(int) - clean[:, right].astype(int)).mean() < 0.5
    assert np.abs(dam[:, :140].astype(int) - clean[:, :140].astype(int)).mean() > 1.0


def test_word_override_changes_only_that_word():
    b = box("one two three", size=0.12, h=0.4)
    plain = render_text_layer([b], 320, 240, full_size=(320, 240))
    b2 = dict(b, overrides=[{"scope": "word", "index": 1, "style": "transmission", "intensity": 1.5}])
    over = render_text_layer([coerce_box(b2)], 320, 240, full_size=(320, 240))
    diff = np.abs(over.astype(int) - plain.astype(int)).sum(-1)
    xs = np.nonzero(diff.sum(0))[0]
    assert len(xs) and 60 < xs.mean() < 260  # the middle word


def test_readability_keeps_a_clean_layer():
    hard = box(style="ghost", intensity=2.0)
    soft = box(style="ghost", intensity=2.0, readability=1.0)
    a = render_text_layer([hard], 320, 240, full_size=(320, 240))
    b = render_text_layer([soft], 320, 240, full_size=(320, 240))
    assert (b[..., 3] > 128).sum() > (a[..., 3] > 128).sum()


def test_transmission_substitution_is_decorative():
    b = box(style="transmission")
    assert b["style_params"]["transmission"]["substitution"] == 0.0  # off by default
    b["style_params"]["transmission"]["substitution"] = 0.5
    b = coerce_box(b)
    render_text_layer([b], 320, 240, full_size=(320, 240))
    assert b["text"] == LYRIC


def test_cut_and_paste_keeps_order_and_honours_locks():
    b = style_preset("cut", box("alpha beta gamma", size=0.1))
    a = render_text_layer([b], 320, 240, full_size=(320, 240))
    locked = dict(b)
    locked["style_params"] = {**b["style_params"], "cut": {**b["style_params"]["cut"],
                                                          "locks": {"1": {"paper": 2, "rot": 0.0}}}}
    l1 = render_text_layer([coerce_box(dict(locked, seed=1))], 320, 240, full_size=(320, 240))
    l2 = render_text_layer([coerce_box(dict(locked, seed=2))], 320, 240, full_size=(320, 240))
    assert not np.array_equal(l1, l2)  # reroll changes the unlocked fragments
    assert a.shape == l1.shape


def test_decorative_repetition_is_separate():
    b = box("echo", repeat="texture", repeat_count=20, repeat_opacity=0.3)
    plain = box("echo")
    a = render_text_layer([b], 320, 240, full_size=(320, 240))
    p = render_text_layer([plain], 320, 240, full_size=(320, 240))
    assert (a[..., 3] > 0).sum() > (p[..., 3] > 0).sum()
    assert b["text"] == "echo"


def test_text_follows_a_path():
    b = layout_preset("path", box("along the curve, " * 3))
    out = render_text_layer([b], 320, 240, full_size=(320, 240))
    ys, xs = np.nonzero(out[..., 3] > 100)
    assert xs.max() - xs.min() > 100 and ys.max() - ys.min() > 15


def test_behind_subject_with_a_manual_mask():
    m = {"enabled": True, "shapes": [{"type": "rect", "mode": "add", "x0": 0.4, "y0": 0.0, "x1": 0.6, "y1": 1.0}]}
    b = box("WALL OF TEXT " * 4, behind=True, subject_mask=m, size=0.1)
    out = render_text_layer([b], 320, 240, full_size=(320, 240))
    assert out[:, 140:180, 3].max() == 0
    assert out[:, :120, 3].max() > 200


# ------------------------------------------------------------------ pipeline
def test_editing_styled_text_updates_the_result(photo):
    b = style_preset("rust", box())
    a = render_full(photo, text_doc(b)).image
    b2 = dict(b, text=LYRIC + "!")
    assert not np.array_equal(a, render_full(photo, text_doc(b2)).image)


def test_text_layer_effects_leave_the_photo_unchanged(photo):
    b = box("TEXT", size=0.15, w=0.5, h=0.3)
    doc = text_doc(b, stack=[EffectInstance.create("channel_shift", seed=3)])
    out = render_full(photo, doc).image
    alpha = render_text_layer([b], 320, 240, full_size=(320, 240))[..., 3]
    far = cv2.dilate((alpha > 0).astype(np.uint8), np.ones((25, 25), np.uint8)) == 0
    assert np.array_equal(out[far], photo[far])


def test_process_photo_and_text_together(photo):
    b = box("TEXT", size=0.15)
    doc = text_doc(b)
    doc.layers.append(Layer(name="Glitch", stack=[EffectInstance.create("scanline_tear", seed=2)], source="below"))
    separate = render_full(photo, text_doc(b)).image
    together = render_full(photo, doc).image
    assert not np.array_equal(separate, together)


def test_transparent_typography_export_has_clean_edges(photo, tmp_path):
    b = box("Clean edges", fill="#ff8800", size=0.12, outline_width=0.0)
    res = render_full(photo, text_doc(b), text_only=True).image
    semi = (res[..., 3] > 10) & (res[..., 3] < 245)
    assert semi.any()
    assert np.abs(res[semi, :3].astype(int) - [255, 136, 0]).max() <= 3  # no dark fringe
    assert res[0, 0, 3] == 0
    p = export_typography_png(photo, text_doc(b), str(tmp_path / "type.png"))
    from glitchlab.engine.image_io import load_image

    assert load_image(p).rgba[..., 3].min() == 0


def test_full_resolution_keeps_the_layout():
    full = fixtures.photo_like(640, 480)
    b = style_preset("thrashed", box(seed=9))
    big = render_full(full, text_doc(b)).image
    small = render_document(cv2.resize(full, (320, 240), interpolation=cv2.INTER_AREA), text_doc(b), scale=0.5,
                            full_size=(640, 480), full_source=full).image
    a = render_text_layer([b], 640, 480, full_size=(640, 480))[..., 3]
    s = render_text_layer([b], 320, 240, full_size=(640, 480))[..., 3]
    a_s = cv2.resize(a, (320, 240), interpolation=cv2.INTER_AREA)
    inter = ((a_s > 128) & (s > 128)).sum()
    union = ((a_s > 128) | (s > 128)).sum()
    assert inter / union > 0.75
    assert big.shape == (480, 640, 4) and small.shape == (240, 320, 4)


def test_project_roundtrip_preserves_text_exactly(tmp_path, photo):
    from glitchlab.engine.image_io import LoadedImage
    from glitchlab.engine.project import load_project, project_from_image, save_project

    b = style_preset("xerox", box(LYRIC))
    doc = text_doc(b)
    li = LoadedImage(rgba=photo, path=None, format="PNG", original_mode="RGBA")
    path = tmp_path / "lyrics.glab"
    save_project(str(path), project_from_image(li, doc))
    loaded = load_project(str(path))
    tl = [layer for layer in loaded.document.layers if layer.kind == "text"][0]
    assert tl.boxes[0]["text"] == LYRIC
    assert tl.boxes[0]["style"] == "xerox" and tl.boxes[0]["seed"] == b["seed"]
    assert np.array_equal(render_full(photo, doc).image, render_full(photo, loaded.document).image)
    assert document_fonts(loaded.document) == {"DejaVu Serif|Bold"}


def test_typography_presets_save_and_apply(tmp_path):
    b = style_preset("industrial", box("keep me", x=0.2, y=0.3))
    save_typography_preset(b, "Label strip", tmp_path)
    pres = load_typography_presets(tmp_path)
    assert [p["name"] for p in pres] == ["Label strip"]
    other = box("other words", x=0.7, y=0.8)
    applied = apply_preset(other, pres[0])
    assert applied["text"] == "other words" and applied["x"] == 0.7 and applied["style"] == "industrial"


def test_contact_sheet(tmp_path):
    p = type_contact_sheet(str(tmp_path / "sheet.png"), fixtures.photo_like(320, 240))
    from glitchlab.engine.image_io import load_image

    img = load_image(p).rgba
    assert img.shape[1] == 3 * 320 and img.shape[0] == 3 * 240
    assert "čćšžđ" in SAMPLE_TEXT
