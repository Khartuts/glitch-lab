"""UI tests using the Qt offscreen platform (no display needed)."""

import os
import time

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("PySide6.QtWidgets")
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from glitchlab.engine import fixtures  # noqa: E402
from glitchlab.engine.image_io import ExportOptions, load_image  # noqa: E402
from glitchlab.engine.model import Document  # noqa: E402
from glitchlab.engine.pipeline import render_document, render_full  # noqa: E402

pytestmark = pytest.mark.ui


@pytest.fixture(scope="module")
def app():
    from glitchlab.ui.theme import apply_theme
    a = QApplication.instance() or QApplication([])
    apply_theme(a)
    return a


def pump(app, seconds=5.0, until=None):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        if until is not None and until():
            app.processEvents()
            return True
        time.sleep(0.005)
    return until() if until else True


@pytest.fixture
def win(app, tmp_path):
    from glitchlab.ui.main_window import MainWindow
    w = MainWindow()
    w.show()
    yield w
    w.dirty = False
    w.close()
    pump(app, 0.1)


def idle(w):
    return (not w._debounce.isActive()) and not any(w.runner.busy(l) for l in w.runner.LANES)


@pytest.fixture
def photo_path(tmp_path):
    p = tmp_path / "photo.png"
    Image.fromarray(fixtures.photo_like(400, 300), "RGBA").save(p)
    return str(p)


def test_import_and_preview(app, win, photo_path):
    assert win.import_image(photo_path, interactive=False)
    assert pump(app, 10, lambda: win.last_result is not None and idle(win))
    assert np.array_equal(win.last_result, win.image.rgba)  # empty stack = original
    assert win.canvas.has_image()


def test_add_effect_undo_redo(app, win, photo_path):
    win.import_image(photo_path, interactive=False)
    pump(app, 10, lambda: idle(win))
    win.add_effect("channel_shift")
    win.add_effect("pixel_sort")
    assert len(win.doc.layer.stack) == 2
    pump(app, 10, lambda: idle(win))
    result_two = win.last_result.copy()
    assert not np.array_equal(result_two, win.image.rgba)
    win.undo()
    assert len(win.doc.layer.stack) == 1
    win.undo()
    assert len(win.doc.layer.stack) == 0
    pump(app, 10, lambda: idle(win))
    assert np.array_equal(win.last_result, win.image.rgba)
    win.redo()
    win.redo()
    pump(app, 10, lambda: idle(win))
    assert np.array_equal(win.last_result, result_two)


def test_param_drag_coalesces_history(app, win, photo_path):
    win.import_image(photo_path, interactive=False)
    win.add_effect("channel_shift")
    uid = win.selected_uid
    n = len(win.history)
    for v in range(1, 30):
        win.on_param_changed(uid, "r_dx", v, False)
    win.on_param_changed(uid, "r_dx", 30, True)
    assert len(win.history) == n + 1
    assert win.find(uid).params["r_dx"] == 30
    win.undo()
    assert win.find(uid).params["r_dx"] == 12


def test_preview_matches_engine_and_ignores_stale(app, win, photo_path):
    win.import_image(photo_path, interactive=False)
    win.add_effect("scanline_tear")
    uid = win.selected_uid
    # fire several changes quickly; only the final state may be displayed
    for d in (10, 50, 200, 333):
        win.on_param_changed(uid, "displacement", d, True)
        win.start_preview()
    assert pump(app, 10, lambda: idle(win))
    expect = render_document(win.preview_src, win.doc, scale=win.preview_scale,
                             full_size=(win.image.width, win.image.height)).image
    assert np.array_equal(win.last_result, expect)


def test_reorder_and_bypass(app, win, photo_path):
    win.import_image(photo_path, interactive=False)
    win.add_effect("pixel_sort")
    a = win.selected_uid
    win.add_effect("block_scramble")
    b = win.selected_uid
    win.reorder_effects([b, a])
    assert [e.uid for e in win.doc.layer.stack] == [b, a]
    win.toggle_effect(a, False)
    assert not win.find(a).enabled
    win.stack_action("bypass", a)
    assert win.find(a).enabled


def test_full_render_cancel_then_render(app, win, photo_path):
    win.import_image(photo_path, interactive=False)
    for _ in range(4):
        win.add_effect("pixel_sort")
    pump(app, 10, lambda: idle(win))
    win.render_full_resolution()
    win.cancel_renders()
    pump(app, 10, lambda: idle(win))
    win.full_result = None
    win.render_full_resolution()
    assert pump(app, 20, lambda: win.full_result is not None and idle(win))
    expect = render_full(win.image.rgba, win.doc).image
    assert np.array_equal(win.full_result[1], expect)
    assert win.showing_full


def test_export_matches_full_pipeline(app, win, photo_path, tmp_path):
    from glitchlab.ui.export_dialog import ExportRequest
    win.import_image(photo_path, interactive=False)
    win.add_effect("scanline_tear")
    win.add_effect("channel_shift")
    out = tmp_path / "out" / "export.png"
    req = ExportRequest(opts=ExportOptions(format="PNG"), folder=str(out.parent), pattern="x",
                        path=str(out), write_recipe=True)
    win.run_export(req)
    assert pump(app, 20, lambda: out.exists() and idle(win))
    exported = load_image(out).rgba
    expect = render_full(win.image.rgba, Document.from_dict(win.doc.to_dict())).image
    assert np.array_equal(exported, expect)
    assert (tmp_path / "out" / "export.json").exists()


def test_presets_and_reset(app, win, photo_path):
    from glitchlab.engine.presets import available_builtin_presets
    win.import_image(photo_path, interactive=False)
    p = available_builtin_presets()[0]
    win.apply_preset(p, "replace")
    assert len(win.doc.layer.stack) == len(p["stack"])
    win.apply_preset(p, "append")
    assert len(win.doc.layer.stack) == 2 * len(p["stack"])
    win.reset_document()
    assert all(not layer.stack for layer in win.doc.layers)
    win.undo()
    assert len(win.doc.layer.stack) == 2 * len(p["stack"])


def test_seed_randomise_updates_stored_seed(app, win, photo_path):
    win.import_image(photo_path, interactive=False)
    win.add_effect("scanline_tear")
    uid = win.selected_uid
    s0 = win.find(uid).seed
    win.start_preview()
    pump(app, 10, lambda: idle(win))
    assert win.find(uid).seed == s0  # preview refresh never changes seeds
    win.randomize_effect_seed(uid)
    assert win.find(uid).seed != s0
    g0 = win.doc.global_seed
    win.randomize_global_seed()
    assert win.doc.global_seed != g0


def test_effect_panel_builds_for_every_effect(app, win, photo_path):
    from glitchlab.engine.effects.base import REGISTRY
    win.import_image(photo_path, interactive=False)
    for eid in sorted(REGISTRY):
        win.add_effect(eid)
        assert win.effect_panel.current_uid() == win.selected_uid


def test_invalid_import_reports_error(app, win, tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"nope")
    assert not win.import_image(str(bad), interactive=False)
    assert win.image is None


# ------------------------------------------------------------------ M2 UI
def test_layers_and_masks(app, win, photo_path):
    win.import_image(photo_path, interactive=False)
    win.add_effect("channel_shift")
    win.add_mask_shape({"type": "rect", "mode": "add", "x0": 0.0, "y0": 0.0, "x1": 0.5, "y1": 1.0})
    assert win.doc.layer.mask.enabled and len(win.doc.layer.mask.shapes) == 1
    assert win.canvas._mask is not None
    pump(app, 10, lambda: idle(win))
    w = win.last_result.shape[1]
    assert np.array_equal(win.last_result[:, w // 2 + 2:], win.preview_src[:, w // 2 + 2:])
    win.on_mask_changed("invert", True, True)
    pump(app, 10, lambda: idle(win))
    assert np.array_equal(win.last_result[:, : w // 2 - 2], win.preview_src[:, : w // 2 - 2])
    win.undo()
    assert not win.doc.layer.mask.invert
    win.layer_action("add")
    assert len(win.doc.layers) == 3 and win.doc.active_layer == 2
    win.on_layer_changed("blend", "difference", True)
    assert win.doc.layers[2].blend == "difference"
    win.select_layer(1)
    assert win.doc.active_layer == 1 and len(win.active_stack()) == 1
    win.layer_action("remove")
    assert len(win.doc.layers) == 2


def test_variation_grid_restore(app, win, photo_path):
    from glitchlab.engine.variations import VariationSettings
    win.import_image(photo_path, interactive=False)
    win.add_effect("scanline_tear")
    win.add_effect("pixel_sort")
    win.generate_variations(VariationSettings(count=4, intensity=0.6, batch_seed=7))
    assert pump(app, 30, lambda: idle(win) and len(win._var_thumbs) == 4)
    target = win._var_docs[2].to_dict()
    win.apply_variation(2)
    assert win.doc.to_dict() == target
    win.favourite_variations([1, 3])
    assert len(win.favourites) == 2
    win.apply_favourite(0)
    assert win.doc.to_dict() == win._var_docs[1].to_dict()


def test_batch_export_variations(app, win, photo_path, tmp_path):
    from glitchlab.engine.variations import VariationSettings
    from glitchlab.ui.export_dialog import ExportRequest
    win.import_image(photo_path, interactive=False)
    win.add_effect("block_scramble")
    win.generate_variations(VariationSettings(count=4, intensity=0.6, batch_seed=7))
    pump(app, 30, lambda: idle(win) and len(win._var_thumbs) == 4)
    out = tmp_path / "batch"
    req = ExportRequest(opts=ExportOptions(format="PNG"), folder=str(out), pattern="v{index}", path="",
                        write_recipe=True)
    win.run_batch_export([win._var_docs[0], win._var_docs[3]], req)
    assert pump(app, 30, lambda: idle(win) and (out / "v02.png").exists())
    img = load_image(out / "v02.png").rgba
    assert np.array_equal(img, render_full(win.image.rgba, win._var_docs[3]).image)
    assert (out / "v01.json").exists()


def test_project_save_open_roundtrip(app, win, photo_path, tmp_path, monkeypatch):
    win.import_image(photo_path, interactive=False)
    win.add_effect("pixel_sort")
    win.add_mask_shape({"type": "stroke", "mode": "add", "points": [[0.1, 0.1], [0.9, 0.9]], "radius": 0.05})
    win.add_current_to_favourites()
    win.render_full_resolution()
    pump(app, 20, lambda: idle(win) and win.full_result is not None)
    path = str(tmp_path / "proj.glab")
    win.project_path = path
    assert win.save_project()
    doc_before = win.doc.to_dict()
    full_before = win.full_result[1].copy()
    win.reset_document()
    win.dirty = False
    assert win.open_project(path, interactive=False)
    assert win.doc.to_dict() == doc_before
    assert len(win.favourites) == 1
    assert np.array_equal(render_full(win.image.rgba, win.doc).image, full_before)
    assert win._project_hash is not None


def test_variation_range_from_ui(app, win, photo_path):
    from glitchlab.engine.variations import VariationSettings, generate_variations
    win.import_image(photo_path, interactive=False)
    win.add_effect("scanline_tear")
    uid = win.selected_uid
    win.on_range_changed(uid, "displacement", [10, 20])
    assert win.find(uid).ranges == {"displacement": [10.0, 20.0]}
    for v in generate_variations(win.doc, VariationSettings(count=16, intensity=1.0, batch_seed=4)):
        assert 10 <= v.layer.stack[0].params["displacement"] <= 20
    win.undo()
    assert win.find(uid).ranges == {}


# ------------------------------------------------------------- v1.1 UI fixes
def test_window_fits_small_screens_and_long_status(app, win):
    """Bug 2: the window must never be forced wider than the screen."""
    from glitchlab.ui.widgets import layout_report, text_scale
    limit = int(800 * text_scale(win))  # wider allowance only where text itself renders wider
    min_w = win.minimumSizeHint().width()
    assert min_w <= limit, layout_report(win)
    win.set_status("x" * 600 + " very long failure message " * 20, "error")
    app.processEvents()
    assert win.minimumSizeHint().width() == min_w
    win.resize(1024, 640)
    app.processEvents()
    assert win.width() <= 1024 and win.height() <= 640


def test_variations_panel_aligned_with_preview(app, win):
    """Bug 1: the variations panel sits directly under the preview, same width and left edge."""
    win.resize(1280, 800)
    app.processEvents()
    c = win.canvas.mapTo(win, win.canvas.rect().topLeft())
    b = win.bottom_panel.mapTo(win, win.bottom_panel.rect().topLeft())
    assert c.x() == b.x() and win.canvas.width() == win.bottom_panel.width()
    assert b.y() > c.y()
    assert win.bottom_panel.parent() is win.center_split


def test_any_variation_count(app, win, photo_path):
    """Bug 3: any count 1..64, not just 4/9/16."""
    win.import_image(photo_path, interactive=False)
    win.add_effect("scanline_tear")
    win.variations.count.setValue(7)
    s = win.variations.settings()
    assert s.count == 7
    win.generate_variations(s)
    assert pump(app, 30, lambda: idle(win) and len(win._var_thumbs) == 7)
    assert win.variations.grid.count() == 7
    assert win.variations.count.maximum() == 64


def test_export_conflict_save_as_new(app, win, photo_path, tmp_path, monkeypatch):
    """Bug 4: exporting to an existing name offers overwrite or save-as-new numbering."""
    from glitchlab.ui import export_dialog as ed
    win.import_image(photo_path, interactive=False)
    folder = tmp_path / "exp"
    folder.mkdir()
    (folder / "photo_glitch_1.png").write_bytes(b"old")

    def make_dialog():
        return ed.ExportDialog(win, render_size=(400, 300), source_name="photo", source_path=photo_path, seed=1,
                               preset="", folder=str(folder), pattern="{name}_glitch_{seed}")

    monkeypatch.setattr(ed, "ask_conflict", lambda *a, **k: "number")
    dlg = make_dialog()
    dlg._accept()
    assert dlg.request.path == str(folder / "photo_glitch_1_01.png") and not dlg.request.opts.overwrite
    win.run_export(dlg.request)
    assert pump(app, 20, lambda: (folder / "photo_glitch_1_01.png").exists() and idle(win))
    assert (folder / "photo_glitch_1.png").read_bytes() == b"old"
    monkeypatch.setattr(ed, "ask_conflict", lambda *a, **k: "overwrite")
    dlg = make_dialog()
    dlg._accept()
    assert dlg.request.path == str(folder / "photo_glitch_1.png") and dlg.request.opts.overwrite
    monkeypatch.setattr(ed, "ask_conflict", lambda *a, **k: None)
    dlg = make_dialog()
    dlg._accept()
    assert dlg.request is None


def test_every_effect_panel_fits_the_dock(app, win):
    """No effect's controls may be wider than the right panel (no clipping)."""
    from glitchlab.engine.effects.base import REGISTRY
    from glitchlab.engine.model import EffectInstance
    app.processEvents()
    viewport = win.right_dock.minimumWidth() - 12
    for eid in sorted(REGISTRY):
        win.effect_panel.set_instance(EffectInstance.create(eid))
        app.processEvents()
        need = win.effect_panel.scroll.widget().minimumSizeHint().width()
        if need > viewport:
            from glitchlab.ui.widgets import layout_report
            body = win.effect_panel.scroll.widget()
            rows = sorted(((w.minimumSizeHint().width(), type(w).__name__, getattr(w, "text", lambda: "")()[:40])
                           for w in body.findChildren(QWidget) if w.isVisibleTo(body)), key=lambda t: -t[0])[:10]
            raise AssertionError(f"{eid} needs {need} > {viewport}\n" + "\n".join(map(str, rows)) + "\n"
                                 + layout_report(win))
    from glitchlab.ui.widgets import text_scale
    assert win.minimumSizeHint().width() <= int(800 * text_scale(win))


def test_long_paths_and_names_do_not_widen_window(app, win):
    """Windows paths cannot wrap at backslashes: they must be elided instead."""
    before = win.minimumSizeHint().width()
    win.browser.folder_label.setText("C:\\Users\\someone\\AppData\\Local\\Temp\\" + "very_long_folder_name\\" * 8)
    win.doc.layer.name = "A" * 300
    win.refresh_stack()
    app.processEvents()
    assert win.minimumSizeHint().width() == before
