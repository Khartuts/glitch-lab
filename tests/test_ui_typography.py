"""Typography in the application (Qt offscreen)."""

import time

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("PySide6.QtWidgets")
from PySide6.QtWidgets import QApplication  # noqa: E402

from glitchlab.engine import fixtures  # noqa: E402
from glitchlab.engine.typography.model import STYLE_IDS  # noqa: E402

pytestmark = pytest.mark.ui
LYRIC = "Ne znam, ali — čujem đavola!\nŠapat, žar & ćelije…"


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


def idle(w):
    return (not w._debounce.isActive()) and not any(w.runner.busy(lane) for lane in w.runner.LANES)


@pytest.fixture
def win(app, tmp_path):
    from glitchlab.ui.main_window import MainWindow

    w = MainWindow()
    w.show()
    p = tmp_path / "photo.png"
    Image.fromarray(fixtures.photo_like(400, 300), "RGBA").save(p)
    w.import_image(str(p), interactive=False)
    pump(app, 10, lambda: idle(w) and w.last_result is not None)
    yield w
    w.dirty = False
    w.close()
    pump(app, 0.1)


def test_add_text_layer_edit_and_undo(app, win):
    before = win.last_result.copy()
    win.add_text_layer()
    uid = win.selected_box
    win.on_box_changed(uid, "text", LYRIC, True)
    assert pump(app, 10, lambda: idle(win))
    assert not np.array_equal(before, win.last_result)
    assert win.find_box(uid)["text"] == LYRIC
    assert win.right_tabs.currentWidget() is win.text_panel
    win.on_box_changed(uid, "style", "rust", True)
    pump(app, 10, lambda: idle(win))
    styled = win.last_result.copy()
    win.undo()
    pump(app, 10, lambda: idle(win))
    assert win.find_box(uid)["style"] == "clean"
    assert not np.array_equal(styled, win.last_result)


def test_text_panel_fits_the_dock_for_every_style(app, win):
    win.add_text_layer()
    win.right_tabs.setCurrentWidget(win.text_panel)
    uid = win.selected_box
    viewport = win.text_panel.scroll.viewport().width()
    for st in STYLE_IDS:
        win.on_box_changed(uid, "style", st, True)
        pump(app, 0.05)
        need = win.text_panel.scroll.widget().minimumSizeHint().width()
        assert need <= viewport + 2, (st, need, viewport)


def test_canvas_geometry_updates_box_and_panel(app, win):
    win.add_text_layer()
    uid = win.selected_box
    win.on_text_geometry(uid, {"x": 0.3, "y": 0.6, "w": 0.4, "h": 0.2, "rotation": 15.0}, True)
    b = win.find_box(uid)
    assert (b["x"], b["y"], b["rotation"]) == (0.3, 0.6, 15.0)
    assert win.text_panel.editor("rotation").value() == pytest.approx(15.0)


def test_presets_reroll_overrides_and_locks(app, win):
    win.add_text_layer()
    uid = win.selected_box
    win.on_box_changed(uid, "text", "one two three", True)
    win.on_text_preset(uid, "layout", "statement")
    b = win.find_box(uid)
    assert b["text"] == "one two three" and b["fit"] and b["align"] == "center"
    win.on_text_preset(uid, "style", "cut")
    seed = win.find_box(uid)["seed"]
    win.on_lock_fragment(uid, 1, True)
    assert "1" in win.find_box(uid)["style_params"]["cut"]["locks"]
    win.on_text_preset(uid, "reroll", "")
    assert win.find_box(uid)["seed"] != seed
    win.on_text_overrides(uid, [{"scope": "word", "index": 0, "style": "xerox", "intensity": 1.0}])
    assert win.find_box(uid)["overrides"][0]["style"] == "xerox"
    pump(app, 10, lambda: idle(win))


def test_text_only_preview_and_typography_export(app, win, tmp_path, monkeypatch):
    win.add_text_layer()
    win.on_text_only(True)
    pump(app, 10, lambda: idle(win))
    assert win.last_result[..., 3].min() == 0  # the photo is hidden
    out = tmp_path / "type.png"
    from PySide6.QtWidgets import QFileDialog

    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), "PNG"))
    win.export_typography()
    from glitchlab.engine.image_io import load_image

    rgba = load_image(str(out)).rgba
    assert rgba[..., 3].min() == 0 and rgba[..., 3].max() == 255


def test_project_roundtrip_and_missing_font(app, win, tmp_path, monkeypatch):
    win.add_text_layer()
    uid = win.selected_box
    win.on_box_changed(uid, "text", LYRIC, True)
    win.on_box_changed(uid, "font", "Nonexistent Sans|Bold", True)
    path = tmp_path / "t.glab"
    win.project_path = None
    from glitchlab.engine.project import project_from_image, save_project

    proj = project_from_image(win.image, win.doc)
    save_project(str(path), proj)
    from PySide6.QtWidgets import QInputDialog

    asked = []
    monkeypatch.setattr(QInputDialog, "getItem", lambda *a, **k: (asked.append(a[2]), ("DejaVu Serif — Book", True))[1])
    win.open_project(str(path), interactive=False)
    pump(app, 2, lambda: bool(asked))
    assert asked and "Nonexistent Sans" in asked[0]
    tl = [la for la in win.doc.layers if la.kind == "text"][0]
    assert tl.boxes[0]["text"] == LYRIC
    assert tl.boxes[0]["font"] == "DejaVu Serif|Book"
