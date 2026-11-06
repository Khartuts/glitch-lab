"""Drive the real main window offscreen and save screenshots.

Usage: python scripts/ui_smoke.py OUTPUT_DIR
"""

import os
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("GLITCHLAB_CONFIG_DIR", tempfile.mkdtemp(prefix="glitchlab-smoke-"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from PIL import Image  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from glitchlab.engine import effects, fixtures  # noqa: E402,F401
from glitchlab.ui.main_window import MainWindow  # noqa: E402
from glitchlab.ui.theme import apply_theme  # noqa: E402
from glitchlab.engine.settings import setup_logging  # noqa: E402


def pump(app, seconds=0.5, until=None):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        if until and until():
            return True
        time.sleep(0.01)
    return False


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    setup_logging(to_file=False)
    app = QApplication.instance() or QApplication(sys.argv)
    apply_theme(app)
    win = MainWindow()
    win.resize(1600, 980)
    win.show()
    img_dir = tempfile.mkdtemp(prefix="glitchlab-imgs-")
    for name in ("photo", "color_bars", "transparent"):
        arr = fixtures.ALL_FIXTURES[name](960, 640) if name == "photo" else fixtures.ALL_FIXTURES[name](480, 320)
        Image.fromarray(arr, "RGBA").save(os.path.join(img_dir, f"{name}.png"))
    assert win.import_image(os.path.join(img_dir, "photo.png"), interactive=False)
    pump(app, 5, until=lambda: win.last_result is not None and not win.runner.busy("preview"))
    for eid in ("pixel_sort", "scanline_tear", "channel_shift"):
        win.add_effect(eid)
    pump(app, 0.3)
    pump(app, 8, until=lambda: not win.runner.busy("preview") and not win._debounce.isActive())
    pump(app, 0.5)
    win.grab().save(os.path.join(out_dir, "main_window.png"))
    win.a_compare.setChecked(True)
    win.canvas.set_split(0.45)
    pump(app, 0.3)
    win.grab().save(os.path.join(out_dir, "compare.png"))
    win.a_compare.setChecked(False)
    # M2: mask + layers + variations
    win.add_mask_shape({"type": "rect", "mode": "add", "x0": 0.05, "y0": 0.05, "x1": 0.6, "y1": 0.7})
    win.add_mask_shape({"type": "stroke", "mode": "add", "points": [[0.6, 0.6], [0.8, 0.85], [0.95, 0.9]], "radius": 0.05})
    win.set_tool("brush")
    from glitchlab.engine.variations import VariationSettings
    win.generate_variations(VariationSettings(count=9, intensity=0.6, batch_seed=3))
    pump(app, 30, until=lambda: len(win._var_thumbs) == 9 and not win.runner.busy("variations") and not win.runner.busy("preview"))
    pump(app, 0.5)
    win.grab().save(os.path.join(out_dir, "masks_variations.png"))
    print("status:", win.status_label.text())
    win.dirty = False
    win.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "screenshots"))
