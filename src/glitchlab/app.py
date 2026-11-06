"""GUI entry point."""

from __future__ import annotations

import logging
import os
import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    # Command-line utilities share the executable: "GlitchLab --selftest" etc.
    if len(argv) > 1 and argv[1] in ("--selftest", "--contact-sheet", "--render", "--version", "--info"):
        from .cli import main as cli_main
        return cli_main(argv[1:])

    from .engine.settings import setup_logging
    log_path = setup_logging()
    log = logging.getLogger("glitchlab")

    from PySide6.QtCore import Qt
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication

    from . import APP_NAME
    from .engine import effects  # noqa: F401  (register effects)
    from .ui.main_window import MainWindow
    from .ui.theme import apply_theme

    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("GlitchLab")
    apply_theme(app)
    icon_path = os.path.join(os.path.dirname(__file__), "resources", "glitchlab.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    def excepthook(exc_type, exc, tb):
        log.critical("Unhandled exception", exc_info=(exc_type, exc, tb))

    sys.excepthook = excepthook
    win = MainWindow()
    win.show()
    if len(argv) > 2 and argv[1] == "--smoke-test":
        return _smoke_test(app, win, argv[2])
    for arg in argv[1:]:
        if os.path.isfile(arg):
            if arg.lower().endswith(".glab") and hasattr(win, "open_project"):
                win.open_project(arg)
            else:
                win.import_image(arg)
            break
    if log_path:
        log.info("Log file: %s", log_path)
    return app.exec()


def _smoke_test(app, win, out_png: str) -> int:
    """Packaged-build check: import a generated image, apply a preset, wait for
    the preview and a full render, save a window screenshot, then exit.
    Exit code 0 = success."""
    import tempfile
    import time

    from PIL import Image

    from .engine import fixtures
    from .engine.presets import available_builtin_presets

    tmp = tempfile.mkdtemp(prefix="glitchlab-smoke-")
    src = os.path.join(tmp, "fixture.png")
    Image.fromarray(fixtures.photo_like(640, 480), "RGBA").save(src)
    ok = win.import_image(src, interactive=False)
    if ok:
        win.apply_preset(available_builtin_presets()[0], "replace")

    def wait(pred, seconds):
        end = time.time() + seconds
        while time.time() < end:
            app.processEvents()
            if pred():
                return True
            time.sleep(0.01)
        return False

    def idle():
        return not win._debounce.isActive() and not any(win.runner.busy(l) for l in win.runner.LANES)

    ok = ok and wait(lambda: win.last_result is not None and idle(), 60)
    if ok:
        win.render_full_resolution()
        ok = wait(lambda: win.full_result is not None and idle(), 120)
    win.grab().save(out_png)
    status = win.status_label.text()
    try:
        print(("SMOKE OK: " if ok else "SMOKE FAILED: ") + status, flush=True)
    except Exception:
        pass
    win.dirty = False
    win.close()
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
