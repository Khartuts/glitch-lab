"""Preferences and diagnostics dialogs."""

from __future__ import annotations

from html import escape

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from ..engine.errors import DependencyError
from ..engine.services import SONIFY_FILTERS, EngineServices, probe_ffmpeg
from ..engine.settings import Settings


def ffmpeg_report(services: EngineServices) -> str:
    path, source = services.locate_ffmpeg()
    if path is None:
        return ("FFmpeg was not found.\n\nSonify needs ffmpeg.exe. Put it in the 'ffmpeg' folder next to "
                "GlitchLab.exe, add it to PATH, or set it in Edit > Preferences.\n\nAll other effects work without it.")
    try:
        info = probe_ffmpeg(path)
    except DependencyError as exc:
        return f"FFmpeg at {path} ({source}) is not usable:\n{exc.message}"
    lines = [f"{info.version}", f"Path: {path} ({source})", "", "Audio filters used by Sonify:"]
    for f in SONIFY_FILTERS:
        lines.append(f"  {'✔' if f in info.filters else '✘ MISSING'}  {f}")
    missing = info.missing()
    lines.append("")
    lines.append("All required filters are available." if not missing else
                 "Some filters are missing - install a fuller FFmpeg build.")
    return "\n".join(lines)


class PreferencesDialog(QDialog):
    def __init__(self, parent, settings: Settings, services: EngineServices):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setMinimumWidth(560)
        self.settings = settings
        self.services = services
        lay = QVBoxLayout(self)
        form = QFormLayout()
        row = QHBoxLayout()
        self.ffmpeg = QLineEdit(settings.get("ffmpeg_path") or "")
        self.ffmpeg.setPlaceholderText("Automatic (bundled 'ffmpeg' folder, then PATH)")
        b = QPushButton("Browse…")
        b.clicked.connect(self._browse)
        t = QPushButton("Test")
        t.clicked.connect(self._test)
        row.addWidget(self.ffmpeg, 1)
        row.addWidget(b)
        row.addWidget(t)
        form.addRow("FFmpeg executable", row)
        self.ffmpeg_status = QLabel("")
        self.ffmpeg_status.setWordWrap(True)
        self.ffmpeg_status.setObjectName("muted")
        form.addRow("", self.ffmpeg_status)
        self.max_mp = QSpinBox()
        self.max_mp.setRange(4, 250)
        self.max_mp.setSuffix(" MP")
        self.max_mp.setValue(int(settings.get("max_import_megapixels")))
        self.max_mp.setToolTip("Larger images are offered a downscaled working copy on import (bounded memory).")
        form.addRow("Working size limit", self.max_mp)
        self.cache_mb = QSpinBox()
        self.cache_mb.setRange(64, 16384)
        self.cache_mb.setSingleStep(128)
        self.cache_mb.setSuffix(" MB")
        self.cache_mb.setValue(int(settings.get("cache_megabytes")))
        self.cache_mb.setToolTip("Memory for cached intermediate results (speeds up editing late in the stack).")
        form.addRow("Render cache", self.cache_mb)
        lay.addLayout(form)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _browse(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Locate ffmpeg executable", "", "FFmpeg (ffmpeg.exe ffmpeg);;All files (*)")
        if p:
            self.ffmpeg.setText(p)
            self._test()

    def _test(self) -> None:
        tmp = EngineServices(ffmpeg_path=self.ffmpeg.text().strip() or None)
        self.ffmpeg_status.setText(escape(ffmpeg_report(tmp)).replace("\n", "<br>"))

    def _ok(self) -> None:
        self.settings.set("ffmpeg_path", self.ffmpeg.text().strip())
        self.settings.set("max_import_megapixels", self.max_mp.value())
        self.settings.set("cache_megabytes", self.cache_mb.value())
        self.accept()
