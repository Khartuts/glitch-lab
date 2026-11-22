"""Dialogs for the optional local AI features: installing the AI engine,
downloading/deleting model weights (with licences), progress and cancel."""

from __future__ import annotations

import threading
from html import escape

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from ..ai import models as ai_models
from ..ai import runtime
from ..engine.cancel import CancelToken
from ..engine.errors import Cancelled, GlitchLabError


class _Bridge(QObject):
    progress = Signal(float, float, str)
    finished = Signal(object)


class DownloadDialog(QDialog):
    """Runs ``fn(progress, cancel)`` in a thread with a progress bar and Cancel."""

    def __init__(self, parent, title: str, intro: str, fn):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(520)
        self.cancel_token = CancelToken()
        self.error: BaseException | None = None
        self.ok = False
        lay = QVBoxLayout(self)
        lab = QLabel(intro)
        lab.setWordWrap(True)
        lay.addWidget(lab)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        lay.addWidget(self.bar)
        self.detail = QLabel("Starting…")
        self.detail.setObjectName("muted")
        self.detail.setWordWrap(True)
        lay.addWidget(self.detail)
        bb = QDialogButtonBox(QDialogButtonBox.Cancel)
        bb.rejected.connect(self._cancel)
        lay.addWidget(bb)
        self._bridge = _Bridge()
        self._bridge.progress.connect(self._on_progress)
        self._bridge.finished.connect(self._on_finished)

        def work():
            try:
                fn(lambda d, t, label: self._bridge.progress.emit(float(d), float(t), str(label)), self.cancel_token)
                self._bridge.finished.emit(None)
            except BaseException as exc:  # noqa: BLE001 - reported in the UI
                self._bridge.finished.emit(exc)

        self._thread = threading.Thread(target=work, name="glitchlab-download", daemon=True)

    def exec(self) -> int:  # noqa: A003
        self._thread.start()
        return super().exec()

    def _on_progress(self, done: float, total: float, label: str) -> None:
        if total > 0:
            self.bar.setValue(int(1000 * min(1.0, done / total)))
            self.detail.setText(f"{done / 2**20:,.0f} of {total / 2**20:,.0f} MB — {label}")
        else:
            self.detail.setText(label)

    def _on_finished(self, exc) -> None:
        self.error = exc
        self.ok = exc is None
        if exc is None:
            self.accept()
        else:
            self.reject()

    def _cancel(self) -> None:
        self.cancel_token.cancel()
        self.detail.setText("Cancelling…")

    def reject(self) -> None:
        if self._thread.is_alive() and self.error is None and not self.ok:
            self.cancel_token.cancel()
            self.detail.setText("Cancelling…")
            return  # closes when the worker stops
        super().reject()


def _report(parent, dlg: DownloadDialog, what: str) -> bool:
    if dlg.ok:
        return True
    if isinstance(dlg.error, Cancelled):
        QMessageBox.information(parent, what, "Cancelled. Already downloaded files are kept and the download "
                                "resumes next time.")
    elif isinstance(dlg.error, GlitchLabError):
        QMessageBox.warning(parent, what, dlg.error.message)
    elif dlg.error is not None:
        QMessageBox.critical(parent, what, f"Unexpected error: {dlg.error}")
    return False


def install_engine(parent) -> bool:
    if runtime.engine_available():
        return True
    if not runtime.can_download():
        QMessageBox.information(parent, "AI engine", runtime.status().message)
        return False
    size = runtime.download_size() / 2**20
    if QMessageBox.question(
            parent, "Install AI engine",
            f"Glitch Lab's AI effects need PyTorch, transformers and diffusers (about {size:.0f} MB, downloaded once "
            "from the Python Package Index and checked against fixed checksums). Everything then runs offline on "
            "this computer, using the processor.\n\nDownload and install now?") != QMessageBox.Yes:
        return False
    dlg = DownloadDialog(parent, "Installing AI engine", "Downloading and unpacking the AI engine…",
                         lambda prog, cancel: runtime.install(prog, cancel))
    dlg.exec()
    if _report(parent, dlg, "AI engine"):
        QMessageBox.information(parent, "AI engine", "The AI engine is installed. AI effects now also need their "
                                "model weights (AI › Models…).")
        return True
    return False


def download_models(parent, ids: list[str]) -> bool:
    ids = [m for m in ids if not ai_models.is_installed(m)]
    if not ids:
        return True
    rows = "".join(f"<li><b>{escape(ai_models.MODELS[m].name)}</b> — {ai_models.MODELS[m].size_text}, licence "
                   f"<a href='{ai_models.MODELS[m].licence_url}'>{escape(ai_models.MODELS[m].licence)}</a></li>"
                   for m in ids)
    box = QMessageBox(parent)
    box.setWindowTitle("Download AI models")
    box.setTextFormat(Qt.RichText)
    box.setText(f"Download these model weights from Hugging Face (pinned revisions, verified checksums)?<ul>{rows}</ul>"
                "By downloading you accept each model's licence. The files are stored on this computer and used "
                "offline.")
    box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    if box.exec() != QMessageBox.Yes:
        return False
    dlg = DownloadDialog(parent, "Downloading AI models", "Downloading model weights…",
                         lambda prog, cancel: ai_models.install(ids, prog, cancel))
    dlg.exec()
    return _report(parent, dlg, "AI models")


class ModelsDialog(QDialog):
    """Overview of the AI engine and every model, with download/delete buttons."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("AI models")
        self.setMinimumWidth(720)
        self._lay = QVBoxLayout(self)
        self._body = None
        self._build()

    def _build(self) -> None:
        if self._body is not None:
            self._lay.removeItem(self._body)
            while self._body.count():
                item = self._body.takeAt(0)
                w = item.widget()
                if w is not None:
                    w.deleteLater()
                elif item.layout() is not None:
                    _clear(item.layout())
        body = QVBoxLayout()
        st = runtime.status()
        row = QHBoxLayout()
        lab = QLabel("<b>AI engine:</b> " + escape(st.message))
        lab.setWordWrap(True)
        row.addWidget(lab, 1)
        if not st.available and st.can_download:
            b = QPushButton("Install…")
            b.clicked.connect(lambda: (install_engine(self), self._build()))
            row.addWidget(b)
        body.addLayout(row)
        note = QLabel("All AI effects run locally on this computer. Models are downloaded only when you ask, from "
                      "fixed revisions; nothing is uploaded.")
        note.setObjectName("muted")
        note.setWordWrap(True)
        body.addWidget(note)
        grid = QGridLayout()
        for c, h in enumerate(("Model", "Used for", "Size", "Licence", "")):
            t = QLabel(f"<b>{h}</b>")
            grid.addWidget(t, 0, c)
        for r, spec in enumerate(ai_models.MODELS.values(), start=1):
            grid.addWidget(QLabel(escape(spec.name)), r, 0)
            used = QLabel(escape(spec.purpose))
            used.setWordWrap(True)
            used.setObjectName("muted")
            grid.addWidget(used, r, 1)
            grid.addWidget(QLabel(spec.size_text), r, 2)
            lic = QLabel(f"<a href='{spec.licence_url}'>{escape(spec.licence)}</a>")
            lic.setOpenExternalLinks(True)
            grid.addWidget(lic, r, 3)
            if ai_models.is_installed(spec.id):
                b = QPushButton("Delete")
                b.setToolTip("Remove the downloaded files")
                b.clicked.connect(lambda _=False, m=spec.id: self._delete(m))
            else:
                b = QPushButton("Download")
                b.clicked.connect(lambda _=False, m=spec.id: (download_models(self, [m]), self._build()))
            grid.addWidget(b, r, 4)
        grid.setColumnStretch(1, 1)
        body.addLayout(grid)
        row = QHBoxLayout()
        unload = QPushButton("Unload models from memory")
        unload.clicked.connect(self._unload)
        clear = QPushButton("Clear AI result cache")
        clear.clicked.connect(self._clear_cache)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        row.addWidget(unload)
        row.addWidget(clear)
        row.addStretch(1)
        row.addWidget(close)
        body.addLayout(row)
        self._lay.addLayout(body)
        self._body = body

    def _delete(self, mid: str) -> None:
        spec = ai_models.MODELS[mid]
        if QMessageBox.question(self, "Delete model", f"Delete {spec.name} ({spec.size_text})?") == QMessageBox.Yes:
            ai_models.delete(mid)
            self._build()

    def _unload(self) -> None:
        n = ai_models.unload()
        QMessageBox.information(self, "AI models", f"Unloaded {n} model(s) from memory.")

    def _clear_cache(self) -> None:
        from ..ai.jobs import manager  # noqa: PLC0415

        mgr = manager()
        size = mgr.disk.size() / 2**20
        mgr.memory.clear()
        mgr.disk.clear()
        QMessageBox.information(self, "AI cache", f"Cleared cached AI results ({size:.0f} MB on disk).")


def _clear(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        if item.widget() is not None:
            item.widget().deleteLater()
        elif item.layout() is not None:
            _clear(item.layout())
