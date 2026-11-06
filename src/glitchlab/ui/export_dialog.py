"""Export options dialog (single image or batch of variations)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from ..engine.errors import ExportError
from ..engine.export import PATTERN_TOKENS, format_filename
from ..engine.image_io import ExportOptions, webp_supported


@dataclass
class ExportRequest:
    opts: ExportOptions
    folder: str
    pattern: str
    path: str            # single export: full path; batch: "" (computed per item)
    write_recipe: bool


class ExportDialog(QDialog):
    def __init__(self, parent, *, render_size: tuple[int, int], source_name: str, source_path: str | None,
                 seed: int, preset: str, folder: str, pattern: str, batch_count: int = 0,
                 has_alpha: bool = False):
        super().__init__(parent)
        self.setWindowTitle("Export variations" if batch_count else "Export image")
        self.setMinimumWidth(520)
        self.render_size = render_size
        self.source_name = source_name
        self.source_path = source_path
        self.seed = seed
        self.preset = preset
        self.batch_count = batch_count
        self.background = QColor("#ffffff")
        self.request: ExportRequest | None = None
        lay = QVBoxLayout(self)

        fmt_box = QGroupBox("Format")
        ff = QFormLayout(fmt_box)
        self.fmt = QComboBox()
        self.fmt.addItem("PNG (lossless, keeps transparency)", "PNG")
        self.fmt.addItem("JPEG (lossy, no transparency)", "JPEG")
        if webp_supported():
            self.fmt.addItem("WebP (keeps transparency)", "WEBP")
        ff.addRow("Format", self.fmt)
        self.quality = QSpinBox()
        self.quality.setRange(1, 100)
        self.quality.setValue(92)
        self.quality.setToolTip("Encoder quality for JPEG and lossy WebP.")
        ff.addRow("Quality", self.quality)
        self.subsampling = QComboBox()
        for s in ("4:4:4", "4:2:2", "4:2:0"):
            self.subsampling.addItem(s, s)
        self.subsampling.setCurrentIndex(0)
        self.subsampling.setToolTip("JPEG chroma subsampling. 4:4:4 keeps colour glitches crisp.")
        ff.addRow("Chroma", self.subsampling)
        self.lossless = QCheckBox("Lossless WebP")
        ff.addRow("", self.lossless)
        bg_row = QHBoxLayout()
        self.bg_btn = QPushButton()
        self.bg_btn.clicked.connect(self._pick_bg)
        bg_row.addWidget(self.bg_btn)
        self.bg_note = QLabel("JPEG cannot store transparency: transparent areas are flattened onto this colour.")
        self.bg_note.setObjectName("muted")
        self.bg_note.setWordWrap(True)
        bg_row.addWidget(self.bg_note, 1)
        ff.addRow("Background", bg_row)
        self.embed = QCheckBox("Embed sRGB ICC profile")
        self.embed.setChecked(True)
        ff.addRow("", self.embed)
        lay.addWidget(fmt_box)

        size_box = QGroupBox("Output size")
        sf = QFormLayout(size_box)
        w0, h0 = render_size
        self.scale = QSpinBox()
        self.scale.setRange(1, 400)
        self.scale.setValue(100)
        self.scale.setSuffix(" %")
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 40000)
        self.width_spin.setValue(w0)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, 40000)
        self.height_spin.setValue(h0)
        self.keep_aspect = QCheckBox("Keep aspect ratio")
        self.keep_aspect.setChecked(True)
        sf.addRow("Scale", self.scale)
        wh = QHBoxLayout()
        wh.addWidget(self.width_spin)
        wh.addWidget(QLabel("×"))
        wh.addWidget(self.height_spin)
        wh.addWidget(self.keep_aspect)
        sf.addRow("Pixels", wh)
        self.resample = QComboBox()
        for label, v in (("Lanczos", "lanczos"), ("Area (best for shrinking)", "area"), ("Bicubic", "cubic"),
                         ("Nearest (crisp pixels)", "nearest")):
            self.resample.addItem(label, v)
        sf.addRow("Resampling", self.resample)
        info = QLabel(f"The image is always rendered at full resolution ({w0}×{h0}) first, then resized.")
        info.setObjectName("muted")
        info.setWordWrap(True)
        sf.addRow(info)
        lay.addWidget(size_box)

        out_box = QGroupBox("Destination")
        of = QFormLayout(out_box)
        frow = QHBoxLayout()
        self.folder = QLineEdit(folder or os.path.expanduser("~"))
        fb = QPushButton("Browse…")
        fb.clicked.connect(self._pick_folder)
        frow.addWidget(self.folder, 1)
        frow.addWidget(fb)
        of.addRow("Folder", frow)
        self.pattern = QLineEdit(pattern or "{name}_glitch_{seed}")
        self.pattern.setToolTip("Tokens: " + " ".join("{" + t + "}" for t in PATTERN_TOKENS))
        of.addRow("File name", self.pattern)
        self.preview = QLabel("")
        self.preview.setObjectName("muted")
        self.preview.setWordWrap(True)
        of.addRow("", self.preview)
        self.recipe = QCheckBox("Write JSON recipe alongside each image")
        self.recipe.setChecked(bool(batch_count))
        self.recipe.setToolTip("Saves the full effect stack, seeds, masks and layers next to the image.")
        of.addRow("", self.recipe)
        lay.addWidget(out_box)

        if batch_count:
            lab = QLabel(f"{batch_count} variation(s) will be rendered at full resolution. "
                         "{index} is added automatically if the pattern lacks it.")
            lab.setWordWrap(True)
            lay.addWidget(lab)
        if has_alpha:
            lab = QLabel("This result contains transparency.")
            lab.setObjectName("muted")
            lay.addWidget(lab)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Export")
        bb.button(QDialogButtonBox.Ok).setObjectName("primary")
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

        self.fmt.currentIndexChanged.connect(self._update)
        self.lossless.toggled.connect(self._update)
        self.scale.valueChanged.connect(self._scale_changed)
        self.width_spin.valueChanged.connect(self._w_changed)
        self.height_spin.valueChanged.connect(self._h_changed)
        self.pattern.textChanged.connect(self._update)
        self.folder.textChanged.connect(self._update)
        self._lock = False
        self._set_bg(self.background)
        self._update()

    # ------------------------------------------------------------------
    def _set_bg(self, c: QColor) -> None:
        self.background = c
        fg = "#000" if c.lightness() > 128 else "#fff"
        self.bg_btn.setText(c.name())
        self.bg_btn.setStyleSheet(f"background:{c.name()}; color:{fg};")

    def _pick_bg(self) -> None:
        c = QColorDialog.getColor(self.background, self, "JPEG background")
        if c.isValid():
            self._set_bg(c)

    def _pick_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Export folder", self.folder.text())
        if d:
            self.folder.setText(d)

    def _scale_changed(self, v: int) -> None:
        if self._lock:
            return
        self._lock = True
        w0, h0 = self.render_size
        self.width_spin.setValue(max(1, round(w0 * v / 100)))
        self.height_spin.setValue(max(1, round(h0 * v / 100)))
        self._lock = False

    def _w_changed(self, v: int) -> None:
        if self._lock or not self.keep_aspect.isChecked():
            return
        self._lock = True
        w0, h0 = self.render_size
        self.height_spin.setValue(max(1, round(v * h0 / w0)))
        self.scale.setValue(max(1, min(400, round(v / w0 * 100))))
        self._lock = False

    def _h_changed(self, v: int) -> None:
        if self._lock or not self.keep_aspect.isChecked():
            return
        self._lock = True
        w0, h0 = self.render_size
        self.width_spin.setValue(max(1, round(v * w0 / h0)))
        self.scale.setValue(max(1, min(400, round(v / h0 * 100))))
        self._lock = False

    def _fmt(self) -> str:
        return self.fmt.currentData()

    def _pattern(self) -> str:
        p = self.pattern.text().strip() or "{name}"
        if self.batch_count and "{index}" not in p:
            p += "_{index}"
        return p

    def _filename(self) -> str:
        ext = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}[self._fmt()]
        return format_filename(self._pattern(), name=self.source_name, seed=self.seed, preset=self.preset,
                               index=1, width=self.width_spin.value(), height=self.height_spin.value(),
                               fmt=self._fmt()) + ext

    def _update(self) -> None:
        fmt = self._fmt()
        self.quality.setEnabled(fmt == "JPEG" or (fmt == "WEBP" and not self.lossless.isChecked()))
        self.subsampling.setEnabled(fmt == "JPEG")
        self.lossless.setEnabled(fmt == "WEBP")
        self.bg_btn.setEnabled(fmt == "JPEG")
        self.bg_note.setEnabled(fmt == "JPEG")
        try:
            name = self._filename()
            self.preview.setText(("First file: " if self.batch_count else "→ ") + os.path.join(self.folder.text(), name))
        except ExportError as exc:
            self.preview.setText(str(exc))

    def options(self) -> ExportOptions:
        c = self.background
        w, h = self.width_spin.value(), self.height_spin.value()
        same = (w, h) == tuple(self.render_size)
        return ExportOptions(
            format=self._fmt(), quality=self.quality.value(), webp_lossless=self.lossless.isChecked(),
            background=(c.red(), c.green(), c.blue()), width=None if same else w, height=None if same else h,
            resample=self.resample.currentData(), jpeg_subsampling=self.subsampling.currentData(),
            embed_srgb=self.embed.isChecked())

    def _accept(self) -> None:
        folder = self.folder.text().strip()
        if not folder:
            QMessageBox.warning(self, "Export", "Choose a destination folder.")
            return
        try:
            name = self._filename()
        except ExportError as exc:
            QMessageBox.warning(self, "Export", str(exc))
            return
        opts = self.options()
        path = "" if self.batch_count else os.path.join(folder, name)
        if path:
            if self.source_path and os.path.abspath(path) == os.path.abspath(self.source_path):
                QMessageBox.warning(self, "Export", "That is the source image. Glitch Lab never overwrites "
                                    "the original - choose another name.")
                return
            targets = [path] + ([os.path.splitext(path)[0] + ".json"] if self.recipe.isChecked() else [])
            existing = [t for t in targets if os.path.exists(t)]
            if existing:
                r = QMessageBox.question(self, "Replace file?",
                                         "These files already exist:\n\n" + "\n".join(existing) +
                                         "\n\nReplace them?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if r != QMessageBox.Yes:
                    return
                opts.overwrite = True
        self.request = ExportRequest(opts=opts, folder=folder, pattern=self._pattern(), path=path,
                                     write_recipe=self.recipe.isChecked())
        self.accept()
