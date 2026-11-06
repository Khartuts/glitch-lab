"""Variation Grid and Favourites panels."""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..engine.params import fparam
from ..engine.variations import VARIATION_COUNTS, VariationSettings
from .param_widgets import NumberEditor
from .qt_image import rgba_to_pixmap

_INTENSITY = fparam("intensity", "Intensity", 0.35, 0.0, 1.0,
                    tooltip="How far unlocked parameters may move from their current values.")


def _grid(icon: int = 128) -> QListWidget:
    lw = QListWidget()
    lw.setViewMode(QListWidget.IconMode)
    lw.setIconSize(QSize(icon, icon))
    lw.setGridSize(QSize(icon + 16, icon + 30))
    lw.setResizeMode(QListWidget.Adjust)
    lw.setMovement(QListWidget.Static)
    lw.setSelectionMode(QAbstractItemView.ExtendedSelection)
    lw.setWordWrap(True)
    return lw


class VariationsPanel(QWidget):
    generate_requested = Signal(object)        # VariationSettings
    cancel_requested = Signal()
    apply_requested = Signal(int)
    favourite_requested = Signal(list)
    export_requested = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Count"))
        self.count = QComboBox()
        for c in VARIATION_COUNTS:
            self.count.addItem(str(c), c)
        self.count.setCurrentIndex(1)
        bar.addWidget(self.count)
        bar.addWidget(QLabel("Intensity"))
        self.intensity = NumberEditor(_INTENSITY, percent=True)
        self.intensity.set_value(0.35)
        self.intensity.setMinimumWidth(180)
        bar.addWidget(self.intensity)
        self.reroll = QCheckBox("Re-roll unlocked seeds")
        self.reroll.setChecked(True)
        self.vary_global = QCheckBox("Vary global seed")
        self.vary_mix = QCheckBox("Vary mix")
        for w in (self.reroll, self.vary_global, self.vary_mix):
            bar.addWidget(w)
        bar.addSpacing(10)
        bar.addWidget(QLabel("Batch seed"))
        self.batch_seed = QLineEdit("1")
        self.batch_seed.setFixedWidth(90)
        self.batch_seed.setToolTip("Same batch seed + same recipe = the same variations.")
        bar.addWidget(self.batch_seed)
        dice = QToolButton()
        dice.setText("🎲")
        dice.setToolTip("New batch seed")
        dice.clicked.connect(self._new_batch_seed)
        bar.addWidget(dice)
        bar.addStretch(1)
        self.gen_btn = QPushButton("Generate")
        self.gen_btn.setObjectName("primary")
        self.gen_btn.setToolTip("Render seeded variations of the current recipe (G).\n"
                                "Only effects with 'Include in variations' and unlocked parameters change.")
        self.gen_btn.clicked.connect(self._generate)
        bar.addWidget(self.gen_btn)
        self.cancel_btn = QPushButton("Stop")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_requested.emit)
        bar.addWidget(self.cancel_btn)
        lay.addLayout(bar)
        self.info = QLabel("Click a variation to restore its exact recipe. Ctrl/Shift-click to select several.")
        self.info.setObjectName("muted")
        lay.addWidget(self.info)
        self.grid = _grid()
        self.grid.itemClicked.connect(self._clicked)
        self.grid.setContextMenuPolicy(Qt.CustomContextMenu)
        self.grid.customContextMenuRequested.connect(self._menu)
        lay.addWidget(self.grid, 1)
        row = QHBoxLayout()
        fav = QPushButton("★ Add selected to favourites")
        fav.clicked.connect(lambda: self.favourite_requested.emit(self.selected_indices()))
        exp = QPushButton("Export selected…")
        exp.setToolTip("Render the selected variations at full resolution and save them (with JSON recipes).")
        exp.clicked.connect(lambda: self.export_requested.emit(self.selected_indices()))
        row.addWidget(fav)
        row.addWidget(exp)
        row.addStretch(1)
        lay.addLayout(row)

    def _new_batch_seed(self) -> None:
        import secrets
        self.batch_seed.setText(str(secrets.randbits(31)))

    def settings(self) -> VariationSettings:
        try:
            bs = int(self.batch_seed.text()) & 0xFFFFFFFF
        except ValueError:
            bs = 1
        return VariationSettings(count=int(self.count.currentData()), intensity=float(self.intensity.value()),
                                 reroll_seeds=self.reroll.isChecked(), vary_global_seed=self.vary_global.isChecked(),
                                 batch_seed=bs, vary_mix=self.vary_mix.isChecked())

    def _generate(self) -> None:
        self.generate_requested.emit(self.settings())

    def set_busy(self, busy: bool) -> None:
        self.cancel_btn.setEnabled(busy)
        self.gen_btn.setEnabled(not busy)

    def start(self, count: int) -> None:
        self.grid.clear()
        for i in range(count):
            it = QListWidgetItem(QIcon(), f"#{i + 1}\nrendering…")
            it.setData(Qt.UserRole, i)
            self.grid.addItem(it)

    def set_thumbnail(self, i: int, img, failed: str = "") -> None:
        it = self.grid.item(i)
        if it is None:
            return
        if img is not None:
            it.setIcon(QIcon(rgba_to_pixmap(img)))
        it.setText(f"#{i + 1}" + (f"\n⚠ {failed[:40]}" if failed else ""))
        if failed:
            it.setToolTip(failed)

    def selected_indices(self) -> list[int]:
        return sorted(it.data(Qt.UserRole) for it in self.grid.selectedItems())

    def _clicked(self, it: QListWidgetItem) -> None:
        mods = QApplication.keyboardModifiers()
        if mods & (Qt.ControlModifier | Qt.ShiftModifier):
            return
        self.apply_requested.emit(int(it.data(Qt.UserRole)))

    def _menu(self, pos) -> None:
        it = self.grid.itemAt(pos)
        if it is None:
            return
        idx = int(it.data(Qt.UserRole))
        sel = self.selected_indices() or [idx]
        m = QMenu(self)
        m.addAction("Restore this recipe", lambda: self.apply_requested.emit(idx))
        m.addAction("Add to favourites", lambda: self.favourite_requested.emit(sel))
        m.addAction("Export…", lambda: self.export_requested.emit(sel))
        m.exec(self.grid.mapToGlobal(pos))


class FavouritesPanel(QWidget):
    apply_requested = Signal(int)
    remove_requested = Signal(list)
    rename_requested = Signal(int)
    export_requested = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        info = QLabel("Favourite recipes are saved in the project. Click one to restore it.")
        info.setObjectName("muted")
        lay.addWidget(info)
        self.grid = _grid(112)
        self.grid.itemClicked.connect(self._clicked)
        lay.addWidget(self.grid, 1)
        row = QHBoxLayout()
        for text, sig in (("Rename…", "rename"), ("Remove", "remove"), ("Export selected…", "export")):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, s=sig: self._act(s))
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)

    def _act(self, what: str) -> None:
        sel = sorted(it.data(Qt.UserRole) for it in self.grid.selectedItems())
        if not sel:
            return
        if what == "rename":
            self.rename_requested.emit(sel[0])
        elif what == "remove":
            self.remove_requested.emit(sel)
        else:
            self.export_requested.emit(sel)

    def _clicked(self, it) -> None:
        if QApplication.keyboardModifiers() & (Qt.ControlModifier | Qt.ShiftModifier):
            return
        self.apply_requested.emit(int(it.data(Qt.UserRole)))

    def set_favourites(self, favs) -> None:
        self.grid.clear()
        for i, f in enumerate(favs):
            it = QListWidgetItem(QIcon(rgba_to_pixmap(f.thumb)) if f.thumb is not None else QIcon(), f.name)
            it.setData(Qt.UserRole, i)
            it.setToolTip(f"{f.name}\n{f.created}\n{sum(len(l.stack) for l in f.document.layers)} effect(s)")
            self.grid.addItem(it)
