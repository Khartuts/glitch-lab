"""Variation Grid and Favourites panels."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QSize, Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..engine.params import fparam
from ..engine.variations import MAX_VARIATIONS, MIN_VARIATIONS, VariationSettings
from .param_widgets import NumberEditor
from .qt_image import rgba_to_pixmap
from .widgets import ElidedLabel, FlowLayout

_INTENSITY = fparam("intensity", "Intensity", 0.35, 0.0, 1.0,
                    tooltip="How far unlocked parameters may move from their current values.")


def flow_container(hspacing: int = 10) -> QWidget:
    """A widget whose children wrap onto new rows; its height follows its width."""
    w = QWidget()
    FlowLayout(w, hspacing=hspacing, vspacing=4)
    sp = w.sizePolicy()
    sp.setHeightForWidth(True)
    sp.setVerticalPolicy(QSizePolicy.Minimum)
    w.setSizePolicy(sp)
    return w


def fit_icons(grid: QListWidget, max_icon: int = 150, min_icon: int = 56) -> None:
    """Size thumbnails so a full row is visible in the grid's current height."""
    avail = grid.viewport().height() - 38
    icon = int(max(min_icon, min(max_icon, avail)))
    if grid.iconSize().width() != icon:
        grid.setIconSize(QSize(icon, icon))
        grid.setGridSize(QSize(icon + 16, icon + 30))


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
        # Controls wrap onto further rows when the panel is narrow, so this
        # panel never forces the window wider than the screen.
        bar_widget = flow_container()
        bar = bar_widget.layout()

        def chip(*widgets) -> QWidget:
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(4)
            for x in widgets:
                h.addWidget(x)
            bar.addWidget(w)
            return w

        self.count = QSpinBox()
        self.count.setRange(MIN_VARIATIONS, MAX_VARIATIONS)
        self.count.setValue(9)
        self.count.setToolTip(f"Number of variations to render ({MIN_VARIATIONS}-{MAX_VARIATIONS}).")
        chip(QLabel("Count"), self.count)
        self.intensity = NumberEditor(_INTENSITY, percent=True)
        self.intensity.set_value(0.35)
        self.intensity.setFixedWidth(190)
        chip(QLabel("Intensity"), self.intensity)
        self.reroll = QCheckBox("Re-roll unlocked seeds")
        self.reroll.setChecked(True)
        self.vary_global = QCheckBox("Vary global seed")
        self.vary_mix = QCheckBox("Vary mix")
        for w in (self.reroll, self.vary_global, self.vary_mix):
            chip(w)
        self.batch_seed = QLineEdit("1")
        self.batch_seed.setFixedWidth(90)
        self.batch_seed.setToolTip("Same batch seed + same recipe = the same variations.")
        dice = QToolButton()
        dice.setText("🎲")
        dice.setToolTip("New batch seed")
        dice.clicked.connect(self._new_batch_seed)
        chip(QLabel("Batch seed"), self.batch_seed, dice)
        self.gen_btn = QPushButton("Generate")
        self.gen_btn.setObjectName("primary")
        self.gen_btn.setToolTip("Render seeded variations of the current recipe (G).\n"
                                "Only effects with 'Include in variations' and unlocked parameters change.")
        self.gen_btn.clicked.connect(self._generate)
        self.cancel_btn = QPushButton("Stop")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_requested.emit)
        chip(self.gen_btn, self.cancel_btn)
        fav = QPushButton("★ Favourite")
        fav.setToolTip("Add the selected variations to Favourites (saved in the project).")
        fav.clicked.connect(lambda: self.favourite_requested.emit(self.selected_indices()))
        exp = QPushButton("Export selected…")
        exp.setToolTip("Render the selected variations at full resolution and save them (with JSON recipes).")
        exp.clicked.connect(lambda: self.export_requested.emit(self.selected_indices()))
        chip(fav, exp)
        lay.addWidget(bar_widget)
        self.info = ElidedLabel("Click a variation to restore its exact recipe. Ctrl/Shift-click to select several.")
        self.info.setObjectName("muted")
        lay.addWidget(self.info)
        self.grid = _grid()
        self.grid.itemClicked.connect(self._clicked)
        self.grid.setContextMenuPolicy(Qt.CustomContextMenu)
        self.grid.customContextMenuRequested.connect(self._menu)
        lay.addWidget(self.grid, 1)
        self.grid.installEventFilter(self)

    def eventFilter(self, obj, ev):  # noqa: N802
        if obj is self.grid and ev.type() == QEvent.Resize:
            fit_icons(self.grid)
        return super().eventFilter(obj, ev)

    def _new_batch_seed(self) -> None:
        import secrets
        self.batch_seed.setText(str(secrets.randbits(31)))

    def settings(self) -> VariationSettings:
        try:
            bs = int(self.batch_seed.text()) & 0xFFFFFFFF
        except ValueError:
            bs = 1
        return VariationSettings(count=int(self.count.value()), intensity=float(self.intensity.value()),
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
        info = ElidedLabel("Favourite recipes are saved in the project. Click one to restore it.")
        info.setObjectName("muted")
        lay.addWidget(info)
        self.grid = _grid(112)
        self.grid.installEventFilter(self)
        self.grid.itemClicked.connect(self._clicked)
        lay.addWidget(self.grid, 1)
        row_widget = flow_container(hspacing=6)
        row = row_widget.layout()
        for text, sig in (("Rename…", "rename"), ("Remove", "remove"), ("Export selected…", "export")):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, s=sig: self._act(s))
            row.addWidget(b)
        lay.addWidget(row_widget)

    def eventFilter(self, obj, ev):  # noqa: N802
        if obj is self.grid and ev.type() == QEvent.Resize:
            fit_icons(self.grid, max_icon=130)
        return super().eventFilter(obj, ev)

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
