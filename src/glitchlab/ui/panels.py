"""Side panels: effect stack, effect library, presets, image browser, log."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor

from PySide6.QtCore import QObject, QSize, Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..engine.effects import effects_by_category
from ..engine.effects.base import get_effect
from ..engine.image_io import list_images, load_thumbnail
from ..engine.model import EffectInstance
from .qt_image import rgba_to_pixmap

log = logging.getLogger(__name__)


def section_label(text: str) -> QLabel:
    lab = QLabel(text.upper())
    lab.setObjectName("sectionTitle")
    return lab


def build_add_menu(parent, callback) -> QMenu:
    menu = QMenu(parent)
    for cat, effs in effects_by_category().items():
        sub = menu.addMenu(cat)
        for eff in effs:
            act = sub.addAction(eff.name)
            act.setToolTip(eff.description)
            act.triggered.connect(lambda _=False, eid=eff.id: callback(eid))
    menu.setToolTipsVisible(True)
    return menu


class StackPanel(QWidget):
    selected = Signal(str)
    toggled = Signal(str, bool)
    reordered = Signal(list)
    add_requested = Signal(str)
    action = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        head = QHBoxLayout()
        self.title = section_label("Effect stack")
        head.addWidget(self.title)
        head.addStretch(1)
        self.add_btn = QToolButton()
        self.add_btn.setText("＋ Add")
        self.add_btn.setToolTip("Add an effect to the active layer's stack")
        self.add_btn.setPopupMode(QToolButton.InstantPopup)
        self.add_btn.setMenu(build_add_menu(self, self.add_requested.emit))
        head.addWidget(self.add_btn)
        lay.addLayout(head)
        self.list = QListWidget()
        self.list.setDragDropMode(QAbstractItemView.InternalMove)
        self.list.setDefaultDropAction(Qt.MoveAction)
        self.list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list.setToolTip("Effects run top to bottom, always starting from the original image.\n"
                             "Drag to reorder; untick to bypass.")
        self.list.model().rowsMoved.connect(self._rows_moved)
        self.list.currentItemChanged.connect(self._current_changed)
        self.list.itemChanged.connect(self._item_changed)
        lay.addWidget(self.list, 1)
        btns = QHBoxLayout()
        btns.setSpacing(2)
        for key, text, tip in (("up", "▲", "Move up (Ctrl+Up)"), ("down", "▼", "Move down (Ctrl+Down)"),
                               ("duplicate", "⧉", "Duplicate (Ctrl+D)"), ("reset", "↺", "Reset parameters"),
                               ("remove", "✕", "Remove (Del)")):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, k=key: self._act(k))
            btns.addWidget(b)
        btns.addStretch(1)
        lay.addLayout(btns)
        self._updating = False

    def _act(self, key: str) -> None:
        uid = self.current_uid()
        if uid:
            self.action.emit(key, uid)

    def current_uid(self) -> str:
        it = self.list.currentItem()
        return it.data(Qt.UserRole) if it else ""

    def set_stack(self, stack: list[EffectInstance], selected_uid: str = "", failures: dict | None = None,
                  layer_name: str = "") -> None:
        failures = failures or {}
        self._updating = True
        self.title.setText(f"EFFECT STACK · {layer_name.upper()}" if layer_name else "EFFECT STACK")
        self.list.clear()
        sel_item = None
        for i, inst in enumerate(stack):
            name = inst.display_name()
            text = f"{i + 1}.  {name}"
            if inst.mix < 1.0:
                text += f"   {inst.mix * 100:.0f}%"
            it = QListWidgetItem(text)
            it.setData(Qt.UserRole, inst.uid)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsDragEnabled)
            it.setCheckState(Qt.Checked if inst.enabled else Qt.Unchecked)
            tip = get_effect(inst.effect_id).description
            if inst.uid in failures:
                it.setText(text + "   ⚠")
                it.setForeground(Qt.red)
                tip = f"FAILED: {failures[inst.uid]}\n\n{tip}"
            elif not inst.enabled:
                it.setForeground(Qt.gray)
            it.setToolTip(tip)
            self.list.addItem(it)
            if inst.uid == selected_uid:
                sel_item = it
        if sel_item is not None:
            self.list.setCurrentItem(sel_item)
        self._updating = False

    def _current_changed(self, cur, _prev) -> None:
        if not self._updating:
            self.selected.emit(cur.data(Qt.UserRole) if cur else "")

    def _item_changed(self, it: QListWidgetItem) -> None:
        if not self._updating:
            self.toggled.emit(it.data(Qt.UserRole), it.checkState() == Qt.Checked)

    def _rows_moved(self, *args) -> None:
        if self._updating:
            return
        uids = [self.list.item(i).data(Qt.UserRole) for i in range(self.list.count())]
        self.reordered.emit(uids)


class EffectLibrary(QWidget):
    add_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(section_label("Effects"))
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setToolTip("Double-click an effect to add it to the active layer.")
        for cat, effs in effects_by_category().items():
            top = QTreeWidgetItem([cat])
            top.setFlags(top.flags() & ~Qt.ItemIsSelectable)
            f = top.font(0)
            f.setBold(True)
            top.setFont(0, f)
            for eff in effs:
                child = QTreeWidgetItem([eff.name])
                child.setData(0, Qt.UserRole, eff.id)
                child.setToolTip(0, eff.description + ("\n\nRequires FFmpeg." if eff.requires_ffmpeg else ""))
                top.addChild(child)
            self.tree.addTopLevelItem(top)
        self.tree.expandAll()
        self.tree.itemDoubleClicked.connect(self._double)
        lay.addWidget(self.tree, 1)
        add = QPushButton("Add to stack")
        add.setToolTip("Add the highlighted effect to the active layer's stack (Enter)")
        add.clicked.connect(self._add_current)
        lay.addWidget(add)

    def _double(self, item, _col) -> None:
        eid = item.data(0, Qt.UserRole)
        if eid:
            self.add_requested.emit(eid)

    def _add_current(self) -> None:
        it = self.tree.currentItem()
        if it and it.data(0, Qt.UserRole):
            self.add_requested.emit(it.data(0, Qt.UserRole))


class PresetList(QWidget):
    apply_requested = Signal(dict, str)  # preset, mode
    save_requested = Signal()
    delete_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(section_label("Presets"))
        self.list = QListWidget()
        self.list.setToolTip("Double-click to replace the active layer's stack with a preset.\n"
                             "Right-click to append instead.")
        self.list.itemDoubleClicked.connect(lambda it: self.apply_requested.emit(it.data(Qt.UserRole), "replace"))
        self.list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._menu)
        lay.addWidget(self.list, 1)
        save = QPushButton("Save stack as preset…")
        save.setToolTip("Store the active layer's stack (parameters and seeds) as a user preset.")
        save.clicked.connect(self.save_requested.emit)
        lay.addWidget(save)

    def set_presets(self, builtin: list[dict], user: list[dict]) -> None:
        self.list.clear()
        for p, is_user in [(p, False) for p in builtin] + [(p, True) for p in user]:
            it = QListWidgetItem(("★ " if is_user else "") + p["name"])
            d = dict(p)
            d["_user"] = is_user
            it.setData(Qt.UserRole, d)
            names = ", ".join(get_effect(fx["effect"]).name for fx in p["stack"])
            it.setToolTip(f"{p.get('description', '')}\n\nStack: {names}")
            self.list.addItem(it)

    def _menu(self, pos) -> None:
        it = self.list.itemAt(pos)
        if not it:
            return
        p = it.data(Qt.UserRole)
        m = QMenu(self)
        m.addAction("Replace stack", lambda: self.apply_requested.emit(p, "replace"))
        m.addAction("Append to stack", lambda: self.apply_requested.emit(p, "append"))
        if p.get("_user"):
            m.addSeparator()
            m.addAction("Delete user preset", lambda: self.delete_requested.emit(p["name"]))
        m.exec(self.list.mapToGlobal(pos))


class _ThumbBridge(QObject):
    ready = Signal(int, str, object)


class ImageBrowser(QWidget):
    open_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        head = QHBoxLayout()
        head.addWidget(section_label("Images"))
        head.addStretch(1)
        b = QToolButton()
        b.setText("Folder…")
        b.setToolTip("Choose a folder to browse")
        b.clicked.connect(self._choose)
        head.addWidget(b)
        lay.addLayout(head)
        self.folder_label = QLabel("")
        self.folder_label.setObjectName("muted")
        self.folder_label.setWordWrap(True)
        lay.addWidget(self.folder_label)
        self.list = QListWidget()
        self.list.setViewMode(QListWidget.IconMode)
        self.list.setIconSize(QSize(72, 72))
        self.list.setGridSize(QSize(88, 100))
        self.list.setResizeMode(QListWidget.Adjust)
        self.list.setMovement(QListWidget.Static)
        self.list.setWordWrap(True)
        self.list.setToolTip("Double-click to import (the source file is never modified).")
        self.list.itemDoubleClicked.connect(lambda it: self.open_requested.emit(it.data(Qt.UserRole)))
        lay.addWidget(self.list, 1)
        self.folder = ""
        self._gen = 0
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="thumbs")
        self._bridge = _ThumbBridge()
        self._bridge.ready.connect(self._thumb_ready)

    def _choose(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Choose image folder", self.folder or os.path.expanduser("~"))
        if d:
            self.set_folder(d)

    def set_folder(self, folder: str) -> None:
        if not folder or not os.path.isdir(folder):
            return
        self.folder = folder
        self.folder_label.setText(folder)
        self._gen += 1
        gen = self._gen
        self.list.clear()
        paths = list_images(folder)[:500]
        placeholder = QIcon()
        for i, path in enumerate(paths):
            it = QListWidgetItem(placeholder, os.path.basename(path))
            it.setData(Qt.UserRole, path)
            it.setToolTip(path)
            self.list.addItem(it)
            self._pool.submit(self._load, gen, i, path)

    def _load(self, gen: int, i: int, path: str) -> None:
        if gen != self._gen:
            return
        try:
            arr = load_thumbnail(path, 72)
        except Exception as exc:  # unreadable files simply show no thumbnail
            log.debug("thumbnail failed for %s: %s", path, exc)
            arr = None
        self._bridge.ready.emit(gen, path, (i, arr))

    def _thumb_ready(self, gen: int, path: str, payload) -> None:
        if gen != self._gen:
            return
        i, arr = payload
        it = self.list.item(i)
        if it is None or it.data(Qt.UserRole) != path:
            return
        if arr is None:
            it.setText(it.text() + "\n(unreadable)")
        else:
            it.setIcon(QIcon(rgba_to_pixmap(arr)))

    def shutdown(self) -> None:
        self._gen += 1
        self._pool.shutdown(wait=False, cancel_futures=True)


class _LogBridge(QObject):
    line = Signal(str, int)


class QtLogHandler(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.bridge = _LogBridge()
        self.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s", "%H:%M:%S"))

    def emit(self, record):
        try:
            self.bridge.line.emit(self.format(record), record.levelno)
        except Exception:  # pragma: no cover
            pass


class LogView(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(2000)
        f = self.font()
        f.setFamily("Consolas")
        f.setStyleHint(f.StyleHint.Monospace)
        f.setPointSizeF(8.5)
        self.setFont(f)

    def add_line(self, text: str, level: int = logging.INFO) -> None:
        color = {logging.WARNING: "#ffb347", logging.ERROR: "#ff5c5c", logging.CRITICAL: "#ff5c5c"}.get(level)
        if color:
            from html import escape
            self.appendHtml(f'<span style="color:{color}">{escape(text)}</span>')
        else:
            self.appendPlainText(text)
