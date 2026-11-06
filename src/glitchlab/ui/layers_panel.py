"""Layers and selection-mask controls."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..engine.model import ALPHA_MODES, BLEND_MODES, Document
from ..engine.params import fparam
from .canvas import TOOL_BRUSH, TOOL_ERASE, TOOL_PAN, TOOL_RECT
from .panels import section_label
from .param_widgets import NumberEditor

_OPACITY = fparam("opacity", "Opacity", 1.0, 0.0, 1.0)
_BRUSH = fparam("brush", "Brush size", 0.03, 0.002, 0.25, decimals=3,
                tooltip="Brush radius as a fraction of the image's long side.")
_FEATHER = fparam("feather", "Feather", 0.0, 0.0, 0.1, decimals=3, tooltip="Softens mask edges (fraction of the long side).")
_LLOW = fparam("luma_low", "Luminance from", 0.0, 0.0, 1.0)
_LHIGH = fparam("luma_high", "Luminance to", 1.0, 0.0, 1.0)
_LSOFT = fparam("luma_softness", "Softness", 0.05, 0.0, 0.5)

_ALPHA_LABELS = {"auto": "Auto (preserve unless an effect affects alpha)", "preserve": "Preserve source alpha",
                 "composite": "Composite layer alpha"}


class LayersPanel(QWidget):
    layer_selected = Signal(int)
    layer_changed = Signal(str, object, bool)   # field, value, final
    layer_action = Signal(str)                  # add / duplicate / remove / up / down
    mask_changed = Signal(str, object, bool)
    mask_clear = Signal()
    tool_changed = Signal(str)
    brush_changed = Signal(float)
    overlay_toggled = Signal(bool)
    alpha_mode_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(scroll)
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(8, 8, 8, 8)
        scroll.setWidget(body)
        self._updating = False

        lay.addWidget(section_label("Layers (top first)"))
        self.list = QListWidget()
        self.list.setMaximumHeight(130)
        self.list.setTextElideMode(Qt.ElideRight)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list.setToolTip("Each layer renders its own effect stack from the original image;\n"
                             "layers are composited bottom-to-top. Select a layer to edit its stack and mask.")
        self.list.currentRowChanged.connect(self._row_changed)
        self.list.itemChanged.connect(self._item_changed)
        lay.addWidget(self.list)
        row = QHBoxLayout()
        for key, text, tip in (("add", "＋", "Add a processing layer above the selected one (Ctrl+Shift+N)"),
                               ("duplicate", "⧉", "Duplicate the selected layer"),
                               ("remove", "✕", "Remove the selected layer"),
                               ("up", "▲", "Move layer up"), ("down", "▼", "Move layer down")):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, k=key: self.layer_action.emit(k))
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)

        box = QGroupBox("Selected layer")
        form = QFormLayout(box)
        self.name = QLineEdit()
        self.name.editingFinished.connect(lambda: self.layer_changed.emit("name", self.name.text().strip() or "Layer", True))
        form.addRow("Name", self.name)
        self.blend = QComboBox()
        for m in BLEND_MODES:
            self.blend.addItem(m.title(), m)
        self.blend.setToolTip("How this layer combines with the layers below it.")
        self.blend.currentIndexChanged.connect(lambda _i: self._emit_layer("blend", self.blend.currentData()))
        form.addRow("Blend mode", self.blend)
        self.opacity = NumberEditor(_OPACITY, percent=True)
        self.opacity.value_changed.connect(lambda v, final: self._emit_layer("opacity", v, final))
        form.addRow("Opacity", self.opacity)
        self.visible = QCheckBox("Visible")
        self.visible.toggled.connect(lambda on: self._emit_layer("visible", bool(on)))
        form.addRow("", self.visible)
        lay.addWidget(box)

        dbox = QGroupBox("Document")
        dform = QFormLayout(dbox)
        self.alpha = QComboBox()
        for m in ALPHA_MODES:
            self.alpha.addItem(_ALPHA_LABELS[m], m)
        self.alpha.setToolTip("Output transparency. By default the source alpha is preserved.")
        self.alpha.currentIndexChanged.connect(lambda _i: (not self._updating) and self.alpha_mode_changed.emit(self.alpha.currentData()))
        dform.addRow("Alpha output", self.alpha)
        lay.addWidget(dbox)

        mbox = QGroupBox("Selection mask (selected layer)")
        mform = QFormLayout(mbox)
        self.mask_on = QCheckBox("Use mask")
        self.mask_on.setToolTip("Limit this layer to the selected area. Drawing a shape enables it automatically.")
        self.mask_on.toggled.connect(lambda on: self._emit_mask("enabled", bool(on)))
        mform.addRow("", self.mask_on)
        tools = QGridLayout()
        tools.setSpacing(2)
        self.tool_group = QButtonGroup(self)
        self.tool_buttons = {}
        for tool, text, tip in ((TOOL_PAN, "Pan", "Pan/zoom the view (no painting)  [H]"),
                                (TOOL_RECT, "Rectangle", "Drag to add a rectangle; right-drag or Alt-drag subtracts  [M]"),
                                (TOOL_BRUSH, "Brush", "Paint the selection  [P]"),
                                (TOOL_ERASE, "Eraser", "Erase from the selection  [E]")):
            b = QToolButton()
            b.setText(text)
            b.setCheckable(True)
            b.setToolTip(tip)
            self.tool_group.addButton(b)
            self.tool_buttons[tool] = b
            b.clicked.connect(lambda _=False, t=tool: self.tool_changed.emit(t))
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            k = len(self.tool_buttons) - 1
            tools.addWidget(b, k // 2, k % 2)
        self.tool_buttons[TOOL_PAN].setChecked(True)
        mform.addRow("Tool", tools)
        self.brush = NumberEditor(_BRUSH)
        self.brush.set_value(0.03)
        self.brush.value_changed.connect(lambda v, final: self.brush_changed.emit(float(v)))
        mform.addRow("Brush size", self.brush)
        self.feather = NumberEditor(_FEATHER)
        self.feather.value_changed.connect(lambda v, final: self._emit_mask("feather", v, final))
        mform.addRow("Feather", self.feather)
        self.invert = QCheckBox("Invert mask")
        self.invert.toggled.connect(lambda on: self._emit_mask("invert", bool(on)))
        mform.addRow("", self.invert)
        self.luma_on = QCheckBox("Luminance range")
        self.luma_on.setToolTip("Only affect pixels of the original whose brightness lies in the range.")
        self.luma_on.toggled.connect(lambda on: self._emit_mask("luma_enabled", bool(on)))
        mform.addRow("", self.luma_on)
        self.luma_low = NumberEditor(_LLOW)
        self.luma_low.value_changed.connect(lambda v, final: self._emit_mask("luma_low", v, final))
        mform.addRow("From", self.luma_low)
        self.luma_high = NumberEditor(_LHIGH)
        self.luma_high.value_changed.connect(lambda v, final: self._emit_mask("luma_high", v, final))
        mform.addRow("To", self.luma_high)
        self.luma_soft = NumberEditor(_LSOFT)
        self.luma_soft.value_changed.connect(lambda v, final: self._emit_mask("luma_softness", v, final))
        mform.addRow("Softness", self.luma_soft)
        crow = QHBoxLayout()
        self.shape_count = QLabel("")
        self.shape_count.setObjectName("muted")
        clear = QPushButton("Clear shapes")
        clear.clicked.connect(self.mask_clear.emit)
        crow.addWidget(self.shape_count, 1)
        crow.addWidget(clear)
        mform.addRow(crow)
        self.overlay = QCheckBox("Show mask overlay")
        self.overlay.setChecked(True)
        self.overlay.setToolTip("Tint the unselected area on the canvas.")
        self.overlay.toggled.connect(self.overlay_toggled.emit)
        mform.addRow("", self.overlay)
        lay.addWidget(mbox)
        lay.addStretch(1)

    def _emit_layer(self, field, value, final=True):
        if not self._updating:
            self.layer_changed.emit(field, value, final)

    def _emit_mask(self, field, value, final=True):
        if not self._updating:
            self.mask_changed.emit(field, value, final)

    def _row_changed(self, row: int) -> None:
        if self._updating or row < 0:
            return
        self.layer_selected.emit(self.list.count() - 1 - row)

    def _item_changed(self, it: QListWidgetItem) -> None:
        if self._updating:
            return
        idx = it.data(Qt.UserRole)
        self.layer_selected.emit(idx)
        self.layer_changed.emit("visible", it.checkState() == Qt.Checked, True)

    def set_tool(self, tool: str) -> None:
        b = self.tool_buttons.get(tool)
        if b:
            b.setChecked(True)

    def set_document(self, doc: Document) -> None:
        self._updating = True
        try:
            self.list.clear()
            n = len(doc.layers)
            for i in reversed(range(n)):
                layer = doc.layers[i]
                fx = len(layer.stack)
                extra = f"{layer.blend}, {layer.opacity * 100:.0f}%, {fx} effect{'s' if fx != 1 else ''}"
                if not layer.mask.is_identity():
                    extra += ", masked"
                it = QListWidgetItem(f"{layer.name}   ({extra})")
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                it.setCheckState(Qt.Checked if layer.visible else Qt.Unchecked)
                it.setData(Qt.UserRole, i)
                self.list.addItem(it)
            self.list.setCurrentRow(n - 1 - doc.active_layer)
            layer = doc.layer
            self.name.setText(layer.name)
            self.blend.setCurrentIndex(max(0, self.blend.findData(layer.blend)))
            self.opacity.set_value(layer.opacity)
            self.visible.setChecked(layer.visible)
            self.alpha.setCurrentIndex(max(0, self.alpha.findData(doc.alpha_mode)))
            m = layer.mask
            self.mask_on.setChecked(m.enabled)
            self.feather.set_value(m.feather)
            self.invert.setChecked(m.invert)
            self.luma_on.setChecked(m.luma_enabled)
            self.luma_low.set_value(m.luma_low)
            self.luma_high.set_value(m.luma_high)
            self.luma_soft.set_value(m.luma_softness)
            n_shapes = len(m.shapes)
            self.shape_count.setText(f"{n_shapes} shape{'s' if n_shapes != 1 else ''}")
        finally:
            self._updating = False
