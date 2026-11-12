"""Editors generated from :class:`glitchlab.engine.params.Param` specs."""

from __future__ import annotations

import os

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..engine.params import BITS, BOOL, CHOICE, COLOR, FILE, FLOAT, HEX, INT, TEXT, Param, parse_hex_bytes
from .widgets import shrinkable_combo


class _Editor(QWidget):
    """Base editor: emits ``value_changed(value, final)``."""

    value_changed = Signal(object, bool)

    def __init__(self, param: Param, parent=None):
        super().__init__(parent)
        self.param = param
        self._block = False

    def set_value(self, v) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def _emit(self, v, final: bool) -> None:
        if not self._block:
            self.value_changed.emit(v, final)


class NumberEditor(_Editor):
    STEPS = 1000

    def __init__(self, param: Param, parent=None, *, percent: bool = False):
        super().__init__(param, parent)
        self.is_int = param.kind == INT
        self.percent = percent
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.slider = QSlider(Qt.Horizontal)
        self.lo, self.hi = float(param.min), float(param.max)
        if self.is_int and (self.hi - self.lo) <= self.STEPS:
            self.slider.setRange(int(self.lo), int(self.hi))
            self._direct = True
        else:
            self.slider.setRange(0, self.STEPS)
            self._direct = False
        if self.is_int:
            self.spin = QSpinBox()
            self.spin.setRange(int(self.lo), int(self.hi))
            self.spin.setSingleStep(int(param.step or 1))
        else:
            self.spin = QDoubleSpinBox()
            dec = param.decimals
            if percent:
                self.spin.setRange(self.lo * 100, self.hi * 100)
                self.spin.setDecimals(0)
                self.spin.setSuffix(" %")
                self.spin.setSingleStep(5)
            else:
                self.spin.setRange(self.lo, self.hi)
                self.spin.setDecimals(dec)
                self.spin.setSingleStep(param.step or (self.hi - self.lo) / 100)
        if param.unit and not percent:
            self.spin.setSuffix(f" {param.unit}")
        self.spin.setKeyboardTracking(False)
        self.spin.setMinimumWidth(70)
        self.slider.setMinimumWidth(60)
        self.spin.setAlignment(Qt.AlignRight)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.spin)
        self.slider.valueChanged.connect(self._slider_moved)
        self.slider.sliderReleased.connect(self._slider_released)
        self.spin.valueChanged.connect(self._spin_changed)
        tip = param.tooltip
        if tip:
            self.slider.setToolTip(tip)
            self.spin.setToolTip(tip)

    def _to_slider(self, v: float) -> int:
        if self._direct:
            return int(round(v))
        if self.hi == self.lo:
            return 0
        return int(round((v - self.lo) / (self.hi - self.lo) * self.STEPS))

    def _from_slider(self, s: int) -> float:
        if self._direct:
            return float(s)
        return self.lo + (self.hi - self.lo) * s / self.STEPS

    def value(self):
        v = self.spin.value()
        if self.percent:
            v = v / 100.0
        return int(round(v)) if self.is_int else float(v)

    def set_value(self, v) -> None:
        self._block = True
        try:
            v = float(v)
            self.slider.setValue(self._to_slider(v))
            self.spin.setValue(int(round(v)) if self.is_int else (v * 100 if self.percent else v))
        finally:
            self._block = False

    def _slider_moved(self, s: int) -> None:
        if self._block:
            return
        v = self._from_slider(s)
        self._block = True
        self.spin.setValue(int(round(v)) if self.is_int else (v * 100 if self.percent else v))
        self._block = False
        self._emit(self.value(), not self.slider.isSliderDown())

    def _slider_released(self) -> None:
        self._emit(self.value(), True)

    def _spin_changed(self, _v) -> None:
        if self._block:
            return
        self._block = True
        self.slider.setValue(self._to_slider(self.value()))
        self._block = False
        self._emit(self.value(), True)


class ChoiceEditor(_Editor):
    def __init__(self, param: Param, parent=None):
        super().__init__(param, parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.combo = QComboBox()
        for value, label in param.choices:
            self.combo.addItem(label, value)
        shrinkable_combo(self.combo)
        self.combo.setToolTip(param.tooltip)
        lay.addWidget(self.combo)
        self.combo.currentIndexChanged.connect(lambda _i: self._emit(self.combo.currentData(), True))

    def set_value(self, v) -> None:
        self._block = True
        i = self.combo.findData(v)
        self.combo.setCurrentIndex(max(0, i))
        self._block = False


class BoolEditor(_Editor):
    def __init__(self, param: Param, parent=None):
        super().__init__(param, parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.box = QCheckBox()
        self.box.setToolTip(param.tooltip)
        lay.addWidget(self.box)
        lay.addStretch(1)
        self.box.toggled.connect(lambda on: self._emit(bool(on), True))

    def set_value(self, v) -> None:
        self._block = True
        self.box.setChecked(bool(v))
        self._block = False


class TextEditor(_Editor):
    """Line edit; for hex parameters the text is validated live."""

    def __init__(self, param: Param, parent=None):
        super().__init__(param, parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self.edit = QLineEdit()
        self.edit.setToolTip(param.tooltip)
        if param.kind == HEX:
            self.edit.setPlaceholderText("e.g. DA  or  0E DA")
            f = self.edit.font()
            f.setFamily("Consolas")
            f.setStyleHint(f.StyleHint.Monospace)
            self.edit.setFont(f)
        self.msg = QLabel("")
        self.msg.setObjectName("error")
        self.msg.setWordWrap(True)
        self.msg.hide()
        lay.addWidget(self.edit)
        lay.addWidget(self.msg)
        self.edit.textEdited.connect(self._validate)
        self.edit.editingFinished.connect(self._finish)

    def _validate(self, text: str) -> bool:
        if self.param.kind != HEX or not text.strip():
            self._set_invalid("")
            return True
        try:
            parse_hex_bytes(text, allow_wildcard="wildcard" in self.param.flags)
            self._set_invalid("")
            return True
        except ValueError as exc:
            self._set_invalid(str(exc))
            return False

    def _set_invalid(self, msg: str) -> None:
        self.edit.setProperty("invalid", bool(msg))
        self.edit.style().unpolish(self.edit)
        self.edit.style().polish(self.edit)
        self.msg.setText(msg)
        self.msg.setVisible(bool(msg))

    def _finish(self) -> None:
        text = self.edit.text()
        if self._validate(text):
            self._emit(text, True)

    def set_value(self, v) -> None:
        self._block = True
        self.edit.setText(str(v or ""))
        self._validate(self.edit.text())
        self._block = False


class FileEditor(_Editor):
    def __init__(self, param: Param, parent=None):
        super().__init__(param, parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.edit = QLineEdit()
        self.edit.setReadOnly(True)
        self.edit.setPlaceholderText("(none)")
        self.edit.setMinimumWidth(40)
        self.edit.setToolTip(param.tooltip)
        b = QPushButton("Browse…")
        b.setMinimumWidth(10)
        c = QToolButton()
        c.setText("✕")
        c.setToolTip("Clear")
        lay.addWidget(self.edit, 1)
        lay.addWidget(b)
        lay.addWidget(c)
        b.clicked.connect(self._browse)
        c.clicked.connect(lambda: (self.set_value(""), self._emit("", True)))
        self._path = ""

    def _browse(self) -> None:
        start = os.path.dirname(self._path) if self._path else ""
        path, _ = QFileDialog.getOpenFileName(self, self.param.label, start, self.param.file_filter or "All files (*)")
        if path:
            self.set_value(path)
            self._emit(path, True)

    def set_value(self, v) -> None:
        self._path = str(v or "")
        self.edit.setText(os.path.basename(self._path))
        self.edit.setToolTip(self._path or self.param.tooltip)


class ColorEditor(_Editor):
    def __init__(self, param: Param, parent=None):
        super().__init__(param, parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.button = QPushButton()
        self.button.setFixedWidth(90)
        self.button.setToolTip(param.tooltip)
        lay.addWidget(self.button)
        lay.addStretch(1)
        self.button.clicked.connect(self._pick)
        self._value = param.default

    def _pick(self) -> None:
        c = QColorDialog.getColor(QColor(self._value), self, self.param.label)
        if c.isValid():
            self.set_value(c.name())
            self._emit(c.name(), True)

    def set_value(self, v) -> None:
        self._value = str(v)
        self.button.setText(self._value)
        q = QColor(self._value)
        fg = "#000000" if q.lightness() > 128 else "#ffffff"
        self.button.setStyleSheet(f"background:{self._value}; color:{fg};")


class BitsEditor(_Editor):
    def __init__(self, param: Param, parent=None):
        super().__init__(param, parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self.buttons = []
        for bit in reversed(range(param.nbits)):
            b = QToolButton()
            b.setText(str(bit))
            b.setCheckable(True)
            b.setFixedWidth(22)
            b.setToolTip(f"Bit plane {bit} ({'most' if bit == param.nbits - 1 else 'least' if bit == 0 else ''} significant)".replace(" ()", ""))
            b.toggled.connect(self._toggled)
            self.buttons.append((bit, b))
            lay.addWidget(b)
        lay.addStretch(1)

    def value(self) -> int:
        return sum(1 << bit for bit, b in self.buttons if b.isChecked())

    def _toggled(self, _on) -> None:
        self._emit(self.value(), True)

    def set_value(self, v) -> None:
        self._block = True
        v = int(v)
        for bit, b in self.buttons:
            b.setChecked(bool(v & (1 << bit)))
        self._block = False


def make_editor(param: Param, parent=None) -> _Editor:
    k = param.kind
    if k in (FLOAT, INT):
        return NumberEditor(param, parent)
    if k == CHOICE:
        return ChoiceEditor(param, parent)
    if k == BOOL:
        return BoolEditor(param, parent)
    if k in (HEX, TEXT):
        return TextEditor(param, parent)
    if k == FILE:
        return FileEditor(param, parent)
    if k == COLOR:
        return ColorEditor(param, parent)
    if k == BITS:
        return BitsEditor(param, parent)
    raise ValueError(k)
