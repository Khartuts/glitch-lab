"""Controls for the selected effect instance (auto-generated)."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QRegularExpressionValidator
from PySide6.QtCore import QRegularExpression
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..engine.effects.base import get_effect
from ..engine.model import EffectInstance
from ..engine.ops import CHANNEL_CHOICES
from ..engine.params import fparam
from .param_widgets import NumberEditor, make_editor

_MIX_PARAM = fparam("mix", "Mix", 1.0, 0.0, 1.0, tooltip="Strength: 0% returns the input unchanged, 100% is the full effect.")


class EffectPanel(QWidget):
    param_changed = Signal(str, str, object, bool)   # uid, name, value, final
    field_changed = Signal(str, str, object, bool)   # uid, field, value, final
    lock_changed = Signal(str, str, bool)
    range_changed = Signal(str, str, object)   # uid, param, [lo, hi] or None
    reset_requested = Signal(str)
    randomize_requested = Signal(str)
    duplicate_requested = Signal(str)
    remove_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._uid: str | None = None
        self.seed_edit = None
        self.status_label = None
        self._editors: dict[str, object] = {}
        self._rows: dict[str, tuple[QWidget, QWidget]] = {}
        self._locks: dict[str, QToolButton] = {}
        self._values: dict[str, object] = {}
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(self.scroll)
        self._empty()

    # ------------------------------------------------------------------
    def current_uid(self) -> str | None:
        return self._uid

    def _empty(self) -> None:
        w = QWidget()
        lay = QVBoxLayout(w)
        lab = QLabel("Select an effect in the stack, or double-click one in the\nEffects library to add it.")
        lab.setObjectName("muted")
        lab.setAlignment(Qt.AlignCenter)
        lay.addStretch(1)
        lay.addWidget(lab)
        lay.addStretch(2)
        self._editors = {}
        self._rows = {}
        self._locks = {}
        self.scroll.setWidget(w)

    def set_instance(self, inst: EffectInstance | None, status: str = "") -> None:
        if inst is None:
            self._uid = None
            self.seed_edit = None
            self.status_label = None
            self._empty()
            return
        self._uid = inst.uid
        self.seed_edit = None
        self.status_label = None
        self._editors = {}
        self._rows = {}
        self._locks = {}
        eff = get_effect(inst.effect_id)
        self._values = dict(inst.params)
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        title = QLabel(inst.display_name())
        title.setObjectName("effectTitle")
        lay.addWidget(title)
        desc = QLabel(eff.description)
        desc.setObjectName("muted")
        desc.setWordWrap(True)
        lay.addWidget(desc)
        if eff.preview_note:
            note = QLabel("Preview note: " + eff.preview_note)
            note.setObjectName("warning")
            note.setWordWrap(True)
            lay.addWidget(note)
        if eff.requires_ffmpeg:
            ff = QLabel("Uses the bundled/system FFmpeg executable (never plays audio).")
            ff.setObjectName("muted")
            ff.setWordWrap(True)
            lay.addWidget(ff)
        self.status_label = QLabel(status)
        self.status_label.setObjectName("error")
        self.status_label.setWordWrap(True)
        self.status_label.setVisible(bool(status))
        lay.addWidget(self.status_label)

        # --- common controls ------------------------------------------------
        common = QGroupBox("Effect")
        form = QFormLayout(common)
        form.setLabelAlignment(Qt.AlignRight)
        row = QHBoxLayout()
        self.enabled_box = QCheckBox("Enabled")
        self.enabled_box.setChecked(inst.enabled)
        self.enabled_box.setToolTip("Bypass the effect without removing it (shortcut: B).")
        self.enabled_box.toggled.connect(lambda on: self.field_changed.emit(self._uid, "enabled", bool(on), True))
        row.addWidget(self.enabled_box)
        row.addStretch(1)
        for text, tip, sig in (("↺ Reset", "Restore default parameters (keeps the seed).", self.reset_requested),
                               ("⧉", "Duplicate this effect (Ctrl+D).", self.duplicate_requested),
                               ("✕", "Remove this effect (Del).", self.remove_requested)):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, s=sig: s.emit(self._uid))
            row.addWidget(b)
        form.addRow(row)
        self.mix_editor = NumberEditor(_MIX_PARAM, percent=True)
        self.mix_editor.set_value(inst.mix)
        self.mix_editor.value_changed.connect(lambda v, final: self.field_changed.emit(self._uid, "mix", v, final))
        form.addRow(self._label("Mix", _MIX_PARAM.tooltip), self.mix_editor)
        if eff.supports_channels:
            self.channel_combo = QComboBox()
            for v, label in CHANNEL_CHOICES:
                self.channel_combo.addItem(label, v)
            self.channel_combo.setCurrentIndex(max(0, self.channel_combo.findData(inst.channels)))
            self.channel_combo.setToolTip("Apply to all colour channels, one channel, or only luminance (Y of YCbCr; chroma kept).")
            self.channel_combo.currentIndexChanged.connect(
                lambda _i: self.field_changed.emit(self._uid, "channels", self.channel_combo.currentData(), True))
            form.addRow(self._label("Channels", self.channel_combo.toolTip()), self.channel_combo)
        if eff.supports_alpha:
            self.alpha_box = QCheckBox("Affect alpha")
            self.alpha_box.setChecked(inst.affect_alpha)
            self.alpha_box.setToolTip("Let this effect move/alter the transparency channel. "
                                      "Off = the source alpha is preserved.")
            self.alpha_box.toggled.connect(lambda on: self.field_changed.emit(self._uid, "affect_alpha", bool(on), True))
            form.addRow("", self.alpha_box)
        if eff.uses_seed:
            seed_row = QHBoxLayout()
            self.seed_edit = QLineEdit(str(inst.seed))
            self.seed_edit.setValidator(QRegularExpressionValidator(QRegularExpression(r"\d{1,10}")))
            self.seed_edit.setToolTip("Per-effect random seed (0-4294967295). Combined with the global seed.")
            self.seed_edit.editingFinished.connect(self._seed_edited)
            dice = QToolButton()
            dice.setText("🎲")
            dice.setToolTip("Randomise this effect's seed (R). The new seed is stored in the recipe.")
            dice.clicked.connect(lambda: self.randomize_requested.emit(self._uid))
            self.seed_lock = QToolButton()
            self.seed_lock.setObjectName("lockButton")
            self.seed_lock.setText("lock")
            self.seed_lock.setCheckable(True)
            self.seed_lock.setChecked(inst.seed_locked)
            self.seed_lock.setToolTip("Keep this seed when generating variations.")
            self.seed_lock.toggled.connect(lambda on: self.field_changed.emit(self._uid, "seed_locked", bool(on), True))
            seed_row.addWidget(self.seed_edit, 1)
            seed_row.addWidget(dice)
            seed_row.addWidget(self.seed_lock)
            form.addRow(self._label("Seed", self.seed_edit.toolTip()), seed_row)
        self.vary_box = QCheckBox("Include in variations")
        self.vary_box.setChecked(inst.vary)
        self.vary_box.setToolTip("Allow the Variation Grid to randomise this effect's unlocked parameters.")
        self.vary_box.toggled.connect(lambda on: self.field_changed.emit(self._uid, "vary", bool(on), True))
        form.addRow("", self.vary_box)
        lay.addWidget(common)

        # --- effect parameters --------------------------------------------
        groups: dict[str, QFormLayout] = {}
        order: list[str] = []
        for prm in eff.params:
            g = prm.group or "Parameters"
            if g not in groups:
                box = QGroupBox(g)
                fl = QFormLayout(box)
                fl.setLabelAlignment(Qt.AlignRight)
                groups[g] = fl
                order.append(g)
                lay.addWidget(box)
            ed = make_editor(prm)
            ed.set_value(inst.params.get(prm.name, prm.default))
            ed.value_changed.connect(lambda v, final, n=prm.name: self._param_edited(n, v, final))
            lock = QToolButton()
            lock.setObjectName("lockButton")
            lock.setText("lock")
            lock.setCheckable(True)
            lock.setChecked(prm.name in inst.locked_params)
            lock.setEnabled(prm.randomizable and prm.kind not in ("hex", "text", "file", "color"))
            lock.setToolTip("Lock: the Variation Grid will not change this parameter.")
            lock.toggled.connect(lambda on, n=prm.name: self.lock_changed.emit(self._uid, n, bool(on)))
            holder = QWidget()
            hl = QHBoxLayout(holder)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.setSpacing(4)
            hl.addWidget(ed, 1)
            hl.addWidget(lock)
            label = self._label(prm.label, prm.tooltip, reset_to=(prm.name, prm.default))
            if prm.kind in ("float", "int") and prm.randomizable:
                rng_set = inst.ranges.get(prm.name)
                if rng_set:
                    label.setText(prm.label + " ⟷")
                    label.setToolTip(label.toolTip() + f"\nVariation range: {rng_set[0]:g} … {rng_set[1]:g}")
                label.setContextMenuPolicy(Qt.CustomContextMenu)
                label.customContextMenuRequested.connect(
                    lambda pos, lab=label, pr=prm, cur=rng_set: self._range_menu(lab, pr, cur, pos))
            groups[g].addRow(label, holder)
            self._editors[prm.name] = ed
            self._rows[prm.name] = (label, holder)
            self._locks[prm.name] = lock
        lay.addStretch(1)
        self.scroll.setWidget(body)
        self._update_visibility()

    def _label(self, text: str, tip: str = "", reset_to=None) -> QLabel:
        lab = QLabel(text)
        tooltip = tip or text
        if reset_to is not None:
            tooltip += "\n(double-click to reset to default)"
            lab.mouseDoubleClickEvent = lambda _e, n=reset_to[0], d=reset_to[1]: self._reset_param(n, d)
        lab.setToolTip(tooltip)
        return lab

    def _range_menu(self, label, prm, current, pos) -> None:
        from PySide6.QtWidgets import QMenu
        m = QMenu(self)
        m.addAction("Set variation range…", lambda: self._edit_range(prm, current))
        if current:
            m.addAction("Clear variation range", lambda: self.range_changed.emit(self._uid, prm.name, None))
        m.exec(label.mapToGlobal(pos))

    def _edit_range(self, prm, current) -> None:
        from PySide6.QtWidgets import QDialog, QDialogButtonBox, QDoubleSpinBox
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Variation range: {prm.label}")
        form = QFormLayout(dlg)
        lo, hi = QDoubleSpinBox(), QDoubleSpinBox()
        for sb, v in ((lo, current[0] if current else prm.min), (hi, current[1] if current else prm.max)):
            sb.setRange(prm.min, prm.max)
            sb.setDecimals(prm.decimals)
            sb.setValue(v)
        form.addRow(QLabel("The Variation Grid will only pick values inside this range."))
        form.addRow("From", lo)
        form.addRow("To", hi)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if dlg.exec():
            a, b = sorted((lo.value(), hi.value()))
            self.range_changed.emit(self._uid, prm.name, [a, b])

    def _reset_param(self, name: str, default) -> None:
        ed = self._editors.get(name)
        if ed is not None:
            ed.set_value(default)
            self._param_edited(name, default, True)

    def _param_edited(self, name: str, value, final: bool) -> None:
        self._values[name] = value
        self._update_visibility()
        if self._uid:
            self.param_changed.emit(self._uid, name, value, final)

    def _seed_edited(self) -> None:
        try:
            v = int(self.seed_edit.text() or "0") & 0xFFFFFFFF
        except ValueError:
            return
        self.seed_edit.setText(str(v))
        self.field_changed.emit(self._uid, "seed", v, True)

    def _update_visibility(self) -> None:
        if not self._uid:
            return
        for name, (label, holder) in self._rows.items():
            prm = self._editors[name].param
            vis = True
            if prm.show_if:
                other, allowed = prm.show_if
                vis = self._values.get(other) in allowed
            label.setVisible(vis)
            holder.setVisible(vis)

    def set_seed(self, seed: int) -> None:
        if self.seed_edit is not None:
            self.seed_edit.setText(str(seed))

    def set_status(self, text: str) -> None:
        if self.status_label is not None:
            self.status_label.setText(text)
            self.status_label.setVisible(bool(text))
