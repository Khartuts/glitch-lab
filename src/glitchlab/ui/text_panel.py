"""The Text panel: lyrics / glitch typography boxes of the selected text layer.

Controls are generated from the same parameter specifications the engine
uses (glitchlab.engine.typography.model), like the effect panel.  The text
editor updates the preview live; the words are stored exactly as typed.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..engine.typography import fonts
from ..engine.typography.model import BOX_PARAMS, LAYOUT_PRESETS, STYLE_IDS, STYLE_NAMES, STYLE_PARAMS
from .param_widgets import _Editor, make_editor
from .widgets import FlowLayout, shrinkable_combo


class FontEditor(_Editor):
    """Installed / bundled / imported font family + style, with Import…."""

    import_requested = Signal()

    def __init__(self, param, parent=None):
        super().__init__(param, parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(3)
        self.family = QComboBox()
        self.family.setEditable(False)
        shrinkable_combo(self.family)
        self.style = QComboBox()
        shrinkable_combo(self.style)
        row = QHBoxLayout()
        row.setSpacing(4)
        row.addWidget(self.style, 1)
        imp = QPushButton("Import…")
        imp.setToolTip("Import a TTF/OTF font file (copied to your Glitch Lab fonts folder)")
        imp.clicked.connect(self.import_requested.emit)
        row.addWidget(imp)
        lay.addWidget(self.family)
        lay.addLayout(row)
        self.warn = QLabel("")
        self.warn.setObjectName("warning")
        self.warn.setWordWrap(True)
        self.warn.hide()
        lay.addWidget(self.warn)
        self._fill_families()
        self.family.currentIndexChanged.connect(self._family_changed)
        self.style.currentIndexChanged.connect(self._style_changed)

    def _fill_families(self) -> None:
        self.family.blockSignals(True)
        self.family.clear()
        for fam in fonts.REGISTRY.families():
            self.family.addItem(fam, fam)
        self.family.blockSignals(False)

    def _family_changed(self, _i) -> None:
        if self._block:
            return
        fam = self.family.currentData()
        styles = fonts.REGISTRY.styles(fam)
        self.style.blockSignals(True)
        self.style.clear()
        for st in styles:
            self.style.addItem(st, st)
        pref = next((i for i, st in enumerate(styles) if st.lower() in ("bold", "regular", "book")), 0)
        self.style.setCurrentIndex(pref)
        self.style.blockSignals(False)
        self._emit(f"{fam}|{self.style.currentData()}", True)

    def _style_changed(self, _i) -> None:
        if not self._block:
            self._emit(f"{self.family.currentData()}|{self.style.currentData()}", True)

    def set_value(self, v) -> None:
        self._block = True
        fam, _, st = str(v or "").partition("|")
        if self.family.findData(fam) < 0:
            self.family.addItem(f"{fam} (missing)", fam)
        self.family.setCurrentIndex(self.family.findData(fam))
        self.style.clear()
        styles = fonts.REGISTRY.styles(fam) or [st]
        for s in styles:
            self.style.addItem(s, s)
        self.style.setCurrentIndex(max(0, self.style.findData(st)))
        missing = fonts.REGISTRY.get(str(v)) is None
        self.warn.setText("This font is not installed on this computer; a substitute is used. Choose another font "
                          "or import the file." if missing else "")
        self.warn.setVisible(missing)
        self._block = False


class TextPanel(QWidget):
    add_text_layer = Signal()
    box_selected = Signal(str)
    box_action = Signal(str, str)                         # action, uid
    box_changed = Signal(str, str, object, bool)          # uid, field, value, final
    style_param_changed = Signal(str, str, str, object, bool)
    canvas_requested = Signal(str, str, object)           # uid, field, request
    preset_requested = Signal(str, str, str)              # uid, kind, name
    overrides_changed = Signal(str, object)
    lock_fragment = Signal(str, int, bool)
    font_import = Signal()
    text_only_toggled = Signal(bool)
    export_typography = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(self.scroll)
        self._uid = ""
        self._box: dict = {}
        self._editors: dict[str, _Editor] = {}
        self._rows: dict[str, tuple] = {}
        self.text_only = False
        self.set_state(None, "", None, [])

    # ---------------------------------------------------------------- build
    def set_state(self, layer, selected: str, report, saved_presets: list) -> None:
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)
        self._editors, self._rows = {}, {}
        if layer is None or layer.kind != "text":
            msg = QLabel("Add a text layer to place lyrics or other text over the photograph. Text stays editable; "
                         "styles are applied to the real font glyphs.")
            msg.setWordWrap(True)
            msg.setObjectName("muted")
            lay.addWidget(msg)
            b = QPushButton("Add text layer")
            b.setObjectName("primary")
            b.clicked.connect(self.add_text_layer.emit)
            lay.addWidget(b)
            lay.addStretch(1)
            self._uid = ""
            self.scroll.setWidget(body)
            return
        boxes = layer.boxes
        if selected not in {b["uid"] for b in boxes}:
            selected = boxes[0]["uid"] if boxes else ""
        self._uid = selected
        top = QLabel(f"Text layer: {layer.name}")
        top.setObjectName("effectTitle")
        lay.addWidget(top)
        self.list = QListWidget()
        self.list.setMaximumHeight(110)
        for i, b in enumerate(boxes):
            first = (b["text"].strip().splitlines() or [""])[0][:28] or "(empty)"
            it = QListWidgetItem(f"{i + 1}. {first}  [{STYLE_NAMES[b['style']]}]")
            it.setData(Qt.UserRole, b["uid"])
            self.list.addItem(it)
            if b["uid"] == selected:
                self.list.setCurrentItem(it)
        self.list.currentItemChanged.connect(lambda it, _p: it and self.box_selected.emit(it.data(Qt.UserRole)))
        lay.addWidget(self.list)
        rw = QWidget()
        row = FlowLayout(rw, hspacing=4, vspacing=3)
        for key, text, tip in (("add", "＋ Text box", "Add a text box"), ("duplicate", "⧉", "Duplicate the text box"),
                               ("remove", "✕", "Remove the text box")):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, k=key: self.box_action.emit(k, self._uid))
            row.addWidget(b)
        move = QToolButton()
        move.setText("Move / resize")
        move.setCheckable(True)
        move.setToolTip("Drag the box to move it, its corners to resize, the top handle to rotate. Guides show the "
                        "centre and the safe margins.")
        move.toggled.connect(lambda on: self.canvas_requested.emit(self._uid, "geometry", {"action": "text",
                                                                                         "on": bool(on)}))
        row.addWidget(move)
        self.move_btn = move
        lay.addWidget(rw)
        box = next((b for b in boxes if b["uid"] == selected), None)
        self._box = dict(box or {})
        if box is None:
            lay.addStretch(1)
            self.scroll.setWidget(body)
            return
        self.status = QLabel("")
        self.status.setObjectName("warning")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        self.set_report(report)
        groups: dict[str, QFormLayout] = {}

        def form(g: str) -> QFormLayout:
            if g not in groups:
                gb = QGroupBox(g)
                fl = QFormLayout(gb)
                fl.setLabelAlignment(Qt.AlignRight)
                fl.setRowWrapPolicy(QFormLayout.WrapLongRows)
                fl.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
                groups[g] = fl
                lay.addWidget(gb)
            return groups[g]

        for p in BOX_PARAMS:
            g = p.group or "Text"
            if p.name == "font":
                ed = FontEditor(p)
                ed.import_requested.connect(self.font_import.emit)
            else:
                ed = make_editor(p)
            ed.set_value(box.get(p.name, p.default))
            ed.value_changed.connect(lambda v, final, n=p.name: self._edited(n, v, final))
            ed.canvas_request.connect(lambda req, n=p.name: self.canvas_requested.emit(self._uid, n, req))
            fl = form(g)
            if p.kind == "bool":
                ed.box.setText(p.label)
                fl.addRow(ed)
                self._rows[p.name] = (ed, ed)
            elif p.name == "text":
                lab = QLabel("Lyrics / text (kept exactly as typed)")
                lab.setWordWrap(True)
                fl.addRow(lab)
                fl.addRow(ed)
                self._rows[p.name] = (lab, ed)
            else:
                lab = QLabel(p.label)
                lab.setToolTip(p.tooltip)
                fl.addRow(lab, ed)
                self._rows[p.name] = (lab, ed)
            self._editors[p.name] = ed
            if p.name == "layout":
                self._add_layout_presets(fl)
            if p.name == "seed":
                rr = QPushButton("Reroll")
                rr.setToolTip("New random damage (the text stays the same)")
                rr.clicked.connect(lambda: self.preset_requested.emit(self._uid, "reroll", ""))
                fl.addRow("", rr)
            if p.name == "style":
                look = QPushButton("Apply style look")
                look.setToolTip("Font and colours suggested for the style (your text and position stay)")
                look.clicked.connect(lambda: self.preset_requested.emit(self._uid, "style", box["style"]))
                fl.addRow("", look)
        # style-specific controls
        st = box["style"]
        sform = form("Style")
        for p in STYLE_PARAMS.get(st, ()):
            if p.kind == "data":
                continue
            ed = make_editor(p)
            ed.set_value(box["style_params"].get(st, {}).get(p.name, p.default))
            ed.value_changed.connect(lambda v, final, n=p.name, s=st: self.style_param_changed.emit(self._uid, s, n, v,
                                                                                                    final))
            if p.kind == "bool":
                ed.box.setText(p.label)
                sform.addRow(ed)
            else:
                sform.addRow(QLabel(p.label), ed)
        if st == "cut":
            self._add_locks(sform, box)
        self._add_overrides(form("Overrides (per word / line)"), box)
        self._add_presets(form("Typography presets"), saved_presets)
        out = form("Output")
        from .param_widgets import BoolEditor  # noqa: PLC0415
        from ..engine.params import bparam  # noqa: PLC0415

        to = BoolEditor(bparam("text_only", "Text only", False))
        to.set_value(self.text_only)
        to.box.setText("Text-only preview")
        to.box.setToolTip("Hide the photograph in the preview to judge the lettering alone.")
        to.value_changed.connect(lambda v, _f: self._text_only(bool(v)))
        out.addRow(to)
        ex = QPushButton("Export text PNG…")
        ex.setToolTip("Export only the typography as a transparent PNG")
        ex.clicked.connect(self.export_typography.emit)
        out.addRow(ex)
        lay.addStretch(1)
        self.scroll.setWidget(body)
        self._update_visibility()

    def _text_only(self, on: bool) -> None:
        self.text_only = on
        self.text_only_toggled.emit(on)

    def _add_layout_presets(self, fl: QFormLayout) -> None:
        combo = QComboBox()
        for key, name in LAYOUT_PRESETS:
            combo.addItem(name, key)
        shrinkable_combo(combo)
        apply = QPushButton("Apply layout")
        apply.clicked.connect(lambda: self.preset_requested.emit(self._uid, "layout", combo.currentData()))
        fl.addRow(QLabel("Layout preset"), combo)
        fl.addRow("", apply)

    def _add_locks(self, fl: QFormLayout, box: dict) -> None:
        n_words = max(1, len(box["text"].split()))
        spin = QSpinBox()
        spin.setRange(1, n_words)
        locks = box["style_params"]["cut"].get("locks") or {}
        info = QLabel(f"Locked: {', '.join(str(int(k) + 1) for k in sorted(locks, key=int)) or 'none'}")
        info.setObjectName("muted")
        lock = QPushButton("Lock")
        lock.setToolTip("Keep this word's font, paper and rotation when rerolling")
        unlock = QPushButton("Unlock")
        lock.clicked.connect(lambda: self.lock_fragment.emit(self._uid, spin.value() - 1, True))
        unlock.clicked.connect(lambda: self.lock_fragment.emit(self._uid, spin.value() - 1, False))
        w = QWidget()
        r = FlowLayout(w, hspacing=4, vspacing=3)
        for x in (spin, lock, unlock):
            r.addWidget(x)
        fl.addRow(QLabel("Fragment (word)"), w)
        fl.addRow(info)

    def _add_overrides(self, fl: QFormLayout, box: dict) -> None:
        lst = QListWidget()
        lst.setMaximumHeight(70)
        for o in box["overrides"]:
            lst.addItem(f"{o['scope']} {o['index'] + 1}: {STYLE_NAMES[o['style']]} ({o['intensity']:.1f})")
        fl.addRow(lst)
        scope = QComboBox()
        scope.addItem("Word", "word")
        scope.addItem("Line", "line")
        idx = QSpinBox()
        idx.setRange(1, 999)
        style = QComboBox()
        for s in STYLE_IDS:
            style.addItem(STYLE_NAMES[s], s)
        shrinkable_combo(style)
        add = QPushButton("Add")
        rem = QPushButton("Remove")
        ovs = list(box["overrides"])
        add.clicked.connect(lambda: self.overrides_changed.emit(self._uid, ovs + [
            {"scope": scope.currentData(), "index": idx.value() - 1, "style": style.currentData(), "intensity": 1.0}]))
        rem.clicked.connect(lambda: lst.currentRow() >= 0 and self.overrides_changed.emit(
            self._uid, [o for i, o in enumerate(ovs) if i != lst.currentRow()]))
        w = QWidget()
        r = FlowLayout(w, hspacing=4, vspacing=3)
        for x in (scope, idx):
            r.addWidget(x)
        fl.addRow(w)
        fl.addRow(style)
        w2 = QWidget()
        r2 = FlowLayout(w2, hspacing=4, vspacing=3)
        r2.addWidget(add)
        r2.addWidget(rem)
        fl.addRow(w2)

    def _add_presets(self, fl: QFormLayout, saved: list) -> None:
        combo = QComboBox()
        for p in saved:
            combo.addItem(p["name"], p["name"])
        shrinkable_combo(combo)
        apply = QPushButton("Apply")
        apply.setEnabled(bool(saved))
        apply.clicked.connect(lambda: self.preset_requested.emit(self._uid, "saved", combo.currentData() or ""))
        save = QPushButton("Save look…")
        save.setToolTip("Save this box's font, colours, layout and style as a reusable typography preset")
        save.clicked.connect(lambda: self.preset_requested.emit(self._uid, "save", ""))
        fl.addRow(combo)
        w = QWidget()
        r = FlowLayout(w, hspacing=4, vspacing=3)
        r.addWidget(apply)
        r.addWidget(save)
        fl.addRow(w)

    # ---------------------------------------------------------------- state
    def set_report(self, report) -> None:
        if not hasattr(self, "status"):
            return
        msgs = []
        if report is not None:
            if report.overflow:
                msgs.append("⚠ The text overflows its box (nothing is cut off): enlarge the box, reduce the size or "
                            "use 'Fit text to the box'.")
            if report.missing_font:
                msgs.append(f"⚠ Font '{report.missing_font.replace('|', ' ')}' is missing; a substitute is used.")
            msgs += [w for w in report.warnings if "font" not in w.lower() and "box" not in w.lower()]
        self.status.setText("\n".join(msgs))
        self.status.setVisible(bool(msgs))

    def set_value(self, name: str, value) -> None:
        ed = self._editors.get(name)
        if ed is not None:
            ed.set_value(value)
        self._box[name] = value
        self._update_visibility()

    def editor(self, name: str):
        return self._editors.get(name)

    def _edited(self, name: str, value, final: bool) -> None:
        self._box[name] = value
        self._update_visibility()
        if self._uid:
            self.box_changed.emit(self._uid, name, value, final)

    def _update_visibility(self) -> None:
        for name, (lab, ed) in self._rows.items():
            prm = next(p for p in BOX_PARAMS if p.name == name)
            vis = True
            if prm.show_if:
                other, allowed = prm.show_if
                vis = self._box.get(other) in allowed
            lab.setVisible(vis)
            ed.setVisible(vis)

    def current_uid(self) -> str:
        return self._uid
