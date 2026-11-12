"""Small layout helpers that keep the window from growing wider than the screen."""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QComboBox, QLabel, QLayout, QSizePolicy, QWidget


class ElidedLabel(QLabel):
    """Single-line label that never forces its parent wider than it is.

    Long text is shown with "…" and the full text is available as tooltip.
    """

    def __init__(self, text: str = "", parent: QWidget | None = None):
        super().__init__(text, parent)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.setMinimumWidth(20)
        self._full = text
        self.setToolTip(text)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt API
        self._full = text or ""
        self.setToolTip(self._full if len(self._full) > 40 else "")
        super().setText(self._full)
        self.update()

    def fullText(self) -> str:  # noqa: N802
        return self._full

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(20, super().minimumSizeHint().height())

    def sizeHint(self) -> QSize:  # noqa: N802
        hint = super().sizeHint()
        return QSize(min(hint.width(), 400), hint.height())

    def paintEvent(self, e):  # noqa: N802
        p = QPainter(self)
        p.setPen(self.palette().color(self.foregroundRole()))
        rect = self.contentsRect()
        text = self.fontMetrics().elidedText(self._full, Qt.ElideRight, rect.width())
        p.drawText(rect, int(self.alignment() | Qt.AlignVCenter), text)
        p.end()


class FlowLayout(QLayout):
    """Lays widgets out left to right and wraps them onto new rows when the
    available width is too small (adapted from the Qt "Flow Layout" example)."""

    def __init__(self, parent: QWidget | None = None, margin: int = 0, hspacing: int = 6, vspacing: int = 4):
        super().__init__(parent)
        self._items = []
        self._h = hspacing
        self._v = vspacing
        self.setContentsMargins(margin, margin, margin, margin)

    def addItem(self, item):  # noqa: N802
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):  # noqa: N802
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):  # noqa: N802
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):  # noqa: N802
        return Qt.Orientations(0)

    def hasHeightForWidth(self):  # noqa: N802
        return True

    def heightForWidth(self, width):  # noqa: N802
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect):  # noqa: N802
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self):  # noqa: N802
        return self.minimumSize()

    def minimumSize(self):  # noqa: N802
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        return size + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _do_layout(self, rect: QRect, test_only: bool) -> int:
        m = self.contentsMargins()
        eff = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y, line_h = eff.x(), eff.y(), 0
        for item in self._items:
            w = item.widget()
            if w is not None and w.isHidden():
                continue
            hint = item.sizeHint()
            next_x = x + hint.width() + self._h
            if next_x - self._h > eff.right() + 1 and line_h > 0:
                x = eff.x()
                y = y + line_h + self._v
                next_x = x + hint.width() + self._h
                line_h = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x
            line_h = max(line_h, hint.height())
        return y + line_h - rect.y() + m.bottom()


def shrinkable_combo(combo: QComboBox, min_chars: int = 8) -> QComboBox:
    """Let a combo box get narrower than its longest entry (the popup still
    shows full texts), so long option names never widen a panel."""
    combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(min_chars)
    combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    view = combo.view()
    view.setMinimumWidth(view.sizeHintForColumn(0) + 24)
    return combo


def fit_rect_to_screen(widget: QWidget, fraction: float = 0.9) -> QRect | None:
    """Return a rectangle (fraction of the available screen area, centred)."""
    screen = widget.screen()
    if screen is None:
        return None
    avail = screen.availableGeometry()
    w = int(avail.width() * fraction)
    h = int(avail.height() * fraction)
    return QRect(avail.x() + (avail.width() - w) // 2, avail.y() + (avail.height() - h) // 2, w, h)

