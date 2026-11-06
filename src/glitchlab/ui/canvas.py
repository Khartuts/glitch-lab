"""Central preview canvas.

Displays the current result (reduced preview or full-resolution render) in
*full-resolution image coordinates*, so zoom 100% always means one source
pixel per screen pixel.  Supports zoom around the cursor, panning, fit,
an original/result split with a draggable divider, a checkerboard behind
transparency and rectangle/brush mask painting.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import QWidget

from . import theme

TOOL_PAN = "pan"
TOOL_RECT = "rect"
TOOL_BRUSH = "brush"
TOOL_ERASE = "erase"


def _checker_tile(size: int = 8) -> QPixmap:
    pm = QPixmap(size * 2, size * 2)
    pm.fill(QColor("#3a3c44"))
    p = QPainter(pm)
    c = QColor("#5a5d66")
    p.fillRect(0, 0, size, size, c)
    p.fillRect(size, size, size, size, c)
    p.end()
    return pm


class PreviewCanvas(QWidget):
    zoom_changed = Signal(float)
    rect_drawn = Signal(dict)
    stroke_drawn = Signal(dict)
    files_dropped = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(320, 240)
        self._result: QPixmap | None = None
        self._original: QPixmap | None = None
        self._mask: QPixmap | None = None
        self._img_w = 0
        self._img_h = 0
        self._zoom = 1.0
        self._offset = QPointF(0, 0)
        self._compare = False
        self._split = 0.5
        self._checker = True
        self._show_mask = True
        self._label = ""
        self._label_kind = "preview"
        self._busy = ""
        self._error = ""
        self._tool = TOOL_PAN
        self.brush_radius = 0.03  # fraction of long side
        self._drag = None  # ("pan", start, offset) | ("split",) | ("rect", p0) | ("stroke", [pts])
        self._cursor_pos: QPointF | None = None
        self._space = False
        self._checker_brush = QBrush(_checker_tile())
        self._fitted_once = False
        self._fit_mode = True

    # ------------------------------------------------------------------ API
    def has_image(self) -> bool:
        return self._result is not None

    def set_images(self, result: QPixmap | None, original: QPixmap | None, size: tuple[int, int],
                   label: str = "", kind: str = "preview") -> None:
        new_size = (int(size[0]), int(size[1]))
        changed = new_size != (self._img_w, self._img_h)
        self._result = result
        self._original = original
        self._img_w, self._img_h = new_size
        self._label = label
        self._label_kind = kind
        if changed or not self._fitted_once:
            self.fit()
            self._fitted_once = True
        self.update()

    def clear(self) -> None:
        self._result = self._original = self._mask = None
        self._img_w = self._img_h = 0
        self._fitted_once = False
        self.update()

    def set_mask_overlay(self, pm: QPixmap | None) -> None:
        self._mask = pm
        self.update()

    def set_show_mask(self, on: bool) -> None:
        self._show_mask = bool(on)
        self.update()

    def set_busy(self, text: str) -> None:
        self._busy = text
        self.update()

    def set_error(self, text: str) -> None:
        self._error = text
        self.update()

    def set_compare(self, on: bool) -> None:
        self._compare = bool(on)
        self.update()

    def compare(self) -> bool:
        return self._compare

    def set_split(self, frac: float) -> None:
        self._split = min(max(float(frac), 0.0), 1.0)
        self.update()

    def split(self) -> float:
        return self._split

    def set_checkerboard(self, on: bool) -> None:
        self._checker = bool(on)
        self.update()

    def set_tool(self, tool: str) -> None:
        self._tool = tool
        self._drag = None
        self.setCursor(Qt.OpenHandCursor if tool == TOOL_PAN else Qt.CrossCursor)
        self.update()

    def tool(self) -> str:
        return self._tool

    def zoom(self) -> float:
        return self._zoom

    def fit(self) -> None:
        if not self._img_w or not self._img_h:
            return
        w, h = max(1, self.width() - 24), max(1, self.height() - 24)
        self._zoom = max(0.01, min(w / self._img_w, h / self._img_h))
        self._center()
        self._fit_mode = True
        self.zoom_changed.emit(self._zoom)
        self.update()

    def actual_size(self) -> None:
        self.set_zoom(1.0)

    def set_zoom(self, z: float, anchor: QPointF | None = None) -> None:
        if not self._img_w:
            return
        z = max(0.02, min(32.0, float(z)))
        anchor = anchor if anchor is not None else QPointF(self.width() / 2, self.height() / 2)
        img_pt = (anchor - self._offset) / self._zoom
        self._zoom = z
        self._offset = anchor - img_pt * z
        self._fit_mode = False
        self.zoom_changed.emit(self._zoom)
        self.update()

    def zoom_in(self) -> None:
        self.set_zoom(self._zoom * 1.25)

    def zoom_out(self) -> None:
        self.set_zoom(self._zoom / 1.25)

    # ------------------------------------------------------------- geometry
    def _center(self) -> None:
        self._offset = QPointF((self.width() - self._img_w * self._zoom) / 2,
                               (self.height() - self._img_h * self._zoom) / 2)

    def _image_rect(self) -> QRectF:
        return QRectF(self._offset.x(), self._offset.y(), self._img_w * self._zoom, self._img_h * self._zoom)

    def _to_norm(self, pos: QPointF) -> tuple[float, float]:
        if not self._img_w:
            return 0.0, 0.0
        p = (pos - self._offset) / self._zoom
        return p.x() / self._img_w, p.y() / self._img_h

    def _split_x(self) -> float:
        r = self._image_rect()
        return r.left() + r.width() * self._split

    def resizeEvent(self, e):  # noqa: N802
        super().resizeEvent(e)
        if self._img_w and self._fit_mode:
            self.fit()

    # --------------------------------------------------------------- paint
    def paintEvent(self, e):  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#0d0e11"))
        if self._result is None:
            self._paint_welcome(p)
            p.end()
            return
        r = self._image_rect()
        if self._checker:
            p.fillRect(r, self._checker_brush)
        else:
            p.fillRect(r, QColor("#000000"))
        p.setRenderHint(QPainter.SmoothPixmapTransform, self._zoom < 2.0)
        if self._compare and self._original is not None:
            sx = self._split_x()
            p.save()
            p.setClipRect(QRectF(r.left(), r.top(), max(0.0, sx - r.left()), r.height()))
            p.drawPixmap(r, self._original, QRectF(self._original.rect()))
            p.restore()
            p.save()
            p.setClipRect(QRectF(sx, r.top(), max(0.0, r.right() - sx), r.height()))
            p.drawPixmap(r, self._result, QRectF(self._result.rect()))
            p.restore()
        else:
            p.drawPixmap(r, self._result, QRectF(self._result.rect()))
        if self._mask is not None and self._show_mask:
            p.drawPixmap(r, self._mask, QRectF(self._mask.rect()))
        p.setRenderHint(QPainter.Antialiasing, True)
        if self._compare and self._original is not None:
            self._paint_split(p, r)
        self._paint_tool(p)
        self._paint_hud(p)
        p.end()

    def _paint_welcome(self, p: QPainter) -> None:
        p.setRenderHint(QPainter.Antialiasing, True)
        rect = self.rect().adjusted(40, 40, -40, -40)
        pen = QPen(QColor("#3a3c46"), 2, Qt.DashLine)
        p.setPen(pen)
        p.drawRoundedRect(rect, 16, 16)
        f = QFont(self.font())
        f.setPointSizeF(20)
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor(theme.ACCENT))
        p.drawText(rect.adjusted(0, -40, 0, -40), Qt.AlignCenter, "GLITCH LAB")
        f.setPointSizeF(10.5)
        f.setBold(False)
        p.setFont(f)
        p.setPen(QColor(theme.MUTED))
        p.drawText(rect.adjusted(0, 30, 0, 30), Qt.AlignCenter,
                   "Import a photograph (Ctrl+O), drop an image here,\n"
                   "or double-click one in the image browser.")

    def _paint_split(self, p: QPainter, r: QRectF) -> None:
        sx = self._split_x()
        p.setPen(QPen(QColor(255, 255, 255, 220), 1.5))
        p.drawLine(QPointF(sx, r.top()), QPointF(sx, r.bottom()))
        cy = min(max(self.height() / 2, r.top() + 14), r.bottom() - 14)
        p.setBrush(QColor(theme.ACCENT))
        p.setPen(QPen(QColor("#ffffff"), 1.5))
        p.drawEllipse(QPointF(sx, cy), 9, 9)
        p.setPen(QPen(QColor("#ffffff"), 1.6))
        p.drawLine(QPointF(sx - 4, cy), QPointF(sx - 1, cy - 3))
        p.drawLine(QPointF(sx - 4, cy), QPointF(sx - 1, cy + 3))
        p.drawLine(QPointF(sx + 4, cy), QPointF(sx + 1, cy - 3))
        p.drawLine(QPointF(sx + 4, cy), QPointF(sx + 1, cy + 3))
        f = QFont(self.font())
        f.setPointSizeF(8)
        f.setBold(True)
        p.setFont(f)
        top = max(r.top() + 6, 6)
        self._tag(p, QPointF(sx - 8, top), "ORIGINAL", right_align=True)
        self._tag(p, QPointF(sx + 8, top), "RESULT")

    def _tag(self, p: QPainter, pt: QPointF, text: str, right_align: bool = False, color=None) -> None:
        fm = p.fontMetrics()
        w = fm.horizontalAdvance(text) + 12
        h = fm.height() + 4
        x = pt.x() - w if right_align else pt.x()
        rr = QRectF(x, pt.y(), w, h)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 170))
        p.drawRoundedRect(rr, 4, 4)
        p.setPen(QColor(color or "#ffffff"))
        p.drawText(rr, Qt.AlignCenter, text)

    def _paint_tool(self, p: QPainter) -> None:
        if self._drag and self._drag[0] == "rect" and self._cursor_pos is not None:
            p0 = self._drag[1]
            rr = QRectF(p0, self._cursor_pos).normalized()
            p.setPen(QPen(QColor(theme.ACCENT_2), 1.5, Qt.DashLine))
            p.setBrush(QColor(47, 224, 255, 40))
            p.drawRect(rr)
        if self._drag and self._drag[0] == "stroke":
            pts = self._drag[1]
            if pts:
                path = QPainterPath(pts[0])
                for q in pts[1:]:
                    path.lineTo(q)
                rad = self._brush_screen_radius()
                col = QColor(theme.ACCENT_2) if self._tool == TOOL_BRUSH else QColor(theme.ACCENT)
                col.setAlpha(110)
                p.setPen(QPen(col, rad * 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                p.setBrush(Qt.NoBrush)
                p.drawPath(path)
        if self._tool in (TOOL_BRUSH, TOOL_ERASE) and self._cursor_pos is not None and self._result is not None:
            p.setPen(QPen(QColor("#ffffff"), 1))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(self._cursor_pos, self._brush_screen_radius(), self._brush_screen_radius())

    def _brush_screen_radius(self) -> float:
        return max(1.0, self.brush_radius * max(self._img_w, self._img_h) * self._zoom)

    def _paint_hud(self, p: QPainter) -> None:
        f = QFont(self.font())
        f.setPointSizeF(8.5)
        f.setBold(True)
        p.setFont(f)
        color = theme.ACCENT_2 if self._label_kind == "full" else theme.WARN
        if self._label:
            self._tag(p, QPointF(10, 10), self._label, color=color)
        self._tag(p, QPointF(self.width() - 10, 10), f"{self._zoom * 100:.0f}%", right_align=True)
        if self._busy:
            self._tag(p, QPointF(10, 34), self._busy, color=theme.ACCENT)
        if self._error:
            f.setBold(False)
            p.setFont(f)
            self._tag(p, QPointF(10, self.height() - 30), self._error, color=theme.ERROR)

    # --------------------------------------------------------------- input
    def wheelEvent(self, e):  # noqa: N802
        if not self._img_w:
            return
        steps = e.angleDelta().y() / 120.0
        if steps:
            self.set_zoom(self._zoom * (1.15 ** steps), e.position())

    def mouseDoubleClickEvent(self, e):  # noqa: N802
        if self._tool == TOOL_PAN:
            self.fit()

    def mousePressEvent(self, e):  # noqa: N802
        pos = e.position()
        if not self._img_w:
            return
        if self._compare and abs(pos.x() - self._split_x()) < 8 and e.button() == Qt.LeftButton:
            self._drag = ("split",)
            return
        if e.button() == Qt.MiddleButton or (e.button() == Qt.LeftButton and (self._tool == TOOL_PAN or self._space)):
            self._drag = ("pan", pos, QPointF(self._offset))
            self.setCursor(Qt.ClosedHandCursor)
            return
        if e.button() in (Qt.LeftButton, Qt.RightButton):
            if self._tool == TOOL_RECT:
                self._drag = ("rect", pos, e.button() == Qt.RightButton or bool(e.modifiers() & Qt.AltModifier))
            elif self._tool in (TOOL_BRUSH, TOOL_ERASE):
                self._drag = ("stroke", [pos])
            self.update()

    def mouseMoveEvent(self, e):  # noqa: N802
        pos = e.position()
        self._cursor_pos = pos
        if self._drag:
            kind = self._drag[0]
            if kind == "pan":
                self._offset = self._drag[2] + (pos - self._drag[1])
                self._fit_mode = False
            elif kind == "split":
                r = self._image_rect()
                if r.width() > 0:
                    self._split = min(max((pos.x() - r.left()) / r.width(), 0.0), 1.0)
            elif kind == "stroke":
                pts = self._drag[1]
                if (pts[-1] - pos).manhattanLength() > 1.5:
                    pts.append(pos)
        elif self._compare and abs(pos.x() - self._split_x()) < 8:
            self.setCursor(Qt.SplitHCursor)
        else:
            self.setCursor(Qt.OpenHandCursor if self._tool == TOOL_PAN else Qt.CrossCursor)
        self.update()

    def mouseReleaseEvent(self, e):  # noqa: N802
        if not self._drag:
            return
        kind = self._drag[0]
        if kind == "rect":
            p0, p1 = self._drag[1], e.position()
            x0, y0 = self._to_norm(p0)
            x1, y1 = self._to_norm(p1)
            x0, x1 = sorted((min(max(x0, 0), 1), min(max(x1, 0), 1)))
            y0, y1 = sorted((min(max(y0, 0), 1), min(max(y1, 0), 1)))
            if (x1 - x0) * self._img_w >= 2 and (y1 - y0) * self._img_h >= 2:
                self.rect_drawn.emit({"type": "rect", "mode": "subtract" if self._drag[2] else "add",
                                      "x0": x0, "y0": y0, "x1": x1, "y1": y1})
        elif kind == "stroke":
            pts = [list(self._to_norm(q)) for q in self._drag[1]]
            pts = [[round(x, 5), round(y, 5)] for x, y in pts][:4000]
            self.stroke_drawn.emit({"type": "stroke", "mode": "subtract" if self._tool == TOOL_ERASE else "add",
                                    "points": pts, "radius": float(self.brush_radius)})
        self._drag = None
        self.setCursor(Qt.OpenHandCursor if self._tool == TOOL_PAN else Qt.CrossCursor)
        self.update()

    def leaveEvent(self, e):  # noqa: N802
        self._cursor_pos = None
        self.update()

    def keyPressEvent(self, e):  # noqa: N802
        if e.key() == Qt.Key_Space and not e.isAutoRepeat():
            self._space = True
            self.setCursor(Qt.OpenHandCursor)
        super().keyPressEvent(e)

    def keyReleaseEvent(self, e):  # noqa: N802
        if e.key() == Qt.Key_Space and not e.isAutoRepeat():
            self._space = False
        super().keyReleaseEvent(e)

    # ---------------------------------------------------------- drag & drop
    def dragEnterEvent(self, e):  # noqa: N802
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):  # noqa: N802
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if paths:
            self.files_dropped.emit(paths)
