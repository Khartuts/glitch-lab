"""numpy <-> Qt image conversion."""

from __future__ import annotations

import numpy as np
from PySide6.QtGui import QIcon, QImage, QPixmap


def rgba_to_qimage(arr: np.ndarray) -> QImage:
    """uint8 RGBA (H, W, 4) -> detached QImage (safe after ``arr`` is freed)."""
    a = np.ascontiguousarray(arr)
    h, w = a.shape[:2]
    qi = QImage(a.data, w, h, 4 * w, QImage.Format_RGBA8888)
    return qi.copy()


def rgba_to_pixmap(arr: np.ndarray) -> QPixmap:
    return QPixmap.fromImage(rgba_to_qimage(arr))


def mask_overlay_qimage(mask: np.ndarray, color=(255, 61, 127), alpha: float = 0.45) -> QImage:
    """Float mask (H, W) -> tinted RGBA overlay of the *unselected* area."""
    h, w = mask.shape
    out = np.zeros((h, w, 4), np.uint8)
    out[..., 0], out[..., 1], out[..., 2] = color
    out[..., 3] = np.clip((1.0 - mask) * 255 * alpha, 0, 255).astype(np.uint8)
    return rgba_to_qimage(out)


def thumbnail_icon(arr: np.ndarray, size: int = 128) -> QIcon:
    from ..engine.image_io import thumbnail_rgba

    return QIcon(rgba_to_pixmap(thumbnail_rgba(arr, size)))
