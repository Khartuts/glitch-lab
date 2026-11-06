"""Dark theme: Fusion style + palette + a compact stylesheet."""

from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

BG = "#141519"
PANEL = "#1b1c21"
PANEL_2 = "#22242b"
BORDER = "#2e3038"
TEXT = "#e4e5ea"
MUTED = "#8d90a0"
ACCENT = "#ff3d7f"     # hot magenta
ACCENT_2 = "#2fe0ff"   # cyan
WARN = "#ffb347"
ERROR = "#ff5c5c"
OK = "#5ee08a"

STYLESHEET = f"""
QWidget {{ color: {TEXT}; font-size: 9.5pt; }}
QMainWindow, QDialog {{ background: {BG}; }}
QDockWidget {{ titlebar-close-icon: none; }}
QDockWidget::title {{
    background: {PANEL}; padding: 5px 8px; border-bottom: 1px solid {BORDER};
    font-weight: 600; letter-spacing: 1px; color: {MUTED};
}}
QToolBar {{ background: {PANEL}; border: 0; border-bottom: 1px solid {BORDER}; spacing: 2px; padding: 3px; }}
QToolBar::separator {{ background: {BORDER}; width: 1px; margin: 4px 6px; }}
QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 5px; padding: 4px 7px; }}
QToolButton:hover {{ background: {PANEL_2}; border-color: {BORDER}; }}
QToolButton:pressed, QToolButton:checked {{ background: #2c1622; border-color: {ACCENT}; }}
QToolButton:disabled {{ color: #555866; }}
QToolButton#lockButton {{ padding: 1px 3px; font-size: 8pt; }}
QToolButton#lockButton:checked {{ background: #2a2410; border-color: {WARN}; color: {WARN}; }}
QPushButton {{
    background: {PANEL_2}; border: 1px solid {BORDER}; border-radius: 5px; padding: 5px 12px;
}}
QPushButton:hover {{ border-color: {ACCENT_2}; }}
QPushButton:pressed {{ background: #1a2a30; }}
QPushButton:disabled {{ color: #5a5d6a; border-color: #26272d; }}
QPushButton#primary {{ background: {ACCENT}; color: #120a0e; font-weight: 700; border: 0; }}
QPushButton#primary:hover {{ background: #ff6b9c; }}
QPushButton#primary:disabled {{ background: #4a2633; color: #806070; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
    background: #101115; border: 1px solid {BORDER}; border-radius: 4px; padding: 3px 5px;
    selection-background-color: {ACCENT};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color: {ACCENT_2}; }}
QLineEdit[invalid="true"] {{ border-color: {ERROR}; }}
QComboBox::drop-down {{ border: 0; width: 18px; }}
QComboBox QAbstractItemView {{ background: {PANEL_2}; border: 1px solid {BORDER}; selection-background-color: #3a1a2a; }}
QSlider::groove:horizontal {{ height: 4px; background: #2b2d35; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {ACCENT_2}, stop:1 {ACCENT}); border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {TEXT}; width: 12px; height: 12px; margin: -5px 0; border-radius: 6px; }}
QSlider::handle:horizontal:hover {{ background: #ffffff; }}
QSlider:disabled::sub-page:horizontal {{ background: #44464f; }}
QListWidget, QTreeWidget, QListView, QTreeView {{
    background: #121317; border: 1px solid {BORDER}; border-radius: 4px; outline: 0;
}}
QListWidget::item, QTreeWidget::item {{ padding: 3px 2px; border-radius: 3px; }}
QListWidget::item:selected, QTreeWidget::item:selected {{ background: #3a1a2a; color: #ffffff; }}
QListWidget::item:hover, QTreeWidget::item:hover {{ background: #22232b; }}
QTabWidget::pane {{ border: 1px solid {BORDER}; border-radius: 4px; top: -1px; background: {PANEL}; }}
QTabBar::tab {{ background: transparent; padding: 5px 12px; border: 0; color: {MUTED}; }}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}
QTabBar::tab:hover {{ color: {TEXT}; }}
QGroupBox {{ border: 1px solid {BORDER}; border-radius: 5px; margin-top: 10px; padding-top: 6px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; color: {ACCENT_2}; font-weight: 600; }}
QScrollArea {{ border: 0; background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: #34363f; border-radius: 4px; min-height: 24px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: #34363f; border-radius: 4px; min-width: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QProgressBar {{ background: #101115; border: 1px solid {BORDER}; border-radius: 4px; text-align: center; height: 14px; }}
QProgressBar::chunk {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {ACCENT_2}, stop:1 {ACCENT}); border-radius: 3px; }}
QCheckBox::indicator {{ width: 14px; height: 14px; border: 1px solid #4a4c57; border-radius: 3px; background: #101115; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QMenu {{ background: {PANEL_2}; border: 1px solid {BORDER}; padding: 4px; }}
QMenu::item {{ padding: 5px 22px; border-radius: 3px; }}
QMenu::item:selected {{ background: #3a1a2a; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 8px; }}
QMenuBar {{ background: {PANEL}; }}
QMenuBar::item:selected {{ background: #3a1a2a; }}
QStatusBar {{ background: {PANEL}; border-top: 1px solid {BORDER}; color: {MUTED}; }}
QToolTip {{ background: #24262e; color: {TEXT}; border: 1px solid {ACCENT_2}; padding: 4px; }}
QLabel#sectionTitle {{ color: {MUTED}; font-weight: 700; letter-spacing: 1px; font-size: 8.5pt; }}
QLabel#effectTitle {{ font-size: 12pt; font-weight: 700; color: #ffffff; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#warning {{ color: {WARN}; }}
QLabel#error {{ color: {ERROR}; }}
QFrame#statusStrip {{ background: {PANEL}; border-bottom: 1px solid {BORDER}; }}
QSplitter::handle {{ background: {BG}; }}
"""


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    pal = QPalette()
    c = QColor
    pal.setColor(QPalette.Window, c(BG))
    pal.setColor(QPalette.WindowText, c(TEXT))
    pal.setColor(QPalette.Base, c("#101115"))
    pal.setColor(QPalette.AlternateBase, c(PANEL))
    pal.setColor(QPalette.ToolTipBase, c(PANEL_2))
    pal.setColor(QPalette.ToolTipText, c(TEXT))
    pal.setColor(QPalette.Text, c(TEXT))
    pal.setColor(QPalette.Button, c(PANEL_2))
    pal.setColor(QPalette.ButtonText, c(TEXT))
    pal.setColor(QPalette.BrightText, c("#ffffff"))
    pal.setColor(QPalette.Highlight, c(ACCENT))
    pal.setColor(QPalette.HighlightedText, c("#120a0e"))
    pal.setColor(QPalette.Link, c(ACCENT_2))
    pal.setColor(QPalette.PlaceholderText, c(MUTED))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        pal.setColor(QPalette.Disabled, role, c("#5a5d6a"))
    app.setPalette(pal)
    f = app.font()
    if f.pointSizeF() < 9:
        f.setPointSizeF(9.5)
    f.setStyleHint(QFont.SansSerif)
    app.setFont(f)
    app.setStyleSheet(STYLESHEET)
