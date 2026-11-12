"""Glitch Lab main window."""

from __future__ import annotations

import logging
import os
import time

import numpy as np
from PySide6.QtCore import QByteArray, Qt, QTimer
from PySide6.QtGui import QAction, QIcon, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStyle,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .. import APP_NAME, __version__
from ..engine.effects.base import get_effect
from ..engine.errors import Cancelled, GlitchLabError, ImageTooLargeError
from ..engine.export import (
    SourceInfo,
    batch_export,
    export_rendered,
    load_recipe,
    path_taken,
    plan_batch_paths,
    render_signature,
)
from ..engine.history import History
from ..engine.image_io import DEFAULT_MAX_PIXELS, file_filter_string, load_image, probe_image
from ..engine.image_io import thumbnail_rgba
from ..engine.masks import rasterize_mask
from ..engine.model import Document, EffectInstance, Layer, random_seed
from ..engine.project import Favourite, Project, load_project, project_from_image, save_project
from ..engine.variations import generate_variations
from ..engine.pipeline import RenderCache, image_hash, make_preview_source, render_document, render_full
from ..engine.presets import (
    apply_preset,
    available_builtin_presets,
    delete_user_preset,
    load_user_presets,
    save_user_preset,
    stack_to_preset,
)
from ..engine.services import EngineServices
from ..engine.settings import Settings
from . import theme
from .canvas import TOOL_BRUSH, TOOL_ERASE, TOOL_PAN, TOOL_RECT, PreviewCanvas
from .effect_panel import EffectPanel
from .export_dialog import ExportDialog, ask_conflict
from .layers_panel import LayersPanel
from .panels import EffectLibrary, ImageBrowser, LogView, PresetList, QtLogHandler, StackPanel
from .qt_image import mask_overlay_qimage, rgba_to_pixmap
from .variations_panel import FavouritesPanel, VariationsPanel
from .widgets import ElidedLabel, fit_rect_to_screen
from .workers import TaskRunner

log = logging.getLogger(__name__)

PREVIEW_SIZES = ((512, "512 px"), (768, "768 px"), (1024, "1024 px"), (1536, "1536 px"), (2048, "2048 px"),
                 (0, "Full resolution"))


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings | None = None, services: EngineServices | None = None):
        super().__init__()
        self.settings = settings or Settings()
        self.services = services or EngineServices(ffmpeg_path=self.settings.get("ffmpeg_path") or None)
        self.runner = TaskRunner(self)
        self.cache = RenderCache(max_bytes=int(self.settings.get("cache_megabytes")) * 1024 * 1024)
        self.history = History()
        self.doc = Document.new()
        self.history.reset(self.doc.to_dict())
        self.image = None            # LoadedImage
        self.preview_src: np.ndarray | None = None
        self.preview_scale = 1.0
        self.source_key = ""
        self.selected_uid = ""
        self.failures: dict[str, str] = {}
        self.full_result: tuple[str, np.ndarray] | None = None
        self.showing_full = False
        self.last_result: np.ndarray | None = None
        self.dirty = False
        self._orig_pm = {}
        self._preview_started = 0.0
        self.favourites: list[Favourite] = []
        self.project_path: str | None = None
        self.original_bytes: bytes | None = None
        self.original_name = ""
        self._var_docs: list[Document] = []
        self._var_queue: list[int] = []
        self._var_src = None
        self._var_scale = 1.0
        self._var_thumbs: dict[int, np.ndarray] = {}
        self._show_mask_overlay = True
        self._project_hash: str | None = None

        self.setWindowTitle(APP_NAME)
        self.resize(1200, 760)
        self._build_ui()
        self._build_actions()
        self._build_menus()
        self._restore_window()

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(140)
        self._debounce.timeout.connect(self.start_preview)

        self._log_handler = QtLogHandler()
        self._log_handler.bridge.line.connect(self.log_view.add_line)
        logging.getLogger().addHandler(self._log_handler)

        self.refresh_presets()
        self.refresh_all()
        folder = self.settings.get("browser_folder")
        if folder and os.path.isdir(folder):
            self.browser.set_folder(folder)
        log.info("%s %s ready", APP_NAME, __version__)

    # ================================================================ layout
    def _build_ui(self) -> None:
        self.setDockNestingEnabled(True)
        self.canvas = PreviewCanvas()
        self.canvas.files_dropped.connect(lambda paths: self.import_image(paths[0]))
        self.canvas.rect_drawn.connect(self.add_mask_shape)
        self.canvas.stroke_drawn.connect(self.add_mask_shape)
        # The preview and the variations / render-status panel share one vertical
        # splitter, so the panel is always exactly as wide as the preview and
        # aligned with it (it can be resized, but never floats or drifts).
        self.center_split = QSplitter(Qt.Vertical)
        self.center_split.setChildrenCollapsible(False)
        self.center_split.addWidget(self.canvas)
        self.setCentralWidget(self.center_split)

        # Left: image browser + effect library + presets
        left = QSplitter(Qt.Vertical)
        self.browser = ImageBrowser()
        self.browser.open_requested.connect(self.import_image)
        self.library = EffectLibrary()
        self.library.add_requested.connect(self.add_effect)
        self.presets = PresetList()
        self.presets.apply_requested.connect(self.apply_preset)
        self.presets.save_requested.connect(self.save_preset)
        self.presets.delete_requested.connect(self.delete_preset)
        for w in (self.browser, self.library, self.presets):
            holder = QWidget()
            hl = QVBoxLayout(holder)
            hl.setContentsMargins(6, 6, 6, 6)
            hl.addWidget(w)
            left.addWidget(holder)
        left.setSizes([260, 380, 220])
        self.left_dock = self._dock("Library", left, Qt.LeftDockWidgetArea, "leftDock")

        # Right: stack + effect controls
        right = QSplitter(Qt.Vertical)
        self.stack_panel = StackPanel()
        self.stack_panel.selected.connect(self.select_effect)
        self.stack_panel.toggled.connect(self.toggle_effect)
        self.stack_panel.reordered.connect(self.reorder_effects)
        self.stack_panel.add_requested.connect(self.add_effect)
        self.stack_panel.action.connect(self.stack_action)
        sp = QWidget()
        spl = QVBoxLayout(sp)
        spl.setContentsMargins(6, 6, 6, 6)
        spl.addWidget(self.stack_panel)
        right.addWidget(sp)
        self.effect_panel = EffectPanel()
        self.effect_panel.param_changed.connect(self.on_param_changed)
        self.effect_panel.field_changed.connect(self.on_field_changed)
        self.effect_panel.lock_changed.connect(self.on_lock_changed)
        self.effect_panel.range_changed.connect(self.on_range_changed)
        self.effect_panel.reset_requested.connect(lambda uid: self.stack_action("reset", uid))
        self.effect_panel.duplicate_requested.connect(lambda uid: self.stack_action("duplicate", uid))
        self.effect_panel.remove_requested.connect(lambda uid: self.stack_action("remove", uid))
        self.effect_panel.randomize_requested.connect(self.randomize_effect_seed)
        right.addWidget(self.effect_panel)
        right.setSizes([260, 600])
        self.right_tabs = QTabWidget()
        self.right_tabs.addTab(right, "Stack")
        self.layers_panel = LayersPanel()
        self.layers_panel.layer_selected.connect(self.select_layer)
        self.layers_panel.layer_changed.connect(self.on_layer_changed)
        self.layers_panel.layer_action.connect(self.layer_action)
        self.layers_panel.mask_changed.connect(self.on_mask_changed)
        self.layers_panel.mask_clear.connect(self.clear_mask)
        self.layers_panel.tool_changed.connect(self.set_tool)
        self.layers_panel.brush_changed.connect(self._brush_changed)
        self.layers_panel.overlay_toggled.connect(self._overlay_toggled)
        self.layers_panel.alpha_mode_changed.connect(self.on_alpha_mode)
        self.right_tabs.addTab(self.layers_panel, "Layers && Mask")
        self.right_dock = self._dock("Effect", self.right_tabs, Qt.RightDockWidgetArea, "rightDock")

        # Bottom: render status + tabs
        bottom = QWidget()
        bl = QVBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(0)
        strip = QFrame()
        strip.setObjectName("statusStrip")
        sl = QHBoxLayout(strip)
        sl.setContentsMargins(8, 4, 8, 4)
        self.status_label = ElidedLabel("Ready")
        self.mode_label = ElidedLabel("")
        self.mode_label.setObjectName("muted")
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setMaximumWidth(220)
        self.progress.setTextVisible(False)
        self.progress.hide()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setToolTip("Cancel the running render (Esc)")
        self.cancel_btn.clicked.connect(self.cancel_renders)
        self.cancel_btn.setEnabled(False)
        sl.addWidget(QLabel("RENDER"))
        sl.addWidget(self.status_label, 1)
        self.mode_label.setMinimumWidth(80)
        self.mode_label.setMaximumWidth(160)
        sl.addWidget(self.mode_label)
        sl.addWidget(self.progress)
        sl.addWidget(self.cancel_btn)
        bl.addWidget(strip)
        self.bottom_tabs = QTabWidget()
        self.variations = VariationsPanel()
        self.variations.generate_requested.connect(self.generate_variations)
        self.variations.cancel_requested.connect(lambda: self.runner.cancel("variations"))
        self.variations.apply_requested.connect(self.apply_variation)
        self.variations.favourite_requested.connect(self.favourite_variations)
        self.variations.export_requested.connect(self.export_variations)
        self.bottom_tabs.addTab(self.variations, "Variations")
        self.favs_panel = FavouritesPanel()
        self.favs_panel.apply_requested.connect(self.apply_favourite)
        self.favs_panel.remove_requested.connect(self.remove_favourites)
        self.favs_panel.rename_requested.connect(self.rename_favourite)
        self.favs_panel.export_requested.connect(self.export_favourites)
        self.bottom_tabs.addTab(self.favs_panel, "Favourites")
        self.log_view = LogView()
        self.bottom_tabs.addTab(self.log_view, "Log")
        bl.addWidget(self.bottom_tabs, 1)
        bottom.setMinimumHeight(150)
        self.bottom_panel = bottom
        self.center_split.addWidget(bottom)
        self.center_split.setStretchFactor(0, 3)
        self.center_split.setStretchFactor(1, 1)
        self.canvas.setMinimumSize(240, 180)
        self.left_dock.setMinimumWidth(180)
        self.right_dock.setMinimumWidth(310)
        self._docks_sized = False

    def showEvent(self, e):  # noqa: N802
        super().showEvent(e)
        if self._docks_sized:
            return
        self._docks_sized = True
        if not self.settings.data.get("window_geometry"):
            r = fit_rect_to_screen(self, 0.92)
            if r is not None:
                self.setGeometry(r)
        else:
            self._clamp_to_screen()
        if not self.settings.data.get("window_state_v2"):
            w = self.width()
            self.resizeDocks([self.left_dock, self.right_dock], [max(190, int(w * 0.17)), max(300, int(w * 0.25))],
                             Qt.Horizontal)
        split = self.settings.data.get("center_split")
        if split:
            self.center_split.restoreState(QByteArray.fromBase64(split.encode()))
        else:
            h = self.center_split.height()
            self.center_split.setSizes([int(h * 0.6), int(h * 0.4)])

    def _clamp_to_screen(self) -> None:
        """Make sure a restored window is fully visible on the current screen."""
        screen = self.screen()
        if screen is None:
            return
        avail = screen.availableGeometry()
        g = self.frameGeometry()
        w = min(self.width(), avail.width() - (g.width() - self.width()))
        h = min(self.height(), avail.height() - (g.height() - self.height()))
        if (w, h) != (self.width(), self.height()):
            self.resize(max(w, 640), max(h, 480))
            g = self.frameGeometry()
        x = min(max(g.x(), avail.x()), avail.right() - g.width() + 1)
        y = min(max(g.y(), avail.y()), avail.bottom() - g.height() + 1)
        self.move(max(x, avail.x()), max(y, avail.y()))

    def _dock(self, title: str, widget: QWidget, area, name: str) -> QDockWidget:
        d = QDockWidget(title, self)
        d.setObjectName(name)
        d.setWidget(widget)
        d.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(area, d)
        return d

    def _icon(self, sp) -> QIcon:
        return self.style().standardIcon(sp)

    def _build_actions(self) -> None:
        def act(text, slot, shortcut=None, tip="", icon=None, checkable=False):
            a = QAction(text, self)
            if icon is not None:
                a.setIcon(self._icon(icon))
            if shortcut:
                a.setShortcuts([QKeySequence(s) for s in (shortcut if isinstance(shortcut, (list, tuple)) else [shortcut])])
            sc = " / ".join(s.toString(QKeySequence.NativeText) for s in a.shortcuts())
            a.setToolTip(f"{tip or text}" + (f"  ({sc})" if sc else ""))
            a.setStatusTip(tip or text)
            a.setCheckable(checkable)
            if checkable:
                a.toggled.connect(slot)
            else:
                a.triggered.connect(slot)
            self.addAction(a)
            return a

        S = QStyle.StandardPixmap
        self.a_import = act("Import…", self.import_dialog, QKeySequence.Open, "Import a photograph (JPEG, PNG, WebP, BMP, TIFF)", S.SP_DialogOpenButton)
        self.a_export = act("Export…", self.export_dialog, "Ctrl+E", "Render at full resolution and export", S.SP_DialogSaveButton)
        self.a_undo = act("Undo", self.undo, QKeySequence.Undo, "Undo", S.SP_ArrowBack)
        self.a_redo = act("Redo", self.redo, ["Ctrl+Y", "Ctrl+Shift+Z"], "Redo", S.SP_ArrowForward)
        self.a_reset = act("Reset", self.reset_document, "Ctrl+Shift+R", "Reset to the original image (removes all effects; undoable)", S.SP_BrowserReload)
        self.a_render = act("Render", self.render_full_resolution, ["F5", "Ctrl+Return"], "Render the full-resolution result", S.SP_MediaPlay)
        self.a_cancel = act("Cancel", self.cancel_renders, "Esc", "Cancel running renders", S.SP_BrowserStop)
        self.a_fit = act("Fit", self.canvas.fit, "Ctrl+0", "Fit image to window")
        self.a_100 = act("100%", self.canvas.actual_size, "Ctrl+1", "View at 100% (one image pixel per screen pixel)")
        self.a_zin = act("Zoom in", self.canvas.zoom_in, ["Ctrl+=", "Ctrl++"], "Zoom in")
        self.a_zout = act("Zoom out", self.canvas.zoom_out, "Ctrl+-", "Zoom out")
        self.a_compare = act("Compare", self.canvas.set_compare, "C", "Split view: original | result (drag the divider)", checkable=True)
        self.a_checker = act("Checkerboard", self._toggle_checker, None, "Show a checkerboard behind transparency", checkable=True)
        self.a_checker.setChecked(bool(self.settings.get("show_checkerboard")))
        self.a_seed = act("Randomise effect seed", lambda: self.randomize_effect_seed(self.selected_uid), "R", "Give the selected effect a new random seed")
        self.a_gseed = act("Randomise global seed", self.randomize_global_seed, "Ctrl+R", "New global seed (affects every seeded effect)")
        self.a_dup = act("Duplicate effect", lambda: self.stack_action("duplicate", self.selected_uid), "Ctrl+D")
        self.a_del = act("Remove effect", lambda: self.stack_action("remove", self.selected_uid), QKeySequence.Delete)
        self.a_up = act("Move effect up", lambda: self.stack_action("up", self.selected_uid), "Ctrl+Up")
        self.a_down = act("Move effect down", lambda: self.stack_action("down", self.selected_uid), "Ctrl+Down")
        self.a_bypass = act("Bypass effect", lambda: self.stack_action("bypass", self.selected_uid), "B", "Toggle the selected effect on/off")
        self.a_open = act("Open project…", self.open_project_dialog, "Ctrl+Shift+O", "Open a Glitch Lab project (.glab)")
        self.a_save = act("Save project", self.save_project, QKeySequence.Save, "Save the project (image copy, layers, stacks, seeds, masks, favourites)")
        self.a_save_as = act("Save project as…", lambda: self.save_project(save_as=True), "Ctrl+Shift+S")
        self.a_recipe = act("Apply recipe (JSON)…", self.apply_recipe_dialog, None, "Load the recipe written next to an exported image")
        self.a_vary = act("Generate variations", lambda: self.generate_variations(self.variations.settings()), "G", "Render a grid of seeded variations")
        self.a_tool_pan = act("Pan tool", lambda: self.set_tool(TOOL_PAN), "H")
        self.a_tool_rect = act("Rectangle mask tool", lambda: self.set_tool(TOOL_RECT), "M")
        self.a_tool_brush = act("Brush mask tool", lambda: self.set_tool(TOOL_BRUSH), "P")
        self.a_tool_erase = act("Eraser mask tool", lambda: self.set_tool(TOOL_ERASE), "E")
        self.a_brush_up = act("Larger brush", lambda: self._brush_step(1.25), "]")
        self.a_brush_down = act("Smaller brush", lambda: self._brush_step(0.8), "[")
        self.a_add_layer = act("Add layer", lambda: self.layer_action("add"), "Ctrl+Shift+N")

        tb = self.addToolBar("Main")
        tb.setObjectName("mainToolbar")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        for a in (self.a_import, self.a_export):
            tb.addAction(a)
        tb.addSeparator()
        for a in (self.a_undo, self.a_redo, self.a_reset):
            tb.addAction(a)
        tb.addSeparator()
        for a in (self.a_render, self.a_cancel):
            tb.addAction(a)
        tb.addSeparator()
        for a in (self.a_fit, self.a_100, self.a_compare):
            tb.addAction(a)
        tb.addSeparator()
        tb.addWidget(QLabel(" Preview "))
        self.preview_combo = QComboBox()
        for v, label in PREVIEW_SIZES:
            self.preview_combo.addItem(label, v)
        cur = int(self.settings.get("preview_max_side"))
        self.preview_combo.setCurrentIndex(max(0, self.preview_combo.findData(cur)))
        self.preview_combo.setToolTip("Long side of the reduced-resolution preview. Export always renders at full resolution.")
        self.preview_combo.currentIndexChanged.connect(self._preview_size_changed)
        tb.addWidget(self.preview_combo)
        tb.addSeparator()
        tb.addWidget(QLabel(" Global seed "))
        self.gseed_edit = QLineEdit()
        self.gseed_edit.setFixedWidth(100)
        self.gseed_edit.setToolTip("Global seed, combined with each effect's own seed.")
        self.gseed_edit.editingFinished.connect(self._gseed_edited)
        tb.addWidget(self.gseed_edit)
        dice = QToolButton()
        dice.setDefaultAction(self.a_gseed)
        dice.setText("🎲")
        tb.addWidget(dice)

    def _build_menus(self) -> None:
        mb = self.menuBar()
        self.m_file = mb.addMenu("&File")
        self.m_file.addAction(self.a_import)
        self.m_recent = self.m_file.addMenu("Recent images")
        self.m_recent.aboutToShow.connect(self._fill_recent)
        self.m_file.addSeparator()
        self.m_file.addAction(self.a_open)
        self.m_recent_proj = self.m_file.addMenu("Recent projects")
        self.m_recent_proj.aboutToShow.connect(self._fill_recent_projects)
        self.m_file.addAction(self.a_save)
        self.m_file.addAction(self.a_save_as)
        self.m_file.addSeparator()
        self.m_file.addAction(self.a_export)
        self.m_file.addAction("Export selected variations…", lambda: self.export_variations(self.variations.selected_indices()))
        self.m_file.addAction(self.a_recipe)
        self.m_file.addSeparator()
        self.m_file.addAction("Quit", self.close, QKeySequence.Quit)
        m = mb.addMenu("&Edit")
        for a in (self.a_undo, self.a_redo, self.a_reset):
            m.addAction(a)
        m.addSeparator()
        m.addAction("Preferences…", self.preferences)
        m = mb.addMenu("&View")
        for a in (self.a_fit, self.a_100, self.a_zin, self.a_zout):
            m.addAction(a)
        m.addSeparator()
        m.addAction(self.a_compare)
        m.addAction(self.a_checker)
        m = mb.addMenu("E&ffects")
        from .panels import build_add_menu
        add = build_add_menu(self, self.add_effect)
        add.setTitle("Add effect")
        m.addMenu(add)
        m.addSeparator()
        for a in (self.a_dup, self.a_del, self.a_up, self.a_down, self.a_bypass, self.a_seed, self.a_gseed):
            m.addAction(a)
        m = mb.addMenu("&Layer")
        m.addAction(self.a_add_layer)
        m.addAction("Duplicate layer", lambda: self.layer_action("duplicate"))
        m.addAction("Remove layer", lambda: self.layer_action("remove"))
        m.addSeparator()
        for a in (self.a_tool_pan, self.a_tool_rect, self.a_tool_brush, self.a_tool_erase, self.a_brush_up, self.a_brush_down):
            m.addAction(a)
        m.addAction("Clear mask shapes", self.clear_mask)
        m.addAction("Invert mask", lambda: self.on_mask_changed("invert", not self.doc.layer.mask.invert, True))
        self.m_render = mb.addMenu("&Render")
        self.m_render.addAction(self.a_render)
        self.m_render.addAction(self.a_cancel)
        self.m_render.addSeparator()
        self.m_render.addAction(self.a_vary)
        self.m_render.addAction("Add current recipe to favourites", self.add_current_to_favourites, "Ctrl+B")
        self.m_render.addSeparator()
        self.m_render.addAction("Verify project render", self.verify_project)
        self.m_help = mb.addMenu("&Help")
        self.m_help.addAction("User guide", self._open_user_guide)
        self.m_help.addAction("Keyboard shortcuts", self.show_shortcuts)
        self.m_help.addSeparator()
        self.m_help.addAction("Generate effect contact sheet…", self.contact_sheet_dialog)
        self.m_help.addAction("Run engine self-test", self.run_selftest)
        self.m_help.addAction("Check FFmpeg…", self.check_ffmpeg)
        self.m_help.addAction("Open log folder", self._open_log_folder)
        self.m_help.addAction("About Glitch Lab", self.about)

    def _restore_window(self) -> None:
        g = self.settings.data.get("window_geometry")
        s = self.settings.data.get("window_state_v2")
        try:
            if g:
                self.restoreGeometry(QByteArray.fromBase64(g.encode()))
            if s:
                self.restoreState(QByteArray.fromBase64(s.encode()))
        except Exception:  # pragma: no cover
            pass

    # ============================================================== helpers
    def active_stack(self) -> list[EffectInstance]:
        return self.doc.layer.stack

    def find(self, uid: str) -> EffectInstance | None:
        _layer, inst = self.doc.find_effect(uid)
        return inst

    def commit(self, label: str, *, coalesce: str | None = None, rebuild_panel: bool = False,
               rebuild_stack: bool = True, preview: bool = True) -> None:
        self.history.push(self.doc.to_dict(), label, coalesce_key=coalesce)
        self.dirty = True
        if rebuild_stack:
            self.refresh_stack()
            self.layers_panel.set_document(self.doc)
        if rebuild_panel:
            self.refresh_effect_panel()
        self.update_actions()
        if preview:
            self.schedule_preview()

    def refresh_all(self) -> None:
        self.refresh_stack()
        self.refresh_effect_panel()
        self.layers_panel.set_document(self.doc)
        self.gseed_edit.setText(str(self.doc.global_seed))
        self.update_mask_overlay()
        self.update_actions()

    def refresh_stack(self) -> None:
        stack = self.active_stack()
        if self.selected_uid and not any(e.uid == self.selected_uid for e in stack):
            self.selected_uid = stack[-1].uid if stack else ""
        self.stack_panel.set_stack(stack, self.selected_uid, self.failures, self.doc.layer.name)

    def refresh_effect_panel(self) -> None:
        inst = self.find(self.selected_uid) if self.selected_uid else None
        self.effect_panel.set_instance(inst, self.failures.get(self.selected_uid, "") if inst else "")

    def update_actions(self) -> None:
        self.a_undo.setEnabled(self.history.can_undo)
        self.a_redo.setEnabled(self.history.can_redo)
        ul, rl = self.history.undo_label(), self.history.redo_label()
        self.a_undo.setToolTip(f"Undo{': ' + ul if ul else ''}  (Ctrl+Z)")
        self.a_redo.setToolTip(f"Redo{': ' + rl if rl else ''}  (Ctrl+Y)")
        has = self.image is not None
        for a in (self.a_export, self.a_render, self.a_fit, self.a_100, self.a_compare):
            a.setEnabled(has)
        has_sel = bool(self.selected_uid)
        for a in (self.a_dup, self.a_del, self.a_up, self.a_down, self.a_bypass, self.a_seed):
            a.setEnabled(has_sel)
        if self.project_path:
            name = os.path.basename(self.project_path)
        else:
            name = (os.path.basename(self.image.path) if (self.image and self.image.path) else
                    (self.original_name or "untitled"))
        for a in (self.a_save, self.a_save_as, self.a_vary):
            a.setEnabled(has)
        self.setWindowTitle(f"{name}{' •' if self.dirty else ''} — {APP_NAME}")

    def set_status(self, text: str, kind: str = "info") -> None:
        color = {"error": theme.ERROR, "warning": theme.WARN, "ok": theme.OK}.get(kind, theme.TEXT)
        self.status_label.setStyleSheet(f"color:{color}")
        self.status_label.setText(text)

    def show_error(self, title: str, exc: BaseException) -> None:
        if isinstance(exc, GlitchLabError):
            msg = exc.message
            if exc.detail:
                log.error("%s: %s (%s)", title, exc.message, exc.detail)
            else:
                log.error("%s: %s", title, exc.message)
        else:
            msg = f"Unexpected error: {exc}"
            log.error("%s", title, exc_info=exc)
        self.set_status(f"{title}: {msg}", "error")
        QMessageBox.warning(self, title, msg)

    # ============================================================= importing
    def import_dialog(self) -> None:
        start = self.settings.get("browser_folder") or os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(self, "Import image", start, file_filter_string())
        if path:
            self.import_image(path)

    def import_image(self, path: str, *, frame: int | None = None, interactive: bool = True) -> bool:
        try:
            probe = probe_image(path)
        except GlitchLabError as exc:
            if interactive:
                self.show_error("Import failed", exc)
            else:
                log.error("Import failed: %s", exc.message)
            return False
        if frame is None:
            frame = 0
            if probe.n_frames > 1 and interactive:
                kind = "page" if probe.format == "TIFF" else "frame"
                frame, ok = QInputDialog.getInt(
                    self, "Multi-frame image",
                    f"This {probe.format} file has {probe.n_frames} {kind}s.\n"
                    f"Glitch Lab works on still images: choose the {kind} to import.",
                    1, 1, probe.n_frames)
                if not ok:
                    return False
                frame -= 1
        limit = int(self.settings.get("max_import_megapixels")) * 1_000_000 or DEFAULT_MAX_PIXELS
        try:
            try:
                li = load_image(path, frame=frame, max_pixels=limit)
            except ImageTooLargeError as exc:
                if not interactive:
                    raise
                r = QMessageBox.question(
                    self, "Large image",
                    f"{exc.message}\n\nImport a downscaled working copy that fits the limit?\n"
                    "(The limit can be changed in Preferences.)")
                if r != QMessageBox.Yes:
                    return False
                li = load_image(path, frame=frame, max_pixels=limit, downscale_to_fit=True)
        except GlitchLabError as exc:
            if interactive:
                self.show_error("Import failed", exc)
            return False
        self.set_image(li)
        self.project_path = None
        self.original_name = os.path.basename(path)
        try:
            self.original_bytes = None
            if os.path.getsize(path) <= 200 * 1024 * 1024:
                with open(path, "rb") as fh:
                    self.original_bytes = fh.read()
        except OSError:
            self.original_bytes = None
        self.update_actions()
        self.settings.add_recent("recent_files", os.path.abspath(path))
        folder = os.path.dirname(os.path.abspath(path))
        if folder != self.browser.folder:
            self.browser.set_folder(folder)
            self.settings.set("browser_folder", folder)
        return True

    def set_image(self, li) -> None:
        self.runner.cancel_all()
        self.image = li
        self.cache.clear()
        self.full_result = None
        self.source_key = li.sha256()[:16]
        self._orig_pm = {}
        self._make_preview_source()
        for n in li.notes:
            log.info("Import: %s", n)
        log.info("Imported %s (%dx%d, %s, %s)", li.path or "image", li.width, li.height, li.format,
                 "with alpha" if li.had_alpha else "opaque")
        self.set_status(f"Imported {os.path.basename(li.path or 'image')} — {li.width}×{li.height}"
                        f"{' · ' + li.notes[0] if li.notes else ''}", "ok")
        self.canvas.clear()
        self._var_docs = []
        self.variations.grid.clear()
        self.update_actions()
        self.update_mask_overlay()
        self.start_preview()

    def _make_preview_source(self) -> None:
        if self.image is None:
            return
        max_side = int(self.preview_combo.currentData() or 0)
        self.preview_src, self.preview_scale = make_preview_source(self.image.rgba, max_side)
        self.preview_src.flags.writeable = False

    def _preview_size_changed(self, _i) -> None:
        self.settings.set("preview_max_side", int(self.preview_combo.currentData()))
        self._orig_pm.pop("preview", None)
        self._make_preview_source()
        self.schedule_preview()

    def _fill_recent(self) -> None:
        self.m_recent.clear()
        for p in self.settings.get("recent_files")[:10]:
            self.m_recent.addAction(p, lambda _=False, q=p: self.import_image(q))
        if self.m_recent.isEmpty():
            self.m_recent.addAction("(none)").setEnabled(False)

    # ============================================================= rendering
    def schedule_preview(self) -> None:
        self.showing_full = False
        self._debounce.start()

    def _original_pixmap(self, kind: str):
        if kind not in self._orig_pm:
            src = self.image.rgba if kind == "full" else self.preview_src
            self._orig_pm[kind] = rgba_to_pixmap(src)
        return self._orig_pm[kind]

    def start_preview(self) -> None:
        if self.image is None or self.preview_src is None:
            return
        doc = self.doc.copy()
        src = self.preview_src
        scale = self.preview_scale
        full_size = (self.image.width, self.image.height)
        prefix = self.source_key + ":p"
        cache, services = self.cache, self.services
        self._preview_started = time.perf_counter()
        self._render_busy("Rendering preview…")

        def work(token, progress):
            return render_document(src, doc, scale=scale, full_size=full_size, services=services, cancel=token,
                                   cache=cache, cache_prefix=prefix, on_error="skip", progress=progress)

        self.runner.submit("preview", work, on_done=self._preview_done, on_error=self._preview_error,
                           on_progress=self._on_progress, on_cancel=lambda: self._render_idle("Preview cancelled", "warning"))

    def _render_busy(self, text: str) -> None:
        self.canvas.set_busy(text)
        self.cancel_btn.setEnabled(True)
        self.progress.show()
        self.progress.setValue(0)

    def _render_idle(self, text: str = "", kind: str = "info") -> None:
        busy = any(self.runner.busy(l) for l in ("preview", "full", "export"))
        if not busy:
            self.canvas.set_busy("")
            self.cancel_btn.setEnabled(False)
            self.progress.hide()
        if text:
            self.set_status(text, kind)

    def _on_progress(self, frac: float, msg: str) -> None:
        self.progress.setValue(int(frac * 1000))
        if msg:
            self.canvas.set_busy(f"Rendering · {msg}")

    def _apply_failures(self, failures) -> None:
        new = {f.uid: f.message for f in failures}
        if new != self.failures:
            self.failures = new
            self.refresh_stack()
            self.effect_panel.set_status(self.failures.get(self.selected_uid, ""))
        if failures:
            first = failures[0]
            self.canvas.set_error(f"⚠ {first.name} failed and was skipped: {first.message}"
                                  + (f"  (+{len(failures) - 1} more)" if len(failures) > 1 else ""))
        else:
            self.canvas.set_error("")

    def _preview_done(self, res) -> None:
        self.last_result = res.image
        self._apply_failures(res.failures)
        w, h = self.image.width, self.image.height
        pw, ph = res.size
        reduced = pw != w
        label = (f"PREVIEW {pw}×{ph} · REDUCED {pw / w * 100:.0f}%" if reduced else f"PREVIEW {pw}×{ph} · FULL SIZE")
        self.canvas.set_images(rgba_to_pixmap(res.image), self._original_pixmap("preview" if reduced else "full"),
                               (w, h), label, "preview")
        for wmsg in res.warnings:
            log.warning(wmsg)
        cached = f" · {res.cache_hits} cached" if res.cache_hits else ""
        kind = "error" if res.failures else ("warning" if res.warnings else "ok")
        text = f"Preview {pw}×{ph} in {res.elapsed:.2f} s{cached}"
        if res.failures:
            text += f" · {len(res.failures)} effect(s) failed: " + "; ".join(f"{f.name}: {f.message}" for f in res.failures)
        elif res.warnings:
            text += " · " + res.warnings[-1]
        self.mode_label.setText("reduced preview" if reduced else "preview at full size")
        self._render_idle(text, kind)
        self.on_preview_rendered(res)

    def on_preview_rendered(self, res) -> None:
        """Hook for panels that react to fresh previews."""

    def _preview_error(self, exc) -> None:
        self._render_idle()
        self.show_error("Preview failed", exc)

    def render_full_resolution(self) -> None:
        if self.image is None:
            return
        doc = self.doc.copy()
        sig = render_signature(doc)
        if self.full_result and self.full_result[0] == sig:
            self._show_full(self.full_result[1])
            return
        src, services, cache = self.image.rgba, self.services, self.cache
        prefix = self.source_key + ":f"
        self._render_busy("Rendering full resolution…")
        self.set_status(f"Rendering {self.image.width}×{self.image.height}…")
        t0 = time.perf_counter()

        def work(token, progress):
            return render_full(src, doc, services=services, cancel=token, progress=progress, cache=cache,
                               cache_prefix=prefix)

        def done(res):
            self.full_result = (sig, res.image)
            self._show_full(res.image)
            self._render_idle(f"Full resolution {res.size[0]}×{res.size[1]} rendered in {time.perf_counter() - t0:.2f} s", "ok")

        def err(exc):
            self._render_idle()
            self.show_error("Full-resolution render failed", exc)

        self.runner.submit("full", work, on_done=done, on_error=err, on_progress=self._on_progress,
                           on_cancel=lambda: self._render_idle("Full render cancelled", "warning"))

    def _show_full(self, img: np.ndarray) -> None:
        self.showing_full = True
        h, w = img.shape[:2]
        self.canvas.set_images(rgba_to_pixmap(img), self._original_pixmap("full"), (w, h),
                               f"FULL RESOLUTION {w}×{h}", "full")
        self.canvas.set_error("")
        self.mode_label.setText("full resolution")

    def cancel_renders(self) -> None:
        any_busy = False
        for lane in ("preview", "full", "export", "variations"):
            any_busy |= self.runner.cancel(lane)
        if any_busy:
            self.set_status("Cancelling…", "warning")

    # ================================================================ export
    def _source_info(self) -> SourceInfo:
        li = self.image
        if li.path:
            name = os.path.splitext(os.path.basename(li.path))[0]
        else:
            name = os.path.splitext(self.original_name)[0] if self.original_name else "image"
        return SourceInfo(name=name, path=li.path or "", sha256=li.sha256(), width=li.width, height=li.height,
                          notes=list(li.notes))

    def export_dialog(self) -> None:
        if self.image is None:
            return
        info = self._source_info()
        folder = self.settings.get("export_folder") or (os.path.dirname(self.image.path) if self.image.path else "")
        dlg = ExportDialog(self, render_size=(self.image.width, self.image.height), source_name=info.name,
                           source_path=self.image.path, seed=self.doc.global_seed, preset=self.doc.preset_name,
                           folder=folder, pattern=self.settings.get("export_pattern"),
                           has_alpha=bool(self.image.had_alpha))
        if dlg.exec() != ExportDialog.Accepted or dlg.request is None:
            return
        req = dlg.request
        self.settings.set("export_folder", req.folder)
        self.settings.set("export_pattern", dlg.pattern.text())
        self.run_export(req)

    def run_export(self, req) -> None:
        """Render at full resolution (or reuse a matching full render) and save."""
        info = self._source_info()
        doc = self.doc.copy()
        sig = render_signature(doc)
        cached = self.full_result[1] if (self.full_result and self.full_result[0] == sig) else None
        src, services, cache = self.image.rgba, self.services, self.cache
        prefix = self.source_key + ":f"
        self._render_busy("Exporting…")

        def work(token, progress):
            if cached is not None:
                img, warnings = cached, []
            else:
                res = render_full(src, doc, services=services, cancel=token, progress=progress, cache=cache,
                                  cache_prefix=prefix)
                img, warnings = res.image, res.warnings
            token.check()
            rep = export_rendered(img, doc, req.path, req.opts, source=info, write_recipe=req.write_recipe,
                                  services=services, warnings=warnings)
            return rep, img

        def done(out):
            rep, img = out
            self.full_result = (sig, img)
            extra = f" + {os.path.basename(rep.recipe_path)}" if rep.recipe_path else ""
            log.info("Exported %s (%dx%d)%s", rep.path, rep.size[0], rep.size[1], extra)
            self._render_idle(f"Exported {os.path.basename(rep.path)} ({rep.size[0]}×{rep.size[1]}){extra}", "ok")

        def err(exc):
            self._render_idle()
            self.show_error("Export failed", exc)

        self.runner.submit("export", work, on_done=done, on_error=err, on_progress=self._on_progress,
                           on_cancel=lambda: self._render_idle("Export cancelled", "warning"))

    # ============================================================ stack edits
    def add_effect(self, effect_id: str) -> None:
        inst = EffectInstance.create(effect_id)
        self.active_stack().append(inst)
        self.selected_uid = inst.uid
        self.commit(f"Add {inst.display_name()}", rebuild_panel=True)

    def select_effect(self, uid: str) -> None:
        if uid == self.selected_uid:
            return
        self.selected_uid = uid
        self.history.break_coalescing()
        self.refresh_effect_panel()
        self.update_actions()

    def toggle_effect(self, uid: str, on: bool) -> None:
        inst = self.find(uid)
        if inst and inst.enabled != on:
            inst.enabled = on
            self.commit(f"{'Enable' if on else 'Bypass'} {inst.display_name()}", rebuild_stack=True,
                        rebuild_panel=(uid == self.selected_uid))

    def reorder_effects(self, uids: list) -> None:
        stack = self.active_stack()
        by = {e.uid: e for e in stack}
        if set(uids) != set(by):
            self.refresh_stack()
            return
        self.doc.layer.stack = [by[u] for u in uids]
        self.commit("Reorder effects")

    def stack_action(self, key: str, uid: str) -> None:
        if not uid:
            return
        stack = self.active_stack()
        i = next((k for k, e in enumerate(stack) if e.uid == uid), -1)
        if i < 0:
            return
        inst = stack[i]
        if key == "up" and i > 0:
            stack[i - 1], stack[i] = stack[i], stack[i - 1]
            self.commit(f"Move {inst.display_name()} up")
        elif key == "down" and i < len(stack) - 1:
            stack[i + 1], stack[i] = stack[i], stack[i + 1]
            self.commit(f"Move {inst.display_name()} down")
        elif key == "duplicate":
            d = inst.duplicate()
            stack.insert(i + 1, d)
            self.selected_uid = d.uid
            self.commit(f"Duplicate {inst.display_name()}", rebuild_panel=True)
        elif key == "remove":
            del stack[i]
            self.selected_uid = stack[min(i, len(stack) - 1)].uid if stack else ""
            self.failures.pop(uid, None)
            self.commit(f"Remove {inst.display_name()}", rebuild_panel=True)
        elif key == "reset":
            inst.reset_params()
            inst.mix = 1.0
            self.commit(f"Reset {inst.display_name()}", rebuild_panel=True)
        elif key == "bypass":
            inst.enabled = not inst.enabled
            self.commit(f"{'Enable' if inst.enabled else 'Bypass'} {inst.display_name()}", rebuild_panel=True)

    def on_param_changed(self, uid: str, name: str, value, final: bool) -> None:
        inst = self.find(uid)
        if inst is None:
            return
        eff = get_effect(inst.effect_id)
        inst.params[name] = eff.param(name).coerce(value)
        self.commit(f"{eff.param(name).label}", coalesce=f"{uid}:{name}", rebuild_stack=False)

    def on_field_changed(self, uid: str, field: str, value, final: bool) -> None:
        inst = self.find(uid)
        if inst is None:
            return
        preview = field not in ("seed_locked", "vary")
        if field == "mix":
            inst.mix = float(min(max(value, 0.0), 1.0))
        elif field == "seed":
            inst.seed = int(value) & 0xFFFFFFFF
        else:
            setattr(inst, field, value)
        labels = {"mix": "Mix", "enabled": "Enable/bypass", "channels": "Channels", "affect_alpha": "Affect alpha",
                  "seed": "Seed", "seed_locked": "Lock seed", "vary": "Include in variations"}
        self.commit(labels.get(field, field), coalesce=f"{uid}:{field}",
                    rebuild_stack=field in ("mix", "enabled"), preview=preview)

    def on_lock_changed(self, uid: str, name: str, locked: bool) -> None:
        inst = self.find(uid)
        if inst is None:
            return
        s = set(inst.locked_params)
        s.add(name) if locked else s.discard(name)
        inst.locked_params = sorted(s)
        self.commit("Lock parameter" if locked else "Unlock parameter", rebuild_stack=False, preview=False)

    def on_range_changed(self, uid: str, name: str, rng) -> None:
        inst = self.find(uid)
        if inst is None:
            return
        if rng is None:
            inst.ranges.pop(name, None)
        else:
            inst.ranges[name] = [float(rng[0]), float(rng[1])]
        self.commit("Variation range", rebuild_stack=False, rebuild_panel=True, preview=False)

    def randomize_effect_seed(self, uid: str) -> None:
        inst = self.find(uid)
        if inst is None:
            return
        inst.seed = random_seed()
        self.effect_panel.set_seed(inst.seed)
        self.commit(f"New seed for {inst.display_name()}", rebuild_stack=False)

    def randomize_global_seed(self) -> None:
        self.doc.global_seed = random_seed()
        self.gseed_edit.setText(str(self.doc.global_seed))
        self.commit("New global seed", rebuild_stack=False)

    def _gseed_edited(self) -> None:
        try:
            v = int(self.gseed_edit.text()) & 0xFFFFFFFF
        except ValueError:
            self.gseed_edit.setText(str(self.doc.global_seed))
            return
        if v != self.doc.global_seed:
            self.doc.global_seed = v
            self.commit("Global seed", rebuild_stack=False)
        self.gseed_edit.setText(str(v))

    def reset_document(self) -> None:
        self.doc = Document.new(self.doc.global_seed)
        self.selected_uid = ""
        self.failures = {}
        self.history.push(self.doc.to_dict(), "Reset to original")
        self.dirty = True
        self.refresh_all()
        self.on_document_replaced()
        self.schedule_preview()

    def on_document_replaced(self) -> None:
        """Called after undo/redo/reset/preset/load replaced ``self.doc``."""
        self.layers_panel.set_document(self.doc)
        self.update_mask_overlay()

    def undo(self) -> None:
        state = self.history.undo()
        if state is not None:
            self._restore_state(state)

    def redo(self) -> None:
        state = self.history.redo()
        if state is not None:
            self._restore_state(state)

    def _restore_state(self, state: dict) -> None:
        self.doc = Document.from_dict(state)
        if self.selected_uid and self.find(self.selected_uid) is None:
            self.selected_uid = ""
        self.dirty = True
        self.refresh_all()
        self.on_document_replaced()
        self.schedule_preview()

    # =============================================================== presets
    def refresh_presets(self) -> None:
        self.presets.set_presets(available_builtin_presets(), load_user_presets())

    def apply_preset(self, preset: dict, mode: str) -> None:
        self.doc = apply_preset(self.doc, preset, mode=mode)
        stack = self.active_stack()
        self.selected_uid = stack[-1].uid if stack else ""
        self.failures = {}
        self.history.push(self.doc.to_dict(), f"Preset {preset['name']}")
        self.dirty = True
        self.refresh_all()
        self.on_document_replaced()
        self.schedule_preview()
        self.set_status(f"Applied preset '{preset['name']}' ({'appended' if mode == 'append' else 'replaced stack'})", "ok")

    def save_preset(self) -> None:
        stack = self.active_stack()
        if not stack:
            QMessageBox.information(self, "Save preset", "The active layer has no effects to save.")
            return
        name, ok = QInputDialog.getText(self, "Save preset", "Preset name:")
        if not ok or not name.strip():
            return
        path = save_user_preset(stack_to_preset(name.strip(), stack))
        log.info("Saved preset %s", path)
        self.refresh_presets()

    def delete_preset(self, name: str) -> None:
        if QMessageBox.question(self, "Delete preset", f"Delete user preset '{name}'?") == QMessageBox.Yes:
            delete_user_preset(name)
            self.refresh_presets()

    # ================================================================= misc
    def _toggle_checker(self, on: bool) -> None:
        self.canvas.set_checkerboard(on)
        self.settings.set("show_checkerboard", bool(on))

    def preferences(self) -> None:
        from .dialogs import PreferencesDialog

        dlg = PreferencesDialog(self, self.settings, self.services)
        if dlg.exec():
            self.services.set_ffmpeg_path(self.settings.get("ffmpeg_path") or None)
            self.cache.max_bytes = int(self.settings.get("cache_megabytes")) * 1024 * 1024
            self.schedule_preview()

    def contact_sheet_dialog(self) -> None:
        start = os.path.join(self.settings.get("export_folder") or os.path.expanduser("~"), "glitchlab_contact_sheet.png")
        path, _ = QFileDialog.getSaveFileName(self, "Save contact sheet", start, "PNG image (*.png)")
        if not path:
            return
        from ..engine.contact_sheet import write_contact_sheet
        services = self.services
        self._render_busy("Building contact sheet…")

        def done(out):
            p, results = out
            bad = [r for r in results if r.status != "ok"]
            for r in bad:
                log.warning("Contact sheet: %s %s: %s", r.name, r.status, r.message)
            self._render_idle(f"Contact sheet saved: {os.path.basename(p)} ({len(results) - len(bad)}/{len(results)} OK)",
                              "ok" if not bad else "warning")

        self.runner.submit("export", lambda token, progress: write_contact_sheet(path, services=services, overwrite=True,
                                                                                 progress=progress),
                           on_done=done, on_error=lambda e: (self._render_idle(), self.show_error("Contact sheet", e)),
                           on_progress=self._on_progress)

    def run_selftest(self) -> None:
        from ..selftest import run_selftest
        services = self.services
        self._render_busy("Running self-test…")

        def done(out):
            ok, report = out
            self._render_idle("Self-test passed" if ok else "Self-test FAILED - see log", "ok" if ok else "error")
            for line in report.splitlines():
                log.info("selftest: %s", line)
            QMessageBox.information(self, "Self-test", report[-3000:])

        self.runner.submit("export", lambda token, progress: run_selftest(services), on_done=done,
                           on_error=lambda e: (self._render_idle(), self.show_error("Self-test", e)))

    def _open_user_guide(self) -> None:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        from ..engine.services import app_base_dir
        for cand in (app_base_dir() / "docs" / "USER_GUIDE.md", app_base_dir() / "USER_GUIDE.md"):
            if cand.exists():
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(cand)))
                return
        QMessageBox.information(self, "User guide", "USER_GUIDE.md was not found next to the application.")

    def show_shortcuts(self) -> None:
        rows = []
        for a in self.actions():
            sc = ", ".join(k.toString(QKeySequence.NativeText) for k in a.shortcuts())
            if sc:
                rows.append(f"<tr><td><b>{sc}</b></td><td>&nbsp;&nbsp;{a.text().replace('&', '')}</td></tr>")
        QMessageBox.information(self, "Keyboard shortcuts", "<table>" + "".join(sorted(rows)) + "</table>"
                                "<p>Canvas: wheel = zoom, middle-drag or Space+drag = pan, double-click = fit.</p>")

    def check_ffmpeg(self) -> None:
        from .dialogs import ffmpeg_report

        QMessageBox.information(self, "FFmpeg", ffmpeg_report(self.services))

    def _open_log_folder(self) -> None:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        from ..engine.settings import log_dir
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(log_dir())))

    def about(self) -> None:
        QMessageBox.about(self, f"About {APP_NAME}",
                          f"<h3>{APP_NAME} {__version__}</h3>"
                          "<p>Still-image glitch art workstation. All processing runs locally and offline.</p>"
                          "<p>Built with Python, PySide6 (Qt, LGPLv3), NumPy, Pillow, OpenCV and FFmpeg "
                          "(invoked as a separate program). See THIRD_PARTY_NOTICES.md.</p>")

    def closeEvent(self, e):  # noqa: N802
        if not self.confirm_discard():
            e.ignore()
            return
        self.runner.shutdown()
        self.browser.shutdown()
        try:
            self.settings.data["window_geometry"] = bytes(self.saveGeometry().toBase64()).decode()
            self.settings.data["window_state_v2"] = bytes(self.saveState().toBase64()).decode()
            self.settings.data["center_split"] = bytes(self.center_split.saveState().toBase64()).decode()
            self.settings.save()
        except Exception:  # pragma: no cover
            pass
        logging.getLogger().removeHandler(self._log_handler)
        super().closeEvent(e)

    def confirm_discard(self) -> bool:
        if not self.dirty or self.image is None:
            return True
        r = QMessageBox.question(self, "Unsaved changes", "Save the current project before continuing?",
                                 QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save)
        if r == QMessageBox.Cancel:
            return False
        if r == QMessageBox.Save:
            return self.save_project()
        return True

    # ================================================================ layers
    def select_layer(self, index: int) -> None:
        if not 0 <= index < len(self.doc.layers) or index == self.doc.active_layer:
            return
        self.doc.active_layer = index
        stack = self.active_stack()
        self.selected_uid = stack[-1].uid if stack else ""
        self.refresh_stack()
        self.refresh_effect_panel()
        self.layers_panel.set_document(self.doc)
        self.update_mask_overlay()
        self.update_actions()

    def on_layer_changed(self, field: str, value, final: bool) -> None:
        layer = self.doc.layer
        if field == "opacity":
            layer.opacity = float(min(max(value, 0.0), 1.0))
        elif field == "blend":
            layer.blend = value
        elif field == "visible":
            layer.visible = bool(value)
        elif field == "name":
            layer.name = str(value)[:60]
        self.commit(f"Layer {field}", coalesce=f"layer:{layer.uid}:{field}", rebuild_stack=field == "name",
                    preview=field != "name")
        if field != "opacity":
            self.layers_panel.set_document(self.doc)

    def layer_action(self, key: str) -> None:
        layers = self.doc.layers
        i = self.doc.active_layer
        if key == "add":
            layers.insert(i + 1, Layer(name=f"Layer {len(layers) + 1}"))
            self.doc.active_layer = i + 1
            self.selected_uid = ""
            label = "Add layer"
        elif key == "duplicate":
            d = Layer.from_dict(layers[i].to_dict())
            from ..engine.model import new_uid
            d.uid = new_uid()
            d.name = layers[i].name + " copy"
            for e in d.stack:
                e.uid = new_uid()
            layers.insert(i + 1, d)
            self.doc.active_layer = i + 1
            label = "Duplicate layer"
        elif key == "remove":
            if len(layers) <= 1:
                QMessageBox.information(self, "Layers", "A document needs at least one layer.")
                return
            del layers[i]
            self.doc.active_layer = max(0, i - 1)
            label = "Remove layer"
        elif key == "up" and i < len(layers) - 1:
            layers[i], layers[i + 1] = layers[i + 1], layers[i]
            self.doc.active_layer = i + 1
            label = "Move layer up"
        elif key == "down" and i > 0:
            layers[i], layers[i - 1] = layers[i - 1], layers[i]
            self.doc.active_layer = i - 1
            label = "Move layer down"
        else:
            return
        stack = self.active_stack()
        if not any(e.uid == self.selected_uid for e in stack):
            self.selected_uid = stack[-1].uid if stack else ""
        self.commit(label, rebuild_panel=True)
        self.update_mask_overlay()

    def on_alpha_mode(self, mode: str) -> None:
        if mode != self.doc.alpha_mode:
            self.doc.alpha_mode = mode
            self.commit("Alpha output", rebuild_stack=False)

    # ================================================================= masks
    def set_tool(self, tool: str) -> None:
        self.canvas.set_tool(tool)
        self.layers_panel.set_tool(tool)
        if tool != TOOL_PAN:
            self.right_tabs.setCurrentWidget(self.layers_panel)
            self.update_mask_overlay()

    def _brush_changed(self, v: float) -> None:
        self.canvas.brush_radius = float(v)
        self.canvas.update()

    def _brush_step(self, factor: float) -> None:
        v = min(0.25, max(0.002, self.canvas.brush_radius * factor))
        self.layers_panel.brush.set_value(v)
        self._brush_changed(v)

    def _overlay_toggled(self, on: bool) -> None:
        self._show_mask_overlay = bool(on)
        self.canvas.set_show_mask(on)

    def add_mask_shape(self, shape: dict) -> None:
        mask = self.doc.layer.mask
        if shape.get("type") == "stroke" and len(shape.get("points", [])) == 0:
            return
        mask.shapes.append(shape)
        mask.enabled = True
        kind = "rectangle" if shape["type"] == "rect" else ("eraser stroke" if shape["mode"] == "subtract" else "brush stroke")
        self.commit(f"Mask {kind}", rebuild_stack=True)
        self.update_mask_overlay()

    def on_mask_changed(self, field: str, value, final: bool) -> None:
        mask = self.doc.layer.mask
        setattr(mask, field, value)
        if field in ("feather", "luma_low", "luma_high", "luma_softness"):
            setattr(mask, field, float(value))
        if field in ("invert", "luma_enabled") and value:
            mask.enabled = True
        self.commit(f"Mask {field}", coalesce=f"mask:{self.doc.layer.uid}:{field}", rebuild_stack=False)
        if field in ("invert", "luma_enabled", "enabled"):
            self.layers_panel.set_document(self.doc)
        self.update_mask_overlay()

    def clear_mask(self) -> None:
        mask = self.doc.layer.mask
        if not mask.shapes:
            return
        mask.shapes = []
        self.commit("Clear mask shapes")
        self.update_mask_overlay()

    def update_mask_overlay(self) -> None:
        if self.preview_src is None:
            self.canvas.set_mask_overlay(None)
            return
        mask = self.doc.layer.mask
        if mask.is_identity():
            self.canvas.set_mask_overlay(None)
            return
        h, w = self.preview_src.shape[:2]
        m = rasterize_mask(mask, w, h, self.preview_src)
        self.canvas.set_mask_overlay(None if m is None else QPixmap.fromImage(mask_overlay_qimage(m)))
        self.canvas.set_show_mask(self._show_mask_overlay)

    # ============================================================ variations
    def generate_variations(self, settings) -> None:
        if self.image is None:
            return
        from ..engine.variations import variable_parameter_count
        if variable_parameter_count(self.doc) == 0:
            QMessageBox.information(self, "Variations", "Nothing can vary: add effects, enable "
                                    "'Include in variations' or unlock some parameters.")
            return
        self.bottom_tabs.setCurrentWidget(self.variations)
        self._var_docs = generate_variations(self.doc.copy(), settings)
        for i, d in enumerate(self._var_docs):
            d.preset_name = self.doc.preset_name
        self._var_thumbs = {}
        self._var_src, self._var_scale = make_preview_source(self.image.rgba, 220)
        self._var_queue = list(range(len(self._var_docs)))
        self.variations.start(len(self._var_docs))
        self.variations.set_busy(True)
        log.info("Generating %d variations (batch seed %d, intensity %.0f%%)", len(self._var_docs),
                 settings.batch_seed, settings.intensity * 100)
        self._next_variation()

    def _next_variation(self) -> None:
        if not self._var_queue:
            self.variations.set_busy(False)
            self.set_status(f"{len(self._var_docs)} variations ready — click one to restore its recipe", "ok")
            return
        i = self._var_queue.pop(0)
        doc = self._var_docs[i]
        src, scale = self._var_src, self._var_scale
        full = (self.image.width, self.image.height)
        services = self.services

        def work(token, progress):
            return render_document(src, doc, scale=scale, full_size=full, services=services, cancel=token,
                                   on_error="skip")

        def done(res, _i=i):
            self._var_thumbs[_i] = res.image
            fail = "; ".join(f"{f.name}: {f.message}" for f in res.failures)
            self.variations.set_thumbnail(_i, res.image, fail)
            self.set_status(f"Variation {_i + 1}/{len(self._var_docs)} rendered")
            self._next_variation()

        def err(exc, _i=i):
            self.variations.set_thumbnail(_i, None, str(exc))
            self._next_variation()

        def cancelled():
            self._var_queue = []
            self.variations.set_busy(False)
            self.set_status("Variation rendering stopped", "warning")

        self.runner.submit("variations", work, on_done=done, on_error=err, on_cancel=cancelled)

    def apply_variation(self, i: int) -> None:
        if not 0 <= i < len(self._var_docs):
            return
        self._replace_doc(self._var_docs[i].copy(), f"Apply variation #{i + 1}")
        self.set_status(f"Restored variation #{i + 1} (global seed {self.doc.global_seed})", "ok")

    def _replace_doc(self, doc: Document, label: str) -> None:
        self.doc = doc
        stack = self.active_stack()
        if not any(e.uid == self.selected_uid for e in stack):
            self.selected_uid = stack[-1].uid if stack else ""
        self.failures = {}
        self.history.push(self.doc.to_dict(), label)
        self.dirty = True
        self.refresh_all()
        self.on_document_replaced()
        self.schedule_preview()

    def favourite_variations(self, indices: list) -> None:
        for i in indices:
            if 0 <= i < len(self._var_docs):
                self.favourites.append(Favourite(name=f"Variation {i + 1} · seed {self._var_docs[i].global_seed}",
                                                 document=self._var_docs[i].copy(), thumb=self._var_thumbs.get(i)))
        self.favs_panel.set_favourites(self.favourites)
        self.dirty = True
        self.update_actions()
        if indices:
            self.set_status(f"Added {len(indices)} favourite(s)", "ok")

    def add_current_to_favourites(self) -> None:
        thumb = thumbnail_rgba(self.last_result, 200) if self.last_result is not None else None
        self.favourites.append(Favourite(name=f"Favourite {len(self.favourites) + 1}", document=self.doc.copy(), thumb=thumb))
        self.favs_panel.set_favourites(self.favourites)
        self.dirty = True

    def apply_favourite(self, i: int) -> None:
        if 0 <= i < len(self.favourites):
            self._replace_doc(self.favourites[i].document.copy(), f"Restore favourite '{self.favourites[i].name}'")

    def remove_favourites(self, indices: list) -> None:
        self.favourites = [f for k, f in enumerate(self.favourites) if k not in set(indices)]
        self.favs_panel.set_favourites(self.favourites)
        self.dirty = True

    def rename_favourite(self, i: int) -> None:
        if not 0 <= i < len(self.favourites):
            return
        name, ok = QInputDialog.getText(self, "Rename favourite", "Name:", text=self.favourites[i].name)
        if ok and name.strip():
            self.favourites[i].name = name.strip()
            self.favs_panel.set_favourites(self.favourites)
            self.dirty = True

    def export_favourites(self, indices: list) -> None:
        docs = [self.favourites[i].document for i in indices if 0 <= i < len(self.favourites)]
        self._batch_export_dialog(docs)

    def export_variations(self, indices: list) -> None:
        docs = [self._var_docs[i] for i in indices if 0 <= i < len(self._var_docs)]
        if not docs:
            QMessageBox.information(self, "Export variations", "Select one or more variations first (Ctrl-click).")
            return
        self._batch_export_dialog(docs)

    def _batch_export_dialog(self, docs: list[Document]) -> None:
        if self.image is None or not docs:
            return
        info = self._source_info()
        dlg = ExportDialog(self, render_size=(self.image.width, self.image.height), source_name=info.name,
                           source_path=self.image.path, seed=self.doc.global_seed, preset=self.doc.preset_name,
                           folder=self.settings.get("export_folder") or "", pattern="{name}_var{index}_{seed}",
                           batch_count=len(docs), has_alpha=bool(self.image.had_alpha))
        if dlg.exec() != ExportDialog.Accepted or dlg.request is None:
            return
        req = dlg.request
        planned = plan_batch_paths(docs, req.folder, req.pattern, req.opts, source=info)
        existing = [p for p in planned if path_taken(p, with_recipe=req.write_recipe)]
        policy = "error"
        if existing:
            policy = ask_conflict(self, existing, batch=True)
            if policy is None:
                return
        self.run_batch_export(docs, req, on_conflict=policy)

    def run_batch_export(self, docs: list[Document], req, on_conflict: str | None = None) -> None:
        info = self._source_info()
        src, services = self.image.rgba, self.services
        docs = [d.copy() for d in docs]
        self.settings.set("export_folder", req.folder)

        def work(token, progress):
            try:
                return batch_export(src, docs, req.folder, req.pattern, req.opts, source=info,
                                    write_recipe=req.write_recipe, services=services, cancel=token, progress=progress,
                                    on_conflict=on_conflict)
            except GlitchLabError as exc:
                if exc.detail == "exists":
                    raise GlitchLabError(exc.message + " Choose another folder or pattern.") from exc
                raise

        def done(reports):
            for r in reports:
                log.info("Exported %s", r.path)
            self._render_idle(f"Exported {len(reports)} image(s) to {req.folder}", "ok")

        def err(exc):
            self._render_idle()
            self.show_error("Batch export failed", exc)

        self._render_busy(f"Exporting {len(docs)} variation(s)…")
        self.runner.submit("export", work, on_done=done, on_error=err, on_progress=self._on_progress,
                           on_cancel=lambda: self._render_idle("Batch export cancelled", "warning"))

    # ============================================================== projects
    def save_project(self, save_as: bool = False) -> bool:
        if self.image is None:
            return False
        path = self.project_path
        if save_as or not path:
            base = os.path.splitext(self.original_name or "untitled")[0] + ".glab"
            start = os.path.join(self.settings.get("export_folder") or os.path.expanduser("~"), base)
            path, _ = QFileDialog.getSaveFileName(self, "Save project", start, "Glitch Lab project (*.glab)")
            if not path:
                return False
            if not path.lower().endswith(".glab"):
                path += ".glab"
        sig = render_signature(self.doc)
        proj = Project(document=self.doc.copy(), source=self.image, original_bytes=self.original_bytes,
                       original_name=self.original_name, favourites=self.favourites,
                       presets=[p for p in load_user_presets() if p["name"] in {self.doc.preset_name}],
                       selection={"selected_uid": self.selected_uid,
                                  "preview_max_side": int(self.preview_combo.currentData() or 0),
                                  "compare": self.canvas.compare(), "split": self.canvas.split(),
                                  "variation_settings": self.variations.settings().to_dict()},
                       render_sha256=image_hash(self.full_result[1]) if (self.full_result and self.full_result[0] == sig) else None)
        try:
            save_project(path, proj)
        except GlitchLabError as exc:
            self.show_error("Save failed", exc)
            return False
        self.project_path = path
        self.dirty = False
        self.settings.add_recent("recent_projects", os.path.abspath(path))
        self.update_actions()
        self.set_status(f"Project saved: {os.path.basename(path)}", "ok")
        log.info("Saved project %s", path)
        return True

    def open_project_dialog(self) -> None:
        if not self.confirm_discard():
            return
        start = self.settings.get("export_folder") or os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(self, "Open project", start, "Glitch Lab project (*.glab)")
        if path:
            self.open_project(path)

    def open_project(self, path: str, *, interactive: bool = True) -> bool:
        try:
            proj = load_project(path)
        except GlitchLabError as exc:
            if interactive:
                self.show_error("Open project failed", exc)
            else:
                log.error("Open project failed: %s", exc.message)
            return False
        self.set_image(proj.source)
        self.original_bytes = proj.original_bytes
        self.original_name = proj.original_name
        self.project_path = path
        self.doc = proj.document
        self.favourites = proj.favourites
        self.favs_panel.set_favourites(self.favourites)
        sel = proj.selection or {}
        self.selected_uid = sel.get("selected_uid", "")
        if self.find(self.selected_uid) is None or self.doc.find_effect(self.selected_uid)[0] is not self.doc.layer:
            stack = self.active_stack()
            self.selected_uid = stack[-1].uid if stack else ""
        if sel.get("compare"):
            self.a_compare.setChecked(True)
            self.canvas.set_split(float(sel.get("split", 0.5)))
        self.history.reset(self.doc.to_dict(), "Open project")
        self.failures = {}
        self.dirty = False
        self.refresh_all()
        self.on_document_replaced()
        self.schedule_preview()
        self.settings.add_recent("recent_projects", os.path.abspath(path))
        self.set_status(f"Opened project {os.path.basename(path)} ({proj.source.width}×{proj.source.height})", "ok")
        if proj.render_sha256:
            log.info("Project records full-render hash %s… (Render > Verify project to check)", proj.render_sha256[:12])
        self._project_hash = proj.render_sha256
        return True

    def verify_project(self) -> None:
        expected = getattr(self, "_project_hash", None)
        if self.image is None or not expected:
            QMessageBox.information(self, "Verify", "Open a project that was saved after a full-resolution render.")
            return
        src, doc, services = self.image.rgba, self.doc.copy(), self.services
        self._render_busy("Verifying project render…")

        def done(res):
            ok = image_hash(res.image) == expected
            self._render_idle("Project verified: full render matches the saved hash" if ok else
                              "Full render differs from the saved hash (different software environment?)",
                              "ok" if ok else "warning")

        self.runner.submit("full", lambda token, progress: render_full(src, doc, services=services, cancel=token,
                                                                       progress=progress),
                           on_done=done, on_error=lambda e: (self._render_idle(), self.show_error("Verify failed", e)),
                           on_progress=self._on_progress, on_cancel=lambda: self._render_idle("Verify cancelled"))

    def _fill_recent_projects(self) -> None:
        self.m_recent_proj.clear()
        for p in self.settings.get("recent_projects")[:10]:
            self.m_recent_proj.addAction(p, lambda _=False, q=p: self.confirm_discard() and self.open_project(q))
        if self.m_recent_proj.isEmpty():
            self.m_recent_proj.addAction("(none)").setEnabled(False)

    def apply_recipe_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Apply recipe", self.settings.get("export_folder") or "",
                                              "Glitch Lab recipe (*.json)")
        if not path:
            return
        try:
            doc, data = load_recipe(path)
        except GlitchLabError as exc:
            self.show_error("Recipe", exc)
            return
        self._replace_doc(doc, f"Apply recipe {os.path.basename(path)}")
        src = data.get("source", {})
        if self.image is not None and src.get("sha256") and src["sha256"] != self.image.sha256():
            self.set_status("Recipe applied — note: it was made for a different source image", "warning")
