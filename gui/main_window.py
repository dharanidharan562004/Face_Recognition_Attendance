from __future__ import annotations
"""
main_window.py
Main application window for the Coin Image Processing Tool.
"""

import os
import cv2
import numpy as np
from typing import Optional, List, Tuple, Dict
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QKeySequence, QIcon, QFont
from PyQt5.QtWidgets import (QDoubleSpinBox, 
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QButtonGroup, QGroupBox,
    QLineEdit, QStatusBar, QAction, QToolBar,
    QSplitter, QFrame, QComboBox, QSpinBox,
    QMessageBox, QShortcut, QSizePolicy,
)

from gui.canvas_widget import CanvasWidget
from processing.image_processor import ImageProcessor
from processing.coin_detector import CoinDetector
from processing.qr_detector import QRDetector
from processing.worker import DetectionWorker
from utils.file_handler import (open_image_dialog, save_image_dialog,
                                ensure_extension, save_image,
                                validate_format, InvalidFormatError)


# ─────────────────────────────────────────────────────────────────────────────
#  Styles
# ─────────────────────────────────────────────────────────────────────────────

DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #f2f4f8;
    color: #1c2333;
    font-family: Arial;
    font-size: 13px;
}

/* ── Menu bar ─────────────────────────────────────────────────── */
QMenuBar {
    background-color: #1c2b45;
    color: #e8edf5;
    padding: 2px 4px;
    font-size: 13px;
    font-weight: bold;
}
QMenuBar::item { padding: 4px 12px; border-radius: 4px; }
QMenuBar::item:selected { background-color: #2e4a72; }
QMenu {
    background-color: #1e3050;
    color: #e8edf5;
    border: 1px solid #3a5a88;
    border-radius: 4px;
    padding: 4px;
}
QMenu::item { padding: 5px 24px; border-radius: 3px; }
QMenu::item:selected { background-color: #3e6188; }
QMenu::separator { background-color: #3a5a88; height: 1px; margin: 4px 8px; }

/* ── Toolbar ──────────────────────────────────────────────────── */
QToolBar {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #f8fafc, stop:1 #eef1f6);
    border-bottom: 2px solid #c0cce0;
    spacing: 3px;
    padding: 4px 8px;
    min-height: 42px;
}
QToolBar QToolButton {
    background-color: #eef1f6;
    color: #1c2333;
    border: 1px solid #c0cce0;
    border-radius: 6px;
    padding: 5px 14px;
    margin: 2px 2px;
    font-size: 13px;
    font-weight: bold;
    font-family: Arial;
    min-height: 30px;
}
QToolBar QToolButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #dce8f8, stop:1 #c8d8f0);
    color: #1a3a6a;
    border: 1px solid #8aaace;
    border-radius: 6px;
}
QToolBar QToolButton:pressed {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #3e6188, stop:1 #2a4f70);
    color: #ffffff;
    border: 1px solid #2a4f70;
    border-radius: 6px;
}
QToolBar::separator {
    background-color: #c0cce0;
    width: 1px;
    margin: 6px 6px;
}

/* ── Side panel buttons ───────────────────────────────────────── */
QPushButton {
    background-color: #eef1f6;
    color: #1c2333;
    border: 1px solid #c0cce0;
    border-radius: 6px;
    padding: 6px 14px;
    font-size: 12px;
    font-weight: bold;
    font-family: Arial;
    min-height: 30px;
}
QPushButton:hover  {
    background-color: #dce8f8;
    color: #1a3a6a;
    border: 1px solid #8aaace;
}
QPushButton:pressed {
    background-color: #3e6188;
    color: #ffffff;
    border: 1px solid #2a5070;
}
QPushButton:checked {
    background-color: #3e6188;
    color: #ffffff;
    border: 1px solid #2a5070;
}
QPushButton:disabled { color: #aaa; background-color: #eaecf0; border: 1px solid #d0d8e8; }

/* ── Group boxes ─────────────────────────────────────────────── */
QGroupBox {
    border: 1px solid #c0cce0;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 8px;
    font-weight: bold;
    background-color: #f8fafc;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    color: #3e6188;
    font-size: 12px;
}

/* ── Input fields ────────────────────────────────────────────── */
QLineEdit {
    background-color: #ffffff;
    color: #1c2333;
    border: 1px solid #c0cce0;
    border-radius: 4px;
    padding: 4px 8px;
}
QLineEdit:focus { border: 1px solid #4a8ac4; }

QSpinBox {
    background-color: #ffffff;
    color: #1c2333;
    border: 1px solid #c0cce0;
    border-radius: 4px;
    padding: 3px 6px;
}

QComboBox {
    background-color: #ffffff;
    color: #1c2333;
    border: 1px solid #c0cce0;
    border-radius: 4px;
    padding: 3px 8px;
}
QComboBox::drop-down { border: none; }

/* ── Status bar ──────────────────────────────────────────────── */
QStatusBar {
    background-color: #1c2b45;
    color: #8aaace;
    font-size: 11px;
    padding: 2px 8px;
    border-top: 1px solid #0d1a2e;
}

/* ── Splitter ────────────────────────────────────────────────── */
QSplitter::handle { background-color: #c0cce0; width: 2px; }

/* ── QR preview label ────────────────────────────────────────── */
QLabel#qr_preview {
    background-color: #ffffff;
    border: 1px solid #c0cce0;
    border-radius: 6px;
}

/* ── Scrollbar ───────────────────────────────────────────────── */
QScrollBar:vertical {
    background: #dde3ec;
    width: 8px;
    border-radius: 4px;
    margin: 0px;
}
QScrollBar::handle:vertical {
    background: #5a7aaa;
    border-radius: 4px;
    min-height: 24px;
}
QScrollBar::handle:vertical:hover {
    background: #3e6188;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }

QScrollBar:horizontal {
    background: #dde3ec;
    height: 8px;
    border-radius: 4px;
    margin: 0px;
}
QScrollBar::handle:horizontal {
    background: #5a7aaa;
    border-radius: 4px;
    min-width: 24px;
}
QScrollBar::handle:horizontal:hover {
    background: #3e6188;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }
"""


# ─────────────────────────────────────────────────────────────────────────────
#  Main window
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Coin Image Processing Tool")

        # ── Window flags — enable max/min/close buttons ────────────────
        from PyQt5.QtCore import Qt as _Qt
        self.setWindowFlags(
            _Qt.Window |
            _Qt.WindowMinimizeButtonHint |
            _Qt.WindowMaximizeButtonHint |
            _Qt.WindowCloseButtonHint
        )

        # ── Initial size + minimum size ────────────────────────────────
        self.resize(1100, 720)
        self.setMinimumSize(800, 560)

        self.setStyleSheet(DARK_STYLESHEET)

        # Back-end objects
        self._processor = ImageProcessor()
        self._detector = CoinDetector()
        self._qr_detector = QRDetector()
        self._worker: Optional[DetectionWorker] = None   # background thread

        # Build UI
        self._build_menu()
        self._build_toolbar()
        self._build_central()
        self._build_status_bar()
        self._connect_signals()

        # ── File shortcuts ────────────────────────────────────────────
        # Ctrl+O, Ctrl+S, Ctrl+Z, Ctrl+Q already on QAction — no duplicate
        QShortcut(QKeySequence("Ctrl+W"),       self, self._detect_qr)   # Detect QR
        # Only add shortcuts NOT already in menu/toolbar actions
        QShortcut(QKeySequence("Delete"),       self, self._delete_image)

        # Rotation (not in menu)
        QShortcut(QKeySequence("Ctrl+Left"),    self, self._rotate_ccw_90)
        QShortcut(QKeySequence("Ctrl+Right"),   self, self._rotate_cw_90)
        QShortcut(QKeySequence("["),            self, self._rotate_ccw)
        QShortcut(QKeySequence("]"),            self, self._rotate_cw)

        # Crop apply
        QShortcut(QKeySequence("Return"),       self, self._apply_crop)
        QShortcut(QKeySequence("Enter"),        self, self._apply_crop)

        # Crop move — global arrow keys
        QShortcut(QKeySequence("Up"),           self, lambda: self._on_crop_move(0, -10))
        QShortcut(QKeySequence("Down"),         self, lambda: self._on_crop_move(0,  10))
        QShortcut(QKeySequence("Left"),         self, lambda: self._on_crop_move(-10, 0))
        QShortcut(QKeySequence("Right"),        self, lambda: self._on_crop_move( 10, 0))

        # Crop resize — global +/-
        QShortcut(QKeySequence("+"),            self, lambda: self._on_crop_resize( 15))
        QShortcut(QKeySequence("-"),            self, lambda: self._on_crop_resize(-15))
        QShortcut(QKeySequence("="),            self, lambda: self._on_crop_resize( 15))

        # Thumbnail navigation
        QShortcut(QKeySequence("Alt+Left"),     self, self._thumb_prev)
        QShortcut(QKeySequence("Alt+Right"),    self, self._thumb_next)
        QShortcut(QKeySequence("F5"),           self, self._refresh_view)
        QShortcut(QKeySequence("Ctrl+R"),       self, self._refresh_view)

        self._update_ui_state()

    # ------------------------------------------------------------------ #
    #  UI construction                                                     #
    # ------------------------------------------------------------------ #

    def _build_menu(self):
        mb = self.menuBar()

        file_menu = mb.addMenu("&File")
        self._act_open = QAction("&Open Image…", self, shortcut="Ctrl+O",
                                 triggered=self._open_image)
        self._act_open_folder = QAction("Open &Folder…", self,
                                         shortcut="Ctrl+Shift+O",
                                         triggered=self._open_folder)
        self._act_save = QAction("&Save Image…", self, shortcut="Ctrl+S",
                                 triggered=self._save_image)
        self._act_reset     = QAction("&Reset (1 step)", self, shortcut="Ctrl+Z",
                                  triggered=self._undo_step)
        self._act_reset_all = QAction("&Reset All",      self, shortcut="Ctrl+Shift+Z",
                                  triggered=self._reset_image)
        self._act_quit = QAction("&Quit", self, shortcut="Ctrl+Q",
                                 triggered=self.close)
        file_menu.addActions([self._act_open, self._act_open_folder,
                               self._act_save,
                               self._act_reset, self._act_reset_all, self._act_quit])

        view_menu = mb.addMenu("&View")

        # Guide lines toggle
        self._act_guides = QAction("Show &Guide Lines", self)
        self._act_guides.setCheckable(True)
        self._act_guides.setChecked(True)
        self._act_guides.toggled.connect(self._toggle_guides)
        view_menu.addAction(self._act_guides)

        view_menu.addSeparator()

        # Zoom fit
        act_fit = QAction("&Fit Image to Window", self,
                          shortcut="Ctrl+F",
                          triggered=self._fit_image)
        view_menu.addAction(act_fit)

        view_menu.addSeparator()

        # Panel visibility
        self._act_show_panel = QAction("Show &Side Panel", self)
        self._act_show_panel.setCheckable(True)
        self._act_show_panel.setChecked(True)
        self._act_show_panel.toggled.connect(self._toggle_panel)
        view_menu.addAction(self._act_show_panel)

        # Panel position sub-menu
        panel_pos_menu = view_menu.addMenu("  Side Panel &Position")
        self._act_panel_right = QAction("▶  Right  (default)", self)
        self._act_panel_right.setCheckable(True)
        self._act_panel_right.setChecked(True)
        self._act_panel_right.triggered.connect(lambda: self._set_panel_side("right"))
        panel_pos_menu.addAction(self._act_panel_right)

        self._act_panel_left = QAction("◀  Left", self)
        self._act_panel_left.setCheckable(True)
        self._act_panel_left.setChecked(False)
        self._act_panel_left.triggered.connect(lambda: self._set_panel_side("left"))
        panel_pos_menu.addAction(self._act_panel_left)

        help_menu = mb.addMenu("&Help")
        help_menu.addAction(QAction("&Keyboard Shortcuts", self,
                                    triggered=self._show_help))

    def _build_toolbar(self):
        from PyQt5.QtCore import Qt as _Qt
        tb = QToolBar("Main Toolbar", self)
        tb.setMovable(False)
        tb.setToolButtonStyle(_Qt.ToolButtonTextOnly)
        self.addToolBar(_Qt.TopToolBarArea, tb)

        def act(label, slot, tip="", name=""):
            a = QAction(label, self)
            a.triggered.connect(slot)
            a.setToolTip(tip)
            if name:
                a.setObjectName(name)
            return a

        tb.addAction(act("⬆  Open",        self._open_image,    "Open JPG (Ctrl+O)"))
        tb.addAction(act("📁  Folder",      self._open_folder,   "Open Folder (Ctrl+Shift+O)"))
        tb.addAction(act("💾  Save",        self._save_image,    "Save (Ctrl+S)"))
        tb.addSeparator()
        tb.addAction(act("⟲  CCW −90°",    self._rotate_ccw_90, "Rotate CCW 90°"))
        tb.addAction(act("⟳  CW  +90°",    self._rotate_cw_90,  "Rotate CW 90°"))
        tb.addSeparator()
        tb.addAction(act("◉  Circle",           self._auto_circle, "Detect Circle"))
        tb.addAction(act("▭  Rectangle",        self._auto_rect,   "Detect Rectangle"))

        # Apply Crop — green
        apply_act = act("✂  Apply Crop", self._apply_crop, "Apply Crop (Enter)", "apply_crop")
        tb.addAction(apply_act)
        apply_btn = tb.widgetForAction(apply_act)
        if apply_btn:
            apply_btn.setStyleSheet("""
                QToolButton {
                    background-color: #eef1f6;
                    color: #1c2333;
                    border: 1px solid #c0cce0;
                    border-radius: 6px;
                    padding: 5px 12px;
                    font-weight: bold;
                    font-size: 12px;
                    min-height: 28px;
                }
                QToolButton:hover {
                    background-color: #c8f0d0;
                    color: #1b5e20;
                    border: 1px solid #66bb6a;
                }
                QToolButton:pressed {
                    background-color: #2e7d32;
                    color: #ffffff;
                    border: 1px solid #1b5e20;
                }
            """)

        tb.addSeparator()
        tb.addAction(act("⊞  Detect QR",        self._detect_qr, "Detect QR (Ctrl+W)"))
        tb.addSeparator()
        tb.addAction(act("↺  Refresh",     self._refresh_view, "Refresh (F5 / Ctrl+R)"))
        tb.addAction(act("⎌  Undo",        self._undo_step,   "Undo 1 step (Ctrl+Z)"))
        tb.addAction(act("⟲  Reset All",   self._reset_image, "Reset to original (Ctrl+Shift+Z)"))
        tb.addSeparator()

        # Delete — red
        del_act = act("🗑  Delete", self._delete_image, "Delete image file (Del)", "delete_btn")
        tb.addAction(del_act)
        del_btn = tb.widgetForAction(del_act)
        if del_btn:
            del_btn.setStyleSheet("""
                QToolButton {
                    background-color: #eef1f6;
                    color: #1c2333;
                    border: 1px solid #c0cce0;
                    border-radius: 6px;
                    padding: 5px 12px;
                    font-weight: bold;
                    font-size: 12px;
                    min-height: 28px;
                }
                QToolButton:hover {
                    background-color: #ffd0d0;
                    color: #b71c1c;
                    border: 1px solid #ef9a9a;
                }
                QToolButton:pressed {
                    background-color: #c62828;
                    color: #ffffff;
                    border: 1px solid #b71c1c;
                }
            """)

    def _build_central(self):
        from gui.thumbnail_panel import ThumbnailPanel
        from PyQt5.QtWidgets import QVBoxLayout as _QVBox

        # Main container
        central = QWidget()
        vbox    = _QVBox(central)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        # Top: canvas + side panel
        self._splitter = QSplitter(Qt.Horizontal)
        self._canvas   = CanvasWidget(self)
        self._splitter.addWidget(self._canvas)
        self._side_panel = self._build_side_panel()
        self._splitter.addWidget(self._side_panel)
        self._splitter.setStretchFactor(0, 3)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setSizes([800, 300])
        vbox.addWidget(self._splitter, 1)

        # Bottom: thumbnail strip — explicit minimum height
        self._thumb_panel = ThumbnailPanel()
        self._thumb_panel.image_selected.connect(self._load_from_thumbnail)
        self._thumb_panel.setMinimumHeight(130)
        self._thumb_panel.setMaximumHeight(130)
        vbox.addWidget(self._thumb_panel, 0)  # 0 = don't stretch

        self.setCentralWidget(central)

    def _build_side_panel(self) -> QWidget:
        from PyQt5.QtWidgets import QScrollArea

        # ── Inner content widget ───────────────────────────────────────
        panel = QWidget()
        panel.setMinimumWidth(270)
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        layout.setContentsMargins(8, 8, 8, 8)

        # ── Rotation ──────────────────────────────────────────────────
        rot_group = QGroupBox("Rotation")
        rot_layout = QVBoxLayout(rot_group)
        row = QHBoxLayout()
        self._btn_ccw = QPushButton("⟲  CCW  −1°")
        self._btn_ccw.setMinimumHeight(36)
        self._btn_ccw.clicked.connect(self._rotate_ccw)
        self._btn_cw = QPushButton("⟳  CW  +1°")
        self._btn_cw.setMinimumHeight(36)
        self._btn_cw.clicked.connect(self._rotate_cw)
        row.addWidget(self._btn_ccw)
        row.addWidget(self._btn_cw)
        rot_layout.addLayout(row)

        # Hidden spinner — keeps backend sync, not shown in UI
        self._spin_angle = QSpinBox()
        self._spin_angle.setRange(0, 270)
        self._spin_angle.setSingleStep(90)
        self._spin_angle.setVisible(False)   # hidden — not shown
        self._spin_angle.valueChanged.connect(self._on_spin_angle)

        layout.addWidget(rot_group)

        # ── Crop type ─────────────────────────────────────────────────
        crop_group = QGroupBox("Crop Mode")
        crop_layout = QVBoxLayout(crop_group)
        crop_layout.setSpacing(6)

        # Circle & Rectangle side by side
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self._btn_circle = QPushButton("◉  Circle")
        self._btn_circle.setCheckable(True)
        self._btn_circle.setMinimumHeight(34)
        self._btn_circle.clicked.connect(self._auto_circle)

        self._btn_rect = QPushButton("▭  Rectangle")
        self._btn_rect.setCheckable(True)
        self._btn_rect.setMinimumHeight(34)
        self._btn_rect.clicked.connect(self._auto_rect)

        self._crop_btn_group = QButtonGroup(self)
        self._crop_btn_group.setExclusive(True)
        self._crop_btn_group.addButton(self._btn_circle)
        self._crop_btn_group.addButton(self._btn_rect)

        btn_row.addWidget(self._btn_circle)
        btn_row.addWidget(self._btn_rect)
        crop_layout.addLayout(btn_row)

        # Pen tool button — disabled until circle/rect detected


        # ── Patch Tool ────────────────────────────────────────────────
        # ── Pen Tool (always enabled) ─────────────────────────────────
        self._btn_pen = QPushButton("✏  Pen Tool")
        self._btn_pen.setCheckable(True)
        self._btn_pen.setMinimumHeight(34)
        self._btn_pen.setEnabled(True)
        self._btn_pen.setToolTip(
            "✏ Pen Tool — Click to place anchor points\n"
            "Click+Drag = curved point\n"
            "Right-click = context menu\n"
            "Enter = Apply Crop  |  Esc = Cancel")
        self._btn_pen.clicked.connect(self._toggle_pen_tool)
        self._btn_pen.setStyleSheet(
            "QPushButton { background:#eef1f6; color:#1c2333; border:1px solid #c0cce0;"
            " border-radius:6px; padding:5px 14px; font-size:12px; font-weight:bold; }"
            "QPushButton:enabled:hover { background:#dce8f8; border:1px solid #8aaace; }"
            "QPushButton:checked { background:#3e6188; color:white; border:1px solid #2a5070; }"
            "QPushButton:disabled { background:#f0f0f0; color:#aaa; border:1px solid #ddd; }")
        crop_layout.addWidget(self._btn_pen)

        self._btn_patch = QPushButton("⬡  Patch Tool")
        self._btn_patch.setCheckable(True)
        self._btn_patch.setMinimumHeight(34)
        self._btn_patch.setToolTip(
            "Patch Tool — Draw a selection around damaged area,\n"
            "then drag to a good area to copy texture.")
        self._btn_patch.clicked.connect(self._toggle_patch_tool)
        self._btn_patch.setStyleSheet(
            "QPushButton { background:#eef1f6; color:#1c2333; border:1px solid #c0cce0;"
            " border-radius:6px; padding:5px 14px; font-size:12px; font-weight:bold; }"
            "QPushButton:enabled:hover { background:#dce8f8; border:1px solid #8aaace; }"
            "QPushButton:checked { background:#3e6188; color:white; border:1px solid #2a5070; }"
            "QPushButton:disabled { background:#f0f0f0; color:#aaa; border:1px solid #ddd; }")
        crop_layout.addWidget(self._btn_patch)

        # Apply Patch button (shown only when patch active)
        self._btn_apply_patch = QPushButton("⬡  Apply Patch")
        self._btn_apply_patch.setMinimumHeight(34)
        self._btn_apply_patch.setToolTip("Apply patch to fix selected area (or press Enter)")
        self._btn_apply_patch.setVisible(False)
        self._btn_apply_patch.clicked.connect(self._apply_patch)
        self._btn_apply_patch.setStyleSheet(
            "QPushButton { background:#3e6188; color:white; border:1px solid #2a5070;"
            " border-radius:6px; padding:5px 14px; font-size:12px; font-weight:bold; }"
            "QPushButton:hover { background:#4a7aaa; }"
            "QPushButton:pressed { background:#2a4f70; }")
        crop_layout.addWidget(self._btn_apply_patch)

        hint = QLabel("Arrow keys: move  |  +/− : resize  |  Enter: apply")
        hint.setStyleSheet("color: #888; font-size: 10px;")
        hint.setWordWrap(True)
        crop_layout.addWidget(hint)

        self._btn_apply_crop = QPushButton("✂  Apply Crop")
        self._btn_apply_crop.setMinimumHeight(36)
        self._btn_apply_crop.setStyleSheet(
            "QPushButton { background-color: #3e6188; color: white; "
            "font-weight: bold; border-radius: 5px; }"
            "QPushButton:hover { background-color: #4a7aaa; }"
            "QPushButton:pressed { background-color: #2a4f70; }"
            "QPushButton:disabled { background-color: #b0bac8; color: #ddd; }"
        )
        self._btn_apply_crop.clicked.connect(self._apply_crop)
        crop_layout.addWidget(self._btn_apply_crop)

        layout.addWidget(crop_group)

        # ── Crop Preview ──────────────────────────────────────────────
        from gui.crop_preview import CropPreviewWidget
        preview_group = QGroupBox("Crop Preview")
        preview_lay   = QVBoxLayout(preview_group)
        preview_lay.setContentsMargins(4, 4, 4, 4)
        preview_lay.setAlignment(Qt.AlignCenter)
        self._crop_preview = CropPreviewWidget()
        preview_lay.addWidget(self._crop_preview)
        layout.addWidget(preview_group)

        # ── Handle Zoom Panel ─────────────────────────────────────────
        from gui.handle_zoom_panel import HandleZoomPanel
        zoom_group = QGroupBox("Handle Zoom (N · E · S · W)")
        zoom_lay   = QVBoxLayout(zoom_group)
        zoom_lay.setContentsMargins(4, 4, 4, 4)
        zoom_lay.setAlignment(Qt.AlignCenter)
        self._handle_zoom = HandleZoomPanel()
        zoom_lay.addWidget(self._handle_zoom)
        layout.addWidget(zoom_group)

        # ── QR / Label ───────────────────────────────────────────────
        qr_group = QGroupBox("QR / Label Detection")
        qr_layout = QVBoxLayout(qr_group)

        self._btn_detect_qr = QPushButton("⊞  Scan QR / Label")
        self._btn_detect_qr.clicked.connect(self._detect_qr)
        qr_layout.addWidget(self._btn_detect_qr)

        # Update QR button
        self._btn_update_qr = QPushButton("🔄  Update QR Code")
        self._btn_update_qr.setMinimumHeight(32)
        self._btn_update_qr.setEnabled(False)
        self._btn_update_qr.setToolTip(
            "Scan QR first, then click to replace QR with a new image file.")
        self._btn_update_qr.clicked.connect(self._update_qr_code)
        self._btn_update_qr.setStyleSheet(
            "QPushButton{background:#eef1f6;color:#1c2333;border:1px solid #c0cce0;"
            "border-radius:6px;padding:5px 14px;font-size:12px;font-weight:bold;}"
            "QPushButton:enabled:hover{background:#dce8f8;border:1px solid #8aaace;}"
            "QPushButton:disabled{background:#f0f0f0;color:#aaa;border:1px solid #ddd;}")
        qr_layout.addWidget(self._btn_update_qr)

        self._lbl_qr_result = QLabel("No code detected yet.")
        self._lbl_qr_result.setWordWrap(True)
        self._lbl_qr_result.setStyleSheet("color: #555; font-size: 11px;")
        qr_layout.addWidget(self._lbl_qr_result)

        self._qr_value_edit = QLineEdit()
        self._qr_value_edit.setPlaceholderText("QR value appears here...")
        self._qr_value_edit.setReadOnly(True)
        self._qr_value_edit.setStyleSheet(
            "background:#f8fafc; color:#1c2333; border:1px solid #c0cce0;"
            "border-radius:4px; padding:3px 6px; font-size:11px;")
        qr_layout.addWidget(self._qr_value_edit)

        self._qr_preview = QLabel()
        self._qr_preview.setObjectName("qr_preview")
        self._qr_preview.setAlignment(Qt.AlignCenter)
        self._qr_preview.setMinimumHeight(100)
        self._qr_preview.setMaximumHeight(100)
        self._qr_preview.setText("QR preview")
        self._qr_preview.setStyleSheet(
            "background-color: #f0f0f0; color: #888; border: 1px solid #c0cce0;"
            "border-radius: 4px; font-size: 11px;"
        )
        self._qr_preview.setScaledContents(False)
        qr_layout.addWidget(self._qr_preview)

        layout.addWidget(qr_group)

        # ── Save ──────────────────────────────────────────────────────
        save_group = QGroupBox("Save")
        save_layout = QVBoxLayout(save_group)
        save_layout.addWidget(QLabel("Filename (from QR/label):"))
        self._file_name_edit = QLineEdit()
        self._file_name_edit.setPlaceholderText("e.g. COIN_001")
        save_layout.addWidget(self._file_name_edit)

        # Auto-scan suffix info
        suffix_lbl = QLabel("Auto: QR found → value-2  |  No QR → filename unchanged")
        suffix_lbl.setStyleSheet("color:#888; font-size:10px;")
        suffix_lbl.setWordWrap(True)
        save_layout.addWidget(suffix_lbl)

        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel("Format:"))
        self._fmt_combo = QComboBox()
        self._fmt_combo.addItems(["JPG"])
        self._fmt_combo.setEnabled(False)
        self._fmt_combo.setToolTip("Only JPG format is supported")
        fmt_row.addWidget(self._fmt_combo)
        save_layout.addLayout(fmt_row)

        self._btn_save = QPushButton("💾  Save Image")
        self._btn_save.clicked.connect(self._save_image)
        save_layout.addWidget(self._btn_save)
        layout.addWidget(save_group)

        layout.addStretch()

        # ── Wrap in scroll area ────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setMinimumWidth(290)
        scroll.setMaximumWidth(340)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("""
            QScrollArea { background: #f2f4f8; border: none; }
            QScrollBar:vertical {
                background: #dde3ec;
                width: 7px;
                border-radius: 3px;
            }
            QScrollBar::handle:vertical {
                background: #5a7aaa;
                border-radius: 3px;
                min-height: 24px;
            }
            QScrollBar::handle:vertical:hover {
                background: #3e6188;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical { height: 0px; }
        """)
        return scroll

    def _build_status_bar(self):
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready — open a coin image to begin.")

    def _connect_signals(self):
        self._canvas.rotate_cw_requested.connect(self._rotate_cw_90)
        self._canvas.rotate_ccw_requested.connect(self._rotate_ccw_90)
        self._canvas.crop_confirmed.connect(self._apply_crop)
        self._canvas.crop_move.connect(self._on_crop_move)
        self._canvas.crop_resize.connect(self._on_crop_resize)
        self._canvas.pen_closed.connect(self._apply_pen_crop)
        self._canvas.patch_apply.connect(self._apply_patch)
        self._canvas.patch_cancelled.connect(
            lambda: self._btn_patch.setChecked(False))
        self._canvas.patch_phase_changed.connect(self._on_patch_phase_changed)
        self._canvas.pen_edit_region.connect(self._edit_text_in_region)
        self._canvas.pen_scan_qr_region.connect(self._scan_qr_in_region)

    # ------------------------------------------------------------------ #
    #  Actions                                                             #
    # ------------------------------------------------------------------ #

    def _open_image(self):
        path = open_image_dialog(self)
        if not path:
            return

        try:
            validate_format(path)
        except InvalidFormatError as e:
            QMessageBox.warning(self, "Invalid Format", str(e))
            return

        self._load_path(path)

        # Load folder thumbnails and highlight current file
        folder = os.path.dirname(path)
        self._thumb_panel.load_folder(folder)
        # Highlight after small delay so thumbnails load first
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(200, lambda: self._thumb_panel.highlight_path(path))

    def _open_folder(self):
        """Open a folder and show all JPG images in the thumbnail strip."""
        from PyQt5.QtWidgets import QFileDialog
        # DontUseNativeDialog prevents crash on Wayland/Linux
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Folder with Coin Images",
            "",
            QFileDialog.DontUseNativeDialog
        )
        if not folder:
            return
        self._thumb_panel.load_folder(folder)
        if self._thumb_panel._paths:
            self._thumb_panel.select_index(0)
        self._status.showMessage(
            f"Folder loaded: {os.path.basename(folder)}  "
            f"({len(self._thumb_panel._paths)} JPG images)")

    def _load_from_thumbnail(self, path: str):
        """Load image selected from thumbnail strip."""
        try:
            validate_format(path)
        except InvalidFormatError:
            return
        self._load_path(path)

    def _load_path(self, path: str):
        """Core image loading — shared by open_image and thumbnail click."""
        self._current_path = path   # remember for delete
        try:
            self._processor.load(path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot open image:\n{e}")
            return

        self._canvas.set_image(self._processor.current)
        self._canvas.set_crop_overlay(None, {})
        self._canvas.set_qr_regions([])
        self._canvas.stop_pen_tool()
        self._lbl_qr_result.setText("No code detected yet.")
        self._lbl_qr_result.setStyleSheet("color: #555; font-size: 11px;")
        self._qr_value_edit.setText("")
        self._qr_preview.setPixmap(QPixmap())
        self._qr_preview.setText("QR preview")
        self._spin_angle.blockSignals(True)
        self._spin_angle.setValue(0)
        self._spin_angle.blockSignals(False)

        base_name = os.path.splitext(os.path.basename(path))[0]

        base_name = os.path.splitext(os.path.basename(path))[0]
        self._file_name_edit.setText(base_name)

        self._update_ui_state()
        self._update_crop_preview()
        self._btn_detect_qr.setEnabled(True)
        self._btn_detect_qr.setText("⊞  Scan QR / Label")
        self._status.showMessage(
            f"Loaded: {os.path.basename(path)}  "
            f"({self._processor.get_size()[0]} × {self._processor.get_size()[1]} px)"
            f"  — Click 'Detect QR / Label' to scan."
        )

    @staticmethod
    def _clean_qr_value(raw: str) -> str:
        """Extract filename-safe value from QR data."""
        import re
        if not raw:
            return ""
        # Strip URL prefix
        val = re.sub(r'^https?://[^/]+/', '', raw.strip())
        val = re.sub(r'^https?://', '', val)
        # Remove unsafe filename chars
        val = re.sub(r'[\\/:*?"<>|]', '_', val)
        val = val.strip().strip('_')
        # If it has numbers, prefer longest number sequence
        nums = re.findall(r'\d+', val)
        if nums:
            longest = max(nums, key=len)
            if len(longest) >= 4:   # meaningful number (e.g. cert no.)
                return longest
        # Return cleaned full value (max 40 chars)
        return val[:40] if val else "coin"

    def _save_image(self):
        if not self._processor.is_loaded:
            return
        name = self._file_name_edit.text().strip() or "coin"
        path = save_image_dialog(self, suggested_name=name + ".jpg")
        if not path:
            return
        path = ensure_extension(path, ".jpg")
        try:
            save_image(self._processor.current, path)
            self._processor.save_checkpoint()   # reset will return here
            self._status.showMessage(f"Saved: {path}  (JPG)")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def _undo_step(self):
        """Undo — patch/pen mode cancel, else undo 1 image step."""
        if not self._processor.is_loaded:
            return

        # ── Patch mode: undo = reset patch selection ───────────────────
        if self._canvas.is_patch_mode():
            self._canvas.stop_patch_tool()
            self._btn_patch.setChecked(False)
            self._status.showMessage("⬡ Patch Tool cancelled.")
            return

        # ── Pen mode: undo = remove last anchor point ──────────────────
        if self._canvas.is_pen_mode():
            pts = self._canvas._pen_points
            if pts:
                pts.pop()
                self._canvas._render_overlay()
                self._status.showMessage(
                    f"✏ Pen undo — {len(pts)} point(s) remaining.")
            else:
                self._status.showMessage("✏ No more pen points to undo.")
            return

        # ── Normal: undo 1 image step ─────────────────────────────────
        result = self._processor.undo_step()
        if result is None:
            self._status.showMessage("Nothing to undo — already at first step.")
            return

        self._canvas.set_image(self._processor.current)
        self._canvas.set_crop_overlay(None, {})
        self._canvas.stop_pen_tool()
        self._spin_angle.blockSignals(True)
        self._spin_angle.setValue(int(self._processor.angle))
        self._spin_angle.blockSignals(False)
        self._btn_circle.setChecked(False)
        self._btn_rect.setChecked(False)
        self._set_detection_buttons_enabled(True)
        self._update_crop_preview()
        self._status.showMessage("↩ Undone 1 step.")

    def _reset_image(self):
        """Reset All — always go back to original image."""
        if not self._processor.is_loaded:
            return

        # Stop pen + patch tools
        self._canvas.stop_pen_tool()
        self._canvas.stop_patch_tool()
        self._btn_patch.setChecked(False)

        self._processor.reset_to_original()

        self._canvas.set_image(self._processor.current)
        self._canvas.fit_view()

        self._spin_angle.blockSignals(True)
        self._spin_angle.setValue(0)
        self._spin_angle.blockSignals(False)

        self._lbl_qr_result.setText("No code detected yet.")
        self._lbl_qr_result.setStyleSheet("color: #555; font-size: 11px;")
        self._qr_preview.setPixmap(QPixmap())
        self._qr_preview.setText("QR preview")
        self._file_name_edit.clear()

        self._btn_circle.setChecked(False)
        self._btn_rect.setChecked(False)
        self._set_detection_buttons_enabled(True)
        self._update_crop_preview()
        self._status.showMessage("↺ Reset All — back to original image.")

    def _delete_image(self):
        """Delete the currently opened image file from disk."""
        # 1. Check image loaded
        if not self._processor.is_loaded:
            QMessageBox.information(self, "Delete", "No image is open.")
            return

        # 2. Get file path
        current_path = getattr(self, "_current_path", "")
        if not current_path and hasattr(self, "_thumb_panel"):
            current_path = self._thumb_panel.current_path()
        current_path = str(current_path).strip()

        if not current_path or not os.path.isfile(current_path):
            QMessageBox.warning(self, "Delete",
                "Cannot find the file path.\n"
                "Open image via Open or Folder first.")
            return

        fname = os.path.basename(current_path)

        # 3. Confirm — custom styled dialog
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton
        from PyQt5.QtCore import Qt as _Qt

        dlg = QDialog(self)
        dlg.setWindowTitle("Confirm Delete")
        dlg.setFixedSize(420, 200)
        dlg.setStyleSheet("QDialog { background:#f2f4f8; }")

        vlay = QVBoxLayout(dlg)
        vlay.setContentsMargins(0, 0, 0, 0)
        vlay.setSpacing(0)

        # Header
        hdr = QLabel("  🗑  Confirm Delete")
        hdr.setFixedHeight(48)
        hdr.setStyleSheet(
            "background:#e57373; color:white; font-size:15px;"
            "font-weight:bold; padding-left:12px;")
        vlay.addWidget(hdr)

        # Message
        msg = QLabel(
            f"<br><b>Permanently delete:</b><br><br>"
            f"&nbsp;&nbsp;&nbsp;{fname}<br><br>"
            f"<span style='color:#c62828;'>This cannot be undone!</span>")
        msg.setAlignment(_Qt.AlignLeft)
        msg.setStyleSheet(
            "background:#f2f4f8; color:#1c2333; font-size:13px;"
            "padding: 16px 24px;")
        msg.setWordWrap(True)
        vlay.addWidget(msg, 1)

        # Buttons row
        btn_row = QWidget()
        btn_row.setStyleSheet("background:#e8ecf4;")
        btn_row.setFixedHeight(58)
        blay = QHBoxLayout(btn_row)
        blay.setContentsMargins(20, 10, 20, 10)
        blay.setSpacing(16)

        btn_no = QPushButton("✕  No, Cancel")
        btn_no.setFixedHeight(36)
        btn_no.setStyleSheet("""
            QPushButton {
                background:#eef1f6; color:#c62828;
                border:2px solid #ef9a9a; border-radius:6px;
                font-size:13px; font-weight:bold; padding:0 20px;
            }
            QPushButton:hover {
                background:#ffd0d0; border:2px solid #c62828;
            }
            QPushButton:pressed { background:#c62828; color:white; }
        """)
        btn_no.clicked.connect(dlg.reject)

        btn_yes = QPushButton("✔  Yes, Delete")
        btn_yes.setFixedHeight(36)
        btn_yes.setStyleSheet("""
            QPushButton {
                background:#eef1f6; color:#1b5e20;
                border:2px solid #66bb6a; border-radius:6px;
                font-size:13px; font-weight:bold; padding:0 20px;
            }
            QPushButton:hover {
                background:#c8f0d0; border:2px solid #2e7d32;
            }
            QPushButton:pressed { background:#2e7d32; color:white; }
        """)
        btn_yes.clicked.connect(dlg.accept)

        blay.addWidget(btn_no)
        blay.addWidget(btn_yes)
        vlay.addWidget(btn_row)

        if dlg.exec_() != QDialog.Accepted:
            return

        # 4. Find next image before deleting
        next_path = ""
        if hasattr(self, "_thumb_panel"):
            paths   = list(self._thumb_panel._paths)
            sel     = self._thumb_panel._sel
            if paths and sel >= 0:
                remaining = [p for p in paths if p != current_path]
                if remaining:
                    next_path = remaining[min(sel, len(remaining) - 1)]

        # 5. Delete file
        try:
            os.remove(current_path)
        except Exception as e:
            QMessageBox.critical(self, "Delete Failed", str(e))
            return

        self._current_path = ""
        self._status.showMessage(f"Deleted: {fname}")

        # 6. Clear canvas & processor
        self._processor.original       = None
        self._processor.current        = None
        self._processor._load_original = None
        self._processor.crop_type      = None
        self._processor.crop_params    = {}
        self._canvas._image            = None
        self._canvas._crop_type        = None
        self._canvas._crop_params      = {}
        self._canvas._qr_regions       = []
        self._canvas._invalidate_cache()
        self._canvas._render_overlay()
        self._file_name_edit.clear()
        self._lbl_qr_result.setText("No code detected yet.")
        self._lbl_qr_result.setStyleSheet("color: #555; font-size: 11px;")
        self._qr_preview.setPixmap(QPixmap())
        self._qr_preview.setText("QR preview")
        self._spin_angle.blockSignals(True)
        self._spin_angle.setValue(0)
        self._spin_angle.blockSignals(False)
        self._update_ui_state()
        if hasattr(self, "_crop_preview"):
            self._crop_preview.clear()

        # 7. Reload strip & load next image
        if hasattr(self, "_thumb_panel"):
            folder = os.path.dirname(current_path)
            self._thumb_panel.load_folder(folder)

        if next_path and os.path.isfile(next_path):
            self._do_load_next(next_path)

    def _do_load_next(self, path: str):
        """Load next image after delete (called with small delay)."""
        from PyQt5.QtCore import QTimer
        def _load():
            self._load_path(path)
            if hasattr(self, "_thumb_panel"):
                self._thumb_panel.highlight_path(path)
        QTimer.singleShot(350, _load)

    def _rotate_pen_points(self, angle_deg: float, img_w: int, img_h: int):
        """Transform pen points to match image after rotation.
        Uses exact pixel mapping — not matrix approximation."""
        if not self._canvas.is_pen_mode() or not self._canvas._pen_points:
            return

        def transform_pt(x, y):
            if angle_deg == 90.0:
                # CW 90°: (x,y) → (img_h-1-y, x)
                return (float(img_h - 1 - y), float(x))
            elif angle_deg == -90.0:
                # CCW 90°: (x,y) → (y, img_w-1-x)
                return (float(y), float(img_w - 1 - x))
            elif abs(angle_deg) == 180.0:
                return (float(img_w - 1 - x), float(img_h - 1 - y))
            else:
                # Fine rotation (+1/-1): rotate around image center
                import math
                cx = img_w / 2.0; cy = img_h / 2.0
                rad   = math.radians(-angle_deg)
                cos_a = math.cos(rad); sin_a = math.sin(rad)
                dx = x - cx; dy = y - cy
                return (dx*cos_a - dy*sin_a + cx, dx*sin_a + dy*cos_a + cy)

        def transform_vec(dx, dy):
            import math
            rad   = math.radians(-angle_deg)
            cos_a = math.cos(rad); sin_a = math.sin(rad)
            return (dx*cos_a - dy*sin_a, dx*sin_a + dy*cos_a)

        for pt_data in self._canvas._pen_points:
            pt_data["pt"]     = transform_pt(*pt_data["pt"])
            pt_data["cp_in"]  = transform_vec(*pt_data["cp_in"])
            pt_data["cp_out"] = transform_vec(*pt_data["cp_out"])

        # Update circle reference
        if self._canvas._pen_circle:
            cp  = self._canvas._pen_circle
            ncx, ncy = transform_pt(cp["x"], cp["y"])
            self._canvas._pen_circle = {**cp, "x": ncx, "y": ncy}

        # Update rect reference
        if self._canvas._pen_rect:
            rp  = self._canvas._pen_rect
            rcx = rp["x"] + rp["width"]  / 2
            rcy = rp["y"] + rp["height"] / 2
            ncx, ncy = transform_pt(rcx, rcy)
            if abs(angle_deg) == 90.0:
                self._canvas._pen_rect = {
                    **rp,
                    "x":      ncx - rp["height"] / 2,
                    "y":      ncy - rp["width"]  / 2,
                    "width":  rp["height"],
                    "height": rp["width"],
                }
            else:
                self._canvas._pen_rect = {
                    **rp,
                    "x": ncx - rp["width"]  / 2,
                    "y": ncy - rp["height"] / 2,
                }

    def _rotate_crop_params(self, angle_deg: float, img_w: int, img_h: int):
        """Rotate crop overlay params to match image after rotation."""
        crop_type = self._canvas._crop_type or self._processor.crop_type
        cp = (self._canvas._crop_params or self._processor.crop_params or {}).copy()
        if not crop_type or not cp:
            return

        import math
        cx_img = img_w / 2.0
        cy_img = img_h / 2.0

        if crop_type == "circle":
            ox, oy = cp["x"], cp["y"]
            if angle_deg == 90.0:
                nx, ny = img_h - 1 - oy, ox
            elif angle_deg == -90.0:
                nx, ny = oy, img_w - 1 - ox
            elif abs(angle_deg) == 180.0:
                nx, ny = img_w - 1 - ox, img_h - 1 - oy
            else:
                rad = math.radians(-angle_deg)
                dx = ox - cx_img; dy = oy - cy_img
                nx = dx*math.cos(rad) - dy*math.sin(rad) + cx_img
                ny = dx*math.sin(rad) + dy*math.cos(rad) + cy_img
            cp["x"] = nx; cp["y"] = ny
            # radius unchanged

        elif crop_type == "rectangle":
            rcx = cp["x"] + cp["width"]  / 2
            rcy = cp["y"] + cp["height"] / 2
            if angle_deg == 90.0:
                nx, ny = img_h - 1 - rcy, rcx
                cp["width"], cp["height"] = cp["height"], cp["width"]
            elif angle_deg == -90.0:
                nx, ny = rcy, img_w - 1 - rcx
                cp["width"], cp["height"] = cp["height"], cp["width"]
            elif abs(angle_deg) == 180.0:
                nx, ny = img_w - 1 - rcx, img_h - 1 - rcy
            else:
                rad = math.radians(-angle_deg)
                dx = rcx - cx_img; dy = rcy - cy_img
                nx = dx*math.cos(rad) - dy*math.sin(rad) + cx_img
                ny = dx*math.sin(rad) + dy*math.cos(rad) + cy_img
            cp["x"] = nx - cp["width"]  / 2
            cp["y"] = ny - cp["height"] / 2

        # Convert all values to int and apply
        for k in ("x","y","width","height","radius"):
            if k in cp:
                cp[k] = int(round(cp[k]))
        self._canvas._crop_type   = crop_type
        self._canvas._crop_params = cp
        self._processor.crop_type   = crop_type
        self._processor.crop_params = cp

    def _sync_crop_to_processor(self):
        """Sync canvas crop params → processor before rotation.
        Pen mode: use pen circle/rect as the crop region."""
        if self._canvas.is_pen_mode():
            # Pen + circle mode → compute center/radius from actual points
            if self._canvas._pen_circle and self._canvas._pen_points:
                import math as _mm
                _pts = self._canvas._pen_points
                _n   = len(_pts)
                _cx  = sum(pt["pt"][0] for pt in _pts) / _n
                _cy  = sum(pt["pt"][1] for pt in _pts) / _n
                _r   = sum(_mm.hypot(pt["pt"][0]-_cx, pt["pt"][1]-_cy)
                           for pt in _pts) / _n
                self._processor.crop_type   = "circle"
                self._processor.crop_params = {
                    "x":      int(round(_cx)),
                    "y":      int(round(_cy)),
                    "radius": int(round(_r)),
                }
            # Pen + rect mode → compute from actual points
            elif self._canvas._pen_rect and self._canvas._pen_points:
                _pts = self._canvas._pen_points
                _xs  = [p["pt"][0] for p in _pts]
                _ys  = [p["pt"][1] for p in _pts]
                self._processor.crop_type   = "rectangle"
                self._processor.crop_params = {
                    "x":      int(round(min(_xs))),
                    "y":      int(round(min(_ys))),
                    "width":  int(round(max(_xs)-min(_xs))),
                    "height": int(round(max(_ys)-min(_ys))),
                }
            else:
                # Free pen — no crop region, rotate whole image
                self._processor.crop_type   = None
                self._processor.crop_params = {}
        elif self._canvas._crop_type and self._canvas._crop_params:
            self._processor.crop_type   = self._canvas._crop_type
            self._processor.crop_params = self._canvas._crop_params.copy()

    def _rotate_cw(self):
        """Right panel: fine +1° rotation — rotates image inside crop only."""
        if not self._processor.is_loaded:
            return
        self._sync_crop_to_processor()
        if not (self._canvas.is_pen_mode() and self._canvas._pen_circle):
            self._rotate_pen_points(1.0, *self._processor.current.shape[1::-1])
        self._processor.rotate_cw()
        self._refresh_canvas()
        self._sync_spin()

    def _rotate_ccw(self):
        """Right panel: fine -1° rotation — rotates image inside crop only."""
        if not self._processor.is_loaded:
            return
        self._sync_crop_to_processor()
        if not (self._canvas.is_pen_mode() and self._canvas._pen_circle):
            self._rotate_pen_points(-1.0, *self._processor.current.shape[1::-1])
        self._processor.rotate_ccw()
        self._refresh_canvas()
        self._sync_spin()

    def _rotate_cw_90(self):
        """Toolbar: coarse +90° rotation — rotates image inside crop only."""
        if not self._processor.is_loaded:
            return
        self._sync_crop_to_processor()
        if not (self._canvas.is_pen_mode() and self._canvas._pen_circle):
            self._rotate_pen_points(90.0, *self._processor.current.shape[1::-1])
        self._processor.rotate_cw_90()
        self._refresh_canvas()
        self._sync_spin()

    def _rotate_ccw_90(self):
        """Toolbar: coarse -90° rotation — rotates image inside crop only."""
        if not self._processor.is_loaded:
            return
        self._sync_crop_to_processor()
        if not (self._canvas.is_pen_mode() and self._canvas._pen_circle):
            self._rotate_pen_points(-90.0, *self._processor.current.shape[1::-1])
        self._processor.rotate_ccw_90()
        self._refresh_canvas()
        self._sync_spin()

    def _on_spin_angle(self, value: int):
        if not self._processor.is_loaded:
            return
        self._processor.rotate_to(float(value))
        self._refresh_canvas()

    def _auto_circle(self):
        if not self._processor.is_loaded:
            return
        if self._canvas.is_pen_mode():
            self._canvas.stop_pen_tool()
        self._start_worker("circle")

    def _auto_rect(self):
        if not self._processor.is_loaded:
            return
        if self._canvas.is_pen_mode():
            self._canvas.stop_pen_tool()
        self._start_worker("rectangle")

    def _toggle_pen_tool(self):
        """Pen Tool — free drawing OR auto-points from detected shape."""
        if not self._processor.is_loaded:
            self._btn_pen.setChecked(False)
            return

        if self._canvas.is_pen_mode():
            self._canvas.stop_pen_tool()
            self._btn_pen.setChecked(False)
            self._status.showMessage("✏ Pen Tool deactivated.")
            return

        # Stop patch if active
        if self._canvas.is_patch_mode():
            self._canvas.stop_patch_tool()
            self._btn_patch.setChecked(False)

        import math
        crop_type = self._canvas._crop_type or self._processor.crop_type
        cp        = self._canvas._crop_params or self._processor.crop_params

        self._btn_pen.setChecked(True)
        self._btn_circle.setChecked(False)
        self._btn_rect.setChecked(False)

        if crop_type == "circle" and cp:
            # ── Mode 2: Circle → 16 points exact on detected circle ───
            import math as _math
            cx = float(cp["x"]); cy = float(cp["y"]); r = float(cp["radius"])
            # Bezier handle for smooth 16-point circle
            h = r * (4/3) * _math.tan(_math.pi / 16)
            pen_pts = []
            for i in range(16):
                angle = _math.radians(i * 22.5 - 90)  # start top, clockwise
                pt_x  = cx + r * _math.cos(angle)
                pt_y  = cy + r * _math.sin(angle)
                tx    = -_math.sin(angle)  # tangent x
                ty    =  _math.cos(angle)  # tangent y
                pen_pts.append({
                    "pt":     (pt_x, pt_y),
                    "cp_out": ( tx * h,  ty * h),
                    "cp_in":  (-tx * h, -ty * h),
                })
            # pen_circle must exactly match detected circle
            pen_circle = {"x": cx, "y": cy, "radius": r}
            self._canvas.start_pen_with_shapes(None, pen_circle)
            self._canvas._pen_points = pen_pts
            self._canvas._pen_closed = True
            self._canvas._render_overlay()
            self._status.showMessage(
                "✏ Pen Tool (Circle) — "
                "Scroll = resize  |  Drag □ = refine  |  "
                "Enter = Apply  |  Esc = Cancel")

        elif crop_type == "rectangle" and cp:
            # ── Mode 2: Rectangle → 8 auto points on edges ────────────
            x, y  = cp["x"], cp["y"]
            w, h  = cp["width"], cp["height"]
            cx2   = x + w/2; cy2 = y + h/2
            pts = [
                (x,    y),    (cx2,  y),    (x+w,  y),
                (x+w,  cy2),  (x+w,  y+h),  (cx2,  y+h),
                (x,    y+h),  (x,    cy2),
            ]
            self._canvas.start_pen_with_shapes(cp.copy(), None)
            self._canvas._pen_points = [
                {"pt": pt, "cp_in": (0.0, 0.0), "cp_out": (0.0, 0.0)}
                for pt in pts
            ]
            self._canvas._pen_closed = True
            self._canvas._render_overlay()
            self._status.showMessage(
                "✏ Pen Tool (Rect) — 8 points placed. "
                "Drag □ handles to refine  |  Enter = Apply  |  Esc = Cancel")

        else:
            # ── Mode 1: Free pen — click to place points ───────────────
            self._canvas.start_pen_tool()
            self._canvas._pen_closed = False
            self._canvas._render_overlay()
            self._status.showMessage(
                "✏ Pen Tool (Free) — Click = corner point  |  "
                "Click+Drag = curve  |  Green ● = close path  |  "
                "Enter = Apply  |  Right-click = menu  |  Esc = Cancel")

    def _toggle_patch_tool(self):
        """Activate / deactivate Patch Tool."""
        if not self._processor.is_loaded:
            self._btn_patch.setChecked(False)
            return
        if self._canvas.is_patch_mode():
            self._canvas.stop_patch_tool()
            self._btn_patch.setChecked(False)
            self._btn_apply_patch.setVisible(False)
            self._status.showMessage("Patch Tool deactivated.")
        else:
            if self._canvas.is_pen_mode():
                self._canvas.stop_pen_tool()
            self._canvas.start_patch_tool()
            self._btn_patch.setChecked(True)
            self._btn_apply_patch.setVisible(False)
            self._status.showMessage(
                "⬡ Patch Tool — Step 1: Draw around damaged area  "
                "Step 2: Drag to good area  Step 3: Click Apply Patch")

    def _update_qr_code(self):
        """Replace QR code in image with a new QR image file."""
        if not self._processor.is_loaded:
            return
        region = getattr(self, '_qr_last_region', None)
        if not region:
            self._status.showMessage("⊞ Scan QR first to detect position.")
            return

        # Open file dialog to select new QR image
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Select New QR Code Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp)",
            options=QFileDialog.DontUseNativeDialog)
        if not path:
            return

        try:
            import cv2, numpy as np

            # Load new QR image
            new_qr = cv2.imread(path)
            if new_qr is None:
                self._status.showMessage("⊞ Cannot load QR image.")
                return

            # Get original QR bounding box
            bx, by, bw, bh = region["bbox"]
            img = self._processor.current.copy()
            h, w = img.shape[:2]

            # Clamp bbox
            bx = max(0, bx); by = max(0, by)
            bw = min(bw, w - bx); bh = min(bh, h - by)
            if bw <= 0 or bh <= 0:
                self._status.showMessage("⊞ QR region invalid.")
                return

            # Resize new QR to fit original QR size
            new_qr_resized = cv2.resize(new_qr, (bw, bh),
                                         interpolation=cv2.INTER_AREA)

            # Replace in image
            img[by:by+bh, bx:bx+bw] = new_qr_resized

            # Save to history
            self._processor._push_history()
            self._processor.current  = img
            self._processor.original = img.copy()
            self._canvas.set_image(img)
            self._canvas.set_qr_regions([])
            self._btn_update_qr.setEnabled(False)
            self._qr_last_region = None
            self._lbl_qr_result.setText("✓ QR code updated successfully.")
            self._lbl_qr_result.setStyleSheet(
                "color:#1a7a1a;font-size:11px;font-weight:bold;")
            self._status.showMessage(
                f"⊞ QR updated at position ({bx},{by}) size {bw}×{bh}px")

        except Exception as e:
            self._status.showMessage(f"⊞ QR update error: {e}")

    def _get_pen_roi(self):
        """Get ROI image from current pen selection (bounding box)."""
        if not self._processor.is_loaded or not self._canvas._pen_points:
            return None, None
        pts  = self._canvas.get_pen_points()
        xs   = [p[0] for p in pts]; ys = [p[1] for p in pts]
        x1   = max(0, int(min(xs))); y1 = max(0, int(min(ys)))
        x2   = int(max(xs)); y2 = int(max(ys))
        img  = self._processor.current
        roi  = img[y1:y2, x1:x2].copy()
        return roi, (x1, y1, x2, y2)

    def _edit_text_in_region(self):
        """OCR text in selected region → let user edit → render back."""
        roi, bbox = self._get_pen_roi()
        if roi is None or roi.size == 0:
            self._status.showMessage("✏ Select a region first.")
            return
        x1, y1, x2, y2 = bbox

        # OCR current text
        current_text = ""
        try:
            import pytesseract, cv2
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thr = cv2.threshold(gray, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            current_text = pytesseract.image_to_string(
                thr, config="--psm 6").strip()
        except Exception:
            pass

        # Show edit dialog
        from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout,
                                      QLabel, QTextEdit, QPushButton,
                                      QFontComboBox, QSpinBox, QColorDialog)
        from PyQt5.QtGui import QColor, QFont
        from PyQt5.QtCore import Qt as _Qt

        dlg = QDialog(self)
        dlg.setWindowTitle("Edit Text in Region")
        dlg.setFixedSize(480, 380)
        dlg.setStyleSheet("QDialog{background:#f2f4f8;}")
        vlay = QVBoxLayout(dlg)
        vlay.setContentsMargins(0,0,0,0)

        # Header
        hdr = QLabel("  ✏  Edit Text in Region")
        hdr.setFixedHeight(44)
        hdr.setStyleSheet("background:#1c2b45;color:#e8edf5;font-size:14px;"
                           "font-weight:bold;padding-left:12px;")
        vlay.addWidget(hdr)

        body = QVBoxLayout()
        body.setContentsMargins(16,12,16,12); body.setSpacing(8)

        # Font controls
        font_row = QHBoxLayout()
        font_row.addWidget(QLabel("Font:"))
        font_cb = QFontComboBox()
        font_cb.setCurrentFont(QFont("Arial"))
        font_row.addWidget(font_cb, 2)
        font_row.addWidget(QLabel("Size:"))
        size_sp = QSpinBox()
        size_sp.setRange(8, 200); size_sp.setValue(32)
        font_row.addWidget(size_sp)

        # Color button
        self._edit_color = QColor(0,0,0)
        btn_color = QPushButton("■ Color")
        btn_color.setStyleSheet(f"background:{self._edit_color.name()};color:white;"
                                 "border-radius:4px;padding:3px 8px;font-weight:bold;")
        def pick_color():
            c = QColorDialog.getColor(self._edit_color, dlg)
            if c.isValid():
                self._edit_color = c
                btn_color.setStyleSheet(f"background:{c.name()};color:white;"
                                         "border-radius:4px;padding:3px 8px;font-weight:bold;")
        btn_color.clicked.connect(pick_color)
        font_row.addWidget(btn_color)
        body.addLayout(font_row)

        # Text edit
        body.addWidget(QLabel("Text (edit below):"))
        txt = QTextEdit()
        txt.setPlainText(current_text)
        txt.setStyleSheet("background:#fff;border:1px solid #c0cce0;border-radius:4px;")
        body.addWidget(txt, 1)

        # Buttons
        btn_row2 = QHBoxLayout()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(dlg.reject)
        btn_apply2  = QPushButton("✔  Apply")
        btn_apply2.setStyleSheet("background:#3e6188;color:white;font-weight:bold;"
                                 "border-radius:5px;padding:6px 16px;")
        btn_apply2.clicked.connect(dlg.accept)
        btn_row2.addWidget(btn_cancel); btn_row2.addStretch(); btn_row2.addWidget(btn_apply2)
        body.addLayout(btn_row2)

        w = QDialog()
        w_inner = QVBoxLayout.__new__(QVBoxLayout)
        inner_w = QDialog.__new__(QDialog)
        from PyQt5.QtWidgets import QWidget as _QW
        inner = _QW(dlg)
        inner.setLayout(body)
        vlay.addWidget(inner, 1)

        if dlg.exec_() != QDialog.Accepted:
            return

        new_text = txt.toPlainText().strip()
        if not new_text:
            return

        # Render text onto image region
        try:
            import cv2, numpy as np
            from PIL import Image, ImageDraw, ImageFont

            img = self._processor.current.copy()
            rw  = x2 - x1; rh = y2 - y1

            # White background on region
            img[y1:y2, x1:x2] = 255

            # Render text with PIL for better font support
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw    = ImageDraw.Draw(pil_img)
            color   = (self._edit_color.red(),
                        self._edit_color.green(),
                        self._edit_color.blue())
            font_name = font_cb.currentFont().family()
            font_size = size_sp.value()
            try:
                pil_font = ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                                               font_size)
            except Exception:
                pil_font = ImageFont.load_default()

            draw.text((x1+4, y1+4), new_text, font=pil_font, fill=color)
            result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            self._processor._push_history()
            self._processor.current  = result
            self._processor.original = result.copy()
            self._canvas.set_image(result)
            self._status.showMessage(f"✏ Text updated in region.")
        except Exception as e:
            self._status.showMessage(f"✏ Text render error: {e}")

    def _scan_qr_in_region(self):
        """Scan QR/barcode in selected pen region."""
        roi, bbox = self._get_pen_roi()
        if roi is None or roi.size == 0:
            self._status.showMessage("⊞ Select a region first.")
            return
        try:
            from processing.qr_detector import QRDetector
            det     = QRDetector()
            results = det.detect(roi)
            if not results:
                self._status.showMessage("⊞ No QR/barcode found in region.")
                return
            import re as _re
            texts = [r["data"] for r in results if r["data"]]
            if texts:
                clean  = self._clean_qr_value(texts[0])
                types  = [r.get("type","qr") for r in results]
                suffix = "-2" if any(t=="qr" for t in types) else "-1"
                clean  = _re.sub(r'-[12]$', '', clean)
                self._file_name_edit.setText(f"{clean}{suffix}")
                self._lbl_qr_result.setText(f"✓ [{types[0]}] {texts[0][:60]}")
                self._lbl_qr_result.setStyleSheet("color:#1a7a1a;font-size:11px;font-weight:bold;")
                self._qr_value_edit.setText(texts[0])
                self._status.showMessage(
                    f"⊞ QR detected in region: {texts[0][:60]} → {clean}{suffix}")
            else:
                self._status.showMessage("⊞ Region detected but no data decoded.")
        except Exception as e:
            self._status.showMessage(f"⊞ Scan error: {e}")

    def _on_patch_phase_changed(self):
        """Called when patch phase changes to draw/drag."""
        if self._canvas._patch_phase == "drag":
            self._btn_apply_patch.setVisible(True)
            self._status.showMessage(
                "⬡ Patch Tool — Drag selection to good texture area, "
                "then click Apply Patch (or press Enter)")
        else:
            self._btn_apply_patch.setVisible(False)

    def _apply_patch(self):
        """Apply patch from source to destination area."""
        if not self._canvas.is_patch_mode():
            return
        if self._canvas._patch_phase != "drag":
            self._status.showMessage(
                "⬡ Draw selection first, then drag to good area, then Enter.")
            return
        try:
            result = self._canvas.apply_patch(self._processor.current)
            if result is not None:
                self._processor._push_history()
                self._processor.current  = result
                self._processor.original = result.copy()
                self._canvas.set_image(result)
                self._status.showMessage("⬡ Patch applied successfully.")
            else:
                self._status.showMessage("⬡ Patch failed — try a larger selection.")
        except Exception as e:
            self._status.showMessage(f"⬡ Patch error: {e}")
        self._canvas.stop_patch_tool()
        self._btn_patch.setChecked(False)
        self._btn_apply_patch.setVisible(False)

    def _apply_pen_crop(self):
        """Apply pen crop — ellipse for circle mode, rect for rect mode, polygon for free."""
        if not self._processor.is_loaded:
            self._status.showMessage("No image loaded.")
            return

        pts = self._canvas.get_pen_points()
        if len(pts) < 3:
            self._status.showMessage("✏ Need at least 3 anchor points to crop.")
            return

        try:
            is_circle_mode = bool(self._canvas._pen_circle)
            is_rect_mode   = bool(self._canvas._pen_rect)

            if is_circle_mode:
                # ── Ellipse crop from bounding box of anchor points ────
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                cx = int(round((min(xs) + max(xs)) / 2))
                cy = int(round((min(ys) + max(ys)) / 2))
                rx = (max(xs) - min(xs)) / 2
                ry = (max(ys) - min(ys)) / 2
                r  = int(round((rx + ry) / 2))
                self._processor.set_circle_crop(cx, cy, r)
                self._processor.apply_crop()

            elif is_rect_mode:
                # ── Rectangle crop from bounding box ──────────────────
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                x  = int(round(min(xs)))
                y  = int(round(min(ys)))
                w  = int(round(max(xs) - min(xs)))
                h  = int(round(max(ys) - min(ys)))
                self._processor.set_rect_crop(x, y, w, h)
                self._processor.apply_crop()

            else:
                # ── Free pen — polygon crop ────────────────────────────
                pts_int = [(int(round(x)), int(round(y))) for x, y in pts]
                bez_pts = list(self._canvas._pen_points)
                self._processor.set_polygon_crop(pts_int, bez_pts)
                self._processor.apply_crop()

        except Exception as e:
            self._status.showMessage(f"✏ Crop failed: {e}")
            import traceback; traceback.print_exc()
            return

        # Reset everything
        self._canvas.stop_pen_tool()
        self._canvas.set_image(self._processor.current)
        self._spin_angle.blockSignals(True)
        self._spin_angle.setValue(0)
        self._spin_angle.blockSignals(False)
        self._lbl_qr_result.setText("No code detected yet.")
        self._lbl_qr_result.setStyleSheet("color:#555;font-size:11px;")
        self._qr_preview.setPixmap(QPixmap())
        self._qr_preview.setText("QR preview")
        self._set_detection_buttons_enabled(True)
        self._btn_circle.setChecked(False)
        self._btn_rect.setChecked(False)
        self._canvas.fit_view()
        self._update_crop_preview()
        self._status.showMessage(
            f"✏ Pen crop applied — "
            f"{self._processor.get_size()[0]} × {self._processor.get_size()[1]} px"
        )

    def _on_crop_move(self, dx: int, dy: int):
        if not self._processor.is_loaded:
            return
        self._processor.adjust_crop(dx, dy)
        self._canvas._crop_params = self._processor.crop_params.copy()
        self._canvas._render_overlay()
        self._update_crop_preview()

    def _on_crop_resize(self, delta: int):
        if not self._processor.is_loaded:
            return
        self._processor.resize_crop(delta)
        self._canvas._crop_params = self._processor.crop_params.copy()
        self._canvas._render_overlay()
        self._update_crop_preview()

    # ── Label Tools ───────────────────────────────────────────────────────

    def _apply_crop(self):
        if not self._processor.is_loaded:
            return

        # ── Pen tool priority — use polygon crop if pen is active ────
        if self._canvas.is_pen_mode():
            pts = self._canvas.get_pen_points()
            if len(pts) >= 3:
                self._apply_pen_crop()
                return
            else:
                self._status.showMessage(
                    "✏ Pen tool active — need at least 3 points to crop. "
                    "Add more points or press Esc to cancel pen.")
                return

        # ── Normal circle / rectangle crop ───────────────────────────
        if self._canvas._crop_params:
            self._processor.crop_params = self._canvas._crop_params.copy()
            self._processor.crop_type   = self._canvas._crop_type

        if not self._processor.crop_type:
            self._status.showMessage("No crop selected — detect circle or rectangle first.")
            return

        self._processor.apply_crop()
        self._canvas.set_image(self._processor.current)
        self._spin_angle.blockSignals(True)
        self._spin_angle.setValue(0)
        self._spin_angle.blockSignals(False)
        self._lbl_qr_result.setText("No code detected yet.")
        self._lbl_qr_result.setStyleSheet("color: #555; font-size: 11px;")
        from PyQt5.QtGui import QPixmap
        self._qr_preview.setPixmap(QPixmap())
        self._qr_preview.setText("QR preview")
        self._set_detection_buttons_enabled(True)
        self._btn_circle.setChecked(False)
        self._btn_rect.setChecked(False)
        self._update_crop_preview()

        self._status.showMessage(
            f"Crop applied — "
            f"{self._processor.get_size()[0]} × {self._processor.get_size()[1]} px  "
            f"| Ready for next operation"
        )

    def _detect_qr(self):
        if not self._processor.is_loaded:
            return
        self._start_worker("qr")

    def _show_no_qr_dialog(self):
        """Custom styled popup — QR not found in image."""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPushButton
        from PyQt5.QtCore import Qt as _Qt

        dlg = QDialog(self)
        dlg.setWindowTitle("QR / Label Detection")
        dlg.setFixedSize(400, 240)
        dlg.setStyleSheet("QDialog{background:#f2f4f8;} QLabel{background:transparent;}")

        vlay = QVBoxLayout(dlg)
        vlay.setContentsMargins(0, 0, 0, 0)
        vlay.setSpacing(0)

        # Header
        hdr = QLabel("  🔍  QR / Label Detection")
        hdr.setFixedHeight(46)
        hdr.setStyleSheet(
            "background:#1c2b45;color:#e8edf5;font-size:14px;"
            "font-weight:bold;padding-left:12px;")
        vlay.addWidget(hdr)

        # Body
        body = QWidget()
        body.setStyleSheet("background:#f2f4f8;")
        blay = QVBoxLayout(body)
        blay.setContentsMargins(24, 20, 24, 16)
        blay.setSpacing(10)

        icon_lbl = QLabel("❌  No QR code or barcode was found.")
        icon_lbl.setStyleSheet(
            "color:#c62828;font-size:14px;font-weight:bold;")
        blay.addWidget(icon_lbl)

        tips_lbl = QLabel(
            "<br><b>Tips to improve detection:</b><br>"
            "• Make sure the QR code is visible and not blurry<br>"
            "• Try rotating the image (CCW / CW) first<br>"
            "• Crop closer to the QR area and try again")
        tips_lbl.setStyleSheet("color:#444;font-size:12px;")
        tips_lbl.setWordWrap(True)
        blay.addWidget(tips_lbl)

        vlay.addWidget(body, 1)

        # OK button
        btn = QPushButton("  OK")
        btn.setFixedHeight(40)
        btn.setStyleSheet(
            "QPushButton{background:#3e6188;color:white;font-weight:bold;"
            "border:none;font-size:13px;border-radius:0;}"
            "QPushButton:hover{background:#4a7aaa;}"
            "QPushButton:pressed{background:#2a4f70;}")
        btn.clicked.connect(dlg.accept)
        vlay.addWidget(btn)

        dlg.exec_()

    # ------------------------------------------------------------------ #
    #  Background worker                                                   #
    # ------------------------------------------------------------------ #

    def _start_worker(self, mode: str):
        """Launch a DetectionWorker thread; show busy cursor while running."""
        if self._worker and self._worker.isRunning():
            return
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import Qt
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self._set_detection_buttons_enabled(False)
        self._status.showMessage(f"Detecting ({mode})… please wait.")
        self._worker = DetectionWorker(self._processor.current, mode, parent=self)
        self._worker.result_ready.connect(self._on_worker_result)
        self._worker.error.connect(self._on_worker_error)
        self._worker.finished.connect(self._on_worker_done)
        self._worker.start()

    def _on_worker_result(self, det_type: str, params: dict):
        if det_type == "circle":
            self._processor.set_circle_crop(
                params["x"], params["y"], params["radius"]
            )
            self._canvas.set_crop_overlay("circle", self._processor.crop_params)
            self._status.showMessage(
                "⊙ Circle detected — drag handles to adjust. Enter = apply."
            )
        elif det_type == "rectangle":
            self._processor.set_rect_crop(
                params["x"], params["y"], params["width"], params["height"]
            )
            self._canvas.set_crop_overlay("rectangle", self._processor.crop_params)
            self._status.showMessage(
                "▭ Rectangle detected — drag handles to adjust. Enter = apply."
            )
        elif det_type == "qr":
            regions = params.get("regions", [])
            self._canvas.set_qr_regions(regions)

            if not regions:
                # ── Nothing found — filename unchanged ────────────────
                self._lbl_qr_result.setText("No QR code or barcode found.")
                self._lbl_qr_result.setStyleSheet("color:#cc3333;font-size:11px;")
                self._qr_preview.setText("No QR found")
                self._status.showMessage("QR detection: Nothing found — filename unchanged.")
                self._show_no_qr_dialog()
                return

            # ── Determine type: QR → -2, Barcode → -1 ─────────────────
            import re as _re
            types_found = [r.get("type", "qr") for r in regions]
            # QR = square matrix code; barcode = linear (CODE128 etc)
            has_qr      = any(t == "qr"      for t in types_found)
            has_barcode = any(t == "barcode"  for t in types_found)
            # If both found, prefer QR suffix
            if has_qr:
                suffix     = "-2"
                type_label = "QR code"
            elif has_barcode:
                suffix     = "-1"
                type_label = "Barcode"
            else:
                suffix     = "-2"   # cv2 fallback = treat as QR
                type_label = "QR code"

            self._lbl_qr_result.setStyleSheet(
                "color: #1a7a1a; font-size: 11px; font-weight: bold;")

            texts = [r["data"] for r in regions if r["data"]]
            if texts:
                self._lbl_qr_result.setText(
                    f"✓  [{type_label}] " + " | ".join(texts))
                self._qr_value_edit.setText(texts[0])
                clean = self._clean_qr_value(texts[0])
                clean = _re.sub(r'-[12]$', '', clean)
                self._file_name_edit.setText(f"{clean}{suffix}")
                self._status.showMessage(
                    f"{type_label} detected: {texts[0][:60]}  →  filename: {clean}{suffix}")
            else:
                base = os.path.splitext(
                    os.path.basename(getattr(self, "_current_path", "coin")))[0]
                base = _re.sub(r'-[12]$', '', base)
                self._lbl_qr_result.setText(
                    f"✓  [{type_label}] {len(regions)} region(s) found.")
                self._qr_value_edit.setText("")
                self._file_name_edit.setText(f"{base}{suffix}")
                self._status.showMessage(
                    f"{type_label} detected — filename: {base}{suffix}")

            self._show_qr_preview(regions[0])
            self._btn_update_qr.setEnabled(True)
            self._qr_last_region = regions[0]
        self._canvas.setFocus()
        # Update preview after detection
        self._update_crop_preview()

    def _on_worker_error(self, msg: str):
        QMessageBox.warning(self, "Detection Error", msg)
        self._status.showMessage("Detection failed.")

    def _on_worker_done(self):
        from PyQt5.QtWidgets import QApplication
        QApplication.restoreOverrideCursor()
        self._set_detection_buttons_enabled(True)

    def _set_detection_buttons_enabled(self, enabled: bool):
        for btn in [self._btn_circle, self._btn_rect, self._btn_detect_qr]:
            btn.setEnabled(enabled)

    def _update_crop_preview(self):
        """Refresh crop preview + handle zoom panel."""
        if not hasattr(self, "_crop_preview"):
            return
        if not self._processor.is_loaded:
            self._crop_preview.clear()
            if hasattr(self, "_handle_zoom"):
                self._handle_zoom.clear()
            return

        img        = self._processor.current
        crop_type  = self._canvas._crop_type  or self._processor.crop_type
        crop_params= self._canvas._crop_params or self._processor.crop_params

        # ── Pen mode: show pen handle positions in zoom panel ─────────
        if self._canvas.is_pen_mode() and self._canvas._pen_points:
            self._crop_preview.update_preview(img, crop_type, crop_params)
            if hasattr(self, "_handle_zoom"):
                import math
                pts = [pt["pt"] for pt in self._canvas._pen_points]
                n   = len(pts)
                # Find N/E/S/W = topmost/rightmost/bottommost/leftmost points
                top = min(pts, key=lambda p: p[1])
                bot = max(pts, key=lambda p: p[1])
                rgt = max(pts, key=lambda p: p[0])
                lft = min(pts, key=lambda p: p[0])
                pen_params = {
                    "x": lft[0], "y": top[1],
                    "width":  rgt[0] - lft[0],
                    "height": bot[1] - top[1],
                    # Store cardinal pts for zoom
                    "_N": top, "_E": rgt, "_S": bot, "_W": lft,
                }
                self._handle_zoom.update_zooms(img, "pen_circle", pen_params)
            return

        if crop_type and crop_params:
            self._crop_preview.update_preview(img, crop_type, crop_params)
            if hasattr(self, "_handle_zoom"):
                self._handle_zoom.update_zooms(img, crop_type, crop_params)
        else:
            self._crop_preview.update_preview(img, None, {})
            if hasattr(self, "_handle_zoom"):
                self._handle_zoom.clear()



    def _show_qr_preview(self, region: dict):
        bx, by, bw, bh = region["bbox"]
        img = self._processor.current
        h, w = img.shape[:2]

        # Tight crop with small padding around exact QR bbox
        pad = max(10, int(max(bw, bh) * 0.08))
        x1 = max(0, bx - pad)
        y1 = max(0, by - pad)
        x2 = min(w, bx + bw + pad)
        y2 = min(h, by + bh + pad)
        crop = img[y1:y2, x1:x2].copy()

        if crop.size == 0:
            return

        # Draw QR polygon on the preview crop
        if region.get("points") is not None:
            pts_local = region["points"].copy()
            pts_local[:, 0] -= x1
            pts_local[:, 1] -= y1
            cv2.polylines(crop, [pts_local.reshape(-1, 1, 2)],
                          True, (0, 220, 0), 2)

        rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        ch, cw = rgb.shape[:2]
        qimg = QImage(rgb.data, cw, ch, cw * 3, QImage.Format_RGB888)
        pm   = QPixmap.fromImage(qimg).scaled(
            self._qr_preview.width()  - 4,
            self._qr_preview.height() - 4,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._qr_preview.setPixmap(pm)
        self._qr_preview.setText("")

    def _thumb_prev(self):
        self._thumb_panel._prev()

    def _thumb_next(self):
        self._thumb_panel._next()

    def _toggle_guides(self, checked: bool):
        self._canvas.set_guides_visible(checked)

    def _fit_image(self):
        """Reset zoom to fit image in window (Ctrl+F)."""
        if self._processor.is_loaded:
            self._canvas.fit_view()
            self._status.showMessage("Fit to window.")

    def closeEvent(self, event):
        """Stop any running worker thread before closing."""
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait(2000)
        # Stop thumbnail loader if running
        if hasattr(self, "_thumb_panel") and self._thumb_panel._loader:
            if self._thumb_panel._loader.isRunning():
                self._thumb_panel._loader._stop = True
                self._thumb_panel._loader.wait(1000)
        event.accept()

    def _refresh_view(self):
        """Refresh — reload folder for new images + redraw canvas (F5 / Ctrl+R)."""
        # Reload current folder to pick up newly added images
        current_path = getattr(self, "_current_path", "")
        if hasattr(self, "_thumb_panel") and self._thumb_panel._paths:
            folder = os.path.dirname(self._thumb_panel._paths[0])
            self._thumb_panel.load_folder(folder)
            # Re-highlight current image if still open
            if current_path and os.path.isfile(current_path):
                from PyQt5.QtCore import QTimer
                QTimer.singleShot(
                    300,
                    lambda: self._thumb_panel.highlight_path(current_path))
            n = len(self._thumb_panel._paths)
            self._status.showMessage(
                f"🔃 Folder refreshed — {n} images found.")
        else:
            self._status.showMessage(
                "🔃 Open a folder first to use Refresh.")
            return

        # Also redraw canvas if image is loaded
        if self._processor.is_loaded:
            self._canvas._invalidate_cache()
            self._canvas._image = self._processor.current.copy()
            if self._processor.crop_type and self._processor.crop_params:
                self._canvas._crop_type   = self._processor.crop_type
                self._canvas._crop_params = self._processor.crop_params.copy()
            self._canvas._render_overlay()
            self._update_crop_preview()

    def _toggle_panel(self, checked: bool):
        """Show or hide the right side panel."""
        self._side_panel.setVisible(checked)
        if checked:
            self._splitter.setSizes([800, 300])
        else:
            self._splitter.setSizes([1, 0])

    def _set_panel_side(self, side: str):
        """Move side panel to left or right of canvas."""
        self._act_panel_right.setChecked(side == "right")
        self._act_panel_left.setChecked(side  == "left")

        # Detach both widgets
        canvas     = self._canvas
        side_panel = self._side_panel
        canvas.setParent(None)
        side_panel.setParent(None)

        # Re-add in correct order
        if side == "right":
            self._splitter.addWidget(canvas)
            self._splitter.addWidget(side_panel)
            self._splitter.setStretchFactor(0, 3)
            self._splitter.setStretchFactor(1, 1)
            self._splitter.setSizes([800, 300])
        else:
            self._splitter.addWidget(side_panel)
            self._splitter.addWidget(canvas)
            self._splitter.setStretchFactor(0, 1)
            self._splitter.setStretchFactor(1, 3)
            self._splitter.setSizes([300, 800])

        self._status.showMessage(f"Side panel → {side}.")

    # ------------------------------------------------------------------ #
    #  Help dialog                                                         #
    # ------------------------------------------------------------------ #

    def _show_help(self):
        from PyQt5.QtWidgets import (QDoubleSpinBox, QDialog, QVBoxLayout, QHBoxLayout,
                                      QScrollArea, QPushButton)
        from PyQt5.QtCore import Qt as _Qt

        dlg = QDialog(self)
        dlg.setWindowTitle("Keyboard Shortcuts")
        dlg.setFixedSize(540, 620)
        dlg.setStyleSheet("QDialog{background:#f2f4f8;} QLabel{background:transparent;}")

        outer = QVBoxLayout(dlg)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Title bar
        title_bar = QLabel("  ⌨  Keyboard Shortcuts")
        title_bar.setFixedHeight(44)
        title_bar.setStyleSheet(
            "background:#1c2b45;color:#e8edf5;font-size:15px;"
            "font-weight:bold;padding-left:12px;")
        outer.addWidget(title_bar)

        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setStyleSheet("background:#f2f4f8;")
        outer.addWidget(scroll, 1)

        content = QWidget()
        content.setStyleSheet("background:#f2f4f8;")
        vlay = QVBoxLayout(content)
        vlay.setContentsMargins(20, 16, 20, 16)
        vlay.setSpacing(3)
        scroll.setWidget(content)

        def section(title):
            lbl = QLabel(title)
            lbl.setFixedHeight(28)
            lbl.setStyleSheet(
                "background:#1c2b45;color:#c8d8f0;font-weight:bold;"
                "font-size:12px;padding-left:10px;border-radius:4px;"
                "margin-top:10px;")
            vlay.addWidget(lbl)

        def row(key, desc):
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(8, 2, 8, 2)
            k = QLabel(key)
            k.setFixedWidth(195)
            k.setStyleSheet(
                "font-family:Courier New;font-size:12px;color:#3e6188;"
                "font-weight:bold;background:transparent;")
            d = QLabel(desc)
            d.setStyleSheet("font-size:12px;color:#1c2333;background:transparent;")
            h.addWidget(k); h.addWidget(d, 1)
            vlay.addWidget(w)

        section("📁  File")
        row("Ctrl + O",             "Open image file")
        row("Ctrl + Shift + O",     "Open folder (thumbnail strip)")
        row("Ctrl + S",             "Save image")
        row("Del",                  "Delete current image file")
        row("Ctrl + Q",             "Quit application")

        section("↩  Undo / Reset")
        row("Ctrl + Z",             "Undo 1 step (previous state)")
        row("Ctrl + Shift + Z",     "Reset All — back to original image")

        section("🔄  Rotation")
        row("◀ -1°  button",        "Fine rotate −1° (right panel)")
        row("+1° ▶  button",        "Fine rotate +1° (right panel)")
        row("↺ CCW −90°  button",   "Rotate counter-clockwise 90°")
        row("↻ CW +90°   button",   "Rotate clockwise 90°")
        row("Ctrl + Left",          "Rotate CCW −90°")
        row("Ctrl + Right",         "Rotate CW  +90°")
        row("[",                    "Rotate CCW −1°  (fine)")
        row("]",                    "Rotate CW  +1°  (fine)")

        section("✂  Crop (after detect)")
        row("Arrow keys",           "Move crop region")
        row("+ / −",                "Resize crop region")
        row("Enter / Return",       "Apply crop")
        row("Mouse drag (inside)",  "Move crop region")
        row("Mouse drag (handle)",  "Resize or reshape corner/edge")
        row("↻ handle (rect)",      "Rotate rectangle")
        row("Mouse wheel",          "Uniform resize")

        section("🔍  Detection")
        row("⭕ Circle  button",     "Detect coin circle")
        row("▭ Rectangle  button",  "Detect slab rectangle")
        row("⊞  Detect QR  button", "Detect QR / Label (manual)")
        row("Ctrl + W",              "Detect QR / Label")

        section("🖼  Thumbnail Navigation")
        row("Alt + Left",           "Previous image in folder")
        row("Alt + Right",          "Next image in folder")
        row("◀ / ▶ buttons",       "Navigate thumbnail strip")
        row("Click thumbnail",      "Load that image")

        section("👁  View")
        row("Ctrl + F",             "Fit image to window")
        row("F5  /  Ctrl + R",         "Refresh folder — pick up new images")
        row("View → Guide Lines",   "Toggle 8 grid lines on/off")
        row("View → Side Panel",    "Show / hide right panel")

        section("🔎  Handle Zoom Panel")
        row("Click any zoom box",   "Change zoom level  (1× → 8×)")

        vlay.addStretch()

        # Close button
        btn = QPushButton("  Close")
        btn.setFixedHeight(38)
        btn.setStyleSheet(
            "QPushButton{background:#3e6188;color:white;font-weight:bold;"
            "border:none;font-size:13px;border-radius:0;}"
            "QPushButton:hover{background:#4a7aaa;}")
        btn.clicked.connect(dlg.accept)
        outer.addWidget(btn)

        dlg.exec_()

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _refresh_canvas(self):
        # Invalidate cache since image pixels changed
        self._canvas._invalidate_cache()
        self._canvas._image = self._processor.current.copy()
        # Keep crop overlay — but don't override if pen mode active
        if not self._canvas.is_pen_mode():
            if self._processor.crop_type and self._processor.crop_params:
                self._canvas._crop_type   = self._processor.crop_type
                self._canvas._crop_params = self._processor.crop_params.copy()
        self._canvas._render_overlay()
        # Always sync preview with latest canvas state
        self._update_crop_preview()

    def _sync_spin(self):
        self._spin_angle.blockSignals(True)
        self._spin_angle.setValue(int(self._processor.angle))
        self._spin_angle.blockSignals(False)

    def _update_ui_state(self):
        loaded = self._processor.is_loaded
        for w in [self._btn_ccw, self._btn_cw, self._btn_circle,
                  self._btn_rect, self._btn_apply_crop,
                  self._btn_detect_qr, self._btn_save,
                  self._spin_angle]:
            w.setEnabled(loaded)