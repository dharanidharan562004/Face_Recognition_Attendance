"""
handle_zoom_panel.py
Shows 4 zoomed handle previews in the side panel.

Circle  : N (top), E (right), S (bottom), W (left)
Rectangle: TL, TR, BR, BL
"""

import cv2
import numpy as np
from PyQt5.QtCore    import Qt
from PyQt5.QtGui     import QPixmap, QImage, QPainter, QPen, QColor, QFont
from PyQt5.QtWidgets import (QWidget, QGridLayout, QVBoxLayout,
                               QLabel, QSizePolicy, QMenu, QAction)

ZOOM_W  = 100   # each zoom box width
ZOOM_H  = 100   # each zoom box height
ZOOM_F  = 5.0  # magnification factor


def _make_zoom_pixmap(image: np.ndarray, ix: float, iy: float,
                       label: str) -> QPixmap:
    """Extract zoomed patch around (ix,iy) in image coords."""
    ih, iw = image.shape[:2]
    sample = int(ZOOM_W / _CURRENT_ZOOM_F[0] / 2) + 1

    sx1 = max(0, int(ix - sample))
    sy1 = max(0, int(iy - sample))
    sx2 = min(iw, int(ix + sample))
    sy2 = min(ih, int(iy + sample))

    if sx2 <= sx1 or sy2 <= sy1:
        pm = QPixmap(ZOOM_W, ZOOM_H)
        pm.fill(QColor("#c8cdd8"))
        return pm

    patch  = image[sy1:sy2, sx1:sx2]
    # Step 1: Upscale 2x first for smoother magnification
    ph, pw = patch.shape[:2]
    patch2x = cv2.resize(patch, (pw*2, ph*2), interpolation=cv2.INTER_CUBIC)

    # Step 2: Unsharp mask for crisp edges
    blurred  = cv2.GaussianBlur(patch2x, (5, 5), 1.0)
    sharpened = cv2.addWeighted(patch2x, 2.0, blurred, -1.0, 0)

    # Step 3: Final resize to zoom box
    zoomed = cv2.resize(sharpened, (ZOOM_W, ZOOM_H),
                        interpolation=cv2.INTER_LANCZOS4)
    rgb    = cv2.cvtColor(zoomed, cv2.COLOR_BGR2RGB)
    qi     = QImage(rgb.data, ZOOM_W, ZOOM_H, ZOOM_W * 3,
                    QImage.Format_RGB888)
    pm     = QPixmap.fromImage(qi).copy()

    # Draw crosshair + border
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.Antialiasing)
    cx, cy = ZOOM_W // 2, ZOOM_H // 2

    # Crosshair shadow (white) for contrast
    painter.setPen(QPen(QColor(255, 255, 255, 180), 3))
    painter.drawLine(cx - 12, cy, cx + 12, cy)
    painter.drawLine(cx, cy - 12, cx, cy + 12)

    # Crosshair red
    painter.setPen(QPen(QColor(220, 40, 40), 1))
    painter.drawLine(cx - 12, cy, cx + 12, cy)
    painter.drawLine(cx, cy - 12, cx, cy + 12)

    # Centre dot
    painter.setBrush(QColor(220, 40, 40))
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(cx - 3, cy - 3, 6, 6)
    painter.setBrush(Qt.NoBrush)

    # Border
    painter.setPen(QPen(QColor(255, 180, 0), 2))
    painter.drawRect(0, 0, ZOOM_W - 1, ZOOM_H - 1)

    painter.end()
    return pm


# Global zoom factor (shared across all cards)
_CURRENT_ZOOM_F = [1.5]   # mutable list so inner functions can update

ZOOM_OPTIONS = [
    ("1×  — wide context",   1.5),
    ("2×  — balanced",       2.5),
    ("3×  — close",          3.5),
    ("5×  — maximum",        5.0),
    ("8×  — ultra",          8.0),
]


class ZoomCard(QWidget):
    """Single zoom card: label + zoomed image. Click to change zoom level."""

    zoom_changed = None   # set by HandleZoomPanel

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.setFixedSize(ZOOM_W + 6, ZOOM_H + 22)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Click to change zoom level")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(1)

        # Top row: label + zoom indicator
        from PyQt5.QtWidgets import QHBoxLayout
        top = QWidget()
        top_lay = QHBoxLayout(top)
        top_lay.setContentsMargins(0,0,0,0)
        top_lay.setSpacing(2)

        self._lbl_name = QLabel(label)
        self._lbl_name.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._lbl_name.setStyleSheet(
            "font-size:9px; font-weight:bold; color:#3e6188; background:transparent;")
        top_lay.addWidget(self._lbl_name)

        self._lbl_zoom = QLabel(f"{_CURRENT_ZOOM_F[0]:.0f}×")
        self._lbl_zoom.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._lbl_zoom.setStyleSheet(
            "font-size:8px; color:#888; background:transparent;")
        top_lay.addWidget(self._lbl_zoom)
        top.setFixedHeight(14)
        lay.addWidget(top)

        self._img_lbl = QLabel()
        self._img_lbl.setFixedSize(ZOOM_W, ZOOM_H)
        self._img_lbl.setAlignment(Qt.AlignCenter)
        self._img_lbl.setStyleSheet(
            "background:#c8cdd8; border:1px solid #9aaac0; border-radius:3px;")
        lay.addWidget(self._img_lbl)

    def set_pixmap(self, pm: QPixmap):
        self._img_lbl.setPixmap(pm)
        self._lbl_zoom.setText(f"{_CURRENT_ZOOM_F[0]:.0f}×")

    def clear(self):
        self._img_lbl.setPixmap(QPixmap())
        self._img_lbl.setStyleSheet(
            "background:#c8cdd8; border:1px solid #9aaac0; border-radius:3px;")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._show_zoom_menu()
        super().mousePressEvent(event)

    def _show_zoom_menu(self):
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background:#1e3050; color:#e8edf5;
                border:1px solid #3a5a88; border-radius:4px;
                padding:4px;
            }
            QMenu::item { padding:6px 20px; border-radius:3px; font-size:12px; }
            QMenu::item:selected { background:#3e6188; }
            QMenu::item:checked { color:#4adf8a; font-weight:bold; }
        """)
        for label, val in ZOOM_OPTIONS:
            act = QAction(label, self)
            act.setCheckable(True)
            act.setChecked(abs(_CURRENT_ZOOM_F[0] - val) < 0.1)
            act.setData(val)
            menu.addAction(act)

        chosen = menu.exec_(self.mapToGlobal(self.rect().bottomLeft()))
        if chosen and chosen.data() is not None:
            _CURRENT_ZOOM_F[0] = chosen.data()
            self._lbl_zoom.setText(f"{_CURRENT_ZOOM_F[0]:.0f}×")
            # Notify panel to refresh all zoom boxes
            if self.zoom_changed:
                self.zoom_changed()


class HandleZoomPanel(QWidget):
    """
    4-quadrant zoom panel for the right side panel.

    Circle layout:
        ┌───┬───┐
        │   │ N │
        ├───┼───┤
        │ W │ E │
        ├───┼───┤
        │   │ S │
        └───┴───┘

    Rectangle layout:
        ┌────┬────┐
        │ TL │ TR │
        ├────┼────┤
        │ BL │ BR │
        └────┴────┘
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setStyleSheet("background:transparent;")

        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(4, 4, 4, 4)
        self._grid.setSpacing(4)

        # Create 4 cards — positions change per crop type
        self._cards = {
            "TL": ZoomCard("TL"),
            "TR": ZoomCard("TR"),
            "BL": ZoomCard("BL"),
            "BR": ZoomCard("BR"),
            "N":  ZoomCard("N"),
            "E":  ZoomCard("E"),
            "S":  ZoomCard("S"),
            "W":  ZoomCard("W"),
        }
        # Connect zoom_changed to re-render all boxes
        for card in self._cards.values():
            card.zoom_changed = self._on_zoom_changed

        self._last_image: np.ndarray | None = None
        self._last_crop_type: str = "rectangle"
        self._last_crop_params: dict = {}
        self._active_keys: list[str] = []
        self._layout_cards("rectangle")   # default

    def _clear_grid(self):
        for key, card in self._cards.items():
            self._grid.removeWidget(card)
            card.setParent(None)

    def _layout_cards(self, crop_type: str):
        self._clear_grid()
        if crop_type in ("circle", "pen_circle"):
            # 3×2 grid: N centre top, W left mid, E right mid, S centre bot
            keys = [("N", 0, 1), ("W", 1, 0), ("E", 1, 2), ("S", 2, 1)]
            self._active_keys = ["N", "E", "S", "W"]
        else:
            # 2×2 grid: TL, TR, BL, BR
            keys = [("TL", 0, 0), ("TR", 0, 1),
                    ("BL", 1, 0), ("BR", 1, 1)]
            self._active_keys = ["TL", "TR", "BL", "BR"]

        for key, row, col in keys:
            card = self._cards[key]
            card.setParent(self)
            self._grid.addWidget(card, row, col, Qt.AlignCenter)

        # Adjust fixed height
        rows = 3 if crop_type in ("circle", "pen_circle") else 2
        self.setFixedHeight(rows * (ZOOM_H + 24) + 8)

    # ── Public ──────────────────────────────────────────────────────────

    def _on_zoom_changed(self):
        """Called when user picks a new zoom level — re-render all."""
        if self._last_image is not None and self._last_crop_params:
            self.update_zooms(self._last_image,
                              self._last_crop_type,
                              self._last_crop_params)

    def update_zooms(self, image: np.ndarray,
                      crop_type: str, crop_params: dict):
        if image is None or not crop_params:
            self.clear()
            return

        # Cache for zoom level change refresh
        self._last_image       = image
        self._last_crop_type   = crop_type
        self._last_crop_params = crop_params.copy()

        self._layout_cards(crop_type)

        s  = 1.0
        ih, iw = image.shape[:2]

        if crop_type == "circle":
            cx = crop_params["x"]
            cy = crop_params["y"]
            r  = crop_params["radius"]
            positions = {
                "N": (cx,   cy-r),
                "E": (cx+r, cy  ),
                "S": (cx,   cy+r),
                "W": (cx-r, cy  ),
            }
        elif crop_type == "pen_circle":
            # Use stored cardinal pen points
            positions = {
                "N": crop_params.get("_N", (0, 0)),
                "E": crop_params.get("_E", (0, 0)),
                "S": crop_params.get("_S", (0, 0)),
                "W": crop_params.get("_W", (0, 0)),
            }
            # Force circle layout
            self._layout_cards("circle")
        else:
            x  = crop_params["x"]
            y  = crop_params["y"]
            w  = crop_params["width"]
            h  = crop_params["height"]
            positions = {
                "TL": (x,   y  ),
                "TR": (x+w, y  ),
                "BR": (x+w, y+h),
                "BL": (x,   y+h),
            }

        for key in self._active_keys:
            ix, iy = positions[key]
            pm = _make_zoom_pixmap(image, ix, iy, key)
            self._cards[key].set_pixmap(pm)

    def clear(self):
        for card in self._cards.values():
            card.clear()