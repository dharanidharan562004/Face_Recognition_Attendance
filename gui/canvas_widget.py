from __future__ import annotations
"""
canvas_widget.py
Image canvas with crop overlays.

Rectangle — single clean overlay:
  • Inside drag       → move
  • Corner handles    → resize that corner (opposite fixed)
  • Mid-edge handles  → push that edge
  • ↻ Rotate handle   → rotate around centre
  • Mouse wheel       → uniform resize

Circle — same as before (N/S/E/W handles).
"""

import math
import cv2
import numpy as np
from PyQt5.QtCore  import Qt, QPoint, QRect, pyqtSignal
from PyQt5.QtGui   import (QImage, QPixmap, QPainter, QPen,
                            QColor, QFont, QBrush, QPolygon)
from PyQt5.QtWidgets import QLabel, QSizePolicy

# ── Hit zones ────────────────────────────────────────────────────────────
_NONE      = 0
_INSIDE    = 1
_EDGE      = 2   # circle arc between handles
_C_N       = 10  # circle North
_C_S       = 11
_C_E       = 12
_C_W       = 13
_R_TL      = 20  # rect corners
_R_TR      = 21
_R_BR      = 22
_R_BL      = 23
_R_T       = 30  # rect edges
_R_R       = 31
_R_B       = 32
_R_L       = 33
_R_ROT     = 40  # rotation handle

HR  = 7    # handle radius (display px)
TOL = 16   # hit tolerance px


class CanvasWidget(QLabel):

    rotate_cw_requested  = pyqtSignal()
    rotate_ccw_requested = pyqtSignal()
    crop_confirmed       = pyqtSignal()
    crop_move            = pyqtSignal(int, int)
    crop_resize          = pyqtSignal(int)
    pen_closed           = pyqtSignal()   # polygon closed → apply pen crop
    patch_apply          = pyqtSignal()
    patch_cancelled      = pyqtSignal()
    patch_phase_changed  = pyqtSignal()
    pen_edit_region      = pyqtSignal()   # edit text in rect region
    pen_scan_qr_region   = pyqtSignal()   # scan QR in region

    KEY_MOVE   = 10
    KEY_RESIZE = 15
    WHEEL_STEP = 20

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(400, 400)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)

        self._image        = None
        self._show_guides  = True
        self._crop_type    = None
        self._crop_params  = {}
        self._qr_regions   = []

        self._scale        = 1.0
        self._offset       = QPoint(0, 0)
        self._base_pixmap  = None
        self._last_img_id  = -1
        self._last_size    = (-1, -1)

        # ── Zoom & Pan ────────────────────────────────────────────────
        self._zoom_factor  = 1.0
        self._fit_scale    = 1.0
        self._pan_start    = None
        self._pan_offset   = QPoint(0, 0)
        self._space_held   = False   # Space key = pan mode

        self._drag_zone    = _NONE
        self._drag_start   = QPoint()
        self._drag_orig    = {}

        # ── Pen tool state ────────────────────────────────────────────
        self._pen_mode     = False
        self._pen_points   = []
        self._pen_hover    = None
        self._pen_drag_idx = -1
        self._pen_drag_part = None
        self._pen_is_drawing = False
        self._pen_new_pt    = None
        self._pen_rect     = None
        self._pen_circle   = None
        self._pen_drag_start_img = None
        self._pen_orig_pts       = []
        self._pen_closed         = False  # path is closed

        # ── Patch tool state ──────────────────────────────────────────
        self._patch_mode    = False
        self._patch_phase   = "draw"   # "draw" | "drag"
        self._patch_lasso   = []       # freehand lasso points (image coords)
        self._patch_src_pts = []       # closed source selection
        self._patch_drag_start = None  # QPoint where drag started
        self._patch_offset  = (0, 0)   # current drag offset (image coords)

        # ── Pen rect drag ─────────────────────────────────────────────
        self._pen_drag_zone  = None   # 'TL','TR','BR','BL','T','R','B','L','M'
        self._pen_drag_start = QPoint()
        self._pen_drag_orig  = {}

        # ── Zoom lens ─────────────────────────────────────────────────
        self._zoom_pos     = None    # QPoint mouse pos in widget coords

    # ── Public ──────────────────────────────────────────────────────────

    def set_image(self, image):
        self._image       = image.copy()
        self._crop_type   = None
        self._crop_params = {}
        self._qr_regions  = []
        self._drag_zone   = _NONE
        self._drag_orig   = {}
        self._pen_mode    = False
        self._pen_points  = []
        self._pen_hover   = None
        self._pen_drag_idx = -1
        self._pen_drag_part = None
        self._pen_is_drawing = False
        self._pen_new_pt  = None
        self._pen_rect    = None
        self._pen_circle  = None
        self._zoom_factor = 1.0        # reset zoom on new image
        self._pan_offset  = QPoint(0,0)
        self.setCursor(Qt.ArrowCursor)
        self._invalidate_cache()
        self._render()

    def set_guides_visible(self, v):
        self._show_guides = v
        if self._image is not None:
            self._render_overlay()

    def set_crop_overlay(self, crop_type, params):
        self._crop_type   = crop_type
        self._crop_params = params.copy() if params else {}
        self._render_overlay()

    def set_qr_regions(self, regions):
        self._qr_regions = regions
        self._render_overlay()

    # ── Pen Tool API ─────────────────────────────────────────────────────

    def start_pen_tool(self):
        """Mode 1 — Free pen: click to place points anywhere."""
        self._pen_mode       = True
        self._pen_points     = []
        self._pen_hover      = None
        self._pen_drag_idx   = -1
        self._pen_drag_part  = None
        self._pen_is_drawing = False
        self._pen_new_pt     = None
        self._pen_rect       = None
        self._pen_circle     = None
        self._pen_closed     = False
        self.setCursor(Qt.CrossCursor)
        self._render_overlay()

    def start_pen_with_shapes(self, rect_params, circle_params):
        """Mode 2 — Auto-points on detected circle/rect edge."""
        self._pen_mode       = True
        self._pen_points     = []
        self._pen_hover      = None
        self._pen_drag_idx   = -1
        self._pen_drag_part  = None
        self._pen_is_drawing = False
        self._pen_new_pt     = None
        self._pen_rect       = rect_params.copy()   if rect_params   else None
        self._pen_circle     = circle_params.copy() if circle_params else None
        self._pen_closed     = True   # Mode 2 starts closed (all points placed)
        self.setCursor(Qt.OpenHandCursor)
        self._render_overlay()

    def stop_pen_tool(self):
        """Exit pen tool mode."""
        self._pen_mode       = False
        self._pen_points     = []
        self._pen_hover      = None
        self._pen_drag_idx   = -1
        self._pen_drag_part  = None
        self._pen_is_drawing = False
        self._pen_new_pt     = None
        self._pen_rect       = None
        self._pen_circle     = None
        self._pen_closed     = False
        self.setCursor(Qt.ArrowCursor)
        self._render_overlay()

    def get_pen_points(self):
        """Return pen anchor points as (x,y) list for polygon crop."""
        return [pt["pt"] for pt in self._pen_points]

    def is_pen_mode(self):
        return self._pen_mode

    # ── Patch Tool API ────────────────────────────────────────────────────
    def start_patch_tool(self):
        self._patch_mode    = True
        self._patch_phase   = "draw"
        self._patch_lasso   = []
        self._patch_src_pts = []
        self._patch_drag_start = None
        self._patch_offset  = (0, 0)
        self.setCursor(Qt.CrossCursor)
        self._render_overlay()

    def stop_patch_tool(self):
        self._patch_mode    = False
        self._patch_phase   = "draw"
        self._patch_lasso   = []
        self._patch_src_pts = []
        self._patch_drag_start = None
        self._patch_offset  = (0, 0)
        self.setCursor(Qt.ArrowCursor)
        self._render_overlay()

    def is_patch_mode(self):
        return self._patch_mode

    def apply_patch(self, image):
        """Apply patch — seamless texture copy from destination to source."""
        import cv2 as _cv2
        import numpy as _np

        if not self._patch_src_pts or len(self._patch_src_pts) < 3:
            return None

        h, w   = image.shape[:2]
        ox, oy = self._patch_offset

        # Simplify lasso points (reduce to 32 max for performance)
        pts = self._patch_src_pts
        if len(pts) > 32:
            step = len(pts) // 32
            pts  = pts[::step]

        src_pts = _np.array([[int(_np.clip(x,1,w-2)),
                               int(_np.clip(y,1,h-2))]
                              for x,y in pts], dtype=_np.int32)

        # Source mask (slightly eroded to avoid edge artifacts)
        src_mask = _np.zeros((h,w), dtype=_np.uint8)
        _cv2.fillPoly(src_mask, [src_pts], 255)
        kernel   = _np.ones((3,3), _np.uint8)
        src_mask = _cv2.erode(src_mask, kernel, iterations=1)

        if src_mask.sum() == 0:
            return None

        # Source center
        M  = _cv2.moments(src_pts)
        if M["m00"] == 0:
            return None
        cx = int(_np.clip(M["m10"]/M["m00"], 1, w-2))
        cy = int(_np.clip(M["m01"]/M["m00"], 1, h-2))

        # Shift image: bring destination texture to source position
        T       = _np.float32([[1,0,-ox],[0,1,-oy]])
        shifted = _cv2.warpAffine(image, T, (w,h),
                                   borderMode=_cv2.BORDER_REFLECT_101)

        # Try seamlessClone (best quality)
        try:
            result = _cv2.seamlessClone(
                shifted, image, src_mask,
                (cx, cy), _cv2.NORMAL_CLONE)
        except Exception:
            # Fallback: Poisson-blend manually using Gaussian feather
            result  = image.copy()
            feather = _cv2.GaussianBlur(
                src_mask.astype(_np.float32)/255.0, (21,21), 7)
            for c in range(3):
                result[:,:,c] = (
                    shifted[:,:,c] * feather +
                    image[:,:,c]   * (1-feather)
                ).astype(_np.uint8)

        return result

    # ── Cache ────────────────────────────────────────────────────────────

    def _invalidate_cache(self):
        self._base_pixmap = None

    def fit_view(self):
        """Reset zoom to fit image in window."""
        self._zoom_factor = 1.0
        self._pan_offset  = QPoint(0, 0)
        self._invalidate_cache()
        self._render_overlay()

    def _ensure_base(self):
        if self._image is None:
            return False
        ww, wh   = self.width(), self.height()
        img_id   = id(self._image)

        # Rebuild pixmap cache only when image or window size changes
        if not (hasattr(self, '_full_pixmap') and
                self._last_img_id == img_id and
                self._last_size == (ww, wh)):
            rgb  = cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            qi   = QImage(rgb.data, w, h, w*3, QImage.Format_RGB888)
            self._full_pixmap = QPixmap.fromImage(qi)
            self._img_w = w;  self._img_h = h

            # Fit scale — reset zoom when new image/size
            margin = 20
            aw, ah = max(1, ww-margin*2), max(1, wh-margin*2)
            self._fit_scale = min(aw/w, ah/h)
            self._last_img_id = img_id
            self._last_size   = (ww, wh)

        # Always recalculate scale + offset (zoom/pan changes every time)
        w = self._img_w;  h = self._img_h
        self._scale = self._fit_scale * self._zoom_factor
        sw = int(w * self._scale);  sh = int(h * self._scale)

        # Centre + pan offset
        cx = (ww - sw) // 2 + self._pan_offset.x()
        cy = (wh - sh) // 2 + self._pan_offset.y()
        self._offset = QPoint(cx, cy)

        # Always render fresh base
        scaled = self._full_pixmap.scaled(
            max(1, sw), max(1, sh),
            Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        base = QPixmap(ww, wh)
        base.fill(QColor("#f2f4f8"))
        p = QPainter(base)
        p.drawPixmap(self._offset, scaled)
        p.end()

        self._base_pixmap = base
        return True

    # ── Render ───────────────────────────────────────────────────────────

    def _render(self):
        if self._image is None:
            self.setText("Open an image to get started…")
            self.setStyleSheet("background:#f2f4f8;color:#555;font-size:16px;")
            return
        self._invalidate_cache()
        self._render_overlay()
        self.setStyleSheet("background:#f2f4f8;")

    def _render_overlay(self):
        if not self._ensure_base():
            self.setText("Open an image to get started…")
            self.setStyleSheet("background:#f2f4f8;color:#555;font-size:16px;")
            return

        canvas = self._base_pixmap.copy()
        p = QPainter(canvas)
        p.setRenderHint(QPainter.Antialiasing)

        if self._show_guides:
            self._draw_guides(p)
        if self._crop_type:
            self._draw_crop(p)
        for r in self._qr_regions:
            self._draw_qr(p, r)
        if self._pen_mode:
            self._draw_pen(p)
        if self._patch_mode:
            self._draw_patch(p)
        # Zoom lens — always show when crop is active and mouse is over image
        if self._crop_type and self._zoom_pos and self._image is not None:
            self._draw_zoom_lens(p)
        p.end()
        self.setPixmap(canvas)
        self.setStyleSheet("background:#f2f4f8;")

    # ── Drawing ──────────────────────────────────────────────────────────

    def _draw_guides(self, p):
        if not hasattr(self, '_img_w'):
            return
        # Image rect in widget coords
        ox  = self._offset.x()
        oy  = self._offset.y()
        sw  = int(self._img_w * self._scale)
        sh  = int(self._img_h * self._scale)

        # Clip painter to image rect — grid never goes outside image
        p.save()
        p.setClipRect(QRect(ox, oy, sw, sh))

        pen = QPen(QColor(0, 140, 255, 110), 1, Qt.DashLine)
        p.setPen(pen)
        for pct in (0.2, 0.4, 0.6, 0.8):
            p.drawLine(ox,          oy + int(sh*pct), ox + sw, oy + int(sh*pct))
            p.drawLine(ox + int(sw*pct), oy,          ox + int(sw*pct), oy + sh)

        # Centre dot
        cx, cy = ox + sw//2, oy + sh//2
        p.setBrush(QColor(0, 200, 100, 180))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPoint(cx, cy), 3, 3)
        p.setBrush(Qt.NoBrush)
        p.restore()

    def _draw_crop(self, p):
        if not self._crop_params:
            return
        s  = self._scale
        ox = self._offset.x()
        oy = self._offset.y()

        glow = QPen(QColor(255, 180, 0, 55), 10)
        main = QPen(QColor(255, 180, 0), 2)
        hpen = QPen(QColor(255, 255, 255), 2)
        hfil = QColor(255, 180, 0)

        if self._crop_type == "circle":
            cx = int(self._crop_params["x"]      * s) + ox
            cy = int(self._crop_params["y"]      * s) + oy
            r  = int(self._crop_params["radius"] * s)

            # ── Square bounding box ──────────────────────────────────
            bx, by = cx - r, cy - r
            bw = bh = r * 2
            sq_pen = QPen(QColor(100, 200, 255, 160), 1, Qt.DashLine)
            p.setPen(sq_pen)
            p.drawRect(QRect(bx, by, bw, bh))

            # ── Internal grid lines (3×3 inside bounding box) ────────
            grid_pen = QPen(QColor(100, 200, 255, 80), 1, Qt.DotLine)
            p.setPen(grid_pen)
            for i in range(1, 3):
                gx = bx + (bw * i) // 3
                gy = by + (bh * i) // 3
                p.drawLine(gx, by, gx, by + bh)   # vertical
                p.drawLine(bx, gy, bx + bw, gy)   # horizontal

            # ── Circle overlay ───────────────────────────────────────
            p.setPen(glow); p.drawEllipse(QPoint(cx,cy), r, r)
            p.setPen(main); p.drawEllipse(QPoint(cx,cy), r, r)

            # Centre cross
            p.setPen(hpen)
            p.drawLine(cx-8,cy,cx+8,cy); p.drawLine(cx,cy-8,cx,cy+8)

            # N/E/S/W handles
            for hx, hy in [(cx,cy-r),(cx+r,cy),(cx,cy+r),(cx-r,cy)]:
                p.setPen(hpen); p.setBrush(hfil)
                p.drawEllipse(QPoint(hx,hy), HR, HR)
                p.setBrush(Qt.NoBrush)

        elif self._crop_type == "rectangle":
            params = self._crop_params
            x  = int(params["x"]      * s) + ox
            y  = int(params["y"]      * s) + oy
            w  = int(params["width"]  * s)
            h  = int(params["height"] * s)
            angle = params.get("angle", 0.0)
            cx, cy = x + w//2, y + h//2

            # Draw rotated via transform
            p.save()
            p.translate(cx, cy)
            p.rotate(angle)
            p.translate(-cx, -cy)

            # Single clean rectangle border
            p.setPen(glow); p.drawRect(QRect(x, y, w, h))
            p.setPen(main); p.drawRect(QRect(x, y, w, h))

            # Centre cross
            p.setPen(hpen)
            p.drawLine(cx-8,cy,cx+8,cy); p.drawLine(cx,cy-8,cx,cy+8)

            # 4 corner handles
            for hx, hy in [(x,y),(x+w,y),(x+w,y+h),(x,y+h)]:
                p.setPen(hpen); p.setBrush(hfil)
                p.drawEllipse(QPoint(hx,hy), HR, HR)
                p.setBrush(Qt.NoBrush)

            # 4 mid-edge handles
            for hx, hy in [(cx,y),(x+w,cy),(cx,y+h),(x,cy)]:
                p.setPen(hpen)
                p.setBrush(QColor(255,180,0,160))
                p.drawEllipse(QPoint(hx,hy), HR-2, HR-2)
                p.setBrush(Qt.NoBrush)

            p.restore()

            # Rotation handle (always above TL, outside transform)
            # Find rotated TL position
            import math as _m
            rad   = _m.radians(angle)
            tlx   = cx + (x-cx)*_m.cos(rad) - (y-cy)*_m.sin(rad)
            tly   = cy + (x-cx)*_m.sin(rad) + (y-cy)*_m.cos(rad)
            rot_hx = int(tlx - 20*_m.sin(rad))
            rot_hy = int(tly - 20*_m.cos(rad))

            p.setPen(QPen(QColor(60,180,255), 1))
            p.setBrush(QColor(60,180,255,220))
            p.drawEllipse(QPoint(rot_hx, rot_hy), 10, 10)
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(255,255,255), 2))
            p.setFont(QFont("Arial", 9, QFont.Bold))
            p.drawText(rot_hx-6, rot_hy+5, "↻")
            p.setPen(QPen(QColor(60,180,255,150), 1, Qt.DotLine))
            p.drawLine(int(tlx), int(tly), rot_hx, rot_hy)
            if abs(angle) > 0.5:
                p.setPen(QColor(60,180,255))
                p.setFont(QFont("Arial", 8))
                p.drawText(rot_hx+13, rot_hy+5, f"{angle:.1f}°")

    def _draw_zoom_lens(self, p):
        """Photoshop-style zoom lens that follows the cursor."""
        if self._image is None or self._zoom_pos is None:
            return

        mx, my = self._zoom_pos.x(), self._zoom_pos.y()
        s  = self._scale
        ox = self._offset.x()
        oy = self._offset.y()

        # Image coords at mouse
        ix = (mx - ox) / s
        iy = (my - oy) / s
        ih, iw = self._image.shape[:2]

        # Only show when inside image bounds
        if not (0 <= ix < iw and 0 <= iy < ih):
            return

        # Lens size & zoom factor
        LW, LH = 160, 160
        ZOOM    = 4.0
        sample  = int(LW / ZOOM / 2) + 1

        sx1 = max(0, int(ix - sample))
        sy1 = max(0, int(iy - sample))
        sx2 = min(iw, int(ix + sample))
        sy2 = min(ih, int(iy + sample))
        if sx2 <= sx1 or sy2 <= sy1:
            return

        patch = self._image[sy1:sy2, sx1:sx2]
        import cv2 as _cv2
        # Sharpen
        blurred  = _cv2.GaussianBlur(patch, (5, 5), 1)
        sharpened = _cv2.addWeighted(patch, 1.8, blurred, -0.8, 0)
        zoomed   = _cv2.resize(sharpened, (LW, LH),
                               interpolation=_cv2.INTER_LANCZOS4)
        rgb      = _cv2.cvtColor(zoomed, _cv2.COLOR_BGR2RGB)
        from PyQt5.QtGui import QImage as _QI
        qi  = _QI(rgb.data, LW, LH, LW * 3, _QI.Format_RGB888)
        pm  = QPixmap.fromImage(qi).copy()

        # Position lens — top-right of cursor, avoid edge clipping
        lx = mx + 20
        ly = my - LH - 20
        if lx + LW > self.width():  lx = mx - LW - 20
        if ly < 0:                   ly = my + 20

        # Draw lens shadow
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(0, 0, 0, 60)))
        p.drawEllipse(QRect(lx+4, ly+4, LW, LH))

        # Draw circular clipping for lens
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        path.addEllipse(lx, ly, LW, LH)
        p.setClipPath(path)
        p.drawPixmap(lx, ly, pm)
        p.setClipping(False)

        # Lens border
        p.setPen(QPen(QColor(255, 220, 80), 2))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QRect(lx, ly, LW, LH))

        # Crosshair inside lens
        lcx, lcy = lx + LW//2, ly + LH//2
        p.setPen(QPen(QColor(255, 80, 80), 1))
        p.drawLine(lcx - 14, lcy, lcx + 14, lcy)
        p.drawLine(lcx, lcy - 14, lcx, lcy + 14)

        # Centre dot
        p.setBrush(QBrush(QColor(255, 80, 80)))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPoint(lcx, lcy), 2, 2)
        p.setBrush(Qt.NoBrush)

        # Coords label
        p.setPen(QPen(QColor(255, 220, 80), 1))
        p.setFont(QFont("Courier New", 8, QFont.Bold))
        p.drawText(lx + 4, ly + LH + 14, f"x:{int(ix)}  y:{int(iy)}")

    def _draw_pen(self, p):
        """Pen tool drawing — 3 modes: free / circle-auto / rect-auto."""
        if self._image is None:
            return
        s  = self._scale
        ox = self._offset.x()
        oy = self._offset.y()

        def wx(ix): return float(ix * s + ox)
        def wy(iy): return float(iy * s + oy)

        # ── Mode 2: Reference overlay ─────────────────────────────────
        if self._pen_circle and isinstance(self._pen_circle, dict) and self._pen_points:
            try:
                import math as _m2
                # Always use centroid of actual points for reference circle
                _pts = self._pen_points
                _n   = len(_pts)
                _ecx = sum(pt["pt"][0] for pt in _pts) / _n
                _ecy = sum(pt["pt"][1] for pt in _pts) / _n
                _er  = sum(_m2.hypot(pt["pt"][0]-_ecx, pt["pt"][1]-_ecy)
                           for pt in _pts) / _n
                cx = int(wx(_ecx)); cy = int(wy(_ecy))
                r  = int(_er * s)
                p.setPen(QPen(QColor(100, 180, 255, 50), 1, Qt.DashLine))
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(cx-r, cy-r, r*2, r*2)
            except Exception:
                pass

        if self._pen_rect and isinstance(self._pen_rect, dict):
            try:
                rx = int(wx(self._pen_rect["x"]))
                ry = int(wy(self._pen_rect["y"]))
                rw = int(self._pen_rect["width"]  * s)
                rh = int(self._pen_rect["height"] * s)
                p.setPen(QPen(QColor(100, 180, 255, 50), 1, Qt.DashLine))
                p.setBrush(Qt.NoBrush)
                p.drawRect(rx, ry, rw, rh)
            except Exception:
                self._pen_rect = None

        pts = self._pen_points
        n   = len(pts)

        # ── Mode 1 empty: show crosshair cursor ───────────────────────
        if n == 0:
            if self._pen_hover and not (self._pen_circle or self._pen_rect):
                hx = int(wx(self._pen_hover[0])); hy = int(wy(self._pen_hover[1]))
                p.setPen(QPen(QColor(255,255,255,160), 1))
                p.drawLine(hx-14, hy, hx+14, hy)
                p.drawLine(hx, hy-14, hx, hy+14)
            return

        pts_f = [(wx(pt["pt"][0]), wy(pt["pt"][1])) for pt in pts]

        # ── Build path ────────────────────────────────────────────────
        from PyQt5.QtGui import QPainterPath as _QPP
        is_circle_mode = bool(getattr(self, '_pen_circle', None))

        is_closed = bool(self._pen_closed) or bool(
            getattr(self, '_pen_circle', None) and n >= 3) or bool(
            getattr(self, '_pen_rect', None) and n >= 3)

        path = _QPP()

        is_rect_mode = bool(getattr(self, '_pen_rect', None))

        if is_circle_mode and n >= 4:
            # ── Perfect circle — center+radius from actual point positions
            import math as _m
            # Center = centroid of all anchor points
            ecx = sum(pt["pt"][0] for pt in pts) / n
            ecy = sum(pt["pt"][1] for pt in pts) / n
            # Radius = average distance from centroid
            er  = sum(_m.hypot(pt["pt"][0]-ecx, pt["pt"][1]-ecy)
                      for pt in pts) / n
            if er < 1: er = 1
            path.addEllipse(wx(ecx) - er*s, wy(ecy) - er*s,
                             er*s*2, er*s*2)
            # Keep _pen_circle in sync
            if self._pen_circle:
                self._pen_circle = {**self._pen_circle,
                                    "x": ecx, "y": ecy, "radius": er}

        elif is_rect_mode and n >= 4:
            # ── Perfect rectangle through anchor points ───────────────
            xs = [pt["pt"][0] for pt in pts]
            ys = [pt["pt"][1] for pt in pts]
            rx1 = min(xs); ry1 = min(ys)
            rx2 = max(xs); ry2 = max(ys)
            path.moveTo(wx(rx1), wy(ry1))
            path.lineTo(wx(rx2), wy(ry1))
            path.lineTo(wx(rx2), wy(ry2))
            path.lineTo(wx(rx1), wy(ry2))
            path.closeSubpath()

        else:
            # ── Free pen — use anchor points as-is ────────────────────
            path.moveTo(pts_f[0][0], pts_f[0][1])
            loop_n = n if is_closed else n - 1
            for i in range(loop_n):
                curr = pts[i % n]
                nxt  = pts[(i + 1) % n]
                ax, ay = curr["pt"]; bx, by = nxt["pt"]
                ox1, oy1 = curr["cp_out"]
                ix2, iy2 = nxt["cp_in"]
                if ox1*ox1 + oy1*oy1 < 4 and ix2*ix2 + iy2*iy2 < 4:
                    path.lineTo(wx(bx), wy(by))
                else:
                    path.cubicTo(wx(ax+ox1), wy(ay+oy1),
                                  wx(bx+ix2), wy(by+iy2),
                                  wx(bx), wy(by))
            if is_closed and n >= 3:
                path.closeSubpath()

        # ── Dim outside closed path ───────────────────────────────────
        if is_closed and n >= 3:
            full = _QPP()
            full.addRect(0, 0, self.width(), self.height())
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(QColor(0, 0, 0, 55)))
            p.drawPath(full.subtracted(path))
            p.setBrush(Qt.NoBrush)

        # ── Path strokes ──────────────────────────────────────────────
        if n >= 2 or (n >= 1 and is_closed):
            if is_circle_mode:
                # Double gold ring (like image 2)
                # Outer ring — wider, lighter
                p.setPen(QPen(QColor(255,200,0,80), 12))
                p.setBrush(Qt.NoBrush)
                p.drawPath(path)
                # Mid ring
                p.setPen(QPen(QColor(200,140,0,120), 5))
                p.drawPath(path)
                # Inner ring — bright gold sharp
                p.setPen(QPen(QColor(255,210,0), 2))
                p.drawPath(path)
            else:
                # Rect / free mode — single gold border
                p.setPen(QPen(QColor(255,180,0,50), 8))
                p.setBrush(Qt.NoBrush)
                p.drawPath(path)
                p.setPen(QPen(QColor(0,0,0,80), 3))
                p.drawPath(path)
                p.setPen(QPen(QColor(255,195,0), 2))
                p.drawPath(path)

        # ── Preview line to cursor (Mode 1 open path) ─────────────────
        if not is_closed and self._pen_hover and n >= 1 and n < 3:
            lx, ly = pts_f[-1]
            hx2 = wx(self._pen_hover[0]); hy2 = wy(self._pen_hover[1])
            p.setPen(QPen(QColor(80,160,255,120), 1, Qt.DashLine))
            p.drawLine(int(lx), int(ly), int(hx2), int(hy2))

        # ── Bezier handles (hidden for circle mode — clean look) ────────
        if not is_circle_mode:
            for i, pt_data in enumerate(pts):
                ax, ay = pt_data["pt"]
                awx, awy = int(wx(ax)), int(wy(ay))
                for key in ("cp_in", "cp_out"):
                    dx, dy = pt_data[key]
                    if dx*dx + dy*dy < 4:
                        continue
                    hx3 = int(wx(ax+dx)); hy3 = int(wy(ay+dy))
                    p.setPen(QPen(QColor(160,160,160), 1))
                    p.drawLine(awx, awy, hx3, hy3)
                    is_act = (i == self._pen_drag_idx and self._pen_drag_part == key)
                    p.setPen(QPen(QColor(0,100,220), 1))
                    p.setBrush(QBrush(QColor(0,160,255) if is_act else QColor(220,240,255)))
                    p.drawEllipse(hx3-4, hy3-4, 8, 8)
                    p.setBrush(Qt.NoBrush)

        # ── Anchor points ─────────────────────────────────────────────
        HS = 5
        for i, (fx, fy) in enumerate(pts_f):
            awx, awy = int(fx), int(fy)
            is_drag  = (i == self._pen_drag_idx and self._pen_drag_part == "pt")
            is_first = (i == 0)

            # Close-path hover ring (Mode 1 only)
            is_hover_close = (is_first and not is_closed and n >= 3 and
                               self._pen_hover and not (self._pen_circle or self._pen_rect) and
                               ((wx(self._pen_hover[0])-fx)**2 +
                                (wy(self._pen_hover[1])-fy)**2) <= 20**2)

            if is_drag:
                p.setPen(QPen(QColor(0,80,200), 1.5))
                p.setBrush(QBrush(QColor(60,140,255)))
                p.drawRect(awx-HS, awy-HS, HS*2, HS*2)
            elif is_first and not is_closed:
                # Green circle = close path target (Mode 1 free)
                p.setPen(QPen(QColor(0,160,40), 1.5))
                p.setBrush(QBrush(QColor(255,255,80) if is_hover_close
                                   else QColor(0,220,80)))
                p.drawEllipse(awx-HS, awy-HS, HS*2, HS*2)
                if is_hover_close:
                    p.setPen(QPen(QColor(255,255,80), 2))
                    p.setBrush(Qt.NoBrush)
                    p.drawEllipse(awx-9, awy-9, 18, 18)
            elif is_circle_mode:
                # Circle mode — round handles with gold border (like image 2)
                p.setPen(QPen(QColor(180,120,0), 1.5))
                p.setBrush(QBrush(QColor(255,255,200)))
                p.drawEllipse(awx-HS, awy-HS, HS*2, HS*2)
            else:
                # Rect / free — white square anchor
                p.setPen(QPen(QColor(80,80,80), 1.5))
                p.setBrush(QBrush(QColor(255,255,255)))
                p.drawRect(awx-HS, awy-HS, HS*2, HS*2)
            p.setBrush(Qt.NoBrush)

        # ── Handle preview while dragging to create new point ─────────
        if self._pen_is_drawing and self._pen_new_pt and self._pen_hover:
            nx, ny   = self._pen_new_pt
            hx4, hy4 = self._pen_hover
            dx = hx4-nx; dy = hy4-ny
            if dx*dx + dy*dy > 25:   # only show when meaningfully dragged
                p.setPen(QPen(QColor(160,160,160), 1))
                p.drawLine(int(wx(nx)), int(wy(ny)), int(wx(hx4)), int(wy(hy4)))
                p.drawLine(int(wx(nx)), int(wy(ny)), int(wx(2*nx-hx4)), int(wy(2*ny-hy4)))
                p.setPen(QPen(QColor(0,100,220), 1))
                p.setBrush(QBrush(QColor(220,240,255)))
                p.drawEllipse(int(wx(hx4))-4, int(wy(hy4))-4, 8, 8)
                p.drawEllipse(int(wx(2*nx-hx4))-4, int(wy(2*ny-hy4))-4, 8, 8)
                p.setBrush(Qt.NoBrush)
    def _pen_remove_last(self):
        if self._pen_points:
            self._pen_points.pop()
            self._pen_drag_idx  = -1
            self._pen_drag_part = None
            self._render_overlay()

    def _hit_pen_rect(self, wp: QPoint):
        """Return drag zone for the pen square overlay."""
        if not self._pen_rect or not self._pen_mode:
            return None
        s  = self._scale
        ox = self._offset.x(); oy = self._offset.y()
        rx = int(self._pen_rect["x"]     * s) + ox
        ry = int(self._pen_rect["y"]     * s) + oy
        rw = int(self._pen_rect["width"] * s)
        rh = int(self._pen_rect["height"]* s)
        T  = 14   # hit tolerance px
        mx, my = wp.x(), wp.y()

        def near(ax, ay): return abs(mx-ax)<=T and abs(my-ay)<=T

        if near(rx,    ry):    return 'TL'
        if near(rx+rw, ry):    return 'TR'
        if near(rx+rw, ry+rh): return 'BR'
        if near(rx,    ry+rh): return 'BL'
        if abs(my-ry)   <= T and rx<=mx<=rx+rw: return 'T'
        if abs(my-ry-rh)<= T and rx<=mx<=rx+rw: return 'B'
        if abs(mx-rx)   <= T and ry<=my<=ry+rh: return 'L'
        if abs(mx-rx-rw)<= T and ry<=my<=ry+rh: return 'R'
        if rx<=mx<=rx+rw and ry<=my<=ry+rh:     return 'M'
        return None

    def _cursor_for_pen_zone(self, zone):
        return {
            'TL': Qt.SizeFDiagCursor, 'BR': Qt.SizeFDiagCursor,
            'TR': Qt.SizeBDiagCursor, 'BL': Qt.SizeBDiagCursor,
            'T':  Qt.SizeVerCursor,   'B':  Qt.SizeVerCursor,
            'L':  Qt.SizeHorCursor,   'R':  Qt.SizeHorCursor,
            'M':  Qt.SizeAllCursor,
        }.get(zone, Qt.CrossCursor)

    def _draw_patch(self, p):
        """Draw patch tool overlay — lasso selection + drag preview."""
        if self._image is None:
            return
        s  = self._scale
        ox = self._offset.x()
        oy = self._offset.y()

        def to_w(ix, iy):
            return QPoint(int(ix*s)+ox, int(iy*s)+oy)

        if self._patch_phase == "draw" and self._patch_lasso:
            # Draw freehand lasso (marching ants style)
            pts_w = [to_w(x, y) for x, y in self._patch_lasso]
            if len(pts_w) >= 2:
                from PyQt5.QtGui import QPolygon
                poly = QPolygon(pts_w)
                p.setPen(QPen(QColor(0,0,0,80), 2))
                p.setBrush(Qt.NoBrush)
                p.drawPolyline(poly)
                ant = QPen(QColor(255,200,0), 1, Qt.CustomDashLine)
                ant.setDashPattern([5,4])
                p.setPen(ant)
                p.drawPolyline(poly)

        elif self._patch_phase == "drag" and self._patch_src_pts:
            ox2, oy2 = self._patch_offset
            # Source selection (red dashed)
            src_w = [to_w(x, y) for x,y in self._patch_src_pts]
            if src_w:
                from PyQt5.QtGui import QPolygon
                sp = QPolygon(src_w)
                p.setPen(QPen(QColor(255,80,80,160), 1, Qt.DashLine))
                p.setBrush(QBrush(QColor(255,80,80,30)))
                p.drawPolygon(sp)

                # Destination (shifted) — green dashed
                dst_w = [to_w(x+ox2, y+oy2) for x,y in self._patch_src_pts]
                dp2 = QPolygon(dst_w)
                p.setPen(QPen(QColor(80,200,80,200), 2, Qt.DashLine))
                p.setBrush(QBrush(QColor(80,200,80,40)))
                p.drawPolygon(dp2)

                # Arrow from source center to dest center
                scx = sum(x for x,y in self._patch_src_pts)/len(self._patch_src_pts)
                scy = sum(y for x,y in self._patch_src_pts)/len(self._patch_src_pts)
                sw2 = to_w(scx, scy)
                dw2 = to_w(scx+ox2, scy+oy2)
                p.setPen(QPen(QColor(255,200,0), 2))
                p.drawLine(sw2, dw2)

    def _draw_qr(self, p, region):
        s = self._scale
        ox, oy = self._offset.x(), self._offset.y()
        bx, by, bw, bh = region["bbox"]
        p.setPen(QPen(QColor(220,50,50), 2))
        p.drawRect(QRect(int(bx*s)+ox, int(by*s)+oy,
                          int(bw*s), int(bh*s)))
        lbl = region.get("data") or region.get("type","")
        if lbl:
            p.setFont(QFont("Arial", 8))
            p.setPen(QColor(220,50,50))
            p.drawText(int(bx*s)+ox, int(by*s)+oy-4, lbl[:40])

    # ── Hit testing ───────────────────────────────────────────────────────

    def _hit(self, wp):
        if not self._crop_params or not self._crop_type:
            return _NONE
        if self._crop_type == "circle":
            return self._hit_circle(wp)
        return self._hit_rect(wp)

    def _hit_circle(self, wp):
        s  = self._scale
        ox, oy = self._offset.x(), self._offset.y()
        cx = int(self._crop_params["x"]      * s) + ox
        cy = int(self._crop_params["y"]      * s) + oy
        r  = int(self._crop_params["radius"] * s)
        tol = max(TOL, int(r*0.07))

        for zone, (hx,hy) in [(_C_N,(cx,cy-r)),(_C_E,(cx+r,cy)),
                                (_C_S,(cx,cy+r)),(_C_W,(cx-r,cy))]:
            if (wp.x()-hx)**2+(wp.y()-hy)**2 <= tol**2:
                return zone
        dist = ((wp.x()-cx)**2+(wp.y()-cy)**2)**0.5
        if dist <= r-tol: return _INSIDE
        if dist <= r+tol: return _EDGE
        return _NONE

    def _hit_rect(self, wp):
        s  = self._scale
        ox, oy = self._offset.x(), self._offset.y()
        params = self._crop_params
        x  = int(params["x"]     * s) + ox
        y  = int(params["y"]     * s) + oy
        w  = int(params["width"] * s)
        h  = int(params["height"]* s)
        angle = params.get("angle", 0.0)
        cx, cy = x+w//2, y+h//2
        tol = TOL

        # Un-rotate the mouse point for hit testing
        import math as _m
        rad  = _m.radians(-angle)
        wpx  = wp.x() - cx;  wpy = wp.y() - cy
        lx   = wpx*_m.cos(rad) - wpy*_m.sin(rad) + cx
        ly   = wpx*_m.sin(rad) + wpy*_m.cos(rad) + cy
        lwp  = QPoint(int(lx), int(ly))

        # Rotation handle (check in screen space)
        rad2  = _m.radians(angle)
        tlx   = cx + (x-cx)*_m.cos(rad2) - (y-cy)*_m.sin(rad2)
        tly   = cy + (x-cx)*_m.sin(rad2) + (y-cy)*_m.cos(rad2)
        rot_hx = int(tlx - 20*_m.sin(rad2))
        rot_hy = int(tly - 20*_m.cos(rad2))
        if (wp.x()-rot_hx)**2+(wp.y()-rot_hy)**2 <= (tol+4)**2:
            return _R_ROT

        # Corners (in local/unrotated space)
        corners = {_R_TL:(x,y),_R_TR:(x+w,y),_R_BR:(x+w,y+h),_R_BL:(x,y+h)}
        for zone,(hx,hy) in corners.items():
            if (lwp.x()-hx)**2+(lwp.y()-hy)**2 <= tol**2:
                return zone

        # Mid edges
        mcx, mcy = x+w//2, y+h//2
        mids = {_R_T:(mcx,y),_R_R:(x+w,mcy),_R_B:(mcx,y+h),_R_L:(x,mcy)}
        for zone,(hx,hy) in mids.items():
            if (lwp.x()-hx)**2+(lwp.y()-hy)**2 <= tol**2:
                return zone

        # Inside
        if x<=lwp.x()<=x+w and y<=lwp.y()<=y+h:
            return _INSIDE
        return _NONE

    # ── Mouse events ─────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        self.setFocus(Qt.MouseFocusReason)

        # ── Patch Tool ────────────────────────────────────────────────
        if self._patch_mode and event.button() == Qt.LeftButton:
            ix = (event.x() - self._offset.x()) / self._scale
            iy = (event.y() - self._offset.y()) / self._scale
            if self._patch_phase == "draw":
                self._patch_lasso   = [(ix, iy)]
                self._patch_src_pts = []
            elif self._patch_phase == "drag":
                self._patch_drag_start = event.pos()
                self._patch_offset  = (0, 0)
            event.accept(); return

        # ── Middle button OR Space+Left = Pan only ───────────────────
        if event.button() == Qt.MidButton or (
                self._space_held and event.button() == Qt.LeftButton):
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept(); return

        # ── Pen tool ──────────────────────────────────────────────────
        if self._pen_mode and event.button() == Qt.LeftButton:
            ix = (event.x() - self._offset.x()) / self._scale
            iy = (event.y() - self._offset.y()) / self._scale

            # Hit existing anchor point → drag it
            for i, pt_data in enumerate(self._pen_points):
                ax, ay = pt_data["pt"]
                awx = int(ax*self._scale) + self._offset.x()
                awy = int(ay*self._scale) + self._offset.y()
                if ((event.x()-awx)**2+(event.y()-awy)**2) <= 12**2:
                    self._pen_drag_idx  = i
                    self._pen_drag_part = "pt"
                    event.accept(); return
                # Hit cp_out/cp_in handle
                for part in ("cp_in", "cp_out"):
                    dx, dy = pt_data[part]
                    if dx == 0 and dy == 0: continue
                    hx2 = int((ax+dx)*self._scale)+self._offset.x()
                    hy2 = int((ay+dy)*self._scale)+self._offset.y()
                    if ((event.x()-hx2)**2+(event.y()-hy2)**2) <= 10**2:
                        self._pen_drag_idx  = i
                        self._pen_drag_part = part
                        event.accept(); return

            # Close path on first point
            if len(self._pen_points) >= 3:
                ax0, ay0 = self._pen_points[0]["pt"]
                p0wx = int(ax0*self._scale)+self._offset.x()
                p0wy = int(ay0*self._scale)+self._offset.y()
                if ((event.x()-p0wx)**2+(event.y()-p0wy)**2) <= 18**2:
                    self.pen_closed.emit()
                    event.accept(); return

            # Click inside polygon → move whole shape (both circle and rect)
            if len(self._pen_points) >= 3:
                pts_xy = [pt["pt"] for pt in self._pen_points]
                import cv2 as _cv2, numpy as _np
                poly = _np.array([[int(x*self._scale+self._offset.x()),
                                    int(y*self._scale+self._offset.y())]
                                   for x,y in pts_xy], dtype=_np.int32)
                if _cv2.pointPolygonTest(poly, (float(event.x()), float(event.y())), False) >= 0:
                    self._pen_drag_idx  = -99   # special: move all
                    self._pen_drag_part = "move_all"
                    self._pen_drag_start_img = (ix, iy)
                    self._pen_orig_pts = [(pt["pt"], pt["cp_in"], pt["cp_out"])
                                          for pt in self._pen_points]
                    event.accept(); return

            # New point — start dragging handle
            self._pen_new_pt     = (ix, iy)
            self._pen_is_drawing = True
            event.accept(); return

        if self._pen_mode and event.button() == Qt.RightButton:
            if not self._pen_points:
                event.accept(); return
            # Context menu — Photoshop style
            from PyQt5.QtWidgets import QMenu
            menu = QMenu(self)
            menu.setStyleSheet(
                "QMenu{background:#1c2b45;color:#e8edf5;border:1px solid #3e6188;}"
                "QMenu::item{padding:6px 24px;}"
                "QMenu::item:selected{background:#3e6188;}")
            if len(self._pen_points) >= 3:
                act_apply = menu.addAction("✂  Make Selection & Apply Crop")
                act_apply.triggered.connect(self.pen_closed.emit)
                menu.addSeparator()
                act_edit = menu.addAction("✏  Edit Text in Region")
                act_edit.triggered.connect(self.pen_edit_region.emit)
                act_qr   = menu.addAction("⊞  Scan QR in Region")
                act_qr.triggered.connect(self.pen_scan_qr_region.emit)
                menu.addSeparator()
            act_del = menu.addAction("⌫  Delete Last Point")
            act_del.triggered.connect(lambda: self._pen_remove_last())
            menu.addSeparator()
            act_cancel = menu.addAction("✕  Cancel Pen Tool")
            act_cancel.triggered.connect(lambda: (
                setattr(self, '_pen_mode', False),
                setattr(self, '_pen_points', []),
                self.setCursor(Qt.ArrowCursor),
                self._render_overlay()))
            menu.exec_(event.globalPos())
            event.accept(); return

        if event.button() == Qt.LeftButton and self._crop_type:
            zone = self._hit(event.pos())
            if zone != _NONE:
                self._drag_zone  = zone
                self._drag_start = event.pos()
                self._drag_orig  = self._crop_params.copy()
                event.accept(); return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._crop_type or self._pen_mode or self._patch_mode:
            self._zoom_pos = event.pos()

        # ── Patch Tool move ───────────────────────────────────────────
        if self._patch_mode and event.buttons() & Qt.LeftButton:
            ix = (event.x() - self._offset.x()) / self._scale
            iy = (event.y() - self._offset.y()) / self._scale
            if self._patch_phase == "draw":
                self._patch_lasso.append((ix, iy))
                self._render_overlay()
                event.accept(); return
            elif self._patch_phase == "drag" and self._patch_drag_start:
                dp = event.pos() - self._patch_drag_start
                self._patch_offset = (dp.x()/self._scale, dp.y()/self._scale)
                self._render_overlay()
                event.accept(); return

        # ── Pan (Middle button OR Space held) — NOT plain left drag ───
        if self._pan_start is not None and (
                event.buttons() & Qt.MidButton or self._space_held):
            dp = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self._pan_offset += dp
            self._invalidate_cache()
            self._render_overlay()
            event.accept(); return

        # ── Pen tool ─────────────────────────────────────────────────
        if self._pen_mode:
            ix = (event.x() - self._offset.x()) / self._scale
            iy = (event.y() - self._offset.y()) / self._scale
            self._pen_hover = (ix, iy)

            # Drawing new point + dragging handle
            if self._pen_is_drawing and self._pen_new_pt:
                self._render_overlay()
                event.accept(); return

            # Dragging existing anchor or handle
            if self._pen_drag_idx == -99 and self._pen_drag_part == "move_all":
                # Move entire pen shape
                sx, sy = self._pen_drag_start_img
                ddx = ix - sx; ddy = iy - sy
                for i, (orig_pt, orig_in, orig_out) in enumerate(self._pen_orig_pts):
                    self._pen_points[i]["pt"]     = (orig_pt[0]+ddx, orig_pt[1]+ddy)
                    self._pen_points[i]["cp_in"]  = orig_in
                    self._pen_points[i]["cp_out"] = orig_out
                # Sync circle/rect center when moving
                if self._pen_circle and isinstance(self._pen_circle, dict):
                    import math as _mm
                    _pts = self._pen_points
                    _n   = len(_pts)
                    _ecx = sum(pt["pt"][0] for pt in _pts) / _n
                    _ecy = sum(pt["pt"][1] for pt in _pts) / _n
                    self._pen_circle = {**self._pen_circle, "x": _ecx, "y": _ecy}
                if self._pen_rect and isinstance(self._pen_rect, dict):
                    _pts = self._pen_points
                    _xs  = [p["pt"][0] for p in _pts]
                    _ys  = [p["pt"][1] for p in _pts]
                    self._pen_rect = {**self._pen_rect,
                                      "x": min(_xs), "y": min(_ys),
                                      "width":  max(_xs)-min(_xs),
                                      "height": max(_ys)-min(_ys)}
                self._render_overlay()
                event.accept(); return
            if self._pen_drag_idx >= 0 and self._pen_drag_part:
                pt_data = self._pen_points[self._pen_drag_idx]
                ax, ay  = pt_data["pt"]
                if self._pen_drag_part == "pt":
                    if bool(self._pen_circle):
                        # Circle: move only this point, update tangent handle
                        import math as _mc
                        n_pts  = len(self._pen_points)
                        pt_data["pt"] = (ix, iy)
                        _cx = sum(p["pt"][0] for p in self._pen_points) / n_pts
                        _cy = sum(p["pt"][1] for p in self._pen_points) / n_pts
                        _angle = _mc.atan2(iy - _cy, ix - _cx)
                        _r     = _mc.hypot(ix - _cx, iy - _cy)
                        _h     = _r * (4/3) * _mc.tan(_mc.pi / 16)
                        tx = -_mc.sin(_angle); ty = _mc.cos(_angle)
                        pt_data["cp_out"] = ( tx * _h,  ty * _h)
                        pt_data["cp_in"]  = (-tx * _h, -ty * _h)
                    elif bool(self._pen_rect):
                        # Rect: move only this point, update rect reference
                        pt_data["pt"] = (ix, iy)
                        _pts2 = self._pen_points
                        _xs2  = [p["pt"][0] for p in _pts2]
                        _ys2  = [p["pt"][1] for p in _pts2]
                        if self._pen_rect:
                            self._pen_rect = {**self._pen_rect,
                                "x": min(_xs2), "y": min(_ys2),
                                "width":  max(_xs2)-min(_xs2),
                                "height": max(_ys2)-min(_ys2)}
                    else:
                        pt_data["pt"] = (ix, iy)
                elif self._pen_drag_part == "cp_out":
                    pt_data["cp_out"] = (ix-ax, iy-ay)
                    pt_data["cp_in"]  = (-(ix-ax), -(iy-ay))
                elif self._pen_drag_part == "cp_in":
                    pt_data["cp_in"] = (ix-ax, iy-ay)
                    pt_data["cp_out"]= (-(ix-ax), -(iy-ay))
                self._render_overlay()
                event.accept(); return

            # Cursor feedback
            cur = Qt.CrossCursor
            for pt_data in self._pen_points:
                ax, ay = pt_data["pt"]
                awx = int(ax*self._scale)+self._offset.x()
                awy = int(ay*self._scale)+self._offset.y()
                if ((event.x()-awx)**2+(event.y()-awy)**2) <= 12**2:
                    cur = Qt.PointingHandCursor; break
            self.setCursor(cur)
            self._render_overlay()
            event.accept(); return

        # ── Crop drag ─────────────────────────────────────────────────
        if self._drag_zone != _NONE:
            dp = event.pos() - self._drag_start
            dx = int(dp.x() / self._scale)
            dy = int(dp.y() / self._scale)
            if self._crop_type == "circle":
                self._move_circle(dx, dy)
            else:
                self._move_rect(dx, dy)
            self._render_overlay()
            event.accept()
        else:
            if self._space_held:
                self.setCursor(Qt.OpenHandCursor)
            else:
                self._set_cursor(event.pos())
            if self._crop_type:
                self._render_overlay()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        # ── Patch Tool release ────────────────────────────────────────
        if self._patch_mode and event.button() == Qt.LeftButton:
            if self._patch_phase == "draw" and len(self._patch_lasso) >= 3:
                # Close lasso → switch to drag phase
                self._patch_src_pts = list(self._patch_lasso)
                self._patch_lasso   = []
                self._patch_phase   = "drag"
                self._patch_offset  = (0, 0)
                self.setCursor(Qt.SizeAllCursor)
                self._render_overlay()
                self.patch_phase_changed.emit()   # notify main_window
            elif self._patch_phase == "drag":
                self._patch_drag_start = None
            event.accept(); return
        if self._pan_start is not None and (
                event.button() == Qt.MidButton or self._space_held):
            self._pan_start = None
            self.setCursor(Qt.OpenHandCursor if self._space_held else Qt.ArrowCursor)
            event.accept(); return

        if self._pen_mode and event.button() == Qt.LeftButton:
            ix = (event.x() - self._offset.x()) / self._scale
            iy = (event.y() - self._offset.y()) / self._scale

            if self._pen_is_drawing and self._pen_new_pt:
                # Finalise new point — handle = drag distance
                nx, ny = self._pen_new_pt
                cp_out = (ix - nx, iy - ny)
                cp_in  = (-(ix - nx), -(iy - ny))
                self._pen_points.append({
                    "pt": (nx, ny),
                    "cp_out": cp_out,
                    "cp_in":  cp_in,
                })
                self._pen_is_drawing = False
                self._pen_new_pt     = None

            self._pen_drag_idx  = -1
            self._pen_drag_part = None
            self._render_overlay()
            event.accept(); return

        if event.button() == Qt.LeftButton and self._drag_zone != _NONE:
            self._drag_zone = _NONE
            self._set_cursor(event.pos())
            event.accept(); return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        mod   = event.modifiers()
        delta = event.angleDelta().y()
        if delta == 0:
            super().wheelEvent(event); return

        # ── Pen mode: scroll = resize circle / expand free pts ──────
        if self._pen_mode and self._pen_points:
            if bool(self._pen_circle):
                # Circle mode — scroll resizes circle (all pts radially)
                import math
                pts  = [pt["pt"] for pt in self._pen_points]
                cx   = sum(x for x,y in pts) / len(pts)
                cy   = sum(y for x,y in pts) / len(pts)
                step = 3.0 if delta > 0 else -3.0
                new_pts = []
                for pt_data in self._pen_points:
                    ax, ay = pt_data["pt"]
                    dx2 = ax - cx; dy2 = ay - cy
                    dist = math.hypot(dx2, dy2)
                    if dist > 0:
                        factor = (dist + step) / dist
                        new_x = cx + dx2 * factor
                        new_y = cy + dy2 * factor
                        # Scale bezier handles proportionally too
                        hf = factor
                        new_pts.append({
                            "pt":     (new_x, new_y),
                            "cp_in":  (pt_data["cp_in"][0]*hf,
                                       pt_data["cp_in"][1]*hf),
                            "cp_out": (pt_data["cp_out"][0]*hf,
                                       pt_data["cp_out"][1]*hf),
                        })
                    else:
                        new_pts.append(pt_data)
                self._pen_points = new_pts
                # Update reference circle radius
                if self._pen_circle and new_pts:
                    new_r = math.hypot(
                        new_pts[0]["pt"][0] - cx,
                        new_pts[0]["pt"][1] - cy)
                    self._pen_circle = {**self._pen_circle,
                                        "radius": new_r,
                                        "x": cx, "y": cy}
                self._render_overlay()
                event.accept(); return
            elif bool(self._pen_rect):
                # Rect mode — scroll resizes (expand/contract around center)
                _pts = self._pen_points
                _n   = len(_pts)
                _cx2 = sum(p["pt"][0] for p in _pts) / _n
                _cy2 = sum(p["pt"][1] for p in _pts) / _n
                step = 3.0 if delta > 0 else -3.0
                new_pts = []
                for pt_data in _pts:
                    ax2, ay2 = pt_data["pt"]
                    dx2 = ax2 - _cx2; dy2 = ay2 - _cy2
                    import math as _mr
                    dist2 = _mr.hypot(dx2, dy2)
                    if dist2 > 0:
                        factor2 = (dist2 + step) / dist2
                        new_pts.append({**pt_data,
                            "pt": (_cx2 + dx2*factor2, _cy2 + dy2*factor2)})
                    else:
                        new_pts.append(pt_data)
                self._pen_points = new_pts
                # Update rect reference
                _xs = [p["pt"][0] for p in new_pts]
                _ys = [p["pt"][1] for p in new_pts]
                if self._pen_rect:
                    self._pen_rect = {**self._pen_rect,
                                      "x": min(_xs), "y": min(_ys),
                                      "width":  max(_xs)-min(_xs),
                                      "height": max(_ys)-min(_ys)}
                self._render_overlay()
                event.accept(); return
            import math
            pts = [pt["pt"] for pt in self._pen_points]
            cx = sum(x for x,y in pts) / len(pts)
            cy = sum(y for x,y in pts) / len(pts)
            step = 3.0 if delta > 0 else -3.0
            new_pts = []
            for pt_data in self._pen_points:
                ax, ay = pt_data["pt"]
                dx2 = ax - cx; dy2 = ay - cy
                dist = math.hypot(dx2, dy2)
                if dist > 0:
                    factor = (dist + step) / dist
                    new_pts.append({
                        **pt_data,
                        "pt": (cx + dx2*factor, cy + dy2*factor)
                    })
                else:
                    new_pts.append(pt_data)
            self._pen_points = new_pts
            self._render_overlay()
            event.accept(); return

        # ── Normal: no crop → canvas zoom ────────────────────────────
        over_crop = (self._crop_type and self._crop_params
                     and self._hit(event.pos()) != _NONE)

        if not over_crop:
            STEP      = 1.2
            factor    = STEP if delta > 0 else 1/STEP
            new_zoom  = max(0.1, min(20.0, self._zoom_factor * factor))
            if new_zoom == self._zoom_factor:
                event.accept(); return

            # Zoom toward cursor
            if hasattr(self, '_img_w'):
                mx, my   = event.x(), event.y()
                old_s    = self._fit_scale * self._zoom_factor
                new_s    = self._fit_scale * new_zoom
                ix_cur   = (mx - self._offset.x()) / old_s
                iy_cur   = (my - self._offset.y()) / old_s
                new_sw   = int(self._img_w * new_s)
                new_sh   = int(self._img_h * new_s)
                base_ox  = (self.width()  - new_sw) // 2
                base_oy  = (self.height() - new_sh) // 2
                self._pan_offset = QPoint(
                    int(mx - ix_cur * new_s - base_ox),
                    int(my - iy_cur * new_s - base_oy)
                )
            self._zoom_factor = new_zoom
            self._invalidate_cache()
            self._render_overlay()
            event.accept(); return

        # ── Wheel over crop handle = resize crop ─────────────────────
        step = self.WHEEL_STEP if delta > 0 else -self.WHEEL_STEP
        if self._crop_type == "circle":
            r = self._crop_params.get("radius", 20)
            self._crop_params["radius"] = max(20, r+step)
        else:
            self._crop_params["width"]  = max(20, self._crop_params.get("width",20)+step)
            self._crop_params["height"] = max(20, self._crop_params.get("height",20)+step)
        self._render_overlay()
        event.accept()

        super().wheelEvent(event)

    # ── Drag logic ───────────────────────────────────────────────────────

    def _move_circle(self, dx, dy):
        orig  = self._drag_orig
        zone  = self._drag_zone
        ox_i, oy_i = orig["x"], orig["y"]
        r           = orig["radius"]

        if zone == _INSIDE:
            self._crop_params["x"] = ox_i+dx
            self._crop_params["y"] = oy_i+dy
        elif zone == _C_N:
            new_r = max(20, r-dy)
            self._crop_params["radius"] = new_r
            self._crop_params["y"]      = (oy_i+r) - new_r
        elif zone == _C_S:
            new_r = max(20, r+dy)
            self._crop_params["radius"] = new_r
            self._crop_params["y"]      = (oy_i-r) + new_r
        elif zone == _C_E:
            new_r = max(20, r+dx)
            self._crop_params["radius"] = new_r
            self._crop_params["x"]      = (ox_i-r) + new_r
        elif zone == _C_W:
            new_r = max(20, r-dx)
            self._crop_params["radius"] = new_r
            self._crop_params["x"]      = (ox_i+r) - new_r
        elif zone == _EDGE:
            cx_w = int(ox_i*self._scale)+self._offset.x()
            cy_w = int(oy_i*self._scale)+self._offset.y()
            cur  = self._drag_start + QPoint(int(dx*self._scale),
                                              int(dy*self._scale))
            d0 = ((self._drag_start.x()-cx_w)**2+
                  (self._drag_start.y()-cy_w)**2)**0.5
            d1 = ((cur.x()-cx_w)**2+(cur.y()-cy_w)**2)**0.5
            self._crop_params["radius"] = max(20, int(r+(d1-d0)))

    def _move_rect(self, dx, dy):
        orig  = self._drag_orig
        zone  = self._drag_zone
        x, y  = orig["x"], orig["y"]
        w, h  = orig["width"], orig["height"]
        angle = orig.get("angle", 0.0)

        if zone == _INSIDE:
            self._crop_params["x"] = x+dx
            self._crop_params["y"] = y+dy

        elif zone == _R_TL:
            self._crop_params.update({
                "x": x+dx, "y": y+dy,
                "width":  max(20, w-dx),
                "height": max(20, h-dy)})
        elif zone == _R_TR:
            self._crop_params.update({
                "y": y+dy,
                "width":  max(20, w+dx),
                "height": max(20, h-dy)})
        elif zone == _R_BR:
            self._crop_params.update({
                "width":  max(20, w+dx),
                "height": max(20, h+dy)})
        elif zone == _R_BL:
            self._crop_params.update({
                "x": x+dx,
                "width":  max(20, w-dx),
                "height": max(20, h+dy)})

        elif zone == _R_T:
            self._crop_params.update({"y": y+dy, "height": max(20, h-dy)})
        elif zone == _R_B:
            self._crop_params["height"] = max(20, h+dy)
        elif zone == _R_L:
            self._crop_params.update({"x": x+dx, "width": max(20, w-dx)})
        elif zone == _R_R:
            self._crop_params["width"] = max(20, w+dx)

        elif zone == _R_ROT:
            import math as _m
            cx_r = x + w//2;  cy_r = y + h//2
            sx = (self._drag_start.x()-self._offset.x())/self._scale
            sy = (self._drag_start.y()-self._offset.y())/self._scale
            cur_x = sx+dx;    cur_y = sy+dy
            a0 = _m.degrees(_m.atan2(sy-cy_r,    sx-cx_r))
            a1 = _m.degrees(_m.atan2(cur_y-cy_r, cur_x-cx_r))
            new_a = (angle+(a1-a0)) % 360
            if new_a > 180: new_a -= 360
            self._crop_params["angle"] = round(new_a, 1)

    # ── Cursor ───────────────────────────────────────────────────────────

    def _set_cursor(self, wp):
        if not self._crop_type:
            self.setCursor(Qt.ArrowCursor); return
        zone = self._hit(wp)
        cur = {
            _INSIDE: Qt.SizeAllCursor,
            _EDGE:   Qt.SizeFDiagCursor,
            _C_N: Qt.SizeVerCursor, _C_S: Qt.SizeVerCursor,
            _C_E: Qt.SizeHorCursor, _C_W: Qt.SizeHorCursor,
            _R_TL: Qt.SizeFDiagCursor, _R_BR: Qt.SizeFDiagCursor,
            _R_TR: Qt.SizeBDiagCursor, _R_BL: Qt.SizeBDiagCursor,
            _R_T: Qt.SizeVerCursor,  _R_B: Qt.SizeVerCursor,
            _R_L: Qt.SizeHorCursor,  _R_R: Qt.SizeHorCursor,
            _R_ROT: Qt.CrossCursor,
        }
        self.setCursor(cur.get(zone, Qt.ArrowCursor))

    # ── Keyboard ─────────────────────────────────────────────────────────

    def keyPressEvent(self, event):
        key = event.key()
        mod = event.modifiers()

        # Space = pan mode (Photoshop)
        if key == Qt.Key_Space and not event.isAutoRepeat():
            self._space_held = True
            self.setCursor(Qt.OpenHandCursor)
            event.accept(); return

        # Patch tool keys
        if self._patch_mode:
            if key == Qt.Key_Return or key == Qt.Key_Enter:
                self.patch_apply.emit()
                event.accept(); return
            if key == Qt.Key_Escape:
                self.stop_patch_tool()
                self.patch_cancelled.emit()
                event.accept(); return

        # Pen tool keys
        if self._pen_mode:
            if key == Qt.Key_Escape:
                # Stop pen mode but keep crop overlay intact
                self._pen_mode     = False
                self._pen_points   = []
                self._pen_hover    = None
                self._pen_drag_idx = -1
                self._pen_rect     = None
                self._pen_circle   = None
                self.setCursor(Qt.ArrowCursor)
                self._render_overlay()
                event.accept(); return
            if key in (Qt.Key_Return, Qt.Key_Enter):
                if len(self._pen_points) >= 3:
                    self.pen_closed.emit()
                event.accept(); return
            if key == Qt.Key_Backspace or key == Qt.Key_Delete:
                if self._pen_points:
                    self._pen_points.pop()
                    self._render_overlay()
                event.accept(); return
            super().keyPressEvent(event); return

        if key == Qt.Key_Left  and mod & Qt.ControlModifier:
            self.rotate_ccw_requested.emit(); event.accept(); return
        if key == Qt.Key_Right and mod & Qt.ControlModifier:
            self.rotate_cw_requested.emit();  event.accept(); return
        if key == Qt.Key_Left:
            self.crop_move.emit(-self.KEY_MOVE, 0); event.accept(); return
        if key == Qt.Key_Right:
            self.crop_move.emit( self.KEY_MOVE, 0); event.accept(); return
        if key == Qt.Key_Up:
            self.crop_move.emit(0, -self.KEY_MOVE); event.accept(); return
        if key == Qt.Key_Down:
            self.crop_move.emit(0,  self.KEY_MOVE); event.accept(); return
        if key in (Qt.Key_Plus,Qt.Key_Equal) or event.text()=='+':
            self.crop_resize.emit( self.KEY_RESIZE); event.accept(); return
        if key == Qt.Key_Minus or event.text()=='-':
            self.crop_resize.emit(-self.KEY_RESIZE); event.accept(); return
        if key in (Qt.Key_Return, Qt.Key_Enter):
            self.crop_confirmed.emit(); event.accept(); return
        super().keyPressEvent(event)

    # ── Resize ───────────────────────────────────────────────────────────

    def leaveEvent(self, event):
        self._zoom_pos = None
        if self._crop_type:
            self._render_overlay()
        super().leaveEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_held = False
            self._pan_start  = None
            self._set_cursor(self.mapFromGlobal(
                self.cursor().pos()))
            event.accept(); return
        super().keyReleaseEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._invalidate_cache()
        self._render_overlay()