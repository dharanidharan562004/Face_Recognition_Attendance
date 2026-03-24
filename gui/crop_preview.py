"""
crop_preview.py
Crop position preview widget — fills the panel width, centered display.
Green handles = edge aligned. Red handles = needs adjustment.
"""

import cv2
import numpy as np
from PyQt5.QtCore    import Qt, QPoint, QRect
from PyQt5.QtGui     import (QPixmap, QImage, QPainter, QPen,
                              QColor, QFont, QBrush)
from PyQt5.QtWidgets import QWidget, QSizePolicy

EDGE_THRESH = 22


class CropPreviewWidget(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        # Fill available width, fixed height
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFixedHeight(120)
        self._image       = None
        self._crop_type   = None
        self._crop_params = {}
        self._pixmap      = None
        self.setToolTip("Green = edge aligned  |  Red = needs adjustment")

    # ------------------------------------------------------------------ #
    #  Public                                                              #
    # ------------------------------------------------------------------ #

    def update_preview(self, image, crop_type, crop_params, active_pt=None):
        self._image       = image
        self._crop_type   = crop_type
        self._crop_params = crop_params.copy() if crop_params else {}
        self._rebuild()
        self.update()

    def clear(self):
        self._image       = None
        self._crop_type   = None
        self._crop_params = {}
        self._pixmap      = None
        self.update()

    # ------------------------------------------------------------------ #
    #  Build                                                               #
    # ------------------------------------------------------------------ #

    def _rebuild(self):
        if self._image is None:
            self._pixmap = None
            return

        pw = max(self.width(), 200)
        ph = self.height()

        ih, iw = self._image.shape[:2]
        scale  = min((pw - 8) / iw, (ph - 8) / ih)
        nw     = max(1, int(iw * scale))
        nh     = max(1, int(ih * scale))
        small  = cv2.resize(self._image, (nw, nh), interpolation=cv2.INTER_AREA)
        rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        qimg   = QImage(rgb.data, nw, nh, nw * 3, QImage.Format_RGB888)

        canvas = QPixmap(pw, ph)
        canvas.fill(QColor("#e8ecf4"))

        p  = QPainter(canvas)
        p.setRenderHint(QPainter.Antialiasing)

        # Centre image
        ox = (pw - nw) // 2
        oy = (ph - nh) // 2
        p.drawPixmap(ox, oy, QPixmap.fromImage(qimg))

        # Subtle border around image
        p.setPen(QPen(QColor("#9aaac8"), 1))
        p.drawRect(ox, oy, nw, nh)

        # 4 grid lines (2H + 2V)
        gpen = QPen(QColor(0, 140, 220, 100), 1, Qt.DashLine)
        p.setPen(gpen)
        p.drawLine(ox, oy + nh//3,    ox + nw, oy + nh//3)
        p.drawLine(ox, oy + 2*nh//3,  ox + nw, oy + 2*nh//3)
        p.drawLine(ox + nw//3,   oy,  ox + nw//3,   oy + nh)
        p.drawLine(ox + 2*nw//3, oy,  ox + 2*nw//3, oy + nh)

        # Edge map for accuracy
        gray_s = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) \
                 if len(small.shape) == 3 else small
        edges  = cv2.Canny(cv2.GaussianBlur(gray_s, (3,3), 1), 30, 90)

        if self._crop_type and self._crop_params:
            if self._crop_type == "circle":
                self._draw_circle(p, scale, ox, oy, edges, nw, nh)
            elif self._crop_type == "rectangle":
                self._draw_rect(p, scale, ox, oy, edges, nw, nh)

        p.end()
        self._pixmap = canvas

    # ------------------------------------------------------------------ #
    #  Accuracy check                                                      #
    # ------------------------------------------------------------------ #

    def _edge_ok(self, edges, px, py, sample=7) -> bool:
        h, w = edges.shape
        x1 = max(0, px - sample);  x2 = min(w, px + sample)
        y1 = max(0, py - sample);  y2 = min(h, py + sample)
        region = edges[y1:y2, x1:x2]
        return region.size > 0 and float(region.mean()) > EDGE_THRESH

    def _pt_color(self, ok):
        return QColor(30, 190, 70) if ok else QColor(220, 40, 40)

    # ------------------------------------------------------------------ #
    #  Circle                                                              #
    # ------------------------------------------------------------------ #

    def _draw_circle(self, p, scale, ox, oy, edges, nw, nh):
        params = self._crop_params
        cx = int(params["x"]      * scale) + ox
        cy = int(params["y"]      * scale) + oy
        r  = int(params["radius"] * scale)
        r  = max(4, min(r, min(nw, nh) // 2))

        handles = {"N":(cx,cy-r), "S":(cx,cy+r),
                   "E":(cx+r,cy), "W":(cx-r,cy)}
        acc = {l: self._edge_ok(edges, hx-ox, hy-oy)
               for l,(hx,hy) in handles.items()}
        good = sum(acc.values())

        ring_col = (QColor(30,190,70) if good>=3
                    else QColor(255,140,0) if good==2
                    else QColor(220,40,40))

        # Glow
        p.setPen(QPen(QColor(ring_col.red(), ring_col.green(),
                             ring_col.blue(), 40), 8))
        p.drawEllipse(QPoint(cx,cy), r, r)
        # Ring
        p.setPen(QPen(ring_col, 2))
        p.drawEllipse(QPoint(cx,cy), r, r)
        # Centre cross
        p.setPen(QPen(QColor(255,255,255,180), 1))
        p.drawLine(cx-6,cy,cx+6,cy)
        p.drawLine(cx,cy-6,cx,cy+6)

        # Handles
        offsets = {"N":(3,-10),"S":(3,13),"E":(9,4),"W":(-14,4)}
        for lbl,(hx,hy) in handles.items():
            col = self._pt_color(acc[lbl])
            p.setPen(QPen(QColor(255,255,255),1)); p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPoint(hx,hy), 6, 6)
            p.setPen(Qt.NoPen); p.setBrush(QBrush(col))
            p.drawEllipse(QPoint(hx,hy), 5, 5)
            p.setBrush(Qt.NoBrush)
            dx,dy = offsets[lbl]
            p.setFont(QFont("Arial",7,QFont.Bold)); p.setPen(col)
            p.drawText(hx+dx, hy+dy, lbl)

        self._badge(p, good, 4)

    # ------------------------------------------------------------------ #
    #  Rectangle                                                           #
    # ------------------------------------------------------------------ #

    def _draw_rect(self, p, scale, ox, oy, edges, nw, nh):
        params = self._crop_params
        x  = int(params["x"]      * scale) + ox
        y  = int(params["y"]      * scale) + oy
        w  = int(params["width"]  * scale)
        h  = int(params["height"] * scale)

        corners = {"TL":(x,y),"TR":(x+w,y),
                   "BR":(x+w,y+h),"BL":(x,y+h)}
        acc = {l: self._edge_ok(edges, hx-ox, hy-oy)
               for l,(hx,hy) in corners.items()}
        good = sum(acc.values())

        rect_col = (QColor(30,190,70) if good>=3
                    else QColor(255,140,0) if good==2
                    else QColor(220,40,40))

        p.setPen(QPen(QColor(rect_col.red(),rect_col.green(),
                             rect_col.blue(),40), 8))
        p.drawRect(QRect(x,y,w,h))
        p.setPen(QPen(rect_col,2))
        p.drawRect(QRect(x,y,w,h))

        cx2,cy2 = x+w//2, y+h//2
        p.setPen(QPen(QColor(255,255,255,180),1))
        p.drawLine(cx2-6,cy2,cx2+6,cy2)
        p.drawLine(cx2,cy2-6,cx2,cy2+6)

        offsets = {"TL":(-14,-8),"TR":(6,-8),
                   "BR":(6,12),"BL":(-16,12)}
        for lbl,(hx,hy) in corners.items():
            col = self._pt_color(acc[lbl])
            p.setPen(QPen(QColor(255,255,255),1)); p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPoint(hx,hy), 6, 6)
            p.setPen(Qt.NoPen); p.setBrush(QBrush(col))
            p.drawEllipse(QPoint(hx,hy), 5, 5)
            p.setBrush(Qt.NoBrush)
            dx,dy = offsets[lbl]
            p.setFont(QFont("Arial",7,QFont.Bold)); p.setPen(col)
            p.drawText(hx+dx, hy+dy, lbl)

        self._badge(p, good, 4)

    # ------------------------------------------------------------------ #
    #  Badge                                                               #
    # ------------------------------------------------------------------ #

    def _badge(self, p, good, total):
        pct = good / total if total else 0
        if pct >= 0.75:
            bg,txt,label = QColor(30,160,60),  QColor(255,255,255), "✓ Good"
        elif pct >= 0.5:
            bg,txt,label = QColor(200,120,0),  QColor(255,255,255), "~ Adjust"
        else:
            bg,txt,label = QColor(200,30,30),  QColor(255,255,255), "✗ Off"

        pw = self.width()
        ph = self.height()
        bw,bh = 62,18
        bx,by = pw-bw-6, ph-bh-6
        p.setPen(Qt.NoPen); p.setBrush(QBrush(bg))
        p.drawRoundedRect(bx,by,bw,bh, 5,5)
        p.setPen(txt)
        p.setFont(QFont("Arial",8,QFont.Bold))
        p.drawText(bx,by,bw,bh, Qt.AlignCenter, label)
        p.setBrush(Qt.NoBrush)

    # ------------------------------------------------------------------ #
    #  Paint                                                               #
    # ------------------------------------------------------------------ #

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rebuild()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        if self._pixmap:
            # Centre pixmap in widget
            px = (self.width()  - self._pixmap.width())  // 2
            py = (self.height() - self._pixmap.height()) // 2
            p.drawPixmap(px, py, self._pixmap)
        else:
            p.fillRect(self.rect(), QColor("#e8ecf4"))
            p.setPen(QColor("#999"))
            p.setFont(QFont("Arial", 9))
            p.drawText(self.rect(), Qt.AlignCenter,
                       "Detect Circle or\nRectangle to preview")
        p.end()