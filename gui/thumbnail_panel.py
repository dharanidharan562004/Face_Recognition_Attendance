"""
thumbnail_panel.py - Bottom image strip with folder navigation.
"""
import os, cv2
import numpy as np
from PyQt5.QtCore    import Qt, pyqtSignal, QThread, QSize
from PyQt5.QtGui     import QPixmap, QImage, QColor, QPainter, QPen, QFont
from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QLabel,
                               QPushButton, QScrollArea, QFrame, QSizePolicy)

THUMB_W = 110
THUMB_H = 90
LABEL_H = 16
ITEM_H  = THUMB_H + LABEL_H + 10
ITEM_W  = THUMB_W + 10
PANEL_H = ITEM_H + 16   # full panel height


class ThumbnailLoader(QThread):
    ready = pyqtSignal(int, QPixmap)

    def __init__(self, paths, parent=None):
        super().__init__(parent)
        self._paths = paths
        self._stop  = False

    def run(self):
        for i, p in enumerate(self._paths):
            if self._stop: break
            try:
                img = cv2.imread(p)
                if img is None: continue
                h, w = img.shape[:2]
                s    = min(THUMB_W/w, THUMB_H/h)
                nw, nh = max(1,int(w*s)), max(1,int(h*s))
                sm   = cv2.resize(img,(nw,nh),interpolation=cv2.INTER_AREA)
                rgb  = cv2.cvtColor(sm, cv2.COLOR_BGR2RGB)
                qi   = QImage(rgb.data, nw, nh, nw*3, QImage.Format_RGB888)
                pm   = QPixmap.fromImage(qi).copy()
                self.ready.emit(i, pm)
            except: pass


class ThumbCard(QWidget):
    clicked = pyqtSignal(int)

    def __init__(self, idx, fname, parent=None):
        super().__init__(parent)
        self.idx = idx
        self.setFixedSize(ITEM_W, ITEM_H)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(fname)

        v = QVBoxLayout(self)
        v.setContentsMargins(4,4,4,2)
        v.setSpacing(2)

        self._pic = QLabel()
        self._pic.setFixedSize(THUMB_W, THUMB_H)
        self._pic.setAlignment(Qt.AlignCenter)
        self._pic.setStyleSheet(
            "background:#c8cdd8;border:1px solid #9aaac0;"
            "border-radius:4px;color:#777;font-size:11px;")
        self._pic.setText("⏳")
        v.addWidget(self._pic)

        short = fname[:14]+"…" if len(fname)>15 else fname
        lbl   = QLabel(short)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setFixedHeight(LABEL_H)
        lbl.setStyleSheet("font-size:9px;color:#333;background:transparent;")
        v.addWidget(lbl)

    def set_pixmap(self, pm):
        # Centre on white background
        canvas = QPixmap(THUMB_W, THUMB_H)
        canvas.fill(QColor("#c8cdd8"))
        p = QPainter(canvas)
        x = (THUMB_W - pm.width())  // 2
        y = (THUMB_H - pm.height()) // 2
        p.drawPixmap(x, y, pm)
        p.end()
        self._pic.setPixmap(canvas)
        self._pic.setText("")

    def set_selected(self, sel):
        if sel:
            self._pic.setStyleSheet(
                "border:3px solid #2a5a9a;border-radius:4px;"
                "background:#d8e8fc;")
        else:
            self._pic.setStyleSheet(
                "border:1px solid #9aaac0;border-radius:4px;"
                "background:#c8cdd8;")

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.clicked.emit(self.idx)
        super().mousePressEvent(e)


class ThumbnailPanel(QWidget):
    image_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(PANEL_H)
        self.setMaximumHeight(PANEL_H)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setStyleSheet(
            "ThumbnailPanel{background:#dde3ec;"
            "border-top:2px solid #8aaac8;}")

        self._paths   = []
        self._cards   = []
        self._sel     = -1
        self._loader  = None
        self._build()

    def _build(self):
        row = QHBoxLayout(self)
        row.setContentsMargins(6,4,6,4)
        row.setSpacing(4)

        self._bl = self._abtn("◀")
        self._bl.clicked.connect(self._prev)
        row.addWidget(self._bl)

        self._sa = QScrollArea()
        self._sa.setMinimumHeight(PANEL_H - 10)
        self._sa.setMaximumHeight(PANEL_H - 10)
        self._sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self._sa.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._sa.setFrameShape(QFrame.NoFrame)
        self._sa.setStyleSheet("""
            QScrollArea { background:transparent; border:none; }
            QScrollBar:horizontal {
                background: #dde3ec;
                height: 8px;
                border-radius: 4px;
                margin: 0px;
            }
            QScrollBar::handle:horizontal {
                background: #5a7aaa;
                border-radius: 4px;
                min-width: 30px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #3e6188;
            }
            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal { width: 0px; }
        """)

        self._inner = QWidget()
        self._inner.setStyleSheet("background:transparent;")
        self._il = QHBoxLayout(self._inner)
        self._il.setContentsMargins(2,2,2,2)
        self._il.setSpacing(6)
        self._il.addStretch()
        self._sa.setWidget(self._inner)
        row.addWidget(self._sa, 1)

        self._br = self._abtn("▶")
        self._br.clicked.connect(self._next)
        row.addWidget(self._br)

        self._upd_arrows()

    def _abtn(self, t):
        b = QPushButton(t)
        b.setFixedSize(32, PANEL_H - 12)
        b.setStyleSheet("""
            QPushButton{background:#b8cce0;border:1px solid #8aaac8;
                border-radius:6px;font-size:16px;font-weight:bold;color:#1a3050;}
            QPushButton:hover{background:#2a5a9a;color:white;}
            QPushButton:pressed{background:#1a4070;color:white;}
            QPushButton:disabled{color:#bbb;background:#dde;}
        """)
        return b

    # ── public ──────────────────────────────────────────────────────────
    def load_folder(self, folder):
        exts = (".jpg",".jpeg")
        paths = sorted([
            os.path.join(folder,f)
            for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in exts
        ])
        if not paths: return

        if self._loader and self._loader.isRunning():
            self._loader._stop = True
            self._loader.wait(300)

        self._paths = paths
        self._sel   = -1
        self._cards = []

        # Clear
        while self._il.count() > 1:
            it = self._il.takeAt(0)
            if it.widget(): it.widget().deleteLater()

        for i,p in enumerate(paths):
            c = ThumbCard(i, os.path.basename(p))
            c.clicked.connect(self._click)
            self._il.insertWidget(i, c)
            self._cards.append(c)

        total_w = len(paths)*(ITEM_W+6)+8
        self._inner.setFixedWidth(total_w)
        self._inner.setFixedHeight(PANEL_H-10)
        self._upd_arrows()

        self._loader = ThumbnailLoader(paths, self)
        self._loader.ready.connect(self._got_thumb)
        self._loader.start()

    def select_index(self, idx):
        if 0 <= idx < len(self._paths):
            self._click(idx)

    def current_path(self) -> str:
        """Return path of currently selected image."""
        if 0 <= self._sel < len(self._paths):
            return self._paths[self._sel]
        return ""

    def highlight_path(self, path):
        try:
            idx = self._paths.index(path)
            if 0 <= self._sel < len(self._cards):
                self._cards[self._sel].set_selected(False)
            self._sel = idx
            self._cards[idx].set_selected(True)
            self._scroll_to(idx)
            self._upd_arrows()
        except (ValueError,IndexError): pass

    # ── private ─────────────────────────────────────────────────────────
    def _got_thumb(self, idx, pm):
        if 0 <= idx < len(self._cards):
            self._cards[idx].set_pixmap(pm)

    def _click(self, idx):
        if 0 <= self._sel < len(self._cards):
            self._cards[self._sel].set_selected(False)
        self._sel = idx
        self._cards[idx].set_selected(True)
        self._scroll_to(idx)
        self._upd_arrows()
        self.image_selected.emit(self._paths[idx])

    def _prev(self):
        if self._sel > 0: self._click(self._sel-1)
        elif self._sel == -1 and self._paths: self._click(0)

    def _next(self):
        if self._sel < len(self._paths)-1: self._click(self._sel+1)

    def _scroll_to(self, idx):
        self._sa.horizontalScrollBar().setValue(idx*(ITEM_W+6))

    def _upd_arrows(self):
        n = len(self._paths)
        self._bl.setEnabled(self._sel > 0)
        self._br.setEnabled(0 <= self._sel < n-1)

    def keyPressEvent(self, e):
        if   e.key()==Qt.Key_Left:  self._prev()
        elif e.key()==Qt.Key_Right: self._next()
        elif e.key() in (Qt.Key_Return, Qt.Key_Enter):
            if self._sel >= 0:
                self.image_selected.emit(self._paths[self._sel])
        else: super().keyPressEvent(e)