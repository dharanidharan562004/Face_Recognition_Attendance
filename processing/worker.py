from __future__ import annotations
"""
worker.py
Background QThread workers for heavy OpenCV operations.

QR detection uses higher resolution (2400px) and also scans
4 corner sub-regions independently so small/off-centre QR codes
are never missed due to downscaling.
"""

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal


class DetectionWorker(QThread):

    result_ready = pyqtSignal(str, dict)
    error        = pyqtSignal(str)

    def __init__(self, image: np.ndarray, mode: str, parent=None):
        super().__init__(parent)
        self._orig   = image
        self._mode   = mode
        # Circle/rect detector now handles its own proxy scaling internally.
        # Keep a 1400px copy only for non-QR modes to save memory.
        self._image  = self._downscale(image, max_side=1400)
        self._scale  = image.shape[1] / self._image.shape[1]

    def run(self):
        try:
            if self._mode == "circle":
                self.result_ready.emit("circle", self._detect_circle())
            elif self._mode == "rectangle":
                self.result_ready.emit("rectangle", self._detect_rectangle())
            elif self._mode == "qr":
                self.result_ready.emit("qr", self._detect_qr())
        except Exception as e:
            self.error.emit(str(e))

    # ------------------------------------------------------------------ #
    #  Circle / Rectangle                                                  #
    # ------------------------------------------------------------------ #

    def _detect_circle(self) -> dict:
        from processing.coin_detector import CoinDetector
        # Detector internally uses 900px proxy — pass original
        return CoinDetector().detect_circle(self._orig)

    def _detect_rectangle(self) -> dict:
        from processing.coin_detector import CoinDetector
        # Detector internally uses 600px proxy — very fast
        return CoinDetector().detect_rectangle(self._orig)

    # ------------------------------------------------------------------ #
    #  QR — multi-resolution + corner-region scan                         #
    # ------------------------------------------------------------------ #

    def _detect_qr(self) -> dict:
        from processing.qr_detector import QRDetector
        det  = QRDetector()
        oh, ow = self._orig.shape[:2]

        # Pass 1: 2400px downscale (fast, handles most cases)
        img_2400 = self._downscale(self._orig, max_side=2400)
        s1       = ow / img_2400.shape[1]
        regions  = det.detect(img_2400)
        if regions:
            return {"regions": self._scale_regions(regions, s1)}

        # Pass 2: Scan label region (right 60% / bottom 40% — where QR usually is)
        h2, w2 = img_2400.shape[:2]
        label_regions_to_try = [
            img_2400[:, w2//3:],          # right 2/3
            img_2400[h2//3:, :],          # bottom 2/3
            img_2400[:h2//2, :],          # top half
            img_2400[:, :w2//2],          # left half
            img_2400[h2//4:3*h2//4, w2//4:3*w2//4],  # centre
        ]
        offsets = [
            (w2//3, 0),
            (0, h2//3),
            (0, 0),
            (0, 0),
            (w2//4, h2//4),
        ]
        for crop, (ox2, oy2) in zip(label_regions_to_try, offsets):
            if crop.size == 0:
                continue
            regions = det.detect(crop)
            if regions:
                # Shift coords back
                for r in regions:
                    bx, by, bw, bh = r["bbox"]
                    r["bbox"] = (bx+ox2, by+oy2, bw, bh)
                    if r["points"] is not None:
                        r["points"] = r["points"] + np.array([[ox2, oy2]])
                return {"regions": self._scale_regions(regions, s1)}

        # Pass 3: full resolution (small QR on large image)
        if oh * ow < 10_000_000:
            regions = det.detect(self._orig)
            if regions:
                return {"regions": regions}

        # Pass 4: 1600px
        img_1600 = self._downscale(self._orig, max_side=1600)
        s2       = ow / img_1600.shape[1]
        regions  = det.detect(img_1600)
        if regions:
            return {"regions": self._scale_regions(regions, s2)}

        return {"regions": []}

    def _scan_corners(self, det, scale_image=None, scale=1.0) -> list:
        """
        Crop each 40% corner of the image and run QR detection.
        Returns results in original image coordinates.
        """
        img   = scale_image if scale_image is not None else self._orig
        h, w  = img.shape[:2]
        cw    = int(w * 0.45)   # corner width
        ch    = int(h * 0.45)   # corner height

        corners = [
            (0,      0,      cw, ch),           # top-left
            (w - cw, 0,      cw, ch),           # top-right
            (0,      h - ch, cw, ch),           # bottom-left
            (w - cw, h - ch, cw, ch),           # bottom-right
        ]

        for ox, oy, cw_, ch_ in corners:
            crop    = img[oy:oy+ch_, ox:ox+cw_]
            regions = det.detect(crop)
            if not regions:
                continue
            # Translate crop-local coords → original image coords,
            # then apply the scale factor to reach true pixel coords
            out = []
            for reg in regions:
                bx, by, bw, bh = reg["bbox"]
                # Add crop offset, then scale to orig image coords
                out.append({
                    **reg,
                    "bbox": (
                        int((bx + ox) * scale),
                        int((by + oy) * scale),
                        int(bw * scale),
                        int(bh * scale),
                    ),
                    "points": (
                        ((reg["points"] + np.array([ox, oy])) * scale
                         ).astype(int)
                        if reg["points"] is not None else None
                    ),
                })
            return out
        return []

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _scale_regions(regions: list, scale: float) -> list:
        out = []
        for reg in regions:
            bx, by, bw, bh = reg["bbox"]
            out.append({
                **reg,
                "bbox":   (int(bx*scale), int(by*scale),
                           int(bw*scale), int(bh*scale)),
                "points": (reg["points"] * scale).astype(int)
                          if reg["points"] is not None else None,
            })
        return out

    @staticmethod
    def _downscale(image: np.ndarray, max_side: int = 1400) -> np.ndarray:
        h, w = image.shape[:2]
        if max(h, w) <= max_side:
            return image
        s     = max_side / max(h, w)
        return cv2.resize(image, (int(w*s), int(h*s)),
                          interpolation=cv2.INTER_AREA)