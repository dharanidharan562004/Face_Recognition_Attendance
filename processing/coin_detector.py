from __future__ import annotations
"""
coin_detector.py
Auto-fit detection — like a document scanner.

Circle  → finds the coin edge precisely using edge gradient analysis
Rectangle → finds the slab boundary like Adobe Scan document detection
Both results are ready to Apply Crop immediately with no manual adjustment.
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple, Dict


class CoinDetector:

    # ================================================================== #
    #  PUBLIC API                                                          #
    # ================================================================== #

    def detect_circle(self, image: np.ndarray) -> dict:
        """
        Precisely detect coin circle boundary.
        Returns { x, y, radius } ready for immediate crop.
        """
        h, w   = image.shape[:2]
        proxy, s = self._make_proxy(image, target=1000)
        gray   = self._gray(proxy)
        ph, pw = gray.shape[:2]

        result = (self._precise_circle(gray, pw, ph)
                  or self._contour_circle(gray, pw, ph))

        return {
            "x":      int(result["x"]      * s),
            "y":      int(result["y"]      * s),
            "radius": int(result["radius"] * s),
        }

    def detect_rectangle(self, image: np.ndarray) -> dict:
        """
        Detect content boundary with equal padding on all 4 sides.
        """
        h, w     = image.shape[:2]
        proxy, s = self._make_proxy(image, target=800)
        gray     = self._gray(proxy)
        ph, pw   = gray.shape[:2]

        result = (self._precise_rect(gray, pw, ph)
                  or {"x": 2, "y": 2, "width": pw - 4, "height": ph - 4})

        # Scale back to original coords
        rx = max(0, int(result["x"]      * s))
        ry = max(0, int(result["y"]      * s))
        rw = max(6, int(result["width"]  * s))
        rh = max(6, int(result["height"] * s))

        # Equal padding on all 4 sides (2% of shorter side)
        pad = int(min(rw, rh) * 0.02)
        pad = max(8, min(pad, 40))   # min 8px, max 40px

        x1 = max(0, rx - pad)
        y1 = max(0, ry - pad)
        x2 = min(w, rx + rw + pad)
        y2 = min(h, ry + rh + pad)

        return {"x": x1, "y": y1,
                "width":  x2 - x1,
                "height": y2 - y1}

    # ================================================================== #
    #  CIRCLE — precise edge-based detection                              #
    # ================================================================== #

    def _precise_circle(self, gray, w, h) -> Optional[dict]:
        """
        Multi-pass HoughCircles with CLAHE + label suppression.
        Picks the circle with strongest, most uniform edge ring.
        """
        # Enhance contrast
        clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Build label mask and suppress it
        label_mask = self._label_mask(gray, w, h)
        suppressed = enhanced.copy()
        suppressed[label_mask == 255] = 128

        blurred = cv2.GaussianBlur(suppressed, (7, 7), 1.5)
        short   = min(w, h)
        edges   = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1), 30, 100)

        candidates = []

        # Collect circles at multiple sensitivity levels
        for dp, p1, p2, min_r_pct, max_r_pct in [
            (1.0, 100, 40, 0.15, 0.48),
            (1.2,  80, 28, 0.12, 0.48),
            (1.5,  60, 18, 0.10, 0.49),
        ]:
            c = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT, dp=dp,
                minDist=int(short * 0.12),
                param1=p1, param2=p2,
                minRadius=int(short * min_r_pct),
                maxRadius=int(short * max_r_pct),
            )
            if c is not None:
                candidates.extend(np.round(c[0]).astype(int).tolist())

        candidates = self._dedup(candidates, tol=15)

        # Hard-filter: remove circles centred in label zone
        filtered = [c for c in candidates
                    if not (0 <= int(c[1]) < h and 0 <= int(c[0]) < w
                            and label_mask[int(c[1]), int(c[0])] == 255)]
        if not filtered:
            filtered = candidates
        if not filtered:
            return None

        # Score and pick best
        return self._score_circles(filtered, gray, edges, label_mask, w, h)

    def _score_circles(self, circles, gray, edges, label_mask, w, h) -> dict:
        best_score = -1e9
        best = circles[0]

        for c in circles:
            cx, cy, r = int(c[0]), int(c[1]), int(c[2])
            score = 0.0

            # A: Interior sample
            interior = np.zeros(gray.shape, np.uint8)
            cv2.circle(interior, (cx, cy), max(1, int(r * 0.82)), 255, -1)
            interior[label_mask == 255] = 0
            pix = gray[interior == 255]
            if len(pix) < 200:
                continue

            mean_v = float(pix.mean())
            std_v  = float(pix.std())

            # B: Coin brightness (not too bright = not plastic)
            if mean_v > 225:   score -= 90
            elif mean_v < 35:  score -= 40
            else:              score += 40 - abs(mean_v - 140) * 0.15

            # C: Some texture (coin detail), penalise blank plastic
            if std_v < 6:     score -= 50
            elif std_v < 40:  score += 20

            # D: Edge ring strength — key metric
            ring = np.zeros(gray.shape, np.uint8)
            cv2.circle(ring, (cx, cy), r, 255, max(4, int(r * 0.06)))
            ring[label_mask == 255] = 0
            edge_val = float(cv2.mean(edges, mask=ring)[0])
            score += edge_val * 0.50   # high weight

            # E: Edge ring uniformity (coin = circular = uniform)
            # Sample 8 arc segments and check variance
            arc_vals = []
            for deg in range(0, 360, 45):
                rad = np.radians(deg)
                arc = np.zeros(gray.shape, np.uint8)
                pts = []
                for d in range(deg, deg + 45):
                    rr = np.radians(d)
                    px = int(cx + r * np.cos(rr))
                    py = int(cy + r * np.sin(rr))
                    if 0 <= px < w and 0 <= py < h:
                        pts.append((px, py))
                if pts:
                    for px, py in pts:
                        arc[py, px] = 255
                    arc_vals.append(float(cv2.mean(edges, mask=arc)[0]))
            if arc_vals and len(arc_vals) > 2:
                uniformity = np.std(arc_vals)
                score -= uniformity * 0.20  # more uniform = better coin

            # F: Penalise label zone centre
            if (0 <= cy < h and 0 <= cx < w
                    and label_mask[cy, cx] == 255):
                score -= 120

            # G: Inner ring brightness (transparent slab well = bright)
            inner_ring = np.zeros(gray.shape, np.uint8)
            cv2.circle(inner_ring, (cx, cy), max(1, int(r * 0.92)), 255, 4)
            cv2.circle(inner_ring, (cx, cy), max(1, int(r * 0.78)), 0,   4)
            ir_mean = float(cv2.mean(gray, mask=inner_ring)[0])
            if ir_mean > 215:
                score -= 60  # plastic slab well, not coin edge

            # H: Penalise oversized
            if r > min(w, h) * 0.47:
                score -= 70

            if score > best_score:
                best_score = score
                best = c

        return {"x": int(best[0]), "y": int(best[1]),
                "radius": int(best[2])}

    def _contour_circle(self, gray, w, h) -> dict:
        """Fallback: largest dark circular contour."""
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        _, thr   = cv2.threshold(blurred, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kern    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        clean   = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kern, iterations=2)
        cnts, _ = cv2.findContours(
            clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts    = [c for c in cnts if cv2.contourArea(c) > w * h * 0.015]
        if not cnts:
            r = int(min(w, h) * 0.38)
            return {"x": w // 2, "y": h // 2, "radius": r}
        best = max(cnts, key=cv2.contourArea)
        (cx, cy), r = cv2.minEnclosingCircle(best)
        return {"x": int(cx), "y": int(cy), "radius": int(r * 0.96)}

    # ================================================================== #
    #  RECTANGLE — Adobe-Scan-style document detection                    #
    # ================================================================== #

    def _precise_rect(self, gray, w, h) -> Optional[dict]:
        """
        Detect rectangular slab boundary using:
        1. Adaptive threshold + contour (like document scanners)
        2. Canny multi-scale
        3. Perspective-aware largest quadrilateral
        """
        blurred = cv2.GaussianBlur(gray, (5, 5), 1)

        # ── Method 1: Adaptive threshold (works on any lighting) ───────
        adapt = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2)
        adapt = cv2.bitwise_not(adapt)
        kern  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        adapt = cv2.dilate(adapt, kern, iterations=2)
        result = self._largest_rect_contour(adapt, w, h, min_fill=0.20)
        if result:
            return result

        # ── Method 2: Multi-threshold Canny ────────────────────────────
        for lo, hi in ((15, 60), (25, 90), (40, 120)):
            edges  = cv2.Canny(blurred, lo, hi)
            kern2  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                                      kern2, iterations=3)
            result = self._largest_rect_contour(closed, w, h, min_fill=0.15)
            if result:
                return result

        # ── Method 3: Otsu threshold ────────────────────────────────────
        _, otsu = cv2.threshold(blurred, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kern3  = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        closed = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kern3, iterations=4)
        result = self._largest_rect_contour(closed, w, h, min_fill=0.10)
        if result:
            return result

        return None

    def _largest_rect_contour(self, binary, w, h,
                               min_fill=0.20) -> Optional[dict]:
        """
        Find the largest contour that looks like a rectangle/document.
        Tries to approximate as quadrilateral (document scanner approach).
        """
        cnts, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None

        img_area  = w * h
        best_area = 0
        best_rect = None

        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < img_area * min_fill:  continue
            if area > img_area * 0.98:      continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw < 6 or bh < 6:           continue

            aspect = bw / bh if bh > 0 else 0
            if not (0.10 < aspect < 10.0):  continue

            # Try to approximate as polygon — fewer sides = more rectangular
            peri    = cv2.arcLength(cnt, True)
            approx  = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            n_sides = len(approx)

            # Score: prefer quadrilaterals (4 sides), penalise complex shapes
            shape_score = 1.0
            if   n_sides == 4: shape_score = 1.5   # perfect quad
            elif n_sides <= 6: shape_score = 1.2   # nearly rectangular
            elif n_sides > 12: shape_score = 0.7   # too complex

            weighted = area * shape_score
            if weighted > best_area:
                best_area = weighted
                pad  = 0   # tight — no extra padding
                best_rect = {
                    "x":      max(0, x),
                    "y":      max(0, y),
                    "width":  min(w - x, bw),
                    "height": min(h - y, bh),
                }

        return best_rect

    # ================================================================== #
    #  LABEL MASK                                                          #
    # ================================================================== #

    def _label_mask(self, gray, w, h) -> np.ndarray:
        mask = np.zeros((h, w), np.uint8)

        # Bright rectangles (label background)
        _, bright = cv2.threshold(gray, 165, 255, cv2.THRESH_BINARY)
        kern      = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 25))
        closed    = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kern)
        cnts, _   = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            if cv2.contourArea(cnt) < w * h * 0.01:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / bh if bh > 0 else 0
            if aspect > 1.2 and (y + bh) < h * 0.65:
                cv2.rectangle(mask, (0, y), (w, y + bh), 255, -1)

        # Top 38% heuristic
        top_h  = int(h * 0.38)
        top_gr = gray[:top_h, :]
        if top_gr.mean() > 130 and top_gr.std() > 25:
            mask[:top_h, :] = 255

        return mask

    # ================================================================== #
    #  HELPERS                                                             #
    # ================================================================== #

    @staticmethod
    def _make_proxy(image: np.ndarray,
                    target: int = 900) -> Tuple[np.ndarray, float]:
        h, w = image.shape[:2]
        long_side = max(h, w)
        if long_side <= target:
            return image, 1.0
        s   = target / long_side
        nw  = max(1, int(w * s))
        nh  = max(1, int(h * s))
        return cv2.resize(image, (nw, nh),
                          interpolation=cv2.INTER_AREA), 1.0 / s

    @staticmethod
    def _dedup(circles: list, tol: int = 15) -> list:
        kept = []
        for c in circles:
            if not any(abs(c[0]-k[0]) < tol and abs(c[1]-k[1]) < tol
                       for k in kept):
                kept.append(c)
        return kept

    @staticmethod
    def _gray(image: np.ndarray) -> np.ndarray:
        return (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                if len(image.shape) == 3 else image)