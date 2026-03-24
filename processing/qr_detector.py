from __future__ import annotations
"""
qr_detector.py — Robust QR / barcode scanner.

Strategy:
  1. Full image → 4 rotations × 7 preprocessing variants
  2. Proxy scale (2400px) for large images
  3. 3×3 grid scan with 30% overlap
  4. 4×4 grid scan with 30% overlap
  5. 2× upscale for small QR codes
  6. pyzbar fallback on every attempt
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple, Dict


class QRDetector:

    def detect(self, image: np.ndarray) -> List[dict]:
        h, w = image.shape[:2]
        results = []

        # ── 1. Proxy (max 2400px long side) ───────────────────────────
        scale  = min(1.0, 2400 / max(h, w))
        if scale < 1.0:
            proxy = cv2.resize(image, (int(w*scale), int(h*scale)),
                               interpolation=cv2.INTER_AREA)
        else:
            proxy = image
            scale = 1.0

        # ── 2. Full image all rotations ────────────────────────────────
        results = self._scan_image(proxy)
        if results:
            return self._rescale(results, 1.0/scale)

        # ── 3. Grid 3×3 ───────────────────────────────────────────────
        results = self._grid_scan(proxy, rows=3, cols=3, overlap=0.30)
        if results:
            return self._rescale(results, 1.0/scale)

        # ── 4. Grid 4×4 ───────────────────────────────────────────────
        results = self._grid_scan(proxy, rows=4, cols=4, overlap=0.30)
        if results:
            return self._rescale(results, 1.0/scale)

        # ── 5. 2× upscale full image ───────────────────────────────────
        ph, pw = proxy.shape[:2]
        big    = cv2.resize(proxy, (pw*2, ph*2), interpolation=cv2.INTER_CUBIC)
        results = self._scan_image(big)
        if results:
            return self._rescale(results, 0.5/scale)

        # ── 6. 2× upscale grid 3×3 ────────────────────────────────────
        results = self._grid_scan(big, rows=3, cols=3, overlap=0.30)
        if results:
            return self._rescale(results, 0.5/scale)

        return []

    # ------------------------------------------------------------------ #
    #  Scan whole image at all rotations                                   #
    # ------------------------------------------------------------------ #

    def _scan_image(self, image: np.ndarray) -> List[dict]:
        gray = self._gray(image)
        h, w = gray.shape

        for angle in (0, 90, 180, 270):
            rot_img  = self._rot(image, angle)
            rot_gray = self._gray(rot_img)
            rh, rw   = rot_gray.shape

            for variant in self._variants(rot_gray):
                res = self._detect_one(variant)
                if res:
                    return self._unrot(res, angle, rw, rh)
        return []

    # ------------------------------------------------------------------ #
    #  Grid scan                                                           #
    # ------------------------------------------------------------------ #

    def _grid_scan(self, image: np.ndarray,
                   rows=3, cols=3, overlap=0.30) -> List[dict]:
        h, w   = image.shape[:2]
        tile_h = int(h / (rows  - overlap * (rows  - 1)))
        tile_w = int(w / (cols  - overlap * (cols  - 1)))
        step_h = int(tile_h * (1 - overlap))
        step_w = int(tile_w * (1 - overlap))
        seen   = []

        for row in range(rows):
            for col in range(cols):
                y1 = min(row * step_h, max(0, h - tile_h))
                x1 = min(col * step_w, max(0, w - tile_w))
                y2 = min(y1 + tile_h, h)
                x2 = min(x1 + tile_w, w)
                tile = image[y1:y2, x1:x2]
                if tile.size == 0:
                    continue

                th, tw = tile.shape[:2]

                for angle in (0, 90, 180, 270):
                    rot      = self._rot(tile, angle)
                    rot_gray = self._gray(rot)
                    rh, rw   = rot_gray.shape

                    for variant in self._variants(rot_gray):
                        res = self._detect_one(variant)
                        if not res:
                            continue

                        # Unrotate to tile coords
                        unrot = self._unrot(res, angle, rw, rh)

                        for r in unrot:
                            bx, by, bw, bh = r["bbox"]
                            # Shift to full image coords
                            fx, fy = bx + x1, by + y1
                            cx, cy = fx + bw//2, fy + bh//2

                            # Deduplicate
                            if any(abs(cx-sc[0])<60 and abs(cy-sc[1])<60
                                   for sc in seen):
                                continue
                            seen.append((cx, cy))

                            pts = r["points"]
                            if pts is not None:
                                pts = pts + np.array([[x1, y1]])

                            return [{"type":   r["type"],
                                     "data":   r["data"],
                                     "points": pts,
                                     "bbox":   (fx, fy, bw, bh)}]
        return []

    # ------------------------------------------------------------------ #
    #  Detect on single preprocessed image                                 #
    # ------------------------------------------------------------------ #

    def _detect_one(self, gray: np.ndarray) -> List[dict]:
        # pyzbar first — correctly identifies QR vs barcode
        r = self._pyzbar(gray)
        if r: return r
        # OpenCV QR fallback — only if pyzbar found nothing
        # cv2.QRCodeDetector can misidentify barcodes as QR, so
        # we only use it when pyzbar returns empty
        r = self._cv_qr(gray)
        if r: return r
        return []

    # ------------------------------------------------------------------ #
    #  Preprocessing variants                                              #
    # ------------------------------------------------------------------ #

    def _variants(self, gray: np.ndarray):
        yield gray                                  # 1. plain
        yield self._clahe(gray)                     # 2. CLAHE
        yield self._sharpen(gray)                   # 3. sharpen
        yield self._otsu(gray)                      # 4. Otsu
        yield self._adaptive(gray)                  # 5. adaptive
        yield self._sharpen(self._clahe(gray))      # 6. CLAHE+sharpen
        yield cv2.bitwise_not(gray)                 # 7. inverted
        # Extra: denoise + CLAHE
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        yield self._clahe(denoised)                 # 8. denoise+CLAHE

    # ------------------------------------------------------------------ #
    #  Core detectors                                                      #
    # ------------------------------------------------------------------ #

    def _cv_qr(self, gray: np.ndarray) -> List[dict]:
        # Try WeChatQR detector first (more accurate)
        try:
            wechat = cv2.wechat_qrcode_WeChatQRCode()
            texts, pts_list = wechat.detectAndDecode(gray)
            results = []
            for text, pts in zip(texts, pts_list):
                if pts is not None and len(pts) > 0:
                    p = pts.astype(int)
                    results.append({"type": "qr", "data": text or "",
                                    "points": p,
                                    "bbox": self._bbox(p, gray.shape)})
            if results:
                return results
        except Exception:
            pass

        # detectAndDecodeMulti — finds multiple QR codes
        det = cv2.QRCodeDetector()
        try:
            ok, texts, pts_list, _ = det.detectAndDecodeMulti(gray)
            if ok and pts_list is not None:
                results = []
                for i, pts in enumerate(pts_list):
                    p = pts.astype(int)
                    text = texts[i] if i < len(texts) else ""
                    results.append({"type": "qr", "data": text or "",
                                    "points": p,
                                    "bbox": self._bbox(p, gray.shape)})
                return results
        except Exception:
            pass

        # Single QR fallback
        try:
            data, pts, _ = det.detectAndDecode(gray)
            if pts is not None:
                p = pts[0].astype(int)
                return [{"type": "qr", "data": data or "",
                         "points": p,
                         "bbox": self._bbox(p, gray.shape)}]
        except Exception:
            pass
        return []

    def _pyzbar(self, gray: np.ndarray) -> List[dict]:
        try:
            from pyzbar import pyzbar
            objs = pyzbar.decode(gray)
        except Exception:
            return []
        out = []
        for obj in objs:
            pts  = np.array([[p.x, p.y] for p in obj.polygon], dtype=int)
            kind = "qr" if obj.type == "QRCODE" else "barcode"
            data = obj.data.decode("utf-8", errors="replace")
            out.append({"type":   kind,
                        "data":   data,
                        "points": pts,
                        "bbox":   self._bbox(pts, gray.shape)})
        return out

    # ------------------------------------------------------------------ #
    #  Preprocessing helpers                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _sharpen(g):
        k = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], np.float32)
        return np.clip(cv2.filter2D(g, -1, k), 0, 255).astype(np.uint8)

    @staticmethod
    def _clahe(g):
        return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(g)

    @staticmethod
    def _adaptive(g):
        return cv2.adaptiveThreshold(
            g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2)

    @staticmethod
    def _otsu(g):
        _, t = cv2.threshold(g, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return t

    # ------------------------------------------------------------------ #
    #  Rotation helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _rot(img, angle):
        if angle == 0:   return img
        if angle == 90:  return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if angle == 180: return cv2.rotate(img, cv2.ROTATE_180)
        if angle == 270: return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img

    @staticmethod
    def _unrot(results, angle, orig_w, orig_h):
        if angle == 0:
            return results
        out = []
        for r in results:
            pts = r["points"].astype(float) if r["points"] is not None else None
            bx, by, bw, bh = r["bbox"]

            if angle == 180:
                if pts is not None:
                    pts[:,0] = orig_w - pts[:,0]
                    pts[:,1] = orig_h - pts[:,1]
                bx = orig_w - bx - bw
                by = orig_h - by - bh

            elif angle == 90:   # CW → unrot CCW
                if pts is not None:
                    new = np.zeros_like(pts)
                    new[:,0] = pts[:,1]
                    new[:,1] = orig_w - pts[:,0]
                    pts = new
                bx, by, bw, bh = by, orig_w - bx - bw, bh, bw

            elif angle == 270:  # CCW → unrot CW
                if pts is not None:
                    new = np.zeros_like(pts)
                    new[:,0] = orig_h - pts[:,1]
                    new[:,1] = pts[:,0]
                    pts = new
                bx, by, bw, bh = orig_h - by - bh, bx, bh, bw

            out.append({**r,
                "points": np.clip(pts,0,None).astype(int) if pts is not None else None,
                "bbox":   (max(0,int(bx)), max(0,int(by)),
                           max(1,int(bw)), max(1,int(bh)))})
        return out

    # ------------------------------------------------------------------ #
    #  Scale results                                                       #
    # ------------------------------------------------------------------ #


    @staticmethod
    def _rescale(regions, s):
        if s == 1.0:
            return regions
        out = []
        for r in regions:
            bx, by, bw, bh = r["bbox"]
            pts = (r["points"] * s).astype(int) \
                  if r["points"] is not None else None
            out.append({**r,
                "bbox":   (int(bx*s), int(by*s),
                           max(1,int(bw*s)), max(1,int(bh*s))),
                "points": pts})
        return out

    # ------------------------------------------------------------------ #
    #  Utility                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _bbox(pts, shape):
        h, w = shape[:2]
        x1 = max(0, int(pts[:,0].min()))
        y1 = max(0, int(pts[:,1].min()))
        x2 = min(w, int(pts[:,0].max()))
        y2 = min(h, int(pts[:,1].max()))
        return (x1, y1, max(1,x2-x1), max(1,y2-y1))

    @staticmethod
    def _gray(img):
        if len(img.shape) == 2:
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)