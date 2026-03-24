from __future__ import annotations
"""
image_processor.py
Core image-processing logic: loading, rotation, circular/rectangular
cropping, and saving.
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple, Dict
from PIL import Image


class ImageProcessor:
    """
    Manages the mutable state of a single coin image as the user edits it.

    Attributes
    ----------
    original   : np.ndarray – the image exactly as loaded from disk
    current    : np.ndarray – the image after all rotations
    angle      : float      – accumulated rotation angle (degrees)
    crop_type  : str        – "circle" | "rectangle" | None
    crop_params: dict       – current crop region (depends on crop_type)
    """

    STEP_FINE   = 1.0    # right panel fine rotation
    STEP_COARSE = 90.0   # toolbar 90° step rotation

    def __init__(self):
        self.original: Optional[np.ndarray] = None
        self.current: Optional[np.ndarray] = None
        self.angle: float = 0.0
        self.crop_type: Optional[str] = None
        self.crop_params: dict = {}
        self._load_original: Optional[np.ndarray] = None
        self._checkpoint: Optional[np.ndarray] = None
        self._checkpoint_angle: float = 0.0
        # Step history for 1-step undo (Reset button)
        self._history: List[tuple] = []   # [(image, angle), ...]
        self._MAX_HISTORY = 20

    # ------------------------------------------------------------------ #
    #  History helpers                                                      #
    # ------------------------------------------------------------------ #

    def _push_history(self):
        """Save current state before an action (for 1-step undo)."""
        if self.current is not None:
            self._history.append((self.current.copy(), self.angle))
            if len(self._history) > self._MAX_HISTORY:
                self._history.pop(0)

    def undo_step(self) -> Optional[np.ndarray]:
        """Undo last action — returns previous image or None if no history."""
        if not self._history:
            return None
        img, angle     = self._history.pop()
        self.original  = img.copy()
        self.current   = img.copy()
        self.angle     = angle
        self.crop_type = None
        self.crop_params = {}
        return self.current

    # ------------------------------------------------------------------ #
    #  Loading                                                             #
    # ------------------------------------------------------------------ #

    def load(self, path: str) -> np.ndarray:
        """Load an image from *path* and reset state. Returns the image."""
        img = cv2.imread(path)
        if img is None:
            # Try via Pillow (handles some formats cv2 misses)
            pil = Image.open(path).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        if img is None:
            raise ValueError(f"Cannot open image: {path}")

        self.original        = img.copy()
        self.current         = img.copy()
        self.angle           = 0.0
        self.crop_type       = None
        self.crop_params     = {}
        self._load_original  = img.copy()   # never overwritten
        self._history        = []           # clear undo history
        self._checkpoint     = img.copy()
        self._checkpoint_angle = 0.0
        return self.current

    def save_checkpoint(self) -> None:
        """Call this after a successful save so reset goes back here."""
        if self.current is not None:
            self._checkpoint       = self.current.copy()
            self._checkpoint_angle = self.angle

    def reset_to_original(self) -> np.ndarray:
        """
        Always reset to the very first loaded image — ignoring any saves.
        This is what the Reset button calls.
        """
        src = self._load_original if self._load_original is not None \
              else self._checkpoint
        if src is None:
            return self.current
        self.original    = src.copy()
        self.current     = src.copy()
        self.angle       = 0.0
        self.crop_type   = None
        self.crop_params = {}
        return self.current

    def reset_to_checkpoint(self) -> np.ndarray:
        """Reset to the last saved state (used internally)."""
        if self._checkpoint is None:
            return self.current
        self.original    = self._checkpoint.copy()
        self.current     = self._checkpoint.copy()
        self.angle       = self._checkpoint_angle
        self.crop_type   = None
        self.crop_params = {}
        return self.current

    # ------------------------------------------------------------------ #
    #  Rotation                                                            #
    # ------------------------------------------------------------------ #

    def rotate_cw(self) -> np.ndarray:
        """Fine +1° — right panel button."""
        return self._rotate_by(self.STEP_FINE)

    def rotate_ccw(self) -> np.ndarray:
        """Fine -1° — right panel button."""
        return self._rotate_by(-self.STEP_FINE)

    def rotate_cw_90(self) -> np.ndarray:
        """Coarse +90° — toolbar button."""
        return self._rotate_by(self.STEP_COARSE)

    def rotate_ccw_90(self) -> np.ndarray:
        """Coarse -90° — toolbar button."""
        return self._rotate_by(-self.STEP_COARSE)

    def rotate_to(self, angle: float) -> np.ndarray:
        delta = (angle % 360) - self.angle
        self.angle = angle % 360
        if self.crop_params and self.crop_type:
            # Crop active → rotate only inside crop region
            self.current = self._rotate_inside_crop(
                self.current, self.crop_type, self.crop_params, delta
            )
            self.original = self.current.copy()
        else:
            self.current = self._apply_rotation(self.original, self.angle)
        return self.current

    def _rotate_by(self, delta: float) -> np.ndarray:
        self._push_history()   # save before rotate
        self.angle = (self.angle + delta) % 360
        if self.crop_params and self.crop_type:
            # Crop active → rotate ONLY the content inside the crop
            # Overlay stays completely fixed
            self.current = self._rotate_inside_crop(
                self.current, self.crop_type, self.crop_params, delta
            )
            # Keep original in sync so reset works
            self.original = self.current.copy()
        else:
            # No crop → rotate whole image
            self.current = self._apply_rotation(self.original, self.angle)
        return self.current

    @staticmethod
    def _rotate_inside_crop(image: np.ndarray,
                             crop_type: str,
                             params: dict,
                             delta_deg: float) -> np.ndarray:
        """
        Rotate only the pixels inside the crop circle/rectangle.
        The overlay position stays fixed — only content rotates.
        """
        result = image.copy()
        h, w   = image.shape[:2]

        if crop_type == "circle":
            cx = params["x"]
            cy = params["y"]
            r  = params["radius"]

            # Extract bounding box of the circle
            x1 = max(0, cx - r);  y1 = max(0, cy - r)
            x2 = min(w, cx + r);  y2 = min(h, cy + r)
            roi = image[y1:y2, x1:x2].copy()

            # Rotate the ROI around its own centre
            rh, rw = roi.shape[:2]
            rcx, rcy = rw / 2, rh / 2
            M    = cv2.getRotationMatrix2D((rcx, rcy), -delta_deg, 1.0)
            rotated_roi = cv2.warpAffine(
                roi, M, (rw, rh),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255)
            )

            # Create circular mask in ROI space
            mask = np.zeros((rh, rw), dtype=np.uint8)
            # Circle centre relative to ROI
            rel_cx = cx - x1
            rel_cy = cy - y1
            cv2.circle(mask, (rel_cx, rel_cy), r, 255, -1)

            # Blend: inside circle = rotated, outside = original
            mask3 = cv2.merge([mask, mask, mask])
            blended = np.where(mask3 == 255, rotated_roi, roi)
            result[y1:y2, x1:x2] = blended

        elif crop_type == "rectangle":
            x  = max(0, params["x"])
            y  = max(0, params["y"])
            x2 = min(w, x + params["width"])
            y2 = min(h, y + params["height"])
            roi = image[y:y2, x:x2].copy()

            rh, rw = roi.shape[:2]
            rcx, rcy = rw / 2, rh / 2
            M = cv2.getRotationMatrix2D((rcx, rcy), -delta_deg, 1.0)
            rotated_roi = cv2.warpAffine(
                roi, M, (rw, rh),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255)
            )
            result[y:y2, x:x2] = rotated_roi

        return result

    @staticmethod
    def _apply_rotation(image: np.ndarray, angle: float) -> np.ndarray:
        h, w = image.shape[:2]
        cx, cy = w / 2, h / 2
        M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)

        # Expand canvas so no content is clipped
        cos_a = abs(M[0, 0])
        sin_a = abs(M[0, 1])
        new_w = int(h * sin_a + w * cos_a)
        new_h = int(h * cos_a + w * sin_a)
        M[0, 2] += (new_w / 2) - cx
        M[1, 2] += (new_h / 2) - cy

        return cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0))

    # ------------------------------------------------------------------ #
    #  Crop region helpers                                                 #
    # ------------------------------------------------------------------ #

    def set_circle_crop(self, x: int, y: int, radius: int):
        self.crop_type = "circle"
        self.crop_params = {"x": x, "y": y, "radius": radius}

    def set_rect_crop(self, x: int, y: int, width: int, height: int):
        """Store actual detected width & height — no forced square."""
        self.crop_type = "rectangle"
        self.crop_params = {"x": x, "y": y, "width": width, "height": height}

    def adjust_crop(self, dx: int, dy: int):
        if not self.crop_params:
            return
        self.crop_params["x"] = self.crop_params.get("x", 0) + dx
        self.crop_params["y"] = self.crop_params.get("y", 0) + dy

    def resize_crop(self, delta: int):
        if self.crop_type == "circle":
            r = self.crop_params.get("radius", 10)
            self.crop_params["radius"] = max(10, r + delta)
        elif self.crop_type == "rectangle":
            self.crop_params["width"]  = max(20, self.crop_params.get("width",  20) + delta)
            self.crop_params["height"] = max(20, self.crop_params.get("height", 20) + delta)

    # ------------------------------------------------------------------ #
    #  Cropping                                                            #
    # ------------------------------------------------------------------ #

    def apply_crop(self) -> np.ndarray:
        """Apply crop — saves step to history first."""
        if self.current is None:
            raise RuntimeError("No image loaded.")
        self._push_history()
        if self.crop_type == "circle":
            result = self._apply_circle_crop(self.current, self.crop_params)
        elif self.crop_type == "rectangle":
            result = self._apply_rect_crop(self.current, self.crop_params)
        elif self.crop_type == "polygon":
            result = self._apply_polygon_crop(self.current, self.crop_params)
        else:
            result = self.current.copy()

        self.original    = result.copy()
        self.current     = result
        self.angle       = 0.0
        self.crop_type   = None
        self.crop_params = {}
        return self.current

    def set_polygon_crop(self, points: list, bezier_pts: list = None):
        """Set polygon crop. bezier_pts = canvas pen_points list of dicts."""
        self.crop_type   = "polygon"
        self.crop_params = {"points": points, "bezier_pts": bezier_pts or []}

    @staticmethod
    def _apply_polygon_crop(image: np.ndarray, params: dict) -> np.ndarray:
        """Crop with smooth Catmull-Rom curve + anti-aliased feathered mask."""
        pts     = params.get("points", [])
        bez_pts = params.get("bezier_pts", [])
        if len(pts) < 3:
            return image.copy()
        h, w = image.shape[:2]

        smooth = []
        n      = len(pts)
        STEPS  = 120   # high resolution

        # Check if any bezier handles are non-zero
        has_handles = (bez_pts and len(bez_pts) == n and
                       any(b["cp_out"] != (0,0) or b["cp_in"] != (0,0)
                           for b in bez_pts))

        if has_handles:
            for i in range(n):
                curr = bez_pts[i]; nxt = bez_pts[(i+1)%n]
                ax,ay = curr["pt"]; bx,by = nxt["pt"]
                ox1,oy1 = curr["cp_out"]; ox2,oy2 = nxt["cp_in"]
                cp1x,cp1y = ax+ox1, ay+oy1
                cp2x,cp2y = bx+ox2, by+oy2
                for s in range(STEPS):
                    t = s/STEPS; mt = 1-t
                    x = mt**3*ax+3*mt**2*t*cp1x+3*mt*t**2*cp2x+t**3*bx
                    y = mt**3*ay+3*mt**2*t*cp1y+3*mt*t**2*cp2y+t**3*by
                    smooth.append([int(np.clip(x,0,w-1)), int(np.clip(y,0,h-1))])
        else:
            # Catmull-Rom — closed smooth curve
            for i in range(n):
                p0 = pts[(i-1)%n]; p1 = pts[i]
                p2 = pts[(i+1)%n]; p3 = pts[(i+2)%n]
                for s in range(STEPS):
                    t = s/STEPS; tt=t*t; ttt=tt*t
                    x = 0.5*((2*p1[0])+(-p0[0]+p2[0])*t+
                              (2*p0[0]-5*p1[0]+4*p2[0]-p3[0])*tt+
                              (-p0[0]+3*p1[0]-3*p2[0]+p3[0])*ttt)
                    y = 0.5*((2*p1[1])+(-p0[1]+p2[1])*t+
                              (2*p0[1]-5*p1[1]+4*p2[1]-p3[1])*tt+
                              (-p0[1]+3*p1[1]-3*p2[1]+p3[1])*ttt)
                    smooth.append([int(np.clip(x,0,w-1)), int(np.clip(y,0,h-1))])

        poly  = np.array(smooth, dtype=np.int32)
        mask  = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)

        # Anti-aliased feather
        mask_f = cv2.GaussianBlur(mask, (7, 7), 1.5)
        alpha  = mask_f.astype(np.float32) / 255.0
        result = image.copy()
        for c in range(3):
            result[:,:,c] = (image[:,:,c]*alpha + 255*(1-alpha)).astype(np.uint8)

        xs=poly[:,0]; ys=poly[:,1]
        x1,y1 = max(0,xs.min()), max(0,ys.min())
        x2,y2 = min(w,xs.max()), min(h,ys.max())
        return result[y1:y2, x1:x2].copy()

    @staticmethod
    def _apply_circle_crop(image: np.ndarray, params: dict) -> np.ndarray:
        x, y, r = params["x"], params["y"], params["radius"]
        h, w = image.shape[:2]

        # Create circular mask
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)

        # Apply mask (background → white)
        result = image.copy()
        result[mask == 0] = 255

        # Tight bounding box
        x1 = max(0, x - r)
        y1 = max(0, y - r)
        x2 = min(w, x + r)
        y2 = min(h, y + r)
        return result[y1:y2, x1:x2]

    @staticmethod
    def _apply_rect_crop(image: np.ndarray, params: dict) -> np.ndarray:
        """Crop rotated rectangle — straighten label + fill bg with detected color."""
        x, y   = params["x"],     params["y"]
        bw, bh = params["width"], params["height"]
        angle  = params.get("angle", 0.0)
        h, w   = image.shape[:2]

        if abs(angle) < 0.1:
            # Simple axis-aligned crop
            x1 = max(0, x);    y1 = max(0, y)
            x2 = min(w, x+bw); y2 = min(h, y+bh)
            return image[y1:y2, x1:x2].copy()

        # ── Rotated crop — deskew label to straight ───────────────────
        cx = x + bw / 2.0
        cy = y + bh / 2.0

        # Rotate entire image to deskew label
        M   = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
        rot = cv2.warpAffine(image, M, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)

        # Crop the now-straight label
        x1 = max(0, int(cx - bw/2))
        y1 = max(0, int(cy - bh/2))
        x2 = min(w, int(cx + bw/2))
        y2 = min(h, int(cy + bh/2))
        cropped = rot[y1:y2, x1:x2].copy()

        # ── Background detection + fill ───────────────────────────────
        # Sample corners of original image to detect background color
        corners = []
        for px, py in [(0,0),(w-1,0),(0,h-1),(w-1,h-1),
                       (w//2,0),(0,h//2),(w-1,h//2),(w//2,h-1)]:
            corners.append(image[py, px].astype(float))
        bg_color = np.median(corners, axis=0).astype(np.uint8)

        # Create output canvas with background color
        out_h = cropped.shape[0]
        out_w = cropped.shape[1]

        # Add padding around label for clean output
        pad = 20
        canvas = np.zeros((out_h + pad*2, out_w + pad*2, 3), dtype=np.uint8)
        canvas[:] = bg_color
        canvas[pad:pad+out_h, pad:pad+out_w] = cropped

        return canvas

    # ------------------------------------------------------------------ #
    #  Saving                                                              #
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        """Save self.current to *path* (format detected from extension)."""
        if self.current is None:
            raise RuntimeError("No image to save.")
        success = cv2.imwrite(path, self.current)
        if not success:
            raise OSError(f"Failed to write image to: {path}")

    # ------------------------------------------------------------------ #
    #  Utility                                                             #
    # ------------------------------------------------------------------ #

    @property
    def is_loaded(self) -> bool:
        return self.current is not None

    def get_size(self) -> Tuple[int, int]:
        if self.current is None:
            return (0, 0)
        h, w = self.current.shape[:2]
        return (w, h)