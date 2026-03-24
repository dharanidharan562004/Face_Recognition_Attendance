from __future__ import annotations
"""
file_handler.py
JPG/JPEG only. Uses system-native file dialogs.
"""

from typing import Optional, List, Tuple, Dict
import os
import cv2
from PyQt5.QtWidgets import QFileDialog

SUPPORTED_READ  = "JPEG Images (*.jpg *.jpeg)"
SUPPORTED_WRITE = "JPEG Image (*.jpg *.jpeg)"
VALID_EXTENSIONS = (".jpg", ".jpeg")


class InvalidFormatError(Exception):
    pass


def open_image_dialog(parent=None) -> "Optional[str]":
    """Open native system file browser."""
    dialog = QFileDialog(parent, "Open Coin Image")
    dialog.setFileMode(QFileDialog.ExistingFile)
    dialog.setNameFilter(SUPPORTED_READ)
    dialog.setOption(QFileDialog.DontUseNativeDialog, True)  # native OS dialog
    if dialog.exec_():
        files = dialog.selectedFiles()
        return files[0] if files else None
    return None


def save_image_dialog(parent=None, suggested_name: str = "coin") -> "Optional[str]":
    """Open native system save dialog."""
    dialog = QFileDialog(parent, "Save Processed Image")
    dialog.setAcceptMode(QFileDialog.AcceptSave)
    dialog.setNameFilter(SUPPORTED_WRITE)
    dialog.setOption(QFileDialog.DontUseNativeDialog, True)  # native OS dialog
    dialog.selectFile(suggested_name)
    if dialog.exec_():
        files = dialog.selectedFiles()
        return files[0] if files else None
    return None


def validate_format(path: str) -> None:
    ext = os.path.splitext(path)[1].lower()
    if ext not in VALID_EXTENSIONS:
        raise InvalidFormatError(
            f"Invalid format: '{ext or '(no extension)'}'\n\n"
            f"Only JPG / JPEG images are supported.\n"
            f"Please select a file ending in  .jpg  or  .jpeg"
        )


def ensure_extension(path: str, default_ext: str = ".jpg") -> str:
    _, ext = os.path.splitext(path)
    if ext.lower() not in VALID_EXTENSIONS:
        return path + ".jpg"
    return path


def save_image(image, path: str) -> None:
    params  = [cv2.IMWRITE_JPEG_QUALITY, 95]
    success = cv2.imwrite(path, image, params)
    if not success:
        raise OSError(f"Failed to write image to: {path}")