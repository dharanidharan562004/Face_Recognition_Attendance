# Coin Image Processing Tool

A standalone desktop application for aligning, rotating, and cropping coin images.
Built with **Python**, **OpenCV**, and **PyQt5**.

---

## Features

| Feature | Description |
|---|---|
| Image Loading | Open JPG / PNG / JPEG coin images |
| Rotation | +1 / −1 degree increments (clockwise / counter-clockwise) |
| Guide Lines | Cross-hair overlay to help align the coin |
| Circle Crop | Auto-detects coin boundary via HoughCircles |
| Rectangle Crop | Auto-detects bounding box via contour analysis |
| Manual Adjustment | Arrow keys move the crop; +/− resize it |
| Crop Confirmation | Press **Enter** (or click Apply Crop) to finalize |
| QR / Label Detection | Highlights QR code or label region; decodes data |
| File Naming | Enter filename manually (auto-filled from QR decode) |
| Save | Export as JPG or PNG |

---

## Installation

### Prerequisites

- Python 3.10+
- pip

### Steps

```bash
# 1. Clone or download the project
cd coin_image_processor

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
python main.py
```

> **Note:** `pyzbar` requires the `zbar` native library.
> - **Ubuntu/Debian:** `sudo apt-get install libzbar0`
> - **macOS:** `brew install zbar`
> - **Windows:** Download the DLL from the pyzbar GitHub page.

---

## Usage

### Basic Workflow

1. **Open** a coin image (`Ctrl+O` or File → Open Image).
2. Use **↺ CCW / ↻ CW** buttons (or `Ctrl+Left/Right`) to rotate the coin.
3. Click **⭕ Circle** or **▭ Rectangle** to auto-detect the crop boundary.
4. Fine-tune with **arrow keys** (move) and **+/−** keys (resize).
5. Press **Enter** (or **✔ Apply Crop**) to finalize the crop.
6. Click **🔍 Detect QR/Label** to identify and read the coin's label.
7. Enter (or confirm) the filename in the **Save** panel.
8. Click **💾 Save Image** (`Ctrl+S`) to export.

### Keyboard Shortcuts

| Key | Action |
|---|---|
| `Ctrl+O` | Open image |
| `Ctrl+S` | Save image |
| `Ctrl+Z` | Reset to original |
| `Ctrl+Left` | Rotate −1° |
| `Ctrl+Right` | Rotate +1° |
| `Arrow keys` | Move crop region |
| `+` / `−` | Resize crop region |
| `Enter` | Apply crop |

---

## Project Structure

```
coin_image_processor/
├── main.py                     # Entry point
├── requirements.txt            # Python dependencies
├── README.md
│
├── gui/
│   ├── __init__.py
│   ├── main_window.py          # Main application window
│   └── canvas_widget.py        # Image display + overlay canvas
│
├── processing/
│   ├── __init__.py
│   ├── image_processor.py      # Rotation, cropping, saving
│   ├── coin_detector.py        # OpenCV coin boundary detection
│   └── qr_detector.py          # QR code / label detection
│
└── utils/
    ├── __init__.py
    └── file_handler.py         # File dialogs and path helpers
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `opencv-python` | Image processing (rotation, detection, masking) |
| `numpy` | Array operations |
| `PyQt5` | Desktop GUI framework |
| `Pillow` | Supplementary image loading |
| `pyzbar` | Optional — QR/barcode decoding fallback |

---

## Future Enhancements (from SRS §5)

- [ ] Automatic coin detection (no manual boundary needed)
- [ ] Automatic QR code reading and filename population
- [ ] Batch image processing
- [ ] AI-based coin alignment
- [ ] Automatic filename generation

---

## License

Open-source — MIT License.
