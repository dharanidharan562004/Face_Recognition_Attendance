"""
main.py — Entry point for the Coin Image Processing Tool.
Usage: python main.py 🗂
"""

import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore    import Qt
from gui.main_window import MainWindow


def main():
    # High-DPI support
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps,    True)

    app = QApplication(sys.argv)
    app.setApplicationName("Coin Image Processing Tool")
    app.setOrganizationName("OpenSource")

    window = MainWindow()
    window.show()                  # normal start
    # window.showMaximized()       # uncomment to start maximized

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()