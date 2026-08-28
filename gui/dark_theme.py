"""
App-wide dark theme, applied once to the QApplication instance so every
window and dialog picks it up automatically. Uses the "Fusion" style,
since native styles largely ignore QPalette colors for many widgets.
"""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QStyleFactory

_WINDOW = QColor(45, 45, 45)
_BASE = QColor(30, 30, 30)
_ALTERNATE_BASE = QColor(45, 45, 45)
_TEXT = QColor(212, 212, 212)
_DISABLED_TEXT = QColor(127, 127, 127)
_BUTTON = QColor(53, 53, 53)
_HIGHLIGHT = QColor(42, 130, 218)
_LINK = QColor(100, 170, 255)


def apply_dark_theme(app: QApplication) -> None:
    app.setStyle(QStyleFactory.create("Fusion"))

    palette = QPalette()
    palette.setColor(QPalette.Window, _WINDOW)
    palette.setColor(QPalette.WindowText, _TEXT)
    palette.setColor(QPalette.Base, _BASE)
    palette.setColor(QPalette.AlternateBase, _ALTERNATE_BASE)
    palette.setColor(QPalette.ToolTipBase, _WINDOW)
    palette.setColor(QPalette.ToolTipText, _TEXT)
    palette.setColor(QPalette.Text, _TEXT)
    palette.setColor(QPalette.Button, _BUTTON)
    palette.setColor(QPalette.ButtonText, _TEXT)
    palette.setColor(QPalette.BrightText, QColor(255, 90, 90))
    palette.setColor(QPalette.Link, _LINK)
    palette.setColor(QPalette.LinkVisited, _LINK)
    palette.setColor(QPalette.Highlight, _HIGHLIGHT)
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    palette.setColor(QPalette.PlaceholderText, _DISABLED_TEXT)

    # Disabled-state colors; defaults assume a light theme.
    palette.setColor(QPalette.Disabled, QPalette.WindowText, _DISABLED_TEXT)
    palette.setColor(QPalette.Disabled, QPalette.Text, _DISABLED_TEXT)
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, _DISABLED_TEXT)
    palette.setColor(QPalette.Disabled, QPalette.Highlight, QColor(80, 80, 80))
    palette.setColor(QPalette.Disabled, QPalette.HighlightedText, _DISABLED_TEXT)

    app.setPalette(palette)
