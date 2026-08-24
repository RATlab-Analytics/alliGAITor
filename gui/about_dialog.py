"""
About dialog: GPLv3 minimum notice, app version, and installed versions
of key dependencies -- the minimum needed to comply with alliGAITor's own
license and the third-party licenses it bundles code from. See
THIRD_PARTY_NOTICES.md (this dialog only points to it, not reproduces it).
"""

from __future__ import annotations

from importlib import metadata
from typing import List

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QPlainTextEdit, QVBoxLayout

import alligaitor

_DEPENDENCIES = [
    "aniposelib", "sleap-nn", "sleap-io", "numpy", "pandas", "openpyxl",
    "PyYAML", "PySide6", "opencv-python", "imageio-ffmpeg",
]


def _dependency_versions() -> List[str]:
    lines = []
    for name in _DEPENDENCIES:
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError:
            version = "not installed"
        lines.append(f"{name}  {version}")
    return lines


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About alliGAITor")
        self.setMinimumWidth(480)

        layout = QVBoxLayout(self)

        title = QLabel(f"<b>alliGAITor</b> v{alligaitor.__version__}")
        layout.addWidget(title)

        notice = QLabel(
            "Copyright (C) 2026 Mitchell Carson\n\n"
            "This program is free software: you can redistribute it and/or modify it "
            "under the terms of the GNU General Public License as published by the Free "
            "Software Foundation, either version 3 of the License, or (at your option) "
            "any later version. This program comes WITHOUT ANY WARRANTY; see the GNU "
            "General Public License for details.\n\n"
            "Portions of alligaitor/calibration.py are adapted from aniposelib "
            "(Copyright (c) 2019-2023, Lili Karashchuk), used under the BSD 2-Clause "
            "License. See THIRD_PARTY_NOTICES.md for the full text and other "
            "third-party attributions."
        )
        notice.setWordWrap(True)
        layout.addWidget(notice)

        layout.addWidget(QLabel("<b>Dependency versions</b>"))
        deps = QPlainTextEdit()
        deps.setReadOnly(True)
        deps.setPlainText("\n".join(_dependency_versions()))
        deps.setMaximumHeight(160)
        layout.addWidget(deps)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.accept)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
