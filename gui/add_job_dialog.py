"""
Load-job dialog: folder pickers for input videos and output destination,
a live "N videos found" readout, and validation against filesystem-unsafe
or duplicate group names. Per-job config (regex, camera roles,
calibration) is handled afterward by group_config_dialog.py.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Set

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit,
    QPushButton, QLabel, QDialogButtonBox, QMessageBox, QFileDialog, QWidget,
)

from alligaitor.discovery import find_videos

# Characters that can't safely appear in a folder name on macOS or Windows.
_UNSAFE_NAME_CHARS = re.compile(r'[\\/:*?"<>|]')


def _folder_row(initial_text: str = ""):
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    edit = QLineEdit(initial_text)
    edit.setReadOnly(True)
    browse_btn = QPushButton("Browse…")
    layout.addWidget(edit)
    layout.addWidget(browse_btn)
    return row, edit, browse_btn


class AddJobDialog(QDialog):
    def __init__(self, default_output_base: str, existing_group_names: Set[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Load Job")
        self.setMinimumWidth(480)

        self._existing_group_names = {n.lower() for n in existing_group_names}
        self.result_job_kwargs: Optional[dict] = None

        self.group_name_edit = QLineEdit()
        self.group_name_edit.setPlaceholderText("e.g. Cohort 1 (saline)")

        input_row, self.input_folder_edit, input_browse_btn = _folder_row()
        output_row, self.output_base_edit, output_browse_btn = _folder_row(default_output_base)

        self.video_count_label = QLabel("Pick an input folder to see how many videos it contains.")
        self.video_count_label.setStyleSheet("color: #9d9d9d;")

        form = QFormLayout()
        form.addRow("Group name:", self.group_name_edit)
        form.addRow("Input video folder:", input_row)
        form.addRow("", self.video_count_label)
        form.addRow("Output base folder:", output_row)
        out_hint = QLabel("Results will be written to <output base>/<group name>/")
        out_hint.setStyleSheet("color: #9d9d9d; font-size: 11px;")
        form.addRow("", out_hint)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Load Job")
        self.buttons.accepted.connect(self._on_accept)
        self.buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.buttons)

        input_browse_btn.clicked.connect(self._browse_input_folder)
        output_browse_btn.clicked.connect(self._browse_output_folder)

    # -- browsing --

    def _update_video_count_label(self, folder):
        videos = find_videos(folder)
        if not videos:
            self.video_count_label.setText("No .mp4 files found in this folder (it's searched recursively).")
            self.video_count_label.setStyleSheet("color: #f28b82;")
        else:
            self.video_count_label.setText(f"{len(videos)} video(s) found.")
            self.video_count_label.setStyleSheet("color: #81c995;")

    def _browse_input_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Choose the folder of videos for this group", self.input_folder_edit.text() or str(Path.home()),
        )
        if not folder:
            return
        self.input_folder_edit.setText(folder)
        self._update_video_count_label(folder)
        if not self.group_name_edit.text().strip():
            self.group_name_edit.setText(Path(folder).name)

    def _browse_output_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Choose where this group's results should be written",
            self.output_base_edit.text() or str(Path.home()),
        )
        if folder:
            self.output_base_edit.setText(folder)

    # -- validation + accept --

    def _on_accept(self):
        group_name = self.group_name_edit.text().strip()
        input_folder = self.input_folder_edit.text().strip()
        output_base = self.output_base_edit.text().strip()

        if not group_name or not input_folder or not output_base:
            QMessageBox.warning(self, "Missing info", "Group name, input folder, and output folder are all required.")
            return

        if _UNSAFE_NAME_CHARS.search(group_name):
            QMessageBox.warning(
                self, "Invalid group name",
                'Group name can\'t contain any of: \\ / : * ? " < > |\n'
                "(it's used directly as an output folder name).",
            )
            return

        if group_name.lower() in self._existing_group_names:
            QMessageBox.warning(self, "Duplicate group", f"A job named '{group_name}' is already in the queue.")
            return

        if not Path(input_folder).is_dir():
            QMessageBox.warning(self, "Folder not found", f"Input folder doesn't exist:\n{input_folder}")
            return

        if not find_videos(input_folder):
            reply = QMessageBox.question(
                self, "No videos found",
                f"No .mp4 files were found under:\n{input_folder}\n\nAdd this job anyway?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        self.result_job_kwargs = dict(
            group_name=group_name,
            input_folder=input_folder,
            output_folder=str(Path(output_base) / group_name),
        )
        self.accept()
