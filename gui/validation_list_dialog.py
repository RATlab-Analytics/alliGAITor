"""
Validation list: one row per session in a completed job, showing each
paw's used-run duration (green if usable, red if not -- see
alligaitor.validation.effective_usability) and the fraction of paws
usable, so a reviewer can spot problem sessions at a glance before
double-clicking into the video (validation_video_dialog.py).

Opened by double-clicking a DONE job in main_window.py's table, or via
its "View Validation..." context-menu action.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView, QMessageBox,
)

from alligaitor import validation
from alligaitor.config import PipelineConfig
from paw_colors import PAW_SHORT_LABELS, COLOR_USABLE, COLOR_UNUSABLE, ordered_paws
from validation_video_dialog import ValidationVideoDialog

_COLOR_NEUTRAL = QColor(150, 150, 150)

_COLUMNS = ["Session"] + [PAW_SHORT_LABELS[p] for p in ordered_paws()] + ["Usable"]


class ValidationListDialog(QDialog):
    def __init__(self, job, parent=None):
        super().__init__(parent)
        self.job = job
        self._rows = []  # row index -> dict, see _refresh()

        self.setWindowTitle(f"Validation -- {job.group_name}")
        self.resize(760, 420)

        try:
            self.config = PipelineConfig.from_yaml(job.config_path)
        except Exception as exc:
            self.config = None
            self._load_error = str(exc)
        else:
            self._load_error = None

        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(_COLUMNS)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for col in range(1, len(_COLUMNS)):
            header.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.doubleClicked.connect(self._on_open_selected)

        self.open_btn = QPushButton("Open Video…")
        self.close_btn = QPushButton("Close")
        self.open_btn.clicked.connect(self._on_open_selected)
        self.close_btn.clicked.connect(self.accept)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(self.open_btn)
        btn_row.addWidget(self.close_btn)

        legend = QLabel(
            f'<span style="color:{COLOR_USABLE.name()};">green = usable</span> &nbsp;&nbsp; '
            f'<span style="color:{COLOR_UNUSABLE.name()};">red = not usable</span>'
        )

        layout = QVBoxLayout(self)
        layout.addWidget(legend)
        layout.addWidget(self.table)
        layout.addLayout(btn_row)

        self._refresh()

    def _refresh(self):
        self._rows = []
        if self.config is None:
            self.table.setRowCount(1)
            self.table.setItem(0, 0, QTableWidgetItem(f"Couldn't load config: {self._load_error}"))
            for col in range(1, len(_COLUMNS)):
                self.table.setItem(0, col, QTableWidgetItem(""))
            self.open_btn.setEnabled(False)
            return

        for session in self.config.sessions:
            summary_path = Path(session.output_dir) / f"{session.name}.validation_summary.json"
            flags_path = Path(session.output_dir) / f"{session.name}.manual_flags.json"
            video_path = Path(self.job.validation_dir) / f"{session.name}.validation.mp4"

            summary = validation.load_validation_summary(summary_path)
            flagged_paws, _note = validation.load_manual_flags(flags_path)
            usability = validation.effective_usability(summary, flagged_paws) if summary is not None else None

            self._rows.append({
                "session": session,
                "summary": summary,
                "usability": usability,
                "video_path": video_path,
            })

        if not self._rows:
            self.table.setRowCount(1)
            self.table.setItem(0, 0, QTableWidgetItem("No sessions in this job."))
            for col in range(1, len(_COLUMNS)):
                self.table.setItem(0, col, QTableWidgetItem(""))
            self.open_btn.setEnabled(False)
            return

        self.table.setRowCount(len(self._rows))
        for i, row in enumerate(self._rows):
            session = row["session"]
            summary = row["summary"]
            usability = row["usability"]

            if usability is None:
                session_item = QTableWidgetItem(f"{session.name}  (no validation data yet)")
                session_item.setForeground(QBrush(_COLOR_NEUTRAL))
                self.table.setItem(i, 0, session_item)
                for col in range(1, len(_COLUMNS)):
                    item = QTableWidgetItem("—")
                    item.setForeground(QBrush(_COLOR_NEUTRAL))
                    self.table.setItem(i, col, item)
                continue

            all_usable = all(usability.values())
            title_color = COLOR_USABLE if all_usable else COLOR_UNUSABLE
            session_item = QTableWidgetItem(session.name)
            session_item.setForeground(QBrush(title_color))
            self.table.setItem(i, 0, session_item)

            for col, paw in enumerate(ordered_paws(), start=1):
                window = summary.get("paws", {}).get(paw)
                text = f"{window['duration_s']:.2f}s" if window else "0.00s"
                color = COLOR_USABLE if usability[paw] else COLOR_UNUSABLE
                item = QTableWidgetItem(text)
                item.setForeground(QBrush(color))
                self.table.setItem(i, col, item)

            n_usable = sum(usability.values())
            usable_item = QTableWidgetItem(f"{n_usable}/{len(usability)}")
            usable_item.setForeground(QBrush(COLOR_USABLE if all_usable else COLOR_UNUSABLE))
            self.table.setItem(i, len(_COLUMNS) - 1, usable_item)

        self.open_btn.setEnabled(True)

    def _selected_row(self):
        rows = sorted({idx.row() for idx in self.table.selectionModel().selectedRows()})
        if not rows or not self._rows or rows[0] >= len(self._rows):
            return None
        return self._rows[rows[0]]

    def _on_open_selected(self):
        row = self._selected_row()
        if row is None:
            return
        if row["summary"] is None:
            QMessageBox.information(
                self, "No validation data",
                f"'{row['session'].name}' hasn't produced validation data yet -- run this job "
                f"(with validation video export enabled) first.",
            )
            return
        if not row["video_path"].exists():
            QMessageBox.information(
                self, "No validation video",
                f"No validation video found for '{row['session'].name}' at {row['video_path']}.",
            )
            return
        dialog = ValidationVideoDialog(self.job, row["session"], self.config.output_xlsx, parent=self)
        dialog.exec()
        self._refresh()
