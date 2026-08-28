# RATlab alliGAITor: an open-source rodent gait analysis pipeline for research
# Copyright (C) 2026 Mitchell Carson
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Qt table model wrapping a JobQueue, so the main window's table view stays
in sync with the underlying job list without duplicating state.
"""

from __future__ import annotations

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QColor

from job_queue import Job, JobStatus

COLUMNS = ["Group", "Status", "Sessions", "Input Folder", "Output Folder"]

_STATUS_LABELS = {
    JobStatus.NEEDS_CONFIG: "Needs config",
    JobStatus.NEEDS_CROP: "Needs crop",
    JobStatus.READY: "Ready",
    JobStatus.QUEUED: "Queued",
    JobStatus.RUNNING: "Running…",
    JobStatus.DONE: "Done",
    JobStatus.FAILED: "Failed",
    JobStatus.CANCELED: "Canceled",
}

# Brightened for readable contrast against the app's dark palette.
_STATUS_COLORS = {
    JobStatus.NEEDS_CONFIG: QColor("#ffca28"),
    JobStatus.NEEDS_CROP: QColor("#ffca28"),
    JobStatus.QUEUED: QColor("#64b5f6"),
    JobStatus.RUNNING: QColor("#64b5f6"),
    JobStatus.DONE: QColor("#81c995"),
    JobStatus.FAILED: QColor("#f28b82"),
}


class JobTableModel(QAbstractTableModel):
    def __init__(self, job_queue, parent=None):
        super().__init__(parent)
        self.job_queue = job_queue

    # -- required QAbstractTableModel overrides --

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.job_queue.jobs)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(COLUMNS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole or orientation != Qt.Horizontal:
            return None
        return COLUMNS[section]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        job = self.job_queue.jobs[index.row()]
        col = index.column()

        if role == Qt.DisplayRole:
            if col == 0:
                return job.group_name
            if col == 1:
                label = _STATUS_LABELS.get(job.status, job.status.value)
                if job.status == JobStatus.FAILED and job.error_message:
                    return f"{label}  ({job.error_message})"
                return label
            if col == 2:
                if job.sessions_total:
                    return f"{job.sessions_done}/{job.sessions_total}"
                return "—"
            if col == 3:
                return job.input_folder
            if col == 4:
                return str(job.output_folder)

        if role == Qt.ForegroundRole and col == 1:
            return _STATUS_COLORS.get(job.status)

        return None

    # -- helpers for the main window --

    def job_at(self, row) -> Job:
        return self.job_queue.jobs[row]

    def refresh(self):
        """Call after the underlying jobs list or any job's fields change."""
        self.beginResetModel()
        self.endResetModel()
