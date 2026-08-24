"""
Qt table model wrapping a JobQueue, so the main window's table view stays
in sync with the underlying job list without duplicating state.

Ported from RATlab-NOR's gui/job_table_model.py, with alliGAITor's two
setup-gate statuses (NEEDS_CONFIG, NEEDS_CROP) added and "videos" renamed
to "sessions" (progress here is tracked per session, not per video --
see job_queue.Job).
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
    JobStatus.RUNNING: "Running…",
    JobStatus.DONE: "Done",
    JobStatus.FAILED: "Failed",
    JobStatus.CANCELED: "Canceled",
}

_STATUS_COLORS = {
    JobStatus.NEEDS_CONFIG: QColor("#8a6d00"),
    JobStatus.NEEDS_CROP: QColor("#8a6d00"),
    JobStatus.RUNNING: QColor("#1565c0"),
    JobStatus.DONE: QColor("#2e7d32"),
    JobStatus.FAILED: QColor("#c0392b"),
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
        """Call after the underlying job_queue.jobs list or any job's
        fields change, to repaint the view."""
        self.beginResetModel()
        self.endResetModel()
