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
GUI-side handle for a batch run of the job queue. Delegates pipeline work
to a separate OS process (``batch_worker_process.run_batch_worker``)
rather than a thread in this process.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
from typing import List, Optional

from PySide6.QtCore import QObject, QTimer, Signal

from batch_worker_process import run_batch_worker
from job_queue import Job


class BatchRunner(QObject):
    log = Signal(str)
    progress = Signal(str)                   # redrawing progress line
    progress_closed = Signal()               # redrawn line's final state was sent
    job_started = Signal(str)               # job_id
    job_progress = Signal(str, int, int)     # job_id, sessions_done, sessions_total
    job_finished = Signal(str, str, str)     # job_id, status, message
    all_finished = Signal()
    finished = Signal()  # fires once the worker process has fully exited

    def __init__(self, jobs: List[Job], repo_dir, device: str = "auto", tracking: bool = False, parent=None):
        super().__init__(parent)
        self._job_dicts = [j.to_dict() for j in jobs]
        self._repo_dir = str(repo_dir)
        self._device = device
        self._tracking = tracking
        self._stop_event = mp.Event()
        self._queue: mp.Queue = mp.Queue()
        self._process: Optional[mp.Process] = None
        self._saw_all_finished = False
        # Set by force_stop() so _poll can tell "killed on purpose" from "crashed".
        self.force_stopped = False

        self._timer = QTimer(self)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self._poll)

    def start(self):
        self._process = mp.Process(
            target=run_batch_worker,
            args=(self._repo_dir, self._job_dicts, self._queue, self._stop_event, self._device, self._tracking),
            daemon=True,
        )
        self._process.start()
        self._timer.start()

    def request_stop(self):
        self._stop_event.set()

    def force_stop(self):
        """Kills the worker process (and its process group on POSIX,
        reaching any `sleap-nn predict` subprocess) immediately, abandoning
        whatever job is in flight. See request_stop() for the graceful
        alternative."""
        self.force_stopped = True
        if self._process is None or not self._process.is_alive():
            return
        pid = self._process.pid
        if pid is None:
            return
        if hasattr(os, "killpg"):
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
                return
            except (ProcessLookupError, PermissionError, OSError):
                pass  # fall through to killing just this process
        self._process.kill()

    def wait(self, timeout_ms: int = 5000):
        """Mirrors QThread.wait()."""
        if self._process is not None:
            self._process.join(timeout=timeout_ms / 1000)

    # -- internal --

    def _poll(self):
        while True:
            try:
                msg = self._queue.get_nowait()
            except Exception:
                break
            self._dispatch(msg)

        if self._process is not None and not self._process.is_alive():
            if not self._saw_all_finished and not self.force_stopped:
                self.log.emit("Batch worker process exited unexpectedly.")
            self._timer.stop()
            self._process.join(timeout=2)
            self.finished.emit()

    def _dispatch(self, msg):
        kind = msg[0]
        if kind == "log":
            self.log.emit(msg[1])
        elif kind == "progress":
            self.progress.emit(msg[1])
        elif kind == "progress_closed":
            self.progress_closed.emit()
        elif kind == "job_started":
            self.job_started.emit(msg[1])
        elif kind == "job_progress":
            self.job_progress.emit(msg[1], msg[2], msg[3])
        elif kind == "job_finished":
            self.job_finished.emit(msg[1], msg[2], msg[3])
        elif kind == "all_finished":
            self._saw_all_finished = True
            self.all_finished.emit()
