"""
GUI-side handle for a batch run of the job queue. Delegates the actual
pipeline work to a separate OS process
(``batch_worker_process.run_batch_worker``) rather than a thread in this
process -- see that module's docstring for why.

Ported from RATlab-NOR's gui/batch_runner.py, with "video" progress
renamed to "session" (alliGAITor's own unit of run progress). The
``progress`` signal below is NOR's same tqdm-redraw mechanism, carried
over for the same reason NOR needed it: without it, the live progress a
long inference run is actually making (see
:mod:`alligaitor.subprocess_streaming`) has nowhere to go, and a
multi-minute session looks hung rather than working.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import List, Optional

from PySide6.QtCore import QObject, QTimer, Signal

from batch_worker_process import run_batch_worker
from job_queue import Job


class BatchRunner(QObject):
    log = Signal(str)
    progress = Signal(str)                   # same redrawing line updating -- see main_window.py's _on_progress_line
    progress_closed = Signal()               # a redrawn line's final state was just sent -- start the next one fresh
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

    def wait(self, timeout_ms: int = 5000):
        """Mirrors QThread.wait() -- used by MainWindow.closeEvent."""
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
            if not self._saw_all_finished:
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
