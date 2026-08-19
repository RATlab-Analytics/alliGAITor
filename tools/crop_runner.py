"""
GUI-side handle for a crop-folder run: the actual work happens in a
separate OS process, polled via a QTimer rather than a QThread -- see
crop_worker_process.py's docstring for why.

Supports two modes (mutually exclusive):
  - uniform (x, y): same crop position for every video in input_folder
    (crop_worker_process.run_crop_worker).
  - positions=[(video_path, x, y), ...]: a specific position per video
    (crop_worker_process.run_crop_worker_positions) -- used by
    CropSetupDialog's "use this position for all remaining" button.

Ported from RATlab-NOR's gui/crop_runner.py unchanged apart from the
nor_classifier_dir -> tools_dir rename (see crop_worker_process.py).

RATlab-NOR is MIT-licensed (Copyright (c) 2026 Mitchell Carson); see
../THIRD_PARTY_NOTICES.md for the full license text.
"""

from __future__ import annotations

import multiprocessing as mp

from PySide6.QtCore import QObject, QTimer, Signal

from crop_worker_process import run_crop_worker, run_crop_worker_positions


class CropRunner(QObject):
    log = Signal(str)
    progress = Signal(int, int)          # done, total
    video_done = Signal(str)             # video_path_str -- only emitted in positions mode
    finished_run = Signal(str, str)      # status ("done"/"failed"/"canceled"), message
    finished = Signal()                  # fires once the worker process has fully exited

    def __init__(self, tools_dir, input_folder, output_folder, width, height,
                 x=None, y=None, positions=None, parent=None, color_grade=False,
                 color_grade_strength=1.0):
        super().__init__(parent)
        if positions is None and (x is None or y is None):
            raise ValueError("CropRunner needs either (x, y) or positions=[...]")

        self._tools_dir = str(tools_dir)
        self._input_folder = str(input_folder)
        self._output_folder = str(output_folder)
        self._width, self._height = width, height
        self._x, self._y = x, y
        self._positions = positions
        self._color_grade = color_grade
        self._color_grade_strength = color_grade_strength

        self._stop_event = mp.Event()
        self._queue = mp.Queue()
        self._process: mp.Process | None = None
        self._saw_finished_run = False

        self._timer = QTimer(self)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self._poll)

    def start(self):
        if self._positions is not None:
            target = run_crop_worker_positions
            args = (
                self._tools_dir, self._input_folder, self._output_folder,
                self._positions, self._width, self._height, self._queue, self._stop_event,
                self._color_grade, self._color_grade_strength,
            )
        else:
            target = run_crop_worker
            args = (
                self._tools_dir, self._input_folder, self._output_folder,
                self._x, self._y, self._width, self._height, self._queue, self._stop_event,
                self._color_grade, self._color_grade_strength,
            )
        self._process = mp.Process(target=target, args=args, daemon=True)
        self._process.start()
        self._timer.start()

    def request_stop(self):
        self._stop_event.set()

    def wait(self, timeout_ms: int = 5000):
        if self._process is not None:
            self._process.join(timeout=timeout_ms / 1000)

    def _poll(self):
        while True:
            try:
                msg = self._queue.get_nowait()
            except Exception:
                break
            self._dispatch(msg)

        if self._process is not None and not self._process.is_alive():
            if not self._saw_finished_run:
                self.log.emit("Crop worker process exited unexpectedly.")
            self._timer.stop()
            self._process.join(timeout=2)
            self.finished.emit()

    def _dispatch(self, msg):
        kind = msg[0]
        if kind == "log":
            self.log.emit(msg[1])
        elif kind == "progress":
            self.progress.emit(msg[1], msg[2])
        elif kind == "video_done":
            self.video_done.emit(msg[1])
        elif kind == "finished":
            self._saw_finished_run = True
            self.finished_run.emit(msg[1], msg[2])
