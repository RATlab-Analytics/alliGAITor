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
Runs queued jobs' pipeline (``alligaitor.pipeline.run_group``) in a
separate OS process, one job at a time, so SLEAP-NN inference and heavy
video I/O can't crash the GUI's Qt event loop.

Stop only takes effect between jobs, not mid-job: ``run_group()`` is one
long blocking call per job with no internal cancellation point.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path


def run_batch_worker(repo_dir: str, job_dicts: list, queue, stop_event,
                      device: str = "auto", tracking: bool = False) -> None:
    # New POSIX process group so BatchRunner.force_stop's os.killpg() also
    # reaches any `sleap-nn predict` subprocess, not just this process.
    if hasattr(os, "setpgrp"):
        os.setpgrp()

    repo_dir = Path(repo_dir)
    for p in (repo_dir, repo_dir / "gui"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    from alligaitor.config import PipelineConfig
    from alligaitor.pipeline import run_group
    from job_queue import Job, JobStatus

    def log(message: str):
        queue.put(("log", message))

    def progress(message: str):
        # Live tqdm-style line from inference; sent separately from "log"
        # so the GUI can redraw it in place instead of appending lines.
        queue.put(("progress", message))

    def on_redraw_closed():
        # Redrawn line's final state was just sent; next progress update
        # should start on a fresh line.
        queue.put(("progress_closed",))

    for job_dict in job_dicts:
        job = Job.from_dict(job_dict)

        if stop_event.is_set():
            queue.put(("job_finished", job.id, JobStatus.CANCELED.value, "Stopped before it started."))
            continue

        queue.put(("job_started", job.id))
        log(f"[{job.group_name}] starting...")

        try:
            config = PipelineConfig.from_yaml(job.config_path)
        except Exception as exc:
            message = f"Couldn't load {job.config_path}: {exc}"
            log(f"[{job.group_name}] {message}")
            queue.put(("job_finished", job.id, JobStatus.FAILED.value, message))
            continue

        def progress_callback(session_name, done, total, _job=job):
            queue.put(("job_progress", _job.id, done, total))
            log(f"[{_job.group_name}] finished session '{session_name}' ({done}/{total})")

        try:
            log(f"[{job.group_name}] calibrating (or loading saved calibration)...")
            output_path = run_group(
                config, device=device, tracking=tracking, progress_callback=progress_callback,
                log=log, progress=progress, html_progress=True, on_redraw_closed=on_redraw_closed,
                validation_dir=job.validation_dir,
            )
            log(f"[{job.group_name}] wrote {output_path}")
            queue.put(("job_finished", job.id, JobStatus.DONE.value, ""))
        except Exception as exc:
            message = str(exc)
            log(f"[{job.group_name}] FAILED: {message}\n{traceback.format_exc()}")
            queue.put(("job_finished", job.id, JobStatus.FAILED.value, message))

    queue.put(("all_finished",))
