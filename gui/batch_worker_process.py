"""
Runs queued jobs' pipeline (``alligaitor.pipeline.run_group``) in a
separate OS process, one job at a time -- kept out of the GUI process for
the same reason ``tools/crop_worker_process.py`` is: SLEAP-NN inference
(itself a subprocess) and heavy video I/O alongside a live Qt event loop
is a real crash risk, not just a performance concern, and a failure here
shouldn't take the GUI down with it.

Stop only takes effect between jobs, not mid-job -- ``run_group()`` is one
long blocking call per job with no internal cancellation point, so a job
already in progress when Stop is requested always finishes (or fails) on
its own before the run actually halts.

Ported from RATlab-NOR's gui/batch_worker_process.py: only the
process/queue plumbing carries over, since alliGAITor's actual per-job
work (``alligaitor.pipeline.run_group``) has nothing in common with NOR's
per-video sniff-scoring loop.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path


def run_batch_worker(repo_dir: str, job_dicts: list, queue, stop_event,
                      device: str = "auto", tracking: bool = False) -> None:
    repo_dir = Path(repo_dir)
    for p in (repo_dir, repo_dir / "gui"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    from alligaitor.config import PipelineConfig
    from alligaitor.pipeline import run_group
    from job_queue import Job, JobStatus

    def log(message: str):
        queue.put(("log", message))

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
            output_path = run_group(config, device=device, tracking=tracking, progress_callback=progress_callback)
            log(f"[{job.group_name}] wrote {output_path}")
            queue.put(("job_finished", job.id, JobStatus.DONE.value, ""))
        except Exception as exc:
            message = str(exc)
            log(f"[{job.group_name}] FAILED: {message}\n{traceback.format_exc()}")
            queue.put(("job_finished", job.id, JobStatus.FAILED.value, message))

    queue.put(("all_finished",))
