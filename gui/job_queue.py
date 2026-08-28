"""
Job queue data model for the alliGAITor GUI.

A "Job" is one group's worth of work: a folder of source videos, an
output location, and the group ``config.yaml`` that
:func:`alligaitor.discovery.discover_sessions` and
:func:`alligaitor.pipeline.run_group` both read. Jobs are queued in the
GUI, then run one at a time by the batch runner.

This module owns the Job dataclass and JobStatus enum, persistence to
``app_data/queue.json``, per-job output layout, and the "is this job
ready to run" check. Pure data/state, no GUI or pipeline-execution code.
"""

from __future__ import annotations

import json
import os
import re
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# App data location
# ---------------------------------------------------------------------------

def default_app_data_dir() -> Path:
    """OS-standard per-user data directory for alliGAITor's settings and job queue."""
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif sys.platform.startswith("win"):
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "alliGAITor"


def default_models_dir(repo_dir: Path) -> Optional[Path]:
    """Dev-checkout convenience fallback: repo_dir/models if it exists, else None."""
    candidate = Path(repo_dir) / "models"
    return candidate if candidate.is_dir() else None


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    NEEDS_CONFIG = "needs_config"  # config.yaml not yet confirmed (regexes, roles, calibration)
    NEEDS_CROP = "needs_crop"      # config confirmed, but not every discovered video is cropped yet
    READY = "ready"                 # fully set up, waiting in the queue
    QUEUED = "queued"               # handed to the batch runner for this run, waiting its turn behind RUNNING
    RUNNING = "running"             # batch runner is actively processing this job
    DONE = "done"                   # finished with no errors
    FAILED = "failed"               # hit an error; batch runner blocked it and moved on
    CANCELED = "canceled"           # user pulled it before/while it ran


# Statuses where re-opening the config editor or crop step is meaningful.
EDITABLE_STATUSES = (JobStatus.NEEDS_CONFIG, JobStatus.NEEDS_CROP, JobStatus.READY)

# Terminal statuses a job can be safely re-queued from.
RERUNNABLE_STATUSES = {JobStatus.FAILED, JobStatus.DONE, JobStatus.CANCELED}


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip()).strip("_")
    return slug or "group"


# ---------------------------------------------------------------------------
# Job
# ---------------------------------------------------------------------------

@dataclass
class Job:
    group_name: str
    input_folder: str
    output_folder: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    status: JobStatus = JobStatus.NEEDS_CONFIG
    error_message: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    # Session-level progress, filled in by the batch runner as it works
    # through the job, so the GUI can show "3/12 sessions" while running.
    sessions_total: int = 0
    sessions_done: int = 0

    @property
    def slug(self) -> str:
        """Filesystem-safe version of the group name."""
        return _slugify(self.group_name)

    @property
    def config_path(self) -> Path:
        """This job's group config.yaml."""
        return Path(self.output_folder) / "config.yaml"

    @property
    def cropped_dir(self) -> Path:
        return Path(self.output_folder) / "cropped"

    @property
    def predictions_dir(self) -> Path:
        return Path(self.output_folder) / "predictions_3d"

    @property
    def reports_dir(self) -> Path:
        return Path(self.output_folder) / "reports"

    @property
    def validation_dir(self) -> Path:
        return Path(self.output_folder) / "validation"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Job":
        d = dict(d)
        d["status"] = JobStatus(d.get("status", "needs_config"))
        return cls(**d)


# ---------------------------------------------------------------------------
# Readiness check
# ---------------------------------------------------------------------------

def refresh_job_readiness(job: Job) -> Job:
    """Recompute NEEDS_CONFIG / NEEDS_CROP / READY from what's on disk.
    Does not touch RUNNING/DONE/FAILED/CANCELED jobs."""
    if job.status not in (JobStatus.NEEDS_CONFIG, JobStatus.NEEDS_CROP, JobStatus.READY):
        return job

    if not job.config_path.exists():
        job.status = JobStatus.NEEDS_CONFIG
        return job

    from alligaitor.config import PipelineConfig  # local import: keep this module import-light otherwise

    try:
        config = PipelineConfig.from_yaml(job.config_path)
    except Exception:
        job.status = JobStatus.NEEDS_CONFIG
        return job

    if not config.sessions:
        job.status = JobStatus.NEEDS_CONFIG
        return job

    job.sessions_total = len(config.sessions)
    missing = any(
        not video_path.exists()
        for session in config.sessions
        for video_path in session.videos.values()
    )
    job.status = JobStatus.NEEDS_CROP if missing else JobStatus.READY
    return job


# ---------------------------------------------------------------------------
# Queue persistence
# ---------------------------------------------------------------------------

class JobQueue:
    """Ordered list of jobs, persisted to app_data/queue.json. Add-order is run order."""

    def __init__(self, app_data_dir: Path):
        self.app_data_dir = Path(app_data_dir)
        self.path = self.app_data_dir / "queue.json"
        self.jobs: List[Job] = []

    # -- persistence --

    def load(self) -> "JobQueue":
        if self.path.exists():
            with open(self.path) as f:
                raw = json.load(f)
            self.jobs = [Job.from_dict(j) for j in raw.get("jobs", [])]
        else:
            self.jobs = []
        self._recover_stale_running_jobs()
        return self

    def _recover_stale_running_jobs(self) -> None:
        """Marks any job left RUNNING or QUEUED (from a crash or kill
        mid-run) as FAILED on load, since nothing will resume it."""
        changed = False
        for job in self.jobs:
            if job.status in (JobStatus.RUNNING, JobStatus.QUEUED):
                job.status = JobStatus.FAILED
                job.error_message = "Interrupted -- the app closed or crashed while this job was running."
                changed = True
        if changed:
            self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump({"jobs": [j.to_dict() for j in self.jobs]}, f, indent=2)
        tmp.replace(self.path)

    # -- CRUD --

    def add(self, job: Job) -> Job:
        self.jobs.append(job)
        self.save()
        return job

    def remove(self, job_id: str) -> None:
        self.jobs = [j for j in self.jobs if j.id != job_id]
        self.save()

    def get(self, job_id: str) -> Optional[Job]:
        return next((j for j in self.jobs if j.id == job_id), None)

    def update(self, job: Job) -> None:
        for i, j in enumerate(self.jobs):
            if j.id == job.id:
                self.jobs[i] = job
                break
        self.save()

    def group_name_taken(self, name: str, exclude_id: Optional[str] = None) -> bool:
        return any(j.group_name.lower() == name.lower() and j.id != exclude_id for j in self.jobs)

    # -- queries the batch runner needs --

    def runnable_jobs(self) -> List[Job]:
        """Jobs in add-order that are ready to be picked up by a run."""
        return [j for j in self.jobs if j.status == JobStatus.READY]
