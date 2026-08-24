"""
Clears cached/generated data for a queued GUI job so you can re-run part
or all of the pipeline from scratch -- handy for testing, or after
changing a model/gait setting that only invalidates some of a job's
output.

Usage:
    python reset.py --job "Cohort 1"               # predictions + 3D output + report (default)
    python reset.py --job "Cohort 1" --output-only  # keep .slp predictions; clear only 3D output + report
    python reset.py --job "Cohort 1" --all          # also clear cropped videos (back to NEEDS_CROP)
    python reset.py --job "Cohort 1" --yes          # skip the confirmation prompt
    python reset.py --list-jobs                     # list queued group names and exit

``crop_positions.json`` is never touched by any of these, including
``--all`` -- unlike RATlab-NOR's object coordinates, it's a position
cache rather than setup state that gates job readiness, so a later
re-crop resumes from cached positions instead of re-asking. This module
is also what ``gui/main_window.py``'s Reset menu calls directly
(``describe_job_targets``/``perform_job_reset``), so the CLI and GUI stay
in exact agreement about what each tier clears.

Modeled on RATlab-NOR's reset.py, simplified: alliGAITor has no "legacy
single-folder" CLI mode to preserve alongside the job queue, since every
run has always been config.yaml/group-based.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List

REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DIR / "gui"))
sys.path.insert(0, str(REPO_DIR))

from job_queue import Job, JobQueue, default_app_data_dir  # noqa: E402
from alligaitor.config import PipelineConfig  # noqa: E402


def _rmtree_contents(folder: Path) -> int:
    folder = Path(folder)
    if not folder.exists():
        return 0
    count = 0
    for child in folder.iterdir():
        if child.name == "crop_positions.json":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
        count += 1
    return count


def _session_names(job: Job) -> List[str]:
    if not job.config_path.exists():
        return []
    try:
        config = PipelineConfig.from_yaml(job.config_path)
    except Exception:
        return []
    return [session.name for session in config.sessions]


# ---------------------------------------------------------------------------
# Describe / perform (shared by the CLI below and gui/main_window.py)
# ---------------------------------------------------------------------------

def describe_job_targets(job: Job, clear_predictions=False, clear_output=False, clear_crops=False) -> List[str]:
    """Human-readable list of what perform_job_reset() would touch, for
    confirmation prompts/dialogs -- doesn't delete anything."""
    lines = []
    sessions = _session_names(job)
    if clear_predictions:
        lines.append(f"{job.predictions_dir}/<session>/{{left,right,bottom}}.predictions.slp "
                      f"({len(sessions)} session(s))")
    if clear_output:
        lines.append(f"{job.predictions_dir}/<session>/*.pose_3d.csv, *.paw_events.csv "
                      f"({len(sessions)} session(s))")
        lines.append(str(job.reports_dir))
    if clear_crops:
        lines.append(f"{job.cropped_dir}  (cropped videos -- crop_positions.json is kept)")
    return lines


def perform_job_reset(job: Job, clear_predictions=False, clear_output=False, clear_crops=False, log=print) -> int:
    """Clears one job's own cache/output -- does not touch any other job.
    Predictions, 3D output, and cropped videos are cleared independently
    (see the module docstring for why clear_output alone keeps cached
    .slp predictions -- a re-run reuses them instead of re-tracking,
    useful after changing a gait-only setting). ``crop_positions.json``
    is preserved in every case."""
    total = 0
    sessions = _session_names(job)

    if clear_predictions:
        for name in sessions:
            session_dir = job.predictions_dir / name
            for role in ("left", "right", "bottom"):
                slp = session_dir / f"{role}.predictions.slp"
                if slp.exists():
                    slp.unlink()
                    total += 1
                    log(f"Deleted {slp}")

    if clear_output:
        for name in sessions:
            session_dir = job.predictions_dir / name
            for suffix in ("pose_3d.csv", "paw_events.csv"):
                f = session_dir / f"{name}.{suffix}"
                if f.exists():
                    f.unlink()
                    total += 1
                    log(f"Deleted {f}")
        n = _rmtree_contents(job.reports_dir)
        total += n
        log(f"Cleared {n} item(s) from {job.reports_dir}")

    if clear_crops:
        n = _rmtree_contents(job.cropped_dir)
        total += n
        log(f"Cleared {n} item(s) from {job.cropped_dir} (kept crop_positions.json)")

    return total


def _find_job_by_group_name(app_data_dir: Path, group_name: str) -> Job:
    queue = JobQueue(app_data_dir).load()
    matches = [j for j in queue.jobs if j.group_name == group_name]
    if not matches:
        available = ", ".join(repr(j.group_name) for j in queue.jobs) or "(no jobs queued)"
        raise SystemExit(f"No job named {group_name!r} found in {queue.path}.\nAvailable: {available}")
    if len(matches) > 1:
        raise SystemExit(f"Multiple jobs are named {group_name!r} -- this shouldn't happen (group names should be unique).")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job", metavar="GROUP_NAME", help="the queued job (group name) to target")
    parser.add_argument("--output-only", action="store_true",
                         help="clear ONLY 3D output + report -- keeps cached .slp predictions, "
                              "so a re-run reuses them instead of re-tracking")
    parser.add_argument("--all", action="store_true",
                         help="also clear cropped videos (job goes back to NEEDS_CROP)")
    parser.add_argument("--yes", "-y", action="store_true", help="skip confirmation prompt")
    parser.add_argument("--list-jobs", action="store_true", help="list queued group names and exit")
    args = parser.parse_args()

    app_data_dir = default_app_data_dir(REPO_DIR)

    if args.list_jobs:
        queue = JobQueue(app_data_dir).load()
        if not queue.jobs:
            print(f"No jobs queued ({queue.path}).")
        else:
            for j in queue.jobs:
                print(f"  {j.group_name!r}  ({j.status.value})")
        return

    if not args.job:
        parser.error("--job GROUP_NAME is required (or pass --list-jobs)")

    if args.output_only and args.all:
        parser.error("--output-only and --all are mutually exclusive -- pick one")

    if args.output_only:
        clear_predictions, clear_output, clear_crops = False, True, False
    elif args.all:
        clear_predictions, clear_output, clear_crops = True, True, True
    else:
        clear_predictions, clear_output, clear_crops = True, True, False

    job = _find_job_by_group_name(app_data_dir, args.job)
    targets = describe_job_targets(job, clear_predictions, clear_output, clear_crops)
    print(f"This will delete, for job '{job.group_name}':")
    for line in targets:
        print(f"  - {line}")
    print("  (crop_positions.json is always kept)")

    if not args.yes:
        answer = input("\nProceed? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("Cancelled.")
            sys.exit(1)

    perform_job_reset(job, clear_predictions, clear_output, clear_crops)

    queue = JobQueue(app_data_dir).load()
    fresh = queue.get(job.id)
    if fresh is not None:
        from job_queue import JobStatus, refresh_job_readiness
        # NEEDS_CROP is always in refresh_job_readiness's refreshable set,
        # regardless of whether this job was previously DONE/FAILED/
        # CANCELED -- refresh then promotes it back to READY if the
        # cropped videos are still all in place (predictions/output-only
        # reset), or leaves it at NEEDS_CROP if not (--all).
        fresh.status = JobStatus.NEEDS_CROP
        refresh_job_readiness(fresh)
        queue.update(fresh)

    print("\nDone.")


if __name__ == "__main__":
    main()
