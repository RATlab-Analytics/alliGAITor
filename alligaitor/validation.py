"""Per-session sidecar files backing the GUI's validation-viewing dialogs.

Two small JSON files live next to each session's ``pose_3d.csv``/
``paw_events.csv`` (see :mod:`alligaitor.pipeline`'s ``run_group``):

* ``<session>.validation_summary.json`` -- the automatic per-paw
  usability window computed once at run time (see
  :func:`alligaitor.gait.paw_usability_windows`), so the GUI can list and
  color every session without re-triangulating or re-running stance
  detection.
* ``<session>.manual_flags.json`` -- a reviewer's manual "this paw's run
  isn't actually trustworthy" override, recorded by the validation video
  dialog's Flag Paw(s) action. Absent by default (= no manual flags).

Kept out of :mod:`alligaitor.gait` so that module stays pure computation
(no filesystem I/O) and this one stays pure I/O (no gait-detection logic).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np

from alligaitor.gait import PAW_NODES, GaitConfig, TrialMetrics, paw_usability_windows

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Automatic per-paw usability summary
# ---------------------------------------------------------------------------

def save_validation_summary(
    trial: Union[TrialMetrics, List[TrialMetrics]],
    times: np.ndarray,
    positions: Dict[str, np.ndarray],
    config: GaitConfig,
    path: PathLike,
) -> None:
    """Compute and write per-paw usability windows (see
    :func:`alligaitor.gait.paw_usability_windows`) to `path`.

    Accepts either one trial or the per-crossing list from
    :func:`alligaitor.gait.compute_crossing_metrics`. Each crossing gets
    its own entry under ``crossings``, since a paw can produce a clean
    run on the way out and nothing usable on the way back.

    ``paws`` stays at the top level, rolled up across crossings: a paw
    appears usable there if *any* crossing produced a usable run for it,
    carrying that crossing's window (the longest one, where several
    qualify). That keeps every existing reader of this file working
    unchanged and answers the question the validation UI actually asks
    -- "is there a good run for this paw in this recording?" -- while
    ``crossings`` holds the detail for a caller that needs per-crossing
    resolution.
    """
    trials = [trial] if isinstance(trial, TrialMetrics) else list(trial)
    if not trials:
        raise ValueError("save_validation_summary needs at least one trial")

    per_crossing = []
    for t in trials:
        windows = paw_usability_windows(t, times, positions, config)
        per_crossing.append({
            "crossing": t.crossing_index + 1,
            "crossing_count": t.crossing_count,
            "window": list(t.crossing_window) if t.crossing_window else None,
            "paws": {paw: (asdict(w) if w is not None else None) for paw, w in windows.items()},
        })

    def _rolled_up(paw: str):
        candidates = [c["paws"].get(paw) for c in per_crossing]
        candidates = [w for w in candidates if w is not None]
        if not candidates:
            return None
        usable = [w for w in candidates if w.get("usable")]
        pool = usable or candidates
        return max(pool, key=lambda w: w.get("duration_s", 0.0))

    raw = {
        "session_name": trials[0].session_name,
        "rat_id": trials[0].rat_id,
        "crossings": per_crossing,
        "paws": {paw: _rolled_up(paw) for paw in PAW_NODES},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(raw, f, indent=2)


def load_validation_summary(path: PathLike) -> Optional[dict]:
    """Load a ``<session>.validation_summary.json`` written by
    :func:`save_validation_summary`, or ``None`` if `path` doesn't exist
    (this session hasn't been run with validation-video export yet).

    Returns the raw dict (``session_name``, ``rat_id``, ``paws`` mapping
    paw name to a :class:`alligaitor.gait.PawWindow`-shaped dict or
    ``None``) rather than reconstructing dataclasses -- the GUI only ever
    reads a handful of fields out of it directly.
    """
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Manual flags
# ---------------------------------------------------------------------------

def load_manual_flags(path: PathLike) -> Tuple[Set[str], str]:
    """Load a ``<session>.manual_flags.json`` sidecar, or ``(set(), "")``
    if `path` doesn't exist (no paw has been manually flagged for this
    session)."""
    path = Path(path)
    if not path.exists():
        return set(), ""
    with open(path) as f:
        raw = json.load(f)
    return set(raw.get("flagged_paws", [])), raw.get("note", "")


def save_manual_flags(path: PathLike, flagged_paws: Set[str], note: str = "") -> None:
    """Write `flagged_paws` (a subset of
    :data:`alligaitor.gait.PAW_NODES`) and an optional shared `note` to
    `path`, replacing whatever was there before -- the validation video
    dialog's Flag Paw(s) popup always writes the full current state, not
    an incremental delta."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"flagged_paws": sorted(flagged_paws), "note": note}, f, indent=2)


# ---------------------------------------------------------------------------
# Combined "should this show as usable" view
# ---------------------------------------------------------------------------

def effective_usability(summary: dict, flagged_paws: Set[str]) -> Dict[str, bool]:
    """Per paw, whether it should display as usable (green) -- the
    automatic call from `summary` (see :func:`load_validation_summary`),
    overridden to unusable for any paw in `flagged_paws`. The single
    source of truth both the validation list and video dialogs read, so
    they can never disagree about a given paw's color."""
    result = {}
    for paw in PAW_NODES:
        window = summary.get("paws", {}).get(paw)
        auto_usable = bool(window["usable"]) if window is not None else False
        result[paw] = auto_usable and paw not in flagged_paws
    return result


# ---------------------------------------------------------------------------
# Manual flags across a whole group (for regenerating the group workbook)
# ---------------------------------------------------------------------------

def load_group_manual_flags(predictions_dir: PathLike, session_names) -> Dict[str, Tuple[set, str]]:
    """Per session name, its ``(flagged_paws, note)`` -- see
    :func:`load_manual_flags` -- for every session in `session_names`.
    Used by :func:`alligaitor.pipeline.run_group` to carry a rerun's
    manual flags forward into the freshly regenerated workbook (see
    :func:`alligaitor.gait.write_group_report`'s ``manual_flags`` arg).
    """
    predictions_dir = Path(predictions_dir)
    result = {}
    for name in session_names:
        flags_path = predictions_dir / name / f"{name}.manual_flags.json"
        flagged, note = load_manual_flags(flags_path)
        if flagged:
            result[name] = (flagged, note)
    return result
