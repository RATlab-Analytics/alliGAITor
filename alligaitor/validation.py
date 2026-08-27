"""Per-session sidecar files backing the GUI's validation-viewing dialogs.

Two small JSON files live next to each session's ``pose_3d.csv``/
``paw_events.csv`` (see :mod:`alligaitor.pipeline`'s ``run_group``):

* ``<session>.validation_summary.json`` -- the automatic per-paw
  usability window computed once at run time (see
  :func:`alligaitor.gait.paw_usability_windows`), so the GUI can list and
  color every session without re-triangulating or re-running stance
  detection.
* ``<session>.manual_flags.json`` -- a reviewer's manual "this paw's run
  on *this crossing* isn't actually trustworthy" override, recorded by
  the validation video dialog's Flag Paw(s) action, keyed by crossing
  number (1-based, matching :attr:`alligaitor.gait.TrialMetrics.crossing_index`
  ``+ 1`` and the ``"crossing"`` field in a validation summary's
  ``crossings`` entries) -- a paw can be flagged on one crossing of a
  recording and left alone on another. Absent by default (= no manual
  flags anywhere in this recording).

Kept out of :mod:`alligaitor.gait` so that module stays pure computation
(no filesystem I/O) and this one stays pure I/O (no gait-detection logic).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np

from alligaitor.gait import (
    BOTTOM_FALLBACK_WARN_THRESHOLD,
    PAW_NODES,
    GaitConfig,
    TrialMetrics,
    paw_usability_windows,
)

# crossing_number (1-based) -> (flagged paw names, shared note)
ManualFlags = Dict[int, Tuple[Set[str], str]]

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
    fallback_mask: Optional[Dict[str, np.ndarray]] = None,
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

    Args:
        fallback_mask: Per paw, an ``(n_frames,)`` boolean array (see
            :func:`alligaitor.gait.load_pose_3d`'s ``fallback`` return
            value), forwarded to :func:`alligaitor.gait.paw_usability_windows`
            so each written window carries its own
            ``bottom_fallback_fraction`` -- how much of that window came
            from :func:`alligaitor.bottom_fallback.fill_gaps` rather than
            real triangulation. ``None`` (the default) leaves every
            window's fraction at ``0.0``.
    """
    trials = [trial] if isinstance(trial, TrialMetrics) else list(trial)
    if not trials:
        raise ValueError("save_validation_summary needs at least one trial")

    per_crossing = []
    for t in trials:
        windows = paw_usability_windows(t, times, positions, config, bottom_fallback_mask=fallback_mask)
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


def crossings_or_fallback(summary: dict) -> List[dict]:
    """``summary["crossings"]`` (see :func:`save_validation_summary`), or
    a single synthetic entry wrapping ``summary["paws"]`` if `summary`
    predates per-crossing data -- a ``validation_summary.json`` written
    before multi-crossing support existed has no ``"crossings"`` key at
    all. Lets a caller iterating crossings (the validation list's
    per-paw visible-crossings fraction, the video dialog's per-crossing
    scrub markers) treat every summary uniformly instead of silently
    seeing an empty list for a job that hasn't been rerun since."""
    crossings = summary.get("crossings")
    if crossings:
        return crossings
    return [{"paws": summary.get("paws", {})}]


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

def load_manual_flags(path: PathLike) -> ManualFlags:
    """Load a ``<session>.manual_flags.json`` sidecar, or ``{}`` if `path`
    doesn't exist (no paw has been manually flagged on any crossing of
    this recording). Returns crossing_number -> (flagged_paws, note)."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path) as f:
        raw = json.load(f)
    return {
        int(crossing_number): (set(entry.get("paws", [])), entry.get("note", ""))
        for crossing_number, entry in raw.get("flags", {}).items()
    }


def save_manual_flags(path: PathLike, flags_by_crossing: ManualFlags) -> None:
    """Write `flags_by_crossing` (crossing_number -> (flagged paw names,
    shared note), a subset of :data:`alligaitor.gait.PAW_NODES` per
    crossing) to `path`, replacing whatever was there before -- the
    validation video dialog's Flag Paw(s) popup always writes the full
    current state for the crossing it just edited, not an incremental
    delta. A crossing with no flagged paws is dropped entirely rather
    than written as an empty entry, so unflagging every paw on a crossing
    cleans the file back up instead of leaving inert clutter."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = {
        str(crossing_number): {"paws": sorted(paws), "note": note}
        for crossing_number, (paws, note) in flags_by_crossing.items()
        if paws
    }
    with open(path, "w") as f:
        json.dump({"flags": flags}, f, indent=2)


# ---------------------------------------------------------------------------
# Combined "should this show as usable" view
# ---------------------------------------------------------------------------

def crossing_flagged_paws(flags_by_crossing: ManualFlags, crossing_number: int) -> Set[str]:
    """The paws flagged on one specific crossing -- ``set()`` if that
    crossing has no flags at all."""
    return flags_by_crossing.get(crossing_number, (set(), ""))[0]


def effective_usability(summary: dict, flags_by_crossing: ManualFlags) -> Dict[str, bool]:
    """Per paw, whether it should display as usable (green) anywhere in
    this recording -- ``True`` if *any* crossing has an automatically-
    usable window for that paw which isn't flagged on that specific
    crossing (see :func:`crossing_flagged_paws`). Recomputed from
    `summary`'s per-crossing data rather than trusting the precomputed
    top-level rollup (see :func:`save_validation_summary`), since that
    rollup was written without knowledge of any flag -- a flag on the
    one crossing that made a paw usable should be able to flip it back to
    unusable, and a flag on a *different*, already-unusable crossing for
    that paw should never affect this at all. The single source of truth
    both the validation list and video dialogs read, so they can never
    disagree about a given paw's color."""
    crossings = crossings_or_fallback(summary)
    result = {}
    for paw in PAW_NODES:
        usable = False
        for crossing in crossings:
            window = crossing.get("paws", {}).get(paw)
            if window is None:
                continue
            crossing_number = crossing.get("crossing", 1)
            if window.get("usable") and paw not in crossing_flagged_paws(flags_by_crossing, crossing_number):
                usable = True
                break
        result[paw] = usable
    return result


def usable_paws_with_fallback_warning(summary: dict, flags_by_crossing: ManualFlags) -> Dict[str, bool]:
    """Per paw, whether the crossing that makes it usable (see
    :func:`effective_usability`) leans heavily on the experimental
    bottom-camera fallback -- ``True`` when that crossing's window has
    ``bottom_fallback_fraction`` over :data:`alligaitor.gait.BOTTOM_FALLBACK_WARN_THRESHOLD`.

    Only meaningful for a paw :func:`effective_usability` already reports
    as usable -- a caller should show this as a yellow caution on top of
    green, never in place of red, since the run is still usable, just one
    worth a second look. Walks the exact same crossing that decides
    usability (not just "any crossing"), so this can never flag a paw
    whose usable run was untouched by the fallback just because some
    *other*, already-unusable crossing happened to lean on it.
    """
    crossings = crossings_or_fallback(summary)
    result = {}
    for paw in PAW_NODES:
        heavy_fallback = False
        for crossing in crossings:
            window = crossing.get("paws", {}).get(paw)
            if window is None:
                continue
            crossing_number = crossing.get("crossing", 1)
            if window.get("usable") and paw not in crossing_flagged_paws(flags_by_crossing, crossing_number):
                heavy_fallback = window.get("bottom_fallback_fraction", 0.0) > BOTTOM_FALLBACK_WARN_THRESHOLD
                break
        result[paw] = heavy_fallback
    return result


# ---------------------------------------------------------------------------
# Manual flags across a whole group (for regenerating the group workbook)
# ---------------------------------------------------------------------------

def load_group_manual_flags(predictions_dir: PathLike, session_names) -> Dict[str, ManualFlags]:
    """Per session name, its :data:`ManualFlags` (crossing_number ->
    (flagged_paws, note)) -- see :func:`load_manual_flags` -- for every
    session in `session_names`. Used by
    :func:`alligaitor.pipeline.run_group` to carry a rerun's manual flags
    forward into the freshly regenerated workbook (see
    :func:`alligaitor.gait.write_group_report`'s ``manual_flags`` arg).
    """
    predictions_dir = Path(predictions_dir)
    result = {}
    for name in session_names:
        flags_path = predictions_dir / name / f"{name}.manual_flags.json"
        flags_by_crossing = load_manual_flags(flags_path)
        if flags_by_crossing:
            result[name] = flags_by_crossing
    return result
