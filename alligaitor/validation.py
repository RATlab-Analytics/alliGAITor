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

"""Per-session sidecar files backing the GUI's validation-viewing dialogs.

Two JSON files live next to each session's ``pose_3d.csv``/``paw_events.csv``:
``<session>.validation_summary.json`` (automatic per-paw usability windows) and
``<session>.manual_flags.json`` (reviewer overrides, keyed by 1-based crossing number).
Kept out of :mod:`alligaitor.gait` so that module stays pure computation.
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
    """Compute and write per-paw usability windows to `path`.

    Accepts either one trial or the per-crossing list from
    :func:`alligaitor.gait.compute_crossing_metrics`. Each crossing gets its own entry under
    ``crossings``; the top-level ``paws`` rolls up across crossings, marking a paw usable if
    any crossing produced a usable run for it.

    Args:
        fallback_mask: Per paw, an ``(n_frames,)`` boolean array marking which frames came
            from the bottom-camera fallback rather than real triangulation. ``None`` leaves
            every window's ``bottom_fallback_fraction`` at ``0.0``.
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
    """``summary["crossings"]``, or a synthetic single entry wrapping ``summary["paws"]``
    for a summary written before multi-crossing support existed."""
    crossings = summary.get("crossings")
    if crossings:
        return crossings
    return [{"paws": summary.get("paws", {})}]


def load_validation_summary(path: PathLike) -> Optional[dict]:
    """Load a ``<session>.validation_summary.json``, or ``None`` if `path` doesn't exist.

    Returns the raw dict rather than reconstructing dataclasses.
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
    """Load a ``<session>.manual_flags.json`` sidecar, or ``{}`` if `path` doesn't exist."""
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
    """Write `flags_by_crossing` to `path`, replacing whatever was there before.

    A crossing with no flagged paws is dropped entirely rather than written as an empty entry.
    """
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
    """The paws flagged on one specific crossing, or ``set()`` if none."""
    return flags_by_crossing.get(crossing_number, (set(), ""))[0]


def effective_usability(summary: dict, flags_by_crossing: ManualFlags) -> Dict[str, bool]:
    """Per paw, whether it should display as usable anywhere in this recording.

    ``True`` if any crossing has an automatically-usable window for that paw not flagged on
    that specific crossing. Recomputed from per-crossing data rather than trusting the
    precomputed top-level rollup, since that rollup has no knowledge of flags.
    """
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
    """Per paw, whether the crossing that makes it usable leans heavily on the bottom-camera
    fallback (``bottom_fallback_fraction`` over :data:`alligaitor.gait.BOTTOM_FALLBACK_WARN_THRESHOLD`).

    Only meaningful for a paw :func:`effective_usability` already reports as usable; intended
    as a caution on top of usable, not a replacement for it.
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


def usable_crossing_counts(summary: dict, flags_by_crossing: ManualFlags) -> Dict[str, int]:
    """Per paw, how many crossings produced a usable, unflagged window for it.

    Not "crossings the paw was merely visible in" -- a paw tracked on every crossing but
    usable on none reads 0, not the crossing count.
    """
    crossings = crossings_or_fallback(summary)
    result = {}
    for paw in PAW_NODES:
        count = 0
        for crossing in crossings:
            window = crossing.get("paws", {}).get(paw)
            if window is None:
                continue
            crossing_number = crossing.get("crossing", 1)
            if window.get("usable") and paw not in crossing_flagged_paws(flags_by_crossing, crossing_number):
                count += 1
        result[paw] = count
    return result


# ---------------------------------------------------------------------------
# Manual flags across a whole group (for regenerating the group workbook)
# ---------------------------------------------------------------------------

def load_group_manual_flags(predictions_dir: PathLike, session_names) -> Dict[str, ManualFlags]:
    """Per session name, its :data:`ManualFlags`, for every session in `session_names`."""
    predictions_dir = Path(predictions_dir)
    result = {}
    for name in session_names:
        flags_path = predictions_dir / name / f"{name}.manual_flags.json"
        flags_by_crossing = load_manual_flags(flags_path)
        if flags_by_crossing:
            result[name] = flags_by_crossing
    return result
