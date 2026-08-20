"""Gait metrics computed from triangulated 3D paw trajectories.

Consumes the ``pose_3d.csv`` written by :func:`alligaitor.pipeline.run_session`
(frame, time_s, node, x, y, z, reprojection_error_px) and derives, per rat
crossing:

* total time to cross the platform,
* average speed,
* average stride length per paw (liftoff position to the following
  touchdown position),
* average step length per paw (forward distance from a paw's touchdown to
  the contralateral paw's most recent prior touchdown), and
* average ground contact time per paw (touchdown to liftoff duration).

Coordinates are taken directly from the 3D reconstruction, which is
already metric (the calibration board's square size is specified in mm),
so no separate pixel-to-length ratio is needed. Ground-contact detection
is speed-only (see :class:`alligaitor.config.GaitConfig`): a
height-above-platform check was tried and dropped, since drawing (or
reasoning about) an accurate height threshold requires knowing each
frame's actual depth across the tunnel's width, and a fixed reference
was visually misleading for paws at a different depth than it was
computed from.

:func:`alligaitor.pipeline.run_group` is the intended entry point for a
future GUI job queue: it runs the full 2D/3D pipeline for every session in
a group, computes each session's :class:`TrialMetrics` here, and writes
one workbook via :func:`write_group_report`. Every detected stance phase
is kept as explicit touchdown/liftoff frame indices (see
:class:`PawEvents`, and :func:`save_paw_events_csv`) rather than folded
straight into an average, so a later validation-video auditor can overlay
exactly the frames this module treated as ground contact.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from alligaitor.config import GaitConfig

PathLike = Union[str, Path]

PAW_NODES = ("left-forepaw", "right-forepaw", "left-hind-paw", "right-hind-paw")

CONTRALATERAL = {
    "left-forepaw": "right-forepaw",
    "right-forepaw": "left-forepaw",
    "left-hind-paw": "right-hind-paw",
    "right-hind-paw": "left-hind-paw",
}

# The side camera on the opposite side of the body from a given paw --
# e.g. "right" for left-forepaw. Measured (see 359a-BL's per-node miss
# rates, even restricted to frames where the rat is clearly present) at
# 64-96% missing, versus 11-26% for the same-side camera: the rat's own
# body occludes it from that angle essentially by construction, not a
# tracking failure. Used to exclude this camera from
# :func:`find_camera_caused_discards`'s attribution -- it not seeing a
# paw it was never going to see isn't a "drop" worth flagging.
FAR_SIDE_CAMERA = {
    "left-forepaw": "right",
    "right-forepaw": "left",
    "left-hind-paw": "right",
    "right-hind-paw": "left",
}

# Body-center node used to time the crossing and establish the direction
# of travel (see minimal_skeleton.json for the full node set).
REFERENCE_NODE = "mid-back"

_INVALID_SHEET_CHARS = set("[]:*?/\\")


@dataclass
class PawEvents:
    """Detected stance (ground-contact) phases for one paw in one trial.

    Each index ``i`` describes one stance phase: the paw is planted from
    ``touchdown_frames[i]``/``touchdown_times[i]`` through
    ``liftoff_frames[i]``/``liftoff_times[i]`` inclusive.
    """

    touchdown_frames: np.ndarray
    liftoff_frames: np.ndarray
    touchdown_times: np.ndarray
    liftoff_times: np.ndarray


@dataclass
class TrialMetrics:
    """Gait metrics for one rat's single crossing of the platform."""

    session_name: str
    rat_id: str
    crossing_time_s: float
    average_speed_mm_s: float
    stride_length_mm: Dict[str, float]
    step_length_mm: Dict[str, float]
    ground_contact_time_s: Dict[str, float]
    n_contacts: Dict[str, int]
    n_strides: Dict[str, int]
    n_steps: Dict[str, int]
    paw_events: Dict[str, PawEvents]


def load_pose_3d(
    csv_path: PathLike,
) -> "tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]":
    """Load a ``pose_3d.csv`` into per-node arrays indexed by frame.

    Returns:
        A ``(times, positions, reprojection_error_px)`` triple: ``times``
        is seconds per frame index; ``positions`` maps node name to an
        ``(n_frames, 3)`` array of that node's ``x``/``y``/``z``, ``NaN``
        where untriangulated; ``reprojection_error_px`` maps node name to
        an ``(n_frames,)`` array of that frame's mean reprojection error
        (see :class:`alligaitor.triangulation.Pose3D`), ``NaN`` to match.
    """
    df = pd.read_csv(csv_path)
    n_frames = int(df["frame"].max()) + 1

    times = np.full(n_frames, np.nan)
    times[df["frame"].to_numpy()] = df["time_s"].to_numpy()

    positions = {}
    reprojection_error_px = {}
    for node, sub in df.groupby("node"):
        arr = np.full((n_frames, 3), np.nan)
        arr[sub["frame"].to_numpy()] = sub[["x", "y", "z"]].to_numpy()
        positions[node] = arr

        err = np.full(n_frames, np.nan)
        err[sub["frame"].to_numpy()] = sub["reprojection_error_px"].to_numpy()
        reprojection_error_px[node] = err
    return times, positions, reprojection_error_px


def bridge_short_gaps(xyz: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly interpolate over short untriangulated runs.

    A run of untriangulated (``NaN``) frames is bridged only when it's at
    most ``max_gap`` frames long *and* bounded by a valid frame on both
    sides -- a gap touching either end of the trial is left as-is, since
    there's nothing to interpolate from/to. ``max_gap <= 0`` is a no-op
    (returns a copy, unchanged).

    Used to keep momentary tracking jitter and brief per-camera dropouts
    (which will always happen, even with better models) from fragmenting
    one real stance phase into pieces too short to individually survive
    :attr:`alligaitor.config.GaitConfig.min_contact_frames` -- see
    :attr:`alligaitor.config.GaitConfig.max_bridge_gap_frames`.
    """
    xyz = xyz.copy()
    if max_gap <= 0:
        return xyz

    valid = ~np.isnan(xyz).any(axis=1)
    n = len(xyz)
    i = 0
    while i < n:
        if valid[i]:
            i += 1
            continue
        j = i
        while j < n and not valid[j]:
            j += 1
        if i > 0 and j < n and (j - i) <= max_gap:
            frac = (np.arange(i, j) - (i - 1)) / (j - (i - 1))
            xyz[i:j] = xyz[i - 1] + (xyz[j] - xyz[i - 1]) * frac[:, None]
        i = j
    return xyz


def _crossing_time_and_speed(
    times: np.ndarray, ref_xyz: np.ndarray
) -> tuple[float, float, np.ndarray]:
    """Return (crossing time, average speed, unit forward direction)."""
    valid = ~np.isnan(ref_xyz).any(axis=1)
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size < 2:
        raise ValueError(
            f"Reference node '{REFERENCE_NODE}' has fewer than 2 triangulated frames; "
            "cannot determine crossing time or direction."
        )

    first_idx, last_idx = valid_idx[0], valid_idx[-1]
    net_displacement = ref_xyz[last_idx] - ref_xyz[first_idx]
    net_distance = np.linalg.norm(net_displacement)
    if net_distance == 0:
        raise ValueError(
            f"Reference node '{REFERENCE_NODE}' shows no net displacement; "
            "cannot determine crossing direction."
        )
    forward = net_displacement / net_distance

    crossing_time_s = times[last_idx] - times[first_idx]

    pts = ref_xyz[valid_idx]
    path_length = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
    average_speed_mm_s = path_length / crossing_time_s

    return float(crossing_time_s), float(average_speed_mm_s), forward


def _raw_stance_candidates(
    times: np.ndarray, xyz: np.ndarray, config: GaitConfig
) -> "tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]":
    """Shared groundwork for stance detection.

    Returns ``(valid, speed, runs)``: whether the paw is triangulated on
    each frame, frame-to-frame speed (backward difference -- ``NaN`` on
    frame 0 and any frame whose predecessor is untriangulated), and every
    maximal run of raw "planted" frames (speed below
    ``config.speed_threshold_mm_s``), before the ``min_contact_frames``
    length filter. Used by both :func:`_detect_paw_events` and
    :func:`find_camera_caused_discards`.
    """
    n = len(times)
    valid = ~np.isnan(xyz).any(axis=1)

    speed = np.full(n, np.nan)
    disp = np.linalg.norm(xyz[1:] - xyz[:-1], axis=1)
    dt = times[1:] - times[:-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        speed[1:] = disp / dt

    planted = valid & (speed < config.speed_threshold_mm_s)

    runs = []
    start = None
    for i, p in enumerate(planted):
        if p and start is None:
            start = i
        elif not p and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, n - 1))

    return valid, speed, runs


def _detect_paw_events(times: np.ndarray, xyz: np.ndarray, config: GaitConfig) -> PawEvents:
    """Segment one paw's trajectory into stance phases.

    A frame counts as planted when the paw is triangulated and its
    frame-to-frame speed (backward difference) is below
    ``config.speed_threshold_mm_s``. Runs of fewer than
    ``config.min_contact_frames`` planted frames are dropped as tracking
    jitter rather than counted as real stance phases -- see
    :func:`find_camera_caused_discards` for identifying which of those
    discards trace back to a specific camera's dropped detection.
    """
    empty = PawEvents(np.array([], dtype=int), np.array([], dtype=int), np.array([]), np.array([]))

    valid, _, runs = _raw_stance_candidates(times, xyz, config)
    if not valid.any():
        return empty

    runs = [(s, e) for s, e in runs if (e - s + 1) >= config.min_contact_frames]
    if not runs:
        return empty

    touchdown_frames = np.array([s for s, _ in runs], dtype=int)
    liftoff_frames = np.array([e for _, e in runs], dtype=int)
    return PawEvents(
        touchdown_frames=touchdown_frames,
        liftoff_frames=liftoff_frames,
        touchdown_times=times[touchdown_frames],
        liftoff_times=times[liftoff_frames],
    )


@dataclass
class DiscardedStance:
    """A raw stance candidate too short to survive ``min_contact_frames``,
    where a triangulation gap right at its boundary looks like the cause.

    Attributes:
        start_frame: First frame of the affected window -- the discarded
            run itself, extended back to the gap frame that broke it.
        end_frame: Last frame of the affected window, similarly extended
            forward if the frame right after the run is also a gap.
        dropped_by: Camera role(s) missing a valid detection at whichever
            gap frame(s) caused the discard.
    """

    start_frame: int
    end_frame: int
    dropped_by: List[str]


def find_camera_caused_discards(
    times: np.ndarray,
    xyz: np.ndarray,
    cam_valid: Dict[str, np.ndarray],
    config: GaitConfig,
    exclude_camera: Optional[str] = None,
) -> List[DiscardedStance]:
    """Find discarded stance candidates a camera dropout plausibly caused.

    A raw candidate run of planted frames (see :func:`_raw_stance_candidates`)
    either survives ``min_contact_frames`` filtering or it doesn't; this
    looks at *why* the discarded ones were short. Speed is a backward
    difference, so a frame right after a triangulation gap has no valid
    predecessor to measure speed against and can never itself be
    classified planted, even though its own position is perfectly good --
    walking back from a short run through that chain of
    "position known, speed undefined" frames finds the actual gap
    responsible, and ``cam_valid`` says which camera(s) were missing
    there. The same check runs forward one frame, since a gap
    immediately after a run also cuts it short.

    Args:
        times: This paw's per-frame timestamps.
        xyz: This paw's ``(n_frames, 3)`` triangulated positions.
        cam_valid: Per camera role, an ``(n_frames,)`` boolean array --
            whether that camera had a valid (aligned) 2D detection for
            this paw on each frame. See
            :func:`alligaitor.triangulation.align_tracks_by_time`.
        config: The same :class:`GaitConfig` used to detect stance.
        exclude_camera: A camera role never counted as having dropped the
            paw -- e.g. :data:`FAR_SIDE_CAMERA`\\ [paw], since that
            camera not seeing this paw is expected, not a failure. A gap
            frame always has at least 2 missing cameras by construction
            (triangulation needs >=2 valid), so excluding one never
            empties a window's attribution entirely.
    """
    n = len(times)
    _, speed, runs = _raw_stance_candidates(times, xyz, config)
    valid = ~np.isnan(xyz).any(axis=1)
    discarded = [(s, e) for s, e in runs if (e - s + 1) < config.min_contact_frames]

    def _cameras_missing_at(f: int) -> List[str]:
        return sorted(role for role, cv in cam_valid.items() if role != exclude_camera and not cv[f])

    results = []
    for run_start, run_end in discarded:
        dropped_by = set()
        window_start, window_end = run_start, run_end

        j = run_start - 1
        while j >= 0:
            if not valid[j]:
                dropped_by.update(_cameras_missing_at(j))
                window_start = j
                break
            elif np.isnan(speed[j]):
                # Valid position, but its own predecessor blocked its
                # speed -- keep walking back through the chain.
                window_start = j
                j -= 1
            else:
                break  # a real (fast) swing frame: a genuine boundary, not a gap.

        if run_end + 1 < n and not valid[run_end + 1]:
            dropped_by.update(_cameras_missing_at(run_end + 1))
            window_end = run_end + 1

        if dropped_by:
            results.append(DiscardedStance(window_start, window_end, sorted(dropped_by)))

    return results


def _stride_lengths(events: PawEvents, xyz: np.ndarray, forward: np.ndarray) -> np.ndarray:
    """Forward distance from each stance phase's liftoff to the next touchdown."""
    if events.touchdown_frames.size < 2:
        return np.array([])
    liftoff_pos = xyz[events.liftoff_frames[:-1]]
    touchdown_pos = xyz[events.touchdown_frames[1:]]
    return (touchdown_pos - liftoff_pos) @ forward


def _step_lengths(
    events: PawEvents,
    contra_events: PawEvents,
    xyz: np.ndarray,
    contra_xyz: np.ndarray,
    forward: np.ndarray,
) -> np.ndarray:
    """Forward distance from each touchdown to the contralateral paw's most recent prior touchdown."""
    if events.touchdown_frames.size == 0 or contra_events.touchdown_frames.size == 0:
        return np.array([])
    contra_idx = np.searchsorted(contra_events.touchdown_frames, events.touchdown_frames, side="right") - 1
    has_prior = contra_idx >= 0
    if not has_prior.any():
        return np.array([])
    this_pos = xyz[events.touchdown_frames[has_prior]]
    contra_frames = contra_events.touchdown_frames[contra_idx[has_prior]]
    contra_pos = contra_xyz[contra_frames]
    return (this_pos - contra_pos) @ forward


def compute_trial_metrics(
    csv_path: PathLike,
    session_name: str,
    rat_id: str,
    config: Optional[GaitConfig] = None,
) -> TrialMetrics:
    """Compute one trial's gait metrics from its triangulated ``pose_3d.csv``.

    Args:
        csv_path: This trial's ``pose_3d.csv``, as written by
            :func:`alligaitor.pipeline.run_session`.
        session_name: Trial identifier, carried through to the report.
        rat_id: Which rat this trial belongs to (see
            :attr:`alligaitor.config.SessionConfig.rat_id`); sessions
            sharing a ``rat_id`` land on the same spreadsheet tab.
        config: Stance/swing detection thresholds; defaults to
            :class:`GaitConfig`'s own defaults.
    """
    config = config or GaitConfig()
    times, positions, _ = load_pose_3d(csv_path)

    missing = [node for node in (REFERENCE_NODE, *PAW_NODES) if node not in positions]
    if missing:
        raise ValueError(f"pose_3d CSV '{csv_path}' is missing required node(s): {missing}")

    crossing_time_s, average_speed_mm_s, forward = _crossing_time_and_speed(times, positions[REFERENCE_NODE])

    # Bridged (short-gap-interpolated) paw positions drive stance
    # detection and every downstream paw measurement, so a touchdown/
    # liftoff landing on a bridged frame still reports a real position
    # rather than mixing bridged and raw coordinates inconsistently.
    bridged = {paw: bridge_short_gaps(positions[paw], config.max_bridge_gap_frames) for paw in PAW_NODES}

    events = {paw: _detect_paw_events(times, bridged[paw], config) for paw in PAW_NODES}

    stride_length_mm, step_length_mm, ground_contact_time_s = {}, {}, {}
    n_contacts, n_strides, n_steps = {}, {}, {}

    for paw in PAW_NODES:
        ev = events[paw]
        contra_ev = events[CONTRALATERAL[paw]]

        strides = _stride_lengths(ev, bridged[paw], forward)
        steps = _step_lengths(ev, contra_ev, bridged[paw], bridged[CONTRALATERAL[paw]], forward)
        contact_durations = ev.liftoff_times - ev.touchdown_times

        stride_length_mm[paw] = float(np.mean(strides)) if strides.size else float("nan")
        step_length_mm[paw] = float(np.mean(steps)) if steps.size else float("nan")
        ground_contact_time_s[paw] = float(np.mean(contact_durations)) if contact_durations.size else float("nan")
        n_contacts[paw] = int(ev.touchdown_frames.size)
        n_strides[paw] = int(strides.size)
        n_steps[paw] = int(steps.size)

    return TrialMetrics(
        session_name=session_name,
        rat_id=rat_id,
        crossing_time_s=crossing_time_s,
        average_speed_mm_s=average_speed_mm_s,
        stride_length_mm=stride_length_mm,
        step_length_mm=step_length_mm,
        ground_contact_time_s=ground_contact_time_s,
        n_contacts=n_contacts,
        n_strides=n_strides,
        n_steps=n_steps,
        paw_events=events,
    )


def planted_mask(trial: TrialMetrics, n_frames: int) -> Dict[str, np.ndarray]:
    """Per-paw boolean array, ``True`` on frames within a detected stance phase.

    Expands :attr:`TrialMetrics.paw_events`' touchdown/liftoff frame pairs
    into a full per-frame mask -- e.g. for a validation video to color a
    paw node by its ground-contact state frame by frame.
    """
    masks = {}
    for paw in PAW_NODES:
        mask = np.zeros(n_frames, dtype=bool)
        ev = trial.paw_events[paw]
        for touchdown, liftoff in zip(ev.touchdown_frames, ev.liftoff_frames):
            mask[touchdown : liftoff + 1] = True
        masks[paw] = mask
    return masks


def save_paw_events_csv(trial: TrialMetrics, csv_path: PathLike) -> None:
    """Write every detected stance phase's frame/time window for one trial.

    Kept separate from the averaged metrics in the group workbook so a
    later validation-video tool can overlay exactly the frames treated as
    ground contact, without recomputing detection.
    """
    rows = [
        {
            "paw": paw,
            "touchdown_frame": int(td_f),
            "liftoff_frame": int(lo_f),
            "touchdown_time_s": float(td_t),
            "liftoff_time_s": float(lo_t),
        }
        for paw, ev in trial.paw_events.items()
        for td_f, lo_f, td_t, lo_t in zip(
            ev.touchdown_frames, ev.liftoff_frames, ev.touchdown_times, ev.liftoff_times
        )
    ]
    df = pd.DataFrame(
        rows, columns=["paw", "touchdown_frame", "liftoff_frame", "touchdown_time_s", "liftoff_time_s"]
    )
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)


def _trial_row(trial: TrialMetrics) -> dict:
    row = {
        "session": trial.session_name,
        "crossing_time_s": trial.crossing_time_s,
        "average_speed_mm_s": trial.average_speed_mm_s,
    }
    for paw in PAW_NODES:
        row[f"{paw}_stride_length_mm"] = trial.stride_length_mm[paw]
        row[f"{paw}_step_length_mm"] = trial.step_length_mm[paw]
        row[f"{paw}_ground_contact_time_s"] = trial.ground_contact_time_s[paw]
        row[f"{paw}_n_contacts"] = trial.n_contacts[paw]
        row[f"{paw}_n_strides"] = trial.n_strides[paw]
        row[f"{paw}_n_steps"] = trial.n_steps[paw]
    return row


def _safe_sheet_name(name: str, used: set) -> str:
    """Sanitize and de-duplicate an Excel sheet name (31-char limit, no ``[]:*?/\\``)."""
    cleaned = "".join(c if c not in _INVALID_SHEET_CHARS else "_" for c in str(name)).strip() or "rat"
    base = cleaned[:31]
    candidate = base
    n = 2
    while candidate in used:
        suffix = f"_{n}"
        candidate = base[: 31 - len(suffix)] + suffix
        n += 1
    used.add(candidate)
    return candidate


def write_group_report(trials: List[TrialMetrics], output_path: PathLike) -> None:
    """Write one group's gait-metrics workbook: one tab per distinct ``rat_id``.

    Each tab holds one row per trial (session) for that rat, covering the
    core parameters -- crossing time, average speed, and per-paw stride
    length, step length, and ground contact time -- plus each metric's
    underlying event count for a quick sanity check on detection quality.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    by_rat: Dict[str, List[TrialMetrics]] = {}
    for trial in trials:
        by_rat.setdefault(trial.rat_id, []).append(trial)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        used_names: set = set()
        if not by_rat:
            pd.DataFrame(columns=["session"]).to_excel(writer, sheet_name="Sheet1", index=False)
        for rat_id, rat_trials in by_rat.items():
            df = pd.DataFrame([_trial_row(trial) for trial in rat_trials])
            df.to_excel(writer, sheet_name=_safe_sheet_name(rat_id, used_names), index=False)
