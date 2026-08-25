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

import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from alligaitor.config import CAMERA_ROLES, GaitConfig
from alligaitor.inference import PoseTrack2D

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


def active_window(times: np.ndarray, ref_xyz: np.ndarray, config: GaitConfig) -> Tuple[int, int]:
    """First/last frame index of sustained whole-body motion.

    Trims any leading/trailing run of at least ``config.min_still_frames``
    frames whose whole-body (reference node) frame-to-frame speed stays
    below ``config.stillness_speed_threshold_mm_s`` -- e.g. a rat that
    stops moving well before the recording ends, whose paws can then
    jitter across the (much more sensitive) per-paw stance-speed
    threshold and look like a run of real steps taken in place. A brief
    slowdown in the middle of the trial isn't trimmed, only a run
    bordering either end. Falls back to the full triangulated range if
    the whole trial reads as "still" by this threshold, rather than
    collapsing to an empty window.
    """
    bridged = bridge_short_gaps(ref_xyz, config.max_bridge_gap_frames)
    valid = ~np.isnan(bridged).any(axis=1)
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size < 2:
        raise ValueError(
            f"Reference node '{REFERENCE_NODE}' has fewer than 2 triangulated frames; "
            "cannot determine an active window."
        )
    first, last = int(valid_idx[0]), int(valid_idx[-1])

    speed = np.full(len(bridged), np.nan)
    disp = np.linalg.norm(bridged[1:] - bridged[:-1], axis=1)
    dt = times[1:] - times[:-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        speed[1:] = disp / dt
    moving = speed >= config.stillness_speed_threshold_mm_s

    start = first
    m = first
    while m < last and not moving[m + 1]:
        m += 1
    if m - first >= config.min_still_frames:
        start = m

    end = last
    k = last
    while k > first and not moving[k]:
        k -= 1
    if last - k >= config.min_still_frames:
        end = k

    if start >= end:
        return first, last
    return start, end


def _crossing_time_and_speed(
    times: np.ndarray, ref_xyz: np.ndarray, config: GaitConfig
) -> "tuple[float, float, np.ndarray, Tuple[int, int]]":
    """Return (crossing time, average speed, unit forward direction, active window).

    Restricted to the active window (see :func:`active_window`) rather
    than the reference node's raw first-to-last triangulated frame, so
    idle time before the rat starts moving or after it stops doesn't
    inflate the crossing time or drag down the average speed.
    """
    start, end = active_window(times, ref_xyz, config)

    window = ref_xyz[start : end + 1]
    valid = ~np.isnan(window).any(axis=1)
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size < 2:
        raise ValueError(
            f"Reference node '{REFERENCE_NODE}' has fewer than 2 triangulated frames "
            "in its active window; cannot determine crossing time or direction."
        )

    first_idx, last_idx = start + valid_idx[0], start + valid_idx[-1]
    net_displacement = ref_xyz[last_idx] - ref_xyz[first_idx]
    net_distance = np.linalg.norm(net_displacement)
    if net_distance == 0:
        raise ValueError(
            f"Reference node '{REFERENCE_NODE}' shows no net displacement; "
            "cannot determine crossing direction."
        )
    forward = net_displacement / net_distance

    crossing_time_s = times[last_idx] - times[first_idx]

    pts = ref_xyz[start + valid_idx]
    path_length = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
    average_speed_mm_s = path_length / crossing_time_s

    return float(crossing_time_s), float(average_speed_mm_s), forward, (start, end)


def _raw_stance_candidates(
    times: np.ndarray, xyz: np.ndarray, config: GaitConfig
) -> "tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]":
    """Shared groundwork for stance detection.

    Returns ``(valid, speed, runs)``: whether the paw is triangulated on
    each frame, frame-to-frame speed, and every maximal run of raw
    "planted" frames (speed below ``config.speed_threshold_mm_s``), before
    the ``min_contact_frames`` length filter. Used by both
    :func:`_detect_paw_events` and :func:`find_camera_caused_discards`.

    Speed at each frame is the *minimum* of its backward- and
    forward-difference estimates (whichever neighbor is triangulated),
    not backward-only. A frame immediately following an untriangulated
    gap has its backward difference span the whole gap, inflating the
    apparent speed right at the frames most likely to be a real stance's
    loading-response onset -- initial paw-ground contact, before the
    animal's weight has fully transferred, which by the standard gait-cycle
    definition (initial contact -> toe-off) still counts as stance even
    though the paw may still show some settling velocity. Preferring
    whichever direction doesn't cross a gap avoids that inflation without
    requiring a second confirming frame, which at this frame rate risks
    dropping genuinely brief stances entirely. (A locally-smoothed/
    regression-based estimate was also tried and rejected -- more variable
    on this rig's sparse, gap-heavy trajectories, sometimes fragmenting a
    stance further instead of un-fragmenting it.)
    """
    n = len(times)
    valid = ~np.isnan(xyz).any(axis=1)

    disp = np.linalg.norm(xyz[1:] - xyz[:-1], axis=1)
    dt = times[1:] - times[:-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        step_speed = disp / dt  # step_speed[k] = speed of the transition frame k -> frame k+1

    backward = np.full(n, np.nan)
    backward[1:] = step_speed
    forward = np.full(n, np.nan)
    forward[:-1] = step_speed
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN slice at isolated valid frames
        speed = np.nanmin(np.stack([backward, forward]), axis=0)

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
    frame-to-frame speed (see :func:`_raw_stance_candidates`) is below
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
    """An untriangulated gap that could plausibly be hiding a real step.

    Attributes:
        start_frame: First frame of the gap.
        end_frame: Last frame of the gap.
        dropped_by: Camera role(s) missing a valid detection somewhere
            in the gap.
    """

    start_frame: int
    end_frame: int
    dropped_by: List[str]


def find_camera_caused_discards(
    bridged_xyz: np.ndarray,
    cam_valid: Dict[str, np.ndarray],
    exclude_camera: Optional[str] = None,
) -> List[DiscardedStance]:
    """Find every untriangulated gap bounded by a triangulated frame on
    both sides, and which camera(s) -- other than ``exclude_camera`` --
    were missing somewhere in it.

    ``bridged_xyz`` should already have short gaps interpolated (see
    :func:`bridge_short_gaps`), so what's left here is exactly the
    gaps long/unresolved enough to matter -- the same trajectory
    :func:`_qualifying_runs` uses to decide whether two detected steps
    can be trusted as truly consecutive. (An earlier version of this
    function instead looked for stance candidates *rejected* by
    ``min_contact_frames`` for being too short; that check went silently
    vacuous once ``min_contact_frames`` defaulted to 1, since nothing is
    ever too short to survive anymore. Checking the bridged trajectory
    directly for real gaps doesn't depend on that threshold at all.)

    A gap touching either end of the trial (no bounding valid frame on
    one side) isn't reported -- there's no before/after step it could
    plausibly be separating.

    Args:
        bridged_xyz: One paw's ``(n_frames, 3)`` bridged positions.
        cam_valid: Per camera role, an ``(n_frames,)`` boolean array --
            whether that camera had a valid (aligned) 2D detection for
            this paw on each frame. See
            :func:`alligaitor.triangulation.align_tracks_by_time`.
        exclude_camera: A camera role never counted as having dropped the
            paw -- e.g. :data:`FAR_SIDE_CAMERA`\\ [paw], since that
            camera not seeing this paw is expected, not a failure. A
            genuine gap always has at least one non-excluded camera
            missing somewhere in it: triangulation needs >=2 valid
            cameras, so if both non-excluded cameras were valid
            throughout, there'd be no gap to find in the first place.
    """
    valid = ~np.isnan(bridged_xyz).any(axis=1)
    n = len(valid)

    def _cameras_missing_at(f: int) -> "set[str]":
        return {role for role, cv in cam_valid.items() if role != exclude_camera and not cv[f]}

    results = []
    i = 0
    while i < n:
        if valid[i]:
            i += 1
            continue
        j = i
        while j < n and not valid[j]:
            j += 1
        if i > 0 and j < n:  # interior gap only -- skip leading/trailing edges
            dropped_by = set()
            for f in range(i, j):
                dropped_by |= _cameras_missing_at(f)
            if dropped_by:
                results.append(DiscardedStance(i, j - 1, sorted(dropped_by)))
        i = j

    return results


def cam_valid_by_paw_from_aligned(aligned: Dict[str, PoseTrack2D]) -> Dict[str, Dict[str, np.ndarray]]:
    """Per paw, per camera role, whether that camera had a valid 2D
    detection on each shared-timeline frame -- the ``cam_valid`` input
    :func:`find_camera_caused_discards` needs, derived from tracks
    already resampled onto the shared timeline (see
    :func:`alligaitor.triangulation.align_tracks_by_time`).
    """
    cam_valid_by_paw = {}
    for paw in PAW_NODES:
        node_idx = {role: aligned[role].node_names.index(paw) for role in CAMERA_ROLES}
        cam_valid_by_paw[paw] = {
            role: ~np.isnan(aligned[role].points[:, node_idx[role], :]).any(axis=1) for role in CAMERA_ROLES
        }
    return cam_valid_by_paw


def compute_discards_by_paw(
    positions: Dict[str, np.ndarray],
    cam_valid_by_paw: Dict[str, Dict[str, np.ndarray]],
    config: GaitConfig,
) -> Dict[str, List[DiscardedStance]]:
    """Per paw, its camera-caused discarded gaps (see
    :func:`find_camera_caused_discards`), bridging short gaps first --
    matching what :func:`compute_trial_metrics` itself does, so this is
    checking against the same trajectory the real stance detection sees.
    """
    return {
        paw: find_camera_caused_discards(
            bridge_short_gaps(positions[paw], config.max_bridge_gap_frames),
            cam_valid_by_paw[paw],
            exclude_camera=FAR_SIDE_CAMERA[paw],
        )
        for paw in PAW_NODES
    }


def find_stride_length_outliers(
    events: PawEvents,
    xyz: np.ndarray,
    forward: np.ndarray,
    ratio: float,
) -> List[Tuple[int, int, float, float]]:
    """Adjacent stride pairs whose length exceeds ``ratio`` times this
    paw's own median stride length in the trial.

    A stride long enough to plausibly be two strides' worth of distance
    rather than one usually means a real stance sat in between that the
    speed classifier failed to recognize -- triangulation can be clean
    the entire way through and still miss a brief plant, which is exactly
    what :func:`find_camera_caused_discards` (a gap-based check) cannot
    catch. The two are independent and complementary, not overlapping.

    Returns a list of ``(liftoff_frame, touchdown_frame, stride_length_mm,
    median_stride_length_mm)``, one entry per outlier stride.
    """
    strides = _stride_lengths(events, xyz, forward)
    if strides.size < 2:
        return []
    median = float(np.median(strides))
    if median <= 0:
        return []
    return [
        (int(events.liftoff_frames[i]), int(events.touchdown_frames[i + 1]), float(s), median)
        for i, s in enumerate(strides)
        if s > median * ratio
    ]


def _qualifying_runs(
    events: PawEvents,
    bridged_xyz: np.ndarray,
    forward: np.ndarray,
    min_run: int,
    stride_length_outlier_ratio: float,
) -> List[Tuple[int, int]]:
    """Maximal runs of consecutive accepted stance events, as ``(start, end)``
    index pairs into ``events``' arrays, at least ``min_run`` events long.

    A pair of adjacent events stays in the same run only if *both* hold:
    no remaining untriangulated frame -- *after* bridging (see
    :func:`bridge_short_gaps`) -- anywhere in the swing between them, and
    the stride between them isn't a
    :func:`find_stride_length_outliers`-flagged outlier. The first check
    catches a triangulation gap that could be hiding a real step; the
    second catches a real step missed despite clean triangulation. Neither
    alone is sufficient -- see :func:`restrict_to_consecutive_runs`.

    This used to check for a :func:`find_camera_caused_discards` window
    instead of the bridged trajectory directly, but that specifically
    flags a raw candidate run *rejected* for being too short -- and with
    :attr:`alligaitor.config.GaitConfig.min_contact_frames` at its
    current default of 1, no candidate is ever too short to survive, so
    that check had gone permanently vacuous (always "no discards found",
    regardless of how much real data was actually missing).
    """
    n = events.touchdown_frames.size
    if n == 0:
        return []

    valid = ~np.isnan(bridged_xyz).any(axis=1)
    gap_clean = np.ones(max(n - 1, 0), dtype=bool)
    for i in range(n - 1):
        gap_start, gap_end = events.liftoff_frames[i], events.touchdown_frames[i + 1]
        gap_clean[i] = bool(valid[gap_start : gap_end + 1].all())

    strides = _stride_lengths(events, bridged_xyz, forward)
    stride_clean = np.ones(max(n - 1, 0), dtype=bool)
    if strides.size:
        median = float(np.median(strides))
        if median > 0:
            stride_clean = strides <= median * stride_length_outlier_ratio

    clean_pair = gap_clean & stride_clean

    runs = []
    start = 0
    for i in range(n - 1):
        if not clean_pair[i]:
            runs.append((start, i))
            start = i + 1
    runs.append((start, n - 1))

    return [(s, e) for s, e in runs if (e - s + 1) >= min_run]


def restrict_to_consecutive_runs(
    trial: TrialMetrics,
    times: np.ndarray,
    positions: Dict[str, np.ndarray],
    config: GaitConfig,
) -> TrialMetrics:
    """Recompute stride/step/ground-contact-time averages from only the
    steps that are part of a qualifying run (see
    :attr:`alligaitor.config.GaitConfig.min_consecutive_steps` and
    :func:`_qualifying_runs`) -- an isolated good detection outside any
    such run doesn't feed the averages, and a paw with no qualifying run
    at all reports ``NaN``. ``trial.paw_events`` (every detected event,
    whether or not it ended up counted) is left untouched, so the
    validation video and raw event log still show everything that was
    actually detected -- only the summary numbers are restricted.

    Args:
        trial: Already-computed metrics (see :func:`compute_trial_metrics`).
        times: This trial's per-frame timestamps (see :func:`load_pose_3d`).
        positions: This trial's per-node positions (see :func:`load_pose_3d`).
        config: The same :class:`GaitConfig` ``trial`` was computed with.
    """
    _, _, forward, (start, end) = _crossing_time_and_speed(times, positions[REFERENCE_NODE], config)

    stride_length_mm, step_length_mm, ground_contact_time_s = {}, {}, {}
    n_contacts, n_strides, n_steps = {}, {}, {}

    for paw in PAW_NODES:
        events = trial.paw_events[paw]
        contra_events = trial.paw_events[CONTRALATERAL[paw]]
        xyz = positions[paw].copy()
        xyz[:start] = np.nan
        xyz[end + 1 :] = np.nan
        bridged = bridge_short_gaps(xyz, config.max_bridge_gap_frames)
        runs = _qualifying_runs(
            events, bridged, forward, config.min_consecutive_steps, config.stride_length_outlier_ratio
        )

        if not runs:
            stride_length_mm[paw] = float("nan")
            step_length_mm[paw] = float("nan")
            ground_contact_time_s[paw] = float("nan")
            n_contacts[paw] = 0
            n_strides[paw] = 0
            n_steps[paw] = 0
            continue

        strides_parts, steps_parts, durations_parts = [], [], []
        total_contacts = 0
        for s, e in runs:
            run_events = PawEvents(
                touchdown_frames=events.touchdown_frames[s : e + 1],
                liftoff_frames=events.liftoff_frames[s : e + 1],
                touchdown_times=events.touchdown_times[s : e + 1],
                liftoff_times=events.liftoff_times[s : e + 1],
            )
            strides_parts.append(_stride_lengths(run_events, positions[paw], forward))
            steps_parts.append(
                _step_lengths(run_events, contra_events, positions[paw], positions[CONTRALATERAL[paw]], forward)
            )
            durations_parts.append(run_events.liftoff_times - run_events.touchdown_times)
            total_contacts += run_events.touchdown_frames.size

        strides = np.concatenate(strides_parts)
        steps = np.concatenate(steps_parts)
        durations = np.concatenate(durations_parts)

        stride_length_mm[paw] = float(np.mean(strides)) if strides.size else float("nan")
        step_length_mm[paw] = float(np.mean(steps)) if steps.size else float("nan")
        ground_contact_time_s[paw] = float(np.mean(durations)) if durations.size else float("nan")
        n_contacts[paw] = total_contacts
        n_strides[paw] = int(strides.size)
        n_steps[paw] = int(steps.size)

    return replace(
        trial,
        stride_length_mm=stride_length_mm,
        step_length_mm=step_length_mm,
        ground_contact_time_s=ground_contact_time_s,
        n_contacts=n_contacts,
        n_strides=n_strides,
        n_steps=n_steps,
    )


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

    crossing_time_s, average_speed_mm_s, forward, (start, end) = _crossing_time_and_speed(
        times, positions[REFERENCE_NODE], config
    )

    # Bridged (short-gap-interpolated) paw positions drive stance
    # detection and every downstream paw measurement, so a touchdown/
    # liftoff landing on a bridged frame still reports a real position
    # rather than mixing bridged and raw coordinates inconsistently.
    # Masked to the active window first (see active_window) so a paw
    # jittering in place after the rat has already stopped moving can't
    # be detected as a stance phase at all.
    bridged = {}
    for paw in PAW_NODES:
        xyz = positions[paw].copy()
        xyz[:start] = np.nan
        xyz[end + 1 :] = np.nan
        bridged[paw] = bridge_short_gaps(xyz, config.max_bridge_gap_frames)

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


@dataclass
class PawWindow:
    """The single time window a validation-video viewer should highlight
    for one paw: its longest qualifying run (see :func:`_qualifying_runs`)
    if it has one, else its longest raw detected run regardless of length
    -- exactly "the used run, or the longest run if none cross the
    threshold" a reviewer needs to see. ``usable`` mirrors which case this
    is, so a caller doesn't have to re-derive it from ``_paw_has_no_usable_run``.
    """

    start_frame: int
    end_frame: int
    start_s: float
    end_s: float
    duration_s: float
    usable: bool


def _longest_raw_run(events: PawEvents) -> Optional[Tuple[int, int]]:
    """Index range (into `events`' arrays) of the longest single detected
    stance phase, or ``None`` if nothing was ever detected at all. Used as
    the fallback window for a paw with zero qualifying runs -- there's no
    "used" run to show, but showing nothing at all would leave a reviewer
    unable to tell where the pipeline even looked."""
    n = events.touchdown_frames.size
    if n == 0:
        return None
    durations = events.liftoff_times - events.touchdown_times
    i = int(np.argmax(durations))
    return i, i


def paw_usability_windows(
    trial: TrialMetrics,
    times: np.ndarray,
    positions: Dict[str, np.ndarray],
    config: GaitConfig,
) -> Dict[str, Optional[PawWindow]]:
    """Per paw, the single window a validation-video viewer should
    highlight -- the longest qualifying run (see :func:`_qualifying_runs`,
    the same run-detection :func:`restrict_to_consecutive_runs` uses for
    its averages) if the paw has one, else its longest raw detected stance
    phase. ``None`` only when the paw was never detected planted at all in
    this trial.
    """
    _, _, forward, (start, end) = _crossing_time_and_speed(times, positions[REFERENCE_NODE], config)

    windows: Dict[str, Optional[PawWindow]] = {}
    for paw in PAW_NODES:
        events = trial.paw_events[paw]
        xyz = positions[paw].copy()
        xyz[:start] = np.nan
        xyz[end + 1 :] = np.nan
        bridged = bridge_short_gaps(xyz, config.max_bridge_gap_frames)
        runs = _qualifying_runs(
            events, bridged, forward, config.min_consecutive_steps, config.stride_length_outlier_ratio
        )

        if runs:
            s, e = max(runs, key=lambda r: events.liftoff_times[r[1]] - events.touchdown_times[r[0]])
            usable = True
        else:
            fallback = _longest_raw_run(events)
            if fallback is None:
                windows[paw] = None
                continue
            s, e = fallback
            usable = False

        windows[paw] = PawWindow(
            start_frame=int(events.touchdown_frames[s]),
            end_frame=int(events.liftoff_frames[e]),
            start_s=float(events.touchdown_times[s]),
            end_s=float(events.liftoff_times[e]),
            duration_s=float(events.liftoff_times[e] - events.touchdown_times[s]),
            usable=usable,
        )
    return windows


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


_PAW_LABELS: Dict[str, str] = {
    "left-forepaw": "Left forepaw",
    "right-forepaw": "Right forepaw",
    "left-hind-paw": "Left hind paw",
    "right-hind-paw": "Right hind paw",
}

# (attribute name on TrialMetrics, column header, per-session number format).
# Counts are true integers per session (a real, possibly-zero tally --
# see _paw_has_no_usable_run's docstring for why that's different from
# NaN), so "0" is the right per-session format; the averages table
# overrides every column to "0.00" instead (see _write_paw_block), since
# an average of integer counts across crossings is itself a fraction, not
# a count.
_STAT_COLUMNS: Tuple[Tuple[str, str, str], ...] = (
    ("stride_length_mm", "Stride Length (mm)", "0.00"),
    ("step_length_mm", "Step Length (mm)", "0.00"),
    ("ground_contact_time_s", "Ground Contact Time (s)", "0.00"),
    ("n_contacts", "Contacts (n)", "0"),
    ("n_strides", "Strides (n)", "0"),
    ("n_steps", "Steps (n)", "0"),
)
_NOTES_LABEL = "Notes"
_N_STAT_COLUMNS = 1 + len(_STAT_COLUMNS)  # "Paw" + the stat columns
_NOTES_COLUMN = _N_STAT_COLUMNS + 1
_N_COLUMNS = _NOTES_COLUMN  # "Paw" + the stat columns + Notes

_TITLE_FONT = Font(bold=True, color="FFFFFF", size=12)
_TITLE_FILL = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
_LABEL_FONT = Font(bold=True)
_HEADER_FONT = Font(bold=True)
_HEADER_FILL = PatternFill(start_color="DCE6F1", end_color="DCE6F1", fill_type="solid")
_BAD_FILL = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
_BAD_FONT = Font(color="9C0006")
_CLEAR_FILL = PatternFill(fill_type=None)
_CLEAR_FONT = Font()


def _paw_has_no_usable_run(trial: TrialMetrics, paw: str) -> bool:
    """True if `paw` never formed a run clean/long enough to trust for
    stride length, step length, or ground-contact time on this crossing
    (all three NaN -- see GaitConfig.min_consecutive_steps). n_contacts
    et al. can still be a real, nonzero count even when this is True: a
    handful of raw detections that never added up to a qualifying run is
    exactly the case this flags, not "zero contacts detected"."""
    return (
        np.isnan(trial.stride_length_mm[paw])
        and np.isnan(trial.step_length_mm[paw])
        and np.isnan(trial.ground_contact_time_s[paw])
    )


def _nanmean(values) -> float:
    with warnings.catch_warnings():
        # An all-NaN slice (a paw with zero qualifying detections across
        # every crossing) warns by default -- NaN is exactly the right
        # answer there, not a bug worth surfacing.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return float(np.nanmean(values))


def _write_cell(ws, row: int, col: int, value, number_format: str, bad: bool):
    display_value = None if value is None or (isinstance(value, float) and np.isnan(value)) else value
    cell = ws.cell(row=row, column=col, value=display_value)
    cell.number_format = number_format
    if bad:
        cell.fill = _BAD_FILL
        cell.font = _BAD_FONT
    return cell


def _write_paw_block(
    ws,
    start_row: int,
    title: str,
    crossing_time_s: float,
    average_speed_cm_s: float,
    get_value: Callable[[str, str], float],
    is_bad: Callable[[str], bool],
    stat_columns: Tuple[Tuple[str, str, str], ...] = _STAT_COLUMNS,
    note_getter: Callable[[str], str] = lambda paw: "",
) -> int:
    """Writes one titled block -- crossing time/speed, then a paw x stat
    table -- starting at `start_row`, and returns the row the next block
    should start at. `get_value(stat, paw)` and `is_bad(paw)` abstract
    over "one trial's own numbers" vs. "this rat's per-paw averages" (see
    write_group_report), so this one function lays out both. `note_getter(paw)`
    fills the trailing Notes column -- empty by default, used to record why
    a paw was manually flagged invalid (see :func:`annotate_manual_flag`).
    """
    row = start_row
    title_cell = ws.cell(row=row, column=1, value=title)
    title_cell.font = _TITLE_FONT
    title_cell.fill = _TITLE_FILL
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=_N_COLUMNS)
    for col in range(1, _N_COLUMNS + 1):
        ws.cell(row=row, column=col).fill = _TITLE_FILL
    row += 1

    ws.cell(row=row, column=1, value="Crossing Time (s)").font = _LABEL_FONT
    _write_cell(ws, row, 2, crossing_time_s, "0.00", bad=False)
    ws.cell(row=row, column=3, value="Average Speed (cm/s)").font = _LABEL_FONT
    _write_cell(ws, row, 4, average_speed_cm_s, "0.00", bad=False)
    row += 2  # blank spacer before the paw table

    ws.cell(row=row, column=1, value="Paw").font = _HEADER_FONT
    ws.cell(row=row, column=1).fill = _HEADER_FILL
    for col, (_, label, _fmt) in enumerate(stat_columns, start=2):
        cell = ws.cell(row=row, column=col, value=label)
        cell.font = _HEADER_FONT
        cell.fill = _HEADER_FILL
    notes_header = ws.cell(row=row, column=_NOTES_COLUMN, value=_NOTES_LABEL)
    notes_header.font = _HEADER_FONT
    notes_header.fill = _HEADER_FILL
    row += 1

    for paw in PAW_NODES:
        bad = is_bad(paw)
        name_cell = ws.cell(row=row, column=1, value=_PAW_LABELS[paw])
        if bad:
            name_cell.fill = _BAD_FILL
            name_cell.font = _BAD_FONT
        for col, (stat, _label, fmt) in enumerate(stat_columns, start=2):
            _write_cell(ws, row, col, get_value(stat, paw), fmt, bad=bad)
        note_cell = ws.cell(row=row, column=_NOTES_COLUMN, value=note_getter(paw) or None)
        if bad:
            note_cell.fill = _BAD_FILL
            note_cell.font = _BAD_FONT
        row += 1

    return row + 2  # blank spacer before the next block


def _format_sheet(ws):
    ws.column_dimensions["A"].width = 22
    for col in range(2, _N_COLUMNS + 1):
        ws.column_dimensions[get_column_letter(col)].width = 20
    ws.column_dimensions[get_column_letter(_NOTES_COLUMN)].width = 40
    for row in ws.iter_rows():
        for cell in row:
            if cell.value is not None:
                cell.alignment = Alignment(horizontal="left" if cell.column == 1 else "right")


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


def write_group_report(
    trials: List[TrialMetrics],
    output_path: PathLike,
    manual_flags: Optional[Dict[str, Tuple[set, str]]] = None,
) -> None:
    """Write one group's gait-metrics workbook: one tab per distinct ``rat_id``.

    Each tab stacks one titled block per trial (session) for that rat --
    crossing time and average speed, then a paw x stat table (rows are
    the four paws, columns are stride length/step length/ground contact
    time/event counts, all in natural-language, unit-bearing headers) --
    followed, when a rat has more than one crossing in this group, by a
    final "Average" block in the same shape. A paw with no run in a
    given crossing clean/long enough to trust for its length/timing
    columns (see :func:`_paw_has_no_usable_run`) has its whole row
    highlighted; the Average block's per-(paw, stat) means are NaN-aware,
    so a paw's bad crossings don't count against its average from the
    crossings where it did produce something usable.

    Args:
        manual_flags: Session name -> (flagged paw names, note), carried
            forward from :func:`alligaitor.validation.load_manual_flags` so
            a regenerated workbook keeps highlighting a paw a reviewer
            already flagged invalid by hand, exactly as
            :func:`annotate_manual_flag` would highlight it on an
            already-written workbook. A paw already bad from
            :func:`_paw_has_no_usable_run` stays bad regardless of what's
            here -- this only ever adds highlighting, never removes it.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manual_flags = manual_flags or {}

    by_rat: Dict[str, List[TrialMetrics]] = {}
    for trial in trials:
        by_rat.setdefault(trial.rat_id, []).append(trial)

    wb = Workbook()
    wb.remove(wb.active)  # replaced by a real sheet per rat below (or the placeholder, if there are none)

    if not by_rat:
        ws = wb.create_sheet("Sheet1")
        ws.cell(row=1, column=1, value="No sessions in this group.")
    else:
        used_names: set = set()
        for rat_id, rat_trials in by_rat.items():
            ws = wb.create_sheet(_safe_sheet_name(rat_id, used_names))
            row = 1
            for trial in rat_trials:
                flagged_paws, note = manual_flags.get(trial.session_name, (set(), ""))
                row = _write_paw_block(
                    ws, row,
                    title=f"Session: {trial.session_name}",
                    crossing_time_s=trial.crossing_time_s,
                    average_speed_cm_s=trial.average_speed_mm_s / 10.0,
                    get_value=lambda stat, paw, t=trial: getattr(t, stat)[paw],
                    is_bad=lambda paw, t=trial, fp=flagged_paws: _paw_has_no_usable_run(t, paw) or paw in fp,
                    note_getter=lambda paw, fp=flagged_paws, n=note: n if paw in fp else "",
                )

            if len(rat_trials) > 1:
                avg_stat_columns = tuple((stat, label, "0.00") for stat, label, _fmt in _STAT_COLUMNS)
                row = _write_paw_block(
                    ws, row,
                    title="Average",
                    crossing_time_s=_nanmean([t.crossing_time_s for t in rat_trials]),
                    average_speed_cm_s=_nanmean([t.average_speed_mm_s for t in rat_trials]) / 10.0,
                    get_value=lambda stat, paw: _nanmean([getattr(t, stat)[paw] for t in rat_trials]),
                    is_bad=lambda paw: all(
                        _paw_has_no_usable_run(t, paw) or paw in manual_flags.get(t.session_name, (set(), ""))[0]
                        for t in rat_trials
                    ),
                    stat_columns=avg_stat_columns,
                )

            _format_sheet(ws)

    wb.save(output_path)


def _find_session_paw_row(ws, session_name: str, paw: str) -> Optional[int]:
    """Row index of `paw`'s row within the ``"Session: {session_name}"``
    block on `ws`, or ``None`` if that block or paw row isn't present --
    e.g. the workbook predates this session, or was regenerated with
    different sessions. Searches a bounded window below the title row
    rather than assuming :func:`_write_paw_block`'s exact row offsets, so
    it keeps working even if that layout changes shape slightly.
    """
    title = f"Session: {session_name}"
    for row in ws.iter_rows(min_col=1, max_col=1):
        cell = row[0]
        if cell.value != title:
            continue
        for r in range(cell.row + 1, cell.row + 16):
            if ws.cell(row=r, column=1).value == _PAW_LABELS[paw]:
                return r
        return None
    return None


def annotate_manual_flag(
    xlsx_path: PathLike,
    rat_id: str,
    session_name: str,
    paw: str,
    auto_usable: bool,
    flagged: bool,
    note: str = "",
) -> bool:
    """Patch one paw's row in an already-written group workbook to reflect
    a reviewer's manual flag, without regenerating the whole report from
    :class:`TrialMetrics`.

    The row is highlighted (and `note` written to its Notes cell) when
    ``flagged`` or ``not auto_usable`` -- unflagging a paw the automatic
    detection already called unusable (``auto_usable=False``) leaves it
    highlighted, since that's a real "no usable run" finding this manual
    action didn't create and shouldn't be able to erase. Only values/number
    formats are left untouched; this only ever changes fill/font/notes.

    Returns:
        ``True`` if the target row was found and patched, ``False`` if the
        workbook has no matching rat sheet / session block / paw row (the
        caller should warn rather than assume the flag took effect).
    """
    from openpyxl import load_workbook

    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        return False

    wb = load_workbook(xlsx_path)
    candidate = _safe_sheet_name(rat_id, set())
    ws = wb[candidate] if candidate in wb.sheetnames else next(
        (wb[name] for name in wb.sheetnames if name.startswith(candidate[: max(1, len(candidate) - 2)])), None
    )
    if ws is None:
        return False

    row = _find_session_paw_row(ws, session_name, paw)
    if row is None:
        return False

    bad = flagged or not auto_usable
    fill = _BAD_FILL if bad else _CLEAR_FILL
    font = _BAD_FONT if bad else _CLEAR_FONT
    for col in range(1, _N_STAT_COLUMNS + 1):
        cell = ws.cell(row=row, column=col)
        cell.fill = fill
        cell.font = font
    notes_cell = ws.cell(row=row, column=_NOTES_COLUMN, value=(note if flagged else None))
    notes_cell.fill = fill
    notes_cell.font = font

    wb.save(xlsx_path)
    return True
