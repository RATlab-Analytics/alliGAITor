"""Experimental monocular fallback for triangulation dropouts.

Standard triangulation (:func:`alligaitor.triangulation.triangulate`) needs
at least 2 of the 3 camera views to have a valid 2D detection for a given
(frame, node); the two side cameras only ever reliably see a paw on their
own near side, so in practice the two load-bearing views for any one paw
are the bottom camera and that paw's near-side camera (see
:mod:`scripts.find_triangulation_gap_frames`'s module docstring). A
big-bodied rat can occlude a paw from its near-side camera too, dropping
below the 2-view minimum even though the bottom camera -- looking straight
up through the tunnel floor -- still sees it fine.

This module recovers exactly those otherwise-NaN (frame, node) points from
the bottom camera alone: it intersects that camera's own viewing ray with
the horizontal plane at a height estimate linearly interpolated from the
same node's nearby *actually-triangulated* frames. This only needs to be
accurate along the platform plane (X/Y), not along height -- gait metrics
are speed-only, not height-based (see :mod:`alligaitor.gait`'s module
docstring), and the bottom camera's image plane is roughly parallel to the
platform, so it strongly constrains X/Y even when the assumed height is a
rough interpolation rather than a real measurement (see
:func:`alligaitor.triangulation.triangulate_axis_prioritized`'s docstring
for the same reasoning applied to the fully-triangulated case).

Toggled by ``PipelineConfig.bottom_fallback`` (see
:mod:`alligaitor.config`) and invoked from exactly two guarded call sites
in :mod:`alligaitor.pipeline` -- :func:`fill_gaps` from ``run_session``,
:func:`guard_against_regression` from ``run_group``. Deleting this file
and those two calls removes the feature entirely -- nothing else in the
pipeline reaches into it, and it reaches into the rest of the pipeline
only through public API (:class:`~alligaitor.triangulation.Pose3D`,
:func:`~alligaitor.triangulation.align_tracks_by_time`,
:func:`~alligaitor.calibration.world_up_direction`, and
:mod:`alligaitor.gait`'s public ``TrialMetrics``/``paw_usability_windows``
surface). The camera-ray/plane math in :func:`fill_gaps` is duplicated
from :mod:`alligaitor.triangulation` rather than imported, so this module
has no dependency on that module's private internals either.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from aniposelib.cameras import Camera, CameraGroup

from alligaitor import calibration
from alligaitor.gait import PAW_NODES, GaitConfig, TrialMetrics, paw_usability_windows
from alligaitor.inference import PoseTrack2D
from alligaitor.triangulation import Pose3D, align_tracks_by_time


def _camera_ray_world(cam: Camera, point_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The 3D ray (origin, unit direction) through one camera's optical
    center and a single 2D pixel, in the calibration's reference frame.
    Same construction as :func:`alligaitor.triangulation._camera_ray_world`
    -- duplicated, not imported, so this module has no dependency on that
    module's private internals (see this module's own docstring).
    """
    undistorted = cam.undistort_points(point_2d.reshape(1, 2)).reshape(2)
    direction_cam = np.array([undistorted[0], undistorted[1], 1.0])
    rotation, _ = cv2.Rodrigues(cam.get_rotation())
    origin = -rotation.T @ cam.get_translation().reshape(3)
    direction = rotation.T @ direction_cam
    direction = direction / np.linalg.norm(direction)
    return origin, direction


def _ray_plane_intersection(
    origin: np.ndarray, direction: np.ndarray, up_direction: np.ndarray, height: float
) -> Optional[np.ndarray]:
    """Where a ray crosses the horizontal plane at ``height``, or ``None``
    if the ray runs (near-)parallel to that plane."""
    denom = float(direction @ up_direction)
    if abs(denom) < 1e-6:
        return None
    s = (height - float(origin @ up_direction)) / denom
    return origin + s * direction


def fill_gaps(
    pose_3d: Pose3D,
    tracks: Dict[str, PoseTrack2D],
    cgroup: CameraGroup,
    fps_by_role: Dict[str, float],
) -> Tuple[Pose3D, np.ndarray]:
    """Fill NaN (frame, node) points in `pose_3d` from the bottom camera
    alone, wherever it has a valid 2D detection there.

    Height at a to-be-filled frame is linearly interpolated (clamped to
    the nearest endpoint outside that range, per ``np.interp``'s default)
    from that same node's own already-triangulated frames elsewhere in the
    session -- there's no independent height measurement available for a
    frame the bottom camera alone can see, so this borrows the node's
    typical height around that point in time rather than assuming a fixed
    platform height (which would be wrong for a paw mid-swing).

    Args:
        pose_3d: A session's standard triangulated result (from
            :func:`alligaitor.triangulation.triangulate`).
        tracks: The same per-camera 2D tracks `pose_3d` was triangulated
            from (mapping camera role to :class:`PoseTrack2D`).
        cgroup: The same calibrated camera group used to produce `pose_3d`.
        fps_by_role: The same per-camera frame rates used to produce
            `pose_3d` -- must match exactly, since the fallback needs the
            bottom camera's 2D points on the identical per-frame timeline
            `pose_3d.points` is already indexed by (see
            :func:`alligaitor.triangulation.align_tracks_by_time`).

    Returns:
        A ``(pose_3d, filled)`` pair: a new :class:`Pose3D`, same shape as
        the input, with any bottom-camera-recoverable gaps filled in
        (filled points carry ``NaN`` reprojection error -- there's no
        second view to reproject against and check); and a boolean
        ``(n_frames, n_nodes)`` array, ``True`` at exactly the points this
        call filled in, for a caller that wants to track how much of a
        session came from this fallback rather than real triangulation
        (see ``PipelineConfig.bottom_fallback``'s docstring).
    """
    cam_names = cgroup.get_names()
    filled = np.zeros(pose_3d.points.shape[:2], dtype=bool)
    if "bottom" not in tracks or "bottom" not in cam_names:
        return pose_3d, filled

    aligned = align_tracks_by_time(tracks, fps_by_role)
    bottom = aligned["bottom"]
    bottom_node_idx = {n: i for i, n in enumerate(bottom.node_names)}
    bottom_cam = cgroup.cameras[cam_names.index("bottom")]
    up_direction = calibration.world_up_direction(cgroup)

    points = pose_3d.points.copy()
    error = pose_3d.reprojection_error.copy()
    n_frames = points.shape[0]

    for node_i, node in enumerate(pose_3d.node_names):
        if node not in bottom_node_idx:
            continue
        valid = ~np.isnan(points[:, node_i, 0])
        if valid.sum() < 2:
            continue  # nothing to interpolate a height reference from

        valid_frames = np.flatnonzero(valid)
        heights_valid = points[valid_frames, node_i, :] @ up_direction
        height_by_frame = np.interp(np.arange(n_frames), valid_frames, heights_valid)

        bidx = bottom_node_idx[node]
        # `bottom` is aligned onto the exact same reference timeline
        # `pose_3d.points` is indexed by (both derived from the same
        # `tracks`/`fps_by_role` via align_tracks_by_time), so it always
        # has exactly `n_frames` rows here.
        bottom_xy = bottom.points[:, bidx]
        need_fill = ~valid & ~np.isnan(bottom_xy).any(axis=1)

        for f in np.flatnonzero(need_fill):
            origin, direction = _camera_ray_world(bottom_cam, bottom_xy[f])
            point = _ray_plane_intersection(origin, direction, up_direction, float(height_by_frame[f]))
            if point is None:
                continue
            points[f, node_i] = point
            error[f, node_i] = np.nan
            filled[f, node_i] = True

    return Pose3D(node_names=pose_3d.node_names, points=points, reprojection_error=error), filled


def guard_against_regression(
    trials: List[TrialMetrics],
    baseline_trials: List[TrialMetrics],
    baseline_times: np.ndarray,
    baseline_positions: Dict[str, np.ndarray],
    config: GaitConfig,
) -> List[TrialMetrics]:
    """Splice every already-usable baseline paw run back into `trials`,
    so the fallback can only ever add usability a paw didn't already have
    -- never take away one it did.

    Filling gaps can shift a paw's *own* stance/swing event detection
    (a newly-recovered frame elsewhere in the trajectory can move the
    active window or bridge/split runs differently -- see
    :func:`alligaitor.gait.paw_usability_windows`), so a run that was
    already clean without the fallback is not otherwise guaranteed to
    come out unchanged with it. This forces that guarantee directly: for
    every paw with a usable run in `baseline_trials` (the same crossings,
    computed with no fallback at all), every one of its per-paw fields --
    stride/step length, ground contact time, event counts, and the raw
    touchdown/liftoff events themselves -- is overwritten with the
    baseline's own values, and its ``bottom_fallback_fraction`` is forced
    to ``0.0`` (accurate: this paw's reported run no longer contains a
    single fallback-derived frame).

    Args:
        trials: The bottom-camera-fallback-enhanced crossings (already
            passed through :func:`alligaitor.gait.restrict_to_consecutive_runs`
            and :func:`alligaitor.gait.attach_bottom_fallback_fraction`).
        baseline_trials: The same session's crossings computed with the
            fallback entirely absent -- same ``config``, same
            :func:`alligaitor.gait.compute_crossing_metrics` /
            :func:`alligaitor.gait.restrict_to_consecutive_runs` pipeline,
            just fed the pre-fallback triangulation.
        baseline_times, baseline_positions: The baseline triangulation's
            own :func:`alligaitor.gait.load_pose_3d` output -- needed to
            recompute each baseline trial's usability windows.

    Returns:
        `trials`, unchanged for any paw that had no usable baseline run,
        spliced for any paw that did. If `trials` and `baseline_trials`
        disagree on crossing count (the reference node's own
        fallback-filled frames shifted crossing detection itself -- rare,
        but possible), `trials` is returned completely untouched, since
        there is no reliable crossing-to-crossing correspondence to guard
        with.
    """
    if len(trials) != len(baseline_trials):
        return trials

    guarded = []
    for trial, baseline_trial in zip(trials, baseline_trials):
        baseline_windows = paw_usability_windows(baseline_trial, baseline_times, baseline_positions, config)
        result = trial
        for paw in PAW_NODES:
            window = baseline_windows.get(paw)
            if window is None or not window.usable:
                continue
            result = replace(
                result,
                stride_length_mm={**result.stride_length_mm, paw: baseline_trial.stride_length_mm[paw]},
                step_length_mm={**result.step_length_mm, paw: baseline_trial.step_length_mm[paw]},
                ground_contact_time_s={**result.ground_contact_time_s, paw: baseline_trial.ground_contact_time_s[paw]},
                n_contacts={**result.n_contacts, paw: baseline_trial.n_contacts[paw]},
                n_strides={**result.n_strides, paw: baseline_trial.n_strides[paw]},
                n_steps={**result.n_steps, paw: baseline_trial.n_steps[paw]},
                paw_events={**result.paw_events, paw: baseline_trial.paw_events[paw]},
                bottom_fallback_fraction={**result.bottom_fallback_fraction, paw: 0.0},
            )
        guarded.append(result)
    return guarded
