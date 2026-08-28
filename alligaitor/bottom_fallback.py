"""Monocular fallback for triangulation dropouts, recovering NaN points from
the bottom camera alone when a paw is occluded from its near-side camera.

Intersects the bottom camera's viewing ray with the horizontal plane at a
height interpolated from the node's nearby triangulated frames; only X/Y
accuracy matters since gait metrics are speed-only. Toggled by
``PipelineConfig.bottom_fallback`` and invoked from
:mod:`alligaitor.pipeline`.
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
    center and a 2D pixel, in the calibration's reference frame."""
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
    alone, wherever it has a valid 2D detection there. Height at a
    to-be-filled frame is interpolated from that node's already-
    triangulated frames elsewhere in the session.

    Returns a ``(pose_3d, filled)`` pair: a new :class:`Pose3D` with gaps
    filled (filled points carry ``NaN`` reprojection error), and a
    boolean ``(n_frames, n_nodes)`` array marking which points were filled.
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
            continue  # not enough points to interpolate a height reference

        valid_frames = np.flatnonzero(valid)
        heights_valid = points[valid_frames, node_i, :] @ up_direction
        height_by_frame = np.interp(np.arange(n_frames), valid_frames, heights_valid)

        bidx = bottom_node_idx[node]
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
    """Splice every already-usable baseline paw run back into `trials`, so
    the fallback can only add usability a paw didn't already have, never
    remove it.

    For every paw with a usable run in `baseline_trials` (crossings
    computed with no fallback), overwrites that paw's per-paw fields in
    `trials` with the baseline's values and forces its
    ``bottom_fallback_fraction`` to ``0.0``. Returns `trials` unchanged if
    it and `baseline_trials` disagree on crossing count.
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
