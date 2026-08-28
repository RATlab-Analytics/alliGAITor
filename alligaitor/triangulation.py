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

"""3D triangulation across the three-camera rig, built on aniposelib.

Combines per-camera 2D pose predictions (see :mod:`alligaitor.inference`)
with a calibrated :class:`~aniposelib.cameras.CameraGroup` to reconstruct
3D keypoint trajectories. The side and bottom SLEAP-NN models declare
their skeleton nodes in different orders, so predictions are always
reindexed to a canonical node order by name before triangulating.

The rig's three cameras do not run at a matched frame rate (see
:mod:`alligaitor.timing`), so a session's per-camera 2D tracks are
resampled onto a shared timeline, by estimated recording time, before
triangulating -- frame index ``i`` in one camera's predictions does not
otherwise correspond to frame index ``i`` in another's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from aniposelib.cameras import Camera, CameraGroup

from alligaitor.config import CAMERA_ROLES
from alligaitor.inference import PoseTrack2D


@dataclass
class Pose3D:
    """Triangulated 3D pose for one session.

    Attributes:
        node_names: Skeleton node names, in the order matching the
            second axis of ``points`` and ``reprojection_error``.
        points: Array of shape ``(n_frames, n_nodes, 3)`` with 3D
            coordinates in the calibration's reference frame. Points that
            could not be triangulated (fewer than 2 cameras with a valid
            2D detection) are ``NaN``.
        reprojection_error: Array of shape ``(n_frames, n_nodes)`` with
            the mean per-point reprojection error, in pixels, averaged
            across cameras that had a valid detection.
    """

    node_names: List[str]
    points: np.ndarray
    reprojection_error: np.ndarray


def _canonical_node_order(tracks: Dict[str, PoseTrack2D]) -> List[str]:
    """Resolve a single node order shared by all camera views, by name."""
    node_sets = {role: set(t.node_names) for role, t in tracks.items()}
    reference = next(iter(node_sets.values()))
    mismatched = {role: nodes for role, nodes in node_sets.items() if nodes != reference}
    if mismatched:
        raise ValueError(
            "Camera views do not all track the same skeleton nodes: "
            f"{ {role: sorted(nodes) for role, nodes in node_sets.items()} }"
        )
    return sorted(reference)


def _reindex(track: PoseTrack2D, node_order: List[str]) -> np.ndarray:
    """Reorder ``track.points`` to match ``node_order``."""
    idx = [track.node_names.index(name) for name in node_order]
    return track.points[:, idx, :]


def _resample_track(track: PoseTrack2D, fps: float, target_times: np.ndarray) -> PoseTrack2D:
    """Resample a 2D track onto ``target_times`` by linear interpolation.

    Each target time is located between the two source frames it falls
    between (by that track's own frame rate) and linearly interpolated.
    Target times outside the track's recorded range, or falling between a
    pair where either source frame is a missing detection (``NaN``),
    resolve to ``NaN`` rather than extrapolating or bridging over gaps.
    """
    n_frames = track.points.shape[0]
    frac_idx = target_times * fps
    idx0 = np.floor(frac_idx).astype(int)
    idx1 = idx0 + 1
    frac = frac_idx - idx0

    in_range = (idx0 >= 0) & (idx1 < n_frames)
    idx0_clipped = np.clip(idx0, 0, n_frames - 1)
    idx1_clipped = np.clip(idx1, 0, n_frames - 1)

    p0, p1 = track.points[idx0_clipped], track.points[idx1_clipped]
    points = p0 + (p1 - p0) * frac[:, None, None]
    points[~in_range] = np.nan

    s0, s1 = track.scores[idx0_clipped], track.scores[idx1_clipped]
    scores = s0 + (s1 - s0) * frac[:, None]
    scores[~in_range] = np.nan

    return PoseTrack2D(node_names=track.node_names, points=points, scores=scores)


def align_tracks_by_time(
    tracks: Dict[str, PoseTrack2D], fps_by_role: Dict[str, float]
) -> Dict[str, PoseTrack2D]:
    """Resample every camera view's 2D track onto a shared timeline.

    The slowest camera's own frame times become the shared timeline; every other view is
    linearly interpolated onto it, never extrapolated beyond its own recorded range.

    Args:
        tracks: Mapping of camera role to that view's 2D predictions.
        fps_by_role: Mapping of camera role to that view's source video frame rate.

    Returns:
        A new mapping of camera role to a 2D track resampled onto the shared timeline.
    """
    reference_role = min(fps_by_role, key=fps_by_role.get)
    reference_fps = fps_by_role[reference_role]
    n_frames = tracks[reference_role].points.shape[0]
    target_times = np.arange(n_frames) / reference_fps

    aligned = {}
    for role, track in tracks.items():
        if role == reference_role:
            aligned[role] = track
        else:
            aligned[role] = _resample_track(track, fps_by_role[role], target_times)
    return aligned


def _stacked_2d_points(
    tracks: Dict[str, PoseTrack2D], cgroup: CameraGroup, fps_by_role: Dict[str, float]
) -> "Tuple[List[str], List[str], np.ndarray]":
    """Shared groundwork for triangulation: validate inputs, align views
    onto a shared timeline, and stack them into aniposelib's expected
    ``(n_cams, n_frames, n_nodes, 2)`` layout, cameras ordered to match
    ``cgroup.get_names()``. Returns ``(node_order, cam_names, stacked)``.
    """
    missing = [role for role in CAMERA_ROLES if role not in tracks]
    if missing:
        raise ValueError(f"Missing camera view(s) for triangulation: {missing}")
    missing_fps = [role for role in CAMERA_ROLES if role not in fps_by_role]
    if missing_fps:
        raise ValueError(f"Missing frame rate for camera view(s): {missing_fps}")

    aligned = align_tracks_by_time(tracks, fps_by_role)
    node_order = _canonical_node_order(aligned)
    cam_names = cgroup.get_names()
    stacked = np.stack([_reindex(aligned[role], node_order) for role in cam_names], axis=0)
    return node_order, cam_names, stacked


def triangulate(
    tracks: Dict[str, PoseTrack2D], cgroup: CameraGroup, fps_by_role: Dict[str, float]
) -> Pose3D:
    """Triangulate 3D keypoints from per-camera 2D predictions.

    Args:
        tracks: Mapping of camera role (``left``/``right``/``bottom``) to that view's 2D
            predictions. All three roles must be present and track the same skeleton nodes.
        cgroup: Calibrated camera group with camera names matching ``left``/``right``/``bottom``.
        fps_by_role: Mapping of camera role to that view's source video frame rate.

    Returns:
        The triangulated 3D trajectory, at the slowest camera's frame rate.
    """
    node_order, cam_names, stacked = _stacked_2d_points(tracks, cgroup, fps_by_role)
    n_frames, n_nodes = stacked.shape[1], stacked.shape[2]

    # aniposelib expects (n_cams, n_points, 2); flatten frames and nodes together.
    points_2d = stacked.reshape(len(cam_names), n_frames * n_nodes, 2)

    points_3d_flat = cgroup.triangulate(points_2d, undistort=True)  # (n_frames*n_nodes, 3)
    error_flat = cgroup.reprojection_error(points_3d_flat, points_2d, mean=True)  # (n_frames*n_nodes,)

    points_3d = points_3d_flat.reshape(n_frames, n_nodes, 3)
    error = error_flat.reshape(n_frames, n_nodes)

    return Pose3D(node_names=node_order, points=points_3d, reprojection_error=error)


def _camera_ray_world(cam: Camera, point_2d: np.ndarray) -> "Tuple[np.ndarray, np.ndarray]":
    """The 3D ray (origin, unit direction) through one camera's optical
    center and a single 2D pixel, in the calibration's reference frame.

    ``cv2.undistortPoints`` (what :meth:`Camera.undistort_points` wraps),
    called without a projection matrix, returns *normalized* camera-plane
    coordinates rather than pixel coordinates -- i.e. already
    ``K^-1``-applied and distortion-corrected -- so ``[x, y, 1]`` in that
    space is directly a ray direction in the camera's own frame.
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
    """Where a ray crosses the horizontal plane at ``height`` (in the same
    ``xyz @ up_direction`` convention used throughout the gait pipeline),
    or ``None`` if the ray runs (near-)parallel to that plane.
    """
    denom = float(direction @ up_direction)
    if abs(denom) < 1e-6:
        return None
    s = (height - float(origin @ up_direction)) / denom
    return origin + s * direction


def triangulate_axis_prioritized(
    tracks: Dict[str, PoseTrack2D],
    cgroup: CameraGroup,
    fps_by_role: Dict[str, float],
    up_direction: np.ndarray,
    blend_weight: float = 0.5,
) -> Pose3D:
    """Like :func:`triangulate`, but where all three cameras have a valid detection, blends in
    an axis-prioritized reconstruction: height comes from the two side cameras alone, X/Y from
    intersecting the bottom camera's ray with the plane at that height.

    The side cameras are better conditioned for height and the bottom camera for X/Y, so this
    can outperform aniposelib's uniform least squares, which blends all three views symmetrically.
    Estimates are blended rather than hard-swapped since uniform least squares still damps
    per-frame detection noise better on disagreeing frames. Falls back to :func:`triangulate`
    wherever fewer than all three cameras have a valid detection.

    Args:
        tracks: Same as :func:`triangulate`.
        cgroup: Same as :func:`triangulate`.
        fps_by_role: Same as :func:`triangulate`.
        up_direction: Unit vector, in the calibration's reference frame, pointing away from
            the platform surface.
        blend_weight: Blend fraction from ``0.0`` (pure uniform least squares) to ``1.0``
            (pure axis-prioritized).

    Returns:
        The triangulated 3D trajectory, at the slowest camera's frame rate.
    """
    baseline = triangulate(tracks, cgroup, fps_by_role)
    node_order, cam_names, stacked = _stacked_2d_points(tracks, cgroup, fps_by_role)
    n_frames, n_nodes = stacked.shape[1], stacked.shape[2]
    cam_idx = {role: cam_names.index(role) for role in CAMERA_ROLES}

    pts_left = stacked[cam_idx["left"]].reshape(-1, 2)
    pts_right = stacked[cam_idx["right"]].reshape(-1, 2)
    pts_bottom = stacked[cam_idx["bottom"]].reshape(-1, 2)
    all_three_valid = (
        ~np.isnan(pts_left).any(axis=1) & ~np.isnan(pts_right).any(axis=1) & ~np.isnan(pts_bottom).any(axis=1)
    )

    points = baseline.points.reshape(-1, 3).copy()
    if all_three_valid.any():
        side_group = CameraGroup([cgroup.cameras[cam_idx["left"]], cgroup.cameras[cam_idx["right"]]])
        bottom_cam = cgroup.cameras[cam_idx["bottom"]]

        idx = np.flatnonzero(all_three_valid)
        side_points = np.stack([pts_left[idx], pts_right[idx]], axis=0)  # (2, n_selected, 2)
        side_points_3d = side_group.triangulate(side_points, undistort=True)  # (n_selected, 3)
        heights = side_points_3d @ up_direction

        for k, i in enumerate(idx):
            origin, direction = _camera_ray_world(bottom_cam, pts_bottom[i])
            intersection = _ray_plane_intersection(origin, direction, up_direction, float(heights[k]))
            if intersection is not None:
                points[i] = (1 - blend_weight) * points[i] + blend_weight * intersection
            # else: near-parallel bottom ray -- keep the baseline point.

    points_2d = stacked.reshape(len(cam_names), n_frames * n_nodes, 2)
    error_flat = cgroup.reprojection_error(points, points_2d, mean=True)

    return Pose3D(
        node_names=node_order,
        points=points.reshape(n_frames, n_nodes, 3),
        reprojection_error=error_flat.reshape(n_frames, n_nodes),
    )
