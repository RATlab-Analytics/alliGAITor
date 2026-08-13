"""3D triangulation across the three-camera rig, built on aniposelib.

Combines per-camera 2D pose predictions (see :mod:`alligaitor.inference`)
with a calibrated :class:`~aniposelib.cameras.CameraGroup` to reconstruct
3D keypoint trajectories. The side and bottom SLEAP-NN models declare
their skeleton nodes in different orders, so predictions are always
reindexed to a canonical node order by name before triangulating.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from aniposelib.cameras import CameraGroup

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


def triangulate(tracks: Dict[str, PoseTrack2D], cgroup: CameraGroup) -> Pose3D:
    """Triangulate 3D keypoints from per-camera 2D predictions.

    Args:
        tracks: Mapping of camera role (``left``/``right``/``bottom``) to
            that view's 2D predictions. All three roles must be present,
            share the same number of frames, and track the same skeleton
            nodes (matched by name, not necessarily the same declared
            order).
        cgroup: Calibrated camera group with camera names matching
            ``left``/``right``/``bottom`` (see :mod:`alligaitor.calibration`).

    Returns:
        The triangulated 3D trajectory.
    """
    missing = [role for role in CAMERA_ROLES if role not in tracks]
    if missing:
        raise ValueError(f"Missing camera view(s) for triangulation: {missing}")

    frame_counts = {role: t.points.shape[0] for role, t in tracks.items()}
    if len(set(frame_counts.values())) != 1:
        raise ValueError(
            "Camera views have mismatched frame counts; 2D predictions must be "
            f"frame-aligned across cameras: {frame_counts}"
        )

    node_order = _canonical_node_order(tracks)
    n_frames = next(iter(frame_counts.values()))
    n_nodes = len(node_order)

    cam_names = cgroup.get_names()
    # (n_cams, n_frames, n_nodes, 2)
    stacked = np.stack([_reindex(tracks[role], node_order) for role in cam_names], axis=0)
    # aniposelib expects (n_cams, n_points, 2); flatten frames and nodes together.
    points_2d = stacked.reshape(len(cam_names), n_frames * n_nodes, 2)

    points_3d_flat = cgroup.triangulate(points_2d, undistort=True)  # (n_frames*n_nodes, 3)
    error_flat = cgroup.reprojection_error(points_3d_flat, points_2d, mean=True)  # (n_frames*n_nodes,)

    points_3d = points_3d_flat.reshape(n_frames, n_nodes, 3)
    error = error_flat.reshape(n_frames, n_nodes)

    return Pose3D(node_names=node_order, points=points_3d, reprojection_error=error)
