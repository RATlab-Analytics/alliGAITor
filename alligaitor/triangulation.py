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

    The slowest camera's own frame times become the shared timeline (the
    other views are only ever interpolated, never extrapolated beyond
    their own recorded range), and every other view is linearly
    interpolated onto it. See :mod:`alligaitor.timing` for why this is
    necessary: the rig's cameras run at different, session-varying frame
    rates, so raw frame index does not correspond to the same recording
    moment across views.

    Args:
        tracks: Mapping of camera role to that view's 2D predictions, one
            entry per frame of that view's own source video.
        fps_by_role: Mapping of camera role to that view's source video
            frame rate (see :func:`alligaitor.timing.video_fps`).

    Returns:
        A new mapping of camera role to a 2D track resampled onto the
        shared timeline; all tracks share the same frame count.
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


def triangulate(
    tracks: Dict[str, PoseTrack2D], cgroup: CameraGroup, fps_by_role: Dict[str, float]
) -> Pose3D:
    """Triangulate 3D keypoints from per-camera 2D predictions.

    Args:
        tracks: Mapping of camera role (``left``/``right``/``bottom``) to
            that view's 2D predictions, one entry per frame of that view's
            own source video. All three roles must be present and track
            the same skeleton nodes (matched by name, not necessarily the
            same declared order).
        cgroup: Calibrated camera group with camera names matching
            ``left``/``right``/``bottom`` (see :mod:`alligaitor.calibration`).
        fps_by_role: Mapping of camera role to that view's source video
            frame rate, used to align views onto a shared timeline before
            triangulating (see :func:`align_tracks_by_time`).

    Returns:
        The triangulated 3D trajectory, at the slowest camera's frame rate.
    """
    missing = [role for role in CAMERA_ROLES if role not in tracks]
    if missing:
        raise ValueError(f"Missing camera view(s) for triangulation: {missing}")
    missing_fps = [role for role in CAMERA_ROLES if role not in fps_by_role]
    if missing_fps:
        raise ValueError(f"Missing frame rate for camera view(s): {missing_fps}")

    aligned = align_tracks_by_time(tracks, fps_by_role)

    node_order = _canonical_node_order(aligned)
    n_frames = next(iter(aligned.values())).points.shape[0]
    n_nodes = len(node_order)

    cam_names = cgroup.get_names()
    # (n_cams, n_frames, n_nodes, 2)
    stacked = np.stack([_reindex(aligned[role], node_order) for role in cam_names], axis=0)
    # aniposelib expects (n_cams, n_points, 2); flatten frames and nodes together.
    points_2d = stacked.reshape(len(cam_names), n_frames * n_nodes, 2)

    points_3d_flat = cgroup.triangulate(points_2d, undistort=True)  # (n_frames*n_nodes, 3)
    error_flat = cgroup.reprojection_error(points_3d_flat, points_2d, mean=True)  # (n_frames*n_nodes,)

    points_3d = points_3d_flat.reshape(n_frames, n_nodes, 3)
    error = error_flat.reshape(n_frames, n_nodes)

    return Pose3D(node_names=node_order, points=points_3d, reprojection_error=error)
