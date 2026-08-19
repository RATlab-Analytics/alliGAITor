"""Annotated validation videos for auditing the 3D gait pipeline by eye.

Stacks a session's three camera views vertically (left, bottom, right --
each the same cropped, model-input clip :mod:`alligaitor.inference`
actually ran on) with the triangulated skeleton reprojected back into
every view: skeleton edges, paw nodes colored by ground-contact state (and
red wherever the contributing cameras substantially disagree), and an
accumulating footprint marker at each detected touchdown.

Meant as the audit trail for tuning :class:`alligaitor.config.GaitConfig`
by eye rather than by staring at numbers alone: every color and marker
here is drawn from exactly the same :class:`alligaitor.gait.TrialMetrics`
the group workbook is built from, not a separate/simplified pass.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from aniposelib.cameras import CameraGroup

from alligaitor import cropping, gait
from alligaitor.config import CAMERA_ROLES, GaitConfig, SessionConfig
from alligaitor.gait import PAW_NODES, TrialMetrics
from alligaitor.timing import video_fps

# Skeleton edges to draw, matching minimal_skeleton.json.
SKELETON_EDGES = (
    ("nose", "neck"),
    ("neck", "mid-back"),
    ("mid-back", "tail-base"),
    ("left-hind-paw", "right-hind-paw"),
    ("right-forepaw", "left-forepaw"),
)

PANEL_ORDER = ("left", "bottom", "right")

# BGR colors (OpenCV convention).
_EDGE_COLOR = (200, 200, 200)
_NODE_COLOR = (0, 210, 255)  # non-paw nodes
_PAW_SWING_COLOR = (255, 140, 0)
_PAW_CONTACT_COLOR = (0, 200, 0)
_DISAGREEMENT_COLOR = (0, 0, 255)
_FOOTPRINT_COLOR = (255, 0, 255)

_NODE_RADIUS = 5
_FOOTPRINT_RADIUS = 4


def _reproject_footprints(
    positions: Dict[str, np.ndarray],
    trial: TrialMetrics,
    cgroup: CameraGroup,
    cam_index: Dict[str, int],
    crop_offset: Dict[str, Tuple[float, float]],
) -> Dict[str, List[Tuple[int, Tuple[float, float]]]]:
    """Per role, a list of ``(touchdown_frame, (px, py))`` footprint markers.

    Reprojected once for the whole video (the rig is static), not per frame.
    """
    entries = [
        (paw, touchdown_frame)
        for paw in PAW_NODES
        for touchdown_frame in trial.paw_events[paw].touchdown_frames
    ]
    footprints_by_role: Dict[str, List[Tuple[int, Tuple[float, float]]]] = {role: [] for role in CAMERA_ROLES}
    if not entries:
        return footprints_by_role

    pts3d = np.stack([positions[paw][frame] for paw, frame in entries])
    proj = cgroup.project(pts3d)  # (n_cams, n_points, 2)
    for role in CAMERA_ROLES:
        cam_proj = proj[cam_index[role]]
        offset = crop_offset[role]
        for (paw, frame), (px, py) in zip(entries, cam_proj):
            footprints_by_role[role].append((int(frame), (float(px - offset[0]), float(py - offset[1]))))
    return footprints_by_role


def _draw_marker(frame: np.ndarray, xy: Tuple[float, float], radius: int, color, thickness: int = -1) -> None:
    x, y = int(round(xy[0])), int(round(xy[1]))
    if -radius <= x <= frame.shape[1] + radius and -radius <= y <= frame.shape[0] + radius:
        cv2.circle(frame, (x, y), radius, color, thickness)


def export_validation_video(
    session: SessionConfig,
    csv_path: Path,
    cgroup: CameraGroup,
    trial: TrialMetrics,
    config: GaitConfig,
    output_path: Path,
    disagreement_threshold_px: float = 20.0,
) -> Path:
    """Write one session's annotated validation video.

    Args:
        session: Session configuration -- ``session.videos`` (the cropped,
            model-input clips) are what's actually rendered.
        csv_path: This trial's ``pose_3d.csv``.
        cgroup: Calibrated camera group.
        trial: This trial's already-computed :class:`gait.TrialMetrics`
            (same instance the group workbook is built from).
        config: The :class:`GaitConfig` ``trial`` was computed with.
        output_path: Destination ``.mp4`` path.
        disagreement_threshold_px: A node is drawn red, regardless of its
            usual color, when its reprojection error exceeds this.

    Returns:
        ``output_path``.
    """
    times, positions, errors = gait.load_pose_3d(csv_path)
    n_frames = len(times)
    planted = gait.planted_mask(trial, n_frames)

    cam_names = cgroup.get_names()
    cam_index = {role: cam_names.index(role) for role in CAMERA_ROLES}

    fps_by_role = {role: video_fps(session.videos[role]) for role in CAMERA_ROLES}
    reference_role = min(fps_by_role, key=fps_by_role.get)
    crop_offset = {role: cropping.crop_offset_for_video(session.videos[role]) for role in CAMERA_ROLES}

    footprints_by_role = _reproject_footprints(positions, trial, cgroup, cam_index, crop_offset)

    caps = {role: cv2.VideoCapture(str(session.videos[role])) for role in CAMERA_ROLES}
    native_frame_counts = {role: int(caps[role].get(cv2.CAP_PROP_FRAME_COUNT)) for role in CAMERA_ROLES}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None

    try:
        for i in range(n_frames):
            t = times[i]
            valid_nodes = [
                node for node, arr in positions.items() if not np.isnan(arr[i]).any()
            ]
            proj_by_role = {}
            if valid_nodes:
                pts3d = np.stack([positions[node][i] for node in valid_nodes])
                proj = cgroup.project(pts3d)  # (n_cams, n_valid_nodes, 2)
                for role in CAMERA_ROLES:
                    offset = crop_offset[role]
                    proj_by_role[role] = {
                        node: (float(px - offset[0]), float(py - offset[1]))
                        for node, (px, py) in zip(valid_nodes, proj[cam_index[role]])
                    }

            panels = []
            for role in PANEL_ORDER:
                native_idx = i if role == reference_role else int(round(t * fps_by_role[role]))
                native_idx = max(0, min(native_idx, native_frame_counts[role] - 1))
                caps[role].set(cv2.CAP_PROP_POS_FRAMES, native_idx)
                ok, panel = caps[role].read()
                if not ok or panel is None:
                    panel = np.zeros((100, 200, 3), dtype=np.uint8)

                node_px = proj_by_role.get(role, {})

                for a, b in SKELETON_EDGES:
                    if a in node_px and b in node_px:
                        pa = tuple(int(round(c)) for c in node_px[a])
                        pb = tuple(int(round(c)) for c in node_px[b])
                        cv2.line(panel, pa, pb, _EDGE_COLOR, 1, cv2.LINE_AA)

                for touchdown_frame, xy in footprints_by_role.get(role, []):
                    if touchdown_frame <= i:
                        _draw_marker(panel, xy, _FOOTPRINT_RADIUS, _FOOTPRINT_COLOR)

                for node, xy in node_px.items():
                    err = errors[node][i]
                    if not np.isnan(err) and err > disagreement_threshold_px:
                        color = _DISAGREEMENT_COLOR
                    elif node in PAW_NODES:
                        color = _PAW_CONTACT_COLOR if planted[node][i] else _PAW_SWING_COLOR
                    else:
                        color = _NODE_COLOR
                    _draw_marker(panel, xy, _NODE_RADIUS, color)

                cv2.putText(
                    panel, f"{role}  frame {native_idx}/{native_frame_counts[role] - 1}",
                    (4, panel.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
                )
                panels.append(panel)

            max_w = max(p.shape[1] for p in panels)
            panels = [p if p.shape[1] == max_w else cv2.resize(p, (max_w, int(p.shape[0] * max_w / p.shape[1]))) for p in panels]
            composite = np.vstack(panels)
            cv2.putText(composite, f"t={t:.3f}s", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_path), fourcc, fps_by_role[reference_role], (composite.shape[1], composite.shape[0]))
            writer.write(composite)
    finally:
        for cap in caps.values():
            cap.release()
        if writer is not None:
            writer.release()

    return output_path
