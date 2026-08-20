"""Annotated validation videos for auditing the 3D gait pipeline by eye.

Stacks a session's three camera views vertically (left, bottom, right --
each the same cropped, model-input clip :mod:`alligaitor.inference`
actually ran on) with the triangulated skeleton reprojected back into
every view: skeleton edges, paw nodes colored by ground-contact state (and
red wherever the contributing cameras substantially disagree), an
accumulating footprint marker at each detected touchdown, and a
per-camera warning banner wherever that specific camera's dropped
detection looks like it broke up a real stance phase into fragments too
short to survive :attr:`alligaitor.config.GaitConfig.min_contact_frames`
(see :func:`alligaitor.gait.find_camera_caused_discards`).

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

from alligaitor import cropping, gait, pipeline, triangulation
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
_DROP_WARNING_COLOR = (0, 0, 220)

_NODE_RADIUS = 5
_FOOTPRINT_RADIUS = 4


def _camera_drop_warnings(
    session: SessionConfig,
    times: np.ndarray,
    positions: Dict[str, np.ndarray],
    config: GaitConfig,
) -> Dict[str, Dict[int, List[str]]]:
    """Per role, per (shared-timeline) frame, which paw(s) that camera's
    dropped detection plausibly cost a stance phase -- see
    :func:`alligaitor.gait.find_camera_caused_discards`.

    Reloads and re-aligns each role's raw 2D predictions (the same cached
    ``<role>.predictions.slp`` triangulation used) to see which camera(s)
    actually had a valid detection on each frame, independent of whether
    the fused 3D point survived. ``positions`` is bridged the same way
    :func:`alligaitor.gait.compute_trial_metrics` bridges it (see
    :func:`alligaitor.gait.bridge_short_gaps`) before looking for
    discards, so this diagnostic only flags gaps actually long enough to
    have mattered to the real classification -- a short gap the trial's
    own stance detection already bridged over isn't a discard to warn
    about.
    """
    tracks = {}
    fps_by_role = {}
    for role in CAMERA_ROLES:
        slp_path = session.output_dir / f"{role}.predictions.slp"
        tracks[role] = pipeline.load_track(session.videos[role], slp_path)
        fps_by_role[role] = video_fps(session.videos[role])
    aligned = triangulation.align_tracks_by_time(tracks, fps_by_role)
    cam_valid_by_paw = gait.cam_valid_by_paw_from_aligned(aligned)
    discards_by_paw = gait.compute_discards_by_paw(positions, cam_valid_by_paw, config)

    warnings_by_role: Dict[str, Dict[int, List[str]]] = {role: {} for role in CAMERA_ROLES}
    for paw in PAW_NODES:
        cam_valid = cam_valid_by_paw[paw]
        exclude_camera = gait.FAR_SIDE_CAMERA[paw]
        for discard in discards_by_paw[paw]:
            # discard.dropped_by is the union of whichever camera(s) were
            # missing at the window's start vs. end boundary -- not every
            # camera in that set was necessarily missing on every frame
            # in between. Re-check per frame so a camera that had a
            # perfectly good detection partway through the window (see
            # frames 80-81 in the 79-82 example this was built against)
            # isn't shown as having dropped it there too.
            for f in range(discard.start_frame, discard.end_frame + 1):
                for role in CAMERA_ROLES:
                    if role != exclude_camera and not cam_valid[role][f]:
                        warnings_by_role[role].setdefault(f, []).append(paw)
    return warnings_by_role


def _draw_drop_warning(panel: np.ndarray, paws: List[str]) -> None:
    text = "DROPPED: " + ", ".join(sorted(set(paws)))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    x0 = max(panel.shape[1] - tw - 12, 0)
    cv2.rectangle(panel, (x0 - 4, 2), (panel.shape[1] - 2, th + 12), _DROP_WARNING_COLOR, -1)
    cv2.putText(panel, text, (x0, th + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


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
    drop_warnings_by_role = _camera_drop_warnings(session, times, positions, config)

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

                active_drops = drop_warnings_by_role.get(role, {}).get(i)
                if active_drops:
                    _draw_drop_warning(panel, active_drops)

                panels.append(panel)

            max_w = max(p.shape[1] for p in panels)
            panels = [p if p.shape[1] == max_w else cv2.resize(p, (max_w, int(p.shape[0] * max_w / p.shape[1]))) for p in panels]
            composite = np.vstack(panels)
            cv2.putText(composite, f"t={t:.3f}s", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if writer is None:
                # "mp4v" (MPEG-4 Part 2) writes a technically-valid stream
                # but many players (QuickTime, Safari, in-app previews)
                # render it as garbled macroblocks rather than falling
                # back gracefully -- "avc1" (H.264) is broadly playable.
                fourcc = cv2.VideoWriter_fourcc(*"avc1")
                writer = cv2.VideoWriter(str(output_path), fourcc, fps_by_role[reference_role], (composite.shape[1], composite.shape[0]))
            writer.write(composite)
    finally:
        for cap in caps.values():
            cap.release()
        if writer is not None:
            writer.release()

    return output_path
