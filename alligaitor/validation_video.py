"""Annotated validation videos for auditing the 3D gait pipeline by eye.

Stacks a session's three camera views vertically with the triangulated skeleton reprojected
back into each: skeleton edges, paw nodes colored by contact state, footprint markers at each
touchdown, and a per-camera warning banner where a dropped detection likely broke up a stance
phase. Colors and markers are drawn from the same :class:`alligaitor.gait.TrialMetrics` the
group workbook is built from.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from dataclasses import replace
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from aniposelib.cameras import CameraGroup
from tqdm import tqdm

from alligaitor import cropping, gait, pipeline, triangulation
from alligaitor.ansi_html import ansi_line_to_html
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
_FOOTPRINT_COLOR_FOREPAW = (255, 0, 255)  # magenta
_FOOTPRINT_COLOR_HINDPAW = (255, 255, 0)  # cyan
_DROP_WARNING_COLOR = (0, 0, 220)

# Footprints from an earlier crossing fade to flat gray once a later crossing has started,
# so an out-and-back recording's trail doesn't read as one continuous crossing.
_FOOTPRINT_FADED_COLOR = (120, 120, 120)
_FOOTPRINT_FADED_ALPHA = 0.35

_NODE_RADIUS = 5
_FOOTPRINT_RADIUS = 4


class _FrameProgress:
    """Throttled tqdm-style progress reporting for this module's per-frame render loop.

    The pure-Python equivalent of :class:`alligaitor.subprocess_streaming.ProgressStreamer`,
    matching the same redraw-in-place/``on_redraw_closed`` contract.
    """

    def __init__(
        self,
        desc: str,
        n_frames: int,
        progress: Optional[Callable[[str], None]],
        html_progress: bool,
        on_redraw_closed: Optional[Callable[[], None]],
        min_interval_s: float = 0.5,
    ):
        self.desc = desc
        self.n_frames = n_frames
        self.progress = progress
        self.html_progress = html_progress
        self.on_redraw_closed = on_redraw_closed
        self.min_interval_s = min_interval_s
        self.start_time = time.monotonic()
        self.last_emit = 0.0

    def update(self, i: int) -> None:
        if self.progress is None:
            return
        now = time.monotonic()
        is_final = i >= self.n_frames - 1
        # Always flush the final frame regardless of throttle, so the bar doesn't stick below 100%.
        if not is_final and now - self.last_emit < self.min_interval_s:
            return
        elapsed = now - self.start_time
        bar = tqdm.format_meter(i + 1, self.n_frames, elapsed, prefix=self.desc)
        text = f"    {bar}"
        self.progress(ansi_line_to_html(text) if self.html_progress else text)
        self.last_emit = now
        if is_final and self.on_redraw_closed is not None:
            self.on_redraw_closed()


def _footprint_color(paw: str):
    """Forepaw and hind-paw footprints get distinct colors."""
    return _FOOTPRINT_COLOR_HINDPAW if "hind" in paw else _FOOTPRINT_COLOR_FOREPAW


def _camera_drop_warnings(
    session: SessionConfig,
    times: np.ndarray,
    positions: Dict[str, np.ndarray],
    config: GaitConfig,
) -> Dict[str, Dict[int, List[str]]]:
    """Per role, per shared-timeline frame, which paw(s) that camera's dropped detection
    plausibly cost a stance phase.

    Reloads and re-aligns each role's raw 2D predictions to see which camera(s) actually had
    a valid detection on each frame, independent of whether the fused 3D point survived.
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
            # Re-check per frame: not every camera missing at the window boundary was
            # necessarily missing on every frame in between.
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


def _crossing_index_of_frame(frame: int, crossing_starts: np.ndarray) -> int:
    """Which crossing (index into ``crossing_starts``) ``frame`` belongs to."""
    return max(0, int(np.searchsorted(crossing_starts, frame, side="right")) - 1)


def _reproject_footprints(
    positions: Dict[str, np.ndarray],
    trial: TrialMetrics,
    cgroup: CameraGroup,
    cam_index: Dict[str, int],
    crop_offset: Dict[str, Tuple[float, float]],
    crossing_starts: np.ndarray,
) -> Dict[str, List[Tuple[int, str, int, Tuple[float, float]]]]:
    """Per role, a list of ``(touchdown_frame, paw, crossing_index, (px, py))`` footprint
    markers, reprojected once for the whole video since the rig is static.
    """
    entries = [
        (paw, touchdown_frame, _crossing_index_of_frame(touchdown_frame, crossing_starts))
        for paw in PAW_NODES
        for touchdown_frame in trial.paw_events[paw].touchdown_frames
    ]
    footprints_by_role: Dict[str, List[Tuple[int, str, int, Tuple[float, float]]]] = {
        role: [] for role in CAMERA_ROLES
    }
    if not entries:
        return footprints_by_role

    pts3d = np.stack([positions[paw][frame] for paw, frame, _ in entries])
    proj = cgroup.project(pts3d)  # (n_cams, n_points, 2)
    for role in CAMERA_ROLES:
        cam_proj = proj[cam_index[role]]
        offset = crop_offset[role]
        for (paw, frame, crossing_idx), (px, py) in zip(entries, cam_proj):
            footprints_by_role[role].append(
                (int(frame), paw, crossing_idx, (float(px - offset[0]), float(py - offset[1])))
            )
    return footprints_by_role


# A reprojected point can land far outside any real pixel coordinate (e.g. a misassigned
# camera role). A huge-but-finite value would otherwise reach cv2's C++ point parser and
# crash; _MAX_COORD bounds anything drawable, safely inside int32 range.
_MAX_COORD = 1_000_000


def _finite_point(xy: Tuple[float, float]) -> Optional[Tuple[int, int]]:
    x, y = xy
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    if abs(x) > _MAX_COORD or abs(y) > _MAX_COORD:
        return None
    return int(round(x)), int(round(y))


def _draw_marker(
    frame: np.ndarray, xy: Tuple[float, float], radius: int, color, thickness: int = -1, alpha: float = 1.0
) -> None:
    """Draw one marker, alpha-blended into the frame when ``alpha < 1`` (used to fade an
    earlier crossing's footprints). Silently skips a non-finite or out-of-range point.
    """
    point = _finite_point(xy)
    if point is None:
        return
    x, y = point
    h, w = frame.shape[:2]
    if not (-radius <= x <= w + radius and -radius <= y <= h + radius):
        return
    if alpha >= 1.0:
        cv2.circle(frame, (x, y), radius, color, thickness)
        return
    x0, x1 = max(x - radius - 1, 0), min(x + radius + 2, w)
    y0, y1 = max(y - radius - 1, 0), min(y + radius + 2, h)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    overlay = roi.copy()
    cv2.circle(overlay, (x - x0, y - y0), radius, color, thickness)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)


_STRIP_HEIGHT = 32
_STRIP_BG_COLOR = (40, 40, 40)
# (label, color), matching every marker color drawn elsewhere in this module.
_LEGEND_ENTRIES = (
    ("body node", _NODE_COLOR),
    ("paw (swing)", _PAW_SWING_COLOR),
    ("paw (contact)", _PAW_CONTACT_COLOR),
    ("reprojection disagreement", _DISAGREEMENT_COLOR),
    ("forepaw footprint", _FOOTPRINT_COLOR_FOREPAW),
    ("hindpaw footprint", _FOOTPRINT_COLOR_HINDPAW),
    ("footprint (earlier crossing)", _FOOTPRINT_FADED_COLOR),
)


def _draw_header_strip(width: int, t: float, frame_idx: int, n_frames: int) -> np.ndarray:
    """A top strip showing shared-timeline time/frame index and a color legend, kept above
    the camera panels rather than overlaid on the video."""
    strip = np.full((_STRIP_HEIGHT, width, 3), _STRIP_BG_COLOR, dtype=np.uint8)
    baseline_y = _STRIP_HEIGHT // 2 + 5

    text = f"t={t:.3f}s  frame {frame_idx}/{n_frames - 1}"
    cv2.putText(strip, text, (8, baseline_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    x = 8 + tw + 28
    for label, color in _LEGEND_ENTRIES:
        if x > width - 20:
            break  # out of room -- drop remaining entries rather than overflow
        cv2.circle(strip, (x, _STRIP_HEIGHT // 2), 5, color, -1)
        cv2.putText(strip, label, (x + 10, baseline_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        x += 10 + lw + 18

    return strip


def _merge_crossings(trial: Union[TrialMetrics, List[TrialMetrics]]) -> TrialMetrics:
    """Collapse a recording's per-crossing trials into one whose ``paw_events`` covers every
    crossing, for drawing purposes. Only ``paw_events`` is meaningful on the result.
    """
    if isinstance(trial, TrialMetrics):
        return trial
    trials = list(trial)
    if not trials:
        raise ValueError("export_validation_video needs at least one trial")
    if len(trials) == 1:
        return trials[0]

    merged_events = {}
    for paw in PAW_NODES:
        parts = [t.paw_events[paw] for t in trials]
        order = np.argsort(np.concatenate([p.touchdown_frames for p in parts]))
        merged_events[paw] = gait.PawEvents(
            touchdown_frames=np.concatenate([p.touchdown_frames for p in parts])[order],
            liftoff_frames=np.concatenate([p.liftoff_frames for p in parts])[order],
            touchdown_times=np.concatenate([p.touchdown_times for p in parts])[order],
            liftoff_times=np.concatenate([p.liftoff_times for p in parts])[order],
        )
    return replace(trials[0], paw_events=merged_events)


def export_validation_video(
    session: SessionConfig,
    csv_path: Path,
    cgroup: CameraGroup,
    trial: Union[TrialMetrics, List[TrialMetrics]],
    config: GaitConfig,
    output_path: Path,
    disagreement_threshold_px: float = 20.0,
    log: Callable[[str], None] = print,
    progress: Optional[Callable[[str], None]] = None,
    html_progress: bool = False,
    on_redraw_closed: Optional[Callable[[], None]] = None,
) -> Path:
    """Write one session's annotated validation video.

    Args:
        session: Session configuration; ``session.videos`` (cropped, model-input clips) is
            what's actually rendered.
        csv_path: This trial's ``pose_3d.csv``.
        cgroup: Calibrated camera group.
        trial: This recording's already-computed :class:`gait.TrialMetrics`, either one, or
            the per-crossing list from :func:`alligaitor.gait.compute_crossing_metrics`.
        config: The :class:`GaitConfig` ``trial`` was computed with.
        output_path: Destination ``.mp4`` path.
        disagreement_threshold_px: A node is drawn red when its reprojection error exceeds this.
        log: Receives discrete one-off messages.
        progress: Receives a live, redrawing tqdm-style progress line as frames render.
        html_progress: Whether ``progress`` wants an HTML-rendered line or plain text.
        on_redraw_closed: Called once the final frame's progress update has been sent.

    Returns:
        ``output_path``.
    """
    if progress is None:
        progress = log

    times, positions, errors, _fallback = gait.load_pose_3d(csv_path)
    n_frames = len(times)
    log(f"  Rendering validation video for '{session.name}' ({n_frames} frames)...")
    frame_progress = _FrameProgress(
        f"{session.name} validation video", n_frames, progress, html_progress, on_redraw_closed
    )
    crossings = [trial] if isinstance(trial, TrialMetrics) else list(trial)
    crossing_starts = np.array(
        sorted(t.crossing_window[0] for t in crossings if t.crossing_window is not None)
    )
    if crossing_starts.size == 0:
        crossing_starts = np.array([0])  # no window info -- treat as one crossing
    trial = _merge_crossings(trial)
    planted = gait.planted_mask(trial, n_frames)

    cam_names = cgroup.get_names()
    cam_index = {role: cam_names.index(role) for role in CAMERA_ROLES}

    fps_by_role = {role: video_fps(session.videos[role]) for role in CAMERA_ROLES}
    reference_role = min(fps_by_role, key=fps_by_role.get)
    crop_offset = {role: cropping.crop_offset_for_video(session.videos[role]) for role in CAMERA_ROLES}

    footprints_by_role = _reproject_footprints(
        positions, trial, cgroup, cam_index, crop_offset, crossing_starts
    )
    drop_warnings_by_role = _camera_drop_warnings(session, times, positions, config)

    caps = {role: cv2.VideoCapture(str(session.videos[role])) for role in CAMERA_ROLES}
    native_frame_counts = {role: int(caps[role].get(cv2.CAP_PROP_FRAME_COUNT)) for role in CAMERA_ROLES}
    # What cap.read() will return next for each role absent a re-seek, so a purely
    # sequential reference role avoids an expensive cap.set() seek on every frame.
    next_native_idx = {role: 0 for role in CAMERA_ROLES}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None

    try:
        for i in range(n_frames):
            t = times[i]
            current_crossing = _crossing_index_of_frame(i, crossing_starts)
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
                if native_idx != next_native_idx[role]:
                    caps[role].set(cv2.CAP_PROP_POS_FRAMES, native_idx)
                ok, panel = caps[role].read()
                next_native_idx[role] = native_idx + 1 if ok else next_native_idx[role]
                if not ok or panel is None:
                    panel = np.zeros((100, 200, 3), dtype=np.uint8)

                node_px = proj_by_role.get(role, {})

                for a, b in SKELETON_EDGES:
                    if a in node_px and b in node_px:
                        pa = _finite_point(node_px[a])
                        pb = _finite_point(node_px[b])
                        if pa is not None and pb is not None:
                            cv2.line(panel, pa, pb, _EDGE_COLOR, 1, cv2.LINE_AA)

                for touchdown_frame, paw, footprint_crossing, xy in footprints_by_role.get(role, []):
                    if touchdown_frame > i:
                        continue
                    if footprint_crossing < current_crossing:
                        _draw_marker(
                            panel, xy, _FOOTPRINT_RADIUS, _FOOTPRINT_FADED_COLOR,
                            alpha=_FOOTPRINT_FADED_ALPHA,
                        )
                    else:
                        _draw_marker(panel, xy, _FOOTPRINT_RADIUS, _footprint_color(paw))

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
            header = _draw_header_strip(max_w, t, i, n_frames)
            composite = np.vstack([header, *panels])

            if writer is None:
                # "avc1" (H.264) is broadly playable; "mp4v" renders as garbled macroblocks
                # in many players (QuickTime, Safari).
                fourcc = cv2.VideoWriter_fourcc(*"avc1")
                writer = cv2.VideoWriter(str(output_path), fourcc, fps_by_role[reference_role], (composite.shape[1], composite.shape[0]))
            writer.write(composite)
            frame_progress.update(i)
    finally:
        for cap in caps.values():
            cap.release()
        if writer is not None:
            writer.release()

    log(f"  Wrote validation video: {output_path}")
    return output_path
