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

# A footprint left over from an earlier crossing (see find_crossings) is drawn
# in this flat gray at reduced opacity instead of its usual fore/hindpaw
# color, once a later crossing has started -- so the accumulating trail from
# an out-and-back recording doesn't read as one continuous (and geometrically
# meaningless, since direction of travel differs) crossing.
_FOOTPRINT_FADED_COLOR = (120, 120, 120)
_FOOTPRINT_FADED_ALPHA = 0.35

_NODE_RADIUS = 5
_FOOTPRINT_RADIUS = 4


class _FrameProgress:
    """Throttled tqdm-style progress reporting for this module's per-frame
    render loop -- the pure-Python equivalent of
    :class:`alligaitor.subprocess_streaming.ProgressStreamer` for a
    subprocess's own tqdm bar, so rendering a validation video shows a
    live progress bar in the GUI log panel that looks and behaves exactly
    like inference's (same ``tqdm.format_meter`` text, same redraw-in-
    place/``on_redraw_closed`` contract -- see
    :func:`alligaitor.inference.run_inference`) instead of the log going
    quiet for however long a render takes.
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
        # Always flush the final frame regardless of the throttle -- same
        # reasoning as ProgressStreamer._flush_progress_final: dropping an
        # intermediate 0.5s tick is harmless (another follows shortly),
        # but silently eating the one update that says "done" leaves the
        # displayed bar stuck below 100% forever.
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
    """Forepaw and hind-paw footprints get distinct colors so a trial's
    two paw categories -- currently tracked with very different
    reliability, see :mod:`alligaitor.gait` -- are visually distinguishable
    at a glance rather than blending into one undifferentiated trail.
    """
    return _FOOTPRINT_COLOR_HINDPAW if "hind" in paw else _FOOTPRINT_COLOR_FOREPAW


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


def _crossing_index_of_frame(frame: int, crossing_starts: np.ndarray) -> int:
    """Which crossing (index into ``crossing_starts``) ``frame`` belongs
    to -- the last crossing whose start is at or before it.

    Used two ways: once per footprint, to fix which crossing *produced*
    it (its touchdown frame always falls inside that crossing's own
    window, so this is exact, not a guess); and once per rendered frame,
    to find the *current* crossing being watched, against which a
    footprint's own crossing is compared to decide whether it's faded
    (see :func:`export_validation_video`). A single-crossing recording
    has one start at or before every frame, so this is always ``0`` and
    nothing ever fades -- unchanged from before crossings existed.
    """
    return max(0, int(np.searchsorted(crossing_starts, frame, side="right")) - 1)


def _reproject_footprints(
    positions: Dict[str, np.ndarray],
    trial: TrialMetrics,
    cgroup: CameraGroup,
    cam_index: Dict[str, int],
    crop_offset: Dict[str, Tuple[float, float]],
    crossing_starts: np.ndarray,
) -> Dict[str, List[Tuple[int, str, int, Tuple[float, float]]]]:
    """Per role, a list of ``(touchdown_frame, paw, crossing_index, (px, py))``
    footprint markers -- ``paw`` is carried through so each marker can be
    colored by fore/hind paw category (see :func:`_footprint_color`), and
    ``crossing_index`` (see :func:`_crossing_index_of_frame`) so a marker
    from an earlier crossing than the one currently playing can be faded.

    Reprojected once for the whole video (the rig is static), not per frame.
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


# A reprojected point can land far outside any real pixel coordinate --
# not just from a genuine calibration/triangulation glitch, but reliably
# from a misassigned camera role (a session whose left/right/bottom video
# mapping doesn't match the calibration's), which sends garbage 3D points
# through a camera model they were never actually seen by. NaN/Inf raise
# in plain Python before reaching cv2 at all (int(round(...)) errors out
# first); a huge-but-finite value instead sails through Python and only
# fails once it hits cv2's own C++ point parser (a `cv::Point` is a 32-bit
# int), surfacing as a cryptic "Can't parse 'pt2' ... wrong type" -- which
# used to abort the rest of this session's video, not just skip the one
# bad node. _MAX_COORD is comfortably beyond any real camera frame but
# safely inside int32 range, so anything past it is treated the same as
# "this node/edge isn't drawable this frame" rather than a crash.
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
    """Draw one marker, alpha-blended into the frame instead of drawn
    solid when ``alpha < 1`` -- used to fade an earlier crossing's
    footprints toward the video underneath them rather than just toward
    a flat color, so they read as "left over" rather than as a still-live
    marker in a different color.

    Blending is done on a small region-of-interest around the circle,
    not the whole panel, so fading many accumulated markers stays cheap
    regardless of panel size. Silently skips a non-finite or wildly
    out-of-range point (see :func:`_finite_point`) rather than raising.
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
# (label, color) -- every marker color actually drawn elsewhere in this
# module, so the legend can't drift out of sync with what's on screen.
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
    """A dedicated top strip: shared-timeline time/frame index on the left,
    a color legend (matching every marker color drawn per-panel) filling
    the rest -- kept in its own band above the camera panels rather than
    overlaid on top of video pixels, so neither obscures the other.
    """
    strip = np.full((_STRIP_HEIGHT, width, 3), _STRIP_BG_COLOR, dtype=np.uint8)
    baseline_y = _STRIP_HEIGHT // 2 + 5

    text = f"t={t:.3f}s  frame {frame_idx}/{n_frames - 1}"
    cv2.putText(strip, text, (8, baseline_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    x = 8 + tw + 28
    for label, color in _LEGEND_ENTRIES:
        if x > width - 20:
            break  # ran out of room on a narrow panel -- drop remaining entries rather than overflow
        cv2.circle(strip, (x, _STRIP_HEIGHT // 2), 5, color, -1)
        cv2.putText(strip, label, (x + 10, baseline_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        x += 10 + lw + 18

    return strip


def _merge_crossings(trial: Union[TrialMetrics, List[TrialMetrics]]) -> TrialMetrics:
    """Collapse a recording's per-crossing trials into one whose
    ``paw_events`` covers every crossing, for drawing purposes.

    The validation video renders the whole recording in one pass, so it
    needs every stance event in a single lookup regardless of which
    crossing produced it. Crossing windows never overlap (see
    :func:`alligaitor.gait.find_crossings`), so concatenating in frame
    order is well-defined. Only ``paw_events`` is meaningful on the
    result -- the averaged metrics are per crossing and are deliberately
    not recombined here; nothing in the drawing path reads them.
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
        session: Session configuration -- ``session.videos`` (the cropped,
            model-input clips) are what's actually rendered.
        csv_path: This trial's ``pose_3d.csv``.
        cgroup: Calibrated camera group.
        trial: This recording's already-computed
            :class:`gait.TrialMetrics` (the same instances the group
            workbook is built from) -- either one, or the per-crossing
            list from :func:`alligaitor.gait.compute_crossing_metrics`.
            The video covers the whole recording, so several crossings
            are merged into one set of stance events for drawing; their
            frame ranges are disjoint by construction, so nothing
            overlaps.
        config: The :class:`GaitConfig` ``trial`` was computed with.
        output_path: Destination ``.mp4`` path.
        disagreement_threshold_px: A node is drawn red, regardless of its
            usual color, when its reprojection error exceeds this.
        log: Receives discrete one-off messages (frame count at the
            start) -- same contract as :func:`alligaitor.inference.run_inference`.
        progress: Receives a live, redrawing tqdm-style progress line as
            frames render (see :class:`_FrameProgress`), throttled the
            same way inference's own progress line is. Defaults to
            ``log`` if not given.
        html_progress: Whether ``progress`` wants an HTML-rendered line
            (a rich-text GUI widget) or plain text -- same meaning as
            :func:`alligaitor.inference.run_inference`'s own flag.
        on_redraw_closed: Called once the final frame's progress update
            has been sent to ``progress``, so a caller redrawing a line
            in place (the GUI does) can start the next thing sharing that
            line -- e.g. the next session's own inference bar -- fresh
            instead of overwriting this bar's completed state.

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
        crossing_starts = np.array([0])  # no window info (e.g. a hand-built trial) -- treat as one crossing
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
    # What cap.read() will return next for each role absent a re-seek --
    # same tracking gui/video_player_widget.py's goto_frame() uses. The
    # reference role's native_idx is *always* i (see below), i.e. purely
    # sequential -- calling cap.set() on every one of its frames anyway
    # was forcing a real seek (nearest-keyframe + decode-forward on
    # compressed footage) for what should just be the next read(), and
    # was a large, easily-avoidable chunk of this function's per-frame
    # cost across all three cameras.
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
                # "mp4v" (MPEG-4 Part 2) writes a technically-valid stream
                # but many players (QuickTime, Safari, in-app previews)
                # render it as garbled macroblocks rather than falling
                # back gracefully -- "avc1" (H.264) is broadly playable.
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
