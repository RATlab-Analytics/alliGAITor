"""End-to-end orchestration: 2D inference, calibration, and triangulation."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from alligaitor import calibration, cropping, inference, triangulation
from alligaitor.config import CAMERA_ROLES, ModelConfig, PipelineConfig, SessionConfig
from alligaitor.inference import PoseTrack2D
from alligaitor.timing import video_fps
from alligaitor.triangulation import Pose3D

from aniposelib.cameras import CameraGroup


def load_track(video_path: Path, slp_path: Path) -> PoseTrack2D:
    """Load one camera view's 2D predictions, corrected into its uncropped source frame.

    ``slp_path`` was predicted on ``video_path``, which is typically a
    crop (the models are trained on cropped footage), so its keypoints
    come out in that crop's own local pixel coordinates. Triangulation
    needs coordinates in the same *uncropped* frame calibration was
    computed against, so this adds back ``video_path``'s crop offset
    (see :mod:`alligaitor.cropping`) -- a no-op if ``video_path`` isn't a
    tracked crop.
    """
    track = inference.load_predictions(slp_path)
    offset_x, offset_y = cropping.crop_offset_for_video(video_path)
    if offset_x or offset_y:
        track.points[..., 0] += offset_x
        track.points[..., 1] += offset_y
    return track


def run_session(
    session: SessionConfig,
    models: ModelConfig,
    cgroup: CameraGroup,
    device: str = "auto",
    tracking: bool = False,
    log: Callable[[str], None] = print,
    progress: Optional[Callable[[str], None]] = None,
    html_progress: bool = False,
    on_redraw_closed: Optional[Callable[[], None]] = None,
) -> Path:
    """Run 2D inference and 3D triangulation for a single session.

    Args:
        session: Session configuration (video paths, output directory).
        models: Trained model directories.
        cgroup: Calibrated camera group (see :mod:`alligaitor.calibration`).
        device: Torch device passed through to SLEAP-NN inference.
        tracking: Whether to run SLEAP-NN's tracker during inference.
        log: Forwarded to :func:`alligaitor.inference.run_inference` for
            each of this session's three camera roles -- discrete
            one-off messages (the command run, full output on failure).
        progress: Forwarded to :func:`alligaitor.inference.run_inference`
            -- the live tqdm-style progress line for whichever camera
            role is currently running inference. Defaults to ``log`` if
            not given (see that function's own docstring).
        html_progress: Forwarded to :func:`alligaitor.inference.run_inference`
            -- whether ``progress`` wants HTML-rendered color or plain text.
        on_redraw_closed: Forwarded to
            :func:`alligaitor.inference.run_inference` -- called whenever
            a redrawn progress line's definitive final state has just
            been sent to ``progress``.

    Returns:
        Path to the written 3D trajectory CSV.
    """
    session.output_dir.mkdir(parents=True, exist_ok=True)

    tracks = {}
    fps_by_role = {}
    for role in CAMERA_ROLES:
        video_path = session.videos[role]
        model_dir = models.model_dir_for_role(role)
        slp_path = session.output_dir / f"{role}.predictions.slp"
        inference.run_inference(
            video_path, model_dir, output_path=slp_path, device=device, tracking=tracking,
            log=log, progress=progress, html_progress=html_progress, on_redraw_closed=on_redraw_closed,
        )
        tracks[role] = load_track(video_path, slp_path)
        fps_by_role[role] = video_fps(video_path)

    # triangulation.triangulate_axis_prioritized() also exists (see its
    # docstring), but isn't used here: on real data it barely applies to
    # paws at all (they almost never have all three cameras valid
    # simultaneously) and measurably worsens body-node reprojection
    # error, apparently because the calibration's side cameras disagree
    # on world "up" by ~30 degrees (see
    # calibration.world_up_direction's own warning), undermining the
    # "sides are reliable for height" assumption it's built on. Revisit
    # once that calibration issue is understood/fixed.
    pose_3d = triangulation.triangulate(tracks, cgroup, fps_by_role)
    # Matches triangulation.align_tracks_by_time's choice of reference
    # timeline: the pose is resampled onto the slowest camera's own frame
    # times, so that view's fps gives the true seconds-per-row of pose_3d.
    reference_fps = min(fps_by_role.values())

    csv_path = session.output_dir / f"{session.name}.pose_3d.csv"
    save_pose_3d_csv(pose_3d, reference_fps, csv_path)
    return csv_path


def save_pose_3d_csv(pose_3d: Pose3D, fps: float, csv_path: Path) -> None:
    """Write a triangulated pose to a long-format CSV: frame, time_s, node, x, y, z, error."""
    n_frames, n_nodes, _ = pose_3d.points.shape
    frame_idx, node_idx = np.meshgrid(np.arange(n_frames), np.arange(n_nodes), indexing="ij")
    node_names = np.array(pose_3d.node_names)

    df = pd.DataFrame(
        {
            "frame": frame_idx.ravel(),
            "time_s": frame_idx.ravel() / fps,
            "node": node_names[node_idx.ravel()],
            "x": pose_3d.points[..., 0].ravel(),
            "y": pose_3d.points[..., 1].ravel(),
            "z": pose_3d.points[..., 2].ravel(),
            "reprojection_error_px": pose_3d.reprojection_error.ravel(),
        }
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)


def _load_or_calibrate(calib_config) -> CameraGroup:
    if calib_config.output_path.exists():
        return calibration.load(calib_config)
    return calibration.calibrate(calib_config)


def aligned_camera_validity(session: SessionConfig) -> "dict":
    """Per paw, per camera role, whether that camera had a valid 2D
    detection on each shared-timeline frame -- reloads and re-aligns
    each role's cached ``<role>.predictions.slp`` (must already exist,
    e.g. from :func:`run_session`) rather than rerunning inference. See
    :func:`alligaitor.gait.cam_valid_by_paw_from_aligned`.
    """
    from alligaitor import gait  # local import: avoids a pipeline<->gait import cycle

    tracks = {}
    fps_by_role = {}
    for role in CAMERA_ROLES:
        slp_path = session.output_dir / f"{role}.predictions.slp"
        tracks[role] = load_track(session.videos[role], slp_path)
        fps_by_role[role] = video_fps(session.videos[role])
    aligned = triangulation.align_tracks_by_time(tracks, fps_by_role)
    return gait.cam_valid_by_paw_from_aligned(aligned)


def run_pipeline(
    config: PipelineConfig,
    device: str = "auto",
    tracking: bool = False,
    log: Callable[[str], None] = print,
    progress: Optional[Callable[[str], None]] = None,
    html_progress: bool = False,
    on_redraw_closed: Optional[Callable[[], None]] = None,
) -> List[Path]:
    """Run calibration (loading a saved one if present) and triangulate every session."""
    cgroup = _load_or_calibrate(config.calibration)
    return [
        run_session(
            session, config.models, cgroup, device=device, tracking=tracking,
            log=log, progress=progress, html_progress=html_progress, on_redraw_closed=on_redraw_closed,
        )
        for session in config.sessions
    ]


def run_group(
    config: PipelineConfig,
    device: str = "auto",
    tracking: bool = False,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    log: Callable[[str], None] = print,
    progress: Optional[Callable[[str], None]] = None,
    html_progress: bool = False,
    on_redraw_closed: Optional[Callable[[], None]] = None,
    validation_dir: Optional[Path] = None,
) -> Path:
    """Run the full pipeline for every session in a group and write its gait-metrics workbook.

    This is the entry point a job queue (see the module docstring in
    :mod:`alligaitor.gait`) should call per queued group: it triangulates
    every session, computes gait metrics for each from its 3D trajectory,
    and writes one Excel workbook for the group with one tab per distinct
    ``rat_id``.

    Args:
        progress_callback: If given, called as
            ``progress_callback(session.name, sessions_done, sessions_total)``
            immediately after each session finishes (inference,
            triangulation, and gait metrics -- everything but the final
            group-wide workbook write, which only happens once at the
            end). Lets a GUI job queue show per-session progress without
            this function needing to know anything about how progress is
            displayed.
        log: Forwarded to :func:`run_session` for each session -- discrete
            one-off messages, not the live per-video progress line.
        progress: Forwarded to :func:`run_session` -- the live tqdm-style
            progress line from whichever camera role is currently running
            inference. This is what makes a long-running session's
            inference stage visible while it's happening, rather than
            the log going quiet for however long it takes (see
            :func:`alligaitor.inference.run_inference`). Distinct from
            ``progress_callback`` above, which only fires once *between*
            sessions, not during one.
        html_progress: Forwarded to :func:`run_session` -- whether
            ``progress`` wants HTML-rendered color (a rich-text GUI
            widget; see :mod:`alligaitor.ansi_html`) or plain text.
        on_redraw_closed: Forwarded to :func:`run_session` -- called
            whenever a redrawn progress line's definitive final state
            has just been sent to ``progress``, so a caller redrawing in
            place can start the next update fresh instead of immediately
            overwriting it (matters when a session's inference prints
            more than one tqdm bar in sequence -- otherwise one bar's
            true completion flashes for an instant before the next bar's
            first redraw overwrites it in the same spot).
        validation_dir: If given, an annotated validation video (see
            :func:`alligaitor.validation_video.export_validation_video`) is
            rendered for every session into
            ``validation_dir/<session.name>.validation.mp4``, reusing the
            same ``log``/``progress``/``html_progress``/``on_redraw_closed``
            as inference above -- so rendering shows its own live,
            tqdm-styled progress line the same way a camera role's
            inference does, rather than the log going quiet for however
            long a render takes. Rendering is best-effort: a failure is
            logged via ``log`` and skipped rather than failing the whole
            run -- the workbook and every other session's video are still
            produced. ``None`` (the default) skips video export entirely,
            as does ``config.skip_validation_videos`` -- a per-group
            opt-out (see :class:`alligaitor.config.PipelineConfig`) that
            takes effect even when a ``validation_dir`` is given.

    Returns:
        Path to the written gait-metrics workbook.
    """
    from alligaitor import gait, validation, validation_video  # local import: avoids a pipeline<->gait import cycle

    cgroup = _load_or_calibrate(config.calibration)

    trials = []
    total = len(config.sessions)
    for i, session in enumerate(config.sessions, start=1):
        csv_path = run_session(
            session, config.models, cgroup, device=device, tracking=tracking,
            log=log, progress=progress, html_progress=html_progress, on_redraw_closed=on_redraw_closed,
        )
        # One recording can hold several crossings (the rat walks the tunnel,
        # turns, walks back, turns, walks again). Each is its own trial with
        # its own direction of travel -- see gait.compute_crossing_metrics.
        crossings = gait.compute_crossing_metrics(
            csv_path,
            session_name=session.name,
            rat_id=session.rat_id,
            config=config.gait,
        )

        times, positions, _ = gait.load_pose_3d(csv_path)
        crossings = [
            gait.restrict_to_consecutive_runs(t, times, positions, config.gait)
            for t in crossings
        ]
        if len(crossings) > 1:
            log(f"[{session.name}] {len(crossings)} crossings detected: "
                + ", ".join(f"frames {t.crossing_window[0]}-{t.crossing_window[1]}" for t in crossings))

        gait.save_paw_events_csv(crossings, session.output_dir / f"{session.name}.paw_events.csv")
        validation.save_validation_summary(
            crossings, times, positions, config.gait,
            session.output_dir / f"{session.name}.validation_summary.json",
        )
        if validation_dir is not None and not config.skip_validation_videos:
            try:
                validation_video.export_validation_video(
                    session, csv_path, cgroup, crossings, config.gait,
                    Path(validation_dir) / f"{session.name}.validation.mp4",
                    log=log, progress=progress, html_progress=html_progress,
                    on_redraw_closed=on_redraw_closed,
                )
            except Exception as exc:
                log(f"[{session.name}] validation video export failed: {exc}")
        trials.extend(crossings)

        if progress_callback is not None:
            progress_callback(session.name, i, total)

    predictions_root = config.sessions[0].output_dir.parent if config.sessions else Path(".")
    manual_flags = validation.load_group_manual_flags(
        predictions_root, [session.name for session in config.sessions]
    )
    gait.write_group_report(trials, config.output_xlsx, manual_flags=manual_flags)
    return config.output_xlsx
