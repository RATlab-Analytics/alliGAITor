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


def _baseline_pose_3d_csv_path(session: SessionConfig) -> Path:
    """Path for the pre-fallback triangulation CSV, shared by :func:`run_session` and :func:`run_group`."""
    return session.output_dir / f"{session.name}.pose_3d.pre_fallback.csv"


def load_track(video_path: Path, slp_path: Path) -> PoseTrack2D:
    """Load one camera view's 2D predictions, shifted from crop-local into uncropped frame coordinates."""
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
    bottom_fallback: bool = False,
) -> Path:
    """Run 2D inference and 3D triangulation for a single session.

    Args:
        session: Session configuration (video paths, output directory).
        models: Trained model directories.
        cgroup: Calibrated camera group.
        device: Torch device passed to SLEAP-NN inference.
        tracking: Whether to run SLEAP-NN's tracker during inference.
        bottom_fallback: See :attr:`alligaitor.config.PipelineConfig.bottom_fallback`.

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

    baseline_pose_3d = triangulation.triangulate(tracks, cgroup, fps_by_role)
    pose_3d = baseline_pose_3d
    fallback_mask = None
    if bottom_fallback:
        from alligaitor import bottom_fallback as _bottom_fallback  # local import: see PipelineConfig.bottom_fallback

        pose_3d, fallback_mask = _bottom_fallback.fill_gaps(baseline_pose_3d, tracks, cgroup, fps_by_role)
    # Pose is resampled onto the slowest camera's frame times, so that fps is the true seconds-per-row.
    reference_fps = min(fps_by_role.values())

    csv_path = session.output_dir / f"{session.name}.pose_3d.csv"
    save_pose_3d_csv(pose_3d, reference_fps, csv_path, fallback_mask=fallback_mask)
    if bottom_fallback:
        save_pose_3d_csv(baseline_pose_3d, reference_fps, _baseline_pose_3d_csv_path(session))
    return csv_path


def save_pose_3d_csv(
    pose_3d: Pose3D, fps: float, csv_path: Path, fallback_mask: Optional[np.ndarray] = None
) -> None:
    """Write a triangulated pose to a long-format CSV: frame, time_s, node, x, y, z, error, fallback.

    Args:
        fallback_mask: Boolean ``(n_frames, n_nodes)`` array, ``True`` where a point came from
            the bottom-fallback fill rather than real triangulation. Defaults to all ``False``.
    """
    n_frames, n_nodes, _ = pose_3d.points.shape
    frame_idx, node_idx = np.meshgrid(np.arange(n_frames), np.arange(n_nodes), indexing="ij")
    node_names = np.array(pose_3d.node_names)
    if fallback_mask is None:
        fallback_mask = np.zeros((n_frames, n_nodes), dtype=bool)

    df = pd.DataFrame(
        {
            "frame": frame_idx.ravel(),
            "time_s": frame_idx.ravel() / fps,
            "node": node_names[node_idx.ravel()],
            "x": pose_3d.points[..., 0].ravel(),
            "y": pose_3d.points[..., 1].ravel(),
            "z": pose_3d.points[..., 2].ravel(),
            "reprojection_error_px": pose_3d.reprojection_error.ravel(),
            "fallback": fallback_mask.ravel(),
        }
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)


def _load_or_calibrate(calib_config) -> CameraGroup:
    if calib_config.output_path.exists():
        return calibration.load(calib_config)
    return calibration.calibrate(calib_config)


def aligned_camera_validity(session: SessionConfig) -> "dict":
    """Per paw, per camera role, whether that camera had a valid 2D detection on each shared-timeline frame.

    Reloads and re-aligns each role's cached ``<role>.predictions.slp`` rather than rerunning inference.
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
            bottom_fallback=config.bottom_fallback,
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

    Triangulates every session, computes gait metrics from each 3D trajectory, and writes one
    Excel workbook for the group with one tab per distinct ``rat_id``.

    Args:
        progress_callback: If given, called as
            ``progress_callback(session.name, sessions_done, sessions_total)`` after each session.
        log: Forwarded to :func:`run_session` for discrete one-off messages.
        progress: Forwarded to :func:`run_session` for the live per-video progress line.
        html_progress: Forwarded to :func:`run_session`; whether ``progress`` wants HTML color or plain text.
        on_redraw_closed: Forwarded to :func:`run_session`.
        validation_dir: If given, an annotated validation video is rendered for every session into
            ``validation_dir/<session.name>.validation.mp4``. Best-effort: a failure is logged and
            skipped rather than failing the run. ``None`` or ``config.skip_validation_videos`` skips export.

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
            bottom_fallback=config.bottom_fallback,
        )
        # One recording can hold several crossings, each its own trial with its own travel direction.
        crossings = gait.compute_crossing_metrics(
            csv_path,
            session_name=session.name,
            rat_id=session.rat_id,
            config=config.gait,
        )

        times, positions, _, fallback = gait.load_pose_3d(csv_path)
        crossings = [
            gait.restrict_to_consecutive_runs(t, times, positions, config.gait)
            for t in crossings
        ]
        crossings = [
            gait.attach_bottom_fallback_fraction(t, times, positions, fallback, config.gait)
            for t in crossings
        ]
        if config.bottom_fallback:
            from alligaitor import bottom_fallback as _bottom_fallback  # local import: see PipelineConfig.bottom_fallback

            baseline_csv_path = _baseline_pose_3d_csv_path(session)
            baseline_crossings = gait.compute_crossing_metrics(
                baseline_csv_path, session_name=session.name, rat_id=session.rat_id, config=config.gait,
            )
            baseline_times, baseline_positions, _, _ = gait.load_pose_3d(baseline_csv_path)
            baseline_crossings = [
                gait.restrict_to_consecutive_runs(t, baseline_times, baseline_positions, config.gait)
                for t in baseline_crossings
            ]
            crossings = _bottom_fallback.guard_against_regression(
                crossings, baseline_crossings, baseline_times, baseline_positions, config.gait,
            )
        if len(crossings) > 1:
            log(f"[{session.name}] {len(crossings)} crossings detected: "
                + ", ".join(f"frames {t.crossing_window[0]}-{t.crossing_window[1]}" for t in crossings))

        gait.save_paw_events_csv(crossings, session.output_dir / f"{session.name}.paw_events.csv")
        validation.save_validation_summary(
            crossings, times, positions, config.gait,
            session.output_dir / f"{session.name}.validation_summary.json",
            fallback_mask=fallback,
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
