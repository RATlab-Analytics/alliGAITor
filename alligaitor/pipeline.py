"""End-to-end orchestration: 2D inference, calibration, and triangulation."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from alligaitor import calibration, inference, triangulation
from alligaitor.config import CAMERA_ROLES, ModelConfig, PipelineConfig, SessionConfig
from alligaitor.timing import video_fps
from alligaitor.triangulation import Pose3D

from aniposelib.cameras import CameraGroup


def run_session(
    session: SessionConfig,
    models: ModelConfig,
    cgroup: CameraGroup,
    device: str = "auto",
    tracking: bool = False,
) -> Path:
    """Run 2D inference and 3D triangulation for a single session.

    Args:
        session: Session configuration (video paths, output directory).
        models: Trained model directories.
        cgroup: Calibrated camera group (see :mod:`alligaitor.calibration`).
        device: Torch device passed through to SLEAP-NN inference.
        tracking: Whether to run SLEAP-NN's tracker during inference.

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
            video_path, model_dir, output_path=slp_path, device=device, tracking=tracking
        )
        tracks[role] = inference.load_predictions(slp_path)
        fps_by_role[role] = video_fps(video_path)

    pose_3d = triangulation.triangulate(tracks, cgroup, fps_by_role)

    csv_path = session.output_dir / f"{session.name}.pose_3d.csv"
    save_pose_3d_csv(pose_3d, csv_path)
    return csv_path


def save_pose_3d_csv(pose_3d: Pose3D, csv_path: Path) -> None:
    """Write a triangulated pose to a long-format CSV: frame, node, x, y, z, error."""
    n_frames, n_nodes, _ = pose_3d.points.shape
    frame_idx, node_idx = np.meshgrid(np.arange(n_frames), np.arange(n_nodes), indexing="ij")
    node_names = np.array(pose_3d.node_names)

    df = pd.DataFrame(
        {
            "frame": frame_idx.ravel(),
            "node": node_names[node_idx.ravel()],
            "x": pose_3d.points[..., 0].ravel(),
            "y": pose_3d.points[..., 1].ravel(),
            "z": pose_3d.points[..., 2].ravel(),
            "reprojection_error_px": pose_3d.reprojection_error.ravel(),
        }
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)


def run_pipeline(config: PipelineConfig, device: str = "auto", tracking: bool = False) -> List[Path]:
    """Run calibration (loading a saved one if present) and triangulate every session."""
    if config.calibration.output_path.exists():
        cgroup = calibration.load(config.calibration)
    else:
        cgroup = calibration.calibrate(config.calibration)

    return [
        run_session(session, config.models, cgroup, device=device, tracking=tracking)
        for session in config.sessions
    ]
