"""Per-camera 2D pose inference via SLEAP-NN.

Wraps the ``sleap-nn predict`` CLI to run a trained model against a
session video, then loads the resulting predictions into a plain numpy
array keyed by skeleton node name.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import sleap_io as sio

DEFAULT_DEVICE = "auto"


@dataclass
class PoseTrack2D:
    """2D pose predictions for one camera view.

    Attributes:
        node_names: Skeleton node names, in the order matching the last
            axis of ``points``.
        points: Array of shape ``(n_frames, n_nodes, 2)`` with pixel
            coordinates. Missing detections are ``NaN``.
        scores: Array of shape ``(n_frames, n_nodes)`` with per-node
            confidence scores.
    """

    node_names: List[str]
    points: np.ndarray
    scores: np.ndarray


def run_inference(
    video_path: Path,
    model_dir: Path,
    output_path: Optional[Path] = None,
    device: str = DEFAULT_DEVICE,
    tracking: bool = False,
) -> Path:
    """Run ``sleap-nn predict`` on a single video and return the output path.

    Args:
        video_path: Video to run inference on.
        model_dir: Trained SLEAP-NN model directory (or its ``best.ckpt`` /
            ``training_config.yaml``, both resolve to the model directory).
        output_path: Destination ``.slp`` path. Defaults to
            ``<video_path>.predictions.slp``.
        device: Torch device to run on (``auto``, ``cpu``, ``cuda``, ``mps``).
        tracking: Whether to run SLEAP-NN's tracker on the predictions.
            These are single-instance models, so tracking is only useful
            here for its identity-smoothing effect on left/right paw
            flicker between frames.

    Returns:
        Path to the written ``.slp`` predictions file.
    """
    video_path = Path(video_path)
    model_dir = Path(model_dir)
    if output_path is None:
        output_path = video_path.with_suffix(video_path.suffix + ".predictions.slp")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "sleap-nn",
        "predict",
        "--data_path",
        str(video_path),
        "--model_paths",
        str(model_dir),
        "--output_path",
        str(output_path),
        "--device",
        device,
    ]
    if tracking:
        cmd.append("--tracking")

    subprocess.run(cmd, check=True)
    return output_path


def load_predictions(slp_path: Path) -> PoseTrack2D:
    """Load a ``.slp`` predictions file into a plain node-indexed array.

    Assumes a single-instance model: if a frame has more than one
    predicted instance, the highest mean-confidence instance is kept.
    """
    labels = sio.load_slp(str(slp_path))
    node_names = list(labels.skeleton.node_names)

    points_all = labels.numpy(untracked=True, return_confidence=True)
    n_frames, n_instances, n_nodes, _ = points_all.shape

    points = np.full((n_frames, n_nodes, 2), np.nan, dtype=np.float64)
    scores = np.full((n_frames, n_nodes), np.nan, dtype=np.float64)

    if n_instances == 0:
        return PoseTrack2D(node_names=node_names, points=points, scores=scores)

    mean_conf = np.nanmean(points_all[..., 2], axis=2)  # (n_frames, n_instances)
    mean_conf = np.where(np.isnan(mean_conf), -np.inf, mean_conf)
    best_instance = np.argmax(mean_conf, axis=1)

    for frame_idx in range(n_frames):
        inst = best_instance[frame_idx]
        points[frame_idx] = points_all[frame_idx, inst, :, :2]
        scores[frame_idx] = points_all[frame_idx, inst, :, 2]

    return PoseTrack2D(node_names=node_names, points=points, scores=scores)
