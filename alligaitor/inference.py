"""Per-camera 2D pose inference via SLEAP-NN.

Wraps the ``sleap-nn predict`` CLI to run a trained model against a
session video, then loads the resulting predictions into a plain numpy
array keyed by skeleton node name.
"""

from __future__ import annotations

import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import sleap_io as sio
import yaml

from alligaitor import preprocessing

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


def _color_mode_flags(model_dir: Path) -> List[str]:
    """Build explicit ``--ensure_rgb``/``--ensure_grayscale`` flags for a model.

    ``sleap-nn predict`` is documented to fall back to the values recorded
    in the model's own ``training_config.yaml`` when these are left unset,
    but that fallback happens on the far side of a subprocess call and a
    silent mismatch (e.g. feeding an RGB-trained model grayscale input, or
    vice versa) degrades accuracy without raising an error. Reading the
    training config here and passing the flags explicitly makes the
    color mode an assertion instead of an assumption.
    """
    config_path = model_dir / "training_config.yaml"
    if not config_path.exists():
        config_path = model_dir / "initial_config.yaml"
    if not config_path.exists():
        warnings.warn(
            f"No training_config.yaml/initial_config.yaml found in {model_dir}; "
            "falling back to sleap-nn's own color-mode default."
        )
        return []

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    try:
        preprocessing = cfg["data_config"]["preprocessing"]
        ensure_rgb = preprocessing["ensure_rgb"]
        ensure_grayscale = preprocessing["ensure_grayscale"]
    except (KeyError, TypeError):
        warnings.warn(
            f"Could not find data_config.preprocessing.ensure_rgb/ensure_grayscale in "
            f"{config_path}; falling back to sleap-nn's own color-mode default."
        )
        return []

    return [
        "--ensure_rgb" if ensure_rgb else "--no-ensure_rgb",
        "--ensure_grayscale" if ensure_grayscale else "--no-ensure_grayscale",
    ]


def run_inference(
    video_path: Path,
    model_dir: Path,
    output_path: Optional[Path] = None,
    device: str = DEFAULT_DEVICE,
    tracking: bool = False,
    force_grayscale: bool = True,
    peak_threshold: Optional[float] = None,
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
        force_grayscale: Both models were trained on achromatic footage,
            so by default the video is re-encoded to true single-channel
            grayscale content (see :mod:`alligaitor.preprocessing`) before
            inference, stripping any color/chroma noise regardless of the
            channel *count* each model expects. Set to ``False`` only if
            ``video_path`` is already a verified grayscale-content file.
        peak_threshold: Minimum confidence map value for a detection to be
            kept; anything below is dropped entirely rather than returned
            as a low-confidence point. Defaults to ``sleap-nn predict``'s
            own default (``0.2``) when left as ``None``. Lower this to
            compare against SLEAP's GUI inference, which may use a
            different default.

    Returns:
        Path to the written ``.slp`` predictions file.
    """
    video_path = Path(video_path)
    model_dir = Path(model_dir)
    if output_path is None:
        output_path = video_path.with_suffix(video_path.suffix + ".predictions.slp")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data_path = video_path
    if force_grayscale:
        data_path = preprocessing.ensure_grayscale_video(video_path, output_path.parent)

    cmd = [
        "sleap-nn",
        "predict",
        "--data_path",
        str(data_path),
        "--model_paths",
        str(model_dir),
        "--output_path",
        str(output_path),
        "--device",
        device,
    ]
    cmd.extend(_color_mode_flags(model_dir))
    if peak_threshold is not None:
        cmd.extend(["--peak_threshold", str(peak_threshold)])
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

    # Mean confidence per (frame, instance), skipping NaN nodes without
    # nanmean's "Mean of empty slice" warning on all-NaN instances (frames
    # with zero real detections still occupy a slot in this array).
    conf = points_all[..., 2]
    valid = ~np.isnan(conf)
    counts = valid.sum(axis=2)
    sums = np.where(valid, conf, 0.0).sum(axis=2)
    mean_conf = np.divide(sums, counts, out=np.full_like(sums, -np.inf), where=counts > 0)
    best_instance = np.argmax(mean_conf, axis=1)

    for frame_idx in range(n_frames):
        inst = best_instance[frame_idx]
        points[frame_idx] = points_all[frame_idx, inst, :, :2]
        scores[frame_idx] = points_all[frame_idx, inst, :, 2]

    return PoseTrack2D(node_names=node_names, points=points, scores=scores)
