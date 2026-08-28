"""Per-camera 2D pose inference via SLEAP-NN.

Wraps the ``sleap-nn predict`` CLI to run a trained model against a
session video, then loads the resulting predictions into a plain numpy
array keyed by skeleton node name.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import sleap_io as sio
import yaml

from alligaitor import preprocessing
from alligaitor.subprocess_streaming import stream_subprocess

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


_COLOR_TOKEN = "color"


def model_trained_on_color(model_dir: Path) -> bool:
    """Whether ``model_dir`` was trained on genuine color content, judged
    by a ``color`` token in the model directory's own name (e.g.
    ``..._color_n=310``). This is separate from the model's
    ``ensure_rgb``/``ensure_grayscale`` channel shape (see
    ``_read_color_mode``), which describes input shape, not training
    content. A directory name without the token is assumed grayscale-only.
    """
    parts = re.split(r"[^a-zA-Z0-9]+", Path(model_dir).name.lower())
    return _COLOR_TOKEN in parts


def _read_color_mode(model_dir: Path) -> "tuple[Optional[bool], Optional[bool]]":
    """Read ``(ensure_rgb, ensure_grayscale)`` from a model's own training
    config, or ``(None, None)`` if that config can't be found/parsed."""
    config_path = model_dir / "training_config.yaml"
    if not config_path.exists():
        config_path = model_dir / "initial_config.yaml"
    if not config_path.exists():
        warnings.warn(
            f"No training_config.yaml/initial_config.yaml found in {model_dir}; "
            "falling back to sleap-nn's own color-mode default."
        )
        return None, None

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    try:
        preprocessing = cfg["data_config"]["preprocessing"]
        return bool(preprocessing["ensure_rgb"]), bool(preprocessing["ensure_grayscale"])
    except (KeyError, TypeError):
        warnings.warn(
            f"Could not find data_config.preprocessing.ensure_rgb/ensure_grayscale in "
            f"{config_path}; falling back to sleap-nn's own color-mode default."
        )
        return None, None


def _color_mode_flags(model_dir: Path) -> List[str]:
    """Build explicit ``--ensure_rgb``/``--ensure_grayscale`` flags for a
    model, so a color-mode mismatch fails loudly instead of silently
    degrading accuracy via ``sleap-nn predict``'s own fallback."""
    ensure_rgb, ensure_grayscale = _read_color_mode(model_dir)
    if ensure_rgb is None:
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
    force_grayscale: Optional[bool] = None,
    peak_threshold: Optional[float] = None,
    log: Callable[[str], None] = print,
    progress: Optional[Callable[[str], None]] = None,
    html_progress: bool = False,
    on_redraw_closed: Optional[Callable[[], None]] = None,
) -> Path:
    """Run ``sleap-nn predict`` on a single video and return the output path.

    Args:
        video_path: Video to run inference on.
        model_dir: Trained SLEAP-NN model directory (or its ``best.ckpt`` /
            ``training_config.yaml``, both resolve to the model directory).
        output_path: Destination ``.slp`` path. Defaults to
            ``<video_path>.predictions.slp``.
        device: Torch device to run on (``auto``, ``cpu``, ``cuda``, ``mps``).
        tracking: Whether to run SLEAP-NN's tracker on the predictions,
            for its identity-smoothing effect on left/right paw flicker.
        force_grayscale: Whether to re-encode the video to grayscale
            before inference. Defaults to :func:`model_trained_on_color`'s
            inverse when left ``None``.
        peak_threshold: Minimum confidence for a detection to be kept.
            Defaults to ``sleap-nn predict``'s own default (``0.2``).
        log: Receives discrete one-off messages.
        progress: Receives ``sleap-nn predict``'s live tqdm progress
            output. Defaults to ``log`` if not given.
        html_progress: If True, ``progress`` receives an HTML rendering of
            the colored progress bar instead of plain text.
        on_redraw_closed: Forwarded to
            :func:`alligaitor.subprocess_streaming.stream_subprocess`.

    Returns:
        Path to the written ``.slp`` predictions file.
    """
    video_path = Path(video_path)
    model_dir = Path(model_dir)
    if output_path is None:
        output_path = video_path.with_suffix(video_path.suffix + ".predictions.slp")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        # Reuse prior predictions rather than re-running the expensive,
        # GPU-bound sleap-nn subprocess. Delete the .slp to force fresh
        # inference.
        log(f"  Reusing existing predictions: {output_path}")
        return output_path

    if force_grayscale is None:
        force_grayscale = not model_trained_on_color(model_dir)

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

    log(f"  $ {' '.join(cmd)}")
    returncode, streamer = stream_subprocess(
        cmd, log, progress, html_progress=html_progress, on_redraw_closed=on_redraw_closed
    )
    if returncode != 0:
        streamer.dump_plain_lines("run failed")
        raise RuntimeError(
            f"sleap-nn predict exited with code {returncode} for {video_path}. "
            f"See its output above for details."
        )
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
    # nanmean's "Mean of empty slice" warning on all-NaN instances.
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
