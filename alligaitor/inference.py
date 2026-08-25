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
    by a ``color`` token in the model directory's own name.

    This is deliberately NOT read from sleap-nn's own
    ``ensure_rgb``/``ensure_grayscale`` training-config fields (see
    ``_read_color_mode``) -- those only describe the channel *shape* a
    model expects (3-channel vs 1-channel), not the actual pixel content
    it was trained on. Content is decided independently, upstream, by
    :mod:`alligaitor.preprocessing`'s grayscale re-encode (or an
    equivalent conversion done by hand before labeling) -- e.g. the
    existing side model is ``ensure_rgb: true``-shaped (matches its
    ConvNeXt backbone) but was still trained on force-grayscale-converted,
    achromatic content, so shape alone would give the wrong answer here.

    Instead this follows the same convention every model directory here
    already uses to record what makes it different from the last one
    (``ConvNeXt``, ``single_instance``, ``n=252``, ...): name a
    color-trained model's directory with ``color`` as one of its
    ``.``/``_``/``-``-delimited components (e.g.
    ``alliGAITor_bottom_slim_v1.1.0.color.n=310``) and it's picked up
    automatically, no separate marker file or registration step needed.
    A directory name without that token is assumed grayscale-only -- the
    correct read for every model trained so far.
    """
    parts = re.split(r"[^a-zA-Z0-9]+", Path(model_dir).name.lower())
    return _COLOR_TOKEN in parts


def _read_color_mode(model_dir: Path) -> "tuple[Optional[bool], Optional[bool]]":
    """Read ``(ensure_rgb, ensure_grayscale)`` from a model's own training
    config, or ``(None, None)`` if that config can't be found/parsed.

    The model directory is the single source of truth for what color mode
    a model expects -- it's set once at training time and every consumer
    (inference's own color-mode flags, and whether to strip color from the
    input video before inference even runs -- see ``run_inference``'s
    ``force_grayscale``) should derive from it rather than assuming every
    model is grayscale, which stops being true the moment a color-trained
    model exists alongside the grayscale ones.
    """
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
    """Build explicit ``--ensure_rgb``/``--ensure_grayscale`` flags for a model.

    ``sleap-nn predict`` is documented to fall back to the values recorded
    in the model's own ``training_config.yaml`` when these are left unset,
    but that fallback happens on the far side of a subprocess call and a
    silent mismatch (e.g. feeding an RGB-trained model grayscale input, or
    vice versa) degrades accuracy without raising an error. Reading the
    training config here and passing the flags explicitly makes the
    color mode an assertion instead of an assumption.
    """
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
        tracking: Whether to run SLEAP-NN's tracker on the predictions.
            These are single-instance models, so tracking is only useful
            here for its identity-smoothing effect on left/right paw
            flicker between frames.
        force_grayscale: Whether to re-encode the video to true
            single-channel grayscale content (see
            :mod:`alligaitor.preprocessing`) before inference, stripping
            any color/chroma content regardless of the channel *count*
            ``model_dir`` expects. Left as ``None`` (the default), this
            is decided by :func:`model_trained_on_color` -- grayscale is
            forced unless ``model_dir``'s own name has a ``color`` token,
            so color-trained and grayscale-trained models can coexist
            without the caller having to know which is which.
            Pass ``True``/``False`` explicitly to override -- e.g.
            ``False`` if ``video_path`` is already a verified
            grayscale-content file and re-encoding would be wasted work.
        peak_threshold: Minimum confidence map value for a detection to be
            kept; anything below is dropped entirely rather than returned
            as a low-confidence point. Defaults to ``sleap-nn predict``'s
            own default (``0.2``) when left as ``None``. Lower this to
            compare against SLEAP's GUI inference, which may use a
            different default.
        log: Receives discrete one-off messages (the command being run;
            the full subprocess output if it fails -- see
            :mod:`alligaitor.subprocess_streaming`).
        progress: Receives ``sleap-nn predict``'s own live tqdm progress
            output as it runs -- repeated calls for what's conceptually
            the same redrawing line, as opposed to ``log``'s discrete
            messages, so a caller that wants to show that in place (the
            GUI does) can tell the two apart. Defaults to ``log`` if not
            given.
        html_progress: If True, ``progress`` receives an HTML rendering
            of ``sleap-nn predict``'s own colored progress bar instead
            of plain text -- for a caller wired up to a rich-text widget
            (the GUI is; see :mod:`alligaitor.ansi_html`). Leave False
            for a plain-text/print()-based ``progress``.
        on_redraw_closed: Forwarded to
            :func:`alligaitor.subprocess_streaming.stream_subprocess` --
            called whenever a redrawn progress line's definitive final
            state has just been sent to ``progress``, so a caller
            redrawing in place can start the next update fresh instead
            of immediately overwriting it.

    Returns:
        Path to the written ``.slp`` predictions file.
    """
    video_path = Path(video_path)
    model_dir = Path(model_dir)
    if output_path is None:
        output_path = video_path.with_suffix(video_path.suffix + ".predictions.slp")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
