"""Crop-offset bookkeeping for videos produced by the crop tool.

The SLEAP-NN models here are trained on -- and must therefore run
inference on -- cropped video (see e.g. ``videos/side-training-data/``,
``videos/bottom_training_data_cropped/``), not the raw, full-frame
recordings. Camera calibration, however, is always run against the raw,
uncropped recordings (see :mod:`alligaitor.calibration`), so the
resulting camera intrinsics/extrinsics describe pixel coordinates in
that uncropped frame -- not the cropped frame a model's keypoints come
out in. Every keypoint from a cropped video needs its crop's top-left
offset added back before it can be triangulated correctly; see
:func:`alligaitor.pipeline.load_track`, where that correction is applied.

The crop tool (``tools/crop_setup_dialog.py``) records each cropped
video's offset, in that video's own uncropped frame's pixel coordinates,
in a ``crop_positions.json`` file alongside the cropped output -- one
entry per video, keyed by filename, value ``[x, y]``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, Union

PathLike = Union[str, Path]


def crop_offset_for_video(video_path: PathLike) -> Tuple[float, float]:
    """Look up ``video_path``'s crop offset from a sibling ``crop_positions.json``.

    Returns ``(0.0, 0.0)`` -- a no-op offset -- if no
    ``crop_positions.json`` exists next to ``video_path``, or if that
    file doesn't list this video by filename (i.e. the video isn't a
    tracked crop, such as a raw calibration recording).
    """
    video_path = Path(video_path)
    positions_path = video_path.parent / "crop_positions.json"
    if not positions_path.exists():
        return (0.0, 0.0)

    with open(positions_path) as f:
        positions = json.load(f)

    offset = positions.get(video_path.name)
    if offset is None:
        return (0.0, 0.0)

    x, y = offset
    return (float(x), float(y))
