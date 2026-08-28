"""Crop-offset bookkeeping for videos produced by the crop tool.

Inference runs on cropped video, but calibration is done on raw
uncropped frames, so each keypoint needs its crop's top-left offset added
back before triangulation (see :func:`alligaitor.pipeline.load_track`).
Offsets are recorded by the crop tool in a ``crop_positions.json`` file
alongside the cropped output, keyed by filename with value ``[x, y]``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, Union

PathLike = Union[str, Path]


def crop_offset_for_video(video_path: PathLike) -> Tuple[float, float]:
    """Look up ``video_path``'s crop offset from a sibling
    ``crop_positions.json``. Returns ``(0.0, 0.0)`` if no such file exists
    or it doesn't list this video."""
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
