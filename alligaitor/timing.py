"""Wall-clock-time-based frame alignment across cameras.

The three-camera rig's cameras do not run at a fixed, matched frame rate:
auto-exposure trades frame rate for brightness independently per camera, so
the achieved rate varies both between cameras within one recording and
between recordings for the same camera. Matching frames across camera
views by raw frame index (aniposelib's default, and a naive equal-length
assumption elsewhere in this pipeline) therefore does not reliably
correspond to the same real moment beyond the very start of a recording.

Every cross-camera correspondence in this pipeline is resolved by
estimated wall-clock time instead: a frame's time is its index divided by
its own video's frame rate, and recordings are assumed to start in sync at
frame 0 (no hardware genlock or per-frame timestamps are available from
this rig).
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2

PathLike = Union[str, Path]


def video_fps(path: PathLike) -> float:
    """Return a video's frame rate as reported by its container."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not fps or fps <= 0:
        raise ValueError(f"Video reports an invalid frame rate: {path}")
    return fps


def frame_time(frame_idx: int, fps: float) -> float:
    """Return the estimated recording time, in seconds, of a frame index."""
    return frame_idx / fps


def shared_frame_key(frame_idx: int, fps: float, grid_fps: float) -> int:
    """Map a frame index to a shared time bucket, given the camera's own fps.

    ``grid_fps`` sets the bucket width (``1 / grid_fps`` seconds); frames
    from different cameras that land in the same bucket are treated as the
    same moment.
    """
    return round(frame_time(frame_idx, fps) * grid_fps)
