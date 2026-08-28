"""Wall-clock-time-based frame alignment across cameras.

Camera frame rates vary independently (auto-exposure), so frames are matched across views by
estimated wall-clock time (index / fps, recordings assumed synced at frame 0) rather than raw
frame index.
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
