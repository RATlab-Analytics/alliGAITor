"""
Small shared helpers that RATlab-NOR's crop_setup_dialog.py originally
pulled in from object_picker.py and object_setup_dialog.py -- neither of
which exists here, and both of which carry a lot of NOR-specific
object-hitbox logic alliGAITor doesn't need. Extracted down to just the
three functions crop_setup_dialog.py actually uses.

RATlab-NOR is MIT-licensed (Copyright (c) 2026 Mitchell Carson); see
../THIRD_PARTY_NOTICES.md for the full license text.
"""

from __future__ import annotations

from pathlib import Path

import cv2
from PySide6.QtGui import QImage


def grab_first_frame(video_path):
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read first frame of {video_path}")
    return frame


def grab_middle_frame(video_path):
    """Like grab_first_frame(), but seeks to the middle of the video --
    used for crop positioning/preview since the rat is much more likely
    to actually be in frame partway through a recording than at frame 0
    (which is often just the empty tunnel/rig before the rat is placed
    in it). Falls back to the first frame if the video's frame count
    can't be read or seeking fails."""
    cap = cv2.VideoCapture(str(video_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, n_frames // 2)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return grab_first_frame(video_path)
    return frame


def bgr_frame_to_qimage(frame) -> QImage:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return qimg.copy()  # own the buffer -- `rgb` goes out of scope after this call


def video_key(video_path, video_folder):
    """Stable identifier for a video, expressed relative to the input
    folder rather than as an absolute path -- so crop_positions.json
    stays valid even if the whole project folder is moved, renamed, or
    copied to another machine."""
    return str(Path(video_path).resolve().relative_to(Path(video_folder).resolve()))
