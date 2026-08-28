# RATlab alliGAITor: an open-source rodent gait analysis pipeline for research
# Copyright (C) 2026 Mitchell Carson
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Small frame-loading and video-identity helpers shared by the crop tools."""

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
    """Like grab_first_frame(), but seeks to the middle of the video, since frame 0 is often
    empty. Falls back to the first frame if seeking fails."""
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
    return qimg.copy()  # own the buffer; `rgb` goes out of scope after this call


def video_key(video_path, video_folder):
    """Stable identifier for a video, relative to the input folder rather than absolute, so
    crop_positions.json stays valid if the project folder is moved or copied."""
    return str(Path(video_path).resolve().relative_to(Path(video_folder).resolve()))
