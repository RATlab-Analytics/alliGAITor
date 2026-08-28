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

"""Default crop target size, editable per-crop in the GUI dialog."""

from __future__ import annotations

from pathlib import Path

CROP_TARGET_WIDTH = 1280
CROP_TARGET_HEIGHT = 170

# Used only if side_crop_size_for_model() can't read a size from the model's training data.
_FALLBACK_SIDE_CROP_WIDTH = 1280
_FALLBACK_SIDE_CROP_HEIGHT = 170


def side_crop_size_for_model(model_dir) -> tuple[int, int]:
    """(width, height) the side model at ``model_dir`` was trained on, read from its
    ``labels_gt.train.0.slp`` (the model's own config has no fixed input size). Falls back to a
    hardcoded default if that file is missing or unreadable."""
    labels_path = Path(model_dir) / "labels_gt.train.0.slp"
    if labels_path.exists():
        try:
            import sleap_io as sio

            labels = sio.load_slp(str(labels_path))
            _, height, width, _ = labels.videos[0].shape
            return int(width), int(height)
        except Exception:
            pass
    return _FALLBACK_SIDE_CROP_WIDTH, _FALLBACK_SIDE_CROP_HEIGHT
