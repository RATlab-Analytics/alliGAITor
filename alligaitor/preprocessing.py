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

"""Frame-level grayscale preprocessing shared by all camera views.

Models are trained on achromatic footage, so video is re-encoded to true single-channel
grayscale content before inference, independent of any channel-count flag used downstream.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def _resolve_ffmpeg() -> str:
    """Locate an ffmpeg binary: a system install if present, else the one bundled with ``imageio-ffmpeg``."""
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def to_grayscale_video(input_path: Path, output_path: Path, crf: int = 12) -> Path:
    """Re-encode a video to true single-channel grayscale content via ffmpeg.

    Output stays a 3-plane video file for downstream compatibility, but every pixel's
    channels are made identical.

    Args:
        input_path: Source video.
        output_path: Destination path for the grayscale copy.
        crf: x264 constant rate factor; lower is higher quality.

    Returns:
        ``output_path``.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _resolve_ffmpeg(),
        "-y",
        "-i",
        str(input_path),
        "-vf",
        "format=gray",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


def ensure_grayscale_video(video_path: Path, cache_dir: Path) -> Path:
    """Return a true-grayscale copy of ``video_path``, encoding it if not cached.

    Cache is keyed on filename under ``cache_dir``; delete a cached copy to force regeneration.
    """
    video_path = Path(video_path)
    cache_dir = Path(cache_dir)
    grayscale_path = cache_dir / f"{video_path.stem}.grayscale{video_path.suffix}"
    if not grayscale_path.exists():
        to_grayscale_video(video_path, grayscale_path)
    return grayscale_path
