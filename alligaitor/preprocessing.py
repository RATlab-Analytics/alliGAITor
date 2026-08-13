"""Frame-level color preprocessing shared by all camera views.

Both the side and bottom SLEAP-NN models were trained on footage that is
achromatic in content; any color present in the raw camera recordings is
noise, not signal (compression chroma artifacts, or the per-channel
differences the bottom-camera color-correction step amplifies -- see the
crop tool's notes on that recipe).

``sleap-nn``'s own ``ensure_rgb``/``ensure_grayscale`` flags only control
the channel *count* fed to a model, not whether that content is genuinely
achromatic: ``ensure_rgb`` on an already-3-channel frame -- which is what
every decoded video frame is, regardless of its visual content -- is a
no-op that passes any color noise straight through unchanged. Video is
therefore explicitly re-encoded to true single-channel grayscale content
here, independent of and prior to whichever channel-count flag a given
model requires downstream.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def _resolve_ffmpeg() -> str:
    """Locate an ffmpeg binary: a system install if present, else the one
    bundled by the ``imageio-ffmpeg`` package (see ``requirements.txt``).

    Note: PyPI has a package literally named ``ffmpeg`` -- it is an
    abandoned, unrelated stub that does not provide the actual tool.
    ``imageio-ffmpeg`` is the real dependency; do not swap it for that one.
    """
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def to_grayscale_video(input_path: Path, output_path: Path, crf: int = 12) -> Path:
    """Re-encode a video to true single-channel grayscale content via ffmpeg.

    The output remains a standard 3-plane video file so downstream tools
    that expect a color-shaped frame still work, but every pixel's
    channels are identical: any real color/chroma noise present in the
    source encoding is discarded, not just visually hidden.

    Args:
        input_path: Source video.
        output_path: Destination path for the grayscale copy.
        crf: x264 constant rate factor for the re-encode. Lower is higher
            quality; 12 is visually close to lossless.

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

    The cache is a simple existence check keyed on filename under
    ``cache_dir`` -- if you edit a source video in place, delete its
    cached copy so it gets regenerated.
    """
    video_path = Path(video_path)
    cache_dir = Path(cache_dir)
    grayscale_path = cache_dir / f"{video_path.stem}.grayscale{video_path.suffix}"
    if not grayscale_path.exists():
        to_grayscale_video(video_path, grayscale_path)
    return grayscale_path
