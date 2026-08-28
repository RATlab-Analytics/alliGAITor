"""Crops videos to a fixed target size, for footage recorded at a resolution other than what a
model was trained on. Uses an explicit pixel-for-pixel crop rather than a resize, since a resize
would distort the pixel-space keypoint coordinates that triangulation depends on.

No GUI/Qt dependency. Pipes raw frames to ffmpeg directly rather than cv2.VideoWriter, whose
built-in encoders visibly degrade quality across a full sequence."""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
from pathlib import Path

import cv2
import numpy as np

# Quality knob for the crop's re-encode. Cropped output feeds back into
# inference (not just human review), so bias toward less lossy.
_FFMPEG_CRF = 12

# --- bottom-up color correction ---------------------------------------
#
# Two stacked Photoshop-style Brightness/Contrast layers. Because B/C applies the same nonlinear
# curve independently to each channel, at high contrast tiny real per-channel differences (color
# cast, sensor noise, fur/skin reflectance) get blown into saturated per-channel outputs -- which
# is how this produces color from a nominally white/gray rat.
#
# _BC_LAYERS is the full-strength (100%) recipe; `strength` on apply_bottom_up_color_correction()
# scales it linearly toward a no-op, since the bottom camera's brighter ambient light needs a
# gentler correction than this was tuned against. Opt-in only (color_grade=False by default) --
# side-angle footage should stay untouched.
_BC_LAYERS = [
    (-100, 100),  # (brightness, contrast), Photoshop's -100..100 dialog range
    (-100, 75),
]


def _apply_brightness_contrast(frame: np.ndarray, brightness: float, contrast: float) -> np.ndarray:
    """One Photoshop-style Brightness/Contrast layer, applied identically to every channel.
    Operates in float and doesn't clip to uint8 internally, so stacking layers (_BC_LAYERS)
    composes without clipping prematurely between them."""
    out = frame.astype(np.float32)
    if brightness != 0:
        if brightness > 0:
            shadow, highlight = brightness, 255
        else:
            shadow, highlight = 0, 255 + brightness
        alpha_b = (highlight - shadow) / 255.0
        out = out * alpha_b + shadow
    if contrast != 0:
        f = 131.0 * (contrast + 127.0) / (127.0 * (131.0 - contrast))
        out = out * f + 127.0 * (1.0 - f)
    return out


def apply_bottom_up_color_correction(
    frame_bgr: np.ndarray, strength: float = 1.0, layers: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """Stacks two Brightness/Contrast layers (see _BC_LAYERS) on a BGR uint8 frame, clipping only
    once at the end. strength (0.0-1.0) scales every layer's brightness/contrast linearly toward
    a no-op. layers, if given, overrides _BC_LAYERS/strength with explicit per-layer values."""
    if layers is None:
        strength = max(0.0, min(1.0, strength))
        layers = [(b * strength, c * strength) for b, c in _BC_LAYERS]
    out = frame_bgr.astype(np.float32)
    for brightness, contrast in layers:
        out = _apply_brightness_contrast(out, brightness, contrast)
    return np.clip(out, 0, 255).astype(np.uint8)


def _get_ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def find_videos(folder) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(folder.rglob("*.mp4"))


def probe_resolution(video_path) -> tuple[int, int] | None:
    """Returns (width, height), or None if the video can't be opened."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if w <= 0 or h <= 0:
        return None
    return w, h


def scan_resolutions(folder) -> dict[tuple[int, int], list[Path]]:
    """Groups every video under `folder` by (width, height)."""
    groups: dict[tuple[int, int], list[Path]] = {}
    for video_path in find_videos(folder):
        res = probe_resolution(video_path)
        if res is None:
            res = (-1, -1)  # unreadable; grouped together rather than dropped silently
        groups.setdefault(res, []).append(video_path)
    return groups


class CropRegionError(ValueError):
    pass


def crop_video(video_path, out_path, x: int, y: int, width: int, height: int,
                log=print, color_grade: bool = False, color_grade_strength: float = 1.0,
                color_grade_layers: list[tuple[float, float]] | None = None) -> Path:
    """Crop a single video to the `width`x`height` window starting at (x, y), writing to out_path.
    Raises CropRegionError if that window doesn't fit inside the source frame -- never silently
    clamps, since a shifted crop would put keypoints at the wrong pixel coordinates unnoticed.

    color_grade=True applies apply_bottom_up_color_correction() to every frame before writing;
    color_grade_strength and color_grade_layers are passed through to it."""
    video_path = Path(video_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    res = probe_resolution(video_path)
    if res is None:
        raise RuntimeError(f"Could not open video for reading: {video_path}")
    src_w, src_h = res

    if x < 0 or y < 0 or x + width > src_w or y + height > src_h:
        raise CropRegionError(
            f"Crop window ({width}x{height} at ({x},{y})) doesn't fit inside "
            f"{video_path.name}'s {src_w}x{src_h} frame."
        )

    ffmpeg_exe = _get_ffmpeg_exe()
    if ffmpeg_exe is None:
        raise RuntimeError(
            "video_crop needs the `imageio_ffmpeg` package -- install it with "
            "`pip install imageio-ffmpeg`."
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for reading: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ffmpeg_cmd = [
        ffmpeg_exe, "-y", "-loglevel", "error",
        # -s must match the cropped frame size being piped below, not the source video's size.
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-crf", str(_FFMPEG_CRF), "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    stderr_chunks = []

    def _drain_stderr():
        for chunk in iter(lambda: proc.stderr.read(4096), b""):
            stderr_chunks.append(chunk)

    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_thread.start()

    try:
        frame_idx = 0
        while n_frames <= 0 or frame_idx < n_frames:
            ok, frame = cap.read()
            if not ok:
                break
            cropped = frame[y:y + height, x:x + width]
            if color_grade:
                cropped = apply_bottom_up_color_correction(
                    cropped, strength=color_grade_strength, layers=color_grade_layers,
                )
            try:
                proc.stdin.write(cropped.tobytes())
            except BrokenPipeError:
                break
            frame_idx += 1
    finally:
        cap.release()
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
        proc.wait()
        stderr_thread.join(timeout=5)

    if proc.returncode != 0:
        stderr = b"".join(stderr_chunks).decode(errors="replace")
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode} while writing {out_path}:\n{stderr}")

    log(f"Cropped {video_path.name} -> {out_path}")
    return out_path


def crop_folder(
    input_folder, output_folder, x: int, y: int, width: int, height: int,
    log=print, on_progress=None, color_grade: bool = False, color_grade_strength: float = 1.0,
    color_grade_layers: list[tuple[float, float]] | None = None,
) -> list[Path]:
    """Crops every video under input_folder into the equivalent relative path under output_folder.
    A video already exactly the target size (at (0,0)) is copied through unchanged rather than
    re-encoded, unless color_grade is set. on_progress(index, total), if given, is called before
    each video."""
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    videos = find_videos(input_folder)
    total = len(videos)
    written = []

    for i, video_path in enumerate(videos, start=1):
        if on_progress:
            on_progress(i, total)

        rel = video_path.relative_to(input_folder)
        out_path = output_folder / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        res = probe_resolution(video_path)
        if res == (width, height) and x == 0 and y == 0 and not color_grade:
            shutil.copy2(video_path, out_path)
            log(f"[{i}/{total}] Already {width}x{height} -- copied {video_path.name} unchanged")
        else:
            log(f"[{i}/{total}] Cropping {video_path.name}...")
            crop_video(video_path, out_path, x, y, width, height, log=log,
                       color_grade=color_grade, color_grade_strength=color_grade_strength,
                       color_grade_layers=color_grade_layers)

        written.append(out_path)

    return written


# --- per-video crop positions ---

def load_positions(positions_path) -> dict:
    positions_path = Path(positions_path)
    if positions_path.exists():
        with open(positions_path) as f:
            return json.load(f)
    return {}


def save_positions(positions_path, positions: dict) -> None:
    positions_path = Path(positions_path)
    positions_path.parent.mkdir(parents=True, exist_ok=True)
    with open(positions_path, "w") as f:
        json.dump(positions, f, indent=2)


def crop_videos_with_positions(
    positions: list[tuple],  # [(video_path, x, y), ...]
    input_folder, output_folder, width: int, height: int,
    log=print, on_progress=None, color_grade: bool = False, color_grade_strength: float = 1.0,
    color_grade_layers: list[tuple[float, float]] | None = None,
) -> list[Path]:
    """Like crop_folder(), but each video gets its own (x, y), for sessions where framing
    shifted between recordings."""
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    total = len(positions)
    written = []

    for i, (video_path, x, y) in enumerate(positions, start=1):
        video_path = Path(video_path)
        if on_progress:
            on_progress(i, total)

        rel = video_path.relative_to(input_folder)
        out_path = output_folder / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        res = probe_resolution(video_path)
        if res == (width, height) and x == 0 and y == 0 and not color_grade:
            shutil.copy2(video_path, out_path)
            log(f"[{i}/{total}] Already {width}x{height} -- copied {video_path.name} unchanged")
        else:
            log(f"[{i}/{total}] Cropping {video_path.name} at ({x},{y})...")
            crop_video(video_path, out_path, x, y, width, height, log=log,
                       color_grade=color_grade, color_grade_strength=color_grade_strength,
                       color_grade_layers=color_grade_layers)

        written.append(out_path)

    return written
