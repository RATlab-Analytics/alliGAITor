"""
Crops videos down to a fixed target size -- for footage recorded at some
resolution other than what a model was trained on (see config.py's
CROP_TARGET_WIDTH/HEIGHT). Deliberately explicit/visual (via a GUI crop
dialog, wired in later) rather than letting the 2D pose model's own
inference-time preprocessing pad or resize oversized frames implicitly,
since a resize (as opposed to a pixel-for-pixel crop) would distort the
pixel-space keypoint coordinates that triangulation depends on.

Ported from RATlab-NOR (github.com/RATlab-Analytics/RATlab-NOR,
video_crop.py) unchanged -- the module was already fully generic (width/
height/x/y are parameters, nothing hardcoded to NOR's own 294x292 crop
size), so alliGAITor's 1280x170 tunnel-strip crop is just a different
config value, not a code change. See config.py's CROP_TARGET_WIDTH/HEIGHT.

No GUI/Qt dependency -- usable from the CLI, tests, or a GUI worker
process alike. Pipes raw frames to ffmpeg directly rather than
cv2.VideoWriter, since cv2.VideoWriter's built-in encoders were confirmed
(in RATlab-NOR) to visibly degrade quality across a full sequence.

alliGAITor-specific note: unlike NOR's roughly-square crop, this is a
wide horizontal strip (1280x170, ~1/10 the source frame height) --
video_crop.py itself needs no changes for that, but when wiring up the
GUI crop dialog later, double-check the crop-rectangle canvas still
renders/drags sanely for a very short, wide box (RATlab-NOR's
CropSetupDialog was built and tested against a much more square crop).
"""

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
# Ground truth from the colleague who actually did this in Photoshop
# (not reverse-engineered from pixel stats): two stacked Brightness/
# Contrast adjustment layers, each set to roughly Brightness -100 /
# Contrast +100 (the first) and Brightness -100 / Contrast +75-100 (the
# second). No colorize/gradient-map/hue-remap step at all -- every
# earlier theory involving a constructed hue gradient or color-balance
# zone shift was wrong (confirmed by testing this exact recipe on
# same-rig side-angle test frames: it reproduces the black background,
# yellow-green rat, and red/green speckle noise closely without any
# hue-remapping step).
#
# Why this alone produces color from a nominally-white/gray rat: PS's
# Brightness/Contrast applies the *same* nonlinear curve independently
# to R, G, and B. At Contrast +100 that curve is steep enough to swing
# from ~0 to ~255 across a narrow input range, so tiny real per-channel
# differences (a faint warm-light color cast, sensor/compression chroma
# noise, real reflectance differences between fur and paw skin) that are
# invisible in the original frame get blown into strongly different
# per-channel outputs -- one channel clips to 255 while another clips to
# 0 -- which is what shows up as saturated, sometimes-inconsistent color.
# Doing it twice compounds that steepness further, which is also why the
# result looks posterized/high-contrast rather than smoothly graded.
#
# _BC_LAYERS is the *full-strength* (100%) recipe -- a plain list so a
# third stacked layer (or different per-layer values) can be added/tuned
# without touching apply_bottom_up_color_correction() itself. Re-tune
# against more side-angle test frames if the vividness looks off once
# this runs on real bottom-up footage.
#
# The bottom camera view has more ambient light than the side-angle test
# footage this was tuned against, so full strength looks starker there
# than intended -- hence `strength` on apply_bottom_up_color_correction()
# below: it linearly scales every layer's brightness/contrast toward 0
# (identity/no-op) at strength=0.0, full recipe at strength=1.0. This is
# what the GUI's strength slider drives (see crop_setup_dialog.py).
#
# Side-angle footage was never run through this originally and should
# stay untouched -- hence this being an opt-in toggle (color_grade=False
# by default everywhere below) rather than always-on.
_BC_LAYERS = [
    (-100, 100),  # (brightness, contrast), Photoshop's -100..100 dialog range
    (-100, 75),
]


def _apply_brightness_contrast(frame: np.ndarray, brightness: float, contrast: float) -> np.ndarray:
    """One Photoshop-style (legacy) Brightness/Contrast adjustment layer,
    applied identically to every channel -- brightness/contrast in
    Photoshop's -100..100 dialog range. Operates in float and does NOT
    clip to uint8 internally, so stacking layers (see _BC_LAYERS) composes
    the way stacked Photoshop adjustment layers do rather than clipping
    prematurely between layers."""
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


def apply_bottom_up_color_correction(frame_bgr: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Stacks the Brightness/Contrast layers in _BC_LAYERS (see module
    docstring above) on a single BGR frame (uint8), clipping to uint8
    only once at the end.

    strength scales every layer's brightness/contrast linearly -- 1.0 is
    the full recipe as documented by the colleague who did this in
    Photoshop, 0.0 is a no-op, values in between fade toward that no-op
    (e.g. for the bottom camera's brighter ambient light needing a less
    stark correction than the side-angle test frames this was tuned
    against)."""
    strength = max(0.0, min(1.0, strength))
    out = frame_bgr.astype(np.float32)
    for brightness, contrast in _BC_LAYERS:
        out = _apply_brightness_contrast(out, brightness * strength, contrast * strength)
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
    """Groups every video under `folder` by (width, height) -- the usual
    first question before cropping: which videos already match the
    target size, and which don't."""
    groups: dict[tuple[int, int], list[Path]] = {}
    for video_path in find_videos(folder):
        res = probe_resolution(video_path)
        if res is None:
            res = (-1, -1)  # unreadable -- grouped together rather than dropped silently
        groups.setdefault(res, []).append(video_path)
    return groups


class CropRegionError(ValueError):
    pass


def crop_video(video_path, out_path, x: int, y: int, width: int, height: int,
                log=print, color_grade: bool = False, color_grade_strength: float = 1.0) -> Path:
    """Crop a single video to the `width`x`height` window starting at
    (x, y), writing to out_path. Raises CropRegionError if that window
    doesn't fit inside the source frame -- never silently clamps, since a
    silently-shifted crop would put keypoints at the wrong pixel
    coordinates without any visible sign something's off.

    color_grade=True applies apply_bottom_up_color_correction() to every
    cropped frame before it's written -- for bottom-up (tunnel) footage
    only. Leave False for side-angle footage, which was never processed
    this way. color_grade_strength (0.0-1.0) scales how strong that
    correction is; ignored when color_grade is False.
    """
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
        # -s must match the CROPPED frame size actually being piped below,
        # not the source video's size -- ffmpeg trusts this blindly for a
        # raw byte stream, it can't infer it from the data itself.
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
                cropped = apply_bottom_up_color_correction(cropped, strength=color_grade_strength)
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
) -> list[Path]:
    """Crops every video under input_folder into the equivalent relative
    path under output_folder. A video already exactly the target size
    (at (0,0)) is copied through as-is rather than re-encoded, to avoid
    a pointless quality-losing round trip through the codec -- unless
    color_grade is set, in which case every video still needs to go
    through crop_video() to actually get color-corrected.

    on_progress(index, total), if given, is called before each video.
    """
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
                       color_grade=color_grade, color_grade_strength=color_grade_strength)

        written.append(out_path)

    return written


# --- per-video crop positions (for a future CropSetupDialog -- the ---
# --- tunnel isn't guaranteed to be framed identically in every video, ---
# --- unlike crop_folder()'s single position applied uniformly) --------

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
) -> list[Path]:
    """Like crop_folder(), but each video gets its own (x, y) -- for
    sessions where the camera/tunnel framing shifted between recordings
    rather than staying fixed for the whole folder."""
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
                       color_grade=color_grade, color_grade_strength=color_grade_strength)

        written.append(out_path)

    return written
