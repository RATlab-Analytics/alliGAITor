"""Validate ChArUco board detection on a short test clip before committing to
a full calibration recording.

Runs every frame (not a sparse sample) through the same board-detection code
path used by ``alligaitor.calibration.calibrate``, reports the detection
rate, and optionally saves annotated frames so you can visually confirm
what's being detected (or why it isn't).

Usage:

    python3 tools/calibration/check_board_detection.py path/to/test_clip.mp4

    # Try a different ArUco dictionary size (default matches
    # alligaitor.calibration.build_board(), i.e. dict_size=50):
    python3 tools/calibration/check_board_detection.py path/to/test_clip.mp4 --dict-size 1000

    # Save annotated frames (both hits and misses) for visual review. Default
    # --out-dir lands under videos/diagnostics/, which is gitignored -- pass
    # your own --out-dir if you want it somewhere else:
    python3 tools/calibration/check_board_detection.py path/to/test_clip.mp4 --save-frames 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
from aniposelib.boards import CharucoBoard

from alligaitor.calibration import BOARD_MARKER_LENGTH_MM, BOARD_SQUARES_X, BOARD_SQUARES_Y, BOARD_SQUARE_LENGTH_MM

try:
    # Use the real pipeline's board builder (includes the checkMarkers=False
    # fix) when the dictionary size wasn't overridden, so this script
    # reflects exactly what `alligaitor.calibration.calibrate` will do.
    from alligaitor.calibration import build_board as _build_default_board
except ImportError:  # pragma: no cover
    _build_default_board = None


def sharpness(gray: np.ndarray) -> float:
    """Laplacian-variance sharpness score; low values indicate blur/noise-only content."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path, help="Test clip to check board detection on.")
    parser.add_argument("--dict-size", type=int, default=50, choices=[50, 100, 250, 1000])
    parser.add_argument("--skip", type=int, default=1, help="Process every Nth frame (default: every frame).")
    parser.add_argument("--save-frames", type=int, default=0, help="Save up to N annotated frames (hits and misses).")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "videos" / "diagnostics" / "board_check",
        help="Where to save annotated frames (only used with --save-frames). Defaults "
        "under videos/diagnostics/, which is gitignored, so diagnostic images never "
        "end up staged for a commit.",
    )
    args = parser.parse_args()

    if args.dict_size == 50 and _build_default_board is not None:
        board = _build_default_board()
    else:
        board = CharucoBoard(
            squaresX=BOARD_SQUARES_X,
            squaresY=BOARD_SQUARES_Y,
            square_length=BOARD_SQUARE_LENGTH_MM,
            marker_length=BOARD_MARKER_LENGTH_MM,
            dict_size=args.dict_size,
        )
        # Same fix as alligaitor.calibration.build_board(): OpenCV's
        # checkMarkers geometric-consistency check (on by default) was
        # found to reject every real, correctly-decoded marker on this
        # footage before corner interpolation ever ran.
        charuco_params = board.charuco_detector.getCharucoParameters()
        charuco_params.checkMarkers = False
        board.charuco_detector.setCharucoParameters(charuco_params)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"Could not open {args.video}")
        raise SystemExit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"video: {args.video}")
    print(f"total frames: {total_frames}, processing every {args.skip} frame(s)")
    print(f"board: {BOARD_SQUARES_X}x{BOARD_SQUARES_Y} squares, dict_size={args.dict_size}")

    if args.save_frames:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    n_processed = 0
    n_with_corners = 0
    corner_counts = []
    sharpness_scores = []
    saved = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % args.skip != 0:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids = board.detect_image(frame)
        n = 0 if (corners is None or ids is None) else len(ids)

        n_processed += 1
        if n > 0:
            n_with_corners += 1
            corner_counts.append(n)
            sharpness_scores.append(sharpness(gray))

        if args.save_frames and saved < args.save_frames and (n > 0 or frame_idx % (args.skip * 30) == 0):
            annotated = frame.copy()
            if n > 0:
                # OpenCV 5.0's CharucoDetector.detectBoard() returns corners/ids
                # squeezed to (N,2)/(N,) instead of the (N,1,2)/(N,1) shape
                # drawDetectedCornersCharuco's internal total()-count assertion
                # still expects -- without this reshape it throws even on a
                # clean, fully-visible board. Harmless no-op on older OpenCV.
                corners_draw = np.asarray(corners).reshape(-1, 1, 2).astype(np.float32)
                ids_draw = np.asarray(ids).reshape(-1, 1).astype(np.int32)
                cv2.aruco.drawDetectedCornersCharuco(annotated, corners_draw, ids_draw)
            tag = "hit" if n > 0 else "miss"
            out_path = args.out_dir / f"frame{frame_idx:06d}_{tag}_{n}corners.png"
            cv2.imwrite(str(out_path), annotated)
            saved += 1

        frame_idx += 1

    cap.release()

    print(f"\nframes processed: {n_processed}")
    print(f"frames with >=1 corner detected: {n_with_corners} ({n_with_corners / max(n_processed, 1):.1%})")
    if corner_counts:
        max_possible = (BOARD_SQUARES_X - 1) * (BOARD_SQUARES_Y - 1)
        print(f"corners per successful frame: mean={np.mean(corner_counts):.1f}, max={np.max(corner_counts)} (board max: {max_possible})")
        print(f"sharpness on successful frames: mean={np.mean(sharpness_scores):.0f}")
    else:
        print("no frames had any detected corners.")

    if args.save_frames:
        print(f"\nannotated frames saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
