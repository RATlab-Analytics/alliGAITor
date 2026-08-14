"""Check whether the calibration board is ever detected simultaneously by two
or more cameras -- which aniposelib's CameraGroup.calibrate_videos() requires
to link every camera pair into one solvable calibration graph.

A "0 boards detected"-per-camera problem is different from this one: a camera
can individually detect the board on plenty of frames and calibration can
still fail with "Could not build calibration graph" if none of those
detections line up with another camera's detections at the same frame index
(aniposelib matches by exact frame number, not by wall-clock time).

Usage:

    python3 tools/calibration/check_multicam_sync.py \\
        --left videos/.../coded_fixed/calibration_cam0_fixed.mp4 \\
        --right videos/.../coded_fixed/calibration_cam1_fixed.mp4 \\
        --bottom videos/.../coded_fixed/calibration_cam2_fixed.mp4

Runs every frame (skip=1) by default -- slower than aniposelib's own
skip=20 default, but this is a one-time diagnostic, not the real
calibration run, and skip=20 would make an already-rare overlap even less
likely to be observed.
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import cv2

from alligaitor.calibration import build_board

MIN_CORNERS = 8  # matches aniposelib.cameras.CameraGroup.calibrate_rows' own filter


def scan(video_path: Path, board, skip: int) -> dict:
    """Return {framenum: n_corners} for every frame with >=1 detected corner."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open {video_path}")

    hits = {}
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % skip == 0:
            corners, ids = board.detect_image(frame)
            n = 0 if ids is None else len(ids)
            if n > 0:
                hits[frame_idx] = n
        frame_idx += 1
    cap.release()
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--bottom", type=Path, required=True)
    parser.add_argument("--skip", type=int, default=1)
    parser.add_argument(
        "--tolerance",
        type=int,
        default=10,
        help="Also check for near-miss overlaps within +/-N frame indices, in case the "
        "cameras' frame counters drifted apart during capture (they're independent, "
        "non-genlocked v4l2 threads, so this is plausible even with similar total "
        "frame counts). Exact-match results (tolerance=0) are always shown too.",
    )
    args = parser.parse_args()

    board = build_board()
    videos = {"left": args.left, "right": args.right, "bottom": args.bottom}

    hits = {}
    for role, path in videos.items():
        print(f"scanning {role}: {path}")
        hits[role] = scan(path, board, args.skip)
        n_strong = sum(1 for n in hits[role].values() if n >= MIN_CORNERS)
        print(f"  {len(hits[role])} frames with any corners, {n_strong} with >={MIN_CORNERS} (aniposelib's own bar)\n")

    print("=== pairwise simultaneous-detection overlap (exact frame-index match) ===")
    for a, b in combinations(videos.keys(), 2):
        frames_a_any = set(hits[a].keys())
        frames_b_any = set(hits[b].keys())
        overlap_any = frames_a_any & frames_b_any

        frames_a_strong = {f for f, n in hits[a].items() if n >= MIN_CORNERS}
        frames_b_strong = {f for f, n in hits[b].items() if n >= MIN_CORNERS}
        overlap_strong = frames_a_strong & frames_b_strong

        print(f"{a} & {b}: {len(overlap_any)} frames with any simultaneous corners, "
              f"{len(overlap_strong)} with >={MIN_CORNERS} in both (this is what calibration actually needs)")

    print("\nIf every pair above shows 0 for the >=8-corner overlap, that's exactly why calibration")
    print("can't build a graph: each camera is detecting the board fine on its own, just never at")
    print("the same moment another camera also gets a strong detection.")

    if args.tolerance > 0:
        print(f"\n=== near-miss check: within +/-{args.tolerance} frame indices (possible camera drift) ===")
        for a, b in combinations(videos.keys(), 2):
            frames_a_strong = sorted(f for f, n in hits[a].items() if n >= MIN_CORNERS)
            frames_b_strong = sorted(f for f, n in hits[b].items() if n >= MIN_CORNERS)

            if not frames_a_strong or not frames_b_strong:
                print(f"{a} & {b}: no >={MIN_CORNERS}-corner frames on at least one side, nothing to compare")
                continue

            near_matches = []
            for fa in frames_a_strong:
                best_offset = min((fb - fa for fb in frames_b_strong), key=abs)
                if abs(best_offset) <= args.tolerance:
                    near_matches.append(best_offset)

            if near_matches:
                offsets = sorted(near_matches)
                print(f"{a} & {b}: {len(near_matches)} near-miss matches within tolerance, "
                      f"offsets range {offsets[0]:+d} to {offsets[-1]:+d} frames "
                      f"(median {offsets[len(offsets)//2]:+d}) -- a consistent non-zero median here "
                      f"means the cameras are drifting apart by roughly that many frames, not that "
                      f"the board was never simultaneously visible.")
            else:
                print(f"{a} & {b}: no matches even within +/-{args.tolerance} frames -- "
                      f"genuinely no simultaneous strong detections, not just an indexing offset.")


if __name__ == "__main__":
    main()
