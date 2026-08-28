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

"""Runs video_crop.crop_folder() (or the per-position variant) in a separate OS process, since
OpenCV video I/O alongside an active Qt event loop off the main thread risks a macOS segfault,
and a full-folder crop can take too long to run synchronously on the main thread.

Reports progress via a multiprocessing.Queue of plain-data tuples, and checks a stop_event
between videos for a clean "finish current video, then stop" cancel."""

from __future__ import annotations

import sys
from pathlib import Path


def run_crop_worker(tools_dir: str, input_folder: str, output_folder: str,
                     x: int, y: int, width: int, height: int, queue, stop_event,
                     color_grade: bool = False, color_grade_strength: float = 1.0,
                     color_grade_layers=None):
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)

    import video_crop as vc

    def log(msg):
        queue.put(("log", msg))

    videos = vc.find_videos(input_folder)
    total = len(videos)
    if total == 0:
        log(f"No .mp4 files found under {input_folder}")
        queue.put(("finished", "done", ""))
        return

    grade_note = ", with bottom-up color correction" if color_grade else ""
    log(f"Cropping {total} video(s) to {width}x{height} at ({x},{y}){grade_note}...")

    def on_progress(i, t):
        queue.put(("progress", i, t))

    written = 0
    for i, video_path in enumerate(vc.find_videos(input_folder), start=1):
        if stop_event.is_set():
            log(f"Stopping after {written}/{total} video(s) (Stop was clicked).")
            queue.put(("finished", "canceled", f"Stopped after {written}/{total} video(s)."))
            return

        rel = video_path.relative_to(Path(input_folder))
        out_path = Path(output_folder) / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            res = vc.probe_resolution(video_path)
            if res == (width, height) and x == 0 and y == 0 and not color_grade:
                import shutil
                shutil.copy2(video_path, out_path)
                log(f"[{i}/{total}] Already {width}x{height} -- copied {video_path.name} unchanged")
            else:
                log(f"[{i}/{total}] Cropping {video_path.name}...")
                vc.crop_video(video_path, out_path, x, y, width, height, log=log,
                               color_grade=color_grade, color_grade_strength=color_grade_strength,
                               color_grade_layers=color_grade_layers)
        except Exception as exc:
            log(f"  ERROR on {video_path.name}: {exc}")
            queue.put(("finished", "failed", f"{video_path.name}: {exc}"))
            return

        written += 1
        queue.put(("progress", i, total))

    log(f"Done. Cropped {written}/{total} video(s) into {output_folder}")
    queue.put(("finished", "done", ""))


def run_crop_worker_positions(tools_dir: str, input_folder: str, output_folder: str,
                               positions: list, width: int, height: int, queue, stop_event,
                               color_grade: bool = False, color_grade_strength: float = 1.0,
                               color_grade_layers=None):
    """Like run_crop_worker(), but each video has its own (x, y): `positions` is a plain list of
    (video_path_str, x, y) tuples, pickle-safe across the process boundary. Emits
    ("video_done", video_path_str) after each successful crop so the GUI can persist that video's
    position incrementally, rather than only finding out all-or-nothing at the end."""
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)

    import shutil
    import video_crop as vc

    def log(msg):
        queue.put(("log", msg))

    total = len(positions)
    if total == 0:
        queue.put(("finished", "done", ""))
        return

    for i, (video_path_str, x, y) in enumerate(positions, start=1):
        if stop_event.is_set():
            log(f"Stopping after {i - 1}/{total} video(s) (Stop was clicked).")
            queue.put(("finished", "canceled", f"Stopped after {i - 1}/{total} video(s)."))
            return

        video_path = Path(video_path_str)
        rel = video_path.relative_to(Path(input_folder))
        out_path = Path(output_folder) / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            res = vc.probe_resolution(video_path)
            if res == (width, height) and x == 0 and y == 0 and not color_grade:
                shutil.copy2(video_path, out_path)
                log(f"[{i}/{total}] Already {width}x{height} -- copied {video_path.name} unchanged")
            else:
                log(f"[{i}/{total}] Cropping {video_path.name} at ({x},{y})...")
                vc.crop_video(video_path, out_path, x, y, width, height, log=log,
                               color_grade=color_grade, color_grade_strength=color_grade_strength,
                               color_grade_layers=color_grade_layers)
        except Exception as exc:
            log(f"  ERROR on {video_path.name}: {exc}")
            queue.put(("finished", "failed", f"{video_path.name}: {exc}"))
            return

        queue.put(("video_done", video_path_str))
        queue.put(("progress", i, total))

    log(f"Done. Cropped {total}/{total} video(s) into {output_folder}")
    queue.put(("finished", "done", ""))
