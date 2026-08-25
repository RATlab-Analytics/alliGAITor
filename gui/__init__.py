"""alliGAITor's GUI job queue."""

from __future__ import annotations

import sys
from pathlib import Path

# The gui/ modules themselves use flat imports among each other (e.g.
# main_window.py does `from job_queue import Job`, not
# `from gui.job_queue import Job`) -- same convention tools/ already uses
# among its own modules -- so gui/ itself needs to be on sys.path, not
# just importable as the `gui` package. tools/ (crop_setup_dialog.py,
# video_crop.py, frame_utils.py, ...) is added for the same reason: it
# has no __init__.py and its modules import each other flatly too. The
# repo root is added so `import alligaitor` works regardless of the
# current working directory the app was launched from. Done once here,
# at gui package import time, rather than in every module that needs it.
REPO_DIR = Path(__file__).resolve().parent.parent
for _p in (REPO_DIR, REPO_DIR / "gui", REPO_DIR / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
