"""
Entry point for the alliGAITor GUI job queue.

Run with:
    python gui/app.py
"""

from __future__ import annotations

import multiprocessing
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent

# Run directly (`python gui/app.py`) rather than as `python -m gui.app`,
# so gui/__init__.py's sys.path setup hasn't necessarily executed yet --
# Python only auto-adds this script's own directory (gui/) to sys.path,
# not the repo root or tools/ that job_queue.py/main_window.py and
# friends need for their own flat imports. Set up the same three
# directories here explicitly; the inserts in gui/__init__.py are
# idempotent no-ops once this has already run.
for _p in (REPO_DIR, REPO_DIR / "gui", REPO_DIR / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from PySide6.QtWidgets import QApplication

from dark_theme import apply_dark_theme
from job_queue import JobQueue, default_app_data_dir
from main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("alliGAITor")
    app.setApplicationDisplayName("alliGAITor")
    apply_dark_theme(app)

    app_data_dir = default_app_data_dir(REPO_DIR)
    job_queue = JobQueue(app_data_dir).load()

    window = MainWindow(job_queue=job_queue, repo_dir=REPO_DIR)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    # Required for multiprocessing (used by the batch runner and the
    # crop tool's bulk-crop runner) to behave correctly once this app is
    # ever frozen into a .app/.exe/AppImage -- must run immediately under
    # this guard, before anything else spawns a process. Harmless as a
    # no-op unfrozen.
    multiprocessing.freeze_support()
    main()
