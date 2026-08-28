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

# Ensure repo root and tools/ are on sys.path in case this runs before
# gui/__init__.py's own setup (e.g. `python gui/app.py` directly).
for _p in (REPO_DIR, REPO_DIR / "gui", REPO_DIR / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

import app_settings
from dark_theme import apply_dark_theme
from job_queue import JobQueue, default_app_data_dir, default_models_dir
from main_window import MainWindow


def _icon_path() -> Path:
    base = Path(getattr(sys, "_MEIPASS", REPO_DIR))
    return base / "packaging" / "icons" / "alligaitor_256.png"


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("alliGAITor")
    app.setApplicationDisplayName("alliGAITor")
    icon_path = _icon_path()
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    apply_dark_theme(app)

    app_data_dir = default_app_data_dir()
    job_queue = JobQueue(app_data_dir).load()
    models_dir = app_settings.get_models_dir(app_data_dir) or default_models_dir(REPO_DIR)

    window = MainWindow(job_queue=job_queue, repo_dir=REPO_DIR, models_dir=models_dir)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    # Needed for multiprocessing once frozen into a .app/.exe/AppImage.
    multiprocessing.freeze_support()
    main()
