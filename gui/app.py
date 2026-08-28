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

"""
Entry point for the alliGAITor GUI job queue.

Run with:
    python gui/app.py
"""

from __future__ import annotations

import multiprocessing
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

REPO_DIR = Path(__file__).resolve().parent.parent


def _uv_tool_bin_dir() -> Path:
    """Default shim directory `uv tool install` (the standard way to install
    sleap-nn) places executables in, per uv's own precedence: UV_TOOL_BIN_DIR,
    then XDG_BIN_HOME, then XDG_DATA_HOME/../bin, then ~/.local/bin."""
    override = os.environ.get("UV_TOOL_BIN_DIR")
    if override:
        return Path(override)
    xdg_bin = os.environ.get("XDG_BIN_HOME")
    if xdg_bin:
        return Path(xdg_bin)
    xdg_data = os.environ.get("XDG_DATA_HOME")
    if xdg_data:
        return Path(xdg_data).parent / "bin"
    return Path.home() / ".local" / "bin"


def _login_shell_path() -> Optional[str]:
    """A GUI app launched from Finder/Dock (or an AppImage/desktop entry on
    Linux) doesn't inherit the user's shell PATH, only Windows does -- so a
    CLI tool only added to PATH in .zshrc/.bashrc is invisible here even
    though it works fine from a terminal."""
    if sys.platform.startswith("win"):
        return None
    shell = os.environ.get("SHELL", "/bin/zsh")
    try:
        result = subprocess.run(
            [shell, "-ilc", 'echo -n "$PATH"'],
            capture_output=True, text=True, timeout=5,
        )
        lines = result.stdout.strip().splitlines()
        return lines[-1] if lines else None
    except Exception:
        return None


def _fix_path_env() -> None:
    """Merge in the user's real login-shell PATH plus uv's tool-shim directory
    so subprocess calls (sleap-nn, etc.) resolve the same way they do from a
    terminal, regardless of how this app was launched."""
    entries = (os.environ.get("PATH") or "").split(os.pathsep)

    shell_path = _login_shell_path()
    if shell_path:
        entries = shell_path.split(":") + entries

    uv_bin = _uv_tool_bin_dir()
    if uv_bin.is_dir():
        entries = [str(uv_bin)] + entries

    merged = list(dict.fromkeys(p for p in entries if p))
    os.environ["PATH"] = os.pathsep.join(merged)

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
    _fix_path_env()
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
