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
A small collapsible "How regex works" panel, shown next to the id/camera
regex fields in the Preferences dialog and the group config editor.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QLabel, QToolButton, QVBoxLayout, QWidget

_HELP_TEXT = """\
<p>Both fields are regular expressions, applied to each video's <b>filename</b>. \
The part in parentheses <b>(...)</b> -- the "capture group" -- is what gets used \
as the session name or camera token; everything outside the parentheses just has \
to match, but isn't kept.</p>
<p><b>Common pieces:</b></p>
<ul>
<li><code>(...)</code> &mdash; the one required capture group</li>
<li><code>.+?</code> &mdash; any characters, as few as possible (stop at the next thing that matches)</li>
<li><code>\\d+</code> &mdash; one or more digits</li>
<li><code>^</code> &mdash; start of the filename</li>
<li>a plain character like <code>_</code> just matches that literal character</li>
</ul>
<p><b>Example</b> &mdash; filename <code>359a-BL_cam0_coded.mp4</code>:</p>
<ul>
<li>ID regex <code>^(.+?)_cam\\d+</code> captures <code>359a-BL</code> (the session name)</li>
<li>Camera regex <code>_(cam\\d+)</code> captures <code>cam0</code> (the camera token)</li>
</ul>
<p>Every video from the same recording should produce the same session name, and \
each of a session's three videos should produce a different camera token.</p>
"""


def shrink_window_to_fit(widget: QWidget) -> None:
    """Force ``widget``'s top-level window to resize down to fit its
    current content. Qt only auto-grows a shown top-level widget on a
    layout change, never auto-shrinks it, so this walks up the parent
    chain recomputing layouts before resizing to a fresh sizeHint.

    Not sufficient when ``widget`` sits inside a QTabWidget, since its
    internal QStackedWidget sizes itself for the largest page it's ever
    shown; see ``_PreferencesDialog._sync_dialog_size`` in main_window.py
    for that case.
    """
    window = widget.window()
    if window is None:
        return
    w = widget
    while w is not None:
        w.updateGeometry()
        if w.layout() is not None:
            w.layout().activate()
        if w is window:
            break
        w = w.parentWidget()
    if window.layout() is not None:
        window.resize(window.layout().sizeHint())
    else:
        window.adjustSize()


def build_regex_help_panel(parent=None, on_toggled=None) -> QWidget:
    """Returns a widget with a collapsed-by-default toggle button; add it
    to a layout right after the regex fields it explains.

    ``on_toggled``, if given, is called after the panel's visibility
    change so the embedder can run its own resize logic (needed inside a
    QTabWidget page). Otherwise defaults to :func:`shrink_window_to_fit`.
    """
    container = QWidget(parent)
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)

    toggle = QToolButton()
    toggle.setText("How regex works")
    toggle.setCheckable(True)
    toggle.setChecked(False)
    toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
    toggle.setArrowType(Qt.RightArrow)

    help_label = QLabel(_HELP_TEXT)
    help_label.setWordWrap(True)
    help_label.setTextFormat(Qt.RichText)
    help_label.setStyleSheet("color: #d4d4d4; background: #353535; padding: 8px; border-radius: 4px;")
    help_label.setVisible(False)

    def _on_toggled(checked: bool):
        toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        help_label.setVisible(checked)
        # Deferred one tick so this runs after the show/hide is processed.
        if on_toggled is not None:
            QTimer.singleShot(0, on_toggled)
        else:
            QTimer.singleShot(0, lambda: shrink_window_to_fit(container))

    toggle.toggled.connect(_on_toggled)

    layout.addWidget(toggle)
    layout.addWidget(help_label)
    return container
