"""
A small collapsible "How regex works" panel, shown next to the id/camera
regex fields in both the Preferences dialog (default regexes) and the
group config editor (per-job regexes) -- these fields are aimed at lab
members setting up a rig, not necessarily anyone who's written a regex
before, so a plain-language reference belongs right where they're typed,
not just in a README.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
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


def build_regex_help_panel(parent=None) -> QWidget:
    """Returns a widget with a collapsed-by-default toggle button; add it
    to a layout right after the regex fields it explains."""
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
    help_label.setStyleSheet("color: #444; background: #f5f5f5; padding: 8px; border-radius: 4px;")
    help_label.setVisible(False)

    def _on_toggled(checked: bool):
        toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        help_label.setVisible(checked)

    toggle.toggled.connect(_on_toggled)

    layout.addWidget(toggle)
    layout.addWidget(help_label)
    return container
