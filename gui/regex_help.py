"""
A small collapsible "How regex works" panel, shown next to the id/camera
regex fields in both the Preferences dialog (default regexes) and the
group config editor (per-job regexes) -- these fields are aimed at lab
members setting up a rig, not necessarily anyone who's written a regex
before, so a plain-language reference belongs right where they're typed,
not just in a README.
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
    current content.

    A plain ``window.adjustSize()`` doesn't shrink a window back down
    after a child widget's visibility changes made its content smaller --
    Qt only auto-*grows* a shown top-level widget to satisfy a layout
    change, never auto-shrinks it. This forces every layout between
    ``widget`` and the top-level window to actually recompute
    (``updateGeometry()`` + ``layout().activate()``, walking up the
    parent chain) before reading a fresh ``sizeHint()`` back off the
    window and resizing to it.

    This is *not* sufficient on its own when ``widget`` sits inside a
    QTabWidget: QTabWidget's internal QStackedWidget is documented to
    size itself for the largest page it's ever shown, not the current
    one, and that has nothing to do with cached/stale sizeHints -- it's
    the actual intended behavior, so no amount of invalidating layouts
    changes what it reports. A tabbed container needs to bypass
    QStackedWidget's own sizeHint entirely (see
    ``_PreferencesDialog._sync_dialog_size`` in main_window.py for how);
    this function alone only handles the non-tabbed case (e.g.
    group_config_dialog.py's scroll area).
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

    ``on_toggled``, if given, is called (with no arguments) after the
    panel's own visibility change -- for an embedder where
    :func:`shrink_window_to_fit` isn't enough on its own (a QTabWidget
    page; see its docstring), so the embedder can run its own resize
    logic instead of the default. When omitted, this panel resizes its
    own top-level window via :func:`shrink_window_to_fit`.
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
    help_label.setStyleSheet("color: #444; background: #f5f5f5; padding: 8px; border-radius: 4px;")
    help_label.setVisible(False)

    def _on_toggled(checked: bool):
        toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        help_label.setVisible(checked)
        # Deferred one event-loop tick so this runs after the show/hide
        # has actually been processed, not before.
        if on_toggled is not None:
            QTimer.singleShot(0, on_toggled)
        else:
            QTimer.singleShot(0, lambda: shrink_window_to_fit(container))

    toggle.toggled.connect(_on_toggled)

    layout.addWidget(toggle)
    layout.addWidget(help_label)
    return container
