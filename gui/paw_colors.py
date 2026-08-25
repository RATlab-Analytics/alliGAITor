"""
Shared per-paw identity colors for the validation-viewing dialogs --
distinct from the red/green *usability* colors used elsewhere (see
job_table_model.py's _STATUS_COLORS), since a paw's own color needs to
stay recognizable across the validation list, the scrub bar's rows, and
the Flag Paw(s) popup regardless of whether that paw is currently usable.

Kept in its own module (rather than duplicated in validation_list_dialog.py
and validation_video_dialog.py) so the two dialogs can't drift apart on
which color means which paw.
"""

from __future__ import annotations

from PySide6.QtGui import QColor

from alligaitor.gait import PAW_NODES

PAW_SHORT_LABELS = {
    "left-forepaw": "LF",
    "right-forepaw": "RF",
    "left-hind-paw": "LH",
    "right-hind-paw": "RH",
}

_PAW_COLORS = {
    "left-forepaw": QColor(255, 167, 38),    # orange
    "right-forepaw": QColor(66, 165, 245),   # blue
    "left-hind-paw": QColor(236, 64, 122),   # pink/magenta
    "right-hind-paw": QColor(38, 198, 218),  # teal/cyan
}

# Semantic usable/unusable colors, matching job_table_model.py's
# _STATUS_COLORS (JobStatus.DONE / JobStatus.FAILED) so "green means good,
# red means bad" reads the same way everywhere in the app.
COLOR_USABLE = QColor("#81c995")
COLOR_UNUSABLE = QColor("#f28b82")


def paw_color(paw: str) -> QColor:
    return _PAW_COLORS[paw]


def grayed_paw_color(paw: str) -> QColor:
    """A desaturated/lightened blend of `paw`'s own color, used on the
    scrub bar for an unusable paw's fallback (longest-raw-run) window --
    muted enough to read as "not trustworthy" while staying identifiable
    as this specific paw rather than collapsing every unusable paw into
    one indistinguishable gray."""
    c = _PAW_COLORS[paw]
    gray = 140
    blend = 0.55  # fraction gray
    return QColor(
        round(c.red() * (1 - blend) + gray * blend),
        round(c.green() * (1 - blend) + gray * blend),
        round(c.blue() * (1 - blend) + gray * blend),
    )


def ordered_paws():
    return list(PAW_NODES)
