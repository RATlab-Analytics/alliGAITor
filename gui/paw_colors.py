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
Shared per-paw identity colors for the validation dialogs, distinct from
the red/green usability colors in job_table_model.py.
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

# Matches job_table_model.py's _STATUS_COLORS for consistent usable/unusable coloring.
COLOR_USABLE = QColor("#81c995")
COLOR_UNUSABLE = QColor("#f28b82")
# Usable but relying heavily on the bottom-camera fallback; only ever
# layered on top of a usable paw, never in place of COLOR_UNUSABLE.
COLOR_FALLBACK_WARNING = QColor("#ffca28")


def paw_color(paw: str) -> QColor:
    return _PAW_COLORS[paw]


def grayed_paw_color(paw: str) -> QColor:
    """Desaturated blend of `paw`'s color, for an unusable paw's fallback window on the scrub bar."""
    c = _PAW_COLORS[paw]
    gray = 120
    blend = 0.85  # fraction gray
    return QColor(
        round(c.red() * (1 - blend) + gray * blend),
        round(c.green() * (1 - blend) + gray * blend),
        round(c.blue() * (1 - blend) + gray * blend),
    )


def ordered_paws():
    return list(PAW_NODES)
