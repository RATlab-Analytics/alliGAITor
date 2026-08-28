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

"""Converts a line of ANSI-colored terminal text into HTML for a Qt rich
text widget, approximating the terminal's colors (e.g. tqdm progress bars).

Handles SGI (color/style) CSI sequences: 16/256-color, truecolor, bold/
italic/underline, reverse video. Other CSI sequences (cursor movement,
line clearing, etc.) are stripped first. Spaces are rendered as
``&nbsp;`` so progress-bar alignment survives regardless of the widget's
``white-space`` handling.
"""

from __future__ import annotations

import html
import re
from typing import Dict, List, Optional

# Matches SGI (color/style) CSI sequences only; run after _NON_SGI_CSI_RE
# has stripped every other CSI sequence.
_SGI_RE = re.compile(r"\x1b\[([0-9;]*)m")

# Matches any CSI sequence except SGI ones, so those are left for _SGI_RE.
_NON_SGI_CSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-ln-~]")

# Approximate xterm/VS-Code palette.
_PALETTE_16: Dict[int, str] = {
    0: "#000000", 1: "#cd3131", 2: "#0dbc79", 3: "#e5e510",
    4: "#2472c8", 5: "#bc3fbc", 6: "#11a8cd", 7: "#e5e5e5",
    8: "#666666", 9: "#f14c4c", 10: "#23d18b", 11: "#f5f543",
    12: "#3b8eea", 13: "#d670d6", 14: "#29b8db", 15: "#e5e5e5",
}


def _ansi_256_to_hex(n: int) -> str:
    if n < 16:
        return _PALETTE_16[n]
    if n < 232:
        n -= 16
        r, g, b = n // 36, (n // 6) % 6, n % 6
        scale = lambda v: 0 if v == 0 else 55 + v * 40
        return f"#{scale(r):02x}{scale(g):02x}{scale(b):02x}"
    gray = 8 + (n - 232) * 10
    return f"#{gray:02x}{gray:02x}{gray:02x}"


class _SgiState:
    """Current text attributes as SGI codes accumulate across a line."""

    def __init__(self):
        self.fg: Optional[str] = None
        self.bg: Optional[str] = None
        self.bold = False
        self.italic = False
        self.underline = False
        self.reverse = False

    def apply(self, codes: List[str]) -> None:
        i = 0
        while i < len(codes):
            raw = codes[i]
            n = int(raw) if raw else 0
            if n == 0:
                self.__init__()
            elif n == 1:
                self.bold = True
            elif n == 3:
                self.italic = True
            elif n == 4:
                self.underline = True
            elif n == 7:
                self.reverse = True
            elif n == 22:
                self.bold = False
            elif n == 23:
                self.italic = False
            elif n == 24:
                self.underline = False
            elif n == 27:
                self.reverse = False
            elif n == 39:
                self.fg = None
            elif n == 49:
                self.bg = None
            elif 30 <= n <= 37:
                self.fg = _PALETTE_16[n - 30]
            elif 90 <= n <= 97:
                self.fg = _PALETTE_16[8 + (n - 90)]
            elif 40 <= n <= 47:
                self.bg = _PALETTE_16[n - 40]
            elif 100 <= n <= 107:
                self.bg = _PALETTE_16[8 + (n - 100)]
            elif n in (38, 48) and i + 1 < len(codes):
                target = "fg" if n == 38 else "bg"
                mode = codes[i + 1]
                if mode == "5" and i + 2 < len(codes):
                    setattr(self, target, _ansi_256_to_hex(int(codes[i + 2])))
                    i += 2
                elif mode == "2" and i + 4 < len(codes):
                    r, g, b = int(codes[i + 2]), int(codes[i + 3]), int(codes[i + 4])
                    setattr(self, target, f"#{r:02x}{g:02x}{b:02x}")
                    i += 4
            i += 1

    def css(self) -> str:
        fg, bg = (self.bg, self.fg) if self.reverse else (self.fg, self.bg)
        styles = []
        if fg:
            styles.append(f"color:{fg}")
        if bg:
            styles.append(f"background-color:{bg}")
        if self.bold:
            styles.append("font-weight:bold")
        if self.italic:
            styles.append("font-style:italic")
        if self.underline:
            styles.append("text-decoration:underline")
        return ";".join(styles)


def _escape_preserving_spaces(text: str) -> str:
    return html.escape(text).replace(" ", "&nbsp;")


def ansi_line_to_html(raw: str) -> str:
    """Converts one line of (possibly) ANSI-colored text to an HTML
    fragment for ``QTextCursor.insertHtml()``. Color/style are rendered
    as inline-styled ``<span>`` elements; other CSI sequences are stripped."""
    raw = _NON_SGI_CSI_RE.sub("", raw)
    state = _SgiState()
    parts: List[str] = []
    pos = 0

    for m in _SGI_RE.finditer(raw):
        text = raw[pos:m.start()]
        if text:
            css = state.css()
            if css:
                parts.append(f'<span style="{css}">')
                parts.append(_escape_preserving_spaces(text))
                parts.append("</span>")
            else:
                parts.append(_escape_preserving_spaces(text))
        codes = m.group(1).split(";") if m.group(1) else ["0"]
        state.apply(codes)
        pos = m.end()

    tail = raw[pos:]
    if tail:
        css = state.css()
        if css:
            parts.append(f'<span style="{css}">')
            parts.append(_escape_preserving_spaces(tail))
            parts.append("</span>")
        else:
            parts.append(_escape_preserving_spaces(tail))

    return "".join(parts)
