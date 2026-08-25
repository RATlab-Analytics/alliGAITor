"""Converts a line of ANSI-colored terminal text into HTML, so a Qt rich
text widget can render it with (approximately) the same colors the
terminal would have -- e.g. sleap-nn's tqdm progress bar, which colors
the bar fill and percentage.

Handles the SGI (Select Graphic Rendition, the ``...m``-terminated CSI
subset that carries color/style rather than cursor movement) parameters
tqdm/rich-based CLIs actually use: the standard and bright 16-color
palette, 256-color, 24-bit truecolor, bold/italic/underline, and reverse
video. Any other CSI sequence (cursor movement, line clearing, show/hide
cursor, and so on) is stripped before the SGI codes are even parsed --
meaningless once redrawn as static HTML rather than replayed into a real
terminal, and left in place they'd otherwise show up as literal garbage
text exactly the way an unstripped ``\\x1b[?25h`` cursor-show code once
did in the plain-text log (see alligaitor.subprocess_streaming's own,
narrower fix for that same class of bug).

Whitespace is preserved with ``&nbsp;`` rather than relying on CSS
``white-space: pre`` support, since a progress bar's alignment (spaces
between the percentage, the bar, and the counts) has to survive however
the HTML ends up rendered.
"""

from __future__ import annotations

import html
import re
from typing import Dict, List, Optional

# Matches only SGI sequences (color/style, "...m") -- run against the
# input *after* _NON_SGI_CSI_RE below has already removed every other
# kind of CSI sequence, so this never has to tell them apart itself.
_SGI_RE = re.compile(r"\x1b\[([0-9;]*)m")

# Any CSI sequence that ISN'T an SGI one -- same [0-?]*[ -/]* parameter/
# intermediate-byte range as alligaitor.subprocess_streaming's own
# _ANSI_ESCAPE_RE, but with the final byte restricted to everything in
# '@'-'~' *except* 'm' (['@'-'l''n'-'~'], since 'm' sits alphabetically
# between them), so SGI sequences are left for _SGI_RE to parse instead
# of being swallowed here too. This runs first specifically because a
# single buffered progress-bar redraw can contain both kinds mixed
# together (e.g. tqdm wrapping a color code around a `\x1b[?25h`
# cursor-show it emits mid-line) -- subprocess_streaming's own stripper
# only ever sees the fully-cleaned *plain* copy it keeps for
# deduplication, never the raw, still-ANSI-coded buffer this module
# actually receives.
_NON_SGI_CSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-ln-~]")

# Approximate xterm/VS-Code-terminal palette -- close enough to whatever
# the real terminal would have shown that colors are still
# distinguishable and read as "the same bar", without claiming to be
# any particular terminal's exact values.
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
    """Current text attributes as SGI codes accumulate -- reused as-is
    across an entire line, since a redrawing progress bar typically only
    resets/recolors partway through, not at the very start."""

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
    fragment suitable for ``QTextCursor.insertHtml()``: color/bold/
    italic/underline/reverse-video are rendered as inline-styled
    ``<span>`` elements, literal text is HTML-escaped, and spaces are
    turned into ``&nbsp;`` so alignment survives regardless of the rich
    text engine's ``white-space`` support. Every CSI sequence that isn't
    a color/style one (cursor movement, line clear, show/hide cursor,
    ...) is stripped first -- see ``_NON_SGI_CSI_RE`` -- so it never
    reaches the parser below and can't leak through as literal text.
    """
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
