"""Runs a subprocess while forwarding its live tqdm-style progress output as it happens.

Distinguishes a redrawing progress line (``'\\r'``-terminated) from ordinary ``'\\n'``-terminated
log messages, so a caller can display the progress line as "the same line updating" separately
from one-off log messages.
"""

from __future__ import annotations

import io
import os
import re
import subprocess
import sys
import time
from typing import Callable, List, Optional, Tuple

from alligaitor.ansi_html import ansi_line_to_html

# Matches a full CSI escape sequence (ESC '[' + params + intermediates + final byte),
# including DEC private-mode sequences like tqdm's cursor show/hide (\x1b[?25h/l).
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class ProgressStreamer:
    """Shared line-splitting/filtering logic for both the pty and pipe implementations below.

    ``'\\r'`` progress-bar redraws go to ``progress`` (throttled to at most one call per
    ``min_interval_s``); ``'\\n'`` lines are buffered silently in ``self.plain_lines`` for the
    caller to surface later via ``dump_plain_lines()`` if needed.
    """

    def __init__(
        self,
        log: Callable[[str], None],
        progress: Callable[[str], None],
        min_interval_s: float,
        html_progress: bool = False,
        on_redraw_closed: Optional[Callable[[], None]] = None,
    ):
        self.log = log
        self.progress = progress
        self.min_interval_s = min_interval_s
        # Whether `progress` wants an HTML-rendered progress line (rich-text GUI) or plain text.
        self.html_progress = html_progress
        # Called right after a redraw's definitive final state is emitted, so a caller redrawing
        # a line in place can tell that a second tqdm bar in sequence is not the same line.
        self.on_redraw_closed = on_redraw_closed
        self.buf = ""
        self.last_emit = 0.0
        self.last_logged: Optional[str] = None
        self.plain_lines: List[str] = []
        # True right after seeing a '\r' whose meaning isn't known yet (needs one char lookahead).
        self._pending_cr = False
        # True once self.buf holds content that began after a confirmed-genuine '\r' redraw,
        # so a following bare '\n' (tqdm's closing newline) is recognized as the bar's final
        # state rather than misfiled as an ordinary line.
        self._in_progress_redraw = False

    @staticmethod
    def _clean(raw: str) -> str:
        # A pty makes some tools emit ANSI codes they'd skip on a plain pipe; strip for the log.
        return _ANSI_ESCAPE_RE.sub("", raw).strip()

    def feed(self, ch: str) -> None:
        # A pty's ONLCR translation turns every plain '\n' into '\r\n', so a print()'d line and
        # a real tqdm redraw both start with '\r'; one character of lookahead (_pending_cr)
        # distinguishes them by whether '\n' immediately follows. _in_progress_redraw further
        # disambiguates a bar's closing '\r\n' (its true final state) from an ordinary line.
        if self._pending_cr:
            self._pending_cr = False
            if ch == "\n":
                if self._in_progress_redraw:
                    self._flush_progress_final()
                else:
                    self._flush_plain()
                self._in_progress_redraw = False
                return
            self._flush_progress()
            self._in_progress_redraw = True
            # fall through -- `ch` still needs to be processed below,
            # it's the start of whatever comes after the redraw

        if ch == "\r":
            self._pending_cr = True
        elif ch == "\n":
            # Reachable without a preceding '\r' only on the Windows pipe fallback (no pty).
            if self._in_progress_redraw:
                self._flush_progress_final()
            else:
                self._flush_plain()
            self._in_progress_redraw = False
        else:
            self.buf += ch

    def _flush_plain(self) -> None:
        line = self._clean(self.buf)
        self.buf = ""
        if line:
            self.plain_lines.append(line)

    def _progress_text(self, raw: str, plain: str) -> str:
        """Render an HTML progress line from ANSI-coded raw bytes, or plain text otherwise."""
        if self.html_progress:
            return ansi_line_to_html("    " + raw)
        return f"    {plain}"

    def _flush_progress(self) -> None:
        """Throttled flush, for a redraw that is not the bar's definitive final state."""
        plain = self._clean(self.buf)
        raw = self.buf
        self.buf = ""
        if plain and plain != self.last_logged:
            now = time.monotonic()
            if now - self.last_emit >= self.min_interval_s:
                self.progress(self._progress_text(raw, plain))
                self.last_emit = now
                self.last_logged = plain
            # else: intermediate redraw dropped by design, another follows shortly.

    def _flush_progress_final(self) -> None:
        """Unconditionally flush self.buf, bypassing the throttle so the bar's true final state
        (or whatever's left buffered at exit) is never silently dropped."""
        plain = self._clean(self.buf)
        raw = self.buf
        self.buf = ""
        if plain and plain != self.last_logged:
            self.progress(self._progress_text(raw, plain))
            self.last_emit = time.monotonic()
            self.last_logged = plain
            if self.on_redraw_closed is not None:
                self.on_redraw_closed()

    def finish(self) -> None:
        # A trailing '\r' at EOF is always the final progress state, never a real line ending.
        if self._pending_cr:
            self._flush_progress_final()
            self._pending_cr = False

        # Flush whatever's left unterminated (no-op if the branch above already consumed it).
        self._flush_progress_final()

    def dump_plain_lines(self, reason: str) -> None:
        """Surface the buffered plain-line output via ``log``, prefixed with ``reason``."""
        if not self.plain_lines:
            return
        self.log(f"    -- full output ({reason}) --")
        for line in self.plain_lines:
            self.log(f"    {line}")


def _stream_subprocess_pty(cmd, env, min_interval_s, log, progress, html_progress, on_redraw_closed) -> Tuple[int, ProgressStreamer]:
    """POSIX (mac/Linux) implementation: runs cmd attached to a pseudo-terminal.

    A pty makes tqdm's isatty() check see a real terminal, so it redraws live instead of
    falling back to a plain-pipe summary mode. A window size is set explicitly since
    openpty() alone leaves it at 0x0, which would otherwise render nothing.
    """
    import errno
    import fcntl
    import pty
    import struct
    import termios

    master_fd, slave_fd = pty.openpty()
    fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 80, 0, 0))
    proc = subprocess.Popen(
        cmd, stdout=slave_fd, stderr=slave_fd, stdin=subprocess.DEVNULL,
        close_fds=True, env=env,
    )
    os.close(slave_fd)  # only the child needs the slave end

    streamer = ProgressStreamer(log, progress, min_interval_s, html_progress, on_redraw_closed)
    # newline="" disables universal-newlines translation, which would otherwise silently
    # rewrite every '\r' to '\n' and hide progress-bar redraws as plain lines.
    with os.fdopen(master_fd, "r", buffering=1, newline="", errors="replace") as master:
        while True:
            try:
                ch = master.read(1)
            except OSError as e:
                if e.errno == errno.EIO:
                    break  # child closed its end -- normal EOF for a pty
                raise
            if ch == "":
                if proc.poll() is not None:
                    break
                continue
            streamer.feed(ch)

    proc.wait()
    streamer.finish()
    return proc.returncode, streamer


def _stream_subprocess_pipe(cmd, env, min_interval_s, log, progress, html_progress, on_redraw_closed) -> Tuple[int, ProgressStreamer]:
    """Windows fallback: uses a plain pipe since a real pty needs platform APIs this
    codebase doesn't otherwise depend on. A tool that disables its progress bar off-tty
    may still print less here than on the pty path.
    """
    # Not using Popen(text=True): it applies universal-newlines translation, silently
    # rewriting '\r' to '\n' and breaking the progress/plain-line distinction. The pipe
    # is opened in raw binary mode and wrapped by hand with newline="" instead.
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, env=env,
    )
    stdout_text = io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="replace", newline="")
    streamer = ProgressStreamer(log, progress, min_interval_s, html_progress, on_redraw_closed)
    try:
        while True:
            ch = stdout_text.read(1)
            if ch == "":
                if proc.poll() is not None:
                    break
                continue
            streamer.feed(ch)
    finally:
        stdout_text.close()

    proc.wait()
    streamer.finish()
    return proc.returncode, streamer


def stream_subprocess(
    cmd: List[str],
    log: Callable[[str], None],
    progress: Optional[Callable[[str], None]] = None,
    min_interval_s: float = 0.5,
    html_progress: bool = False,
    on_redraw_closed: Optional[Callable[[], None]] = None,
) -> Tuple[int, ProgressStreamer]:
    """Run ``cmd``, forwarding its live tqdm-style progress line through ``progress`` as it runs.

    ``progress`` receives repeated updates for the same redrawing line; ``log`` is for discrete
    one-off messages. Ordinary ``'\\n'``-terminated output is buffered rather than logged live.

    Args:
        html_progress: If True, ``progress`` receives an HTML rendering of the command's ANSI
            styling instead of plain text.
        on_redraw_closed: Forwarded to ProgressStreamer; called when a redrawn line's final
            state has just been sent to ``progress``.

    Returns:
        ``(exit code, the ProgressStreamer used)`` -- the caller can surface buffered plain
        output via the streamer's ``dump_plain_lines()``.
    """
    if progress is None:
        progress = log

    # CPython block-buffers stdout off-tty regardless of flush calls; force unbuffered
    # output from the child so progress output isn't delayed behind an internal buffer.
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if sys.platform.startswith("win"):
        return _stream_subprocess_pipe(cmd, env, min_interval_s, log, progress, html_progress, on_redraw_closed)
    return _stream_subprocess_pty(cmd, env, min_interval_s, log, progress, html_progress, on_redraw_closed)
