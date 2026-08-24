"""Runs a subprocess while forwarding its live tqdm-style progress output
as it happens, instead of staying silent until the process exits.

``alligaitor.inference.run_inference`` shells out to ``sleap-nn
predict``, which -- like most CLI tools -- prints a tqdm progress bar
that rewrites a single line via carriage returns (``'\\r'``). Run under
``subprocess.run()`` with inherited stdio, that output either goes
nowhere the GUI can see (a spawned batch-worker process's stdio isn't
the GUI's own), or floods a log with page after page of "same" redrawn
line if captured naively -- both leave a long inference run looking
hung. This module tells the two kinds of output SLEAP-NN CLIs produce
apart (a redrawing progress line vs. ordinary ``'\\n'``-terminated
messages) and lets a caller wire the redrawing line to something that
displays "the same line updating" (see ``gui/main_window.py``'s
``_on_progress_line``) separately from one-off log messages.

Ported from RATlab-NOR's sleap_inference.py (its ``_ProgressStreamer``,
``_stream_subprocess_pty``, ``_stream_subprocess_pipe``, and
``_stream_subprocess``), generalized to any command rather than being
tracking-specific, and exposed as public names since it's now meant to
be reused (alliGAITor only has the one such subprocess call today, but
nothing here is inference-specific).
"""

from __future__ import annotations

import io
import os
import re
import subprocess
import sys
import time
from typing import Callable, List, Optional, Tuple

# A CSI (Control Sequence Introducer) escape sequence is ESC '[', then
# parameter bytes (0x30-0x3F: digits plus ':;<=>?'), then intermediate
# bytes (0x20-0x2F), then one final byte (0x40-0x7E) -- matched here as
# [0-?]*[ -/]*[@-~] rather than the narrower [0-9;]*[a-zA-Z] this used to
# be, which missed anything using the parameter-byte range beyond plain
# digits/semicolons. In particular '?' (0x3F) marks a DEC private-mode
# sequence -- \x1b[?25h/\x1b[?25l (show/hide cursor) are exactly what
# tqdm emits around every redraw, and the old pattern let them straight
# through as literal "[?25h" text once the (actually-invisible) ESC byte
# itself was gone.
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class ProgressStreamer:
    """Shared line-splitting/filtering logic for both the pty and pipe
    implementations below -- only how characters get onto the wire
    differs between them.

    A subprocess's output is two very different kinds of thing mixed
    into one stream: a tqdm-style progress bar that rewrites a single
    line via carriage returns (``'\\r'``), and everything else (config
    dumps, device/model info, warnings) as ordinary ``'\\n'``-terminated
    lines. The progress bar is the one thing actually worth watching
    live during a long run -- the rest is mostly startup noise that
    would flood the log around it. So: ``'\\r'`` updates go to
    ``progress`` (throttled to at most one call per ``min_interval_s`` --
    a tqdm bar can emit dozens a second) rather than ``log``,
    specifically so a caller that wants to redraw a single line in place
    (the GUI does -- see ``main_window.py``'s ``_on_progress_line``) can
    tell "this is the same bar updating" apart from "this is a new,
    separate message". ``'\\n'`` lines go to neither live -- they're
    buffered silently (``self.plain_lines``) so a caller can decide
    afterwards whether they're worth surfacing (via ``log``) -- see
    ``dump_plain_lines()``. Not every "the process technically exited 0"
    run is actually fine, so that decision is left to the caller rather
    than being tied to exit code alone in here.
    """

    def __init__(self, log: Callable[[str], None], progress: Callable[[str], None], min_interval_s: float):
        self.log = log
        self.progress = progress
        self.min_interval_s = min_interval_s
        self.buf = ""
        self.last_emit = 0.0
        self.last_logged: Optional[str] = None
        self.plain_lines: List[str] = []
        # True right after seeing a '\r' whose meaning isn't known yet --
        # see feed()'s in-place comment for why this needs one character
        # of lookahead.
        self._pending_cr = False

    @staticmethod
    def _clean(raw: str) -> str:
        # Forcing a pty (see _stream_subprocess_pty) makes some tools emit
        # ANSI color/cursor codes they'd otherwise skip when writing to a
        # plain pipe -- strip those so the log shows plain text.
        return _ANSI_ESCAPE_RE.sub("", raw).strip()

    def feed(self, ch: str) -> None:
        # A pty's own line-ending translation (ONLCR) rewrites every
        # plain '\n' a child writes into '\r\n' before we ever see it --
        # so an ordinary print()'d line and a real tqdm carriage-return
        # redraw both start by handing us a '\r'; the only way to tell
        # them apart is whether a '\n' immediately follows. Hence the
        # one-character lookahead via _pending_cr: a '\r' immediately
        # followed by '\n' is an ordinary line ending (buffered, not
        # shown live); a '\r' followed by anything else is a genuine
        # progress-bar redraw (shown live, throttled).
        if self._pending_cr:
            self._pending_cr = False
            if ch == "\n":
                self._flush_plain()
                return
            self._flush_progress()
            # fall through -- `ch` still needs to be processed below,
            # it's the start of whatever comes after the redraw

        if ch == "\r":
            self._pending_cr = True
        elif ch == "\n":
            self._flush_plain()
        else:
            self.buf += ch

    def _flush_plain(self) -> None:
        line = self._clean(self.buf)
        self.buf = ""
        if line:
            self.plain_lines.append(line)

    def _flush_progress(self) -> None:
        line = self._clean(self.buf)
        self.buf = ""
        if line and line != self.last_logged:
            now = time.monotonic()
            if now - self.last_emit >= self.min_interval_s:
                self.progress(f"    {line}")
                self.last_emit = now
                self.last_logged = line
            # else: an intermediate progress-bar redraw, dropped by
            # design -- the trailing flush in finish() still catches it
            # if the process happens to exit right after.

    def finish(self) -> None:
        # A trailing '\r' with nothing after it (no more input, EOF) is
        # ambiguous -- but in practice that's always the final progress
        # state right before the process exits, never a real line ending
        # (those get their '\n' before EOF), so treat it as progress.
        if self._pending_cr:
            self._flush_progress()
            self._pending_cr = False

        # Flush whatever's left unterminated -- e.g. the final "100%"
        # state if it wasn't itself followed by a '\r'/'\n' before exit.
        tail = self._clean(self.buf)
        if tail and tail != self.last_logged:
            self.progress(f"    {tail}")

    def dump_plain_lines(self, reason: str) -> None:
        """Surfaces the buffered ``'\\n'`` output (see class docstring)
        via ``log``, prefixed with ``reason`` -- call this when the
        caller decides, by whatever criteria, that this run's output is
        worth digging into. A no-op if nothing was buffered."""
        if not self.plain_lines:
            return
        self.log(f"    -- full output ({reason}) --")
        for line in self.plain_lines:
            self.log(f"    {line}")


def _stream_subprocess_pty(cmd, env, min_interval_s, log, progress) -> Tuple[int, ProgressStreamer]:
    """POSIX (mac/Linux) implementation: runs cmd with its stdout/stderr
    attached to a pseudo-terminal instead of a plain pipe.

    This matters beyond just buffering: tqdm (and most other CLI progress
    bars) explicitly check isatty() and behave very differently
    depending on the answer -- connected to a plain pipe (as a normal
    subprocess.PIPE would be), many of them redraw far less often, or
    only print a final summary, specifically to avoid spamming a log
    file. A pty makes the child see what looks like a real interactive
    terminal, so it behaves the same way it would if you'd run it
    directly in a terminal yourself.

    Also sets a window size on the pty (openpty() alone leaves it at
    0x0) -- without this, tqdm can't determine a terminal width and
    renders nothing at all rather than falling back to a sane default,
    which looks identical to "the progress bar is broken" from here even
    though isatty() is already True.
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

    streamer = ProgressStreamer(log, progress, min_interval_s)
    # newline="" disables universal-newlines translation -- the default
    # (newline=None) silently rewrites every '\r' to '\n' before this code
    # ever sees it (true of both os.fdopen and subprocess's own
    # text=True/PIPE, which is why the plain-pipe path below doesn't use
    # that either), which would make every tqdm update look like a plain
    # '\n' line (buffered, not shown live) instead of a progress-bar redraw.
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


def _stream_subprocess_pipe(cmd, env, min_interval_s, log, progress) -> Tuple[int, ProgressStreamer]:
    """Windows fallback: a real pty needs platform APIs this codebase
    doesn't otherwise depend on (pywinpty/conpty), so this uses a plain
    pipe instead. PYTHONUNBUFFERED (see stream_subprocess) still fixes
    the buffering half of the problem; a tool that specifically disables
    its progress bar when not attached to a real terminal may still
    print less on Windows than on mac/Linux -- see
    _stream_subprocess_pty's docstring for why that's pty-specific.
    """
    # Deliberately *not* using Popen(text=True) here: it applies the same
    # universal-newlines translation as os.fdopen's default (silently
    # rewriting every '\r' to '\n' before this code ever sees it,
    # breaking the progress-bar/plain-line distinction ProgressStreamer
    # relies on), and Popen has no way to override that translation
    # itself -- so the pipe is opened in raw binary mode and wrapped by
    # hand with newline="" instead.
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, env=env,
    )
    stdout_text = io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="replace", newline="")
    streamer = ProgressStreamer(log, progress, min_interval_s)
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
) -> Tuple[int, ProgressStreamer]:
    """Runs ``cmd``, forwarding its live tqdm-style progress output
    through ``progress`` as it runs (throttled -- see ProgressStreamer),
    rather than staying silent for however long the command takes and
    only surfacing everything after the fact. ``progress`` is called
    repeatedly for what is conceptually the *same* single line being
    redrawn -- as opposed to ``log``, which is for discrete one-off
    messages -- so a caller that wants to show that redraw in place (the
    GUI does; see main_window.py's ``_on_progress_line``) can tell the
    two apart. Defaults to ``log`` if not given, e.g. for plain
    print()-based callers where "in place" doesn't apply.

    Ordinary ``'\\n'``-terminated output (config dumps, device info,
    warnings -- not the progress bar) is intentionally kept out of the
    live log entirely. Returns ``(exit code, the ProgressStreamer used)``
    -- the caller decides whether that buffered output is worth
    surfacing via the streamer's ``dump_plain_lines()``.
    """
    if progress is None:
        progress = log

    # sleap-nn is itself a Python program. CPython only line-buffers
    # stdout when it's attached to a real terminal -- writing to a plain
    # pipe makes it fall back to full block buffering (~8KB) regardless
    # of how the writer flushes. PYTHONUNBUFFERED forces CPython's
    # stdout/stderr to be unbuffered for the child process (same effect
    # as `python -u`) -- kept even with the pty path below, since a pty
    # fixes tqdm's own tty-detection behavior but this is a separate,
    # lower-level buffering layer underneath it.
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if sys.platform.startswith("win"):
        return _stream_subprocess_pipe(cmd, env, min_interval_s, log, progress)
    return _stream_subprocess_pty(cmd, env, min_interval_s, log, progress)
