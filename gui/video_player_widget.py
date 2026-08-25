"""
Reusable video-playback widget: frame-by-frame stepping, real-time
Play/Pause, and Rewind/Fast-Forward acceleration (2x, 5x, 10x -- resetting
back to a configurable "normal" speed rather than jumping). Used by
validation_video_dialog.py to play a session's already-annotated
validation video (see alligaitor/validation_video.py) -- no overlay
drawing of its own.

Ported from RATlab-NOR's gui/video_player_widget.py: transport controls are
carried over essentially verbatim (see that module's own docstring for why
this stays plain OpenCV + Qt paint events rather than QMediaPlayer -- a
past macOS segfault with threaded video decode). The one real change is
_ScrubBar -> _MultiRowScrubBar: alliGAITor needs one highlighted range per
paw, stacked in its own labeled row so overlapping paw windows stay
visually distinct, rather than NOR's single overlapping-marker line.
"""

from __future__ import annotations

import cv2
from PySide6.QtCore import Qt, QTimer, Signal, QPointF, QRectF
from PySide6.QtGui import QImage, QPixmap, QPainter, QShortcut, QKeySequence, QFont, QPen, QBrush, QColor
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel


def _monospace_font() -> QFont:
    """A fixed-pitch font via Qt's own font-matching rather than a
    "font-family: monospace" stylesheet string -- see NOR's original for
    why (avoids a startup "missing font family" warning on some
    platforms)."""
    font = QFont()
    font.setStyleHint(QFont.Monospace)
    font.setFixedPitch(True)
    return font


# Rewind/Fast-Forward acceleration steps -- each successive click in the
# same direction advances to the next (higher) magnitude, clamped at the
# last entry; a click in the other direction, or Play/Pause, resets back
# to the current "normal" speed (see NORMAL_SPEEDS/default_speed).
ACCEL_STEPS = [2.0, 5.0, 10.0]

# Baseline ("normal") forward speed, cycled by the Speed button --
# independent of Rewind/Fast-Forward's own acceleration. Play/Pause (and
# switching accel direction) always resets back to whichever of these is
# currently selected.
NORMAL_SPEEDS = [0.25, 0.5, 1.0, 2.0, 4.0]


def bgr_frame_to_qimage(frame) -> QImage:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return qimg.copy()  # own the buffer -- `rgb` goes out of scope after this call


_SCRUB_MARGIN = 8
_SCRUB_LABEL_WIDTH = 92
_SCRUB_ROW_HEIGHT = 22
_SCRUB_TRACK_COLOR = QColor(190, 190, 190)
_SCRUB_HANDLE_COLOR = QColor(40, 90, 200)
_SCRUB_HANDLE_BORDER = QColor(20, 50, 120)
_SCRUB_LABEL_COLOR = QColor(220, 220, 220)


class _MultiRowScrubBar(QWidget):
    """Click/drag horizontal seek bar with one labeled row per paw, each
    showing that paw's single highlighted window (see
    alligaitor.gait.PawWindow) as a colored segment on its own track --
    rows stack top to bottom rather than overlapping on one line, so e.g.
    all four paws' windows stay independently readable even when they
    overlap in time. A single vertical playhead line spans every row, and
    clicking/dragging anywhere (regardless of which row's y) seeks, since
    every row shares the same frame -> x mapping.
    """

    seek_requested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.n_frames = 1
        self.current_frame = 0
        # (label, start_frame or None, end_frame or None, QColor)
        self.rows: list[tuple[str, "int | None", "int | None", QColor]] = []
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)
        self._relayout()

    def _relayout(self):
        n_rows = max(1, len(self.rows))
        self.setFixedHeight(n_rows * _SCRUB_ROW_HEIGHT)

    def set_range(self, n_frames: int):
        self.n_frames = max(1, n_frames)
        self.update()

    def set_paw_windows(self, rows):
        """`rows`: iterable of (label, start_frame_or_None,
        end_frame_or_None, QColor), one entry per paw, in the order they
        should be stacked top to bottom. A ``None`` start/end means that
        paw has nothing to highlight (never detected at all this trial)."""
        self.rows = list(rows)
        self._relayout()
        self.update()

    def set_current_frame(self, frame_idx: int):
        self.current_frame = frame_idx
        self.update()

    def _track_bounds(self):
        return _SCRUB_MARGIN + _SCRUB_LABEL_WIDTH, self.width() - _SCRUB_MARGIN

    def _frame_to_x(self, frame_idx: float) -> float:
        left, right = self._track_bounds()
        track_w = max(1, right - left)
        if self.n_frames <= 1:
            return left
        return left + track_w * frame_idx / (self.n_frames - 1)

    def _x_to_frame(self, x: float) -> int:
        left, right = self._track_bounds()
        track_w = right - left
        if track_w <= 0:
            return 0
        frac = (x - left) / track_w
        frac = max(0.0, min(1.0, frac))
        return round(frac * (self.n_frames - 1))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        left, right = self._track_bounds()

        track_pen = QPen(_SCRUB_TRACK_COLOR, 3)
        track_pen.setCapStyle(Qt.RoundCap)
        marker_pen = QPen()
        marker_pen.setWidth(9)
        marker_pen.setCapStyle(Qt.FlatCap)

        rows = self.rows or [("", None, None, _SCRUB_HANDLE_COLOR)]
        for i, (label, start, stop, color) in enumerate(rows):
            mid_y = i * _SCRUB_ROW_HEIGHT + _SCRUB_ROW_HEIGHT / 2

            painter.setPen(QPen(_SCRUB_LABEL_COLOR))
            painter.drawText(
                QRectF(0, i * _SCRUB_ROW_HEIGHT, _SCRUB_LABEL_WIDTH + _SCRUB_MARGIN, _SCRUB_ROW_HEIGHT),
                Qt.AlignRight | Qt.AlignVCenter, label,
            )

            painter.setPen(track_pen)
            painter.drawLine(QPointF(left, mid_y), QPointF(right, mid_y))

            if start is not None and stop is not None:
                x0 = self._frame_to_x(start)
                x1 = max(self._frame_to_x(max(stop, start)), x0 + 1)
                marker_pen.setColor(color)
                painter.setPen(marker_pen)
                painter.drawLine(QPointF(x0, mid_y), QPointF(x1, mid_y))

        handle_x = self._frame_to_x(self.current_frame)
        painter.setPen(QPen(QColor(120, 120, 120), 1))
        painter.drawLine(QPointF(handle_x, 0), QPointF(handle_x, self.height()))
        painter.setPen(QPen(_SCRUB_HANDLE_BORDER, 1))
        painter.setBrush(QBrush(_SCRUB_HANDLE_COLOR))
        painter.drawEllipse(QPointF(handle_x, self.height() / 2), 6, 6)
        painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.seek_requested.emit(self._x_to_frame(event.position().x()))

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.seek_requested.emit(self._x_to_frame(event.position().x()))


class _PlayerCanvas(QWidget):
    """Paints the current frame -- no overlay hook needed here, since
    every alliGAITor caller plays an already-annotated video."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap: QPixmap | None = None

    def set_frame(self, qimage: QImage):
        self.pixmap = QPixmap.fromImage(qimage)
        self.setFixedSize(self.pixmap.size())
        self.update()

    def paintEvent(self, event):
        if self.pixmap is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.drawPixmap(0, 0, self.pixmap)
        painter.end()


class VideoPlayerWidget(QWidget):
    """Drop into any layout. Owns its own cv2.VideoCapture -- call
    release() when done with it (e.g. from the host dialog's
    closeEvent/reject)."""

    frame_changed = Signal(int)

    def __init__(self, video_path, fps_hint=None, enable_space_shortcut=True, parent=None):
        super().__init__(parent)
        self.video_path = str(video_path)
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or fps_hint or 30.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.current_frame_idx = 0
        self._decoder_next_idx = 0  # what cap.read() will return next, absent a re-seek

        self.default_speed = 1.0
        self.playback_rate = self.default_speed
        self._accel_dir = None   # "ff", "rw", or None -- which button last accelerated
        self._accel_idx = -1     # index into ACCEL_STEPS for the current accel direction
        self.playing = False

        self.canvas = _PlayerCanvas()

        self.scrub_bar = _MultiRowScrubBar()
        self.scrub_bar.set_range(self.n_frames)
        self.scrub_bar.seek_requested.connect(self._on_scrub_seek)

        self.frame_label = QLabel()
        self.frame_label.setFont(_monospace_font())

        self.rewind_btn = QPushButton("⏪ Rewind")
        self.step_back_btn = QPushButton("◀| -1f")
        self.play_btn = QPushButton("▶ Play")
        self.step_fwd_btn = QPushButton("1f |▶")
        self.ff_btn = QPushButton("Fast-Forward ⏩")
        self.speed_label = QLabel(self._speed_label())
        self.speed_label.setFont(_monospace_font())
        self.speed_label.setMinimumWidth(60)
        self.speed_label.setAlignment(Qt.AlignCenter)
        self.speed_btn = QPushButton(self._default_speed_label())

        self.rewind_btn.clicked.connect(self._on_rewind)
        self.step_back_btn.clicked.connect(lambda: self.step_frames(-1))
        self.play_btn.clicked.connect(self.toggle_play_pause)
        self.step_fwd_btn.clicked.connect(lambda: self.step_frames(1))
        self.ff_btn.clicked.connect(self._on_fast_forward)
        self.speed_btn.clicked.connect(self._on_cycle_default_speed)

        transport_row = QHBoxLayout()
        for w in (
            self.rewind_btn, self.step_back_btn, self.play_btn, self.step_fwd_btn, self.ff_btn,
            self.speed_label, self.speed_btn,
        ):
            transport_row.addWidget(w)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_timer_tick)

        self._space_shortcut = None
        if enable_space_shortcut:
            # WindowShortcut (not WidgetWithChildrenShortcut) -- this
            # widget is normally embedded inside a larger dialog alongside
            # other focusable widgets (e.g. the Flag Paw(s) button) that
            # are its *siblings*, not descendants -- see NOR's original
            # for the full reasoning.
            self._space_shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
            self._space_shortcut.setContext(Qt.WindowShortcut)
            self._space_shortcut.activated.connect(self.toggle_play_pause)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas, alignment=Qt.AlignHCenter)
        layout.addWidget(self.scrub_bar)
        layout.addWidget(self.frame_label, alignment=Qt.AlignHCenter)
        layout.addLayout(transport_row)

        self.goto_frame(0)

    # -- frame navigation --

    def goto_frame(self, idx: int):
        idx = max(0, min(idx, self.n_frames - 1))
        if idx != self._decoder_next_idx:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok:
            self.pause()
            return
        self._decoder_next_idx = idx + 1
        self.current_frame_idx = idx
        qimg = bgr_frame_to_qimage(frame)
        self.canvas.set_frame(qimg)
        self.scrub_bar.set_current_frame(idx)
        self._update_frame_label()
        self.frame_changed.emit(idx)
        if idx >= self.n_frames - 1 or idx <= 0:
            self.pause()

    def _on_scrub_seek(self, frame_idx: int):
        self.pause()
        self.goto_frame(frame_idx)

    def set_paw_windows(self, rows):
        """Mark each paw's highlighted window on the scrub bar -- see
        _MultiRowScrubBar.set_paw_windows."""
        self.scrub_bar.set_paw_windows(rows)

    def _update_frame_label(self):
        t = self.current_frame_idx / self.fps
        self.frame_label.setText(f"Frame {self.current_frame_idx} / {self.n_frames - 1}      t = {t:.3f}s")

    def step_frames(self, delta: int):
        self.pause()
        self.goto_frame(self.current_frame_idx + delta)

    # -- playback --

    def _restart_timer(self):
        interval_ms = max(1, round(1000.0 / (self.fps * abs(self.playback_rate))))
        self.timer.start(interval_ms)

    def _start_playback(self):
        if self.current_frame_idx >= self.n_frames - 1 and self.playback_rate > 0:
            self.goto_frame(0)
        elif self.current_frame_idx <= 0 and self.playback_rate < 0:
            self.goto_frame(self.n_frames - 1)
        self.playing = True
        self.play_btn.setText("⏸ Pause")
        self._restart_timer()

    def pause(self):
        if not self.playing:
            return
        self.playing = False
        self.play_btn.setText("▶ Play")
        self.timer.stop()

    def _reset_speed(self):
        """Drop any Rewind/Fast-Forward acceleration and go back to the
        current "normal" speed (self.default_speed, set via the Speed
        button -- 1x unless changed)."""
        self._accel_dir = None
        self._accel_idx = -1
        self.playback_rate = self.default_speed
        self.speed_label.setText(self._speed_label())
        if self.playing:
            self._restart_timer()

    def toggle_play_pause(self):
        """Space and the Play/Pause button both land here -- always
        resets any Rewind/Fast-Forward acceleration back to the current
        normal speed, then toggles play/pause."""
        was_playing = self.playing
        self._reset_speed()
        if was_playing:
            self.pause()
        else:
            self._start_playback()

    def _on_cycle_default_speed(self):
        idx = NORMAL_SPEEDS.index(self.default_speed) if self.default_speed in NORMAL_SPEEDS else NORMAL_SPEEDS.index(1.0)
        self.default_speed = NORMAL_SPEEDS[(idx + 1) % len(NORMAL_SPEEDS)]
        self.speed_btn.setText(self._default_speed_label())
        self._reset_speed()

    def _on_fast_forward(self):
        if self._accel_dir == "ff":
            self._accel_idx = min(self._accel_idx + 1, len(ACCEL_STEPS) - 1)
        else:
            self._accel_idx = 0
        self._accel_dir = "ff"
        self.playback_rate = ACCEL_STEPS[self._accel_idx]
        self.speed_label.setText(self._speed_label())
        self._start_playback()

    def _on_rewind(self):
        if self._accel_dir == "rw":
            self._accel_idx = min(self._accel_idx + 1, len(ACCEL_STEPS) - 1)
        else:
            self._accel_idx = 0
        self._accel_dir = "rw"
        self.playback_rate = -ACCEL_STEPS[self._accel_idx]
        self.speed_label.setText(self._speed_label())
        self._start_playback()

    def _on_timer_tick(self):
        step = 1 if self.playback_rate > 0 else -1
        next_idx = self.current_frame_idx + step
        if next_idx < 0 or next_idx > self.n_frames - 1:
            self.pause()
            return
        self.goto_frame(next_idx)

    def _speed_label(self):
        return f"{self.playback_rate:+g}x" if self.playback_rate != 1.0 else "1x"

    def _default_speed_label(self):
        return f"Normal: {self.default_speed:g}x"

    # -- lifecycle --

    def release(self):
        self.pause()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
