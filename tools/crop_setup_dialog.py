"""
"Crop Videos…" dialog -- walks through videos one at a time, letting you
drag a fixed-size (width x height) crop window into place per video.
The camera/tunnel framing isn't guaranteed to repeat exactly across
every recording, so this defaults to per-video positioning rather than
one uniform crop for a whole folder.

Confirming a video (Forward) crops it immediately (a single video, on
the main thread -- brief and bounded). A "Use This Position for All
Remaining" button skips the rest of the walkthrough when the framing
hasn't moved: it crops every remaining video in one shot via CropRunner,
a separate process (see crop_worker_process.py for why cropping a whole
batch can't just happen inline here).

Positions are cached per video (video_crop.load_positions/
save_positions, keyed with frame_utils.video_key()) at
<output_folder>/crop_positions.json, so reopening this dialog on the
same input/output folder resumes rather than re-asking for videos
already cropped.

Ported from RATlab-NOR's gui/crop_setup_dialog.py, with two changes:

  1. The `object_picker`/`object_setup_dialog` imports (NOR-specific
     modules that don't exist here) are replaced with frame_utils.py, a
     minimal extraction of just the three functions this dialog
     actually used from them.
  2. _CropCanvas now scales the displayed frame (and crop box with it)
     to fit whatever size the dialog window currently is, letterboxed
     to preserve aspect ratio, instead of forcing the window to the
     video's native resolution -- NOR's original version called
     setFixedSize(pixmap.size()), which made sense for its ~1000px
     arena footage but is unusable for e.g. a 1920x1080 source frame on
     a laptop screen. self.x/self.y/self.crop_w/self.crop_h stay in
     native-frame pixel coordinates throughout (the same coordinates
     video_crop.crop_video() takes); only paintEvent and the mouse
     handlers convert to/from the current display scale.

RATlab-NOR is MIT-licensed (Copyright (c) 2026 Mitchell Carson); see
../THIRD_PARTY_NOTICES.md for the full license text.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QRectF, QPointF, QSize
from PySide6.QtGui import QPainter, QPen, QColor, QPixmap
from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSpinBox, QMessageBox, QPlainTextEdit, QProgressBar, QApplication,
    QSizePolicy, QRadioButton, QButtonGroup, QSlider,
)

from frame_utils import video_key, grab_middle_frame, bgr_frame_to_qimage
import video_crop as vc
from crop_runner import CropRunner

_RECT_COLOR = QColor(255, 140, 0)
_RECT_COLOR_BAD = QColor(200, 40, 40)

# Floor on the letterboxed display scale -- without this, a very small
# dialog on a small screen could shrink a 1280-wide frame down to where
# the crop rectangle becomes un-clickable (sub-pixel wide). The user can
# still resize the window bigger; this just stops things from going
# unusably tiny rather than silently degrading dragging.
_MIN_DISPLAY_SCALE = 0.05


class _CropCanvas(QWidget):
    """Shows the loaded preview frame, letterboxed/scaled to fill
    whatever size this widget currently is, with a draggable rectangle
    marking the crop window (drag, click-to-place, arrow-key nudge).
    self.x/self.y/self.crop_w/self.crop_h are always in *frame*
    (native-resolution) pixel coordinates -- the same coordinates
    video_crop.crop_video() takes -- regardless of how the frame is
    currently being scaled for display; only paintEvent and the mouse
    handlers ever deal with the display-space <-> frame-space
    conversion."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(240, 120)
        self.pixmap = None
        self.frame_w = 0
        self.frame_h = 0
        self.crop_w = 0
        self.crop_h = 0
        self.x = 0
        self.y = 0
        self._dragging = False
        self._drag_offset = (0, 0)  # in frame-space
        self.on_change = None  # callable(x, y), invoked after any position change

    def set_frame(self, qimage, crop_w, crop_h):
        self.pixmap = QPixmap.fromImage(qimage)
        self.frame_w, self.frame_h = self.pixmap.width(), self.pixmap.height()
        self.set_crop_size(crop_w, crop_h)
        self.update()

    def set_crop_size(self, crop_w, crop_h):
        self.crop_w, self.crop_h = crop_w, crop_h
        self.set_position(self.x, self.y)  # re-clamp

    def set_position(self, x, y):
        max_x = max(0, self.frame_w - self.crop_w)
        max_y = max(0, self.frame_h - self.crop_h)
        self.x = min(max(0, x), max_x)
        self.y = min(max(0, y), max_y)
        self.update()
        if self.on_change:
            self.on_change(self.x, self.y)

    def fits(self) -> bool:
        return self.crop_w <= self.frame_w and self.crop_h <= self.frame_h

    def sizeHint(self):
        if self.frame_w and self.frame_h:
            return QSize(self.frame_w, self.frame_h)
        return QSize(640, 480)

    # -- display-space <-> frame-space mapping --

    def _display_geometry(self):
        """Returns (scale, offset_x, offset_y) for the current widget
        size: scale maps frame-space lengths to displayed/widget-space
        lengths, offsets center the (aspect-ratio-preserved) scaled
        frame within whatever space is available -- letterboxed, so a
        wide/short 1280x170-shaped frame doesn't get stretched into a
        squarer widget."""
        if not self.pixmap or self.frame_w <= 0 or self.frame_h <= 0:
            return 1.0, 0.0, 0.0
        avail_w = max(1, self.width())
        avail_h = max(1, self.height())
        scale = max(_MIN_DISPLAY_SCALE, min(avail_w / self.frame_w, avail_h / self.frame_h))
        disp_w = self.frame_w * scale
        disp_h = self.frame_h * scale
        offset_x = (avail_w - disp_w) / 2
        offset_y = (avail_h - disp_h) / 2
        return scale, offset_x, offset_y

    def _to_frame_pt(self, widget_pt: QPointF) -> QPointF:
        scale, ox, oy = self._display_geometry()
        return QPointF((widget_pt.x() - ox) / scale, (widget_pt.y() - oy) / scale)

    def paintEvent(self, event):
        if self.pixmap is None:
            return
        scale, ox, oy = self._display_geometry()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        target_rect = QRectF(ox, oy, self.frame_w * scale, self.frame_h * scale)
        painter.drawPixmap(target_rect, self.pixmap, QRectF(0, 0, self.frame_w, self.frame_h))
        painter.setPen(QPen(_RECT_COLOR if self.fits() else _RECT_COLOR_BAD, 2))
        painter.drawRect(QRectF(ox + self.x * scale, oy + self.y * scale, self.crop_w * scale, self.crop_h * scale))
        painter.end()

    def _in_rect(self, frame_pt: QPointF) -> bool:
        return (self.x <= frame_pt.x() <= self.x + self.crop_w
                and self.y <= frame_pt.y() <= self.y + self.crop_h)

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton or self.pixmap is None:
            return
        self.setFocus()  # so arrow-key nudging works right after a click, not just after Tab
        pt = self._to_frame_pt(event.position())
        if self._in_rect(pt):
            self._dragging = True
            self._drag_offset = (pt.x() - self.x, pt.y() - self.y)
        else:
            self.set_position(int(pt.x() - self.crop_w / 2), int(pt.y() - self.crop_h / 2))

    def mouseMoveEvent(self, event):
        if not self._dragging:
            return
        pt = self._to_frame_pt(event.position())
        self.set_position(int(pt.x() - self._drag_offset[0]), int(pt.y() - self._drag_offset[1]))

    def mouseReleaseEvent(self, event):
        self._dragging = False

    # No keyPressEvent here on purpose -- CropSetupDialog handles every
    # keyboard shortcut (including arrow-key nudging) itself, rather than
    # splitting handling between this canvas and its parent dialog and
    # relying on Qt's focus/event-bubbling behavior to route things
    # correctly.
    #
    # No resizeEvent override needed either -- Qt already schedules a
    # repaint on resize, and paintEvent recomputes _display_geometry()
    # from self.width()/height() fresh every time, so the frame and crop
    # box just rescale automatically as the dialog is resized.


class CropSetupDialog(QDialog):
    def __init__(self, video_paths, input_folder, output_folder, tools_dir,
                 width, height, parent=None, force_review=False, camera_angle="side"):
        super().__init__(parent)
        self.setWindowTitle("Crop Setup")
        self.setFocusPolicy(Qt.StrongFocus)

        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.tools_dir = tools_dir
        self.runner: CropRunner | None = None
        self._bulk_xy = None

        self.positions_path = self.output_folder / "crop_positions.json"
        self.positions = vc.load_positions(self.positions_path)

        if force_review:
            self.session_videos = list(video_paths)
        else:
            self.session_videos = [
                v for v in video_paths if video_key(v, self.input_folder) not in self.positions
            ]
            if not self.session_videos:
                # Everything's already cropped -- fall back to reviewing
                # all of them rather than opening an empty dialog.
                self.session_videos = list(video_paths)

        self.last_known_xy = None
        if self.positions:
            self.last_known_xy = list(next(reversed(self.positions.values())))

        self.idx = 0
        self._current_raw_frame = None  # set by _load_video; re-rendered by _update_preview()

        # -- widgets --
        self.canvas = _CropCanvas()

        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 20000)
        self.width_spin.setValue(width)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, 20000)
        self.height_spin.setValue(height)
        self.width_spin.valueChanged.connect(self._on_size_changed)
        self.height_spin.valueChanged.connect(self._on_size_changed)

        # Camera angle toggle -- side-angle footage is cropped as-is;
        # bottom-up (tunnel) footage additionally gets
        # video_crop.apply_bottom_up_color_correction() applied per frame
        # at crop time, reproducing the paw/body color-contrast boost the
        # original (now-lost) bottom-up preprocessing produced, without
        # the noise it also introduced. See video_crop.py's module
        # docstring for why.
        self.side_radio = QRadioButton("Side angle")
        self.bottomup_radio = QRadioButton("Bottom-up (apply color correction)")
        self.angle_group = QButtonGroup(self)
        self.angle_group.addButton(self.side_radio)
        self.angle_group.addButton(self.bottomup_radio)
        if camera_angle == "bottom-up":
            self.bottomup_radio.setChecked(True)
        else:
            self.side_radio.setChecked(True)
        self.side_radio.toggled.connect(self._on_angle_changed)
        self.bottomup_radio.toggled.connect(self._on_angle_changed)

        # Strength slider -- the bottom camera has more ambient light than
        # the side-angle footage this correction was tuned against, so the
        # full-strength recipe (video_crop._BC_LAYERS) can look starker
        # than intended on real bottom-up footage. Scales
        # apply_bottom_up_color_correction()'s effect linearly, 0% = no-op,
        # 100% = the documented recipe as-is. Only meaningful (and only
        # enabled) when Bottom-up is selected.
        self.strength_slider = QSlider(Qt.Horizontal)
        self.strength_slider.setRange(0, 100)
        self.strength_slider.setValue(100)
        self.strength_label = QLabel("100%")
        self.strength_label.setMinimumWidth(40)
        self.strength_slider.valueChanged.connect(self._on_strength_changed)
        self.strength_slider.setEnabled(self.bottomup_radio.isChecked())
        self.strength_label.setEnabled(self.bottomup_radio.isChecked())

        self.info_label = QLabel()
        self.info_label.setWordWrap(True)
        self.progress_label = QLabel()

        self.back_btn = QPushButton("< Back")
        self.forward_btn = QPushButton("Forward >")
        self.skip_btn = QPushButton("Skip (Esc)")
        self.apply_all_btn = QPushButton("Use This Position for All Remaining")
        for b in (self.back_btn, self.forward_btn):
            b.setMinimumHeight(34)
            b.setMinimumWidth(95)

        self.back_btn.clicked.connect(self._go_back)
        self.forward_btn.clicked.connect(self._go_forward)
        self.skip_btn.clicked.connect(self._skip)
        self.apply_all_btn.clicked.connect(self._on_apply_to_remaining)

        self.bulk_progress = QProgressBar()
        self.bulk_progress.setVisible(False)
        self.bulk_log = QPlainTextEdit()
        self.bulk_log.setReadOnly(True)
        self.bulk_log.setMaximumBlockCount(1000)
        self.bulk_log.setFixedHeight(90)
        self.bulk_log.setVisible(False)

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Crop size:"))
        size_row.addWidget(self.width_spin)
        size_row.addWidget(QLabel("x"))
        size_row.addWidget(self.height_spin)
        size_row.addStretch()
        size_row.addWidget(self.progress_label)

        angle_row = QHBoxLayout()
        angle_row.addWidget(QLabel("Camera angle:"))
        angle_row.addWidget(self.side_radio)
        angle_row.addWidget(self.bottomup_radio)
        angle_row.addStretch()

        strength_row = QHBoxLayout()
        strength_row.addWidget(QLabel("Correction strength:"))
        strength_row.addWidget(self.strength_slider, stretch=1)
        strength_row.addWidget(self.strength_label)

        nav_row = QHBoxLayout()
        nav_row.addWidget(self.back_btn)
        nav_row.addWidget(self.forward_btn)
        nav_row.addWidget(self.skip_btn)

        layout = QVBoxLayout(self)
        layout.addLayout(size_row)
        layout.addLayout(angle_row)
        layout.addLayout(strength_row)
        layout.addWidget(self.canvas, stretch=1)  # the one thing that should grow when the dialog is resized
        layout.addWidget(self.info_label)
        layout.addLayout(nav_row)
        layout.addWidget(self.apply_all_btn)
        layout.addWidget(self.bulk_progress)
        layout.addWidget(self.bulk_log)

        # Dialog is resizable by default (no setFixedSize anywhere in this
        # class); just pick a sensible starting size instead of whatever
        # Qt's layout would otherwise shrink-to-fit, and cap it well under
        # the screen so it opens fully on-screen regardless of the source
        # video's native resolution.
        screen = QApplication.primaryScreen()
        avail = screen.availableGeometry() if screen else None
        if avail is not None:
            start_w = min(1000, int(avail.width() * 0.8))
            start_h = min(750, int(avail.height() * 0.8))
        else:
            start_w, start_h = 1000, 750
        self.resize(start_w, start_h)

        if self.session_videos:
            self._load_video(0)
        else:
            self.info_label.setText("No videos to crop.")
            self.forward_btn.setEnabled(False)
            self.apply_all_btn.setEnabled(False)

    @property
    def color_grade(self) -> bool:
        return self.bottomup_radio.isChecked()

    @property
    def color_grade_strength(self) -> float:
        return self.strength_slider.value() / 100.0

    # -- preview --

    def _on_angle_changed(self, checked=None):
        self.strength_slider.setEnabled(self.bottomup_radio.isChecked())
        self.strength_label.setEnabled(self.bottomup_radio.isChecked())
        self._update_preview()

    def _on_strength_changed(self, value):
        self.strength_label.setText(f"{value}%")
        self._update_preview()

    def _update_preview(self):
        """Re-renders the canvas's displayed frame from the last-loaded
        raw frame -- called whenever the angle toggle or strength slider
        changes, so the preview reflects what the crop will actually
        produce without re-reading the video. Preserves the current crop
        box position (set_frame re-clamps rather than resetting it)."""
        if self._current_raw_frame is None:
            return
        frame = self._current_raw_frame
        if self.color_grade:
            frame = vc.apply_bottom_up_color_correction(frame, strength=self.color_grade_strength)
        qimage = bgr_frame_to_qimage(frame)
        self.canvas.set_frame(qimage, self.width_spin.value(), self.height_spin.value())

    # -- per-video navigation --

    def _starting_xy_for(self, key):
        if key in self.positions:
            return list(self.positions[key])
        if self.last_known_xy is not None:
            return list(self.last_known_xy)
        return None

    def _load_video(self, idx):
        self.idx = idx
        video_path = self.session_videos[idx]
        try:
            # A frame from partway through the recording, not frame 0 --
            # frame 0 is frequently just the empty tunnel/rig before the
            # rat is placed in it, which makes positioning the crop box
            # (and judging the color-correction preview) impossible.
            frame = grab_middle_frame(video_path)
        except Exception as exc:
            QMessageBox.warning(self, "Couldn't load video", f"Could not read {video_path.name}:\n{exc}")
            self._advance()
            return

        self._current_raw_frame = frame
        self._update_preview()

        key = video_key(video_path, self.input_folder)
        starting = self._starting_xy_for(key)
        if starting is not None:
            self.canvas.set_position(starting[0], starting[1])
        else:
            self.canvas.set_position(
                (self.canvas.frame_w - self.canvas.crop_w) // 2,
                (self.canvas.frame_h - self.canvas.crop_h) // 2,
            )

        self.setWindowTitle(f"Crop Setup -- {video_path.name}  ({idx + 1}/{len(self.session_videos)})")
        self.progress_label.setText(f"{idx + 1} / {len(self.session_videos)}")
        self.back_btn.setEnabled(idx > 0)
        self._update_info()
        self.canvas.setFocus()

    def _update_info(self):
        if not self.canvas.fits():
            self.info_label.setText(
                f"Crop size {self.width_spin.value()}x{self.height_spin.value()} is larger than this "
                f"video's {self.canvas.frame_w}x{self.canvas.frame_h} frame -- can't crop this one."
            )
            self.info_label.setStyleSheet("color: #a94442;")
        else:
            self.info_label.setText("Drag the box into place (or arrow keys to nudge, Shift for 10px).")
            self.info_label.setStyleSheet("color: #666;")
        self.forward_btn.setEnabled(self.canvas.fits())
        self.apply_all_btn.setEnabled(self.canvas.fits())

    def _on_size_changed(self):
        self.canvas.set_crop_size(self.width_spin.value(), self.height_spin.value())
        self._update_info()

    def _out_path_for(self, video_path):
        rel = Path(video_path).relative_to(self.input_folder)
        return self.output_folder / rel

    def _go_forward(self):
        if not self.canvas.fits():
            return
        video_path = self.session_videos[self.idx]
        x, y = self.canvas.x, self.canvas.y

        self.info_label.setText(f"Cropping {video_path.name}...")
        self.info_label.repaint()
        QApplication.processEvents()

        try:
            vc.crop_video(
                video_path, self._out_path_for(video_path), x, y,
                self.width_spin.value(), self.height_spin.value(),
                color_grade=self.color_grade, color_grade_strength=self.color_grade_strength,
            )
        except Exception as exc:
            QMessageBox.warning(self, "Crop failed", f"Couldn't crop {video_path.name}:\n{exc}")
            self._update_info()
            return

        key = video_key(video_path, self.input_folder)
        self.positions[key] = [x, y]
        self.last_known_xy = [x, y]
        vc.save_positions(self.positions_path, self.positions)

        self._advance()

    def _skip(self):
        self._advance()

    def _advance(self):
        if self.idx >= len(self.session_videos) - 1:
            self.accept()
            return
        self._load_video(self.idx + 1)

    def _go_back(self):
        if self.idx == 0:
            return
        self._load_video(self.idx - 1)

    # -- bulk "use this position for all remaining" --

    def _on_apply_to_remaining(self):
        if not self.canvas.fits():
            return
        x, y = self.canvas.x, self.canvas.y
        remaining = self.session_videos[self.idx:]
        reply = QMessageBox.question(
            self, "Crop all remaining?",
            f"Crop the remaining {len(remaining)} video(s) using this same position "
            f"({x},{y}), {self.width_spin.value()}x{self.height_spin.value()}?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._bulk_xy = (x, y)
        positions = [(str(v), x, y) for v in remaining]

        self._set_nav_enabled(False)
        self.bulk_progress.setVisible(True)
        self.bulk_progress.setValue(0)
        self.bulk_log.setVisible(True)
        self.bulk_log.appendPlainText(f"Cropping {len(remaining)} remaining video(s) at ({x},{y})...")

        self.runner = CropRunner(
            self.tools_dir, str(self.input_folder), str(self.output_folder),
            self.width_spin.value(), self.height_spin.value(), positions=positions,
            color_grade=self.color_grade, color_grade_strength=self.color_grade_strength,
        )
        self.runner.log.connect(self.bulk_log.appendPlainText)
        self.runner.progress.connect(self._on_bulk_progress)
        self.runner.video_done.connect(self._on_bulk_video_done)
        self.runner.finished_run.connect(self._on_bulk_finished_run)
        self.runner.finished.connect(self._on_bulk_process_finished)
        self.runner.start()

    def _on_bulk_progress(self, done, total):
        self.bulk_progress.setMaximum(total)
        self.bulk_progress.setValue(done)

    def _on_bulk_video_done(self, video_path_str):
        # Recorded incrementally (rather than only at the very end) so a
        # later failure in the same batch doesn't leave already-cropped
        # videos looking un-positioned in the cache.
        key = video_key(Path(video_path_str), self.input_folder)
        self.positions[key] = list(self._bulk_xy)
        vc.save_positions(self.positions_path, self.positions)

    def _on_bulk_finished_run(self, status, message):
        suffix = f" -- {message}" if message else ""
        self.bulk_log.appendPlainText(f"Bulk crop finished: {status}{suffix}")

    def _on_bulk_process_finished(self):
        self.runner = None
        self._set_nav_enabled(True)
        self.accept()  # nothing left to position either way, successful or not

    def _set_nav_enabled(self, enabled):
        self.back_btn.setEnabled(enabled and self.idx > 0)
        self.forward_btn.setEnabled(enabled and self.canvas.fits())
        self.skip_btn.setEnabled(enabled)
        self.apply_all_btn.setEnabled(enabled and self.canvas.fits())
        self.width_spin.setEnabled(enabled)
        self.height_spin.setEnabled(enabled)
        self.side_radio.setEnabled(enabled)
        self.bottomup_radio.setEnabled(enabled)
        self.strength_slider.setEnabled(enabled and self.bottomup_radio.isChecked())
        self.strength_label.setEnabled(enabled and self.bottomup_radio.isChecked())

    # -- keyboard shortcuts (dialog-level, so they work regardless of --
    # -- exactly which child widget currently has focus) --

    def keyPressEvent(self, event):
        key = event.key()
        step = 10 if event.modifiers() & Qt.ShiftModifier else 1
        if key == Qt.Key_Left:
            self.canvas.set_position(self.canvas.x - step, self.canvas.y)
        elif key == Qt.Key_Right:
            self.canvas.set_position(self.canvas.x + step, self.canvas.y)
        elif key == Qt.Key_Up:
            self.canvas.set_position(self.canvas.x, self.canvas.y - step)
        elif key == Qt.Key_Down:
            self.canvas.set_position(self.canvas.x, self.canvas.y + step)
        elif key in (Qt.Key_Return, Qt.Key_Enter):
            self._go_forward()
        elif key == Qt.Key_Escape:
            self._skip()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        if self.runner is not None:
            reply = QMessageBox.question(
                self, "Crop in progress",
                "A bulk crop is still running. Close anyway? The current video will be interrupted.",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                event.ignore()
                return
            self.runner.request_stop()
            self.runner.wait(5000)
        event.accept()
