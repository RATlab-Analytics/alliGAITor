"""Crop Videos dialog -- walks through videos one at a time, letting you drag a fixed-size
crop window into place per video, since framing can shift between recordings.

Confirming a video (Forward) crops it immediately on the main thread. "Use This Position for All
Remaining" crops every remaining video in one shot via CropRunner, a separate process. Positions
are cached per video at ``<output_folder>/crop_positions.json``, so reopening this dialog resumes
rather than re-asking for videos already cropped.

The ``mode`` constructor argument controls how the color-correction UI is offered:

  - ``"manual"`` (default): a side/bottom-up radio pair, for the standalone
    ``scripts/crop_tool_main.py`` entry point.
  - ``"bottom"``: a single "Apply bottom-up color correction" checkbox, pre-checked from
    ``default_color_grade``.
  - ``"side"``: no color-correction UI -- ``color_grade`` is always ``False``.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QRectF, QPointF, QSize
from PySide6.QtGui import QPainter, QPen, QColor, QPixmap
from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSpinBox, QMessageBox, QPlainTextEdit, QProgressBar, QApplication,
    QSizePolicy, QRadioButton, QButtonGroup, QSlider, QCheckBox,
)

from frame_utils import video_key, grab_middle_frame, bgr_frame_to_qimage
import video_crop as vc
from video_crop import _BC_LAYERS
from crop_runner import CropRunner

_RECT_COLOR = QColor(255, 140, 0)
_RECT_COLOR_BAD = QColor(200, 40, 40)

# Floor on the letterboxed display scale, so the crop rectangle never shrinks below clickable size.
_MIN_DISPLAY_SCALE = 0.05


class _CropCanvas(QWidget):
    """Shows the loaded preview frame, letterboxed/scaled to fill this widget, with a draggable
    rectangle marking the crop window. self.x/self.y/self.crop_w/self.crop_h are always in native
    frame pixel coordinates; only paintEvent and the mouse handlers convert to/from display scale."""

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
        """(scale, offset_x, offset_y) for the current widget size: scale maps frame-space
        lengths to widget-space, offsets center the letterboxed, aspect-ratio-preserved frame."""
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

    # No keyPressEvent here -- CropSetupDialog handles keyboard shortcuts itself. No resizeEvent
    # needed either -- paintEvent recomputes _display_geometry() from current widget size each time.


class CropSetupDialog(QDialog):
    def __init__(self, video_paths, input_folder, output_folder, tools_dir,
                 width, height, parent=None, force_review=False,
                 mode="manual", default_color_grade=True):
        super().__init__(parent)
        self.setWindowTitle("Crop Setup")
        self.setFocusPolicy(Qt.StrongFocus)

        if mode not in ("manual", "bottom", "side"):
            raise ValueError(f"mode must be 'manual', 'bottom', or 'side', got {mode!r}")
        self.mode = mode

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
                # Everything's already cropped; review all rather than open an empty dialog.
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

        # Color-correction UI, only meaningful for bottom-up (tunnel) footage. Offered as a
        # radio pair in "manual" mode, a single checkbox in "bottom" mode, or not at all in "side".
        self.side_radio = None
        self.bottomup_radio = None
        self.grade_checkbox = None
        if self.mode == "manual":
            self.side_radio = QRadioButton("Side angle")
            self.bottomup_radio = QRadioButton("Bottom-up (apply color correction)")
            self.angle_group = QButtonGroup(self)
            self.angle_group.addButton(self.side_radio)
            self.angle_group.addButton(self.bottomup_radio)
            self.side_radio.setChecked(True)
            self.side_radio.toggled.connect(self._on_grade_changed)
            self.bottomup_radio.toggled.connect(self._on_grade_changed)
        elif self.mode == "bottom":
            self.grade_checkbox = QCheckBox("Apply bottom-up color correction")
            self.grade_checkbox.setChecked(default_color_grade)
            self.grade_checkbox.toggled.connect(self._on_grade_changed)
        # mode == "side": self.color_grade stays False unconditionally.

        # Strength slider: scales apply_bottom_up_color_correction()'s effect linearly,
        # 0% = no-op, 100% = full recipe. Shown/enabled only where grading is offered.
        self.strength_slider = None
        self.strength_label = None
        if self.mode != "side":
            self.strength_slider = QSlider(Qt.Horizontal)
            self.strength_slider.setRange(0, 100)
            self.strength_slider.setValue(100)
            self.strength_label = QLabel("100%")
            self.strength_label.setMinimumWidth(40)
            self.strength_slider.valueChanged.connect(self._on_strength_changed)
            self.strength_slider.setEnabled(self.color_grade)
            self.strength_label.setEnabled(self.color_grade)

        # "Advanced" mode exposes the per-layer brightness/contrast params (video_crop._BC_LAYERS)
        # directly, instead of the single strength knob that scales both layers together.
        self.advanced_checkbox = None
        self.layer_sliders = None  # [(brightness_slider, brightness_label, contrast_slider, contrast_label), ...]
        if self.mode != "side":
            self.advanced_checkbox = QCheckBox("Advanced (per-layer brightness/contrast)")
            self.advanced_checkbox.setChecked(False)
            self.advanced_checkbox.toggled.connect(self._on_advanced_toggled)
            self.advanced_checkbox.setEnabled(self.color_grade)

            self.layer_sliders = []
            for brightness, contrast in _BC_LAYERS:
                b_slider = QSlider(Qt.Horizontal)
                b_slider.setRange(-100, 100)
                b_slider.setValue(int(brightness))
                b_label = QLabel(f"{int(brightness)}")
                b_label.setMinimumWidth(40)
                b_slider.valueChanged.connect(self._on_layer_slider_changed)

                c_slider = QSlider(Qt.Horizontal)
                c_slider.setRange(-100, 100)
                c_slider.setValue(int(contrast))
                c_label = QLabel(f"{int(contrast)}")
                c_label.setMinimumWidth(40)
                c_slider.valueChanged.connect(self._on_layer_slider_changed)

                self.layer_sliders.append((b_slider, b_label, c_slider, c_label))

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

        angle_row = None
        if self.mode == "manual":
            angle_row = QHBoxLayout()
            angle_row.addWidget(QLabel("Camera angle:"))
            angle_row.addWidget(self.side_radio)
            angle_row.addWidget(self.bottomup_radio)
            angle_row.addStretch()
        elif self.mode == "bottom":
            angle_row = QHBoxLayout()
            angle_row.addWidget(self.grade_checkbox)
            angle_row.addStretch()

        strength_row = None
        if self.strength_slider is not None:
            strength_row = QHBoxLayout()
            strength_row.addWidget(QLabel("Correction strength:"))
            strength_row.addWidget(self.strength_slider, stretch=1)
            strength_row.addWidget(self.strength_label)

        advanced_row = None
        layer_rows = None
        if self.advanced_checkbox is not None:
            advanced_row = QHBoxLayout()
            advanced_row.addWidget(self.advanced_checkbox)
            advanced_row.addStretch()

            layer_rows = []
            for i, (b_slider, b_label, c_slider, c_label) in enumerate(self.layer_sliders, start=1):
                row = QHBoxLayout()
                row.addWidget(QLabel(f"Layer {i} brightness:"))
                row.addWidget(b_slider, stretch=1)
                row.addWidget(b_label)
                row.addWidget(QLabel(f"Layer {i} contrast:"))
                row.addWidget(c_slider, stretch=1)
                row.addWidget(c_label)
                # Wrapped in a QWidget so the whole row can be hidden as a unit (QLayout has no setVisible).
                row_widget = QWidget()
                row_widget.setLayout(row)
                row_widget.setVisible(False)  # advanced mode starts off
                layer_rows.append(row_widget)

        nav_row = QHBoxLayout()
        nav_row.addWidget(self.back_btn)
        nav_row.addWidget(self.forward_btn)
        nav_row.addWidget(self.skip_btn)

        layout = QVBoxLayout(self)
        layout.addLayout(size_row)
        if angle_row is not None:
            layout.addLayout(angle_row)
        if strength_row is not None:
            layout.addLayout(strength_row)
        if advanced_row is not None:
            layout.addLayout(advanced_row)
        self.layer_rows = layer_rows
        if layer_rows is not None:
            for row_widget in layer_rows:
                layout.addWidget(row_widget)
        layout.addWidget(self.canvas, stretch=1)  # grows when the dialog is resized
        layout.addWidget(self.info_label)
        layout.addLayout(nav_row)
        layout.addWidget(self.apply_all_btn)
        layout.addWidget(self.bulk_progress)
        layout.addWidget(self.bulk_log)

        # Pick a sensible starting size, capped well under the screen so it opens fully on-screen.
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
        if self.mode == "manual":
            return self.bottomup_radio.isChecked()
        if self.mode == "bottom":
            return self.grade_checkbox.isChecked()
        return False  # "side": grading isn't offered

    @property
    def color_grade_strength(self) -> float:
        if self.strength_slider is None:
            return 0.0
        return self.strength_slider.value() / 100.0

    @property
    def advanced(self) -> bool:
        return self.advanced_checkbox is not None and self.advanced_checkbox.isChecked()

    @property
    def color_grade_layers(self) -> list[tuple[float, float]] | None:
        """Explicit per-layer (brightness, contrast) values from the Advanced sliders, or None to
        fall back to color_grade_strength scaling the default recipe."""
        if not self.advanced or self.layer_sliders is None:
            return None
        return [(b_slider.value(), c_slider.value()) for b_slider, _, c_slider, _ in self.layer_sliders]

    # -- preview --

    def _on_grade_changed(self, checked=None):
        if self.strength_slider is not None:
            self.strength_slider.setEnabled(self.color_grade and not self.advanced)
            self.strength_label.setEnabled(self.color_grade and not self.advanced)
        if self.advanced_checkbox is not None:
            self.advanced_checkbox.setEnabled(self.color_grade)
        self._set_layer_sliders_enabled(self.color_grade and self.advanced)
        self._update_preview()

    def _on_strength_changed(self, value):
        self.strength_label.setText(f"{value}%")
        self._update_preview()

    def _on_advanced_toggled(self, checked):
        if self.strength_slider is not None:
            self.strength_slider.setEnabled(self.color_grade and not checked)
            self.strength_label.setEnabled(self.color_grade and not checked)
        if self.layer_rows is not None:
            for row_widget in self.layer_rows:
                row_widget.setVisible(checked)
        self._set_layer_sliders_enabled(self.color_grade and checked)
        self._update_preview()

    def _on_layer_slider_changed(self, value):
        for b_slider, b_label, c_slider, c_label in self.layer_sliders:
            b_label.setText(str(b_slider.value()))
            c_label.setText(str(c_slider.value()))
        self._update_preview()

    def _set_layer_sliders_enabled(self, enabled):
        if self.layer_sliders is None:
            return
        for b_slider, b_label, c_slider, c_label in self.layer_sliders:
            b_slider.setEnabled(enabled)
            b_label.setEnabled(enabled)
            c_slider.setEnabled(enabled)
            c_label.setEnabled(enabled)

    def _update_preview(self):
        """Re-renders the canvas's displayed frame from the last-loaded raw frame, reflecting
        the current color-grade settings without re-reading the video."""
        if self._current_raw_frame is None:
            return
        frame = self._current_raw_frame
        if self.color_grade:
            frame = vc.apply_bottom_up_color_correction(
                frame, strength=self.color_grade_strength, layers=self.color_grade_layers,
            )
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
            # A frame from partway through the recording, not frame 0, which is often empty.
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
                color_grade_layers=self.color_grade_layers,
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
            color_grade_layers=self.color_grade_layers,
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
        # Recorded incrementally so a later failure doesn't leave already-cropped videos unpositioned.
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
        if self.mode == "manual":
            self.side_radio.setEnabled(enabled)
            self.bottomup_radio.setEnabled(enabled)
        elif self.mode == "bottom":
            self.grade_checkbox.setEnabled(enabled)
        if self.strength_slider is not None:
            self.strength_slider.setEnabled(enabled and self.color_grade and not self.advanced)
            self.strength_label.setEnabled(enabled and self.color_grade and not self.advanced)
        if self.advanced_checkbox is not None:
            self.advanced_checkbox.setEnabled(enabled and self.color_grade)
        self._set_layer_sliders_enabled(enabled and self.color_grade and self.advanced)

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
