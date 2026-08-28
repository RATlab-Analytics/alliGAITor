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
Group config editor: builds one job's config.yaml from a folder of
videos, an id/camera regex pair, a manual camera-role assignment, and a
calibration. Opens automatically after a job is added, and again on
every double-click. Model selection is app-wide (via the Model menu),
not a field here; this dialog just blocks Save until one's picked.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Set

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLineEdit,
    QPushButton, QLabel, QDialogButtonBox, QMessageBox, QFileDialog, QWidget,
    QComboBox, QCheckBox, QTableWidget, QTableWidgetItem, QPlainTextEdit,
    QHeaderView, QScrollArea,
)

from alligaitor.config import CalibrationConfig, DiscoveryConfig, GaitConfig, ModelConfig, PipelineConfig
from alligaitor.discovery import camera_tokens, discover_sessions, find_videos, representative_video_for_token
from frame_utils import bgr_frame_to_qimage, grab_middle_frame
from job_queue import Job
from regex_help import build_regex_help_panel

_UNSAFE_NAME_CHARS = re.compile(r'[\\/:*?"<>|]')
_ROLES = ("left", "right", "bottom")
_ROLE_LABELS = {"left": "Left camera", "right": "Right camera", "bottom": "Bottom camera"}
_BOARD_PRESETS = ("apriltag", "original", "strip")
_THUMB_SIZE = (160, 90)


def _folder_row(initial_text: str = ""):
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    edit = QLineEdit(initial_text)
    edit.setReadOnly(True)
    browse_btn = QPushButton("Browse…")
    layout.addWidget(edit)
    layout.addWidget(browse_btn)
    return row, edit, browse_btn


def _file_row(initial_text: str = ""):
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    edit = QLineEdit(initial_text)
    edit.setReadOnly(True)
    browse_btn = QPushButton("Browse…")
    layout.addWidget(edit)
    layout.addWidget(browse_btn)
    return row, edit, browse_btn


class GroupConfigDialog(QDialog):
    def __init__(self, job: Job, models_dir: Optional[Path], app_data_dir: Path,
                 existing_group_names: Set[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Group Config — {job.group_name}")
        self.setMinimumSize(760, 700)

        self.job = job
        self.app_data_dir = Path(app_data_dir)
        self.models_dir = Path(models_dir) if models_dir else None
        self._existing_group_names = {n.lower() for n in existing_group_names}

        self.saved = False
        self._sessions: List = []
        self._problems: List[str] = []
        self._rat_id_overrides: Dict[str, str] = {}
        self._existing_config: Optional[PipelineConfig] = None
        if job.config_path.exists():
            try:
                self._existing_config = PipelineConfig.from_yaml(job.config_path)
            except Exception:
                self._existing_config = None

        self._build_ui()
        self._load_from_job()
        self._rescan()

    # -- construction --

    def _build_ui(self):
        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll, stretch=1)
        content = QWidget()
        scroll.setWidget(content)
        layout = QVBoxLayout(content)

        # -- group info --
        info_box = QGroupBox("Group")
        info_form = QFormLayout(info_box)
        self.group_name_edit = QLineEdit()
        input_row, self.input_folder_edit, input_browse_btn = _folder_row()
        output_row, self.output_folder_edit, output_browse_btn = _folder_row()
        info_form.addRow("Group name:", self.group_name_edit)
        info_form.addRow("Input video folder:", input_row)
        info_form.addRow("Output folder:", output_row)
        input_browse_btn.clicked.connect(self._browse_input_folder)
        output_browse_btn.clicked.connect(self._browse_output_folder)
        self.skip_validation_check = QCheckBox("Skip validation video generation for this job")
        self.skip_validation_check.setToolTip(
            "Skips rendering an annotated validation video per session on every run of this "
            "job -- the spreadsheet and paw-usability summary are still produced either way."
        )
        info_form.addRow("Validation videos:", self.skip_validation_check)
        self.bottom_fallback_check = QCheckBox("Use bottom-camera fallback for triangulation gaps (experimental)")
        self.bottom_fallback_check.setToolTip(
            "Fills triangulation gaps using the bottom camera's own monocular view wherever it alone "
            "has a valid 2D detection. Can only add usability a paw didn't already have -- a paw run "
            "that already worked without it is left untouched. Defaults to off."
        )
        info_form.addRow("Bottom fallback:", self.bottom_fallback_check)
        layout.addWidget(info_box)

        # -- discovery regexes --
        regex_box = QGroupBox("Session discovery")
        regex_form = QFormLayout(regex_box)
        self.id_regex_edit = QLineEdit()
        self.id_regex_edit.setToolTip("Applied to each filename; group 1 is the session name.")
        self.camera_regex_edit = QLineEdit()
        self.camera_regex_edit.setToolTip("Applied to each filename; group 1 is a camera token (e.g. \"cam0\").")
        regex_form.addRow("ID regex:", self.id_regex_edit)
        regex_form.addRow("Camera regex:", self.camera_regex_edit)
        regex_form.addRow(build_regex_help_panel(self))
        self.id_regex_edit.textChanged.connect(self._rescan)
        self.camera_regex_edit.textChanged.connect(self._on_camera_regex_changed)
        layout.addWidget(regex_box)

        # -- camera roles --
        # Thumbnails are keyed by token, not by role, so previews stay
        # correct regardless of the current dropdown selections.
        roles_box = QGroupBox("Camera roles")
        roles_layout = QVBoxLayout(roles_box)

        roles_layout.addWidget(QLabel("Detected camera tokens (from the camera regex above):"))
        self.token_gallery = QWidget()
        self.token_gallery_layout = QHBoxLayout(self.token_gallery)
        self.token_gallery_layout.setContentsMargins(0, 0, 0, 0)
        roles_layout.addWidget(self.token_gallery)

        # Side by side rather than stacked, to match the gallery above.
        role_row = QHBoxLayout()
        self.role_combos: Dict[str, QComboBox] = {}
        for role in _ROLES:
            role_cell = QVBoxLayout()
            role_cell.addWidget(QLabel(f"{_ROLE_LABELS[role]}:"))
            combo = QComboBox()
            combo.setEditable(True)
            combo.currentTextChanged.connect(self._rescan)
            role_cell.addWidget(combo)
            role_row.addLayout(role_cell)
            self.role_combos[role] = combo
        roles_layout.addLayout(role_row)
        layout.addWidget(roles_box)

        # -- sessions --
        sessions_box = QGroupBox("Sessions")
        sessions_layout = QVBoxLayout(sessions_box)
        self.multi_session_check = QCheckBox("Multiple sessions per rat")
        self.multi_session_check.setToolTip(
            "When checked, edit Rat ID per session below to group multiple sessions (separate "
            "videos) of the same rat onto one spreadsheet tab (with an averaged summary row)."
        )
        self.multi_session_check.toggled.connect(self._on_multi_session_toggled)
        sessions_layout.addWidget(self.multi_session_check)

        self.session_table = QTableWidget(0, 6)
        self.session_table.setHorizontalHeaderLabels(
            ["Session", "Rat ID", "Left video", "Right video", "Bottom video", "Status"]
        )
        self.session_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.session_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.session_table.horizontalHeader().setStretchLastSection(True)
        self.session_table.setColumnHidden(1, True)
        self.session_table.itemChanged.connect(self._on_session_item_changed)
        sessions_layout.addWidget(self.session_table)

        sessions_layout.addWidget(QLabel("Unmatched / incomplete:"))
        self.problems_view = QPlainTextEdit()
        self.problems_view.setReadOnly(True)
        self.problems_view.setMaximumHeight(90)
        sessions_layout.addWidget(self.problems_view)
        layout.addWidget(sessions_box)

        # -- calibration --
        calib_box = QGroupBox("Calibration")
        calib_layout = QVBoxLayout(calib_box)
        self.use_existing_calib_check = QCheckBox("Use an existing calibration file instead of recording new videos")
        self.use_existing_calib_check.toggled.connect(self._on_calib_mode_toggled)
        calib_layout.addWidget(self.use_existing_calib_check)

        self.calib_videos_widget = QWidget()
        calib_videos_form = QFormLayout(self.calib_videos_widget)
        calib_videos_form.setContentsMargins(0, 0, 0, 0)
        self.calib_video_rows: Dict[str, QLineEdit] = {}
        for role in _ROLES:
            row, edit, browse_btn = _file_row()
            browse_btn.clicked.connect(lambda _checked=False, r=role: self._browse_calib_video(r))
            calib_videos_form.addRow(f"{_ROLE_LABELS[role]} calibration video:", row)
            self.calib_video_rows[role] = edit
        self.board_preset_combo = QComboBox()
        self.board_preset_combo.addItems(_BOARD_PRESETS)
        calib_videos_form.addRow("Board preset:", self.board_preset_combo)
        calib_layout.addWidget(self.calib_videos_widget)

        self.calib_existing_row, self.calib_existing_edit, calib_existing_browse_btn = _file_row()
        calib_existing_browse_btn.clicked.connect(self._browse_existing_calibration)
        self.calib_existing_label = QLabel("Calibration file (.toml):")
        calib_existing_form = QFormLayout()
        calib_existing_form.setContentsMargins(0, 0, 0, 0)
        calib_existing_form.addRow(self.calib_existing_label, self.calib_existing_row)
        self.calib_existing_widget = QWidget()
        self.calib_existing_widget.setLayout(calib_existing_form)
        calib_layout.addWidget(self.calib_existing_widget)

        self.calib_output_label = QLabel()
        self.calib_output_label.setStyleSheet("color: #9d9d9d; font-size: 11px;")
        calib_layout.addWidget(self.calib_output_label)
        layout.addWidget(calib_box)

        # -- buttons --
        self.buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self._on_save)
        self.buttons.rejected.connect(self.reject)
        outer.addWidget(self.buttons)

        self._on_calib_mode_toggled(False)

    # -- loading existing state --

    def _load_from_job(self):
        self.group_name_edit.setText(self.job.group_name)
        self.input_folder_edit.setText(self.job.input_folder)
        self.output_folder_edit.setText(self.job.output_folder)

        from app_settings import (
            get_default_bottom_fallback, get_default_camera_tokens, get_default_regexes,
            get_default_skip_validation_videos,
        )
        default_id_regex, default_camera_regex = get_default_regexes(self.app_data_dir)
        self.skip_validation_check.setChecked(
            self._existing_config.skip_validation_videos if self._existing_config is not None
            else get_default_skip_validation_videos(self.app_data_dir)
        )
        self.bottom_fallback_check.setChecked(
            self._existing_config.bottom_fallback if self._existing_config is not None
            else get_default_bottom_fallback(self.app_data_dir)
        )

        # Block textChanged -> _rescan while both fields are set, so a
        # rescan doesn't fire with one field still at its old/empty text.
        self.id_regex_edit.blockSignals(True)
        self.camera_regex_edit.blockSignals(True)
        config = self._existing_config
        if config is not None and config.discovery is not None:
            disc = config.discovery
            self.id_regex_edit.setText(disc.id_regex)
            self.camera_regex_edit.setText(disc.camera_regex)
            self._rat_id_overrides = dict(disc.rat_id_overrides)
            role_to_token = {role: token for token, role in disc.camera_role_map.items()}
        else:
            self.id_regex_edit.setText(default_id_regex)
            self.camera_regex_edit.setText(default_camera_regex)
            role_to_token = get_default_camera_tokens(self.app_data_dir)
        self.id_regex_edit.blockSignals(False)
        self.camera_regex_edit.blockSignals(False)
        self._populate_role_combos(role_to_token=role_to_token)

        has_overrides = bool(self._rat_id_overrides) or any(
            s.rat_id != s.name for s in (config.sessions if config else [])
        )
        self.multi_session_check.setChecked(has_overrides)

        if config is not None:
            calib = config.calibration
            self.board_preset_combo.setCurrentText(calib.board_preset)
            # Save() sets every role's video to calib.output_path itself
            # when "use existing calibration file" was checked -- the
            # signal used to round-trip which mode was in effect.
            used_existing_file = all(
                calib.videos.get(role) == calib.output_path for role in _ROLES
            )
            self.use_existing_calib_check.setChecked(used_existing_file)
            if used_existing_file:
                self.calib_existing_edit.setText(str(calib.output_path))
            else:
                for role in _ROLES:
                    if role in calib.videos:
                        self.calib_video_rows[role].setText(str(calib.videos[role]))

    def _populate_role_combos(self, role_to_token: Dict[str, str]):
        videos = find_videos(self.input_folder_edit.text()) if self.input_folder_edit.text() else []
        tokens = camera_tokens(videos, self.camera_regex_edit.text() or ".^")  # ".^" matches nothing if empty
        for i, role in enumerate(_ROLES):
            combo = self.role_combos[role]
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(tokens)
            preset = role_to_token.get(role)
            if preset:
                combo.setCurrentText(preset)
            elif i < len(tokens):
                combo.setCurrentText(tokens[i])
            combo.blockSignals(False)

    # -- browsing --

    def _browse_input_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Choose the folder of videos for this group", self.input_folder_edit.text() or str(Path.home()),
        )
        if folder:
            self.input_folder_edit.setText(folder)
            self._populate_role_combos(role_to_token={r: c.currentText() for r, c in self.role_combos.items()})
            self._rescan()

    def _browse_output_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Choose where this group's results should be written",
            self.output_folder_edit.text() or str(Path.home()),
        )
        if folder:
            self.output_folder_edit.setText(folder)
            self._rescan()

    def _browse_calib_video(self, role: str):
        path, _ = QFileDialog.getOpenFileName(
            self, f"Choose the {role} calibration video", str(Path.home()), "Video files (*.mp4 *.mov *.avi)"
        )
        if path:
            self.calib_video_rows[role].setText(path)

    def _browse_existing_calibration(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose an existing calibration file", str(Path.home()), "Calibration files (*.toml)"
        )
        if path:
            self.calib_existing_edit.setText(path)

    # -- calibration mode --

    def _on_calib_mode_toggled(self, checked: bool):
        self.calib_videos_widget.setVisible(not checked)
        self.calib_existing_widget.setVisible(checked)
        self._update_calib_output_label()

    def _update_calib_output_label(self):
        if self.use_existing_calib_check.isChecked():
            self.calib_output_label.setText("Loaded directly -- no calibration will be re-run.")
        else:
            out = Path(self.output_folder_edit.text() or ".") / "calibration.toml"
            self.calib_output_label.setText(f"Will be written to: {out}")

    # -- camera roles / discovery --

    def _on_camera_regex_changed(self, text: str):
        self._populate_role_combos(role_to_token={r: c.currentText() for r, c in self.role_combos.items()})
        self._rescan()

    def _on_multi_session_toggled(self, checked: bool):
        self.session_table.setColumnHidden(1, not checked)
        if not checked:
            self._rat_id_overrides.clear()
        self._rescan()

    def _on_session_item_changed(self, item: QTableWidgetItem):
        if item.column() != 1:
            return
        session_name = self.session_table.item(item.row(), 0).text()
        new_rat_id = item.text().strip() or session_name
        if new_rat_id == session_name:
            self._rat_id_overrides.pop(session_name, None)
        else:
            self._rat_id_overrides[session_name] = new_rat_id

    def _current_discovery(self) -> Optional[DiscoveryConfig]:
        input_folder = self.input_folder_edit.text().strip()
        if not input_folder:
            return None
        role_map = {}
        for role in _ROLES:
            token = self.role_combos[role].currentText().strip()
            if token:
                role_map[token] = role
        try:
            re.compile(self.id_regex_edit.text())
            re.compile(self.camera_regex_edit.text())
        except re.error:
            return None
        return DiscoveryConfig(
            input_dir=Path(input_folder),
            id_regex=self.id_regex_edit.text(),
            camera_regex=self.camera_regex_edit.text(),
            camera_role_map=role_map,
            rat_id_overrides=dict(self._rat_id_overrides),
        )

    def _rescan(self):
        discovery = self._current_discovery()
        self._update_token_gallery(discovery)
        self._update_calib_output_label()

        if discovery is None or not Path(discovery.input_dir).is_dir():
            self._sessions, self._problems = [], []
            self._refresh_session_table()
            return

        job_output = self.output_folder_edit.text().strip() or self.job.output_folder
        cropped_dir = Path(job_output) / "cropped"
        predictions_dir = Path(job_output) / "predictions_3d"
        self._sessions, self._problems = discover_sessions(discovery, cropped_dir, predictions_dir)
        self._refresh_session_table()

    def _update_token_gallery(self, discovery: Optional[DiscoveryConfig]):
        """Rebuilds the token->thumbnail gallery: one cell per distinct
        camera token found in the input folder, each with a preview
        frame from a representative video."""
        while self.token_gallery_layout.count():
            item = self.token_gallery_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        videos = find_videos(discovery.input_dir) if discovery is not None else []
        camera_regex = self.camera_regex_edit.text() or ".^"
        tokens = camera_tokens(videos, camera_regex)

        if not tokens:
            placeholder = QLabel("(no camera tokens detected yet)")
            placeholder.setStyleSheet("color: #9d9d9d;")
            self.token_gallery_layout.addWidget(placeholder)
            return

        for token in tokens:
            cell = QWidget()
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(4, 4, 4, 4)

            thumb = QLabel()
            thumb.setFixedSize(*_THUMB_SIZE)
            thumb.setStyleSheet("background: #222; color: #888;")
            thumb.setAlignment(Qt.AlignCenter)
            video = representative_video_for_token(videos, camera_regex, token)
            if video is None:
                thumb.setText("no preview")
            else:
                try:
                    frame = grab_middle_frame(video)
                    qimage = bgr_frame_to_qimage(frame)
                    pixmap = QPixmap.fromImage(qimage).scaled(
                        *_THUMB_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation
                    )
                    thumb.setPixmap(pixmap)
                except Exception:
                    thumb.setText("preview failed")

            token_label = QLabel(token)
            token_label.setAlignment(Qt.AlignCenter)
            token_label.setStyleSheet("font-weight: bold;")

            cell_layout.addWidget(thumb)
            cell_layout.addWidget(token_label)
            self.token_gallery_layout.addWidget(cell)

        self.token_gallery_layout.addStretch()

    def _refresh_session_table(self):
        self.session_table.blockSignals(True)
        self.session_table.setRowCount(len(self._sessions))
        for row, session in enumerate(self._sessions):
            name_item = QTableWidgetItem(session.name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.session_table.setItem(row, 0, name_item)

            rat_item = QTableWidgetItem(session.rat_id)
            self.session_table.setItem(row, 1, rat_item)

            for col, role in ((2, "left"), (3, "right"), (4, "bottom")):
                path_item = QTableWidgetItem(session.videos[role].name)
                path_item.setFlags(path_item.flags() & ~Qt.ItemIsEditable)
                self.session_table.setItem(row, col, path_item)

            status_item = QTableWidgetItem("complete")
            status_item.setFlags(status_item.flags() & ~Qt.ItemIsEditable)
            self.session_table.setItem(row, 5, status_item)
        self.session_table.blockSignals(False)
        self.problems_view.setPlainText("\n".join(self._problems) if self._problems else "(none)")

    # -- save --

    def _on_save(self):
        group_name = self.group_name_edit.text().strip()
        input_folder = self.input_folder_edit.text().strip()
        output_folder = self.output_folder_edit.text().strip()

        if not group_name or not input_folder or not output_folder:
            QMessageBox.warning(self, "Missing info", "Group name, input folder, and output folder are all required.")
            return
        if _UNSAFE_NAME_CHARS.search(group_name):
            QMessageBox.warning(self, "Invalid group name", 'Group name can\'t contain any of: \\ / : * ? " < > |')
            return
        if group_name.lower() in self._existing_group_names and group_name.lower() != self.job.group_name.lower():
            QMessageBox.warning(self, "Duplicate group", f"A job named '{group_name}' is already in the queue.")
            return
        if not Path(input_folder).is_dir():
            QMessageBox.warning(self, "Folder not found", f"Input folder doesn't exist:\n{input_folder}")
            return

        tokens_used = [self.role_combos[r].currentText().strip() for r in _ROLES]
        if not all(tokens_used):
            QMessageBox.warning(self, "Missing camera role", "Every camera role needs a token assigned.")
            return
        if len(set(tokens_used)) != len(tokens_used):
            QMessageBox.warning(self, "Duplicate camera token", "Each camera role must use a different token.")
            return

        discovery = self._current_discovery()
        if discovery is None:
            QMessageBox.warning(self, "Invalid regex", "The ID regex and camera regex must both be valid.")
            return

        from app_settings import get_default_gait_overrides, get_default_min_corners_extrinsic, get_selected_model
        side_model = get_selected_model(self.app_data_dir, "side")
        bottom_model = get_selected_model(self.app_data_dir, "bottom")
        if not side_model or not bottom_model:
            QMessageBox.warning(
                self, "No model selected",
                "Pick a side model and a bottom model from the Model menu before saving a job.",
            )
            return
        models = ModelConfig(
            side_model_dir=self.models_dir / side_model,
            bottom_model_dir=self.models_dir / bottom_model,
        )

        if self.use_existing_calib_check.isChecked():
            existing_path = self.calib_existing_edit.text().strip()
            if not existing_path or not Path(existing_path).exists():
                QMessageBox.warning(self, "Calibration file missing", "Pick an existing calibration (.toml) file.")
                return
            output_path = Path(existing_path).resolve()
            # These paths are never read; only the output_path is used.
            calib_videos = {role: output_path for role in _ROLES}
        else:
            calib_videos = {}
            for role in _ROLES:
                text = self.calib_video_rows[role].text().strip()
                if not text:
                    QMessageBox.warning(self, "Missing calibration video", f"Pick the {role} calibration video.")
                    return
                calib_videos[role] = Path(text).resolve()
            output_path = Path(output_folder) / "calibration.toml"
        calibration = CalibrationConfig(
            videos=calib_videos,
            output_path=output_path,
            board_preset=self.board_preset_combo.currentText(),
            min_corners_extrinsic=get_default_min_corners_extrinsic(self.app_data_dir),
        )

        cropped_dir = Path(output_folder) / "cropped"
        predictions_dir = Path(output_folder) / "predictions_3d"
        sessions, problems = discover_sessions(discovery, cropped_dir, predictions_dir)
        if not sessions:
            QMessageBox.warning(
                self, "No complete sessions",
                "No session has a video for every camera role yet -- check the regexes, "
                "camera role assignment, and the Unmatched panel.",
            )
            return
        if problems:
            reply = QMessageBox.question(
                self, "Unmatched videos",
                f"{len(problems)} video(s)/session(s) couldn't be grouped and will be skipped. Save anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        config = PipelineConfig(
            models=models,
            calibration=calibration,
            sessions=sessions,
            name=group_name,
            output_xlsx=Path(output_folder) / "reports" / f"{group_name}.gait_metrics.xlsx",
            gait=GaitConfig(**get_default_gait_overrides(self.app_data_dir)),
            discovery=discovery,
            skip_validation_videos=self.skip_validation_check.isChecked(),
            bottom_fallback=self.bottom_fallback_check.isChecked(),
        )

        # Update job fields (config_path derives from output_folder) before writing.
        self.job.group_name = group_name
        self.job.input_folder = input_folder
        self.job.output_folder = output_folder
        config.to_yaml(self.job.config_path)

        self.saved = True
        self.accept()
