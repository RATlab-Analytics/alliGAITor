"""
Main window shell for the alliGAITor GUI: job queue table, Load/Edit/
Remove/Run controls (both a toolbar row and mirrored menu actions), a
log panel, and a progress bar + ETA shown while a run is in flight.

Ported from RATlab-NOR's gui/main_window.py's overall shape (menu bar,
table+log+progress layout, BatchRunner wiring, Reset menu), with
alliGAITor's two setup gates (config editor, then crop) replacing NOR's
one (object-coordinate setup), and Model/Settings/Help menus built for
alliGAITor's own needs (two model roles, discovery regex defaults, a
GPLv3 About dialog).
"""

from __future__ import annotations

import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QTimer, QModelIndex
from PySide6.QtGui import QActionGroup, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTableView,
    QPushButton, QTextEdit, QDialog, QDialogButtonBox, QFormLayout,
    QLineEdit, QMessageBox, QAbstractItemView, QSplitter,
    QProgressBar, QLabel, QTabWidget, QSpinBox, QDoubleSpinBox, QMenu,
    QCheckBox,
)

from job_queue import Job, JobQueue, JobStatus, refresh_job_readiness
from job_table_model import JobTableModel
from add_job_dialog import AddJobDialog
from group_config_dialog import GroupConfigDialog
from batch_runner import BatchRunner
from about_dialog import AboutDialog
from validation_list_dialog import ValidationListDialog
import app_settings
import reset as reset_module

from alligaitor.config import PipelineConfig
from alligaitor.discovery import find_videos

from crop_setup_dialog import CropSetupDialog
from crop_config import CROP_TARGET_WIDTH, CROP_TARGET_HEIGHT, side_crop_size_for_model
from regex_help import build_regex_help_panel


def _double_spin(value: float, minimum: float, maximum: float, decimals: int = 1, suffix: str = "",
                 tooltip: str = "", step: Optional[float] = None):
    box = QDoubleSpinBox()
    box.setRange(minimum, maximum)
    box.setDecimals(decimals)
    box.setSuffix(suffix)
    if step is not None:
        box.setSingleStep(step)
    box.setValue(value)
    if tooltip:
        box.setToolTip(tooltip)
    return box


def _int_spin(value: int, minimum: int, maximum: int, suffix: str = "", tooltip: str = ""):
    box = QSpinBox()
    box.setRange(minimum, maximum)
    box.setSuffix(suffix)
    box.setValue(value)
    if tooltip:
        box.setToolTip(tooltip)
    return box


class _PreferencesDialog(QDialog):
    """Settings > Preferences, in two tabs:

    - **General**: the id/camera regex and per-role default tokens a
      newly loaded job's config editor is pre-filled with, and the
      default output-base folder offered by the Load Job dialog. The
      default tokens (e.g. "cam0" for Left) are a separate setting from
      the camera regex itself: the regex says how to *extract* a token
      from a filename, the tokens say which extracted value this lab
      expects for each role, so a new job's role combos start pre-filled
      with an actual guess instead of an arbitrary
      first/second/third-discovered-token fallback that has no
      connection to which camera is which.
    - **Scoring & Triangulation**: the stance/swing detection thresholds
      (alligaitor.config.GaitConfig) and the calibration
      min_corners_extrinsic tunable, baked into every newly-saved job's
      config.yaml the same way the selected models are (see
      group_config_dialog.py's _on_save) -- not something an individual
      job overrides in the editor UI, just what "new job" starts from.

    Small enough to keep inline here rather than its own module.
    """

    def __init__(self, app_data_dir: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.app_data_dir = app_data_dir

        # Built before the tabs' own content, since the regex help panel
        # inside the General tab needs to call back into
        # _sync_dialog_size(), which needs self._tabs to already exist.
        self._tabs = QTabWidget()
        self._tabs.currentChanged.connect(lambda _index: QTimer.singleShot(0, self._sync_dialog_size))

        self._tabs.addTab(self._build_general_tab(), "General")
        self._tabs.addTab(self._build_scoring_tab(), "Scoring && Triangulation")

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self._tabs)
        layout.addWidget(buttons)

    def _sync_dialog_size(self):
        """Resizes this dialog to fit whichever tab is current.

        QTabWidget's internal QStackedWidget is documented Qt behavior
        to size itself for the *largest* page it has ever shown, not the
        current one -- so switching to a shorter tab, or collapsing the
        regex help panel back down within the current tab, leaves the
        dialog oversized under plain sizeHint()-based resizing (see
        shrink_window_to_fit's docstring; that approach alone isn't
        enough here). The only reliable fix is to bypass QStackedWidget's
        own sizeHint entirely: give the tab widget an explicit fixed
        height computed from the current page's own sizeHint(), which --
        unlike the tab widget's -- always stays accurate regardless of
        what other tabs have been shown.
        """
        page = self._tabs.currentWidget()
        if page is None:
            return
        tabbar_h = self._tabs.tabBar().sizeHint().height()
        frame_padding = 20  # tab frame border; small and stable across styles
        self._tabs.setFixedHeight(page.sizeHint().height() + tabbar_h + frame_padding)
        self.layout().activate()
        self.resize(self.width(), self.layout().sizeHint().height())

    def _build_general_tab(self) -> QWidget:
        id_regex, camera_regex = app_settings.get_default_regexes(self.app_data_dir)
        tokens = app_settings.get_default_camera_tokens(self.app_data_dir)
        self.id_regex_edit = QLineEdit(id_regex)
        self.camera_regex_edit = QLineEdit(camera_regex)
        self.left_token_edit = QLineEdit(tokens["left"])
        self.right_token_edit = QLineEdit(tokens["right"])
        self.bottom_token_edit = QLineEdit(tokens["bottom"])
        self.output_base_edit = QLineEdit(app_settings.get_default_output_base(self.app_data_dir))
        self.skip_validation_check = QCheckBox("Skip validation video generation")
        self.skip_validation_check.setChecked(app_settings.get_default_skip_validation_videos(self.app_data_dir))
        self.skip_validation_check.setToolTip(
            "Starting default for a newly-saved job's config editor -- still editable per job there. "
            "Unchecked means every run renders an annotated validation video per session."
        )
        self.bottom_fallback_check = QCheckBox("Use bottom-camera fallback for triangulation gaps (experimental)")
        self.bottom_fallback_check.setChecked(app_settings.get_default_bottom_fallback(self.app_data_dir))
        self.bottom_fallback_check.setToolTip(
            "Starting default for a newly-saved job's config editor -- still editable per job there. "
            "Fills triangulation gaps using the bottom camera's own monocular view wherever it alone "
            "has a valid 2D detection; can only add usability a paw didn't already have."
        )

        form = QFormLayout()
        form.addRow("Default ID regex:", self.id_regex_edit)
        form.addRow("Default camera regex:", self.camera_regex_edit)
        form.addRow("Default token — Left camera:", self.left_token_edit)
        form.addRow("Default token — Right camera:", self.right_token_edit)
        form.addRow("Default token — Bottom camera:", self.bottom_token_edit)
        form.addRow("Default output base folder:", self.output_base_edit)
        form.addRow("Validation videos:", self.skip_validation_check)
        form.addRow("Bottom fallback:", self.bottom_fallback_check)
        form.addRow(build_regex_help_panel(self, on_toggled=self._sync_dialog_size))

        tab = QWidget()
        tab.setLayout(form)
        return tab

    def _build_scoring_tab(self) -> QWidget:
        gait = app_settings.get_default_gait_overrides(self.app_data_dir)
        min_corners = app_settings.get_default_min_corners_extrinsic(self.app_data_dir)

        self.speed_threshold_spin = _double_spin(
            gait["speed_threshold_mm_s"], 0, 100000, 1, " mm/s",
            "Maximum frame-to-frame speed for a paw to count as planted.",
        )
        self.min_contact_frames_spin = _int_spin(
            gait["min_contact_frames"], 0, 1000, "",
            "Minimum consecutive frames a paw must stay under the speed threshold to count as a real stance.",
        )
        self.max_bridge_gap_spin = _int_spin(
            gait["max_bridge_gap_frames"], 0, 1000, "",
            "Untriangulated runs up to this many frames are interpolated before stance is computed. 0 disables bridging.",
        )
        self.min_consecutive_strides_spin = _int_spin(
            gait["min_consecutive_strides"], 0, 1000, "",
            "A paw's reported averages only use strides from a run of at least this many consecutive clean stances.",
        )
        self.stillness_window_spin = _double_spin(
            gait["stillness_window_seconds"], 0.01, 60, 2, " s",
            "Whole-body speed is measured across a window this wide, so reconstruction jitter at rest "
            "doesn't read as motion.",
            step=0.05,
        )
        self.stillness_threshold_spin = _double_spin(
            gait["stillness_window_speed_mm_s"], 0, 100000, 1, " mm/s",
            "Below this windowed whole-body speed, the rat counts as not translating (for trimming a "
            "stationary start/end).",
        )
        self.min_still_seconds_spin = _double_spin(
            gait["min_still_seconds"], 0, 600, 2, " s",
            "How long the rat must stay stationary at the very start/end of a trial to trim it as \"stopped\".",
            step=0.05,
        )
        self.min_valid_steps_spin = _int_spin(
            gait["min_valid_steps"], 0, 1000, "",
            "Fewest valid steps a paw needs before its average step length is reported; below this "
            "it is blank and flagged on its own.",
        )
        self.stride_outlier_spin = _double_spin(
            gait["stride_length_outlier_ratio"], 0.1, 100, 2, "×",
            "A stride longer than this many times a paw's own median stride length is flagged as a likely missed stance.",
        )
        self.min_corners_extrinsic_spin = _int_spin(
            min_corners, 1, 10000, "",
            "Minimum matched points a frame needs to link two cameras' poses during AprilTag calibration.",
        )

        gait_form = QFormLayout()
        gait_form.addRow("Speed threshold (planted):", self.speed_threshold_spin)
        gait_form.addRow("Min contact frames:", self.min_contact_frames_spin)
        gait_form.addRow("Max bridge gap:", self.max_bridge_gap_spin)
        gait_form.addRow("Min consecutive strides:", self.min_consecutive_strides_spin)
        gait_form.addRow("Stillness window:", self.stillness_window_spin)
        gait_form.addRow("Stillness speed threshold:", self.stillness_threshold_spin)
        gait_form.addRow("Min still duration:", self.min_still_seconds_spin)
        gait_form.addRow("Min valid steps (step length):", self.min_valid_steps_spin)
        gait_form.addRow("Stride length outlier ratio:", self.stride_outlier_spin)

        triangulation_form = QFormLayout()
        triangulation_form.addRow("Min corners (extrinsic, AprilTag):", self.min_corners_extrinsic_spin)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("<b>Scoring</b> (stance/swing detection defaults for new jobs)"))
        layout.addLayout(gait_form)
        layout.addWidget(QLabel("<b>Triangulation</b> (calibration defaults for new jobs)"))
        layout.addLayout(triangulation_form)
        layout.addStretch()

        tab = QWidget()
        tab.setLayout(layout)
        return tab

    def _on_save(self):
        app_settings.set_default_regexes(
            self.app_data_dir, self.id_regex_edit.text(), self.camera_regex_edit.text()
        )
        app_settings.set_default_camera_tokens(self.app_data_dir, {
            "left": self.left_token_edit.text().strip(),
            "right": self.right_token_edit.text().strip(),
            "bottom": self.bottom_token_edit.text().strip(),
        })
        app_settings.set_default_output_base(self.app_data_dir, self.output_base_edit.text())
        app_settings.set_default_skip_validation_videos(self.app_data_dir, self.skip_validation_check.isChecked())
        app_settings.set_default_bottom_fallback(self.app_data_dir, self.bottom_fallback_check.isChecked())
        app_settings.set_default_gait_overrides(self.app_data_dir, {
            "speed_threshold_mm_s": self.speed_threshold_spin.value(),
            "min_contact_frames": self.min_contact_frames_spin.value(),
            "max_bridge_gap_frames": self.max_bridge_gap_spin.value(),
            "min_consecutive_strides": self.min_consecutive_strides_spin.value(),
            "stillness_window_seconds": self.stillness_window_spin.value(),
            "stillness_window_speed_mm_s": self.stillness_threshold_spin.value(),
            "min_still_seconds": self.min_still_seconds_spin.value(),
            "min_valid_steps": self.min_valid_steps_spin.value(),
            "stride_length_outlier_ratio": self.stride_outlier_spin.value(),
        })
        app_settings.set_default_min_corners_extrinsic(self.app_data_dir, self.min_corners_extrinsic_spin.value())
        self.accept()


class MainWindow(QMainWindow):
    def __init__(self, job_queue: JobQueue, repo_dir: Path, parent=None):
        super().__init__(parent)
        self.job_queue = job_queue
        self.repo_dir = Path(repo_dir)
        self.app_data_dir = job_queue.app_data_dir
        self.runner: BatchRunner | None = None

        # Overall-run progress tracking (every job in the current run,
        # not just the job currently executing), keyed by job id.
        self._run_total_by_job: dict = {}
        self._run_done_by_job: dict = {}
        self._run_start_time: float | None = None
        # ETA is a countdown snapshot, re-anchored only when real
        # progress happens -- see _recompute_eta_snapshot's docstring for
        # why recomputing the estimate on every 1Hz tick (against an
        # ever-growing elapsed time with a numerator that's fixed between
        # updates) made the displayed ETA count up instead of down.
        self._eta_remaining_s: float | None = None
        self._eta_snapshot_time: float | None = None
        # EMA of whole-session wall-clock duration, measured between
        # consecutive _on_job_progress calls (a session finishing) --
        # deliberately session-granularity only, not per-bar/per-video.
        # A finer-grained ETA (tracking each camera role's own inference
        # progress live, accounting for bottom running faster than side,
        # video-length differences between sessions, etc.) was tried and
        # scrapped: it needed identifying *which* role a given progress
        # bar belongs to, which nothing in the progress text actually
        # says (see alligaitor.pipeline.run_session -- no per-role log
        # message precedes each role's inference), so the correlation
        # was too fragile to trust. Per-session is coarser (only updates
        # once per session, which can be minutes) but robust: real data,
        # no guessing at bar identity.
        self._eta_session_duration_ema: float | None = None
        self._eta_last_session_done_time: float | None = None

        # Whether the last line written to the log panel is a live
        # progress-bar redraw (as opposed to a discrete log message) --
        # see _on_progress_line/_log. Lets repeated redraws of "the same"
        # tqdm line overwrite each other in place instead of each one
        # adding a new line.
        self._progress_line_open = False

        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(1000)
        self._progress_timer.timeout.connect(self._update_progress_ui)

        self.setWindowTitle("alliGAITor")
        self.resize(1080, 620)

        self._build_menu_bar()

        self.model = JobTableModel(job_queue)
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        self.table.doubleClicked.connect(lambda _index: self._on_table_double_clicked())
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._on_table_context_menu)

        # QTextEdit rather than QPlainTextEdit: sleap-nn's tqdm progress
        # bar is ANSI-colored, and rendering that (see
        # alligaitor.ansi_html/_on_progress_line) needs a rich-text
        # widget. Dark background + monospace font so the colors read
        # the way they would in a real terminal and the bar's
        # '|####..|'-style characters stay aligned; QTextEdit has no
        # setMaximumBlockCount (that's QPlainTextEdit-only), so
        # _trim_log_panel() does the equivalent by hand.
        self.log_panel = QTextEdit()
        self.log_panel.setReadOnly(True)
        self.log_panel.setPlaceholderText("Run log will appear here…")
        self.log_panel.setStyleSheet(
            "QTextEdit {"
            " background-color: #1e1e1e; color: #d4d4d4;"
            " font-family: Menlo, Consolas, 'Courier New', monospace;"
            " font-size: 12px;"
            "}"
        )

        self.load_btn = QPushButton("Load Jobs…")
        self.edit_btn = QPushButton("Edit Job…")
        self.remove_btn = QPushButton("Remove Selected")
        self.run_selected_btn = QPushButton("Run Selected")
        self.run_btn = QPushButton("Run All")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)

        self.load_btn.clicked.connect(self._on_load_jobs)
        self.edit_btn.clicked.connect(self._on_edit_job)
        self.remove_btn.clicked.connect(self._on_remove_selected)
        self.run_selected_btn.clicked.connect(self._on_run_selected)
        self.run_btn.clicked.connect(self._on_run_queue)
        self.stop_btn.clicked.connect(self._on_stop_queue)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.load_btn)
        btn_row.addWidget(self.edit_btn)
        btn_row.addWidget(self.remove_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.run_selected_btn)
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.stop_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%v / %m sessions (%p%)")
        self.progress_bar.setVisible(False)
        self.eta_label = QLabel("")
        self.eta_label.setVisible(False)
        progress_row = QHBoxLayout()
        progress_row.addWidget(self.progress_bar, stretch=1)
        progress_row.addWidget(self.eta_label)

        top = QWidget()
        top_layout = QVBoxLayout(top)
        # No left/right margins: `top` (holding the button row and the
        # table) and log_panel are sibling children of the splitter
        # below, but only `top` wraps its content in its own QVBoxLayout
        # -- that layout's default margins were insetting the table by
        # ~9px on each side while log_panel, added straight to the
        # splitter with no such wrapper, had none, so the two ended up
        # different widths despite both notionally spanning "the window".
        top_layout.setContentsMargins(0, top_layout.contentsMargins().top(), 0, top_layout.contentsMargins().bottom())
        top_layout.addLayout(btn_row)
        top_layout.addLayout(progress_row)
        top_layout.addWidget(self.table)

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(top)
        splitter.addWidget(self.log_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.addWidget(splitter)
        self.setCentralWidget(central)

        self.statusBar().showMessage(f"{len(self.job_queue.jobs)} job(s) queued")
        self._log(f"Loaded queue from {self.job_queue.path}")

    # -- menu bar --

    def _build_menu_bar(self):
        job_menu = self.menuBar().addMenu("Job")
        job_menu.addAction("Load Jobs…", self._on_load_jobs)
        job_menu.addAction("Edit Job…", self._on_edit_job)
        job_menu.addAction("Remove Selected", self._on_remove_selected)
        job_menu.addSeparator()
        job_menu.addAction("Re-crop Selected…", self._on_recrop_selected)
        job_menu.addSeparator()
        job_menu.addAction("Retry Failed Job(s)…", self._on_retry_failed)

        reset_menu = self.menuBar().addMenu("Reset")
        reset_menu.addAction(
            "Selected Job(s): Predictions + 3D Output + Report…",
            lambda: self._on_reset_selected(clear_predictions=True, clear_output=True, clear_crops=False),
        )
        reset_menu.addAction(
            "Selected Job(s): 3D Output + Report Only…",
            lambda: self._on_reset_selected(clear_predictions=False, clear_output=True, clear_crops=False),
        )
        reset_menu.addAction(
            "Selected Job(s): Everything…",
            lambda: self._on_reset_selected(clear_predictions=True, clear_output=True, clear_crops=True),
        )

        model_menu = self.menuBar().addMenu("Model")
        self._build_role_model_submenu(model_menu, "Side Model", "side")
        self._build_role_model_submenu(model_menu, "Bottom Model", "bottom")

        run_menu = self.menuBar().addMenu("Run")
        run_menu.addAction("Run All", self._on_run_queue)
        run_menu.addAction("Run Selected Job(s)", self._on_run_selected)
        run_menu.addSeparator()
        run_menu.addAction("Stop", self._on_stop_queue)

        settings_menu = self.menuBar().addMenu("Settings")
        settings_menu.addAction("Preferences…", self._on_open_preferences)

        help_menu = self.menuBar().addMenu("Help")
        help_menu.addAction("About alliGAITor…", self._on_open_about)

    def _build_role_model_submenu(self, parent_menu, label: str, role: str):
        submenu = parent_menu.addMenu(label)
        models_dir = self.repo_dir / "models"
        candidates = sorted(
            p.name for p in models_dir.iterdir() if p.is_dir() and role in p.name.lower()
        ) if models_dir.is_dir() else []
        if not candidates:
            action = submenu.addAction(f"(no {role} models found in models/)")
            action.setEnabled(False)
            return
        current = app_settings.get_selected_model(self.app_data_dir, role)
        group = QActionGroup(self)
        group.setExclusive(True)
        for name in candidates:
            action = submenu.addAction(name)
            action.setCheckable(True)
            action.setChecked(name == current)
            action.triggered.connect(lambda _checked=False, n=name, r=role: self._on_select_model(r, n))
            group.addAction(action)

    def _on_select_model(self, role: str, name: str):
        app_settings.set_selected_model(self.app_data_dir, role, name)
        updated = self._propagate_model_selection(role, name)
        suffix = f" -- updated {updated} saved job(s)." if updated else " -- no saved jobs to update."
        self._log(f"{role.capitalize()} model set to '{name}'{suffix}")

    def _propagate_model_selection(self, role: str, name: str) -> int:
        """Writes the newly selected model straight into every already-saved
        job's config.yaml, rather than leaving it to only apply the next time
        each job's config editor happens to be re-saved -- checking a model in
        this menu should mean every job now runs with it, not silently keep
        stale jobs on whatever model they were saved with last."""
        field = f"{role}_model_dir"
        model_path = self.repo_dir / "models" / name
        updated = 0
        for job in self.job_queue.jobs:
            if job.status in (JobStatus.RUNNING, JobStatus.QUEUED):
                continue  # never rewrite the config out from under an active/pending run
            if not job.config_path.exists():
                continue
            try:
                config = PipelineConfig.from_yaml(job.config_path)
            except Exception:
                continue
            if getattr(config.models, field) == model_path:
                continue
            config.models = replace(config.models, **{field: model_path})
            config.to_yaml(job.config_path)
            refresh_job_readiness(job)
            self.job_queue.update(job)
            updated += 1
        if updated:
            self.model.refresh()
        return updated

    def _on_open_preferences(self):
        dialog = _PreferencesDialog(self.app_data_dir, parent=self)
        if dialog.exec() == QDialog.Accepted:
            self._log("Preferences saved.")

    def _on_open_about(self):
        AboutDialog(parent=self).exec()

    # -- job selection helpers --

    def _selected_jobs(self) -> list[Job]:
        rows = sorted({index.row() for index in self.table.selectionModel().selectedRows()})
        return [self.model.job_at(r) for r in rows]

    def _selected_job(self) -> Job | None:
        jobs = self._selected_jobs()
        return jobs[0] if len(jobs) == 1 else None

    # -- right-click menu --

    def _on_table_context_menu(self, pos):
        """Right-click menu on the job table: Edit / Crop / Run / Remove
        / Reset, all scoped to whichever job(s) the click applies to --
        same underlying handlers as the toolbar buttons and menu bar
        actions, just reachable without leaving the row."""
        index = self.table.indexAt(pos)
        if not index.isValid():
            return

        # Right-clicking a row outside the current selection selects just
        # that row first (standard table convention) -- right-clicking
        # within an existing multi-row selection leaves it alone, so a
        # batch action (e.g. Reset) can still apply to all of them.
        selection_model = self.table.selectionModel()
        if not selection_model.isRowSelected(index.row(), QModelIndex()):
            self.table.selectRow(index.row())

        jobs = self._selected_jobs()
        if not jobs:
            return

        menu = self._build_job_context_menu(jobs)
        menu.exec(self.table.viewport().mapToGlobal(pos))

    def _build_job_context_menu(self, jobs: list[Job]) -> QMenu:
        """Split out from _on_table_context_menu so it can be constructed
        (and its contents inspected) without also invoking the blocking,
        modal QMenu.exec() -- handy for tests, and keeps the "what's in
        the menu" logic separate from "where/when it pops up"."""
        menu = QMenu(self)
        menu.addAction("Edit Job…", self._on_edit_job)
        menu.addAction("Crop…", self._on_recrop_selected)
        menu.addSeparator()
        menu.addAction("Run Selected Job(s)", self._on_run_selected)
        retry_action = menu.addAction("Retry Failed Job(s)…", self._on_retry_failed)
        retry_action.setEnabled(any(j.status == JobStatus.FAILED for j in jobs))
        menu.addSeparator()
        validation_action = menu.addAction("View Validation…", self._on_view_validation_selected)
        validation_action.setEnabled(len(jobs) == 1 and jobs[0].status == JobStatus.DONE)
        menu.addSeparator()
        menu.addAction("Remove Selected", self._on_remove_selected)
        menu.addSeparator()

        reset_menu = menu.addMenu("Reset")
        reset_menu.addAction(
            "Predictions + 3D Output + Report…",
            lambda: self._on_reset_selected(clear_predictions=True, clear_output=True, clear_crops=False),
        )
        reset_menu.addAction(
            "3D Output + Report Only…",
            lambda: self._on_reset_selected(clear_predictions=False, clear_output=True, clear_crops=False),
        )
        reset_menu.addAction(
            "Everything…",
            lambda: self._on_reset_selected(clear_predictions=True, clear_output=True, clear_crops=True),
        )

        return menu

    # -- load / edit / remove --

    def _on_load_jobs(self):
        existing_names = {j.group_name for j in self.job_queue.jobs}
        default_output_base = app_settings.get_default_output_base(self.app_data_dir)
        dialog = AddJobDialog(default_output_base, existing_names, parent=self)
        if dialog.exec() != QDialog.Accepted or dialog.result_job_kwargs is None:
            return
        job = Job(**dialog.result_job_kwargs)
        self.job_queue.add(job)
        self.model.refresh()
        self.statusBar().showMessage(f"{len(self.job_queue.jobs)} job(s) queued")
        self._log(f"Loaded job '{job.group_name}'.")
        self._open_config_editor(job)  # opens on first load, per the design brief

    def _open_config_editor(self, job: Job):
        existing_names = {j.group_name for j in self.job_queue.jobs if j.id != job.id}
        dialog = GroupConfigDialog(job, self.repo_dir, self.app_data_dir, existing_names, parent=self)
        dialog.exec()
        if not dialog.saved:
            return  # canceled -- job's fields are untouched, nothing to persist

        refresh_job_readiness(job)
        self.job_queue.update(job)
        self.model.refresh()
        self._log(f"Saved config for '{job.group_name}' -- now {job.status.value}.")

        if job.status == JobStatus.NEEDS_CROP:
            self._open_crop_step(job)

    def _job_locked_by_run(self, job: Job) -> bool:
        """True while `job` is either actually RUNNING or still QUEUED
        behind the running job in the current BatchRunner -- BatchRunner
        snapshotted its own copy of every job at start() (see
        batch_runner.py), so editing/removing/resetting a queued job
        here wouldn't reach it and would leave the run working from
        stale data."""
        return job.status in (JobStatus.RUNNING, JobStatus.QUEUED)

    def _on_edit_job(self):
        job = self._selected_job()
        if job is None:
            QMessageBox.information(self, "Select a job", "Select exactly one job to edit.")
            return
        if self._job_locked_by_run(job):
            QMessageBox.information(self, "Job running", "Can't edit a job while it's running.")
            return
        self._open_config_editor(job)

    def _on_table_double_clicked(self):
        """A completed job opens the validation viewer, matching the
        double-click's meaning in every other status ("show me what this
        row means") -- everything else still opens the config editor."""
        job = self._selected_job()
        if job is not None and job.status == JobStatus.DONE:
            self._open_validation_view(job)
        else:
            self._on_edit_job()

    def _open_validation_view(self, job: Job):
        ValidationListDialog(job, parent=self).exec()

    def _on_view_validation_selected(self):
        job = self._selected_job()
        if job is not None and job.status == JobStatus.DONE:
            self._open_validation_view(job)

    def _confirm_retry(self, jobs: list[Job]) -> bool:
        """Shows each job's actual failure reason and asks to confirm
        retrying it -- a job failed for a specific, possibly still-
        unresolved reason, so retrying it silently would either waste
        however long it takes to fail the same way again, or (worse)
        succeed having silently masked whatever was actually wrong."""
        if len(jobs) == 1:
            job = jobs[0]
            reason = job.error_message or "(no error message recorded)"
            detail = f"'{job.group_name}' previously failed with:\n\n{reason}\n\nRetry this job anyway?"
        else:
            reason_lines = "\n\n".join(
                f"'{j.group_name}': {j.error_message or '(no error message recorded)'}" for j in jobs
            )
            detail = f"Retry {len(jobs)} failed job(s) anyway? Their errors:\n\n{reason_lines}"
        reply = QMessageBox.warning(
            self, "Retry failed job(s)", detail, QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        return reply == QMessageBox.Yes

    def _retry_jobs(self, jobs: list[Job]) -> None:
        """Moves FAILED job(s) back to READY (or NEEDS_CROP, if
        refresh_job_readiness finds the crop step itself is why it's not
        ready -- see that function) so they're picked up by Run All/Run
        Selected again, same as any other READY job. Caller's
        responsibility to confirm first (see _confirm_retry) and to
        refresh the model/table after."""
        for job in jobs:
            job.error_message = ""
            # refresh_job_readiness deliberately leaves FAILED (and
            # RUNNING/DONE/CANCELED) alone -- those are runner-owned
            # states it won't recompute out of (see its docstring).
            # READY is the assumption to recompute *from*; it'll still
            # correctly downgrade to NEEDS_CROP/NEEDS_CONFIG below if
            # disk state has drifted since this job failed (e.g. a crop
            # output got cleaned up).
            job.status = JobStatus.READY
            refresh_job_readiness(job)
            self.job_queue.update(job)
            self._log(f"'{job.group_name}' retry requested -- now {job.status.value}.")

    def _on_retry_failed(self):
        jobs = [j for j in self._selected_jobs() if j.status == JobStatus.FAILED]
        if not jobs or not self._confirm_retry(jobs):
            return
        self._retry_jobs(jobs)
        self.model.refresh()

    def _on_remove_selected(self):
        jobs = self._selected_jobs()
        if not jobs:
            return
        if any(self._job_locked_by_run(j) for j in jobs):
            QMessageBox.information(self, "Job running", "Stop the run before removing a running job.")
            return
        reply = QMessageBox.question(
            self, "Remove job(s)",
            f"Remove {len(jobs)} job(s) from the queue? (Output already written to disk is not deleted.)",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        for job in jobs:
            self.job_queue.remove(job.id)
        self.model.refresh()
        self.statusBar().showMessage(f"{len(self.job_queue.jobs)} job(s) queued")
        self._log(f"Removed {len(jobs)} job(s).")

    # -- crop step --

    def _on_recrop_selected(self):
        job = self._selected_job()
        if job is None:
            QMessageBox.information(self, "Select a job", "Select exactly one job to re-crop.")
            return
        if self._job_locked_by_run(job):
            QMessageBox.information(self, "Job running", "Can't re-crop a job while it's running.")
            return
        self._open_crop_step(job)

    def _open_crop_step(self, job: Job):
        if not job.config_path.exists():
            QMessageBox.warning(self, "No config", "Save this job's config before cropping.")
            return
        try:
            config = PipelineConfig.from_yaml(job.config_path)
        except Exception as exc:
            QMessageBox.warning(self, "Can't crop", f"Couldn't load {job.config_path}:\n{exc}")
            return
        if config.discovery is None:
            QMessageBox.warning(
                self, "Can't crop", "This job's config has no discovery info -- re-save it from the config editor."
            )
            return

        disc = config.discovery
        all_videos = find_videos(disc.input_dir)
        token_pattern = re.compile(disc.camera_regex)

        def videos_for_role(role):
            matched = []
            for v in all_videos:
                m = token_pattern.search(v.name)
                if m and disc.camera_role_map.get(m.group(1)) == role:
                    matched.append(v)
            return matched

        bottom_videos = videos_for_role("bottom")
        side_videos = videos_for_role("left") + videos_for_role("right")
        cropped_dir = job.cropped_dir
        tools_dir = str(self.repo_dir / "tools")

        if bottom_videos:
            grade_all = QMessageBox.question(
                self, "Bottom footage",
                "Apply bottom-up color correction to all bottom videos in this job?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
            ) == QMessageBox.Yes
            dialog = CropSetupDialog(
                bottom_videos, disc.input_dir, cropped_dir, tools_dir,
                CROP_TARGET_WIDTH, CROP_TARGET_HEIGHT, parent=self,
                mode="bottom", default_color_grade=grade_all,
            )
            dialog.exec()

        if side_videos:
            side_model = app_settings.get_selected_model(self.app_data_dir, "side")
            if side_model:
                side_w, side_h = side_crop_size_for_model(self.repo_dir / "models" / side_model)
            else:
                side_w, side_h = CROP_TARGET_WIDTH, CROP_TARGET_HEIGHT
            dialog = CropSetupDialog(
                side_videos, disc.input_dir, cropped_dir, tools_dir,
                side_w, side_h, parent=self, mode="side",
            )
            dialog.exec()

        refresh_job_readiness(job)
        self.job_queue.update(job)
        self.model.refresh()
        self._log(f"'{job.group_name}' now {job.status.value}.")

    # -- reset --

    def _on_reset_selected(self, clear_predictions: bool, clear_output: bool, clear_crops: bool):
        jobs = self._selected_jobs()
        targetable = [j for j in jobs if not self._job_locked_by_run(j)]
        if len(targetable) < len(jobs):
            self._log("Skipping running job(s) -- can't reset while running.")
        if not targetable:
            return

        lines = []
        for job in targetable:
            lines.append(f"{job.group_name}:")
            lines.extend(
                f"    {t}" for t in reset_module.describe_job_targets(job, clear_predictions, clear_output, clear_crops)
            )
        reply = QMessageBox.question(
            self, "Confirm reset",
            "This will delete:\n\n" + "\n".join(lines) + "\n\n(crop_positions.json is always kept)\n\nProceed?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        for job in targetable:
            reset_module.perform_job_reset(job, clear_predictions, clear_output, clear_crops, log=self._log)
            job.status = JobStatus.NEEDS_CROP
            job.error_message = ""
            refresh_job_readiness(job)
            self.job_queue.update(job)
            self._log(f"'{job.group_name}' reset -- now {job.status.value}.")
        self.model.refresh()

    # -- run --

    def _on_run_queue(self):
        # A FAILED job sitting in the queue is exactly the case
        # _retry_jobs exists for, and "Run All" is a reasonable place to
        # offer that rather than requiring the user to already know
        # about the separate Retry action (see _on_retry_failed) first.
        # Checked unconditionally, not just when there's nothing else
        # READY to run -- a FAILED job alongside a READY one should
        # still get offered a retry, not silently skipped in favor of
        # whatever else happens to be runnable.
        failed = [j for j in self.job_queue.jobs if j.status == JobStatus.FAILED]
        if failed and self._confirm_retry(failed):
            self._retry_jobs(failed)
            self.model.refresh()
        self._run_jobs(self.job_queue.runnable_jobs())

    def _on_run_selected(self):
        selected = self._selected_jobs()
        failed = [j for j in selected if j.status == JobStatus.FAILED]
        if failed and self._confirm_retry(failed):
            self._retry_jobs(failed)
            self.model.refresh()
        jobs = [j for j in selected if j.status == JobStatus.READY]
        self._run_jobs(jobs)

    def _run_jobs(self, jobs: list[Job]):
        if not jobs:
            QMessageBox.information(self, "Nothing to run", "No Ready job(s) to run.")
            return
        if self.runner is not None:
            QMessageBox.information(self, "Already running", "A run is already in progress.")
            return

        # Only the job actually executing should read RUNNING -- the rest
        # of this batch reads QUEUED (not READY) until BatchRunner
        # reaches it (see _on_job_started, which flips a job to RUNNING
        # one at a time as it's dequeued). QUEUED, not READY, so it's
        # still locked from editing/removing/resetting in the meantime --
        # see _job_locked_by_run.
        for job in jobs:
            job.status = JobStatus.QUEUED
            job.sessions_done = 0
            self.job_queue.update(job)
        self.model.refresh()

        self._run_total_by_job = {j.id: (j.sessions_total or 1) for j in jobs}
        self._run_done_by_job = {j.id: 0 for j in jobs}
        self._run_start_time = time.time()
        self._eta_remaining_s = None
        self._eta_snapshot_time = None
        self._eta_session_duration_ema = None
        self._eta_last_session_done_time = self._run_start_time

        self.runner = BatchRunner(jobs, self.repo_dir, device="auto", tracking=False, parent=self)
        self.runner.log.connect(self._log)
        self.runner.progress.connect(self._on_progress_line)
        self.runner.progress_closed.connect(self._on_progress_closed)
        self.runner.job_started.connect(self._on_job_started)
        self.runner.job_progress.connect(self._on_job_progress)
        self.runner.job_finished.connect(self._on_job_finished)
        self.runner.all_finished.connect(self._on_all_finished)
        self.runner.finished.connect(self._on_runner_finished)
        self.runner.start()

        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(sum(self._run_total_by_job.values()))
        self.progress_bar.setValue(0)
        self.eta_label.setVisible(True)
        self.eta_label.setText("ETA: estimating…")
        self._progress_timer.start()
        self.run_btn.setEnabled(False)
        self.run_selected_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._log(f"Starting run of {len(jobs)} job(s)...")

    def _on_job_started(self, job_id: str):
        job = self.job_queue.get(job_id)
        if job is not None:
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now(timezone.utc).isoformat()
            self.job_queue.update(job)
            self.model.refresh()
        self._log(f"Started '{job.group_name if job else job_id}'.")

    def _on_job_progress(self, job_id: str, done: int, total: int):
        self._run_done_by_job[job_id] = done
        self._run_total_by_job[job_id] = total
        job = self.job_queue.get(job_id)
        if job is not None:
            job.sessions_done = done
            job.sessions_total = total
            self.job_queue.update(job)
            self.model.refresh()

        # A whole session just finished -- fold its wall-clock duration
        # into the session-duration EMA (see __init__'s docstring for why
        # this, rather than a cumulative since-start average: a real
        # slowdown partway through a run should pull the estimate down
        # within a few sessions, not stay diluted forever by however fast
        # the run started out).
        now = time.time()
        if self._eta_last_session_done_time is not None:
            duration = now - self._eta_last_session_done_time
            if duration > 0:
                if self._eta_session_duration_ema is None:
                    self._eta_session_duration_ema = duration
                else:
                    alpha = 0.3
                    self._eta_session_duration_ema = alpha * duration + (1 - alpha) * self._eta_session_duration_ema
        self._eta_last_session_done_time = now

        self._recompute_eta_snapshot()
        self._update_progress_ui()

    def _on_job_finished(self, job_id: str, status: str, message: str):
        job = self.job_queue.get(job_id)
        if job is None:
            return
        job.status = JobStatus(status)
        job.error_message = message
        job.finished_at = datetime.now(timezone.utc).isoformat()
        self.job_queue.update(job)
        self.model.refresh()
        suffix = f" -- {message}" if message else ""
        self._log(f"'{job.group_name}' finished: {status}{suffix}")

    def _on_all_finished(self):
        self._log("Run complete.")

    def _on_runner_finished(self):
        self.runner = None
        self._progress_timer.stop()
        self.progress_bar.setVisible(False)
        self.eta_label.setVisible(False)
        self.run_btn.setEnabled(True)
        self.run_selected_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _on_stop_queue(self):
        if self.runner is not None:
            self.runner.request_stop()
            self._log("Stop requested -- the current job will finish; remaining queued job(s) will be canceled.")

    def _recompute_eta_snapshot(self):
        """Re-anchors the ETA countdown to a fresh remaining-time estimate.

        Called only when real progress actually happens -- a session
        finishing (_on_job_progress) -- NOT every second; see
        _update_progress_ui for the 1Hz countdown between calls. An
        earlier version recomputed ``(total - done) / (done /
        elapsed_since_start)`` on every 1Hz tick, with
        ``elapsed_since_start`` growing continuously while ``done``
        stayed fixed between session completions (often several minutes
        apart) -- so the *average rate* it computed kept dropping every
        tick, and the *remaining time* estimate grew right along with
        it: the ETA counted up, not down, for however long a session
        took. Recomputing only on real progress fixed the counting-up
        bug, but a single ``done / elapsed_since_start`` rate was still
        a *cumulative* average over the whole run -- once a run has done
        a lot of fast progress early on, that history keeps dragging the
        average up even after the run has since slowed right down, so
        the ETA stayed optimistic and could never fully recover.

        ``_eta_session_duration_ema`` (see __init__'s docstring) fixes
        both: it's recency-weighted, so a real slowdown pulls the
        estimate down within a few sessions, and it's only ever updated
        from an actual completed session's real duration -- no attempt
        is made to track progress *within* a session (that was tried and
        scrapped; see __init__'s docstring for why). No estimate is
        shown until at least one session has finished.
        """
        if self._run_start_time is None:
            return
        sessions_done = sum(self._run_done_by_job.values())
        sessions_total = sum(self._run_total_by_job.values()) or 1
        sessions_remaining = max(0, sessions_total - sessions_done)

        if self._eta_session_duration_ema is None or sessions_remaining <= 0:
            self._eta_remaining_s = None
            self._eta_snapshot_time = None
            return
        self._eta_remaining_s = sessions_remaining * self._eta_session_duration_ema
        self._eta_snapshot_time = time.time()

    def _update_progress_ui(self):
        done = sum(self._run_done_by_job.values())
        total = sum(self._run_total_by_job.values()) or 1
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(done)

        if self._eta_remaining_s is None or self._eta_snapshot_time is None:
            self.eta_label.setText("ETA: estimating…")
            return

        displayed_remaining = max(0.0, self._eta_remaining_s - (time.time() - self._eta_snapshot_time))
        m, s = divmod(int(displayed_remaining), 60)
        self.eta_label.setText(f"ETA: {m}m {s:02d}s")

    # -- logging --

    def _append_plain_line(self, text: str):
        """QTextEdit has no appendPlainText() (that's QPlainTextEdit-only,
        and this panel needs to be QTextEdit -- see its construction);
        this reproduces the same "always start a new paragraph, insert
        literal (non-HTML) text" behavior by hand.

        Explicitly resets the cursor's character format first: a colored
        progress-bar redraw (see _on_progress_line) very often leaves a
        color "open" at the end of its content -- real tqdm output
        constantly does this, since the color reset code (if any) is
        wherever tqdm itself put it, not necessarily right at the end of
        the line -- and insertBlock() alone does NOT clear that; it
        carries the current character format into the new paragraph.
        insertText() (unlike insertHtml() with genuinely unstyled
        content, which does get a clean default) then uses that stale
        format verbatim, so without this reset every plain log line
        after any colored one would silently inherit its color.
        """
        cursor = self.log_panel.textCursor()
        cursor.movePosition(QTextCursor.End)
        if not self.log_panel.document().isEmpty():
            cursor.insertBlock()
        cursor.setCharFormat(QTextCharFormat())
        cursor.insertText(text)
        self.log_panel.setTextCursor(cursor)
        self.log_panel.ensureCursorVisible()

    def _trim_log_panel(self, max_blocks: int = 5000):
        """QPlainTextEdit's setMaximumBlockCount() has no QTextEdit
        equivalent -- this drops the oldest paragraphs by hand once the
        document grows past `max_blocks`, so an overnight run's log
        doesn't grow unbounded."""
        doc = self.log_panel.document()
        excess = doc.blockCount() - max_blocks
        if excess <= 0:
            return
        cursor = QTextCursor(doc)
        cursor.movePosition(QTextCursor.Start)
        cursor.movePosition(QTextCursor.NextBlock, QTextCursor.KeepAnchor, excess)
        cursor.removeSelectedText()

    def _log(self, message: str):
        self._append_plain_line(message)
        # A discrete message always ends any progress-bar redraw in
        # progress -- the next one (a different camera role's inference,
        # most likely) should start its own new line rather than
        # overwriting whatever was just logged here.
        self._progress_line_open = False
        self._trim_log_panel()

    def _on_progress_line(self, html_message: str):
        """Live inference-progress updates (see
        alligaitor.subprocess_streaming's `progress` callback, HTML-
        rendered by alligaitor.ansi_html since html_progress=True is
        what the batch worker passes) -- redraws the same line in place,
        the same colored way the tqdm bar this mirrors does in a real
        terminal, instead of adding a new line to the log for every
        redraw. Purely a log-panel display concern -- the ETA is
        session-granularity only (see __init__'s docstring on
        _eta_session_duration_ema for why), so nothing here feeds it."""
        cursor = self.log_panel.textCursor()
        cursor.movePosition(QTextCursor.End)
        if self._progress_line_open:
            # Selects from the end of the document back to the start of
            # the last line (not the whole document) -- KeepAnchor stops
            # the selection from also eating the newline before it, so
            # only that one line's content gets replaced.
            cursor.movePosition(QTextCursor.StartOfBlock, QTextCursor.KeepAnchor)
            cursor.removeSelectedText()
        elif not self.log_panel.document().isEmpty():
            cursor.insertBlock()
        cursor.insertHtml(html_message)
        self.log_panel.setTextCursor(cursor)
        self._progress_line_open = True
        self.log_panel.ensureCursorVisible()
        self._trim_log_panel()

    def _on_progress_closed(self):
        """A redrawn progress line's definitive final state was just
        shown (see BatchRunner.progress_closed /
        alligaitor.subprocess_streaming's `on_redraw_closed`) -- ends
        that redraw so the *next* progress update starts a fresh line
        instead of immediately overwriting the state that was just
        displayed. Without this, a session that prints more than one
        tqdm bar in sequence (e.g. one for loading, a separate one for
        predicting) would have the second bar's very first redraw
        instantly overwrite the first bar's just-shown completion --
        which reads as that line flashing before the "real" bar
        reappears in its place, even though nothing was actually lost."""
        self._progress_line_open = False

    # -- shutdown --

    def closeEvent(self, event):
        if self.runner is not None:
            reply = QMessageBox.question(
                self, "Run in progress",
                "A run is still in progress. Quit anyway? The current job will be interrupted.",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                event.ignore()
                return
            self.runner.request_stop()
            self.runner.wait(5000)
        event.accept()
