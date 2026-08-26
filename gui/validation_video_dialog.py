"""
Validation video viewer: plays one session's already-annotated validation
video (see alligaitor/validation_video.py) with the standard transport
controls (video_player_widget.py) and one scrub-bar row per paw showing
its used (or, if unusable, longest raw) run per crossing -- see
alligaitor.gait.paw_usability_windows. A recording with several crossings
(the rat walks out, turns, walks back) gets one colored segment per
crossing on that paw's row, since a paw can be usable on one crossing and
not another; the row's label shows how many of the recording's crossings
that paw was even visible in.

"Flag Paw(s)..." lets a reviewer mark a paw's run as not actually
trustworthy on one specific crossing, despite passing the automatic
threshold (or clear a previous flag on that crossing) -- a paw can be
fine on crossing 1 and flagged on crossing 2 of the same recording. The
popup defaults to whichever crossing the playhead is currently in. The
flag is stored per session, keyed by crossing (validation.save_manual_flags)
and immediately patched into just that crossing's block in the group's
Excel report (gait.annotate_manual_flag) -- no pipeline rerun needed.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QPushButton, QLabel,
    QCheckBox, QLineEdit, QComboBox, QDialogButtonBox, QMessageBox,
)

from alligaitor import gait, validation
from paw_colors import PAW_SHORT_LABELS, COLOR_USABLE, COLOR_UNUSABLE, paw_color, grayed_paw_color, ordered_paws
from video_player_widget import VideoPlayerWidget


class _FlagPawsDialog(QDialog):
    """Small popup: which paw(s) to flag as invalid on one crossing, plus
    an optional shared note -- pre-checked to whatever's currently
    flagged on that crossing, so it doubles as an editor (checking/
    unchecking a box adds/removes that paw's flag) rather than a purely
    additive action. A recording with more than one crossing gets a
    crossing picker up top (defaulting to `initial_crossing_number`);
    switching it reloads the checkboxes/note for whichever crossing is
    now selected. Only the crossing selected when OK is pressed is
    actually saved -- edit one crossing per popup invocation."""

    def __init__(self, crossings: list, initial_crossing_number: int, flags_by_crossing: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Flag Paw(s)")
        self.crossings = crossings
        self.flags_by_crossing = flags_by_crossing

        layout = QVBoxLayout(self)

        self.crossing_combo = None
        if len(crossings) > 1:
            self.crossing_combo = QComboBox()
            for c in crossings:
                n = c.get("crossing", 1)
                self.crossing_combo.addItem(f"Crossing {n} of {len(crossings)}", n)
            start_index = next(
                (i for i, c in enumerate(crossings) if c.get("crossing") == initial_crossing_number), 0
            )
            self.crossing_combo.setCurrentIndex(start_index)
            self.crossing_combo.currentIndexChanged.connect(lambda _i: self._load_crossing())
            layout.addWidget(QLabel("Crossing:"))
            layout.addWidget(self.crossing_combo)

        self.checkboxes = {}
        form = QFormLayout()
        for paw in ordered_paws():
            box = QCheckBox(f"{PAW_SHORT_LABELS[paw]} -- {paw.replace('-', ' ')}")
            box.setStyleSheet(f"QCheckBox {{ color: {paw_color(paw).name()}; }}")
            self.checkboxes[paw] = box
            form.addRow(box)
        layout.addLayout(form)

        self.note_edit = QLineEdit()
        self.note_edit.setPlaceholderText("Why is this run not actually valid? (optional)")
        form.addRow("Note:", self.note_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._load_crossing()

    def current_crossing_number(self) -> int:
        if self.crossing_combo is not None:
            return self.crossing_combo.currentData()
        return self.crossings[0].get("crossing", 1)

    def _load_crossing(self):
        crossing_number = self.current_crossing_number()
        flagged_paws = validation.crossing_flagged_paws(self.flags_by_crossing, crossing_number)
        _flagged, note = self.flags_by_crossing.get(crossing_number, (set(), ""))
        crossing = next((c for c in self.crossings if c.get("crossing") == crossing_number), {})
        paws_here = crossing.get("paws", {})
        for paw, box in self.checkboxes.items():
            box.setChecked(paw in flagged_paws)
            visible = paws_here.get(paw) is not None
            # A paw never detected on this crossing has nothing to flag --
            # disabled unless it's somehow already flagged (shouldn't
            # normally happen, but let the user still clear it if so).
            box.setEnabled(visible or paw in flagged_paws)
            box.setToolTip("" if visible else "Not detected on this crossing")
        self.note_edit.setText(note)

    def result_flags(self):
        flagged = {paw for paw, box in self.checkboxes.items() if box.isChecked()}
        return self.current_crossing_number(), flagged, self.note_edit.text().strip()


class ValidationVideoDialog(QDialog):
    """Plays `session`'s validation video with per-paw, per-crossing scrub
    markers and the Flag Paw(s) action. `xlsx_path` is the group's
    already-written report (alligaitor.config.PipelineConfig.output_xlsx)
    -- patched in place, one crossing's block at a time, when a flag
    changes."""

    def __init__(self, job, session, xlsx_path, parent=None):
        super().__init__(parent)
        self.job = job
        self.session = session
        self.xlsx_path = Path(xlsx_path)

        self._summary_path = Path(session.output_dir) / f"{session.name}.validation_summary.json"
        self._flags_path = Path(session.output_dir) / f"{session.name}.manual_flags.json"
        self.summary = validation.load_validation_summary(self._summary_path) or {"paws": {}}
        self.flags_by_crossing = validation.load_manual_flags(self._flags_path)

        self.setWindowTitle(f"Validation -- {session.name}")

        video_path = Path(job.validation_dir) / f"{session.name}.validation.mp4"
        self.player = VideoPlayerWidget(video_path)

        self.title_label = QLabel()
        self.title_label.setStyleSheet("font-weight: bold; font-size: 13px;")

        self.flag_btn = QPushButton("Flag Paw(s)…")
        self.close_btn = QPushButton("Close")
        self.flag_btn.clicked.connect(self._on_flag_paws)
        self.close_btn.clicked.connect(self.accept)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.flag_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.close_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(self.title_label)
        layout.addWidget(self.player)
        layout.addLayout(btn_row)

        self._refresh_display()

    # -- crossing lookup --

    def _crossings(self) -> list:
        return validation.crossings_or_fallback(self.summary)

    def _crossing_at_frame(self, frame_idx: int) -> int:
        """Which crossing (1-based) `frame_idx` falls inside, for
        defaulting the Flag Paw(s) popup to wherever the playhead
        currently is -- falls back to the first crossing if none of
        them carry frame-range info, or the frame lands in a gap between
        crossings (a pause/turn)."""
        for crossing in self._crossings():
            window = crossing.get("window")
            if window and window[0] <= frame_idx <= window[1]:
                return crossing.get("crossing", 1)
        return self._crossings()[0].get("crossing", 1)

    # -- display --

    def _effective_usability(self):
        return validation.effective_usability(self.summary, self.flags_by_crossing)

    def _refresh_display(self):
        usability = self._effective_usability()
        crossings = self._crossings()
        n_crossings = len(crossings)

        rows = []
        for paw in ordered_paws():
            segments = []
            n_visible = 0
            for crossing in crossings:
                window = crossing.get("paws", {}).get(paw)
                if window is None:
                    continue
                n_visible += 1
                crossing_number = crossing.get("crossing", 1)
                flagged_here = paw in validation.crossing_flagged_paws(self.flags_by_crossing, crossing_number)
                crossing_usable = bool(window.get("usable")) and not flagged_here
                color = paw_color(paw) if crossing_usable else grayed_paw_color(paw)
                segments.append((window["start_frame"], window["end_frame"], color, crossing_usable))

            if n_crossings > 1:
                label = f"{PAW_SHORT_LABELS[paw]} {n_visible}/{n_crossings}"
            else:
                window = self.summary.get("paws", {}).get(paw)
                duration = f"{window['duration_s']:.2f}s" if window else "—"
                label = f"{PAW_SHORT_LABELS[paw]} {duration}"

            rows.append((label, segments))
        self.player.set_paw_windows(rows)

        n_usable = sum(usability.values())
        all_usable = n_usable == len(usability)
        color = COLOR_USABLE.name() if all_usable else COLOR_UNUSABLE.name()
        self.title_label.setText(
            f'<span style="color:{color};">{self.session.name} -- {n_usable}/{len(usability)} paws usable</span>'
        )

    # -- flagging --

    def _on_flag_paws(self):
        crossings = self._crossings()
        current_crossing = self._crossing_at_frame(self.player.current_frame_idx)
        dialog = _FlagPawsDialog(crossings, current_crossing, self.flags_by_crossing, parent=self)
        if dialog.exec() != QDialog.Accepted:
            return
        crossing_number, new_flagged, new_note = dialog.result_flags()
        old_flagged = validation.crossing_flagged_paws(self.flags_by_crossing, crossing_number)
        changed_paws = old_flagged.symmetric_difference(new_flagged)
        self.flags_by_crossing[crossing_number] = (new_flagged, new_note)
        if not changed_paws:
            return

        validation.save_manual_flags(self._flags_path, self.flags_by_crossing)

        crossing = next((c for c in crossings if c.get("crossing") == crossing_number), {})
        crossing_count = len(crossings)
        missing = []
        for paw in changed_paws:
            window = crossing.get("paws", {}).get(paw)
            auto_usable = bool(window["usable"]) if window else False
            ok = gait.annotate_manual_flag(
                self.xlsx_path, self.session.rat_id, self.session.name, crossing_number, crossing_count, paw,
                auto_usable=auto_usable, flagged=(paw in new_flagged), note=new_note,
            )
            if not ok:
                missing.append(paw)

        self._refresh_display()

        if missing:
            QMessageBox.warning(
                self, "Flag saved, but the report wasn't updated",
                f"The flag was saved for crossing {crossing_number}, but couldn't be written into "
                f"{self.xlsx_path.name} for: {', '.join(missing)} -- the report may not "
                f"have a matching session/crossing/rat entry yet (e.g. it hasn't been run since "
                f"this session was added).",
            )

    # -- lifecycle --

    def closeEvent(self, event):
        self.player.release()
        super().closeEvent(event)
