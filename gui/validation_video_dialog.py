"""
Validation video viewer: plays one session's already-annotated validation
video (see alligaitor/validation_video.py) with the standard transport
controls (video_player_widget.py) and one scrub-bar row per paw showing
its used (or, if unusable, longest raw) run -- see
alligaitor.gait.paw_usability_windows.

"Flag Paw(s)..." lets a reviewer mark a paw's run as not actually
trustworthy despite passing the automatic threshold (or clear a previous
flag); the flag is stored per session (validation.save_manual_flags) and
immediately patched into the group's Excel report
(gait.annotate_manual_flag) -- no pipeline rerun needed.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QPushButton, QLabel,
    QCheckBox, QLineEdit, QDialogButtonBox, QMessageBox,
)

from alligaitor import gait, validation
from paw_colors import PAW_SHORT_LABELS, COLOR_USABLE, COLOR_UNUSABLE, paw_color, grayed_paw_color, ordered_paws
from video_player_widget import VideoPlayerWidget


class _FlagPawsDialog(QDialog):
    """Small popup: which paw(s) to flag as invalid, plus an optional
    shared note -- pre-checked to whatever's currently flagged, so it
    doubles as an editor (checking/unchecking a box adds/removes that
    paw's flag) rather than a purely additive action."""

    def __init__(self, flagged_paws: set, note: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Flag Paw(s)")

        self.checkboxes = {}
        form = QFormLayout()
        for paw in ordered_paws():
            box = QCheckBox(f"{PAW_SHORT_LABELS[paw]} -- {paw.replace('-', ' ')}")
            box.setChecked(paw in flagged_paws)
            box.setStyleSheet(f"QCheckBox {{ color: {paw_color(paw).name()}; }}")
            self.checkboxes[paw] = box
            form.addRow(box)

        self.note_edit = QLineEdit(note)
        self.note_edit.setPlaceholderText("Why is this run not actually valid? (optional)")
        form.addRow("Note:", self.note_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def result_flags(self):
        flagged = {paw for paw, box in self.checkboxes.items() if box.isChecked()}
        return flagged, self.note_edit.text().strip()


class ValidationVideoDialog(QDialog):
    """Plays `session`'s validation video with per-paw scrub markers and
    the Flag Paw(s) action. `xlsx_path` is the group's already-written
    report (alligaitor.config.PipelineConfig.output_xlsx) -- patched in
    place when a flag changes."""

    def __init__(self, job, session, xlsx_path, parent=None):
        super().__init__(parent)
        self.job = job
        self.session = session
        self.xlsx_path = Path(xlsx_path)

        self._summary_path = Path(session.output_dir) / f"{session.name}.validation_summary.json"
        self._flags_path = Path(session.output_dir) / f"{session.name}.manual_flags.json"
        self.summary = validation.load_validation_summary(self._summary_path) or {"paws": {}}
        self.flagged_paws, self.note = validation.load_manual_flags(self._flags_path)

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

    # -- display --

    def _effective_usability(self):
        return validation.effective_usability(self.summary, self.flagged_paws)

    def _refresh_display(self):
        usability = self._effective_usability()
        rows = []
        for paw in ordered_paws():
            window = self.summary.get("paws", {}).get(paw)
            usable = usability[paw]
            color = paw_color(paw) if usable else grayed_paw_color(paw)
            duration = f"{window['duration_s']:.2f}s" if window else "—"
            label = f"{PAW_SHORT_LABELS[paw]} {duration}"
            start = window["start_frame"] if window else None
            end = window["end_frame"] if window else None
            rows.append((label, start, end, color))
        self.player.set_paw_windows(rows)

        n_usable = sum(usability.values())
        all_usable = n_usable == len(usability)
        color = COLOR_USABLE.name() if all_usable else COLOR_UNUSABLE.name()
        self.title_label.setText(
            f'<span style="color:{color};">{self.session.name} -- {n_usable}/{len(usability)} paws usable</span>'
        )

    # -- flagging --

    def _on_flag_paws(self):
        dialog = _FlagPawsDialog(self.flagged_paws, self.note, parent=self)
        if dialog.exec() != QDialog.Accepted:
            return
        new_flagged, new_note = dialog.result_flags()
        changed_paws = self.flagged_paws.symmetric_difference(new_flagged)
        if not changed_paws:
            self.flagged_paws, self.note = new_flagged, new_note
            return

        validation.save_manual_flags(self._flags_path, new_flagged, new_note)

        missing = []
        for paw in changed_paws:
            window = self.summary.get("paws", {}).get(paw)
            auto_usable = bool(window["usable"]) if window else False
            ok = gait.annotate_manual_flag(
                self.xlsx_path, self.session.rat_id, self.session.name, paw,
                auto_usable=auto_usable, flagged=(paw in new_flagged), note=new_note,
            )
            if not ok:
                missing.append(paw)

        self.flagged_paws, self.note = new_flagged, new_note
        self._refresh_display()

        if missing:
            QMessageBox.warning(
                self, "Flag saved, but the report wasn't updated",
                f"The flag was saved for this session, but couldn't be written into "
                f"{self.xlsx_path.name} for: {', '.join(missing)} -- the report may not "
                f"have a matching session/rat entry yet (e.g. it hasn't been run since "
                f"this session was added).",
            )

    # -- lifecycle --

    def closeEvent(self, event):
        self.player.release()
        super().closeEvent(event)
