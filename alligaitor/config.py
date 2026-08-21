"""Configuration schema for the alliGAITor calibration and triangulation pipeline.

Camera role assignment (``left`` / ``right`` / ``bottom``) is resolved
per session rather than by a fixed camera index, because the physical
camera that lands on a given device index (``cam0`` / ``cam1`` / ``cam2``)
is not consistent across recording sessions. Each :class:`SessionConfig`
explicitly maps roles to video files; calibration is captured once against
the same three role names and reused across sessions as long as the
cameras have not been physically moved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml

CAMERA_ROLES = ("left", "right", "bottom")

PathLike = Union[str, Path]


def _resolve(base_dir: Path, path: PathLike) -> Path:
    """Resolve ``path`` relative to ``base_dir`` unless it is already absolute."""
    p = Path(path)
    return p if p.is_absolute() else (base_dir / p).resolve()


def _require_roles(videos: Dict[str, Path], context: str) -> None:
    missing = [role for role in CAMERA_ROLES if role not in videos]
    if missing:
        raise ValueError(f"{context} is missing video(s) for role(s): {missing}")
    extra = [role for role in videos if role not in CAMERA_ROLES]
    if extra:
        raise ValueError(f"{context} has unknown camera role(s): {extra}; expected one of {CAMERA_ROLES}")


@dataclass
class ModelConfig:
    """Paths to trained SLEAP-NN model directories.

    Attributes:
        side_model_dir: Trained side-angle model directory, used for both
            the left and right camera views.
        bottom_model_dir: Trained bottom-up (tunnel) model directory.
    """

    side_model_dir: Path
    bottom_model_dir: Path

    def model_dir_for_role(self, role: str) -> Path:
        """Return the model directory that predicts on the given camera role."""
        if role not in CAMERA_ROLES:
            raise ValueError(f"Unknown camera role '{role}'; expected one of {CAMERA_ROLES}.")
        return self.bottom_model_dir if role == "bottom" else self.side_model_dir


@dataclass
class CalibrationConfig:
    """Paths to the ChArUco calibration recordings, one per camera role.

    Calibration is captured once per physical camera rig and reused across
    sessions, provided the cameras have not moved. Re-record and re-run
    calibration if a camera is bumped or repositioned.

    Attributes:
        videos: Mapping of camera role to calibration video path.
        output_path: Where the resulting camera calibration (aniposelib
            ``CameraGroup``, saved as TOML) is written or loaded from.
        board_preset: Which physical calibration board this recording
            used — either ``"apriltag"`` (the flat AprilTag marker-grid
            board, see :class:`alligaitor.calibration.AprilGridBoard`) or
            a key into :data:`alligaitor.calibration.BOARD_PRESETS` for a
            ChArUco board (currently ``"original"``, the 8x8/15mm board, or
            ``"strip"``, the narrow 4x5/35mm board — both superseded by
            ``"apriltag"`` for the bottom camera's 45-degree slit view).
            Different recordings may use different physical boards; this
            says which one to expect when detecting corners/markers for
            this particular calibration. Also determines
            :attr:`calibration_standard`.
        min_corners_extrinsic: Minimum matched points a frame needs to
            link two cameras' poses during calibration (see
            :data:`alligaitor.calibration.MIN_CORNERS_EXTRINSIC`). Only
            used when :attr:`calibration_standard` is ``"apriltag"`` —
            ChArUco calibration always uses aniposelib's own hardcoded
            floor (8) instead, since making it configurable there was
            found to make calibration quality worse rather than better.
    """

    videos: Dict[str, Path]
    output_path: Path
    board_preset: str = "original"
    min_corners_extrinsic: int = 8

    def __post_init__(self) -> None:
        _require_roles(self.videos, "Calibration config")

    @property
    def calibration_standard(self) -> str:
        """Which calibration algorithm
        :func:`alligaitor.calibration.calibrate` should run for this
        recording: ``"apriltag"`` for the ``"apriltag"`` board preset, or
        ``"charuco"`` for every ChArUco board preset. Derived from
        ``board_preset`` rather than stored separately, since the two
        can't disagree — a recording's calibration algorithm is
        determined by which physical board it used.
        """
        return "apriltag" if self.board_preset == "apriltag" else "charuco"


@dataclass
class SessionConfig:
    """One gait-recording session: one video per camera role.

    Attributes:
        name: Session identifier, used for output file naming.
        videos: Mapping of camera role to this session's video path.
        output_dir: Directory where 2D predictions and 3D output for this
            session are written.
        rat_id: Which rat this trial belongs to. Video/session names
            (e.g. ``"359a-BL"``) encode a trial letter and condition, not
            a reliable rat identity, so this is a separate, explicit
            field rather than something parsed from ``name``. Defaults to
            ``name`` (i.e. each session is its own rat) when not given.
            Sessions sharing a ``rat_id`` within one group are combined
            onto that rat's tab in the gait-metrics spreadsheet (see
            :mod:`alligaitor.gait`), one row per trial.
    """

    name: str
    videos: Dict[str, Path]
    output_dir: Path
    rat_id: Optional[str] = None

    def __post_init__(self) -> None:
        _require_roles(self.videos, f"Session '{self.name}'")
        if not self.rat_id:
            self.rat_id = self.name


@dataclass
class GaitConfig:
    """Tunable thresholds for stance/swing (paw ground-contact) detection.

    A paw is considered planted on a frame when its frame-to-frame speed
    stays below ``speed_threshold_mm_s``; see :mod:`alligaitor.gait` for
    how that feeds into stride, step, and ground-contact-time
    calculations.

    A height-above-platform check was tried and dropped: drawing an
    accurate height-threshold reference line requires knowing the current
    frame's actual depth across the tunnel's width, and a single-anchor
    approximation was visually misleading (a paw on the far side of the
    tunnel could appear to cross a threshold line that was only valid for
    a different depth) -- speed alone avoids that failure mode.

    Attributes:
        speed_threshold_mm_s: Maximum frame-to-frame speed, in mm/s, for a
            paw to count as planted.
        min_contact_frames: Minimum number of consecutive frames a paw
            must satisfy that threshold to count as a real stance phase.
            Originally meant to filter out single-frame tracking jitter,
            but on real data (this rig, ~12.5fps) speed cleanly separates
            into two well-defined clusters with almost nothing between
            them -- a single frame landing in the low cluster is real
            evidence of a (likely brief) stance, not noise near a fuzzy
            boundary, and requiring 2 consecutive frames was discarding
            most genuine forepaw stances (which are often only one frame
            long at this frame rate). Defaults to ``1`` accordingly.
        max_bridge_gap_frames: Untriangulated runs of at most this many
            frames, bounded by a valid frame on both sides, are linearly
            interpolated before speed/stance is computed at all -- jitter
            and brief per-camera dropouts will always happen even with
            good models, and this keeps a real stance phase from being
            fragmented into pieces too short to individually survive
            ``min_contact_frames`` just because of a momentary gap.
            Longer gaps are left as real gaps (see
            :func:`alligaitor.gait.find_camera_caused_discards`). ``0``
            disables bridging entirely. Defaults to ``4``: measured gap
            lengths on real trials cluster at 2-4 frames (motion-blur
            dropouts during a single swing, not multi-swing outages), and
            the previous default of ``2`` was leaving nearly every one of
            those just barely un-bridged -- fragmenting genuinely
            consecutive steps into short runs for no reason tied to data
            quality. A bridged gap also stops counting as a camera-caused
            discard (see ``find_camera_caused_discards``), which is why a
            second, independent check
            (``stride_length_outlier_ratio``) exists for steps a gap
            *doesn't* explain: one silently missing entirely inside a
            span that still reads as clean.
        min_consecutive_steps: A paw's reported stride/step/ground-contact
            averages are computed only from steps that are part of a run
            of at least this many consecutive accepted stance events with
            no camera-caused discard (see
            :func:`alligaitor.gait.find_camera_caused_discards`) in the
            swing between any pair -- an isolated good detection that
            isn't part of such a run doesn't count, and a paw with no
            qualifying run at all reports ``NaN`` rather than an average
            built from too little (or too suspect) data. See
            :func:`alligaitor.gait.restrict_to_consecutive_runs`.
        stillness_speed_threshold_mm_s: Below this frame-to-frame speed,
            in mm/s, the whole-body reference node (see
            ``alligaitor.gait.REFERENCE_NODE``) counts as not translating.
            Used to trim any leading/trailing stretch where the rat has
            stopped moving -- once the body itself is stationary, a
            paw's own position can still jitter across
            ``speed_threshold_mm_s`` from tracking noise alone, which
            would otherwise look like a run of real steps in place. This
            is a much lower bar than ``speed_threshold_mm_s``: it's
            asking whether the *animal* is translating at all, not
            whether one paw is currently planted mid-stride. Starting
            default, not derived from measured data the way
            ``min_contact_frames`` was -- tune against real trials with
            ``scripts/debug_gait.py``.
        min_still_frames: How many consecutive frames of sub-threshold
            body speed, bordering either end of the trial, counts as
            "stopped" rather than an ordinary brief slowdown mid-stride.
            Only a run touching the very start or very end of the trial
            is ever trimmed -- a pause in the middle is left alone. See
            :func:`alligaitor.gait.restrict_to_consecutive_runs`.
        stride_length_outlier_ratio: A stride longer than this many times
            a paw's own median stride length in the trial is flagged as
            likely hiding a missed step -- e.g. a real stance the speed
            classifier failed to recognize -- rather than being one
            genuine stride, and breaks a qualifying run the same way a
            camera-caused discard does. This catches exactly what
            ``max_bridge_gap_frames`` cannot: triangulation can be clean
            the entire way through and still miss a real, brief stance.
            See :func:`alligaitor.gait.find_stride_length_outliers`.
    """

    speed_threshold_mm_s: float = 50.0
    min_contact_frames: int = 1
    max_bridge_gap_frames: int = 4
    min_consecutive_steps: int = 5
    stillness_speed_threshold_mm_s: float = 20.0
    min_still_frames: int = 15
    stride_length_outlier_ratio: float = 1.8


@dataclass
class PipelineConfig:
    """Top-level configuration: models, calibration, and one or more sessions.

    Attributes:
        models: Trained model directories.
        calibration: Calibration recordings and output path.
        sessions: The group's trials -- one video-triplet per crossing.
        name: Group identifier, used to name the gait-metrics workbook.
            Defaults to the config file's stem.
        output_xlsx: Where the gait-metrics workbook for this group (one
            tab per distinct ``rat_id`` across ``sessions``) is written.
            Defaults to ``<config dir>/reports/<name>.gait_metrics.xlsx``.
        gait: Stance/swing detection thresholds shared by every session
            in this group.
    """

    models: ModelConfig
    calibration: CalibrationConfig
    sessions: List[SessionConfig]
    name: str = "group"
    output_xlsx: Optional[Path] = None
    gait: GaitConfig = field(default_factory=GaitConfig)

    @classmethod
    def from_yaml(cls, path: PathLike) -> "PipelineConfig":
        """Load a :class:`PipelineConfig` from a YAML file.

        Relative paths in the file are resolved against the file's parent
        directory. See ``configs/session_example.yaml`` for the expected
        schema.
        """
        path = Path(path)
        base_dir = path.parent
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        models_raw = raw["models"]
        models = ModelConfig(
            side_model_dir=_resolve(base_dir, models_raw["side_model_dir"]),
            bottom_model_dir=_resolve(base_dir, models_raw["bottom_model_dir"]),
        )

        calib_raw = raw["calibration"]
        calibration = CalibrationConfig(
            videos={role: _resolve(base_dir, p) for role, p in calib_raw["videos"].items()},
            output_path=_resolve(base_dir, calib_raw["output_path"]),
            board_preset=calib_raw.get("board_preset", "original"),
            min_corners_extrinsic=calib_raw.get("min_corners_extrinsic", 8),
        )

        sessions = [
            SessionConfig(
                name=session_raw["name"],
                videos={role: _resolve(base_dir, p) for role, p in session_raw["videos"].items()},
                output_dir=_resolve(base_dir, session_raw["output_dir"]),
                rat_id=session_raw.get("rat_id"),
            )
            for session_raw in raw["sessions"]
        ]

        name = raw.get("name", path.stem)
        output_xlsx_raw = raw.get("output_xlsx")
        output_xlsx = (
            _resolve(base_dir, output_xlsx_raw)
            if output_xlsx_raw
            else base_dir / "reports" / f"{name}.gait_metrics.xlsx"
        )
        gait = GaitConfig(**raw.get("gait", {}))

        return cls(
            models=models,
            calibration=calibration,
            sessions=sessions,
            name=name,
            output_xlsx=output_xlsx,
            gait=gait,
        )
