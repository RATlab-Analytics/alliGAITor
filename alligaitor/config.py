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

    A paw is considered planted on a frame when both its frame-to-frame
    speed and its height above the platform stay below their respective
    thresholds; see :mod:`alligaitor.gait` for how these feed into stride,
    step, and ground-contact-time calculations.

    Attributes:
        speed_threshold_mm_s: Maximum frame-to-frame speed, in mm/s, for a
            paw to count as planted.
        height_threshold_mm: Maximum height above this trial's estimated
            platform surface (see ``platform_baseline_percentile``), in
            mm, for a paw to count as planted.
        platform_baseline_percentile: Percentile of a paw's height trace,
            within one trial, used to estimate that platform's surface
            height for that paw -- low enough to reflect genuine contact
            frames without being thrown off by a handful of outlier low
            readings.
        min_contact_frames: Minimum number of consecutive frames a paw
            must satisfy both thresholds to count as a real stance phase,
            filtering out single-frame tracking jitter.

    Note there is no fixed "which axis is up" setting here: the
    calibration board's orientation (not gravity) sets the reconstruction's
    coordinate frame, so height is measured against a world-up direction
    derived from the calibrated rig instead -- see
    :func:`alligaitor.calibration.world_up_direction` and
    :func:`alligaitor.gait.compute_trial_metrics`.
    """

    speed_threshold_mm_s: float = 50.0
    height_threshold_mm: float = 5.0
    platform_baseline_percentile: float = 5.0
    min_contact_frames: int = 2


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
