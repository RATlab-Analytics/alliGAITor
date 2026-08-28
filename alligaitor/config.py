"""Configuration schema for the alliGAITor calibration and triangulation pipeline.

Camera role assignment (``left`` / ``right`` / ``bottom``) is resolved
per session rather than by a fixed camera index, since the physical
camera on a given device index can vary across sessions. Each
:class:`SessionConfig` explicitly maps roles to video files; calibration
is captured once and reused across sessions while the cameras stay fixed.
"""

from __future__ import annotations

import dataclasses
import warnings
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


def _relativize(base_dir: Path, path: PathLike) -> str:
    """Inverse of :func:`_resolve`: render ``path`` relative to ``base_dir``
    for writing back out to YAML, falling back to an absolute path string
    if it isn't actually under ``base_dir``."""
    p = Path(path)
    try:
        return str(p.relative_to(base_dir))
    except ValueError:
        return str(p)


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
            used — ``"apriltag"`` (see
            :class:`alligaitor.calibration.AprilGridBoard`) or a key into
            :data:`alligaitor.calibration.BOARD_PRESETS` for a ChArUco
            board. Also determines :attr:`calibration_standard`.
        min_corners_extrinsic: Minimum matched points a frame needs to
            link two cameras' poses during calibration (see
            :data:`alligaitor.calibration.MIN_CORNERS_EXTRINSIC`). Only
            used when :attr:`calibration_standard` is ``"apriltag"``.
    """

    videos: Dict[str, Path]
    output_path: Path
    board_preset: str = "original"
    min_corners_extrinsic: int = 8

    def __post_init__(self) -> None:
        _require_roles(self.videos, "Calibration config")

    @property
    def calibration_standard(self) -> str:
        """Calibration algorithm to run: ``"apriltag"`` or ``"charuco"``,
        derived from ``board_preset``."""
        return "apriltag" if self.board_preset == "apriltag" else "charuco"


@dataclass
class SessionConfig:
    """One gait-recording session: one video per camera role.

    Attributes:
        name: Session identifier, used for output file naming.
        videos: Mapping of camera role to this session's video path.
        output_dir: Directory where 2D predictions and 3D output for this
            session are written.
        rat_id: Which rat this trial belongs to. Defaults to ``name``
            (each session is its own rat) when not given. Sessions
            sharing a ``rat_id`` within one group are combined onto that
            rat's tab in the gait-metrics spreadsheet (see
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

    A paw is planted on a frame when its frame-to-frame speed stays below
    ``speed_threshold_mm_s``; see :mod:`alligaitor.gait` for how that
    feeds into stride, step, and ground-contact-time calculations.

    Attributes:
        speed_threshold_mm_s: Maximum frame-to-frame speed, in mm/s, for a
            paw to count as planted.
        min_contact_frames: Minimum consecutive frames below threshold to
            count as a real stance phase.
        max_bridge_gap_frames: Untriangulated runs of at most this many
            frames, bounded by a valid frame on both sides, are linearly
            interpolated before speed/stance is computed. Longer gaps are
            left as real gaps (see
            :func:`alligaitor.gait.find_camera_caused_discards`). ``0``
            disables bridging.
        min_consecutive_strides: A paw's reported stride/step/ground-
            contact averages are computed only from strides in a run of
            at least this many consecutive accepted stance events with no
            camera-caused discard between them; otherwise ``NaN``. See
            :func:`alligaitor.gait.restrict_to_consecutive_runs`.
        stillness_window_seconds: Width, in seconds, of the window
            whole-body speed is measured across for stillness detection
            (see :func:`alligaitor.gait.windowed_body_speed`); a window
            cancels frame-to-frame reconstruction jitter that a single
            node's instantaneous speed would not.
        stillness_window_speed_mm_s: Below this whole-body speed, in
            mm/s, measured across ``stillness_window_seconds``, the
            reference node (``alligaitor.gait.REFERENCE_NODE``) counts as
            not translating. Used to trim leading/trailing stretches
            where the rat has stopped moving.
        min_still_seconds: How long a stretch of sub-threshold body speed
            bordering either end of the trial counts as "stopped" rather
            than a brief mid-stride slowdown. See
            :func:`alligaitor.gait.active_window`.
        min_valid_steps: Fewest valid steps (see
            :func:`alligaitor.gait._step_lengths`) a paw needs before its
            average step length is reported; below this it is ``NaN``.
            Step length depends on the contralateral paw's most recent
            touchdown, so it has its own evidence bar separate from
            ``min_consecutive_strides``.
        stride_length_outlier_ratio: A stride longer than this many times
            a paw's own median stride length is flagged as likely hiding
            a missed stance, and breaks a qualifying run the same way a
            camera-caused discard does. A stride with zero or negative
            net forward progress always breaks a run, unconditionally.
            See :func:`alligaitor.gait.find_stride_length_outliers`.
    """

    speed_threshold_mm_s: float = 50.0
    min_contact_frames: int = 1
    max_bridge_gap_frames: int = 4
    min_consecutive_strides: int = 5
    stillness_window_seconds: float = 0.4
    stillness_window_speed_mm_s: float = 100.0
    min_still_seconds: float = 0.4
    min_valid_steps: int = 5
    stride_length_outlier_ratio: float = 1.8

    @classmethod
    def from_raw(cls, raw: Optional[Dict[str, object]]) -> "GaitConfig":
        """Build from a config file's (or ``settings.json``'s) ``gait``
        mapping, dropping any keys that are no longer fields (falling back
        to that field's default, with a warning) instead of raising."""
        values = dict(raw or {})
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            warnings.warn(
                f"Ignoring unrecognized gait setting(s) {unknown}; using current defaults "
                "instead. Re-save this job's config (or Settings > Preferences) to clear this.",
                stacklevel=2,
            )
        return cls(**{k: v for k, v in values.items() if k in known})


@dataclass
class DiscoveryConfig:
    """Rules for auto-discovering a group's sessions from a folder of
    videos, used by the GUI's config editor (see
    :mod:`alligaitor.discovery`) to build :attr:`PipelineConfig.sessions`.
    Not consumed by the plain CLI pipeline itself; ``sessions`` is always
    written out fully resolved.

    Attributes:
        input_dir: Folder of source (pre-crop) videos to discover
            sessions from.
        id_regex: Applied to each video's filename; group 1 is the
            session name. Videos sharing a session name, one per camera
            role, form one session.
        camera_regex: Applied to each video's filename; group 1 is a
            camera token, mapped to a role via :attr:`camera_role_map`.
        camera_role_map: Camera token -> role
            (``"left"``/``"right"``/``"bottom"``).
        rat_id_overrides: Session name -> rat_id, for sessions where the
            same rat crosses more than once within this group (see
            :attr:`SessionConfig.rat_id`).
    """

    input_dir: Path
    id_regex: str
    camera_regex: str
    camera_role_map: Dict[str, str] = field(default_factory=dict)
    rat_id_overrides: Dict[str, str] = field(default_factory=dict)


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
        discovery: How the GUI's config editor auto-discovered
            ``sessions`` from a folder of videos, if it did. Round-tripped
            for editing but not read by :func:`run_pipeline`/`run_group`.
        skip_validation_videos: If ``True``, skip rendering an annotated
            validation video for every session in this group (see
            :func:`alligaitor.validation_video.export_validation_video`).
            Defaults to ``False``.
        bottom_fallback: If ``True``, fill triangulation gaps using the
            bottom-camera monocular fallback (see
            :mod:`alligaitor.bottom_fallback`). Defaults to ``False``.
    """

    models: ModelConfig
    calibration: CalibrationConfig
    sessions: List[SessionConfig]
    name: str = "group"
    output_xlsx: Optional[Path] = None
    gait: GaitConfig = field(default_factory=GaitConfig)
    discovery: Optional[DiscoveryConfig] = None
    skip_validation_videos: bool = False
    bottom_fallback: bool = False

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
        gait = GaitConfig.from_raw(raw.get("gait"))

        discovery_raw = raw.get("discovery")
        discovery = None
        if discovery_raw:
            discovery = DiscoveryConfig(
                input_dir=_resolve(base_dir, discovery_raw["input_dir"]),
                id_regex=discovery_raw["id_regex"],
                camera_regex=discovery_raw["camera_regex"],
                camera_role_map=dict(discovery_raw.get("camera_role_map", {})),
                rat_id_overrides=dict(discovery_raw.get("rat_id_overrides", {})),
            )

        return cls(
            models=models,
            calibration=calibration,
            sessions=sessions,
            name=name,
            output_xlsx=output_xlsx,
            gait=gait,
            discovery=discovery,
            skip_validation_videos=bool(raw.get("skip_validation_videos", False)),
            bottom_fallback=bool(raw.get("bottom_fallback", False)),
        )

    def to_yaml(self, path: PathLike) -> None:
        """Write this config back out to YAML, in the same schema
        :meth:`from_yaml` reads. Paths are written relative to ``path``'s
        parent directory where possible, falling back to absolute."""
        path = Path(path)
        base_dir = path.parent

        raw: dict = {
            "name": self.name,
            "models": {
                "side_model_dir": _relativize(base_dir, self.models.side_model_dir),
                "bottom_model_dir": _relativize(base_dir, self.models.bottom_model_dir),
            },
            "calibration": {
                "videos": {
                    role: _relativize(base_dir, p) for role, p in self.calibration.videos.items()
                },
                "output_path": _relativize(base_dir, self.calibration.output_path),
                "board_preset": self.calibration.board_preset,
                "min_corners_extrinsic": self.calibration.min_corners_extrinsic,
            },
            "sessions": [
                {
                    "name": session.name,
                    "rat_id": session.rat_id,
                    "output_dir": _relativize(base_dir, session.output_dir),
                    "videos": {
                        role: _relativize(base_dir, p) for role, p in session.videos.items()
                    },
                }
                for session in self.sessions
            ],
            "gait": {
                "speed_threshold_mm_s": self.gait.speed_threshold_mm_s,
                "min_contact_frames": self.gait.min_contact_frames,
                "max_bridge_gap_frames": self.gait.max_bridge_gap_frames,
                "min_consecutive_strides": self.gait.min_consecutive_strides,
                "stillness_window_seconds": self.gait.stillness_window_seconds,
                "stillness_window_speed_mm_s": self.gait.stillness_window_speed_mm_s,
                "min_still_seconds": self.gait.min_still_seconds,
                "min_valid_steps": self.gait.min_valid_steps,
                "stride_length_outlier_ratio": self.gait.stride_length_outlier_ratio,
            },
            "skip_validation_videos": self.skip_validation_videos,
            "bottom_fallback": self.bottom_fallback,
        }
        if self.output_xlsx is not None:
            raw["output_xlsx"] = _relativize(base_dir, self.output_xlsx)
        if self.discovery is not None:
            raw["discovery"] = {
                "input_dir": _relativize(base_dir, self.discovery.input_dir),
                "id_regex": self.discovery.id_regex,
                "camera_regex": self.discovery.camera_regex,
                "camera_role_map": dict(self.discovery.camera_role_map),
                "rat_id_overrides": dict(self.discovery.rat_id_overrides),
            }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(raw, f, sort_keys=False, default_flow_style=False)
