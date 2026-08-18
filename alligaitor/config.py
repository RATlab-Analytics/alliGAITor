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

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

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
        board_preset: Which physical ChArUco board this recording used —
            a key into :data:`alligaitor.calibration.BOARD_PRESETS`
            (currently ``"original"``, the 8x8/15mm board, or ``"strip"``,
            the narrow 4x5/35mm board sized for the bottom camera's slit
            view). Different recordings may use different physical
            boards; this says which one to expect when detecting corners
            for this particular calibration.
        min_corners_extrinsic: Minimum ChArUco corners a frame needs to
            link two cameras' poses during calibration (see
            :data:`alligaitor.calibration.MIN_CORNERS_EXTRINSIC`). Defaults
            to aniposelib's own hardcoded value, 8; lower it for a
            recording where the bottom camera's slit view can't reach 8
            corners while a frame is simultaneously visible to a side
            camera.
    """

    videos: Dict[str, Path]
    output_path: Path
    board_preset: str = "original"
    min_corners_extrinsic: int = 8

    def __post_init__(self) -> None:
        _require_roles(self.videos, "Calibration config")


@dataclass
class SessionConfig:
    """One gait-recording session: one video per camera role.

    Attributes:
        name: Session identifier, used for output file naming.
        videos: Mapping of camera role to this session's video path.
        output_dir: Directory where 2D predictions and 3D output for this
            session are written.
    """

    name: str
    videos: Dict[str, Path]
    output_dir: Path

    def __post_init__(self) -> None:
        _require_roles(self.videos, f"Session '{self.name}'")


@dataclass
class PipelineConfig:
    """Top-level configuration: models, calibration, and one or more sessions."""

    models: ModelConfig
    calibration: CalibrationConfig
    sessions: List[SessionConfig]

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
            )
            for session_raw in raw["sessions"]
        ]

        return cls(models=models, calibration=calibration, sessions=sessions)
