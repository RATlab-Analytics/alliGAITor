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
            consecutive strides into short runs for no reason tied to data
            quality. A bridged gap also stops counting as a camera-caused
            discard (see ``find_camera_caused_discards``), which is why a
            second, independent check
            (``stride_length_outlier_ratio``) exists for strides a gap
            *doesn't* explain: one silently missing entirely inside a
            span that still reads as clean.
        min_consecutive_strides: A paw's reported stride/step/ground-contact
            averages are computed only from strides that are part of a run
            of at least this many consecutive accepted stance events with
            no camera-caused discard (see
            :func:`alligaitor.gait.find_camera_caused_discards`) in the
            swing between any pair -- an isolated good detection that
            isn't part of such a run doesn't count, and a paw with no
            qualifying run at all reports ``NaN`` rather than an average
            built from too little (or too suspect) data. A step (paw vs.
            the contralateral paw) is a different measurement from a
            stride (paw vs. itself) and has its own evidence bar -- see
            ``min_valid_steps`` below -- so this field governs stride,
            ground-contact-time, and (indirectly, via the same run) step
            length, but is named for what it actually counts: consecutive
            strides. See :func:`alligaitor.gait.restrict_to_consecutive_runs`.
        stillness_window_seconds: Width, in seconds, of the window the
            whole-body speed below is measured across (see
            :func:`alligaitor.gait.windowed_body_speed`). Frame-to-frame
            speed of a single node is useless for this: measured on real
            trials, ``mid-back`` swings +/-12mm from reconstruction
            jitter alone while the rat stands still, which at ~12.5fps
            pose sampling reads as 150-225 mm/s -- above any threshold
            that would still sit below a real walking speed. Net
            displacement across a window cancels that jitter (the node
            comes back to nearly the same place) while leaving genuine
            translation untouched. Defaults to ``0.4``: trimmed the same
            frames at ``0.32`` across every measured trial, so it isn't
            balanced on a knife's edge, while ``0.48`` started eating
            genuine slow walking on the slowest trial.
        stillness_window_speed_mm_s: Below this whole-body speed, in
            mm/s, measured across ``stillness_window_seconds`` (not
            between adjacent frames), the reference node (see
            ``alligaitor.gait.REFERENCE_NODE``) counts as not
            translating. Used to trim any leading/trailing stretch where
            the rat has stopped moving -- once the body itself is
            stationary, a paw's own position can still jitter across
            ``speed_threshold_mm_s`` from tracking noise alone, which
            would otherwise look like a run of real steps in place. This
            asks whether the *animal* is translating at all, not whether
            one paw is currently planted mid-stride, so it is not
            comparable to ``speed_threshold_mm_s`` -- the two measure
            different things over different spans. Defaults to ``100``,
            which separated stopped stretches (windowed speed under
            ~80) from walking (240+) on every measured trial, including
            the slowest. Tune against real trials with
            ``scripts/debug_gait.py``.
        min_still_seconds: How long a stretch of sub-threshold body
            speed, bordering either end of the trial, counts as
            "stopped" rather than an ordinary brief slowdown mid-stride.
            Only a stretch touching the very start or very end of the
            trial is ever trimmed -- a pause in the middle is left
            alone. In seconds rather than frames because ``pose_3d`` is
            sampled at the slowest camera's frame rate (see
            :func:`alligaitor.pipeline.run_session`), so a frame count
            means a different real duration on every rig. See
            :func:`alligaitor.gait.active_window`.
        min_valid_steps: Fewest valid steps (see
            :func:`alligaitor.gait._step_lengths`) a paw needs before its
            average step length is reported at all; below this it is
            ``NaN`` and flagged on its own, independently of stride
            length and ground contact time. Step length is the one
            metric that depends on a *second* paw -- it measures forward
            distance from the contralateral paw's most recent touchdown
            -- so it can be untrustworthy on a crossing where everything
            else about this paw is clean, and needs a verdict of its
            own. Defaults to ``5``, holding step length to the same
            evidence bar a qualifying run must clear
            (``min_consecutive_strides``); kept a separate field so
            lowering that threshold to rescue runs doesn't silently
            lower this one too.
        stride_length_outlier_ratio: A stride longer than this many times
            a paw's own median stride length in the trial is flagged as
            likely hiding a missed stance -- e.g. a real stance the speed
            classifier failed to recognize -- rather than being one
            genuine stride, and breaks a qualifying run the same way a
            camera-caused discard does. This catches exactly what
            ``max_bridge_gap_frames`` cannot: triangulation can be clean
            the entire way through and still miss a real, brief stance.
            A stride with zero or negative net forward progress breaks a
            run the same way, unconditionally -- not scaled by this
            ratio, since no multiple of a real stride's length justifies
            a genuinely backward one. Investigated on real trials: these
            traced to genuine, cleanly-triangulated paw movement during
            a brief mid-crossing pause (grooming, sniffing) rather than
            to turning or walking backward, but the *cause* doesn't
            matter for this purpose -- directed locomotion shouldn't
            produce a non-positive stride, so one is never trustworthy
            regardless of why. This paw's own median (used for the
            ratio check above) is computed from its positive strides
            only, so a paused/backward one can't drag it down and mask
            a real too-long outlier next to it.
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
        mapping, dropping keys that are no longer fields instead of
        raising.

        Keeps a ``config.yaml`` or ``app_data/settings.json`` written
        before a tunable was renamed loadable -- a stale key would
        otherwise be a ``TypeError`` on every load. Anything dropped
        falls back to that field's current default, warned about rather
        than done silently.

        Both stillness tunables were renamed when
        :func:`alligaitor.gait.active_window` moved to a windowed speed:
        ``min_still_frames`` -> ``min_still_seconds`` (a frame count
        means a different duration on every rig), and
        ``stillness_speed_threshold_mm_s`` ->
        ``stillness_window_speed_mm_s``. The second is a rename on
        purpose even though the units didn't change: it compares against
        a speed measured across ``stillness_window_seconds`` rather than
        between adjacent frames, so an old file's value is off by more
        than an order of magnitude and silently honoring it would leave
        the check doing nothing.

        ``min_consecutive_steps`` -> ``min_consecutive_strides`` was a
        naming fix, not a behavior change: the field always counted
        consecutive stance events feeding stride/ground-contact-time
        averages (paw vs. itself), never the contralateral-paw
        measurement a "step" actually is (see ``min_valid_steps``) -- an
        old file's value means exactly what it always meant and just
        gets dropped-and-defaulted like any other unrecognized key here.
        """
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
    videos -- used by the GUI's config editor (see
    :mod:`alligaitor.discovery`) to (re)build :attr:`PipelineConfig.sessions`
    from ``input_dir`` rather than requiring them to be hand-written.

    Not consumed by the plain CLI pipeline itself: ``sessions`` is always
    written out fully resolved (see :meth:`PipelineConfig.to_yaml`), so a
    GUI-authored config file stays a complete, valid :class:`PipelineConfig`
    that ``python -m alligaitor.cli run`` can run standalone, with or
    without this block.

    Attributes:
        input_dir: Folder of source (pre-crop) videos to discover
            sessions from.
        id_regex: Applied to each video's filename; group 1 is the
            session name (e.g. ``"359a-BL"`` from
            ``"359a-BL_cam0_coded.mp4"``). Videos sharing a session name,
            one per camera role, form one session.
        camera_regex: Applied to each video's filename; group 1 is a
            camera token (e.g. ``"cam0"``), mapped to a role via
            :attr:`camera_role_map`.
        camera_role_map: Camera token -> role (``"left"``/``"right"``/
            ``"bottom"``). One mapping applies to every video in the
            group -- a rig's physical camera assignment doesn't change
            mid-group, unlike across groups (see the port-order caveat
            on :class:`SessionConfig`).
        rat_id_overrides: Session name -> rat_id, for sessions where the
            same rat crosses more than once within this group (see
            :attr:`SessionConfig.rat_id`). Empty unless the group
            actually has repeat crossings.
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
            for editing but not read by :func:`run_pipeline`/`run_group` --
            ``sessions`` is always the source of truth for a run.
    """

    models: ModelConfig
    calibration: CalibrationConfig
    sessions: List[SessionConfig]
    name: str = "group"
    output_xlsx: Optional[Path] = None
    gait: GaitConfig = field(default_factory=GaitConfig)
    discovery: Optional[DiscoveryConfig] = None

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
        )

    def to_yaml(self, path: PathLike) -> None:
        """Write this config back out to YAML, in the same schema
        :meth:`from_yaml` reads. Paths are written relative to ``path``'s
        parent directory where possible (matching :meth:`from_yaml`'s
        resolution convention), falling back to absolute if a path isn't
        actually under it.

        Used by the GUI's config editor to persist a group's
        ``config.yaml`` after auto-discovery -- the written file always
        has ``sessions`` fully resolved, so it stays a complete, valid
        config independent of the GUI (``discovery``, if present, is
        extra metadata for re-opening the editor, not something a plain
        ``alligaitor.cli run`` needs).
        """
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
