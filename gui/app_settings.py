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
App-wide GUI settings, persisted to ``app_data/settings.json``. Covers
selected models, default id/camera regex and camera tokens, default
gait/calibration tunables, and default output folder. Editable from
Settings > Preferences.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

from alligaitor.config import CalibrationConfig, GaitConfig

# Starting-point defaults; editable via Settings > Preferences.
DEFAULT_ID_REGEX = r"^(.+?)_cam\d+"
DEFAULT_CAMERA_REGEX = r"_(cam\d+)"

# Default per-role token pre-filled in a new job's config editor. Must
# match what DEFAULT_CAMERA_REGEX's capture group produces.
DEFAULT_CAMERA_TOKENS = {"left": "cam0", "right": "cam1", "bottom": "cam2"}


def _dataclass_field_default(cls, name: str):
    for f in dataclasses.fields(cls):
        if f.name == name:
            return f.default
    raise KeyError(name)


# Read from the dataclasses' own field defaults to avoid drift.
DEFAULT_GAIT = dataclasses.asdict(GaitConfig())
DEFAULT_MIN_CORNERS_EXTRINSIC = _dataclass_field_default(CalibrationConfig, "min_corners_extrinsic")

_DEFAULTS = {
    "models_dir": None,
    "selected_side_model": None,
    "selected_bottom_model": None,
    "default_id_regex": DEFAULT_ID_REGEX,
    "default_camera_regex": DEFAULT_CAMERA_REGEX,
    "default_camera_tokens": dict(DEFAULT_CAMERA_TOKENS),
    "default_output_base": "",
    "default_gait": dict(DEFAULT_GAIT),
    "default_min_corners_extrinsic": DEFAULT_MIN_CORNERS_EXTRINSIC,
    "default_skip_validation_videos": False,
    "default_bottom_fallback": False,
}


def _settings_path(app_data_dir: Path) -> Path:
    return Path(app_data_dir) / "settings.json"


def load_settings(app_data_dir: Path) -> dict:
    path = _settings_path(app_data_dir)
    if not path.exists():
        return dict(_DEFAULTS)
    with open(path) as f:
        raw = json.load(f)
    settings = dict(_DEFAULTS)
    settings.update(raw)
    return settings


def save_settings(app_data_dir: Path, settings: dict) -> None:
    path = _settings_path(app_data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(settings, f, indent=2)
    tmp.replace(path)


# -- models directory --

def get_models_dir(app_data_dir: Path) -> Optional[Path]:
    """User-configured folder containing model subdirectories, or ``None`` if unset."""
    raw = load_settings(app_data_dir).get("models_dir")
    return Path(raw) if raw else None


def set_models_dir(app_data_dir: Path, models_dir: Optional[Path]) -> None:
    settings = load_settings(app_data_dir)
    settings["models_dir"] = str(models_dir) if models_dir else None
    save_settings(app_data_dir, settings)


# -- model selection --

def get_selected_model(app_data_dir: Path, role: str) -> Optional[str]:
    """``role`` is ``"side"`` or ``"bottom"``. Returns the selected
    ``models/`` subdirectory name, or ``None`` if unset."""
    return load_settings(app_data_dir).get(f"selected_{role}_model")


def set_selected_model(app_data_dir: Path, role: str, model_dir_name: Optional[str]) -> None:
    settings = load_settings(app_data_dir)
    settings[f"selected_{role}_model"] = model_dir_name
    save_settings(app_data_dir, settings)


# -- default regexes (Settings > Preferences) --

def get_default_regexes(app_data_dir: Path) -> Tuple[str, str]:
    settings = load_settings(app_data_dir)
    return settings["default_id_regex"], settings["default_camera_regex"]


def set_default_regexes(app_data_dir: Path, id_regex: str, camera_regex: str) -> None:
    settings = load_settings(app_data_dir)
    settings["default_id_regex"] = id_regex
    settings["default_camera_regex"] = camera_regex
    save_settings(app_data_dir, settings)


def get_default_camera_tokens(app_data_dir: Path) -> Dict[str, str]:
    """Role -> token used to pre-fill a new job's camera role combos."""
    settings = load_settings(app_data_dir)
    tokens = dict(DEFAULT_CAMERA_TOKENS)
    tokens.update(settings.get("default_camera_tokens") or {})
    return tokens


def set_default_camera_tokens(app_data_dir: Path, tokens: Dict[str, str]) -> None:
    settings = load_settings(app_data_dir)
    settings["default_camera_tokens"] = dict(tokens)
    save_settings(app_data_dir, settings)


# -- default scoring (gait) and triangulation/calibration tunables --
# (Settings > Preferences > Scoring & Triangulation)

def get_default_gait_overrides(app_data_dir: Path) -> Dict[str, float]:
    """Values used to build the GaitConfig baked into a newly-saved job's
    config.yaml. Keys no longer present on GaitConfig are dropped and
    fall back to that field's current default."""
    settings = load_settings(app_data_dir)
    saved = settings.get("default_gait") or {}
    values = dict(DEFAULT_GAIT)
    values.update({k: v for k, v in saved.items() if k in DEFAULT_GAIT})
    return values


def set_default_gait_overrides(app_data_dir: Path, values: Dict[str, float]) -> None:
    settings = load_settings(app_data_dir)
    settings["default_gait"] = dict(values)
    save_settings(app_data_dir, settings)


def get_default_min_corners_extrinsic(app_data_dir: Path) -> int:
    settings = load_settings(app_data_dir)
    return settings.get("default_min_corners_extrinsic", DEFAULT_MIN_CORNERS_EXTRINSIC)


def set_default_min_corners_extrinsic(app_data_dir: Path, value: int) -> None:
    settings = load_settings(app_data_dir)
    settings["default_min_corners_extrinsic"] = value
    save_settings(app_data_dir, settings)


# -- default validation-video skip (Settings > Preferences) --

def get_default_skip_validation_videos(app_data_dir: Path) -> bool:
    """Whether a newly-saved job's config editor starts with "skip
    validation videos" checked. Defaults to ``False``."""
    return bool(load_settings(app_data_dir).get("default_skip_validation_videos", False))


def set_default_skip_validation_videos(app_data_dir: Path, value: bool) -> None:
    settings = load_settings(app_data_dir)
    settings["default_skip_validation_videos"] = bool(value)
    save_settings(app_data_dir, settings)


# -- default bottom-camera fallback (Settings > Preferences) --

def get_default_bottom_fallback(app_data_dir: Path) -> bool:
    """Whether a newly-saved job's config editor starts with "bottom
    fallback" checked. Defaults to ``False``."""
    return bool(load_settings(app_data_dir).get("default_bottom_fallback", False))


def set_default_bottom_fallback(app_data_dir: Path, value: bool) -> None:
    settings = load_settings(app_data_dir)
    settings["default_bottom_fallback"] = bool(value)
    save_settings(app_data_dir, settings)


# -- default output base folder --

def get_default_output_base(app_data_dir: Path) -> str:
    return load_settings(app_data_dir)["default_output_base"]


def set_default_output_base(app_data_dir: Path, output_base: str) -> None:
    settings = load_settings(app_data_dir)
    settings["default_output_base"] = output_base
    save_settings(app_data_dir, settings)
