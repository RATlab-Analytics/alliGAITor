"""
App-wide settings for the alliGAITor GUI, persisted to
``app_data/settings.json`` alongside the job queue. Covers things that
apply to every job unless overridden: which side/bottom model the Model
menu currently has selected, the default id/camera regex used to seed a
newly added job's config editor (Settings > Preferences), and the
default output-base folder offered by the Add Job dialog.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

# Matches this rig's current "<session>_camN_coded.mp4" filenames (see
# configs/session_example.yaml) -- just a starting point. Any lab can
# repoint these via Settings > Preferences; nothing elsewhere assumes
# this exact convention.
DEFAULT_ID_REGEX = r"^(.+?)_cam\d+"
DEFAULT_CAMERA_REGEX = r"_(cam\d+)"

# Default token each role starts pre-filled with in a new job's config
# editor (see group_config_dialog.py's _populate_role_combos) -- a guess
# at this lab's usual wiring, always visible and editable per group, and
# never assumed anywhere a session is actually resolved (the saved
# per-group camera_role_map is what's actually used at run time). These
# are what the camera regex's capture group needs to produce for a token
# to match -- e.g. DEFAULT_CAMERA_REGEX above captures "cam0"/"cam1"/
# "cam2" verbatim, so the defaults below are those same strings.
DEFAULT_CAMERA_TOKENS = {"left": "cam0", "right": "cam1", "bottom": "cam2"}

_DEFAULTS = {
    "selected_side_model": None,
    "selected_bottom_model": None,
    "default_id_regex": DEFAULT_ID_REGEX,
    "default_camera_regex": DEFAULT_CAMERA_REGEX,
    "default_camera_tokens": dict(DEFAULT_CAMERA_TOKENS),
    "default_output_base": "",
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


# -- model selection --

def get_selected_model(app_data_dir: Path, role: str) -> Optional[str]:
    """``role`` is ``"side"`` or ``"bottom"``. Returns the ``models/``
    subdirectory name currently selected via the Model menu, or ``None``
    if nothing's been picked yet."""
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
    """Role -> token a new job's config editor pre-fills each camera
    role combo with (see group_config_dialog.py's _populate_role_combos),
    before the user has assigned anything for that particular group."""
    settings = load_settings(app_data_dir)
    tokens = dict(DEFAULT_CAMERA_TOKENS)
    tokens.update(settings.get("default_camera_tokens") or {})
    return tokens


def set_default_camera_tokens(app_data_dir: Path, tokens: Dict[str, str]) -> None:
    settings = load_settings(app_data_dir)
    settings["default_camera_tokens"] = dict(tokens)  # copy -- never share the caller's dict
    save_settings(app_data_dir, settings)


# -- default output base folder --

def get_default_output_base(app_data_dir: Path) -> str:
    return load_settings(app_data_dir)["default_output_base"]


def set_default_output_base(app_data_dir: Path, output_base: str) -> None:
    settings = load_settings(app_data_dir)
    settings["default_output_base"] = output_base
    save_settings(app_data_dir, settings)
