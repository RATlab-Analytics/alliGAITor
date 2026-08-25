"""Auto-discovery of a group's sessions from a folder of videos, using
regex-captured filename tokens for session identity and camera role.

Used by the GUI's config editor (``gui/group_config_dialog.py``) to build
:class:`alligaitor.config.SessionConfig` entries without hand-writing them
-- see :class:`alligaitor.config.DiscoveryConfig` for the rules this reads.
Nothing here is lab-specific: id/camera regexes and the camera-token-to-
role mapping are all supplied by the caller, not assumed.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from alligaitor.config import CAMERA_ROLES, DiscoveryConfig, SessionConfig

PathLike = Union[str, Path]

VIDEO_EXTENSIONS = (".mp4",)


def find_videos(folder: PathLike) -> List[Path]:
    """Every video file under ``folder``, searched recursively, sorted."""
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS)


def camera_tokens(videos: List[Path], camera_regex: str) -> List[str]:
    """Distinct camera tokens found across ``videos`` matching
    ``camera_regex``, in first-encounter order -- used to populate the
    config editor's per-role dropdowns."""
    pattern = re.compile(camera_regex)
    tokens: List[str] = []
    seen = set()
    for video in videos:
        m = pattern.search(video.name)
        if m and m.group(1) not in seen:
            seen.add(m.group(1))
            tokens.append(m.group(1))
    return tokens


def representative_video_for_token(videos: List[Path], camera_regex: str, token: str) -> Optional[Path]:
    """First video whose ``camera_regex`` match equals ``token`` -- used
    to grab a preview-thumbnail frame for the config editor's per-role
    field, since one token is assumed to apply to every video in the
    group (a rig's physical camera assignment doesn't change mid-group)."""
    pattern = re.compile(camera_regex)
    for video in videos:
        m = pattern.search(video.name)
        if m and m.group(1) == token:
            return video
    return None


def discover_sessions(
    discovery: DiscoveryConfig,
    cropped_dir: PathLike,
    predictions_dir: PathLike,
) -> Tuple[List[SessionConfig], List[str]]:
    """Group ``discovery.input_dir``'s videos into sessions.

    Applies ``discovery.id_regex`` (group 1 = session name) and
    ``discovery.camera_regex`` (group 1 = camera token, mapped to a role
    via ``discovery.camera_role_map``) to every video under
    ``discovery.input_dir``. A session is only emitted if it ends up with
    exactly one video per role in :data:`alligaitor.config.CAMERA_ROLES`;
    anything else (an unmatched filename, an unassigned camera token, a
    role with zero or more than one video) is reported back in the second
    return value instead of being silently dropped.

    Each emitted session's ``videos[role]`` points at where cropping is
    expected to write that video's cropped copy: ``<cropped_dir>/<path
    relative to discovery.input_dir>`` -- not at the raw video in
    ``discovery.input_dir``, and not namespaced by role (same-role
    filenames are already unique across a group, since they still carry
    their distinct camera token). This mirrors
    ``tools/crop_setup_dialog.py``'s own ``_out_path_for`` convention
    exactly, so ``config.yaml`` never needs rewriting once cropping
    actually happens -- cropping just has to produce files at these
    paths, which it already does when pointed at ``discovery.input_dir``
    / ``cropped_dir`` as its input/output folders. It also means one
    shared ``crop_positions.json`` under ``cropped_dir`` covers every
    role, since ``video_key()`` is already relative-path-based.
    ``output_dir`` is set to ``<predictions_dir>/<session name>``.

    Returns:
        ``(sessions, problems)`` -- ``problems`` is a list of
        human-readable strings describing anything that couldn't be
        grouped, for the config editor to surface.
    """
    id_pattern = re.compile(discovery.id_regex)
    camera_pattern = re.compile(discovery.camera_regex)
    cropped_dir = Path(cropped_dir)
    predictions_dir = Path(predictions_dir)

    # A regex that compiles but has no capture group (e.g. "" during a
    # config editor field mid-edit, or a plain literal with no
    # parentheses) would otherwise crash below at .group(1) instead of
    # being reported like any other bad regex.
    if id_pattern.groups < 1:
        return [], [f"id regex has no capture group: {discovery.id_regex!r}"]
    if camera_pattern.groups < 1:
        return [], [f"camera regex has no capture group: {discovery.camera_regex!r}"]

    videos = find_videos(discovery.input_dir)
    problems: List[str] = []

    by_session: Dict[str, Dict[str, Path]] = {}
    collisions: Dict[str, Dict[str, List[Path]]] = {}

    for video in videos:
        id_match = id_pattern.search(video.name)
        camera_match = camera_pattern.search(video.name)
        if not id_match:
            problems.append(f"{video.name}: id regex didn't match")
            continue
        if not camera_match:
            problems.append(f"{video.name}: camera regex didn't match")
            continue

        session_name = id_match.group(1)
        token = camera_match.group(1)
        role = discovery.camera_role_map.get(token)
        if role is None:
            problems.append(f"{video.name}: camera token '{token}' isn't assigned to a role")
            continue
        if role not in CAMERA_ROLES:
            problems.append(f"{video.name}: '{role}' isn't a valid camera role {CAMERA_ROLES}")
            continue

        session_roles = by_session.setdefault(session_name, {})
        if role in session_roles:
            collisions.setdefault(session_name, {}).setdefault(role, [session_roles[role]]).append(video)
            continue
        session_roles[role] = video

    for session_name, roles in collisions.items():
        for role, paths in roles.items():
            names = ", ".join(p.name for p in paths)
            problems.append(f"session '{session_name}': more than one '{role}' video ({names})")

    sessions: List[SessionConfig] = []
    for session_name, roles in sorted(by_session.items()):
        if session_name in collisions:
            continue  # already reported above; don't also emit a partial session for it
        missing = [r for r in CAMERA_ROLES if r not in roles]
        if missing:
            have = ", ".join(f"{r}={p.name}" for r, p in roles.items()) or "none"
            problems.append(f"session '{session_name}': missing {missing} (have: {have})")
            continue

        cropped_videos = {
            role: cropped_dir / roles[role].relative_to(discovery.input_dir) for role in CAMERA_ROLES
        }
        rat_id = discovery.rat_id_overrides.get(session_name, session_name)
        sessions.append(
            SessionConfig(
                name=session_name,
                videos=cropped_videos,
                output_dir=predictions_dir / session_name,
                rat_id=rat_id,
            )
        )

    return sessions, problems
